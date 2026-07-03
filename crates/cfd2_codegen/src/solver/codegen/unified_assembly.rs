use super::bc_table::BcTable;
use super::coupled_common::{
    base_assembly_items, coefficient_value_expr, coupled_offsets, coupled_unknown_components,
    kernel_bindings_from_items,
};
use super::dsl as typed;
use super::state_access::{find_slot, state_component_slot};
use super::wgsl_ast::{
    AccessMode, AssignOp, Attribute, Block, Expr, Function, Item, Module, Param, Stmt, Type,
};
use super::wgsl_bindings::storage_var;
use super::wgsl_dsl as dsl;
use super::KernelWgsl;
use crate::solver::codegen::ir::{DiscreteOpKind, DiscreteSystem};
use crate::solver::codegen::reconstruction::scalar_reconstruction_stmts;
use crate::solver::gpu::enums::GpuBcKind;
use crate::solver::ir::ports::{ParamSpec, ResolvedStateSlotsSpec};
use crate::solver::ir::{
    Coefficient, Discretization, DispatchDomain, FieldKind, KernelProgram, LaunchSemantics, TermOp,
};

const UNIFIED_ASSEMBLY_WORKGROUP_SIZE: u32 = 64;

fn unified_assembly_needs_fluxes(system: &DiscreteSystem) -> bool {
    system.equations.iter().any(|eq| {
        eq.ops
            .iter()
            .any(|op| op.kind == DiscreteOpKind::Convection)
    })
}

fn validate_unified_assembly_inputs(needs_fluxes: bool, flux_stride: u32) {
    if needs_fluxes && flux_stride == 0 {
        panic!("unified_assembly requires flux_stride > 0 when convection ops are present");
    }
}

/// ALE marker, derived from the discrete system: any convection op consuming
/// its face flux relative to the mesh (`Term::relative_to_mesh`). Gates the
/// ALE storage-binding emission — static (non-ALE) models emit no new item
/// and stay byte-identical.
fn unified_assembly_needs_mesh_fluxes(system: &DiscreteSystem) -> bool {
    system.is_ale()
}

/// Fail-fast validation of the ALE v1 scope, called whenever a system is ALE
/// (mirrors the ALE+`dt_local` assert in time_integration.rs). Two silent
/// mishandlings are rejected at codegen time instead of compiling into
/// GCL-violating kernels:
///
/// * **Explicit flagged terms**: every mesh-relative subtraction site gates on
///   `Discretization::Implicit`, so an explicit `Div` term flagged
///   `relative_to_mesh` would get the bindings / moving-volume ddt /
///   continuity source emitted but NO flux subtraction — an inconsistent ALE
///   discretization. (`Term::with_mesh_relative` also rejects this at model
///   construction; this is the authoritative consumer-side backstop for
///   directly-constructed IR.)
///
/// * **Variable density**: `ale_relative_flux_expr` hardcodes the flux's
///   density factor as the uniform `constants.density`. The flux derivation
///   (src/solver/model/flux_derivation.rs `density_face_expr`) uses that same
///   uniform exactly when the state layout has NO `rho` field; a state-layout
///   `rho` means the flux carries an upwinded face density and the subtraction
///   would be wrong-by-ρ. v1 ALE scope is constant-density (incompressible)
///   only — variable-density fluxes need a persisted face density (flagged
///   follow-up, see `Term::relative_to_mesh` docs).
fn validate_ale_unified_assembly(system: &DiscreteSystem, slots: &ResolvedStateSlotsSpec) {
    for eq in &system.equations {
        for op in &eq.ops {
            assert!(
                !op.relative_to_mesh || op.discretization == Discretization::Implicit,
                "ALE (relative_to_mesh) is only supported on IMPLICIT Div/DivFlux terms: \
                 the mesh-relative flux subtraction is emitted at the implicit convection \
                 consumption points, so an explicit flagged term ({:?} on target '{}') would \
                 silently keep the absolute flux (GCL-violating)",
                op.term_op,
                eq.target.name(),
            );
        }
    }
    assert!(
        !slots.slots.iter().any(|s| s.name == "rho"),
        "ALE (relative_to_mesh) on a variable-density model is unsupported: the state \
         layout carries a 'rho' field, so the derived face flux uses an upwinded face \
         density, while the ALE subtraction assumes the uniform constants.density \
         (v1 constant-density scope; a variable-density ALE flux needs a persisted \
         face density)"
    );
}

/// `mesh_fluxes` storage binding (group 0 / binding 8, the first free mesh
/// slot): per-face volumetric swept rate `V̇_f = A_swept(f)/dt` (Volume/Time),
/// signed along the stored face normal (owner convention, exactly like the
/// `fluxes` mass flux). Computed host-side from swept-face geometry (SCL by
/// construction), NEVER from a velocity dotted with a normal. Allocated
/// zero-filled always at runtime, so an ALE model over a static mesh binds
/// zeros and the mesh-relative subtraction vanishes bitwise.
fn mesh_fluxes_item() -> Item {
    storage_var("mesh_fluxes", Type::array(Type::F32), 0, 8, AccessMode::Read)
}

/// ALE volume-history bindings (group 0 / bindings 9 and 15 — the remaining
/// free mesh slots; 14 is `face_wrap_shift` in the flux modules and is left
/// untouched so fused kernels can never collide). `cell_vols_old` = V^n,
/// `cell_vols_old_old` = V^{n-1}; both rotated by the ALE step seam
/// (`begin_ale_step`: old_old ← old ← current, BEFORE the new volumes are
/// uploaded) and seeded equal to `cell_vols` by `initialize_history`.
/// Consumed by the moving-volume ddt and the ALE volume rates
/// (`ale_volume_locals_setup`, time_integration.rs).
fn ale_vols_history_items() -> Vec<Item> {
    vec![
        storage_var(
            "cell_vols_old",
            Type::array(Type::F32),
            0,
            9,
            AccessMode::Read,
        ),
        storage_var(
            "cell_vols_old_old",
            Type::array(Type::F32),
            0,
            15,
            AccessMode::Read,
        ),
    ]
}

/// The convective face flux actually consumed by a convection op: the stored
/// (absolute) mass flux for static terms, or the mesh-relative flux
/// `phi_rel = phi - rho_f * mesh_fluxes[face_idx]` for `relative_to_mesh`
/// terms. The subtraction sits BEFORE the non-owner sign flip so `phi_rel`
/// inherits the flip exactly like `phi` — both cells of a shared face see one
/// consistent relative flux. Every downstream consumer (upwind matrix
/// coefficients, deferred correction, the `bounded` diagonal correction, the
/// `DivFlux` RHS and its pressure linearization) reads the accumulator, so
/// this is the single subtraction point for the whole assembly (and the
/// rhs_only / fused kernel variants are synthesized from this same
/// `KernelProgram`, inheriting it).
///
/// `rho_f` is the constant density coefficient (`constants.density`): v1 ALE
/// scope is constant-density (incompressible) mass fluxes, where
/// `phi = rho * (U·n) A` uses the same constant. Variable-density fluxes
/// (compressible/allmach upwinded `rho_f`) would need the flux kernel to
/// persist its face density — flagged follow-up, not supported here.
fn ale_relative_flux_expr(
    conv_op: &crate::solver::codegen::ir::DiscreteOp,
    flux_val_expr: Expr,
) -> Expr {
    if !conv_op.relative_to_mesh {
        return flux_val_expr;
    }
    flux_val_expr
        - Expr::ident("constants").field("density")
            * dsl::array_access("mesh_fluxes", Expr::ident("face_idx"))
}

pub fn generate_unified_assembly_wgsl(
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    flux_stride: u32,
    needs_gradients: bool,
    eos_params: &[ParamSpec],
) -> KernelWgsl {
    let coupled_stride = coupled_unknown_components(system).len() as u32;

    struct Dispatch<'a> {
        system: &'a DiscreteSystem,
        slots: &'a ResolvedStateSlotsSpec,
        flux_stride: u32,
        needs_gradients: bool,
        eos_params: &'a [ParamSpec],
    }
    impl typed::DispatchByStride<KernelWgsl> for Dispatch<'_> {
        fn call<Ax: typed::CoupledAxis>(&self) -> KernelWgsl {
            let needs_fluxes = unified_assembly_needs_fluxes(self.system);
            validate_unified_assembly_inputs(needs_fluxes, self.flux_stride);

            let mut module = Module::new();
            module.push(Item::Comment(
                "GENERATED BY CFD2 CODEGEN (unified_assembly)".to_string(),
            ));
            module.push(Item::Comment("DO NOT EDIT MANUALLY".to_string()));
            module.extend(base_assembly_items(
                self.needs_gradients,
                needs_fluxes,
                self.eos_params,
            ));
            if unified_assembly_needs_mesh_fluxes(self.system) {
                validate_ale_unified_assembly(self.system, self.slots);
                module.push(mesh_fluxes_item());
                for item in ale_vols_history_items() {
                    module.push(item);
                }
            }
            module.push(Item::Function(main_assembly_fn::<Ax>(
                self.system,
                self.slots,
                self.flux_stride,
                self.needs_gradients,
            )));
            KernelWgsl::from(module)
        }
    }
    typed::dispatch_by_coupled_stride(
        coupled_stride,
        Dispatch {
            system,
            slots,
            flux_stride,
            needs_gradients,
            eos_params,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

pub fn generate_unified_assembly_kernel_program(
    id: &str,
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    flux_stride: u32,
    needs_gradients: bool,
    eos_params: &[ParamSpec],
) -> Result<KernelProgram, String> {
    let coupled_stride = coupled_unknown_components(system).len() as u32;

    struct Dispatch<'a> {
        id: &'a str,
        system: &'a DiscreteSystem,
        slots: &'a ResolvedStateSlotsSpec,
        flux_stride: u32,
        needs_gradients: bool,
        eos_params: &'a [ParamSpec],
    }
    impl typed::DispatchByStride<Result<KernelProgram, String>> for Dispatch<'_> {
        fn call<Ax: typed::CoupledAxis>(&self) -> Result<KernelProgram, String> {
            let needs_fluxes = unified_assembly_needs_fluxes(self.system);
            validate_unified_assembly_inputs(needs_fluxes, self.flux_stride);

            let mut items = base_assembly_items(self.needs_gradients, needs_fluxes, self.eos_params);
            if unified_assembly_needs_mesh_fluxes(self.system) {
                validate_ale_unified_assembly(self.system, self.slots);
                items.push(mesh_fluxes_item());
                items.extend(ale_vols_history_items());
            }
            let bindings = kernel_bindings_from_items(&items)?;
            let main = main_assembly_fn::<Ax>(
                self.system,
                self.slots,
                self.flux_stride,
                self.needs_gradients,
            );
            let (launch, consumed_stmts) = launch_from_main_statements(&main.body.stmts)?;
            let kernel_stmts = &main.body.stmts[consumed_stmts..];

            let mut program = KernelProgram::new(self.id, DispatchDomain::Cells, launch, bindings);
            program.body = kernel_stmts.to_vec();
            program.eos_params = self.eos_params.to_vec();
            Ok(program)
        }
    }
    typed::dispatch_by_coupled_stride(
        coupled_stride,
        Dispatch {
            id,
            system,
            slots,
            flux_stride,
            needs_gradients,
            eos_params,
        },
    )?
}

fn launch_from_main_statements(stmts: &[Stmt]) -> Result<(LaunchSemantics, usize), String> {
    let idx_expr = match stmts.first() {
        Some(Stmt::Let { name, expr, .. }) if name == "idx" => expr.to_string(),
        _ => {
            return Err(
                "unified_assembly: expected first statement to define idx launch expression"
                    .to_string(),
            )
        }
    };
    let bounds_expr = match stmts.get(1) {
        Some(Stmt::If {
            cond,
            then_block,
            else_block,
        }) if else_block.is_none()
            && then_block.stmts.len() == 1
            && matches!(then_block.stmts.first(), Some(Stmt::Return(None))) =>
        {
            cond.to_string()
        }
        _ => {
            return Err(
                "unified_assembly: expected second statement to be idx bounds guard".to_string(),
            )
        }
    };
    Ok((
        LaunchSemantics::new(
            [UNIFIED_ASSEMBLY_WORKGROUP_SIZE, 1, 1],
            idx_expr,
            Some(bounds_expr),
        ),
        2,
    ))
}

fn main_assembly_fn<Ax: typed::CoupledAxis>(
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    flux_stride: u32,
    needs_gradients: bool,
) -> Function {
    let _stride = slots.stride;
    let unknowns = coupled_unknown_components(system);
    let coupled_stride = unknowns.len() as u32;
    let acc = typed::CoupledAccumulators::new(coupled_stride);
    let offsets = coupled_offsets(system);

    assert_eq!(
        Ax::STRIDE,
        coupled_stride,
        "main_assembly_fn: axis stride ({}) != coupled_stride ({coupled_stride})",
        Ax::STRIDE,
    );

    let params = vec![Param::new(
        "global_id",
        Type::vec3_u32(),
        vec![Attribute::Builtin("global_invocation_id".to_string())],
    )];

    let block_matrix = typed::NamedBlockCsrSoaMatrix::<Ax>::from_start_row_prefix(
        "matrix_values",
        "start_row",
        typed::ScalarType::F32,
        typed::UnitDim::dimensionless(),
    );

    let mut stmts = vec![
        dsl::let_expr(
            "idx",
            Expr::ident("global_id").field("y") * Expr::ident("constants").field("stride_x")
                + Expr::ident("global_id").field("x"),
        ),
        dsl::if_block_expr(
            Expr::ident("idx").ge(Expr::call_named(
                "arrayLength",
                vec![Expr::ident("cell_vols").addr_of()],
            )),
            dsl::block(vec![Stmt::Return(None)]),
            None,
        ),
        dsl::let_expr(
            "center",
            dsl::array_access("cell_centers", Expr::ident("idx")),
        ),
        dsl::let_expr("vol", dsl::array_access("cell_vols", Expr::ident("idx"))),
        dsl::let_expr(
            "start",
            dsl::array_access("cell_face_offsets", Expr::ident("idx")),
        ),
        dsl::let_expr(
            "end",
            dsl::array_access("cell_face_offsets", Expr::ident("idx") + 1u32),
        ),
        dsl::let_expr(
            "scalar_offset",
            dsl::array_access("scalar_row_offsets", Expr::ident("idx")),
        ),
        dsl::let_expr(
            "diag_rank",
            dsl::array_access("diagonal_indices", Expr::ident("idx"))
                - Expr::ident("scalar_offset"),
        ),
        dsl::let_expr(
            "num_neighbors",
            dsl::array_access("scalar_row_offsets", Expr::ident("idx") + 1u32)
                - Expr::ident("scalar_offset"),
        ),
    ];
    stmts
        .extend(acc.declare_start_rows(Expr::ident("scalar_offset"), Expr::ident("num_neighbors")));

    // Clear all block entries for this cell's rows.
    stmts.push(dsl::for_loop_expr(
        dsl::for_init_var_expr("rank", 0u32),
        Expr::ident("rank").lt(Expr::ident("num_neighbors")),
        dsl::for_step_increment_expr(Expr::ident("rank")),
        dsl::block(dsl::for_each_mat_entry_block(
            coupled_stride as usize,
            |r, c| {
                vec![dsl::assign_expr(
                    block_matrix
                        .entry(
                            &Expr::ident("rank"),
                            typed::block_row::<Ax>(r as u32),
                            typed::block_col::<Ax>(c as u32),
                        )
                        .expr,
                    0.0,
                )]
            },
        )),
    ));

    // diag_i / rhs_i accumulators
    stmts.extend(acc.declare());

    // Time derivative contributions (implicit only).
    stmts.extend(super::coupled_common::emit_ddt_contributions(
           system, slots, &offsets, &acc,
    ));

    // Source terms.
    for equation in &system.equations {
        for source_op in equation
            .ops
            .iter()
            .filter(|op| op.kind == DiscreteOpKind::Source)
        {
            let base_offset = *offsets
                .get(equation.target.name())
                .expect("missing target offset");

            let field_name = source_op.field.name();
            let field_offset_opt = offsets.get(field_name).copied();
            if source_op.discretization == Discretization::Implicit {
                let val =
                    coefficient_value_expr(slots, source_op.coeff.as_ref(), "idx", 0.0.into());
                // Static implicit diagonal (`sp`): the coefficient is the bare
                // diagonal contribution, NOT volume-integrated like an ordinary
                // implicit reaction source (`S_p * V`).
                let term = if source_op.static_diag {
                    val
                } else {
                    val * Expr::ident("vol")
                };
                if source_op.field.kind() != equation.target.kind() {
                    panic!(
                        "implicit source currently requires field.kind == target.kind (target={}, field={})",
                        equation.target.name(),
                        field_name
                    );
                }
                let field_offset = field_offset_opt.unwrap_or_else(|| {
                    panic!(
                        "implicit source requires '{}' to be a coupled unknown field",
                        field_name
                    )
                });
                let diag_block = block_matrix.row_entry(&Expr::ident("diag_rank"));

                for component in 0..equation.target.kind().component_count() as u32 {
                    let row_u_idx = base_offset + component;
                    let col_u_idx = field_offset + component;
                    if row_u_idx == col_u_idx {
                        // LHS -= term * phi.
                        // Assuming term is the coefficient S_p where S = S_p * phi.
                        // Contribution to diagonal is -S_p * V.
                        stmts.push(acc.sub_diag(row_u_idx, term.clone()));
                    } else {
                        // Cross-coupled source term: contribute to the (row, col) block entry.
                        stmts.push(dsl::assign_op_expr(
                            AssignOp::Sub,
                            diag_block
                                .entry(
                                    typed::block_row::<Ax>(row_u_idx),
                                    typed::block_col::<Ax>(col_u_idx),
                                )
                                .expr,
                            term.clone(),
                        ));
                    }
                }
            } else {
                // Explicit source: RHS += value * vol, per target component.
                //
                // Vector targets need per-component source values: a plain
                // vector-field coefficient of the target's kind is read
                // component-wise. Broadcasting one scalar into every
                // component is almost always a declaration bug (it produced
                // identical x/y sources), so it is rejected.
                let target_kind = equation.target.kind();
                let vector_source = match source_op.coeff.as_ref() {
                    Some(Coefficient::Field(f))
                        if f.kind() == target_kind && f.kind() != FieldKind::Scalar =>
                    {
                        Some(*f)
                    }
                    _ => None,
                };
                if let Some(direction) = source_op.direction.as_ref() {
                    // Directional source: scalar coefficient tree times a
                    // constant per-component direction multiplier.
                    assert_eq!(
                        direction.len(),
                        target_kind.component_count(),
                        "directional source on '{}' has {} direction components, target has {}",
                        equation.target.name(),
                        direction.len(),
                        target_kind.component_count()
                    );
                    let val = coefficient_value_expr(
                        slots,
                        source_op.coeff.as_ref(),
                        "idx",
                        0.0.into(),
                    );
                    for (component, dir) in direction.iter().enumerate() {
                        if *dir == 0.0 {
                            continue;
                        }
                        let u_idx = base_offset + component as u32;
                        stmts.push(acc.add_rhs(
                            u_idx,
                            val.clone() * Expr::lit_f32(*dir as f32) * Expr::ident("vol"),
                        ));
                    }
                } else if let Some(source_field) = vector_source {
                    let slot =
                        find_slot(slots, source_field.name()).unwrap_or_else(|| {
                            panic!(
                                "explicit vector source '{}' is not in the state layout",
                                source_field.name()
                            )
                        });
                    for component in 0..target_kind.component_count() as u32 {
                        let u_idx = base_offset + component;
                        let value = state_component_slot(
                            slots.stride,
                            "state",
                            "idx",
                            slot,
                            component,
                        );
                        stmts.push(acc.add_rhs(u_idx, value * Expr::ident("vol")));
                    }
                } else if target_kind.component_count() > 1 {
                    panic!(
                        "explicit source on vector target '{}' requires a vector-field \
                         coefficient of matching kind (per-component values); scalar \
                         broadcast is not supported",
                        equation.target.name()
                    );
                } else {
                    let val = coefficient_value_expr(
                        slots,
                        source_op.coeff.as_ref(),
                        "idx",
                        0.0.into(),
                    );
                    stmts.push(acc.add_rhs(base_offset, val * Expr::ident("vol")));
                }
            }
        }
    }

    // Bounded convection (OpenFOAM's `bounded Gauss`): accumulate the cell's
    // net outflow sum_f(phi_f) during the face loop and subtract it from the
    // diagonal afterwards — fvm::div(phi, u) - Sp(div(phi), u). This keeps
    // convection bounded while the flux field is not exactly divergence-free
    // (outer iterations, imperfectly converged steady states).
    // Per bounded unknown: (packed component index, term is mesh-relative).
    let bounded_unknowns: Vec<(u32, bool)> = system
        .equations
        .iter()
        .flat_map(|equation| {
            let base_offset = *offsets
                .get(equation.target.name())
                .expect("missing target offset");
            let components = equation.target.kind().component_count() as u32;
            equation
                .ops
                .iter()
                .filter(|op| {
                    op.kind == DiscreteOpKind::Convection
                        && op.discretization == Discretization::Implicit
                        && op.term_op == TermOp::Div
                        && op.bounded
                })
                .flat_map(move |op| {
                    (0..components).map(move |c| (base_offset + c, op.relative_to_mesh))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    for &(u_idx, _) in &bounded_unknowns {
        stmts.push(dsl::var_typed_expr(
            &format!("bounded_sum_phi_{u_idx}"),
            Type::F32,
            Some(0.0.into()),
        ));
    }

    // Face loop for diffusion contributions (implicit only).
    let face_loop_body = {
        let mut body = vec![
            dsl::let_expr(
                "face_idx",
                dsl::array_access("cell_faces", Expr::ident("k")),
            ),
            dsl::let_expr(
                "owner",
                dsl::array_access("face_owner", Expr::ident("face_idx")),
            ),
            dsl::let_expr(
                "neighbor_raw",
                dsl::array_access("face_neighbor", Expr::ident("face_idx")),
            ),
            dsl::let_expr(
                "boundary_type",
                dsl::array_access("face_boundary", Expr::ident("face_idx")),
            ),
            dsl::let_expr(
                "area",
                dsl::array_access("face_areas", Expr::ident("face_idx")),
            ),
            dsl::let_expr(
                "f_center",
                dsl::array_access("face_centers", Expr::ident("face_idx")),
            ),
            dsl::var_typed_expr(
                "normal",
                Type::Custom("Vector2".to_string()),
                Some(dsl::array_access("face_normals", Expr::ident("face_idx"))),
            ),
            dsl::var_typed_expr("is_boundary", Type::Bool, Some(false.into())),
            dsl::var_typed_expr("other_idx", Type::U32, Some(Expr::ident("idx"))),
            dsl::var_typed_expr("other_center", Type::Custom("Vector2".to_string()), None),
            // Make normal outward from `idx`.
            dsl::if_block_expr(
                Expr::ident("owner").ne(Expr::ident("idx")),
                dsl::block(vec![
                    dsl::assign_expr(
                        Expr::ident("normal").field("x"),
                        -Expr::ident("normal").field("x"),
                    ),
                    dsl::assign_expr(
                        Expr::ident("normal").field("y"),
                        -Expr::ident("normal").field("y"),
                    ),
                ]),
                None,
            ),
        ];

        let interior_block = dsl::block(vec![
            dsl::let_expr(
                "neighbor",
                Expr::call_named("u32", vec![Expr::ident("neighbor_raw")]),
            ),
            dsl::assign_expr(Expr::ident("other_idx"), Expr::ident("neighbor")),
            dsl::if_block_expr(
                Expr::ident("owner").ne(Expr::ident("idx")),
                dsl::block(vec![dsl::assign_expr(
                    Expr::ident("other_idx"),
                    Expr::ident("owner"),
                )]),
                None,
            ),
            dsl::assign_expr(
                Expr::ident("other_center"),
                dsl::array_access("cell_centers", Expr::ident("other_idx")),
            ),
        ]);

        let boundary_block = dsl::block(vec![
            dsl::assign_expr(Expr::ident("is_boundary"), true),
            dsl::assign_expr(Expr::ident("other_idx"), Expr::ident("idx")),
            dsl::assign_expr(Expr::ident("other_center"), Expr::ident("f_center")),
        ]);

        body.push(dsl::if_block_expr(
            Expr::ident("neighbor_raw").ne(-1),
            interior_block,
            Some(boundary_block),
        ));

        body.push(dsl::let_expr(
            "dx",
            Expr::ident("other_center").field("x") - Expr::ident("center").field("x"),
        ));
        body.push(dsl::let_expr(
            "dy",
            Expr::ident("other_center").field("y") - Expr::ident("center").field("y"),
        ));
        body.push(dsl::let_expr(
            "dist_proj",
            dsl::abs(
                Expr::ident("dx") * Expr::ident("normal").field("x")
                    + Expr::ident("dy") * Expr::ident("normal").field("y"),
            ),
        ));
        body.push(dsl::let_expr(
            "dist_euc",
            dsl::sqrt(
                Expr::ident("dx") * Expr::ident("dx") + Expr::ident("dy") * Expr::ident("dy"),
            ),
        ));
        body.push(dsl::var_typed_expr(
            "dist",
            Type::F32,
            Some(dsl::max("dist_euc", 1e-6)),
        ));
        body.push(dsl::if_block_expr(
            Expr::ident("dist_proj").gt(1e-6),
            dsl::block(vec![dsl::assign_expr(
                Expr::ident("dist"),
                Expr::ident("dist_proj"),
            )]),
            None,
        ));

        // Distance-weighted face interpolation weight (owner weight =
        // d_neigh / (d_own + d_neigh), the standard FV linear weight).
        // Matches the derived Rhie-Chow flux kernel's Lerp convention —
        // the assembly's face coefficients MUST interpolate identically to
        // the flux module's face d_p or the pressure system loses
        // consistency. On uniform meshes lambda = 0.5 (the previous
        // arithmetic mean); on graded meshes the mean is only first-order.
        // (Vector2 is the custom STRUCT; convert member-wise for distance().)
        body.push(dsl::let_expr(
            "lam_f_center_v",
            dsl::vec2_f32(
                Expr::ident("f_center").field("x"),
                Expr::ident("f_center").field("y"),
            ),
        ));
        body.push(dsl::let_expr(
            "lam_d_own",
            dsl::distance(
                dsl::vec2_f32(
                    Expr::ident("center").field("x"),
                    Expr::ident("center").field("y"),
                ),
                Expr::ident("lam_f_center_v"),
            ),
        ));
        body.push(dsl::let_expr(
            "lam_d_neigh",
            dsl::distance(
                dsl::vec2_f32(
                    Expr::ident("other_center").field("x"),
                    Expr::ident("other_center").field("y"),
                ),
                Expr::ident("lam_f_center_v"),
            ),
        ));
        body.push(dsl::let_expr(
            "lam_total",
            Expr::ident("lam_d_own") + Expr::ident("lam_d_neigh"),
        ));
        body.push(dsl::var_typed_expr("lambda_f", Type::F32, Some(0.5.into())));
        body.push(dsl::if_block_expr(
            Expr::ident("lam_total").gt(1e-6),
            dsl::block(vec![dsl::assign_expr(
                Expr::ident("lambda_f"),
                Expr::ident("lam_d_neigh") / Expr::ident("lam_total"),
            )]),
            None,
        ));

        body.push(dsl::let_expr(
            "scalar_mat_idx",
            dsl::array_access("cell_face_matrix_indices", Expr::ident("k")),
        ));
        body.push(dsl::let_expr(
            "neighbor_rank",
            Expr::ident("scalar_mat_idx") - Expr::ident("scalar_offset"),
        ));

        // Diffusion contributions per equation.
        for equation in &system.equations {
            let base_offset = *offsets
                .get(equation.target.name())
                .expect("missing target offset");

            // All implicit diffusion ops on this equation. Most equations have
            // at most one (the viscous / heat-conduction Laplacian), but a
            // conserved equation can carry a SECOND implicit diffusion — the
            // biharmonic stabilizer `laplacian(eps4*c, lap_X)` diffusing the
            // auxiliary undivided-Laplacian unknown into the conserved row.
            let implicit_diffusion_ops: Vec<_> = equation
                .ops
                .iter()
                .filter(|op| {
                    op.kind == DiscreteOpKind::Diffusion
                        && op.discretization == Discretization::Implicit
                })
                .collect();
            let multiple_implicit_diffusion = implicit_diffusion_ops.len() > 1;
            for diff_op in implicit_diffusion_ops {
                if diff_op.field.kind() != equation.target.kind() {
                    panic!(
                        "implicit diffusion currently requires field.kind == target.kind (target={}, field={})",
                        equation.target.name(),
                        diff_op.field.name()
                    );
                }

                let field_name = diff_op.field.name();
                let field_base_offset = offsets.get(field_name).copied().unwrap_or_else(|| {
                    panic!(
                        "implicit diffusion currently requires '{}' to be a coupled unknown field",
                        field_name
                    )
                });

                // Face-interpolated coefficient for implicit diffusion.
                // This is critical for consistency with the Rhie-Chow flux formula:
                // the pressure Laplacian coefficient (rho * d_p) must match the
                // face-interpolated d_p used in the momentum flux computation.
                let kappa_own =
                    coefficient_value_expr(slots, diff_op.coeff.as_ref(), "idx", 1.0.into());
                let kappa_other =
                    coefficient_value_expr(slots, diff_op.coeff.as_ref(), "other_idx", 1.0.into());
                // Distance-weighted for interior faces; owner value at boundaries.
                let kappa = dsl::select(
                    kappa_own.clone(),
                    kappa_own * Expr::ident("lambda_f")
                        + kappa_other * (Expr::from(1.0) - Expr::ident("lambda_f")),
                    !Expr::ident("is_boundary"),
                );

                // Keep the historical single-op name `diff_coeff_<target>`
                // (preserves byte-identical WGSL for every existing model);
                // disambiguate by field only when an equation carries more than
                // one implicit diffusion op (e.g. viscous + biharmonic).
                let diff_coeff_name = if multiple_implicit_diffusion {
                    format!("diff_coeff_{}_{}", equation.target.name(), field_name)
                } else {
                    format!("diff_coeff_{}", equation.target.name())
                };
                body.push(dsl::let_expr(
                    &diff_coeff_name,
                    kappa.clone() * Expr::ident("area") / Expr::ident("dist"),
                ));

                for component in 0..equation.target.kind().component_count() as u32 {
                    let row_u_idx = base_offset + component;
                    let col_u_idx = field_base_offset + component;

                    // Boundary values are taken from the field being diffused (the "column"
                    // variable), not from the equation target.
                    let bc = BcTable::new(Expr::ident("face_idx"), coupled_stride);
                    let (bc_kind_expr, bc_value_expr) = bc.lookup(col_u_idx);

                    let diag_block = block_matrix.row_entry(&Expr::ident("diag_rank"));

                    let interior_contrib = dsl::block(vec![
                        // Diagonal contribution.
                        if row_u_idx == col_u_idx {
                            acc.add_diag(row_u_idx, Expr::ident(&diff_coeff_name))
                        } else {
                            dsl::assign_op_expr(
                                AssignOp::Add,
                                diag_block
                                    .entry(
                                        typed::block_row::<Ax>(row_u_idx),
                                        typed::block_col::<Ax>(col_u_idx),
                                    )
                                    .expr,
                                Expr::ident(&diff_coeff_name),
                            )
                        },
                        // Neighbor contribution.
                        dsl::assign_op_expr(
                            AssignOp::Sub,
                            block_matrix
                                .entry(
                                    &Expr::ident("neighbor_rank"),
                                    typed::block_row::<Ax>(row_u_idx),
                                    typed::block_col::<Ax>(col_u_idx),
                                )
                                .expr,
                            Expr::ident(&diff_coeff_name),
                        ),
                    ]);

                    // Neumann value is the outward normal gradient g = dphi/dn. The implicit
                    // diffusion operator on the LHS is -div(kappa grad phi); its known boundary
                    // face contribution -kappa*g*A moves to the RHS as +kappa*g*A.
                    let neumann_rhs = kappa.clone() * Expr::ident("area") * bc_value_expr.clone();
                    let boundary_contrib = {
                        let is_velocity_field = matches!(field_name, "u" | "U" | "rho_u" | "rhoU");
                        let is_slipwall = Expr::ident("boundary_type").eq(Expr::from(4u32));

                        // SlipWall: emulate OpenFOAM `type slip` by projecting out the normal
                        // component at the boundary. This cannot be represented by the scalar
                        // per-component BC table, so handle it here (diffusion terms only).
                        let slip_bc_value = if is_velocity_field
                            && equation.target.kind().component_count() >= 2
                            && component <= 1
                        {
                            let field_slot = slots
                                .slots
                                .iter()
                                .find(|s| s.name == field_name)
                                .unwrap_or_else(|| {
                                    panic!("missing field '{}' in resolved state slots", field_name)
                                });
                            let vx =
                                state_component_slot(slots.stride, "state", "idx", field_slot, 0);
                            let vy =
                                state_component_slot(slots.stride, "state", "idx", field_slot, 1);
                            let nx = Expr::ident("normal").field("x");
                            let ny = Expr::ident("normal").field("y");
                            let un = vx.clone() * nx.clone() + vy.clone() * ny.clone();

                            Some(if component == 0 {
                                vx - un * nx
                            } else {
                                vy - un * ny
                            })
                        } else {
                            None
                        };

                        let diag_add = if row_u_idx == col_u_idx {
                            acc.add_diag(row_u_idx, Expr::ident(&diff_coeff_name))
                        } else {
                            dsl::assign_op_expr(
                                AssignOp::Add,
                                diag_block
                                    .entry(
                                        typed::block_row::<Ax>(row_u_idx),
                                        typed::block_col::<Ax>(col_u_idx),
                                    )
                                    .expr,
                                Expr::ident(&diff_coeff_name),
                            )
                        };

                        let slip_block = slip_bc_value.map(|value| {
                            dsl::block(vec![
                                diag_add.clone(),
                                acc.add_rhs(row_u_idx, Expr::ident(&diff_coeff_name) * value),
                            ])
                        });

                        let default_block = dsl::block(vec![dsl::if_block_expr(
                            bc_kind_expr.eq(GpuBcKind::Dirichlet),
                            dsl::block(vec![
                                diag_add,
                                acc.add_rhs(
                                    row_u_idx,
                                    Expr::ident(&diff_coeff_name) * bc_value_expr,
                                ),
                            ]),
                            Some(dsl::block(vec![dsl::if_block_expr(
                                bc_kind_expr.eq(GpuBcKind::Neumann),
                                dsl::block(vec![acc.add_rhs(row_u_idx, neumann_rhs)]),
                                None,
                            )])),
                        )]);

                        if let Some(slip_block) = slip_block {
                            dsl::block(vec![dsl::if_block_expr(
                                is_slipwall,
                                slip_block,
                                Some(default_block),
                            )])
                        } else {
                            default_block
                        }
                    };

                    body.push(dsl::if_block_expr(
                        !Expr::ident("is_boundary"),
                        interior_contrib,
                        Some(boundary_contrib),
                    ));
                }
            }

            // Explicit diffusion (RHS-only), supporting `laplacian(k, field)` where `field` may
            // differ from the equation target (e.g., viscous term for conserved momentum using
            // the derived velocity `u`).
            if let Some(diff_op) = equation.ops.iter().find(|op| {
                op.kind == DiscreteOpKind::Diffusion
                    && op.discretization == Discretization::Explicit
                    && !op.transpose_dev2
            }) {
                if diff_op.field.kind() != equation.target.kind() {
                    panic!(
                        "explicit diffusion currently requires field.kind == target.kind (target={}, field={})",
                        equation.target.name(),
                        diff_op.field.name()
                    );
                }

                let field_name = diff_op.field.name();

                let kappa_own =
                    coefficient_value_expr(slots, diff_op.coeff.as_ref(), "idx", 1.0.into());
                let kappa_other =
                    coefficient_value_expr(slots, diff_op.coeff.as_ref(), "other_idx", 1.0.into());
                let kappa_face = dsl::select(
                    kappa_own.clone(),
                    kappa_own.clone() * Expr::ident("lambda_f")
                        + kappa_other * (Expr::from(1.0) - Expr::ident("lambda_f")),
                    !Expr::ident("is_boundary"),
                );

                let diff_coeff_name =
                    format!("diff_coeff_exp_{}_{}", equation.target.name(), field_name);
                body.push(dsl::let_expr(
                    &diff_coeff_name,
                    kappa_face * Expr::ident("area") / Expr::ident("dist"),
                ));

                let field_offset_opt = offsets.get(field_name).copied();
                let is_derived_u = matches!(field_name, "u" | "U")
                    && offsets.contains_key("rho")
                    && offsets.contains_key("rho_u");

                for component in 0..equation.target.kind().component_count() as u32 {
                    let u_idx = base_offset + component;
                    let field_slot = slots
                        .slots
                        .iter()
                        .find(|s| s.name == field_name)
                        .unwrap_or_else(|| {
                            panic!("missing field '{}' in resolved state slots", field_name)
                        });
                    let phi_own =
                        state_component_slot(slots.stride, "state", "idx", field_slot, component);
                    let phi_neigh = state_component_slot(
                        slots.stride,
                        "state",
                        "other_idx",
                        field_slot,
                        component,
                    );

                    let interior_contrib =
                        dsl::block(vec![acc.add_rhs(
                            u_idx,
                            Expr::ident(&diff_coeff_name) * (phi_neigh - phi_own.clone()),
                        )]);

                    let boundary_contrib = if let Some(field_base_offset) = field_offset_opt {
                        let field_u_idx = field_base_offset + component;
                        let bc = BcTable::new(Expr::ident("face_idx"), coupled_stride);
                        let (bc_kind_expr, bc_value_expr) = bc.lookup(field_u_idx);

                        // Outward-gradient Neumann: explicit +div(kappa grad phi) on the RHS
                        // gains +kappa*g*A at the boundary face (same convention as the
                        // implicit path above).
                        let neumann_rhs =
                            kappa_own.clone() * Expr::ident("area") * bc_value_expr.clone();

                        dsl::block(vec![dsl::if_block_expr(
                            bc_kind_expr.eq(GpuBcKind::Dirichlet),
                            dsl::block(vec![acc.add_rhs(
                                u_idx,
                                Expr::ident(&diff_coeff_name) * (bc_value_expr - phi_own),
                            )]),
                            Some(dsl::block(vec![dsl::if_block_expr(
                                bc_kind_expr.eq(GpuBcKind::Neumann),
                                dsl::block(vec![acc.add_rhs(u_idx, neumann_rhs)]),
                                None,
                            )])),
                        )])
                    } else if is_derived_u {
                        // Derive a velocity boundary value from conserved BCs: u = rho_u / rho.
                        let rho_idx = *offsets.get("rho").expect("missing rho offset");
                        let rho_u_base = *offsets.get("rho_u").expect("missing rho_u offset");
                        let rho_u_idx = rho_u_base + component;

                        let bc = BcTable::new(Expr::ident("face_idx"), coupled_stride);
                        let (bc_rho_kind, bc_rho_val) = bc.lookup(rho_idx);
                        let (bc_rho_u_kind, bc_rho_u_val) = bc.lookup(rho_u_idx);

                        let rho_slot =
                            slots
                                .slots
                                .iter()
                                .find(|s| s.name == "rho")
                                .unwrap_or_else(|| {
                                    panic!("missing field 'rho' in resolved state slots")
                                });
                        let rho_own =
                            state_component_slot(slots.stride, "state", "idx", rho_slot, 0);
                        let rho_bc =
                            dsl::select(rho_own, bc_rho_val, bc_rho_kind.eq(GpuBcKind::Dirichlet));
                        let rho_bc_safe = dsl::max(rho_bc, 1e-12);
                        let u_bc = bc_rho_u_val / rho_bc_safe;
                        let u_other =
                            dsl::select(phi_own.clone(), u_bc, bc_rho_u_kind.eq(GpuBcKind::Dirichlet));

                        dsl::block(vec![acc.add_rhs(
                            u_idx,
                            Expr::ident(&diff_coeff_name) * (u_other - phi_own),
                        )])
                    } else {
                        // No BC information for this derived field; default to zero-gradient.
                        dsl::block(vec![])
                    };

                    body.push(dsl::if_block_expr(
                        !Expr::ident("is_boundary"),
                        interior_contrib,
                        Some(boundary_contrib),
                    ));
                }
            }

            // Explicit transpose/deviatoric viscous correction (RHS-only):
            //   -div(coeff * dev2((grad field)^T)),  dev2(A) = A - (2/3) tr(A) I
            // in the sum-to-zero residual. Explicit residual terms accumulate
            // negated on the RHS (cf. grad p: `rhs -= p_f * n * A`), so the
            // outward face flux F_j = n_i d(u_i)/d(x_j) - (2/3) divU n_j
            // enters as `rhs_j += coeff_f * A * F_j` — the same orientation
            // by which the explicit laplacian realizes -div(coeff grad phi)
            // via `rhs += coeff*A/d*(phi_N - phi_P)`.
            //
            // Face gradients are the owner/neighbor average of grad_state
            // cell gradients; at boundary faces `other_idx == idx`, so this
            // degrades to the owner gradient, whose BC ghost handling is
            // already folded in by packed_state_gradients — no bc branch.
            //
            // grad_state is STATE-OFFSET keyed (slots base_offset), NOT the
            // coupled-rank `offsets` map — the known rank-vs-offset latent
            // bug class (two prior engine bugs).
            for dev2_op in equation.ops.iter().filter(|op| {
                op.kind == DiscreteOpKind::Diffusion
                    && op.discretization == Discretization::Explicit
                    && op.transpose_dev2
            }) {
                if dev2_op.field.kind() != FieldKind::Vector2
                    || equation.target.kind() != FieldKind::Vector2
                {
                    panic!(
                        "transpose_dev2 requires a Vector2 field and target (target={}, field={})",
                        equation.target.name(),
                        dev2_op.field.name()
                    );
                }

                let field_name = dev2_op.field.name();
                let field_offset = slots
                    .slots
                    .iter()
                    .find(|s| s.name == field_name)
                    .map(|s| s.base_offset)
                    .unwrap_or_else(|| {
                        panic!("missing field '{}' in resolved state slots", field_name)
                    });
                let field_slot = slots
                    .slots
                    .iter()
                    .find(|s| s.name == field_name)
                    .expect("slot checked above");

                // Face-averaged gradient of velocity component `c`, as a
                // vec2<f32> (grad_state elements are the Vector2 STRUCT,
                // which WGSL cannot add — convert member-wise first, like
                // the convection reconstruction does).
                let grad_face = |c: u32| -> Expr {
                    if needs_gradients {
                        let own = dsl::array_access_linear(
                            "grad_state",
                            Expr::ident("idx"),
                            slots.stride,
                            field_offset + c,
                        );
                        let neigh = dsl::array_access_linear(
                            "grad_state",
                            Expr::ident("other_idx"),
                            slots.stride,
                            field_offset + c,
                        );
                        let own_v = dsl::vec2_f32(own.clone().field("x"), own.field("y"));
                        let neigh_v =
                            dsl::vec2_f32(neigh.clone().field("x"), neigh.field("y"));
                        (own_v + neigh_v) * 0.5
                    } else {
                        // Two-point fallback (same as the convection
                        // reconstruction fallback). This variant is never
                        // scheduled once the dev2 term forces the gradients
                        // pipeline on, but the kernel must still compile;
                        // at boundary faces it degrades to zero.
                        let phi_own = state_component_slot(
                            slots.stride,
                            "state",
                            "idx",
                            field_slot,
                            c,
                        );
                        let phi_neigh = state_component_slot(
                            slots.stride,
                            "state",
                            "other_idx",
                            field_slot,
                            c,
                        );
                        let diff = phi_neigh - phi_own;
                        let denom = dsl::max(
                            Expr::ident("dx") * Expr::ident("dx")
                                + Expr::ident("dy") * Expr::ident("dy"),
                            1e-12,
                        );
                        dsl::vec2_f32(
                            diff.clone() * Expr::ident("dx") / denom.clone(),
                            diff * Expr::ident("dy") / denom,
                        )
                    }
                };

                let prefix = format!("dev2_{}_{}", equation.target.name(), field_name);
                let gx_name = format!("{prefix}_gx");
                let gy_name = format!("{prefix}_gy");
                let div_name = format!("{prefix}_div");
                let mu_name = format!("{prefix}_mu");

                body.push(dsl::let_expr(&gx_name, grad_face(0)));
                body.push(dsl::let_expr(&gy_name, grad_face(1)));
                body.push(dsl::let_expr(
                    &div_name,
                    Expr::ident(&gx_name).field("x") + Expr::ident(&gy_name).field("y"),
                ));

                let kappa_own =
                    coefficient_value_expr(slots, dev2_op.coeff.as_ref(), "idx", 1.0.into());
                let kappa_other =
                    coefficient_value_expr(slots, dev2_op.coeff.as_ref(), "other_idx", 1.0.into());
                body.push(dsl::let_expr(
                    &mu_name,
                    dsl::select(
                        kappa_own.clone(),
                        kappa_own * Expr::ident("lambda_f")
                            + kappa_other * (Expr::from(1.0) - Expr::ident("lambda_f")),
                        !Expr::ident("is_boundary"),
                    ),
                ));

                // F_x = n_x ∂u_x/∂x + n_y ∂u_y/∂x − (2/3) divU n_x
                let flux_x = Expr::ident("normal").field("x") * Expr::ident(&gx_name).field("x")
                    + Expr::ident("normal").field("y") * Expr::ident(&gy_name).field("x")
                    - Expr::from(2.0 / 3.0)
                        * Expr::ident(&div_name)
                        * Expr::ident("normal").field("x");
                // F_y = n_x ∂u_x/∂y + n_y ∂u_y/∂y − (2/3) divU n_y
                let flux_y = Expr::ident("normal").field("x") * Expr::ident(&gx_name).field("y")
                    + Expr::ident("normal").field("y") * Expr::ident(&gy_name).field("y")
                    - Expr::from(2.0 / 3.0)
                        * Expr::ident(&div_name)
                        * Expr::ident("normal").field("y");

                body.push(acc.add_rhs(
                    base_offset,
                    Expr::ident(&mu_name) * Expr::ident("area") * flux_x,
                ));
                body.push(acc.add_rhs(
                    base_offset + 1,
                    Expr::ident(&mu_name) * Expr::ident("area") * flux_y,
                ));
            }

            // 2. Convection
            if let Some(conv_op) = equation.ops.iter().find(|op| {
                op.kind == DiscreteOpKind::Convection
                    && op.discretization == Discretization::Implicit
            }) {
                // Flux buffer is interpreted as a packed per-unknown-component face flux table.
                // Indexing is `fluxes[face * flux_stride + u_idx]`, where `u_idx` is the packed
                // unknown component index in the coupled system.
                //
                // Flux population is handled by an optional flux module kernel; assembly only
                // assumes the packed `(face_idx, u_idx)` layout.

                // `DivFlux` terms represent conservative flux divergence:
                // the face flux is precomputed (e.g., KT) and should contribute RHS-only:
                //   RHS -= sum_face(sign * flux_face_component)
                // This must not be treated like a scalar convection operator.
                if conv_op.term_op == TermOp::DivFlux {
                    for component in 0..equation.target.kind().component_count() as u32 {
                        let u_idx = base_offset + component;
                        let flux_val_expr = dsl::array_access_linear(
                            "fluxes",
                            Expr::ident("face_idx"),
                            flux_stride,
                            u_idx,
                        );
                        let flux_val_expr = ale_relative_flux_expr(conv_op, flux_val_expr);

                        body.push(acc.declare_phi(u_idx, flux_val_expr));
                        body.push(dsl::if_block_expr(
                            Expr::ident("owner").ne(Expr::ident("idx")),
                            dsl::block(vec![dsl::assign_op_expr(
                                AssignOp::Sub,
                                acc.phi(u_idx),
                                acc.phi(u_idx) * 2.0,
                            )]),
                            None,
                        ));
                        body.push(acc.sub_rhs(u_idx, acc.phi(u_idx)));

                        // Deferred-correction Newton linearization of this
                        // mass-flux divergence against the pressure (see
                        // `Term::linearize_pressure_flux`). The face mass flux
                        // `phi = rho_f * (U.n) * A` depends on p through the EOS
                        // density `rho_f = rho_ref + psi*p`, so its Jacobian is
                        //   a_f = d(phi)/dp = psi_f * (U.n) * A = phi * psi/rho.
                        // We add an implicit upwind convection of p by `a_f` to
                        // the matrix AND the same operator applied to the FROZEN
                        // state pressure to the RHS: the two cancel at outer
                        // convergence (so the converged solution is untouched —
                        // low-Mach and steady-MMS results are unchanged), while
                        // the implicit coupling damps the transonic/supersonic
                        // iteration that the explicit `div(rho_f U)` feedback
                        // would otherwise run to vacuum. `acc.phi(u_idx)` here is
                        // already the OUTWARD-from-`idx` mass flux, so `a_f`
                        // inherits the correct upwind sign for free.
                        if let Some(lin_coeff) = &conv_op.linearize_pressure_flux {
                            let p_slot = slots
                                .slots
                                .iter()
                                .find(|s| s.name == equation.target.name())
                                .unwrap_or_else(|| {
                                    panic!(
                                        "linearize_pressure_flux: missing pressure field '{}' in state slots",
                                        equation.target.name()
                                    )
                                });
                            let rho_slot = slots
                                .slots
                                .iter()
                                .find(|s| s.name == "rho")
                                .unwrap_or_else(|| {
                                    panic!(
                                        "linearize_pressure_flux requires a 'rho' state field"
                                    )
                                });
                            let psi_own = coefficient_value_expr(
                                slots,
                                Some(lin_coeff),
                                "idx",
                                Expr::from(0.0),
                            );
                            let rho_own = dsl::max(
                                state_component_slot(slots.stride, "state", "idx", rho_slot, 0),
                                1.0e-30,
                            );
                            // a_f = phi * psi_own / rho_own (Jacobian d(div phi)/dp,
                            // owner-cell approximation — exact value only affects the
                            // damped iteration path, not the converged solution).
                            let a_name = format!("a_lin_{u_idx}");
                            body.push(dsl::var_typed_expr(
                                &a_name,
                                Type::F32,
                                Some(acc.phi(u_idx) * psi_own / rho_own),
                            ));
                            let a_f = Expr::ident(&a_name);
                            let flux_pos = dsl::max(a_f.clone(), 0.0);
                            let flux_neg = dsl::min(a_f, 0.0);
                            let p_own_state =
                                state_component_slot(slots.stride, "state", "idx", p_slot, 0);
                            let p_neigh_state =
                                state_component_slot(slots.stride, "state", "other_idx", p_slot, 0);

                            // Interior face: full upwind stencil (diagonal +
                            // neighbor coupling) with the frozen-state deferred RHS.
                            let interior_lin = dsl::block(vec![
                                acc.add_diag(u_idx, flux_pos.clone()),
                                dsl::assign_op_expr(
                                    AssignOp::Add,
                                    block_matrix
                                        .entry(
                                            &Expr::ident("neighbor_rank"),
                                            typed::block_row::<Ax>(u_idx),
                                            typed::block_col::<Ax>(u_idx),
                                        )
                                        .expr,
                                    flux_neg.clone(),
                                ),
                                acc.add_rhs(
                                    u_idx,
                                    flux_pos.clone() * p_own_state.clone()
                                        + flux_neg * p_neigh_state,
                                ),
                            ]);
                            // Boundary face: no valid neighbor. Pin only the
                            // OUTFLOW (diagonal) part — this is exactly the
                            // upwind supersonic-outlet closure that anchors the
                            // extrapolated exit pressure to the interior.
                            let boundary_lin = dsl::block(vec![
                                acc.add_diag(u_idx, flux_pos.clone()),
                                acc.add_rhs(u_idx, flux_pos * p_own_state),
                            ]);
                            body.push(dsl::if_block_expr(
                                !Expr::ident("is_boundary"),
                                interior_lin,
                                Some(boundary_lin),
                            ));
                        }
                    }
                } else {
                    // Reconstruct field at face (scalar convection operator)
                    for component in 0..equation.target.kind().component_count() as u32 {
                        let u_idx = base_offset + component;
                        let field_name = equation.target.name();

                        let flux_val_expr = dsl::array_access_linear(
                            "fluxes",
                            Expr::ident("face_idx"),
                            flux_stride,
                            u_idx,
                        );
                        let flux_val_expr = ale_relative_flux_expr(conv_op, flux_val_expr);
                        body.push(acc.declare_phi(u_idx, flux_val_expr));
                        body.push(dsl::if_block_expr(
                            Expr::ident("owner").ne(Expr::ident("idx")),
                            dsl::block(vec![dsl::assign_op_expr(
                                AssignOp::Sub,
                                acc.phi(u_idx),
                                acc.phi(u_idx) * 2.0,
                            )]),
                            None,
                        ));

                        if conv_op.bounded {
                            body.push(dsl::assign_op_expr(
                                AssignOp::Add,
                                Expr::ident(format!("bounded_sum_phi_{u_idx}")),
                                acc.phi(u_idx),
                            ));
                        }

                        let field_slot = slots
                            .slots
                            .iter()
                            .find(|s| s.name == field_name)
                            .unwrap_or_else(|| {
                                panic!("missing field '{}' in resolved state slots", field_name)
                            });
                        let phi_own = state_component_slot(
                            slots.stride,
                            "state",
                            "idx",
                            field_slot,
                            component,
                        );
                        let phi_neigh = state_component_slot(
                            slots.stride,
                            "state",
                            "other_idx",
                            field_slot,
                            component,
                        );

                        // Advection scheme selection.
                        //
                        // A scheme declared on the term (model math declaration) is baked at
                        // codegen time (only that variant is emitted); otherwise the solver
                        // drives the selection at runtime through `constants.scheme` (an
                        // if/else-if chain — each face computes only the ACTIVE variant).
                        let scheme_src = if conv_op.scheme_declared {
                            super::reconstruction::SchemeSource::Baked(conv_op.scheme)
                        } else {
                            super::reconstruction::SchemeSource::Runtime(
                                Expr::ident("constants").field("scheme"),
                            )
                        };

                        // Reconstruction gradients.
                        //
                        // If a packed `grad_state` buffer exists, use it. Otherwise, fall back to
                        // a simple two-point gradient estimate based on neighbor differences.
                        // This keeps the scheme knob meaningful without requiring a dedicated
                        // gradients kernel for every model.
                        // Helper to find slot offset by field name.
                        let field_offset = slots
                            .slots
                            .iter()
                            .find(|s| s.name == field_name)
                            .map(|s| s.base_offset)
                            .unwrap_or_else(|| {
                                panic!("missing field '{}' in resolved state slots", field_name)
                            });
                        let (grad_own, grad_neigh) = if needs_gradients {
                            let grad_own = dsl::array_access_linear(
                                "grad_state",
                                Expr::ident("idx"),
                                slots.stride,
                                field_offset + component,
                            );
                            let grad_neigh = dsl::array_access_linear(
                                "grad_state",
                                Expr::ident("other_idx"),
                                slots.stride,
                                field_offset + component,
                            );
                            (grad_own, grad_neigh)
                        } else {
                            let diff = phi_neigh.clone() - phi_own.clone();
                            let denom = dsl::max(
                                Expr::ident("dx") * Expr::ident("dx")
                                    + Expr::ident("dy") * Expr::ident("dy"),
                                1e-12,
                            );
                            let g_x = diff.clone() * Expr::ident("dx") / denom.clone();
                            let g_y = diff * Expr::ident("dy") / denom;
                            let grad = dsl::vec2_f32(g_x, g_y);
                            (grad.clone(), grad)
                        };

                        let (rec_stmts, rec) = scalar_reconstruction_stmts(
                            &format!("rec_{u_idx}"),
                            scheme_src,
                            acc.phi(u_idx),
                            phi_own,
                            phi_neigh,
                            grad_own,
                            grad_neigh,
                            super::reconstruction::GeometryPoints {
                                center: Expr::ident("center"),
                                other_center: Expr::ident("other_center"),
                                face_center: Expr::ident("f_center"),
                            },
                        );

                        let flux_pos = dsl::max(acc.phi(u_idx), 0.0);
                        let flux_neg = dsl::min(acc.phi(u_idx), 0.0);

                        let dc_term = acc.phi(u_idx) * (rec.phi_ho - rec.phi_upwind);

                        // The reconstruction locals live at the head of the
                        // interior branch: boundary faces never needed them (the
                        // deferred-correction term is interior-only), so they
                        // skip the reconstruction arithmetic entirely.
                        let mut interior_stmts = rec_stmts;
                        interior_stmts.extend([
                            acc.add_diag(u_idx, flux_pos.clone()),
                            dsl::assign_op_expr(
                                AssignOp::Add,
                                block_matrix
                                    .entry(
                                        &Expr::ident("neighbor_rank"),
                                        typed::block_row::<Ax>(u_idx),
                                        typed::block_col::<Ax>(u_idx),
                                    )
                                    .expr,
                                flux_neg.clone(),
                            ),
                            acc.sub_rhs(u_idx, dc_term),
                        ]);
                        let interior_contrib = dsl::block(interior_stmts);

                        let bc = BcTable::new(Expr::ident("face_idx"), coupled_stride);
                        let (bc_kind_expr, bc_value_expr) = bc.lookup(u_idx);

                        let boundary_contrib = dsl::block(vec![dsl::if_block_expr(
                            bc_kind_expr.eq(GpuBcKind::Dirichlet),
                            dsl::block(vec![
                                acc.add_diag(u_idx, flux_pos),
                                acc.sub_rhs(u_idx, flux_neg * bc_value_expr),
                            ]),
                            Some(dsl::block(vec![acc.add_diag(u_idx, acc.phi(u_idx))])),
                        )]);

                        body.push(dsl::if_block_expr(
                            !Expr::ident("is_boundary"),
                            interior_contrib,
                            Some(boundary_contrib),
                        ));
                    }
                }
            }

            // 3. Gradient
            if let Some(grad_op) = equation
                .ops
                .iter()
                .find(|op| op.kind == DiscreteOpKind::Gradient)
            {
                let field_name = grad_op.field.name();
                let p_offset_opt = offsets.get(field_name);
                let field_slot = slots
                    .slots
                    .iter()
                    .find(|s| s.name == field_name)
                    .unwrap_or_else(|| {
                        panic!("missing field '{}' in resolved state slots", field_name)
                    });
                let phi_own = state_component_slot(slots.stride, "state", "idx", field_slot, 0);
                let phi_neigh =
                    state_component_slot(slots.stride, "state", "other_idx", field_slot, 0);

                let coeff =
                    coefficient_value_expr(slots, grad_op.coeff.as_ref(), "idx", 1.0.into());
                let factor = coeff * 0.5 * Expr::ident("area");

                for component in 0..equation.target.kind().component_count() as u32 {
                    let u_idx = base_offset + component;
                    let n_comp = if component == 0 {
                        Expr::ident("normal").field("x")
                    } else {
                        Expr::ident("normal").field("y")
                    };

                    let term_common = factor.clone() * n_comp;

                    if let Some(&p_idx) = p_offset_opt {
                        if grad_op.discretization == Discretization::Implicit {
                            let diag_block = block_matrix.row_entry(&Expr::ident("diag_rank"));

                            // Owner p contribution -> Diag block (u, p)
                            body.push(dsl::assign_op_expr(
                                AssignOp::Add,
                                diag_block
                                    .entry(
                                        typed::block_row::<Ax>(u_idx),
                                        typed::block_col::<Ax>(p_idx),
                                    )
                                    .expr,
                                term_common.clone(),
                            ));
                            // Neighbor p contribution -> Neighbor block (u, p)
                            body.push(dsl::assign_op_expr(
                                AssignOp::Add,
                                block_matrix
                                    .entry(
                                        &Expr::ident("neighbor_rank"),
                                        typed::block_row::<Ax>(u_idx),
                                        typed::block_col::<Ax>(p_idx),
                                    )
                                    .expr,
                                term_common,
                            ));
                        } else {
                            let val = term_common * (phi_own.clone() + phi_neigh.clone());
                            body.push(acc.sub_rhs(u_idx, val));
                        }
                    } else {
                        let val = term_common * (phi_own.clone() + phi_neigh.clone());
                        body.push(acc.sub_rhs(u_idx, val));
                    }
                }
            }
        }

        body
    };

    stmts.push(dsl::for_loop_expr(
        dsl::for_init_var_expr("k", Expr::ident("start")),
        Expr::ident("k").lt(Expr::ident("end")),
        dsl::for_step_increment_expr(Expr::ident("k")),
        dsl::block(face_loop_body),
    ));

    // Bounded convection: subtract the accumulated continuity defect from
    // the diagonal (LHS gains -(sum_f phi_f) * u_P).
    //
    // ALE (mesh-relative) bounded terms: the subtraction must remove
    // `U_P × (discrete mass residual)`, and on a moving mesh that residual is
    // `ρ·dV/dt + Σ_f φ_rel` (mass in the cell changes because the volume
    // does), with the volume rate taken at the SAME time-scheme weights as
    // the momentum ddt (`ale_dvdt_ddt`, see time_integration.rs). Augment the
    // face-accumulated `Σ_f φ_rel` accordingly. This is what makes a uniform
    // flow an exact fixed point of the moving-mesh momentum equation under
    // both Euler and BDF2: ddt contributes `ρU·dV/dt|_scheme`, upwind
    // convection of a uniform U contributes `U·Σφ_rel`, and the bounded
    // subtraction removes both. With a static mesh (equal volume history)
    // `ale_dvdt_ddt` is exactly `0.0` and the augmentation is the IEEE
    // identity `x + 0.0` (the accumulated sum is never `-0.0`: +0-initialized
    // f32 additions cannot produce it), keeping the zero-flux equivalence
    // gate bitwise.
    for &(u_idx, ale) in &bounded_unknowns {
        if ale {
            stmts.push(dsl::assign_op_expr(
                AssignOp::Add,
                Expr::ident(format!("bounded_sum_phi_{u_idx}")),
                Expr::ident("constants").field("density") * Expr::ident("ale_dvdt_ddt"),
            ));
        }
        stmts.push(acc.sub_diag(u_idx, Expr::ident(format!("bounded_sum_phi_{u_idx}"))));
    }

    // ALE continuity volume source (design S2.3 option b): an equation whose
    // `DivFlux` mass-flux divergence is mesh-relative gains the compensating
    // per-cell volume-change source. Mass balance on a moving cell (constant
    // density): ρ·(V^{n+1}−V^n)/dt + Σ_f φ_rel = 0, so the RHS (which already
    // accumulated `−Σ_f φ_rel` in the face loop) gains `−ρ·ale_dvdt_scl`.
    // The rate is the SCL/BDF1 rate — exactly what the mesh-flux closure
    // guarantees `Σ_f mesh_fluxes` sums to (src/solver/mesh/ale.rs), so at a
    // divergence-free absolute flux the two cancel to f32 roundoff under any
    // time scheme. Static mesh: `ale_dvdt_scl == 0.0` bitwise and
    // `rhs -= ρ·0.0` is the IEEE identity `x - 0.0`.
    for equation in &system.equations {
        let has_ale_div_flux = equation.ops.iter().any(|op| {
            op.kind == DiscreteOpKind::Convection
                && op.discretization == Discretization::Implicit
                && op.term_op == TermOp::DivFlux
                && op.relative_to_mesh
        });
        if !has_ale_div_flux {
            continue;
        }
        assert_eq!(
            equation.target.kind().component_count(),
            1,
            "ALE DivFlux continuity source requires a scalar target equation (got '{}')",
            equation.target.name()
        );
        let base_offset = *offsets
            .get(equation.target.name())
            .expect("missing target offset");
        stmts.push(acc.sub_rhs(
            base_offset,
            Expr::ident("constants").field("density") * Expr::ident("ale_dvdt_scl"),
        ));
    }

    // Write diagonal block and RHS.
    let diag_entry = block_matrix.row_entry(&Expr::ident("diag_rank"));
    stmts.extend(acc.writeback(
        |i| {
            diag_entry
                .entry(typed::block_row::<Ax>(i), typed::block_col::<Ax>(i))
                .expr
        },
        "rhs",
        Expr::ident("idx"),
    ));

    Function::new(
        "main",
        params,
        None,
        vec![
            Attribute::Compute,
            Attribute::WorkgroupSize(UNIFIED_ASSEMBLY_WORKGROUP_SIZE),
        ],
        Block::new(stmts),
    )
}
