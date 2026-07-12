use super::bc_table::BcTable;
use super::coupled_common::{
    base_assembly_items, base_assembly_items_structured, coefficient_value_expr, coupled_offsets,
    coupled_unknown_components, kernel_bindings_from_items,
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
use crate::solver::ir::ports::{ParamSpec, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
use crate::solver::ir::{
    Coefficient, Discretization, DispatchDomain, FieldKind, KernelProgram, LaunchSemantics, TermOp,
    TopologyMode,
};

const UNIFIED_ASSEMBLY_WORKGROUP_SIZE: u32 = 64;

fn unified_assembly_needs_fluxes(system: &DiscreteSystem) -> bool {
    system.equations.iter().any(|eq| {
        eq.ops
            .iter()
            .any(|op| op.kind == DiscreteOpKind::Convection)
    })
}

fn validate_unified_assembly_inputs(
    system: &DiscreteSystem,
    needs_fluxes: bool,
    flux_stride: u32,
    coupled_stride: u32,
) {
    let face_channels =
        super::explicit_liveness::ExplicitFaceChannelLiveness::from_discrete_system(system);
    debug_assert_eq!(face_channels.coupled_stride(), coupled_stride);
    let expected = face_channels.storage_stride();
    if needs_fluxes && flux_stride != expected {
        panic!(
            "unified_assembly requires flux_stride ({flux_stride}) == live face-channel stride ({expected}) when convection ops are present (coupled stride {coupled_stride})"
        );
    }
}

/// ALE marker: true when any convection op consumes its face flux relative to
/// the mesh. Gates the ALE storage-binding emission.
fn unified_assembly_needs_mesh_fluxes(system: &DiscreteSystem) -> bool {
    system.is_ale()
}

/// Fail-fast validation of the ALE scope. Two silent mishandlings are rejected
/// at codegen time instead of compiling into GCL-violating kernels:
///
/// * **Explicit flagged terms**: every mesh-relative subtraction site gates on
///   `Discretization::Implicit`, so an explicit `Div` term flagged
///   `relative_to_mesh` would get the bindings / moving-volume ddt /
///   continuity source emitted but NO flux subtraction — an inconsistent ALE
///   discretization.
///
/// * **Variable density**: when the state layout carries a `rho` field the flux
///   is a variable-density mass flux `phi = rho_f*(U·n)A`, so the mesh-relative
///   subtraction must use the SAME face density `rho_f` (not `constants.density`).
///   `ale_relative_flux_expr` reconstructs `rho_f` from the per-cell `rho` state
///   with the EXACT interpolation the model's flux module uses (central Lerp for
///   the barotropic family, upwind blend for the `t_ref` thermal family — both
///   uniform-exact so a uniform free stream is preserved bitwise), keeping the
///   constant-density path byte-identical for incompressible models. This lifts
///   the former constant-density-only scope to the all-Mach compressible
///   (`allmach_*_ale`) models.
fn validate_ale_unified_assembly(system: &DiscreteSystem, _slots: &ResolvedStateSlotsSpec) {
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
}

/// The per-cell Brinkman momentum-penalty mask carried by immersed-boundary
/// (structured IBM) models. Field-name gate — literal to mirror the dp_update
/// Brinkman mask (src/solver/model/modules/rhie_chow.rs) and the flux
/// derivation's min-based face d_p (src/solver/model/flux_derivation.rs);
/// non-IBM models never carry this slot, so every gate keyed on it is
/// byte-inert for them.
const IBM_PENALTY_SLOT: &str = "ibm_penalty_U";

/// Split an implicit-diffusion coefficient into its unique `D_P`-unit scalar
/// field factor (the Rhie–Chow pressure-velocity coupling `d_p`) and the
/// remaining coefficient (`None` when the whole coefficient IS the d_p
/// field). Returns `None` when the tree carries no `D_P` factor (viscous /
/// thermal / biharmonic laplacians) — only the pressure Laplacian
/// `laplacian(rho*d_p, p)` matches.
fn split_dp_coeff_factor(
    coeff: &Coefficient,
) -> Option<(crate::solver::ir::FieldRef, Option<Coefficient>)> {
    match coeff {
        Coefficient::Field(f) if f.unit() == cfd2_ir::units::si::D_P => Some((*f, None)),
        Coefficient::Product(l, r) => {
            if let Some((f, rest_l)) = split_dp_coeff_factor(l) {
                let rest = Some(match rest_l {
                    Some(rl) => Coefficient::Product(Box::new(rl), r.clone()),
                    None => (**r).clone(),
                });
                Some((f, rest))
            } else if let Some((f, rest_r)) = split_dp_coeff_factor(r) {
                let rest = Some(match rest_r {
                    Some(rr) => Coefficient::Product(l.clone(), Box::new(rr)),
                    None => (**l).clone(),
                });
                Some((f, rest))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build the interior-face diffusion coefficient, including the structured-IBM
/// Rhie--Chow conductance rule. Both implicit assembly and the matrix-free
/// explicit residual must call this helper: using different interpolation for
/// `laplacian(rho*d_p,p)` than the flux module uses for the two pressure-bracket
/// terms reintroduces a collocated pressure null mode.
fn interior_diffusion_coefficient(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    kappa_own: Expr,
    kappa_other: Expr,
) -> Expr {
    let interpolated = kappa_own.clone() * Expr::ident("lambda_f")
        + kappa_other.clone() * (Expr::from(1.0) - Expr::ident("lambda_f"));
    let ibm_dp_seal = find_slot(slots, IBM_PENALTY_SLOT).and_then(|pen_slot| {
        coeff
            .and_then(split_dp_coeff_factor)
            .map(|split| (pen_slot, split))
    });
    let Some((pen_slot, (dp_field, rest))) = ibm_dp_seal else {
        return interpolated;
    };

    let dp_slot = find_slot(slots, dp_field.name()).unwrap_or_else(|| {
        panic!(
            "IBM d_p seal: missing '{}' in resolved state slots",
            dp_field.name()
        )
    });
    let dp_own = state_component_slot(slots.stride, "state", "idx", dp_slot, 0);
    let dp_neigh = state_component_slot(slots.stride, "state", "other_idx", dp_slot, 0);
    let dp_min = dsl::min(dp_own, dp_neigh);
    let sealed = match rest {
        Some(rest) => {
            let rest_own = coefficient_value_expr(slots, Some(&rest), "idx", 1.0.into());
            let rest_other = coefficient_value_expr(slots, Some(&rest), "other_idx", 1.0.into());
            (rest_own * Expr::ident("lambda_f")
                + rest_other * (Expr::from(1.0) - Expr::ident("lambda_f")))
                * dp_min
        }
        None => dp_min,
    };

    // Fluid rows use the min-based impermeable-wall conductance. Solid rows
    // retain the unsealed interpolation so their otherwise flux-isolated
    // pressure remains slaved to neighbouring pressure instead of resonating.
    let sp_own_abs = dsl::abs(state_component_slot(
        slots.stride,
        "state",
        "idx",
        pen_slot,
        0,
    ));
    dsl::select(sealed, interpolated, sp_own_abs.gt(0.0))
}

/// `mesh_fluxes` storage binding (group 0 / binding 8): per-face volumetric
/// swept rate `V̇_f = A_swept(f)/dt` (Volume/Time), signed along the stored
/// face normal (owner convention, like the `fluxes` mass flux). Computed
/// host-side from swept-face geometry (SCL by construction), NEVER from a
/// velocity dotted with a normal. Zero-filled when static, so the subtraction
/// vanishes bitwise.
fn mesh_fluxes_item() -> Item {
    storage_var(
        "mesh_fluxes",
        Type::array(Type::F32),
        0,
        8,
        AccessMode::Read,
    )
}

/// Periodic-seam shift that lifts the wrapped neighbour centre into the face
/// owner's coordinate frame. It is zero on ordinary faces/static domains.
fn face_wrap_shift_item() -> Item {
    storage_var(
        "face_wrap_shift",
        Type::array(Type::Custom("Vector2".to_string())),
        0,
        14,
        AccessMode::Read,
    )
}

/// ALE volume-history bindings (group 0 / bindings 9 and 15; binding 14 is
/// `face_wrap_shift` in the flux modules and is left untouched so fused kernels
/// can never collide). `cell_vols_old` = V^n, `cell_vols_old_old` = V^{n-1};
/// both rotated by the ALE step seam (`begin_ale_step`: old_old ← old ←
/// current, BEFORE the new volumes are uploaded) and seeded equal to
/// `cell_vols` by `initialize_history`.
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
/// consistent relative flux. This is the single subtraction point for the
/// whole assembly; every downstream consumer reads the accumulator.
///
/// `rho_f` is the EXACT face density the model's flux module bakes into the
/// convective mass flux — that is the load-bearing INVARIANT of this function
/// (mirrored in `derive_rhie_chow`'s `density_face_expr`,
/// src/solver/model/flux_derivation.rs): `phi` and its mesh-relative
/// subtraction must carry the SAME face density, or every face with a density
/// gradient on a moving mesh gains a spurious mass source
/// ∝ (rho mismatch) × (mesh velocity). Three family cases, matching the flux
/// deriver case-for-case:
///
/// * no `rho` slot (incompressible): the constant `constants.density` —
///   byte-unchanged, preserving the zero-flux equivalence and static gates;
/// * `rho` slot, no `t_ref` (barotropic all-Mach): the flux module's central
///   distance-weighted Lerp, written `rho_other + lambda_f*(rho[idx]-rho[other])`
///   so a uniform density reduces to EXACTLY `rho_other` bitwise (free stream
///   preserved); `other_idx == idx` on a boundary face, so the read is always
///   a valid index;
/// * `rho` AND `t_ref` slots (real-EOS thermal): the flux module's UPWIND
///   blend `0.5*(rho_o+rho_n) + sgn(u_n_rel)*0.5*(rho_o-rho_n)` — precomputed
///   once per face as the `ale_rho_f` local (see
///   `ale_thermal_upwind_face_density_stmts`, emitted in the face-loop head).
///   The former central Lerp here removed a DIFFERENT mass flux than the
///   upwinding flux module added: an O(rho jump)×(mesh velocity) spurious
///   source wherever grad(rho) != 0 on a moving mesh.
///
/// All three are uniform-exact (uniform density ⇒ `rho_f == rho` bitwise), so
/// GCL/free-stream preservation is untouched, and multiply `mesh_fluxes ≡ 0`
/// on a static mesh, so static results are bitwise unchanged.
fn ale_relative_flux_expr(
    conv_op: &crate::solver::codegen::ir::DiscreteOp,
    flux_val_expr: Expr,
    slots: &ResolvedStateSlotsSpec,
) -> Expr {
    if !conv_op.relative_to_mesh {
        return flux_val_expr;
    }
    let rho_f = match slots.slots.iter().find(|s| s.name == "rho") {
        Some(rho_slot) => {
            if ale_thermal_upwind_rho_slot(slots).is_some() {
                // Real-EOS thermal family: the flux module UPWINDS rho_f, so the
                // subtraction must too. The blend is hoisted to one per-face local
                // (every mesh-relative convection site of every equation consumes
                // the same face density, exactly like the flux module writes the
                // same rho_f into every component's flux slot).
                Expr::ident("ale_rho_f")
            } else {
                let rho_idx = state_component_slot(slots.stride, "state", "idx", rho_slot, 0);
                let rho_other =
                    state_component_slot(slots.stride, "state", "other_idx", rho_slot, 0);
                // Distance-weighted (`lambda_f`) face density, MATCHING the flux module's
                // central Lerp for the barotropic family (owner weight `lambda_f =
                // d_neigh/(d_own+d_neigh)`, the same weight the Rhie-Chow d_p face
                // interpolation uses). Written `rho_other + lambda_f*(rho_idx - rho_other)`
                // so a uniform density (`rho_idx == rho_other`) reduces to EXACTLY
                // `rho_other` bitwise — free stream preserved. On a uniform mesh
                // `lambda_f = 0.5` recovers the standard average. `lambda_f` is the
                // per-face weight already emitted into the assembly face loop.
                rho_other.clone() + Expr::ident("lambda_f") * (rho_idx - rho_other)
            }
        }
        None => Expr::ident("constants").field("density"),
    };
    flux_val_expr - rho_f * dsl::array_access("mesh_fluxes", Expr::ident("face_idx"))
}

/// Slots gate for the ALE THERMAL upwind face density: mirrors the flux
/// deriver's marker check (`density_face_expr` upwinds `rho_f` iff the layout
/// carries a scalar Temperature `t_ref` field — the real-EOS thermal family).
/// Returns the `rho` slot when the upwind path is active. Keep this predicate
/// in lockstep with `derive_rhie_chow`: if the two gates diverge, the
/// mesh-relative subtraction reconstructs a different face density than the
/// flux module baked into `phi`.
fn ale_thermal_upwind_rho_slot(slots: &ResolvedStateSlotsSpec) -> Option<&ResolvedStateSlotSpec> {
    use cfd2_ir::dimensions::{Temperature, UnitDimension};
    let rho = find_slot(slots, "rho")?;
    let t_ref_ok = slots.slots.iter().any(|s| {
        s.name == "t_ref"
            && s.kind == crate::solver::ir::ports::PortFieldKind::Scalar
            && s.unit == <Temperature as UnitDimension>::UNIT
    });
    t_ref_ok.then_some(rho)
}

/// Per-face locals for the ALE THERMAL upwind face density `ale_rho_f`: the
/// EXACT mirror — same blend, same distance-weighted velocity Lerp, same
/// mesh-RELATIVE sgn argument, same 1e-12 regularisation — of the face density
/// the derived Rhie–Chow flux kernel bakes into the mass flux for `t_ref`
/// layouts (see `density_face_expr` in src/solver/model/flux_derivation.rs and
/// the INVARIANT documented there and on `ale_relative_flux_expr`).
///
/// Emitted in the face-loop head (after `lambda_f`) so every mesh-relative
/// convection site reads one shared local instead of re-expanding the blend.
///
/// Orientation: the flux module evaluates in OWNER orientation (owner-outward
/// normal, owner-signed `mesh_fluxes`); this face loop runs in `idx`
/// orientation (`normal` already flipped outward from `idx`). The blend is
/// orientation-invariant — negating the normal negates both `u_n_rel` (the
/// mesh flux is re-signed via the `owner == idx` select below) and the
/// `rho_idx - rho_other` half-jump, and their product is exactly the
/// owner-oriented value (IEEE negation is exact) — so the idx-oriented locals
/// reproduce the flux module's face value. Residual sub-ulp differences (the
/// flux module's `lambda_other = 1.0 - lambda` vs this loop's complementary
/// weight, `dot()` contraction) can only perturb `sgn` on faces where the
/// relative normal velocity sits inside f32 rounding noise of zero — where the
/// upwind side is genuinely ambiguous and the eps-regularised sgn is tiny.
///
/// Boundary faces: `other_idx == idx` makes the half-jump exactly 0.0, so
/// `ale_rho_f == rho[idx]` bitwise regardless of sgn — consistent with the flux
/// module, whose boundary neighbor rho collapses to the owner value. Uniform
/// density: avg is exact and the half-jump is 0.0, so `ale_rho_f == rho`
/// bitwise (free stream/GCL preserved). Static mesh: `mesh_fluxes ≡ 0` makes
/// `u_n_rel = u_n - 0.0/area = u_n` bitwise AND the whole subtraction inert.
fn ale_thermal_upwind_face_density_stmts(
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
) -> Vec<Stmt> {
    let rho_slot = ale_thermal_upwind_rho_slot(slots)
        .expect("ale_thermal_upwind_face_density_stmts: gate must be checked by the caller");
    // The advecting velocity: the unique Vector2 coupled target (the momentum
    // equation) — the same field the flux deriver's `u_face` Lerp reads.
    let mut vec_targets = system
        .equations
        .iter()
        .map(|eq| &eq.target)
        .filter(|t| t.kind() == FieldKind::Vector2);
    let u_target = vec_targets
        .next()
        .expect("ALE thermal upwind face density requires a Vector2 momentum target");
    assert!(
        vec_targets.next().is_none(),
        "ALE thermal upwind face density requires a UNIQUE Vector2 target (ambiguous advecting velocity)"
    );
    let u_slot = find_slot(slots, u_target.name()).unwrap_or_else(|| {
        panic!(
            "ALE thermal upwind face density: momentum target '{}' missing from state slots",
            u_target.name()
        )
    });

    let s = |idx: &str, slot, comp| state_component_slot(slots.stride, "state", idx, slot, comp);
    let lambda_f = || Expr::ident("lambda_f");
    let w_other = || Expr::from(1.0) - Expr::ident("lambda_f");

    let mf = dsl::array_access("mesh_fluxes", Expr::ident("face_idx"));
    vec![
        // `mesh_fluxes` is owner-signed; re-sign outward from `idx` like `normal`.
        dsl::let_expr(
            "ale_mesh_flux_out",
            dsl::select(-mf.clone(), mf, Expr::ident("owner").eq(Expr::ident("idx"))),
        ),
        // Distance-weighted face velocity (the flux module's `U` Lerp, idx-oriented).
        dsl::let_expr(
            "ale_u_f_x",
            s("idx", u_slot, 0) * lambda_f() + s("other_idx", u_slot, 0) * w_other(),
        ),
        dsl::let_expr(
            "ale_u_f_y",
            s("idx", u_slot, 1) * lambda_f() + s("other_idx", u_slot, 1) * w_other(),
        ),
        // Mesh-RELATIVE face-normal velocity: the convecting speed under ALE.
        dsl::let_expr(
            "ale_u_n_rel",
            Expr::ident("ale_u_f_x") * Expr::ident("normal").field("x")
                + Expr::ident("ale_u_f_y") * Expr::ident("normal").field("y")
                - Expr::ident("ale_mesh_flux_out") / Expr::ident("area"),
        ),
        // sgn(u_n_rel) = u_n_rel / max(|u_n_rel|, 1e-12) — same regularisation
        // as the flux module (finite for u_n_rel == 0, so `sgn * 0.0` stays 0.0).
        dsl::let_expr(
            "ale_upwind_sgn",
            Expr::ident("ale_u_n_rel") / dsl::max(dsl::abs(Expr::ident("ale_u_n_rel")), 1.0e-12),
        ),
        // rho_f = 0.5*(rho_o+rho_n) + sgn*0.5*(rho_o-rho_n): upwind cell's rho.
        dsl::let_expr(
            "ale_rho_f",
            Expr::from(0.5) * (s("idx", rho_slot, 0) + s("other_idx", rho_slot, 0))
                + Expr::ident("ale_upwind_sgn")
                    * (Expr::from(0.5) * (s("idx", rho_slot, 0) - s("other_idx", rho_slot, 0))),
        ),
    ]
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
            validate_unified_assembly_inputs(
                self.system,
                needs_fluxes,
                self.flux_stride,
                Ax::STRIDE,
            );

            let mut module = Module::new();
            module.push(Item::Comment(
                "GENERATED BY CFD2 CODEGEN (unified_assembly)".to_string(),
            ));
            module.push(Item::Comment("DO NOT EDIT MANUALLY".to_string()));
            if self.system.topology() == TopologyMode::Structured2D {
                module.extend(base_assembly_items_structured(
                    self.needs_gradients,
                    needs_fluxes,
                    self.eos_params,
                ));
            } else {
                module.extend(base_assembly_items(
                    self.needs_gradients,
                    needs_fluxes,
                    self.eos_params,
                ));
                module.push(face_wrap_shift_item());
            }
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
                false,
                None,
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
            validate_unified_assembly_inputs(
                self.system,
                needs_fluxes,
                self.flux_stride,
                Ax::STRIDE,
            );

            let mut items = if self.system.topology() == TopologyMode::Structured2D {
                base_assembly_items_structured(self.needs_gradients, needs_fluxes, self.eos_params)
            } else {
                base_assembly_items(self.needs_gradients, needs_fluxes, self.eos_params)
            };
            if self.system.topology() == TopologyMode::Unstructured {
                items.push(face_wrap_shift_item());
            }
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
                false,
                None,
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

/// Generate a matrix-free method-of-lines residual kernel.
///
/// The returned program binds no global matrix, no block-CSR row metadata and
/// no solution vector.  It evaluates the declared spatial discretization
/// directly into `rhs = S(q) - L(q)` for consumption by an explicit RK stage.
pub fn generate_matrix_free_residual_kernel_program(
    id: &str,
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    flux_stride: u32,
    needs_gradients: bool,
    eos_params: &[ParamSpec],
) -> Result<KernelProgram, String> {
    let explicit_layout = super::explicit_liveness::ExplicitRkLayout::from_discrete_system(system);
    let rhs_projection: Vec<u32> = explicit_layout
        .differential_components()
        .iter()
        .map(|component| component.coupled_rank)
        .collect();
    let residual_system = system.matrix_free_spatial_residual();
    let coupled_stride = coupled_unknown_components(&residual_system).len() as u32;

    struct Dispatch<'a> {
        id: &'a str,
        system: &'a DiscreteSystem,
        slots: &'a ResolvedStateSlotsSpec,
        flux_stride: u32,
        needs_gradients: bool,
        eos_params: &'a [ParamSpec],
        rhs_projection: &'a [u32],
    }
    impl typed::DispatchByStride<Result<KernelProgram, String>> for Dispatch<'_> {
        fn call<Ax: typed::CoupledAxis>(&self) -> Result<KernelProgram, String> {
            let needs_fluxes = unified_assembly_needs_fluxes(self.system);
            validate_unified_assembly_inputs(
                self.system,
                needs_fluxes,
                self.flux_stride,
                Ax::STRIDE,
            );
            let mut items = super::coupled_common::base_matrix_free_residual_items(
                self.system.topology(),
                self.needs_gradients,
                needs_fluxes,
                self.eos_params,
            );
            if self.system.topology() == TopologyMode::Unstructured {
                items.push(face_wrap_shift_item());
            }
            let bindings = kernel_bindings_from_items(&items)?;
            let main = main_assembly_fn::<Ax>(
                self.system,
                self.slots,
                self.flux_stride,
                self.needs_gradients,
                true,
                Some(self.rhs_projection),
            );
            let (launch, consumed_stmts) = launch_from_main_statements(&main.body.stmts)?;
            let mut program = KernelProgram::new(self.id, DispatchDomain::Cells, launch, bindings);
            program.body = main.body.stmts[consumed_stmts..].to_vec();
            program.eos_params = self.eos_params.to_vec();
            Ok(program)
        }
    }

    typed::dispatch_by_coupled_stride(
        coupled_stride,
        Dispatch {
            id,
            system: &residual_system,
            slots,
            flux_stride,
            needs_gradients,
            eos_params,
            rhs_projection: &rhs_projection,
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

/// Emit the STRUCTURED per-face descriptor for direction `k` (the runtime loop
/// variable, 0..4 over South/West/East/North). Binds the SAME named locals the
/// unstructured gather prologue does — `normal` (already outward from `idx`),
/// `area`, `f_center`, `is_boundary`, `other_idx`, `other_center`,
/// `boundary_type`, `face_idx` — but entirely from index arithmetic on the dense
/// Cartesian grid (`grid.nx/ny/dx/dy`; cell `idx = gj*nx + gi`), reading NO
/// connectivity or geometry buffers. Downstream the shared geometry tail
/// (`dx`/`dy`/`dist`/`lambda_f`) and the term-contribution physics consume these
/// locals unchanged, so the structured operator is the identical finite-volume
/// discretisation with an implicit, indirection-free stencil.
///
/// Direction order matches the ascending-column band layout of the 5-point
/// operator (row columns `[idx-nx, idx-1, idx, idx+1, idx+nx]`): `k=0` South,
/// `1` West, `2` East, `3` North; the diagonal is band rank 2, bound in the
/// preamble. `neighbor_rank` (the off-diagonal band slot) is bound by the caller.
fn structured_face_head() -> Vec<Stmt> {
    let k = || Expr::ident("k");
    let id = |s: &str| Expr::ident(s);
    let grid = |f: &str| Expr::ident("grid").field(f);
    let center = |f: &str| Expr::ident("center").field(f);
    let vec2 = |x: Expr, y: Expr| Expr::call_named("Vector2", vec![x, y]);
    // West/East faces have an x-aligned normal; South/North a y-aligned one.
    let axis_is_x = k().ge(1u32) & k().le(2u32);
    // East (k=2) and North (k=3) point in the +axis direction; South/West in -.
    let k_pos = k().ge(2u32);
    vec![
        // Every structured face "belongs" to `idx` with its normal already
        // outward, so `owner == idx` unconditionally — this keeps the shared
        // convection/reconstruction orientation guards (`owner != idx`) inert.
        dsl::let_expr("owner", id("idx")),
        dsl::let_expr("center_frame", id("center")),
        dsl::let_expr("axis_is_x", axis_is_x),
        dsl::let_expr("sign_f", dsl::select(-1.0, 1.0, k_pos.clone())),
        dsl::let_expr(
            "normal",
            vec2(
                dsl::select(0.0, id("sign_f"), id("axis_is_x")),
                dsl::select(id("sign_f"), 0.0, id("axis_is_x")),
            ),
        ),
        // Face area = the OTHER axis' spacing; cell-to-cell spacing = this axis'.
        dsl::let_expr("area", dsl::select(grid("dx"), grid("dy"), id("axis_is_x"))),
        dsl::let_expr(
            "spacing",
            dsl::select(grid("dy"), grid("dx"), id("axis_is_x")),
        ),
        dsl::let_expr("half", Expr::from(0.5) * id("spacing")),
        // Boundary when the face sits on a domain edge (min edge for -dir faces,
        // max edge for +dir faces) along this face's axis.
        dsl::let_expr("coord", dsl::select(id("gj"), id("gi"), id("axis_is_x"))),
        dsl::let_expr("ext", dsl::select(grid("ny"), grid("nx"), id("axis_is_x"))),
        dsl::let_expr(
            "is_boundary",
            dsl::select(
                id("coord").eq(id("ext") - 1u32),
                id("coord").eq(0u32),
                k().lt(2u32),
            ),
        ),
        dsl::let_expr("off_u", dsl::select(grid("nx"), 1u32, id("axis_is_x"))),
        dsl::let_expr(
            "neighbor",
            dsl::select(id("idx") - id("off_u"), id("idx") + id("off_u"), k_pos),
        ),
        dsl::let_expr(
            "other_idx",
            dsl::select(id("neighbor"), id("idx"), id("is_boundary")),
        ),
        dsl::let_expr(
            "f_center",
            vec2(
                center("x") + id("half") * id("normal").field("x"),
                center("y") + id("half") * id("normal").field("y"),
            ),
        ),
        // Interior neighbour is a full cell away; a boundary "ghost" sits on the
        // face (half a cell), giving the standard Dirichlet ghost distance.
        dsl::let_expr(
            "mult",
            dsl::select(id("spacing"), id("half"), id("is_boundary")),
        ),
        dsl::let_expr(
            "other_center",
            vec2(
                center("x") + id("mult") * id("normal").field("x"),
                center("y") + id("mult") * id("normal").field("y"),
            ),
        ),
        // SlipWall (boundary_type==4) is not represented on the structured grid
        // yet; a neutral tag keeps the diffusion/convection physics on its
        // default BC path (Dirichlet/Neumann via the bc table).
        // Per-(cell,direction) face id for the BC table lookup (sized N*4).
        dsl::let_expr("face_idx", id("idx") * 4u32 + k()),
        // Real boundary TYPE per (cell,dir) from the structured `face_boundary`
        // array (0 interior) — full parity with `face_boundary[face_idx]`, so the
        // SlipWall projection (boundary_type==4) works.
        dsl::let_expr(
            "boundary_type",
            dsl::array_access("face_boundary", id("face_idx")),
        ),
    ]
}

fn main_assembly_fn<Ax: typed::CoupledAxis>(
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    flux_stride: u32,
    needs_gradients: bool,
    matrix_free: bool,
    matrix_free_rhs_projection: Option<&[u32]>,
) -> Function {
    let _stride = slots.stride;
    let unknowns = coupled_unknown_components(system);
    let coupled_stride = unknowns.len() as u32;
    let acc = typed::CoupledAccumulators::new(coupled_stride);
    let offsets = coupled_offsets(system);
    let face_channels =
        super::explicit_liveness::ExplicitFaceChannelLiveness::from_discrete_system(system);

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

    let structured = system.topology() == TopologyMode::Structured2D;

    // Launch index (identical for both topologies; structured runs set
    // `constants.stride_x = nx` so `idx = gj*nx + gi`).
    let mut stmts = vec![dsl::let_expr(
        "idx",
        Expr::ident("global_id").field("y") * Expr::ident("constants").field("stride_x")
            + Expr::ident("global_id").field("x"),
    )];

    if structured {
        // Dense Cartesian grid: no cell_vols / cell_centers / scalar_row_offsets
        // buffers. The bound, cell geometry and CSR row layout are all arithmetic
        // on `grid.{nx,ny,dx,dy}`, and the operator is a fixed 5-point band
        // (`matrix_values` sized N*5, diagonal at rank 2). This `if` MUST remain
        // the 2nd statement so `launch_from_main_statements` reads its condition
        // as the dispatch bounds guard.
        let f32_of = |s: &str| Expr::call_named("f32", vec![Expr::ident(s)]);
        stmts.push(dsl::if_block_expr(
            Expr::ident("idx")
                .ge(Expr::ident("grid").field("nx") * Expr::ident("grid").field("ny")),
            dsl::block(vec![Stmt::Return(None)]),
            None,
        ));
        stmts.push(dsl::let_expr(
            "gi",
            Expr::ident("idx").modulo(Expr::ident("grid").field("nx")),
        ));
        stmts.push(dsl::let_expr(
            "gj",
            Expr::ident("idx") / Expr::ident("grid").field("nx"),
        ));
        stmts.push(dsl::let_expr(
            "center",
            Expr::call_named(
                "Vector2",
                vec![
                    (f32_of("gi") + Expr::from(0.5)) * Expr::ident("grid").field("dx"),
                    (f32_of("gj") + Expr::from(0.5)) * Expr::ident("grid").field("dy"),
                ],
            ),
        ));
        stmts.push(dsl::let_expr(
            "vol",
            Expr::ident("grid").field("dx") * Expr::ident("grid").field("dy"),
        ));
        if !matrix_free {
            stmts.push(dsl::let_expr("scalar_offset", Expr::ident("idx") * 5u32));
            stmts.push(dsl::let_expr("diag_rank", Expr::from(2u32)));
            stmts.push(dsl::let_expr("num_neighbors", Expr::from(5u32)));
        }
    } else {
        stmts.push(dsl::if_block_expr(
            Expr::ident("idx").ge(Expr::call_named(
                "arrayLength",
                vec![Expr::ident("cell_vols").addr_of()],
            )),
            dsl::block(vec![Stmt::Return(None)]),
            None,
        ));
        stmts.push(dsl::let_expr(
            "center",
            dsl::array_access("cell_centers", Expr::ident("idx")),
        ));
        stmts.push(dsl::let_expr(
            "vol",
            dsl::array_access("cell_vols", Expr::ident("idx")),
        ));
        stmts.push(dsl::let_expr(
            "start",
            dsl::array_access("cell_face_offsets", Expr::ident("idx")),
        ));
        stmts.push(dsl::let_expr(
            "end",
            dsl::array_access("cell_face_offsets", Expr::ident("idx") + 1u32),
        ));
        if !matrix_free {
            stmts.push(dsl::let_expr(
                "scalar_offset",
                dsl::array_access("scalar_row_offsets", Expr::ident("idx")),
            ));
            stmts.push(dsl::let_expr(
                "diag_rank",
                dsl::array_access("diagonal_indices", Expr::ident("idx"))
                    - Expr::ident("scalar_offset"),
            ));
            stmts.push(dsl::let_expr(
                "num_neighbors",
                dsl::array_access("scalar_row_offsets", Expr::ident("idx") + 1u32)
                    - Expr::ident("scalar_offset"),
            ));
        }
    }
    if !matrix_free {
        stmts.extend(
            acc.declare_start_rows(Expr::ident("scalar_offset"), Expr::ident("num_neighbors")),
        );

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
    }

    // diag_i / rhs_i accumulators
    stmts.extend(acc.declare());

    // Time derivative contributions (implicit only).
    if !matrix_free {
        stmts.extend(super::coupled_common::emit_ddt_contributions(
            system, slots, &offsets, &acc,
        ));
    }

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

            // Viscous-dissipation energy source Phi = tau:grad(U). Read the owner
            // cell's velocity-gradient tensor from grad_state (STATE-OFFSET keyed,
            // like the dev2 term) and add `coeff * Phi_grad * V` to the (scalar)
            // energy row, where
            //   Phi_grad = 2[(du/dx)^2 + (dv/dy)^2 + 0.5(du/dy+dv/dx)^2 - (1/3)(divU)^2]
            // (>= 0, so it always heats). Handled before the generic source paths.
            if source_op.viscous_dissipation {
                if source_op.field.kind() != FieldKind::Vector2 {
                    panic!(
                        "viscous_dissipation requires a Vector2 velocity field (field={field_name})"
                    );
                }
                // grad_state is keyed by the velocity's STATE base offset (slots),
                // NOT the coupled-rank `offsets` map (rank-vs-offset bug class).
                let vel_offset = slots
                    .slots
                    .iter()
                    .find(|s| s.name == field_name)
                    .map(|s| s.base_offset)
                    .unwrap_or_else(|| {
                        panic!(
                            "viscous_dissipation: velocity '{field_name}' missing from state slots"
                        )
                    });
                // Only the grad_state assembly variant binds grad_state. Declaring a
                // viscous_dissipation term forces that variant on (needs_gradients),
                // so the non-grad variant (never scheduled for such a model) simply
                // omits the term and still compiles.
                if needs_gradients {
                    // gx = grad(U_x) = (du/dx, du/dy); gy = grad(U_y) = (dv/dx, dv/dy).
                    let grad_comp = |c: u32| {
                        dsl::array_access_linear(
                            "grad_state",
                            Expr::ident("idx"),
                            slots.stride,
                            vel_offset + c,
                        )
                    };
                    let gx = grad_comp(0);
                    let gy = grad_comp(1);
                    let dudx = gx.clone().field("x");
                    let dudy = gx.field("y");
                    let dvdx = gy.clone().field("x");
                    let dvdy = gy.field("y");
                    let shear = dudy + dvdx;
                    let div_u = dudx.clone() + dvdy.clone();
                    // Phi_grad = 2[ (du/dx)^2 + (dv/dy)^2 + 0.5*shear^2 - (1/3)*divU^2 ].
                    let phi_grad = Expr::lit_f32(2.0)
                        * (dudx.clone() * dudx
                            + dvdy.clone() * dvdy
                            + Expr::lit_f32(0.5) * shear.clone() * shear
                            - Expr::lit_f32(1.0 / 3.0) * div_u.clone() * div_u);
                    let coeff_val =
                        coefficient_value_expr(slots, source_op.coeff.as_ref(), "idx", 0.0.into());
                    stmts.push(acc.add_rhs(base_offset, coeff_val * phi_grad * Expr::ident("vol")));
                }
                continue;
            }

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
                if source_op.explicit_reaction {
                    let coeff =
                        coefficient_value_expr(slots, source_op.coeff.as_ref(), "idx", 0.0.into());
                    let scale = if source_op.static_diag {
                        coeff
                    } else {
                        coeff * Expr::ident("vol")
                    };
                    let source_slot = slots
                        .slots
                        .iter()
                        .find(|s| s.name == field_name)
                        .unwrap_or_else(|| {
                            panic!(
                                "matrix-free reaction source field '{}' is missing from state slots",
                                field_name
                            )
                        });
                    if source_op.field.kind() != equation.target.kind() {
                        panic!(
                            "matrix-free reaction source requires field.kind == target.kind (target={}, field={})",
                            equation.target.name(),
                            field_name
                        );
                    }
                    for component in 0..equation.target.kind().component_count() as u32 {
                        let value = state_component_slot(
                            slots.stride,
                            "state",
                            "idx",
                            source_slot,
                            component,
                        );
                        stmts.push(acc.add_rhs(base_offset + component, scale.clone() * value));
                    }
                    continue;
                }

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
                    let val =
                        coefficient_value_expr(slots, source_op.coeff.as_ref(), "idx", 0.0.into());
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
                    let slot = find_slot(slots, source_field.name()).unwrap_or_else(|| {
                        panic!(
                            "explicit vector source '{}' is not in the state layout",
                            source_field.name()
                        )
                    });
                    for component in 0..target_kind.component_count() as u32 {
                        let u_idx = base_offset + component;
                        let value =
                            state_component_slot(slots.stride, "state", "idx", slot, component);
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
                    let val =
                        coefficient_value_expr(slots, source_op.coeff.as_ref(), "idx", 0.0.into());
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
        let mut body: Vec<Stmt> = Vec::new();
        if structured {
            // Structured Cartesian face descriptor: neighbours, face geometry and
            // the boundary flag all from arithmetic on `grid` (no connectivity
            // reads). Binds the same named locals the unstructured gather does.
            body.extend(structured_face_head());
        } else {
            body.extend(vec![
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
                dsl::let_expr(
                    "wrap_shift",
                    dsl::array_access("face_wrap_shift", Expr::ident("face_idx")),
                ),
                dsl::var_typed_expr(
                    "center_frame",
                    Type::Custom("Vector2".to_string()),
                    Some(Expr::ident("center")),
                ),
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
                        // The face centre is stored in the face owner's frame.
                        // Lift the wrapped neighbour-row centre into that same
                        // frame before computing d and interpolation weights.
                        dsl::assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("center_frame").field("x"),
                            Expr::ident("wrap_shift").field("x"),
                        ),
                        dsl::assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("center_frame").field("y"),
                            Expr::ident("wrap_shift").field("y"),
                        ),
                    ]),
                    None,
                ),
            ]);

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
                // Owner row: lift the wrapped neighbour into the owner's
                // frame. Neighbour row already lifted `center_frame` above and
                // its `other_center` is the unshifted face owner.
                dsl::if_block_expr(
                    Expr::ident("owner").eq(Expr::ident("idx")),
                    dsl::block(vec![
                        dsl::assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("other_center").field("x"),
                            Expr::ident("wrap_shift").field("x"),
                        ),
                        dsl::assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("other_center").field("y"),
                            Expr::ident("wrap_shift").field("y"),
                        ),
                    ]),
                    None,
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
        }

        body.push(dsl::let_expr(
            "dx",
            Expr::ident("other_center").field("x") - Expr::ident("center_frame").field("x"),
        ));
        body.push(dsl::let_expr(
            "dy",
            Expr::ident("other_center").field("y") - Expr::ident("center_frame").field("y"),
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

        // Distance-weighted face interpolation weight (weight of `idx` =
        // d_other / (d_idx + d_other), the standard FV linear weight).
        // Matches the derived Rhie-Chow flux kernel's Lerp convention —
        // the assembly's face coefficients MUST interpolate identically to
        // the flux module's face d_p or the pressure system loses
        // consistency. On uniform meshes lambda = 0.5.
        //
        // Distances are the face-normal PROJECTED cell-to-face distances
        // `|dot(f_center - center, n)|` — the OpenFOAM
        // `surfaceInterpolation::weights()` form and EXACTLY what the flux
        // module computes (`d_own`/`d_neigh` in the flux kernels); `abs` makes
        // the normal's orientation irrelevant. A Euclidean `distance()` (the
        // former form) is identical on orthogonal meshes but diverges at
        // O(skew) on CVT/deformed ALE meshes, so the assembly's kappa/rho_f
        // interpolation carried different weights than the flux it must match.
        body.push(dsl::let_expr(
            "lam_d_own",
            dsl::abs(
                (Expr::ident("f_center").field("x") - Expr::ident("center_frame").field("x"))
                    * Expr::ident("normal").field("x")
                    + (Expr::ident("f_center").field("y") - Expr::ident("center_frame").field("y"))
                        * Expr::ident("normal").field("y"),
            ),
        ));
        body.push(dsl::let_expr(
            "lam_d_neigh",
            dsl::abs(
                (Expr::ident("other_center").field("x") - Expr::ident("f_center").field("x"))
                    * Expr::ident("normal").field("x")
                    + (Expr::ident("other_center").field("y") - Expr::ident("f_center").field("y"))
                        * Expr::ident("normal").field("y"),
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

        // ALE THERMAL upwind face density: per-face locals consumed by
        // `ale_relative_flux_expr` at every mesh-relative convection site (must
        // reproduce the flux module's upwinded rho_f EXACTLY — see the invariant
        // on `ale_relative_flux_expr` / `ale_thermal_upwind_face_density_stmts`).
        if unified_assembly_needs_mesh_fluxes(system)
            && ale_thermal_upwind_rho_slot(slots).is_some()
        {
            assert!(
                !structured,
                "ALE thermal upwind face density is unstructured-only \
                 (no structured ALE model exists; the structured assembly has no mesh_fluxes)"
            );
            body.extend(ale_thermal_upwind_face_density_stmts(system, slots));
        }

        if !matrix_free && structured {
            // Off-diagonal band slot for this direction: S=0, W=1, (diag=2),
            // E=3, N=4 — i.e. `k` for k<2 else `k+1` (skipping the diagonal).
            body.push(dsl::let_expr(
                "neighbor_rank",
                dsl::select(
                    Expr::ident("k") + 1u32,
                    Expr::ident("k"),
                    Expr::ident("k").lt(2u32),
                ),
            ));
        } else if !matrix_free {
            body.push(dsl::let_expr(
                "scalar_mat_idx",
                dsl::array_access("cell_face_matrix_indices", Expr::ident("k")),
            ));
            body.push(dsl::let_expr(
                "neighbor_rank",
                Expr::ident("scalar_mat_idx") - Expr::ident("scalar_offset"),
            ));
        }

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
                // IBM (Brinkman) impermeable-wall seal, matching the flux
                // module's min-based face d_p (flux_derivation.rs `ibm_seal`;
                // the consistency invariant above is why the two MUST change
                // together): when the layout carries the `ibm_penalty_U` mask
                // AND this diffusion coefficient contains the D_P-unit d_p
                // factor (i.e. this is the pressure Laplacian), the interior
                // face value on FLUID rows uses min(d_p_own, d_p_neigh) in
                // place of the distance-weighted d_p blend. The dp_update
                // Brinkman mask collapses d_p to ~0 INSIDE the solid, but an
                // arithmetic face blend at the solid/fluid jump re-admits half
                // the fluid-side conductance, so the pressure equation leaks
                // mass through the immersed wall (interface velocity speckle).
                // A wall is a SERIES conductance — the blocked side must
                // dominate — and min is exactly that. The remaining factor
                // (rho) keeps the existing lambda_f blend. Fluid-interior
                // faces see min(a,a) == a for the uniform ClosedForm d_p, so
                // the fluid operator is unchanged; non-IBM models (no
                // `ibm_penalty_U` slot) emit byte-identical code.
                let kappa_interior = interior_diffusion_coefficient(
                    slots,
                    diff_op.coeff.as_ref(),
                    kappa_own.clone(),
                    kappa_other,
                );
                // Distance-weighted for interior faces; owner value at boundaries.
                let kappa = dsl::select(kappa_own, kappa_interior, !Expr::ident("is_boundary"));

                // Single-op name `diff_coeff_<target>`; disambiguate by field
                // only when an equation carries more than one implicit
                // diffusion op (e.g. viscous + biharmonic).
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
            for diff_op in equation.ops.iter().filter(|op| {
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
                let kappa_interior = interior_diffusion_coefficient(
                    slots,
                    diff_op.coeff.as_ref(),
                    kappa_own.clone(),
                    kappa_other,
                );
                let kappa_face = dsl::select(
                    kappa_own.clone(),
                    kappa_interior,
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

                    let interior_contrib = dsl::block(vec![acc.add_rhs(
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
                        let u_other = dsl::select(
                            phi_own.clone(),
                            u_bc,
                            bc_rho_u_kind.eq(GpuBcKind::Dirichlet),
                        );

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
            // coupled-rank `offsets` map (rank-vs-offset latent bug class).
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
                        let neigh_v = dsl::vec2_f32(neigh.clone().field("x"), neigh.field("y"));
                        (own_v + neigh_v) * 0.5
                    } else {
                        // Two-point fallback (same as the convection
                        // reconstruction fallback). This variant is never
                        // scheduled once the dev2 term forces the gradients
                        // pipeline on, but the kernel must still compile;
                        // at boundary faces it degrades to zero.
                        let phi_own =
                            state_component_slot(slots.stride, "state", "idx", field_slot, c);
                        let phi_neigh =
                            state_component_slot(slots.stride, "state", "other_idx", field_slot, c);
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
            if let Some(conv_op) = equation
                .ops
                .iter()
                .find(|op| op.kind == DiscreteOpKind::Convection)
            {
                // Semantic `u_idx` remains the full coupled unknown index for
                // row accumulation. Face storage uses the independent dense
                // producer/consumer rank, so dead algebraic channels consume
                // no buffer slot. The optional flux-module producer derives
                // this same map from the lowered residual operations.

                // `DivFlux` terms represent conservative flux divergence:
                // the face flux is precomputed (e.g., KT) and should contribute RHS-only:
                //   RHS -= sum_face(sign * flux_face_component)
                // This must not be treated like a scalar convection operator.
                if conv_op.term_op == TermOp::DivFlux {
                    for component in 0..equation.target.kind().component_count() as u32 {
                        let u_idx = base_offset + component;
                        let face_channel = face_channels
                            .storage_rank_for_coupled(u_idx)
                            .expect("DivFlux row must own a live face channel");
                        let flux_val_expr = dsl::array_access_linear(
                            "fluxes",
                            Expr::ident("face_idx"),
                            flux_stride,
                            face_channel,
                        );
                        let flux_val_expr = ale_relative_flux_expr(conv_op, flux_val_expr, slots);

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
                                    panic!("linearize_pressure_flux requires a 'rho' state field")
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
                        let face_channel = face_channels
                            .storage_rank_for_coupled(u_idx)
                            .expect("convection row must own a live face channel");
                        let field_name = equation.target.name();

                        let flux_val_expr = dsl::array_access_linear(
                            "fluxes",
                            Expr::ident("face_idx"),
                            flux_stride,
                            face_channel,
                        );
                        let flux_val_expr = ale_relative_flux_expr(conv_op, flux_val_expr, slots);
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

                        if conv_op.bounded && conv_op.discretization == Discretization::Implicit {
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
                            phi_own.clone(),
                            phi_neigh,
                            grad_own,
                            grad_neigh,
                            super::reconstruction::GeometryPoints {
                                center: Expr::ident("center"),
                                other_center: Expr::ident("other_center"),
                                face_center: Expr::ident("f_center"),
                            },
                        );

                        let bc = BcTable::new(Expr::ident("face_idx"), coupled_stride);
                        let (bc_kind_expr, bc_value_expr) = bc.lookup(u_idx);

                        if conv_op.discretization == Discretization::Explicit {
                            // Direct finite-volume flux residual.  Unlike the
                            // implicit path below there is no upwind matrix plus
                            // deferred correction: the selected reconstructed
                            // face value is consumed in one expression.
                            let mut interior_stmts = rec_stmts;
                            interior_stmts.push(acc.sub_rhs(u_idx, acc.phi(u_idx) * rec.phi_ho));
                            if conv_op.bounded {
                                // bounded div(phi,q) = div(phi q) - div(phi) q
                                interior_stmts
                                    .push(acc.add_rhs(u_idx, acc.phi(u_idx) * phi_own.clone()));
                            }
                            let interior_contrib = dsl::block(interior_stmts);

                            let flux_pos = dsl::max(acc.phi(u_idx), 0.0);
                            let flux_neg = dsl::min(acc.phi(u_idx), 0.0);
                            let mut dirichlet = vec![acc.sub_rhs(
                                u_idx,
                                flux_pos * phi_own.clone() + flux_neg * bc_value_expr,
                            )];
                            let mut extrapolated =
                                vec![acc.sub_rhs(u_idx, acc.phi(u_idx) * phi_own.clone())];
                            if conv_op.bounded {
                                dirichlet
                                    .push(acc.add_rhs(u_idx, acc.phi(u_idx) * phi_own.clone()));
                                extrapolated.push(acc.add_rhs(u_idx, acc.phi(u_idx) * phi_own));
                            }
                            let boundary_contrib = dsl::block(vec![dsl::if_block_expr(
                                bc_kind_expr.eq(GpuBcKind::Dirichlet),
                                dsl::block(dirichlet),
                                Some(dsl::block(extrapolated)),
                            )]);
                            body.push(dsl::if_block_expr(
                                !Expr::ident("is_boundary"),
                                interior_contrib,
                                Some(boundary_contrib),
                            ));
                        } else {
                            let flux_pos = dsl::max(acc.phi(u_idx), 0.0);
                            let flux_neg = dsl::min(acc.phi(u_idx), 0.0);
                            // IBM (Brinkman): kill the deferred high-order
                            // correction on a face touching a penalty cell.
                            // The implicit matrix side remains first-order
                            // upwind and is unchanged.
                            let dc_raw = acc.phi(u_idx) * (rec.phi_ho - rec.phi_upwind);
                            let dc_term = if let Some(pen_slot) = find_slot(slots, IBM_PENALTY_SLOT)
                            {
                                let sp_own = dsl::abs(state_component_slot(
                                    slots.stride,
                                    "state",
                                    "idx",
                                    pen_slot,
                                    0,
                                ));
                                let sp_neigh = dsl::abs(state_component_slot(
                                    slots.stride,
                                    "state",
                                    "other_idx",
                                    pen_slot,
                                    0,
                                ));
                                dc_raw
                                    * dsl::select(
                                        Expr::from(1.0),
                                        Expr::from(0.0),
                                        (sp_own + sp_neigh).gt(0.0),
                                    )
                            } else {
                                dc_raw
                            };

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
                            // BC-aware boundary closure for the explicit
                            // gradient force. At a boundary face
                            // `other_idx == idx`, so the mirror sum
                            // `phi_own + phi_neigh` reduces to `2·p_P` and the
                            // force never sees a Dirichlet/Neumann boundary
                            // value. At a Dirichlet-p (outlet-gauge) face that
                            // FLIPS the sign of the boundary cell's
                            // pressure→velocity feedback (δp_P < 0 ⇒ extra
                            // outward force ⇒ more outflux ⇒ the continuity
                            // row's Dirichlet anchor drives p_P further down)
                            // — the outlet-band checkerboard that blows up on
                            // polygonal (CVT) meshes, whose band-parallel
                            // Rhie-Chow damping is geometrically weaker than a
                            // grid's. Substitute the ghost (face) value at
                            // full weight instead: Dirichlet → bc_value,
                            // Neumann → p_P + g·d_own, ZeroGradient → p_P —
                            // consistent with the RC grad kernels, the flux
                            // module, and the continuity anchor. Interior
                            // faces and ZeroGradient boundaries are bitwise
                            // unchanged (`0.5·(g + g) ≡ 0.5·(2·g)` in IEEE).
                            let bc = BcTable::new(Expr::ident("face_idx"), coupled_stride);
                            let ghost = bc.ghost_value(p_idx, phi_own.clone(), Expr::ident("dist"));
                            body.push(dsl::if_block_expr(
                                !Expr::ident("is_boundary"),
                                dsl::block(vec![acc.sub_rhs(
                                    u_idx,
                                    term_common.clone() * (phi_own.clone() + phi_neigh.clone()),
                                )]),
                                Some(dsl::block(vec![
                                    acc.sub_rhs(u_idx, term_common * (ghost * 2.0))
                                ])),
                            ));
                        }
                    } else {
                        // No coupled-unknown offset ⇒ no bc-table column for
                        // this field; keep the mirror closure (no shipped
                        // model takes the explicit gradient of a non-unknown).
                        let val = term_common * (phi_own.clone() + phi_neigh.clone());
                        body.push(acc.sub_rhs(u_idx, val));
                    }
                }
            }
        }

        body
    };

    // Structured grids visit a fixed 4-neighbour stencil (`k` = 0..4 over
    // S/W/E/N); unstructured grids walk the cell's face list `start..end`.
    let (loop_init, loop_cond) = if structured {
        (
            dsl::for_init_var_expr("k", Expr::from(0u32)),
            Expr::ident("k").lt(Expr::from(4u32)),
        )
    } else {
        (
            dsl::for_init_var_expr("k", Expr::ident("start")),
            Expr::ident("k").lt(Expr::ident("end")),
        )
    };
    stmts.push(dsl::for_loop_expr(
        loop_init,
        loop_cond,
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
    // the momentum ddt (`ale_dvdt_ddt`). Augment the face-accumulated
    // `Σ_f φ_rel` accordingly. This is what makes a uniform
    // flow an exact fixed point of the moving-mesh momentum equation under
    // both Euler and BDF2: ddt contributes `ρU·dV/dt|_scheme`, upwind
    // convection of a uniform U contributes `U·Σφ_rel`, and the bounded
    // subtraction removes both. With a static mesh (equal volume history)
    // `ale_dvdt_ddt` is exactly `0.0` and the augmentation is the IEEE
    // identity `x + 0.0` (the accumulated sum is never `-0.0`: +0-initialized
    // f32 additions cannot produce it), keeping the zero-flux equivalence
    // gate bitwise.
    // Variable-density models: the mass residual's ddt part is `rho_P·dV/dt` at
    // the per-cell density (matching the `ddt(rho,U)` momentum coefficient), so a
    // uniform-velocity but spatially-varying-density field stays a fixed point.
    // Incompressible models (no `rho` slot) keep `constants.density` bitwise.
    let ale_bounded_density = match slots.slots.iter().find(|s| s.name == "rho") {
        Some(rho_slot) => state_component_slot(slots.stride, "state", "idx", rho_slot, 0),
        None => Expr::ident("constants").field("density"),
    };
    for &(u_idx, ale) in &bounded_unknowns {
        if ale {
            stmts.push(dsl::assign_op_expr(
                AssignOp::Add,
                Expr::ident(format!("bounded_sum_phi_{u_idx}")),
                ale_bounded_density.clone() * Expr::ident("ale_dvdt_ddt"),
            ));
        }
        stmts.push(acc.sub_diag(u_idx, Expr::ident(format!("bounded_sum_phi_{u_idx}"))));
    }

    // ALE continuity volume source: an equation whose `DivFlux` mass-flux
    // divergence is mesh-relative gains the compensating per-cell volume-change
    // source. Mass balance on a moving cell:
    // ρ·(V^{n+1}−V^n)/dt + Σ_f φ_rel = 0, so the RHS (which already accumulated
    // `−Σ_f φ_rel` in the face loop) gains `−ρ·ale_dvdt_scl`. The rate is the
    // SCL/BDF1 rate — exactly what the mesh-flux closure guarantees
    // `Σ_f mesh_fluxes` sums to, so at a divergence-free absolute flux the two
    // cancel to f32 roundoff under any time scheme. Static mesh:
    // `ale_dvdt_scl == 0.0` bitwise and `rhs -= ρ·0.0` is the IEEE identity.
    //
    // Density is the PER-CELL `rho` (`ale_bounded_density`, = the `rho` state slot
    // when present, else `constants.density`) — NOT `constants.density` — so it
    // matches the per-cell `rho_f` the mesh-relative `Σ_f φ_rel` carries and the two
    // geometric `rho·dV/dt` terms cancel EXACTLY for any density (not just `ρ≡ρ_ref`).
    // Paired with a `non_conservative_ale` acoustic ddt (which then contributes NO
    // geometric part), this makes the moving-mesh continuity row both 2nd-order and
    // free-stream-preserving for a real (thermal-EOS) variable density. For the
    // incompressible models (no `rho` slot) this is `constants.density`, bitwise the
    // former behaviour.
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
            ale_bounded_density.clone() * Expr::ident("ale_dvdt_scl"),
        ));
    }

    if matrix_free {
        if let Some(projection) = matrix_free_rhs_projection {
            stmts.extend(acc.write_rhs_projection("rhs", Expr::ident("idx"), projection));
        } else {
            stmts.extend(acc.write_rhs("rhs", Expr::ident("idx")));
        }
    } else {
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
    }

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

#[cfg(test)]
mod input_validation_tests {
    use super::validate_unified_assembly_inputs;
    use crate::solver::codegen::ir::lower_system;
    use crate::solver::ir::{
        fvm, surface_scalar_dim, vol_scalar_dim, Equation, EquationSystem, SchemeRegistry,
    };
    use crate::solver::scheme::Scheme;
    use cfd2_ir::dimensions::Dimensionless;

    fn interleaved_flux_system() -> super::DiscreteSystem {
        let q = vol_scalar_dim::<Dimensionless>("q");
        let closure = vol_scalar_dim::<Dimensionless>("closure");
        let r = vol_scalar_dim::<Dimensionless>("r");
        let phi = surface_scalar_dim::<Dimensionless>("phi");
        let mut q_eq = Equation::new(q);
        q_eq.add_term(fvm::div(phi, q));
        let closure_eq = Equation::new(closure);
        let mut r_eq = Equation::new(r);
        r_eq.add_term(fvm::div(phi, r));
        let mut system = EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(closure_eq);
        system.add_equation(r_eq);
        lower_system(&system, &SchemeRegistry::new(Scheme::Upwind)).unwrap()
    }

    #[test]
    #[should_panic(expected = "flux_stride (1) == live face-channel stride (2)")]
    fn rejects_nonzero_but_undersized_flux_stride() {
        validate_unified_assembly_inputs(&interleaved_flux_system(), true, 1, 3);
    }

    #[test]
    fn accepts_exact_or_unused_flux_stride() {
        let system = interleaved_flux_system();
        validate_unified_assembly_inputs(&system, true, 2, 3);
        validate_unified_assembly_inputs(&system, false, 0, 3);
    }
}
