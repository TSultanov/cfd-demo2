//! Matrix-free classical RK4 stage kernels.
//!
//! Spatial residuals are produced by `unified_assembly` with every spatial
//! operator evaluated explicitly.  This module consumes that integrated
//! residual, builds the *local* mass block declared by the model's `ddt`
//! terms, solves it with an unrolled per-cell elimination, and applies one
//! classical RK4 stage.  No global sparse/banded matrix or linear solution
//! vector is present in the kernel interface.

use super::coeff_expr::coeff_cell_expr;
use super::constants::constants_struct;
use super::coupled_common::kernel_bindings_from_items;
use super::explicit_liveness::ExplicitRkLayout;
use super::primitive_expr::resolve_field_refs;
use super::state_access::state_component_slot;
use super::wgsl_ast::{AccessMode, AssignOp, Block, Expr, Item, Stmt, Type};
use super::wgsl_bindings::{storage_var, uniform_var};
use super::wgsl_dsl as dsl;
use crate::solver::codegen::ir::{DiscreteOpKind, DiscreteSystem};
use crate::solver::ir::ports::{ParamSpec, ResolvedStateSlotsSpec};
use crate::solver::ir::{
    Coefficient, DispatchDomain, KernelProgram, LaunchSemantics, TopologyMode,
};

const WORKGROUP_SIZE: u32 = 64;
const MASS_PIVOT_REL_FLOOR: f32 = 1.0e-6;
const MASS_MATCH_QUALITY_FLOOR: f32 = 1.0e-5;
const MASS_EQUILIBRATION_FLOOR: f32 = 1.907_348_6e-6; // 2^-19
const MASS_QUANTUM_AMPLIFICATION_FLOOR: f32 = 2.384_185_8e-7; // 2^-22

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rk4Stage {
    First,
    Second,
    Third,
    Fourth,
}

fn stage_items(topology: TopologyMode, eos_params: &[ParamSpec]) -> Vec<Item> {
    let mut items = match topology {
        TopologyMode::Structured2D => vec![
            Item::Struct(constants_struct(eos_params)),
            Item::Struct(super::coupled_common::structured_grid_struct()),
            uniform_var("grid", Type::Custom("StructuredGrid".to_string()), 0, 0),
        ],
        TopologyMode::Unstructured => vec![
            Item::Struct(constants_struct(eos_params)),
            storage_var("cell_vols", Type::array(Type::F32), 0, 5, AccessMode::Read),
        ],
    };
    items.extend([
        storage_var("state", Type::array(Type::F32), 1, 0, AccessMode::ReadWrite),
        uniform_var("constants", Type::Custom("Constants".to_string()), 1, 3),
        storage_var("rhs", Type::array(Type::F32), 2, 0, AccessMode::Read),
        storage_var(
            "rk_base",
            Type::array(Type::F32),
            2,
            1,
            AccessMode::ReadWrite,
        ),
        storage_var(
            "rk_accum",
            Type::array(Type::F32),
            2,
            2,
            AccessMode::ReadWrite,
        ),
    ]);
    items
}

fn mass_name(row: u32, col: u32) -> String {
    format!("mass_{row}_{col}")
}

fn rate_name(row: u32) -> String {
    format!("rate_{row}")
}

fn scale_name(row: u32) -> String {
    format!("mass_scale_{row}")
}

fn column_scale_name(column: u32) -> String {
    format!("mass_column_scale_{column}")
}

fn component_solution_scale_name(component: usize) -> String {
    format!("mass_solution_scale_{component}")
}

fn component_conditioning_scale_name(component: usize) -> String {
    format!("mass_conditioning_scale_{component}")
}

fn component_volume_conditioning_scale_name(component: usize) -> String {
    format!("mass_volume_conditioning_scale_{component}")
}

fn volume_precision_risk_name(row: u32) -> String {
    format!("volume_precision_risk_{row}")
}

fn safe_pivot(
    expr: Expr,
    column_scale: Option<Expr>,
    extra_invalid: Option<Expr>,
) -> Expr {
    let magnitude = dsl::abs(expr.clone());
    let invalid = dsl::bitcast("f32", 0x7fc0_0000u32);
    let mut is_invalid = magnitude.clone().lt(dsl::bitcast("f32", 1u32))
        | magnitude.clone().gt(dsl::bitcast("f32", 0x7f7f_ffffu32))
        | magnitude.clone().ne(magnitude.clone());
    if let Some(scale) = column_scale {
        let normalized = magnitude / dsl::max(scale, dsl::bitcast("f32", 1u32));
        is_invalid = is_invalid | normalized.lt(Expr::lit_f32(MASS_PIVOT_REL_FLOOR));
    }
    if let Some(extra) = extra_invalid {
        is_invalid = is_invalid | extra;
    }
    dsl::select(expr, invalid, is_invalid)
}

fn safe_scaled_rate(
    value: Expr,
    solution_scale: Expr,
    extra_invalid: Option<Expr>,
) -> Expr {
    let invalid = dsl::bitcast("f32", 0x7fc0_0000u32);
    let magnitude = dsl::abs(value.clone());
    let mut is_invalid = value.clone().ne(0.0)
        & magnitude.clone().lt(dsl::bitcast("f32", 0x0080_0000u32))
        & (magnitude / dsl::max(solution_scale, dsl::bitcast("f32", 1u32)))
            .ge(dsl::bitcast("f32", 0x0080_0000u32));
    if let Some(extra) = extra_invalid {
        is_invalid = is_invalid | extra;
    }
    dsl::select(value, invalid, is_invalid)
}

/// Structural partial pivoting. Codegen emits comparisons only for lower rows
/// whose pivot-column entry can be nonzero, then selects the largest magnitude
/// in the same sequential form as ordinary partial pivoting. Restricting swaps
/// to sub-floor diagonals is not numerically sufficient: a small but nonzero
/// leading entry can make a well-conditioned block catastrophically unstable.
fn append_guarded_row_pivot(
    body: &mut Vec<Stmt>,
    pivot: u32,
    stride: u32,
    support: &mut [Vec<bool>],
) {
    for candidate in (pivot + 1)..stride {
        if !support[candidate as usize][pivot as usize] {
            continue;
        }
        let pivot_entry = Expr::ident(mass_name(pivot, pivot));
        let candidate_entry = Expr::ident(mass_name(candidate, pivot));
        let condition = dsl::abs(candidate_entry).gt(dsl::abs(pivot_entry));
        let mut swaps = Vec::new();
        for column in pivot..stride {
            if !support[pivot as usize][column as usize]
                && !support[candidate as usize][column as usize]
            {
                continue;
            }
            let pivot_value = Expr::ident(mass_name(pivot, column));
            let candidate_value = Expr::ident(mass_name(candidate, column));
            let temporary = format!("pivot_swap_{pivot}_{candidate}_{column}");
            swaps.push(dsl::let_expr(&temporary, pivot_value.clone()));
            swaps.push(dsl::assign_expr(pivot_value, candidate_value.clone()));
            swaps.push(dsl::assign_expr(candidate_value, Expr::ident(temporary)));
        }
        let pivot_rate = Expr::ident(rate_name(pivot));
        let candidate_rate = Expr::ident(rate_name(candidate));
        let temporary_rate = format!("pivot_swap_rate_{pivot}_{candidate}");
        swaps.push(dsl::let_expr(&temporary_rate, pivot_rate.clone()));
        swaps.push(dsl::assign_expr(pivot_rate, candidate_rate.clone()));
        swaps.push(dsl::assign_expr(
            candidate_rate,
            Expr::ident(temporary_rate),
        ));
        body.push(dsl::if_block_expr(condition, Block::new(swaps), None));

        // The swap is data-dependent. Conservatively carry the union of both
        // structural rows into either branch so later fill-in pruning can
        // never erase an entry that exists on one runtime path.
        for column in pivot..stride {
            let live = support[pivot as usize][column as usize]
                || support[candidate as usize][column as usize];
            support[pivot as usize][column as usize] = live;
            support[candidate as usize][column as usize] = live;
        }
    }
}

fn coefficient_may_be_nonzero(coefficient: Option<&Coefficient>) -> bool {
    match coefficient {
        None => true,
        Some(Coefficient::Constant { value, .. }) => (*value as f32) != 0.0,
        Some(Coefficient::Product(lhs, rhs)) => {
            coefficient_may_be_nonzero(Some(lhs)) && coefficient_may_be_nonzero(Some(rhs))
        }
        Some(Coefficient::Field(_) | Coefficient::MagSqr(_)) => true,
    }
}

/// Enforce the structured all-Mach immersed solid as an algebraic velocity
/// constraint at every RK abscissa. This removes the artificial `-1e5 U`
/// Brinkman stiffness from the explicit stability spectrum while preserving
/// its intended limit (U=0 in solid cells). Fluid cells are byte-identical.
fn append_ibm_velocity_projection(body: &mut Vec<Stmt>, slots: &ResolvedStateSlotsSpec) {
    let Some(penalty) = slots.slots.iter().find(|slot| slot.name == "ibm_penalty_U") else {
        return;
    };
    let Some(velocity) = slots.slots.iter().find(|slot| slot.name == "U") else {
        return;
    };
    if velocity.kind.component_count() < 2 {
        return;
    }

    let penalty_value = state_component_slot(slots.stride, "state", "idx", penalty, 0);
    let solid = penalty_value.lt(0.0);
    for component in 0..2 {
        let value = state_component_slot(slots.stride, "state", "idx", velocity, component);
        body.push(dsl::assign_expr(
            value.clone(),
            dsl::select(value, 0.0, solid.clone()),
        ));
    }
}

/// Generate the stage-local algebraic closure used immediately before an
/// explicit residual evaluation. `primitives` is the same topologically
/// ordered closure applied at the end of every RK stage; running it here as
/// well makes stage 1 correct after host seeding/parameter edits and lets
/// gradient-dependent closures observe the gradient of the current stage.
pub fn generate_primitive_recovery_kernel_program(
    id: &str,
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    primitives: &[(u32, Expr)],
    eos_params: &[ParamSpec],
) -> Result<KernelProgram, String> {
    let items = stage_items(system.topology(), eos_params);
    let bindings = kernel_bindings_from_items(&items)?;
    let idx = Expr::ident("idx");
    let mut body = Vec::<Stmt>::with_capacity(primitives.len());

    for (offset, expr) in primitives {
        let value = resolve_field_refs(expr, slots, idx.clone(), "state");
        body.push(dsl::assign_expr(
            dsl::array_access_linear("state", idx.clone(), slots.stride, *offset),
            value,
        ));
    }
    append_ibm_velocity_projection(&mut body, slots);

    let bounds = match system.topology() {
        TopologyMode::Structured2D => "idx >= grid.nx * grid.ny".to_string(),
        TopologyMode::Unstructured => "idx >= arrayLength(&cell_vols)".to_string(),
    };
    let launch = LaunchSemantics::new(
        [WORKGROUP_SIZE, 1, 1],
        "global_id.y * constants.stride_x + global_id.x",
        Some(bounds),
    );
    let mut program = KernelProgram::new(id, DispatchDomain::Cells, launch, bindings);
    program.body = body;
    program.eos_params = eos_params.to_vec();
    Ok(program)
}

/// Generate one classical RK4 stage.  `primitives` must already be ordered so
/// a derived field may depend on an earlier derived field.
pub fn generate_rk4_stage_kernel_program(
    id: &str,
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    primitives: &[(u32, Expr)],
    stage: Rk4Stage,
    eos_params: &[ParamSpec],
) -> Result<KernelProgram, String> {
    generate_rk4_stage_kernel_program_with_mass_floor(
        id,
        system,
        slots,
        primitives,
        stage,
        eos_params,
        MASS_EQUILIBRATION_FLOOR,
    )
}

/// Generate one RK stage with a model-owned f32 mass-equilibration floor.
/// The model layer validates the proof policy before selecting this entry.
pub fn generate_rk4_stage_kernel_program_with_mass_floor(
    id: &str,
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    primitives: &[(u32, Expr)],
    stage: Rk4Stage,
    eos_params: &[ParamSpec],
    mass_equilibration_floor: f32,
) -> Result<KernelProgram, String> {
    let layout = ExplicitRkLayout::from_discrete_system(system);
    let stride = layout.differential_stride();
    if stride == 0 {
        return Err("RK4 requires at least one differential unknown".to_string());
    }

    let items = stage_items(system.topology(), eos_params);
    let bindings = kernel_bindings_from_items(&items)?;
    let idx = Expr::ident("idx");
    let mut body = Vec::<Stmt>::new();
    let mut mass_support = vec![vec![false; stride as usize]; stride as usize];

    let volume = match system.topology() {
        TopologyMode::Structured2D => {
            Expr::ident("grid").field("dx") * Expr::ident("grid").field("dy")
        }
        TopologyMode::Unstructured => dsl::array_access("cell_vols", idx.clone()),
    };
    body.push(dsl::let_expr("vol", volume));

    for component in layout.differential_components() {
        let row = component
            .differential_rank
            .expect("differential component has a compact rank");
        for col in 0..stride {
            body.push(dsl::var_typed_expr(
                &mass_name(row, col),
                Type::F32,
                Some(0.0.into()),
            ));
        }
        let rhs_name = format!("raw_rhs_{row}");
        let initial_rate_name = format!("initial_rate_{row}");
        body.push(dsl::let_expr(
            &rhs_name,
            dsl::array_access_linear("rhs", idx.clone(), stride, row),
        ));
        body.push(dsl::let_expr(
            &initial_rate_name,
            Expr::ident(&rhs_name) / dsl::max("vol", 1.0e-30),
        ));
        body.push(dsl::let_expr(
            &volume_precision_risk_name(row),
            Expr::ident(rhs_name).ne(0.0)
                & dsl::abs(Expr::ident(&initial_rate_name))
                    .lt(dsl::bitcast("f32", 0x0080_0000u32)),
        ));
        body.push(dsl::var_typed_expr(
            &rate_name(row),
            Type::F32,
            Some(Expr::ident(initial_rate_name)),
        ));
    }

    // Build the local differential mass block directly from the model's ddt
    // declaration. Recoverable algebraic rows have zero rate in the incumbent
    // explicit semantics, so columns that point at them make no contribution
    // and are deliberately absent from the compact block. Differential cross-
    // ddt entries remain live and retain their declared row/column ordering.
    for equation in &system.equations {
        let full_row_base = layout
            .coupled_offset(equation.target.name())
            .ok_or_else(|| format!("missing coupled offset for '{}'", equation.target.name()))?;
        let Some(row_base) = layout.differential_rank_for_coupled(full_row_base) else {
            continue;
        };
        for ddt in equation
            .ops
            .iter()
            .filter(|op| op.kind == DiscreteOpKind::TimeDerivative)
        {
            let full_col_base = layout.coupled_offset(ddt.field.name()).ok_or_else(|| {
                format!(
                    "RK4 ddt field '{}' in row '{}' is not a solved unknown",
                    ddt.field.name(),
                    equation.target.name()
                )
            })?;
            let row_components = equation.target.kind().component_count() as u32;
            let col_components = ddt.field.kind().component_count() as u32;
            if row_components != col_components {
                return Err(format!(
                    "RK4 ddt component mismatch in row '{}': target has {}, field '{}' has {}",
                    equation.target.name(),
                    row_components,
                    ddt.field.name(),
                    col_components
                ));
            }
            let coeff = coeff_cell_expr(slots, ddt.coeff.as_ref(), "idx", 1.0.into());
            let coefficient_live = coefficient_may_be_nonzero(ddt.coeff.as_ref());
            for component in 0..row_components {
                let Some(col) = layout.differential_rank_for_coupled(full_col_base + component)
                else {
                    // Algebraic rates are exactly zero and their values are
                    // refreshed by the ordered primitive closure below.
                    continue;
                };
                if !coefficient_live {
                    continue;
                }
                mass_support[(row_base + component) as usize][col as usize] = true;
                body.push(dsl::assign_op_expr(
                    AssignOp::Add,
                    Expr::ident(mass_name(row_base + component, col)),
                    coeff.clone(),
                ));
            }
        }
    }

    // Partition the structural mass graph into independent blocks. A row and
    // column index share a node; any possible off-diagonal entry joins their
    // components. Pivoting cannot cross these components, so conditioning
    // quality must be accumulated per component rather than multiplied across
    // unrelated physics blocks.
    let mut component_for = vec![usize::MAX; stride as usize];
    let mut component_sizes = Vec::<usize>::new();
    for seed in 0..stride as usize {
        if component_for[seed] != usize::MAX {
            continue;
        }
        let component = component_sizes.len();
        let mut stack = vec![seed];
        component_for[seed] = component;
        let mut size = 0usize;
        while let Some(node) = stack.pop() {
            size += 1;
            for other in 0..stride as usize {
                if component_for[other] == usize::MAX
                    && (mass_support[node][other] || mass_support[other][node])
                {
                    component_for[other] = component;
                    stack.push(other);
                }
            }
        }
        component_sizes.push(size);
    }

    // Eliminate coupled blocks in row-normalized coordinates. Scaling both M's
    // row and its RHS preserves M x = b, makes ordinary partial-pivot factors
    // at most one, and prevents finite wrong answers from raw factor overflow
    // or underflow under equation-unit changes. Isolated scalar blocks retain
    // the one-division fast path because no elimination factor exists.
    for row in 0..stride {
        if component_sizes[component_for[row as usize]] == 1 {
            continue;
        }
        let mut scale = Expr::lit_f32(0.0);
        for column in 0..stride {
            if mass_support[row as usize][column as usize] {
                scale = dsl::max(scale, dsl::abs(Expr::ident(mass_name(row, column))));
            }
        }
        body.push(dsl::let_expr(&scale_name(row), scale));
    }
    for row in 0..stride {
        if component_sizes[component_for[row as usize]] == 1 {
            continue;
        }
        let scale = Expr::ident(scale_name(row));
        for column in 0..stride {
            if mass_support[row as usize][column as usize] {
                let mass = Expr::ident(mass_name(row, column));
                let normalized = mass.clone() / scale.clone();
                let lost_precision = mass.clone().ne(0.0)
                    & dsl::abs(normalized.clone())
                        .lt(dsl::bitcast("f32", 0x0080_0000u32));
                body.push(dsl::assign_expr(
                    mass,
                    dsl::select(
                        normalized,
                        dsl::bitcast("f32", 0x7fc0_0000u32),
                        lost_precision,
                    ),
                ));
            }
        }
        let rate = Expr::ident(rate_name(row));
        let normalized_rate = rate.clone() / scale;
        body.push(dsl::assign_expr(
            rate,
            normalized_rate,
        ));
    }

    // Column scales complete the conditioning certificate without changing
    // the unknowns. Coupled rows are already normalized, so this is a max/abs
    // reduction only; scalar components need neither the scale nor its ALU.
    for column in 0..stride {
        if component_sizes[component_for[column as usize]] == 1 {
            continue;
        }
        let mut scale = Expr::lit_f32(0.0);
        for row in 0..stride {
            if mass_support[row as usize][column as usize] {
                scale = dsl::max(scale, dsl::abs(Expr::ident(mass_name(row, column))));
            }
        }
        body.push(dsl::let_expr(&column_scale_name(column), scale));
    }
    let mut component_members = vec![Vec::<u32>::new(); component_sizes.len()];
    for (rank, &component) in component_for.iter().enumerate() {
        component_members[component].push(rank as u32);
    }
    for (component, members) in component_members.iter().enumerate() {
        if members.len() == 1 {
            let row = members[0];
            let mass = dsl::abs(Expr::ident(mass_name(row, row)));
            let material_volume_risk = Expr::ident(volume_precision_risk_name(row))
                & mass
                    .clone()
                    .lt(Expr::lit_f32(MASS_QUANTUM_AMPLIFICATION_FLOOR));
            let rate = Expr::ident(rate_name(row));
            body.push(dsl::assign_expr(
                rate.clone(),
                safe_scaled_rate(rate, mass, Some(material_volume_risk)),
            ));
            continue;
        }
        let mut scale = Expr::lit_f32(f32::MAX);
        for &column in members {
            scale = dsl::min(scale, Expr::ident(column_scale_name(column)));
        }
        body.push(dsl::let_expr(
            &component_solution_scale_name(component),
            scale,
        ));
    }

    // Solve the fully equilibrated coupled block A' y = b', where
    // A' = D_r^-1 A D_c^-1 and y = D_c x. This prevents subnormal matrix
    // columns from reintroducing factor loss after row normalization. The
    // physical rates x are restored after back substitution.
    for row in 0..stride {
        if component_sizes[component_for[row as usize]] == 1 {
            continue;
        }
        for column in 0..stride {
            if mass_support[row as usize][column as usize] {
                let mass = Expr::ident(mass_name(row, column));
                body.push(dsl::assign_expr(
                    mass.clone(),
                    mass / Expr::ident(column_scale_name(column)),
                ));
            }
        }
    }

    // The capability contract limits runtime-dependent connected blocks to
    // rank two. For such a block, compare the emitted determinant with the
    // largest perfect-matching product. The ratio is invariant under arbitrary
    // row *and column* diagonal scaling (therefore under variable units), unlike
    // multiplying row-relative pivots. Constant blocks of larger rank receive
    // an exact build-time equilibrated-rcond audit and need no runtime product.
    for (component, members) in component_members.iter().enumerate() {
        if members.len() != 2 {
            continue;
        }
        let (i, j) = (members[0], members[1]);
        let diagonal =
            Expr::ident(mass_name(i, i)) * Expr::ident(mass_name(j, j));
        let cross = Expr::ident(mass_name(i, j)) * Expr::ident(mass_name(j, i));
        let diagonal_name = format!("mass_match_diagonal_{component}");
        let cross_name = format!("mass_match_cross_{component}");
        body.push(dsl::let_expr(&diagonal_name, diagonal));
        body.push(dsl::let_expr(&cross_name, cross));
        let diagonal = Expr::ident(diagonal_name);
        let cross = Expr::ident(cross_name);
        let denominator = dsl::max(
            dsl::max(dsl::abs(diagonal.clone()), dsl::abs(cross.clone())),
            dsl::bitcast("f32", 1u32),
        );
        let raw_quality_name = format!("mass_match_raw_quality_{component}");
        body.push(dsl::let_expr(
            &raw_quality_name,
            dsl::abs(diagonal - cross) / denominator,
        ));
        let raw_quality = Expr::ident(raw_quality_name);
        let quality = dsl::min(raw_quality.clone(), Expr::lit_f32(1.0));
        body.push(dsl::let_expr(
            &format!("mass_match_quality_{component}"),
            quality,
        ));
        body.push(dsl::let_expr(
            &format!("mass_quality_bad_{component}"),
            raw_quality
                .clone()
                .lt(Expr::lit_f32(MASS_MATCH_QUALITY_FLOOR))
                | raw_quality.clone().ne(raw_quality),
        ));
    }

    // For a 2x2 equilibrated block, capped q_match bounds inverse
    // amplification. The unconditional 2^-19 floor limits amplification of
    // ordinary normal row-division roundoff (and is the strongest audited
    // threshold stable through the shipped nozzle trajectory); the separate 2^-22 test
    // catches a volume division that loses a nonzero f32 quantum before row
    // scaling. Larger constant blocks conservatively use their build-time
    // rcond floor. The one-quantum argument assumes retained subnormals; FTZ
    // backends are outside that extreme-scale certificate, while shipped GUI
    // states remain many orders above the denormal boundary.
    for (component, members) in component_members.iter().enumerate() {
        if members.len() == 1 {
            continue;
        }
        let quality = if members.len() == 2 {
            Expr::ident(format!("mass_match_quality_{component}"))
        } else {
            Expr::lit_f32(MASS_MATCH_QUALITY_FLOOR)
        };
        let conditioning_scale =
            Expr::ident(component_solution_scale_name(component)) * quality;
        body.push(dsl::let_expr(
            &component_conditioning_scale_name(component),
            conditioning_scale,
        ));
        let mut minimum_row_scale = Expr::lit_f32(f32::MAX);
        let mut volume_precision_risk = Expr::lit_bool(false);
        for &row in members {
            minimum_row_scale = dsl::min(minimum_row_scale, Expr::ident(scale_name(row)));
            volume_precision_risk =
                volume_precision_risk | Expr::ident(volume_precision_risk_name(row));
        }
        body.push(dsl::let_expr(
            &component_volume_conditioning_scale_name(component),
            minimum_row_scale * Expr::ident(component_conditioning_scale_name(component)),
        ));
        let conditioning_unsafe = Expr::ident(component_conditioning_scale_name(component))
            .lt(Expr::lit_f32(mass_equilibration_floor));
        let material_risk = conditioning_unsafe.clone()
            | (volume_precision_risk
                & Expr::ident(component_volume_conditioning_scale_name(component))
                    .lt(Expr::lit_f32(MASS_QUANTUM_AMPLIFICATION_FLOOR)));
        for &row in members {
            let rate = Expr::ident(rate_name(row));
            body.push(dsl::assign_expr(
                rate.clone(),
                dsl::select(
                    rate,
                    dsl::bitcast("f32", 0x7fc0_0000u32),
                    material_risk.clone(),
                ),
            ));
        }
    }

    // Unrolled Gaussian elimination. The capability gate rejects a mass block
    // whose closure-aware symbolic determinant is identically zero. Structural
    // partial pivoting gives stable finite solves without paying comparisons
    // against entries proven zero. A truly singular or denormal runtime block
    // receives a quiet NaN pivot so the existing non-finite state gate stops
    // the step; silently flooring it would manufacture a finite wrong rate.
    for pivot in 0..stride {
        append_guarded_row_pivot(&mut body, pivot, stride, &mut mass_support);
        let raw_pivot = Expr::ident(mass_name(pivot, pivot));
        let component = component_for[pivot as usize];
        let column_scale = (component_sizes[component] > 1)
            .then(|| Expr::lit_f32(1.0));
        let pivot_expr = safe_pivot(raw_pivot, column_scale, None);
        body.push(dsl::let_expr(&format!("pivot_{pivot}"), pivot_expr));
        for row in (pivot + 1)..stride {
            if !mass_support[row as usize][pivot as usize] {
                continue;
            }
            body.push(dsl::let_expr(
                &format!("factor_{pivot}_{row}"),
                Expr::ident(mass_name(row, pivot)) / Expr::ident(format!("pivot_{pivot}")),
            ));
            for col in pivot..stride {
                if !mass_support[pivot as usize][col as usize] {
                    continue;
                }
                body.push(dsl::assign_op_expr(
                    AssignOp::Sub,
                    Expr::ident(mass_name(row, col)),
                    Expr::ident(format!("factor_{pivot}_{row}"))
                        * Expr::ident(mass_name(pivot, col)),
                ));
                if col != pivot {
                    mass_support[row as usize][col as usize] = true;
                }
            }
            mass_support[row as usize][pivot as usize] = false;
            let correction_name = format!("rate_correction_{pivot}_{row}");
            let correction = Expr::ident(format!("factor_{pivot}_{row}"))
                * Expr::ident(rate_name(pivot));
            body.push(dsl::let_expr(&correction_name, correction));
            let updated = Expr::ident(rate_name(row)) - Expr::ident(correction_name);
            body.push(dsl::assign_expr(
                Expr::ident(rate_name(row)),
                updated,
            ));
        }
    }
    for row in (0..stride).rev() {
        let solved_name = format!("backsolve_rate_{row}");
        body.push(dsl::var_typed_expr(
            &solved_name,
            Type::F32,
            Some(Expr::ident(rate_name(row))),
        ));
        let component = component_for[row as usize];
        for col in (row + 1)..stride {
            if !mass_support[row as usize][col as usize] {
                continue;
            }
            let correction_name = format!("backsolve_correction_{row}_{col}");
            body.push(dsl::let_expr(
                &correction_name,
                Expr::ident(mass_name(row, col)) * Expr::ident(rate_name(col)),
            ));
            let updated = Expr::ident(&solved_name) - Expr::ident(correction_name);
            body.push(dsl::assign_expr(
                Expr::ident(&solved_name),
                updated,
            ));
        }
        let component_bad = (component_sizes[component] == 2)
            .then(|| Expr::ident(format!("mass_quality_bad_{component}")));
        body.push(dsl::assign_expr(
            Expr::ident(rate_name(row)),
            Expr::ident(solved_name)
                / safe_pivot(
                    Expr::ident(mass_name(row, row)),
                    (component_sizes[component] > 1)
                        .then(|| Expr::lit_f32(1.0)),
                    component_bad,
                ),
        ));
    }
    for row in 0..stride {
        if component_sizes[component_for[row as usize]] == 1 {
            continue;
        }
        let scaled_rate = Expr::ident(rate_name(row));
        body.push(dsl::assign_expr(
            scaled_rate.clone(),
            scaled_rate / Expr::ident(column_scale_name(row)),
        ));
    }

    // Classical RK4 low-storage form. `rk_base` and `rk_accum` contain only
    // differential components; full state retains algebraic/derived storage.
    for component in layout.differential_components() {
        let rank = component
            .differential_rank
            .expect("differential component has a compact rank");
        let slot = slots
            .slots
            .iter()
            .find(|slot| slot.name == component.field.name())
            .ok_or_else(|| format!("missing RK4 state slot '{}'", component.field.name()))?;
        let state = state_component_slot(slots.stride, "state", "idx", slot, component.component);
        let base = dsl::array_access_linear("rk_base", idx.clone(), stride, rank);
        let accum = dsl::array_access_linear("rk_accum", idx.clone(), stride, rank);
        let rate = Expr::ident(rate_name(rank));
        let dt = Expr::ident("constants").field("dt");
        match stage {
            Rk4Stage::First => {
                body.push(dsl::assign_expr(base.clone(), state.clone()));
                body.push(dsl::assign_expr(accum, rate.clone() / 6.0));
                body.push(dsl::assign_expr(state, base + dt * 0.5 * rate));
            }
            Rk4Stage::Second => {
                body.push(dsl::assign_op_expr(
                    AssignOp::Add,
                    accum,
                    rate.clone() / 3.0,
                ));
                body.push(dsl::assign_expr(state, base + dt * 0.5 * rate));
            }
            Rk4Stage::Third => {
                body.push(dsl::assign_op_expr(
                    AssignOp::Add,
                    accum,
                    rate.clone() / 3.0,
                ));
                body.push(dsl::assign_expr(state, base + dt * rate));
            }
            Rk4Stage::Fourth => {
                body.push(dsl::assign_expr(state, base + dt * (accum + rate / 6.0)));
            }
        }
    }

    // Refresh model-declared derived primitives before the next residual.
    for (offset, expr) in primitives {
        let value = resolve_field_refs(expr, slots, idx.clone(), "state");
        body.push(dsl::assign_expr(
            dsl::array_access_linear("state", idx.clone(), slots.stride, *offset),
            value,
        ));
    }
    append_ibm_velocity_projection(&mut body, slots);

    let bounds = match system.topology() {
        TopologyMode::Structured2D => "idx >= grid.nx * grid.ny".to_string(),
        TopologyMode::Unstructured => "idx >= arrayLength(&cell_vols)".to_string(),
    };
    let launch = LaunchSemantics::new(
        [WORKGROUP_SIZE, 1, 1],
        "global_id.y * constants.stride_x + global_id.x",
        Some(bounds),
    );
    let mut program = KernelProgram::new(id, DispatchDomain::Cells, launch, bindings);
    program.body = body;
    program.eos_params = eos_params.to_vec();
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::fusion::lower_kernel_program_to_wgsl;
    use crate::solver::codegen::ir::lower_system;
    use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
    use crate::solver::ir::{fvm, vol_scalar_dim, Equation, EquationSystem, SchemeRegistry};
    use crate::solver::scheme::Scheme;
    use cfd2_ir::dimensions::Dimensionless;

    #[test]
    fn stage_uses_compact_differential_rhs_and_history() {
        let q = vol_scalar_dim::<Dimensionless>("q");
        let closure = vol_scalar_dim::<Dimensionless>("closure");
        let r = vol_scalar_dim::<Dimensionless>("r");

        let mut q_eq = Equation::new(q);
        q_eq.add_term(fvm::ddt(q));
        q_eq.add_term(fvm::ddt(r));
        let closure_eq = Equation::new(closure);
        let mut r_eq = Equation::new(r);
        r_eq.add_term(fvm::ddt(r));
        r_eq.add_term(fvm::ddt(closure));

        let mut source = EquationSystem::new();
        source.add_equation(q_eq);
        source.add_equation(closure_eq);
        source.add_equation(r_eq);
        let discrete = lower_system(&source, &SchemeRegistry::new(Scheme::Upwind)).unwrap();

        let slots = ResolvedStateSlotsSpec {
            stride: 3,
            slots: vec![
                ResolvedStateSlotSpec {
                    name: "q".into(),
                    kind: PortFieldKind::Scalar,
                    unit: q.unit(),
                    base_offset: 0,
                },
                ResolvedStateSlotSpec {
                    name: "closure".into(),
                    kind: PortFieldKind::Scalar,
                    unit: closure.unit(),
                    base_offset: 1,
                },
                ResolvedStateSlotSpec {
                    name: "r".into(),
                    kind: PortFieldKind::Scalar,
                    unit: r.unit(),
                    base_offset: 2,
                },
            ],
        };
        let primitives = vec![(1, Expr::ident("q") + Expr::ident("r"))];
        let residual_program =
            super::super::unified_assembly::generate_matrix_free_residual_kernel_program(
                "mixed_residual",
                &discrete,
                &slots,
                0,
                false,
                &[],
            )
            .unwrap();
        let residual_wgsl = lower_kernel_program_to_wgsl(&residual_program)
            .unwrap()
            .to_wgsl();
        let program = generate_rk4_stage_kernel_program(
            "mixed_stage",
            &discrete,
            &slots,
            &primitives,
            Rk4Stage::First,
            &[],
        )
        .unwrap();
        let wgsl = lower_kernel_program_to_wgsl(&program).unwrap().to_wgsl();

        assert!(residual_wgsl.contains("rhs[idx * 2u + 0u] = rhs_0"));
        assert!(residual_wgsl.contains("rhs[idx * 2u + 1u] = rhs_2"));
        assert!(!residual_wgsl.contains("rhs[idx * 3u"));
        assert!(wgsl.contains("rhs[idx * 2u + 0u]"));
        assert!(wgsl.contains("rhs[idx * 2u + 1u]"));
        assert!(!wgsl.contains("rhs[idx * 3u"));
        assert!(wgsl.contains("rk_base[idx * 2u + 0u]"));
        assert!(wgsl.contains("rk_base[idx * 2u + 1u]"));
        assert!(!wgsl.contains("rk_base[idx * 3u"));
        assert!(
            wgsl.contains("mass_0_1 += 1.0"),
            "differential cross-ddt must survive compaction:\n{wgsl}"
        );
        assert!(
            !wgsl.contains("mass_1_2"),
            "algebraic ddt columns have zero rate and must not enter the compact block"
        );
        assert!(
            wgsl.contains("state[idx * 3u + 1u]"),
            "algebraic closure must remain materialized in full state"
        );
    }

    #[test]
    fn stage_emits_guarded_row_swap_for_invertible_zero_pivot_block() {
        let q = vol_scalar_dim::<Dimensionless>("q");
        let r = vol_scalar_dim::<Dimensionless>("r");
        let s = vol_scalar_dim::<Dimensionless>("s");
        let mut q_eq = Equation::new(q);
        q_eq.add_term(fvm::ddt(q));
        q_eq.add_term(fvm::ddt(r));
        let mut r_eq = Equation::new(r);
        r_eq.add_term(fvm::ddt(q));
        r_eq.add_term(fvm::ddt(r));
        r_eq.add_term(fvm::ddt(s));
        let mut s_eq = Equation::new(s);
        s_eq.add_term(fvm::ddt(r));
        s_eq.add_term(fvm::ddt(s));
        let mut source = EquationSystem::new();
        source.add_equation(q_eq);
        source.add_equation(r_eq);
        source.add_equation(s_eq);
        let discrete = lower_system(&source, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
        let slots = ResolvedStateSlotsSpec {
            stride: 3,
            slots: [q, r, s]
                .into_iter()
                .enumerate()
                .map(|(offset, field)| ResolvedStateSlotSpec {
                    name: field.name().into(),
                    kind: PortFieldKind::Scalar,
                    unit: field.unit(),
                    base_offset: offset as u32,
                })
                .collect(),
        };
        let program = generate_rk4_stage_kernel_program(
            "pivot_stage",
            &discrete,
            &slots,
            &[],
            Rk4Stage::First,
            &[],
        )
        .unwrap();
        let wgsl = lower_kernel_program_to_wgsl(&program).unwrap().to_wgsl();
        assert!(
            wgsl.contains("pivot_swap_1_2_1"),
            "stage lacks row-swap temporaries for the zero second pivot:\n{wgsl}"
        );
        assert!(
            wgsl.contains("abs(mass_2_1) > abs(mass_1_1)"),
            "stage does not select the largest pivot after row normalization:\n{wgsl}"
        );
        assert!(
            wgsl.contains("mass_column_scale_1") && wgsl.contains("< 0.000001"),
            "stage lacks an equilibrated ill-conditioning cutoff:\n{wgsl}"
        );
        assert!(
            wgsl.contains("bitcast<f32>(2143289344u)"),
            "singular runtime mass block is not routed to the non-finite gate:\n{wgsl}"
        );
    }

    #[test]
    fn stage_prunes_structurally_zero_mass_elimination_edges() {
        let q = vol_scalar_dim::<Dimensionless>("q");
        let r = vol_scalar_dim::<Dimensionless>("r");
        let s = vol_scalar_dim::<Dimensionless>("s");
        let mut source = EquationSystem::new();
        for field in [q, r, s] {
            let mut equation = Equation::new(field);
            equation.add_term(fvm::ddt(field));
            source.add_equation(equation);
        }
        let discrete = lower_system(&source, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
        let slots = ResolvedStateSlotsSpec {
            stride: 3,
            slots: [q, r, s]
                .into_iter()
                .enumerate()
                .map(|(offset, field)| ResolvedStateSlotSpec {
                    name: field.name().into(),
                    kind: PortFieldKind::Scalar,
                    unit: field.unit(),
                    base_offset: offset as u32,
                })
                .collect(),
        };
        let program = generate_rk4_stage_kernel_program(
            "diagonal_stage",
            &discrete,
            &slots,
            &[],
            Rk4Stage::First,
            &[],
        )
        .unwrap();
        let wgsl = lower_kernel_program_to_wgsl(&program).unwrap().to_wgsl();
        assert!(
            !wgsl.contains("factor_") && !wgsl.contains("pivot_swap_"),
            "diagonal mass graph emitted impossible elimination edges:\n{wgsl}"
        );
        assert!(wgsl.contains("rate_0 / select(mass_0_0"));
        assert!(wgsl.contains("rate_1 / select(mass_1_1"));
        assert!(wgsl.contains("rate_2 / select(mass_2_2"));
    }
}
