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
use super::coupled_common::{
    coupled_offsets, coupled_unknown_components, kernel_bindings_from_items,
};
use super::primitive_expr::resolve_field_refs;
use super::state_access::state_component_slot;
use super::wgsl_ast::{AccessMode, AssignOp, Expr, Item, Stmt, Type};
use super::wgsl_bindings::{storage_var, uniform_var};
use super::wgsl_dsl as dsl;
use crate::solver::codegen::ir::{DiscreteOpKind, DiscreteSystem};
use crate::solver::ir::ports::{ParamSpec, ResolvedStateSlotsSpec};
use crate::solver::ir::{DispatchDomain, KernelProgram, LaunchSemantics, TopologyMode};
use std::collections::HashSet;

const WORKGROUP_SIZE: u32 = 64;

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

fn safe_pivot(expr: Expr) -> Expr {
    let eps = Expr::lit_f32(1.0e-20);
    let floor = dsl::select(-eps.clone(), eps.clone(), expr.clone().ge(0.0));
    dsl::select(expr.clone(), floor, dsl::abs(expr).lt(eps))
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
    let unknowns = coupled_unknown_components(system);
    let stride = unknowns.len() as u32;
    if stride == 0 {
        return Err("RK4 requires at least one differential unknown".to_string());
    }
    let offsets = coupled_offsets(system);
    let mut algebraic_rows = HashSet::<u32>::new();
    for equation in &system.equations {
        let row_base = offsets[equation.target.name()];
        let has_own_ddt = equation
            .ops
            .iter()
            .any(|op| op.kind == DiscreteOpKind::TimeDerivative && op.field == equation.target);
        if !has_own_ddt {
            for component in 0..equation.target.kind().component_count() as u32 {
                algebraic_rows.insert(row_base + component);
            }
        }
    }

    let items = stage_items(system.topology(), eos_params);
    let bindings = kernel_bindings_from_items(&items)?;
    let idx = Expr::ident("idx");
    let mut body = Vec::<Stmt>::new();

    let volume = match system.topology() {
        TopologyMode::Structured2D => {
            Expr::ident("grid").field("dx") * Expr::ident("grid").field("dy")
        }
        TopologyMode::Unstructured => dsl::array_access("cell_vols", idx.clone()),
    };
    body.push(dsl::let_expr("vol", volume));

    for row in 0..stride {
        for col in 0..stride {
            body.push(dsl::var_typed_expr(
                &mass_name(row, col),
                Type::F32,
                Some(0.0.into()),
            ));
        }
        let rate = if algebraic_rows.contains(&row) {
            0.0.into()
        } else {
            dsl::array_access_linear("rhs", idx.clone(), stride, row) / dsl::max("vol", 1.0e-30)
        };
        body.push(dsl::var_typed_expr(&rate_name(row), Type::F32, Some(rate)));
        // A recoverable algebraic row is not time-integrated. Giving it a
        // local identity mass keeps the unrolled solve nonsingular; the model's
        // explicit primitive closure overwrites the corresponding state slot
        // after every stage.
        if algebraic_rows.contains(&row) {
            body.push(dsl::assign_expr(Expr::ident(mass_name(row, row)), 1.0));
        }
    }

    // Build the local mass block directly from the model's ddt declaration.
    for equation in &system.equations {
        let row_base = *offsets
            .get(equation.target.name())
            .ok_or_else(|| format!("missing coupled offset for '{}'", equation.target.name()))?;
        if algebraic_rows.contains(&row_base) {
            continue;
        }
        for ddt in equation
            .ops
            .iter()
            .filter(|op| op.kind == DiscreteOpKind::TimeDerivative)
        {
            let col_base = *offsets.get(ddt.field.name()).ok_or_else(|| {
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
            for component in 0..row_components {
                body.push(dsl::assign_op_expr(
                    AssignOp::Add,
                    Expr::ident(mass_name(row_base + component, col_base + component)),
                    coeff.clone(),
                ));
            }
        }
    }

    // Unrolled Gaussian elimination.  The explicit-capability gate requires a
    // non-zero own-variable ddt on every row, so pivoting is unnecessary for
    // the shipped diagonal/triangular physical mass blocks.  A signed floor
    // prevents a transient denormal from turning into NaN.
    for pivot in 0..stride {
        let pivot_expr = safe_pivot(Expr::ident(mass_name(pivot, pivot)));
        body.push(dsl::let_expr(&format!("pivot_{pivot}"), pivot_expr));
        for row in (pivot + 1)..stride {
            body.push(dsl::let_expr(
                &format!("factor_{pivot}_{row}"),
                Expr::ident(mass_name(row, pivot)) / Expr::ident(format!("pivot_{pivot}")),
            ));
            for col in pivot..stride {
                body.push(dsl::assign_op_expr(
                    AssignOp::Sub,
                    Expr::ident(mass_name(row, col)),
                    Expr::ident(format!("factor_{pivot}_{row}"))
                        * Expr::ident(mass_name(pivot, col)),
                ));
            }
            body.push(dsl::assign_op_expr(
                AssignOp::Sub,
                Expr::ident(rate_name(row)),
                Expr::ident(format!("factor_{pivot}_{row}")) * Expr::ident(rate_name(pivot)),
            ));
        }
    }
    for row in (0..stride).rev() {
        let mut solved = Expr::ident(rate_name(row));
        for col in (row + 1)..stride {
            solved = solved - Expr::ident(mass_name(row, col)) * Expr::ident(rate_name(col));
        }
        body.push(dsl::assign_expr(
            Expr::ident(rate_name(row)),
            solved / safe_pivot(Expr::ident(mass_name(row, row))),
        ));
    }

    // Classical RK4 low-storage form. `rk_base` and `rk_accum` are packed by
    // differential rank; the full state retains derived/storage fields.
    for (rank, (field, component)) in unknowns.iter().enumerate() {
        let rank = rank as u32;
        let slot = slots
            .slots
            .iter()
            .find(|slot| slot.name == field.name())
            .ok_or_else(|| format!("missing RK4 state slot '{}'", field.name()))?;
        let state = state_component_slot(slots.stride, "state", "idx", slot, *component);
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
