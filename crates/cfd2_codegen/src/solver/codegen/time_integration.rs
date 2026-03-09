//! Time integration scheme contributions for coupled assembly kernels.
//!
//! This module decouples time-integration math (BDF1, BDF2, dual-time stepping)
//! from the generic assembler files.  The assembler calls
//! [`emit_ddt_contributions`] which dispatches to the active
//! [`TimeIntegrator`], keeping the assembler free of scheme-specific details.

use super::coupled_common::coefficient_value_expr;
use super::dsl::CoupledAccumulators;
use super::state_access::state_component_slot;
use super::wgsl_ast::{Expr, Stmt};
use super::wgsl_dsl as dsl;
use crate::solver::codegen::ir::{DiscreteOpKind, DiscreteSystem};
use crate::solver::ir::Discretization;
use crate::solver::ir::ports::ResolvedStateSlotsSpec;
use std::collections::HashMap;

fn local_dual_time_scale_setup() -> Vec<Stmt> {
    let dtau_safe = Expr::call_named(
        "max",
        vec![Expr::ident("constants").field("dtau"), Expr::from(1e-12)],
    );

    let mut stmts = vec![
        dsl::let_expr("dtau_safe", dtau_safe.clone()),
        dsl::let_expr("global_dual_time_scale", Expr::ident("vol") / dtau_safe),
        dsl::var_typed_expr(
            "perimeter_sum",
            super::wgsl_ast::Type::F32,
            Some(Expr::from(0.0)),
        ),
    ];

    stmts.push(dsl::for_loop_expr(
        dsl::for_init_var_expr("k", Expr::ident("start")),
        Expr::ident("k").lt(Expr::ident("end")),
        dsl::for_step_increment_expr(Expr::ident("k")),
        dsl::block(vec![
            dsl::let_expr(
                "area",
                dsl::array_access("face_areas", dsl::array_access("cell_faces", Expr::ident("k"))),
            ),
            dsl::assign_expr(
                Expr::ident("perimeter_sum"),
                Expr::ident("perimeter_sum") + Expr::ident("area"),
            ),
        ]),
    ));

    stmts.push(dsl::let_expr(
        "face_metric_scale",
        Expr::call_named(
            "max",
            vec![
                Expr::from(1.0),
                (Expr::ident("perimeter_sum") * Expr::ident("perimeter_sum"))
                    / Expr::call_named(
                        "max",
                        vec![Expr::from(16.0) * Expr::ident("vol"), Expr::from(1e-12)],
                    ),
            ],
        ),
    ));
    stmts.push(dsl::let_expr(
        "dual_time_scale",
        Expr::ident("global_dual_time_scale") * Expr::ident("face_metric_scale"),
    ));

    stmts
}

// ---------------------------------------------------------------------------
// TimeIntegrator trait
// ---------------------------------------------------------------------------

/// Produces AST statements for the time-derivative contribution of a single
/// equation component.
///
/// Implementations encapsulate the specific time-stepping scheme (BDF1, BDF2,
/// dual-time, etc.) so that the generic assembler remains scheme-agnostic.
pub trait TimeIntegrator {
    /// Emit statements for one (equation, component) pair.
    ///
    /// # Arguments
    ///
    /// * `acc`  – coupled accumulator (provides `add_diag`, `add_rhs`, …)
    /// * `u_idx` – packed unknown component index
    /// * `base_coeff` – `vol * rho / dt`
    /// * `dual_time_coeff` – `vol * rho / dtau`
    /// * `phi_n` – solution at previous time level (state_old)
    /// * `phi_nm1` – solution at time level n-1 (state_old_old)
    /// * `phi_iter` – solution at current outer iteration (state_iter, for dual-time)
    fn emit_component(
        &self,
        acc: &CoupledAccumulators,
        u_idx: u32,
        base_coeff: Expr,
        dual_time_coeff: Expr,
        phi_n: Expr,
        phi_nm1: Expr,
        phi_iter: Expr,
    ) -> Vec<Stmt>;
}

// ---------------------------------------------------------------------------
// BdfDualTimeIntegrator  (BDF1 + optional BDF2 + optional dual-time)
// ---------------------------------------------------------------------------

/// The default time integrator: BDF1 with runtime-switchable BDF2 correction
/// and optional pseudo-transient continuation (dual-time stepping via `dtau`).
///
/// This reproduces the original inline logic that was duplicated across both
/// assembler files.
pub struct BdfDualTimeIntegrator;

impl TimeIntegrator for BdfDualTimeIntegrator {
    fn emit_component(
        &self,
        acc: &CoupledAccumulators,
        u_idx: u32,
        base_coeff: Expr,
        dual_time_coeff: Expr,
        phi_n: Expr,
        phi_nm1: Expr,
        phi_iter: Expr,
    ) -> Vec<Stmt> {
        use crate::solver::gpu::enums::TimeScheme;
        use super::dsl::EnumExpr;

        let dt = Expr::ident("constants").field("dt");
        let dt_old = Expr::ident("constants").field("dt_old");
        let dtau = Expr::ident("constants").field("dtau");
        let time_scheme =
            EnumExpr::<TimeScheme>::from_expr(Expr::ident("constants").field("time_scheme"));

        let mut stmts = Vec::new();

        // BDF1 base.
        stmts.push(acc.add_diag(u_idx, base_coeff.clone()));
        stmts.push(acc.add_rhs(u_idx, base_coeff.clone() * phi_n.clone()));

        // Optional BDF2 correction.
        stmts.push(dsl::if_block_expr(
            time_scheme.eq(TimeScheme::BDF2),
            dsl::block(vec![
                dsl::let_expr("r", dt / dt_old),
                dsl::let_expr(
                    "diag_bdf2",
                    base_coeff.clone() * (Expr::ident("r") * 2.0 + 1.0)
                        / (Expr::ident("r") + 1.0),
                ),
                dsl::let_expr("factor_n", Expr::ident("r") + 1.0),
                dsl::let_expr(
                    "factor_nm1",
                    (Expr::ident("r") * Expr::ident("r")) / (Expr::ident("r") + 1.0),
                ),
                acc.set_diag(
                    u_idx,
                    acc.diag(u_idx) - base_coeff.clone() + Expr::ident("diag_bdf2"),
                ),
                acc.set_rhs(
                    u_idx,
                    acc.rhs(u_idx) - base_coeff.clone() * phi_n.clone()
                        + base_coeff.clone()
                            * (Expr::ident("factor_n") * phi_n
                                - Expr::ident("factor_nm1") * phi_nm1),
                ),
            ]),
            None,
        ));

        // Optional pseudo-time continuation (dual-time stepping).
        stmts.push(dsl::if_block_expr(
            dtau.gt(0.0),
            dsl::block(vec![
                acc.add_diag(u_idx, dual_time_coeff.clone()),
                acc.add_rhs(u_idx, dual_time_coeff * phi_iter),
            ]),
            None,
        ));

        stmts
    }
}

// ---------------------------------------------------------------------------
// Public driver: emit_ddt_contributions
// ---------------------------------------------------------------------------

/// Emit AST statements for all implicit time-derivative contributions across
/// all equations in the system, using the given [`TimeIntegrator`].
///
/// This is the primary entry point called by the assembly code generators.
pub fn emit_ddt_contributions(
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    offsets: &HashMap<String, u32>,
    acc: &CoupledAccumulators,
    integrator: &dyn TimeIntegrator,
) -> Vec<Stmt> {
    let mut stmts = local_dual_time_scale_setup();

    for equation in &system.equations {
        let Some(ddt_op) = equation.ops.iter().find(|op| {
            op.kind == DiscreteOpKind::TimeDerivative
                && op.discretization == Discretization::Implicit
        }) else {
            continue;
        };

        let base_offset = *offsets
            .get(equation.target.name())
            .expect("missing target offset");
        let rho_expr = coefficient_value_expr(slots, ddt_op.coeff.as_ref(), "idx", 1.0.into());
        let base_coeff =
            Expr::ident("vol") * rho_expr.clone() / Expr::ident("constants").field("dt");
        let dual_time_coeff = rho_expr * Expr::ident("dual_time_scale");

        for component in 0..equation.target.kind().component_count() as u32 {
            let u_idx = base_offset + component;
            let target_slot = slots
                .slots
                .iter()
                .find(|s| s.name == equation.target.name())
                .unwrap_or_else(|| {
                    panic!(
                        "missing field '{}' in resolved state slots",
                        equation.target.name()
                    )
                });
            let phi_n =
                state_component_slot(slots.stride, "state_old", "idx", target_slot, component);
            let phi_nm1 = state_component_slot(
                slots.stride,
                "state_old_old",
                "idx",
                target_slot,
                component,
            );
            let phi_iter =
                state_component_slot(slots.stride, "state_iter", "idx", target_slot, component);

            stmts.extend(integrator.emit_component(
                acc,
                u_idx,
                base_coeff.clone(),
                dual_time_coeff.clone(),
                phi_n,
                phi_nm1,
                phi_iter,
            ));
        }
    }

    stmts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the trait object dispatch compiles and produces non-empty output.
    #[test]
    fn bdf_dual_time_integrator_emits_stmts() {
        let acc = CoupledAccumulators::new(1);
        let integrator = BdfDualTimeIntegrator;

        let stmts = integrator.emit_component(
            &acc,
            0,
            Expr::lit_f32(1.0),
            Expr::lit_f32(0.5),
            Expr::ident("phi_n"),
            Expr::ident("phi_nm1"),
            Expr::ident("phi_iter"),
        );

        // BDF1 base (2 stmts) + BDF2 if-block (1 stmt) + dtau if-block (1 stmt)
        assert_eq!(stmts.len(), 4, "expected 4 top-level statements from BDF+dual-time integrator");
    }
}
