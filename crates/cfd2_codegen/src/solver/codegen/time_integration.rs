//! Time integration scheme contributions for coupled assembly kernels.
//!
//! This module decouples time-integration math (BDF1, BDF2, dual-time stepping)
//! from the generic assembler files.  The assembler calls
//! [`emit_ddt_contributions`] which dispatches to the active
//! [`TimeIntegrator`], keeping the assembler free of scheme-specific details.

use super::coupled_common::coefficient_value_expr;
use super::dsl::CoupledAccumulators;
use super::state_access::state_component_slot;
use super::wgsl_ast::{AssignOp, Expr, Stmt};
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

/// ALE (moving-mesh) locals for the moving-volume ddt and the volume-change
/// rates consumed by the bounded-correction augmentation and the continuity
/// volume source (unified_assembly.rs). Emitted at the head of the ddt
/// contributions — i.e. into the top-level scope of the assembly `main`, so
/// every later statement (face loop, bounded finalization, writeback) can
/// reference them — **only for ALE systems** (`DiscreteSystem::is_ale`);
/// non-ALE lowering emits nothing and stays byte-identical.
///
/// Definitions (all f32, from the `cell_vols` / `cell_vols_old{,_old}`
/// buffers rotated by the ALE step seam — `begin_ale_step`):
///
/// * `ale_vol_ratio_n = V^n / V^{n+1}`, `ale_vol_ratio_nm1 = V^{n-1} / V^{n+1}`:
///   history-state weights for the moving-volume ddt. The lowering multiplies
///   `phi^n` / `phi^{n-1}` by these ratios and then applies the UNCHANGED
///   static integrator whose coefficient is `base_coeff = V^{n+1}·c/dt`:
///
///     BDF1:  diag += V^{n+1}c/dt,   rhs += V^{n+1}c/dt · (V^n/V^{n+1})·φⁿ
///                                        = V^n c/dt · φⁿ            (+rounding)
///     BDF2:  rhs += V^{n+1}c/dt · (factor_n·(V^n/V^{n+1})φⁿ
///                                  − factor_nm1·(V^{n−1}/V^{n+1})φⁿ⁻¹)
///
///   which is exactly the variable-dt BDF2 stencil applied to the CONSERVED
///   PRODUCT V·φ:  d(Vφ)/dt ≈ [a(Vφ)^{n+1} − b(Vφ)^n + c(Vφ)^{n−1}]/dt with
///   a=(2r+1)/(r+1), b=r+1, c=r²/(r+1), r=dt/dt_old — the same Newton
///   backward-difference weights as the static form (the stencil is a property
///   of the time levels; on a moving mesh it acts on V·φ, which is what the
///   FV cell integral conserves). The ratio formulation (instead of separate
///   `V^n·c/dt` coefficients) is deliberate: with equal volumes the ratios
///   are exactly `1.0` (pinned by the select guard below), `1.0*φ` is
///   bitwise `φ`, and under strict IEEE evaluation the emitted arithmetic
///   reduces bit-for-bit to the static ddt — the zero-flux equivalence gate
///   (tests/ale_zero_flux_equivalence_test.rs) verifies this bitwise on both
///   CPU engines (Euler AND BDF2) and on GPU/Euler. GPU/BDF2 is the one
///   documented exception: Metal fast math reassociates the two textually
///   different rhs chains differently (~1 ulp per assembly even at ratio ==
///   1.0; isolated by tests/ale_metal_fastmath_evidence.rs), so that leg is
///   tolerance-gated instead.
///
/// * `ale_dvdt_scl = (V^{n+1} − V^n)/dt`: the SCL (per-step swept) volume
///   rate. This is the rate the mesh-flux closure targets
///   (`Σ_f mesh_fluxes[f] == (V^{n+1}−V^n)/dt` per cell, src/solver/mesh/ale.rs),
///   so the continuity volume source `+ρ·ale_dvdt_scl` cancels the mesh-flux
///   part of `Σ_f φ_rel` EXACTLY (to f32 roundoff) under any time scheme.
///
/// * `ale_dvdt_ddt`: the volume rate at the **ddt scheme's weights**,
///   runtime-switched exactly like the ddt itself (Euler ⇒ `ale_dvdt_scl`;
///   BDF2 ⇒ `[aV^{n+1} − bV^n + cV^{n−1}]/dt`). Emitted in the INCREMENTAL
///   form `a·(V^{n+1}−V^n)/dt − c·(V^n−V^{n−1})/dt` (algebraically equal since
///   a−b+c=0) so equal volumes give differences of exactly `0.0` and the rate
///   vanishes bitwise. The bounded-correction augmentation uses this rate:
///   the bounded form subtracts `U_P × (discrete mass residual)`, and on a
///   moving mesh that residual's ddt part is `ρ·d(V)/dt` at the SAME weights
///   as the momentum ddt — with any other choice a uniform flow is NOT a
///   fixed point of the momentum equation (the GCL gate fails under BDF2).
fn ale_volume_locals_setup(slots: &ResolvedStateSlotsSpec) -> Vec<Stmt> {
    use super::dsl::EnumExpr;
    use crate::solver::gpu::enums::TimeScheme;

    // The ALE volume rates divide by the GLOBAL dt; a per-cell `dt_local`
    // (LTS) would make them inconsistent with the ddt's `dt_eff`. No ALE v1
    // model carries dt_local (incompressible only); fail fast at codegen time.
    assert!(
        !slots.slots.iter().any(|s| s.name == "dt_local"),
        "ALE (relative_to_mesh) + local time stepping (dt_local) is unsupported: \
         the ALE volume rates use the global constants.dt"
    );

    let dt = Expr::ident("constants").field("dt");
    let dt_old = Expr::ident("constants").field("dt_old");
    let time_scheme =
        EnumExpr::<TimeScheme>::from_expr(Expr::ident("constants").field("time_scheme"));
    let vol = Expr::ident("vol");
    let vol_old = Expr::ident("vol_old");
    let vol_old_old = Expr::ident("vol_old_old");

    vec![
        dsl::let_expr(
            "vol_old",
            dsl::array_access("cell_vols_old", Expr::ident("idx")),
        ),
        dsl::let_expr(
            "vol_old_old",
            dsl::array_access("cell_vols_old_old", Expr::ident("idx")),
        ),
        // The equal-volume guard on the ratios hardens `x / x == 1.0`
        // against backends that lower `/` to an approximate reciprocal: with
        // bitwise-equal volume history the ratio is a compile-visible exact
        // 1.0 on every backend; under real motion the ratio only needs f32
        // accuracy (it is a discretization weight). NOTE this does NOT buy
        // GPU byte-identity of the BDF2 ddt on a static mesh: Metal fast
        // math reassociates the (textually different) static and ALE rhs
        // chains differently, producing ~1-ulp differences even with the
        // ratio pinned to exactly 1.0 — isolated and evidenced by
        // tests/ale_metal_fastmath_evidence.rs; the zero-flux gate is
        // therefore bitwise on CPU (both engines, Euler AND BDF2) and on
        // GPU/Euler, and tolerance-gated on GPU/BDF2.
        dsl::let_expr(
            "ale_vol_ratio_n",
            dsl::select(
                vol_old.clone() / vol.clone(),
                Expr::from(1.0),
                vol_old.clone().eq(vol.clone()),
            ),
        ),
        dsl::let_expr(
            "ale_vol_ratio_nm1",
            dsl::select(
                vol_old_old.clone() / vol.clone(),
                Expr::from(1.0),
                vol_old_old.clone().eq(vol.clone()),
            ),
        ),
        dsl::let_expr(
            "ale_dvdt_scl",
            (vol.clone() - vol_old.clone()) / dt.clone(),
        ),
        dsl::var_typed_expr(
            "ale_dvdt_ddt",
            super::wgsl_ast::Type::F32,
            Some(Expr::ident("ale_dvdt_scl")),
        ),
        dsl::if_block_expr(
            time_scheme.eq(TimeScheme::BDF2),
            dsl::block(vec![
                dsl::let_expr("r_ale", dt.clone() / dt_old),
                dsl::assign_expr(
                    Expr::ident("ale_dvdt_ddt"),
                    ((Expr::ident("r_ale") * 2.0 + 1.0) / (Expr::ident("r_ale") + 1.0)
                        * (vol - vol_old.clone())
                        - (Expr::ident("r_ale") * Expr::ident("r_ale"))
                            / (Expr::ident("r_ale") + 1.0)
                            * (vol_old - vol_old_old))
                        / dt,
                ),
            ]),
            None,
        ),
    ]
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

    // ALE (moving-mesh) systems get the moving-volume ddt: volume-history
    // locals here, ratio-weighted history states below. Non-ALE systems take
    // the exact pre-ALE path (byte-identical generated code).
    let ale = system.is_ale();
    if ale {
        stmts.extend(ale_volume_locals_setup(slots));
    }

    // Local Time Stepping (steady-state acceleration): if the model carries a
    // per-cell `dt_local` field, use it as the effective physical timestep when it
    // has been filled (> 0), otherwise fall back to the global `constants.dt`.
    // Models without the field emit the global `dt` verbatim — byte-identical.
    let dt_eff = match slots.slots.iter().find(|s| s.name == "dt_local") {
        Some(dt_local_slot) => {
            let dtl = state_component_slot(slots.stride, "state", "idx", dt_local_slot, 0);
            dsl::select(
                Expr::ident("constants").field("dt"),
                dtl.clone(),
                dtl.gt(0.0),
            )
        }
        None => Expr::ident("constants").field("dt"),
    };

    for equation in &system.equations {
        // Every implicit time-derivative op on this equation. Normally there is
        // exactly one and its field is the equation's own target (the diagonal
        // d/dt). A model may additionally declare a CROSS-variable ddt — a ddt of
        // a DIFFERENT field inside this equation (e.g. the thermal-expansion
        // `rho_dT * dT/dt` in the continuity/pressure row of `allmach_thermal`,
        // where d(rho)/dt = psi*dp/dt + rho_dT*dT/dt). The own-variable ddt keeps
        // the full implicit BDF1/BDF2/dual-time treatment; a cross-variable ddt is
        // emitted as an EXPLICIT lagged source (the per-cell diagonal accumulators
        // cannot carry an off-diagonal p<-T entry here).
        for ddt_op in equation.ops.iter().filter(|op| {
            op.kind == DiscreteOpKind::TimeDerivative
                && op.discretization == Discretization::Implicit
        }) {
            if ddt_op.field.name() == equation.target.name() {
                // ---- own-variable d/dt: implicit (unchanged for every model) ----
                let base_offset = *offsets
                    .get(equation.target.name())
                    .expect("missing target offset");
                let rho_expr =
                    coefficient_value_expr(slots, ddt_op.coeff.as_ref(), "idx", 1.0.into());
                let base_coeff = Expr::ident("vol") * rho_expr.clone() / dt_eff.clone();
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
                    let phi_n = state_component_slot(
                        slots.stride,
                        "state_old",
                        "idx",
                        target_slot,
                        component,
                    );
                    let phi_nm1 = state_component_slot(
                        slots.stride,
                        "state_old_old",
                        "idx",
                        target_slot,
                        component,
                    );
                    let phi_iter = state_component_slot(
                        slots.stride,
                        "state_iter",
                        "idx",
                        target_slot,
                        component,
                    );

                    // Moving-volume ddt (ALE): weight the history states by
                    // V^n/V^{n+1} and V^{n-1}/V^{n+1}, then apply the
                    // UNCHANGED static integrator (whose base coefficient is
                    // V^{n+1}·c/dt). This yields the variable-dt BDF stencil
                    // on the conserved product V·φ — see the derivation at
                    // `ale_volume_locals_setup` — and reduces bitwise to the
                    // static form when the volume history is equal (ratio ==
                    // 1.0). The dual-time (dtau) term inside the integrator
                    // stays on V^{n+1} by construction.
                    let (phi_n, phi_nm1) = if ale {
                        (
                            Expr::ident("ale_vol_ratio_n") * phi_n,
                            Expr::ident("ale_vol_ratio_nm1") * phi_nm1,
                        )
                    } else {
                        (phi_n, phi_nm1)
                    };

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
            } else {
                // Cross-variable ddt on a moving mesh would need the same
                // V^n-weighted history treatment; no v1 ALE model declares one
                // (incompressible only — allmach_thermal owns the cross ddt).
                // Fail fast at codegen time rather than silently mis-weighting.
                assert!(
                    !ale,
                    "cross-variable ddt (d({})/dt in the '{}' equation) is unsupported on \
                     ALE models (v1)",
                    ddt_op.field.name(),
                    equation.target.name()
                );
                // ---- cross-variable d/dt: IMPLICIT off-diagonal coupling ----
                // ddt(coeff, field) with field != target (e.g. the thermal-
                // expansion rho_dT*dT/dt in the continuity/pressure row). BDF1
                // implicit: the field_new coefficient goes to the within-cell
                // off-diagonal matrix block [eqn_row, field_col], and the field_old
                // part to the equation RHS — so the residual carries
                // coeff*(field_new - field_old)/dt with the p<-T coupling living in
                // the Jacobian/Schur block. Implicit (NOT a Picard lag), so it
                // does not stall in a fully-pinned closed box and stays stable at
                // finite compressibility. Requires the coupled-assembly context
                // (matrix_values, diag_rank, start_row_*), which is in scope
                // wherever a cross-variable ddt is declared. Scalar fields only.
                let eqn_offset = *offsets
                    .get(equation.target.name())
                    .expect("missing target offset");
                let field_col = *offsets
                    .get(ddt_op.field.name())
                    .expect("missing cross-ddt field offset");
                let rho_expr =
                    coefficient_value_expr(slots, ddt_op.coeff.as_ref(), "idx", 1.0.into());
                let base_coeff = Expr::ident("vol") * rho_expr / dt_eff.clone();
                let field_slot = slots
                    .slots
                    .iter()
                    .find(|s| s.name == ddt_op.field.name())
                    .unwrap_or_else(|| {
                        panic!(
                            "missing cross-ddt field '{}' in resolved state slots",
                            ddt_op.field.name()
                        )
                    });
                let field_old =
                    state_component_slot(slots.stride, "state_old", "idx", field_slot, 0);

                // matrix_values[start_row_{eqn} + diag_rank*coupled_stride + field_col] += base_coeff
                let entry_index = acc.start_row(eqn_offset)
                    + dsl::linear_index(
                        Expr::ident("diag_rank"),
                        acc.coupled_stride,
                        field_col,
                    );
                stmts.push(dsl::assign_op_expr(
                    AssignOp::Add,
                    dsl::array_access("matrix_values", entry_index),
                    base_coeff.clone(),
                ));
                // RHS += base_coeff * field_old  (BDF1 old-time part).
                stmts.push(acc.add_rhs(eqn_offset, base_coeff * field_old));
            }
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
