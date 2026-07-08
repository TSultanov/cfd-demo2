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
/// volume source. Emitted at the head of the ddt contributions (top-level scope
/// of the assembly `main`) so every later statement can reference them — only
/// for ALE systems (`DiscreteSystem::is_ale`); non-ALE lowering emits nothing.
///
/// All f32, from the `cell_vols` / `cell_vols_old{,_old}` buffers:
///
/// * `ale_vol_ratio_n = V^n / V^{n+1}`, `ale_vol_ratio_nm1 = V^{n-1} / V^{n+1}`:
///   history-state weights. The lowering multiplies `phi^n` / `phi^{n-1}` by
///   these and applies the unchanged static integrator (base coeff
///   `V^{n+1}·c/dt`), giving the variable-dt BDF stencil on the conserved
///   product V·φ (a=(2r+1)/(r+1), b=r+1, c=r²/(r+1), r=dt/dt_old). The ratio
///   formulation (rather than separate `V^n·c/dt` coefficients) is deliberate:
///   with equal volumes the ratios are exactly `1.0` (pinned by the select
///   guard below), so under strict IEEE the emitted arithmetic reduces
///   bit-for-bit to the static ddt. GPU/BDF2 is the exception: Metal fast math
///   reassociates the two rhs chains differently (~1 ulp even at ratio 1.0), so
///   that leg is tolerance-gated.
///
/// * `ale_dvdt_scl = (V^{n+1} − V^n)/dt`: the SCL (per-step swept) volume rate,
///   which the mesh-flux closure targets (`Σ_f mesh_fluxes[f] ==
///   (V^{n+1}−V^n)/dt` per cell), so the continuity volume source
///   `+ρ·ale_dvdt_scl` cancels the mesh-flux part of `Σ_f φ_rel` exactly (to f32
///   roundoff) under any time scheme.
///
/// * `ale_dvdt_ddt`: the volume rate at the ddt scheme's weights (Euler ⇒
///   `ale_dvdt_scl`; BDF2 ⇒ `[aV^{n+1} − bV^n + cV^{n−1}]/dt`). Emitted in the
///   incremental form `a·(V^{n+1}−V^n)/dt − c·(V^n−V^{n−1})/dt` (a−b+c=0) so
///   equal volumes vanish bitwise. The bounded-correction augmentation needs
///   the ddt part of the mass residual `ρ·d(V)/dt` at the SAME weights as the
///   momentum ddt, else a uniform flow is not a fixed point (GCL fails).
fn ale_volume_locals_setup(_slots: &ResolvedStateSlotsSpec) -> Vec<Stmt> {
    use super::dsl::EnumExpr;
    use crate::solver::gpu::enums::TimeScheme;

    // The ALE volume rates divide by the GLOBAL dt (they must match the
    // host-computed swept mesh fluxes, which are closed against the global step
    // dt: `Σ_f mesh_fluxes == (V^{n+1}-V^n)/dt_global`). A per-cell `dt_local`
    // (LTS) would make the ddt's `dt_eff` inconsistent with these rates and break
    // the GCL. The all-Mach models carry a `dt_local` field for steady-state LTS
    // acceleration, but the moving-mesh driver keeps it ≡ 0 (a moving mesh is
    // time-accurate, not a steady march), so `dt_eff` falls back to the global
    // `constants.dt` and the rates stay consistent. The presence of the field is
    // therefore permitted under ALE; a nonzero `dt_local` under motion is a driver
    // contract violation, not a codegen one.
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
        // The equal-volume guard on the ratios hardens `x / x == 1.0` against
        // backends that lower `/` to an approximate reciprocal: with
        // bitwise-equal volume history the ratio is a compile-visible exact 1.0;
        // under real motion the ratio only needs f32 accuracy (it is a
        // discretization weight).
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
                    // V^n/V^{n+1} and V^{n-1}/V^{n+1}, then apply the unchanged
                    // static integrator (base coeff V^{n+1}·c/dt). See the
                    // derivation at `ale_volume_locals_setup`. The dual-time
                    // (dtau) term stays on V^{n+1} by construction.
                    //
                    // A `non_conservative_ale` own-variable ddt (e.g. the
                    // compressibility/pseudo-acoustic `ddt(psi_precond, p)` in the
                    // continuity row) OPTS OUT of the conservative weighting: it is an
                    // intensive rate at V^{n+1} (like a cross-variable ddt), so its
                    // history stays unweighted. This keeps its geometric mass-change
                    // part OUT of the ddt (it would otherwise be a BDF2-rate
                    // `psi*p*dV/dt` that mismatches the SCL-rate volume source + mesh
                    // flux and caps the moving-mesh order at 1); the full per-cell
                    // `rho*dV/dt` is instead carried by the continuity volume source at
                    // the SCL rate, cancelling the mesh flux exactly. See
                    // `Term::non_conservative_ale`.
                    let (phi_n, phi_nm1) = if ale && !ddt_op.non_conservative_ale {
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
                let field_old_old =
                    state_component_slot(slots.stride, "state_old_old", "idx", field_slot, 0);

                let entry_index = acc.start_row(eqn_offset)
                    + dsl::linear_index(
                        Expr::ident("diag_rank"),
                        acc.coupled_stride,
                        field_col,
                    );
                stmts.push(dsl::assign_op_expr(
                    AssignOp::Add,
                    dsl::array_access("matrix_values", entry_index.clone()),
                    base_coeff.clone(),
                ));
                // BDF1 old-time part. NOTE: a cross-variable ddt on a moving mesh
                // is NOT conservative-weighted. These terms are the `V·∂ρ/∂ξ·dξ/dt`
                // pieces of the non-conservative continuity expansion
                // `dρ/dt = V(ψ·dp/dt + ρ_dT·dT/dt) + ρ·dV/dt` (e.g. the thermal
                // expansion `ρ_dT·dT/dt` in the pressure row, the compression
                // heating `dp/dt` in the temperature row): an INTENSIVE rate times
                // the CURRENT volume, `V^{n+1}·coeff·(ξ^{n+1}-ξ^n)/dt`. Both times
                // therefore carry `V^{n+1}` (the `base_coeff`), with NO `V^n/V^{n+1}`
                // weight — weighting `ξ^n` would make it `d(coeff·ξ·V)/dt`, which
                // spuriously injects `coeff·ξ·dV/dt` at a uniform field (`dξ/dt = 0`
                // but `ξ ≠ 0`), breaking free-stream preservation on a moving mesh.
                // The `ρ·dV/dt` piece lives in the barotropic continuity volume
                // source; the conservative moving-volume weighting is applied only
                // to the PRIMARY ddts (paired with the bounded correction).
                stmts.push(acc.add_rhs(eqn_offset, base_coeff.clone() * field_old.clone()));

                // Optional BDF2 correction for the cross-variable (off-diagonal) ddt,
                // mirroring the own-variable diagonal path (`BdfDualTimeIntegrator::
                // emit_component`). Written INCREMENTALLY (subtract the just-emitted
                // BDF1 part, add the variable-dt BDF2 part) so Euler/BDF1 runs — and
                // every model without a cross-variable ddt — stay byte-identical, and
                // the whole block is inert unless `time_scheme == BDF2`. `field_old_old`
                // reads the SAME cross-field slot from `state_old_old` (already bound,
                // and consumed by the diagonal BDF2 correction above). CRUCIALLY, unlike
                // the own-variable ddt, the cross history is NOT `ale_vol_ratio`-weighted:
                // both time levels stay on `V^{n+1}` (`base_coeff`) so free-stream is
                // preserved on a moving mesh — the identity `(2r+1)/(r+1) == factor_n −
                // factor_nm1` makes the unweighted stencil vanish exactly at a
                // uniform-in-time field (vol-ratio weighting would inject `coeff·ξ·dV/dt`
                // and break the GCL; see the BDF1 note above). `r` uses the global
                // `constants.dt/dt_old`, matching the diagonal path whose `base_coeff`
                // likewise carries `dt_eff` while `r` uses the global step.
                {
                    use super::dsl::EnumExpr;
                    use crate::solver::gpu::enums::TimeScheme;
                    let dt = Expr::ident("constants").field("dt");
                    let dt_old = Expr::ident("constants").field("dt_old");
                    let time_scheme = EnumExpr::<TimeScheme>::from_expr(
                        Expr::ident("constants").field("time_scheme"),
                    );
                    stmts.push(dsl::if_block_expr(
                        time_scheme.eq(TimeScheme::BDF2),
                        dsl::block(vec![
                            dsl::let_expr("r_x", dt / dt_old),
                            dsl::let_expr(
                                "diag_bdf2_x",
                                base_coeff.clone() * (Expr::ident("r_x") * 2.0 + 1.0)
                                    / (Expr::ident("r_x") + 1.0),
                            ),
                            dsl::let_expr("factor_n_x", Expr::ident("r_x") + 1.0),
                            dsl::let_expr(
                                "factor_nm1_x",
                                (Expr::ident("r_x") * Expr::ident("r_x"))
                                    / (Expr::ident("r_x") + 1.0),
                            ),
                            // matrix: swap the BDF1 coeff for the BDF2 coeff.
                            dsl::assign_op_expr(
                                AssignOp::Add,
                                dsl::array_access("matrix_values", entry_index),
                                Expr::ident("diag_bdf2_x") - base_coeff.clone(),
                            ),
                            // rhs: swap the BDF1 old-time part for the two-level part.
                            acc.add_rhs(
                                eqn_offset,
                                base_coeff.clone()
                                    * (Expr::ident("factor_n_x") * field_old.clone()
                                        - Expr::ident("factor_nm1_x") * field_old_old)
                                    - base_coeff * field_old,
                            ),
                        ]),
                        None,
                    ));
                }
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
