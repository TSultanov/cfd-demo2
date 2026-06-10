# Compressible Dual-Time Improvement Plan

Goal: improve the compressible dual-time path so it behaves more physically at low Mach, converges more reliably, and avoids inlet-localized artifacts, while explicitly keeping the current zero-velocity initial field in place.

## Scope

In scope:
- Improve the dual-time iteration path so each physical step is driven by meaningful pseudo-time convergence rather than fixed damping alone.
- Reduce reliance on aggressive under-relaxation as the primary stabilizer.
- Make pseudo-time stepping more locally appropriate for cut-cell and backstep flows.
- Improve compressible low-Mach boundary treatment and positivity robustness.
- Add diagnostics and regression coverage for the dual-time path.

Out of scope:
- Changing the current zero-velocity interior initialization used by the UI and tests.
- Replacing the compressible model formulation wholesale.
- Refactoring unrelated mesh generation, rendering, or OpenFOAM reference infrastructure.

## Completed milestones

- **Phase 1–2**: Dual-time acceptance/status plumbing, retry/reject/rollback handling, relaxation-default cleanup, primitive-recovery rho guards, solver-level positivity minima/counter diagnostics, positivity-triggered rollback/retry fallback, UI surfacing of positivity stats, and full regression coverage.
- **Phase 3 (local `dtau`)**: Face-based geometric local scaling implemented in generated dual-time operator; regular square cells keep unit scale, distorted/cut cells receive stronger pseudo-time damping. Validated on structured and cut-cell meshes with dedicated regressions and OpenFOAM comparison.
- **Phase 4 (boundary refresh)**: Inlet thermodynamic refresh via recurring Preparation-phase runtime kernel; outlet static-pressure Dirichlet BC seeded through existing helper path; outlet non-pressure state extrapolated from interior per iteration; BC refresh now runs per outer iteration (not just per step). Validated with backstep/structured-channel regressions and OpenFOAM comparison.
- **Phase 6 (validation)**: UI-like regressions cover acceptance status, scaled residuals, retry/backoff, velocity bounds, positivity diagnostics, and the forced positivity-fallback path. Local-`dtau` validation covers structured and cut-cell cases. Low-Mach multi-step physical regression added. OpenFOAM comparisons completed for retry-policy, positivity-protection, and per-iteration BC refresh milestones with no metric regression.

## Phase 3: Local pseudo-time stepping — remaining decision

Decide whether to keep the current face-based geometric metric as the production formulation or replace or augment it with a spectral-radius form such as `dtau_i = CFL_tau * V_i / sum_f(|lambda_f| A_f)`. Any later spectral form must include acoustic and advective contributions consistently with the low-Mach preconditioned pseudo-time operator.

Key files: [generic_coupled.rs](src/solver/gpu/lowering/programs/generic_coupled.rs), [generic_coupled_ports.rs](src/solver/model/modules/generic_coupled_ports.rs)

## Phase 4: Subsonic compressible boundary treatment — remaining items

1. **Inlet policy evolution**: Decide whether the current pressure-following reduced inlet set (inlet refreshes p, T, rho_e from interior pressure; preserves prescribed rho and u) is sufficient, or whether a fuller characteristic-total-state form is needed.

2. **Outlet policy evolution**: The current outlet extrapolates non-pressure state from interior and keeps the host-seeded Dirichlet pressure. Decide whether this should evolve toward a characteristic-based treatment (e.g., wave-transmissive outlet) or remain as-is.

Key files: [solver_ext.rs](src/solver/model/helpers/solver_ext.rs), [compressible.rs](src/solver/model/definitions/compressible.rs) (inlet/outlet are declared `BoundaryExpr` sets lowered by [bc_expr.rs](src/solver/model/modules/bc_expr.rs))

## Phase 5: Positivity fallback — remaining decision

Decide whether rollback/backoff is the final bounded positivity response or whether a more local bounded fallback (e.g. limited update scaling) is warranted for recoverable violations, now that boundary treatment and retry policy are settled.

Key files: [primitives.rs](src/solver/model/primitives.rs), [app.rs](src/ui/app.rs)

## Recommended next steps

1. Reassess Phase 3 spectral-radius decision now that boundary treatment is stable.
2. Reassess Phase 5 positivity fallback granularity.
3. Run OpenFOAM reference suite after any further major changeset.

## Validation commands

- `cargo test --features 'meshgen dev-tests' --test ui_compressible_dual_time_backstep_regression_test -- --nocapture`
- `cargo test --features 'meshgen dev-tests' --test gpu_compressible_dual_time_uniform_state_test -- --nocapture`
- `cargo test --features 'meshgen dev-tests' --test gpu_compressible_low_mach_multi_step_regression_test -- --nocapture`
- `CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh`

## Non-goal reminder

This plan intentionally keeps the current zero-velocity initial interior field unchanged. Any implementation work under this plan must preserve that startup behavior unless a separate follow-up plan explicitly changes it.