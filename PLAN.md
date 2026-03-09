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

## Current issues to address

1. Dual-time convergence is weakly enforced.
The current path already computes outer-loop correction norms and can break early, but the practical behavior still allows underconverged pseudo-time iterations to advance the physical state. Relevant paths:
- [src/solver/gpu/lowering/programs/generic_coupled.rs](src/solver/gpu/lowering/programs/generic_coupled.rs)
- [src/solver/model/helpers/solver_ext.rs](src/solver/model/helpers/solver_ext.rs)

2. Compressible dual-time stability depends too heavily on velocity under-relaxation.
The compressible model currently sets a conservative default `alpha_u = 0.2` for `dtau > 0`. This improves robustness, but it also makes the method behave like a damped fixed-point loop rather than a strongly converged pseudo-time solve. Relevant path:
- [src/solver/model/definitions/compressible.rs](src/solver/model/definitions/compressible.rs)

3. Pseudo-time stepping uses a global `dtau`.
A single global pseudo-time scale is a poor fit for low-Mach internal flows on nonuniform or cut-cell meshes, where local spectral radii differ substantially across the domain.

4. Low-Mach preconditioning is present, but the pseudo-time path can still march on with insufficient pseudo-time convergence.
Relevant path:
- [src/solver/model/flux_schemes.rs](src/solver/model/flux_schemes.rs)

5. Primitive recovery remains sensitive to local density undershoots.
Velocity is recovered directly as `rho_u / rho`, so any local rho undershoot near the inlet can create a visually large velocity spike.
Relevant path:
- [src/solver/model/primitives.rs](src/solver/model/primitives.rs)

6. Inlet boundary treatment is stronger than ideal for subsonic compressible inflow.
The current inlet setter writes a fully specified isothermal state for multiple variables. That is usable, but it is more rigid than a characteristic subsonic treatment.
Relevant path:
- [src/solver/model/helpers/solver_ext.rs](src/solver/model/helpers/solver_ext.rs)

## Implementation principles

1. Preserve the current physical-time semantics.
Low-Mach preconditioning and pseudo-time controls should accelerate convergence to the physical implicit step, not alter the target physical solution.

2. Prefer convergence controls over damping.
The first line of defense should be pseudo-time residual reduction and adaptive pseudo-time control, not stronger relaxation.

3. Keep the zero-velocity initial field.
All improvements must work with the existing startup behavior where the interior field begins at rest.

4. Add observability before tightening behavior.
The implementation should expose enough residual, positivity, and pseudo-time statistics to explain why a step converged, stalled, or was forced to back off.

## Progress Update

Status as of 2026-03-09:
- Completed: dual-time acceptance/status plumbing, nonconverged-step retry/reject handling, relaxation-default cleanup, primitive-recovery rho guards, solver-level positivity minima/counter diagnostics, positivity-triggered rollback/retry fallback, UI surfacing of positivity stats, and explicit regression coverage for the clean-case, retry/backoff, and forced positivity-fallback paths.
- Completed: the required OpenFOAM pre/post comparisons for both the retry-policy milestone and the first positivity-protection milestone showed unchanged reported `[openfoam]` discrepancy metrics.
- Completed: the first Phase 3 local pseudo-time slice now uses a face-based geometric local scaling in the generated dual-time operator, normalized so regular square cells keep the baseline global scale while more distorted cells receive stronger pseudo-time damping.
- Completed: the Phase 3 face-metric slice passed targeted codegen tests, the UI-like compressible dual-time backstep regression, and a required OpenFOAM pre/post comparison without worsening tracked `[openfoam]` discrepancy metrics; the worst reported tracked discrepancy improved slightly versus the fresh baseline.
- Completed: the dedicated structured local-`dtau` regression now verifies that regular square cells keep the unit face-metric scale and that a structured compressible dual-time channel step remains bounded while surfacing explicit pseudo-time acceptance statistics.
- Active remaining work: decide whether the current face metric is final or should evolve toward a spectral-radius form, implement Phase 4 boundary-condition changes, complete the Phase 5 fallback-granularity follow-up, and add a low-Mach physical regression case.

## Phase 3: Add local pseudo-time stepping

Objective: replace the single global `dtau` with a more physical per-cell or locally varying pseudo-time scale.

Status:
- Partially completed. The first implementation slice now scales the dual-time diagonal and RHS by a per-cell face-based geometric metric derived from `perimeter_sum^2 / (16 * vol)`, with the existing global `dtau` retained as the user-facing scaling knob.
- Remaining work is to decide whether this face-based metric is the final representation or an intermediate approximation toward a fuller spectral-radius form.

Tasks:
1. Decide whether the current face metric is the final production formulation.
   - Done for validation coverage: cut-cell behavior is covered indirectly by the UI-like compressible dual-time backstep regression, and the structured-grid regression now verifies unit local scale on regular square cells while recording bounded structured-channel dual-time behavior.
   - Done for the first slice: the generated assembly now uses a local face-based geometric scale built from cell face areas and normalized to equal `1` on square cells.
   - Remaining: decide whether to keep the face-based metric as the production formulation or replace/augment it with a spectral-radius form such as `dtau_i = CFL_tau * V_i / sum_f(|lambda_f| A_f)`.
   - Ensure any later spectral form includes acoustic and advective contributions consistently with the low-Mach preconditioned pseudo-time operator.

2. Reassess whether the transient representation is sufficient.
   - Done for the first slice: the local scale is derived transiently in the generated assembly kernels rather than stored in a dedicated field.
   - Remaining: only revisit a persistent `dtau_local` field if later diagnostics or a spectral-radius variant clearly require it.

Expected file focus:
- [src/solver/gpu/lowering/programs/generic_coupled.rs](src/solver/gpu/lowering/programs/generic_coupled.rs)
- [src/solver/model/modules/generic_coupled_ports.rs](src/solver/model/modules/generic_coupled_ports.rs)
- generated coupled assembly/update paths
- [src/ui/app.rs](src/ui/app.rs)

Acceptance criteria:
- Status: mostly completed.
- Done: the pseudo-time operator now uses a locally varying per-cell scale in generated assembly, cut-cell behavior is exercised by the UI-like backstep regression, and the dedicated structured-grid regression verifies unit local scaling on regular square cells.
- Remaining: decide whether the face-based metric is sufficient or should evolve toward a fuller spectral-radius formulation.

## Phase 4: Improve subsonic compressible boundary treatment

Objective: make the inlet and outlet behavior more physically appropriate for low-Mach subsonic flow.

Tasks:
1. Define a characteristic-style subsonic inlet policy.
   - Prefer specifying inflow through thermodynamic totals and flow direction, or another characteristic-consistent reduced set, instead of fully forcing all primitive quantities.
   - Preserve a compatibility wrapper so existing callers can still request `set_compressible_inlet_isothermal_x(...)` while internally mapping to the improved treatment.

2. Reassess outlet pressure treatment.
   - Ensure the outlet primarily constrains the appropriate outgoing characteristic or static pressure, rather than overconstraining the thermodynamic state.

3. Verify compatibility with the current generic BC table mechanism.
   - The current GPU path expands BCs per face and per component in the generic coupled backend.
   - Confirm the new inlet treatment can be expressed cleanly in that structure or introduce a specialized compressible boundary update path if needed.

Expected file focus:
- [src/solver/model/helpers/solver_ext.rs](src/solver/model/helpers/solver_ext.rs)
- [src/solver/model/definitions/compressible.rs](src/solver/model/definitions/compressible.rs)
- [src/solver/gpu/program/generic_coupled_backend.rs](src/solver/gpu/program/generic_coupled_backend.rs)

Acceptance criteria:
- Compressible subsonic inlet and outlet behavior are less rigid and better aligned with physical wave propagation.
- Existing tests can be updated without changing the zero-velocity interior initialization.

## Phase 5: Positivity fallback follow-up

Objective: decide whether the current rollback/backoff response is the final positivity policy or whether later solver changes warrant a more local bounded fallback.

Tasks:
1. Reassess fallback granularity after the rollback path lands.
   - Status: active follow-up. The current implementation rejects or rolls back nonphysical dual-time updates, records positivity diagnostics, and regression coverage now exercises the forced fallback path.
   - Remaining: decide whether rollback/backoff alone is sufficient or whether a more local bounded response, such as limited update scaling, is still warranted for recoverable violations.

Expected file focus:
- [src/solver/model/primitives.rs](src/solver/model/primitives.rs)
- update kernel generation paths
- diagnostics in [src/ui/app.rs](src/ui/app.rs)

Acceptance criteria:
- Done: the solver now surfaces rho/p minima and undershoot counts, guards primitive recovery against tiny rho values, routes nonphysical compressible dual-time outcomes through rollback/retry instead of silently advancing them, and has a dedicated regression that exercises the positivity-triggered fallback path.
- Remaining: decide whether rollback/backoff is the final bounded response or just the first implementation stage.

## Phase 6: Validation and regression coverage

Objective: make the new dual-time path measurable and maintainable.

Tasks:
1. Extend existing UI-like regressions.
   - Status: completed. The existing UI-like compressible dual-time regressions now assert explicit acceptance status, scaled conserved residuals, rejected-retry/`dt`-`dtau` backoff behavior, maximum-velocity bounds, positivity minima/counter behavior, and the forced positivity-triggered fallback path, and focused unit tests cover retry-status classification and retry-control plumbing.

2. Add focused tests for local pseudo-time stepping.
   - Verify that local pseudo-time behaves sensibly on structured and cut-cell meshes.

3. Add a low-Mach physical regression case.
   - Keep the zero-velocity initial field.
   - Verify that the solution progresses physically over repeated dual-time-corrected steps without relying on extreme damping.

4. Run the OpenFOAM reference suite after major implementation milestones.
   - Status: completed for the retry-policy and first positivity-protection milestones. The required pre/post runs were executed, `[openfoam]` discrepancy metrics were emitted in both runs, and the before/after diffs were empty for both changesets.
   - Follow [AGENTS.md](AGENTS.md) requirements.
   - Treat any worse OpenFOAM discrepancy as a regression unless explicitly justified.

Suggested validation commands:
- `cargo test --features 'meshgen dev-tests' --test ui_compressible_dual_time_backstep_regression_test -- --nocapture`
- `CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh`

Acceptance criteria:
- Status: partial.
- Done: the dual-time path has explicit regression coverage for pseudo-time acceptance status, scaled conserved residuals, rejected-retry/backoff behavior, maximum-velocity bounds, rho/p minima, positivity undershoot counters, and the forced positivity-fallback path, plus focused unit coverage for retry-control plumbing and retry-status classification.
- Done: focused local-`dtau` validation now covers both the structured square-cell case and the cut-cell/backstep path.
- Done: the required OpenFOAM pre/post comparisons for the retry-policy and first positivity-protection milestones showed unchanged reported discrepancy metrics.
- Remaining: add a low-Mach physical regression case.

## Recommended implementation order

1. Phase 3: add local pseudo-time stepping.
2. Phase 4: improve subsonic compressible boundary treatment.
3. Phase 5 follow-up: reassess whether rollback/backoff remains the desired final positivity response after the local-`dtau` and boundary changes land.
4. Phase 6: add local-`dtau` validation, add a low-Mach physical regression case, and rerun reference sweeps.

Rationale:
- The dual-time acceptance, retry, and positivity-observability slices are now implemented and regression-covered, so the remaining work shifts to local pseudo-time scaling and boundary semantics.
- Phase 3 improves scaling and robustness now that fallback behavior is already exposed and tested.
- Phase 4 changes BC semantics and should come after the core pseudo-time loop is trustworthy.

## Non-goal reminder

This plan intentionally keeps the current zero-velocity initial interior field unchanged. Any implementation work under this plan must preserve that startup behavior unless a separate follow-up plan explicitly changes it.