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

Status as of 2026-03-07:
- Completed: Phase 1 observability/status plumbing for compressible dual-time acceptance.
- Completed: Phase 6 partial regression coverage for explicit pseudo-time acceptance status.
- Partial: Phase 1 convergence classification now uses scaled pseudo-time correction norms for conserved compressible variables, but step rejection/retry is not implemented yet.
- Partial: Phase 2 relaxation policy now defaults the compressible dual-time path to full updates and reserves `nonconverged_relax` for explicitly nonconverged pseudo-time steps or linear-solver failures.
- Not started: local pseudo-time stepping, boundary-condition changes, and positivity protection.

## Phase 1: Strengthen pseudo-time convergence control

Objective: ensure the physical state is only accepted after the pseudo-time loop has sufficiently reduced the nonlinear residual.

Tasks:
1. Promote adaptive outer-loop convergence to the default compressible dual-time mode.
   - Status: audited. The generic coupled path already defaults to adaptive outer-loop break behavior; this changeset preserved that default and added explicit end-of-step status reporting for the dual-time path.
   - Audit how `outer_break_enabled`, `outer_tol`, and `outer_tol_abs` are initialized and used in [src/solver/gpu/lowering/programs/generic_coupled.rs](src/solver/gpu/lowering/programs/generic_coupled.rs).
   - Ensure compressible dual-time steps evaluate convergence every pseudo-iteration unless the user explicitly opts into fixed-iteration mode.

2. Tighten what “converged” means for compressible dual-time.
   - Status: partial. Compressible dual-time status classification now checks scaled correction norms for `rho`, `rho_u`, and `rho_e` when those targets are present, instead of relying only on linear-solver convergence.
   - Use scaled residuals for at least `rho`, `rho_u`, and `rho_e` rather than depending only on linear solver convergence.
   - Define convergence on the pseudo-time correction norm, not on the final Krylov residual alone.

3. Add a failure path for nonconverged pseudo-time steps.
   - Status: partial. Nonconverged dual-time steps are now surfaced explicitly as `accepted_nonconverged`, but retry/reject behavior is still pending.
   - If the outer loop hits `outer_iters` without sufficient reduction, do not silently accept the state as if it were converged.
   - Introduce one of these behaviors:
     - reject the physical step and retry with reduced `dt`, or
     - keep the step but mark it as nonconverged and trigger an automatic reduction of `dt` and or `dtau` on the next attempt.
   - Prefer the first option for correctness, but gate it behind a runtime setting if necessary to preserve existing workflows.

4. Improve runtime diagnostics.
   - Status: completed. The solver/UI/test path now reports explicit pseudo-time acceptance status (`accepted_converged` or `accepted_nonconverged`) alongside scaled and absolute outer residuals.
   - Emit the per-step scaled pseudo-time residuals already available in the plan state.
   - Add an explicit status such as `accepted_converged`, `accepted_nonconverged`, or `rejected_retry`.

Expected file focus:
- [src/solver/gpu/lowering/programs/generic_coupled.rs](src/solver/gpu/lowering/programs/generic_coupled.rs)
- [src/solver/model/helpers/solver_ext.rs](src/solver/model/helpers/solver_ext.rs)
- [src/ui/app.rs](src/ui/app.rs)

Acceptance criteria:
- Status: partial.
- Done: compressible dual-time runs report scaled outer residuals for conserved variables.
- Done: a pseudo-time loop that fails to converge is detectable and no longer silently indistinguishable from a converged step.
- Done: existing non-dual-time paths remain unchanged.
- Remaining: nonconverged steps are still accepted rather than retried/rejected.

## Phase 2: Reduce dependence on aggressive relaxation

Objective: stop using `alpha_u = 0.2` as the default mechanism holding the solver together.

Tasks:
1. Revisit compressible dual-time relaxation defaults in [src/solver/model/definitions/compressible.rs](src/solver/model/definitions/compressible.rs).
   - Status: partial. The compressible model now defaults dual-time updates to `alpha_u = 1.0` and `alpha_p = 1.0`; further tuning still needs validation against tougher low-Mach cases.
   - Raise the default `alpha_u` toward 1.0.
   - Keep `alpha_p = 1.0` unless a later study shows a better physically consistent treatment.

2. Use relaxation as a fallback, not the primary path.
   - Status: partial. The generic coupled update now only applies `nonconverged_relax` when a dual-time step is explicitly classified as nonconverged, or when the linear solve itself fails before a pseudo-time status is available.
   - Keep or extend the existing `nonconverged_relax` control so that relaxation is only strengthened when the pseudo-time loop is demonstrably struggling.
   - Distinguish between normal pseudo-time updates and degraded fallback behavior.

3. Tie relaxation policy to pseudo-time convergence state.
   - Status: partial. Apply-time damping now keys off `accepted_converged` versus `accepted_nonconverged` status for dual-time steps, but there is still no automatic retry/backoff path.
   - Converged or nearly converged pseudo-iterations should use minimal damping.
   - Repeated stalled iterations may activate a fallback relaxation policy for robustness.

Expected file focus:
- [src/solver/model/definitions/compressible.rs](src/solver/model/definitions/compressible.rs)
- [src/solver/model/kernel.rs](src/solver/model/kernel.rs)
- [src/solver/model/modules/generic_coupled.rs](src/solver/model/modules/generic_coupled.rs)

Acceptance criteria:
- Status: partial.
- Done: the compressible dual-time path now defaults to materially less damping than the previous `alpha_u = 0.2` baseline.
- Done: fallback damping is keyed to pseudo-time convergence classification instead of being the default path.
- Remaining: validate whether the higher-default path stays robust enough across broader low-Mach regression coverage.

## Phase 3: Add local pseudo-time stepping

Objective: replace the single global `dtau` with a more physical per-cell or locally varying pseudo-time scale.

Tasks:
1. Define the local pseudo-time formula.
   - Base it on local control-volume size and local spectral radius.
   - Candidate form:
     - `dtau_i = CFL_tau * V_i / sum_f(|lambda_f| A_f)`
   - Ensure acoustic and advective contributions are included consistently with the low-Mach preconditioned pseudo-time operator.

2. Decide representation.
   - Either add a new per-cell state or field buffer for `dtau_local`, or derive it transiently in the relevant kernels.
   - Prefer a runtime field if it improves diagnostics and avoids redundant recomputation.

3. Thread local pseudo-time through assembly and update.
   - The current assembly adds `vol/dtau` style diagonal terms.
   - Replace or augment this with `vol/dtau_i` on a per-cell basis.

4. Preserve the existing global `dtau` UI control as a scaling knob.
   - Reinterpret the current scalar `dtau` as a target pseudo-CFL or global multiplier for local pseudo-time.
   - Avoid breaking the current UI contract abruptly.

Expected file focus:
- [src/solver/gpu/lowering/programs/generic_coupled.rs](src/solver/gpu/lowering/programs/generic_coupled.rs)
- [src/solver/model/modules/generic_coupled_ports.rs](src/solver/model/modules/generic_coupled_ports.rs)
- generated coupled assembly/update paths
- [src/ui/app.rs](src/ui/app.rs)

Acceptance criteria:
- The pseudo-time operator uses a locally varying scale or an equivalent local spectral-radius formulation.
- Dual-time convergence becomes less sensitive to the smallest or stiffest cells controlling the whole domain.

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

## Phase 5: Add positivity protection for pseudo-time updates

Objective: prevent local rho or p undershoots from producing unphysical spikes during pseudo-time convergence.

Tasks:
1. Add conservative positivity checks after update.
   - Detect cells with rho or p below a threshold after each pseudo-time update.
   - Report counts and extrema in diagnostics.

2. Introduce a bounded fallback for nonphysical updates.
   - Options include update clipping, pseudo-time backtracking, or limited update scaling.
   - Prefer reducing the update magnitude over hard clipping where practical.

3. Make primitive recovery robust.
   - Guard `u = rho_u / rho` against tiny rho values in a way that prevents artificial visual spikes while preserving diagnostic visibility into the underlying problem.

Expected file focus:
- [src/solver/model/primitives.rs](src/solver/model/primitives.rs)
- update kernel generation paths
- diagnostics in [src/ui/app.rs](src/ui/app.rs)

Acceptance criteria:
- The solver no longer produces extreme visual velocity spikes from a local rho undershoot without surfacing a diagnostic.
- Positivity-protection logic is only active when needed and does not materially alter converged solutions.

## Phase 6: Validation and regression coverage

Objective: make the new dual-time path measurable and maintainable.

Tasks:
1. Extend existing UI-like regressions.
   - Status: partial. The existing UI-like compressible dual-time regression now asserts that each dual-time step surfaces an explicit acceptance status.
   - Start from [tests/ui_compressible_dual_time_backstep_regression_test.rs](tests/ui_compressible_dual_time_backstep_regression_test.rs).
   - Add assertions on:
     - scaled outer residual reduction
     - whether a step was accepted converged vs nonconverged
     - maximum velocity bounds
     - positivity counters or rho/p minima

2. Add focused tests for local pseudo-time stepping.
   - Verify that local pseudo-time behaves sensibly on structured and cut-cell meshes.

3. Add a low-Mach physical regression case.
   - Keep the zero-velocity initial field.
   - Verify that the solution progresses physically over repeated dual-time-corrected steps without relying on extreme damping.

4. Run the OpenFOAM reference suite after major implementation milestones.
   - Status: attempted for this milestone. The required pre/post runs were executed, but the current workspace baseline fails during compilation in the OpenFOAM reference tests before any `[openfoam]` discrepancy metrics are emitted.
   - Follow [AGENTS.md](AGENTS.md) requirements.
   - Treat any worse OpenFOAM discrepancy as a regression unless explicitly justified.

Suggested validation commands:
- `cargo test --features 'meshgen dev-tests' --test ui_compressible_dual_time_backstep_regression_test -- --nocapture`
- `CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh`

Acceptance criteria:
- Status: partial.
- Done: the dual-time path has explicit regression coverage for pseudo-time acceptance status, in addition to the existing blow-up-avoidance checks.
- Remaining: add scaled-residual-reduction and positivity assertions.
- Blocked by existing reference-suite compile failures: OpenFOAM discrepancy metrics are not currently available in this workspace baseline, so this milestone could only verify that the suite failure mode did not worsen.

## Recommended implementation order

1. Phase 1: strengthen pseudo-time convergence control.
2. Phase 2: reduce dependence on aggressive relaxation.
3. Phase 6 partial: add diagnostics and regression assertions for the new behavior.
4. Phase 3: add local pseudo-time stepping.
5. Phase 4: improve subsonic compressible boundary treatment.
6. Phase 5: add positivity protection and primitive-recovery guards.
7. Phase 6 full: run reference and regression sweeps.

Rationale:
- Phases 1 and 2 address the main physical weakness without changing initialization.
- Phase 3 improves scaling and robustness once convergence semantics are reliable.
- Phase 4 changes BC semantics and should come after the core pseudo-time loop is trustworthy.
- Phase 5 is essential for robustness, but it should be layered over a better-converged dual-time method rather than used to mask the underlying deficiencies.

## Non-goal reminder

This plan intentionally keeps the current zero-velocity initial interior field unchanged. Any implementation work under this plan must preserve that startup behavior unless a separate follow-up plan explicitly changes it.