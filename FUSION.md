# DSL-Based Kernel Fusion Checklist

This checklist tracks the implementation of full compile-time, DSL-based kernel fusion.

## Success Criteria (Full DSL-Based Compile-Time Fusion)

- Fusion replacement kernels are synthesized at build time from DSL `KernelProgram` inputs.
- Fusion-capable kernel implementations are authored via structured DSL/AST construction (WGSL DSL statements/expressions), not hand-written `Vec<String>` program sections.
- Runtime selection remains recipe/schedule-driven (`fusion_schedule_registry`) with no ad-hoc
  fusion pass during stepping.
- `KernelFusionPolicy::Off|Safe|Aggressive` semantics are explicit and test-covered.
- Fused kernel IDs stay stable so runtime graph wiring does not churn across migrations.

## 0) Scope and Contracts

- [x] Define and document "full DSL-based fusion" success criteria (compile-time only, no runtime fusion pass).
- [x] Require structured DSL/AST kernel encoding for fusion-capable kernels (no manual string-array kernel body encoding).
- [x] Confirm fusion remains recipe-schedule driven in `src/solver/gpu/recipe.rs` via `fusion_schedule_registry::schedule_for_model`.
- [x] Add a contract test that fused kernels are resolved through generated registries (not handwritten runtime lookup glue).
- [x] Add a contract test that runtime code does not call `apply_model_fusion_rules`.

## 1) Kernel Program IR (Fusion Input)

- [x] Add a kernel-program IR module under `crates/cfd2_ir/src/solver/ir` (for fusion-capable kernels).
- [x] Model dispatch domain in IR (cells/faces/custom) and preserve launch semantics.
- [x] Model bind interface in IR (group/binding/name/access) for deterministic merge checks.
- [x] Represent kernel body/preamble/indexing in IR so two kernels can be safely composed.
- [x] Add side-effect metadata in IR (read/write sets, optional barriers/atomics flags).
- [x] Re-export IR types from `crates/cfd2_ir/src/solver/ir/mod.rs`.

## 2) Generator API Upgrade

- [x] Extend model kernel generator API to support DSL program artifacts in addition to WGSL output.
- [x] Keep backward compatibility so existing WGSL-only generators continue to work.
- [x] Add helper constructors/adapters for "DSL program -> WGSL" lowering.
- [x] Add tests for mixed-mode models (some kernels DSL-capable, some WGSL-only).

## 3) Fusion Compiler Pass (Codegen)

- [x] Add fusion pass module under `crates/cfd2_codegen/src/solver/codegen`.
- [x] Implement candidate matching based on applied fusion rules and ordered kernel list.
- [x] Implement Safe policy checks:
  - [x] same dispatch domain,
  - [x] compatible preamble/indexing,
  - [x] no unsafe hazards (RAW/WAR/WAW conflicts, barriers/atomics restrictions).
- [x] Implement deterministic symbol-renaming to avoid local-name collisions.
- [x] Merge bind interfaces deterministically and reject incompatible interfaces with clear errors.
- [x] Emit fused DSL program and lower it to WGSL through existing AST/lowering pipeline.
- [x] Add unit tests for all pass stages (matching, hazards, symbol merge, bind merge, output determinism).

## 4) Build-Time Synthesis and Registry Wiring

- [x] In `build.rs`, synthesize replacement kernels when fusion schedule references a replacement without explicit generator output.
- [x] Emit synthesized fused WGSL files into `src/solver/gpu/shaders/generated` with stable names.
- [x] Ensure `generate_kernel_registry_map` includes synthesized fused kernels.
- [x] Ensure `generate_fusion_schedule_registry` validates resolvability of synthesized outputs.
- [x] Keep generated registry entries deterministic across builds.
- [x] Add a contract/integration test that generated registry contains synthesized fused entries.

## 5) Pilot Migration: Rhie-Chow

- [x] Migrate `rhie_chow:dp_update_store_grad_p_v1` to synthesized fusion output.
- [x] Remove manual fused kernel generator/body once synthesized path is validated.
- [x] Keep stable fused kernel id (`rhie_chow/dp_update_store_grad_p_fused`) to avoid runtime churn.
- [x] Keep schedule behavior unchanged for `Off`/`Safe`/`Aggressive` until aggressive semantics are expanded.
- [x] Add parity test: fused vs unfused Rhie-Chow results remain numerically equivalent within tolerance.

## 6) Policy Semantics and Expansion

- [x] Codify policy semantics:
  - [x] `Off`: no fusion,
  - [x] `Safe`: strict hazard-checked fusion,
  - [x] `Aggressive`: broader candidate set + optional algebraic cleanup passes.
- [x] Implement explicit aggressive-only transforms behind policy guard.
- [x] Add tests proving `Aggressive` can differ from `Safe` only where intended.

## 7) Broaden Coverage Beyond Pilot

- [x] Identify all model-generated kernels eligible for DSL fusion (update/assembly-heavy chains first).
- [x] Migrate eligible generators from WGSL-only to DSL-capable artifacts.
- [x] Keep non-DSL kernels as standalone passes (no forced migration).
- [x] Add per-model coverage report (which kernels are fusion-capable vs legacy).

## 8) Validation Matrix

- [x] Unit tests:
  - [x] fusion matcher,
  - [x] hazard analysis,
  - [x] symbol/bind merge,
  - [x] deterministic output naming.
- [x] Contract tests:
  - [x] compile-time-only schedule lookup,
  - [x] no runtime fusion path,
  - [x] registry completeness for fused kernels.
- [x] Numerical regression tests:
  - [x] `Off` vs `Safe` parity,
  - [x] `Safe` vs `Aggressive` parity where expected.
- [ ] Performance checks:
  - [x] dispatch count reduction (compile-time schedule tests now validate `Off -> Safe` update dispatch drop of 2 and `Safe -> Aggressive` drop of 2 for incompressible coupled; runtime dispatch-counter test also confirms `Aggressive` executes fewer kernel-graph dispatches than `Safe`),
  - [ ] no regression in wall-clock for representative cases.

## 9) Rollout and Cleanup

- [ ] Roll out in small PR sequence (scaffolding -> pass -> pilot -> expansion).
- [ ] Keep feature-gated fallback until pilot and validation matrix are green.
- [ ] Remove obsolete handwritten fused kernel code after migration completion.
- [ ] Update docs for model authors (how to define fusion-capable kernels/rules).
- [ ] Add troubleshooting notes for common fusion synthesis failures.

## 10) Exit Criteria (Definition of Done)

- [x] Fused kernels are generated from DSL at compile-time (not handwritten WGSL bodies).
- [x] Fusion-capable kernels are encoded through structured DSL/AST builders, not manual `Vec<String>` sections.
- [ ] Runtime remains registry-driven and model-driven, with no ad-hoc fusion logic.
- [x] At least one production fusion path (Rhie-Chow) fully migrated and validated.
- [x] Safe and Aggressive policy semantics are implemented, tested, and documented.
- [ ] Full validation matrix passes in CI.

## 11) Concrete Kernel Fusion Worklist

- [x] `dp_update_from_diag` + `rhie_chow/store_grad_p` (Safe and Aggressive via `rhie_chow:dp_update_store_grad_p_v1`).
- [x] `dp_update_from_diag` + `rhie_chow/store_grad_p` + `rhie_chow/grad_p_update` (Aggressive-only via `rhie_chow:dp_update_store_grad_p_grad_p_update_v1`).
- [x] `dp_update_from_diag` + `rhie_chow/store_grad_p` + `rhie_chow/grad_p_update` + `rhie_chow/correct_velocity_delta` (Aggressive-only via `rhie_chow:dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1`).
- [x] `dp_init` + `dp_update_from_diag` + `rhie_chow/store_grad_p` + `rhie_chow/grad_p_update` + `rhie_chow/correct_velocity_delta` (Aggressive-only via `rhie_chow:dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1`).
- [x] `rhie_chow/store_grad_p` + `rhie_chow/grad_p_update` as a standalone declared pair rule (Aggressive-only via `rhie_chow:store_grad_p_grad_p_update_v1`).
- [x] `rhie_chow/grad_p_update` + `rhie_chow/correct_velocity_delta` as a standalone declared pair rule (Aggressive-only via `rhie_chow:grad_p_update_correct_velocity_delta_v1`).
- [x] `generic_coupled_update` + `dp_init` (Safe-only in coupled stepping via `generic_coupled:update_dp_init_v1`, guarded with `ExactPolicy(Safe)`).
- [x] `generic_coupled_assembly` + `generic_coupled_assembly_grad_state` DSL migration prerequisite completed (both kernels now emitted as DSL artifacts; rule declaration remains optional because conditions are mutually exclusive at runtime).

## 12) Fusion Opportunity Reassessment (Post Assembly DSL Migration)

- [x] Regenerate `FUSION_COVERAGE.md` after migrating `generic_coupled_assembly*` to DSL.
- [x] Reconfirm active Rhie-Chow fusion opportunities and rule synthesis status.
- [x] Reconfirm `generic_coupled_assembly -> generic_coupled_assembly_grad_state` is technically synthesizeable (`Safe` and `Aggressive`) once both kernels are DSL.
- [x] Migrate `generic_coupled_update` to DSL and regenerate coverage (legacy DSL-migration-eligible per-model kernels now reduced to zero).
- [x] Decide whether to declare an explicit assembly-pair fusion rule despite mutually-exclusive kernel conditions (decision: no explicit rule; runtime kernel conditions are mutually exclusive so the pair is never adjacent in an active schedule).

## 13) Dispatch-Reduction Investigation (Post DSL-Driven Fusion)

- [x] Reassess the next update-phase fusion candidate for dispatch-count reduction: `generic_coupled_update` + `dp_init` + `dp_update_from_diag` + `rhie_chow/store_grad_p` + `rhie_chow/grad_p_update` + `rhie_chow/correct_velocity_delta`.
- [x] Prototype the aggressive full-chain candidate and run parity checks (`tests/rhie_chow_fusion_parity_test.rs`) plus schedule dispatch checks.
- [x] Document outcome: candidate can synthesize, but is currently **rejected** for rollout because parity regresses (`safe` vs `aggressive` mismatch above tolerance) when dispatch boundaries are removed before neighbor-dependent gradient updates.
- [x] Reconfirm current stable update dispatch floor for coupled incompressible path: `Off=6`, `Safe=4`, `Aggressive=2` (best stable schedule today).
- [x] Investigate single-submission outer iterations (FGMRES-restart style batching analogy).
- [x] Document current blockers for one-submission outer loop:
  - Outer-loop control in `src/solver/gpu/recipe.rs` is host-interleaved (`coupled:before_iter`, `coupled:solve`).
  - Program executor in `src/solver/gpu/program/plan.rs` executes `Host` nodes between graph nodes every outer iteration.
  - Coupled solve and convergence/break decisions in `src/solver/gpu/lowering/programs/generic_coupled.rs` require host-visible solver stats/readback (`host_solve_linear_system`, `host_after_solve`, `repeat_break`).
- [ ] Future path to true one-submission outer loops (not implemented yet):
  - Move outer convergence evaluation and break signaling to GPU-visible buffers.
  - Provide a GPU-driven outer-iteration loop primitive (or fixed-iteration batch mode) that avoids per-iteration host branching.
  - Keep feature-gated fallback to current host-driven loop until numerical parity and diagnostics are preserved.

## 14) One-Submission Outer Loop Plan (Execution Roadmap)

### Phase 1: Control-Mode Scaffolding (fixed outer count vs adaptive break)

- [x] Add a coupled outer-loop runtime mode switch so we can run fixed outer iterations without adaptive host break logic.
- [x] Wire mode through named params and solver helpers as `outer_fixed_iterations_mode` (bool).
- [x] Keep default behavior unchanged (`adaptive` break remains default).
- [x] Add regression coverage that fixed-iteration mode executes all configured outer iterations.

### Phase 2: GPU-Visible Outer Convergence State

- [x] Allocate/write a GPU-visible outer-convergence status buffer (instead of host-only `repeat_break` decisions).
- [x] Move convergence metric reduction and tolerance check to GPU kernels.
- [x] Preserve host-readable diagnostics by optional post-step readback.

### Phase 3: Batched Outer-Loop Program Form

- [x] Add fixed-iteration batched-tail scaffolding (`outer_batched_mode`) using `coupled:batch_tail` to execute remaining outer iterations inside one host op.
- [x] Keep host-driven path as fallback and default until parity/perf gates pass.
- [x] Ensure recipe/program-spec wiring can select host-driven vs batched path deterministically.
- [x] Measure current impact of batched-tail scaffolding: fixed outer-iteration counts are preserved, kernel-graph dispatch count does not increase in batched mode, and queue submissions are reduced in measured runtime coverage (default validated path: `non_batched=178`, `batched=172`, `delta=6` in `tests/rhie_chow_fusion_parity_test.rs`; experimental full one-submission path can be opt-in tested separately).
- [x] Add a batched coupled program path that runs outer iterations as a fixed GPU-driven batch (`coupled:before_iter` and `coupled:batch_tail` both have a guarded full one-submission implementation behind `CFD2_ENABLE_FULL_ONE_SUBMISSION_OUTER=1`, with fallback to the validated batched-tail path by default).

### Phase 4: Toward True One-Submission Outer Step

- [x] Refactor linear solve entrypoints to support encode-only solve passes suitable for inclusion in a batched submission path (FGMRES restart-body encode/finalize API is now available via `KrylovSolveModule::{encode_solve_once, finish_encoded_solve_once}`).
- [x] Collapse per-iteration assembly/update/solve into a single submission for fixed-iteration mode (implemented as an opt-in experimental path behind `CFD2_ENABLE_FULL_ONE_SUBMISSION_OUTER=1`; fallback path remains default).
- [x] Add queue-submission instrumentation (`submission_counter`) and runtime coverage proving batched mode does not increase submissions (default validated path: `178 -> 172`, delta `6`; experimental opt-in path reaches `178 -> 4` in targeted coverage).
- [x] Add hard validation gates: numerical parity, dispatch/submission counters, OpenFOAM drift non-regression (`tests/rhie_chow_fusion_parity_test.rs` now includes fixed-batched vs fixed-nonbatched snapshot parity + low submission-budget assertion, and `scripts/run_one_submission_hard_gates.sh` runs parity/counter checks plus OpenFOAM diagnostic diff against a provided baseline metrics file).

#### Phase 4A: Detailed Execution Plan (Current)

- [x] Add reusable graph-encoding API (`ModuleGraph::encode_into`) so multiple graph segments can be recorded into one command encoder.
- [x] Start collapsing graph-only submissions in batched tail by pipelining `update(i)` with `assembly(i+1)` into one submission.
- [x] Extend preconditioner setup to encode into caller-provided command encoders (avoid per-iteration `prepare` submits in one-submission path): `FgmresPreconditionerModule::encode_prepare` is wired through `KrylovSolveModule::solve_once_with_prepare`, and runtime/schur preconditioners now provide encode-based setup paths.
- [x] Extend FGMRES residual-seed/normalization path with encode-only variants (remove host-side residual submit/readback before restart-body encode) and fix command-buffer ordering hazards by using in-encoder buffer copies for seed/solver params/scalars and restart setup tables/indirect args in `src/solver/gpu/linear_solver/fgmres.rs`.
- [x] Eliminate remaining race-like ordering risk in encoded FGMRES setup by writing `params` via in-encoder copies before `encode_prepare` and before restart-body encode (`src/solver/gpu/modules/krylov_solve.rs`, `src/solver/gpu/linear_solver/fgmres.rs`); repeated parity/submission probes with fixed settings are deterministic run-to-run.
- [x] Add bounded multi-restart encode chunking for one-submission fixed-iteration solves in `src/solver/gpu/modules/linear_solver.rs` (`CFD2_ONE_SUBMISSION_RESTART_BUDGET` per restart chunk, `CFD2_ONE_SUBMISSION_TOTAL_ITERS` per-solve total budget; optional tuning knobs `CFD2_ONE_SUBMISSION_CHUNKS`, `CFD2_ONE_SUBMISSION_MIN_TAIL`, `CFD2_ONE_SUBMISSION_SOLUTION_OMEGA`, and `CFD2_ONE_SUBMISSION_TAIL_OMEGA`; current default tuning is `12`/`27` for the closest measured strict-parity behavior while preserving one-submission dispatch reduction).
- [x] Add fixed-iteration encoded outer-loop runner that records `assembly + solve + update` for all outer iterations into one command buffer submission.
- [x] Keep a guarded fallback to current batched-tail path until parity and OpenFOAM drift checks remain stable.
- [ ] Close remaining strict parity gap for opt-in encoded-seed / full one-submission path (current targeted parity measurements: encoded-seed batched mode `max_rel=1.659893e-3` vs `1e-3` tolerance; full one-submission mode default best measured `max_rel=1.007846e-3` with `CFD2_ONE_SUBMISSION_RESTART_BUDGET=12` and `CFD2_ONE_SUBMISSION_TOTAL_ITERS=27`; tuned tail-damped variant reaches `max_rel=1.000047e-3` with `CFD2_ONE_SUBMISSION_TAIL_OMEGA=1.02580`, still slightly above `1e-3`).
