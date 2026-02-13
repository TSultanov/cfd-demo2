# DSL-Based Kernel Fusion — Remaining Work and Gaps

This document tracks **remaining gaps and future work** for the kernel fusion system.
All completed items from the original checklist have been removed. What follows is the
set of verified-incomplete tasks and newly discovered gaps.

For full coverage status of all models, see `FUSION_COVERAGE.md`
(`cargo run --bin fusion_coverage -- --write FUSION_COVERAGE.md`).

## Current State (Summary)

The core fusion pipeline is complete and production-default:

- Fusion-capable kernels are encoded as structured `KernelProgram` IR (DSL).
- Fusion synthesis (candidate matching, hazard analysis, binding merge, symbol renaming,
  WGSL lowering) runs at build time in `build.rs`.
- Fusion schedules are precomputed for all `(model, stepping, grad_state, policy)` tuples
  and served at runtime via `fusion_schedule_registry::schedule_for_model`.
- Contract tests enforce: no runtime fusion pass, registry-driven lookup, handwritten fused
  generators removed.
- One-submission outer loop is default-enabled (`DEFAULT_OUTER_BATCHED_MODE = true`) with
  GPU-driven adaptive outer break, FGMRES + CG encoded solver paths, and convergence
  diagnostics readback.
- All DSL-eligible model kernels (8 of 12) are migrated; 4 WGSL-only kernels remain by
  design (different dispatch domain or phase).
- 7 synthesized fused WGSL shaders exist for `incompressible_momentum`.
- Policy semantics (`Off`/`Safe`/`Aggressive`) are implemented and tested.
- Dispatch floor for incompressible coupled: `Off=6`, `Safe=4`, `Aggressive=2` update
  dispatches.
- Numerical parity: fusion policies (Off/Safe/Aggressive) produce **bitwise-identical**
  results across all fields. Batched-vs-non-batched and one-submission-vs-host-driven
  paths have known ~7e-3 grad_p discrepancies from update-ordering differences.

## 1) Validation and Testing Gaps

### 1a) Wall-Clock Performance Regression Checks (P1)

No benchmarks exist comparing fusion policies on representative cases.

- [x] Add a Criterion (or equivalent) benchmark that measures step wall-clock time for
  `Off`, `Safe`, and `Aggressive` policies on the incompressible momentum model.
  *(Added in `benches/fusion_policy_benchmark.rs` — `fusion_policy` benchmark group.)*
- [x] Add a wall-clock benchmark comparing the one-submission path vs the multi-submission
  host-driven path on a representative mesh.
  *(Added in `benches/fusion_policy_benchmark.rs` — `submission_path` benchmark group.)*

### 1b) Cross-Model Fusion Integration Tests (P1)

All runtime parity and dispatch-count tests use `incompressible_momentum` only.
The `compressible` model has a fuseable `assembly → assembly_grad_state` pair
(both DSL, same dispatch domain) that is untested at runtime.

- [x] Add a runtime fusion parity test for the `compressible` model (even though no
  explicit fusion rule is currently declared, verify schedule correctness across policies).
  *(Added in `tests/compressible_fusion_parity_test.rs` — 2 tests: snapshot identity + dispatch count identity.)*
- [x] Add runtime fusion parity tests for `generic_diffusion_demo` /
  `generic_diffusion_demo_neumann` models.
  *(Added in `tests/generic_diffusion_fusion_parity_test.rs` — 4 tests: both models × snapshot + dispatch count.)*

### 1c) Stepping-Mode Coverage (P2)

All runtime parity tests use `Coupled` stepping. Fusion schedules are precomputed for
`Explicit` and `Implicit` stepping too, but never tested at runtime.

- [x] Add a runtime parity test exercising fusion under `Implicit` stepping for a model
  that supports it.
  *(Added 2 tests in `tests/rhie_chow_fusion_parity_test.rs`: Off vs Safe and Safe vs Aggressive under Implicit stepping. Both produce bitwise-identical results.)*
- [x] Verify that `Explicit` stepping schedules produce correct dispatch counts and kernel
  lists (at least at the schedule level, runtime execution may not be feasible for all
  models).
  *(Added 2 unit tests in `fusion_schedule_registry.rs`: `explicit_stepping_schedule_excludes_implicit_only_kernels` and `explicit_stepping_no_fusion_models_identical_across_policies`.)*

### 1e) Tight Parity for Safe Policy (P3)

Safe fusion should be semantically equivalent to unfused execution (identical dispatch
order, just fewer dispatches). ~~Current parity tolerance is `1e-3`.~~

- [x] Investigate whether Safe policy can achieve tighter parity (e.g., `1e-6` or bitwise
  match) and add a tighter gate if so.
  *(Investigation complete: all fusion policy comparisons (Off/Safe/Aggressive) produce **bitwise-identical** results across all fields (u, p, d_p, grad_p_old) under both Coupled and Implicit stepping. Tolerances tightened from `1e-3` to exact `0.0` in all fusion-policy parity tests. The only tests with nonzero error are batched-vs-non-batched and host-driven-vs-one-submission comparisons, which have known ~7e-3 grad_p discrepancies from update-ordering differences, not from fusion.)*

## 2) Fusion Compiler Gaps

### 2a) No Hazard Analysis for Aggressive Policy (P2)

`ensure_safe_composition()` in `crates/cfd2_codegen/src/solver/codegen/fusion.rs:181`
is gated on `policy == Safe`. Under Aggressive, the entire safety check is skipped.
Rule authors must manually guarantee correctness.

- [ ] Add at least a warning-level hazard report for Aggressive policy fusions (log
  detected hazards without rejecting, so authors have visibility).
- [ ] Consider adding a `--aggressive-hazard-report` flag to `fusion_coverage` binary.

### 2b) No Cross-Kernel Value Forwarding (P3 — optimization opportunity)

Kernel bodies are stored as `Vec<String>` (WGSL text lines). When kernel A writes a
value that kernel B reads (from the same buffer slot), the fused kernel still performs
a memory round-trip (store then load). A proper AST-based body representation would
enable eliminating redundant loads/stores across fused kernel boundaries.

- [ ] Evaluate the cost/benefit of replacing `Vec<String>` body with a typed expression
  AST (the face-expression AST `FaceScalarExpr`/`FaceVec2Expr` already exists in the IR
  but is not used for kernel bodies).
- [ ] If adopted, implement a simple load-after-store elimination pass in
  `synthesize_fused_program`.

### 2c) Binding Access Promotion (P3)

`merge_bindings()` rejects bindings where one kernel reads a slot and another writes
the same slot with different `BindingAccess`. A smarter merge could safely promote
`ReadOnlyStorage` to `ReadWriteStorage` when the write is in a later kernel.

- [ ] Evaluate whether any real fusion candidate is blocked by this (check
  `FUSION_COVERAGE.md` for bind-merge rejections vs hazard rejections).
- [ ] If blocking, implement promotion logic with clear documentation of safety
  guarantees.

### 2d) EOS Parameter Detection Is Fragile (P3)

`constants_extra_params_for_program()` in `fusion.rs:388` scans for hardcoded EOS
field names (`eos.gamma`, etc.) via string containment. Adding new EOS parameters
requires updating this function.

- [ ] Replace string-scan heuristic with a structured EOS parameter declaration in the
  `KernelProgram` IR (e.g., an `eos_params_used: BTreeSet<String>` field).

## 3) Fusion Coverage Expansion

### 3a) Compressible Model Has No Fusion Rules (P2)

`FUSION_COVERAGE.md` shows the compressible model has a fuseable `assembly →
assembly_grad_state` pair (both DSL, same dispatch domain, `Cells`), but no rule is
declared. The pair is theoretically safe and aggressive-fuseable.

However, these two kernels use mutually-exclusive runtime conditions
(`RequiresNoGradState` / `RequiresGradState`), so they are never adjacent in an
active schedule. This was a deliberate decision (FUSION.md section 12).

- [ ] Evaluate whether the compressible model has other update-phase fusion opportunities
  (e.g., `generic_coupled_update` + any model-specific update kernels) that are not
  blocked by mutually-exclusive conditions.
- [ ] Document the decision not to fuse `assembly + assembly_grad_state` in the
  compressible model.

### 3b) Cross-Phase Fusion (P3 — future investigation)

Current fusion is limited to same-phase, same-dispatch-domain kernels. No investigation
has been done into:

- Gradients + Assembly phase fusion (e.g., `packed_state_gradients` + `assembly`)
- Assembly + FluxComputation fusion (blocked by dispatch domain: Cells vs Faces)

- [ ] Investigate whether cross-phase fusion is feasible for any model's kernel chain
  (likely requires relaxing the phase-match constraint in the pattern matcher).

## 4) Documentation (P2)

### 4a) Model Author Guide

No documentation exists for model authors on how to:
- Define a fusion-capable kernel (return `ModelKernelArtifact::DslProgram` from generator)
- Declare `SideEffectMetadata` read/write sets
- Write `ModelKernelFusionRule` with guards
- Use the `fusion_coverage` binary for coverage auditing

- [ ] Write a model-author guide (e.g., `docs/fusion-authoring.md`) covering the full
  workflow from kernel generator to fusion rule to validated synthesized output.

### 4b) Troubleshooting Guide

No guide exists for debugging fusion synthesis failures (hazard rejections, bind-merge
conflicts, dispatch-domain mismatches).

- [ ] Add troubleshooting notes covering:
  - How to read `FUSION_COVERAGE.md` hazard rejection messages
  - How to fix common RAW/WAR/WAW hazard errors
  - How to debug bind-merge incompatibilities
  - How to validate a new fusion rule with the parity test framework

## 5) One-Submission Path Gaps

### 5a) Fallback Path After Multi-Submission Loop Removal (P1)

The multi-submission fallback loop in `host_coupled_batch_tail` was removed per the
rollout plan. When `try_host_coupled_batch_tail_one_submission` returns `false`
(unsupported solver config), the code prints a warning but does not execute the
remaining outer iterations. For edge-case solver configurations that don't match
FGMRES, CG, or Schur paths, the solver would silently run only 1 outer iteration.

- [ ] Add a graceful fallback: when one-submission fails, fall back to per-iteration
  recipe-level loop (re-enable `plan.repeat_break = false` so the recipe continues).
- [ ] Add a test that exercises the fallback path with an unsupported solver config.

### 5b) Env-Var Reads in Hot Path (P3)

The `CFD2_ONE_SUBMISSION_*` env vars (`RESTART_BUDGET`, `TOTAL_ITERS`, `CHUNKS`,
`MIN_TAIL`, `CG_CHUNK_SIZE`) are read via `std::env::var()` on every call to the
chunked submission functions (`linear_solver.rs:249-269`, `431-464`, `624`).

- [ ] Cache env-var reads in a `once_cell::sync::Lazy` or similar, or move them to
  solver config / named params.

### 5c) `CFD2_ENABLE_ENCODED_SEED_BASIS0` Still Opt-In (P3)

GPU-side `r0 = b - Ax` basis seeding is validated but default-off for the host-driven
solve path (`generic_coupled.rs:2640-2644`). The batched path always uses it, but the
host-driven path gates it behind both `outer_batched_mode` and the env var.

- [ ] Promote `CFD2_ENABLE_ENCODED_SEED_BASIS0` to default-on after confirming parity
  in the host-driven path, or remove the env var entirely.

### 5d) Future: GPU-Driven Outer Loop Without Fixed-Iteration Requirement

The one-submission path currently works with both adaptive and fixed-iteration modes.
Future optimization opportunities:

- [ ] Move outer convergence evaluation and break signaling to fully GPU-visible buffers
  (partially done via `OuterConvergenceMonitor`).
- [ ] Provide a GPU-driven outer-iteration loop primitive that avoids per-iteration host
  branching entirely (currently, adaptive break still requires iteration-counter readback
  after submission).
- [ ] Keep feature-gated fallback to host-driven loop until numerical parity and
  diagnostics are preserved.

## 6) Build and Release

### 6a) Feature-Gated Fallback (P3)

Fusion is controlled by `KernelFusionPolicy` (runtime enum), not Cargo feature flags.
The `Off` policy serves as the opt-out mechanism. No compile-time feature gate exists.

- [ ] Decide whether a `--no-default-features` compile-time opt-out is needed for
  environments where even the codegen overhead of fusion synthesis is undesirable.
  Current assessment: not needed — `Off` policy is sufficient.

### 6b) Codegen-Level `FusionPatternRule` Cleanup (P3)

`FusionPatternRule` and `match_fusion_candidates()` in
`crates/cfd2_codegen/src/solver/codegen/fusion.rs` appear unused in production.
The model-level `ModelKernelFusionRule` / `apply_model_fusion_rules()` has replaced
them. They are only used in `fusion.rs` unit tests.

- [ ] Mark codegen-level `FusionPatternRule` as `#[cfg(test)]` or document it as
  test-only infrastructure.
