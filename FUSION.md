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
- All DSL-eligible model kernels (9 of 12) are migrated; 3 WGSL-only kernels remain by
  design (different dispatch domain or phase). `packed_state_gradients` was the latest
  migration (Gradients phase, Cells dispatch).
- 8 synthesized fused WGSL shaders exist for `incompressible_momentum`, plus 1 cross-phase
  fused shader (`packed_state_gradients + assembly_grad_state`) generated for all 4 models.
- Cross-phase fusion (Gradients → Assembly) is supported via per-atom phase tags in
  `KernelPatternAtom` and `BindingRemap` for binding layout harmonization.
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

- [x] Add at least a warning-level hazard report for Aggressive policy fusions (log
  detected hazards without rejecting, so authors have visibility).
  **Implemented:** Extracted `detect_hazards()` as a public function returning a
  `HazardReport` (with `HazardKind` enum: RAW/WAR/WAW/BarriersOrAtomics).
  `synthesize_fused_program_with_report()` exposes hazard info alongside the fused
  program. `ensure_safe_composition()` now delegates to `detect_hazards()` internally.
  9 new unit tests cover all hazard kinds and report formatting.
- [x] Consider adding a `--aggressive-hazard-report` flag to `fusion_coverage` binary.
  **Implemented:** `fusion_coverage --aggressive-hazard-report` renders per-rule hazard
  reports for all aggressive-policy fusions.

### 2b) No Cross-Kernel Value Forwarding (P3 — optimization opportunity)

Kernel bodies are stored as `Vec<String>` (WGSL text lines). When kernel A writes a
value that kernel B reads (from the same buffer slot), the fused kernel still performs
a memory round-trip (store then load). A proper AST-based body representation would
enable eliminating redundant loads/stores across fused kernel boundaries.

- [x] Evaluate the cost/benefit of replacing `Vec<String>` body with a typed expression
  AST (the face-expression AST `FaceScalarExpr`/`FaceVec2Expr` already exists in the IR
  but is not used for kernel bodies).
  **Evaluation result:** Only 3 of 7 fusion rules have cross-kernel store-load
  forwarding opportunities (all aggressive-only). Maximum 5 scalar `f32` store-load
  pairs could be eliminated, but these values are almost certainly in L1 cache after
  the preceding store, so the performance benefit is negligible.
  `FaceScalarExpr`/`FaceVec2Expr` are face-centric and unsuitable as general body AST.
  Migrating `body: Vec<String>` to `body: Vec<Stmt>` would be a deep structural change
  touching every kernel generator. **Cost far exceeds benefit — deferred.**
- [ ] If adopted in the future, implement a simple load-after-store elimination pass in
  `synthesize_fused_program`. A lightweight string-pattern rewrite pass (~50–100 lines)
  could achieve the same result without AST migration if the need arises.

### 2c) Binding Access Promotion (P3)

`merge_bindings()` originally rejected bindings where one kernel reads a slot and another
writes the same slot with different `BindingAccess`. Promotion logic has now been
implemented.

- [x] Evaluate whether any real fusion candidate is blocked by this (check
  `FUSION_COVERAGE.md` for bind-merge rejections vs hazard rejections).
  **Result: The cross-phase `packed_state_gradients + assembly_grad_state` fusion
  requires access promotion** — `state` at (1,0) is ReadOnly in gradients and ReadWrite
  in assembly. Without promotion, this fusion would be rejected.
- [x] Implement promotion logic with clear documentation of safety guarantees.
  **Implemented:** `promote_access()` helper in `fusion.rs` promotes `ReadOnly` +
  `ReadWrite` → `ReadWrite` when both kernels reference the same `(group, binding)`
  slot. The later kernel's access mode is used. `merge_bindings()` calls
  `promote_access()` instead of rejecting mismatches.

### 2d) EOS Parameter Detection Is Fragile (P3)

`constants_extra_params_for_program()` in `fusion.rs:488` scans for hardcoded EOS
field names (`eos.gamma`, etc.) via string containment. Adding new EOS parameters
requires updating this function.

- [x] Replace string-scan heuristic with a structured EOS parameter declaration in the
  `KernelProgram` IR.
  **Implemented:** Added `eos_params: Vec<ParamSpec>` field to `KernelProgram`.
  Generators (`generate_generic_coupled_update_kernel_program`,
  `generate_unified_assembly_kernel_program`) now populate `eos_params`.
  Fusion synthesis merges `eos_params` via `merge_eos_params()`.
  `lower_kernel_program_to_wgsl` prefers `program.eos_params` when non-empty,
  falling back to the legacy string-scan for backward compatibility.

## 3) Fusion Coverage Expansion

### 3a) Compressible Model Fusion Opportunities — CLOSED

The compressible model's only same-phase fuseable pair (`assembly` / `assembly_grad_state`)
uses mutually-exclusive runtime conditions (`RequiresNoGradState` / `RequiresGradState`),
so they are never adjacent in an active schedule. No same-phase fusion is possible.

However, cross-phase fusion (see 3b below) identified a viable opportunity:
`packed_state_gradients` (Gradients) → `assembly_grad_state` (Assembly). These are
directly adjacent in all `has_grad_state=true` schedules.

- [x] Evaluate whether the compressible model has other fusion opportunities beyond
  the mutually-exclusive `assembly + assembly_grad_state` pair.
  **Result:** The only viable pair is `packed_state_gradients + assembly_grad_state`
  (cross-phase, Gradients → Assembly). The compressible model has only one
  update-phase kernel (`generic_coupled_update`), so no update-phase chaining is
  possible. Implemented as fusion rule `generic_coupled:gradients_assembly_grad_state_v1`.
- [x] Document the decision not to fuse `assembly + assembly_grad_state`.
  **Documented here:** These kernels have mutually-exclusive `RequiresNoGradState` /
  `RequiresGradState` guards, making them never co-present in a schedule. Fusion is
  structurally impossible.

### 3b) Cross-Phase Fusion — CLOSED

Cross-phase fusion has been investigated, implemented, and validated.

**Findings:**
- Gradients + Assembly fusion (`packed_state_gradients` + `assembly_grad_state`) is
  feasible and safe. These kernels are directly adjacent in all `has_grad_state=true`
  schedules for all 4 models (compressible, incompressible_momentum,
  generic_diffusion_demo, generic_diffusion_demo_neumann).
- Assembly + FluxComputation fusion remains blocked by dispatch domain mismatch
  (Cells vs Faces).

**Implementation (3 sub-tasks):**

- [x] **3b-1: DSL migration of `packed_state_gradients`.** Migrated the WGSL kernel to
  a `KernelProgram` generator (`generate_packed_state_gradients_kernel_program()` in
  `packed_state_gradients.rs`). Generated WGSL is cosmetically different but
  semantically identical to the original.
- [x] **3b-2: Cross-phase pattern matching.** Added per-atom `phase: Option<KernelPhaseId>`
  field to `KernelPatternAtom` and relaxed the same-phase constraint in `rule_matches_at()`.
  The IR-level `ensure_safe_composition()` already did not check phase — only dispatch
  domain, launch semantics, and indexing.
- [x] **3b-3: Binding layout harmonization via `BindingRemap`.** The two kernels use
  incompatible binding layouts (gradients: bc at group 2, grad_state at (1,4); assembly:
  bc at group 3, grad_state at (1,5), matrix/rhs at group 2). Added `BindingRemap` struct
  and `apply_binding_remaps()` to fusion.rs. The fusion rule remaps gradients bindings
  to assembly's layout before `merge_bindings`. Access mode promotion (`ReadOnly` +
  `ReadWrite` → `ReadWrite`) was also added to `merge_bindings()`.
- [x] **3b-4: Fusion rule declaration.** Rule
  `generic_coupled:gradients_assembly_grad_state_v1` declared in `generic_coupled.rs`
  with `MinPolicy(Safe)` + guards for `RequiresGradState`, `RequiresStepping(Coupled)`,
  and `RequiresModule("generic_coupled")`.

**Validation:**
- Fused WGSL shaders generated for all 4 models with correct binding superset,
  body ordering, and `k1_` symbol renaming.
- All `cargo test` pass (1 pre-existing failure unrelated to fusion).
- OpenFOAM reference metrics are **bit-identical** before and after the change
  (compressible_acoustic u_x max_rel=0.008001 — pre-existing).

## 4) Documentation (P2)

### 4a) Model Author Guide

Model-author documentation is now available for how to:
- Define a fusion-capable kernel (return `ModelKernelArtifact::DslProgram` from generator)
- Declare `SideEffectMetadata` read/write sets
- Write `ModelKernelFusionRule` with guards
- Use the `fusion_coverage` binary for coverage auditing

- [x] Write a model-author guide (e.g., `docs/fusion-authoring.md`) covering the full
  workflow from kernel generator to fusion rule to validated synthesized output.
  *(Added `docs/fusion-authoring.md`.)*

### 4b) Troubleshooting Guide

Troubleshooting notes are now available for fusion synthesis failures (hazard
rejections, bind-merge conflicts, dispatch-domain mismatches).

- [x] Add troubleshooting notes covering:
  - How to read `FUSION_COVERAGE.md` hazard rejection messages
  - How to fix common RAW/WAR/WAW hazard errors
  - How to debug bind-merge incompatibilities
  - How to validate a new fusion rule with the parity test framework
  *(Added `docs/fusion-troubleshooting.md`.)*

## 5) One-Submission Path Gaps

### 5a) Fallback Path After Multi-Submission Loop Removal (P1)

The fallback behavior is now restored: when one-submission cannot be used, the
recipe-level per-iteration loop continues.

- [x] Add a graceful fallback: when one-submission fails, fall back to per-iteration
  recipe-level loop (re-enable `plan.repeat_break = false` so the recipe continues).
  *(Implemented in `host_coupled_batch_tail`: failure path now clears
  `plan.repeat_break` before returning to recipe-driven looping.)*
- [x] Add a test that exercises the fallback path with an unsupported solver config.
  *(Added unit test in `generic_coupled.rs`:
  `batch_tail_fallback_clears_repeat_break_for_unsupported_solver_state`.)*

### 5b) Env-Var Reads in Hot Path (P3)

The `CFD2_ONE_SUBMISSION_*` env vars (`RESTART_BUDGET`, `TOTAL_ITERS`, `CHUNKS`,
`MIN_TAIL`, `CG_CHUNK_SIZE`) are now cached once per process and reused by the
chunked submission functions.

- [x] Cache env-var reads in a `once_cell::sync::Lazy` or similar, or move them to
  solver config / named params.
  *(Implemented via process-lifetime cached `OneSubmissionEnvTunables` in
  `linear_solver.rs` using `std::sync::OnceLock`.)*

### 5c) `CFD2_ENABLE_ENCODED_SEED_BASIS0` Default Policy (P3)

GPU-side `r0 = b - Ax` basis seeding now defaults on for host-driven
multi-outer solves (`outer_iters > 1`), while batched one-submission paths
continue to force encoded seeding.

- [x] Promote `CFD2_ENABLE_ENCODED_SEED_BASIS0` to default-on after confirming parity
  in the host-driven path, or remove the env var entirely.
  **Implemented (scope-limited):** host-driven solves now default to encoded basis
  seeding when `outer_iters > 1` (`generic_coupled.rs`), with env-var override
  preserved (`CFD2_ENABLE_ENCODED_SEED_BASIS0=0` disables, nonzero enables).
  Single-outer implicit solves keep default-off unless explicitly opted in, because
  that path remains more sensitive in OpenFOAM diagnostics.
  Added parity test `host_driven_encoded_seed_basis0_default_on_matches_opt_out`
  in `tests/rhie_chow_fusion_parity_test.rs` (tolerance `1e-2`).

### 5d) Future: GPU-Driven Outer Loop Without Fixed-Iteration Requirement

The one-submission path currently works with both adaptive and fixed-iteration modes.
Future optimization opportunities:

- [x] Move outer convergence evaluation and break signaling to fully GPU-visible buffers.
  **Implemented:** Host-driven adaptive break now evaluates convergence directly from
  GPU-resident `b_delta`/`b_scale` buffers and reads back only `b_break_status`
  (`generic_coupled.rs`, `host_after_solve`). When convergence diagnostics are enabled,
  per-field residual readback is still performed for reporting, but break signaling stays
  buffer-driven.
- [x] Provide a GPU-driven outer-iteration loop primitive that avoids per-iteration host
  branching entirely (currently, adaptive break still requires iteration-counter readback
  after submission).
  **Implemented (experimental):** `CFD2_ENABLE_GPU_OUTER_LOOP_PRIMITIVE=1` enables an
  adaptive one-submission path that skips the `outer_gate:counter_readback` submission,
  so outer-loop progression remains GPU-driven for the full encoded batch.
- [x] Keep feature-gated fallback to host-driven loop until numerical parity and
  diagnostics are preserved.
  **Implemented:** with the primitive enabled, the solver falls back to the recipe-level
  host-driven loop when adaptive gate resources are unavailable or when
  `collect_convergence_stats` is enabled. This preserves diagnostics behavior while the
  primitive remains feature-gated.

## 6) Build and Release

### 6a) Feature-Gated Fallback (P3)

Fusion is controlled by `KernelFusionPolicy` (runtime enum), not Cargo feature flags.
The `Off` policy serves as the opt-out mechanism. No compile-time feature gate exists.

- [x] Decide whether a `--no-default-features` compile-time opt-out is needed for
  environments where even the codegen overhead of fusion synthesis is undesirable.
  **Decision:** not needed — runtime `KernelFusionPolicy::Off` remains the supported
  opt-out and no compile-time fusion feature gate is added.

### 6b) Codegen-Level `FusionPatternRule` Cleanup (P3)

`FusionPatternRule` and `match_fusion_candidates()` in
`crates/cfd2_codegen/src/solver/codegen/fusion.rs` appear unused in production.
The model-level `ModelKernelFusionRule` / `apply_model_fusion_rules()` has replaced
them. They are only used in `fusion.rs` unit tests.

- [x] Mark codegen-level `FusionPatternRule` as `#[cfg(test)]` or document it as
  test-only infrastructure.
  *(Implemented: `FusionPatternRule`, `FusionCandidate`, and
  `match_fusion_candidates()` are now `#[cfg(test)]`.)*
