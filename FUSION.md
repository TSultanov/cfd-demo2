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
  - [ ] dispatch count reduction (via dispatch counter),
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
- [x] `rhie_chow/store_grad_p` + `rhie_chow/grad_p_update` as a standalone declared pair rule (Aggressive-only via `rhie_chow:store_grad_p_grad_p_update_v1`).
- [x] `rhie_chow/grad_p_update` + `rhie_chow/correct_velocity_delta` as a standalone declared pair rule (Aggressive-only via `rhie_chow:grad_p_update_correct_velocity_delta_v1`).
- [x] `generic_coupled_assembly` + `generic_coupled_assembly_grad_state` DSL migration prerequisite completed (both kernels now emitted as DSL artifacts; rule declaration remains optional because conditions are mutually exclusive at runtime).

## 12) Fusion Opportunity Reassessment (Post Assembly DSL Migration)

- [x] Regenerate `FUSION_COVERAGE.md` after migrating `generic_coupled_assembly*` to DSL.
- [x] Reconfirm active Rhie-Chow fusion opportunities and rule synthesis status.
- [x] Reconfirm `generic_coupled_assembly -> generic_coupled_assembly_grad_state` is technically synthesizeable (`Safe` and `Aggressive`) once both kernels are DSL.
- [ ] Decide whether to declare an explicit assembly-pair fusion rule despite mutually-exclusive kernel conditions.
- [ ] Migrate `generic_coupled_update` to DSL to unlock reassessment of additional update-phase fusion opportunities.
