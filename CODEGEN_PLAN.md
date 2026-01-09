# CFD2 Codegen + Unified Solver Plan

This file tracks *remaining* codegen + solver orchestration work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering + codegen path (no separate compressible/incompressible lowerers/compilers)
- a single runtime orchestration path (no per-family step loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- kernels + auxiliary passes derived automatically from `ModelSpec` + solver config (no handwritten WGSL)

## Status (Reality Check)
- Runtime orchestration is largely unified via `GpuProgramPlan` + `ProgramSpec`, and lowering now routes through a single entrypoint (`src/solver/gpu/lowering/model_driven.rs`) with template schedules in `src/solver/gpu/lowering/templates.rs`.
- Op dispatch is now **registry-only**: lowering builds a `ProgramOpRegistry` via per-model `register_ops(...)`, validates that every `ProgramSpecNode` has a handler, and runs directly through the registry (no legacy/hybrid fallback).
- Codegen emits WGSL at build time, and Rust-side pipeline/bind-group wiring is becoming metadata-driven (build-generated binding metadata + generic bind-group builder), but many modules still own bespoke resource-resolution glue.
- Several critical kernels are still handwritten WGSL (`src/solver/gpu/shaders/*.wgsl`), including solver-family-specific ones (`compressible_*`, `schur_precond`).
- Kernel lookup is now unified via `kernel_registry::kernel_source(model_id, KernelKind)` (generated shader strings + binding metadata + pipeline constructors), and multiple pipelines now consume it directly.
- Time integration semantics are now centralized via `TimeIntegrationModule` (`time/dt/dt_old/step_count`) so host ops/plans stop doing ad-hoc bookkeeping.

## Main Blockers
- Op dispatch is still **structured by solver family** (`GraphOpKind`/`HostOpKind` enums and per-family resource containers). The registry exists and is validated, but we still lack a composable “module registry” and module-owned op IDs.
- Solver-family resource containers (`GpuSolver`, `CompressiblePlanResources`, `GenericCoupledProgramResources`) still own most pipelines/bind groups and dictate wiring.
- Kernel lookup is unified and build-generated, but it still relies on build-time tables (i.e. not yet derived automatically from scheme expansion / `KernelPlan`).
- Scheme expansion and aux-pass discovery are not yet driving required buffers/kernels/schedule end-to-end.

## Recently Implemented
- Single template-driven lowering entrypoint: `src/solver/gpu/lowering/model_driven.rs`.
- Centralized schedules + typed op kinds: `src/solver/gpu/lowering/templates.rs`.
- Removed per-family lowering entrypoints (`lower_parts`); model-specific code is now runtime hooks/providers only.
- Phase 1 op-dispatch registry refactor:
  - `ProgramOpRegistry` in `src/solver/gpu/plans/program.rs`.
  - Per-model `register_ops(...)` in `src/solver/gpu/lowering/models/*`.
  - Lowering wires the registry in `src/solver/gpu/lowering/model_driven.rs`.
- Phase 2 op-dispatch unification:
  - `ProgramSpec` validation (missing handler detection) and removal of legacy/hybrid fallback.
  - Unit tests covering validation behavior.
- Introduced shared GPU resource modules for solver state and constants:
  - `PingPongState` (triple-buffer + shared step index) and `ConstantsModule` (CPU copy + uniform buffer).
  - Compressible/incompressible/generic-coupled paths now share the same state+constants abstraction (no per-family ping-pong bookkeeping).
- Time integration module:
  - `TimeIntegrationModule` owns `time/dt/dt_old/step_count` and updates `ConstantsModule`.
  - All solver families now advance time at step prepare and rotate `dt_old` at step finalization through the module.
- Linear-solver/preconditioner pluggability groundwork:
  - `FgmresPreconditionerModule::prepare(...)` hook (default no-op) so preconditioners can own their “build” phase.
- Unified kernel lookup + binding metadata (build time):
  - `kernel_registry::kernel_source(model_id, KernelKind)` provides shader + pipeline + binding metadata for generated kernels.
  - Bind groups can now be constructed from build-generated WGSL binding metadata via `src/solver/gpu/wgsl_reflect.rs`.
- Fully codegen-driven kernel registry mapping:
  - `build.rs` generates `kernel_registry_map.rs` with complete `KernelKind -> (shader, pipeline, bindings)` mapping.
  - `kernel_registry.rs` now uses `generated::kernel_entry(kind)` eliminating all hand-written match statements for non-generic kernels.
  - Generic coupled kernels remain model-id-based via `generic_coupled_pair(model_id)`.

## Next Steps (Prioritized)
1. **Make op dispatch truly module-owned (beyond Phase 2)**
   - Replace global `GraphOpKind`/`HostOpKind` enums with module-owned IDs (newtypes or namespaced IDs) so adding a module doesn’t require editing a central enum.
   - Introduce a small `ModuleRegistry` concept (composition): modules register ops *and* declare what resources/ports they provide.
   - (Optional) generate op IDs and registrations from codegen output to reduce central wiring.

2. **Finish module-owned resources (beyond state/constants/time)**
   - Move remaining solver-owned buffers (linear solver scratch, flux/gradient buffers, BC tables, etc.) behind module boundaries with explicit `PortSpace` contracts.
   - Extend `TimeIntegrationModule` to cover restart/init + scheme-specific history rotation (e.g. BDF2 bootstrap) without duplicating time advancement.

3. **Expand registry-driven kernel wiring**
   - ✅ Generate the `KernelKind -> KernelSource` mapping from codegen output (hand-written match eliminated).
   - Derive kernel registry entries from `KernelPlan` / scheme expansion so adding a kernel doesn’t require updating build-time tables.
   - Extend binding metadata from `{group, binding, name}` to explicit `PortSpace`/resource contracts so modules can resolve resources without string matching.
   - Endgame: generate specialized kernel sets per (temporal scheme + spatial scheme) choice, each with its own optimized memory layout, rather than a single generic layout + `constants.scheme/time_scheme` runtime switching.

4. **Spatial/temporal scheme pluggability (end-to-end)**
   - Use scheme expansion (`expand_schemes`) to drive:
     - required aux buffers (e.g. gradients),
     - required aux passes,
     - required kernel variants,
     - schedule structure (not just runtime `constants.scheme`).
   - Stop using “scheme implies gradients” as an ad-hoc per-plan heuristic; make it a lowering-time contract.

5. **Linear solver + preconditioner pluggability (end-to-end)**
   - Unify the FGMRES “driver” logic (restart loop, residual checks, timing) into a shared module so coupled/compressible don’t diverge.
   - Move solver selection to a registry/factory keyed by typed config (no strings), and let solvers/preconditioners own their scratch buffers.

6. **Eliminate handwritten WGSL (incremental)**
   - Start with solver-family-specific shaders (`compressible_*`, `schur_precond`) and migrate them into the codegen WGSL pipeline.
   - Keep handwritten WGSL only as a temporary bootstrapping layer.

7. **Replace `PlanParam` with typed config deltas**
   - Generate `SolverConfigDelta` + module-specific deltas from `ModelSpec` + module capabilities; remove ad-hoc host callbacks.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
