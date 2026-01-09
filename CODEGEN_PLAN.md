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
- Codegen emits WGSL at build time, but Rust-side pipeline/bind-group wiring is still handwritten and often solver-family-specific.
- Several critical kernels are still handwritten WGSL (`src/solver/gpu/shaders/*.wgsl`), including solver-family-specific ones (`compressible_*`, `schur_precond`).

## Main Blockers
- Program schedules now use typed op-kinds, but op dispatch is still centralized in per-family dispatchers rather than module-owned `OpKind` implementations (no module registry yet).
- Solver-family resource containers (`GpuSolver`, `CompressiblePlanResources`, `GenericCoupledProgramResources`) still own most pipelines/bind groups and dictate wiring.
- Kernel selection still relies on manual dispatch logic (e.g. string-based model-id matches in `src/solver/gpu/lowering/model_driven.rs`) instead of a generated registry keyed by `ModelSpec`/`KernelPlan`.
- Scheme expansion and aux-pass discovery are not yet driving required buffers/kernels/schedule end-to-end.

## Recently Implemented
- Single template-driven lowering entrypoint: `src/solver/gpu/lowering/model_driven.rs`.
- Centralized schedules + typed op kinds: `src/solver/gpu/lowering/templates.rs`.
- Removed per-family lowering entrypoints (`lower_parts`); model-specific code is now runtime hooks/providers only.
- Replaced fn-pointer op registries with typed op kinds + dispatcher (`GraphOpKind`/`HostOpKind`/`CondOpKind`/`CountOpKind` + `ProgramOpDispatcher`).
- Introduced shared GPU resource modules for solver state and constants:
  - `PingPongState` (triple-buffer + shared step index) and `ConstantsModule` (CPU copy + uniform buffer).
  - Compressible/incompressible/generic-coupled paths now share the same state+constants abstraction (no per-family ping-pong bookkeeping).

## Next Steps (Prioritized)
1. **Finish module-owned resources (beyond state/constants)**
   - Move remaining solver-owned buffers (linear solver scratch, flux/gradient buffers, BC tables, etc.) behind module boundaries with explicit `PortSpace` contracts.
   - Introduce a small time-integration module that owns history rotation + `dt/dt_old/time` semantics so host ops stop doing manual book-keeping.
     - Fix `dt_old` semantics: it should always represent the **previous step's** `dt` and be updated by the temporal scheme, not by ad-hoc heuristics in constants writes.
2. **Generate kernel bindings + dispatch**
   - Emit binding metadata alongside WGSL (ports/resources per bind group) and add a generic bind-group builder.
   - Replace string-based per-model kernel selection with a generated registry (`ModelSpec.id` + kernel kind -> pipeline/bind builders).
3. **Eliminate handwritten WGSL (incremental)**
   - Start with solver-family-specific shaders (`compressible_*`, `schur_precond`) and migrate them into the codegen WGSL pipeline.
   - Keep handwritten WGSL only as a temporary bootstrapping layer.
4. **Replace `PlanParam` with typed config deltas**
   - Generate `SolverConfigDelta` + module-specific deltas from `ModelSpec` + module capabilities; remove ad-hoc host callbacks.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
