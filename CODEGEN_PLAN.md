# CFD2 Codegen + Unified Solver Plan

This file tracks *codegen + solver orchestration* work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering/compilation path (no compressible/incompressible-specific compiler/lowerer forks)
- a single runtime orchestration path (no per-family solver loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- numerical schemes + auxiliary computations derived automatically from `ModelSpec` + solver config

## Status
- The solver entrypoint is unified (`GpuUnifiedSolver`), but there are still **three plan families** under the hood: incompressible (`GpuSolver`), compressible (`CompressiblePlanResources`), and generic coupled (`GpuGenericCoupledSolver`).
- Lowering is routed through `src/solver/gpu/lowering/mod.rs`, and both solver config and the caller-provided `ModelSpec` are now passed into plan construction.
- Orchestration is increasingly plan-driven (`ExecutionPlan`/`ModuleGraph`), and shared helpers are replacing per-plan boilerplate.
- `ModelGpuProgramSpec` + `GpuProgramPlan` now exist and are used for **generic coupled** models (first adopter of the “universal plan runtime” idea), including state-buffer access and state writes via the spec.

## Assessment (Are We Approaching The Goal?)
- Yes: the public surface is unified, config flows into lowering, and the runtime is moving toward module-graph execution with shared Krylov/FGMRES helpers.
- Not yet: we still have solver-family resource containers, solver-family kernel binding/dispatch wiring, and multiple lowering/initialization paths. Generated WGSL exists, but the Rust-side binding/plumbing is not model-driven.

## Remaining Gaps (Concrete)
- `ModelGpuProgramSpec` exists but is only adopted by generic coupled; compressible/incompressible still build solver-family structs directly.
- Kernel wiring is still handwritten per plan (bind group creation, pipeline selection, ping-pong choices, and pass ordering).
- “Modules own their own resources” is only partially true; many pipelines/bind groups still live on solver-family structs.
- Generic coupled remains intentionally incomplete (limited terms/BCs), and scheme expansion is not yet driving required auxiliary passes automatically.

## Next Steps (Prioritized)
1. **Introduce `ModelGpuProgramSpec`**
   - Expand `ModelGpuProgramSpec` beyond the generic-coupled prototype: unify “time/constants source”, state-buffer selection, and linear-solve ports.
   - Keep generated-per-model WGSL, but move solver-family-specific binding/dispatch wiring into the program spec layer.
2. **Create a single universal GPU plan runtime**
   - Make `GpuProgramPlan` the only runtime plan type used by lowering.
 - Plans become *data* (spec + module registry), not code with bespoke control flow.
3. **Migrate plans iteratively**
   - Done: `GpuGenericCoupledSolver` → `GpuProgramPlan`.
   - Next: compressible → `GpuProgramPlan`, then incompressible → `GpuProgramPlan`.
   - Delete the corresponding legacy plan structs after each migration.
4. **Elevate “first-class modules”**
   - Krylov / preconditioners / AMG become pluggable modules with explicit ports and self-owned resources.
   - Lowering composes modules; modules do not construct each other’s buffers/bindings.
5. **Drive scheme expansion end-to-end**
   - Scheme selection expands to required auxiliary passes (gradients/reconstruction/history) for arbitrary fields/models.
6. **Unify codegen/lowering**
   - Remove separate compressible/incompressible lowerers/compilers; keep only the model-driven `ModelGpuProgramSpec` path.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
