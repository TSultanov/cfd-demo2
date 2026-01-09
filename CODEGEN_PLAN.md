# CFD2 Codegen + Unified Solver Plan

This file tracks *codegen + solver orchestration* work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering/compilation path (no compressible/incompressible-specific compiler/lowerer forks)
- a single runtime orchestration path (no per-family solver loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- numerical schemes + auxiliary computations derived automatically from `ModelSpec` + solver config

## Status
- The solver entrypoint is unified (`GpuUnifiedSolver`), and **all models now lower to a single runtime plan type**: `GpuProgramPlan` + `ModelGpuProgramSpec`.
- Lowering is routed through `src/solver/gpu/lowering/mod.rs` via a single `lower_program(...)` entrypoint, and both solver config and the caller-provided `ModelSpec` are passed into program construction.
- Orchestration is increasingly plan-driven (`ExecutionPlan`/`ModuleGraph`), and shared helpers are replacing per-plan boilerplate.
- `ModelGpuProgramSpec` currently wraps existing solver-family resource containers (incompressible `GpuSolver`, compressible `CompressiblePlanResources`, generic coupled resources) and forwards `state_buffer`, params, and linear-debug hooks through the spec.
- Compressible stepping (explicit + implicit outer loop) is now expressed as a program schedule (host nodes + module graphs + control flow) instead of calling legacy `step_with_stats()`.

## Assessment (Are We Approaching The Goal?)
- Yes: the public surface is unified, config flows into lowering, and the runtime is moving toward module-graph execution with shared Krylov/FGMRES helpers.
- Not yet: we still have solver-family resource containers and solver-family kernel binding/dispatch wiring. Generated WGSL exists, but the Rust-side binding/plumbing is not model-driven.

## Remaining Gaps (Concrete)
- `ModelGpuProgramSpec` exists, but it is mostly **function-pointer glue** over solver-family containers rather than a first-class “ports + module graph + dispatch plan” spec.
- Kernel wiring is still handwritten per plan (bind group creation, pipeline selection, ping-pong choices, and pass ordering).
- “Modules own their own resources” is only partially true; many pipelines/bind groups still live on solver-family structs.
- Incompressible stepping still happens inside legacy plan code; the runtime does not yet “own” its per-pass schedule.
- Generic coupled remains intentionally incomplete (limited terms/BCs), and scheme expansion is not yet driving required auxiliary passes automatically.

## Next Steps (Prioritized)
1. **Make `ModelGpuProgramSpec` a real program spec**
   - Replace remaining “call legacy `step()`/`step_with_stats()`” with explicit per-pass schedules (module graphs + host nodes) in the spec for incompressible.
   - Goal: the runtime owns pass ordering and dispatch; solver-family code becomes pure “lowering/building modules”, not orchestration.
2. **Delete migrated legacy plan types**
   - Once incompressible/compressible schedules are moved into specs, delete `GpuSolver`/`CompressiblePlanResources` as `GpuPlanInstance` implementations (keep only as internal lowered resources/modules if still needed).
3. **Elevate “first-class modules”**
   - Krylov / preconditioners / AMG become pluggable modules with explicit ports and self-owned resources.
   - Lowering composes modules; modules do not construct each other’s buffers/bindings.
4. **Drive scheme expansion end-to-end**
   - Scheme selection expands to required auxiliary passes (gradients/reconstruction/history) for arbitrary fields/models.
5. **Reduce handwritten kernel plumbing**
   - Move bind-group/pipeline construction toward “generated bindings + generic port wiring”, so model-specific code shrinks to declaring ports and assembling module graphs.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
