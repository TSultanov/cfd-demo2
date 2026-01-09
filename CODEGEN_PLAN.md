# CFD2 Codegen + Unified Solver Plan

This file tracks *codegen + solver orchestration* work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering/compilation path (no compressible/incompressible-specific compiler/lowerer forks)
- a single runtime orchestration path (no per-family solver loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- numerical schemes + auxiliary computations derived automatically from `ModelSpec` + solver config

## Current State
- One unified runtime plan type: `GpuProgramPlan` + `ModelGpuProgramSpec`.
- `GpuPlanInstance` is implemented only by `GpuProgramPlan`; solver-family structs (`GpuSolver`, `CompressiblePlanResources`) are internal resources.
- One unified lowering entrypoint: `src/solver/gpu/lowering/mod.rs` `lower_program(...)`.
- Program runtime supports control flow (`If`, `Repeat`, `While`) + graph dispatch + host nodes.
- Compressible stepping (explicit + implicit outer loop) is expressed as a program schedule (graphs + host nodes), not legacy `step_with_stats()`.
- Incompressible coupled stepping is expressed as a program schedule (graphs + host nodes + outer-loop control flow), not legacy `step_coupled_impl()`.

## Assessment (Are We Approaching The Goal?)
- Yes: the public surface is unified, config flows into lowering, and the runtime is moving toward module-graph execution with shared Krylov/FGMRES helpers.
- Not yet: we still have solver-family resource containers and solver-family kernel binding/dispatch wiring. Generated WGSL exists, but the Rust-side binding/plumbing is not model-driven.

## Remaining Gaps (Concrete)
- `ModelGpuProgramSpec` is still mostly **function-pointer glue** over solver-family containers rather than a first-class “ports + module graph + dispatch plan” spec.
- Kernel wiring is still handwritten per plan (bind group creation, pipeline selection, ping-pong choices, and pass ordering).
- “Modules own their own resources” is only partially true; many pipelines/bind groups still live on solver-family structs.
- Generic coupled remains intentionally incomplete (limited terms/BCs), and scheme expansion is not yet driving required auxiliary passes automatically.
- `src/solver/gpu/plans/plan_instance.rs` still exposes a “kitchen sink” interface; several methods/params only make sense for some plan instances. This should become a minimal, universally meaningful interface with optional capabilities expressed via `ModelGpuProgramSpec`.

## Next Steps (Prioritized)
1. **Normalize plan instance interfaces**
   - Shrink `GpuPlanInstance` to a minimal universally meaningful surface (time/dt/state IO + stepping).
   - Express optional capabilities (history init, profiling hooks, linear debug, step stats) via `ModelGpuProgramSpec` / capability traits rather than mandatory methods/params.
2. **Delete migrated legacy plan types**
   - Once incompressible/compressible schedules live in specs, remove `GpuSolver`/`CompressiblePlanResources` as `GpuPlanInstance` implementations (keep only as internal lowered resources/modules).
3. **Elevate “first-class modules”**
   - Krylov / preconditioners / AMG become pluggable modules with explicit ports and self-owned resources.
4. **Reduce handwritten kernel plumbing**
   - Move bind-group/pipeline construction toward “generated bindings + generic port wiring” so model-specific code becomes mostly “declare ports + compose module graphs”.
5. **Drive scheme expansion end-to-end**
   - Scheme selection expands to required auxiliary passes (gradients/reconstruction/history) for arbitrary fields/models.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
