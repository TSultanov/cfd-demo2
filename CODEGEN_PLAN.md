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
- `build_plan_instance(...)` returns a concrete `GpuProgramPlan` (no dyn plan layer).
- Solver-family structs (`GpuSolver`, `CompressiblePlanResources`) are internal resources only (not part of the plan API).
- One unified lowering entrypoint: `src/solver/gpu/lowering/mod.rs` `lower_program(...)`.
- Program runtime supports control flow (`If`, `Repeat`, `While`) + graph dispatch + host nodes.
- Compressible stepping (explicit + implicit outer loop) is expressed as a program schedule (graphs + host nodes).
- Incompressible coupled stepping is expressed as a program schedule (graphs + host nodes + outer-loop control flow).
- Legacy per-family step loops (`step_with_stats`/`step_coupled_impl`) were removed; solver-family structs now exist only as internal resource containers + helpers.

## Assessment (Are We Approaching The Goal?)
- Yes: the public surface is unified, config flows into lowering, and the runtime is moving toward module-graph execution with shared Krylov/FGMRES helpers.
- Not yet: we still have solver-family resource containers and solver-family kernel binding/dispatch wiring. Generated WGSL exists, but the Rust-side binding/plumbing is not model-driven.

## Remaining Gaps (Concrete)
- `ModelGpuProgramSpec` is still mostly **function-pointer glue** over solver-family containers rather than a first-class “ports + module graph + dispatch plan” spec.
- Lowering still has per-family program builders (`compressible_program.rs` / `incompressible_program.rs` / `generic_coupled_program.rs`); schedules are handwritten rather than derived from `ModelSpec` + config.
- Kernel wiring is still largely handwritten per plan (bind group creation, pipeline selection, ping-pong choices, and pass ordering), especially for generated-per-model kernels.
- “Modules own their own resources” is only partially true; many pipelines/bind groups still live on solver-family structs.
- Generic coupled remains intentionally incomplete (limited terms/BCs), and scheme expansion is not yet driving required auxiliary passes automatically.
- Plan configuration is still driven by a global `PlanParam` enum and a handful of ad-hoc host callbacks; this should become a typed config/update path derived from `ModelSpec` and module capabilities.

## Next Steps (Prioritized)
1. **Make lowering truly model-driven**
   - Introduce a small “program spec IR” (schedule + module-graph composition + ports) that can be **generated per model**.
   - Target: remove per-family handwritten program builders and use one generic `lower_program(...)` path for all models.
2. **Elevate “first-class modules”**
   - Krylov / preconditioners / AMG become pluggable modules with explicit ports and self-owned resources.
   - Replace solver-family resource containers with module-owned resources reachable only through ports.
3. **Reduce handwritten kernel plumbing (incremental)**
   - Add small shared helpers/macros to reduce bind-group/pipeline boilerplate in existing builders (start with `generic_coupled_program.rs`).
   - Then move bind-group/pipeline construction toward “generated bindings + generic port wiring” so model-specific code becomes mostly “declare ports + compose module graphs”.
4. **Drive scheme expansion end-to-end**
   - Scheme selection expands to required auxiliary passes (gradients/reconstruction/history) for arbitrary fields/models.
5. **Replace `PlanParam` with typed config updates**
   - Replace `PlanParam/PlanParamValue` with typed `SolverConfigDelta` (and/or module-specific deltas) generated from `ModelSpec` + module capabilities.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
