# CFD2 Codegen + Unified Solver Plan

This file tracks *codegen + solver orchestration* work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering/compilation path (no compressible/incompressible-specific compiler/lowerer forks)
- a single runtime orchestration path (no per-family solver loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- numerical schemes + auxiliary computations derived automatically from `ModelSpec` + solver config

## Current State (Implemented)
- Typed WGSL DSL: real AST (`src/solver/codegen/wgsl_ast.rs`), expressions are arena-backed `Copy`, no string-based API.
- Units/type checking in the DSL with SI-printable dimensions (`TypedExpr` + `UnitDim`), erased at WGSL emission.
- `UnifiedSolver` is the only public entrypoint; UI and tests go through it (`src/solver/gpu/unified_solver.rs`, `src/ui/app.rs`).
- `ExecutionPlan` + `ModuleGraph` exist and are used to sequence dispatches and host/plugin steps.
- Linear system resources are increasingly passed as typed ports/views (`LinearSystemPorts`, `LinearSystemView`) instead of raw buffers.
- Readback staging buffers are shared via `StagingBufferCache`.
- `GpuPlanInstance` is now a small universal interface (`src/solver/gpu/plans/plan_instance.rs`) with typed `PlanParam`/`PlanAction` plus `PlanStepStats` (reducing plan-specific downcasts in `UnifiedSolver`).

## Gap Check (What Still Blocks The Goal)
- Still multiple plan builders/resource containers (compressible/incompressible/generic coupled); lowering is not unified.
- Some pipelines/bind groups are still owned by solver-family structs instead of module-owned resources.
- `UnifiedSolver` still contains plan-specific convenience methods implemented via downcasts; this needs a universal “capabilities/parameters” surface.
- Generic coupled kernels are incomplete (convection/source/cross-coupling, broader BC support, closures).
- Scheme expansion is only partially wired end-to-end (some auxiliary passes are still hand-selected in plan code).

## Next Milestones (Ordered)
1. **Plan instance surface**
   - Replace `UnifiedSolver` downcasts with a typed “capabilities/parameters” API (e.g. `PlanParam::{Viscosity,Density,...}`, `PlanCapability::{Profiling,LinearDebug,...}`).
   - Keep `GpuPlanInstance` methods universal (avoid plan-only method creep; `src/solver/gpu/plans/plan_instance.rs`).
2. **Single model-driven lowerer**
   - Introduce `ModelLowerer` that produces a `ModelGpuProgramSpec` from `ModelSpec` + `SolverConfig` (ports, resource sizing, module list, execution plan).
   - Keep “generated-per-model WGSL” but unify the Rust-side binding/dispatch wiring.
3. **Module-owned GPU resources**
   - Move remaining pipeline/bind-group ownership into modules; plans describe composition, not layouts.
   - Ensure Krylov/preconditioners/AMG manage buffers internally and expose only typed ports.
4. **Generic scheme expansion end-to-end**
   - Expand schemes into required auxiliary computations (gradients/reconstruction/history) and drive kernel selection/dispatch from that output.
5. **Generic kernels and BCs**
   - Extend generic assembly/apply/update and boundary conditions until incompressible and compressible can be expressed as closure modules + generic operators.
6. **Delete legacy per-family solvers**
   - Remove remaining specialized solver loops and ensure all tests + UI still pass.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
