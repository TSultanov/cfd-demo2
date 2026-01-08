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
- `GpuPlanInstance` is now a small universal interface (`src/solver/gpu/plans/plan_instance.rs`) with typed `PlanParam`/`PlanAction` plus `PlanStepStats` and a few optional debug hooks (reducing plan-specific downcasts in `UnifiedSolver`).
- `GpuPlanInstance` no longer requires `Any`/downcasts (`src/solver/gpu/plans/plan_instance.rs`).
- Plan selection is centralized in `build_plan_instance` (`src/solver/gpu/plans/plan_instance.rs`), keeping `UnifiedSolver` orchestration generic.
- Generic coupled plan no longer depends on legacy `GpuSolver`; it uses a minimal scalar runtime (`src/solver/gpu/runtime.rs`) that owns mesh + constants + scalar CG resources.
- The “linear debug” hooks (`set_linear_system`/`solve_linear_system_*`/`get_linear_solution`) are implemented across all current plans (scalar CG for scalar plans, FGMRES for compressible), reducing plan-only method gaps in `src/solver/gpu/plans/plan_instance.rs`.

## Gap Check (What Still Blocks The Goal)
- Still multiple plan builders/resource containers (compressible/incompressible/generic coupled); lowering/compilation is not unified.
- Some pipelines/bind groups are still owned by solver-family structs instead of module-owned resources.
- `GpuPlanInstance` still exposes several plan-only hooks; `src/solver/gpu/plans/plan_instance.rs` needs to converge toward a small, universal “ports + params + actions” surface.
- Generic coupled kernels are incomplete (convection/source/cross-coupling, broader BC support, closures).
- Scheme expansion is only partially wired end-to-end (some auxiliary passes are still hand-selected in plan code).

## Next Milestones (Ordered)
1. **Tighten the plan surface**
   - Reduce/replace plan-only methods in `src/solver/gpu/plans/plan_instance.rs` by routing through typed `PlanParam`/`PlanAction` and typed ports/views (capability-gated).
   - Make all remaining `GpuPlanInstance` methods meaningful for *every* plan (or move them behind explicit capabilities).
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
