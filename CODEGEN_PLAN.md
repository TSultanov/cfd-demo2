# CFD2 Codegen + Unified Solver Plan

This file tracks *codegen + solver orchestration* work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering/compilation path (no compressible/incompressible-specific compiler/lowerer forks)
- a single runtime orchestration path (no per-family solver loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- numerical schemes + auxiliary computations derived automatically from `ModelSpec` + solver config

## Current State (Implemented)
- Typed WGSL DSL is a real AST (no string-based ops) with unit/type checking; units are erased at WGSL emission.
- `UnifiedSolver` is the only public entrypoint; tests/UI use it.
- `ExecutionPlan`/`ModuleGraph` exist and drive dispatch sequencing.
- Readback is unified via cached staging buffers (`StagingBufferCache` + `read_buffer_cached`).
- `GpuRuntimeCommon` consolidates GPU context + mesh + profiling + cached readback and is used by all current plans (compressible/incompressible/generic coupled/scalar runtimes).
- `GpuPlanInstance` is universal and configured via typed `PlanParam`/`PlanParamValue` and queried via `PlanCapability` (no downcasts).
- Linear debug path is supported across current plans (`set_linear_system`, `solve_linear_system_with_size`, `get_linear_solution`) with plan-appropriate Krylov under the hood (CG for scalar systems, FGMRES for compressible coupled).

## Gap Check (What Still Blocks The Goal)
- Still multiple plan builders/resource containers (compressible/incompressible/generic coupled); lowering/compilation is not unified.
- Some pipelines/bind groups are still owned by solver-family structs instead of module-owned resources.
- `GpuPlanInstance` still carries non-core debug/inspection hooks; long-term these should be moved behind explicit debug modules/ports.
- Generic coupled kernels are incomplete (convection/source/cross-coupling, broader BC support, closures).
- Scheme expansion is only partially wired end-to-end (some auxiliary passes are still hand-selected in plan code).

## Next Milestones (Ordered)
1. **Finish tightening the plan surface**
   - Keep `GpuPlanInstance` minimal (time/state I/O + stepping + param setting).
   - Move remaining debug/inspection hooks behind explicit debug modules/ports (still capability-gated), so `plan_instance.rs` does not accumulate plan-family APIs.
2. **Single model-driven lowerer**
   - Introduce a shared `ModelLowerer` that produces a `ModelGpuProgramSpec` from `ModelSpec` + `SolverConfig` (ports, module list, execution plan, sizing).
   - Keep “generated-per-model WGSL” but unify the Rust-side binding/dispatch wiring and remove solver-family-specific lowerer/compile paths.
3. **Module-owned GPU resources**
   - Move remaining pipeline/bind-group ownership into modules; plans describe composition, not layouts.
   - Make Krylov/preconditioners/AMG fully own their internal buffers and expose only typed ports.
4. **Generic scheme expansion end-to-end**
   - Expand schemes into required auxiliary computations (gradients/reconstruction/history) and drive kernel selection/dispatch from that output for arbitrary models/fields.
5. **Generic kernels and BCs**
   - Extend generic assembly/apply/update + boundary conditions until compressible/incompressible reduce to closure modules + generic operators.
6. **Delete legacy per-family solvers**
   - Remove remaining solver-family structs/loops and ensure all tests + UI still pass.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
