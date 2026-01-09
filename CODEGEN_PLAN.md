# CFD2 Codegen + Unified Solver Plan

This file tracks *remaining* codegen + solver orchestration work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering + codegen path (no separate compressible/incompressible lowerers/compilers)
- a single runtime orchestration path (no per-family step loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- kernels + auxiliary passes derived automatically from `ModelSpec` + solver config (no handwritten WGSL)

## Status (Reality Check)
- Runtime orchestration is largely unified via `GpuProgramPlan` + `ProgramSpec`, but lowering still forks by `ModelFields` and manually builds schedules/ops in `src/solver/gpu/lowering/*_program.rs`.
- Codegen emits WGSL at build time, but Rust-side pipeline/bind-group wiring is still handwritten and often solver-family-specific.
- Several critical kernels are still handwritten WGSL (`src/solver/gpu/shaders/*.wgsl`), including solver-family-specific ones (`compressible_*`, `schur_precond`).

## Main Blockers
- `ProgramSpec` is data, but `ProgramOps` is still a per-plan fn-pointer registry (IDs -> functions) rather than typed “op kinds” implemented by modules.
- Solver-family resource containers (`GpuSolver`, `CompressiblePlanResources`, `GenericCoupledProgramResources`) still own most pipelines/bind groups and dictate wiring.
- Kernel selection still relies on manual dispatch logic (e.g. string-based model-id matches in `generic_coupled_program.rs`) instead of a generated registry keyed by `ModelSpec`/`KernelPlan`.
- Scheme expansion and aux-pass discovery are not yet driving required buffers/kernels/schedule end-to-end.

## Next Steps (Prioritized)
1. **Make lowering truly model-driven**
   - Introduce a single lowerer that consumes a `ProgramTemplate`/`KernelPlan` derived from `ModelSpec` + config.
   - Delete per-family schedule authorship (`compressible_program.rs`, `incompressible_program.rs`, `generic_coupled_program.rs`); keep only reusable module/resource providers.
2. **Replace fn-pointer ops with typed module ops**
   - Replace `ProgramGraphId`/`ProgramHostId` registries with typed `OpKind` enums backed by modules (graph/host/cond/count).
   - Target: program specs become stable data that can be generated and unit-tested without a plan instance.
3. **Make resources module-owned and family-agnostic**
   - Move pipelines/bind groups/buffers out of `GpuSolver` / `*PlanResources` into modules with explicit `PortSpace` contracts.
   - Standardize ping-pong state, linear-system ports, and time integration as shared modules.
4. **Generate kernel bindings + dispatch**
   - Emit binding metadata alongside WGSL (ports/resources per bind group) and add a generic bind-group builder.
   - Replace string-based per-model kernel selection with a generated registry (`ModelSpec.id` + kernel kind -> pipeline/bind builders).
5. **Eliminate handwritten WGSL (incremental)**
   - Start with solver-family-specific shaders (`compressible_*`, `schur_precond`) and migrate them into the codegen WGSL pipeline.
   - Keep handwritten WGSL only as a temporary bootstrapping layer.
6. **Replace `PlanParam` with typed config deltas**
   - Generate `SolverConfigDelta` + module-specific deltas from `ModelSpec` + module capabilities; remove ad-hoc host callbacks.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
