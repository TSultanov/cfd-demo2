# CFD2 Codegen + Unified Solver Plan

This file tracks *remaining* codegen + solver orchestration work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering + codegen path (no separate compressible/incompressible lowerers/compilers)
- a single runtime orchestration path (no per-family step loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- kernels + auxiliary passes derived automatically from `ModelSpec` + solver config (no handwritten WGSL)

## Status (Reality Check)
- **Unified Lowering:** Lowering now routes through a single entrypoint (`src/solver/gpu/lowering/model_driven.rs`) with template schedules in `src/solver/gpu/lowering/templates.rs`. Per-family lowering entrypoints have been removed.
- **Registry-Based Op Dispatch:** Op dispatch is handled via `ProgramOpRegistry` (in `src/solver/gpu/plans/program.rs`). Models register their ops via `register_ops(...)` in `src/solver/gpu/lowering/models/*`. `ProgramSpec` validation ensures all nodes have handlers.
- **Shared State & Constants:** `PingPongState` and `ConstantsModule` provide shared abstractions for solver state and constants across all solver families.
- **Time Integration:** `TimeIntegrationModule` centralizes time advancement and bookkeeping, used by all solver families.
- **Kernel Lookup:** Unified via `kernel_registry::kernel_source(model_id, KernelKind)`, providing generated shader strings, pipeline constructors, and binding metadata.
- **Codegen:** Emits WGSL at build time. Binding metadata is generated, but resource resolution still largely relies on solver-family specific glue.
- **Handwritten WGSL:** Critical kernels (compressible, schur_precond) are still handwritten.

## Main Blockers
- **Global Op Enums:** Op dispatch is still structured by global `GraphOpKind`/`HostOpKind` enums. Adding a module requires editing these central enums, preventing true modularity.
- **Monolithic Resource Containers:** Solver-family resource containers (`CompressiblePlanResources`, `GenericCoupledProgramResources`) still own most pipelines, bind groups, and wiring logic.
- **Hardcoded Scheme Assumptions:** Runtime lowering often assumes worst-case schemes (e.g., SOU for generic coupled) to allocate resources, rather than deriving exact needs from `ModelSpec` + config.
- **Build-Time Kernel Tables:** Kernel lookup relies on build-time generated tables (`kernel_registry_map.rs`), not yet dynamically derived from scheme expansion.

## Next Steps (Prioritized)

1. **Make Op Dispatch Module-Owned**
   - Replace global `GraphOpKind`/`HostOpKind` enums with module-owned IDs (e.g., namespaced IDs or newtypes).
   - Introduce a `ModuleRegistry` where modules register ops and declare provided resources/ports.
   - Goal: Add a new solver feature without touching `src/solver/gpu/plans/program.rs`.

2. **Modularize Resources & Pipelines**
   - Break down monolithic resource containers (`CompressiblePlanResources` etc.) into smaller, self-contained modules.
   - Move solver-owned buffers (flux/gradient buffers, BC tables, linear solver scratch) behind module boundaries with explicit `PortSpace` contracts.
   - Extend `TimeIntegrationModule` to cover restart/init and scheme-specific history rotation.

3. **Expand Registry-Driven Kernel Wiring**
   - Derive kernel registry entries from `KernelPlan` / scheme expansion dynamically where possible.
   - Extend binding metadata to use explicit `PortSpace`/resource contracts instead of string matching, allowing modules to resolve resources automatically.

4. **End-to-End Scheme Pluggability**
   - Use `expand_schemes` to drive:
     - Required aux buffers (e.g., gradients).
     - Required aux passes.
     - Schedule structure (not just runtime branching).
   - Remove "scheme implies gradients" heuristics; make it a strict lowering-time contract.

5. **Linear Solver & Preconditioner Pluggability**
   - Unify FGMRES driver logic into a shared module.
   - Move solver selection to a typed registry/factory.
   - Let solvers/preconditioners own their scratch buffers.

6. **Eliminate Handwritten WGSL**
   - Migrate solver-family-specific shaders (`compressible_*`, `schur_precond`) into the codegen WGSL pipeline.

7. **Typed Config Deltas**
   - Replace `PlanParam` with generated `SolverConfigDelta` + module-specific deltas to remove ad-hoc host callbacks.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; build-time-only modules must be added there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.