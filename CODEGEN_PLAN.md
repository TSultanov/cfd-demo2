# CFD2 Codegen + Unified Solver Plan

This file tracks *remaining* codegen + solver orchestration work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering + codegen path (no separate compressible/incompressible lowerers/compilers)
- a single runtime orchestration path (no per-family step loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- kernels + auxiliary passes derived automatically from `ModelSpec` + solver config (no handwritten WGSL)

## Status (Reality Check)
- **Op Dispatch:** Fully module-owned via string IDs (`GraphOpKind` et al). Global enums in `program.rs` have been replaced with newtypes wrapping `&'static str`. Models now register ops using decentralized constants defined in `templates.rs`.
- **Resource Modularization:** `CompressiblePlanResources` has been refactored into a coordinator of self-contained modules:
  - `CompressibleLinearSolver`: Manages FGMRES resources, AMG setup, and solve logic.
  - `CompressibleGraphs`: Manages graph construction and execution (explicit/implicit/update).
  - `CompressibleFieldResources`: Owns fields and buffers.
  - `TimeIntegrationModule`: Owns time stepping logic.
- **Unified Lowering:** Lowering routes through `src/solver/gpu/lowering/model_driven.rs`.
- **Shared State & Constants:** `PingPongState` and `ConstantsModule` used across solver families.
- **Kernel Lookup:** Unified via `kernel_registry::kernel_source`.
- **Linear Solver:** FGMRES logic extracted into generic `solve_fgmres` in `src/solver/gpu/modules/linear_solver.rs`, decoupling algorithm from resource management.
- **Codegen:** Emits WGSL at build time.

## Main Blockers
- **Monolithic Resource Containers (Generic/Incompressible):** `GenericCoupledProgramResources` and `IncompressibleProgramResources` still need similar modularization to what was done for Compressible.
- **Hardcoded Scheme Assumptions:** Runtime lowering often assumes worst-case schemes (e.g., SOU for generic coupled) to allocate resources.
- **Build-Time Kernel Tables:** Kernel lookup relies on build-time generated tables (`kernel_registry_map.rs`), not yet dynamically derived from scheme expansion.

## Next Steps (Prioritized)

1. **Modularize Resources & Pipelines (Remaining Families)**
   - Apply the modularization pattern (Linear Solver, Graphs modules) to `Incompressible` and `GenericCoupled` paths.
   - Ensure all solver-owned buffers are behind module boundaries.

2. **Expand Registry-Driven Kernel Wiring**
   - Derive kernel registry entries from `KernelPlan` / scheme expansion dynamically where possible.
   - Extend binding metadata to use explicit `PortSpace`/resource contracts instead of string matching.

3. **End-to-End Scheme Pluggability**
   - Use `expand_schemes` to drive:
     - Required aux buffers (e.g., gradients).
     - Required aux passes.
     - Schedule structure (not just runtime branching).
   - Remove "scheme implies gradients" heuristics; make it a strict lowering-time contract.

4. **Linear Solver & Preconditioner Pluggability**
   - [x] Extract FGMRES driver logic to `LinearSolverModule` (done).
   - [ ] Genericize `CompressibleLinearSolver` to be a shared `LinearSolverModule` usable by other solvers.
   - [ ] Move solver selection to a typed registry/factory.

5. **Eliminate Handwritten WGSL**
   - Migrate solver-family-specific shaders (`compressible_*`, `schur_precond`) into the codegen WGSL pipeline.

6. **Typed Config Deltas**
   - Replace `PlanParam` with generated `SolverConfigDelta` + module-specific deltas to remove ad-hoc host callbacks.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; build-time-only modules must be added there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
