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
- **SolverRecipe Abstraction:** `src/solver/gpu/recipe.rs` defines `SolverRecipe` — a unified specification derived from `ModelSpec + SolverConfig` that captures:
  - Required kernel passes (`KernelSpec` with `KernelPhase`)
  - Required auxiliary buffers (`BufferSpec` for gradients, history, etc.)
  - Linear solver configuration (`LinearSolverSpec`)
  - Time integration requirements (`TimeIntegrationSpec`)
  - Stepping mode (`SteppingMode`: Explicit/Implicit/Coupled)
  - Fields requiring gradient computation
  - `build_program_spec()` method to derive ProgramSpec from stepping mode
- **Generic Linear Solver Module:** `src/solver/gpu/modules/generic_linear_solver.rs` provides a parameterized `GenericLinearSolverModule<P>` that can be instantiated with different preconditioners, decoupling solver infrastructure from specific physics families.
- **derive_kernel_plan:** Function in `recipe.rs` to derive kernel requirements from `EquationSystem` structure rather than hardcoding in `ModelSpec::kernel_plan()`.
- **UnifiedFieldResources (NEW):** `src/solver/gpu/modules/unified_field_resources.rs` provides unified field storage (PingPongState, gradients, constants) derived from SolverRecipe. Integrated into GenericCoupled path.
- **UnifiedOpRegistryBuilder (NEW):** `src/solver/gpu/lowering/unified_registry.rs` builds op registries dynamically from SolverRecipe stepping mode instead of hardcoded templates.
- **UnifiedGraphModule (NEW):** `src/solver/gpu/modules/unified_graph.rs` provides trait and helpers for building compute graphs from SolverRecipe kernel specifications.
- **Recipe-Driven GenericCoupled (NEW):** GenericCoupledScalar now uses:
  - `recipe.build_program_spec()` instead of hardcoded template spec
  - `register_ops_from_recipe()` instead of manual op registration
  - `UnifiedFieldResources` for field storage

## Main Blockers
- **Monolithic Resource Containers (Compressible/Incompressible):** These paths still use hardcoded resource containers. Need to migrate to recipe-driven approach like GenericCoupled.
- **Hardcoded Scheme Assumptions:** Runtime lowering often assumes worst-case schemes (e.g., SOU for generic coupled) to allocate resources.
- **Build-Time Kernel Tables:** Kernel lookup relies on build-time generated tables (`kernel_registry_map.rs`), not yet dynamically derived from scheme expansion.
- **Two Remaining Template Paths:** `ProgramTemplateKind::Compressible` and `ProgramTemplateKind::IncompressibleCoupled` still have separate resource containers and op handlers.

## Next Steps (Prioritized)

1. **Migrate IncompressibleCoupled to Recipe-Driven Approach** ✅ Partially done for GenericCoupled
   - Apply same pattern as GenericCoupled: use `UnifiedFieldResources`, `register_ops_from_recipe()`, recipe's `build_program_spec()`.
   - Update `IncompressibleProgramResources` to use unified field storage.

2. **Migrate Compressible to Recipe-Driven Approach**
   - More complex due to explicit/implicit branching and FGMRES integration.
   - May need to extend `UnifiedOpRegistryBuilder` to handle conditional ops.

3. **Implement UnifiedGraphModule for Modules**
   - Have GenericCoupledKernelsModule implement `UnifiedGraphModule` trait.
   - Use `build_graph_for_phase()` instead of manual graph construction.

4. **Migrate derive_kernel_plan to Production**
   - Replace `ModelSpec::kernel_plan()` hardcoded matches with calls to `derive_kernel_plan()`.
   - Extend `derive_kernel_plan` to handle all equation term types.

5. **Linear Solver & Preconditioner Pluggability**
   - [x] Extract FGMRES driver logic to `LinearSolverModule` (done).
   - [x] Create `GenericLinearSolverModule<P>` with pluggable preconditioner (done).
   - [ ] Migrate `CompressibleLinearSolver` to use `GenericLinearSolverModule`.
   - [ ] Add `PreconditionerFactory` implementations for Jacobi, AMG, Schur.

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
