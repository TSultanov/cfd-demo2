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
- **SolverRecipe Abstraction (NEW):** `src/solver/gpu/recipe.rs` defines `SolverRecipe` — a unified specification derived from `ModelSpec + SolverConfig` that captures:
  - Required kernel passes (`KernelSpec` with `KernelPhase`)
  - Required auxiliary buffers (`BufferSpec` for gradients, history, etc.)
  - Linear solver configuration (`LinearSolverSpec`)
  - Time integration requirements (`TimeIntegrationSpec`)
  - Stepping mode (`SteppingMode`: Explicit/Implicit/Coupled)
  - Fields requiring gradient computation
- **Generic Linear Solver Module (NEW):** `src/solver/gpu/modules/generic_linear_solver.rs` provides a parameterized `GenericLinearSolverModule<P>` that can be instantiated with different preconditioners, decoupling solver infrastructure from specific physics families.
- **derive_kernel_plan (NEW):** Function in `recipe.rs` to derive kernel requirements from `EquationSystem` structure rather than hardcoding in `ModelSpec::kernel_plan()`.

## Main Blockers
- **Monolithic Resource Containers (Generic/Incompressible):** `GenericCoupledProgramResources` and `IncompressibleProgramResources` still need similar modularization to what was done for Compressible.
- **Hardcoded Scheme Assumptions:** Runtime lowering often assumes worst-case schemes (e.g., SOU for generic coupled) to allocate resources.
- **Build-Time Kernel Tables:** Kernel lookup relies on build-time generated tables (`kernel_registry_map.rs`), not yet dynamically derived from scheme expansion.
- **Three Template Paths:** `ProgramTemplateKind` still has three hardcoded variants with separate resource containers and op handlers.

## Next Steps (Prioritized)

1. **Wire SolverRecipe to Resource Allocation**
   - Use `SolverRecipe::from_model()` during lowering to drive buffer allocation.
   - Replace heuristic gradient detection with `recipe.needs_gradients()`.
   - Use `recipe.aux_buffers` to allocate auxiliary storage systematically.

2. **Unify Resource Containers**
   - Create `UnifiedPlanResources` composed of:
     - `FieldResourcesModule` (ping-pong state, gradients)
     - `GenericLinearSolverModule<P>` (FGMRES/CG with pluggable preconditioner)
     - `ComputeGraphsModule` (GPU execution graphs)
     - `TimeIntegrationModule`
     - `ModelKernelsModule` (compiled pipelines from registry)
   - Migrate `IncompressibleProgramResources` and `GenericCoupledProgramResources` to use this unified container.

3. **Derive Program Spec from Recipe**
   - Replace `build_program_spec()` functions in `templates.rs` with `build_program_from_recipe(recipe)`.
   - Use `recipe.stepping` to determine loop structure (explicit vs implicit vs coupled outer loops).
   - Use `recipe.kernels` to populate graph ops.

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
