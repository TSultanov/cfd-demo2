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
- **UnifiedFieldResources (EXTENDED):** `src/solver/gpu/modules/unified_field_resources.rs` provides unified field storage (PingPongState, gradients, constants) derived from SolverRecipe:
  - Supports flux buffers for face-based storage
  - Supports low-mach preconditioning params buffer
  - Builder pattern for flexible configuration (`with_flux_buffer()`, `with_low_mach_params()`, `with_gradient_fields()`)
  - **Model-agnostic**: No solver-family-specific factory methods; callers configure via builder
- **FieldProvider Trait (NEW):** `src/solver/gpu/modules/field_provider.rs` provides trait abstraction for field buffer access:
  - Common interface for `CompressibleFieldResources` and `UnifiedFieldResources`
  - Enables gradual migration to unified resources without breaking existing code
  - `buffer_for_binding()` method for shader reflection-based binding
- **UnifiedOpRegistryBuilder (NEW):** `src/solver/gpu/lowering/unified_registry.rs` builds op registries dynamically from SolverRecipe stepping mode instead of hardcoded templates.
- **UnifiedGraphModule (NEW):** `src/solver/gpu/modules/unified_graph.rs` provides trait and helpers for building compute graphs from SolverRecipe kernel specifications.
- **Composite Phase Graphs (NEW):** `src/solver/gpu/modules/unified_graph.rs` now supports building a single graph from an ordered sequence of phases (`build_graph_for_phases`, `build_optional_graph_for_phases`). This enables multi-pass graphs like `gradients -> flux -> update -> primitive_recovery` to be recipe-driven.
- **GenericCoupledKernelsModule (UPDATED):** Now implements `UnifiedGraphModule` trait, enabling recipe-driven graph construction via `build_graph_for_phase()`.
- **Recipe-Driven Graph Building (NEW):** `GenericCoupledProgramResources::new()` now uses `build_graph_for_phase(recipe, phase, module, label)` to derive compute graphs from the recipe instead of hardcoded graph builders (with fallbacks for legacy recipes).
- **Recipe-Driven GenericCoupled (NEW):** GenericCoupledScalar now uses:
  - `recipe.build_program_spec()` instead of hardcoded template spec
  - `register_ops_from_recipe()` instead of manual op registration
  - `UnifiedFieldResources` for field storage
- **Recipe-Driven Compressible (NEW):** CompressiblePlanResources::new() now takes SolverRecipe
  - Uses `recipe.needs_gradients()` instead of runtime scheme detection
  - Uses `recipe.stepping` for outer iterations count
- **Recipe-Driven IncompressibleCoupled (NEW):** GpuSolver::new() now takes SolverRecipe
  - Uses `recipe.needs_gradients()` for scheme_needs_gradients
  - Uses `recipe.stepping` for n_outer_correctors
- **FieldProvider-based bind groups (NEW):** `model_kernels.rs` now uses `FieldProvider::buffer_for_binding()` for compressible bind group creation:
  - `field_binding()` helper reduces bind group creation from ~40 lines to single call
  - Enables bind group code to work with either `CompressibleFieldResources` or `UnifiedFieldResources`

## Main Blockers
- **Compressible/Incompressible still use legacy field containers:** The `FieldProvider` trait provides the migration path, and bind group creation now uses `field_binding()` helper. Actual switch to `UnifiedFieldResources` requires callers to use the builder pattern to configure needed buffers.
- **Hardcoded Scheme Assumptions:** Runtime lowering often assumes worst-case schemes (e.g., SOU for generic coupled) to allocate resources.
- **Build-Time Kernel Tables:** Kernel lookup relies on build-time generated tables (`kernel_registry_map.rs`), not yet dynamically derived from scheme expansion.
- **Template ProgramSpec not yet recipe-driven for Compressible/Incompressible:** These templates still use hardcoded `build_program_spec()` functions in templates.rs.

## Next Steps (Prioritized)

1. **Switch Compressible field storage to UnifiedFieldResources**
   - Bind group creation already uses `FieldProvider` trait via `field_binding()` helper
   - Use builder: `UnifiedFieldResourcesBuilder::new(...).with_flux_buffer(...).with_gradient_fields(&[...]).build()`
   - Keep solver-family knowledge in the compressible plan, not in unified modules

2. **Implement UnifiedGraphModule for ModelKernelsModule**
   - [x] GenericCoupledKernelsModule now implements `UnifiedGraphModule` trait (done).
   - [x] GenericCoupledProgramResources uses `build_graph_for_phase()` (done).
   - [x] ModelKernelsModule (compressible) implements `UnifiedGraphModule` trait (done).
   - [x] CompressiblePlanResources uses `CompressibleGraphs::from_recipe()` (done).

3. **Migrate derive_kernel_plan to Production**
   - Replace `ModelSpec::kernel_plan()` hardcoded matches with calls to `derive_kernel_plan()`.
   - Extend `derive_kernel_plan` to handle all equation term types.

4. **Linear Solver & Preconditioner Pluggability**
   - [x] Extract FGMRES driver logic to `LinearSolverModule` (done).
   - [x] Create `GenericLinearSolverModule<P>` with pluggable preconditioner (done).
   - [ ] Migrate `CompressibleLinearSolver` to use `GenericLinearSolverModule`.
   - [ ] Add `PreconditionerFactory` implementations for Jacobi, AMG, Schur.

5. **Eliminate Handwritten WGSL**
   - Migrate solver-family-specific shaders (`compressible_*`, `schur_precond`) into the codegen WGSL pipeline.

6. **Typed Config Deltas**
   - Replace `PlanParam` with generated `SolverConfigDelta` + module-specific deltas to remove ad-hoc host callbacks.

## Roadmap: Truly Model-Agnostic Unified Solver

The current unified pieces (`SolverRecipe`, `UnifiedFieldResources`, unified graph builder) are moving in the right direction, but a few remaining “legacy glue points” prevent *arbitrary* models/methods from fitting without touching handwritten matches.

This roadmap focuses on removing those glue points in small, testable steps.

### Milestone A: Make the recipe authoritative for kernel scheduling

**Goal:** No handwritten `KernelKind -> phase` mapping is required for correct execution. The recipe explicitly describes kernel order for each solver mode.

- [x] **A1. Move phase assignment into recipe construction**
  - Today `phase_for_kernel()` in `src/solver/gpu/recipe.rs` is a central match on `KernelKind`. This should become an implementation detail of *legacy recipes only*.
  - New path: `SolverRecipe::from_model(...)` assigns phases when emitting `KernelSpec`s.
- [x] **A2. Validate required phases are non-empty**
  - `build_graph_for_phase()` should error on “required but empty” phases to avoid silent no-ops (the BDF2 acoustic regression was caused by an empty apply graph being accepted).

### Milestone B: Decouple orchestration from `KernelKind`

**Goal:** Add new kernels via codegen/model definitions without editing handwritten enums/matches.

- [x] **B1. Introduce a generated kernel identifier**
  - Add `KernelId(&'static str)` (or similar) alongside the existing `KernelKind` bridge.
  - Update `KernelSpec` to carry `KernelId` (and optionally keep `KernelKind` during migration).
- [x] **B2. Extend `kernel_registry` to lookup by `KernelId`**
  - Codegen emits a per-model kernel table mapping `KernelId -> (wgsl source, bind metadata)`.
  - The runtime graph builder uses only `KernelId`.
- [x] **B3. Shrink `KernelKind` usage to UI/debug only**
  - Once recipes are emitted in terms of `KernelId`, `KernelKind` can become optional legacy.

  Status: Orchestration/graph building is now `KernelId`-based; `KernelKind` is still carried in recipes as a migration/debug bridge.

### Milestone C: Full recipe-driven graphs (no solver-family graph builders)

**Goal:** `CompressibleGraphs`, `IncompressibleGraphs`, etc. become thin wrappers (or disappear). Execution structure is derived from the recipe.

- [x] **C1. Add “composite phase” support to the recipe/program spec**
  - Explicit compressible needs sequences like `gradients -> flux -> explicit_update -> primitive_recovery`.
  - Implemented as recipe-driven composite graph building from ordered phase sequences (`build_graph_for_phases` / `build_optional_graph_for_phases`), and wired into `CompressibleGraphs::from_recipe()`.
- [x] **C2. Convert existing solver-family plans to `register_ops_from_recipe()`**
  - GenericCoupled is done; migrate Compressible and IncompressibleCoupled.

    Status: GenericCoupled, Compressible, and IncompressibleCoupled now register ops via recipe-aware entrypoints.

### Milestone D: Resources fully derived from recipe specs

**Goal:** No solver-family plan decides “which buffers exist”. It only supplies numerics and initial/boundary conditions.

- [ ] **D1. Allocate `UnifiedFieldResources` from `BufferSpec`/field requirements**
  - Replace builder calls with a recipe-driven allocator.
- [ ] **D2. Bind groups generated from reflection + `FieldProvider`/resource registry**
  - Existing `field_binding()` helper is the migration path; end state is uniform reflection-driven binds.

### Execution order (recommended)

1. A2 (safety) → A1 (phase assignment in recipe) → C2 (use unified registry) to eliminate correctness footguns.
2. B1/B2 (KernelId) to unlock “arbitrary models” without touching handwritten enums.
3. C1/C2 to remove solver-family graph code.
4. D1/D2 to finish resource unification.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; build-time-only modules must be added there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
