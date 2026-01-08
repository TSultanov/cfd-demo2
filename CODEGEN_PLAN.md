# CFD2 Codegen Plan

This file tracks *codegen + solver orchestration* work. Pure physics/tuning tasks should live elsewhere.

## Current Status (Done)
- WGSL is generated from a real AST (`src/solver/codegen/wgsl_ast.rs`) with a precedence-aware renderer.
- `Expr` is a `Copy` handle backed by an arena; node construction is private so expressions must be built via overloaded operators + helper methods.
- The string-based WGSL DSL is removed; kernels build typed AST directly (`src/solver/codegen/*`).
- The DSL has:
  - builtin WGSL function wrappers (`src/solver/codegen/wgsl_dsl.rs`)
  - small-vector/matrix algebra helpers (`src/solver/codegen/dsl/tensor.rs`)
  - CSR/block-CSR helpers (`src/solver/codegen/dsl/matrix.rs`)
  - typed/unit-carrying expressions (`TypedExpr`) with SI `UnitDim` (units erased when emitting WGSL).
- Typed WGSL enums exist in the DSL and on the GPU side (e.g. boundary/time schemes), and kernels use them for branching.
- 1D regression tests that save plots exist (acoustic pulse + Sod shock tube) and are the main safety net for refactors.
- Public module layout is now separated so most users don’t touch backend internals:
  - `solver::model` (physics/spec)
  - `solver::compiler` (typed AST + WGSL emission)
  - `solver::kernels` (kernel sources/artifacts; currently still stored under `src/solver/gpu/shaders`)
  - `solver::backend` (runtime backend implementations)
  - `solver::options` (solver config enums/types)
  - `solver::UnifiedSolver` (single user-facing entrypoint)

## Unified Solver Progress (So Far)
- Added `GpuUnifiedSolver` + `SolverConfig` as a single entrypoint wrapper over the existing incompressible/compressible GPU solvers (`src/solver/gpu/unified_solver.rs`).
- Removed `ModelSpec.kernel_plan` storage; kernel plans are derived via `ModelSpec::kernel_plan()` (still model-family specific for now).
- Shared block-CSR expansion (`src/solver/gpu/csr.rs`) and derived compressible unknown count from `EquationSystem::unknowns_per_cell()`.
- Derived incompressible coupled block size from `EquationSystem::unknowns_per_cell()` as well (removing a runtime magic `3` from the coupled linear-solver init path).
- Introduced a small runtime `KernelGraph` executor (`src/solver/gpu/kernel_graph.rs`) and migrated the compressible explicit KT timestep path to it.
- Migrated the compressible implicit path’s GPU dispatch blocks (grad/assembly/apply/update) to `KernelGraph` as well (FGMRES remains a host-controlled plugin step for now).
- Migrated the incompressible coupled solver’s GPU dispatch blocks (prepare/assembly/update) to prebuilt `KernelGraph`s; the coupled FGMRES solve + convergence logic remains host-controlled for now.
- Added an `ExecutionPlan` layer that can sequence GPU `KernelGraph` nodes and host/plugin steps uniformly (`src/solver/gpu/execution_plan.rs`); compressible explicit stepping uses it as the first integration point.
- Planized the compressible implicit outer-iteration body (grad+assembly → host FGMRES → snapshot/apply) using `ExecutionPlan`, keeping behavior but moving sequencing into the common plan layer.
- Planized the incompressible coupled iteration’s assembly+solve stage and update stage using `ExecutionPlan` (assembly via `KernelGraph`, host FGMRES solve as a plugin node, update kernel as `KernelGraph`).
- Planized the incompressible coupled solver’s init/prepare stage (host constant update + init prepare kernel) using `ExecutionPlan` as well (`src/solver/gpu/coupled_solver.rs`).
- Expanded `GpuUnifiedSolver` to provide a real shared surface (common setters + `get_u/get_p/get_rho`) and migrated the 1D regression test harness to run compressible/incompressible cases through it (`tests/gpu_1d_structured_regression_test.rs`).
- Migrated the UI app to `GpuUnifiedSolver` and switched plotting to use `StateLayout`-derived offsets instead of hardcoded packing (`src/ui/app.rs`).
- Removed `FluidState` struct dependence from the public incompressible GPU read/write helpers by using `StateLayout` offsets instead (`src/solver/gpu/solver.rs`).
- Switched incompressible state buffer initialization to use `StateLayout` stride (so the host no longer needs a matching `FluidState` struct) (`src/solver/gpu/init/fields.rs`).
- Added cached default `ModelSpec`s for GPU codepaths that need access to layout/unknown metadata without threading a `ModelSpec` everywhere (`src/solver/gpu/model_defaults.rs`).
- Added a generic (model-driven) scheme expansion pass that reports required auxiliary computations (currently: gradient needs for higher-order convection) (`src/solver/model/backend/scheme_expansion.rs`).
- Used the scheme expansion to skip compressible gradient dispatches for first-order runs by selecting first-order vs reconstruction kernel graphs at runtime (`src/solver/gpu/compressible_solver.rs`).
- Plumbed the same scheme expansion result into the incompressible coupled solver loop via `GpuSolver.scheme_needs_gradients` (replacing direct `scheme != 0` checks) (`src/solver/gpu/solver.rs`, `src/solver/gpu/coupled_solver.rs`).
- Removed duplicated CSR adjacency computation by reusing mesh-derived scalar CSR for linear-solver initialization (`src/solver/gpu/init/mesh.rs`, `src/solver/gpu/init/linear_solver/mod.rs`).
- Decoupled mesh buffer initialization from any single kernel’s bind-group layout (mesh init no longer constructs bind groups; plans/modules do) (`src/solver/gpu/init/mesh.rs`, `src/solver/gpu/init/mod.rs`).
- Split the low-Mach “preconditioning” parameters out of the shared constants buffer into a dedicated `GpuLowMachParams` uniform bound in compressible kernels (`src/solver/gpu/structs.rs`, `src/solver/gpu/init/compressible_fields.rs`, `src/solver/codegen/compressible_*`).
- Moved preconditioner selection out of `GpuConstants` into typed host-side selection (`PreconditionerType`) and refactored coupled/compressible FGMRES to use first-class GPU modules (`src/solver/gpu/modules/krylov_precond.rs`, `src/solver/gpu/modules/compressible_krylov.rs`, `src/solver/gpu/modules/coupled_schur.rs`).
- Encapsulated GPU FGMRES internals in `FgmresWorkspace`: plans allocate via `FgmresWorkspace::new()` and interact via accessors (e.g. `params_layout()` for plugin preconditioner pipeline layouts), instead of rebuilding GMRES bind-groups/pipelines.
- Added a typed buffer “port” system for lowered GPU resources (`PortSpace`) and migrated compressible linear RHS/X/scalar-row-offsets to it (`src/solver/gpu/modules/ports.rs`, `src/solver/gpu/modules/compressible_lowering.rs`).
- Migrated compressible block-CSR buffers (row/col/value) into `PortSpace` and removed `MatrixResources` dependency from `CompressibleKernelsModule` (`src/solver/gpu/modules/compressible_lowering.rs`, `src/solver/gpu/modules/compressible_kernels.rs`).
- Added typed ports for the legacy scalar linear system (CSR + RHS + X) and plumbed them through `GpuSolver` initialization as a no-behavior-change step toward module-owned resources (`src/solver/gpu/modules/linear_system.rs`, `src/solver/gpu/init/linear_solver/mod.rs`, `src/solver/gpu/structs.rs`).
- Encapsulated the scalar CG solve sequence into a `ScalarCgModule` (owning its bindgroups/pipelines and using shared linear buffers) and migrated `GpuSolver::solve_linear_system_cg*` to delegate to it (`src/solver/gpu/modules/scalar_cg.rs`, `src/solver/gpu/linear_solver/common.rs`).
- Added a compute-only `ModuleGraph` executor and migrated the compressible explicit KT step’s compute dispatches to it (`src/solver/gpu/modules/graph.rs`, `src/solver/gpu/plans/compressible.rs`).
- Encapsulated incompressible coupled kernels (prepare/assembly/update) into `IncompressibleKernelsModule` (module-owned bind groups + pipelines) and deleted the now-unused `init/physics.rs` pipeline plumbing (`src/solver/gpu/modules/incompressible_kernels.rs`, `src/solver/gpu/plans/coupled.rs`, `src/solver/gpu/init/mod.rs`).
- Migrated the coupled solver’s compute graphs to `ModuleGraph<IncompressibleKernelsModule>` and removed its remaining `KernelGraph` usage (execution is driven via `ExecutionPlan` host nodes as a bridge) (`src/solver/gpu/plans/coupled.rs`, `src/solver/gpu/structs.rs`).
- Began separating “plans” from “resource containers”: `GpuUnifiedSolver::step()` now dispatches through plan modules (`src/solver/gpu/compressible_solver.rs` `plan::*`, `src/solver/gpu/coupled_solver.rs` `plan::*`) instead of calling backend `.step()` methods directly; coupled currently delegates to `step_coupled_impl` as an intermediate step.
- Moved legacy per-family GPU implementations under `src/solver/gpu/plans/*` and renamed the unified dispatch enum from a “backend” to `PlanInstance` (`src/solver/gpu/unified_solver.rs`), so internal code is expressed as plan resources + plan functions rather than separate solver modules.
- Added generic (model-driven) `assembly/apply/update` WGSL generators for arbitrary coupled systems (currently supports implicit `ddt` + implicit `laplacian` only) and a small demo model to force build-time shader emission (`src/solver/codegen/generic_coupled_kernels.rs`, `src/solver/model/definitions.rs`).
- Added a first **generic boundary-condition** representation (`BoundarySpec` on `ModelSpec`) plus a helper to build GPU BC tables and a generic BC path in the generic coupled assembly kernel (Dirichlet + Neumann/zeroGradient for diffusion) (`src/solver/model/definitions.rs`, `src/solver/codegen/generic_coupled_kernels.rs`).
- Added `ModelSpec.id` and emit generic coupled kernels under id-suffixed WGSL names (so multiple generated-per-model variants can coexist) and a first runtime backend for `ModelFields::GenericCoupled` in `GpuUnifiedSolver` (`src/solver/codegen/emit.rs`, `src/solver/gpu/generic_coupled_solver.rs`, `src/solver/gpu/unified_solver.rs`).
- `GpuGenericCoupledSolver` now dispatches generic coupled kernels by `ModelSpec.id` (no single-id hardcode); added a second demo model variant (homogeneous Neumann) and a regression test to ensure both codegen+runtime paths work.
- All integration tests and the UI now run through `GpuUnifiedSolver` only; legacy solver modules/types are crate-internal (no longer exported from `cfd2::solver::gpu`).

## Current Focus: Unified Solver Loop (Planned)
Goal: a single GPU solver loop that can run *any* coupled model described by `ModelSpec` (`src/solver/model/definitions.rs`), including:
- incompressible (coupled)
- compressible (KT flux)
- single-scalar models from tests (Laplace/Poisson/Heat/etc.)

The user should only need to:
- provide a `ModelSpec`
- select solver method + preconditioner(s)
- select temporal + spatial discretization schemes

Everything else (unknown layout, kernel sequencing, auxiliary computations, matrix sizing, etc.) should be derived automatically.

### Architecture Sketch
1. **Model lowering**: `EquationSystem` + `SchemeRegistry` → `DiscreteSystem` (already exists: `src/solver/codegen/ir.rs`).
2. **Scheme expansion**: inspect `DiscreteSystem` and add required auxiliary computations (e.g. gradients for SOU/QUICK, extra history buffers for BDF2, limiter/reconstruction inputs).
3. **Execution plan derivation**: `ModelSpec` + `SolverConfig` → `KernelGraph` (DAG) + resource plan (state/flux/aux/matrix buffers).
4. **Codegen**: `KernelGraph` → WGSL kernels + Rust bindings (build-time via `build.rs` + `wgsl_bindgen`).
5. **Runtime**: a generic dispatcher runs the `KernelGraph` for each timestep/nonlinear iteration and calls a generic linear solver with the chosen preconditioner chain.

### Milestones (Iterative)
1. **Unify the runtime interface (no behavior change)** (DONE: wrapper + config)
   - Introduce `SolverConfig` (time scheme, spatial scheme(s), linear solver, preconditioner chain, tolerances).
   - Add a thin `GpuUnifiedSolver` wrapper that can host the *existing* incompressible and compressible solvers behind one entrypoint (temporary adapters).
2. **Derive `kernel_plan` instead of hardcoding it** (DONE: derived accessor, still uses model-family mapping)
   - Replace `ModelSpec.kernel_plan` with a derived plan: `derive_kernel_plan(model, config)`.
   - Start by matching existing pipelines (incompressible: prepare→assemble→pressure→update→flux; compressible: flux→grad→assembly→apply→update).
3. **Unify state packing + unknown layout**
   - Make all models use `StateLayout`-driven packed state (like compressible) so init/update/plotting don’t need per-model structs.
   - Compute unknown count from equation targets (sum of field component counts); build CSR expansion generically for any block size.
   - Status: UI + read/write helpers + buffer init now use `StateLayout`; coupled solver still has a few `*3` assumptions outside init/debug paths.
4. **Generic scheme expansion**
   - Implement “needs gradients?” and “needs reconstruction?” detection per term/scheme.
   - Auto-generate gradient kernels per required field (instead of hardcoded `grad_rho`, `grad_u`, etc.).
   - Status: gradient *requirements* are derived from the model + `SchemeRegistry`; runtime can now skip gradient passes for first-order compressible stepping.
5. **Generic assembly/apply/update kernels for arbitrary coupled systems**
   - Generate matrix + RHS assembly for implicit terms and RHS-only contributions for explicit terms (from `DiscreteSystem`).
   - Generate an `apply(A, x)` kernel (SpMV + optional explicit parts) so the same linear solver kernels can be reused.
   - Generate time-integration updates (Euler/BDF2) from config (and required history buffers).
   - Status: generic `ddt` (Euler/BDF2) + `laplacian` assembly, generic CSR SpMV apply, and generic state update are emitted; BC handling and convection/source/cross-coupling are not yet implemented.
6. **Model “closures” as plugins (for incompressible + compressible specifics)**
   - Represent computed fluxes (Rhie–Chow, KT) and EOS/primitive recovery as explicit plan nodes (“closure kernels”) with declared inputs/outputs.
   - Keep these as small, swappable components so other models can plug in different closures.
7. **Boundary conditions become first-class and per-field**
   - Add `BoundarySpec` to `ModelSpec` (or a parallel structure) that maps (patch, field) → BC type + parameters with units.
   - Use typed enums in WGSL for BC kinds; codegen derives required BC handling for each operator.
   - Status: `BoundarySpec` exists and can be lowered to `(bc_kind, bc_value)` tables; generic coupled diffusion assembly uses it via a `bc_kind/bc_value` buffer (other kernels still use legacy hardcoded inlet/outlet/wall behavior).
8. **Test/validation automation**
   - Run all 1D cases across all combinations of temporal (Euler/BDF2) × spatial (Upwind/SOU/QUICK) schemes.
   - Store plots under `target/test_plots/*` with a consistent naming scheme; add a small summary index (text/HTML) to spot regressions quickly.
9. **Eliminate legacy public solvers** (DONE: public API)
   - Make `GpuUnifiedSolver` the only public GPU solver entrypoint.
   - Keep legacy implementations crate-internal until the generic runtime fully replaces them.
10. **Eliminate legacy solver *implementations* (Compressible/Coupled)**
   Goal: no `GpuCompressibleSolver` / `GpuSolver` “solvers” at all; only `GpuUnifiedSolver` instances with different plans + resources.
   Hard requirement: **no separate per-family lowering/compilation entrypoints**; everything flows through a single model-driven pipeline, with any “special sauce” expressed as pluggable closure modules.

   - Introduce a single model-driven “program” representation:
     - `ModelGpuProgramSpec`: a pure data description derived from `ModelSpec + SolverConfig + SchemeRegistry` that declares:
       - required resources (typed ports): mesh, packed state + history, auxiliary fields (gradients), linear system buffers (CSR + values + RHS/X), BC tables, closure-specific params.
       - compute nodes (in execution order or as a DAG): generic operators (assembly/apply/update/gradients) + closure kernels (KT flux, Rhie–Chow, primitive recovery/EOS, etc.).
       - typed option enums required by kernels (time scheme, advection scheme, BC kinds, closure variants).
     - `GpuProgramLowerer`: consumes `ModelGpuProgramSpec` and produces `ModelGpuProgram` (GPU resources):
       - allocates all buffers via `PortSpace`
       - builds pipelines + bind groups inside modules
       - performs layout compatibility checks (ports/bindings) during lowering (the only place that knows memory layout).

   - Make “closures” first-class modules (no branching in the lowering/compilation pipeline):
     - Each closure module declares its IO ports and required runtime dims, and owns its internal resources.
     - Closures are selected by the program builder based on `ModelSpec` (e.g. compressible gets KT flux + primitive recovery; incompressible gets Rhie–Chow/pressure-velocity coupling).

   - Unify compilation around `DiscreteSystem` where possible:
     - Generic operators (grad/assembly/apply/update) are emitted from `DiscreteSystem` for *all* models.
     - Family-specific kernels remain only as closures until their behavior can be expressed in the generic operator set.
     - Build-time codegen emits both generic operators and closures and registers them under a single `ModelGpuProgramSpec`.

   - Concrete incremental path (keep tests passing at every step):
     1) Add a generic `ModelLowerer` scaffold (mesh init + `PortSpace` + common sizing helpers) and migrate existing lowerers to use it.
     2) Port linear system resources (CSR, values, RHS/X) to typed ports everywhere (compressible → coupled → generic coupled).
     3) Convert plan execution to module graphs (`ModuleGraph`/`ExecutionPlan`) where every compute dispatch goes through a module node.
     4) Move remaining “init” logic (state/history/BC tables) into the unified lowerer so plan resources don’t allocate ad-hoc buffers.
     5) Replace `PlanInstance::{Compressible,Incompressible,GenericCoupled}` branching with a single `ModelGpuProgram` built from `ModelGpuProgramSpec`.

   - Introduce a plan/runtime split:
     - `GpuRuntimeCommon`: mesh buffers, constants, state ping-pong/history, and generic dispatch helpers.
     - `GpuPlanInstance`: plan-specific GPU resources (buffers/bind groups/pipelines) + an `ExecutionPlan` describing the timestep/nonlinear loop.
   - Move per-family resources out of “solver” structs:
     - `GpuCompressibleSolver` → `CompressiblePlanResources` (state offsets, flux/grad buffers, compressible linear buffers, graphs).
     - Coupled portion of `GpuSolver` → `CoupledPlanResources` (coupled matrix buffers, max-diff buffers, coupled graphs, async reader).
     - Keep `GpuSolver` only as a thin generic linear-solver helper (or fold into `GpuRuntimeCommon`) and delete coupled/compressible stepping methods entirely.
   - Remove `UnifiedSolverBackend` enum:
     - `GpuUnifiedSolver { common: GpuRuntimeCommon, plan: Box<dyn GpuPlanInstance> }` (or an internal `enum PlanInstance` if we want monomorphization without trait objects).
     - Construction selects a plan builder based on `ModelSpec` + `SolverConfig` (and registered closure plugins).
   - Make “closures” explicit plan nodes (already directionally true):
     - compressible: KT flux + primitive recovery
     - incompressible: flux update / pressure-velocity coupling kernels
   - Delete or move legacy files after migration:
     - `src/solver/gpu/compressible_solver.rs`, `src/solver/gpu/coupled_solver.rs` become plan modules/resources, not solvers.
     - update tests + UI to ensure no direct references remain (already routed through `GpuUnifiedSolver`).

### Gap Check (Are We Approaching The Goal?)
- The common *runtime sequencing* layer now exists (`KernelGraph` + `ExecutionPlan`) and both incompressible and compressible paths are progressively being expressed in it.
- The solver still relies on model-family-specific resource containers and some host-side control flow for convergence + linear solves; the next material step is to make **plans + plan resources** the only backend variants and to remove “solver” structs for each family.
- Even though the public API is unified, the runtime still uses legacy per-family structs internally. The remaining work to truly delete them is to (a) introduce `GpuRuntimeCommon` + `GpuPlanInstance`, (b) migrate compressible/coupled resources into plan resources, and (c) remove `UnifiedSolverBackend` + legacy step methods.

### Concrete Plan (Next Iteration: Module-First Runtime)
Goal: remove solver-owned pipelines/bind-groups and move *all GPU dispatch* behind first-class modules with typed ports, so lowering is the only place that knows layouts.

1. **Coupled (incompressible) kernel module**
   - Introduce `IncompressibleKernelsModule` (pipelines + bind groups + ping-pong state indexing).
   - Migrate `plans/coupled.rs` to use module-owned bind groups/pipelines (no `GpuSolver.bg_mesh/bg_fields`, no `GpuSolver.pipeline_*` physics fields).
   - Delete unused physics pipeline plumbing (`init/physics.rs`, `pipeline_pressure_assembly`, `pipeline_flux_rhie_chow`).
2. **Graph unification**
   - Standardize on `ModuleGraph` for compute dispatch (keep `ExecutionPlan` host/plugin steps).
   - Either (a) extend `ExecutionPlan` to support `ModuleGraph`, or (b) wrap module-graph execution in host nodes as a bridge, then delete `KernelGraph`.
3. **Ports everywhere**
   - Move coupled block-CSR + RHS/X into `PortSpace` like compressible (currently coupled has module preconditioners, but core coupled buffers are still embedded in `CoupledSolverResources`).
   - Migrate `GpuGenericCoupledSolver` to use the same port+module pattern (no bespoke bind group wiring).
4. **Single Krylov module**
   - Replace per-family FGMRES/GMRES glue with one `KrylovSolveModule` that owns its work buffers and bind groups and consumes a `LinearSystemPorts` + preconditioner module.
5. **Collapse backend variants**
   - Replace `PlanInstance::{Incompressible,Compressible,GenericCoupled}` with a single `ModelGpuProgram` built from a `ModelGpuProgramSpec` (closure modules selected from the model).
   - Remove specialized solver types/files once all tests + UI are routed through the unified program path.

### Decisions (Chosen)
- **Generated-per-model kernels**: keep the current approach (no runtime compilation/reflection).
- **Flux closures**: represent as plugin nodes in the `KernelGraph` (declared inputs/outputs), not as inline AST expressions.
- **Preconditioning**: selection is host-side (typed enums); compressible low-Mach parameters live in a dedicated uniform buffer, and linear-solver preconditioning is expressed via small pluggable modules.
- **Nonlinear iteration**: configurable (e.g. Picard/Newton); `SolverConfig` selects the strategy.

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
