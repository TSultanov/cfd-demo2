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

## Unified Solver Progress (So Far)
- Added `GpuUnifiedSolver` + `SolverConfig` as a single entrypoint wrapper over the existing incompressible/compressible GPU solvers (`src/solver/gpu/unified_solver.rs`).
- Removed `ModelSpec.kernel_plan` storage; kernel plans are derived via `ModelSpec::kernel_plan()` (still model-family specific for now).
- Shared block-CSR expansion (`src/solver/gpu/csr.rs`) and derived compressible unknown count from `EquationSystem::unknowns_per_cell()`.
- Introduced a small runtime `KernelGraph` executor (`src/solver/gpu/kernel_graph.rs`) and migrated the compressible explicit KT timestep path to it.
- Migrated the compressible implicit path’s GPU dispatch blocks (grad/assembly/apply/update) to `KernelGraph` as well (FGMRES remains a host-controlled plugin step for now).
- Migrated the incompressible coupled solver’s GPU dispatch blocks (prepare/assembly/update) to prebuilt `KernelGraph`s; the coupled FGMRES solve + convergence logic remains host-controlled for now.

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
4. **Generic scheme expansion**
   - Implement “needs gradients?” and “needs reconstruction?” detection per term/scheme.
   - Auto-generate gradient kernels per required field (instead of hardcoded `grad_rho`, `grad_u`, etc.).
5. **Generic assembly/apply/update kernels for arbitrary coupled systems**
   - Generate matrix + RHS assembly for implicit terms and RHS-only contributions for explicit terms (from `DiscreteSystem`).
   - Generate an `apply(A, x)` kernel (SpMV + optional explicit parts) so the same linear solver kernels can be reused.
   - Generate time-integration updates (Euler/BDF2) from config (and required history buffers).
6. **Model “closures” as plugins (for incompressible + compressible specifics)**
   - Represent computed fluxes (Rhie–Chow, KT) and EOS/primitive recovery as explicit plan nodes (“closure kernels”) with declared inputs/outputs.
   - Keep these as small, swappable components so other models can plug in different closures.
7. **Boundary conditions become first-class and per-field**
   - Add `BoundarySpec` to `ModelSpec` (or a parallel structure) that maps (patch, field) → BC type + parameters with units.
   - Use typed enums in WGSL for BC kinds; codegen derives required BC handling for each operator.
8. **Test/validation automation**
   - Run all 1D cases across all combinations of temporal (Euler/BDF2) × spatial (Upwind/SOU/QUICK) schemes.
   - Store plots under `target/test_plots/*` with a consistent naming scheme; add a small summary index (text/HTML) to spot regressions quickly.

### Decisions (Chosen)
- **Generated-per-model kernels**: keep the current approach (no runtime compilation/reflection).
- **Flux closures**: represent as plugin nodes in the `KernelGraph` (declared inputs/outputs), not as inline AST expressions.
- **Preconditioning**: default to generic Jacobi for all models; compressible can add specialized preconditioners.
- **Nonlinear iteration**: configurable (e.g. Picard/Newton); `SolverConfig` selects the strategy.

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; new build-time-only modules must be added there.
- Build scripts are std-only (plus `wgsl_bindgen`/`glob` build-deps); avoid pulling runtime-only crates into code referenced by `build.rs`.
