# CFD2 Codegen + Unified Solver Plan

This file tracks *remaining* codegen + solver orchestration work. Pure physics/tuning tasks should live elsewhere.

## Goal
One **model-driven** GPU solver pipeline with:
- a single lowering + codegen path (no separate compressible/incompressible lowerers/compilers)
- a single runtime orchestration path (no per-family step loops)
- solver features implemented as **pluggable modules** that own their own GPU resources
- kernels + auxiliary passes derived automatically from `ModelSpec` + solver config (no handwritten WGSL)

## Current State (Implemented)
- **Single lowering entrypoint exists:** Lowering routes through `src/solver/gpu/lowering/model_driven.rs` and is recipe-aware.
- **Recipe abstraction exists:** `src/solver/gpu/recipe.rs` defines `SolverRecipe` which drives ProgramSpec generation and resource requirements.
- **Graph building is recipe-driven:** `src/solver/gpu/modules/unified_graph.rs` builds graphs from recipe phases and supports composite phase sequences.
- **Unified resources exist:** `UnifiedFieldResources` allocates from recipe requirements; `FieldProvider` + bind-group builder use shader binding names.
- **Kernel identifiers exist:** `KernelId(&'static str)` exists and is used for id-based kernel lookup in parts of the runtime.
- **Kernel requirements are derived:** `ModelSpec::kernel_plan()` routes through `derive_kernel_plan(system)`.
- **Generic solver infrastructure exists:** `GenericLinearSolverModule<P>` exists; FGMRES driver logic is extracted.

## Remaining Gaps (Blocking “any model + any method”)
- **Lowering still selects a template family:** `ProgramTemplateKind` + `lowering/templates.rs` are still in the runtime path.
- **KernelKind still leaks into orchestration:** `SolverRecipe` and some plans still iterate kernels via `KernelKind` / `kernel_registry::kernel_source(...)`.
- **Kernel registry is not fully scheme-expanded:** kernel lookup relies on build-time generated tables (`kernel_registry_map.rs`) that are not purely derived from “model + selected numerics”.
- **Model GPU requirements are not fully generated:** `ModelGpuSpec`/resource policies still carry some handwritten or heuristic requirements.
- **Bind groups are not fully reflection-driven:** current `field_binding()` is a migration helper; end state should not require plan/module-specific binding code.
- **Config deltas are ad-hoc:** `PlanParam` is still used for host → plan configuration.
- **Some handwritten WGSL remains:** solver-family-specific shaders still exist outside codegen.
- **Preconditioner pluggability incomplete:** compressible linear solver hasn’t fully converged onto the generic module + factory abstraction.

## Remaining Work (Single Prioritized List)

1. **Make the recipe the sole lowering/orchestration authority (remove template selection).**
   - Remove `ProgramTemplateKind` routing from `model_driven`.
   - Ensure op registration and ProgramSpec derivation are entirely recipe-driven.
   - Success: adding a new model/method does not require edits to `lowering/templates.rs` or new template kinds.

2. **Eliminate `KernelKind` from runtime orchestration (keep only as debug/UI bridge).**
   - Make recipes and graphs operate purely on `KernelId`.
   - Remove remaining `kernel_registry::kernel_source(model.id, KernelKind::...)` call sites from plans.
   - Success: kernel scheduling/execution never matches on `KernelKind`.

3. **Make the kernel registry fully derived from codegen + selected numerics (scheme expansion).**
   - Codegen should emit the authoritative `KernelId -> wgsl + binding metadata (+ workgroup sizes if needed)` mapping.
   - Runtime lookup should be id-based everywhere (`kernel_source_by_id`).
   - Success: registering a new kernel requires only codegen output changes, not handwritten tables.

4. **Make GPU resource requirements fully derived (minimize/retire handwritten `ModelGpuSpec`).**
   - Move “always required buffers” (flux stride, gradient storage policy, low-mach params, history buffers) to recipe/specs generated from model terms + method choices.
   - Keep manual overrides only where unavoidable and explicitly documented.
   - Success: model constructors stop encoding solver-method-specific GPU resource policy.

5. **Finish reflection-driven bind groups and retire remaining per-family bind logic.**
   - Use codegen-emitted binding metadata + `FieldProvider`/resource registry to build bind groups uniformly.
   - Success: adding a kernel with new bindings requires no handwritten host-side bind-group assembly.

6. **Replace `PlanParam` with typed config deltas (`SolverConfigDelta` + module-specific deltas).**
   - Generate deltas where possible; route updates through module-owned handlers.
   - Success: no new features add ad-hoc `PlanParam` matches.

7. **Complete linear solver & preconditioner pluggability.**
   - Migrate `CompressibleLinearSolver` to the generic module infrastructure.
   - Add `PreconditionerFactory` implementations for Jacobi, AMG, Schur.
   - Success: preconditioner selection is config-driven and solver-family-agnostic.

8. **Eliminate remaining handwritten WGSL.**
   - Migrate remaining solver-family shaders (e.g. `compressible_*`, `schur_precond`) into codegen.
   - Success: runtime consumes generated WGSL only.

9. **Legacy glue cleanup.**
   - Remove unused init modules/exports and dead legacy bind-group ownership codepaths.
   - Success: a single owner exists for each GPU resource and each runtime feature lives in a module.

## Decisions (Locked In)
- **Generated-per-model WGSL** stays (no runtime compilation/reflection).
- **Flux/EOS closures** are explicit plan/modules, not inline expressions.
- **Options selection** uses typed enums (no strings).

## Notes / Constraints
- `build.rs` `include!()`s codegen modules; build-time-only modules must be added there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
