# CFD2 Codegen + Unified Solver Plan

This file tracks *remaining* codegen + solver orchestration work needed to reach a **single unified solver** where runtime behavior is derived from **(ModelSpec + selected numerical methods)** only.

## Goal
One model-driven GPU solver pipeline with:
- a single lowering + codegen path (no compressible/incompressible “families” in lowering)
- a single runtime orchestration path (no per-family step loops or templates)
- solver features as pluggable modules that own their GPU resources
- kernels + auxiliary passes derived automatically from model structure + method selection (no handwritten WGSL)

## Non-negotiable invariants (for “any model + any method”)
- **No solver-family switches in orchestration:** runtime must not branch on “compressible vs incompressible vs generic”.
- **No handwritten kernel scheduling:** ordering/phase membership is part of the recipe emitted from model+methods.
- **No handwritten kernel lookup tables:** runtime only asks for `KernelId` and receives generated WGSL + binding metadata.
- **No handwritten bind groups:** host binding is assembled from generated binding metadata + a uniform resource registry.
- **Extensibility contract:** adding a new PDE term or method should require edits only in codegen/model metadata + the term/method expander, not in runtime orchestration.

## Current State (Implemented)
- Single lowering entrypoint exists and is recipe-aware (`src/solver/gpu/lowering/model_driven.rs`).
- `SolverRecipe` exists and already drives:
  - many resource requirements (gradients/history/flux/low-mach params)
  - `KernelId`-based kernel specs
  - ProgramSpec derivation in some paths
- Recipe-driven graph building exists (phase graphs + composite phase sequences).
- Unified field allocation exists (`UnifiedFieldResources`) and bind-group building is partially unified via `FieldProvider` helpers.
- `KernelId(&'static str)` exists and `kernel_registry::kernel_source_by_id` exists.
- Kernel requirements are structurally derived (`derive_kernel_plan(system)`), and `ModelSpec::kernel_plan()` routes through it.
- Generic linear solver infrastructure exists (`GenericLinearSolverModule<P>`) and FGMRES driver logic is extracted.

## Remaining Work (Single Prioritized List)

1. **Make `SolverRecipe` authoritative for kernel ordering and phase assignment (no central matches).**
   - Today: recipe still has `phase_for_id(...)` and `dispatch_for_id(...)` matches, and kernel planning still originates as `KernelKind`.
   - Change: recipe construction assigns phase/dispatch as it emits `KernelSpec` from scheme expansion + method selection.
   - Done when: no new kernel requires editing a central `match` in runtime to place it in a phase/dispatch.

2. **Remove template-driven orchestration (`lowering/templates.rs`, `ProgramTemplateKind`) from the runtime path.**
   - Today: model lowering still selects per-family “template op sets” (compressible/incompressible) and registers the full template ops.
   - Change: ProgramSpec and op registration become recipe-driven for all models (including current compressible/incompressible).
   - Done when: adding a new model does not require a new template kind or edits to template registries.

3. **Eliminate `KernelKind` from orchestration (keep only as optional debug/UI bridge).**
   - Today: `derive_kernel_plan` synthesizes `KernelKind`, and some runtime code still bridges through it.
   - Change: kernel planning and recipes are in terms of `KernelId` end-to-end.
   - Done when: graph building/execution never matches on or iterates `KernelKind`.

4. **Make the kernel registry 100% codegen-derived and scheme-expanded (no handwritten kernel tables).**
   - Today: there is still a build-time generated map, but it is not guaranteed to be the sole authoritative output of “model + numerics expansion”.
   - Change: codegen emits, per model, the complete `KernelId -> (pipeline ctor, bindings, wgsl)` table for exactly the expanded kernels.
   - Done when: introducing a new kernel requires only codegen output changes (and its generator), not runtime registry edits.

5. **Finish reflection-driven bind groups via a uniform resource registry (retire plan-specific binding logic).**
   - Today: `FieldProvider` + helpers reduce boilerplate, but some bindings are still wired in plan/module code.
   - Change: bind group layouts + entries are derived entirely from generated binding metadata, with a resolver that maps binding names to buffers/uniforms.
   - Done when: adding a binding to a kernel never requires handwritten host-side bind-group assembly.

6. **Derive GPU resource requirements fully from recipe specs (minimize/retire handwritten GPU policies in models).**
   - Today: some “always required” buffers/policies remain expressed as handwritten `ModelGpuSpec`/heuristics.
   - Change: recipe/spec generation (from model terms + method selection) declares all required buffers and storage policies.
   - Done when: model constructors no longer encode method-dependent resource policy, except for explicitly documented overrides.

7. **Replace `PlanParam` with typed config deltas (module-owned).**
   - Today: `PlanParam` is still used to push ad-hoc config into plans.
   - Change: generated (or strongly typed) `SolverConfigDelta` + module-specific deltas, routed to the owning module.
   - Done when: configuration updates do not add new global enum cases or stringly plumbing.

8. **Complete linear solver & preconditioner pluggability (family-agnostic).**
   - Migrate compressible/incompressible linear solvers onto `GenericLinearSolverModule<P>`.
   - Add/finish `PreconditionerFactory` for Jacobi, AMG, Schur.
   - Done when: selecting a preconditioner is config-only and independent of model family.

9. **Eliminate remaining handwritten WGSL (all kernels come from codegen output).**
   - Migrate remaining solver-family shaders (e.g. `compressible_*`, `schur_precond`) into the codegen pipeline.
   - Done when: runtime consumes generated WGSL only; no handwritten WGSL is required for correctness.

10. **Delete legacy glue paths and enforce the contract with tests.**
   - Remove unused init modules/exports and dead code paths.
   - Add regression tests that fail if:
     - a required phase is empty but executed
     - new kernels require edits to central phase/dispatch matches
     - template kinds are needed for new models

## Near-term recommended sequence (practical)
1) (1)+(3): move phase/dispatch + planning to recipe in terms of `KernelId`.
2) (2): rip out template-driven op sets and make all programs recipe-driven.
3) (4)+(5): make codegen tables + bindings authoritative and bind-group assembly uniform.

## Decisions (Locked In)
- Generated-per-model WGSL stays (no runtime compilation).
- Flux/EOS closures remain explicit plan/modules (not inline expressions).
- Options selection uses typed enums.

## Notes / Constraints
- `build.rs` uses `include!()`; build-time-only modules must be wired there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
