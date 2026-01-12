# CFD2 Model-Driven Solver + Codegen Plan

This file tracks *remaining* work to reach a **fully model-agnostic solver** where the set of model-dependent kernels is derived automatically from the **math in `ModelSpec.system`** (as declared in `src/solver/model/definitions.rs`) plus selected numerical methods.

## Target State (definition of done)
- A single runtime orchestration path (no per-family stepping templates or “compressible/incompressible” branches).
- A single lowering path (no per-family plan selection; one universal plan/resource model).
- Kernel schedule emitted as part of a recipe derived from `(ModelSpec + method selection + runtime config)`.
- All model-dependent WGSL is generated (no handwritten physics kernels).
- Host bind groups are assembled purely from generated binding metadata + a uniform resource registry (no `KernelId`-specific bind-group code).
- Adding a new model/PDE requires editing only `src/solver/model/definitions.rs` (and model-side helpers under `src/solver/model/*`), not lowering/codegen/runtime glue.

## Non-Negotiable Invariants
- **No solver-family switches in orchestration:** runtime must not branch on “compressible vs incompressible vs generic”.
- **No handwritten kernel scheduling:** ordering/phase membership is emitted by the model+method recipe.
- **No handwritten kernel lookup tables:** runtime asks for `(model_id, KernelId)` and receives generated WGSL + binding metadata.
- **No handwritten bind groups:** host binding is assembled from metadata + a uniform resource registry.
- **Codegen is PDE-agnostic:** `crates/cfd2_codegen/src/solver/codegen` must not depend on `src/solver/model` (enforced by `build.rs:enforce_codegen_ir_boundary`).

## Current Baseline (already implemented, for context only)
- Models are expressed as `EquationSystem` via the fvm/fvc AST in `src/solver/model/definitions.rs`.
- The model emits a kernel schedule/dispatch plan that the recipe consumes (`src/solver/model/kernel.rs`, `src/solver/gpu/recipe.rs`).
- Build-time codegen exists (`crates/cfd2_codegen`) and emits WGSL into `src/solver/gpu/shaders/generated` via `src/solver/compiler/emit.rs`.
- Lowering uses a single universal spec/ops path (no per-family selection in `lower_program_model_driven`).
- Kernel registry lookup is uniform for all kernels via a single generated mapping function (`src/solver/gpu/lowering/kernel_registry.rs`).
- Shared field allocation exists and is recipe-driven (`src/solver/gpu/modules/unified_field_resources.rs`).

## Remaining Gaps (what blocks “fully model-agnostic”)

### 1) Bindings (metadata-driven bind groups)
- Expand the uniform `ResourceRegistry` (now exists) to cover all binding roles used by kernels (mesh buffers, ping-pong state, gradients, fluxes, BC tables, solver workspaces, uniforms).
- Build bind groups exclusively from generated binding metadata + `ResourceRegistry` (no per-kernel bind-group wiring); `GeneratedKernelsModule` is the reference implementation.
- Continue migrating existing partial helpers (`BindGroupBuilder`, `UnifiedFieldResources::buffer_for_binding`, ad-hoc closures) behind this uniform registry.

### 2) Kernel Modules (single generated-kernel module)
- Finish migrating all solver families to a single generated-kernel module (`src/solver/gpu/modules/generated_kernels.rs`) that:
  - iterates `recipe.kernels`
  - builds pipelines via `(model_id, KernelId)`
  - builds bind groups via binding metadata + `ResourceRegistry`
- Progress: generic-coupled and coupled paths now use `GeneratedKernelsModule` (no per-kernel host bind-group wiring).
- Progress: explicit/implicit now also uses `GeneratedKernelsModule`; `ModelKernelsModule` is deleted.
- Done when: adding a kernel does not require editing host-side module code or adding new `match` arms.

### 3) Plan Resources (single universal runtime resource graph)
- Converge `ExplicitImplicitPlanResources` and `GenericCoupledPlanResources` into one universal plan/resources layer (recipe-driven graphs + shared resource registry).
- Done when: new solver modes do not introduce new plan/resource structs or wiring paths.

### 4) Codegen (remove model-specific generators; IR-driven kernels)
- Retire field-name-specific bridges (e.g. `CodegenIncompressibleMomentumFields`) and kernel generators that assume fixed layouts; drive assembly/update kernels from IR + layout only.
- Define a stable “flux module contract” so KT/Rhie–Chow become just module configurations that write packed face fluxes consistent with `FluxLayout` (no special-case scheduling/bindings).
- Add derived-primitive dependency validation/toposort (or explicitly forbid derived→derived at validation time).
- Converge duplicated WGSL AST sources (`src/solver/shared/wgsl_ast.rs` vs `crates/cfd2_codegen/src/solver/codegen/wgsl_ast.rs`).

### 5) Handwritten WGSL (treat infrastructure the same way)
- Move remaining handwritten solver infrastructure shaders under `src/solver/gpu/shaders` behind the same registry/metadata mechanism and treat them as generated artifacts (even if template-generated).
- Done when: runtime consumes only “registry-provided” WGSL (no ad-hoc `include_str!` modules).

### 6) Build-Time Model Discovery (edit `definitions.rs` only)
- Remove hard-coded model enumeration in `build.rs` (the explicit calls to `*_model()`).
- Expose a model registry from `src/solver/model/definitions.rs` (e.g. `pub fn all_models() -> Vec<ModelSpec>`), and have `build.rs` iterate it to emit kernels/registries.
- Done when: adding a model requires editing only `src/solver/model/definitions.rs`.

### 7) Retire `PlanParam` as global plumbing
- Replace `PlanParam`-based “global knobs” with typed, module-owned uniforms/config deltas routed through the recipe.
- Done when: new configuration does not add `PlanParam` enum cases.

### 8) Contract Tests
- Add regression tests that fail if:
  - `kernel_registry` has special-case lookup paths
  - lowering selects a backend by “family”
  - adding a kernel requires editing a central `match` (bind groups / pipeline selection)
  - `build.rs` contains per-model hardcoding

## Recommended Sequence (high leverage)
1) Migrate bind-group construction to metadata-driven assembly (`ResourceRegistry` everywhere).
2) Merge kernel modules into a single generated-kernel module and delete per-family modules.
3) Converge plan resources into one universal runtime graph/resources layer.
4) Retire model-specific codegen bridges and handwritten physics kernels; keep solver infrastructure next.
5) Remove build-time per-model hardcoding (discover models from `definitions.rs`).

## Decisions (Locked In)
- Generated-per-model WGSL stays (no runtime compilation).
- Options selection uses typed enums.

## Notes / Constraints
- `build.rs` uses `include!()`; build-time-only modules must be wired there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
