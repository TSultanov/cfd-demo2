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
- A unified recipe exists and emits a `ProgramSpec` (`src/solver/gpu/recipe.rs`).
- Build-time codegen exists (`crates/cfd2_codegen`) and emits WGSL into `src/solver/gpu/shaders/generated` via `src/solver/compiler/emit.rs`.
- Stable `KernelId` lookup exists via build-time generated tables (`src/solver/gpu/lowering/kernel_registry.rs`).
- Shared field allocation exists and is recipe-driven (`src/solver/gpu/modules/unified_field_resources.rs`).

## Remaining Gaps (what blocks “fully model-agnostic”)

### 1) Lowering (stop per-family plan selection)
- Converge `ExplicitImplicitPlanResources` and `GenericCoupledPlanResources` into a single universal plan/resources path (recipe-driven graphs + shared resource registry).
- Done when: `lower_program_model_driven` never switches lowering paths based on kernel presence or “stepping family”.

### 2) Kernel Registry (one uniform lookup path)
- Remove the generic-coupled special-case (`generated::generic_coupled_pair(model_id)`) in `src/solver/gpu/lowering/kernel_registry.rs`.
- Emit a single table keyed by `(model_id, KernelId)` for *all* kernels (including per-model kernels and “infrastructure” kernels; use a sentinel model id like `__global__` for non-model kernels).
- Done when: `kernel_registry` has exactly one lookup function and no special cases.

### 3) Bindings (metadata-driven bind groups)
- Introduce a uniform `ResourceRegistry` keyed by binding name/role (mesh buffers, ping-pong state, gradients, fluxes, BC tables, solver workspaces, uniforms).
- Build bind groups exclusively from generated binding metadata + `ResourceRegistry` (no per-kernel bind-group wiring).
- Migrate existing partial helpers (`wgsl_reflect`, `UnifiedFieldResources::buffer_for_binding`) into this uniform registry.

### 4) Kernel Modules (single generated-kernel module)
- Merge per-family kernel modules (`src/solver/gpu/modules/model_kernels.rs`, `src/solver/gpu/modules/generic_coupled_kernels.rs`) into one module that:
  - iterates `recipe.kernels`
  - builds pipelines via `(model_id, KernelId)`
  - builds bind groups via binding metadata + `ResourceRegistry`
- Done when: adding a kernel does not require editing host-side module code or adding new `match` arms.

### 5) Codegen (remove model-specific generators; IR-driven kernels)
- Retire field-name-specific bridges (e.g. `CodegenIncompressibleMomentumFields`) and kernel generators that assume fixed layouts; drive assembly/update kernels from IR + layout only.
- Define a stable “flux module contract” so KT/Rhie–Chow become just module configurations that write packed face fluxes consistent with `FluxLayout` (no special-case scheduling/bindings).
- Add derived-primitive dependency validation/toposort (or explicitly forbid derived→derived at validation time).
- Converge duplicated WGSL AST sources (`src/solver/shared/wgsl_ast.rs` vs `crates/cfd2_codegen/src/solver/codegen/wgsl_ast.rs`).

### 6) Handwritten WGSL (treat infrastructure the same way)
- Move remaining handwritten solver infrastructure shaders under `src/solver/gpu/shaders` behind the same registry/metadata mechanism and treat them as generated artifacts (even if template-generated).
- Done when: runtime consumes only “registry-provided” WGSL (no ad-hoc `include_str!` modules).

### 7) Build-Time Model Discovery (edit `definitions.rs` only)
- Remove hard-coded model enumeration in `build.rs` (the explicit calls to `*_model()`).
- Expose a model registry from `src/solver/model/definitions.rs` (e.g. `pub fn all_models() -> Vec<ModelSpec>`), and have `build.rs` iterate it to emit kernels/registries.
- Done when: adding a model requires editing only `src/solver/model/definitions.rs`.

### 8) Retire `PlanParam` as global plumbing
- Replace `PlanParam`-based “global knobs” with typed, module-owned uniforms/config deltas routed through the recipe.
- Done when: new configuration does not add `PlanParam` enum cases.

### 9) Contract Tests
- Add regression tests that fail if:
  - `kernel_registry` has special-case lookup paths
  - lowering selects a backend by “family”
  - adding a kernel requires editing a central `match` (bind groups / pipeline selection)
  - `build.rs` contains per-model hardcoding

## Recommended Sequence (high leverage)
1) Unify kernel registry to a single `(model_id, KernelId)` lookup.
2) Add `ResourceRegistry` and migrate bind-group construction to metadata-driven assembly.
3) Merge kernel modules into a single generated-kernel module and delete per-family modules.
4) Remove backend selection in `lower_parts_for_model` by making all lowering use the same universal resources/graphs.
5) Retire model-specific codegen bridges and handwritten physics kernels; keep solver infrastructure next.

## Decisions (Locked In)
- Generated-per-model WGSL stays (no runtime compilation).
- Options selection uses typed enums.

## Notes / Constraints
- `build.rs` uses `include!()`; build-time-only modules must be wired there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
