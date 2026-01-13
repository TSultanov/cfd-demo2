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
- Build-time codegen exists (`crates/cfd2_codegen`) and emits WGSL into `src/solver/gpu/shaders/generated` from `build.rs` (no runtime compilation).
- Kernel schedule/phase/dispatch is model-owned and recipe-driven (`src/solver/model/kernel.rs`, `src/solver/gpu/recipe.rs`).
- Lowering uses a single universal spec/ops path (no per-family selection in `lower_program_model_driven`).
- Kernel registry lookup is uniform for all kernels via `(model_id, KernelId)` (`src/solver/gpu/lowering/kernel_registry.rs`).
- Host pipelines + bind groups for recipe kernels are built from binding metadata + a uniform `ResourceRegistry` (`src/solver/gpu/modules/generated_kernels.rs`).
- All major stepping backends (coupled, explicit/implicit, generic-coupled) use `GeneratedKernelsModule` (no kernel-specific host bind-group wiring).

## Completed Milestones (implemented, removed from “gaps”)
- Metadata-driven bind groups for all recipe kernels via `ResourceRegistry` + `GeneratedKernelsModule`.
- Single generated-kernel module used across stepping backends (no `ModelKernelsModule`, no per-family kernel modules).
- Generic-coupled integrated as a backend variant inside `UniversalProgramResources` (no separate plan selection path).
- Kernel constants/time integration are owned by the recipe-level `UnifiedFieldResources` (no per-backend constants buffer).
- `build.rs` discovers models via `definitions.rs` (adding a model does not require editing `build.rs`).
- Build-time “compiler” code is model/PDE-agnostic and lives in `crates/cfd2_codegen` (no `src/solver/compiler` module).
- Handwritten infrastructure kernels can be routed through the same kernel registry + binding-metadata path (e.g. `generic_coupled_schur_setup`).
- Removed field-name-specific bridges (`CodegenIncompressibleMomentumFields`); coupled kernels accept names and models provide per-kernel derived codegen fields (`derive_kernel_codegen_fields_for_model`).
- Derived primitive recovery ordering is dependency-ordered (toposort) so derived→derived references are well-defined.
- Mesh-level CSR adjacency buffers are owned by mesh resources (no per-backend duplication), and CSR binding semantics are explicitly DOF/system-level where required.
- Removed transitional `system_main.wgsl` artifact/hack (no “representative model” generation).

## Remaining Gaps (what blocks “fully model-agnostic”)

### 1) Plan Resources (single universal runtime resource graph)
- Finish converging plan/runtime plumbing between explicit/implicit and generic-coupled so new solver modes do not introduce new plan/resource structs or wiring paths.
- Document the canonical linear-system representation(s) (DOF/system CSR vs scalar CSR used for mesh adjacency), and make all kernel bindings reflect that choice consistently.

### 2) Codegen (remove model-specific generators; IR-driven kernels)
- Retire legacy coupled-incompressible kernel generators (`prepare_coupled`, `coupled_assembly`, `pressure_assembly`, `update_fields_from_coupled`, `flux_rhie_chow`) and drive assembly/update/flux kernels from IR + layout only (no 2D-only assumptions baked into the generator).
- Define a stable “flux module contract” so KT/Rhie–Chow become module configurations writing packed face fluxes consistent with `FluxLayout` (no special-case scheduling/bindings).
- Reduce build-time kernel-specific glue:
  - `build.rs` does not implement per-kernel generation; it loops models and delegates to model/kernel helpers for emission.
- Converge duplicated WGSL AST sources (`src/solver/shared/wgsl_ast.rs` vs `crates/cfd2_codegen/src/solver/codegen/wgsl_ast.rs`).

### 3) Retire `PlanParam` as global plumbing
- Replace `PlanParam`-based “global knobs” with typed, module-owned uniforms/config deltas routed through the recipe.
- Done when: new configuration does not add `PlanParam` enum cases.

### 4) Handwritten WGSL (treat infrastructure the same way)
- Move handwritten solver infrastructure shaders under `src/solver/gpu/shaders` behind the same registry/metadata mechanism and treat them as registry-provided artifacts (even if template-generated).
- Done when: runtime consumes only registry-provided WGSL (no ad-hoc `include_str!` modules).

### 5) Contract Tests
- Add regression tests that fail if:
  - `kernel_registry` has special-case lookup paths
  - lowering selects a backend by “family”
  - adding a kernel requires editing a central `match` (bind groups / pipeline selection)
  - `build.rs` contains per-model hardcoding or kernel-kind-specific codegen glue

## Recommended Sequence (high leverage)
1) Converge plan resources into one universal runtime graph/resources layer.
2) Retire model-specific codegen bridges and handwritten physics kernels; keep solver infrastructure next.
3) Retire `PlanParam` and add contract tests to lock in invariants.

## Decisions (Locked In)
- Generated-per-model WGSL stays (no runtime compilation).
- Options selection uses typed enums.

## Notes / Constraints
- `build.rs` uses `include!()`; build-time-only modules must be wired there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
