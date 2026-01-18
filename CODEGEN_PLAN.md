# CFD2 Model-Driven Solver + Codegen Plan

This file tracks *remaining* work to reach a **fully model-agnostic solver** where the set of model-dependent kernels is derived automatically from the **math in `ModelSpec.system`** (as declared in `src/solver/model/definitions.rs`) plus selected numerical methods.

## Target State (definition of done)
- A single runtime orchestration path (no per-family stepping templates or “compressible/incompressible” branches).
- A single lowering path (no per-family plan selection; one universal plan/resource model).
- Kernel schedule emitted as part of a recipe derived from `(ModelSpec + method selection + runtime config)`.
- All model-dependent WGSL is generated (no handwritten physics kernels).
- Host bind groups are assembled purely from generated binding metadata + a uniform resource registry (no `KernelId`-specific bind-group code).
- Adding a new model/PDE requires editing only `src/solver/model/definitions.rs` (and model-side helpers under `src/solver/model/*`), not lowering/codegen/runtime glue.
- The *public* solver API is model-agnostic: no built-in notions of `U/p/rho`, `gamma`, “inlet velocity”, or “incompressible-only” controls in `UnifiedSolver` itself (those belong in model-specific helper layers, not the core solver).

## Non-Negotiable Invariants
- **No solver-family switches in orchestration:** runtime must not branch on “compressible vs incompressible vs generic”.
- **No handwritten kernel scheduling:** ordering/phase membership is emitted by the model+method recipe.
- **No handwritten kernel lookup tables:** runtime asks for `(model_id, KernelId)` and receives generated WGSL + binding metadata.
- **No handwritten bind groups:** host binding is assembled from metadata + a uniform resource registry.
- **Codegen is PDE-agnostic:** `crates/cfd2_codegen/src/solver/codegen` must not depend on `src/solver/model` (enforced by `build.rs:enforce_codegen_ir_boundary`).

## Development Cadence (required)
To keep unification/refactor work low-risk, follow a strict loop:

- **test → patch → test → commit** (small, frequent commits)
- **OpenFOAM reference tests are the primary quality gate** for solver behavior:
  - `bash scripts/run_openfoam_reference_tests.sh`
    - (equivalent to `cargo test -p cfd2 --tests openfoam_ -- --nocapture`, but fails loudly if 0 tests are discovered)
- Prefer a short, targeted test before/after each patch (e.g. `cargo test contract_` or the specific failing test), but do not skip OpenFOAM when the change can affect numerics/orchestration.
- If tests are skipped or filtered unintentionally (CI/sandbox/GPU env), call it out explicitly in the PR/notes.

## Current Baseline (already implemented, for context only)
- Models are expressed as `EquationSystem` via the fvm/fvc AST in `src/solver/model/definitions.rs`.
- Build-time codegen exists (`crates/cfd2_codegen`) and emits WGSL into `src/solver/gpu/shaders/generated` from `build.rs` (no runtime compilation).
- Kernel schedule/phase/dispatch is model-owned and recipe-driven (`src/solver/model/kernel.rs`, `src/solver/gpu/recipe.rs`).
- Model modules carry a small manifest (method/flux selection, named params, typed invariants) that is validated explicitly.
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
- Converged duplicated WGSL AST sources (single canonical implementation in codegen).
- Explicit stepping uses the same generic-coupled backend resources (no separate explicit/implicit plan/resources path).
- Documented CSR binding/shape contract (`src/solver/gpu/CSR_CONTRACT.md`) and clarified scalar-vs-DOF naming in mesh resources.
- Retired the legacy coupled backend resources (`GpuSolver`); coupled stepping runs on the same universal `GenericCoupledProgramResources` path.
- Stabilized the face-flux buffer contract: flux modules write a packed per-unknown-component face table (`flux_stride = system.unknowns_per_cell()`), and unified assembly indexes fluxes as `fluxes[face * flux_stride + u_idx]` (no scalar-stride special-casing).
- Flux-module kernels are emitted per-model when they depend on `(ModelSpec.system, StateLayout, FluxLayout)` (no shared `flux_*.wgsl` artifacts across models).
- Flux-module scheduling/lookup is unified behind stable `KernelId`s (`flux_module_gradients`, `flux_module`) rather than per-method kernel ids.
- Retired handwritten flux WGSL generators (`flux_kt`, `flux_rhie_chow`) in favor of a PDE-agnostic IR-driven `flux_module` codegen path; flux math is declared in `src/solver/model/definitions.rs` and compiled generically.
- Added initial contract tests to prevent regressions toward kernel-id/model-id special-casing (`tests/contract_codegen_invariants_test.rs`).
- Moved low-Mach/preconditioning knobs off the core solver type and into helper traits (`src/solver/model/helpers/solver_ext.rs`) to keep `GpuUnifiedSolver` closer to model-agnostic.

## Remaining Gaps (what blocks “fully model-agnostic”)

### 0) Factor per-model “kernels” into pluggable model-defined modules
We recently added additional **per-model generated kernels** (e.g. `dp_update_from_diag_*`, `rhie_chow_correct_velocity_*`) as first steps towards better Rhie–Chow behavior.

Those kernels *should not* become permanent solver-core concepts. In the target architecture:
- The solver core is **model-agnostic** and does not grow new kernel kinds/IDs whenever a numerical method needs an extra pass.
- Models/methods provide **pluggable modules** (small kernel bundles) that can be composed into a recipe.
- `build.rs` does **not** need per-prefix/per-kernel glue to discover “special” per-model WGSL files.

Target:
- Replace the ad-hoc “KernelKind grows forever” pattern with a **module interface**:
  - A module declares a set of `ModelKernelSpec { id, phase, dispatch }` entries.
  - A module declares (or can emit) its WGSL kernels + binding metadata.
  - A module declares a **manifest** (method/flux selection, named params, invariants) that is merged with explicit conflicts.
  - The unified solver runs *modules*; it does not interpret “Rhie–Chow” or “d_p” directly.
- Models select and configure modules (or a method selects modules) via `ModelSpec` (or a derived recipe), e.g.:
  - `flux_module` (KT, Rhie–Chow, …)
  - `aux_field_updates` (e.g. `d_p <- momentum_diag`)
  - `corrections` (e.g. `U <- U - d_p * grad(p)`), if still needed
- Eliminate `build.rs` per-kernel discovery by generating a **kernel manifest** at build time:
  - A single generated table of `(model_id, kernel_id) -> { wgsl, bindings, pipeline_ctor }`
  - Populated from the *model-derived module list*, not from hardcoded filename prefixes.

Done when:
- Adding a new numerical “module” does **not** require editing:
  - `src/solver/model/kernel.rs` (no new `KernelKind` variants),
  - `build.rs` (no new per-prefix discovery code),
  - central kernel registries/matches in runtime.
- Module composition (order/phase/dispatch) is emitted by `(ModelSpec + method selection + runtime config)` and validated by contract tests.

Progress (partial):
- `ModelSpec.modules` is the primary mechanism for adding kernel bundles (schedule + WGSL generators) without central registries.
- Modules now carry a `ModuleManifest` that can contribute:
  - method selection (exactly one required)
  - flux module configuration (0 or 1 allowed)
  - named parameter keys accepted by the runtime plan
- typed invariants validated early (e.g. Rhie–Chow dp coupling requirements)
- Extracted Rhie–Chow aux bundle into a dedicated module factory (`src/solver/model/modules/rhie_chow.rs`), eliminating per-model inline `KernelBundleModule` literals for this pass.
- Formalized EOS as a module (`src/solver/model/modules/eos.rs`) and removed `ModelSpec.eos` entirely; EOS/low-mach named keys are now purely manifest-driven (no solver-core injection in `ModelSpec::named_param_keys`).
- Build-script stability for modules: `build.rs` includes `src/solver/model/modules/mod.rs`, so adding a new module file under `src/solver/model/modules/` does not require editing `build.rs` (validated on 2026-01-18).
- Rhie–Chow module-specific kernel IDs (`dp_init`, `dp_update_from_diag`, `rhie_chow/correct_velocity`) are now module-local (no longer central `KernelId` constants), so adding similar auxiliary modules does not require editing `src/solver/model/kernel.rs` (validated on 2026-01-18).
- Contract hardening: `contract_gap0_module_defined_kernel_id_is_module_driven` ensures a locally-defined `KernelId("contract/...")` (not a builtin kernel id) can still be scheduled and generated via module wiring (`src/solver/model/kernel.rs`), so future re-centralization breaks the test (added on 2026-01-18).
- Removed the last central “builtin model-generated kernel id” whitelist (`is_builtin_model_generated_kernel_id`); kernel specs/generators now flow only from the model’s module list.
- Consolidated coupled method identity into `MethodSpec::Coupled(CoupledCapabilities)`; solver strategy (implicit/coupled) is selected via `SolverConfig.stepping`.
- Validation gate: OpenFOAM reference tests passed (`bash scripts/run_openfoam_reference_tests.sh`) on 2026-01-18.

### 1) Flux Modules (reconstruction + method knobs)
- Flux kernels are IR-driven and now consume boundary conditions consistently with assembly.
  - `bc_kind`/`bc_value` are treated as **per-face × unknown-component** buffers (indexed by `face_idx`/`idx`, not by `boundary_type`).
  - This eliminates the mismatch where boundary conditions affected assembly but not face fluxes.
  - Validation gate: OpenFOAM reference tests passed (`bash scripts/run_openfoam_reference_tests.sh`) on 2026-01-16.

Status:
- **Reconstruction/limiter selection is now IR-driven for CentralUpwind** via `FluxReconstructionSpec::FirstOrder|Muscl{limiter}`.
- **Defaults remain `FirstOrder` for shipped models**, so OpenFOAM reference targets are unchanged.

Progress (partial):
- Added `FluxReconstructionSpec` + `LimiterSpec` as **IR-level** knobs and plumbed them from the `flux_module` manifest into `FluxModuleKernelSpec::CentralUpwind`.
  - Default remains `FirstOrder` (behavior-identical).
  - Added `FaceVec2Builtin::CellToFace { side }` to the IR and codegen lowering so schemes can express limited-linear/MUSCL-style reconstruction purely in IR terms.
- Implemented MUSCL face-state reconstruction in CentralUpwind lowering (`src/solver/model/flux_schemes.rs`):
  - Uses gradients + `CellToFace` (PDE-agnostic geometry builtin) to build left/right face states.
  - Honors `LimiterSpec::{None, MinMod, VanLeer}`.
- Enforced the required gradients stage + required `grad_*` fields when MUSCL is selected (early manifest validation).
- Fixed `CellToFace{Neighbor}` semantics on boundary faces so reconstruction is geometry-correct.
- Fixed boundary BC preservation for MUSCL: neighbor-side `grad_*` vectors are treated as zero on boundary faces so reconstruction cannot modify Dirichlet/Neumann ghost values.

Follow-ups:
- Contract/validation tests ensure reconstruction/limiter knobs are honored (MUSCL + unified_assembly scheme/limiter variants).
- VanLeer-limited scheme variants guard against opposite-signed slopes (prevents new extrema).

Long-term: derive flux-module specs from `EquationSystem` where possible; otherwise require explicit flux formulas as part of `ModelSpec` (still keeping codegen PDE-agnostic).

### 2) Make `UnifiedSolver` truly model-agnostic (API + host-side math)
Goal: keep the *public* solver API model-agnostic.

Historically, `GpuUnifiedSolver` (`src/solver/gpu/unified_solver.rs`) contained model-specific assumptions (field-name heuristics, hard-coded thermodynamics, and "inlet velocity" style helpers). Those belong in model helper layers, not in the core solver.

Target:
- Core solver exposes only generic operations: `set_param(id, value)`, `set_field(name, values)`, `step()`, `read_state()`, etc.
- Any “physics-aware” helpers live alongside models (e.g. `model::helpers::euler::pack_conservative_state(...)`), not in the solver core.
- Boundary driving (e.g. inlet velocity) is described declaratively by the model (or recipe), not by ad-hoc host-side special-casing.

Progress:
- `GpuUnifiedSolver` exposes generic state/field IO (`read_state_f32` / `write_state_f32`, `set_field_scalar`, `set_field_vec2`, etc.) and a generic boundary update API (`set_boundary_scalar/set_boundary_vec2/set_boundary_values`).
- Inlet-driving special-casing was removed; tests/UI drive inlet BCs via the generic boundary API + model helpers.
- Add/expand contract tests to prevent re-introducing solver-core physics assumptions.

### 3) Retire `PlanParam` as global plumbing
- Replace global param plumbing with typed, module-owned uniforms/config deltas routed through the recipe (or a model-declared parameter table).
- Done when: new configuration does not add new “known key” entries to a universal string match, and runtime controls are described by model/method metadata instead of global plumbing.

Progress (transitional):
- Added a string-keyed “named parameter” path (`GpuProgramPlan::set_named_param`) so new runtime knobs can be introduced without growing global enums.
- Removed the `PlanParam` enum and the `set_param` path; all runtime knobs now route through named parameters.
- Removed the universal string-key match; lowering registers named-parameter handlers in the plan spec.
- Moved named-parameter registration into the backend module (`lowering/models/generic_coupled.rs`) so universal lowering no longer owns a “known keys” list.
- Made named parameters **module-driven**: the runtime only registers handlers for keys declared by the model's modules.
  - EOS + low-Mach named keys are declared by the EOS module manifest (not implied by solver-core).

### 4) Handwritten WGSL (treat infrastructure the same way)
- Move handwritten solver infrastructure shaders under `src/solver/gpu/shaders` behind the same registry/metadata mechanism and treat them as registry-provided artifacts (even if template-generated).
- Done when: runtime consumes only registry-provided WGSL (no ad-hoc `include_str!` modules).

Progress (partial):
- `build.rs` derives handwritten kernel registry entries by scanning `src/solver/gpu/shaders/*.wgsl` for `@compute` entrypoints (removes the manual `id_only_entries` list and reduces per-kernel glue).

### 5) Contract Tests
- Add regression tests that fail if:
  - `kernel_registry` has special-case lookup paths
  - lowering selects a backend by “family”
  - adding a kernel requires editing a central `match` (bind groups / pipeline selection)
  - `build.rs` contains per-model hardcoding or kernel-kind-specific codegen glue

Progress:
- Source-level contract tests exist in `tests/contract_codegen_invariants_test.rs` and should be expanded as new invariants are introduced.

### 6) Validation Tests (OpenFOAM reference suite)
- After any solver or numerical-method change, run the OpenFOAM reference tests to catch whole-field regressions:
  - `bash scripts/run_openfoam_reference_tests.sh`
- If a change intentionally alters the reference target, regenerate the datasets and re-run:
  - `bash scripts/regenerate_openfoam_reference_data.sh`
 - Treat these tests as a **validation gate** during solver unification work (especially when reshaping kernel/module orchestration).

## Recommended Sequence (high leverage)
1) Make `UnifiedSolver` surface model-agnostic (remove solver-core physics assumptions; push helpers into model-side code).
2) Retire `PlanParam` and replace with model/method-declared runtime controls.
3) Finish closing remaining “numerics knobs” gaps by keeping selection manifest/IR-driven (reconstruction/limiters, optional method-specific passes) and adding contract tests that prevent silent ignored settings.
4) Add/expand contract tests and keep OpenFOAM reference tests green (or intentionally updated) during refactors.

## Next Tasks (smallest high-impact first)
Completed:
- Added a contract/validation test proving `FluxReconstructionSpec::Muscl{...}` affects generated WGSL (while shipped models remain `FirstOrder`).
- Made `unified_assembly`’s advection reconstruction configurable via the runtime `constants.scheme` knob, including limited SOU/QUICK variants.
- Added regression/contract coverage so limited SOU/QUICK variants can’t silently degrade.
- Gap 0 hardening: Rhie–Chow aux kernel IDs are module-local; plus a contract test proving a module-defined `KernelId("contract/...")` can be scheduled/generated without editing `src/solver/model/kernel.rs`.

Next:
1) Decide whether `unified_assembly` reconstruction should be retired in favor of the flux-module IR path (single source of truth for reconstruction/limiters), or kept as a lightweight fallback.
2) Continue Gap 0 cleanup: move `flux_module_module(...)` and `generic_coupled_module(...)` into `src/solver/model/modules/` so `src/solver/model/kernel.rs` is primarily shared infra + generator plumbing.

## Decisions (Locked In)
- Generated-per-model WGSL stays (no runtime compilation).
- Options selection uses typed enums.

## Notes / Constraints
- `build.rs` uses `include!()`; model modules are wired for build-time use via `src/solver/model/modules/mod.rs`.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
- EOS/low-Mach named params are declared via the EOS module manifest (not solver-core implied).
