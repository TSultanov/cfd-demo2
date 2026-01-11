# CFD2 Codegen + Unified Solver Plan

This file tracks *remaining* codegen + solver orchestration work needed to reach a **single unified solver** where runtime behavior is derived from **(ModelSpec + selected numerical methods)** only.

## Goal
One model-driven GPU solver pipeline with:
- a single lowering + codegen path (no compressible/incompressible “families” in lowering)
- a single runtime orchestration path (no per-family step loops or templates)
- solver features as pluggable modules that own their GPU resources
- kernels + auxiliary passes derived automatically from model structure + method selection (no handwritten WGSL)
- **system size = model definition**: `unknowns_per_cell` is always `EquationSystem`-derived
- **derived primitives are model-defined** via an expression language (reuse the existing DSL/AST, but keep codegen physics-agnostic)
- **fluxes are pluggable modules** configured in `src/solver/model/definitions.rs` (codegen emits generic loops; physics lives in model-spec configuration)
- **preconditioner choice is model-owned** and validated (no env overrides, no silent fallback)

## Non-negotiable invariants (for “any model + any method”)
- **No solver-family switches in orchestration:** runtime must not branch on “compressible vs incompressible vs generic”.
- **No handwritten kernel scheduling:** ordering/phase membership is part of the recipe emitted from model+methods.
- **No handwritten kernel lookup tables:** runtime only asks for `KernelId` and receives generated WGSL + binding metadata.
- **No handwritten bind groups:** host binding is assembled from generated binding metadata + a uniform resource registry.
- **Codegen agnostic to the PDE model** - `src/solver/codegen` must ot have dependencies on `src/solver/model`. Adding new PDE should only require editing `src/solver/model` without touching the codegen.
- **No specialized EI kernels**: EI becomes just a particular combination of generic modules (flux + primitive recovery + assembly/apply/update) derived from the model.

## Status Snapshot (as of 2026-01-11)

### Landed / Working
- Model-driven lowering is recipe-first (`src/solver/gpu/lowering/model_driven.rs`).
- Template-driven orchestration is effectively gone: program structure is emitted by `SolverRecipe::build_program_spec()` and op registration is unified (`src/solver/gpu/lowering/unified_registry.rs`).
- `KernelId` exists and runtime can look up generated kernel source via `kernel_registry::kernel_source_by_id` (with build-time generated tables).
- Generated model WGSL exists under `src/solver/gpu/shaders/generated`, and generated binding metadata is available via `src/solver/gpu/wgsl_meta.rs`.
- **Transitional backend:** compressible uses `MethodSpec::ConservativeCompressible` and selects `KernelId::CONSERVATIVE_*` (mapped to legacy `ei_*` WGSL ids).
- Build-time kernel registry now includes the legacy `ei_*` shaders so the transitional path can run without special host tables.
- EI uses `FluxLayout` (named component offsets/stride) and unified BC tables (`bc_kind`/`bc_value`) with consecutive bind groups.
- EI codegen no longer depends on `CompressibleFields` for EI kernel emission (still Euler-name-specific internally).
- EI kernel implementations live under `src/solver/codegen/ei/*`; legacy `compressible_*` EI modules have been deleted.
- Regression validation: `gpu_compressible_solver_preserves_uniform_state` passes.

### Newly Landed (as of 2026-01-11)
- Removed unused GPU env helper (`src/solver/gpu/env_utils.rs`); solver behavior is not env-driven.
- Generic coupled GPU path now has an internal CSR runtime that can allocate the linear system for `num_dofs = num_cells * unknowns_per_cell` (still using scalar CSR kernels over the expanded system).
- `ModelKernelsModule` no longer assumes a coupled-only anchor kernel; it can build pipelines/bind groups for conservative (legacy EI) kernels by iterating the recipe and using registry binding metadata.
- **Model-owned Schur preconditioner (bridge):** `ModelSpec.linear_solver` can request `ModelPreconditionerSpec::Schur { omega, layout }`, which wires the generic-coupled path to FGMRES + a SIMPLE-like Schur preconditioner. Compatibility is validated at model load.

### Still Violating the Goal (remaining family-ness)
- Kernel planning still performs structural “family selection” (method enum variants imply solver families; eventually these should be emitted from term/module expansion).
- Recipe still contains central mapping matches for phase/dispatch (`src/solver/gpu/recipe.rs`).
- Runtime still has solver-family branches (coupled vs conservative vs generic) during module construction; goal is a single fully data-driven resource registry.
- Lowering still has an explicit special-case path for generic-coupled plan resources.
- Bind group assembly is only partially unified: several kernels still rely on plan/module-specific wiring (e.g. solver buffers and per-kernel binding arrays).

### Still Violating the Goal (new invariant gaps)
- Derived primitives are still hard-coded in a few places (e.g. Euler primitive recovery), but EOS parameters are now model-driven (no more `gamma=1.4` in kernels).
- Derived primitive expressions must be dependency-safe. Today we require that each expression can be evaluated from **conserved/state fields only** (no references to other derived primitives) unless we add an explicit dependency DAG + topo sort.
- Flux computation is still method-specific (EI KT flux and incompressible Rhie–Chow exist as dedicated kernels rather than pluggable “flux modules” configured in the model).
- Incompressible still runs through a coupled-family path (pressure-correction / Rhie–Chow family) instead of the generic discretization pipeline.

### Blockers / Known Limitations (generic incompressible bridge)
- The model-owned Schur preconditioner is intentionally narrow: it assumes a 2D saddle-point split (one Vector2 “velocity-like” block + one Scalar “pressure-like” unknown). The per-cell indices are model-declared via `SchurBlockLayout` (no solver-side `[U_x,U_y,p]` ordering assumption).
- Generic-coupled rejects attempts to set `PlanParam::Preconditioner` when a model-owned preconditioner is active (e.g. Schur), and `GpuUnifiedSolver::new()` uses the same validation helper as lowering so config/runtime cannot override model-owned preconditioning.
- The UI disables the preconditioner selector when the active model owns preconditioning (e.g. Schur), keeping UI state consistent with enforcement.

## Remaining Work (Prioritized)

1. **Stop inferring “families” from structure; emit kernels/stepping from (terms + selected numerics).**
   - Today: `derive_kernel_ids` and `derive_stepping_mode` encode compressible/coupled/generic heuristics.
   - Change: scheme/term expansion emits a recipe (or kernel list + stepping) directly; method selection decides Rhie–Chow vs KT vs generic, etc.
   - Done when: adding a new method/term updates only expansion metadata + codegen; no structural “family selection” exists.

2. **Make `SolverRecipe` authoritative for ordering/phase/dispatch without central ID matches.**
   - Today: `SolverRecipe::from_model` assigns `KernelPhase` and `DispatchKind` with a central `match` on `KernelId`.
   - Change: recipe construction assigns phase/dispatch as kernels are emitted (by the method/term expanders), not by a global mapping.
   - Done when: a new kernel never requires editing a central mapping in `recipe.rs`.

3. **Unify runtime kernel module construction (remove per-family constructors + hard-coded model ids).**
   - Today: `ModelKernelsModule::new_compressible/new_incompressible` hard-code kernel sets, binding arrays, and `kernel_source_by_id("compressible"|"incompressible_momentum", ...)`.
   - Change: build pipelines + bind groups by iterating `recipe.kernels`, using generated binding metadata and a uniform resource resolver.
   - Done when: runtime never contains lists like “compressible kernel ids” or strings like "compressible"/"incompressible_momentum".

4. **Make kernel registry fully per-model and remove special-case lookups.**
   - Today: `kernel_registry` is partly global-by-`KernelId`, with a special-case `generic_coupled_pair(model_id)` path.
   - Change: codegen emits a single table keyed by `(model_id, KernelId)` for *all* kernels (including per-model generic coupled), so runtime lookup is uniform.
   - Done when: there is one lookup API (no generic-coupled special case) and no handwritten kernel tables.

5. **Finish reflection-driven bind groups via a uniform resource registry.**
   - Today: reflection helpers exist, but some bindings are still wired via per-kernel metadata constants and custom host matches.
   - Change: bind group layouts + entries are derived from generated binding metadata; a `ResourceRegistry` resolves binding names to buffers/uniforms.
   - Done when: adding a binding to a kernel never requires host-side bind-group edits.

6. **Reduce codegen duplication: migrate per-family WGSL generators to term/method-driven emitters.**
   - Today: codegen still has dedicated `compressible_*` generators and `emit.rs` matches on `KernelKind`.
   - Change: emit WGSL by iterating recipe/kernel specs (prefer `KernelId`), and build kernel WGSL from the same IR expansion regardless of “family”.
   - Done when: codegen doesn’t need separate “compressible vs incompressible” entrypoints to emit kernels.

    ### Concrete sub-project: remove `compressible_*` codegen by turning EI into a `MethodModule`

    **Decisions (as of 2026-01-10):**
    - Implement EI as a `MethodModule`.
    - Unify boundary-condition handling now (do not keep EI-only BC wiring).
    - Introduce a new named-component flux table (do not reuse `FluxKind`).

    **Current EI ("compressible") generators to retire:**
    - `src/solver/codegen/compressible_apply.rs`
    - `src/solver/codegen/compressible_assembly.rs`
    - `src/solver/codegen/compressible_explicit_update.rs`
    - `src/solver/codegen/compressible_flux_kt.rs`
    - `src/solver/codegen/compressible_gradients.rs`
    - `src/solver/codegen/compressible_update.rs`

   **Remaining work (as of 2026-01-11):**
    - **Delete the legacy module names:** done (legacy `src/solver/codegen/compressible_*` EI modules removed).
   - ✅ **DONE (2026-01-11):** EI assembly no longer hard-codes the Euler 4×4 layout; it accepts `unknowns_per_cell >= 4` and derives ordering from `EquationSystem`.
   - ✅ **DONE (2026-01-11):** EOS is model-driven via `ModelSpec.eos`; EI kernels consume `EosSpec` instead of hard-coding `gamma = 1.4`.
   - **Finalize recipe/module ownership:** the EI method module should declare its resources and emit its kernel list + phase/dispatch without central matches.

    **Notes (updated 2026-01-12):**
    - EOS is now plumbed through `ModelSpec.eos` into EI WGSL generators (no hard-coded `gamma`).
    - EI assembly accepts `unknowns_per_cell >= 4` and derives component ordering from `EquationSystem`.

    **What replaces them:**
    - A method module, e.g. `ExplicitImplicitConservativeMethodModule`, that:
       - declares required resources (state buffers, flux buffers, gradients, linear solver vectors/matrices)
       - emits a per-model kernel list + phases/dispatch into the `SolverRecipe`
       - uses the same BC expansion path as other methods (no EI-only BC logic)
    - A new `FluxLayout`/named-component table used uniformly by:
       - face flux kernels (write fluxes by component name)
       - cell update kernels (read fluxes by component name)
       - implicit assembly (select the same flux components when linearizing)

    **Minimal metadata required (deriveable from `ModelSpec + MethodSpec`):**
    - `MethodSpec::ExplicitImplicitConservative { flux_method, reconstruction, eos, low_mach_precond, time_integration }`
    - `ModelSpec` must provide:
       - conserved unknown list (e.g. `[rho, rho_u_x, rho_u_y, rho_E]` for Euler)
       - primitive recovery requirements (which primitives must exist for method/terms)
       - EOS spec (required for kernels; no hard-coded `gamma`)
    - `FluxLayout` provides named components and per-component packing (offset/stride)

    **BC unification requirement (do now):**
    - EI flux/update/assembly must consult the same boundary-condition expansion/metadata as unified kernels.
    - "Special" EI BC handling is not allowed as an intermediate state.

   **Next steps (practical):**
   1) Replace Euler-specific primitive recovery and assembly with EOS/unknown-driven variants (or make the constraints explicit in `MethodSpec`).
   2) Make the EI method module the authoritative source of its kernel list + phase/dispatch in the recipe.
   3) Delete the legacy `compressible_*` EI modules once unused.

    **Done when:**
    - Kernel planning no longer infers EI from "compressible" structure.
    - Runtime orchestration is unchanged when adding/removing EI kernels (the module emits recipe entries).
    - No `compressible_*` modules remain in `src/solver/codegen`.

7. **Eliminate remaining handwritten WGSL required for correctness (including solver infrastructure).**
   - Today: several solver/preconditioner/GMRES shaders exist as handwritten WGSL under `src/solver/gpu/shaders`.
   - Change: treat these as generated artifacts (same registry + binding metadata) or move them into the same codegen pipeline.
   - Done when: runtime consumes generated WGSL only.

8. **Retire `PlanParam` as global plumbing; move to typed module-owned config deltas.**
   - Today: `PlanParam` is still used to push ad-hoc config through plans.
   - Change: route configuration updates to the owning module using strongly typed deltas.
   - Done when: new configuration does not add global enum cases or stringly plumbing.

9. **Delete legacy glue and enforce the contract with tests.**
   - Add regression tests that fail if:
     - kernel planning uses structural “family selection” heuristics
     - runtime contains solver-family switches / hard-coded model ids
     - kernel phase/dispatch requires editing a central match

## Near-term recommended sequence (practical)
1) Introduce the IR boundary: model lowering produces a physics-erased IR and codegen depends only on that IR.
2) Make generic coupled fully size-generic (`unknowns_per_cell` everywhere), including solver allocations and CSR expansion.
3) Convert incompressible to generic: represent pressure coupling, flux module, and derived primitives via model configuration (no special kernels).
4) Convert EI/compressible to generic: KT/Riemann flux module + EOS/primitive recovery module + generic assembly/apply/update.
5) Delete EI-specific kernel kinds, WGSL generators, and bindings once no model emits EI kernels.

## Decisions (Locked In)
- Generated-per-model WGSL stays (no runtime compilation).
- Flux/EOS closures remain explicit plan/modules (not inline expressions).
- Options selection uses typed enums.

## Blockers / Unclear Requirements
- **Expression language for derived primitives**: a neutral `src/solver/shared` module exists, but there is still duplication between `src/solver/shared/wgsl_ast.rs` and `src/solver/codegen/wgsl_ast.rs`. We should converge on a single AST to avoid type-split bugs.
- **Generic incompressible solve**: the incompressible coupled system is typically indefinite; the current generic coupled plan still uses scalar-CG infrastructure. Routing incompressible through generic path requires a Krylov method (FGMRES) and a system-agnostic preconditioner that works for the expanded CSR.
- **Generic flux modules**: compressible currently uses `DivFlux(phi_rho, rho)` etc. The generic pipeline wants a packed face-flux buffer with stable offsets; this implies replacing per-equation named flux fields with a model-provided flux module that writes a packed flux buffer consistent with a derived `FluxLayout`.

## Notes / Constraints
- `build.rs` uses `include!()`; build-time-only modules must be wired there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
