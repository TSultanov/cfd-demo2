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
- **Codegen agnostic to the PDE model** - `crates/cfd2_codegen/src/solver/codegen` must not have dependencies on `src/solver/model`. Adding a new PDE should only require editing `src/solver/model` without touching the codegen.
- **No specialized EI kernels**: EI becomes just a particular combination of generic modules (flux + primitive recovery + assembly/apply/update) derived from the model.

## Current Code State (as of 2026-01-12)

### Landed / Working
- Recipe-first lowering (`src/solver/gpu/lowering/model_driven.rs`) and program structure emitted by `SolverRecipe::build_program_spec()`.
- Codegen lives in `crates/cfd2_codegen/src/solver/codegen` and is invoked via `src/solver/compiler/emit.rs` to emit WGSL into `src/solver/gpu/shaders/generated`.
- Runtime kernel lookup is `kernel_registry::kernel_source_by_id(model_id, KernelId)` using build-time generated tables (`src/solver/gpu/lowering/kernel_registry.rs`).
- Model-side configuration exists for method selection, flux modules, and primitive recovery (`src/solver/model/*`).
- Generic-coupled pipeline can run:
  - `compressible_model()` routed through `MethodSpec::GenericCoupledImplicit` with KT flux + `PrimitiveDerivations`.
  - `incompressible_momentum_generic_model()` routed through `MethodSpec::GenericCoupled` with a model-owned Schur preconditioner.
- Model-owned preconditioning is enforced (Schur disables config/runtime overrides).

### Still Violating the Goal (remaining family-ness)
- Lowering still selects backends with structural branches (`src/solver/gpu/lowering/model_driven.rs:lower_parts_for_model`), including a dedicated generic-coupled path.
- Kernel planning still encodes method/family branching in `src/solver/model/kernel.rs` (`derive_kernel_ids_for_model`).
- Recipe phase/dispatch assignment still relies on central `KernelId` matches in `src/solver/gpu/recipe.rs`.
- Kernel modules and bind group wiring are still per-family (`src/solver/gpu/modules/model_kernels.rs`, `src/solver/gpu/modules/generic_coupled_kernels.rs`).

### Still Violating the Goal (invariant gaps)
- Kernel registry still has a generic-coupled special-case (`generated::generic_coupled_pair(model_id)` in `src/solver/gpu/lowering/kernel_registry.rs`).
- Bind group assembly is only partially reflection-driven; there is no uniform `ResourceRegistry`, and kernel → bind-group selection is still handwritten.
- Derived primitives are expression-based but lack dependency validation/toposort (no derived→derived DAG).
- Several solver/preconditioner/GMRES kernels remain handwritten WGSL under `src/solver/gpu/shaders`.
- `PlanParam` is still used as global plumbing.

## Remaining Work (Prioritized)

1. **Make model+method expansion emit the recipe directly (no “family inference” shims).**
   - Today: kernel selection/ordering is still centralized in `src/solver/model/kernel.rs` and stepping is derived in `src/solver/gpu/recipe.rs`.
   - Change: method modules (flux, primitives, assembly/solve/update) emit `KernelSpec { id, phase, dispatch }` plus loop/stepping structure.
   - Done when: adding a new method/term does not require editing “family selection” code.

2. **Remove central `KernelId` → phase/dispatch matches from `SolverRecipe::from_model`.**
   - Today: `src/solver/gpu/recipe.rs` assigns `KernelPhase` and `DispatchKind` with `match id`.
   - Change: phase/dispatch are owned by the module that emits the kernel.
   - Done when: a new kernel never requires editing `recipe.rs`.

3. **Introduce a uniform `ResourceRegistry` and build bind groups purely from generated metadata.**
   - Today: each plan/module wires buffers manually, even when `wgsl_reflect` is used.
   - Change: a registry maps binding names → concrete resources (mesh buffers, state/aux buffers, solver buffers, BC tables, uniforms).
   - Done when: adding a kernel binding does not require host-side bind-group edits beyond registering the resource.

4. **Unify kernel module construction into one generated-kernel module.**
   - Today: separate kernel modules with handwritten bind-group choices and ping-pong handling.
   - Change: a single module iterates `recipe.kernels`, builds pipelines/bind groups via `(model_id, KernelId)` + binding metadata + `ResourceRegistry`.
   - Done when: runtime has no per-family kernel modules and no `KernelId`-specific bind-group matches.

5. **Make kernel registry fully uniform across per-model and “infrastructure” kernels.**
   - Today: `GENERIC_COUPLED_*` are looked up via `generic_coupled_pair(model_id)` and others via `kernel_entry_by_id`.
   - Change: codegen emits one table keyed by `(model_id, KernelId)` for all kernels; “global” kernels use a sentinel model id (e.g. `__global__`).
   - Done when: `kernel_registry` has exactly one lookup path.

6. **Retire remaining family-specific WGSL generators by moving to IR-driven kernels + modules.**
   - Today: several kernels still have dedicated generators in `crates/cfd2_codegen/src/solver/codegen`, and `src/solver/compiler/emit.rs` matches on `KernelKind`.
   - Change: incompressible and compressible become configurations of the same IR/method-module pipeline; remove field-name-specific helpers like `CodegenIncompressibleMomentumFields`.
   - Done when: models differ only by `ModelSpec` configuration (no special-case kernel generators).

7. **Eliminate remaining handwritten WGSL required for correctness (including solver infrastructure).**
   - Today: many solver/preconditioner/GMRES shaders are handwritten under `src/solver/gpu/shaders`.
   - Change: treat these as generated artifacts (same registry + binding metadata) or move into codegen.
   - Done when: runtime consumes generated WGSL only.

8. **Retire `PlanParam` as global plumbing; move to typed module-owned config deltas.**
   - Today: `PlanParam` is still used to push ad-hoc config through plans.
   - Change: route configuration updates to the owning module using strongly typed deltas/uniforms.
   - Done when: new configuration does not add global enum cases.

9. **Delete legacy glue and enforce the contract with tests.**
   - Add regression tests that fail if:
     - `kernel_registry` has special-case lookup paths
     - lowering selects backends by “family”
     - adding a new kernel requires editing a central `match` (phase/dispatch/bind groups)

## Near-term recommended sequence (practical)
1) Unify kernel registry to a single `(model_id, KernelId)` lookup.
2) Add `ResourceRegistry` and migrate bind-group construction to metadata-driven assembly.
3) Merge kernel modules into a single generated-kernel module and delete per-family modules.
4) Remove backend selection in `lower_parts_for_model` by making all plans use the same module/resource registry.
5) Retire handwritten infrastructure WGSL and lock invariants with tests.

## Decisions (Locked In)
- Generated-per-model WGSL stays (no runtime compilation).
- Options selection uses typed enums.

## Blockers / Unclear Requirements
- **Derived primitive dependencies:** decide whether to support derived→derived expressions (add a DAG + topo sort) or enforce “state-only” at validation time.
- **AST duplication:** converge `src/solver/shared/wgsl_ast.rs` and `crates/cfd2_codegen/src/solver/codegen/wgsl_ast.rs` to a single source to avoid type-split bugs.
- **Flux modules:** KT and Rhie–Chow still rely on dedicated kernels/bindings; define a stable “flux module contract” that writes packed face fluxes consistent with `FluxLayout`.

## Notes / Constraints
- `build.rs` uses `include!()`; build-time-only modules must be wired there.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
