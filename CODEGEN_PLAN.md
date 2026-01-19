# CFD2 Model-Driven Solver + Codegen Plan

This file tracks *remaining* work to reach a **fully model-agnostic solver** where the set of model-dependent kernels is derived automatically from the **math in `ModelSpec.system`** (as declared in `src/solver/model/definitions.rs`) plus selected numerical methods.

This is intentionally **not a changelog**: once a gap is closed, remove it from here (or summarize it in high-level “Completed Milestones”) so this file stays focused on what remains.

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
- **test → patch → test → commit** (small, frequent commits)
- **OpenFOAM reference tests are the primary quality gate** for solver behavior:
  - `bash scripts/run_openfoam_reference_tests.sh`
    - (equivalent to `cargo test -p cfd2 --tests openfoam_ -- --nocapture`, but fails loudly if 0 tests are discovered)
- Prefer a short, targeted test before/after each patch (e.g. `cargo test contract_` or the specific failing test), but do not skip OpenFOAM when the change can affect numerics/orchestration.

## Current Baseline (already implemented, for context only)
- Models are expressed as `EquationSystem` via the fvm/fvc AST in `src/solver/model/definitions.rs`.
- Build-time codegen exists (`crates/cfd2_codegen`) and emits WGSL into `src/solver/gpu/shaders/generated` from `build.rs` (no runtime compilation).
- Kernel schedule/phase/dispatch is model-owned and recipe-driven (`src/solver/model/kernel.rs`, `src/solver/gpu/recipe.rs`).
- Kernel registry lookup is uniform for all kernels via `(model_id, KernelId)` (`src/solver/gpu/lowering/kernel_registry.rs`).
- Host pipelines + bind groups for recipe kernels are built from binding metadata + a uniform `ResourceRegistry` (`src/solver/gpu/modules/generated_kernels.rs`).

## Completed Milestones (high-level)
- Model-owned modules + manifests drive kernel scheduling, named params, and optional passes (contract tests protect against re-centralization).
- Runtime wiring is metadata-driven (no kernel-id switches for bind groups/pipeline layouts; registry provides bindings + pipelines).
- A single model-driven lowering path is used across stepping modes (explicit/implicit/coupled share the universal backend wiring).

## Remaining Gaps (current blockers)

### 1) Low-Mach preconditioning is still a dead knob
Today the runtime exposes `low_mach.*` named parameters and allocates/updates a low-Mach params buffer, but generated kernels do not consume those values, so toggling low-Mach has no effect.

Done when:
- Low-Mach settings influence **generated** compressible kernels (starting with `EulerCentralUpwind` wave-speed bounds) in a way that is derived from model/module configuration.
- Default behavior remains unchanged for the OpenFOAM reference suite.
- Contract coverage prevents regressing to a “declared but unused” runtime knob.

### 2) Ongoing hardening (evergreen)
- Add/expand contract tests as new invariants are introduced (keep “no special casing” gaps closed).
- Prefer binding/manifest-driven derivation for optional resources/stages (no solver-side aliases/special cases).

## Next
1) Wire Low-Mach params into compressible flux wave speeds (default Off).
2) Remove/retire unused low-Mach buffer plumbing once kernels consume the values.
3) Expand low-Mach regression coverage (at least one non-ignored smoke test runnable by default, if feasible).

## Notes / Constraints
- `build.rs` uses `include!()`; model modules are wired for build-time use via `src/solver/model/modules/mod.rs`.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
- EOS/low-Mach named params are declared via the EOS module manifest (not solver-core implied).
