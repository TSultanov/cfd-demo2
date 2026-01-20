# CFD2 Model-Driven Solver + Simplification Plan

This file tracks *remaining* work to keep `cfd2` a **fully model-driven FVM-based 2D PDE solver** while **aggressively simplifying** the repository (delete unused code, remove accidental complexity, and isolate optional surfaces like UI/meshgen).

This is intentionally **not a changelog**: once a gap is closed, remove it from here so this file stays focused on what remains.

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
- **Commit generated WGSL remains policy:** do not move runtime-critical shader sources to runtime generation.

## Core Keep List (reference)
- **Core (must keep):**
  - Model-driven GPU solver pipeline (`src/solver/model/*`, `src/solver/gpu/*`, `crates/cfd2_ir`, `crates/cfd2_codegen`).
  - Minimal mesh + structured mesh constructors used by OpenFOAM reference tests (`src/solver/mesh/structs.rs`, `src/solver/mesh/structured.rs`).
  - Validation signal: OpenFOAM reference tests + contract tests under `tests/` (and the `scripts/run_openfoam_reference_tests.sh` gate).
- **Optional surfaces (may be feature-gated / moved / deleted):**
  - UI (`src/ui/*`, binary entrypoint).
  - Mesh generation experiments (`meshgen` feature: cut-cell/delaunay/voronoi/quadtree/geometry + smoothing).
  - Profiling (`profiling` feature: profiling plumbing + long-running profiling workloads).
  - Reproduction harnesses (`examples/reproduce_*` and any future repro utilities).

## Development Cadence (required)
- **test → patch → test → commit** (small, frequent commits)
- **OpenFOAM reference tests are the primary quality gate** for solver behavior:
  - `bash scripts/run_openfoam_reference_tests.sh`
    - (equivalent to `cargo test -p cfd2 --tests openfoam_ -- --nocapture`, but fails loudly if 0 tests are discovered)
  - Core-only mode: `CFD2_CORE_ONLY=1 bash scripts/run_openfoam_reference_tests.sh` (runs with `--no-default-features`)
- Prefer a short, targeted test before/after each patch (e.g. `cargo test contract_` or the specific failing test), but do not skip OpenFOAM when the change can affect numerics/orchestration.

## Current Audit Notes (concrete simplification targets)
(Empty; add new concrete targets as they’re discovered.)

## Remaining Gaps (simplification + pruning plan)

### 1) UI responsiveness (optional surface)
- Eliminate UI-thread blocking while the solver is running (no blocking `Mutex` waits in the UI thread).
- Decouple visualization/readback cadence from console logging cadence.
- Make solver re-init / solver-kind switching reliable (stop current worker, then init) without appearing to hang.

### 2) Ongoing hardening (evergreen)
- Add/expand contract tests as new invariants are introduced (keep “no special casing” gaps closed).
- Prefer binding/manifest-driven derivation for optional resources/stages (no solver-side aliases/special cases).

## Verification Checklist (use after every pruning PR)
- `cargo test -p cfd2 --lib`
- `bash scripts/run_openfoam_reference_tests.sh`
- `bash scripts/check_generated_wgsl.sh` (or equivalent local verification)

## Notes / Constraints
- `build.rs` uses `include!()`; model modules are wired for build-time use via `src/solver/model/modules/mod.rs`.
- Build scripts are std-only; avoid pulling runtime-only crates into code referenced by `build.rs`.
- Generated WGSL + `src/solver/gpu/bindings.rs` are committed per `GENERATED_WGSL_POLICY.md`.
