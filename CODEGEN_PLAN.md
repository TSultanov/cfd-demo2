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

## Development Cadence (required)
- **test → patch → test → commit** (small, frequent commits)
- **OpenFOAM reference tests are the primary quality gate** for solver behavior:
  - `bash scripts/run_openfoam_reference_tests.sh`
    - (equivalent to `cargo test -p cfd2 --tests openfoam_ -- --nocapture`, but fails loudly if 0 tests are discovered)
- Prefer a short, targeted test before/after each patch (e.g. `cargo test contract_` or the specific failing test), but do not skip OpenFOAM when the change can affect numerics/orchestration.

## Current Audit Notes (concrete simplification targets)
- `src/solver/gpu/async_buffer.rs` appears unused (no references in `src/` or `tests/`).
- `src/solver/model/fluid.rs` is UI-only (only referenced from `src/ui/app.rs`).
- `src/solver/mesh/tests.rs` is compiled into non-test builds via `pub mod tests;` in `src/solver/mesh/mod.rs` (should be `#[cfg(test)]` or moved).
- Suspected unused deps in `Cargo.toml`: `geo`, `num-traits` (verify with `cargo udeps`/equivalent or grep-based audit, then remove).
- Many `tests/reproduce_*` and `tests/*profile*` are exploratory/profiling artifacts; keep the knowledge but move them out of the default test suite.

## Remaining Gaps (simplification + pruning plan)

### 1) Define the “core keep list” (what must remain)
- Document the minimal supported product as **core**: model-driven solver + OpenFOAM reference suite + contract tests.
- Explicitly list which parts are “optional surfaces”: UI, mesh generation experiments, profiling tooling, reproduction harnesses.

### 2) Isolate optional surfaces (UI / meshgen / profiling)
Prefer **Cargo feature-gating** first; split into new workspace crates only if feature-gating becomes awkward.
- Add an `ui` feature that gates `src/ui/*` + the GUI binary entrypoint.
- Add a `meshgen` feature that gates cut-cell/delaunay/voronoi/quadtree generators + geometry helpers used only for mesh generation.
- Add a `profiling` feature for GPU profiling plumbing and long-running profiling tests.
- Update scripts so OpenFOAM reference tests can run in “core-only” mode (e.g. `--no-default-features` if `ui` is default).

### 3) Move UI-only domain helpers out of solver-core
- Move `Fluid` presets out of `src/solver/model/` (UI concern) into `src/ui/` (or a small `ui_support` module behind the `ui` feature).
- Keep model helper traits in `src/solver/model/helpers/*`, but ensure they remain opt-in and do not leak into solver-core.

### 4) Prune / simplify the mesh module
- Keep a minimal mesh representation + structured mesh constructor(s) needed by OpenFOAM reference tests.
- Move/delete advanced mesh generation, smoothing, and geometry SDF machinery unless it is part of the “core keep list”.
- Remove parallel/vectorized dependencies from core mesh if they’re only used for experimental meshgen.

### 5) Delete unused code + unused dependencies (mechanical)
- Delete `src/solver/gpu/async_buffer.rs` (after confirming no downstream users).
- Gate `src/solver/mesh/tests.rs` behind `#[cfg(test)]` or convert to integration tests under `tests/`.
- Remove unused deps (`geo`, `num-traits`, and any others discovered) and re-run OpenFOAM refs as the regression gate.

### 6) Test + bench consolidation (keep signal, drop noise)
- Keep as default: OpenFOAM reference tests + contract tests + a small set of GPU smoke tests.
- Move profiling workloads out of `tests/` into `benches/` or `examples/` (or mark `#[ignore]` and run only via scripts).
- Move reproduction tests (`tests/reproduce_*`) into `examples/` or a `docs/repro/` area; keep them runnable but not part of the default suite.

### 7) Structural cleanup (rename and collapse transitional modules)
- Align naming to reflect the “one universal backend” reality (reduce `generic_coupled` vs `universal` vs `plans` confusion).
- Consider collapsing pure re-export modules (`options.rs`, `profiling.rs`, etc.) if they don’t provide real API value after the crate boundaries are cleaned up.

### 8) Ongoing hardening (evergreen)
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
