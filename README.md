# 2D CFD Solver

Math-driven 2D CFD solver in Rust and WGPU: physics is declared as typed,
unit-checked equations (plus flux, EOS, and boundary-condition declarations),
and all GPU numerics — assembly, flux modules, auxiliary kernels, boundary
refresh — are generated from those declarations at build time.

## Authoring a model

A model is one file under `src/solver/model/definitions/` plus a registry
entry in `all_models()`: equations built with `typed_fvm`/`typed_fvc`
(per-term `.scheme(..)` and `.bounded()` available), algebraic relations via
`typed_alg`, a derived or declared flux (`derive_rhie_chow`,
`FluxSchemeSpec::CentralUpwind(decl)`, `FluxExprSpec`), boundary conditions
(constants or `BoundaryExpr` expressions over interior/prescribed/param
values), and solver policy structs. No hand-written WGSL, no per-model
kernels. See `docs/math-surface.md` for the normative math/engine boundary
and `definitions/buoyant_incompressible.rs` for a complete worked example
(coupled momentum + pressure + temperature with Boussinesq buoyancy).

Correctness is gated by MMS convergence-order suites (`tests/mms_*`, the
primary oracle) and the OpenFOAM reference suite (secondary, no-growth
policy) — see `AGENTS.md`.

## Docs

- Math/engine boundary (model authoring): `docs/math-surface.md`
- Codegen + solver unification plan: `CODEGEN_PLAN.md`
- Generated WGSL policy: `GENERATED_WGSL_POLICY.md`
- Port refactor plan: `PORT_REFACTOR_PLAN.md`
- Port refactor migration guide: `MIGRATION_PORT_REFACTOR.md` (type-level dimensions + port-based field access)
- Fusion authoring guide: `docs/fusion-authoring.md`
- Fusion troubleshooting guide: `docs/fusion-troubleshooting.md`

## How to Run

1. Ensure you have Rust installed.
2. Run the application:
   ```bash
   cargo run --release --features ui
   ```

## Testing

### OpenFOAM Reference Tests

The OpenFOAM reference comparison tests are marked with `#[ignore]` and require extended timeout due to GPU compute requirements:

```bash
# Incompressible tests (faster)
cargo test --test openfoam_incompressible_lid_driven_cavity_reference_test -- --ignored --timeout 120
cargo test --test openfoam_incompressible_channel_reference_test -- --ignored --timeout 120
cargo test --test openfoam_incompressible_backwards_step_reference_test -- --ignored --timeout 120

# Compressible tests (slower, require extended timeout)
cargo test --test openfoam_compressible_lid_driven_cavity_reference_test -- --ignored --timeout 300
cargo test --test openfoam_compressible_backwards_step_reference_test -- --ignored --timeout 120
cargo test --test openfoam_compressible_acoustic_reference_test -- --ignored --timeout 120
cargo test --test openfoam_compressible_supersonic_wedge_reference_test -- --ignored --timeout 120
```

**Note**: The compressible lid-driven cavity test shows ~60% velocity error at t=0.003s. This is a known difference due to solver formulation at early transient times (implicit BDF2 + low-Mach preconditioning vs OpenFOAM's explicit central-upwind). See `COMPRESSIBLE_SOLVER_INVESTIGATION.md` for details.

### Solver Comparison Tests

Compare compressible and incompressible solvers (should give similar results at low Mach):

```bash
cargo test --test lid_driven_cavity_compressible_vs_incompressible -- --nocapture --timeout 180
```
