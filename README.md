# 2D CFD Solver

2D CFD solver for incompressible laminar flow implemented in Rust and WGPU.

## Docs

- Codegen + solver unification plan: `CODEGEN_PLAN.md`
- Generated WGSL policy: `GENERATED_WGSL_POLICY.md`
- Port refactor plan: `PORT_REFACTOR_PLAN.md`
- Port refactor migration guide: `MIGRATION_PORT_REFACTOR.md` (type-level dimensions + port-based field access)

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
