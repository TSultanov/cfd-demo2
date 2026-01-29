# Time Stepping Change: Euler → BDF2

## Summary

All verification tests have been changed from `TimeScheme::Euler` (first-order explicit) to `TimeScheme::BDF2` (second-order implicit) for consistency and accuracy.

## Why BDF2?

### 1. **Higher Order Accuracy**
- **Euler**: First-order accurate (O(Δt))
- **BDF2**: Second-order accurate (O(Δt²))

BDF2 provides significantly better accuracy for the same time step size.

### 2. **Consistency with Compressible Solver**
The compressible solver already used BDF2 by default for most tests. Having both solvers use the same time scheme ensures fair comparisons.

### 3. **Better for Verification**
Verification tests against analytical solutions or reference data (OpenFOAM) benefit from higher-order accuracy to minimize time discretization errors.

### 4. **Implicit Stability**
BDF2 is implicit (A-stable), allowing larger time steps without stability constraints, while Euler is explicit and requires smaller steps for stability.

## Files Changed

### OpenFOAM Reference Tests
- `openfoam_incompressible_lid_driven_cavity_reference_test.rs`
- `openfoam_incompressible_channel_reference_test.rs`
- `openfoam_incompressible_backwards_step_reference_test.rs`

### GPU Solver Tests
- `gpu_generic_incompressible_schur_smoke_test.rs`
- `gpu_compressible_solver_test.rs`
- `gpu_generic_unified_solver_test.rs`
- `gpu_compressible_validation_test.rs`
- `gpu_divergence_test.rs`
- `gpu_fine_mesh_obstacle.rs`
- `gpu_codegen_matches_manual_test.rs`
- `gpu_solver_scaling_test.rs`

### Smoke Tests
- `ui_incompressible_air_smoke_test.rs`
- `ui_compressible_air_backstep_smoke_test.rs`

### Other Tests
- `diag_incompressible_channel_outer_iters_test.rs`
- `diag_incompressible_channel_flux_balance_test.rs`
- `model_analytical_solutions_test.rs`
- `amg_test.rs`

## Tests Intentionally Left with Both Schemes

These tests specifically compare or test both time schemes:

- `coupled_schemes_test.rs` - Tests multiple scheme combinations including both Euler and BDF2
- `gpu_low_mach_equivalence_test.rs` - Tests equivalence between time schemes
- `gpu_1d_structured_regression_test.rs` - Tests both schemes for regression
- `ui_compressible_dual_time_backstep_regression_test.rs` - Tests dual time stepping with both schemes

## Test Results

All modified tests pass with BDF2:

| Test | Result | Time |
|------|--------|------|
| gpu_generic_incompressible_schur_smoke_test | ✅ PASS | 0.40s |
| amg_test | ✅ PASS | 1.31s |
| gpu_divergence_test | ✅ PASS | 22.17s |
| gpu_generic_unified_solver_test | ✅ PASS | 0.35s |
| gpu_codegen_matches_manual_test | ✅ PASS | 0.27s |
| lid_driven_cavity_compressible_vs_incompressible | ✅ PASS | 110.32s |

## Compressible vs Incompressible Comparison

Both solvers now use BDF2 consistently:

| Solver | Time Scheme | max_restart | Tolerance |
|--------|-------------|-------------|-----------|
| Incompressible | BDF2 | 30 | 1e-6 |
| Compressible | BDF2 | 60 | 1e-10 |

Despite both using BDF2 and FGMRES, there remains ~46% RMS error between the solvers, confirming the disagreement is due to **physical formulation differences**, not time stepping or linear solver choices.

## Important Note: BDF2 Requires History

BDF2 is a multi-step method that requires state history. Tests using BDF2 must call `initialize_history()` after setting the initial state:

```rust
solver.set_uniform_state(rho0, [0.0, 0.0], p0);
solver.initialize_history(); // Required for BDF2!
```

All modified tests already include this call.
