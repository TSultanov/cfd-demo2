# OpenFOAM Reference Discrepancy Investigation (cfd2)

Goal: reduce discrepancies between `cfd2` and OpenFOAM v2512 reference solutions, with extra focus on compressible cases and late-stage behavior (stability and end-time fields).

## Current findings (Feb 2026)
- Compressible reference tests currently failing:
  - `compressible_lid_driven_cavity`: stable with time-accurate stepping, but large velocity mismatch vs OpenFOAM (~60% max cell rel at t=0.003s); mean-free pressure (`dp`) differs significantly even when absolute `p≈101325` looks close.
  - `compressible_acoustic_box`: pressure matches very closely; `u_x` mismatch ~0.8% max cell rel.
  - `compressible_backwards_step`: `U` mismatch ~0.35% max cell rel; `p` mismatch ~0.18% max cell rel.
  - `compressible_supersonic_wedge`: `U` mismatch ~0.085% max cell rel; `p` mismatch ~0.18% max cell rel.

## Working hypotheses
- Low-Mach preconditioning should be treated as a pseudo-transient/dual-time tool (`dtau > 0`) rather than used in time-accurate mode (`dtau == 0`).
- The low-Mach cavity mismatch is likely dominated by incompressible-limit pressure/velocity coupling and/or viscous split differences vs `rhoCentralFoam`.
- For low-Mach cases, absolute pressure metrics are ill-conditioned; mean-free pressure (`p - mean(p)`) is more meaningful.

## Checklist
1. Align low-Mach cavity end-time fields to OpenFOAM reference (reduce ~60% U error).
2. Re-evaluate viscous split (`laplacian(mu,U)` + `div(tauMC)` + energy viscous-work) against rhoCentralFoam behavior on orthogonal meshes.
3. Investigate residual mismatches in other compressible cases (acoustic/backstep/wedge), focusing on end-time behavior.
4. Consider whether some cases should be treated as steady via pseudo-transient/dual-time (dtau) rather than strictly transient, and what reference outputs should be used for comparison.

## Validation commands
- Regenerate OpenFOAM CSVs: `bash scripts/regenerate_openfoam_reference_data.sh`
- Run reference tests: `bash scripts/run_openfoam_reference_tests.sh`
- Focused compressible diagnostics:
  - `CFD2_OPENFOAM_DIAG=1 cargo test -p cfd2 --no-fail-fast --test openfoam_compressible_acoustic_reference_test --test openfoam_compressible_backwards_step_reference_test --test openfoam_compressible_lid_driven_cavity_reference_test --test openfoam_compressible_supersonic_wedge_reference_test -- --ignored --nocapture`

