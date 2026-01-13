# Reference solutions (OpenFOAM)

This repo includes regression tests that compare `cfd2` against reference data generated with OpenFOAM v2512.

The OpenFOAM case templates live in:

- `reference/openfoam/incompressible_channel`
- `reference/openfoam/compressible_acoustic_box`

The committed reference datasets live in:

- `tests/openfoam_reference/data/incompressible_channel_centerline.csv`
- `tests/openfoam_reference/data/incompressible_channel_full_field.csv`
- `tests/openfoam_reference/data/compressible_acoustic_centerline.csv`
- `tests/openfoam_reference/data/compressible_acoustic_full_field.csv`

## Regenerating reference data

1. Ensure OpenFOAM v2512 is available (this machine provides the `openfoam` wrapper and `/Volumes/OpenFOAM-v2512`).
2. Run:
   - `bash scripts/regenerate_openfoam_reference_data.sh`

This will run each case in `target/openfoam_reference_runs/` and overwrite the CSV files under `tests/openfoam_reference/data/`.

## Case details

### Incompressible channel (`simpleFoam`)

- Geometry: 2D channel, `L=1.0`, `H=0.2` (single cell thick in z, `empty` front/back).
- Mesh: `40 x 20`.
- BCs:
  - `U`: inlet `fixedValue (1 0 0)`, outlet `zeroGradient`, walls `noSlip`
  - `p`: inlet `zeroGradient`, outlet `fixedValue 0`, walls `zeroGradient`
- Model: laminar (`constant/turbulenceProperties`).
- Transport: kinematic viscosity `nu=0.01` (`constant/transportProperties`).
- Numerics:
  - `div(phi,U)` upwind (`system/fvSchemes`)
  - Convergence: `system/fvSolution` uses `SIMPLE.residualControl` with `p` and `U` at `1e-12`.
- Sampling:
  - 20 probes placed at structured cell centers along `x=0.4875`, spanning `y=0.005..0.195`.
  - Full-field: 800 probes placed at all structured cell centers (`40x20`), written via a second `probesAll` function object.

### Compressible acoustic box (`rhoCentralFoam`)

- Geometry: 2D box, `L=1.0`, `H=0.05` (single cell thick in z, `empty` front/back).
- Mesh: `200 x 1`.
- BCs: slip walls (reflecting) on all 4 sides (`U` uses `slip`, `p` uses `zeroGradient`).
- Thermo:
  - Ideal gas configured to give `gamma≈1.4` (see `constant/thermophysicalProperties`).
  - Viscosity is set to `mu=0` (Euler-like).
- Initial condition:
  - `p(x) = 1 * (1 + 1e-3*cos(pi*x))` applied with `setExprFields` (`system/setExprFieldsDict`).
  - `U=0`, `T=1` initially.
- Time stepping:
  - Fixed `deltaT=5e-4`, `endTime=0.05` (`system/controlDict`).
  - Kurganov (central-upwind) flux scheme (`system/fvSchemes`) to match the style of flux used by `cfd2`'s compressible solver.
- Sampling:
  - 200 probes placed at structured cell centers along the midline (`y=0.025`).
  - Full-field: identical to the midline for `Ny=1`, written via a second `probesAll` function object.
