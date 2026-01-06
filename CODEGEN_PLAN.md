# CFD2: Codegen + Solver Roadmap (Trimmed)

## Current State (What Exists)
- WGSL codegen pipeline is in place and emits generated shaders via `build.rs`.
- GPU incompressible coupled solver and GPU compressible solver exist.
- Compressible solver still struggles in low‑Mach: slow/fragile convergence and pressure/velocity decoupling symptoms (stripey/checkerboard).

## Primary Guiding Principle (For Future Work)
Prefer cheap, 1D/“1.5D” diagnostics + plots over expensive 2D vortex-street runs. Do not iterate on obstacle/vortex-street behavior until the low‑Mach 1D suite is clean and stable.

## Reduce Knobs (Make It Automatic)
Goal: “works by default” with a small, high-level parameter surface.

### What To Remove / Avoid
- Per-run tuning via a pile of ad-hoc env vars.
- Exposed setters for parameters that are ineffective or only needed for debugging.
- Dead or unused pipelines/stages in the runtime path.

### What To Prefer
- One small `*Config` struct per solver with strong defaults.
- Automatic scaling based on cheap local sensors (Mach, smoothness/gradient sensors, CFL-ish estimates).
- Adaptive iteration controls: stop early when converged; only tighten tolerances when needed.

## Low‑Mach Verification Suite (Must-Have)
Tests should:
- Run quickly (few cells, few steps).
- Save plots to `target/test_plots/...` for visual sanity checks.
- Compare compressible vs incompressible wherever physics overlap (low‑Mach regimes).
- Include analytical/known approximations where feasible (especially in 1D).

### Building Blocks
- Deterministic structured meshes for 1D-style problems (avoid noisy mesh-generation variability).
- Simple PNG line plots (don’t rely only on scalar norms/metrics).

### Existing Lightweight Pieces (Implemented)
- `generate_structured_rect_mesh(...)` for deterministic rectangular grids.
- 1D acoustic pulse regression with plots: `tests/gpu_1d_structured_regression_test.rs`.
- Diagnostic low‑Mach comparison harness (currently `#[ignore]`) that produces plots.

## Next Steps (What’s Not Done Yet)
1. **Expand 1D analytical comparisons**
   - Add an exact/standard Riemann problem (Sod) sampler and compare profiles (ρ, u, p) against the compressible solver.
   - Add viscous 1D decay/transport tests in regimes where incompressible and compressible should match.
2. **Make compressible low‑Mach robust by default**
   - Harden pressure/velocity coupling (remove checkerboarding) with automatic, sensor-gated coupling terms.
   - Remove reliance on manual `dtau`/relaxation tuning by selecting stabilization automatically from mesh + state scales.
3. **Collapse solver API surface**
   - Replace many individual setters with a single `configure(...)` call + a few essential runtime controls (`dt`, `inlet_velocity`, material props).
4. **Only then: return to 2D vortex street**
   - Use the 1D suite to validate changes before running long obstacle/vortex cases.
   - When re-enabled, add a reduced “fast” 2D smoke case (coarse mesh, short runtime) to catch regressions early.

