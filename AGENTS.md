# AGENTS.md instructions for /Volumes/sources/cfd2

## Mandatory MMS gate (primary correctness oracle)

For any change touching discretization, schemes, boundary conditions, codegen, or time
integration, run the MMS convergence-order suite and require it green:

```bash
cargo test --features dev-tests --test mms_diffusion_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_scalar_transport_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_incompressible_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_buoyant_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_compressible_order_test -- --test-threads 1
```

(Glob any additional `tests/mms_*` suites as they are added.) These tests verify the
discrete operators converge at design order against manufactured exact solutions; an order
drop or a blown error cap is a hard regression regardless of OpenFOAM metrics. The
incompressible (saddle-point) suite is the slowest (~3 min, dominated by the SOU n=64
level); the compressible suite (~3 min) marches each level a fixed minimum t=6 to pass
the slow thermal transient. Keep new MMS cases lean — steady cases settle in ~13 steps
for the incompressible models, so cost is per-step outer iterations, not step count.
NOTE: the compressible suite asserts the physical-NS viscous operator (EXTRA_SHEAR = 0);
its `#[ignore]` probe (doubled-shear sources) must SATURATE — order-2 convergence there
means the pre-June-2026 double-counted laplacian came back (see the header of
`tests/mms_compressible_order_test.rs`).

## Mandatory ALE gate (moving-mesh path)

For any change touching the ALE codegen (`Term::relative_to_mesh`, moving-volume ddt,
`mesh_fluxes`/volume-history plumbing, `begin_ale_step`, `src/solver/mesh/ale.rs`) or the
mesh-refresh seam, run the ALE suite and require it green — same declaration style as the
MMS gate above:

```bash
cargo test --test static_wgsl_snapshot_test                    # static models byte-identical (hash-pinned)
cargo test --features meshgen,cpu --test ale_zero_flux_equivalence_test -- --test-threads 1
cargo test --features meshgen,cpu --test ale_gcl_test -- --test-threads 1
cargo test --features cpu --test ale_conservation_test -- --test-threads 1
cargo test --features dev-tests --test mms_ale_order_test -- --test-threads 1
```

The snapshot test pins every generated WGSL file's content hash (bless ritual in its
header); zero-flux equivalence pins ALE-with-zero-fluxes == static bitwise on CPU;
the GCL gate pins free-stream preservation on a deforming mesh (Euler AND BDF2, both
backends); the conservation audit pins the f64 mesh-side mass/area identities and
zero spurious ALE-injected velocity; the MMS-ALE suite (dev-tests tier, ~8 min, the
n=64 spatial level and the 80-step temporal level dominate) pins spatial order ~2 and
temporal BDF2 order ~2 on prescribed motion.

## Mandatory OpenFOAM drift check (before/after each major changeset)

For every **major** changeset, run the OpenFOAM reference suite both **before** and **after** the edits, then compare failure magnitudes.

- The suite passes at per-case error bands (`reference_bands()` in `tests/openfoam_reference/common.rs`).
- The key rule is: **error values must not grow** vs baseline, and bands only ratchet down.
- If any tracked error grows, treat it as a regression and call it out explicitly.

Runtime note / skip policy:

- The full suite takes ~8 minutes wall clock (the incompressible lid ~4 min
  and backstep ~3 min are the long poles; the channel converges from rest
  in ~20 s). The incompressible references are machine-converged STEADY
  states (audited June 2026; the old lid/backstep snapshots were
  mid-transient) and their tests march to steady at the canonical
  configuration dt=0.02/alpha_u=0.7 — dt is NOT a free knob (d_p ∝
  alpha_u·dt is part of the spatial discretization). The compressible
  references are matched-time transients by construction (provenance
  notes in the test headers). `CFD2_LIN_TOL` overrides the linear
  tolerance for sweeps/diagnostics.
- For changesets that are **provably numerics-neutral** — generated WGSL
  byte-identical (zero diffs under `shaders/generated/`, clean
  `check_generated_wgsl.sh`) AND no runtime/host solver-path changes — the
  full before/after run may be skipped; say so explicitly in the changeset
  report. Numerics-affecting changesets still require the before/after run.

Recommended workflow:

```bash
mkdir -p target/openfoam_reference_logs

# Baseline (before edits)
CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh \
  2>&1 | tee target/openfoam_reference_logs/before.log || true

# Post-change (after edits)
CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh \
  2>&1 | tee target/openfoam_reference_logs/after.log || true

# Compare reported OpenFOAM diagnostics
rg '^\[openfoam\]' target/openfoam_reference_logs/before.log > target/openfoam_reference_logs/before.metrics
rg '^\[openfoam\]' target/openfoam_reference_logs/after.log > target/openfoam_reference_logs/after.metrics
diff -u target/openfoam_reference_logs/before.metrics target/openfoam_reference_logs/after.metrics || true
```

Reporting requirement for major changesets:

- Include baseline vs post-change OpenFOAM metric summary.
- Explicitly state whether worst-case error is unchanged/improved/regressed.
