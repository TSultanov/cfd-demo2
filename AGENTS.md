# AGENTS.md instructions for /Volumes/sources/cfd2

## Mandatory MMS gate (primary correctness oracle)

For any change touching discretization, schemes, boundary conditions, codegen, or time
integration, run the MMS convergence-order suite and require it green:

```bash
cargo test --features dev-tests --test mms_diffusion_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_scalar_transport_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_incompressible_order_test -- --test-threads 1
```

(Glob any additional `tests/mms_*` suites as they are added.) These tests verify the
discrete operators converge at design order against manufactured exact solutions; an order
drop or a blown error cap is a hard regression regardless of OpenFOAM metrics. The
incompressible (saddle-point) suite is the slowest (~3 min, dominated by the SOU n=64
level); keep new MMS cases lean — steady cases here settle in ~13 steps, so cost is
per-step outer iterations, not step count.

## Mandatory OpenFOAM drift check (before/after each major changeset)

For every **major** changeset, run the OpenFOAM reference suite both **before** and **after** the edits, then compare failure magnitudes.

- The suite passes at per-case error bands (`reference_bands()` in `tests/openfoam_reference/common.rs`).
- The key rule is: **error values must not grow** vs baseline, and bands only ratchet down.
- If any tracked error grows, treat it as a regression and call it out explicitly.

Runtime note / skip policy:

- The full suite takes ~13–15 minutes wall clock (the incompressible backstep,
  channel, and lid cases are the long poles at several minutes each).
  **TODO: investigate suite performance** — per-case step counts/convergence
  policies look like the dominant cost, not the comparison machinery; profile
  and trim before adding more reference cases.
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
