# AGENTS.md instructions for /Volumes/sources/cfd2

## Mandatory OpenFOAM drift check (before/after each major changeset)

For every **major** changeset, run the OpenFOAM reference suite both **before** and **after** the edits, then compare failure magnitudes.

- These tests are expected to fail today in some cases.
- The key rule is: **error values must not grow** vs baseline.
- If any tracked error grows, treat it as a regression and call it out explicitly.

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
