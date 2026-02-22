# Fusion Authoring Guide

This guide is for model authors adding or extending DSL-based kernel fusion.

## 1) Make kernels fusion-capable (DSL)

Fusion synthesis only works for kernels emitted as `ModelKernelArtifact::DslProgram`.

1. In your module spec, register DSL generators with `ModelKernelGeneratorSpec::new_dsl` (or `new_shared_dsl` for shared kernels).
2. Return a populated `KernelProgram` from the generator.
3. Keep `dispatch`, launch semantics, indexing expressions, and binding interfaces accurate; the fusion pass validates compatibility.

Example locations:

- `src/solver/model/modules/generic_coupled.rs`
- `src/solver/model/kernel.rs`
- `crates/cfd2_codegen/src/solver/codegen/*.rs`

## 2) Declare side effects for hazard checks

Populate `KernelProgram.side_effects: SideEffectMetadata` so safe-policy checks can reason about hazards:

- `read_set`: resources read by the kernel
- `write_set`: resources written by the kernel
- `uses_barriers` / `uses_atomics`: mark when present

Under `KernelFusionPolicy::Safe`, detected hazards reject synthesis. Under `Aggressive`, hazards are reported for visibility.

## 3) Declare fusion rules

Rules are model-level `ModelKernelFusionRule` entries in module specs.

Key fields:

- `pattern`: ordered `KernelPatternAtom` sequence to match
- `replacement`: fused kernel spec inserted into the schedule
- `guards`: policy/stepping/module/grad-state constraints
- `binding_remaps`: optional slot remaps when layouts differ

Use `FusionGuard::MinPolicy(KernelFusionPolicy::Safe)` for safe-default rules; reserve aggressive-only rules for explicitly unsafe opportunities.

## 4) Generate and audit coverage

Regenerate coverage report:

```bash
cargo run --bin fusion_coverage -- --write FUSION_COVERAGE.md
```

Include aggressive hazard diagnostics:

```bash
cargo run --bin fusion_coverage -- --aggressive-hazard-report --write FUSION_COVERAGE.md
```

Use the report to verify:

- which kernels are DSL vs WGSL
- which adjacent pairs are fuseable/safe/aggressive-only
- why synthesis was rejected (hazard / bind / dispatch mismatch)

## 5) Validate runtime behavior

For new rules, validate both numerics and execution shape:

1. Snapshot parity across `Off` / `Safe` / `Aggressive`
2. Dispatch count expectations (reduced where intended)
3. Stepping coverage (`Coupled`, and `Implicit`/`Explicit` where relevant)

Representative tests live under `tests/*fusion_parity_test.rs`.

## 6) OpenFOAM drift check for major fusion changes

For major fusion changes, run before/after OpenFOAM diagnostics and confirm no metric regression:

```bash
mkdir -p target/openfoam_reference_logs

CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh \
  2>&1 | tee target/openfoam_reference_logs/before.log || true

CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh \
  2>&1 | tee target/openfoam_reference_logs/after.log || true

rg '^\[openfoam\]' target/openfoam_reference_logs/before.log > target/openfoam_reference_logs/before.metrics
rg '^\[openfoam\]' target/openfoam_reference_logs/after.log > target/openfoam_reference_logs/after.metrics
diff -u target/openfoam_reference_logs/before.metrics target/openfoam_reference_logs/after.metrics || true
```
