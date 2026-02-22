# Fusion Troubleshooting

Use this guide when fusion synthesis or parity checks fail.

## 1) Read `FUSION_COVERAGE.md` first

Regenerate with:

```bash
cargo run --bin fusion_coverage -- --write FUSION_COVERAGE.md
```

For aggressive diagnostics:

```bash
cargo run --bin fusion_coverage -- --aggressive-hazard-report --write FUSION_COVERAGE.md
```

Look at:

- **Adjacent Fusion Reassessment**: whether a pair is safe/aggressive fuseable
- **Declared Fusion Rule Reassessment**: whether the rule synthesizes
- rejection notes: hazard type, bind-interface conflict, dispatch mismatch

## 2) Hazard rejection fixes (RAW / WAR / WAW)

- `RAW` (later reads earlier write): usually reorder-sensitive; keep unfused for `Safe` unless sequencing is provably preserved.
- `WAR` (later writes earlier read): usually indicates clobber risk; split pattern or move write later.
- `WAW` (both write same resource): can be legal only with strict overwrite intent; avoid in `Safe`.
- `barriers/atomics`: not safe-fused without dedicated transforms.

Authoring checklist:

1. Verify each kernel's `SideEffectMetadata` is accurate.
2. Confirm both kernels use compatible dispatch/indexing/launch semantics.
3. Restrict risky rules with `FusionGuard::MinPolicy(KernelFusionPolicy::Aggressive)` when appropriate.

## 3) Bind-merge incompatibilities

Common causes:

- same `(group,binding)` slot but different type/name semantics
- logical same buffer placed at different slots in each kernel

Fixes:

1. Use `BindingRemap` in the rule to align logical resources to one layout.
2. Ensure types and names for remapped slots match after remap.
3. For mixed read/write access on same slot, rely on access promotion logic in fusion synthesis.

Reference example: cross-phase gradients+assembly rule in `src/solver/model/modules/generic_coupled.rs`.

## 4) Dispatch-domain / phase mismatch

If synthesis says kernels are incompatible:

- check `DispatchKindId` and `DispatchDomain` (Cells vs Faces must match)
- verify `KernelPatternAtom::with_phase(...)` for cross-phase patterns
- keep pattern order identical to runtime schedule order

## 5) Runtime parity validation workflow

After adding/changing a rule:

```bash
cargo test --test rhie_chow_fusion_parity_test -- --nocapture
cargo test --test compressible_fusion_parity_test -- --nocapture
cargo test --test generic_diffusion_fusion_parity_test -- --nocapture
```

Validate:

- field snapshots (`u`, `p`, gradients, auxiliaries) match expected policy parity
- dispatch counts do not increase unexpectedly

## 6) OpenFOAM regression guard (major changes)

Run before/after diagnostics and compare `[openfoam]` lines; worst-case errors must not grow.
