# ARCH_FIX_5: Consolidate Field Layout Authority

## Problem (Architecture Review Issue #5)

There are three overlapping representations of state buffer memory layout:

1. **`StateLayout`** (`cfd2_ir::equation::state_layout`) — The original field-to-offset map.
   Constructed from `Vec<FieldRef>` at model definition time. Used in `PortRegistry`, codegen
   (`packed_state_gradients`), and flux module resolver passes.

2. **`PortRegistry`** (`src/solver/model/ports/registry.rs`) — A newer typed port system that
   *wraps* `StateLayout`, adding dimension-checked `FieldPort`/`ParamPort`/`BufferPort`
   registration with idempotency. Stores a `StateLayout` internally and delegates field lookups
   to it.

3. **`ResolvedStateSlotsSpec`** (`cfd2_ir::ports`) — An IR-safe snapshot of resolved field offsets,
   stored on `PortManifest`. Created by `flux_module_resolver_pass` to avoid probing `StateLayout`
   at WGSL generation time. Consumed by codegen (`state_access.rs`, `coupled_common.rs`,
   `unified_assembly.rs`, etc.).

The problem: `PortRegistry` wraps `StateLayout` and must be kept in sync. The resolver pass
manually maps between `StateLayout` and `ResolvedStateSlotsSpec`. Model definitions construct
*both* a `StateLayout` and feed it into `PortRegistry`, creating two sources of truth that can
diverge.

## Design Goal

Make **`PortRegistry`** the single authority for field layout. Eliminate the pattern where
`StateLayout` is constructed independently and then separately wrapped by `PortRegistry`.
`ResolvedStateSlotsSpec` stays as the IR-safe codegen projection — but it should be derivable
from `PortRegistry` without a separate resolver pass that re-scans `StateLayout`.

## Constraints

- **Pure refactor** — no behavioral changes; all tests and OpenFOAM metrics must be unchanged.
- **`cfd2_ir` stays runtime-free** — `PortRegistry` lives in the main crate, so `cfd2_ir` cannot
  depend on it. `StateLayout` (in `cfd2_ir`) must remain available as a lightweight value type.
- **Codegen boundary** — `cfd2_codegen` can only depend on `cfd2_ir`, not the main crate. It will
  continue to consume `ResolvedStateSlotsSpec` (from `cfd2_ir::ports`).

## Phased Plan

### Phase 1: PortRegistry::from_fields() — ✅ DONE

Added `PortRegistry::from_fields(Vec<FieldRef>)` constructor that builds `StateLayout` internally.
Model definitions (`compressible.rs`, `incompressible_momentum.rs`, `generic_diffusion_demo.rs`)
now construct state layouts via `PortRegistry::from_fields()` instead of `StateLayout::new()`.

`StateLayout::new()` is no longer called directly in production model definition code. All
remaining `StateLayout::new()` calls are in:
- Test helpers (intentional — tests may construct directly)
- `cfd2_codegen` and `cfd2_ir` crates (can't use `PortRegistry`)
- `PortRegistry::from_fields()` itself (the single delegation point)

Also added:
- `PortRegistry::into_state_layout()` for transitional extraction
- `ModelSpec::state_stride()` convenience method

### Phase 2: PortRegistry-based resolver — ✅ DONE

Added `PortRegistry::to_resolved_state_slots()` and `to_resolved_state_slots_for()` methods
that produce `ResolvedStateSlotsSpec` from the registry's state layout.

Updated flux module resolver pass:
- Added `resolve_flux_module_state_slots_via_registry()` and
  `resolve_flux_module_state_slots_runtime_scheme_via_registry()` — registry-based entry points
- `flux_module.rs::resolve_state_slots_for_flux()` now creates a `PortRegistry` and uses
  the registry-based resolvers
- Old layout-based resolvers marked `#[cfg(test)]` for equivalence verification
- `kernel.rs::resolved_slots_from_layout()` now delegates to
  `PortRegistry::to_resolved_state_slots()`, removing duplicated conversion logic

Equivalence test confirms registry-based and layout-based resolvers produce identical results.

### Phase 3: Remove ModelSpec.state_layout field — DEFERRED

`ModelSpec.state_layout` remains as a public field. Removing it would require updating ~76
references across many files (unified_solver, recipe, UI, tests, benchmarks). The field now
serves as a cache: it is always constructed via `PortRegistry::from_fields()`, so there is
no divergence risk. When a future refactor introduces `PortRegistry` storage on `ModelSpec`,
the field can be removed.

### Phase 4: Restrict StateLayout visibility — DEFERRED

`StateLayout` remains `pub` in `cfd2_ir` because `cfd2_codegen` directly consumes it in
`packed_state_gradients.rs`. Converting those codegen call sites to use
`ResolvedStateSlotsSpec` would be a larger API change.

## Verification

- `cargo build` — clean (only pre-existing codegen warnings)
- `cargo test` — 156 lib tests pass, only pre-existing `block_jacobi` integration test failure
- OpenFOAM reference tests — zero metric drift (before/after identical)
- No external API changes
- `StateLayout` is no longer independently constructed outside `PortRegistry` in production code
