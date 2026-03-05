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

## Current Usage Map

### `StateLayout` direct consumers (non-test)
- `ModelSpec.state_layout` field — the canonical model-level layout
- `PortRegistry::new(state_layout)` — wraps it
- `PortRegistry` field lookups — delegates to internal `state_layout`
- `flux_module_resolver_pass` — builds `ResolvedStateSlotsSpec` from it
- `packed_state_gradients` (codegen) — reads `.stride()`, `.fields()`
- `flux_module_gradients_wgsl` (codegen) — reads `.stride()`, field offsets
- `flux_module_wgsl` (codegen) — reads field offsets via `ResolvedStateSlotsSpec`

### `PortRegistry` consumers (non-test)
- Model definitions (`compressible.rs`, `incompressible_momentum.rs`, etc.) — register fields
- `recipe.rs` — creates registry, registers manifests, passes `Arc<PortRegistry>` to plan
- `generic_coupled.rs` (lowering) — reads `PortRegistry` for field entries
- `unified_solver.rs` — stores `Arc<PortRegistry>`

### `ResolvedStateSlotsSpec` consumers (codegen)
- `state_access.rs` (`StateAccessor`) — the primary codegen interface for field offsets
- `coupled_common.rs` — assembly constants setup
- `primitive_expr.rs` — primitive variable code generation
- `time_integration.rs` — temporal term code generation
- `unified_assembly.rs` — assembly kernel generation
- `generic_coupled_kernels.rs` — generic coupled kernel generation

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
- **Build script** — `build.rs` `include!()`s files that reference `StateLayout`; paths must
  remain valid in the build-script module tree.

## Phased Plan

### Phase 1: Make `PortRegistry` derive from `EquationSystem` fields (not raw `StateLayout`)

Currently, model definitions do:
```rust
let layout = StateLayout::new(vec![field_a, field_b, ...]);
// ... separately ...
let mut registry = PortRegistry::new(layout.clone());
```

Change `PortRegistry::new()` to accept a `&[FieldRef]` (or `&EquationSystem`) and construct the
internal `StateLayout` itself. This eliminates the external `StateLayout` construction at model
definition sites.

**Files affected:**
- `src/solver/model/ports/registry.rs` — change constructor
- `src/solver/model/definitions/compressible.rs` — stop constructing `StateLayout` externally
- `src/solver/model/definitions/incompressible_momentum.rs` — same
- `src/solver/model/definitions/generic_diffusion_demo.rs` — same
- `src/solver/model/definitions.rs` — `ModelSpec.state_layout` → derived from registry
- `src/solver/model/modules/flux_module.rs` — use registry instead of layout
- `src/solver/model/modules/rhie_chow.rs` — use registry instead of layout (tests)
- `src/solver/gpu/recipe.rs` — use registry
- Various test helpers

### Phase 2: Add `to_resolved_state_slots()` method on `PortRegistry`

Currently, `flux_module_resolver_pass` manually walks `StateLayout` to build
`ResolvedStateSlotsSpec`. Add a `to_resolved_state_slots()` method on `PortRegistry` that produces
the same `ResolvedStateSlotsSpec` from the already-registered field ports — eliminating the need
for the separate resolver pass to scan `StateLayout` directly.

**Files affected:**
- `src/solver/model/ports/registry.rs` — add `to_resolved_state_slots()` method
- `src/solver/model/modules/flux_module_resolver_pass.rs` — simplify to use registry method
- `src/solver/model/modules/flux_module.rs` — pass registry instead of layout to resolver

### Phase 3: Remove `ModelSpec.state_layout` field

Once `PortRegistry` is the sole authority, `ModelSpec.state_layout` becomes redundant.
Replace it with a method that delegates to the registry (or remove it and update callers).

**Files affected:**
- `src/solver/model/definitions.rs` — remove `state_layout` field, add accessor
- All files that read `model.state_layout` — switch to `model.port_registry().state_layout()`
  or use `PortRegistry` methods directly

### Phase 4: Restrict `StateLayout` to internal use

Mark `StateLayout` as `pub(crate)` within `cfd2_ir` if possible, or document that it is an
internal implementation detail of `PortRegistry`. The public API for field offsets should be
`PortRegistry::get_field_entry()` and `ResolvedStateSlotsSpec`.

**Files affected:**
- `cfd2_ir::equation::state_layout` — visibility change (if feasible without breaking codegen)
- `cfd2_ir::kernel::mod.rs` — update re-exports

**Note:** This phase may be limited because `cfd2_codegen` (the codegen crate) directly consumes
`StateLayout` in `packed_state_gradients.rs`. Those codegen functions receive `StateLayout` from
the build script. A full removal would require converting those codegen call sites to use
`ResolvedStateSlotsSpec` instead, which is a larger change.

## Verification Criteria

1. `cargo build` — clean (no new warnings)
2. `cargo test` — all tests pass (only pre-existing `block_jacobi` failure)
3. OpenFOAM reference tests — zero metric drift
4. No external API changes visible to downstream consumers
5. `StateLayout` is no longer independently constructed outside `PortRegistry` in production code
   (test helpers may still construct it directly)

## Risk Assessment

- **Low risk**: Phases 1-2 are mechanical — changing constructor signatures and adding a method.
- **Medium risk**: Phase 3 touches many files but is still mechanical (field removal + accessor).
- **Higher risk**: Phase 4 may be blocked by codegen dependencies; defer if needed.

## Out of Scope

- Replacing `StateLayout` inside `cfd2_codegen` with `ResolvedStateSlotsSpec` throughout
  (that would be a separate, larger refactor affecting the codegen API surface)
- Changing `PortManifest` structure
- Modifying the `cfd2_macros` proc-macro
