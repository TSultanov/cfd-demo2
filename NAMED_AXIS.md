# Named Axis Adoption in DSL Kernels — Progress

This document tracks progress on adopting the existing `NamedVecExpr`/`NamedMatExpr`
named axis tensor types in the DSL kernel generators. The goal is to replace raw
numeric indices with type-safe named axis access for improved readability, safety,
and maintainability.

## Current State (Summary)

The codebase has a **fully implemented** named axis system in
`cfd2_codegen::solver::codegen::dsl::tensor`:

| Type | Purpose | Status |
|------|---------|--------|
| `Axis<N>` trait | Maps symbolic index to positional `usize` | Implemented |
| `AxisXY` / `XY::{X, Y}` | 2D spatial directions | Implemented |
| `AxisCons` / `Cons::{Rho, Ru, Rv, Re}` | 4-component conservative variables | Implemented |
| `NamedVecExpr<N, Ax>` | Type-safe vector with named components | Implemented |
| `NamedMatExpr<R, C, RowAx, ColAx>` | Type-safe matrix with named row/col axes | Implemented |

These types were only exercised in unit tests — no production kernel code used them.

## Tier 1: XY Named Axis for Spatial Components

Replace `component(0)` / `component(1)` with `XY::X` / `XY::Y` across kernel
generators that deal with 2D spatial vector fields.

### 1a) XY Utility Methods

Add convenience methods to the `XY` enum to eliminate repeated `match component { 0 => "x", 1 => "y" }` patterns.

- [x] `XY::from_index(u32)` — convert numeric index to named axis value
- [x] `XY::suffix() -> &'static str` — return `"x"` or `"y"`
- [x] `XY::ALL` constant — `[XY::X, XY::Y]` for iteration
- [x] `XY::to_usize()` — direct method on the enum (not just via `Axis` trait)

### 1b) NamedVecExpr Convenience Constructors

- [x] `NamedVecExpr::from_components()` — construct from array of `Expr`
- [x] `NamedVecExpr::zeros()` — zero vector with axis tag

### 1c) Kernel Generator Refactors

- [x] `flux_module_gradients_wgsl.rs` — Replace `match target.base_component { 0 => ..., 1 => ... }` and manual `idx * stride + offset` patterns with XY-aware helpers
- [x] `flux_module_wgsl.rs` — Replace `component: 0` / `component: 1` in `StateKey`, `state_var_name`, `state_component_at_side_resolver`, `apply_slipwall_velocity_reflection_resolver` with XY enum usage (6 change sites; `resolve_state_field_component_resolver` left as-is since it handles `"z"=>2`)
- [x] `reconstruction.rs` — Replace `component(0)` / `component(1)` and `[0]` / `[1]` indexing in `vec2_reconstruction_xy` with `XY::X.to_usize()` / `XY::Y.to_usize()`
- [x] `rhie_chow.rs` — Replace 12 `FieldPort::component(0)` / `component(1)` calls with `XY::X.to_usize() as u32` / `XY::Y.to_usize() as u32`

## Tier 2: Typed Coupled-Unknown Accumulators

Replace `format!("diag_{u_idx}")` / `format!("rhs_{u_idx}")` / `format!("phi_{u_idx}")` /
`format!("start_row_{row}")` patterns (~50 occurrences across `generic_coupled_kernels.rs`
and `unified_assembly.rs`) with a typed `CoupledAccumulators` abstraction. Also deduplicate
~8 shared helper functions that were copy-pasted between the two assembly files.

### 2a) Deduplicate Shared Helpers

- [x] Create `coupled_common.rs` with 8 shared functions: `coupled_unknown_components`,
  `coupled_offsets`, `coefficient_value_expr`, `base_mesh_items`, `base_state_items`,
  `base_assembly_items`, `kernel_bindings_from_items`, `group_binding_from_attributes`
- [x] Update imports in `generic_coupled_kernels.rs`, `unified_assembly.rs`,
  `packed_state_gradients.rs`, `flux_module_wgsl.rs`, `flux_module_gradients_wgsl.rs`

### 2b) CoupledAccumulators Type

- [x] Design `CoupledAccumulators` struct (Vec-based, runtime `coupled_stride`)
- [x] Implement `AccIndex` wrapper with `From<u32>`, `From<i32>`, `From<usize>`
- [x] Methods: `declare()`, `declare_start_rows()`, `declare_phi()`, `diag()`, `rhs()`,
  `phi()`, `start_row()`, `add_diag()`, `sub_diag()`, `set_diag()`, `add_rhs()`,
  `sub_rhs()`, `set_rhs()`, `writeback()`
- [x] 6 unit tests

### 2c) Kernel Generator Refactors

- [x] `generic_coupled_kernels.rs` — 17 accumulator pattern occurrences replaced
  (start_row declarations, diag/rhs declarations, BDF1/BDF2/dtau time derivative,
  diffusion interior/boundary diag/rhs, writeback loop)
- [x] `unified_assembly.rs` — ~33 accumulator pattern occurrences replaced
  (start_row declarations, diag/rhs declarations, BDF1/BDF2/dtau time derivative,
  implicit/explicit source terms, implicit/explicit diffusion diag/rhs, DivFlux phi/rhs,
  scalar convection phi/diag/rhs, gradient rhs, writeback loop)
- [x] `flux_module_wgsl.rs` — evaluated; uses `phi_{off}` where `off` is a flux layout
  offset (not a coupled unknown index), so `CoupledAccumulators` does not apply

### Verification

- All 135 codegen tests pass
- All 151 lib tests pass
- OpenFOAM reference suite: no regressions (before/after metrics are identical)

## Tier 3: Named Block-CSR Matrix

Add `NamedBlockCsrSoaMatrix<Ax>` phantom-typed wrapper around `BlockCsrSoaMatrix`
to prevent row/col argument swaps in `block_matrix.entry(rank, row, col)` calls
by requiring `BlockRow<Ax>` and `BlockCol<Ax>` typed indices instead of raw `u8`.

### Design Decisions

- Block matrices are always square (same `coupled_stride` for rows and cols),
  so a single axis type parameter `Ax: CoupledAxis` covers both rows and columns.
- `CoupledAxis` is a separate trait from `Axis<N>` — it is not const-generic
  because the stride is determined at runtime via `dispatch_by_coupled_stride()`.
- Dispatch uses a `DispatchByStride<R>` trait pattern (not generic closures)
  since Rust doesn't support `impl Fn<Ax: CoupledAxis>()`.
- Dispatch happens inside codegen public API functions; the model layer
  (`kernel.rs`) needs no changes.

### 3a) CoupledAxis Trait + Concrete Enums

- [x] `CoupledAxis` trait with `STRIDE`, `to_u8()`, `from_u32()`, `all()` methods
- [x] `ScalarAxis` (stride 1): `Phi`
- [x] `IncompressibleAxis2D` (stride 3): `Ux`, `Uy`, `P`
- [x] `IncompressibleAxis3D` (stride 4): `Ux`, `Uy`, `Uz`, `P`
- [x] `CompressibleAxis2D` (stride 8): `Rho`, `RhoUx`, `RhoUy`, `RhoE`, `Ux`, `Uy`, `P`, `T`

### 3b) BlockRow / BlockCol Newtypes

- [x] `BlockRow<Ax>` / `BlockCol<Ax>` zero-cost wrappers with `new()`, `to_u8()`, `axis()`
- [x] `block_row<Ax>(index: u32)` / `block_col<Ax>(index: u32)` convenience constructors

### 3c) NamedBlockCsrSoaMatrix / NamedBlockCsrSoaEntry Wrappers

- [x] `NamedBlockCsrSoaMatrix<Ax>`: `new()`, `from_start_row_prefix()`, typed `entry()`,
  `row_entry()`, `inner()` escape hatch
- [x] `NamedBlockCsrSoaEntry<Ax>`: typed `entry()`, `index_expr()`, `access_expr()`,
  `inner()` escape hatch

### 3d) Exports + Dispatch Helper

- [x] `dsl/mod.rs` updated to export all new types and functions
- [x] `dispatch_by_coupled_stride()` function + `DispatchByStride<R>` trait in `tensor.rs`

### 3e) Unit Tests

- [x] CoupledAxis enum roundtrip tests (4 enums, including out-of-range panics)
- [x] BlockRow / BlockCol preservation and free-function tests
- [x] `dispatch_by_coupled_stride` — all 4 supported strides + unsupported error
- [x] NamedBlockCsrSoaMatrix construction, stride mismatch panic, typed entry equivalence
- [x] NamedBlockCsrSoaEntry delegation (entry, index_expr, access_expr)
- [x] `scatter_assign_to_named_block_entry_scaled` equivalence with untyped variant

### 3f) generic_coupled_kernels.rs Refactored

- [x] `main_assembly_fn` made generic over `Ax: CoupledAxis`
- [x] All 3 entry call sites updated to use `BlockRow`/`BlockCol`
- [x] `generate_generic_coupled_assembly_wgsl` updated with `DispatchByStride` dispatch

### 3g) unified_assembly.rs Refactored

- [x] `main_assembly_fn` made generic over `Ax: CoupledAxis`
- [x] All 9 entry call sites updated (zeroing loop, source cross-coupling,
  diffusion interior/boundary, convection neighbor, gradient diag/neighbor,
  final writeback)
- [x] Both `generate_unified_assembly_wgsl` and `generate_unified_assembly_kernel_program`
  updated with `DispatchByStride` dispatch structs

### 3h) Model-Layer Wrappers

- [x] No changes needed — dispatch is internal to codegen public functions

### 3i) scatter_assign_to_named_block_entry_scaled

- [x] New typed variant added to `MatExpr<R, C>` alongside original untyped method

### Verification

- All 156 codegen tests pass (20 new unit tests for Tier 3 types)
- All 151 lib tests pass
- OpenFOAM reference suite: no regressions (before/after metrics are identical)

## Tier 4: BC Table Accessor

Replace manual `face_idx * coupled_stride + u_idx` arithmetic with typed
`BcTable` (codegen DSL) and `HostBcTable` (host-side) accessors.

### New types

- `BcTable` (`crates/cfd2_codegen/src/solver/codegen/bc_table.rs`): codegen DSL
  wrapper that hides index arithmetic for `bc_kind[]`/`bc_value[]` lookups.
  Methods: `kind()`, `kind_raw()`, `value()`, `lookup()`, `ghost_value()`.
- `HostBcTable` (same file): host-side helper with `offset()`, `byte_offset()`,
  `row_base()` for CPU-side table construction and runtime patching.
- `BOUNDARY_TYPE_COUNT` (same file): constant `6` replacing hardcoded magic
  numbers for boundary type row count.
- `BoundaryType::bc_table_index()` (`src/solver/mesh/structs.rs`): maps
  `BoundaryType` variants to their BC table row index (1-5), replacing
  duplicated match arms.

### GPU codegen sites refactored (7 sites across 5 files)

- [x] `generic_coupled_kernels.rs` — 1 site → `BcTable::lookup()`
- [x] `unified_assembly.rs` — 4 sites → `BcTable::lookup()`
- [x] `packed_state_gradients.rs` — 1 site → `BcTable::ghost_value()`
- [x] `flux_module_wgsl.rs` — 1 site → `BcTable::kind_raw()` + `BcTable::value()`
- [x] `flux_module_gradients_wgsl.rs` — 1 site → `BcTable::ghost_value()`
- [x] `rhie_chow.rs` — 1 site → `BcTable::ghost_value()`

### Host-side sites refactored (3 sites across 3 files)

- [x] `definitions.rs` `to_gpu_tables()` — `HostBcTable::offset()`
- [x] `generic_coupled_backend.rs` — `BoundaryType::bc_table_index()` +
      `HostBcTable::row_base()` + `BOUNDARY_TYPE_COUNT`
- [x] `generic_coupled.rs` `spec_set_bc_value()` — `HostBcTable::byte_offset()` +
      `BOUNDARY_TYPE_COUNT`

### Verification

- All 165 codegen tests pass (9 new unit tests for BcTable/HostBcTable)
- All 151 lib tests pass
- OpenFOAM reference suite: no regressions (before/after metrics byte-identical)
