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

## Tier 3: Named Block-CSR Matrix (Future)

Add `NamedBlockCsrSoaMatrix<R, C, RowAx, ColAx>` phantom wrapper to prevent
row/col argument swaps in `block_matrix.entry(rank, u_idx as u8, u_idx as u8)`.

- [ ] Implement `NamedBlockCsrSoaMatrix` wrapper
- [ ] Refactor assembly kernels to use it

## Tier 4: BC Table Accessor (Future)

Replace manual `face_idx * coupled_stride + u_idx` arithmetic (~8 occurrences)
with a typed `BcTable` accessor.

- [ ] Implement `BcTable` type with `kind(face_idx, unknown)` / `value(face_idx, unknown)`
- [ ] Refactor assembly and gradient kernels
