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

## Tier 2: Typed Coupled-Unknown Accumulators (Future)

Replace `format!("diag_{u_idx}")` / `format!("rhs_{u_idx}")` patterns (~40+
occurrences across `generic_coupled_kernels.rs` and `unified_assembly.rs`) with
a typed `CoupledAccumulators` abstraction.

- [ ] Design `CoupledAccumulators` type backed by `MatExpr` or newtype array
- [ ] Refactor `generic_coupled_kernels.rs`
- [ ] Refactor `unified_assembly.rs`

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
