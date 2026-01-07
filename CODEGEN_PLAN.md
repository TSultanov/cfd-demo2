# CFD2 Codegen Plan: Typed AST + High-Level DSL

This file tracks *codegen* work only. Solver physics/tuning tasks should live elsewhere.

## Current Status
- WGSL is generated from a real AST (`src/solver/codegen/wgsl_ast.rs`) with a precedence-aware renderer.
- `wgsl_dsl` is AST-only (no string parsing helpers remain); `src/solver/codegen/*` kernels do not build expressions via strings.
- Tensor helpers exist in `src/solver/codegen/dsl/tensor.rs`:
  - `VecExpr<const N>` and `MatExpr<const R, const C>` for unrolled small-vector/matrix algebra and scatters.
  - Named-axis wrappers (`NamedVecExpr`, `NamedMatExpr`) with `AxisXY` (`XY`) and `AxisCons` (`Cons`) plus basic broadcasting (`mul_row_broadcast`, `mul_col_broadcast`) and reductions (`contract_rows`).
- Units/types scaffolding exists in `src/solver/codegen/dsl/*`:
  - `UnitDim` uses SI base dimensions with rational exponents (`src/solver/units.rs`).
  - `DslType`/`Shape`/`TypedExpr` support unit propagation and validation (units erased when emitting WGSL).
  - Model-side FV validation (`EquationSystem::validate_units`) is wired into lowering and `StateLayout` stores per-field units.
- Sparse matrix types exist in `src/solver/codegen/dsl/matrix.rs`:
  - `CsrPattern`, `CsrMatrix`, `BlockCsrMatrix`, `BlockCsrSoaMatrix`, etc.
- Kernel codegen status:
  - All kernels under `src/solver/codegen/` are AST-based (no `wgsl_dsl::expr`, no stringly-typed `let_/assign/if_block/for_loop`).
- 1D regression tests that save plots exist (acoustic pulse + Sod shock tube) and are the primary safety net for refactors.

## Recently Completed
- Removed the string-based WGSL DSL API from `src/solver/codegen/wgsl_dsl.rs` and updated downstream kernels/tests to build `wgsl_ast::Expr` directly.
- Migrated matrix-heavy compressible assembly sections to `MatExpr`/`VecExpr` and replaced large hand-expanded derivative boilerplate with vectorized tensor ops.
- Updated generated WGSL shaders under `src/solver/gpu/shaders/generated/*` to match the new AST emit path.

## Next Work (Planned)
1. **Make tensors typed/unit-aware**
   - Add `TypedVecExpr` / `TypedMatExpr` (or a generic wrapper) that carries `DslType` + `UnitDim` and supports common ops (`+ - * / dot mul_mat`), enforcing unit rules.
   - Prefer `state_*_typed` accessors in kernels so unit validation happens during expression construction.
2. **Expand named axes + broadcasting**
   - Add additional standard axes (e.g. `{xx,xy,yx,yy}` for gradients/strain, or `{rho,u,v,e}` vs conserved) and generalized broadcast/reduction APIs.
   - Use these to eliminate remaining per-component/per-variable boilerplate in viscous and Jacobian code.
3. **Higher-level sparse assembly API**
   - Wrap `BlockCsrSoaEntry` scatters/accumulates behind a small API that accepts `MatExpr`/`NamedMatExpr` directly (and later typed matrices), minimizing manual indexing.
4. **Migration tracker + regressions**
   - Track (per kernel) remaining places where we still do direct `.x/.y` component math or manual index arithmetic.
   - Keep 1D plot regressions (acoustic pulse + Sod) as behavior guardrails; add “golden shader” parity checks only for kernels actively being refactored.

## Notes / Gotchas
- WGSL `dot(...)` works on `vecN<f32>` types, not the project’s `Vector2` struct; convert via helpers (e.g. `vec2<f32>(v.x, v.y)`).
- `build.rs` manually `include!()`s some codegen modules; new DSL modules must be added there as well.
