# CFD2 Codegen Plan: Typed AST + High-Level DSL

This file tracks *codegen* work only. Solver physics/tuning tasks should live elsewhere.

## Current Status
- WGSL codegen emits generated shaders via `build.rs` into `src/solver/gpu/shaders/generated/*`.
- A WGSL AST exists (`src/solver/codegen/wgsl_ast.rs`) with a precedence-aware renderer; kernels can build expressions without parsing strings.
- Typed DSL scaffolding exists (`src/solver/codegen/dsl/*`):
  - `DslType`/`Shape` + `TypedExpr`
  - `UnitDim` uses **SI base dimensions** (M/L/T) and supports **rational exponents** (e.g. `sqrt`).
- Dense tensor helpers exist in the DSL (`src/solver/codegen/dsl/tensor.rs`):
  - `MatExpr<const R, const C>` for building small dense matrix expressions and emitting unrolled assigns/scatters.
- Sparse matrix types exist in the DSL (`src/solver/codegen/dsl/matrix.rs`):
  - `CsrPattern`, `CsrMatrix`, `BlockCsrMatrix`, `BlockCsrSoaMatrix`, `BlockCsrSoaEntry`, `BlockShape`
- Migration is in progress:
  - Some matrix indexing/scatters in hot kernels are AST-first (no parser for `matrix_values[...]` writes).
  - `wgsl_dsl` now has helpers for linear array indexing (`idx * stride + offset`) to avoid `format!(...)`-built indices.
- 1D regression tests that save plots exist (acoustic pulse + Sod shock tube) and are the primary safety net for refactors.

## Goals
1. **No stringly-typed expressions in codegen**: build expressions from a real AST (no `format!(...)` + parse in kernels).
2. **A proper DSL for matrix/tensor/multidimensional operations** that lowers to WGSL reliably.
3. **Physical-units-aware type-checking** in the DSL (units erased when lowered to WGSL).
4. Incremental migration with strong regression tests; keep generated WGSL stable until deliberate changes are made.

## Roadmap (Remaining Work)

### 1) Eliminate String Parsing From Kernels
**Deliverable:** kernel codegen does not call `wgsl_dsl::expr(&str)` in hot paths.

- Keep adding small AST helpers where repeated patterns show up (array indexing, block scatters, component swizzles).
- Migrate kernels one-by-one, preferring the most string-fragile sections first:
  - `coupled_assembly`: continue replacing remaining string-based math; diagonal block scatter is now `MatExpr`-based.
  - `compressible_assembly`: continue replacing remaining string-based math; CSR-SoA scatters and boundary block ops are now `MatExpr`-based.
  - `pressure_assembly`, `update_fields_from_coupled`, `compressible_flux_kt`, `compressible_gradients`.
- Once migrations are sufficiently complete, gate `wgsl_dsl::expr(&str)` behind tests/debug (or deprecate it).

### 2) Make the Typed DSL More Complete (Units + Shapes)
**Deliverable:** common numerical building blocks are expressible with `TypedExpr` without falling back to strings.

- Add typed helpers for common intrinsics and comparisons (`min/max/abs/select/clamp`, comparisons, `dot`, component access).
- Make unit/type diagnostics actionable (include operation + field/var names where possible).

### 3) Attach SI Units to Model/State Access (Enforce Them)
**Deliverable:** state loads/stores and coefficients carry units and are checked in codegen.

- Define an SI unit map for core fields and constants (at minimum):
  - `rho` [kg/m³], `p` [Pa], `U` [m/s], `rhoU` [kg/(m²·s)], `rhoE` [Pa], `mu` [Pa·s], `dt` [s].
- Provide typed state/coeff access helpers that return `TypedExpr` with `UnitDim` + `DslType`.
- Add unit tests for invalid operations (e.g. `p + U`, `sqrt(rho)` without compensating units) with clear error messages.

### 4) Higher-Level Matrix DSL (Block/CSR Assembly)
**Deliverable:** matrix-heavy kernels stop hand-writing `_00/_01/...` patterns.

- Unify sparse storage layouts behind a small API (contiguous vs row-split SoA) and provide safe scatter/accumulate helpers.
- Expand dense tensor support:
  - Current: `MatExpr<const R, const C>` is an Expr-level helper for unrolled ops/scatters.
  - Missing: a typed variant (`TypedMat`) that tracks per-entry `DslType`/`UnitDim` and supports more ops (e.g. block transforms and structured assembly).
  - Missing: vector helpers (`VecExpr` / typed component access) to avoid `.x/.y` string usage in kernels.

### 5) Discovery Notes (Gotchas)
- `build.rs` manually `include!()`s the codegen DSL modules; new DSL files must be added there as well (e.g. `dsl/tensor.rs`).

### 6) Kernel Migration Tracker + Parity Tests
**Deliverable:** migrated kernels remain behaviorally stable while removing strings.

- Track per-kernel migration status (AST-only vs mixed vs string-heavy).
- Add “golden WGSL” parity tests (string compare against committed generated files) *only* for kernels under active migration.
- Keep the 1D plot regressions as the runtime validation layer for solver behavior.
