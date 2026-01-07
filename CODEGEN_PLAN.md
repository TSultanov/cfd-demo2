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
    - Includes `identity()`, `from_entries(...)`, diagonal updates (`assign_op_diag`), matrix multiplication (`mul_mat`), and prefix var helpers (`var_prefix`).
  - `VecExpr<const N>` for `vecN<f32>` expressions and basic vector algebra (used to remove per-component loops).
    - Includes bridging helpers like `to_vector2_struct()` for writing into `Vector2` storage buffers.
  - Named axes + broadcasting are now available for the most common CFD dimensions:
    - `AxisXY`/`XY` for `{x,y}` and `AxisCons`/`Cons` for `{rho, rhoU.x, rhoU.y, rhoE}`.
    - `NamedVecExpr`/`NamedMatExpr` wrap `VecExpr`/`MatExpr` and provide `at(...)` access, broadcast multiplies, and row-contraction (`contract_rows`) for dot/reduction-style operations.
- Sparse matrix types exist in the DSL (`src/solver/codegen/dsl/matrix.rs`):
  - `CsrPattern`, `CsrMatrix`, `BlockCsrMatrix`, `BlockCsrSoaMatrix`, `BlockCsrSoaEntry`, `BlockShape`
- Migration is in progress:
  - Some matrix indexing/scatters in hot kernels are AST-first (no parser for `matrix_values[...]` writes).
  - `wgsl_dsl` now has helpers for linear array indexing (`idx * stride + offset`) to avoid `format!(...)`-built indices.
  - `wgsl_dsl` has basic vector helpers (`vec2<f32>` construction, `dot/min/max`) used by reconstruction and flux kernels.
  - `reconstruction.rs` is AST-first (no string parsing) and is shared by incompressible + compressible kernels.
  - `compressible_gradients` removed x/y component loops in favor of `VecExpr` ops (accumulate `grad += scalar * area * normal`).
  - Incompressible face-based kernels now use `VecExpr` for geometry + dot products (`normal_vec`, `face_vec`) to avoid x/y duplication:
    - `prepare_coupled`, `flux_rhie_chow`, `pressure_assembly`
  - Compressible per-face kernels now use `VecExpr` for momentum/velocity algebra:
    - `compressible_apply`/`compressible_update` treat momentum updates as `vec2` and use AST assigns for state writes.
    - `compressible_flux_kt` computes KT momentum flux as a `vec2`, uses `dot(u, normal_vec)`, vectorizes viscous diffusion and pressure-coupling updates, and orients normals via `dot(face_center - owner_center, normal_vec)`.
  - `compressible_assembly` migrated the matrix-heavy parts to `MatExpr`:
    - Jacobians (`jac_l`, `jac_r`) are built from matrix algebra and scattered via `BlockCsrSoaEntry`.
    - Inlet/wall boundary conditions are expressed as `jac_l + jac_r * T_bc` using `mul_mat`.
    - Removed per-entry temporary scalars like `jac_l_00` and `A_l_10` in favor of matrix notation (`a_l_mat`, `a_r_mat`).
    - Viscous-energy Jacobian terms now use named tensor ops (`du_dU`, `d_diff`, `du_face`, `d_e_visc`) instead of hand-expanded `du_lx_drho`/`d_e_visc_l_rho`-style scalar boilerplate.
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
  - `compressible_flux_kt`, `prepare_coupled`, `flux_rhie_chow`, `pressure_assembly`, `compressible_apply`, `compressible_update`: major x/y duplication removed via `VecExpr`, but many scalar `dsl::let_(&str)` remain; continue converting to `let_expr`/AST where practical.
  - `compressible_assembly`: matrix/tensor portions are now `MatExpr`-based; remaining work is converting scalar/intermediate math away from `dsl::let_(&str)` and other parse-based helpers.
  - `coupled_assembly`: continue replacing remaining string-based math (still some `dsl::let_(&str)` + `Vector2` component access in face loops).
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
  - Current: `MatExpr<const R, const C>` and `VecExpr<const N>` are Expr-level helpers for unrolled ops/scatters, vector algebra, and small matrix multiplication.
  - Current (new): `NamedVecExpr`/`NamedMatExpr` enable axis-aware indexing and reduction/broadcast operations for `XY` and conserved-variable axes.
  - Missing: a typed variant (`TypedMat`) that tracks per-entry `DslType`/`UnitDim` and supports more ops (e.g. block transforms and structured assembly).
  - Partial: basic vector helpers exist in `wgsl_dsl` (`vec2<f32>` construction + `dot/min/max`).
  - Missing: typed vectors/matrices (units + shapes) and higher-level ops (e.g. mat-vec, vector-valued reconstruction, structured tensor access).

### 5) Discovery Notes (Gotchas)
- `build.rs` manually `include!()`s the codegen DSL modules; new DSL files must be added there as well (e.g. `dsl/tensor.rs`).
- WGSL `dot(...)` only accepts `vecN<...>` types, not the project’s `Vector2` struct; convert via `vec2<f32>(v.x, v.y)` (helper: `wgsl_dsl::vec2_f32_from_xy_fields`).
- For face-based kernels, don’t assume mesh normals are consistently oriented; re-orient with a local check like `dot(face_center - owner_center, normal_vec) < 0` and negate when needed.

### 6) Kernel Migration Tracker + Parity Tests
**Deliverable:** migrated kernels remain behaviorally stable while removing strings.

- Track per-kernel migration status (AST-only vs mixed vs string-heavy).
  - Mostly `VecExpr`/AST (still some strings): `compressible_apply`, `compressible_update`, `compressible_flux_kt`, `prepare_coupled`, `flux_rhie_chow`, `pressure_assembly`, `coupled_assembly`.
  - `MatExpr` for block assembly but still string-heavy elsewhere: `compressible_assembly`.
- Add “golden WGSL” parity tests (string compare against committed generated files) *only* for kernels under active migration.
- Keep the 1D plot regressions as the runtime validation layer for solver behavior.
