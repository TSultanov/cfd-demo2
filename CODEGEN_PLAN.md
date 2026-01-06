# CFD2 Codegen Plan: Typed AST + High-Level DSL

This file tracks *codegen* work only. Solver physics/tuning tasks should live elsewhere.

## Where We Are Today
- WGSL codegen exists and emits generated shaders via `build.rs` into `src/solver/gpu/shaders/generated/*`.
- There is a WGSL AST (`src/solver/codegen/wgsl_ast.rs`) for items/statements/expressions with a precedence-aware renderer.
- **But** most kernels still build expressions by string formatting + parsing (`wgsl_dsl::expr("a + b")`). This is fragile and makes higher-level operations (matrices/tensors) awkward.
- Some “higher-order” helpers exist to reduce repetition (e.g., component loops and 4×4 matrix helpers), but they still operate on strings.
- 1D regression tests that save plots exist (acoustic pulse + Sod shock tube) and should be the primary safety net for future refactors.

## Goals
1. **No stringly-typed expressions in codegen**: build expressions from a real AST, not `format!(...)` + parse.
2. **A proper DSL for tensor/matrix/multidimensional operations** that can be lowered to WGSL reliably.
3. **Physical-units-aware type-checking** in the DSL (units erased when lowered to WGSL).
4. Incremental migration with strong regression tests; keep generated WGSL stable until deliberate changes are made.

## Non-Goals (For Now)
- Rewriting the entire solver IR in one shot.
- Adding “smart” optimizations before correctness/ergonomics are in place.
- Requiring a heavy dependency stack for units/type-level algebra; start with a lightweight checker.

## Plan

### Phase 1: Make `wgsl_ast::Expr` Buildable (No Parser in the Hot Path)
**Deliverable:** codegen can build expressions without parsing strings.

- Add constructors/builders for `wgsl_ast::Expr`:
  - `Expr::ident("foo")`, `Expr::lit_f32(1.0)`, `Expr::field(base, "x")`, `Expr::index(base, idx)`,
  - `Expr::call("min", [a, b])`, `Expr::binary(a, Add, b)`, `Expr::unary(Negate, a)`, etc.
- Update `wgsl_dsl` to support `*_expr` variants that accept `Expr` directly:
  - `let_expr(name, Expr)`, `assign_expr(target: Expr, value: Expr)`, etc.
- Keep the string parser as a **test/debug** tool only (or behind a feature flag), not as the primary API.
- Add unit tests that ensure builder-rendered WGSL matches the previous string-based output for representative expressions.

### Phase 2: Introduce a Typed DSL IR (Shapes + Units) Above WGSL
**Deliverable:** a new IR that can express matrix/tensor operations and type-check them before lowering.

- Add a new DSL layer (e.g., `src/solver/codegen/dsl/*`) with:
  - `DslType`: scalar/vec2/vec3/matNxM (+ element type).
  - `UnitDim`: dimension exponents for base units (M, L, T, Θ, …) or a minimal subset to start.
  - `TypedExpr { ty: DslType, unit: UnitDim, expr: DslExpr }`.
  - `DslExpr`: identifiers, literals, calls, indexing/fields, unary/binary ops, select/if, etc.
- Implement a type-checker/inferencer for `DslExpr`:
  - Addition/subtraction requires identical units and shapes.
  - Multiplication/division combines units and checks compatible shapes.
  - Comparisons require compatible units (usually equal units).
  - Function calls validate argument counts/types/units for a supported set (`min/max/abs/sqrt`, etc.).
- Implement lowering: `TypedExpr -> wgsl_ast::Expr` (units erased, shape mapped to WGSL types).

### Phase 3: First-Class Matrix/Tensor Operations in the DSL
**Deliverable:** matrix-heavy kernels can be expressed without manual `_00/_01/...` variables.

- Provide a small tensor library on top of `TypedExpr`:
  - `Vec<N>` and `Mat<R,C>` with indexing, component access, and common ops (`mul`, `mad`, `transpose`, `diag`, `outer`).
  - Helpers for “block CSR” assembly patterns:
    - scatter a block `Mat<R,C>` into `matrix_values` given base indices,
    - accumulate diagonal blocks cleanly,
    - write RHS vectors with proper strides.
- Lower these ops to explicit WGSL AST statements (loops unrolled at codegen time when dimensions are const).

### Phase 4: Attach Units to Model Fields (and Enforce Them)
**Deliverable:** field loads/stores come with units; illegal operations are caught during codegen.

- Extend model field metadata (likely in `src/solver/model/backend/*`) to include units:
  - density, momentum, velocity, pressure, energy, viscosity, etc.
- Update state/coeff access helpers (e.g., `state_access.rs`, `coeff_expr.rs`) to return `TypedExpr` with the correct `UnitDim` and shape.
- Add a small suite of unit tests that intentionally try invalid operations and assert on the diagnostics (error messages should mention field names and operations).

### Phase 5: Incremental Kernel Migration + Parity Tests
**Deliverable:** major kernels use the typed DSL; generated WGSL remains stable.

- Migrate in the order of highest “string fragility” + biggest payoff:
  1. `compressible_assembly` (matrix-heavy, many repeated blocks)
  2. `coupled_assembly` (block matrix + reconstruction terms)
  3. `pressure_assembly`
  4. `compressible_flux_kt` + `compressible_gradients`
- Add a “golden WGSL” parity test for migrated kernels (string compare against committed generated files) until the migration is complete.
- Keep the 1D plot regressions as the runtime validation layer for behavior.

### Phase 6: Deprecate and Remove String-Based APIs
**Deliverable:** strings are no longer used for expression construction in codegen.

- Delete or gate `wgsl_dsl::expr(&str)` and friends.
- Remove now-obsolete string-format helpers once all kernels are migrated.
- Keep the expression parser only if it continues to add value for tests/debugging.
