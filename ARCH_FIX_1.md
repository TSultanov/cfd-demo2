# ARCH_FIX_1: Resolving "The Façade of Type Safety (Ports & Dimensions)"

## Problem Summary

Architecture Review §1 identifies three interrelated problems in the dimension/port/expression system:

1. **Type Erasure & `AnyDimension` Escape Hatch** — A complex const-generic type-level dimension system (`Dim<M_NUM, M_DEN, …>`, the `UnitDimension` trait) is maintained across ~1,500 lines in `cfd2_ir::dimensions`, `cfd2_ir::units`, and `src/solver/model/ports/dimensions.rs`. At the GPU boundary, all of this is erased into runtime `UnitDim` structs stored in `ResolvedStateSlotSpec`. The `AnyDimension` sentinel type (exponents `(1000,1)`) exists solely to bypass checks that the type system was supposed to enforce, and is used in at least `flux_module.rs` and `rhie_chow.rs`.

2. **Runtime Panics for Compile-Time Concepts** — The `#[derive(PortSet)]` macro generates code that calls `registry.register_field::<D, K>(…)`, which validates dimensions at runtime inside `PortRegistry`. Mismatches produce `PortRegistryError` variants that surface as panics during solver init rather than compile errors.

3. **Redundant Expression Trees** — `PrimitiveExpr` (in `cfd2_ir::flux`) and `Expr` (in `cfd2_ir::ast::expr`) are structurally identical expression trees (literals, field refs, binary ops, unary ops). `PrimitiveExpr` exists only for EOS/thermodynamic derivations and requires a separate lowering pass (`lower_primitive_expr_dyn`) to convert into the codegen `Expr`.

---

## Design Principles

- **Runtime-first dimension validation.** The GPU compute-graph must be assembled at runtime (field names, offsets, strides are not known until `UnifiedSolver::new`). Compile-time dimension types add complexity without preventing the errors that actually occur.
- **Single expression tree.** One canonical AST should serve model definitions and codegen.
- **Incremental migration.** Each phase must leave the project in a compilable and testable state. OpenFOAM reference tests must not regress.

---

## Phase 1: Unify the Expression Trees (`PrimitiveExpr` → `Expr`)

**Goal:** Eliminate `PrimitiveExpr` and its lowering pass entirely.

### 1.1 Audit all `PrimitiveExpr` usage sites

| File | Usage |
|------|-------|
| `crates/cfd2_ir/src/flux.rs` | Definition of `PrimitiveExpr` enum (50 lines) |
| `crates/cfd2_codegen/src/solver/codegen/primitive_expr.rs` | `lower_primitive_expr` / `lower_primitive_expr_dyn` (293 lines) |
| `crates/cfd2_codegen/src/solver/codegen/generic_coupled_kernels.rs` | Calls `lower_primitive_expr` |
| `crates/cfd2_codegen/src/solver/mod.rs` | Re-exports |
| `src/solver/model/primitives.rs` | Builds `PrimitiveExpr` trees for EOS |
| `src/solver/model/kernel.rs` | References `PrimitiveExpr` |
| `src/solver/model/modules/flux_module_wgsl.rs` | Calls lowering |
| `src/solver/model/modules/flux_module_resolver_pass.rs` | Passes `PrimitiveExpr` through |
| `src/solver/model/modules/flux_module.rs` | Stores `PrimitiveExpr` |

### 1.2 Migration steps

1. **Add builder helpers to `Expr`** to match `PrimitiveExpr` ergonomics:
   - `Expr::field_ref(name: &str) -> Expr` — wraps `Expr::ident(name)` (already exists)
   - `Expr::sqrt(self) -> Expr` — wraps `Expr::call_named("sqrt", vec![self])`
   - Verify existing `From<f32>` for `Expr` covers the `Literal` case.

2. **Replace `PrimitiveExpr` with `Expr` at each model definition site** (in `src/solver/model/primitives.rs`). EOS definitions will build `Expr` trees directly. Example migration:
   ```rust
   // Before:
   PrimitiveExpr::Div(
       Box::new(PrimitiveExpr::Field("rho_u_x".into())),
       Box::new(PrimitiveExpr::Field("rho".into())),
   )
   // After:
   Expr::ident("rho_u_x") / Expr::ident("rho")
   ```

3. **Replace `lower_primitive_expr` callsites** with a single `resolve_field_refs(expr: &Expr, slots: &ResolvedStateSlotsSpec, cell_idx: Expr, state_array: &str) -> Expr` function that walks the `Expr` tree and replaces bare `Ident` nodes matching field names with the correct `state[idx * stride + offset]` index expression. This is a simpler tree walk than the current lowering because `Expr` is already the output type.

4. **Preserve runtime unit tracking** by keeping the `DynExpr` wrapper in the DSL layer. The new `resolve_field_refs` will also produce a `DynExpr` with runtime `UnitDim` propagation (functionally identical to what `lower_primitive_expr_dyn` does today).

5. **Delete `PrimitiveExpr`** from `cfd2_ir::flux` and `lower_primitive_expr*` from `cfd2_codegen::solver::codegen::primitive_expr`.

### 1.3 Verification

- All existing `primitive_expr` tests are ported to use `Expr`-based equivalents.
- `cargo test --workspace` passes.
- OpenFOAM reference tests: run before/after, verify no metric regression.

### Estimated diff

~350 lines removed (PrimitiveExpr + lowering), ~80 lines added (resolve helper + builder sugar).

---

## Phase 2: Consolidate to Runtime-Only Dimension Validation

**Goal:** Remove the type-level `UnitDimension` trait, `Dim<…>` struct, and all const-generic dimension machinery. Replace with `UnitDim` (the runtime struct that already exists and works).

### 2.1 New runtime-checked port API

Replace the generic-parameterized registration API:
```rust
// Before — dimension is a type parameter:
registry.register_field::<Velocity, Vector2>("U")?;

// After — dimension is a runtime value:
registry.register_field("U", FieldKind::Vector2, si::VELOCITY)?;
```

Concrete changes to `PortRegistry`:

```rust
impl PortRegistry {
    pub fn register_field(
        &mut self,
        name: &str,
        kind: FieldKind,
        expected_unit: UnitDim,
    ) -> Result<FieldPort, PortRegistryError> { … }

    pub fn register_param(
        &mut self,
        key: &'static str,
        wgsl_field: &'static str,
        param_type: ParamTypeKind,
        expected_unit: UnitDim,
    ) -> Result<ParamPort, PortRegistryError> { … }
}
```

### 2.2 Simplify port types

Remove generic dimension/kind parameters from `FieldPort` and `ParamPort`:

```rust
// Before:
pub struct FieldPort<D: UnitDimension, K: FieldKind> { … }

// After:
pub struct FieldPort {
    id: PortId,
    name: &'static str,
    offset: u32,
    stride: u32,
    kind: FieldKind,        // runtime enum, not type param
    unit: UnitDim,           // runtime struct, not type param
}
```

The `DimensionalPort` trait is removed. `unit()` becomes a plain method.

### 2.3 Handle `AnyDimension`

Replace the sentinel type with an explicit `Option<UnitDim>`:

```rust
// Callers that previously used AnyDimension:
registry.register_field("rho_u", FieldKind::Vector2, None)?;  // skip unit check

// Callers that know the unit:
registry.register_field("U", FieldKind::Vector2, Some(si::VELOCITY))?;
```

Internally, `None` means "skip validation" — the same semantics as `AnyDimension` but without the magic sentinel value `(1000,1)`.

### 2.4 Update `#[derive(PortSet)]` macro

The macro currently extracts `D` and `K` from `FieldPort<D, K>`. After simplification:

```rust
// Before:
#[derive(PortSet)]
struct MyPorts {
    #[field(name = "U")]
    u: FieldPort<Velocity, Vector2>,
    #[param(name = "dt", wgsl = "dt")]
    dt: ParamPort<F32, Time>,
}

// After:
#[derive(PortSet)]
struct MyPorts {
    #[field(name = "U", kind = "vector2", unit = "si::VELOCITY")]
    u: FieldPort,
    #[param(name = "dt", wgsl = "dt", type = "f32", unit = "si::TIME")]
    dt: ParamPort,
}
```

The macro generates `registry.register_field("U", FieldKind::Vector2, Some(si::VELOCITY))` — all runtime.

### 2.5 Update model definitions

All three model definition files need updating:

| File | Changes |
|------|---------|
| `definitions/incompressible_momentum.rs` | Replace `TypedFieldRef::<Velocity, Vector2>` → `FieldRef::new("U", FieldKind::Vector2, si::VELOCITY)` |
| `definitions/compressible.rs` | Same pattern; ~40 type-level dimension references |
| `definitions/generic_diffusion_demo.rs` | Simplest case, fewest fields |

The typed AST wrapper layer (`cfd2_ir::equation::typed_ast`) has two migration options:

- **Option A (Recommended):** Keep `TypedFieldRef<D, K>` as a thin convenience wrapper that erases to `FieldRef` + `UnitDim` at construction time. No const-generic dimension traits needed — `D` becomes a marker type whose only role is documentation/readability. The `UnitDimension` trait is replaced by a simple `fn unit() -> UnitDim` method. This gives model authors readable code (`TypedFieldRef::<Velocity, Vector2>`) while all checking is runtime.

- **Option B:** Remove `TypedFieldRef` entirely; pass `UnitDim` constants directly. More mechanical but loses readability.

Recommendation: **Option A** for model definition ergonomics, but with `D` no longer enforcing anything at the type level — it's purely a named constant holder.

### 2.6 Files deleted or substantially simplified

| File | Action |
|------|--------|
| `crates/cfd2_ir/src/dimensions.rs` (511 lines) | **Delete entirely**. Replace all imports with `use cfd2_ir::units::{UnitDim, si};` |
| `src/solver/model/ports/dimensions.rs` (127 lines) | **Delete entirely**. `AnyDimension` replaced by `Option<UnitDim>`. |
| `src/solver/model/ports/field.rs` (383 lines) | Simplify — remove `D`/`K` generics (~100 lines removed) |
| `src/solver/model/ports/params.rs` (506 lines) | Simplify — remove `D` generic (~80 lines removed) |
| `src/solver/model/ports/registry.rs` (2109 lines) | Simplify registrations, remove `TypeId::of::<AnyDimension>()` checks (~200 lines removed) |
| `crates/cfd2_macros/src/lib.rs` (697 lines) | Simplify — no longer extracts type generics for dim/kind (~150 lines simplified) |
| `crates/cfd2_ir/src/equation/typed_ast.rs` (1041 lines) | Under Option A: keep but simplify trait bounds. Under Option B: delete. |

### 2.7 Keep `UnitDim` and `si::*` constants

The runtime `UnitDim` struct (`crates/cfd2_ir/src/units.rs`, 388 lines) is well-designed and stays as-is. The `si` module of named constants (`si::VELOCITY`, `si::PRESSURE`, etc.) remains the single source of truth for dimension values. Callers simply pass these constants instead of type parameters.

### 2.8 Validation point

The key invariant — "dimensions are checked when the solver graph is assembled" — is preserved. `PortRegistry` continues to validate `expected_unit` against `StateLayout` field units at registration time. The difference is that errors are `Result<_, PortRegistryError>` (already the case) rather than compile errors (which never truly worked due to `AnyDimension`).

### 2.9 Verification

- `cargo test --workspace` passes.
- All trybuild UI tests for `PortSet` macro updated.
- OpenFOAM reference tests: run before/after, verify no metric regression.

### Estimated diff

~1,200 lines removed, ~200 lines added/changed.

---

## Phase 3: Improve Error Reporting at the GPU Boundary

**Goal:** Ensure dimension/type mismatches produce clear, actionable errors instead of panics.

### 3.1 Replace panics with `Result` propagation in init path

Currently `PortRegistryError` is returned as `Result`, but many callers in `unified_solver.rs` and the init modules call `.unwrap()` or `.expect()`. Audit and replace with proper `?` propagation:

```rust
// Before (unified_solver.rs):
let port = registry.register_field::<Velocity, Vector2>("U").unwrap();

// After:
let port = registry.register_field("U", FieldKind::Vector2, Some(si::VELOCITY))?;
```

### 3.2 Structured error type for solver init

Create a top-level `SolverInitError` enum that wraps:
- `PortRegistryError` (port/dimension issues)
- `WgpuError` (GPU device issues)
- `MeshError` (mesh incompatibility)

The `UnifiedSolver::new()` signature changes from returning the solver directly (with panics) to `Result<UnifiedSolver, SolverInitError>`.

### 3.3 UI-safe initialization

For the `ui` feature (egui), `SolverInitError` is displayed in a dialog rather than crashing. This is a UI-layer change and doesn't affect the core solver.

### 3.4 Verification

- Manual testing with invalid mesh/solver combos to confirm error messages.
- OpenFOAM reference tests: no regression.

---

## Phase 4: Clean Up Runtime Unit Tracking in Codegen DSL

**Goal:** Ensure the `DynExpr` / `DslType` layer in codegen remains consistent after Phases 1–3.

### 4.1 Simplify `DynExpr` imports

After removing `cfd2_ir::dimensions`, all `DynExpr` operations import `UnitDim` from `cfd2_ir::units` directly. The `DslType` / `DslError` machinery in `crates/cfd2_codegen/src/solver/codegen/dsl/expr.rs` remains — it provides valuable runtime unit checking during WGSL generation.

### 4.2 Remove dead `UnitDimension` trait references from codegen

Grep for remaining `UnitDimension` trait bounds in codegen paths (`state_access.rs`, `coeff_expr.rs`, `generic_coupled_kernels.rs`) and replace with `UnitDim` values.

### 4.3 Verification

- `cargo test --workspace`
- OpenFOAM reference tests: final comparison against Phase 1 baseline.

---

## Migration Order & Dependencies

```
Phase 1 (Expr unification)
    │
    ▼
Phase 2 (Runtime dimensions)
    │
    ├──► Phase 3 (Error handling)  [can run in parallel with Phase 4]
    │
    └──► Phase 4 (Codegen cleanup) [can run in parallel with Phase 3]
```

Phases 1 and 2 are sequential (Phase 2 depends on the simpler expression tree from Phase 1). Phases 3 and 4 are independent and can proceed in parallel after Phase 2.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| OpenFOAM test regression | Low | These changes are structural/compile-time only; runtime numerical behavior is unchanged. Run before/after comparison per `AGENTS.md`. |
| Breakage in `#[derive(PortSet)]` macro | Medium | The macro needs non-trivial rewrite in Phase 2. Trybuild UI tests will catch regressions. |
| Model definition churn | Medium | Three model files + boundary condition helpers need updating. Mechanical but tedious. Test coverage exists. |
| `typed_ast.rs` compatibility | Low | Option A preserves the API surface; callers barely change. |
| Merge conflicts with concurrent work | Medium | Phases are designed to be individually mergeable. Each phase ends in a green CI state. |

---

## Line Count Impact (Estimated)

| Category | Lines Removed | Lines Added | Net |
|----------|--------------|-------------|-----|
| Phase 1 (PrimitiveExpr) | ~350 | ~80 | **−270** |
| Phase 2 (Dimensions) | ~1,200 | ~200 | **−1,000** |
| Phase 3 (Error handling) | ~50 | ~150 | **+100** |
| Phase 4 (Codegen cleanup) | ~100 | ~30 | **−70** |
| **Total** | **~1,700** | **~460** | **−1,240** |

---

## Success Criteria

1. `crates/cfd2_ir/src/dimensions.rs` is deleted.
2. `AnyDimension` sentinel type is deleted; replaced by `Option<UnitDim>`.
3. `PrimitiveExpr` enum is deleted; all EOS derivations use `Expr`.
4. `FieldPort` and `ParamPort` have no type-level dimension parameters.
5. All dimension validation happens at runtime in `PortRegistry` via `UnitDim` equality checks.
6. `cargo test --workspace` is green.
7. OpenFOAM reference test metrics are unchanged or improved vs. baseline.
