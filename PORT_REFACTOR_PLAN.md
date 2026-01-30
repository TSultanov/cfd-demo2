# Port-Based Refactoring Implementation Plan

This document tracks the implementation of the port-based system to replace hardcoded string field names with statically-typed ports.

## Adjusted Plan Overview

Based on design discussions, the implementation follows these principles:
- **Automatic derivations** using proc-macros (`#[derive(ModulePorts)]`, `#[derive(PortSet)]`)
- **Gradual migration** starting from low-risk modules
- **End-state full replacement** of string-based codegen (temporary dual API during migration is OK; hard cutoff removes it)
- **PortRegistry created on-demand** during model initialization
- **Both compile-time and runtime validation**
- **Respect the codegen/IR boundary**: `crates/cfd2_codegen` must not depend on `src/solver/model/*` types
- **Direct WGSL generation** from port metadata
- **Hard cutoff** migration (remove old APIs after migration complete)

## Status Overview

| Phase | Status | Progress | Description |
|-------|--------|----------|-------------|
| 1. Macro Infrastructure | **In Progress** | ~30% | `cfd2_macros` exists; `ModulePorts` skeleton + `#[port(module = ...)]` parsing implemented; `PortSet` derive is still stubbed |
| 2. Core Port Runtime | **In Progress** | ~70% | `src/solver/model/ports/*` exists (ports + registry + tests); missing provider wiring + some runtime validation |
| 3. PortManifest + Module Integration | Pending | 0% | Decide IR-facing manifest shape + wire into model init/lowering (replacing/augmenting `ModuleManifest`) |
| 4. Low-Risk Migration | Pending | 0% | eos, generic_coupled modules |
| 5. Field-Access Migration | Pending | 0% | flux_module, rhie_chow |
| 6. Codegen Replacement | Pending | 0% | Replace string-based codegen |
| 7. Hard Cutoff | Pending | 0% | Remove deprecated APIs |

## Implementation Phases

### Phase 1: Macro Infrastructure (In Progress)

**Goal**: Create proc-macro crate and basic derive macros

**Done**:
- [x] Create `cfd2_macros` crate structure
- [x] Implement `#[derive(ModulePorts)]` with `#[port(module = "...")]` attribute parsing

**Remaining**:
- [ ] Implement `#[derive(PortSet)]` field parsing + codegen (currently `register()` is a no-op; `from_registry()` is `todo!()`)
- [ ] Add support for `#[param]` / `#[field]` / (optional) `#[buffer]` attributes and generate registrations/lookups
- [ ] Fix crate-path resolution (macros currently emit `::cfd2::...`; in-crate usage needs `extern crate self as cfd2;` or equivalent)
- [ ] Add compile-time validation (duplicates, unknown attrs, missing WGSL names, etc.)
- [ ] Add macro tests (e.g. `trybuild`) and at least one end-to-end “migrated module compiles” test

**Files Created**:
```
crates/cfd2_macros/
├── Cargo.toml
└── src/lib.rs          # Proc-macro implementations (PortSet, ModulePorts)
```

**Deliverables**:
- `#[derive(ModulePorts)]` - Derives `ModulePortsTrait` (currently delegates to `PortSetTrait`)
- `#[derive(PortSet)]` - Intended to derive parameter/field/buffer port collections (not yet usable)
- `#[port(module = "name")]` attribute parsing

### Phase 2: Core Port Runtime (In Progress)

**Goal**: Provide the runtime port types + registry that macros and modules will use

**Done**:
- [x] Define core traits (`ModulePortsTrait`, `PortSetTrait`, `PortValidationError`, `ModulePortRegistry`)
- [x] Implement port primitives (`FieldPort`, `ParamPort`, `BufferPort`) + WGSL helpers
- [x] Implement compile-time dimension tracking (`UnitDimension` + common dimensions)
- [x] Implement `PortRegistry` (field/param/buffer registration) + basic errors
- [x] Add unit tests for the port runtime pieces

**Remaining**:
- [ ] Wire `ParamPortProvider` / `FieldPortProvider` so `PortSetTrait::from_registry()` can be derived/implemented
- [ ] Add missing runtime validation (e.g. dimension mismatch checks on field registration; richer `PortRegistryError`)
- [ ] Decide and implement an IR-facing “port manifest” representation that can cross into `crates/cfd2_codegen` without violating the IR boundary

**Files Created**:
```
src/solver/model/ports/
├── buffer.rs
├── dimensions.rs
├── field.rs
├── mod.rs
├── params.rs
├── registry.rs
├── traits.rs
└── tests.rs
```

### Phase 3: PortManifest + Module Integration (Pending)

**Goal**: Replace/augment `ModuleManifest` so modules can declare ports and the solver can build/validate a port registry

**Tasks**:
- [ ] Decide where `PortManifest` lives (if `crates/cfd2_codegen` needs it directly, it must live in `cfd2_ir::solver::ir`)
- [ ] Define `PortManifest` (fields/params/buffers + WGSL names + units/kinds) as pure data
- [ ] Wire model initialization to build a `PortRegistry` on-demand (requires a strategy for getting `ModulePortsTrait` implementations from `ModelSpec.modules`)
- [ ] Integrate with existing build-time generation (WGSL emission + `named_params_registry` generation currently depends on `ModuleManifest.named_params`)
- [ ] Add validation helpers (“missing field”, “wrong kind”, “dimension mismatch”, etc.)
- [ ] Add `ports::prelude` for convenient imports for module authors

**Files to Create**:
```
crates/cfd2_ir/src/solver/ir/
└── ports.rs            # IR-facing PortManifest + port specs (re-export from ir/mod.rs)

src/solver/model/ports/
└── prelude.rs          # Convenient imports for module authors (and re-exports)
```

### Phase 4: Low-Risk Module Migration (Weeks 3-4)

**Goal**: Migrate modules with only parameter declarations

**Modules**:
- [ ] `eos` - EOS configuration (proof of concept)
- [ ] `generic_coupled_apply` - Simple module (1 kernel)
- [ ] `generic_coupled` - Complex but safe (16 params, 5 kernels)

**Process per module**:
1. Create new port-based module (e.g., `eos_ports.rs`)
2. Add `#[deprecated]` to old module function
3. Update model definitions
4. Run full test suite
5. Verify no regressions

**Example Migration**:
```rust
// BEFORE (string-based)
pub fn eos_module(eos: EosSpec) -> KernelBundleModule {
    let mut manifest = ModuleManifest::default();
    manifest.named_params = vec![
        NamedParamKey::Key("eos.gamma"),
        // ...
    ];
    KernelBundleModule { ... }
}

// AFTER (port-based)
#[derive(ModulePorts)]
#[port(module = "eos")]
pub struct EosModule {
    #[param(name = "eos.gamma", wgsl = "gamma")]
    pub gamma: ParamPort<F32, Dimensionless>,
}
```

### Phase 5: Field-Access Module Migration (Weeks 5-6)

**Goal**: Migrate modules with field lookups

**Modules**:
- [ ] `flux_module` - Gradient and flux computation
- [ ] `rhie_chow` - Pressure-velocity coupling

**Special considerations**:
- Integrate with `PortRegistry` for field validation
- Validate field coupling invariants at runtime
- Update WGSL generation to use ports

## Module Migration Playbook (All string-based field-name users)

This section enumerates every current module/file that does string-based state lookups
(`StateLayout::{field, offset_for, component_offset}` or equivalent) and provides explicit
migration steps to move that logic onto the port infrastructure.

### 0) Prerequisites (one-time, required before bulk migration)

- [ ] Finish Phase 1: `#[derive(PortSet)]` must be usable for real structs (registration + lookup)
- [ ] Finish Phase 3: define an IR-safe `PortManifest` (`cfd2_ir::solver::ir`) and a way to attach it to model modules (likely via `KernelBundleModule.manifest`)
- [x] Unify/upgrade the dimension system across crates (see “Type-Level Dimensions Migration” below). This must happen before we can rely on `FieldPort<Dim, Kind>` throughout the repo (including in IR/codegen).
- [ ] Add (or decide against) dimension enforcement policy in `PortRegistry`:
  - For “semantic” fields (p, rho, mu, grad_p, etc) enforce runtime dimension matches compile-time `UnitDimension`
  - For “system-driven” fields whose dimensions vary by model, provide a supported escape hatch:
    - Option A: an explicit “any-dimension” port type (no dimension check)
    - Option B: a separate untyped field port that tracks kind only

### 1) Common refactor recipe (apply to every module below)

1. **Inventory**: list every string-based lookup in the module (field name(s), expected kind, expected unit).
2. **Define ports**:
   - Add a `PortSet` struct describing required fields/params/buffers.
   - If the set is dynamic (depends on model/system), use a manifest “record list” (Vec of resolved port specs) rather than a fixed struct.
3. **Resolve once**:
   - Add a single resolver function (model init / build.rs time) that turns names into ports/offsets using `StateLayout` + invariants.
   - Store the results in `PortManifest` (IR-safe) and/or in a runtime `PortRegistry` cached on the solver/model.
4. **Refactor generators**: change WGSL/codegen functions to accept ports/offsets (no direct `StateLayout` lookups inside generators).
5. **Delete old lookups**: remove `offset_for/component_offset/field` usage from the migrated module.
6. **Verify**:
   - `cargo test`
   - Generated WGSL diff is empty (or manually reviewed and accepted)
   - OpenFOAM reference tests (when applicable)

### 2) Per-module migration steps (current repo inventory)

#### A) `src/solver/model/modules/rhie_chow.rs` (Rhie–Chow aux module)

**String lookups to remove** (today):
- `model.state_layout.field(dp_field)` / `offset_for(dp_field)`
- `component_offset("grad_<p>", c)` / `component_offset("grad_<p>_old", c)`
- `component_offset(momentum.name(), c)`

**Migration steps**:
- [ ] Define dimension aliases needed by this module in the canonical IR type-level dimensions module (see “Type-Level Dimensions Migration”):
  - `PressureGradient = DivDim<Pressure, Length>`
  - `DP = DivDim<MulDim<Volume, Time>, Mass>` (matches `si::D_P`)
- [ ] Create a port set / manifest entries for:
  - `dp_field`: scalar field (`DP`, Scalar) (name comes from module config)
  - `grad_p` + `grad_p_old`: vector2 fields (`PressureGradient`, Vector2) (names derived from inferred pressure field)
  - `momentum`: vector2 field (dimension is model-dependent; use the chosen escape hatch)
- [ ] Add a single resolver:
  - Infer `(momentum, pressure)` via `infer_unique_momentum_pressure_coupling_referencing_dp`
  - Build all ports (and derived `grad_*` names) from that inference
- [ ] Change the kernel generators in this module to take resolved ports/offsets:
  - Replace `StateLayout` lookups with `FieldPort::{offset,stride}` / `ComponentOffset::full_offset()` equivalents
  - Keep `rhie_chow_wgsl.rs` unchanged (it already accepts offsets/stride)
- [ ] Update/extend tests in `src/solver/model/modules/rhie_chow.rs`:
  - Keep the existing dp-field-name contract test
  - Add a test that the resolver fails with a clear error when `grad_<p>`/`grad_<p>_old` are missing or wrong kind

#### B) `src/solver/model/modules/flux_module_gradients_wgsl.rs` (Flux gradients stage)

**String lookups to remove** (today):
- Scans `StateLayout` for `grad_*` targets and calls `layout.field(...)` / `component_offset(...)`.

**Migration steps**:
- [ ] Move “gradient target discovery” out of WGSL generation:
  - During model validation (or during port registry build), compute an explicit list of gradient targets:
    - base field name + (optional) base component
    - grad field name (vector2)
    - resolved offsets for base/grad components
    - bc unknown offset (from `FluxLayout`)
- [ ] Define a compact IR-safe record type (in `cfd2_ir::solver::ir::ports`) for gradient targets and attach it to the module’s `PortManifest`.
- [ ] Change `generate_flux_module_gradients_wgsl(...)` to accept the resolved target records instead of a `StateLayout`.
- [ ] Delete all `layout.field/component_offset` calls from `flux_module_gradients_wgsl.rs`.
- [ ] Add/keep tests to ensure:
  - The resolved target list matches the legacy “scan `grad_` fields” behavior
  - WGSL output remains identical

#### C) `src/solver/model/modules/flux_module_wgsl.rs` (Flux kernel WGSL generation)

**String lookups to remove** (today):
- `state_layout.offset_for(name)` and `state_layout.component_offset(base, c)` inside helper routines.
- Various “does this field/component exist?” checks using `StateLayout`.

**Migration steps**:
- [ ] Add a lowering pass that resolves every state-field reference used by flux codegen into a “resolved state slot”:
  - Traverse `FluxModuleKernelSpec` + any `PrimitiveExpr` used by flux evaluation
  - Collect all referenced field names (including component-suffixed names like `<field>_x`)
  - Resolve them once via `StateLayout` into `(stride, offset[, component_count], unit)` records
- [ ] Store this resolved mapping in the module’s `PortManifest` (IR-safe) and pass it into `generate_flux_module_wgsl*`.
- [ ] Replace all `StateLayout` lookups in `flux_module_wgsl.rs` with reads from the resolved mapping / ports.
- [ ] Keep the public API surface stable:
  - The module can still accept a `FluxModuleSpec` by value
  - Internally, it should pre-resolve ports before WGSL generation
- [ ] Confirm WGSL output is unchanged against committed generated shaders.

#### D) `src/solver/gpu/lowering/programs/generic_coupled.rs` (GPU lowering: generic coupled)

**String lookups to remove** (today):
- Uses `layout.offset_for(name)` / `layout.component_offset(name, comp)` while building program bindings for unknowns.

**Migration steps**:
- [ ] Pre-resolve per-model “unknown → state slot” mapping at model build time:
  - For each equation target field in the system, compute offsets for each component
  - Store as an IR-safe mapping (e.g. `Vec<StateSlot>` ordered by unknown index) in `PortManifest`
- [ ] Update lowering to use the mapping instead of calling `StateLayout` directly.
- [ ] Add a regression test that the mapping order matches the legacy lowering behavior.

#### E) `src/solver/model/helpers/solver_ext.rs` (Host-side helpers)

**String lookups to remove** (today):
- Repeated `state_layout.offset_for("rho")`, `"rho_u"`, `"rho_e"`, `"p"`, `"T"`, `"u"/"U"` when seeding initial conditions.

**Migration steps**:
- [ ] Define “canonical field port sets” for the helper surfaces:
  - Compressible: `rho`, `rho_u` (vec2), `rho_e` (+ optional `p`, `T`, `u|U`)
  - Incompressible: `U|u`, `p`, (and any optional diagnostics fields used by stats helpers)
- [ ] Resolve these ports once per solver/model and cache them (avoid per-call `StateLayout` scanning).
- [ ] Update helper implementations to use cached ports for offsets/stride.
- [ ] Leave boundary APIs string-keyed for now, but centralize the names as constants next to the port set definitions.

#### F) `src/solver/model/definitions/incompressible_momentum.rs` (Model construction)

**String lookups to remove** (today):
- `layout.component_offset("U", c)` and `layout.offset_for("p")` used as early validation.

**Migration steps**:
- [ ] Replace ad-hoc layout checks with port resolution:
  - Build `FieldPort`s for `U` (vec2) and `p` (scalar) and fail early if missing/wrong kind
- [ ] Prefer using the same ports later for any host-side indexing needs (avoid duplicating offset logic).

#### G) `src/solver/model/definitions.rs` (Model validation and invariants)

**String lookups to migrate** (today):
- Flux-module gradients validation scans `grad_*` names and checks base existence via `state_layout.field(...)`.
- Invariant validation checks required fields via `state_layout.field(...)` / `component_offset(...)`.

**Migration steps**:
- [ ] Make `PortManifest` the single source of truth for “required fields” and “gradient targets”.
- [ ] Convert validation to:
  - Resolve all required ports once, returning structured `PortValidationError`
  - Avoid repeated `StateLayout` probing across multiple validation passes

#### H) `src/ui/app.rs` (UI-only state inspection)

**String lookups to migrate** (today):
- `layout.offset_for("U"/"u")` and `layout.offset_for("p")` for plotting/inspection.

**Migration steps** (optional; do after core solver is migrated):
- [ ] Expose a small, stable “UI port set” from the solver/model (optional ports for common fields)
- [ ] Update UI to use that port set instead of raw `StateLayout` probing

#### I) `crates/cfd2_codegen/src/solver/codegen/*` (Codegen helper modules)

These are not “model modules”, but they are the largest remaining concentration of string-based
state lookups and will block completing Phase 6 unless migrated.

**Migration steps**:
- [ ] Introduce an IR-safe “resolved state slot” type (offset + stride + kind/unit metadata as needed).
- [ ] Update `state_access.rs` to accept resolved slots instead of `(StateLayout, field_name)`.
- [ ] Update callers (`generic_coupled_kernels`, `unified_assembly`, `primitive_expr`, `coeff_expr`) to use slots/ports provided by model/module manifests.

## Type-Level Dimensions Migration (entire codebase, including IR)

Today, the codebase uses **runtime** dimensional analysis via `cfd2_ir::solver::units::UnitDim`:
- IR/model backend validates term units at runtime: `crates/cfd2_ir/src/solver/model/backend/ast.rs` (`EquationSystem::validate_units`)
- Codegen DSL checks units at runtime: `crates/cfd2_codegen/src/solver/codegen/dsl/expr.rs` (`TypedExpr.unit: UnitDim`)
- Port infrastructure previously defined a separate **type-level** dimension system, but it has now been unified:
  - Canonical type-level dimensions live in `crates/cfd2_ir/src/solver/dimensions.rs`
  - Ports re-export the canonical system via `src/solver/model/ports/dimensions.rs`

**Goal**: make the *type-level* dimension system the canonical source of truth across **IR + codegen + ports**,
while keeping `UnitDim` as the runtime/serialization/debug representation.

### 1) Create a canonical type-level dimension system in `cfd2_ir`

- [x] Add a new IR module that defines type-level dimensions with **rational exponents** (to match `UnitDim`):
  - Location: `crates/cfd2_ir/src/solver/dimensions.rs` (re-export via `crates/cfd2_ir/src/solver/mod.rs`)
  - Uses const-generics to represent rationals as `(NUM, DEN)` tuples for each base dimension exponent:
    - Mass (kg), Length (m), Time (s), Temperature (K)
  - Provides type constructors:
    - `MulDim<A, B>`, `DivDim<A, B>`, `PowDim<A, NUM, DEN>`, `SqrtDim<A>`
  - Provides a `to_runtime() -> UnitDim` mapping and round-trip tests
  - Added `UnitDim::from_rational()` const fn constructor for building `UnitDim` from rational exponents
- [x] Define typed equivalents for the existing `si::*` dimensions (e.g. `Pressure`, `Velocity`, `Density`, `D_P`, `PressureGradient`) in that module.
- [x] Tests verify that type-level dimensions match `units::si::*` runtime constants exactly (see `derived_dimensions_match_si_constants` test)

### 2) Make ports reuse the canonical IR dimension types

- [x] Refactor `src/solver/model/ports/dimensions.rs` to re-export `cfd2_ir::solver::dimensions::*` (removed duplicate impls).
- [x] Created `src/solver/dimensions.rs` re-export module for convenient access from main crate.
- [x] `FieldPort` / `ParamPort` / `PortRegistry` already use the `UnitDimension` trait via the re-export.
- [x] Added regression tests that port dimension types match `crate::solver::units::si::*` exactly (see `port_specific_dimensions_match_si` test).

**Note**: The old ports dimension system used `i8` exponents and had a buggy `SqrtDim` definition (`PowDim<A, 1>` instead of `A^(1/2)`). The new canonical system uses rational `(i32, i32)` exponents and correctly implements `SqrtDim<A> = PowDim<A, 1, 2>`.

### 3) Add a typed IR builder layer (units checked by Rust types)

This avoids making the existing untyped IR (`FieldRef { unit: UnitDim }`) generic-heavy, while still moving unit correctness to types.

- [x] Add a typed "builder/front-end" in IR that erases into the existing IR structs:
  - Location: `crates/cfd2_ir/src/solver/model/backend/typed_ast.rs`
  - Provide typed wrappers:
    - `TypedFieldRef<D, K>` / `TypedFluxRef<D, K>` (dimension `D`, kind `K`)
    - `TypedCoeff<D>` for coefficients with compile-time dimension checking
    - `TypedTerm<D>` and `TypedTermSum<D>` for terms with integrated unit checking
    - `TypedEquation<D, K>` and `TypedEquationSystem` for equation building
  - Provide typed term constructors (`typed_fvm::*`, `typed_fvc::*`) that compute integrated units in the type system
  - `Add` trait implementation ensures terms can only be added when integrated units match (compile-time check)
  - Keep `EquationSystem::validate_units()` as a runtime backstop for untyped callers (and during migration)
- [x] Fix typed-IR builder gaps found while migrating models:
  - `div()` accepts independent flux/field kinds (scalar face flux with vector unknowns)
  - Explicit `source` integrated unit matches runtime semantics (`coeff * Volume`)
  - `mag_sqr()` returns squared units
- [ ] Canonicalize type-level dimensions so equivalent exponent vectors become the same Rust type:
  - Introduce a canonical dimension carrier type (e.g. `Dim<...exponents...>`)
  - Make `MulDim`/`DivDim`/`PowDim` (or a `Canonical<D>` helper) resolve to that canonical type
  - Add compile-time regression tests for “same units, different expressions” (e.g. ddt vs div force balance)
- [ ] Migrate model constructors in `src/solver/model/definitions/*` to use the typed builder APIs (still producing the same `EquationSystem` as output).
  - **Current**: model definitions still use the untyped builder + `EquationSystem::validate_units()` assertions as a runtime backstop.
- [ ] Update `crates/cfd2_codegen/src/solver/codegen/ir.rs` (`lower_system`) to:
  - Prefer typed-built systems (no unit errors expected)
  - Keep `validate_units()` for any remaining untyped system construction paths
- [ ] Update `crates/cfd2_ir/src/solver/model/backend/scheme_expansion.rs` (`expand_schemes`) similarly:
  - Prefer typed-built systems
  - Keep `validate_units()` as a backstop during migration

### 4) Port the codegen DSL unit checks to type-level dimensions

- [ ] Introduce a typed variant of `TypedExpr` parameterized by dimension type:
  - `TypedExpr<D>` where `D: UnitDimension`
  - Addition/subtraction only type-checks for identical `D`
  - Multiplication/division returns `TypedExpr<MulDim<...>>` / `TypedExpr<DivDim<...>>`
  - `sqrt()` returns `TypedExpr<SqrtDim<D>>`
  - Keep an ergonomic escape hatch for genuinely dynamic units (explicitly marked, avoids silent “any unit”)
- [ ] Incrementally migrate codegen helpers to the typed DSL, starting with the leaf utilities:
  - `crates/cfd2_codegen/src/solver/codegen/state_access.rs`
  - `crates/cfd2_codegen/src/solver/codegen/primitive_expr.rs`
  - `crates/cfd2_codegen/src/solver/codegen/coeff_expr.rs`
- [ ] Remove (or quarantine) the runtime unit mismatch checks once all call sites are typed.

### 5) End-state success criteria for dimensions

- [ ] All dimensional invariants are enforced by Rust types wherever construction happens in Rust (models, IR builders, port definitions, codegen DSL).
- [ ] Runtime `UnitDim` remains only for:
  - Serialization/debug output
  - Reading units off runtime `StateLayout` metadata
  - Validating runtime layouts/configurations against typed expectations
- [ ] No duplicate dimension systems remain (single canonical definition in `cfd2_ir`).

### Phase 6: Codegen Replacement (Weeks 7-8)

**Goal**: Replace string-based codegen with port-based

**Files to update**:
- [ ] `crates/cfd2_codegen/src/solver/codegen/state_access.rs`
- [ ] `crates/cfd2_codegen/src/solver/codegen/coeff_expr.rs`
- [ ] `crates/cfd2_codegen/src/solver/codegen/primitive_expr.rs`
- [ ] All `*_wgsl.rs` files in modules

**Changes**:
- Add IR-level (boundary-safe) field/param access helpers that accept port metadata (e.g. offsets/stride or `PortManifest` entries) rather than string field names
- Keep string-based helpers temporarily (mark as `#[deprecated]` during migration)
- Update kernel generators to pass port metadata instead of raw strings (without importing `src/solver/model/*` into codegen)

### Phase 7: Hard Cutoff (Week 9)

**Goal**: Remove all legacy code

- [ ] Delete deprecated module functions
- [ ] Remove `NamedParamKey` enum
- [ ] Remove old `ModuleManifest`
- [ ] Delete string-based codegen functions
- [ ] Update documentation
- [ ] Breaking change commit with migration guide

## Design Decisions (Confirmed)

### 1. Proc-Macro Architecture

```rust
// crates/cfd2_macros/src/lib.rs
#[proc_macro_derive(ModulePorts, attributes(port, param, field, buffer))]
pub fn derive_module_ports(input: TokenStream) -> TokenStream {
    // Generate:
    // - impl ModulePorts
    // - Port registration logic
    // - Validation code
}
```

### 2. Module Declaration Pattern

```rust
#[derive(ModulePorts)]
#[port(module = "eos")]
#[port(eos_provider)]  // Marks as EOS provider
pub struct EosModule {
    // Parameters with compile-time type/dimension checking
    #[param(name = "eos.gamma", wgsl = "gamma")]
    pub gamma: ParamPort<F32, Dimensionless>,
    
    // Fields with automatic validation
    #[field(name = "p", kind = Scalar)]
    pub pressure: FieldPort<Pressure, Scalar>,
}
```

### 3. On-Demand PortRegistry

```rust
impl ModelSpec {
    pub fn create_port_registry(&self) -> Result<PortRegistry, PortRegistryError> {
        let mut registry = PortRegistry::new(self.state_layout.clone());
        // NOTE: `ModelSpec.modules` is currently `Vec<KernelBundleModule>`.
        // Before this can work, decide how port-declaring modules are stored:
        // - Option A: introduce an object-safe port trait for storage/iteration (e.g. `ModelModulePorts`)
        //   (Note: `ModulePortsTrait` itself is not directly usable as a heterogeneous trait object due to its `PortSet` assoc type.)
        // - Option B: keep `KernelBundleModule` but attach a `PortManifest` to it (or its manifest)
        Ok(registry)
    }
}
```

### 4. Validation Strategy

**Compile-time** (via macros):
- ParamPort<T, D> type checking
- Dimension compatibility
- Field kind verification

**Runtime** (via PortRegistry):
- Field existence in StateLayout
- Field kind matching
- Parameter existence/type/dimension checks

### 5. Migration Compatibility

During migration, old and new APIs coexist:

```rust
// Old API (deprecated during migration)
#[deprecated = "Use EosModule with ModulePorts"]
pub fn eos_module(eos: EosSpec) -> KernelBundleModule { ... }

// New API
#[derive(ModulePorts)]
pub struct EosModule { ... }

// Both implement ModelModule trait
```

**Hard cutoff** at end of Phase 6 removes all deprecated code.

## Testing Strategy

**For each phase**:
- Unit tests for macro-generated code
- Integration tests with existing models
- OpenFOAM reference tests (if applicable)
- WGSL generation verification

**Test files**:
```
src/solver/model/ports/*.rs
crates/cfd2_macros/tests/*        # (planned) trybuild + compile-fail tests
```

## Success Criteria

- [ ] `cargo test` passes
- [ ] Zero string-based field lookups in migrated modules
- [ ] Compile-time dimension checking works
- [ ] Runtime validation catches invalid configs
- [ ] Generated WGSL diffs are empty (or equivalent check passes)
- [ ] Hard cutoff removes all deprecated APIs

## Notes

- **Breaking changes** documented in migration guide
- **No feature flags** - hard cutoff approach
- **Documentation** updated after each phase
- **CHANGELOG** updated with breaking changes

## Current Status

As of **2026-01-30**:
- Canonical type-level dimensions (rational exponents) live in `crates/cfd2_ir/src/solver/dimensions.rs` and are re-exported through the main crate for ports
- A typed IR builder layer exists at `crates/cfd2_ir/src/solver/model/backend/typed_ast.rs` for compile-time unit checking when constructing `EquationSystem`s in Rust (currently best-effort; complex systems still rely on runtime `validate_units()` until dimension types are canonicalized)
- Port runtime types + registry exist under `src/solver/model/ports/*`
- Proc-macro scaffolding exists, but `PortSet` is not implemented yet (and `ModulePorts` delegates to it)
- No model modules have been migrated to ports yet
- `PortManifest` does not exist yet; `ModuleManifest`/string-based codegen remains the source of truth

**Next (recommended)**:
- Type-Level Dimensions Migration: canonicalize type-level dimensions so equivalent unit expressions become the same Rust type, then re-attempt typed model definitions.
- Phase 1: finish derive macros (`PortSet`) so modules can be migrated with minimal boilerplate.
- Phase 3: define `PortManifest` + module integration, then migrate `eos`.

**Deliverables Ready**:
- ✅ `src/solver/model/ports/*` runtime primitives + tests
- ✅ Core traits (`ModulePortsTrait`, `PortSetTrait`, etc.)
- ✅ Typed IR builder layer for compile-time unit checking (`crates/cfd2_ir/src/solver/model/backend/typed_ast.rs`)
- ⚠️ `cfd2_macros` derive macros exist but are not yet usable for real module migrations

**Ready for**: Completing derive macros + deciding the module/container integration strategy
