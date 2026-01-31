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
| 1. Macro Infrastructure | **Mostly Done** | ~90% | `cfd2_macros` has `PortSet` derive (param/field/buffer) + trybuild tests; `ModulePorts` exists but needs integration strategy (how/when modules materialize port sets) |
| 2. Core Port Runtime | **Done** | 100% | `src/solver/model/ports/*` exists (ports + registry + tests); registry supports idempotent registration + conflict errors; IR-safe manifests can be registered dynamically; dimension enforcement policy is implemented (with an explicit escape hatch for dynamic dimensions) |
| 3. PortManifest + Module Integration | **Done** | 100% | IR-safe `PortManifest` is wired through runtime + build-time; named-param allowlisting consumes `port_manifest.params`; WGSL `Constants` generation consumes manifest params (no ad-hoc string lists) |
| 4. Low-Risk Migration | **Done** | 100% | `eos` + `generic_coupled` publish `PortManifest` for uniform params; `generic_coupled_apply` wrapper removed (kernel remains in `generic_coupled`) |
| 5. Field-Access Migration | **In Progress** | ~90% | `flux_module_gradients_wgsl` no longer scans `StateLayout` during WGSL generation (targets pre-resolved); `flux_module_wgsl` consumes pre-resolved `ResolvedStateSlotsSpec` from `PortManifest` (no `StateLayout` probing) and has WGSL goldens; `rhie_chow` runtime generators migrated to PortRegistry (build-script fallback remains); `generic_coupled` GPU lowering migrated to use pre-resolved unknown-to-state mapping via PortRegistry (runtime path works correctly with all StateLayout fields pre-registered; tests cover the runtime resolver); flux-module gradient targets now stored in IR-safe `PortManifest` |
| 6. Codegen Replacement | Pending | 0% | Replace string-based codegen |
| 7. Hard Cutoff | Pending | 0% | Remove deprecated APIs |

## Implementation Phases

### Phase 1: Macro Infrastructure (Mostly Done)

**Goal**: Create proc-macro crate and basic derive macros

**Done**:
- [x] Create `cfd2_macros` crate structure
- [x] Implement `#[derive(ModulePorts)]` with `#[port(module = "...")]` attribute parsing

**Remaining**:
- [x] Implement `#[derive(PortSet)]` field parsing + codegen with idempotent registration
- [x] Add support for `#[param]` / `#[field]` / `#[buffer]` attributes and generate registrations/lookups
- [x] Fix crate-path resolution (added `extern crate self as cfd2;` to src/lib.rs)
- [x] Add compile-time validation (duplicates, unknown attrs, missing WGSL names, etc.)
- [x] Add macro tests with `trybuild` (passing and failing cases)
- [x] Add at least one end-to-end “migrated module compiles” test that uses both `ModulePorts` + `PortSet` (not just `PortSet` in isolation)

**Files Created**:
```
crates/cfd2_macros/
├── Cargo.toml
└── src/lib.rs          # Proc-macro implementations (PortSet, ModulePorts)
```

**Deliverables**:
- `#[derive(PortSet)]` - Derives parameter/field/buffer port collections (`from_registry` registers ports idempotently)
- `#[derive(ModulePorts)]` - Exists but needs a clarified integration story (module instances are created after a registry exists, so “register ports” vs “build port set” must be reconciled)
- `#[port(module = "name")]` attribute parsing

### Phase 2: Core Port Runtime (Done)

**Goal**: Provide the runtime port types + registry that macros and modules will use

**Done**:
- [x] Define core traits (`ModulePortsTrait`, `PortSetTrait`, `PortValidationError`, `ModulePortRegistry`)
- [x] Implement port primitives (`FieldPort`, `ParamPort`, `BufferPort`) + WGSL helpers
- [x] Implement compile-time dimension tracking (`UnitDimension` + common dimensions)
- [x] Implement `PortRegistry` (field/param/buffer registration) + basic errors
- [x] Make `PortRegistry` registration idempotent (name/key indexes) + return clear conflicts when re-registered with incompatible specs
- [x] Add unit tests for the port runtime pieces

**Remaining**:
- [x] Add missing runtime validation (dimension mismatch checks + clear errors):
  - Decide whether `PortRegistry` enforces `D::to_runtime() == layout.unit()` for “semantic” fields/params
  - Provide an explicit escape hatch for genuinely dynamic-dimension fields (per prerequisites)

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

### Phase 3: PortManifest + Module Integration (Done)

**Goal**: Replace/augment `ModuleManifest` so modules can declare ports and the solver can build/validate a port registry

**Done**:
- [x] Define IR-safe `PortManifest` in `cfd2_ir::solver::ir::ports` (pure data, no `src/solver/model/*` deps)
- [x] Add `port_manifest: Option<PortManifest>` to `ModuleManifest`
- [x] Teach `#[derive(PortSet)]` to emit `port_manifest()` method
- [x] Migrate `eos` module to publish `PortManifest` for uniform params
- [x] Add `PortRegistry::register_manifest(module, &PortManifest)` for dynamic manifest registration + validation
- [x] Store `PortRegistry` on `SolverRecipe` (`SolverRecipe.port_registry`) and populate it from module manifests in `SolverRecipe::from_model()`
- [x] Consume `port_manifest.params` for the named-parameter surface area (`ModelSpec::named_param_keys()` + `named_params_for_model(...)`) so modules don’t have to duplicate uniform params in both `named_params` and `port_manifest`
- [x] Correct EOS port-manifest units (including temperature exponent for `eos.r`) and add a regression test
- [x] Align EOS `PortManifest` `wgsl_field` names with the `Constants` uniform struct (e.g. `eos_*`) and add regression assertions

**Remaining**:
- [x] Integrate with existing build-time generation (WGSL emission uses `PortManifest`; build-time plumbing avoids ad-hoc string lists once port manifests are the single source of truth)
  - Ensure module `port_manifest` is populated in build-script compilation (no `#[cfg(cfd2_build_script)]` gating for IR-safe manifests).
  - WGSL `Constants` generation consumes `PortManifest.params` (e.g. EOS) rather than hard-coded field lists.
- [x] Add validation helpers ("missing field", "wrong kind", "dimension mismatch", etc.) (`PortRegistry::validate_*` in `src/solver/model/ports/registry.rs`)
- [x] Add `ports::prelude` for convenient imports for module authors

**Notes**:
- Keep `manifest.named_params` as an escape hatch for params that aren’t simple WGSL uniform fields (solver knobs, flags, higher-level enums, etc.) until they are representable in `PortManifest` (or can be lowered to it cleanly).

**Files Created**:
```
crates/cfd2_ir/src/solver/ir/
└── ports.rs            # IR-facing PortManifest + port specs (re-export from ir/mod.rs)
```

### Phase 4: Low-Risk Module Migration (Weeks 3-4)

**Goal**: Migrate modules with only parameter declarations

**Modules**:
- [x] `eos` - EOS configuration (proof of concept) - publishes `PortManifest` for uniform params
- [x] `generic_coupled` - Publishes `PortManifest` for uniform params and removes duplication in `manifest.named_params`
- [x] `generic_coupled_apply` - Removed as a model module wrapper (kernel is already provided by `generic_coupled`)

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
- [x] `flux_module_gradients_wgsl` - Gradients stage (targets pre-resolved; WGSL generator no longer scans `StateLayout`; build-script fallback remains)
- [x] `flux_module_wgsl` - Flux kernel WGSL generation (state slot references are pre-resolved into IR-safe `ResolvedStateSlotsSpec` stored on `PortManifest`; WGSL generator consumes only that mapping; WGSL goldens cover compressible + incompressible_momentum)
- [x] `rhie_chow` - Pressure-velocity coupling (runtime generators use PortRegistry; build-script fallback remains)

**Special considerations**:
- Integrate with `PortRegistry` for field validation
- Validate field coupling invariants at runtime
- Update WGSL generation to use ports

## Module Migration Playbook (All string-based field-name users)

This section enumerates every current module/file that does string-based state lookups
(`StateLayout::{field, offset_for, component_offset}` or equivalent) and provides explicit
migration steps to move that logic onto the port infrastructure.

### 0) Prerequisites (one-time, required before bulk migration)

- [x] `#[derive(PortSet)]` is usable for real structs (registration + lookup)
- [x] Phase 3 baseline: IR-safe `PortManifest` exists and can be attached to model modules via `KernelBundleModule.manifest.port_manifest`
- [x] Unify/upgrade the dimension system across crates (see "Type-Level Dimensions Migration" below). This must happen before we can rely on `FieldPort<Dim, Kind>` throughout the repo (including in IR/codegen).
- [x] Support **derived/dynamic field names** (e.g. `format!("grad_{}", pressure.name())`) in the port system:
  - Implemented as a centralized string interner: `src/solver/model/ports/intern.rs`.
  - `PortRegistry` field registration APIs accept `&str` and internally intern to `&'static str` for storage.
- [x] Add (or decide against) dimension enforcement policy in `PortRegistry`:
  - For "semantic" fields (p, rho, mu, grad_p, etc) enforce runtime dimension matches compile-time `UnitDimension`
  - For "system-driven" fields whose dimensions vary by model, provide a supported escape hatch:
    - Option A: an explicit "any-dimension" port type (no dimension check)
    - Option B: a separate untyped field port that tracks kind only

### 1) Common refactor recipe (apply to every module below)

1. **Inventory**: list every string-based lookup in the module (field name(s), expected kind, expected unit).
2. **Define ports**:
   - Add a `PortSet` struct describing required fields/params/buffers.
   - If the set is dynamic (depends on model/system), use a manifest "record list" (Vec of resolved port specs) rather than a fixed struct.
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
- [x] Define dimension aliases needed by this module in the canonical IR type-level dimensions module (see "Type-Level Dimensions Migration"):
  - `PressureGradient = DivDim<Pressure, Length>`
  - `D_P = DivDim<MulDim<Volume, Time>, Mass>` (matches `si::D_P`)
- [x] Create a port set / dynamic manifest entries for:
  - `dp_field`: scalar field (`D_P`, Scalar) (name comes from module config)
  - `grad_p` + `grad_p_old`: vector2 fields (`PressureGradient`, Vector2) (names derived from inferred pressure field)
  - `momentum`: vector2 field (dimension is model-dependent; use the chosen escape hatch)
- [x] Add a single resolver:
  - Infer `(momentum, pressure)` via `infer_unique_momentum_pressure_coupling_referencing_dp`
  - Build all ports (and derived `grad_*` names) from that inference once (avoid duplicating inference/registration per-kernel)
- [x] Change the kernel generators in this module to use ports/offsets in the runtime build:
  - Re-introduce a `#[cfg(cfd2_build_script)]` runtime path that uses `PortRegistry` (derived names via interner)
  - Keep a `#[cfg(not(cfd2_build_script))]` build-script fallback path using `StateLayout`
  - Keep `rhie_chow_wgsl.rs` unchanged (it already accepts offsets/stride)
- [x] Update/extend tests in `src/solver/model/modules/rhie_chow.rs`:
  - Keep the existing dp-field-name contract test
  - Restore a regression test that missing `grad_<p>_old` fails with a clear error (guard against silent behavior changes)

#### B) `src/solver/model/modules/flux_module_gradients_wgsl.rs` (Flux gradients stage)

**String lookups to remove** (today):
- Gradient target discovery + offset resolution happens in `resolve_flux_module_gradients_targets(...)` (currently scans `StateLayout`).
- WGSL generator no longer uses `StateLayout` (consumes pre-resolved targets).

**Migration steps**:
- [x] Port runtime offset resolution to `PortRegistry` (keep build-script fallback).
- [x] Move "gradient target discovery" out of WGSL generation:
  - Implemented as `ResolvedGradientTarget` + `resolve_flux_module_gradients_targets(...)` in `src/solver/model/modules/flux_module.rs`.
- [x] Move/duplicate `ResolvedGradientTarget` into an IR-safe record type (in `cfd2_ir::solver::ir::ports`) and attach it to the module's `PortManifest` (for Phase 6 / build-time codegen plumbing).
  - Created `ResolvedGradientTargetSpec` in `crates/cfd2_ir/src/solver/ir/ports.rs` with all necessary fields
  - Added `gradient_targets: Vec<ResolvedGradientTargetSpec>` field to `PortManifest`
  - Updated `flux_module_module()` to accept `&EquationSystem` and `&StateLayout` and pre-resolve targets
  - Gradient targets are now attached to `manifest.port_manifest` when gradients are enabled
  - Updated all call sites in `compressible.rs`, `incompressible_momentum.rs`, and `definitions.rs`
  - Added `PortRegistry::register_state_field()` method for bulk field registration
  - Modified `SolverRecipe::from_model()` to pre-register all StateLayout fields
  - Tests updated to use runtime path with properly populated PortRegistry
- [x] Change `generate_flux_module_gradients_wgsl(...)` to accept resolved targets instead of a `StateLayout`.
- [x] Delete all `StateLayout` lookups from `flux_module_gradients_wgsl.rs` (generator no longer probes layout).
- [x] Add/keep tests to ensure WGSL output remains identical (or explicitly bless diffs).

#### C) `src/solver/model/modules/flux_module_wgsl.rs` (Flux kernel WGSL generation)

**String lookups to remove** (today):
- `state_layout.offset_for(name)` and `state_layout.component_offset(base, c)` inside helper routines.
- Various "does this field/component exist?" checks using `StateLayout`.

**Migration steps**:
- [x] Port runtime offset resolution to `PortRegistry` (keep build-script fallback).
  - Created `OffsetResolver` trait abstracting over PortRegistry and StateLayout
  - Runtime (`#[cfg(cfd2_build_script)]`) uses `PortRegistryResolver` with Dimensionless placeholder
  - Build-script (`#[cfg(not(cfd2_build_script))]`) keeps using StateLayout directly
  - Updated `state_component_at`, `state_component_at_side`, `apply_slipwall_velocity_reflection`, `resolve_state_field_component`
- [x] Thread a single resolver through WGSL lowering (no per-helper registry creation):
  - Both `generate_flux_module_wgsl` and `generate_flux_module_wgsl_runtime_scheme` now create a single resolver at the top
  - Helper functions now accept `&dyn OffsetResolver` instead of repeatedly probing `StateLayout`
  - `state_var_name` now uses `resolver.field_kind()` instead of `layout.field()`
- [x] Add a lowering pass that resolves every state-field reference used by flux codegen into a "resolved state slot" *outside* WGSL generation:
  - Introduced IR-safe `ResolvedStateSlotsSpec`/`ResolvedStateSlotSpec` and attached as `PortManifest.resolved_state_slots`
  - Added a resolver pass that traverses `FluxModuleKernelSpec` + referenced `PrimitiveExpr`s and resolves required state slots once
  - Updated `generate_flux_module_wgsl*` to consume only the resolved mapping (no `StateLayout` probing)
- [x] Add/keep regression coverage to ensure WGSL output remains identical (or explicitly bless diffs).

#### D) `src/solver/gpu/lowering/programs/generic_coupled.rs` (GPU lowering: generic coupled)

**String lookups to remove** (today):
- Uses `layout.offset_for(name)` / `layout.component_offset(name, comp)` while building program bindings for unknowns.

**Migration steps**:
- [x] Introduce a pre-resolved per-model "unknown → state slot" mapping:
  - Created `ResolvedUnknownMapping` struct to hold the mapping from equation targets to state offsets
  - Implemented `resolve_unknown_mapping_build_script()` for tests/build-script fallback using `StateLayout`
  - Mapping is computed once in `GenericCoupledProgramResources::new()` and stored in the struct
- [x] Update lowering to use the mapping instead of calling `StateLayout` directly:
  - `OuterConvergenceMonitor::new()` now consumes `ResolvedUnknownMapping`
  - `validate_schur_model()` now consumes `ResolvedUnknownMapping`
- [x] Fix the runtime mapping resolver to use the PortRegistry correctly:
  - Added `PortRegistry::register_state_field()` method for bulk-registration without typed returns
  - Modified `SolverRecipe::from_model()` to pre-register all `StateLayout` fields into `port_registry` before module manifests
  - Runtime path now works correctly - all equation target fields are available in PortRegistry
- [x] Restore runtime-path tests so this can't regress:
  - Updated `schur_rejects_invalid_layout_indices` and `schur_accepts_vector3_velocity_layout` to use runtime path
  - Added `runtime_mapping_fails_when_field_not_registered` regression test
  - Added `runtime_mapping_succeeds_with_all_fields_registered` test covering scalar + Vector3 fields

#### E) `src/solver/model/helpers/solver_ext.rs` (Host-side helpers)

**String lookups to remove** (today):
- Repeated `state_layout.offset_for("rho")`, `"rho_u"`, `"rho_e"`, `"p"`, `"T"`, `"u"/"U"` when seeding initial conditions.

**Migration steps**:
- [x] Update compressible state seeding helpers to use cached `PortRegistry` field entries for offsets (no per-call `StateLayout` cloning).
- [x] Define "canonical field port sets" for the helper surfaces:
  - Compressible: `rho`, `rho_u` (vec2), `rho_e` (+ optional `p`, `T`, `u|U`)
  - Incompressible: `U|u`, `p`, (and any optional diagnostics fields used by stats helpers)
- [x] Resolve these ports once per solver/model and cache them (avoid per-call `StateLayout` scanning): store `Arc<PortRegistry>` in plan resources and expose via `GpuUnifiedSolver::port_registry()`.
- [x] Update helper implementations to use cached ports for offsets/stride (compressible state seeding).
- [x] Leave boundary APIs string-keyed for now, but centralize the names as constants next to the port set definitions.

#### F) `src/solver/model/definitions/incompressible_momentum.rs` (Model construction)

**String lookups to remove** (today):
- `layout.component_offset("U", c)` and `layout.offset_for("p")` used as early validation.

**Migration steps**:
- [x] Replace ad-hoc layout checks with port resolution (runtime path; build-script fallback remains):
  - Build `FieldPort`s for `U` (vec2) and `p` (scalar) and fail early if missing/wrong kind
- [ ] Prefer using the same ports later for any host-side indexing needs (avoid duplicating offset logic).

#### G) `src/solver/model/definitions.rs` (Model validation and invariants)

**String lookups to migrate** (today):
- Flux-module gradients validation scans `grad_*` names and checks base existence via `state_layout.field(...)`.
- Invariant validation checks required fields via `state_layout.field(...)` / `component_offset(...)`.

**Migration steps**:
- [x] Make `PortManifest.gradient_targets` the single source of truth for flux-module gradients validation (no `StateLayout` `grad_*` scanning in `ModelSpec::validate_module_manifests()`).
- [x] Validate `port_manifest.fields` against the model `StateLayout` (existence + kind + unit) in `ModelSpec::validate_module_manifests()`.
- [x] Convert validation to:
  - Resolve all required ports once, returning structured `PortValidationError`
  - Avoid repeated `StateLayout` probing across multiple validation passes

#### H) `src/ui/app.rs` (UI-only state inspection)

**String lookups to migrate** (today):
- `layout.offset_for("U"/"u")` and `layout.offset_for("p")` for plotting/inspection.

**Migration steps** (optional; do after core solver is migrated):
- [x] Expose a small, stable "UI port set" from the solver/model (optional ports for common fields)
- [x] Update UI to use that port set instead of raw `StateLayout` probing

#### I) `crates/cfd2_codegen/src/solver/codegen/*` (Codegen helper modules)

These are not "model modules", but they are the largest remaining concentration of string-based
state lookups and will block completing Phase 6 unless migrated.

**Migration steps**:
- [x] Introduce an IR-safe "resolved state slot" type (offset + stride + kind/unit metadata as needed).
- [x] Update `state_access.rs` to accept resolved slots instead of `(StateLayout, field_name)`.
- [x] Update callers (`generic_coupled_kernels`, `unified_assembly`, `primitive_expr`, `coeff_expr`) to use slots/ports provided by model/module manifests.

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
- [x] Attempted to canonicalize type-level dimensions (BLOCKED by Rust limitations):
  - Introduced `Dim<...exponents...>` canonical carrier type in `dimensions.rs`
  - **BLOCKER**: Rust doesn't allow type parameters in const generic expressions in a way that makes them "used"
  - Cannot make `MulDim<A, B>` automatically resolve to `Dim<...>` even when exponents are computable
  - The typed builder works for structurally identical dimensions but cannot prove equivalence of semantically-equal-but-structurally-different dimensions (e.g., `Volume/Time` vs `Area²/(Time·Length)`)
  - **Workaround**: Typed builder supports explicit `cast_to()` for semantically-equal units (runtime-checked), enabling incremental adoption while keeping `validate_units()` as the backstop
- [x] Migrate model constructors in `src/solver/model/definitions/*` to use the typed builder APIs (still producing the same `EquationSystem` as output).
  - **Current**: all model definitions use the typed builder layer (with `cast_to()` where needed).
  - ✅ Migrated: `generic_diffusion_demo` (+ Neumann variant) uses typed builder + `cast_to()`
  - ✅ Migrated: `incompressible_momentum` uses typed builder + `cast_to()` (`Force` / `MassFlux`)
  - ✅ Migrated: `compressible` uses typed builder + `cast_to()` (`Force` / `MassFlux` / `Power`)
- [x] Update `crates/cfd2_codegen/src/solver/codegen/ir.rs` (`lower_system`) to:
  - Prefer typed-built systems (no unit errors expected)
  - Keep `validate_units()` for any remaining untyped system construction paths
- [x] Update `crates/cfd2_ir/src/solver/model/backend/scheme_expansion.rs` (`expand_schemes`) similarly:
  - Prefer typed-built systems
  - Keep `validate_units()` as a backstop during migration

### 4) Port the codegen DSL unit checks to type-level dimensions

- [x] Introduce a typed variant of `TypedExpr` parameterized by dimension type:
  - `TypedExpr<D>` where `D: UnitDimension`
  - Addition/subtraction only type-checks for identical `D`
  - Multiplication/division returns `TypedExpr<MulDim<...>>` / `TypedExpr<DivDim<...>>`
  - `sqrt()` returns `TypedExpr<SqrtDim<D>>`
  - Keep an ergonomic escape hatch for genuinely dynamic units (explicitly marked, avoids silent "any unit")
- [ ] Incrementally migrate codegen helpers to the typed DSL, starting with the leaf utilities:
  - ✅ `crates/cfd2_codegen/src/solver/codegen/state_access.rs` (`state_*_dim<D> -> TypedExpr<D>` with runtime unit validation; `DynExpr` helpers remain the escape hatch)
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
        // Iterate module manifests and register/validate their port manifests.
        // NOTE: this requires a dynamic "register by spec" API on PortRegistry,
        // since PortManifest is pure data (no type parameters available here).
        for module in &self.modules {
            if let Some(manifest) = &module.manifest.port_manifest {
                registry.register_manifest(module.name, manifest)?;
            }
        }
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
crates/cfd2_macros/tests/*        # trybuild + compile-fail tests
```

## Success Criteria

- [ ] `cargo test` passes
- [ ] Zero string-based field lookups in migrated modules
- [ ] Compile-time dimension checking works where Rust can express it (ports + structurally-identical typed IR); runtime `validate_units()` remains the backstop for semantically-equal-but-structurally-different dimensions
- [ ] Runtime validation catches invalid configs
- [ ] Generated WGSL diffs are empty (or equivalent check passes)
- [ ] Hard cutoff removes all deprecated APIs

## Notes

- **Breaking changes** documented in migration guide
- **No feature flags** - hard cutoff approach
- **Documentation** updated after each phase
- **CHANGELOG** updated with breaking changes

## Current Status

As of **2026-01-31**:
- Canonical type-level dimensions (rational exponents) live in `crates/cfd2_ir/src/solver/dimensions.rs` and are re-exported through the main crate for ports
- A typed IR builder layer exists at `crates/cfd2_ir/src/solver/model/backend/typed_ast.rs` for **best-effort** compile-time unit checking when building `EquationSystem`s in Rust
- Attempted full dimension canonicalization is **blocked** on stable Rust today; the type-level dimension system cannot prove equivalence of semantically-equal-but-structurally-different expressions (see “Type-Level Dimensions Migration”, Step 3)
- Port runtime types + registry exist under `src/solver/model/ports/*`
- `#[derive(PortSet)]` exists (param/field/buffer) with trybuild tests; `PortRegistry` registration is idempotent
- `#[derive(ModulePorts)]` exists but still needs a clarified integration story (how/when modules materialize a port set and expose it to codegen)
- IR-safe `PortManifest` exists (`crates/cfd2_ir/src/solver/ir/ports.rs`) and is attached to `ModuleManifest` via `ModuleManifest.port_manifest`
- `PortManifest` field specs support an `ANY_DIMENSION` unit sentinel to skip unit validation for dynamic-dimension fields
- Named parameter allowlisting now includes `port_manifest.params[*].key`, so modules no longer need to duplicate uniform params in `manifest.named_params`
- First module migrated: `eos` publishes a `PortManifest` for its uniform params and no longer duplicates those uniform keys in `manifest.named_params` (keeps only non-uniform/escape-hatch keys there)
- Second module migrated: `generic_coupled` publishes a `PortManifest` for its uniform params and no longer duplicates those uniform keys in `manifest.named_params` (keeps only host-only keys there)
- Derived/dynamic field name support is implemented via a centralized interner (`src/solver/model/ports/intern.rs`); `PortRegistry::{register_field,register_scalar_field,register_vector2_field,register_vector3_field}` accept `&str`
- `rhie_chow` publishes a `PortManifest` (dp, grad_p, grad_p_old, momentum) and pre-resolves momentum/pressure coupling at module creation; kernel generators use `PortRegistry` (runtime build) to resolve offsets/stride; the build script compilation context still uses the legacy `StateLayout` code paths
- `flux_module_gradients_wgsl` consumes pre-resolved `PortManifest.gradient_targets` during kernel WGSL generation; target discovery still scans `StateLayout` during module creation; the build script compilation context still uses legacy `StateLayout` offsets
- `flux_module_wgsl` consumes pre-resolved `PortManifest.resolved_state_slots` (IR-safe `ResolvedStateSlotsSpec`) and no longer probes `StateLayout`; golden WGSL tests cover compressible + incompressible_momentum
- `SolverRecipe::from_model()` builds/stores a `PortRegistry` (`SolverRecipe.port_registry`) from module port manifests when present
- `cfd2_build_script` cfg is currently used as a build-time hack to exclude runtime-only manifest attachment from the build script’s `include!()` compilation context; regression coverage exists to ensure the runtime recipe populates `port_registry` from the EOS manifest
- Build-time codegen consumes port manifests for uniform params (e.g. EOS `Constants` fields); string-based lookups remain in core hotspots (see “Module Migration Playbook”)

**Next (recommended)**:
- ✅ `generic_coupled_apply` module wrapper removed (kernel remains in `generic_coupled`).
- ✅ Phase 3: consume `port_manifest` at build time (codegen):
  - Plumb module `PortManifest` data into the build-time pipeline (uniform params + bindings), reducing reliance on ad-hoc string lists.
  - Keep an escape hatch for params that are not yet representable as `ParamPort<...>` (e.g. `low_mach.model` as `u32` enum).
- Continue migrating `flux_module`:
  - ✅ Attach resolved gradients targets to `PortManifest` (IR-safe) so build-time codegen can consume them
  - ✅ Add a lowering pass for flux codegen that pre-resolves all state slots used by a flux spec and passes them into WGSL generation (eliminate remaining `StateLayout` probing in `flux_module_wgsl.rs`)
- ✅ Dimension enforcement policy in `PortRegistry` implemented (unit mismatch errors + `AnyDimension` escape hatch).
- Keep dimensional correctness enforced by runtime `EquationSystem::validate_units()` for complex systems until/unless we adopt a different type-level encoding (nightly features or type-encoded exponents).

**Deliverables Ready**:
- ✅ `src/solver/model/ports/*` runtime primitives + tests
- ✅ Core traits (`ModulePortsTrait`, `PortSetTrait`, etc.)
- ✅ `#[derive(PortSet)]` + trybuild tests (param/field/buffer)
- ✅ IR-safe `PortManifest` types (`crates/cfd2_ir/src/solver/ir/ports.rs`) + re-export through `cfd2_ir::solver::ir`
- ✅ `ModuleManifest.port_manifest` wiring (model boundary-safe)
- ✅ `eos` publishes a first `PortManifest` (uniform params)
- ✅ Canonical dimension carrier type `Dim<...>` (usable directly, but not auto-derived from expressions)
- ✅ Typed IR builder layer (`crates/cfd2_ir/src/solver/model/backend/typed_ast.rs`) for structurally-matching unit expressions
- ⚠️ `#[derive(ModulePorts)]` exists but needs integration work before real module migrations

**Ready for**: Wiring model init/codegen to consume `PortManifest`, then migrating additional modules
