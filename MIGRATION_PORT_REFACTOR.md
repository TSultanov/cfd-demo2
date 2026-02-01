# Port Refactor Migration Guide

This guide explains how to migrate code from the legacy string-based architecture to the new port-based system with type-level dimensions.

## Overview

The port refactor introduces:
- **Type-level dimensions** for compile-time unit checking
- **PortRegistry** for centralized field/param/buffer management
- **PortManifest** for IR-safe module contracts
- **Pre-resolved metadata** to eliminate ad-hoc `StateLayout` probing

## What Changed

### 1. Type-Level Dimensions (Canonical)

Dimensional analysis moved from runtime (`UnitDim`) to compile-time types where possible.

**Before:**
```rust
use crate::solver::units::si;

let rho = vol_scalar("rho", si::DENSITY);
```

**After:**
```rust
use crate::solver::dimensions::Density;
use crate::solver::model::backend::ast::vol_scalar_dim;

let rho = vol_scalar_dim::<Density>("rho");
```

### 2. Port-Based Field Access

Direct `StateLayout` probing is replaced with `PortRegistry` registration and lookup.

**Before:**
```rust
// Direct StateLayout probing
if let Some(field) = layout.field("rho") {
    let offset = field.offset();
}
```

**After:**
```rust
use crate::solver::model::ports::{PortRegistry, Density};

let mut registry = PortRegistry::new(layout.clone());
let rho_port = registry.register_scalar_field::<Density>("rho")?;
let offset = rho_port.offset();
```

### 3. Module Manifests (PortManifest)

Modules now declare their ports via `PortManifest` attached to `KernelBundleModule`.

**Before:**
```rust
KernelBundleModule {
    name: "eos",
    named_params: vec!["eos.gamma", "eos.r"],
    ..Default::default()
}
```

**After:**
```rust
use crate::solver::ir::ports::{PortManifest, ParamSpec};

KernelBundleModule {
    name: "eos",
    port_manifest: Some(PortManifest {
        params: vec![
            ParamSpec { key: "eos.gamma", wgsl_field: "eos_gamma", wgsl_type: "f32", unit: /* dimensionless */ },
            ParamSpec { key: "eos.r", wgsl_field: "eos_r", wgsl_type: "f32", unit: /* temperature exponent */ },
        ],
        ..Default::default()
    }),
    ..Default::default()
}
```

## How to Update Code

### Type-Level Dimensions Migration

#### Use `*_dim` Constructors

Replace runtime unit parameters with compile-time type parameters:

| Before | After |
|--------|-------|
| `vol_scalar(name, unit)` | `vol_scalar_dim::<D>(name)` |
| `vol_vector(name, unit)` | `vol_vector_dim::<D>(name)` |
| `surface_scalar(name, unit)` | `surface_scalar_dim::<D>(name)` |

Available dimension types (in `crate::solver::dimensions`):
- `Density`, `Pressure`, `Velocity`, `Temperature`
- `Force`, `MassFlux`, `Power`, `InvTime`
- `D_P` (diagonal coefficient), `PressureGradient`
- `DivDim<A, B>`, `MulDim<A, B>`, `SqrtDim<A>` for compound dimensions

#### Typed IR Builder

For equation system construction, use the typed builder API:

```rust
use crate::solver::model::backend::typed_ast::{
    TypedFieldRef, Scalar, Vector2, typed_fvm, typed_fvc
};
use crate::solver::dimensions::{Velocity, Pressure, Force};

let u_typed = TypedFieldRef::<Velocity, Vector2>::new("U");
let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");

// Terms are type-checked at compile time
let ddt_term = typed_fvm::ddt_coeff(rho_coeff, u_typed);
let grad_term = typed_fvc::grad(p_typed);

// Cast to unify dimensions when needed
let momentum_eqn = (ddt_term.cast_to::<Force>() + grad_term.cast_to::<Force>()).eqn(u_typed);
```

### Port-Based Field Access Migration

#### Avoid Direct StateLayout Probing

**Don't do this:**
```rust
// Legacy pattern - direct StateLayout probing
layout.field("U").map(|f| f.offset())
layout.offset_for("p")
layout.component_offset("U", 0)
```

**Do this instead:**
```rust
// New pattern - PortRegistry registration
let mut registry = PortRegistry::new(layout.clone());

// Register fields with type+kind validation
let u_port = registry.register_vector2_field::<Velocity>("U")?;
let p_port = registry.register_scalar_field::<Pressure>("p")?;

// Access offsets through ports
let u0_offset = u_port.component(0).unwrap().full_offset();
let p_offset = p_port.offset();
```

#### Pre-Resolved Metadata Pattern

For modules that need to resolve multiple fields once at initialization:

```rust
/// Precomputed layout metadata for efficient lookups.
struct LayoutMetadata {
    fields_by_name: HashMap<String, FieldMetadata>,
}

struct FieldMetadata {
    kind: FieldKind,
    offset: u32,
    component_count: u32,
}

fn build_layout_metadata(layout: &StateLayout) -> LayoutMetadata {
    let mut fields_by_name = HashMap::new();
    for f in layout.fields() {
        fields_by_name.insert(
            f.name().to_string(),
            FieldMetadata {
                kind: f.kind(),
                offset: f.offset(),
                component_count: f.component_count(),
            },
        );
    }
    LayoutMetadata { fields_by_name }
}
```

#### PortManifest for Module Contracts

Modules should pre-resolve their field requirements and store in `PortManifest`:

```rust
pub fn my_module() -> KernelBundleModule {
    // ... module setup ...
    
    let port_manifest = PortManifest {
        params: vec![/* param specs */],
        fields: vec![/* field specs */],
        buffers: vec![/* buffer specs */],
        // Pre-resolved metadata for WGSL generation
        gradient_targets: vec![/* ResolvedGradientTargetSpec */],
        resolved_state_slots: Some(/* ResolvedStateSlotsSpec */),
    };
    
    KernelBundleModule {
        name: "my_module",
        port_manifest: Some(port_manifest),
        ..Default::default()
    }
}
```

### Validation Migration

Replace ad-hoc validation with `PortRegistry` validation:

```rust
// Validate required fields exist with correct kind/dimension
registry.validate_scalar_field::<Pressure>("my_module", "p")?;
registry.validate_vector2_field::<Velocity>("my_module", "U")?;

// Validation returns structured errors
match result {
    Err(PortValidationError::MissingField { module, field }) => { /* ... */ }
    Err(PortValidationError::FieldKindMismatch { field, expected, found }) => { /* ... */ }
    Err(PortValidationError::DimensionMismatch { field, expected, found }) => { /* ... */ }
    Ok(()) => { /* field is valid */ }
}
```

## Notable Breaking Changes

1. **`StateLayout::field()` probing in modules** - Modules should no longer call `layout.field()` directly. Use `PortRegistry` or pre-resolved metadata.

2. **`StateLayout::offset_for()` / `component_offset()`** - These are deprecated for module use. Register fields with `PortRegistry` and query ports for offsets.

3. **Runtime unit parameters** - `vol_scalar(name, UnitDim)` constructors are replaced with `vol_scalar_dim::<D>(name)` type-parameterized versions.

4. **`solver::units::si` usage** - Prefer canonical type-level dimensions from `crate::solver::dimensions`. The `si` module is retained only for runtime validation and debug output.

5. **Named params allowlisting** - `PortManifest.params[*].key` entries are now included in named-param allowlisting. Modules should not duplicate uniform param keys in `KernelBundleModule.named_params` (keep `named_params` for non-uniform/host-only escape hatches).

## Escape Hatches

### AnyDimension

For fields with genuinely dynamic dimensions, use the `AnyDimension` escape hatch:

```rust
use crate::solver::model::ports::dimensions::AnyDimension;

// Skips dimension validation
let dynamic_field = registry.register_scalar_field::<AnyDimension>("dynamic")?;
```

### DynExpr for Codegen

For codegen with dynamic units, use `DynExpr`:

```rust
use cfd2_codegen::solver::codegen::dsl::DynExpr;
use cfd2_codegen::solver::units::UnitDim;

// Runtime unit tracking/validation happens when you combine DynExpr values.
let expr = DynExpr::f32(1.0, UnitDim::dimensionless());
```

## Testing Migrated Code

After migration, verify:

1. **Unit tests pass**: `cargo test -p cfd2 --lib`
2. **WGSL generation unchanged**: `./scripts/check_generated_wgsl.sh`
3. **OpenFOAM reference tests** (if applicable): `cargo test --test openfoam_* -- --ignored`

## Related Documentation

- `PORT_REFACTOR_PLAN.md` - Detailed implementation plan and status
- `GENERATED_WGSL_POLICY.md` - WGSL generation and verification workflow
- `CODEGEN_PLAN.md` - Overall codegen architecture
