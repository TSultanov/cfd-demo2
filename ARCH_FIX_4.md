# ARCH_FIX_4: Flatten and Rename `cfd2_ir` Module Hierarchy

## Problem Statement

`cfd2_ir` is an independent crate, but its internal module paths mirror the main crate's hierarchy:

```
cfd2_ir::solver::model::backend::ast       ← mirrors cfd2::solver::model::backend::ast
cfd2_ir::solver::model::backend::scheme    ← mirrors cfd2::solver::model::backend::scheme
cfd2_ir::solver::ir::*                     ← mirrors cfd2::solver::ir
cfd2_ir::solver::gpu::enums                ← mirrors cfd2::solver::gpu::enums
cfd2_ir::solver::scheme                    ← mirrors cfd2::solver::scheme
cfd2_ir::solver::units                     ← mirrors cfd2::solver::units
cfd2_ir::solver::dimensions                ← mirrors cfd2::solver::dimensions
cfd2_ir::solver::shared                    ← mirrors cfd2::solver::shared
```

The main crate's modules are thin re-export shims:
```rust
// src/solver/model/backend/ast.rs
pub use cfd2_ir::solver::model::backend::ast::*;

// src/solver/ir/mod.rs
pub use cfd2_ir::solver::ir::*;
```

**Why it's bad:** The `solver::model::backend` path inside `cfd2_ir` conflates the abstract syntax
tree with the physical model layer. The wrapping `solver::` module inside `cfd2_ir` is pure
hierarchy cargo-cult — the IR crate *is* the solver IR, it doesn't contain a "solver" sub-module.
This makes imports unnecessarily deep (`cfd2_ir::solver::model::backend::typed_ast::TypedFieldRef`)
and confuses which crate owns the concept.

---

## Current Module Layout

```
cfd2_ir/src/
├── lib.rs                          → pub mod ast; pub mod solver;
├── ast/                            → WGSL AST types (Expr, Stmt, Type)
│   ├── mod.rs
│   ├── expr.rs                     (997 lines)
│   ├── stmt.rs                     (420 lines)
│   └── types.rs                    (160 lines)
└── solver/
    ├── mod.rs                      → pub mod dimensions/gpu/ir/model/scheme/shared/units
    ├── dimensions.rs               (511 lines) — type-level physical dimensions
    ├── units.rs                    (388 lines) — runtime UnitDim
    ├── scheme.rs                   (158 lines) — Scheme enum
    ├── shared/
    │   ├── mod.rs
    │   └── expr.rs                 (50 lines) — PrimitiveExpr
    ├── gpu/
    │   ├── mod.rs
    │   └── enums.rs                (34 lines) — GpuBoundaryType, TimeScheme, etc.
    ├── model/
    │   ├── mod.rs                  → pub mod backend
    │   └── backend/
    │       ├── mod.rs              (17 lines) — re-exports
    │       ├── ast.rs              (844 lines) — EquationSystem, Term, FieldRef, etc.
    │       ├── typed_ast.rs        (1041 lines) — TypedFieldRef, TypedEquation, etc.
    │       ├── scheme.rs           (154 lines) — SchemeRegistry
    │       ├── scheme_expansion.rs (102 lines) — expand_schemes
    │       └── state_layout.rs     (135 lines) — StateLayout
    └── ir/
        ├── mod.rs                  (375 lines) — FaceScalarExpr, FluxLayout, FluxModuleKernelSpec, etc.
        ├── kernel_program.rs       (229 lines) — KernelProgram, KernelBinding
        ├── ports.rs                (216 lines) — PortManifest, FieldSpec, etc.
        └── reconstruction.rs       (157 lines) — ReconstructionBuilder trait + algorithms
```

---

## Proposed Layout

Flatten `solver/` away. Group by *what things are*, not by mirroring the consumer's hierarchy.

```
cfd2_ir/src/
├── lib.rs                          → pub mod ast; pub mod equation; pub mod types;
│                                      pub mod kernel; pub mod ports; pub mod flux;
│                                      pub mod gpu_enums; pub mod scheme;
│                                      pub mod units; pub mod dimensions;
├── ast/                            → (unchanged) WGSL AST types
│   ├── mod.rs
│   ├── expr.rs
│   ├── stmt.rs
│   └── types.rs
├── equation/                       → (was solver/model/backend/)
│   ├── mod.rs                      re-exports
│   ├── ast.rs                      EquationSystem, Term, FieldRef, Coefficient, etc.
│   ├── typed_ast.rs                TypedFieldRef, TypedEquation, etc.
│   ├── scheme.rs                   SchemeRegistry
│   ├── scheme_expansion.rs         expand_schemes
│   └── state_layout.rs             StateLayout
├── kernel/                         → (was solver/ir/)
│   ├── mod.rs                      FaceScalarExpr, FluxLayout, FluxModuleKernelSpec, etc.
│   ├── program.rs                  KernelProgram, KernelBinding  (was kernel_program.rs)
│   └── reconstruction.rs           ReconstructionBuilder trait
├── ports.rs                        → (was solver/ir/ports.rs) PortManifest, FieldSpec
├── flux.rs                         → (was solver/shared/expr.rs) PrimitiveExpr
├── gpu_enums.rs                    → (was solver/gpu/enums.rs) GpuBoundaryType, TimeScheme
├── scheme.rs                       → (was solver/scheme.rs) Scheme enum
├── units.rs                        → (was solver/units.rs) UnitDim
└── dimensions.rs                   → (was solver/dimensions.rs) type-level dimensions
```

### Naming rationale

| Old path | New path | Why |
|----------|----------|-----|
| `solver::model::backend::ast` | `equation::ast` | These are *equation* definitions (EquationSystem, Term, fvm/fvc), not generic "AST" |
| `solver::model::backend::typed_ast` | `equation::typed_ast` | Typed wrappers over the equation types |
| `solver::model::backend::scheme` | `equation::scheme` | SchemeRegistry is per-equation-term |
| `solver::model::backend::scheme_expansion` | `equation::scheme_expansion` | Expands schemes over equations |
| `solver::model::backend::state_layout` | `equation::state_layout` | Layout derived from equation fields |
| `solver::ir` | `kernel` | These are kernel IR types (FaceScalarExpr, FluxLayout, KernelProgram) |
| `solver::ir::kernel_program` | `kernel::program` | KernelProgram is a kernel concept |
| `solver::ir::ports` | `ports` | Standalone port manifest types |
| `solver::ir::reconstruction` | `kernel::reconstruction` | Reconstruction is kernel-level |
| `solver::shared::expr` | `flux` | PrimitiveExpr is used for flux/primitive derivations |
| `solver::gpu::enums` | `gpu_enums` | Flat — only 34 lines |
| `solver::scheme` | `scheme` | Already a leaf |
| `solver::units` | `units` | Already a leaf |
| `solver::dimensions` | `dimensions` | Already a leaf |

---

## Implementation Phases

### Phase 1: Move files inside `cfd2_ir` (internal restructuring)

Move files to the new locations. Update `lib.rs` and internal `use`/`mod` declarations.
All `cfd2_ir`-internal cross-references must be updated.

**Intra-crate references to update:**
- `crate::solver::model::backend::ast` → `crate::equation::ast`
- `crate::solver::units` → `crate::units`
- `crate::solver::dimensions` → `crate::dimensions`
- `crate::solver::scheme` → `crate::scheme`
- `crate::solver::ir` → `crate::kernel`
- `crate::ast` → `crate::ast` (unchanged)

### Phase 2: Update consumers in `cfd2_codegen`

`cfd2_codegen` references `cfd2_ir::solver::*` in both its `solver/mod.rs` re-export shims
and direct imports in codegen files.

**Changes:**
- `crates/cfd2_codegen/src/solver/mod.rs` — update re-export paths
- All files doing `use cfd2_ir::solver::ir::*` → `use cfd2_ir::kernel::*`
- All files doing `use cfd2_ir::solver::dimensions::*` → `use cfd2_ir::dimensions::*`
- etc.

### Phase 3: Update consumers in main crate (`cfd2`)

The main crate has thin re-export shims in `src/solver/`. These continue to exist (they define
the main crate's public API), but their `use` paths change:

```rust
// src/solver/ir/mod.rs
// Before: pub use cfd2_ir::solver::ir::*;
// After:  pub use cfd2_ir::kernel::*;

// src/solver/model/backend/mod.rs
// Before: pub use cfd2_ir::solver::model::backend::*;
// After:  pub use cfd2_ir::equation::*;
```

Direct `cfd2_ir::` references in source files (e.g., `src/solver/model/kernel.rs`) also need updating.

### Phase 4: Delete empty `solver/` directory in `cfd2_ir`

Remove the now-empty `solver/` tree and `lib.rs`'s `pub mod solver`.

---

## Files Affected

### `cfd2_ir` crate (internal moves + edits)

| File | Action |
|------|--------|
| `src/lib.rs` | Rewrite: new top-level module declarations |
| `src/solver/` (entire tree) | **Delete** after moving contents |
| `src/equation/` (new) | Move from `solver/model/backend/` |
| `src/kernel/` (new) | Move from `solver/ir/` |
| `src/ports.rs` (new) | Move from `solver/ir/ports.rs` |
| `src/flux.rs` (new) | Move from `solver/shared/expr.rs` |
| `src/gpu_enums.rs` (new) | Move from `solver/gpu/enums.rs` |
| `src/scheme.rs` (new) | Move from `solver/scheme.rs` |
| `src/units.rs` (new) | Move from `solver/units.rs` |
| `src/dimensions.rs` (new) | Move from `solver/dimensions.rs` |

### `cfd2_codegen` crate

| File | Change |
|------|--------|
| `src/solver/mod.rs` | Update re-export paths |
| ~12 source files | Update `use cfd2_ir::solver::*` → `use cfd2_ir::*` |

### Main crate (`cfd2`)

| File | Change |
|------|--------|
| `src/solver/ir/mod.rs` | `cfd2_ir::solver::ir::*` → `cfd2_ir::kernel::*` |
| `src/solver/model/backend/mod.rs` | `cfd2_ir::solver::model::backend::*` → `cfd2_ir::equation::*` |
| `src/solver/model/backend/ast.rs` | Update re-export path |
| `src/solver/model/backend/scheme.rs` | Update re-export path |
| `src/solver/model/backend/scheme_expansion.rs` | Update re-export path |
| `src/solver/model/backend/state_layout.rs` | Update re-export path |
| `src/solver/gpu/enums.rs` | Update re-export path |
| `src/solver/scheme.rs` | Update re-export path |
| `src/solver/units.rs` | Update re-export path |
| `src/solver/dimensions.rs` | Update re-export path |
| `src/solver/shared/mod.rs` | Update re-export path |
| `src/solver/model/kernel.rs` | Update direct `cfd2_ir::` references |
| `src/bin/fusion_coverage.rs` | Update direct `cfd2_ir::` reference |

---

## Out of Scope

- Renaming the main crate's re-export modules (`src/solver/model/backend/` stays as-is — it
  defines the main crate's public API, not the IR crate's internal structure)
- Changing semantics of any type — this is purely a file/module rename
- The `ast/` module (WGSL AST) — already at the correct level
- Deprecating `StateLayout` in favor of `PortRegistry` (that's issue #5)

---

## Verification

- **Compilation**: `cargo build` for all three crates
- **Tests**: All tests pass across `cfd2_ir`, `cfd2_codegen`, and `cfd2`
- **OpenFOAM**: Metrics must not regress (pure refactor, zero behavioral change)
- **No public API change**: Main crate re-exports remain at same paths

---

## Risk Assessment

**Medium risk.** This is a large mechanical rename touching ~25 files across 3 crates.
The risk is entirely in missing a path update (caught immediately by the compiler).
No behavioral changes — same types, same trait impls, same code, just different module paths.

The `cfd2_ir` crate has no external consumers beyond `cfd2_codegen` and `cfd2`, so
there is no semver concern.
