# ARCH_FIX_4: Flatten and Rename `cfd2_ir` Module Hierarchy — COMPLETE

## Status: ✅ COMPLETE (single commit)

Flattened `cfd2_ir`'s module hierarchy from deeply nested `solver::model::backend::*` paths
to flat, semantically-named top-level modules.

### What changed

| Old path (`cfd2_ir::solver::...`) | New path (`cfd2_ir::...`) |
|-----------------------------------|---------------------------|
| `model::backend::ast` | `equation::ast` |
| `model::backend::typed_ast` | `equation::typed_ast` |
| `model::backend::scheme` | `equation::scheme` |
| `model::backend::scheme_expansion` | `equation::scheme_expansion` |
| `model::backend::state_layout` | `equation::state_layout` |
| `ir::*` (FaceScalarExpr, FluxLayout...) | `kernel::*` |
| `ir::kernel_program` | `kernel::program` |
| `ir::ports` | `ports` |
| `ir::reconstruction` | `kernel::reconstruction` |
| `shared::expr` (PrimitiveExpr) | `flux` |
| `gpu::enums` | `gpu_enums` |
| `scheme` | `scheme` |
| `units` | `units` |
| `dimensions` | `dimensions` |

### Files affected

- **56 files** across 3 crates (`cfd2_ir`, `cfd2_codegen`, `cfd2`)
- Deleted: `solver/gpu/mod.rs`, `solver/mod.rs`, `solver/model/mod.rs`, `solver/shared/mod.rs`
- Updated: `build.rs` (added top-level aliases for `include!()`'d file dual-context compilation)
- Updated: 1 contract test (`contract_codegen_invariants_test.rs`) for new file paths

### Verification

- **Tests**: 186+ pass (only pre-existing `block_jacobi` failure)
- **OpenFOAM**: Zero metric drift
- **Net**: +13/-160 restructured lines (same code, flat module paths)
- **Main crate API**: Unchanged (re-export shims in `src/solver/` still expose same paths)
