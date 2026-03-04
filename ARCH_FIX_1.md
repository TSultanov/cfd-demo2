# ARCH_FIX_1: Eliminate String-Based Fusion Compiler

## Problem Statement

The kernel fusion system (`crates/cfd2_codegen/src/solver/codegen/fusion.rs`) manipulated
kernel bodies as `Vec<String>` line arrays. Optimisation passes like load-after-store
forwarding and no-op self-assign cleanup used `KernelBodyIrOp` which tracked raw
`line_index: usize` values into these string arrays, then mutated them in place:

```rust
program.body[line_index] = line;   // fragile string replacement
```

Any upstream change that reformats WGSL output, adds a blank line, or changes indentation
would silently mis-align `line_index`, producing corrupted shaders.

**Goal:** `KernelProgram` should carry a typed `Vec<Stmt>` body instead of `Vec<String>`.
Fusion and optimisation passes should traverse/mutate the AST directly. String emission
(`lower_kernel_program_to_wgsl`) should be the absolute final step.

---

## Implementation Status — ✅ COMPLETE

All production code now flows through `Vec<Stmt>` AST types. Legacy `Vec<String>` fields
remain as parallel caches but are no longer authoritative. The lowerer, fusion synthesis,
and cleanup passes all use the AST path when `*_ast` fields are present (which they
always are in production).

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ DONE | AST arena reform — `Expr(u32)` → `Expr(Arc<ExprNode>)` in `cfd2_ir::ast` |
| Phase 1 | ✅ DONE | Added `body_ast`, `preamble_ast`, `indexing_ast` to `KernelProgram` |
| Phase 2 | ✅ DONE | All producers populate `*_ast` fields (body, preamble, indexing) |
| Phase 3 | ✅ DONE | Fusion synthesis builds `body_ast` alongside string body |
| Phase 4 | ✅ DONE | `lower_kernel_program_to_wgsl` uses `*_ast` when available |
| Phase 5 | ✅ DONE | AST-based cleanup passes; last manual-string site converted; all tests updated |
| Cleanup | ⏳ FUTURE | Remove legacy `Vec<String>` fields and `body_ir_ops` entirely |

### OpenFOAM drift check

All phases completed with **zero metric drift** vs baseline. The pre-existing
`openfoam_compressible_acoustic_matches_reference_profile` failure is unchanged.

---

## Architecture (Current)

### Data flow

```
Producer (Stmt-based DSL)
  ──────────────────────▶ KernelProgram.body_ast      (Vec<Stmt>, authoritative)
  ──────────────────────▶ KernelProgram.preamble_ast   (Vec<Stmt>, authoritative)
  ──────────────────────▶ KernelProgram.indexing_ast   (Vec<Stmt>, authoritative)
  ──render_stmt_lines()─▶ KernelProgram.body           (Vec<String>, parallel cache)
  ──render_stmt_lines()─▶ KernelProgram.preamble       (Vec<String>, parallel cache)
  ──render_stmt_lines()─▶ KernelProgram.indexing       (Vec<String>, parallel cache)

Fusion synthesis
  ── body_ast present ──▶  concatenate + rename_stmts() → fused.body_ast
  ── body_ast absent ──▶   concatenate + rename_lines() → fused.body  (legacy fallback)

Aggressive cleanup (when body_ast present)
  ── apply_ast_load_after_store_forwarding()  (AST walk, no line indices)
  ── apply_ast_noop_self_assign_cleanup()     (retain filter, no line indices)

Constants field detection
  ── body_ast present ──▶  ast_stmt_references_field() (structural AST search)
  ── body_ast absent ──▶   string .contains() search (legacy fallback)

WGSL lowering
  ── *_ast present ──▶  render from AST directly
  ── *_ast absent ──▶   emit from Vec<String>  (legacy fallback)
```

### AST types (`cfd2_ir::ast`)

Self-contained, `Clone + Send + Sync + PartialEq + Eq`:

| Type | Location | Notes |
|------|----------|-------|
| `Expr(Arc<ExprNode>)` | `cfd2_ir/src/ast/expr.rs` | Replaces `Expr(u32)` arena |
| `ExprNode` | `cfd2_ir/src/ast/expr.rs` | `Ident`, `Literal`, `Binary`, `Unary`, `Call`, `Index`, `Field` |
| `Stmt` | `cfd2_ir/src/ast/stmt.rs` | `Let`, `Var`, `Assign`, `If`, `For`, `Loop`, `While`, etc. |
| `Block` | `cfd2_ir/src/ast/stmt.rs` | `Vec<Stmt>` wrapper |
| `Type` | `cfd2_ir/src/ast/types.rs` | `F32`, `U32`, `Vec2`, `Array`, `Custom`, etc. |

`cfd2_codegen::wgsl_ast` re-exports all types from `cfd2_ir::ast` and adds:
`Module`, `Item`, `Function`, `CseBuilder`, `render_stmt_lines_with_ir`.

### KernelProgram fields

```rust
pub struct KernelProgram {
    // Authoritative AST (preferred by lowerer and fusion)
    pub body_ast:     Option<Vec<Stmt>>,      // always Some in production
    pub preamble_ast: Option<Vec<Stmt>>,      // always Some in production
    pub indexing_ast: Option<Vec<Stmt>>,      // always Some in production

    // Legacy string arrays (parallel cache; will be removed in future cleanup)
    pub body:         Vec<String>,
    pub preamble:     Vec<String>,
    pub indexing:     Vec<String>,
    pub body_ir_ops:  Vec<KernelBodyIrOp>,    // only used by legacy cleanup fallback
    pub local_symbols: Vec<String>,
}
```

### Fusion AST functions (`fusion.rs`)

| Function | Purpose |
|----------|---------|
| `rename_expr(expr, map)` | Structural identifier rename in expression tree |
| `rename_stmt(stmt, map)` | Structural identifier rename in statement tree |
| `rename_stmts(stmts, map)` | Rename identifiers in a slice of statements |
| `rename_block(block, map)` | Rename identifiers in a block |
| `rename_for_init(init, map)` | Rename identifiers in for-loop initializer |
| `rename_for_step(step, map)` | Rename identifiers in for-loop step |
| `apply_ast_load_after_store_forwarding(program)` | AST-based load-store forwarding |
| `apply_ast_noop_self_assign_cleanup(program)` | AST-based noop assignment removal |
| `ast_buffer_access_key(expr)` | Extract `(base, index)` key from index expression |
| `ast_collect_buffer_accesses(expr)` | Collect all buffer accesses in expression tree |
| `ast_expr_references_field(expr, base, field)` | Check if expression references `base.field` |
| `ast_stmt_references_field(stmt, base, field)` | Check if statement references `base.field` |

---

## Producer Coverage

All 15 producer sites across 8 files now set all three `*_ast` fields:

| File | Sites | `body_ast` | `preamble_ast` | `indexing_ast` |
|------|-------|------------|----------------|----------------|
| `packed_state_gradients.rs` | 1 | ✅ | ✅ | — (none) |
| `generic_coupled_kernels.rs` | 2 | ✅ | — | ✅ |
| `unified_assembly.rs` | 1 | ✅ | — | — |
| `flux_module_wgsl.rs` | 2 | ✅ | — | — |
| `flux_module_gradients_wgsl.rs` | 1 | ✅ | — | — |
| `rhie_chow.rs` | 5 | ✅ | ✅ (3 sites) | ✅ |
| `kernel.rs` (test) | 1 | ✅ | ✅ | ✅ |
| `fusion.rs` (test `sample_program`) | 1+ | ✅ | ✅ | — |

---

## Completed Work Detail

### Phase 0: AST arena reform ✅

Created `cfd2_ir::ast` module with self-contained types:
- `cfd2_ir/src/ast/mod.rs` — module root
- `cfd2_ir/src/ast/expr.rs` — `Expr(Arc<ExprNode>)`, operator impls, WGSL rendering
- `cfd2_ir/src/ast/stmt.rs` — `Stmt`, `Block`, `ForInit`, `ForStep`, rendering
- `cfd2_ir/src/ast/types.rs` — `Type`, `AddressSpace`, `RenderContext`

Rewrote `cfd2_codegen::wgsl_ast` to re-export from `cfd2_ir::ast`. Removed the
thread-local `EXPR_ARENA`. Fixed ~255 compilation errors across ~20 files caused
by `Expr` no longer being `Copy`.

### Phase 1: Added AST fields to `KernelProgram` ✅

Added `body_ast: Option<Vec<Stmt>>`, `preamble_ast: Option<Vec<Stmt>>`,
`indexing_ast: Option<Vec<Stmt>>` to `KernelProgram`, initialised to `None`.

### Phase 2: Migrated all producers ✅

All producer sites now set `body_ast = Some(stmts)`, and also `preamble_ast`
and `indexing_ast` where applicable.

### Phase 3: AST-based fusion ✅

Fusion synthesis (`synthesize_fused_program_with_report_remapped`) now:
1. Checks if all input programs have `body_ast`
2. If yes: concatenates AST bodies via `rename_stmts()` (structural rename)
3. Propagates `indexing_ast` and `preamble_ast` through fusion
4. Adds synthesis marker comment to both string and AST preambles
5. String-based concatenation still runs in parallel for backward compat

### Phase 4: AST-first WGSL lowering ✅

`lower_kernel_program_to_wgsl` now prefers `*_ast` fields for rendering when
present, falling back to string arrays when `None`.

### Phase 5: AST-based cleanup + final conversions ✅

1. Converted the last manual-string producer (`kernel.rs` test helper) to use
   `Stmt` AST
2. Added AST-based cleanup passes:
   - `apply_ast_load_after_store_forwarding` — walks `body_ast`, no line indices
   - `apply_ast_noop_self_assign_cleanup` — `retain()` filter on `body_ast`
3. `apply_aggressive_cleanup` now dispatches to AST path when `body_ast` is present
4. `program_references_constants_field` now uses AST-based field search when
   `*_ast` is available
5. All fusion test helpers (`sample_program`, cleanup tests) updated to set
   `body_ast`/`preamble_ast`
6. All `rhie_chow.rs` sites now set `indexing_ast`
7. `generic_coupled_kernels.rs` now sets `indexing_ast`
8. `packed_state_gradients.rs` now sets `preamble_ast`

---

## What Remains (Future Cleanup)

These are non-essential cleanup tasks. All production code already uses AST paths.

### Remove `Vec<String>` fields from `KernelProgram`

Remove `body`, `preamble`, `indexing` fields. Make `body_ast` / `preamble_ast` /
`indexing_ast` non-optional (`Vec<Stmt>` instead of `Option<Vec<Stmt>>`).

**Scope:** Every test and producer that sets string fields. One fallback test
(`lowering_falls_back_to_string_scan_when_eos_params_empty`) intentionally uses
string-only body to test the legacy path — it would need redesign.

### Remove `body_ir_ops` and `KernelBodyIrOp`

Remove the `KernelBodyIrOp` enum from `cfd2_ir`, the `body_ir_ops` field from
`KernelProgram`, the legacy `apply_ir_*` cleanup functions, and
`render_stmt_lines_with_ir` from `wgsl_ast.rs`.

### Remove legacy string helpers from `fusion.rs`

Remove `rename_lines`, `rename_text`, `rename_identifier`, `is_ident_boundary`,
`is_member_access_field`, `is_ident_char` — all replaced by structural AST rename.

### Remove `local_symbols` field

Once fusion operates exclusively on AST, local symbols can be derived structurally
by walking the `Let`/`Var` statements.

---

## Risk Assessment

| Risk | Severity | Status |
|------|----------|--------|
| `Expr` arena removal breaks CSE | Medium | ✅ Resolved — structural hashing on `Arc<ExprNode>` |
| `Arc` overhead vs `u32` copy | Low | Acceptable — codegen is build-time, not hot-path |
| Thread-local arena leak (latent bug) | Low | ✅ Fixed — removed entirely |
| Large merge conflicts | Medium | ✅ Mitigated — optional fields, no breaking changes |
| `Eq`/`Hash` on `f32` in `Expr` | Low | ✅ Handled — floats stored as `String` in `Literal::Float` |
| Dual-path inconsistency (AST vs string) | Low | Both paths kept in sync; AST is authoritative |

---

## Testing Strategy

All phases verified with:
1. `cargo test` — all 186+ tests pass (only pre-existing `block_jacobi` failure)
2. OpenFOAM reference suite — zero metric drift at every phase
3. Fusion contract tests pass end-to-end through new AST paths
