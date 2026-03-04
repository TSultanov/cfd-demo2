# ARCH_FIX_1: Eliminate String-Based Fusion Compiler — COMPLETE

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

## Status: ✅ FULLY COMPLETE

All legacy fields and types have been removed. `KernelProgram` now uses typed AST
(`Vec<Stmt>`) for all code sections. No string-based code manipulation remains.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ DONE | AST arena reform — `Expr(u32)` → `Expr(Arc<ExprNode>)` in `cfd2_ir::ast` |
| Phase 1 | ✅ DONE | Added `body_ast`, `preamble_ast`, `indexing_ast` to `KernelProgram` |
| Phase 2 | ✅ DONE | All producers populate `*_ast` fields |
| Phase 3 | ✅ DONE | Fusion synthesis builds AST bodies |
| Phase 4 | ✅ DONE | `lower_kernel_program_to_wgsl` uses AST |
| Phase 5 | ✅ DONE | AST-based cleanup passes replace line-index IR ops |
| Cleanup | ✅ DONE | Removed all legacy fields, types, and string-based functions |

### OpenFOAM drift check

All phases completed with **zero metric drift** vs baseline. The pre-existing
`openfoam_compressible_acoustic_matches_reference_profile` failure is unchanged.

---

## Architecture (Final)

### Data flow

```
Producer (Stmt-based DSL)
  ──────────────────────▶ KernelProgram.body       (Vec<Stmt>)
  ──────────────────────▶ KernelProgram.preamble   (Vec<Stmt>)
  ──────────────────────▶ KernelProgram.indexing   (Vec<Stmt>)

Fusion synthesis
  ── rename_stmts() ──▶  concatenate AST bodies → fused.body

Aggressive cleanup
  ── apply_ast_load_after_store_forwarding()  (AST walk, no line indices)
  ── apply_ast_noop_self_assign_cleanup()     (retain filter, no line indices)

Constants field detection
  ── ast_stmt_references_field()  (structural AST search)

WGSL lowering (lower_kernel_program_to_wgsl)
  ── render_stmt_lines() on indexing, preamble, body → String output
```

### AST types (`cfd2_ir::ast`)

Self-contained, `Clone + Send + Sync + PartialEq + Eq`:

| Type | Location | Notes |
|------|----------|-------|
| `Expr(Arc<ExprNode>)` | `cfd2_ir/src/ast/expr.rs` | Self-contained expression tree |
| `ExprNode` | `cfd2_ir/src/ast/expr.rs` | `Ident`, `Literal`, `Binary`, `Unary`, `Call`, `Index`, `Field` |
| `Stmt` | `cfd2_ir/src/ast/stmt.rs` | `Let`, `Var`, `Assign`, `If`, `For`, `Loop`, `While`, etc. |
| `Block` | `cfd2_ir/src/ast/stmt.rs` | `Vec<Stmt>` wrapper |
| `Type` | `cfd2_ir/src/ast/types.rs` | `F32`, `U32`, `Vec2`, `Array`, `Custom`, etc. |

### KernelProgram (final)

```rust
pub struct KernelProgram {
    pub id: String,
    pub dispatch: DispatchDomain,
    pub launch: LaunchSemantics,
    pub bindings: Vec<KernelBinding>,
    pub helper_functions: Vec<String>,
    pub preamble: Vec<Stmt>,       // typed AST
    pub indexing: Vec<Stmt>,       // typed AST
    pub body: Vec<Stmt>,           // typed AST
    pub side_effects: SideEffectMetadata,
    pub eos_params: Vec<ParamSpec>,
}
```

Local symbols are derived on-the-fly via `program.local_symbols()`, which walks
`Let`/`Var` declarations in preamble + body (excluding indexing, since indexing
is shared across fused programs).

---

## What Was Removed (Cleanup Phase)

### Types removed from `cfd2_ir`
- `KernelBodyIrOp` enum (Store, LetLoad, NoopSelfAssign, Invalidate)
- `KernelBufferAccess` struct

### Fields removed from `KernelProgram`
- `body: Vec<String>` (replaced by `body: Vec<Stmt>`)
- `preamble: Vec<String>` (replaced by `preamble: Vec<Stmt>`)
- `indexing: Vec<String>` (replaced by `indexing: Vec<Stmt>`)
- `body_ir_ops: Vec<KernelBodyIrOp>`
- `local_symbols: Vec<String>` (now `pub fn local_symbols()` method)

### Fields renamed
- `body_ast: Option<Vec<Stmt>>` → `body: Vec<Stmt>`
- `preamble_ast: Option<Vec<Stmt>>` → `preamble: Vec<Stmt>`
- `indexing_ast: Option<Vec<Stmt>>` → `indexing: Vec<Stmt>`

### Functions removed from `fusion.rs`
- `rename_lines()` — string-based line rename
- `rename_text()` — string-based text rename
- `rename_identifier()` — character-level identifier rename
- `is_ident_boundary()` — helper for string rename
- `is_member_access_field()` — helper for string rename
- `is_ident_char()` — helper for string rename
- `rename_symbols()` — symbol list rename
- `rename_and_offset_body_ir_ops()` — IR op offset/rename
- `apply_ir_load_after_store_forwarding()` — legacy string-based forwarding
- `apply_ir_noop_self_assign_cleanup()` — legacy string-based cleanup

### Functions removed from `wgsl_ast.rs`
- `render_stmt_lines_with_ir()` — rendered stmts + emitted `KernelBodyIrOp` metadata
- `expr_as_buffer_access()` — helper
- `collect_buffer_accesses()` — helper

### Functions removed from `rhie_chow.rs`
- `rhie_chow_section_lines()` — `Vec<Stmt>` → `Vec<String>` conversion
- `rhie_chow_segment_ir_ops()` — `KernelBodyIrOp` construction
- `rhie_chow_collect_local_symbols_sections()` — local symbol collection

### Tests removed/rewritten
- `kernel_program_initializes_ir_ops_empty` — tested removed field
- `symbol_rename_skips_member_access_fields` — tested removed function
- `render_stmt_lines_with_ir_*` — tested removed function
- All cleanup tests rewritten to use AST-only assertions

---

## Testing Strategy

All phases verified with:
1. `cargo test` — 186+ tests pass (only pre-existing `block_jacobi` failure)
2. `cargo test -p cfd2_codegen -p cfd2_ir` — 245 crate tests pass
3. OpenFOAM reference suite — zero metric drift across all phases
4. Build script validates generated WGSL via naga
