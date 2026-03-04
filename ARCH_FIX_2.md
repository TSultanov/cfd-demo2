# ARCH_FIX_2: Decouple Time Integration from Generic Assemblers

## Problem Statement

The "generic" coupled assembly kernels (`generic_coupled_kernels.rs` and
`unified_assembly.rs`) are intended to be PDE-agnostic — they iterate over
`DiscreteOp` elements from the IR and emit the corresponding matrix/RHS
contributions.  However, the time-derivative loop in both files directly
hardcodes **BDF2** math, **dual-time stepping** (pseudo-transient continuation
via `dtau`), and the runtime `time_scheme` enum dispatch:

```rust
// Inside main_assembly_fn (which should be generic)
stmts.push(dsl::if_block_expr(
    time_scheme.eq(TimeScheme::BDF2),   // ← hardcoded scheme knowledge
    dsl::block(vec![ /* BDF2 correction math */ ]),
    None,
));
stmts.push(dsl::if_block_expr(
    dtau.clone().gt(0.0),               // ← hardcoded pseudo-transient logic
    dsl::block(vec![ /* dtau diagonal+RHS */ ]),
    None,
));
```

### Why it's bad

1. **Code duplication:** The BDF2 + dtau block is copy-pasted almost verbatim
   between `generic_coupled_kernels.rs::main_assembly_fn` and
   `unified_assembly.rs::main_assembly_fn` (~80 lines each, differing only in
   trivial `.clone()` calls).

2. **Closed to extension:** Adding a new time integration scheme (e.g., RK4,
   Crank-Nicolson, SDIRK) requires modifying *both* generic assembler files,
   plus the `TimeScheme` enum, plus the constants struct — all of which are in
   the "infrastructure" layer rather than the "model" layer.

3. **Leaky abstraction:** A generic assembler should not know about BDF2 or
   `dtau`.  It should process a list of IR-level contributions (diagonal
   coefficient, RHS contribution) that were already pre-lowered by the model
   or time-integration module.

### The fix

Time integration terms should be expressed as standard `DiscreteOp` IR
contributions (diagonal coefficients + RHS source terms) **before** the generic
assembler processes them.  The assembler should see `ddt` as just another
"add to diagonal, add to RHS" operator with pre-computed coefficients, where
the BDF2/dtau branching has already been resolved into conditional IR nodes.

---

## Current Architecture

```
Model definition (e.g., incompressible_momentum.rs)
  ── fvm::ddt(u) ──────────────────────▶  Term { op: Ddt, field: u }
                                              │
                                              ▼
                                        lower_system()
                                              │
                                              ▼
                                        DiscreteOp { kind: TimeDerivative }
                                              │
                                              ▼
                          ┌───────────────────┴───────────────────┐
                          │  main_assembly_fn  (BOTH files)       │
                          │  ── hardcodes BDF1 base               │
                          │  ── if BDF2: corrects diag + RHS      │
                          │  ── if dtau > 0: adds dtau term       │
                          └───────────────────────────────────────┘
```

### Affected files

| File | Role | Lines affected |
|------|------|---------------|
| `crates/cfd2_codegen/src/solver/codegen/generic_coupled_kernels.rs` | Generic assembly (no flux) | ~80 lines (ddt block) |
| `crates/cfd2_codegen/src/solver/codegen/unified_assembly.rs` | Unified assembly (with flux) | ~80 lines (ddt block, identical) |
| `crates/cfd2_codegen/src/solver/codegen/ir.rs` | `DiscreteOp` / `DiscreteOpKind` | Will gain `TimeIntegration` metadata |
| `crates/cfd2_ir/src/solver/gpu/enums.rs` | `TimeScheme` enum | May move or be consumed at lowering |
| `crates/cfd2_codegen/src/solver/codegen/constants.rs` | Constants struct | `time_scheme`, `dt_old`, `dtau` fields |

### Duplication inventory

The BDF2 + dtau code block (lines ~380–460 in `generic_coupled_kernels.rs` and
~285–370 in `unified_assembly.rs`) is structurally identical, producing:

- **BDF1 base:** `diag += vol * rho / dt`, `rhs += vol * rho / dt * phi_n`
- **BDF2 correction:** conditional on `time_scheme == BDF2`, replaces diag/rhs
  with variable-step BDF2 coefficients using `dt` and `dt_old`
- **Dual-time:** conditional on `dtau > 0`, adds `vol * rho / dtau` to diag and
  corresponding `phi_iter` term to rhs

---

## Proposed Architecture

```
Model definition
  ── fvm::ddt(u) ──▶  Term { op: Ddt, field: u }
                           │
                           ▼
                     lower_system()
                           │
                           ▼
                     DiscreteOp { kind: TimeDerivative, ... }
                           │
                           ▼
              ┌────────────┴────────────┐
              │  emit_ddt_contributions │  ◄── NEW shared function
              │  (produces AST Stmts)   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌────────────┴────────────┐
              │  main_assembly_fn       │  ◄── calls emit_ddt_contributions()
              │  (generic, scheme-free) │
              └─────────────────────────┘
```

---

## Implementation Plan

### Phase 0: Baseline (before any edits)

Capture OpenFOAM reference metrics + full test suite as baseline.

### Phase 1: Extract shared `emit_ddt_contributions` helper

**Goal:** Eliminate the code duplication without changing semantics.

1. Create a new function in a shared location (e.g., `coupled_common.rs` or a
   new `time_integration.rs` module):

   ```rust
   /// Emit time-derivative AST contributions for one equation component.
   ///
   /// Produces BDF1 base terms, optional BDF2 correction, and optional
   /// dual-time (dtau) terms.  Returns a Vec<Stmt> to be appended to
   /// the assembly body.
   pub fn emit_ddt_stmts(
       acc: &CoupledAccumulators,
       u_idx: u32,
       base_coeff: Expr,          // vol * rho / dt
       dual_time_coeff: Expr,     // vol * rho / dtau
       phi_n: Expr,               // state_old[component]
       phi_nm1: Expr,             // state_old_old[component]
       phi_iter: Expr,            // state_iter[component]
       dt: Expr,
       dt_old: Expr,
       dtau: Expr,
       time_scheme: EnumExpr<TimeScheme>,
   ) -> Vec<Stmt>
   ```

2. Both `generic_coupled_kernels.rs::main_assembly_fn` and
   `unified_assembly.rs::main_assembly_fn` call this shared function instead
   of inlining the BDF1/BDF2/dtau code.

3. **Verification:** All tests pass, OpenFOAM metrics unchanged, generated
   WGSL output is identical.

### Phase 2: Introduce `TimeIntegrationScheme` trait in the IR

**Goal:** Make the time integration scheme a first-class IR concept rather than
a runtime enum checked inside the assembler.

1. Add a `TimeIntegrationContrib` struct to the codegen IR:

   ```rust
   /// Pre-lowered time-integration contribution for one equation component.
   pub struct TimeIntegrationContrib {
       /// Statements to add to the assembly body (may contain if-blocks for
       /// runtime scheme selection).
       pub stmts: Vec<Stmt>,
   }
   ```

2. Add a `TimeIntegrator` trait:

   ```rust
   pub trait TimeIntegrator {
       fn emit_contributions(
           &self,
           acc: &CoupledAccumulators,
           u_idx: u32,
           coeff: Expr,
           phi_n: Expr,
           phi_nm1: Expr,
           phi_iter: Expr,
       ) -> Vec<Stmt>;
   }
   ```

3. Implement `BdfTimeIntegrator` (handles BDF1/BDF2 with runtime switch) and
   `DualTimeIntegrator` (wraps an inner integrator + dtau contribution).

4. The model definition or lowering layer selects the `TimeIntegrator` based
   on solver configuration, rather than the assembler hardcoding the logic.

5. **Verification:** All tests pass, OpenFOAM metrics unchanged.

### Phase 3: Remove time-scheme knowledge from assemblers

**Goal:** The generic assemblers become truly scheme-agnostic for time
integration.

1. `main_assembly_fn` receives a `&dyn TimeIntegrator` (or the pre-lowered
   `TimeIntegrationContrib` per equation) and simply appends the statements
   it produces.  It no longer references `TimeScheme`, `BDF2`, `dtau`, or
   `dt_old` directly.

2. The `constants.time_scheme` and `constants.dt_old` fields remain in the
   Constants struct (they are still consumed at runtime by the emitted
   if-blocks), but the assembler code itself has no knowledge of them.

3. If desired, a future phase can push the BDF1-vs-BDF2 selection entirely
   to the host side (pre-computing coefficients and passing them as uniform
   constants), eliminating the runtime branch from the shader entirely.  This
   is explicitly **out of scope** for this fix.

4. **Verification:** All tests pass, OpenFOAM metrics unchanged.

### Phase 4: Cleanup

1. Remove any dead code from Phase 1's extraction.
2. Update comments and doc-strings.
3. Ensure no `TimeScheme::BDF2` or `dtau.gt(0.0)` patterns exist in the
   assembler files — they should only appear in the time-integration module.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Generated WGSL changes semantically | High | Diff generated shaders before/after each phase |
| BDF2 coefficients subtly change during extraction | High | Existing OpenFOAM tests catch regressions |
| Phase 2 trait design over-engineered for current needs | Medium | Phase 1 alone is valuable; Phase 2/3 can be deferred |
| Constants struct field ordering matters for GPU alignment | Medium | No fields added/removed in Phase 1–3 |
| `dtau` handling interacts with relaxation in update kernel | Low | Update kernel is separate; only assembly is refactored |

---

## Success Criteria

1. **No BDF2/dtau knowledge in assembler files:** After Phase 3,
   `generic_coupled_kernels.rs` and `unified_assembly.rs` should have zero
   references to `TimeScheme`, `BDF2`, or `dtau.gt(0.0)`.

2. **Generated WGSL identical:** Before/after diff of generated WGSL for all
   test cases shows no changes.

3. **Tests pass:** Full test suite (186+ tests) with zero regressions.

4. **OpenFOAM drift:** Zero metric drift across all phases.

5. **Extensibility demonstrated:** Adding a hypothetical BDF3 scheme should
   require changes only in the time-integration module, not in the assemblers.

---

## Out of Scope

- Moving BDF coefficient computation to the host side (eliminating runtime
  if-blocks from generated WGSL)
- Refactoring the `Constants` struct layout
- Addressing the relaxation (`alpha_u` / `alpha_p`) logic in the update kernel
- Addressing the `Scheme` (advection scheme) enum dispatch — same pattern but
  independent concern
- Rhie-Chow's own time-step handling (uses `dt` but not BDF2/dtau)
