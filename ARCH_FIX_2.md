# ARCH_FIX_2: Decouple Time Integration from Generic Assemblers — COMPLETE

## Problem Statement

The "generic" coupled assembly kernels (`generic_coupled_kernels.rs` and
`unified_assembly.rs`) hardcoded BDF2 math, dual-time stepping (`dtau`), and
runtime `time_scheme` enum dispatch — ~80 lines copy-pasted between both files.

A generic assembler should not know about BDF2 or `dtau`. It should process
IR-level contributions where the time-integration math has been factored out.

---

## Status: ✅ COMPLETE

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ DONE | Baseline captured (50 WGSL files + OpenFOAM metrics) |
| Phase 1 | ✅ DONE | Extract shared `emit_ddt_contributions` into `coupled_common.rs` |
| Phase 2 | ✅ DONE | Introduce `TimeIntegrator` trait + `BdfDualTimeIntegrator` in new `time_integration.rs` |
| Phase 3 | ✅ DONE | Assembler files have zero `TimeScheme`/`BDF2`/`dtau` references in assembly logic |
| Phase 4 | ✅ DONE | Cleanup: unused imports removed, doc-strings updated |

### Verification

- **Generated WGSL:** Identical before/after (all 50 shaders diffed)
- **Tests:** 186+ pass (only pre-existing `block_jacobi` failure)
- **Crate tests:** 246 pass (168 codegen + 78 ir)
- **OpenFOAM drift:** Zero

---

## Architecture (Final)

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
              │ time_integration.rs     │
              │ emit_ddt_contributions  │
              │  ├─ iterates equations  │
              │  └─ delegates to:       │
              │    BdfDualTimeIntegrator │  ← implements TimeIntegrator trait
              └────────────┬────────────┘
                           │
                           ▼
              ┌────────────┴────────────┐
              │  main_assembly_fn       │  ← calls emit_ddt_contributions()
              │  (scheme-agnostic)      │     via coupled_common wrapper
              └─────────────────────────┘
```

### New module: `time_integration.rs`

| Type | Description |
|------|-------------|
| `TimeIntegrator` trait | `emit_component(acc, u_idx, base_coeff, dual_time_coeff, phi_n, phi_nm1, phi_iter) -> Vec<Stmt>` |
| `BdfDualTimeIntegrator` | BDF1 base + runtime BDF2 correction + dtau pseudo-transient |
| `emit_ddt_contributions()` | Driver: loops equations, computes coefficients, calls `&dyn TimeIntegrator` |

### Extensibility

Adding a new time integration scheme (e.g., Crank-Nicolson) requires:
1. Implement `TimeIntegrator` for the new scheme
2. Pass it to `emit_ddt_contributions()` instead of `BdfDualTimeIntegrator`

No changes needed in `generic_coupled_kernels.rs` or `unified_assembly.rs`.

---

## Files Changed

| File | Change |
|------|--------|
| `crates/cfd2_codegen/src/solver/codegen/time_integration.rs` | **NEW** — trait + BDF implementation + driver |
| `crates/cfd2_codegen/src/solver/codegen/mod.rs` | Register new module |
| `crates/cfd2_codegen/src/solver/codegen/coupled_common.rs` | Thin wrapper delegates to `time_integration` |
| `crates/cfd2_codegen/src/solver/codegen/generic_coupled_kernels.rs` | Removed 80-line inline ddt block, calls wrapper |
| `crates/cfd2_codegen/src/solver/codegen/unified_assembly.rs` | Removed 80-line inline ddt block, calls wrapper |
