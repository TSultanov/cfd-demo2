# ARCH_FIX_3: Replace Type-Erased `ProgramResources` with Strongly-Typed Backend

## Problem Statement

`GpuProgramPlan` uses a type-erased `ProgramResources` bag (`HashMap<TypeId, Box<dyn Any + Send>>`)
to hold backend state. Every handler must downcast at runtime:

```rust
fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    plan.resources
        .get::<UniversalProgramResources>()
        .and_then(|u| u.generic_coupled())
        .expect("missing GenericCoupledProgramResources backend")  // Runtime panic!
}
```

This ECS-like pattern circumvents the borrow checker and moves compile-time type safety to runtime
panics. The type set stored in `ProgramResources` is **fully known at compile time**:

| Stored type | Inserted by | Read by |
|---|---|---|
| `UniversalProgramResources` | `model_driven.rs:101` | `generic_coupled.rs: res(), res_mut()` via `universal.rs: linear_debug_provider()` |
| `Arc<PortRegistry>` | `model_driven.rs:103` | `unified_solver.rs: port_registry()` |

There is exactly **one** backend variant (`GenericCoupled`) wrapped in `UniversalProgramResources`,
and one auxiliary resource (`Arc<PortRegistry>`). No dynamic dispatch is needed.

---

## Proposed Fix

Replace the `HashMap<TypeId, Box<dyn Any>>` with a strongly-typed struct whose fields are known at
compile time.

### Phase 1: Define `PlanResources` struct (safe extraction)

Replace `ProgramResources` with a new struct that has named, typed fields:

```rust
pub(crate) struct PlanResources {
    /// The solver backend (currently always GenericCoupled via UniversalProgramResources).
    pub backend: UniversalProgramResources,
    /// Cached port registry for field offset lookups.
    pub port_registry: Arc<PortRegistry>,
}
```

**Changes:**
- `src/solver/gpu/program/plan.rs` — replace `ProgramResources` struct with `PlanResources`
- `src/solver/gpu/lowering/model_driven.rs` — construct `PlanResources { backend, port_registry }` directly
- `src/solver/gpu/lowering/types.rs` — update `LoweredProgramParts.resources` type

**Result:** `ProgramResources::new()`, `.insert()`, `.get::<T>()`, `.get_mut::<T>()` are all deleted.
No `TypeId`, no `Box<dyn Any>`, no runtime panics.

### Phase 2: Inline `res()` / `res_mut()` accessors

With typed fields, the two-step accessor pattern collapses:

```rust
// Before (runtime downcast)
fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    plan.resources
        .get::<UniversalProgramResources>()
        .and_then(|u| u.generic_coupled())
        .expect("missing GenericCoupledProgramResources backend")
}

// After (compile-time safe)
fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    plan.resources.backend.generic_coupled()
        .expect("missing GenericCoupledProgramResources backend")
}
```

Or, since `UniversalProgramResources` currently wraps only `GenericCoupledProgramResources`:

```rust
fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    &plan.resources.backend.plan
}
```

**Changes:**
- `src/solver/gpu/lowering/programs/generic_coupled.rs` — simplify `res()`, `res_mut()`
- `src/solver/gpu/lowering/programs/universal.rs` — simplify `linear_debug_provider()`
- `src/solver/gpu/unified_solver.rs` — simplify `port_registry()`

### Phase 3: Simplify `UniversalProgramResources` (optional)

Currently `UniversalProgramResources` is a thin wrapper around `GenericCoupledProgramResources`
with `generic_coupled()` / `generic_coupled_mut()` returning `Option<&_>` (always `Some`).

Since there is now only one backend variant, consider:
- **Option A**: Flatten — remove `UniversalProgramResources`, store `GenericCoupledProgramResources`
  directly in `PlanResources.backend`.
- **Option B**: Convert to enum for future extensibility:
  ```rust
  pub(crate) enum SolverBackend {
      GenericCoupled(GenericCoupledProgramResources),
      // future: DirectSolve(...), Multigrid(...), etc.
  }
  ```

Decision: **Option A** is simpler and matches the current codebase. If a second backend is ever
added, converting a struct field to an enum is trivial.

### Phase 4: Cleanup

- Delete `ProgramResources` (the old `HashMap<TypeId, Box<dyn Any>>` implementation)
- Remove `use std::any::{Any, TypeId}` from `plan.rs`
- Update any tests that construct `ProgramResources`

---

## Files Affected

| File | Change |
|------|--------|
| `src/solver/gpu/program/plan.rs` | Replace `ProgramResources` with `PlanResources`; remove `Any`/`TypeId` |
| `src/solver/gpu/lowering/model_driven.rs` | Construct `PlanResources` struct directly |
| `src/solver/gpu/lowering/types.rs` | Update `LoweredProgramParts.resources` type |
| `src/solver/gpu/lowering/programs/generic_coupled.rs` | Simplify `res()` / `res_mut()` |
| `src/solver/gpu/lowering/programs/universal.rs` | Simplify `linear_debug_provider()` |
| `src/solver/gpu/unified_solver.rs` | Simplify `port_registry()` |

---

## Out of Scope

- Refactoring `ProgramOpDispatcher` / `ProgramOpRegistry` (the op-dispatch system uses `&'static str`
  keys, not `TypeId`, and serves a different purpose — it *is* runtime-configured)
- Refactoring `ModelGpuProgramSpec` or `ModelGpuProgramSpecParts` function-pointer tables
- The UI-side `eframe::egui_wgpu::CallbackResources` (separate type, unrelated)
- Adding new solver backend variants (that's future work)

---

## Verification

- **Tests**: All existing tests must pass (186+)
- **WGSL diff**: Not applicable (this is runtime, not codegen)
- **OpenFOAM**: Metrics must not regress
- **Compilation**: Zero `Any`/`TypeId` imports remaining in `plan.rs` after Phase 4

---

## Risk Assessment

**Low risk.** This is a mechanical refactoring:
- The stored types are fully known — no dynamic dispatch discovery needed
- All access sites are already in `pub(crate)` scope within the same crate
- The number of call sites is small (~6 total)
- No behavioral changes — same data, same lifetimes, just type-safe access
