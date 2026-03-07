# ARCH_FIX_6: Defensive Programming vs. Panicking

**Issue**: Architecture review item #6 — "Severe inconsistency in how errors are handled."

Two sub-problems:
1. **Setup-path panics**: GPU init and model construction use `.unwrap()` / `.expect()` / `panic!()`
   which crash the UI application when an incompatible configuration is selected.
2. **Silent failures in abstractions**: Some code paths silently substitute defaults (e.g. boundary
   neighbor reads use owner's index, `ping_pong_indices` has an unreachable `_ =>` arm) instead of
   failing explicitly.

## Scope

### Audit summary (non-test `.unwrap()`/`.expect()`/`panic!()`)

| Category | Count | Treatment |
|----------|-------|-----------|
| Test-only | 199 | Leave as-is |
| `Mutex::lock().unwrap()` | 33 | Leave as-is (poisoned mutex = unrecoverable) |
| GPU shader/kernel registry lookups | 29 | **Phase 1** — convert to `Result` |
| GPU init (`context.rs`, `readback.rs`) | 4 | **Phase 2** — convert to `Result` |
| GPU runtime (`generic_coupled.rs`, `fgmres.rs`, etc.) | 13 | **Phase 3** — convert to `Result` |
| Model definitions (`.expect()` in constructors) | 14 | **Phase 4** — convert to `Result` |
| Port registry internal (`"entry exists"`) | 7 | **Phase 5** — convert to internal error |
| Meshgen (sort, map lookup) | 5 | Leave as-is (internal invariant assertions) |
| UI (`app.rs`) | 4 | **Phase 6** — improve error handling |
| `fusion_schedule_registry.rs` `panic!()` | 2 | **Phase 1** — already returns `Result`, just convert panics |

### Out of scope

- `Mutex::lock().unwrap()` — standard Rust idiom; a poisoned mutex means another thread
  panicked, which is already unrecoverable.
- Meshgen internal invariant unwraps (e.g. `on_segment.sort_by(|a, b| a.partial_cmp(&b).unwrap())`)
  — these are algorithmic invariants on finite, non-NaN geometry values.
- Test code — tests are expected to unwrap.

## Phases

### Phase 1: GPU init pipeline construction → `Result` propagation

**Files**: `src/solver/gpu/init/linear_solver/pipelines.rs`, `src/solver/gpu/init/scalars.rs`,
`src/solver/gpu/init/mesh.rs`, `src/solver/gpu/lowering/fusion_schedule_registry.rs`

**What**: Convert `unwrap_or_else(|e| panic!(...))` to `?` propagation.
- `init_pipelines()` → returns `Result<PipelineResources, String>`
- `init_scalars()` → returns `Result<..., String>`
- Mesh diagonal lookup → return `Err` instead of `panic!`
- `fusion_schedule_registry` → remove 2 `panic!()` calls, already returns `Result`

**Size**: ~15 sites, ~40 lines changed.

### Phase 2: GPU context creation → `Result` propagation

**Files**: `src/solver/gpu/context.rs`, `src/solver/gpu/readback.rs`

**What**: `GpuContext::new()` already returns `Self` via async. Change to return
`Result<Self, String>` so missing GPU adapter or failed device creation surfaces as an error
rather than crashing.
- `request_adapter().await.unwrap()` → `request_adapter().await.ok_or("no GPU adapter")?`
- `request_device().await.unwrap()` → `request_device().await.map_err(|e| e.to_string())?`
- `readback.rs` channel recv / map_async unwraps → return `Result`

**Size**: ~4 sites in context.rs, ~5 sites in readback.rs, ~25 lines changed.
**Propagation**: callers of `GpuContext::new()` must handle the Result.

### Phase 3: GPU runtime unwraps → `Result` propagation

**Files**: `src/solver/gpu/lowering/programs/generic_coupled.rs`,
`src/solver/gpu/modules/linear_solver.rs`, `src/solver/gpu/modules/coupled_schur.rs`,
`src/solver/gpu/linear_solver/fgmres.rs`, `src/solver/gpu/linear_solver/amg.rs`,
`src/solver/gpu/modules/scalar_cg.rs`, `src/solver/gpu/modules/generic_coupled_schur.rs`,
`src/solver/gpu/modules/runtime_preconditioner.rs`

**What**: Convert runtime unwraps to `Result`:
- `generic_coupled.rs`: `r.outer_gate.as_ref().unwrap()` (5 sites) → `ok_or("...")?`
- `linear_solver.rs`: `rel_scale.unwrap()` → `ok_or("...")?`
- `coupled_schur.rs`: `.expect("AMG override bind group missing")` → `ok_or("...")?`
- `amg.rs`: `.expect("AMG hierarchy must contain at least one level")` → `ok_or("...")?`
- `fgmres.rs`: channel recv unwraps (2 sites) → `ok_or("...")?`
- `scalar_cg.rs`: channel recv unwraps (4 sites) → `ok_or("...")?`
- `generic_coupled_schur.rs`, `runtime_preconditioner.rs`: map_async channel sends (2 sites)

**Size**: ~18 sites, ~50 lines changed.
**Note**: Many of these functions already return `Result`. The unwraps are in code paths
that *should* be infallible (e.g. `outer_gate` is always `Some` when outer loop is active),
but converting to `Result` prevents obscure panics if invariants are violated.

### Phase 4: Model definition constructors → `Result` propagation

**Files**: `src/solver/model/definitions/incompressible_momentum.rs`,
`src/solver/model/definitions/compressible.rs`,
`src/solver/model/definitions/generic_diffusion_demo.rs`,
`src/solver/model/primitives.rs`, `src/solver/model/kernel.rs`

**What**: Model factory functions (`incompressible_momentum_model()`, `compressible_model()`,
`generic_diffusion_demo_model()`) use `.expect()` for system validation, field registration,
module creation. Convert to `Result<ModelSpec, String>`.
- `incompressible_momentum.rs`: 11 `.expect()` → `?` (function returns `Result`)
- `compressible.rs`: 2 `.expect()` → `?`
- `generic_diffusion_demo.rs`: 2 `.expect()` → `?`
- `primitives.rs`: 8 `.expect()` in topo sort → `?`
- `kernel.rs`: unwraps in kernel emit functions → `?`

**Size**: ~25 sites, ~60 lines changed.
**Propagation**: callers (tests, UI, recipe) must handle `Result`.
**Risk**: Medium — changes public API signatures.

### Phase 5: Port registry internal unwraps → internal errors

**Files**: `src/solver/model/ports/registry.rs`

**What**: 7 sites use `.expect("entry exists")` after a map lookup with an ID that was
just inserted. These are true internal invariant assertions (the ID came from an insert
that just succeeded). Convert to `debug_assert!` + `unwrap_or_else` pattern so they
remain assertions in debug builds but don't abort in release:

```rust
// Before:
let entry = self.param_ports.get(&id).expect("entry exists");
// After:
let entry = self.param_ports.get(&id).ok_or_else(|| {
    format!("internal error: port entry {id:?} not found after insertion")
})?;
```

**Size**: ~7 sites, ~20 lines changed.

### Phase 6: UI error handling improvements

**Files**: `src/ui/app.rs`

**What**: 4 sites:
1. `.expect("default UI model must exist")` → show error in UI status bar
2. `.take().unwrap()` on pending init request → guard with `if let Some(...)`
3. `lock().unwrap()` on renderer (2 sites) — leave as mutex unwrap (standard)

**Size**: ~2 sites to change, ~10 lines.

## Sub-problem B: Silent failures

### Phase 7: Audit and document intentional silent fallbacks

**Files**: `src/solver/gpu/modules/state.rs`, `src/solver/model/modules/flux_module_wgsl.rs`

**What**: The architecture review calls out two items:
1. `ping_pong_indices` has `_ => (0, 1, 2)` — this is an unreachable arm (i % 3 can only
   be 0, 1, or 2). Replace with `unreachable!()` or remove via exhaustive match.
2. Boundary face neighbor reads substitute owner's index — this is **intentional** physics
   (zero-gradient boundary condition), not a bug. Add doc comments explaining the intent.

**Size**: ~5 lines changed.

## Estimated total scope

- ~75 unwrap/expect/panic sites converted to `Result` propagation
- ~14 files modified
- ~200 lines touched
- Risk: **Medium** — Phases 1–3 are mechanical. Phase 4 changes public API signatures.

## Execution order

Phases 1–3 should be done together (GPU stack, bottom-up).
Phase 4 can be done independently (model layer).
Phases 5–7 are small and independent.

Recommended: **Phase 7 → Phase 5 → Phase 2 → Phase 1 → Phase 3 → Phase 4 → Phase 6**
(simplest/safest first, API-breaking changes last)

## Verification

1. `cargo test --features meshgen --lib` — all tests pass
2. `cargo clippy --features meshgen` — no new warnings
3. OpenFOAM reference metrics unchanged (before/after diff)
4. Manual: trigger error conditions (missing GPU adapter, invalid model combo) and verify
   error message surfaces instead of crash
