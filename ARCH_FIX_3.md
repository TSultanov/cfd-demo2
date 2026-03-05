# ARCH_FIX_3: Replace Type-Erased `ProgramResources` with Strongly-Typed Backend — COMPLETE

## Status: ✅ COMPLETE (single commit)

Replaced `HashMap<TypeId, Box<dyn Any + Send>>` with a strongly-typed `PlanResources` struct.
Flattened the `UniversalProgramResources` wrapper. Removed all `Any`/`TypeId` imports.

### What changed

| File | Before | After |
|------|--------|-------|
| `program/plan.rs` | `ProgramResources` with `insert::<T>()`, `get::<T>()` via `Any` | `PlanResources { backend, port_registry }` — two typed fields |
| `generic_coupled.rs` | `res()` → `.get::<Universal>().and_then(\|u\| u.generic_coupled()).expect(…)` | `res()` → `&plan.resources.backend` |
| `universal.rs` | `UniversalProgramResources` wrapper struct + `PlanLinearSystemDebug` delegate | Deleted — backend stored directly |
| `unified_solver.rs` | `.get::<Arc<PortRegistry>>().map(…)` | `Some(plan.resources.port_registry.as_ref())` |
| `model_driven.rs` | `ProgramResources::new()` + two `.insert()` calls | `PlanResources { backend, port_registry }` struct literal |
| `lowering/types.rs` | `resources: ProgramResources` | `resources: PlanResources` |

### Verification

- **Tests**: 186+ pass (only pre-existing `block_jacobi` failure)
- **OpenFOAM**: Zero metric drift
- **Removed**: `Any`, `TypeId`, all runtime downcast paths
- **Net**: -77 lines
