# ARCH_FIX_7: Fix Potential Memory Leak in Readback Staging Buffer Cache

## Problem (Architecture Review Issue #7)

`StagingBufferCache` in `src/solver/gpu/readback.rs` stores GPU staging buffers
keyed by byte-size in a `HashMap<u64, wgpu::Buffer>`.  Buffers are borrowed via
`take_or_create` and returned via `put`, but there is **no eviction policy**.

If the set of requested sizes changes over the lifetime of a solver (e.g. dynamic
mesh refinement, switching between different equation systems, or multiple solves
with different meshes in the same process), old buffers accumulate in the map and
are never freed — leaking GPU memory.

There are two independent cache instances:
1. `GpuProgramPlan::staging_cache` — used by the main solver loop
   (outer-convergence readback, adaptive iteration counters)
2. `GpuRuntimeCommon::readback_cache` — used by general `read_buffer` calls

Both exhibit the same leak pattern.

### Current call sites

| Location | Sizes used |
|----------|-----------|
| `generic_coupled.rs` outer convergence bits | `(zero_out_words.len() * 4)` — mesh-dependent |
| `generic_coupled.rs` break status | constant `4` |
| `generic_coupled.rs` iter counter readback | constant `4` |
| `plan.rs` `read_buffer` | arbitrary caller-provided size |
| `runtime_common.rs` `read_buffer` | arbitrary caller-provided size |

In a single-mesh steady-state run the leak is benign: only a handful of sizes
are ever requested.  The risk appears when sizes change across solver
re-creations or dynamic scenarios.

---

## Goals

1. **Bounded memory**: guarantee the cache cannot grow without limit.
2. **Zero regression**: cache hit rate for the common case (fixed-mesh,
   fixed-equation) must stay identical.
3. **Minimal API change**: callers (`take_or_create` / `put`) should not need
   to change.

---

## Plan

### Phase 1: Add `clear()` and call it on solver re-creation

Add a `StagingBufferCache::clear()` method that drops all cached buffers, and
call it when a new solver / program plan is constructed.  This is the minimal
fix: if the mesh changes, the cache is reset.

**Files**: `readback.rs`, `plan.rs`, `runtime_common.rs`

### Phase 2: Add an LRU eviction cap

Replace `HashMap<u64, wgpu::Buffer>` with a bounded container that evicts the
least-recently-used entry when the number of distinct sizes exceeds a threshold
(e.g. 16).  This caps worst-case GPU memory waste without affecting the
steady-state fast path.

Implementation options (in order of preference):
- **a)** Use a `Vec<(u64, wgpu::Buffer)>` with move-to-front on hit and
  truncation at capacity.  Simple, no extra deps, O(n) lookup with n ≤ 16.
- **b)** Pull in the `lru` crate for a proper O(1) LRU map.
- **c)** Use a `BTreeMap` with an access-time sidecar for eviction.

Option (a) is preferred given the tiny expected working set.

**Files**: `readback.rs`

### Phase 3: Add diagnostic / metric tracking (optional)

Log or expose cache statistics (hits, misses, evictions, total GPU bytes held)
behind the existing `profiling` feature flag.  This gives visibility into
whether the cache is effective.

**Files**: `readback.rs`, possibly `profiling.rs`

---

## Verification

- `cargo build` clean
- `cargo test` — no regressions
- OpenFOAM reference metrics — zero drift (this change is runtime-only,
  should not affect numerics)
