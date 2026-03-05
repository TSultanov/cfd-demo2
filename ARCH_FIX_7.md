# ARCH_FIX_7: Fix Potential Memory Leak in Readback Staging Buffer Cache

**Status: COMPLETE**

## Problem (Architecture Review Issue #7)

`StagingBufferCache` in `src/solver/gpu/readback.rs` stores GPU staging buffers
keyed by byte-size in a `HashMap<u64, wgpu::Buffer>`.  Buffers are borrowed via
`take_or_create` and returned via `put`, but there is **no eviction policy**.

If the set of requested sizes changes over the lifetime of a solver (e.g. dynamic
mesh refinement, switching between different equation systems, or multiple solves
with different meshes in the same process), old buffers accumulate in the map and
are never freed — leaking GPU memory.

## Solution

Replaced the unbounded `HashMap<u64, wgpu::Buffer>` with a bounded LRU cache
backed by a `Vec<(u64, wgpu::Buffer)>` (capacity 16).  Most-recently-used
entries are kept at the front; when the cache exceeds capacity the
least-recently-used entry is evicted (dropped, releasing GPU memory).

### Changes

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Added `clear()` method on `StagingBufferCache` | ✅ |
| 2 | Replaced `HashMap` with bounded LRU `Vec` (capacity 16) | ✅ |
| 3 | Added `CacheCounters` (hits/misses/evictions/cached_bytes) via `stats()` | ✅ |

### Files changed

- `src/solver/gpu/readback.rs` — Rewrote `StagingBufferCache` with LRU eviction,
  `clear()`, `stats()`, `CacheCounters`, and unit tests.

### Verification

- `cargo build` — clean
- `cargo test` — all pass (1 pre-existing contract test failure unchanged)
- OpenFOAM reference metrics — zero drift (identical before/after)
