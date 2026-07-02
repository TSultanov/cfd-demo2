//! Runtime-selectable parallel-for over a kernel dispatch domain.
//!
//! All threading goes through this single module so the mechanism stays
//! swappable. The user is (rightly) skeptical of per-dispatch fork/join overhead
//! from a work-stealing pool, so the helpers split the domain into a few
//! **coarse contiguous chunks**; each worker owns a cache-friendly cell/face
//! range with no per-index task overhead. Execution runs on the persistent
//! worker pool in [`super::pool`] (parked threads, ~1-5 µs region launch); the
//! previous scoped-thread spawn+join (~100-280 µs per region at 16 threads —
//! the measured wall of the Schur inner solve) remains available via
//! `CFD2_CPU_POOL=0`.
//!
//! Soundness: callers run CPU kernels that write disjoint per-cell/face buffer
//! slots, and the buffer store is relaxed-atomic, so concurrent invocations for
//! distinct indices never race. Determinism: every chunk split below depends
//! only on `n`/`threads` and fixed constants — never on which pool thread runs
//! a chunk — and each output element is produced exactly once with identical
//! arithmetic, so results are bit-identical across thread counts and across
//! pool/scoped mechanisms.

use super::pool;

/// Run `f(idx)` for every `idx` in `0..n`, using up to `threads` workers.
pub fn parallel_for<F>(n: usize, threads: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if threads <= 1 || n <= 1 {
        for idx in 0..n {
            f(idx);
        }
        return;
    }

    let workers = threads.min(n);
    let chunk = n.div_ceil(workers * pool::OVERSPLIT).max(1);
    let tasks = n.div_ceil(chunk);
    pool::run(tasks, workers, |w| {
        let start = w * chunk;
        let end = (start + chunk).min(n);
        for idx in start..end {
            f(idx);
        }
    });
}

/// Run `f(start, end)` over contiguous disjoint index ranges covering `0..n`,
/// using up to `threads` workers — the range-granular sibling of
/// [`parallel_for`] (identical chunk math) for callers that amortize per-chunk
/// setup, e.g. the transpiled kernels' chunk-range entry points, which resolve
/// their buffer handles once per range instead of once per index.
pub fn parallel_ranges<F>(n: usize, threads: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if threads <= 1 || n <= 1 {
        f(0, n);
        return;
    }
    let workers = threads.min(n);
    let chunk = n.div_ceil(workers * pool::OVERSPLIT).max(1);
    let tasks = n.div_ceil(chunk);
    pool::run(tasks, workers, |w| {
        let start = w * chunk;
        let end = (start + chunk).min(n);
        f(start, end);
    });
}

/// Minimum output elements per worker for the fine-grained (BLAS-1 style)
/// parallel helpers. Below this, region-launch latency exceeds the memory-bound
/// work itself, so the helpers scale the worker count down (bit-exactness is
/// unaffected: every element is still produced by identical arithmetic).
/// 8k f64 ≈ 64 KB per worker ≈ the break-even for a persistent-pool region
/// (~1-5 µs launch) against ~10 GB/s-per-core streaming. (The scoped-thread era
/// used 64k: spawn+join cost ~20-30 µs/worker; the pool moves the knee down and
/// lets mid-size vectors — e.g. the 118k-cell obstacle's BLAS-1 — actually use
/// the machine instead of being capped at 1-2 workers.)
const MIN_ELEMS_PER_WORKER: usize = 8 * 1024;

/// Same idea for the row-wise chunk helpers ([`parallel_cell_chunks_mut`] and
/// friends), whose per-element work (a CSR row gather, a block solve) is
/// several times heavier than BLAS-1 streaming: coarse AMG levels (a few k
/// rows) run serial instead of fanning out 16 workers for microseconds of
/// work, while every production-size dispatch keeps its full worker count.
const MIN_CELL_ELEMS_PER_WORKER: usize = 4 * 1024;

/// Worker count for the deterministic dot helpers (shared with the f32
/// mixed-precision twin in `linalg`).
#[inline]
pub(crate) fn par_dot_workers(threads: usize, total_elems: usize) -> usize {
    effective_workers(threads, total_elems)
}

#[inline]
fn effective_workers(threads: usize, total_elems: usize) -> usize {
    threads
        .min(total_elems.div_ceil(MIN_ELEMS_PER_WORKER))
        .max(1)
}

#[inline]
fn effective_row_workers(threads: usize, total_elems: usize) -> usize {
    threads
        .min(total_elems.div_ceil(MIN_CELL_ELEMS_PER_WORKER))
        .max(1)
}

/// Run `f(chunk_start_cell, y_chunk)` over contiguous cell ranges, where the
/// output `y` (length `num_cells * width`) is split into disjoint per-range
/// mutable sub-slices. Each worker owns `y[start*width .. end*width]` and no
/// other, so writes never race — the caller is responsible only for reading
/// shared immutable inputs. `width` is the per-cell stride (S for block-CSR, 1
/// for scalar). The split is the same coarse contiguous chunking as
/// [`parallel_for`]; results are independent of `threads` (each output element is
/// produced by exactly one worker with the identical arithmetic).
pub fn parallel_cell_chunks_mut<T, F>(num_cells: usize, width: usize, threads: usize, y: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    debug_assert_eq!(y.len(), num_cells * width, "y must be num_cells*width");
    if threads <= 1 || num_cells <= 1 {
        f(0, y);
        return;
    }
    let workers = effective_row_workers(threads, num_cells * width).min(num_cells);
    if workers <= 1 {
        f(0, y);
        return;
    }
    let chunk = num_cells.div_ceil(workers * pool::OVERSPLIT).max(1);
    let tasks = num_cells.div_ceil(chunk);
    let base = pool::MutSlicePtr::new(y);
    pool::run(tasks, workers, |w| {
        let start = w * chunk;
        let end = (start + chunk).min(num_cells);
        // SAFETY: tasks own disjoint cell ranges, so the reconstructed
        // sub-slices never overlap; `base` outlives the (blocking) run call.
        let head = unsafe { base.slice(start * width, (end - start) * width) };
        f(start, head);
    });
}

/// Like [`parallel_cell_chunks_mut`], but splits TWO output slices (with
/// per-cell widths `w1`, `w2`) over the SAME contiguous cell ranges, so a single
/// pass can produce two disjoint per-cell outputs (e.g. the Schur velocity
/// predict `z` and the Schur RHS `gp`). Same determinism guarantee: each output
/// element is written by exactly one worker with identical arithmetic.
pub fn parallel_cell_chunks_mut2<F>(
    num_cells: usize,
    w1: usize,
    w2: usize,
    threads: usize,
    y1: &mut [f64],
    y2: &mut [f64],
    f: F,
) where
    F: Fn(usize, &mut [f64], &mut [f64]) + Sync,
{
    debug_assert_eq!(y1.len(), num_cells * w1, "y1 must be num_cells*w1");
    debug_assert_eq!(y2.len(), num_cells * w2, "y2 must be num_cells*w2");
    if threads <= 1 || num_cells <= 1 {
        f(0, y1, y2);
        return;
    }
    let workers = effective_row_workers(threads, num_cells * (w1 + w2)).min(num_cells);
    if workers <= 1 {
        f(0, y1, y2);
        return;
    }
    let chunk = num_cells.div_ceil(workers * pool::OVERSPLIT).max(1);
    let tasks = num_cells.div_ceil(chunk);
    let base1 = pool::MutSlicePtr::new(y1);
    let base2 = pool::MutSlicePtr::new(y2);
    pool::run(tasks, workers, |w| {
        let start = w * chunk;
        let end = (start + chunk).min(num_cells);
        // SAFETY: disjoint cell ranges per task; bases outlive the run call.
        let head1 = unsafe { base1.slice(start * w1, (end - start) * w1) };
        let head2 = unsafe { base2.slice(start * w2, (end - start) * w2) };
        f(start, head1, head2);
    });
}

/// Deterministic parallel dot product. The summation is FIXED-CHUNKED: partial
/// sums are computed per `DOT_CHUNK`-element chunk (4-wide SIMD within a chunk)
/// and then reduced serially in chunk order. The result depends only on the
/// input (and the fixed chunk size) — NOT on `threads` — so the linear solvers
/// stay bit-identical across thread counts while the O(n) reduction runs on all
/// cores. (This is a different summation ORDER from a plain serial loop, i.e. a
/// one-time rounding-level change, validated by the tolerance-based suites.)
pub fn par_dot(threads: usize, a: &[f64], b: &[f64]) -> f64 {
    const DOT_CHUNK: usize = 8192;
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let nchunks = n.div_ceil(DOT_CHUNK).max(1);
    let chunk_partial = |c: usize| -> f64 {
        let s = c * DOT_CHUNK;
        let e = (s + DOT_CHUNK).min(n);
        dot_simd_range(&a[s..e], &b[s..e])
    };
    let workers = effective_workers(threads, n);
    if workers <= 1 || nchunks == 1 {
        return (0..nchunks).map(chunk_partial).sum();
    }
    let mut partials = vec![0.0f64; nchunks];
    let per = nchunks.div_ceil(workers * pool::OVERSPLIT).max(1);
    let tasks = nchunks.div_ceil(per);
    let base = pool::MutSlicePtr::new(&mut partials);
    pool::run(tasks, workers, |w| {
        let c0 = w * per;
        let c1 = (c0 + per).min(nchunks);
        // SAFETY: disjoint partial ranges per task; `partials` outlives run.
        let head = unsafe { base.slice(c0, c1 - c0) };
        for (li, o) in head.iter_mut().enumerate() {
            *o = chunk_partial(c0 + li);
        }
    });
    // Serial reduction in fixed chunk order — deterministic.
    partials.iter().sum()
}

/// 4-wide f64 SIMD dot over one chunk (the per-chunk kernel of [`par_dot`]).
/// FP summation is non-associative, so this fixed lane/tail order is part of
/// the determinism contract.
#[inline]
fn dot_simd_range(a: &[f64], b: &[f64]) -> f64 {
    use wide::f64x4;
    let n = a.len();
    let mut acc = f64x4::splat(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = f64x4::from([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let vb = f64x4::from([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        acc += va * vb;
        i += 4;
    }
    let lanes = acc.to_array();
    let mut s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Parallel elementwise map-into: `out[i] = f(i)` for every `i`, over contiguous
/// disjoint index chunks. Each element is produced by exactly one worker with the
/// same arithmetic, so the result is BIT-IDENTICAL to the serial loop regardless
/// of `threads` (no reduction / summation reorder). `f` reads only shared
/// immutable inputs.
pub fn par_map_into<F>(threads: usize, out: &mut [f64], f: F)
where
    F: Fn(usize) -> f64 + Sync,
{
    let n = out.len();
    let threads = effective_workers(threads, n);
    if threads <= 1 || n <= 1 {
        for (i, o) in out.iter_mut().enumerate() {
            *o = f(i);
        }
        return;
    }
    let workers = threads.min(n);
    let chunk = n.div_ceil(workers * pool::OVERSPLIT).max(1);
    let tasks = n.div_ceil(chunk);
    let base = pool::MutSlicePtr::new(out);
    pool::run(tasks, workers, |w| {
        let start = w * chunk;
        let end = (start + chunk).min(n);
        // SAFETY: disjoint index ranges per task; `out` outlives run.
        let head = unsafe { base.slice(start, end - start) };
        for (li, o) in head.iter_mut().enumerate() {
            *o = f(start + li);
        }
    });
}

/// Parallel elementwise in-place update: `f(i, &mut out[i])` for every `i`, over
/// contiguous disjoint index chunks. BIT-IDENTICAL to the serial loop (each
/// element updated once, same arithmetic). `f` reads only shared immutable inputs
/// besides its own `&mut` element.
pub fn par_update<F>(threads: usize, out: &mut [f64], f: F)
where
    F: Fn(usize, &mut f64) + Sync,
{
    let n = out.len();
    let threads = effective_workers(threads, n);
    if threads <= 1 || n <= 1 {
        for (i, o) in out.iter_mut().enumerate() {
            f(i, o);
        }
        return;
    }
    let workers = threads.min(n);
    let chunk = n.div_ceil(workers * pool::OVERSPLIT).max(1);
    let tasks = n.div_ceil(chunk);
    let base = pool::MutSlicePtr::new(out);
    pool::run(tasks, workers, |w| {
        let start = w * chunk;
        let end = (start + chunk).min(n);
        // SAFETY: disjoint index ranges per task; `out` outlives run.
        let head = unsafe { base.slice(start, end - start) };
        for (li, o) in head.iter_mut().enumerate() {
            f(start + li, o);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_for_covers_every_index_once() {
        for &threads in &[1usize, 2, 4, 8] {
            let n = 1000usize;
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            parallel_for(n, threads, |i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                "threads={threads}: every index must run exactly once"
            );
        }
    }

    #[test]
    fn parallel_for_sum_matches_serial() {
        let n = 10_000usize;
        let acc = AtomicUsize::new(0);
        parallel_for(n, 4, |i| {
            acc.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(acc.load(Ordering::Relaxed), (0..n).sum::<usize>());
    }

    #[test]
    fn cell_chunks_mut_covers_and_matches_serial() {
        // width=3 (block stride): each cell's 3 slots filled from its global index;
        // threaded result must equal the serial fill exactly. Sizes straddle the
        // work-size guard so both the serial-collapse and multi-worker paths run.
        for &num_cells in &[1000usize, 50_000] {
            let width = 3usize;
            let expect: Vec<f64> = (0..num_cells * width).map(|k| k as f64).collect();
            for &threads in &[1usize, 3, 4, 7, 16] {
                let mut y = vec![-1.0f64; num_cells * width];
                parallel_cell_chunks_mut(num_cells, width, threads, &mut y, |cell0, chunk| {
                    for (li, slot) in chunk.iter_mut().enumerate() {
                        *slot = ((cell0 * width) + li) as f64;
                    }
                });
                assert_eq!(y, expect, "num_cells={num_cells} threads={threads}");
            }
        }
    }

    #[test]
    fn par_dot_thread_count_invariant() {
        // par_dot's fixed-chunk summation must be BIT-IDENTICAL across thread
        // counts (including 1): the chunk partials and their reduction order are
        // independent of the worker split.
        for &n in &[1usize, 100, 8192, 8193, 100_000, 1_000_000] {
            let a: Vec<f64> = (0..n).map(|i| ((i % 97) as f64) * 0.37 - 1.0).collect();
            let b: Vec<f64> = (0..n).map(|i| ((i % 89) as f64) * -0.21 + 0.5).collect();
            let ref_v = par_dot(1, &a, &b);
            for &threads in &[2usize, 3, 8, 16] {
                let v = par_dot(threads, &a, &b);
                assert!(
                    v.to_bits() == ref_v.to_bits(),
                    "n={n} threads={threads}: {v:e} != {ref_v:e}"
                );
            }
        }
    }

    #[test]
    fn cell_chunks_mut2_covers_and_matches_serial() {
        let (num_cells, w1, w2) = (5000usize, 3usize, 1usize);
        let e1: Vec<f64> = (0..num_cells * w1).map(|k| k as f64).collect();
        let e2: Vec<f64> = (0..num_cells * w2).map(|k| (k * 2) as f64).collect();
        for &threads in &[1usize, 4, 16] {
            let mut y1 = vec![-1.0f64; num_cells * w1];
            let mut y2 = vec![-1.0f64; num_cells * w2];
            parallel_cell_chunks_mut2(num_cells, w1, w2, threads, &mut y1, &mut y2, |c0, a, b| {
                for (li, slot) in a.iter_mut().enumerate() {
                    *slot = (c0 * w1 + li) as f64;
                }
                for (li, slot) in b.iter_mut().enumerate() {
                    *slot = ((c0 * w2 + li) * 2) as f64;
                }
            });
            assert_eq!(y1, e1, "threads={threads}");
            assert_eq!(y2, e2, "threads={threads}");
        }
    }

    #[test]
    fn par_map_and_update_bit_identical() {
        let n = 9973usize; // prime-ish, forces uneven final chunk
        let a: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5 - 3.0).collect();
        let serial_map: Vec<f64> = a.iter().map(|&v| v * v - 0.25).collect();
        for &threads in &[1usize, 2, 4, 8, 16] {
            let mut out = vec![0.0f64; n];
            par_map_into(threads, &mut out, |i| a[i] * a[i] - 0.25);
            assert_eq!(out, serial_map, "map threads={threads}");

            let mut upd = a.clone();
            par_update(threads, &mut upd, |i, o| *o = *o * 2.0 + (i as f64));
            let serial_upd: Vec<f64> = a.iter().enumerate().map(|(i, &v)| v * 2.0 + i as f64).collect();
            assert_eq!(upd, serial_upd, "update threads={threads}");
        }
    }
}
