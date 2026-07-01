//! Runtime-selectable parallel-for over a kernel dispatch domain.
//!
//! All threading goes through this single function so the mechanism stays
//! swappable. The user is (rightly) skeptical of per-dispatch fork/join overhead
//! from a work-stealing pool, so the default deliberately avoids rayon: it splits
//! the domain into a few **coarse contiguous chunks** and runs them on scoped
//! threads. Each worker owns a cache-friendly cell/face range; there is no
//! per-index task overhead. A persistent worker pool could replace the body here
//! without touching any kernel or the interpreter.
//!
//! Soundness: callers run CPU kernels that write disjoint per-cell/face buffer
//! slots, and the buffer store is relaxed-atomic, so concurrent invocations for
//! distinct indices never race.

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
    let chunk = n.div_ceil(workers);
    std::thread::scope(|s| {
        for w in 0..workers {
            let start = w * chunk;
            if start >= n {
                break;
            }
            let end = (start + chunk).min(n);
            let fr = &f;
            s.spawn(move || {
                for idx in start..end {
                    fr(idx);
                }
            });
        }
    });
}

/// Minimum output elements per worker for the fine-grained (BLAS-1 style)
/// parallel helpers. Below this, thread-spawn latency exceeds the memory-bound
/// work itself, so the helpers scale the worker count down (bit-exactness is
/// unaffected: every element is still produced by identical arithmetic).
/// 64k f64 ≈ 512 KB per worker ≈ the measured break-even for scoped-thread
/// spawn+join (~20-30 µs/worker) against ~10 GB/s-per-core streaming.
const MIN_ELEMS_PER_WORKER: usize = 64 * 1024;

#[inline]
fn effective_workers(threads: usize, total_elems: usize) -> usize {
    threads
        .min(total_elems.div_ceil(MIN_ELEMS_PER_WORKER))
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
pub fn parallel_cell_chunks_mut<F>(num_cells: usize, width: usize, threads: usize, y: &mut [f64], f: F)
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    debug_assert_eq!(y.len(), num_cells * width, "y must be num_cells*width");
    if threads <= 1 || num_cells <= 1 {
        f(0, y);
        return;
    }
    let workers = threads.min(num_cells);
    let chunk = num_cells.div_ceil(workers);
    std::thread::scope(|s| {
        let mut rest: &mut [f64] = y;
        let mut start = 0usize;
        while start < num_cells {
            let end = (start + chunk).min(num_cells);
            let take = (end - start) * width;
            let (head, tail) = rest.split_at_mut(take);
            rest = tail;
            let fr = &f;
            s.spawn(move || fr(start, head));
            start = end;
        }
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
    let workers = effective_workers(threads, num_cells * (w1 + w2)).min(num_cells);
    if workers <= 1 {
        f(0, y1, y2);
        return;
    }
    let chunk = num_cells.div_ceil(workers);
    std::thread::scope(|s| {
        let mut rest1: &mut [f64] = y1;
        let mut rest2: &mut [f64] = y2;
        let mut start = 0usize;
        while start < num_cells {
            let end = (start + chunk).min(num_cells);
            let (head1, tail1) = rest1.split_at_mut((end - start) * w1);
            let (head2, tail2) = rest2.split_at_mut((end - start) * w2);
            rest1 = tail1;
            rest2 = tail2;
            let fr = &f;
            s.spawn(move || fr(start, head1, head2));
            start = end;
        }
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
    let per = nchunks.div_ceil(workers);
    std::thread::scope(|s| {
        let mut rest: &mut [f64] = &mut partials;
        let mut c0 = 0usize;
        while c0 < nchunks {
            let c1 = (c0 + per).min(nchunks);
            let (head, tail) = rest.split_at_mut(c1 - c0);
            rest = tail;
            let cp = &chunk_partial;
            s.spawn(move || {
                for (li, o) in head.iter_mut().enumerate() {
                    *o = cp(c0 + li);
                }
            });
            c0 = c1;
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
    let chunk = n.div_ceil(workers);
    std::thread::scope(|s| {
        let mut rest: &mut [f64] = out;
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let (head, tail) = rest.split_at_mut(end - start);
            rest = tail;
            let fr = &f;
            s.spawn(move || {
                for (li, o) in head.iter_mut().enumerate() {
                    *o = fr(start + li);
                }
            });
            start = end;
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
    let chunk = n.div_ceil(workers);
    std::thread::scope(|s| {
        let mut rest: &mut [f64] = out;
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let (head, tail) = rest.split_at_mut(end - start);
            rest = tail;
            let fr = &f;
            s.spawn(move || {
                for (li, o) in head.iter_mut().enumerate() {
                    fr(start + li, o);
                }
            });
            start = end;
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
        // threaded result must equal the serial fill exactly.
        let (num_cells, width) = (1000usize, 3usize);
        let expect: Vec<f64> = (0..num_cells * width).map(|k| k as f64).collect();
        for &threads in &[1usize, 3, 4, 7, 16] {
            let mut y = vec![-1.0f64; num_cells * width];
            parallel_cell_chunks_mut(num_cells, width, threads, &mut y, |cell0, chunk| {
                for (li, slot) in chunk.iter_mut().enumerate() {
                    *slot = ((cell0 * width) + li) as f64;
                }
            });
            assert_eq!(y, expect, "threads={threads}");
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
