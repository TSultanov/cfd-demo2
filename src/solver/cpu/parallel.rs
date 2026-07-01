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
