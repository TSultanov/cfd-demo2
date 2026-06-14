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
}
