//! Persistent worker pool: the execution mechanism behind [`super::parallel`].
//!
//! The scoped-thread body it replaces spawned and joined fresh OS threads for
//! EVERY parallel region (~100-280 µs per region at 16 threads on this class of
//! hardware, flat in region size). The obstacle bench issues ~15-20k regions
//! per step inside the Schur inner solve alone, which made the CPU backend
//! launch-LATENCY-bound rather than bandwidth-bound: a 118k-cell BLAS-1 op
//! measured 283 µs on 16 scoped threads vs 66 µs serial. Persistent workers
//! park between bursts and spin briefly between back-to-back regions, cutting
//! region launch overhead to the few-µs range.
//!
//! Handoff design notes (each rejected alternative was MEASURED to regress the
//! fat-chunk nozzle regime):
//! - Per-worker `AtomicPtr` job slots + `park`/`unpark`, NOT a shared
//!   mutex+condvar: `notify_all` makes 15 woken workers serially re-acquire one
//!   mutex (thundering-herd convoy), delaying region STARTUP enough to inflate
//!   every fat parallel phase ~10-20%.
//! - DYNAMIC task claiming (atomic counter), with callers OVER-SPLITTING fat
//!   regions into more tasks than workers: this machine class is
//!   heterogeneous (12 P + 4 E cores), so equal static chunks make the E-core
//!   chunks the region's critical path — static assignment measured ~15%
//!   SLOWER than scoped threads at 16 threads (a parked pool worker stays on
//!   its E-core with a full-size chunk; fresh scoped threads get re-placed
//!   every region). With over-split dynamic claiming, fast cores absorb more
//!   tasks and a late-waking or slow worker costs at most one small task, not
//!   a full chunk. Task boundaries never affect results (per-element
//!   arithmetic is index-determined; reductions use fixed sub-chunks).
//! - The job lives on the caller's stack; no per-region allocation.
//!
//! The pool owns NO chunking policy: callers pass a task count and a task body,
//! and every index in `0..tasks` runs exactly once on some thread (the calling
//! thread participates too). Which thread runs which task is irrelevant to
//! results — the bit-exactness contract lives entirely in the callers' chunk
//! math, which is unchanged from the scoped-thread era.
//!
//! Concurrency: one job runs at a time. A caller that finds the pool busy
//! (another solver instance mid-region, or a nested call) executes its tasks
//! inline on its own thread — correct (same arithmetic), merely unaccelerated —
//! so concurrent test solvers and accidental nesting can never deadlock.
//!
//! `CFD2_CPU_POOL=0` restores the scoped-thread mechanism (A/B + escape hatch);
//! `CFD2_CPU_POOL_SPIN` overrides the worker spin budget (0 = park immediately).

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::Thread;

/// Base pointer of a caller-owned `&mut [T]`, for reconstructing DISJOINT
/// sub-slices inside pool tasks (the closure-capture equivalent of the
/// `split_at_mut` chains the scoped-thread code used). Callers guarantee the
/// ranges they reconstruct never overlap and that the backing slice outlives
/// the (blocking) [`run`] call.
pub(crate) struct MutSlicePtr<T>(*mut T);
unsafe impl<T: Send> Send for MutSlicePtr<T> {}
unsafe impl<T: Send> Sync for MutSlicePtr<T> {}

impl<T> MutSlicePtr<T> {
    pub(crate) fn new(s: &mut [T]) -> Self {
        MutSlicePtr(s.as_mut_ptr())
    }

    /// # Safety
    /// `start..start+len` must lie within the original slice, must not overlap
    /// any range reconstructed by a concurrent task, and the original slice
    /// must outlive the use.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// One parallel region. Heap-allocated (`Arc`): each published mailbox slot
/// holds an OWNED reference (`Arc::into_raw`), so a straggler worker that only
/// wakes after the region completed still finds live memory — it claims
/// `next >= tasks` and exits without ever touching the closure. (A stack-job
/// variant segfaulted exactly there: with dynamic claiming, fast threads can
/// finish ALL tasks and let [`run`] return while a slow worker's mailbox still
/// points at the dead stack frame.)
///
/// `data`/`call` type-erase the caller's closure without a fat-pointer
/// transmute: `call` is a monomorphized shim that downcasts. `data` borrows
/// the caller's stack, but is only dereferenced for claimed indices
/// `< tasks`, all of which complete before [`run`] returns (`done == tasks`).
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
    tasks: usize,
    /// Dynamic claim counter (heterogeneous-core load balance; see module docs).
    next: AtomicUsize,
    done: AtomicUsize,
}
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    /// Claim-and-run loop shared by workers and the issuing caller.
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.tasks {
                break;
            }
            // SAFETY: `i` is uniquely claimed and < tasks, so the caller's
            // closure is still alive (see the struct docs) and never runs
            // twice for one index.
            unsafe { (self.call)(self.data, i) };
            // Release: publishes the task's writes to the caller's Acquire
            // load in `run`'s completion wait.
            self.done.fetch_add(1, Ordering::Release);
        }
    }
}

/// Upper bound on pool size; requests beyond it are clamped (tasks still all
/// run, just with less parallelism). Generous vs any real `threads` config.
const MAX_WORKERS: usize = 64;

/// Chunk-helper over-split factor: regions are split into up to
/// `workers * OVERSPLIT` tasks so dynamic claiming can balance heterogeneous
/// (P/E) cores — a slow core finishes early tasks late but only ever holds one
/// small task, not a `1/workers` share of the region. Measured on the 750k-cell
/// nozzle at 16 threads (12P+4E): equal static chunks lose ~15% to the E-core
/// critical path. Task boundaries never affect results (see module docs).
pub(crate) const OVERSPLIT: usize = 4;

struct Shared {
    /// Per-worker job mailboxes. A publish stores the job pointer with
    /// `Release`; the worker claims it with a swap-to-null `Acquire`.
    slots: [AtomicPtr<Job>; MAX_WORKERS],
    /// Grow-on-demand registry of worker thread handles (for `unpark`).
    /// Indexed by worker id; guarded by `grow`.
    handles: Mutex<Vec<Thread>>,
    /// Workers spawned so far (mirror of `handles.len()` readable without the
    /// lock; only written while holding `grow`).
    spawned: AtomicUsize,
}

struct Pool {
    shared: &'static Shared,
    /// Serializes job issuance; contended callers run inline (see module docs).
    issue: Mutex<()>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

fn pool_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CFD2_CPU_POOL").map(|v| v != "0").unwrap_or(true))
}

/// Spin iterations a worker burns waiting for the next region before parking
/// (~tens of µs by default: covers the serial gaps between the back-to-back
/// regions of an inner solve, so mid-solve handoffs skip the unpark syscall).
fn worker_spin() -> usize {
    static SPIN: OnceLock<usize> = OnceLock::new();
    *SPIN.get_or_init(|| {
        std::env::var("CFD2_CPU_POOL_SPIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1 << 14)
    })
}

fn worker_loop(shared: &'static Shared, id: usize) {
    let slot = &shared.slots[id];
    loop {
        let raw = slot.swap(std::ptr::null_mut(), Ordering::Acquire);
        if !raw.is_null() {
            // SAFETY: the slot held an owned reference created by
            // `Arc::into_raw` in `publish`; reconstituting transfers that
            // ownership here (dropped at end of scope).
            let job = unsafe { Arc::from_raw(raw as *const Job) };
            job.work();
            continue;
        }
        // Spin briefly for the next region, then park. `unpark` tokens make
        // the park/publish race lossless (a publish that lands mid-park-entry
        // leaves a token; park returns immediately).
        let mut hot = false;
        for _ in 0..worker_spin() {
            if !slot.load(Ordering::Relaxed).is_null() {
                hot = true;
                break;
            }
            std::hint::spin_loop();
        }
        if !hot {
            std::thread::park();
        }
    }
}

impl Pool {
    fn new() -> Self {
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            slots: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
            handles: Mutex::new(Vec::new()),
            spawned: AtomicUsize::new(0),
        }));
        Pool { shared, issue: Mutex::new(()) }
    }

    /// Grow the pool to `helpers` workers if needed, then hand an owned
    /// reference of `job` to workers `0..helpers`.
    fn publish(&self, job: &Arc<Job>, helpers: usize) {
        let shared = self.shared;
        if shared.spawned.load(Ordering::Relaxed) < helpers {
            let mut handles = shared.handles.lock().unwrap();
            while handles.len() < helpers {
                let id = handles.len();
                let t = std::thread::Builder::new()
                    .name(format!("cfd2-pool-{id}"))
                    .spawn(move || worker_loop(shared, id))
                    .expect("spawn cfd2 cpu pool worker");
                handles.push(t.thread().clone());
                shared.spawned.store(handles.len(), Ordering::Release);
            }
        }
        let handles = shared.handles.lock().unwrap();
        for w in 0..helpers {
            let raw = Arc::into_raw(Arc::clone(job)) as *mut Job;
            let stale = shared.slots[w].swap(raw, Ordering::AcqRel);
            if !stale.is_null() {
                // The worker never woke for an earlier job (its tasks were
                // absorbed by faster threads); reclaim that unconsumed ref.
                // SAFETY: owned reference from a previous `Arc::into_raw`.
                unsafe { drop(Arc::from_raw(stale as *const Job)) };
            }
            // Cheap when the worker is spinning (sets a token, no syscall);
            // a real wake for parked workers.
            handles[w].unpark();
        }
    }
}

/// Execute `f(0)..f(tasks-1)`, each exactly once, on up to `workers` threads
/// (pool workers plus the calling thread); returns after all tasks completed.
/// Callers size `workers` from their work-size guards and `tasks` at a finer
/// granularity (over-splitting) so dynamic claiming can balance heterogeneous
/// cores — see the module docs.
pub(crate) fn run<F: Fn(usize) + Sync>(tasks: usize, workers: usize, f: F) {
    if tasks == 0 {
        return;
    }
    if tasks == 1 || workers <= 1 {
        for i in 0..tasks {
            f(i);
        }
        return;
    }
    if !pool_enabled() {
        run_scoped(tasks, workers, &f);
        return;
    }
    let pool = POOL.get_or_init(Pool::new);
    // Busy pool (concurrent solver or nested region): run inline — same
    // results, no deadlock risk.
    let Ok(_guard) = pool.issue.try_lock() else {
        for i in 0..tasks {
            f(i);
        }
        return;
    };
    unsafe fn shim<F: Fn(usize) + Sync>(data: *const (), i: usize) {
        unsafe { (*(data as *const F))(i) }
    }
    let helpers = (workers.min(tasks) - 1).min(MAX_WORKERS);
    let job = Arc::new(Job {
        data: (&raw const f).cast::<()>(),
        call: shim::<F>,
        tasks,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
    });
    pool.publish(&job, helpers);
    job.work();
    // Wait out stragglers; Acquire pairs with each task's Release increment.
    // Once done == tasks every closure call has completed, so the borrow of
    // `f` may end; any slot refs a sleeping worker never consumed stay valid
    // (heap Arc) and get reclaimed by the next publish.
    let mut spins = 0u32;
    while job.done.load(Ordering::Acquire) != job.tasks {
        spins = spins.wrapping_add(1);
        if spins < 1 << 13 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

/// Scoped-thread mechanism behind `CFD2_CPU_POOL=0` (A/B + escape hatch):
/// fresh threads per region as before the pool, with the same dynamic task
/// claiming as the pool path.
fn run_scoped<F: Fn(usize) + Sync>(tasks: usize, workers: usize, f: &F) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..workers.min(tasks) {
            let next = &next;
            let fr = &f;
            s.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= tasks {
                    break;
                }
                fr(i);
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn run_covers_every_task_once() {
        for &tasks in &[1usize, 2, 3, 7, 16, 33, 64, 200] {
            for &workers in &[2usize, 8, 16] {
                let counts: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                run(tasks, workers, |i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(
                    counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                    "tasks={tasks} workers={workers}: every task must run exactly once"
                );
            }
        }
    }

    #[test]
    fn run_publishes_writes() {
        // Writes from worker-executed tasks must be visible after run returns.
        let n = 100_000usize;
        let tasks = 16usize;
        let chunk = n.div_ceil(tasks);
        let out: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
        run(tasks, tasks, |w| {
            let start = w * chunk;
            let end = (start + chunk).min(n);
            for i in start..end {
                out[i].store((i as u64) * 3 + 1, Ordering::Relaxed);
            }
        });
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v.load(Ordering::Relaxed), (i as u64) * 3 + 1, "index {i}");
        }
    }

    #[test]
    fn run_back_to_back_regions() {
        // Exercises the spin path (regions issued in a tight burst) and the
        // park/wake path (a pause between bursts).
        let acc = AtomicUsize::new(0);
        for burst in 0..3 {
            for _ in 0..500 {
                run(8, 8, |_| {
                    acc.fetch_add(1, Ordering::Relaxed);
                });
            }
            if burst < 2 {
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
        }
        assert_eq!(acc.load(Ordering::Relaxed), 3 * 500 * 8);
    }

    #[test]
    fn run_concurrent_callers_fall_back_inline() {
        // Two threads hammering the pool concurrently must both complete with
        // full coverage (loser of the issue try_lock runs inline).
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    for _ in 0..200 {
                        let counts: Vec<AtomicUsize> =
                            (0..16).map(|_| AtomicUsize::new(0)).collect();
                        run(16, 16, |i| {
                            counts[i].fetch_add(1, Ordering::Relaxed);
                        });
                        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
                    }
                });
            }
        });
    }
}
