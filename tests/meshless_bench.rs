//! Meshless-engine performance report (M0.6, design §8 budget table).
//!
//! Times every engine phase on the ChannelWithObstacle geometry (the class
//! the roadmap budgets were written for) at ~75k and ~300k cells, plus the
//! incumbent `generate_voronoi_mesh` baseline the CVT total is budgeted
//! against. Reports the full table for the roadmap's acceptance review
//! (diagram ≤ 100 ms @300k, Lloyd iter ≤ 120 ms, assembly ≤ 250 ms, CVT
//! total ≤ incumbent wall time, all at 16T), and ASSERTS only at 2× budget
//! (the roadmap's "red flag" threshold) on the ≥250k-seed case — generous
//! enough not to flake on machine noise (measured headroom ≥ 3×), loud
//! enough that a real regression fails instead of hiding in a report.
//!
//! Poisson seeding is timed separately on purpose: it is inherently
//! sequential, common to BOTH pipelines, and must not be booked as engine
//! speedup (design §8).
//!
//! Run (release is what the budget table means):
//!
//! ```sh
//! cargo test --release --features "dev-tests meshgen" \
//!     --test meshless_bench -- --nocapture
//! ```
//!
//! Env knobs:
//!   CFD2_BENCH_SIZE  single target cell size (default: 0.005 AND 0.0025,
//!                    ≈75k and ≈300k cells on the 3×1 obstacle channel)
//!   CFD2_BENCH_REPS  repetitions for the cheap phases (default 5; the
//!                    reported number is the min — the honest "cost when
//!                    not fighting the machine" statistic)

#![cfg(all(feature = "dev-tests", feature = "meshgen"))]

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, lloyd_relax, meshless_seed_points, EngineConfig, LloydConfig,
    MeshlessInput, SeedGrid,
};
use cfd2::meshgen::{generate_voronoi_mesh, ChannelWithObstacle, Geometry, MeshgenTolerances};
use nalgebra::{Point2, Vector2};
use std::time::Instant;

const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;

fn obstacle() -> (ChannelWithObstacle, Vector2<f64>) {
    (
        ChannelWithObstacle {
            length: LENGTH,
            height: HEIGHT,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        },
        Vector2::new(LENGTH, HEIGHT),
    )
}

/// Min + mean over `reps` timed runs of `f` (result dropped).
fn time_reps<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, f64) {
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(f());
        times.push(t.elapsed().as_secs_f64());
    }
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    (min, mean)
}

fn ms(t: f64) -> f64 {
    t * 1e3
}

fn bench_case(size: f64, reps: usize) {
    let (geo, domain) = obstacle();
    let growth = 1.2;

    // --- Seeding (common to both pipelines; NOT engine time) --------------
    let t = Instant::now();
    let (seeds, kinds, spec) = meshless_seed_points(&geo, size, size, growth, domain);
    let t_seed = t.elapsed().as_secs_f64();
    let n = seeds.len();
    let tol = MeshgenTolerances::from_geometry(size, domain);
    let cfg = EngineConfig::default();

    // --- SeedGrid build ----------------------------------------------------
    let (grid_min, grid_mean) = time_reps(reps, || SeedGrid::build(&seeds, domain));

    // --- build_diagram -----------------------------------------------------
    let input = MeshlessInput {
        seeds: &seeds,
        kinds: &kinds,
        boundary: &spec,
        domain,
        tol: &tol,
        cfg,
    };
    let (diag_min, diag_mean) = time_reps(reps, || build_diagram(&input));
    let diagram = build_diagram(&input);
    let (ok, esc, ovf, empty, failed) = diagram.status_counts();

    // --- assemble_mesh -----------------------------------------------------
    let (asm_min, asm_mean) = time_reps(reps, || assemble_mesh(&input, &diagram));

    // --- one Lloyd iteration (grid + diagram + move, NO assembly) ----------
    let sizing = |p: Point2<f64>| -> f64 {
        let dist = geo.sdf(&p).abs();
        (size + (growth - 1.0f64).max(0.0) * dist).min(size)
    };
    let one = LloydConfig {
        max_iters: 1,
        tol_disp: 0.0,
        ..LloydConfig::default()
    };
    let (lloyd_min, lloyd_mean) = time_reps(reps, || {
        let mut s = seeds.clone();
        lloyd_relax(&mut s, &kinds, &spec, &sizing, domain, &tol, &cfg, &one)
    });

    // --- generate_cvt_mesh total (seeding + default-budget Lloyd + assembly)
    let t = Instant::now();
    let cvt = cfd2::meshgen::generate_cvt_mesh(&geo, size, size, growth, domain, &LloydConfig::default());
    let t_cvt = t.elapsed().as_secs_f64();

    // --- incumbent baseline: generate_voronoi_mesh, plus the GUI's smooth --
    let t = Instant::now();
    let mut incumbent = generate_voronoi_mesh(&geo, size, size, growth, domain);
    let t_inc = t.elapsed().as_secs_f64();
    let t = Instant::now();
    incumbent.smooth(&geo, 0.3, 50);
    let t_inc_smooth = t.elapsed().as_secs_f64();

    println!("== meshless bench: size={size} n_seeds={n} (cvt cells={}, incumbent cells={}) ==", cvt.num_cells(), incumbent.num_cells());
    println!("  statuses: ok={ok} esc={esc} ovf={ovf} empty={empty} failed={failed}");
    println!("  poisson seeding (both pipelines)  {:9.1} ms", ms(t_seed));
    println!("  SeedGrid::build      min/mean     {:9.1} / {:7.1} ms", ms(grid_min), ms(grid_mean));
    println!("  build_diagram        min/mean     {:9.1} / {:7.1} ms   [budget @300k: <= 100 ms]", ms(diag_min), ms(diag_mean));
    println!("  assemble_mesh        min/mean     {:9.1} / {:7.1} ms   [budget @300k: <= 250 ms]", ms(asm_min), ms(asm_mean));
    println!("  Lloyd iteration      min/mean     {:9.1} / {:7.1} ms   [budget @300k: <= 120 ms]", ms(lloyd_min), ms(lloyd_mean));
    println!("  generate_cvt_mesh total           {:9.1} ms   [budget: <= incumbent voronoi]", ms(t_cvt));
    println!("  incumbent generate_voronoi_mesh   {:9.1} ms   (+ GUI smooth(0.3,50): {:.1} ms)", ms(t_inc), ms(t_inc_smooth));

    // Regression gates at 2x budget (roadmap red-flag line), on the ~300k
    // case only. `min` over reps is the noise-robust statistic.
    if n >= 250_000 {
        assert!(diag_min <= 0.200, "build_diagram {:.1} ms > 2x budget (200 ms) @ {n} seeds", ms(diag_min));
        assert!(asm_min <= 0.500, "assemble_mesh {:.1} ms > 2x budget (500 ms) @ {n} seeds", ms(asm_min));
        assert!(lloyd_min <= 0.240, "Lloyd iteration {:.1} ms > 2x budget (240 ms) @ {n} seeds", ms(lloyd_min));
        assert!(
            t_cvt <= 2.0 * (t_inc + t_inc_smooth),
            "generate_cvt_mesh {:.2} s > 2x incumbent+smooth {:.2} s @ {n} seeds",
            t_cvt,
            t_inc + t_inc_smooth
        );
    }
}

#[test]
fn meshless_bench() {
    let reps = std::env::var("CFD2_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5usize)
        .max(1);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    println!("[bench] hw_par={threads} rayon_threads={} reps={reps}", rayon::current_num_threads());

    // ~75k and ~300k cells on the 3x1 obstacle channel (cells ~= 1.92/h^2,
    // measured on this geometry), overridable to a single size via env.
    let sizes: Vec<f64> = match std::env::var("CFD2_BENCH_SIZE") {
        Ok(v) => vec![v.parse().expect("CFD2_BENCH_SIZE must be f64")],
        Err(_) => vec![0.005, 0.0025],
    };
    for size in sizes {
        bench_case(size, reps);
    }
}
