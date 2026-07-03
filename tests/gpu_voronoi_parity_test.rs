//! M1 stage-1 parity gates for the GPU meshless Voronoi engine
//! (`src/solver/gpu/voronoi/`), interior-only bbox configuration:
//!
//! 1. For every GPU `SUCCESS` cell, the eps_face-filtered neighbor set
//!    (bisector ids AND bbox sides) equals the CPU f64 `build_diagram`
//!    result on the SAME f32-rounded seed values (review F4: the oracle
//!    runs on what the kernel saw), eps_face = 1e-6 · local h applied to
//!    BOTH sides;
//! 2. geometry parity: cell area rel < 1e-5, centroid (seed-relative)
//!    < 1e-5 · local h;
//! 3. mismatch budget ≤ 0.1% of cells pre-fallback, zero overflow statuses;
//! 4. run-to-run byte stability of all deterministic outputs on the same
//!    device;
//! 5. a first kernel-timing datapoint at ~30k seeds.
//!
//! Run with:
//!
//! ```sh
//! cargo test --features meshgen --test gpu_voronoi_parity_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use std::collections::BTreeSet;

use cfd2::meshgen::meshless::{
    build_diagram, BoundarySpec, CellStatus, EngineConfig, MeshlessInput, PlaneTag,
    MAX_CLIP_VERTS,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{status, GpuVoronoiEngine, K_FACE_MAX, NBR_NONE};
use nalgebra::{Point2, Vector2};
use rand::{Rng, SeedableRng};

const DOMAIN: Vector2<f64> = Vector2::new(2.0, 1.0);

/// Deterministic Poisson-like set: jittered lattice (jitter ≤ 0.4·spacing,
/// so seeds stay pairwise separated and strictly interior).
fn jittered_seeds(nx: usize, ny: usize, amp: f64, rng_seed: u64) -> Vec<Point2<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let (sx, sy) = (DOMAIN.x / nx as f64, DOMAIN.y / ny as f64);
    let mut pts = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let jx = (rng.gen::<f64>() - 0.5) * amp * sx;
            let jy = (rng.gen::<f64>() - 0.5) * amp * sy;
            pts.push(Point2::new(
                (i as f64 + 0.5) * sx + jx,
                (j as f64 + 0.5) * sy + jy,
            ));
        }
    }
    pts
}

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

struct CaseReport {
    n: usize,
    mismatches: usize,
    overflows: usize,
}

/// Full parity run for one seed set. `spacing` is the lattice pitch used
/// for tolerances.
fn run_parity_case(name: &str, seeds: Vec<Point2<f64>>, spacing: f64, timing: bool) -> CaseReport {
    let Some(ctx) = gpu_context() else {
        return CaseReport {
            n: 0,
            mismatches: 0,
            overflows: 0,
        };
    };
    let n = seeds.len();
    let tol = MeshgenTolerances::from_geometry(0.5 * spacing, DOMAIN);

    // f32-rounded seeds: what the kernel sees AND what the CPU oracle gets
    // (widened back to f64 — exact).
    let seeds_f32: Vec<f32> = seeds
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32])
        .collect();
    let rounded: Vec<Point2<f64>> = (0..n)
        .map(|i| Point2::new(seeds_f32[2 * i] as f64, seeds_f32[2 * i + 1] as f64))
        .collect();
    let flags = vec![0u32; n];

    // GPU diagram.
    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, DOMAIN, &tol);
    engine.upload_seeds(&ctx.device, &ctx.queue, &seeds_f32, &flags);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
    let cache = StagingBufferCache::default();
    let gpu = engine.read_cells(&ctx, &cache);

    // CPU f64 oracle on the identical (f32-rounded) inputs.
    let boundary = BoundarySpec::empty();
    let input = MeshlessInput::interior_only(
        &rounded,
        &boundary,
        DOMAIN,
        &tol,
        EngineConfig::default(),
    );
    let cpu = build_diagram(&input);

    // Per-cell comparison.
    let mut mismatches = 0usize;
    let mut topo_mismatches = 0usize;
    let mut geom_mismatches = 0usize;
    let mut non_success = 0usize;
    let mut overflows = 0usize;
    let mut examples: Vec<String> = Vec::new();
    let note = |s: String, examples: &mut Vec<String>| {
        if examples.len() < 12 {
            examples.push(s);
        }
    };

    for i in 0..n {
        if gpu.status[i] == status::VERT_OVERFLOW || gpu.status[i] == status::FACE_OVERFLOW {
            overflows += 1;
        }
        if gpu.status[i] != status::SUCCESS {
            non_success += 1;
            mismatches += 1;
            note(
                format!("cell {i}: gpu status {}", gpu.status[i]),
                &mut examples,
            );
            continue;
        }
        let cpu_ok = matches!(cpu.status[i], CellStatus::Ok | CellStatus::OkEscalated(_));
        if !cpu_ok {
            mismatches += 1;
            note(
                format!("cell {i}: cpu status {:?}", cpu.status[i]),
                &mut examples,
            );
            continue;
        }

        // Local h = nearest-neighbor distance (nearest neighbor is always a
        // Voronoi neighbor, so the unfiltered CPU ring suffices).
        let p = rounded[i];
        let len = cpu.ring_len[i] as usize;
        let mut h_i = f64::INFINITY;
        for e in 0..len {
            if let PlaneTag::Bisector(j) = cpu.ring_plane[i * MAX_CLIP_VERTS + e] {
                h_i = h_i.min((rounded[j as usize] - p).norm());
            }
        }
        assert!(h_i.is_finite(), "cell {i} has no bisector neighbors");
        let eps_face = 1e-6 * h_i;

        // CPU neighbor/box sets post eps_face filter.
        let mut cpu_nbrs: BTreeSet<u32> = BTreeSet::new();
        let mut cpu_sides: BTreeSet<u32> = BTreeSet::new();
        for e in 0..len {
            let v0 = cpu.ring_xy[i * MAX_CLIP_VERTS + e];
            let v1 = cpu.ring_xy[i * MAX_CLIP_VERTS + (e + 1) % len];
            let elen = ((v1[0] - v0[0]).powi(2) + (v1[1] - v0[1]).powi(2)).sqrt();
            if elen <= eps_face {
                continue;
            }
            match cpu.ring_plane[i * MAX_CLIP_VERTS + e] {
                PlaneTag::Bisector(j) => {
                    cpu_nbrs.insert(j);
                }
                PlaneTag::Box(s) => {
                    cpu_sides.insert(s as u32);
                }
                PlaneTag::Boundary(_) => unreachable!("bbox-only configuration"),
            }
        }

        // GPU sets post the same filter.
        let mut gpu_nbrs: BTreeSet<u32> = BTreeSet::new();
        let mut gpu_sides: BTreeSet<u32> = BTreeSet::new();
        for s in 0..gpu.nfaces[i] as usize {
            let slot = i * K_FACE_MAX + s;
            let flen = gpu.face_geom[slot][2] as f64;
            if flen <= eps_face {
                continue;
            }
            let nbr = gpu.nbr_ids[slot];
            if nbr == NBR_NONE {
                gpu_sides.insert(gpu.face_bc[slot]);
            } else {
                gpu_nbrs.insert(nbr);
            }
        }

        if cpu_nbrs != gpu_nbrs || cpu_sides != gpu_sides {
            mismatches += 1;
            topo_mismatches += 1;
            note(
                format!(
                    "cell {i}: nbrs cpu-only {:?} gpu-only {:?}; sides cpu {:?} gpu {:?}",
                    cpu_nbrs.difference(&gpu_nbrs).collect::<Vec<_>>(),
                    gpu_nbrs.difference(&cpu_nbrs).collect::<Vec<_>>(),
                    cpu_sides,
                    gpu_sides
                ),
                &mut examples,
            );
            continue;
        }

        // Geometry parity.
        let cpu_area = cpu.area[i];
        let gpu_area = gpu.area[i] as f64;
        let area_rel = (gpu_area - cpu_area).abs() / cpu_area;
        let cpu_rel = [cpu.centroid[i][0] - p.x, cpu.centroid[i][1] - p.y];
        let dcx = gpu.centroid_rel[i][0] as f64 - cpu_rel[0];
        let dcy = gpu.centroid_rel[i][1] as f64 - cpu_rel[1];
        let cen_err = (dcx * dcx + dcy * dcy).sqrt();
        if area_rel >= 1e-5 || cen_err >= 1e-5 * h_i {
            mismatches += 1;
            geom_mismatches += 1;
            note(
                format!(
                    "cell {i}: area_rel {area_rel:.2e} cen_err {cen_err:.2e} (h {h_i:.2e})"
                ),
                &mut examples,
            );
        }
    }

    println!(
        "[{name}] n={n} mismatches={mismatches} (topo={topo_mismatches} geom={geom_mismatches} \
         non_success={non_success} overflow={overflows}) flagged={}",
        gpu.flagged.len()
    );
    for e in &examples {
        println!("[{name}]   {e}");
    }

    // Byte stability: regen again on the same device, all deterministic
    // outputs must be byte-identical.
    let raw1 = engine.read_raw_outputs(&ctx, &cache);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
    let raw2 = engine.read_raw_outputs(&ctx, &cache);
    assert_eq!(raw1, raw2, "[{name}] regen outputs not byte-stable run-to-run");

    if timing {
        // First kernel-timing datapoint: wall clock around submit + poll.
        let mut best = f64::INFINITY;
        let mut total = 0.0;
        const REPS: usize = 10;
        for _ in 0..REPS {
            let t0 = std::time::Instant::now();
            let idx = engine.run_regen(&ctx.device, &ctx.queue);
            let _ = ctx.device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            });
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            best = best.min(dt);
            total += dt;
        }
        println!(
            "[{name}] regen wall time over {REPS} reps: best {best:.3} ms, mean {:.3} ms",
            total / REPS as f64
        );
    }

    CaseReport {
        n,
        mismatches,
        overflows,
    }
}

fn assert_budget(r: &CaseReport, name: &str) {
    if r.n == 0 {
        return; // skipped (no GPU)
    }
    let budget = ((r.n as f64) * 1e-3).ceil() as usize;
    assert!(
        r.mismatches <= budget,
        "[{name}] {} mismatched cells exceed the 0.1% pre-fallback budget ({})",
        r.mismatches,
        budget
    );
    assert_eq!(
        r.overflows, 0,
        "[{name}] VERT/FACE_OVERFLOW must not fire on Poisson-like sets"
    );
}

#[test]
fn gpu_voronoi_interior_parity_5k() {
    let seeds = jittered_seeds(100, 50, 0.8, 0x5EED_CFD2);
    let spacing = DOMAIN.y / 50.0;
    let r = run_parity_case("5k", seeds, spacing, false);
    assert_budget(&r, "5k");
}

#[test]
fn gpu_voronoi_interior_parity_30k() {
    let seeds = jittered_seeds(245, 123, 0.8, 0xCFD2_5EED);
    let spacing = DOMAIN.y / 123.0;
    let r = run_parity_case("30k", seeds, spacing, true);
    assert_budget(&r, "30k");
}
