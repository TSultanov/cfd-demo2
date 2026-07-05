//! CPU-only regression coverage for the mutual-orphan endpoint reconciliation pass
//! in `assemble_mesh`: the pass changes the shared assembler for every consumer, but
//! was otherwise exercised only by GPU-gated tests that silently skip without an adapter.
//!
//! The trigger is f32 QUANTIZATION of the seeds/boundary, not the GPU kernel: on
//! f32-rounded seed sets a reflex-vertex guard pair's mutual bisector misses the
//! polyline vertex by ~1 f32 position ulp, so the two cells canonicalize the shared
//! face endpoint through DIFFERENT tag pairs into different dedup bins. No GPU needed:
//!
//! 1. f32-rounded seeds + spec → `build_diagram` → `assemble_mesh` must produce a
//!    fully CLOSED mesh (the condition the pass repairs); the pass must fire on at
//!    least one standard case and accepted gaps must stay at the f32 noise scale.
//! 2. On unrounded f64 seeds the pass still fires (obstacle circle 4× with gaps
//!    ~1.7e-8) — but only BELOW the 1e-6·h quantized-dedup pitch (5e-8 here):
//!    sub-pitch coincidences that straddle a quantize-bin boundary. The gate: every
//!    f64-accepted gap ≤ the dedup pitch (the pass only welds what the dedup was
//!    designed to absorb, never real topology), and the mesh closes.
//!
//! ```sh
//! cargo test --features meshgen --test meshless_orphan_cpu_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use std::sync::atomic::Ordering::Relaxed;

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, reset_mutual_orphan_stats, EngineConfig,
    MeshlessInput, MUTUAL_ORPHAN_MAX_GAP, MUTUAL_ORPHAN_UNIONS,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::voronoi::boundary_spec_f32;
use cfd2::solver::mesh::{ChannelWithObstacle, Geometry, Mesh, Nozzle};
use nalgebra::{Point2, Vector2};

/// Worst per-cell relative closure defect `|Σ A·n̂| / Σ A` and the number of
/// cells over 1e-6 (the validate_mesh closure gate).
fn closure_defects(mesh: &Mesh) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut open = 0usize;
    for c in 0..mesh.num_cells() {
        let start = mesh.cell_face_offsets[c];
        let end = mesh.cell_face_offsets[c + 1];
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut perim = 0.0;
        for &f in &mesh.cell_faces[start..end] {
            let sign = if mesh.face_owner[f] == c { 1.0 } else { -1.0 };
            sx += sign * mesh.face_area[f] * mesh.face_nx[f];
            sy += sign * mesh.face_area[f] * mesh.face_ny[f];
            perim += mesh.face_area[f];
        }
        if perim > 0.0 {
            let rel = (sx * sx + sy * sy).sqrt() / perim;
            worst = worst.max(rel);
            if rel > 1e-6 {
                open += 1;
            }
        }
    }
    (worst, open)
}

/// Assemble one geometry case; returns (fires, max_gap, mesh).
fn assemble_case(
    name: &str,
    geo: &(impl Geometry + Sync),
    domain: Vector2<f64>,
    hmin: f64,
    round_f32: bool,
) -> (u64, f64, Mesh) {
    let (seeds, kinds, spec) = meshless_seed_points(geo, hmin, hmin, 1.2, domain);
    let tol = MeshgenTolerances::from_geometry(hmin, domain);
    let (seeds, spec) = if round_f32 {
        (
            seeds
                .iter()
                .map(|p| Point2::new(p.x as f32 as f64, p.y as f32 as f64))
                .collect::<Vec<_>>(),
            boundary_spec_f32(&spec),
        )
    } else {
        (seeds, spec)
    };
    let input = MeshlessInput {
        seeds: &seeds,
        kinds: &kinds,
        boundary: &spec,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let diag = build_diagram(&input);
    reset_mutual_orphan_stats();
    let mesh = assemble_mesh(&input, &diag);
    let fires = MUTUAL_ORPHAN_UNIONS.load(Relaxed);
    let max_gap = f64::from_bits(MUTUAL_ORPHAN_MAX_GAP.load(Relaxed));
    let (worst, open) = closure_defects(&mesh);
    println!(
        "[{name}/{}] n={} fires={fires} max_gap={max_gap:.3e} worst_closure={worst:.3e} open={open}",
        if round_f32 { "f32" } else { "f64" },
        mesh.num_cells()
    );
    assert_eq!(
        open, 0,
        "[{name}] {open} cells not closed (worst rel {worst:.3e})"
    );
    (fires, max_gap, mesh)
}

#[test]
fn mutual_orphan_pass_fires_and_closes_on_f32_rounded_seeds() {
    let obstacle = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let nozzle = Nozzle {
        length: 3.0,
        height: 1.0,
        throat_height: 0.40,
        throat_frac: 0.40,
        exit_height: 0.80,
    };
    let domain = Vector2::new(3.0, 1.0);

    let (f_obst, _, _) = assemble_case("obstacle", &obstacle, domain, 0.05, true);
    let (f_nozz, gap_nozz, _) = assemble_case("nozzle", &nozzle, domain, 0.05, true);

    // At least one standard f32-rounded case must exercise the pass —
    // otherwise this test has lost its trigger and needs a new case
    // (the reflex-guard pattern lives on the nozzle/obstacle walls).
    assert!(
        f_obst + f_nozz > 0,
        "mutual-orphan pass no longer fires on f32-rounded obstacle/nozzle — \
         regression coverage lost (did seeding or the threshold change?)"
    );
    // Accepted gaps must stay at the f32-position-noise scale the pass is
    // justified by (64 ulps of domain scale ≈ 1.1e-5 for domain 3).
    let noise = 64.0 * 2f64.powi(-24) * 3.0;
    assert!(
        gap_nozz <= noise,
        "accepted endpoint gap {gap_nozz:.3e} exceeds the f32 noise scale {noise:.3e}"
    );
}

#[test]
fn mutual_orphan_pass_f64_firings_stay_sub_dedup_pitch() {
    let obstacle = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let nozzle = Nozzle {
        length: 3.0,
        height: 1.0,
        throat_height: 0.40,
        throat_frac: 0.40,
        exit_height: 0.80,
    };
    let domain = Vector2::new(3.0, 1.0);
    let hmin = 0.05;

    let (f_obst, gap_obst, _) = assemble_case("obstacle", &obstacle, domain, hmin, false);
    let (f_nozz, gap_nozz, _) = assemble_case("nozzle", &nozzle, domain, hmin, false);

    // The pass DOES fire on the unrounded f64 obstacle (4 sites) — but only for
    // endpoint gaps BELOW the quantized-dedup pitch (vertex_merge = 1e-6·h): the
    // sub-pitch bin-straddle coincidence class the dedup was built to absorb. Anything
    // above the pitch on f64 would mean the pass rewrites genuine topology on the
    // pure-CPU path — that is the gate.
    let dedup_pitch = 1e-6 * hmin;
    let max_gap = gap_obst.max(gap_nozz);
    println!(
        "[f64] fires: obstacle {f_obst} nozzle {f_nozz}; max gap {max_gap:.3e} \
         (dedup pitch {dedup_pitch:.3e})"
    );
    assert!(
        max_gap <= dedup_pitch,
        "f64 mutual-orphan union accepted a gap {max_gap:.3e} ABOVE the dedup \
         pitch {dedup_pitch:.3e} — real topology is being rewritten"
    );
}
