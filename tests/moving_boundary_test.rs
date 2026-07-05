//! M6 stage 1 gates (meshless/moving-mesh roadmap §M6): a PRESCRIBED-motion
//! boundary (an oscillating cylinder in a channel) whose boundary-bound seeds
//! move RIGIDLY with it each step, staying exactly on the moving wall, driven
//! through the M4 moving-mesh loop on CPU.
//!
//! Stage 1 delivers the mesh/seed-motion machinery only — the `MovingWall`
//! Dirichlet BC (fluid feels the wall's material velocity) is stage 2, so the
//! obstacle contour is still tagged `Wall` here and these gates validate mesh
//! integrity + seed tracking + the static-boundary do-no-harm anchor, NOT the
//! flow response.
//!
//! Gates:
//!   * `moving_obstacle_mesh_stays_valid` — oscillating cylinder, N steps; every
//!     step the obstacle-contour faces exist + are tagged `Wall`, zero untagged
//!     boundary faces, watertight (Wall length ≈ moving circumference), positive
//!     cell areas, closure, fixed cell count. The obstacle demonstrably moves.
//!   * `boundary_seeds_track_the_wall` — every moving boundary seed stays on the
//!     analytic moving obstacle (its distance to the moved centre equals the
//!     chord-midpoint radius to f64 roundoff) every step.
//!   * `static_boundary_is_byte_identical_do_no_harm` — a static obstacle (both
//!     `BoundaryMotionSpec::Static` AND a zero-amplitude `RigidLoop`) reproduces
//!     the M4 frozen behaviour: the regen is BYTE-IDENTICAL to the initial mesh
//!     every step and `w_wall` is all-zero.
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{ChannelWithObstacle, LloydConfig};
use cfd2::sim::{BoundaryMotionSpec, MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the CPU tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

// --- shared scenario constants ------------------------------------------------
const LX: f64 = 2.0;
const LY: f64 = 1.0;
const OBS_CX: f64 = 0.6;
const OBS_CY: f64 = 0.5;
const OBS_R: f64 = 0.1;
const H: f64 = 0.05;
/// Oscillation amplitude in x (< H so frozen interior seeds are never swallowed).
const AMP: f64 = 0.02;
/// Oscillation angular frequency (period 0.4).
const OMEGA: f64 = std::f64::consts::TAU / 0.4;
/// The obstacle is loop 1 of `ChannelWithObstacle::get_boundary_loops`
/// (loop 0 = outer channel box).
const OBSTACLE_LOOP: usize = 1;

/// Oscillating-cylinder rigid transform: x-translation `A·sin(ω t)`, identity at
/// t=0 (the build-mesh contract). A plain `fn` so it fits `BoundaryMotionSpec`.
fn oscillate(t: f64, p: [f64; 2]) -> [f64; 2] {
    [p[0] + AMP * (OMEGA * t).sin(), p[1]]
}

/// Zero-amplitude rigid transform (exact identity) — the do-no-harm probe that
/// exercises the `RigidLoop` plumbing while moving nothing.
fn identity_motion(_t: f64, p: [f64; 2]) -> [f64; 2] {
    p
}

fn base_params(dt: f32) -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: dt,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 100_000,
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 3,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 1.0,
        density: 1.0,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn obstacle_geo() -> ChannelWithObstacle {
    ChannelWithObstacle {
        length: LX,
        height: LY,
        obstacle_center: Point2::new(OBS_CX, OBS_CY),
        obstacle_radius: OBS_R,
    }
}

/// Faces on the obstacle contour (reused verbatim from
/// `tests/voronoi_obstacle_wall_test.rs:30`): open faces whose centre is near
/// the (moved) circle.
fn obstacle_faces(mesh: &Mesh, center: Point2<f64>, radius: f64) -> Vec<usize> {
    (0..mesh.num_faces())
        .filter(|&f| {
            mesh.face_neighbor[f].is_none() && {
                let d = (Point2::new(mesh.face_cx[f], mesh.face_cy[f]) - center).norm();
                (d - radius).abs() < 0.5 * radius
            }
        })
        .collect()
}

/// Untagged boundary faces (validate_mesh #6): open faces with no BoundaryType.
fn untagged_boundary_faces(mesh: &Mesh) -> usize {
    (0..mesh.num_faces())
        .filter(|&f| mesh.face_neighbor[f].is_none() && mesh.face_boundary[f].is_none())
        .count()
}

/// Worst per-cell closure ‖Σ σ·A·n‖ / perimeter and the count of non-positive
/// cell volumes.
fn mesh_health(mesh: &Mesh) -> (f64, usize) {
    let mut worst = 0.0f64;
    for c in 0..mesh.num_cells() {
        let (s, e) = (mesh.cell_face_offsets[c], mesh.cell_face_offsets[c + 1]);
        let (mut sx, mut sy, mut perim) = (0.0, 0.0, 0.0);
        for &f in &mesh.cell_faces[s..e] {
            let sign = if mesh.face_owner[f] == c { 1.0 } else { -1.0 };
            sx += sign * mesh.face_area[f] * mesh.face_nx[f];
            sy += sign * mesh.face_area[f] * mesh.face_ny[f];
            perim += mesh.face_area[f];
        }
        if perim > 0.0 {
            worst = worst.max((sx * sx + sy * sy).sqrt() / perim);
        }
    }
    let nonpos = mesh.cell_vol.iter().filter(|&&v| !(v > 0.0)).count();
    (worst, nonpos)
}

fn build_cvt() -> cfd2::meshgen::meshless::CvtMeshSeeds {
    let geo = obstacle_geo();
    let domain = Vector2::new(LX, LY);
    generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default())
}

#[test]
fn moving_obstacle_mesh_stays_valid() {
    let _g = ENV_LOCK.lock().unwrap();
    let cvt = build_cvt();
    let n_cells = cvt.mesh.num_cells();
    // Chord count of the obstacle loop (== the moving-wall face count target).
    let n_chords = cvt.spec.loops[OBSTACLE_LOOP].pts.len();
    let params = base_params(0.01);

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen, // interior seeds frozen; only the wall moves
        &vec![(1.0, 0.0); n_cells],
        &vec![0.0; n_cells],
        None,
        None,
    ))
    .expect("build moving obstacle driver");
    moving.driver_mut().apply_params(&params);
    moving.set_boundary_motion(BoundaryMotionSpec::RigidLoop {
        loop_index: OBSTACLE_LOOP,
        transform: oscillate,
    });

    let steps = 40usize;
    let mut max_disp = 0.0f64;
    let mut sim_t = 0.0f64;
    for step in 0..steps {
        let (_outcome, stats) = moving.step(false).expect("moving step");
        sim_t += stats.dt;

        // Cell count fixed (fixed-seed v1).
        assert_eq!(
            stats.n_cells, n_cells,
            "step {step}: cell count changed {} -> {}",
            n_cells, stats.n_cells
        );

        let mesh = moving.mesh();
        // Analytic moved obstacle centre at the committed time.
        let cx = OBS_CX + AMP * (OMEGA * sim_t).sin();
        max_disp = max_disp.max((cx - OBS_CX).abs());
        let center = Point2::new(cx, OBS_CY);

        // Obstacle-contour faces exist and are tagged Wall (stage 1 keeps Wall;
        // MovingWall arrives in stage 2).
        let contour = obstacle_faces(mesh, center, OBS_R);
        assert!(
            contour.len() >= n_chords - 2,
            "step {step}: obstacle contour under-resolved: {} faces (chords {n_chords})",
            contour.len()
        );
        let mut wall_len = 0.0;
        for &f in &contour {
            assert_eq!(
                mesh.face_boundary[f],
                Some(BoundaryType::Wall),
                "step {step}: obstacle face {f} at ({:.3},{:.3}) not tagged Wall",
                mesh.face_cx[f],
                mesh.face_cy[f]
            );
            wall_len += mesh.face_area[f];
        }
        // Watertight: chord perimeter ≈ circumference (chords slightly shorter).
        let circumference = std::f64::consts::TAU * OBS_R;
        assert!(
            wall_len > 0.95 * circumference && wall_len < 1.02 * circumference,
            "step {step}: obstacle wall length {wall_len:.4} vs circumference {circumference:.4}"
        );

        // Zero untagged boundary faces; positive areas; closed cells.
        assert_eq!(
            untagged_boundary_faces(mesh),
            0,
            "step {step}: untagged boundary faces (free-slip holes)"
        );
        let (worst_closure, nonpos) = mesh_health(mesh);
        assert_eq!(nonpos, 0, "step {step}: {nonpos} non-positive cell volumes");
        assert!(
            worst_closure < 1e-6,
            "step {step}: worst cell closure {worst_closure:.3e} (Σ A·n not ~0)"
        );

        // GCL: the swept fluxes stay conservative (roundoff-scale defect).
        assert!(
            stats.scl_defect < 1e-6,
            "step {step}: SCL defect {:.3e} (moving-boundary swept fluxes not closed)",
            stats.scl_defect
        );
    }
    // The wall demonstrably moved over the run.
    assert!(
        max_disp > 0.5 * AMP,
        "obstacle barely moved: max |Δx| = {max_disp:.4} vs amplitude {AMP}"
    );
    println!(
        "moving_obstacle_mesh_stays_valid: {steps} steps, {n_cells} cells, max wall disp {max_disp:.4}, sim_t {sim_t:.3}"
    );
}

#[test]
fn boundary_seeds_track_the_wall() {
    let _g = ENV_LOCK.lock().unwrap();
    let cvt = build_cvt();
    let n_cells = cvt.mesh.num_cells();
    let kinds = cvt.kinds.clone();
    let seeds0: Vec<Point2<f64>> = cvt.seeds.clone();
    // Moving-loop segment range: seeds whose adjacent segment is in loop 1.
    let (seg_lo, seg_hi) = (
        cvt.spec.seg_offsets[OBSTACLE_LOOP],
        cvt.spec.seg_offsets[OBSTACLE_LOOP + 1],
    );
    let n_chords = cvt.spec.loops[OBSTACLE_LOOP].pts.len();
    // Chord-midpoint radius: every obstacle seed is a reflex-vertex guard at a
    // chord midpoint, i.e. r·cos(π/n) from the centre (rigid-motion invariant).
    let mid_radius = OBS_R * (std::f64::consts::PI / n_chords as f64).cos();

    let is_moving_wall_seed = |i: usize| -> bool {
        matches!(
            kinds[i],
            cfd2::meshgen::meshless::SeedKind::Boundary { seg_next, .. }
                if (seg_next as usize) >= seg_lo && (seg_next as usize) < seg_hi
        )
    };
    // Every moving-wall seed sits at the chord-midpoint radius at t=0.
    let n_wall_seeds = (0..n_cells).filter(|&i| is_moving_wall_seed(i)).count();
    assert!(n_wall_seeds >= n_chords - 1, "too few obstacle seeds: {n_wall_seeds}");
    for i in 0..n_cells {
        if is_moving_wall_seed(i) {
            let d = (seeds0[i] - Point2::new(OBS_CX, OBS_CY)).norm();
            assert!(
                (d - mid_radius).abs() < 1e-9,
                "seed {i} not at chord-midpoint radius at t=0: {d:.6} vs {mid_radius:.6}"
            );
        }
    }

    let params = base_params(0.01);
    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen,
        &vec![(1.0, 0.0); n_cells],
        &vec![0.0; n_cells],
        None,
        None,
    ))
    .expect("build");
    moving.driver_mut().apply_params(&params);
    moving.set_boundary_motion(BoundaryMotionSpec::RigidLoop {
        loop_index: OBSTACLE_LOOP,
        transform: oscillate,
    });

    let mut sim_t = 0.0f64;
    let mut worst = 0.0f64;
    for step in 0..30 {
        let (_o, stats) = moving.step(false).expect("step");
        sim_t += stats.dt;
        let cx = OBS_CX + AMP * (OMEGA * sim_t).sin();
        let center = Point2::new(cx, OBS_CY);
        let seeds = moving.seeds();
        let w_wall = moving.w_wall();
        for i in 0..n_cells {
            if is_moving_wall_seed(i) {
                // Stays on the analytic moving wall (chord-midpoint radius).
                let d = (seeds[i] - center).norm();
                worst = worst.max((d - mid_radius).abs());
                assert!(
                    (d - mid_radius).abs() < 1e-9,
                    "step {step} seed {i}: off the moving wall, dist-to-centre {d:.6} vs {mid_radius:.6}"
                );
                // Its recorded material velocity is the analytic wall velocity
                // A·ω·cos(ω t̄) in x, 0 in y (finite-difference over the step).
                assert!(
                    w_wall[i][1].abs() < 1e-9,
                    "step {step} seed {i}: spurious y wall velocity {:.3e}",
                    w_wall[i][1]
                );
            } else {
                // Interior + static-boundary seeds carry zero wall velocity.
                assert_eq!(w_wall[i], [0.0, 0.0], "step {step} seed {i}: nonzero w_wall off the moving wall");
            }
        }
    }
    println!("boundary_seeds_track_the_wall: worst radial error {worst:.3e} (tol 1e-9)");
}

#[test]
fn static_boundary_is_byte_identical_do_no_harm() {
    let _g = ENV_LOCK.lock().unwrap();

    // Reference: the initial regen mesh vertices (frozen ⇒ every step must
    // reproduce these bit-for-bit).
    let cvt_ref = build_cvt();
    let vx0 = cvt_ref.mesh.vx.clone();
    let vy0 = cvt_ref.mesh.vy.clone();
    let n_cells = cvt_ref.mesh.num_cells();
    drop(cvt_ref);

    // Two static-boundary configurations that must both be byte-identical to a
    // pure frozen run: explicit Static, and a zero-amplitude RigidLoop (the
    // plumbing runs but moves nothing).
    for (label, bmotion) in [
        ("Static", BoundaryMotionSpec::Static),
        (
            "RigidLoop(A=0)",
            BoundaryMotionSpec::RigidLoop {
                loop_index: OBSTACLE_LOOP,
                transform: identity_motion,
            },
        ),
    ] {
        let cvt = build_cvt();
        let params = base_params(0.01);
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(1.0, 0.0); n_cells],
            &vec![0.0; n_cells],
            None,
            None,
        ))
        .expect("build");
        moving.driver_mut().apply_params(&params);
        moving.set_boundary_motion(bmotion);

        for step in 0..15 {
            let (_o, stats) = moving.step(false).expect("step");
            let mesh = moving.mesh();
            // Byte-identical vertices every step (frozen + zero boundary motion).
            assert!(
                mesh.vx == vx0 && mesh.vy == vy0,
                "[{label}] step {step}: regen mesh NOT byte-identical to the initial mesh"
            );
            // Zero swept flux (byte-identical regen) and zero wall velocity.
            assert_eq!(
                stats.scl_defect, 0.0,
                "[{label}] step {step}: nonzero SCL defect on a frozen step"
            );
            assert!(
                moving.w_wall().iter().all(|w| *w == [0.0, 0.0]),
                "[{label}] step {step}: nonzero w_wall under static boundary"
            );
            assert!(!stats.flipped, "[{label}] step {step}: unexpected flip on a frozen step");
        }
        println!("static_boundary_is_byte_identical_do_no_harm[{label}]: 15 steps byte-identical");
    }
}
