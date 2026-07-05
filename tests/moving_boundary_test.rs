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

// =============================================================================
// M6 stage 2 — MovingWall ALE BC: the fluid feels the wall's material velocity.
//
// The BC path (documented in the report): `MovingWall` (bc index 5) is ALREADY
// a per-face Dirichlet velocity in the incompressible_momentum(_ale) model,
// consumed by the assembly identically to the Inlet Dirichlet (kind 1 →
// `bc_neighbor_scalar` returns the prescribed value). Stage 2 reuses it: each
// regen re-tags the moving obstacle's open faces `MovingWall` and, after the ALE
// refresh, sets their per-face `bc_value` to the recorded wall velocity `w_wall`
// (re-applied every step because the topology seam resets per-face overrides).
// No codegen / WGSL change — the smallest correct path.
// =============================================================================

/// Obstacle+free-stream translation velocity for the rigid-body gate.
const W_TRANS: f64 = 0.1;

/// Rigidly TRANSLATING obstacle: constant velocity `W_TRANS` in x, identity at
/// t=0 (the build-mesh contract). Small total displacement over the short run
/// keeps the fixed interior seeds from being swallowed.
fn translate(t: f64, p: [f64; 2]) -> [f64; 2] {
    [p[0] + W_TRANS * t, p[1]]
}

/// Slip-channel retag that stamps ONLY the outer domain box (Inlet left, Outlet
/// right, SlipWall top/bottom) and LEAVES the obstacle contour faces for the
/// driver's `MovingWall` retag. A uniform free stream `(W_TRANS,0)` satisfies
/// inlet Dirichlet, outlet zero-gradient and slip walls, so any drift off it is
/// a GCL / wall-BC artifact — never outer-BC physics.
fn tag_slip_box(mesh: &mut Mesh) {
    let eps = 1e-6;
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_some() {
            continue;
        }
        let (x, y) = (mesh.face_cx[f], mesh.face_cy[f]);
        let bt = if x < eps {
            BoundaryType::Inlet
        } else if x > LX - eps {
            BoundaryType::Outlet
        } else if y < eps || y > LY - eps {
            BoundaryType::SlipWall
        } else {
            continue; // obstacle contour — left for the MovingWall retag
        };
        mesh.face_boundary[f] = Some(bt);
    }
}

/// Tag every outer domain-box face `Wall` (closed box); obstacle contour faces
/// are left for the driver's `MovingWall` retag.
fn tag_wall_box(mesh: &mut Mesh) {
    let eps = 1e-6;
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_some() {
            continue;
        }
        let (x, y) = (mesh.face_cx[f], mesh.face_cy[f]);
        if x < eps || x > LX - eps || y < eps || y > LY - eps {
            mesh.face_boundary[f] = Some(BoundaryType::Wall);
        }
    }
}

/// Read per-cell `(U, p)` from the solver state (f64-widened f32).
fn read_state(moving: &MovingMeshDriver) -> (Vec<[f32; 2]>, Vec<f32>) {
    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;
    let p_off = layout.offset_for("p").expect("p offset") as usize;
    let state = pollster::block_on(moving.driver().solver().read_state_f32());
    let n = moving.mesh().num_cells();
    let mut uv = Vec::with_capacity(n);
    let mut pr = Vec::with_capacity(n);
    for c in 0..n {
        uv.push([state[c * stride + u_off], state[c * stride + u_off + 1]]);
        pr.push(state[c * stride + p_off]);
    }
    (uv, pr)
}

struct FsOut {
    /// max over cells of |U − U0| (U0 = the free stream = wall velocity).
    max_du: f32,
    /// max over cells of |p| (gauge 0).
    max_dp: f32,
    /// max over MovingWall faces of |(U_owner − w)·n| — the no-penetration
    /// residual (the fluid does not cross the moving wall).
    max_no_pen: f32,
    /// max per-step SCL defect.
    max_scl: f64,
    /// how many MovingWall faces were seen (the wall is actually tagged/driven).
    n_wall_faces: usize,
    /// how many steps flipped the Voronoi adjacency.
    flips: usize,
}

/// Drive a rigidly-translating obstacle through a uniform free stream equal to
/// the obstacle velocity. With `moving_wall_on` the MovingWall Dirichlet BC =
/// `w_wall`; with it off (control) the obstacle stays a zero-velocity `Wall`
/// while the mesh moves identically — so the only difference is the wall BC.
fn run_freestream(scheme: TimeScheme, moving_wall_on: bool) -> FsOut {
    const STEPS_FS: usize = 12;
    let cvt = build_cvt();
    let n = cvt.mesh.num_cells();
    let mut params = base_params(0.01);
    params.time_scheme = scheme;
    params.inlet_velocity = W_TRANS as f32; // free stream carried by the scalar inlet BC

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen, // interior frozen; only the obstacle translates
        &vec![(W_TRANS, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("build freestream driver");
    moving.set_boundary_retag(Some(tag_slip_box));
    moving.driver_mut().apply_params(&params);
    moving.set_boundary_motion(BoundaryMotionSpec::RigidLoop {
        loop_index: OBSTACLE_LOOP,
        transform: translate,
    });
    moving.set_moving_wall_bc(moving_wall_on);

    let mut out = FsOut {
        max_du: 0.0,
        max_dp: 0.0,
        max_no_pen: 0.0,
        max_scl: 0.0,
        n_wall_faces: 0,
        flips: 0,
    };

    for step in 0..STEPS_FS {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("step {step} (moving_wall={moving_wall_on}) failed: {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        assert_eq!(stats.n_cells, n, "step {step}: cell count changed (seed swallowed)");
        out.max_scl = out.max_scl.max(stats.scl_defect);
        if stats.flipped {
            out.flips += 1;
        }

        let (uv, pr) = read_state(&moving);
        let w_wall = moving.w_wall();
        let mesh = moving.mesh();
        for c in 0..n {
            out.max_du = out
                .max_du
                .max((uv[c][0] - W_TRANS as f32).abs())
                .max((uv[c][1] - 0.0).abs());
            out.max_dp = out.max_dp.max(pr[c].abs());
        }
        // No-penetration on the moving wall: fluid normal velocity relative to
        // the wall's material velocity.
        let mut wall_faces = 0usize;
        for f in 0..mesh.num_faces() {
            if mesh.face_neighbor[f].is_none()
                && mesh.face_boundary[f] == Some(BoundaryType::MovingWall)
            {
                wall_faces += 1;
                let owner = mesh.face_owner[f];
                let rel = [
                    uv[owner][0] as f64 - w_wall[owner][0],
                    uv[owner][1] as f64 - w_wall[owner][1],
                ];
                let rn = (rel[0] * mesh.face_nx[f] + rel[1] * mesh.face_ny[f]).abs();
                out.max_no_pen = out.max_no_pen.max(rn as f32);
            }
        }
        out.n_wall_faces = out.n_wall_faces.max(wall_faces);
    }
    out
}

fn run_freestream_cpu(scheme: TimeScheme, moving_wall_on: bool) -> FsOut {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_freestream(scheme, moving_wall_on));
    std::env::remove_var("CFD2_BACKEND");
    match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    }
}

/// The core stage-2 physics gate: a rigidly translating obstacle in a uniform
/// free stream equal to its velocity. With the MovingWall BC ON the fluid moves
/// WITH the wall (free stream preserved, no-penetration to GCL scale); with it
/// OFF (identical mesh motion, wall BC = 0) the fluid is dragged off the free
/// stream — proving the wall's material velocity actually drives the fluid.
fn moving_wall_freestream(scheme: TimeScheme, label: &str) {
    let on = run_freestream_cpu(scheme, true);
    let off = run_freestream_cpu(scheme, false);
    println!(
        "[m6.2-freestream] cpu/{label} ON : max|U-U0| = {:.3e}, max|p| = {:.3e}, \
         no-pen = {:.3e}, SCL = {:.3e}, wall faces = {}, flips = {}",
        on.max_du, on.max_dp, on.max_no_pen, on.max_scl, on.n_wall_faces, on.flips
    );
    println!(
        "[m6.2-freestream] cpu/{label} OFF: max|U-U0| = {:.3e}, max|p| = {:.3e} (control)",
        off.max_du, off.max_dp
    );

    // The wall is actually tagged MovingWall and driven.
    assert!(on.n_wall_faces > 10, "too few MovingWall faces: {}", on.n_wall_faces);
    // GCL: the swept fluxes stay closed even through the moving-wall motion.
    assert!(on.max_scl < 1e-6, "SCL defect {:.3e}", on.max_scl);
    // No-penetration: the fluid does not cross the moving wall (relative normal
    // velocity is at the solve/GCL scale, ≪ the wall speed W_TRANS).
    assert!(
        (on.max_no_pen as f64) < 0.1 * W_TRANS,
        "no-penetration residual {:.3e} not ≪ wall speed {W_TRANS}",
        on.max_no_pen
    );
    // Free stream preserved with the wall velocity fed in: U stays ≈ U0.
    assert!(
        (on.max_du as f64) < 0.1 * W_TRANS,
        "free stream drift {:.3e} too large (wall velocity not preserving U0)",
        on.max_du
    );
    // The controlled experiment: the zero-velocity wall (OFF) drags the fluid
    // off the free stream far more than the material-velocity wall (ON). This is
    // the proof the fluid FEELS the wall velocity — identical mesh motion, only
    // the BC differs.
    assert!(
        off.max_du > 3.0 * on.max_du.max(1e-6),
        "moving-wall BC made no difference: ON drift {:.3e} vs OFF {:.3e}",
        on.max_du, off.max_du
    );
}

#[test]
fn moving_wall_freestream_preserved_cpu_euler() {
    moving_wall_freestream(TimeScheme::Euler, "euler");
}

#[test]
fn moving_wall_freestream_preserved_cpu_bdf2() {
    moving_wall_freestream(TimeScheme::BDF2, "bdf2");
}

/// Conservation: a closed all-walls box with a rigidly OSCILLATING internal
/// MovingWall obstacle. The rigid obstacle's area is invariant, so the fluid
/// area (box − obstacle) is constant ⇒ Σρ·V must not drift, and from rest the
/// wall-driven flow stays bounded.
#[test]
fn conservation_moving_wall_closed_box_cpu() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let cvt = build_cvt();
        let n = cvt.mesh.num_cells();
        let params = base_params(0.01);
        let rho = params.density as f64;

        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(0.0, 0.0); n], // from rest
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("build closed-box driver");
        moving.set_boundary_retag(Some(tag_wall_box));
        moving.driver_mut().apply_params(&params);
        moving.set_boundary_motion(BoundaryMotionSpec::RigidLoop {
            loop_index: OBSTACLE_LOOP,
            transform: oscillate,
        });
        moving.set_moving_wall_bc(true);

        let mass = |m: &Mesh| -> f64 { rho * m.cell_vol.iter().sum::<f64>() };
        let m0 = mass(moving.mesh());
        let mut m_prev = m0;
        let (mut max_step, mut max_total, mut max_u, mut max_scl) = (0.0, 0.0, 0.0f32, 0.0);
        let mut saw_moving_wall = false;

        for step in 0..30usize {
            let (outcome, stats) = moving
                .step(false)
                .unwrap_or_else(|e| panic!("step {step} failed: {e}"));
            assert!(outcome.diverged.is_none(), "step {step} diverged");
            assert_eq!(stats.n_cells, n, "step {step}: cell count changed");
            max_scl = f64::max(max_scl, stats.scl_defect);

            let m_n = mass(moving.mesh());
            max_step = f64::max(max_step, (m_n - m_prev).abs() / m0);
            max_total = f64::max(max_total, (m_n - m0).abs() / m0);
            m_prev = m_n;

            let mesh = moving.mesh();
            saw_moving_wall |= (0..mesh.num_faces())
                .any(|f| mesh.face_boundary[f] == Some(BoundaryType::MovingWall));

            let (uv, _p) = read_state(&moving);
            for c in 0..n {
                assert!(uv[c][0].is_finite() && uv[c][1].is_finite(), "step {step}: non-finite U");
                max_u = max_u.max(uv[c][0].abs()).max(uv[c][1].abs());
            }
        }
        (max_step, max_total, max_u, max_scl, saw_moving_wall)
    });
    std::env::remove_var("CFD2_BACKEND");
    let (max_step, max_total, max_u, max_scl, saw_moving_wall) = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    println!(
        "[m6.2-conservation] cpu/bdf2: per-step mass drift = {:.3e}, total = {:.3e}, \
         max|U| = {:.3e}, SCL = {:.3e}, moving-wall tagged = {saw_moving_wall} (30 steps)",
        max_step, max_total, max_u, max_scl
    );
    assert!(saw_moving_wall, "obstacle was never tagged MovingWall");
    // Closed rigid box+hole: fluid area is invariant ⇒ Σρ·V is constant to f64.
    assert!(max_step < 1e-12, "per-step mass drift {:.3e}", max_step);
    assert!(max_total < 1e-12, "total mass drift {:.3e}", max_total);
    // The wall-driven flow stays bounded (no blow-up from the moving BC).
    assert!(max_u.is_finite() && max_u < 1.0, "unbounded wall-driven velocity {:.3e}", max_u);
    assert!(max_scl < 1e-6, "SCL defect {:.3e}", max_scl);
}

/// Do-no-harm: enabling `moving_wall_bc` on a STATIC boundary must change
/// nothing — no face is tagged MovingWall, the regen is byte-identical, w_wall
/// stays zero. (Complements the stage-1 static do-no-harm anchor.)
#[test]
fn moving_wall_bc_static_do_no_harm() {
    let _g = ENV_LOCK.lock().unwrap();
    let cvt_ref = build_cvt();
    let vx0 = cvt_ref.mesh.vx.clone();
    let vy0 = cvt_ref.mesh.vy.clone();
    let n_cells = cvt_ref.mesh.num_cells();
    drop(cvt_ref);

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
    // Enable the moving-wall BC but leave the boundary Static.
    moving.set_moving_wall_bc(true);
    moving.set_boundary_motion(BoundaryMotionSpec::Static);

    for step in 0..12 {
        let (_o, stats) = moving.step(false).expect("step");
        let mesh = moving.mesh();
        assert!(
            mesh.vx == vx0 && mesh.vy == vy0,
            "step {step}: regen NOT byte-identical with moving_wall_bc on a Static boundary"
        );
        assert_eq!(stats.scl_defect, 0.0, "step {step}: nonzero SCL on a static step");
        assert!(
            (0..mesh.num_faces()).all(|f| mesh.face_boundary[f] != Some(BoundaryType::MovingWall)),
            "step {step}: a face was tagged MovingWall under a Static boundary"
        );
        assert!(
            moving.w_wall().iter().all(|w| *w == [0.0, 0.0]),
            "step {step}: nonzero w_wall under a Static boundary"
        );
    }
    println!("moving_wall_bc_static_do_no_harm: 12 steps byte-identical, no MovingWall faces");
}
