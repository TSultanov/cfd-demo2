//! Prescribed-motion moving-boundary gates: an oscillating cylinder in a channel
//! whose boundary-bound seeds move RIGIDLY with it each step, staying exactly on
//! the moving wall, driven through the moving-mesh loop on CPU. Here the obstacle
//! contour is tagged `Wall` (not `MovingWall`); these gates validate mesh integrity
//! + seed tracking + the static do-no-harm anchor, not the flow response.
//!
//! Gates:
//!   * `moving_obstacle_mesh_stays_valid` — every step the obstacle-contour faces
//!     exist + are tagged `Wall`, zero untagged boundary faces, watertight (Wall
//!     length ≈ moving circumference), positive cell areas, closure, fixed cell
//!     count.
//!   * `boundary_seeds_track_the_wall` — every moving boundary seed stays on the
//!     analytic moving obstacle (distance to the moved centre equals the
//!     chord-midpoint radius to f64 roundoff) every step.
//!   * `static_boundary_is_byte_identical_do_no_harm` — a static obstacle (both
//!     `BoundaryMotionSpec::Static` AND a zero-amplitude `RigidLoop`): the regen is
//!     BYTE-IDENTICAL to the initial mesh every step and `w_wall` is all-zero.
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{ChannelWithObstacle, LloydConfig};
use cfd2::sim::{BoundaryMotionSpec, MeshMotionSpec, MovingMeshDriver, OscAxis, RuntimeParams};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the CPU tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

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

/// Faces on the obstacle contour: open faces whose centre is near the (moved)
/// circle.
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

/// Untagged boundary faces: open faces with no BoundaryType.
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

        // Cell count fixed (fixed-seed).
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

        // Obstacle-contour faces exist and are tagged Wall.
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
    let mut worst_wvel = 0.0f64;
    for step in 0..30 {
        let (_o, stats) = moving.step(false).expect("step");
        let dt = stats.dt;
        sim_t += stats.dt;
        let cx = OBS_CX + AMP * (OMEGA * sim_t).sin();
        let center = Point2::new(cx, OBS_CY);
        // The analytic wall velocity in x is the finite difference of the rigid
        // x-translation over exactly this step, `(oscillate(t̄) − oscillate(t̄−dt))/dt`
        // — the SAME (new−old)/dt `record_wall_velocity` computes. All moving-wall
        // seeds share it (rigid translation), so it is a single scalar.
        let w_analytic_x = AMP * ((OMEGA * sim_t).sin() - (OMEGA * (sim_t - dt)).sin()) / dt;
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
                // Its recorded material velocity MATCHES the analytic wall velocity
                // in x (magnitude, not just sign) and is 0 in y. Checking the
                // magnitude catches a wrong scale factor or a wrong-dt division —
                // things the y==0 / free-stream-drift checks alone would miss.
                worst_wvel = worst_wvel.max((w_wall[i][0] - w_analytic_x).abs());
                assert!(
                    (w_wall[i][0] - w_analytic_x).abs() < 1e-12,
                    "step {step} seed {i}: recorded w_wall.x {:.6e} != analytic {:.6e}",
                    w_wall[i][0], w_analytic_x
                );
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
    println!(
        "boundary_seeds_track_the_wall: worst radial error {worst:.3e} (tol 1e-9), \
         worst w_wall.x vs analytic {worst_wvel:.3e} (tol 1e-12)"
    );
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

// MovingWall ALE BC. `MovingWall` (bc index 5) is a per-face Dirichlet velocity in
// the incompressible_momentum(_ale) model, consumed by the assembly identically to
// the Inlet Dirichlet (kind 1 → `bc_neighbor_scalar` returns the prescribed value).
// Each regen re-tags the moving obstacle's open faces `MovingWall` and, after the
// ALE refresh, sets their per-face `bc_value` to the recorded wall velocity `w_wall`
// (re-applied every step because the topology seam resets per-face overrides).

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

/// The core physics gate: a rigidly translating obstacle in a uniform
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

/// Area preservation + boundedness: a closed all-walls box with a rigidly
/// OSCILLATING internal MovingWall obstacle, driven from rest.
///
/// **What Σρ·V measures.** For constant ρ, `mass = ρ·Σ cell_vol = ρ·(box_area −
/// obstacle_area)`. A RIGID map preserves a polygon's area exactly in f64, so
/// this sum is invariant *by construction* — a GEOMETRIC identity of the regen
/// (the obstacle polygon's area is conserved as it moves), NOT a statement about
/// the solver's mass conservation (the solved U/p field never enters the number).
/// So the sub-roundoff drift below is an area-preservation check, and the genuine
/// per-cell **mass-conservation-under-mesh-motion** guarantee is the SCL defect
/// `Σ_f σ·flux = ΔV_i/dt` (asserted here too, `< 1e-6`).
///
/// The solved field IS exercised — the from-rest wall-driven flow must stay
/// finite + bounded (the moving BC does not blow up).
#[test]
fn rigid_obstacle_area_preserved_and_bounded_cpu() {
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
        "[m6.2-area] cpu/bdf2: Σρ·V area-preservation drift = {:.3e}/step, {:.3e} total, \
         max|U| = {:.3e}, SCL (real mass conservation) = {:.3e}, moving-wall tagged = {saw_moving_wall} (30 steps)",
        max_step, max_total, max_u, max_scl
    );
    assert!(saw_moving_wall, "obstacle was never tagged MovingWall");
    // Area preservation: the rigid obstacle polygon's area (hence Σ cell_vol) is
    // invariant under the motion — a geometric identity of the regen, to f64.
    assert!(max_step < 1e-12, "per-step area drift {:.3e}", max_step);
    assert!(max_total < 1e-12, "total area drift {:.3e}", max_total);
    // Mass conservation under mesh motion is the SCL defect (the field-relevant
    // per-cell check the area sum does NOT capture).
    assert!(max_scl < 1e-6, "SCL defect {:.3e}", max_scl);
    // The from-rest wall-driven flow stays finite + bounded (no blow-up).
    assert!(max_u.is_finite() && max_u < 1.0, "unbounded wall-driven velocity {:.3e}", max_u);
}

/// No-penetration measurement over a genuinely NON-co-moving field: a
/// cross-stream-oscillating obstacle in a QUIESCENT closed box.
struct NoPenOut {
    /// max over contour faces & steps of `|(U_owner − w_wall)·n̂|` — the near-wall
    /// relative NORMAL velocity at the OWNER-cell centroid. NB: this is inherently
    /// O(wall speed), NOT ≪ it — exact no-penetration is enforced at the wall FACE
    /// (Dirichlet `U_face = w_wall` + the ALE mesh flux ⇒ zero relative flux); the
    /// owner cell is a finite distance out, where the field varies by O(wall speed)
    /// over one cell. The discriminating signal is ON-vs-OFF (see the gate).
    max_no_pen: f64,
    /// max `|w_wall·n̂|` — the wall's own normal speed (proves `w_wall·n ≠ 0`, so
    /// this is NOT the co-moving/free-stream-preservation regime).
    max_wall_normal: f64,
    /// max `|U_owner − w_wall|` — the field is genuinely OFF the wall velocity
    /// (O(wall speed) ⇒ this is a real disturbed flow, not the trivial `U ≡ w_wall`
    /// co-moving field where the original no-pen number collapsed to free-stream
    /// preservation).
    max_rel: f64,
    max_scl: f64,
    /// min per-step contour-face count seen (the wall is resolved).
    min_faces: usize,
}

/// Drive the cross-stream-oscillating obstacle in a QUIESCENT closed box (all
/// outer walls, from rest) and measure the no-penetration residual over the
/// obstacle contour. With `moving_wall_on` the contour is a MovingWall Dirichlet
/// = `w_wall`; the control (`false`) oscillates the identical mesh but leaves the
/// contour a zero-velocity `Wall`, so the fluid is NOT told to track the wall.
///
/// Unlike `run_freestream` (where `U ≡ w_wall` everywhere, so the no-pen number is
/// free-stream preservation restated), here there is NO free stream — the ONLY
/// motion is driven by the oscillating wall, and `w_wall·n ≠ 0`, so `(U − w_wall)·n`
/// is a real, independent no-penetration measure (a streamwise stream instead swamps
/// the owner-cell normal with flow AROUND the cylinder). In the ON run the MovingWall
/// face set is asserted EQUAL to the geometric contour set (no untagged / plain-Wall
/// hole in the moving wall).
fn run_nopen(moving_wall_on: bool) -> NoPenOut {
    use std::collections::HashSet;
    let cvt = build_cvt();
    let n = cvt.mesh.num_cells();
    let mut params = base_params(0.01);
    params.viscosity = 1e-2;
    params.inlet_velocity = 0.0; // quiescent — no free stream to preserve
    params.time_scheme = TimeScheme::BDF2;

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen, // interior frozen; only the obstacle oscillates
        &vec![(0.0, 0.0); n],   // from rest
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("build no-pen driver");
    moving.set_boundary_retag(Some(tag_wall_box)); // closed box (all outer walls)
    moving.driver_mut().apply_params(&params);
    moving.set_boundary_motion(BoundaryMotionSpec::Oscillation {
        loop_index: OBSTACLE_LOOP,
        amplitude: OSC_AMP,
        omega: OSC_OMEGA,
        axis: OscAxis::CrossStream,
    });
    moving.set_moving_wall_bc(moving_wall_on);

    let mut out = NoPenOut {
        max_no_pen: 0.0,
        max_wall_normal: 0.0,
        max_rel: 0.0,
        max_scl: 0.0,
        min_faces: usize::MAX,
    };
    let mut sim_t = 0.0f64;

    for step in 0..80usize {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("nopen step {step} (on={moving_wall_on}): {e}"));
        assert!(outcome.diverged.is_none(), "nopen step {step} diverged");
        assert_eq!(stats.n_cells, n, "nopen step {step}: cell count changed");
        out.max_scl = out.max_scl.max(stats.scl_defect);
        sim_t += stats.dt;

        let (uv, _p) = read_state(&moving);
        let w_wall = moving.w_wall();
        let mesh = moving.mesh();

        // Geometric obstacle contour at the committed (moved) centre.
        let cy = OBS_CY + OSC_AMP * (OSC_OMEGA * sim_t).sin();
        let contour: Vec<usize> = obstacle_faces(mesh, Point2::new(OBS_CX, cy), OBS_R);
        out.min_faces = out.min_faces.min(contour.len());

        if moving_wall_on {
            // The MovingWall-tagged set == the geometric contour set, so no
            // obstacle face is a static-Wall no-penetration hole.
            let geo: HashSet<usize> = contour.iter().copied().collect();
            let tagged: HashSet<usize> = (0..mesh.num_faces())
                .filter(|&f| mesh.face_boundary[f] == Some(BoundaryType::MovingWall))
                .collect();
            assert_eq!(
                geo, tagged,
                "nopen step {step}: MovingWall set != geometric contour set (a moving-wall hole)"
            );
        }

        for &f in &contour {
            let owner = mesh.face_owner[f];
            let (nx, ny) = (mesh.face_nx[f], mesh.face_ny[f]);
            let rel = [
                uv[owner][0] as f64 - w_wall[owner][0],
                uv[owner][1] as f64 - w_wall[owner][1],
            ];
            out.max_no_pen = out.max_no_pen.max((rel[0] * nx + rel[1] * ny).abs());
            out.max_rel = out.max_rel.max((rel[0] * rel[0] + rel[1] * rel[1]).sqrt());
            let wn = (w_wall[owner][0] * nx + w_wall[owner][1] * ny).abs();
            out.max_wall_normal = out.max_wall_normal.max(wn);
        }
    }
    out
}

fn run_nopen_cpu(moving_wall_on: bool) -> NoPenOut {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_nopen(moving_wall_on));
    std::env::remove_var("CFD2_BACKEND");
    match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    }
}

/// The moving wall drives near-wall no-penetration in a genuinely NON-co-moving
/// field. A cross-stream-oscillating
/// cylinder (`w_wall·n ≠ 0`) in a QUIESCENT closed box — the ONLY motion is
/// wall-driven, so `(U−w)·n` is a real, independent measure, NOT the free-stream
/// preservation the co-moving `run_freestream` test collapses to.
///
/// HONEST framing: exact no-penetration is enforced at the wall FACE (Dirichlet
/// `U_face = w_wall` + ALE mesh flux ⇒ zero relative flux) — structural, no
/// residual to measure. At the OWNER-cell centroid the relative normal velocity
/// is inherently O(wall speed) (the field varies by that much over one near-wall
/// cell). The discriminating, non-trivial result is the CONTROL: imposing the
/// wall velocity (ON) pulls the near-wall normal velocity substantially toward the
/// wall's, well below the zero-velocity-wall control (OFF), whose residual equals
/// the full wall normal speed (the fluid, held at rest, doesn't follow the wall).
#[test]
fn no_penetration_cross_stream_oscillation_cpu() {
    let on = run_nopen_cpu(true);
    let off = run_nopen_cpu(false);
    let wall_speed = OSC_AMP * OSC_OMEGA;
    println!(
        "[m6.4-nopen] cross-stream osc in QUIESCENT box (wall speed A·ω = {:.3e}):",
        wall_speed
    );
    println!(
        "[m6.4-nopen]   ON : max|(U-w)·n| = {:.3e}, max|w·n| = {:.3e}, max|U-w| = {:.3e}, SCL = {:.3e}, faces >= {}",
        on.max_no_pen, on.max_wall_normal, on.max_rel, on.max_scl, on.min_faces
    );
    println!(
        "[m6.4-nopen]   OFF: max|(U-w)·n| = {:.3e} (zero-velocity wall control)",
        off.max_no_pen
    );

    // The wall genuinely moves normal to itself (this is NOT the co-moving regime).
    assert!(
        on.max_wall_normal > 0.3 * wall_speed,
        "wall normal speed {:.3e} too small vs A·ω {:.3e}",
        on.max_wall_normal, wall_speed
    );
    // The field is a REAL disturbed flow (|U−w| is O(wall speed)), NOT the trivial
    // U ≡ w_wall co-moving field where the original no-pen number was free-stream
    // preservation restated — so this measure is independent of that.
    assert!(
        on.max_rel > 0.3 * wall_speed,
        "field is (near) co-moving; no-pen not independent (|U-w| {:.3e})",
        on.max_rel
    );
    // The OFF control's residual is essentially the FULL wall normal speed — the
    // rest-held fluid does not follow the wall at all (a sanity anchor).
    assert!(
        off.max_no_pen > 0.8 * on.max_wall_normal,
        "OFF control residual {:.3e} unexpectedly small vs wall normal speed {:.3e}",
        off.max_no_pen, on.max_wall_normal
    );
    // CONTROLLED no-penetration: imposing w_wall drives the near-wall normal
    // velocity substantially toward the wall's — the ON residual is well below the
    // zero-velocity-wall OFF control (the proof the wall velocity, not the mesh
    // motion, drives no-penetration). Measured ≈33% reduction; gate at 15%.
    assert!(
        on.max_no_pen < 0.85 * off.max_no_pen,
        "moving-wall BC did not reduce penetration: ON {:.3e} vs OFF {:.3e}",
        on.max_no_pen, off.max_no_pen
    );
    assert!(on.max_scl < 1e-6, "SCL defect {:.3e}", on.max_scl);
}

/// Do-no-harm: enabling `moving_wall_bc` on a STATIC boundary must change
/// nothing — no face is tagged MovingWall, the regen is byte-identical, w_wall
/// stays zero.
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

/// GUI-minima oscillation (amplitude 0.005, freq 0.1 Hz — the slider floors):
/// the step-0 displacement `A·sin(ω·dt) ≈ 3e-5` is far below the near-wall cell
/// spacing, so the Voronoi CONNECTIVITY is unchanged and the driver would take
/// the surgical GEOMETRY seam. Identity at t=0 (`sin 0 = 0`).
fn oscillate_tiny(t: f64, p: [f64; 2]) -> [f64; 2] {
    [p[0] + 0.005 * (std::f64::consts::TAU * 0.1 * t).sin(), p[1]]
}

/// Regression: a `Wall → MovingWall` retag on a step whose motion is too small to
/// change the Voronoi connectivity must route
/// through the TOPOLOGY seam (which rebuilds `face_boundary` + BC tables), NOT
/// the geometry seam (which asserts identical `face_boundary` and would hard-error
/// on the tag flip — the crash reachable directly from the GUI slider minima).
#[test]
fn tiny_amplitude_moving_wall_first_step_ok_cpu() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let cvt = build_cvt();
        let n = cvt.mesh.num_cells();
        let params = base_params(0.01);
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(1.0, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("build");
        moving.driver_mut().apply_params(&params);
        moving.set_boundary_motion(BoundaryMotionSpec::RigidLoop {
            loop_index: OBSTACLE_LOOP,
            transform: oscillate_tiny,
        });
        moving.set_moving_wall_bc(true);
        // Step 0: tiny motion + the Wall→MovingWall retag. Must NOT error.
        let (_o, stats) = moving
            .step(false)
            .expect("step 0 must succeed (tag flip routes through the topology seam)");
        let mesh = moving.mesh();
        let n_mw = (0..mesh.num_faces())
            .filter(|&f| mesh.face_boundary[f] == Some(BoundaryType::MovingWall))
            .count();
        (stats.topo_changed, n_mw)
    });
    std::env::remove_var("CFD2_BACKEND");
    let (topo_changed, n_mw) = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    // The obstacle was actually re-tagged MovingWall (the retag ran)...
    assert!(n_mw > 10, "obstacle not tagged MovingWall: {n_mw} faces");
    // ...and the tag flip forced the topology seam (which rebuilds face_boundary).
    assert!(
        topo_changed,
        "the Wall→MovingWall tag flip must route through the topology seam"
    );
    println!(
        "tiny_amplitude_moving_wall_first_step_ok_cpu: {n_mw} MovingWall faces, topo_changed={topo_changed}"
    );
}

// The headline demo: a cross-stream forced sinusoidal cylinder oscillation
// (`BoundaryMotionSpec::Oscillation`) driven through the moving-mesh loop + the
// MovingWall BC over 2+ forcing periods, with:
//   * bounded/finite solution (max|U| < 10·U_scale over the whole run),
//   * wall boundary integrity every step (obstacle faces tagged MovingWall,
//     watertight, zero untagged, positive volumes, SCL closed),
//   * a NEAR-WALL QUALITY instrument (max skew + min cell volume within 3 cell
//     layers of the moving wall) reported + bounded every step (seeds
//     crowding/starving at a moving wall),
//   * a measurable FLOW RESPONSE to the forcing: the downstream transverse
//     velocity in the forced run is many times the static-control run (NOT a claim
//     of a specific shedding lock-in; Re is low so the static case is
//     steady/symmetric and any transverse signal is forced).

/// Cross-stream oscillation amplitude (< H so the frozen interior seeds near the
/// obstacle are squeezed but never swallowed — the near-wall instrument watches
/// exactly this margin).
const OSC_AMP: f64 = 0.015;
/// Oscillation angular frequency (period 0.4 s).
const OSC_OMEGA: f64 = std::f64::consts::TAU / 0.4;
/// Free-stream / forcing velocity scale for the demo (low Re ⇒ steady symmetric
/// static control, so the forced transverse response is unambiguous).
const U_SCALE: f32 = 0.5;

/// A downstream wake probe box (just behind the obstacle, spanning the wake) —
/// where a genuine flow RESPONSE to the cross-stream forcing shows up as
/// transverse velocity, as opposed to the transverse velocity the MovingWall BC
/// imposes right AT the wall.
fn in_wake_probe(x: f64, y: f64) -> bool {
    x > OBS_CX + 1.5 * OBS_R && x < OBS_CX + 6.0 * OBS_R && y > 0.3 && y < 0.7
}

/// Near-wall cell quality within `layers` cell layers of the moving wall.
/// Returns `(max_skew, min_vol, n_near_cells)` over the near-wall cell set:
/// `max_skew` is the worst face skew `1 − |d̂·n̂|` on any face incident to a
/// near-wall cell (same definition as `Mesh::calculate_max_skewness`), `min_vol`
/// the smallest near-wall cell area. Wall cells = owners of a `MovingWall` face;
/// the set is grown `layers` times over the cell-face adjacency (BFS).
fn near_wall_quality(mesh: &Mesh, layers: usize) -> (f64, f64, usize) {
    let n = mesh.num_cells();
    let mut near = vec![false; n];
    // Seed: cells owning a MovingWall boundary face.
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_none()
            && mesh.face_boundary[f] == Some(BoundaryType::MovingWall)
        {
            near[mesh.face_owner[f]] = true;
        }
    }
    // Grow the layer set over face adjacency.
    for _ in 0..layers {
        let mut add = vec![false; n];
        for c in 0..n {
            if !near[c] {
                continue;
            }
            let (s, e) = (mesh.cell_face_offsets[c], mesh.cell_face_offsets[c + 1]);
            for &f in &mesh.cell_faces[s..e] {
                let other = if mesh.face_owner[f] == c {
                    mesh.face_neighbor[f]
                } else {
                    Some(mesh.face_owner[f])
                };
                if let Some(o) = other {
                    add[o] = true;
                }
            }
        }
        for c in 0..n {
            near[c] |= add[c];
        }
    }
    let mut max_skew = 0.0f64;
    let mut min_vol = f64::INFINITY;
    let mut count = 0usize;
    for c in 0..n {
        if !near[c] {
            continue;
        }
        count += 1;
        min_vol = min_vol.min(mesh.cell_vol[c]);
        let (s, e) = (mesh.cell_face_offsets[c], mesh.cell_face_offsets[c + 1]);
        for &f in &mesh.cell_faces[s..e] {
            let owner = mesh.face_owner[f];
            let (dx, dy) = if let Some(nb) = mesh.face_neighbor[f] {
                (mesh.cell_cx[nb] - mesh.cell_cx[owner], mesh.cell_cy[nb] - mesh.cell_cy[owner])
            } else {
                (mesh.face_cx[f] - mesh.cell_cx[owner], mesh.face_cy[f] - mesh.cell_cy[owner])
            };
            let dn = (dx * dx + dy * dy).sqrt();
            if dn > 1e-12 {
                let dot = (dx / dn) * mesh.face_nx[f] + (dy / dn) * mesh.face_ny[f];
                max_skew = max_skew.max(1.0 - dot.abs());
            }
        }
    }
    if !min_vol.is_finite() {
        min_vol = 0.0;
    }
    (max_skew, min_vol, count)
}

struct DemoOut {
    /// Peak |U| over all cells and steps (bounded-solution watchdog).
    max_u: f32,
    /// Peak |v| (transverse) anywhere — dominated by the imposed wall BC.
    max_v_global: f32,
    max_scl: f64,
    /// Worst near-wall skew and smallest near-wall cell area over the run.
    worst_near_skew: f64,
    min_near_vol: f64,
    n_wall_faces: usize,
    flips: usize,
    steps: usize,
    periods: f64,
    saw_nonfinite: bool,
    /// Per-step SIGNED spatial mean of the transverse velocity `v` over the
    /// symmetric downstream wake box. In the steady + symmetric static control
    /// the up/down `v` cancels ⇒ this is ~0 and flat in time; cross-stream
    /// forcing pushes the wake fluid coherently ⇒ it oscillates with the
    /// forcing. Its temporal amplitude (peak-to-peak over the tail) is the
    /// honest forced-response signal (steady spatial |v| is a bad discriminator
    /// — flow diverting around a cylinder has large local |v| even when steady).
    wake_mean_v: Vec<f32>,
}

/// Drive the oscillating (or, as the control, static) cylinder demo. With
/// `oscillate_on` the obstacle cross-stream-oscillates and the MovingWall BC is
/// live; the control holds the obstacle static (`Frozen` interior, `Static`
/// boundary) in the identical channel so the ONLY difference is the forcing.
fn run_demo(oscillate_on: bool) -> DemoOut {
    let cvt = build_cvt();
    let n = cvt.mesh.num_cells();
    let mut params = base_params(0.01);
    params.viscosity = 1e-2;
    params.inlet_velocity = U_SCALE;
    params.time_scheme = TimeScheme::BDF2;

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen, // interior seeds frozen; only the wall moves
        &vec![(U_SCALE as f64, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("build oscillating-cylinder demo driver");
    moving.driver_mut().apply_params(&params);
    if oscillate_on {
        moving.set_boundary_motion(BoundaryMotionSpec::Oscillation {
            loop_index: OBSTACLE_LOOP,
            amplitude: OSC_AMP,
            omega: OSC_OMEGA,
            axis: OscAxis::CrossStream,
        });
        moving.set_moving_wall_bc(true);
    }

    let steps = 100usize; // 2.5 forcing periods at dt=0.01, period 0.4
    let mut out = DemoOut {
        max_u: 0.0,
        max_v_global: 0.0,
        max_scl: 0.0,
        worst_near_skew: 0.0,
        min_near_vol: f64::INFINITY,
        n_wall_faces: 0,
        flips: 0,
        steps,
        periods: 0.0,
        saw_nonfinite: false,
        wake_mean_v: Vec::with_capacity(steps),
    };
    let mut sim_t = 0.0f64;

    for step in 0..steps {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("demo step {step} (osc={oscillate_on}) failed: {e}"));
        assert!(outcome.diverged.is_none(), "demo step {step} diverged");
        assert_eq!(stats.n_cells, n, "demo step {step}: cell count changed (seed swallowed)");
        sim_t += stats.dt;
        out.max_scl = out.max_scl.max(stats.scl_defect);
        if stats.flipped {
            out.flips += 1;
        }

        let mesh = moving.mesh();

        // --- wall boundary integrity every step ---------------------------
        if oscillate_on {
            // Obstacle contour tagged MovingWall + watertight + zero untagged.
            let cy = OBS_CY + OSC_AMP * (OSC_OMEGA * sim_t).sin();
            let center = Point2::new(OBS_CX, cy);
            let contour = obstacle_faces(mesh, center, OBS_R);
            let mut wall_len = 0.0;
            let mut wall_faces = 0usize;
            for &f in &contour {
                assert_eq!(
                    mesh.face_boundary[f],
                    Some(BoundaryType::MovingWall),
                    "demo step {step}: obstacle face {f} not tagged MovingWall"
                );
                wall_len += mesh.face_area[f];
                wall_faces += 1;
            }
            out.n_wall_faces = out.n_wall_faces.max(wall_faces);
            let circ = std::f64::consts::TAU * OBS_R;
            assert!(
                wall_len > 0.9 * circ && wall_len < 1.05 * circ,
                "demo step {step}: obstacle wall length {wall_len:.4} vs circumference {circ:.4}"
            );
        }
        assert_eq!(
            untagged_boundary_faces(mesh),
            0,
            "demo step {step}: untagged boundary faces (free-slip holes)"
        );
        let (worst_closure, nonpos) = mesh_health(mesh);
        assert_eq!(nonpos, 0, "demo step {step}: {nonpos} non-positive cell volumes");
        assert!(worst_closure < 1e-6, "demo step {step}: worst closure {worst_closure:.3e}");
        assert!(stats.scl_defect < 1e-6, "demo step {step}: SCL defect {:.3e}", stats.scl_defect);

        // --- near-wall quality instrument ---------------------------------
        if oscillate_on {
            let (near_skew, near_min_vol, near_n) = near_wall_quality(mesh, 3);
            out.worst_near_skew = out.worst_near_skew.max(near_skew);
            out.min_near_vol = out.min_near_vol.min(near_min_vol);
            assert!(near_n > 0, "demo step {step}: empty near-wall cell set");
            assert!(
                near_skew < 0.7,
                "demo step {step}: near-wall skew {near_skew:.3} exceeded 0.7 (seeds crowding)"
            );
            assert!(
                near_min_vol > 0.0,
                "demo step {step}: near-wall min cell volume {near_min_vol:.3e} not positive (starved)"
            );
        }

        // --- bounded/finite + response signal -----------------------------
        let (uv, _p) = read_state(&moving);
        let (mut wake_sum, mut wake_n) = (0.0f64, 0usize);
        for c in 0..n {
            let (u, v) = (uv[c][0], uv[c][1]);
            if !u.is_finite() || !v.is_finite() {
                out.saw_nonfinite = true;
            }
            out.max_u = out.max_u.max(u.abs()).max(v.abs());
            out.max_v_global = out.max_v_global.max(v.abs());
            if in_wake_probe(mesh.cell_cx[c], mesh.cell_cy[c]) {
                wake_sum += v as f64;
                wake_n += 1;
            }
        }
        out.wake_mean_v
            .push(if wake_n > 0 { (wake_sum / wake_n as f64) as f32 } else { 0.0 });
    }
    if !out.min_near_vol.is_finite() {
        out.min_near_vol = 0.0;
    }
    out.periods = sim_t / 0.4;
    out
}

/// The headline demo gate: an oscillating cylinder in the channel, forced for
/// 2+ periods on the CPU, bounded + wall-integrity + near-wall quality + a
/// measurable flow response vs the static control.
#[test]
fn oscillating_cylinder_demo_responds_to_forcing_cpu() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let forced = run_demo(true);
        let control = run_demo(false);
        (forced, control)
    });
    std::env::remove_var("CFD2_BACKEND");
    let (forced, control) = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };

    // Temporal peak-to-peak of the signed wake-mean transverse velocity over the
    // TAIL (last 1.5 periods = 60 steps) — skips the startup transient. This is
    // the forced-response amplitude: the steady symmetric control is ~flat, the
    // forcing oscillates it.
    let peak_to_peak = |series: &[f32]| -> f32 {
        let tail = &series[series.len().saturating_sub(60)..];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in tail {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if hi >= lo { hi - lo } else { 0.0 }
    };
    let forced_amp = peak_to_peak(&forced.wake_mean_v);
    let control_amp = peak_to_peak(&control.wake_mean_v);

    println!(
        "[m6.3-demo] FORCED  : {} steps ({:.2} periods), max|U| = {:.3e}, max|v|(global) = {:.3e}, \
         wake-mean-v pk-pk = {:.3e}, near-wall skew = {:.3}, near-wall min vol = {:.3e}, \
         SCL = {:.3e}, wall faces = {}, flips = {}",
        forced.steps, forced.periods, forced.max_u, forced.max_v_global, forced_amp,
        forced.worst_near_skew, forced.min_near_vol, forced.max_scl, forced.n_wall_faces,
        forced.flips
    );
    println!(
        "[m6.3-demo] CONTROL : max|U| = {:.3e}, max|v|(global) = {:.3e}, wake-mean-v pk-pk = {:.3e} \
         (static obstacle, identical channel)",
        control.max_u, control.max_v_global, control_amp
    );

    // At least 2 forcing periods actually elapsed.
    assert!(forced.periods >= 2.0, "only {:.2} forcing periods elapsed", forced.periods);
    // The obstacle was actually tagged + driven as a MovingWall.
    assert!(forced.n_wall_faces > 10, "too few MovingWall faces: {}", forced.n_wall_faces);
    // Bounded + finite over the whole run.
    assert!(!forced.saw_nonfinite, "forced run produced a non-finite velocity");
    assert!(
        (forced.max_u as f64) < 10.0 * U_SCALE as f64,
        "forced max|U| {:.3e} not bounded by 10·U_scale = {}",
        forced.max_u, 10.0 * U_SCALE as f64
    );
    // GCL held through the whole forced run.
    assert!(forced.max_scl < 1e-6, "forced SCL defect {:.3e}", forced.max_scl);
    // Near-wall quality stayed healthy for the whole run.
    assert!(
        forced.worst_near_skew < 0.7 && forced.min_near_vol > 0.0,
        "near-wall quality degraded: skew {:.3}, min vol {:.3e}",
        forced.worst_near_skew, forced.min_near_vol
    );
    // FLOW RESPONSE: the forced run's downstream wake oscillates transversely
    // many times more than the steady symmetric control — a measurable signal
    // correlated with the forcing (NOT a claim of a specific shedding lock-in;
    // Re is low so the static case is steady + symmetric ⇒ ~flat wake-mean v).
    assert!(
        forced_amp > 5.0 * control_amp.max(1e-4),
        "no measurable forced response: forced wake-mean-v pk-pk {:.3e} vs control {:.3e}",
        forced_amp, control_amp
    );
    // The response is a real fraction of the wall speed (the transverse momentum
    // the wall injects reaches the wake).
    assert!(
        (forced_amp as f64) > 0.02 * (OSC_AMP * OSC_OMEGA),
        "forced wake response {:.3e} implausibly small vs wall speed {:.3e}",
        forced_amp, OSC_AMP * OSC_OMEGA
    );
}
