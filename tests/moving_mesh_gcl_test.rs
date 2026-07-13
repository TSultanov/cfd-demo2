//! Prescribed seed motion through the full Voronoi-regen + swept-flux + refresh
//! + ALE loop. The seeds move and the Voronoi `Mesh` is regenerated every step,
//! so swept fluxes are built through the seed-set vertex correspondence
//! (`align_old_vertices_by_seed_set`, vertex ≡ seed-triple). `MovingMeshDriver`
//! owns the cycle; these tests prescribe the motion and audit the physics.
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::{assemble_meshless_from_seeds, generate_cvt_mesh_with_seeds};
use cfd2::meshgen::{LloydConfig, RectangularChannel};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::mesh::{
    align_old_vertices_by_seed_set, swept_mesh_fluxes_closed, BoundaryType, Mesh,
};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const LX: f64 = 1.0;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const DT: f32 = 0.005;
const STEPS: usize = 120;
/// Motion period (~1.5 swirl cycles over the run).
const PERIOD: f64 = 80.0 * DT as f64;
/// Horizontal free stream. Uniform `(U,0)` is an exact discrete fixed point:
/// inlet Dirichlet `(U,0)`, outlet zero-gradient, and slip walls `U·n=0` on the
/// (horizontal) top/bottom are all satisfied — so any drift is a GCL/solve
/// artifact, not BC physics. Horizontal lets the scalar inlet BC carry it, so
/// the topology seam's bc rebuild needs no vector re-override each step.
const U0: (f32, f32) = (1.0, 0.0);

/// Flip-free interior swirl: rotate each seed about the domain centre by an
/// angle modulated by a boundary-vanishing bump `sin²(πx/L)·sin²(πy/L)`, so the
/// deformation is smooth, zero at the walls (boundary faces stay static), and
/// small enough to never flip the Voronoi adjacency. Evaluated from the t=0
/// label (no incremental drift).
fn swirl(p: [f64; 2], t: f64) -> [f64; 2] {
    let (cx, cy) = (0.5 * LX, 0.5 * LY);
    let bump = (std::f64::consts::PI * p[0] / LX).sin().powi(2)
        * (std::f64::consts::PI * p[1] / LY).sin().powi(2);
    // Peak rotation ~0.03 rad at the centre, tapering to 0 at the walls — small
    // enough to keep the Voronoi adjacency flip-free over the run.
    let theta = 0.03 * (2.0 * std::f64::consts::PI * t / PERIOD).sin() * bump;
    let (dx, dy) = (p[0] - cx, p[1] - cy);
    let (c, s) = (theta.cos(), theta.sin());
    [cx + c * dx - s * dy, cy + s * dx + c * dy]
}

/// Rigid translation of the interior seed block (boundary seeds fixed) — the
/// flip-prone motion: fixed boundary seeds shear against the translating
/// interior. Used only by the flip probe.
fn rigid_at(p: [f64; 2], amp: f64) -> [f64; 2] {
    [p[0] + amp, p[1] + 0.6 * amp]
}

fn test_params(time_scheme: TimeScheme) -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: DT,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 1000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 6,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: U0.0,
        density: 1.0,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn square() -> (RectangularChannel, Vector2<f64>) {
    (
        RectangularChannel {
            length: LX,
            height: LY,
        },
        Vector2::new(LX, LY),
    )
}

/// Rewrite the boundary tags to a slip channel: Inlet left, Outlet right,
/// SlipWall top+bottom (the engine tags bottom/top as no-slip `Wall`, which
/// would kill a uniform flow). A `fn` pointer so it can be the driver's
/// per-regen retag hook. Only boundary faces (no neighbour) are touched.
fn tag_slip_channel(mesh: &mut Mesh) {
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
            continue;
        };
        mesh.face_boundary[f] = Some(bt);
    }
}

/// Rewrite every boundary face to `Wall` (the closed-box conservation case).
fn tag_all_walls(mesh: &mut Mesh) {
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_none() {
            mesh.face_boundary[f] = Some(BoundaryType::Wall);
        }
    }
}

struct GclOut {
    max_du: f32,
    max_dp: f32,
    early_du: f32,
    late_du: f32,
    early_dp: f32,
    late_dp: f32,
    max_scl_defect: f64,
    max_identity_err: f64,
    max_skew: f64,
    rebuilds: usize,
}

/// Drive the uniform-diagonal-flow GCL protocol through the moving-mesh driver.
fn run_gcl(time_scheme: TimeScheme) -> GclOut {
    let (geo, domain) = square();
    let params = test_params(time_scheme);
    let mut cvt =
        generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_slip_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("moving driver build");
    // Re-stamp the slip-channel tags on every regen (the topology seam rebuilds
    // the bc tables from the regenerated mesh's tags each step).
    moving.set_boundary_retag(Some(tag_slip_channel));
    moving.driver_mut().apply_params(&params);

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;
    let p_off = layout.offset_for("p").expect("p offset") as usize;

    let mut out = GclOut {
        max_du: 0.0,
        max_dp: 0.0,
        early_du: 0.0,
        late_du: 0.0,
        early_dp: 0.0,
        late_dp: 0.0,
        max_scl_defect: 0.0,
        max_identity_err: 0.0,
        max_skew: 0.0,
        rebuilds: 0,
    };

    for step in 0..STEPS {
        // A genuine flip makes step() return Err; the flip-free swirl must never
        // trigger it.
        let (outcome, stats) = moving.step(false).unwrap_or_else(|e| {
            panic!("step {step} failed (a flip means the amplitude is too large): {e}")
        });
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        out.max_scl_defect = out.max_scl_defect.max(stats.scl_defect);
        out.max_identity_err = out.max_identity_err.max(stats.identity_err);
        out.max_skew = out.max_skew.max(stats.max_skew);
        // `topo_changed` counts topology-seam rebuilds (any motion step), not
        // flips — flips would have Err'd above.
        if stats.topo_changed {
            out.rebuilds += 1;
        }
        assert!(
            stats.identity_err < 1e-10,
            "step {step}: f64 swept-quad identity {:.3e} above roundoff",
            stats.identity_err
        );
        assert!(
            stats.scl_defect < 1e-7,
            "step {step}: f32 SCL closure defect {:.3e} above roundoff",
            stats.scl_defect
        );

        if step % 10 != 9 && step + 1 != STEPS {
            continue;
        }
        let state = pollster::block_on(moving.driver().solver().read_state_f32());
        assert_eq!(state.len(), n * stride, "unexpected state layout");
        let (mut du, mut dp) = (0.0f32, 0.0f32);
        for c in 0..n {
            du = du
                .max((state[c * stride + u_off] - U0.0).abs())
                .max((state[c * stride + u_off + 1] - U0.1).abs());
            dp = dp.max(state[c * stride + p_off].abs());
        }
        out.max_du = out.max_du.max(du);
        out.max_dp = out.max_dp.max(dp);
        if (STEPS / 4..STEPS / 2).contains(&step) {
            out.early_du = out.early_du.max(du);
            out.early_dp = out.early_dp.max(dp);
        }
        if step >= 3 * STEPS / 4 {
            out.late_du = out.late_du.max(du);
            out.late_dp = out.late_dp.max(dp);
        }
        if std::env::var("CFD2_GCL_TRACE").as_deref() == Ok("1") {
            println!("[m4.2-gcl-trace] step {:4}: max|U-U0| = {du:.3e}, max|p| = {dp:.3e}", step + 1);
        }
    }
    out
}

fn run_gcl_cpu(scheme: TimeScheme, label: &str) -> GclOut {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_gcl(scheme));
    std::env::remove_var("CFD2_BACKEND");
    let out = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    println!(
        "[m4.2-gcl] cpu/{label}: max|U-U0| = {:.3e} (early {:.3e}, late {:.3e}), \
         max|p| = {:.3e} (early {:.3e}, late {:.3e}), SCL defect = {:.3e}, \
         f64 identity = {:.3e}, max skew = {:.3e}, topology rebuilds = {} ({STEPS} steps)",
        out.max_du, out.early_du, out.late_du, out.max_dp, out.early_dp, out.late_dp,
        out.max_scl_defect, out.max_identity_err, out.max_skew, out.rebuilds
    );
    out
}

/// Caps are ~3-4× the measured drift (125-cell CVT square, swirl θ_peak=0.03).
/// Completing all `STEPS` proves flip-free (a genuine flip Err's out of `step`).
fn assert_gcl_caps(out: &GclOut) {
    assert!(out.max_identity_err < 1e-11, "f64 identity {:.3e}", out.max_identity_err);
    assert!(out.max_scl_defect < 1e-8, "SCL defect {:.3e}", out.max_scl_defect);
    assert!(out.max_du < 6e-6, "U drift {:.3e} above pinned cap", out.max_du);
    assert!(out.max_dp < 8e-5, "p drift {:.3e} above pinned cap", out.max_dp);
    // Non-compounding: the final quarter must not exceed the early window by
    // much (a GCL violation compounds; solve noise saturates).
    assert!(
        out.late_du <= (out.early_du * 2.0).max(6e-6),
        "late U drift {:.3e} vs early {:.3e}: GCL error compounds",
        out.late_du, out.early_du
    );
    assert!(
        out.late_dp <= (out.early_dp * 2.0).max(8e-5),
        "late p drift {:.3e} vs early {:.3e}: GCL error compounds",
        out.late_dp, out.early_dp
    );
}

#[test]
fn gcl_moving_uniform_flow_preserved_cpu_euler() {
    let out = run_gcl_cpu(TimeScheme::Euler, "euler");
    assert_gcl_caps(&out);
}

#[test]
fn gcl_moving_uniform_flow_preserved_cpu_bdf2() {
    let out = run_gcl_cpu(TimeScheme::BDF2, "bdf2");
    assert_gcl_caps(&out);
}

struct ConsOut {
    max_area_err: f64,
    max_step_drift: f64,
    max_total_drift: f64,
    max_u: f32,
    max_scl_defect: f64,
    max_identity_err: f64,
}

fn run_conservation(time_scheme: TimeScheme) -> ConsOut {
    let (geo, domain) = square();
    let params = test_params(time_scheme);
    let mut cvt =
        generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_all_walls(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();
    let rho = params.density as f64;

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(0.0, 0.0); n], // from rest
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("moving driver build");
    // Keep every regen a closed box (the topology seam rebuilds bc from tags).
    moving.set_boundary_retag(Some(tag_all_walls));
    moving.driver_mut().apply_params(&params);

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;

    let area = LX * LY;
    let mass = |m: &Mesh| -> f64 { rho * m.cell_vol.iter().sum::<f64>() };
    let m0 = mass(moving.mesh());
    let mut m_prev = m0;

    let mut out = ConsOut {
        max_area_err: 0.0,
        max_step_drift: 0.0,
        max_total_drift: 0.0,
        max_u: 0.0,
        max_scl_defect: 0.0,
        max_identity_err: 0.0,
    };

    for step in 0..STEPS {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("step {step} failed: {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        out.max_scl_defect = out.max_scl_defect.max(stats.scl_defect);
        out.max_identity_err = out.max_identity_err.max(stats.identity_err);

        let m_n = mass(moving.mesh());
        out.max_step_drift = out.max_step_drift.max((m_n - m_prev).abs() / m0);
        out.max_total_drift = out.max_total_drift.max((m_n - m0).abs() / m0);
        out.max_area_err = out
            .max_area_err
            .max((moving.mesh().cell_vol.iter().sum::<f64>() - area).abs() / area);
        m_prev = m_n;

        if step % 10 == 9 || step + 1 == STEPS {
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            for c in 0..n {
                let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                assert!(ux.is_finite() && uy.is_finite(), "step {step}: non-finite U");
                out.max_u = out.max_u.max(ux.abs()).max(uy.abs());
            }
        }
    }
    out
}

#[test]
fn conservation_moving_closed_box_cpu() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_conservation(TimeScheme::BDF2));
    std::env::remove_var("CFD2_BACKEND");
    let out = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    println!(
        "[m4.2-conservation] cpu/bdf2: area err = {:.3e}, per-step mass drift = {:.3e}, \
         total = {:.3e}, max|U| = {:.3e}, SCL defect = {:.3e}, f64 identity = {:.3e} ({STEPS} steps)",
        out.max_area_err, out.max_step_drift, out.max_total_drift, out.max_u,
        out.max_scl_defect, out.max_identity_err
    );
    // Mesh-side identities: Σρ·V = ρ·(box area) exactly (the regen tiles the
    // fixed box) — f64 geometry precision; caps pinned with headroom.
    assert!(out.max_area_err < 1e-12, "Σ V vs box area: {:.3e}", out.max_area_err);
    assert!(out.max_step_drift < 1e-12, "per-step mass drift: {:.3e}", out.max_step_drift);
    assert!(out.max_total_drift < 1e-12, "total mass drift: {:.3e}", out.max_total_drift);
    // Zero-flow free-stream preservation: the exact solution is still fluid, so
    // any velocity is ALE-injected noise — f32 scale, non-compounding.
    assert!(out.max_u < 5e-4, "spurious ALE velocity: max|U| = {:.3e}", out.max_u);
    assert!(out.max_identity_err < 1e-10, "f64 identity {:.3e}", out.max_identity_err);
    assert!(out.max_scl_defect < 1e-7, "SCL defect {:.3e}", out.max_scl_defect);
}

/// How large a rigid interior translation the persistent-topology swept-flux
/// path tolerates before the Voronoi adjacency flips: at the swirl amplitude
/// the GCL/conservation gates use, zero flips; a rigid translation past a few
/// hundredths of h starts flipping.
#[test]
fn flip_frequency_vs_amplitude() {
    let (geo, domain) = square();
    let cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    let n = cvt.mesh.num_cells();
    println!("[m4.2-flip-probe] CVT square: {n} cells, {} faces, h = {H}", cvt.mesh.num_faces());

    // Rigid translation of interior seeds only (boundary fixed). Sweep the
    // amplitude as a fraction of h and report the first flip + a running count.
    for &frac in &[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0] {
        let amp = frac * H;
        let mut new_seeds: Vec<Point2<f64>> = cvt.seeds.clone();
        for (i, s0) in cvt.seeds.iter().enumerate() {
            if cvt.kinds[i] == cfd2::meshgen::meshless::SeedKind::Interior {
                let p = rigid_at([s0.x, s0.y], amp);
                new_seeds[i] = Point2::new(p[0], p[1]);
            }
        }
        let new_mesh = assemble_meshless_from_seeds(
            &new_seeds,
            &cvt.kinds,
            &cvt.spec,
            cvt.domain,
            cvt.min_cell_size,
        );
        let same_cells = new_mesh.num_cells() == n;
        let face_flip = !same_cells
            || new_mesh.num_faces() != cvt.mesh.num_faces()
            || new_mesh.face_owner != cvt.mesh.face_owner
            || new_mesh.face_neighbor != cvt.mesh.face_neighbor
            || new_mesh.cell_faces != cvt.mesh.cell_faces;
        // Vertex-level: does the seed-set correspondence lose any vertex?
        let (unmatched, identity) = if same_cells {
            let (ovx, ovy, um) =
                align_old_vertices_by_seed_set(&cvt.mesh, &new_mesh).expect("align");
            let id = swept_mesh_fluxes_closed(&new_mesh, &ovx, &ovy, DT as f64)
                .map(|s| s.max_identity_err_rel)
                .unwrap_or(f64::NAN);
            (um, id)
        } else {
            (usize::MAX, f64::NAN)
        };
        println!(
            "[m4.2-flip-probe] amp = {frac:.2}·h: face_flip = {face_flip}, unmatched verts = {}, \
             f64 identity = {:.3e}",
            if unmatched == usize::MAX { -1i64 } else { unmatched as i64 },
            identity
        );
    }
}
