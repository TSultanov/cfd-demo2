#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

//! Solver-level regression test: on a Voronoi (and Delaunay) obstacle-channel
//! mesh the obstacle contour faces must act as a no-slip wall, not as an open
//! zero-gradient hole (BC-table row 0), so the incompressible solve cannot let
//! mass flow straight through the circle.

use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{
    generate_delaunay_mesh, generate_voronoi_mesh, BoundaryType, ChannelWithObstacle, Mesh,
};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::ops::ControlFlow;

fn obstacle_geo() -> ChannelWithObstacle {
    ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    }
}

/// Faces on the obstacle contour: open faces whose center is near the circle.
fn obstacle_faces(mesh: &Mesh, geo: &ChannelWithObstacle) -> Vec<usize> {
    (0..mesh.num_faces())
        .filter(|&f| {
            mesh.face_neighbor[f].is_none() && {
                let d = (Point2::new(mesh.face_cx[f], mesh.face_cy[f]) - geo.obstacle_center)
                    .norm();
                (d - geo.obstacle_radius).abs() < 0.5 * geo.obstacle_radius
            }
        })
        .collect()
}

fn run_case(kind: &str) {
    let geo = obstacle_geo();
    let domain_size = Vector2::new(geo.length, geo.height);
    let mut mesh = match kind {
        "voronoi" => generate_voronoi_mesh(&geo, 0.05, 0.05, 1.2, domain_size),
        "delaunay" => generate_delaunay_mesh(&geo, 0.05, 0.05, 1.2, domain_size),
        _ => unreachable!(),
    };
    mesh.smooth(&geo, 0.3, 50);

    // Mesh-level: the contour must be closed as walls.
    let contour = obstacle_faces(&mesh, &geo);
    assert!(
        contour.len() >= 8,
        "[{kind}] expected a resolved obstacle contour, found {} faces",
        contour.len()
    );
    for &f in &contour {
        assert_eq!(
            mesh.face_boundary[f],
            Some(BoundaryType::Wall),
            "[{kind}] obstacle contour face {f} at ({:.3},{:.3}) is not tagged Wall",
            mesh.face_cx[f],
            mesh.face_cy[f]
        );
    }

    // Solver-level: the wall must block through-flow. Uniform IC at the inlet
    // velocity; with no-slip contour faces the normal-velocity proxy through
    // the circle collapses within a few implicit steps, with an open hole it
    // stays at the free-stream level.
    let inlet: f32 = 1.0;
    let params = RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.001,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 8,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: inlet,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    };

    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        incompressible_momentum_model().expect("model"),
        &params,
        &vec![(f64::from(inlet), 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);

    let mut u: Vec<(f64, f64)> = Vec::new();
    let result = driver.run_steps(30, 1, |step, outcome| {
        if let Some(reason) = &outcome.diverged {
            panic!("[{kind}] step {step}: diverged: {reason:?}");
        }
        if let Some(rb) = &outcome.readback {
            assert_eq!(rb.stats.nonfinite_u, 0, "[{kind}] step {step}: non-finite u");
            u = rb.u.clone();
        }
        ControlFlow::Continue(())
    });
    assert!(
        result.diverged.is_none(),
        "[{kind}] diverged at step {:?}",
        result.stop_step
    );
    assert_eq!(u.len(), n, "[{kind}] missing final readback");

    // Normal-velocity proxy through the contour, area-weighted, from the
    // owner-cell velocities.
    let mut flux = 0.0;
    let mut area = 0.0;
    for &f in &contour {
        let o = mesh.face_owner[f];
        let un = u[o].0 * mesh.face_nx[f] + u[o].1 * mesh.face_ny[f];
        flux += mesh.face_area[f] * un.abs();
        area += mesh.face_area[f];
    }
    let mean_un = flux / area;
    println!("[{kind}] mean |U·n| at obstacle contour after 30 steps: {mean_un:.4}");
    assert!(
        mean_un < 0.3 * f64::from(inlet),
        "[{kind}] obstacle is leaking: mean |U·n| = {mean_un:.4} (inlet {inlet})"
    );
}

#[test]
fn voronoi_obstacle_contour_is_a_wall() {
    run_case("voronoi");
}

#[test]
fn delaunay_obstacle_contour_is_a_wall() {
    run_case("delaunay");
}
