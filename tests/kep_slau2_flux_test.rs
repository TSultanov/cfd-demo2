//! Physics gates for the KEP and SLAU2 compressible flux families.
//!
//! These run on the CPU kernel interpreter (the executable oracle for the
//! typed codegen programs — no GPU adapter needed) and pin the properties
//! the two schemes were introduced for:
//!
//!  * free-stream preservation (exact quiescent state on both topologies),
//!  * KEP: the ONLY momentum dissipation is the physical viscosity — a
//!    resolved Taylor–Green vortex decays at the analytic 4*nu*k^2 rate,
//!    with NO wavespeed-scale artificial viscosity (the c-scale Kurganov
//!    dissipation measured ~1000x physical on the explicit obstacle at low
//!    Mach is absent by construction),
//!  * SLAU2: flow-speed-scale upwind dissipation only — the same vortex
//!    stays within a whisker of the physical decay rate,
//!  * contrast: first-order central-upwind (Kurganov) dissipation on the
//!    same vortex is several times the physical rate, pinning that the gate
//!    actually discriminates.
#![cfg(all(feature = "dev-tests", feature = "cpu"))]

use cfd2::solver::cpu::structured::{BcComp, StructuredGrid, StructuredModelSolver};
use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, generate_structured_rect_mesh_periodic, BoundarySides,
};
use cfd2::solver::model::{compressible_model, compressible_structured_model};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SteppingMode, TimeScheme};

/// Uniform rest state on the unstructured topology must be preserved exactly
/// (up to f32 roundoff) by both new flux families: any sign error, missing
/// gauge grouping, or asymmetric pressure term shows up here immediately.
#[test]
fn kep_and_slau2_uniform_freestream_is_preserved_unstructured() {
    for scheme in [Scheme::Kep, Scheme::Slau2] {
        let mesh = generate_structured_rect_mesh(4, 4, 1.0, 1.0, BoundarySides::wall());
        let mut solver = CpuSolver::with_stepping(
            &mesh,
            compressible_model().expect("compressible"),
            scheme,
            TimeScheme::RK4,
            SteppingMode::Explicit,
            CpuBackendConfig::default(),
        )
        .expect("explicit compressible solver");
        assert!(solver.debug_is_fully_matrix_free());
        solver.set_dt(1.0e-3);
        solver.set_density(1.0);
        assert!(solver.set_eos_param("eos.gm1", 0.4));
        assert!(solver.set_eos_param("eos.r", 1.0));
        solver
            .set_field_scalar("rho", &vec![1.0; mesh.num_cells()])
            .unwrap();
        solver
            .set_field_vec2("rho_u", &vec![(0.0, 0.0); mesh.num_cells()])
            .unwrap();
        solver
            .set_field_scalar("rho_e", &vec![2.5; mesh.num_cells()])
            .unwrap();
        solver
            .set_field_scalar("p", &vec![1.0; mesh.num_cells()])
            .unwrap();
        solver
            .set_field_scalar("T", &vec![1.0; mesh.num_cells()])
            .unwrap();
        solver
            .set_field_vec2("u", &vec![(0.0, 0.0); mesh.num_cells()])
            .unwrap();
        solver.initialize_history();

        solver.step();

        for (name, exact) in [("rho", 1.0), ("rho_e", 2.5), ("p", 1.0), ("T", 1.0)] {
            let error = solver
                .get_field_scalar(name)
                .unwrap()
                .into_iter()
                .map(|v| (v - exact).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error < 2.0e-6,
                "{scheme:?}: freestream {name} drifted by {error:e}"
            );
        }
    }
}

/// Same rest-state preservation on the dense-Cartesian structured topology.
#[test]
fn kep_and_slau2_uniform_freestream_is_preserved_structured() {
    for scheme in [Scheme::Kep, Scheme::Slau2] {
        let model = compressible_structured_model().expect("structured compressible");
        let mut solver = StructuredModelSolver::with_config(
            StructuredGrid::new(4, 4, 1.0, 1.0),
            &model,
            1.0e-3,
            1,
            scheme,
            TimeScheme::RK4,
        )
        .expect("structured explicit compressible solver");
        assert!(solver.debug_is_fully_matrix_free());
        solver.set_named_field("rho", |_x, _y| 1.0);
        solver.set_named_field("rho_u", |_x, _y| 0.0);
        let rho_u_y = solver.field_offset("rho_u").unwrap() + 1;
        solver.set_state(rho_u_y, |_x, _y| 0.0);
        solver.set_named_field("rho_e", |_x, _y| 2.5);
        solver.set_named_field("u", |_x, _y| 0.0);
        let u_y = solver.field_offset("u").unwrap() + 1;
        solver.set_state(u_y, |_x, _y| 0.0);
        solver.set_named_field("p", |_x, _y| 1.0);
        solver.set_named_field("T", |_x, _y| 1.0);
        let unknowns = solver.unknowns();
        solver.set_boundaries(move |_edge, _x, _y| {
            (
                3,
                vec![
                    BcComp {
                        kind: 0,
                        value: 0.0
                    };
                    unknowns
                ],
            )
        });

        solver.step();

        for (name, exact) in [("rho", 1.0), ("rho_e", 2.5), ("p", 1.0), ("T", 1.0)] {
            let values = solver.state_field(solver.field_offset(name).unwrap());
            let error = values
                .into_iter()
                .map(|v| (v - exact).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error < 2.0e-6,
                "{scheme:?}: structured freestream {name} drifted by {error:e}"
            );
        }
    }
}

/// Kinetic-energy decay rate of a resolved low-Mach Taylor–Green vortex on a
/// fully periodic mesh, per scheme. The analytic incompressible rate is
/// 4*nu*k^2 (each velocity component decays as exp(-2 nu k^2 t)).
fn taylor_green_decay_rate(scheme: Scheme, n: usize, mu: f64, steps: usize, dt: f64) -> f64 {
    let mesh = generate_structured_rect_mesh_periodic(n, n, 1.0, 1.0);
    let mut solver = CpuSolver::with_stepping(
        &mesh,
        compressible_model().expect("compressible"),
        scheme,
        TimeScheme::RK4,
        SteppingMode::Explicit,
        CpuBackendConfig::default(),
    )
    .expect("explicit compressible solver");
    assert!(solver.debug_is_fully_matrix_free());
    solver.set_dt(dt as f32);
    solver.set_density(1.0);
    solver.set_viscosity(mu as f32);
    assert!(solver.set_eos_param("eos.gm1", 0.4));
    assert!(solver.set_eos_param("eos.r", 1.0));

    let k = 2.0 * std::f64::consts::PI;
    let u0 = 0.02; // Mach ~ 0.017 at c = sqrt(1.4)
    let centers: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|c| (mesh.cell_cx[c], mesh.cell_cy[c]))
        .collect();
    let vel: Vec<(f64, f64)> = centers
        .iter()
        .map(|&(x, y)| {
            (
                u0 * (k * x).sin() * (k * y).cos(),
                -u0 * (k * x).cos() * (k * y).sin(),
            )
        })
        .collect();
    // Consistent incompressible TG pressure (rho = 1) and ideal-gas energy.
    let p: Vec<f64> = centers
        .iter()
        .map(|&(x, y)| 1.0 - (u0 * u0 / 4.0) * ((2.0 * k * x).cos() + (2.0 * k * y).cos()))
        .collect();
    let rho_e: Vec<f64> = p
        .iter()
        .zip(vel.iter())
        .map(|(&p, &(ux, uy))| p / 0.4 + 0.5 * (ux * ux + uy * uy))
        .collect();

    solver
        .set_field_scalar("rho", &vec![1.0; mesh.num_cells()])
        .unwrap();
    solver.set_field_vec2("rho_u", &vel).unwrap();
    solver.set_field_scalar("rho_e", &rho_e).unwrap();
    solver.set_field_scalar("p", &p).unwrap();
    solver.set_field_scalar("T", &p).unwrap(); // T = p/(rho*R), rho = R = 1
    solver.set_field_vec2("u", &vel).unwrap();
    solver.initialize_history();

    let ke = |solver: &CpuSolver| -> f64 {
        solver
            .get_field_vec2("u")
            .unwrap()
            .into_iter()
            .map(|(ux, uy)| 0.5 * (ux * ux + uy * uy))
            .sum()
    };
    let ke0 = ke(&solver);
    for _ in 0..steps {
        solver.step();
    }
    let ke1 = ke(&solver);
    assert!(
        ke1.is_finite() && ke1 > 0.0,
        "{scheme:?}: Taylor–Green KE not positive/finite after {steps} steps: {ke1}"
    );
    -(ke1 / ke0).ln() / (steps as f64 * dt)
}

/// KEP carries ZERO artificial momentum dissipation: the resolved vortex
/// decays at the physical viscous rate. SLAU2's upwind dissipation follows
/// the flow speed, so at Mach ~0.02 it stays close to physical. First-order
/// central-upwind (Kurganov) dissipation runs at the SOUND speed and is
/// several times physical on the same vortex — pinning the discrimination
/// this arc exists for.
#[test]
fn taylor_green_decay_matches_physical_viscosity_per_scheme() {
    let n = 24;
    let mu = 5.0e-3;
    let dt = 0.01;
    let steps = 40;
    let k = 2.0 * std::f64::consts::PI;
    let expected = 4.0 * mu * k * k; // 0.7896 for mu = 5e-3

    let kep = taylor_green_decay_rate(Scheme::Kep, n, mu, steps, dt);
    eprintln!("Taylor–Green decay rates (physical {expected:.4}): kep={kep:.4}");
    assert!(
        (kep / expected - 1.0).abs() < 0.2,
        "KEP Taylor–Green decay rate {kep:.4} differs from physical {expected:.4} by more than 20%"
    );

    let slau2 = taylor_green_decay_rate(Scheme::Slau2, n, mu, steps, dt);
    assert!(
        (slau2 / expected - 1.0).abs() < 0.35,
        "SLAU2 Taylor–Green decay rate {slau2:.4} differs from physical {expected:.4} by more \
         than 35%"
    );

    let upwind = taylor_green_decay_rate(Scheme::Upwind, n, mu, steps, dt);
    assert!(
        upwind > 2.0 * expected,
        "first-order Kurganov decay rate {upwind:.4} is not >> physical {expected:.4}; the gate \
         has lost its discrimination"
    );

    let vanleer = taylor_green_decay_rate(Scheme::SecondOrderUpwindVanLeer, n, mu, steps, dt);
    eprintln!("slau2={slau2:.4} upwind={upwind:.4} vanleer={vanleer:.4}");
    assert!(
        vanleer >= kep * 0.98,
        "vanLeer central-upwind rate {vanleer:.4} below the dissipation-free KEP rate {kep:.4}"
    );
}
