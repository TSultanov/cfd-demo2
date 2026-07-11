//! Manufactured-solution gates for the matrix-free explicit RK4 path.
//!
//! These run directly on the CPU kernel interpreter, which is the executable
//! oracle for the typed codegen programs and does not require a GPU adapter.
#![cfg(all(feature = "dev-tests", feature = "cpu"))]

use cfd2::solver::cpu::structured::{BcComp, StructuredGrid, StructuredModelSolver};
use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SolverRecipe;
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, generate_structured_rect_mesh_periodic, BoundarySides,
    BoundaryType,
};
use cfd2::solver::model::{
    allmach_pressure_model, allmach_thermal_compressible_mms_model, allmach_thermal_model,
    allmach_thermal_structured_model, compressible_mms_biharmonic_model, compressible_model,
    compressible_structured_model, generic_diffusion_demo_mms_model,
    generic_diffusion_demo_structured_ibm_model, incompressible_momentum_ale_model,
    incompressible_momentum_model, scalar_transport_model, KernelId, ADVECTING_VELOCITY_FIELD,
    ALLMACH_MMS_SOURCE_P_FIELD, ALLMACH_MMS_SOURCE_T_FIELD, ALLMACH_MMS_SOURCE_U_FIELD,
    MMS_SOURCE_FIELD, SCALAR_TRANSPORT_FIELD, SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SteppingMode, TimeScheme};

/// A spatially uniform manufactured field makes the discrete Laplacian exactly
/// zero. This therefore pins residual signs, local ddt inversion, and all four
/// stage updates independently of spatial truncation error.
#[test]
fn uniform_source_is_integrated_exactly() {
    let mesh = generate_structured_rect_mesh(8, 8, 1.0, 1.0, BoundarySides::wall());
    let model = generic_diffusion_demo_mms_model().expect("model");
    let mut solver = CpuSolver::with_stepping(
        &mesh,
        model,
        Scheme::Upwind,
        TimeScheme::RK4,
        SteppingMode::Explicit,
        CpuBackendConfig::default(),
    )
    .expect("explicit solver");
    assert!(solver.debug_is_fully_matrix_free());
    solver.set_dt(0.125);
    solver
        .set_field_scalar("phi", &vec![1.0; mesh.num_cells()])
        .expect("phi");
    solver
        .set_field_scalar(MMS_SOURCE_FIELD, &vec![-2.0; mesh.num_cells()])
        .expect("source");
    solver.initialize_history();

    solver.step();

    let phi = solver.get_field_scalar("phi").expect("read phi");
    let max_err = phi
        .iter()
        .map(|&v| (v - 0.75_f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 2.0e-6, "uniform RK4 source error = {max_err:e}");
}

fn observed_orders(dts: &[f64], errors: &[f64]) -> Vec<f64> {
    dts.windows(2)
        .zip(errors.windows(2))
        .map(|(h, e)| (e[0] / e[1]).ln() / (h[0] / h[1]).ln())
        .collect()
}

/// Temporal MMS on the unstructured face-based operator. On an orthogonal
/// Cartesian mesh, cos(pi*x)cos(pi*y) is an exact eigenvector of the discrete
/// zero-Neumann Laplacian. Its discrete eigenvalue is known analytically, so
/// spatial error is removed and the test measures only RK4 time error.
#[test]
fn unstructured_temporal_order_is_four() {
    let n = 4usize;
    let t_end = 0.2_f64;
    let lambda_h = 8.0 * (n * n) as f64 * (std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2);
    let exact_scale = (-lambda_h * t_end).exp();
    let mut errors = Vec::new();
    let dts = [0.02_f64, 0.01, 0.005];

    for &dt in &dts {
        let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
        let model = generic_diffusion_demo_mms_model().expect("model");
        let mut solver = CpuSolver::with_stepping(
            &mesh,
            model,
            Scheme::Upwind,
            TimeScheme::RK4,
            SteppingMode::Explicit,
            CpuBackendConfig::default(),
        )
        .expect("explicit solver");
        solver.set_dt(dt as f32);
        let initial: Vec<f64> = (0..mesh.num_cells())
            .map(|i| {
                (std::f64::consts::PI * mesh.cell_cx[i]).cos()
                    * (std::f64::consts::PI * mesh.cell_cy[i]).cos()
            })
            .collect();
        solver.set_field_scalar("phi", &initial).expect("phi");
        solver
            .set_field_scalar(MMS_SOURCE_FIELD, &vec![0.0; mesh.num_cells()])
            .expect("source");
        solver.initialize_history();
        for _ in 0..(t_end / dt).round() as usize {
            solver.step();
        }
        let got = solver.get_field_scalar("phi").expect("read phi");
        let l2 = (got
            .iter()
            .zip(initial.iter())
            .map(|(&v, &v0)| (v - exact_scale * v0).powi(2))
            .sum::<f64>()
            / mesh.num_cells() as f64)
            .sqrt();
        errors.push(l2);
    }

    let orders = observed_orders(&dts, &errors);
    eprintln!("[mms][explicit-rk4][unstructured] dt={dts:?} error={errors:?} order={orders:?}");
    assert!(
        orders.iter().all(|&p| p > 3.5),
        "unstructured RK4 order below four: {orders:?}, errors={errors:?}"
    );
}

/// Structured temporal MMS using the declared Brinkman reaction term. A
/// uniform field and zero-gradient faces remove diffusion, leaving the exact
/// autonomous ODE phi'=-4 phi. This also exercises the transformation of an
/// implicit reaction declaration into a direct matrix-free residual.
#[test]
fn structured_temporal_order_is_four() {
    let t_end = 0.5_f64;
    let exact = (-4.0 * t_end).exp();
    let dts = [0.125_f64, 0.0625, 0.03125];
    let mut errors = Vec::new();

    for &dt in &dts {
        let model = generic_diffusion_demo_structured_ibm_model().expect("model");
        let grid = StructuredGrid::new(4, 4, 1.0, 1.0);
        let mut solver = StructuredModelSolver::with_config(
            grid,
            &model,
            dt,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured explicit solver");
        assert!(solver.debug_is_fully_matrix_free());
        solver.set_named_field("phi", |_x, _y| 1.0);
        solver.set_named_field("ibm_penalty", |_x, _y| -4.0);
        solver.set_boundaries(|_edge, _x, _y| {
            (
                3,
                vec![BcComp {
                    kind: 0,
                    value: 0.0,
                }],
            )
        });
        for _ in 0..(t_end / dt).round() as usize {
            solver.step();
        }
        let phi = solver.state_field(solver.field_offset("phi").expect("phi offset"));
        let l2 = (phi.iter().map(|&v| (v - exact).powi(2)).sum::<f64>() / phi.len() as f64).sqrt();
        errors.push(l2);
    }

    let orders = observed_orders(&dts, &errors);
    eprintln!("[mms][explicit-rk4][structured] dt={dts:?} error={errors:?} order={orders:?}");
    assert!(
        orders.iter().all(|&p| p > 3.5),
        "structured RK4 order below four: {orders:?}, errors={errors:?}"
    );
}

/// Every runtime reconstruction option must remain reachable through the
/// matrix-free residual. This lean manufactured transient is intentionally a
/// smoke/order-seam test: the stronger spatial convergence rates remain pinned
/// by `mms_scalar_transport_order_test` and are shared with the implicit path.
#[test]
fn unstructured_supports_every_spatial_scheme() {
    let sides = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Outlet,
        bottom: BoundaryType::Wall,
        top: BoundaryType::Wall,
    };
    let mesh = generate_structured_rect_mesh(6, 6, 1.0, 1.0, sides);
    let exact = |x: f64, y: f64| {
        1.0 + 0.1 * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
    };
    let schemes = [
        Scheme::Upwind,
        Scheme::SecondOrderUpwind,
        Scheme::QUICK,
        Scheme::SecondOrderUpwindMinMod,
        Scheme::SecondOrderUpwindVanLeer,
        Scheme::QUICKMinMod,
        Scheme::QUICKVanLeer,
    ];

    for scheme in schemes {
        let mut solver = CpuSolver::with_stepping(
            &mesh,
            scalar_transport_model().expect("model"),
            scheme,
            TimeScheme::RK4,
            SteppingMode::Explicit,
            CpuBackendConfig::default(),
        )
        .expect("explicit scalar solver");
        assert!(solver.debug_is_fully_matrix_free());
        solver.set_dt(1.0e-3);
        solver
            .set_field_scalar(
                SCALAR_TRANSPORT_FIELD,
                &(0..mesh.num_cells())
                    .map(|i| exact(mesh.cell_cx[i], mesh.cell_cy[i]))
                    .collect::<Vec<_>>(),
            )
            .expect("scalar");
        solver
            .set_field_vec2(
                ADVECTING_VELOCITY_FIELD,
                &vec![(0.2, 0.1); mesh.num_cells()],
            )
            .expect("velocity");
        solver
            .set_field_scalar(
                SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
                &vec![0.0; mesh.num_cells()],
            )
            .expect("source");
        let face_value =
            |face: u32| exact(mesh.face_cx[face as usize], mesh.face_cy[face as usize]) as f32;
        for boundary in [
            GpuBoundaryType::Inlet,
            GpuBoundaryType::Outlet,
            GpuBoundaryType::Wall,
        ] {
            solver
                .set_boundary_values_per_face(boundary, SCALAR_TRANSPORT_FIELD, 0, &face_value)
                .expect("boundary values");
        }
        solver.initialize_history();
        solver.step();
        let field = solver
            .get_field_scalar(SCALAR_TRANSPORT_FIELD)
            .expect("read scalar");
        assert!(
            field
                .iter()
                .all(|v| v.is_finite() && (0.5..1.5).contains(v)),
            "{scheme:?} produced an invalid explicit field"
        );
    }
}

#[test]
fn explicit_capability_is_derived_from_the_model_math() {
    for model in [
        generic_diffusion_demo_mms_model().expect("diffusion"),
        scalar_transport_model().expect("transport"),
        compressible_model().expect("compressible"),
        compressible_structured_model().expect("structured compressible"),
    ] {
        model
            .validate_explicit_rk4()
            .unwrap_or_else(|e| panic!("{} should support explicit RK4: {e}", model.id));
    }

    let saddle = incompressible_momentum_model().expect("incompressible");
    assert!(
        saddle.validate_explicit_rk4().is_err(),
        "a pressure-constraint row must not be presented as an explicit ODE"
    );
    let ale = incompressible_momentum_ale_model().expect("ALE");
    assert!(
        ale.validate_explicit_rk4().is_err(),
        "ALE is intentionally deferred"
    );
    let biharmonic = compressible_mms_biharmonic_model().expect("biharmonic");
    assert!(
        biharmonic.validate_explicit_rk4().is_err(),
        "auxiliary elliptic constraint rows need a dedicated explicit closure"
    );
    // Pressure-based all-Mach models declare a stage-local EOS + Rhie--Chow
    // closure, making their U/p(/T) rows a genuine local-mass ODE.
    for model in [
        allmach_pressure_model().expect("allmach"),
        allmach_thermal_model().expect("allmach thermal"),
        allmach_thermal_structured_model().expect("structured allmach thermal"),
    ] {
        model
            .validate_explicit_rk4()
            .unwrap_or_else(|e| panic!("{} should support explicit RK4: {e}", model.id));
    }
}

#[test]
fn allmach_explicit_recipe_closes_before_flux_without_implicit_corrections() {
    let model = allmach_thermal_model().expect("allmach thermal");
    let recipe = SolverRecipe::from_model(
        &model,
        Scheme::Upwind,
        TimeScheme::RK4,
        PreconditionerType::Jacobi,
        SteppingMode::Explicit,
    )
    .expect("explicit recipe");
    let ids: Vec<&str> = recipe
        .kernels
        .iter()
        .map(|kernel| kernel.id.as_str())
        .collect();
    let pos = |id: KernelId| {
        ids.iter()
            .position(|candidate| *candidate == id.as_str())
            .unwrap_or_else(|| panic!("missing explicit kernel {}", id.as_str()))
    };
    assert!(pos(KernelId::FLUX_MODULE_GRADIENTS) < pos(KernelId::EXPLICIT_PRIMITIVE_RECOVERY));
    assert!(pos(KernelId::EXPLICIT_PRIMITIVE_RECOVERY) < pos(KernelId::FLUX_MODULE));
    assert!(pos(KernelId::FLUX_MODULE) < pos(KernelId::EXPLICIT_RESIDUAL_GRAD_STATE));
    for implicit_id in [
        "dp_init",
        "dp_update_from_diag",
        "rhie_chow/store_grad_p",
        "rhie_chow/grad_p_update",
        "rhie_chow/correct_velocity_delta",
        "generic_coupled_update",
    ] {
        assert!(
            !ids.contains(&implicit_id),
            "implicit-only kernel {implicit_id} leaked into explicit recipe: {ids:?}"
        );
    }
}

fn run_uniform_allmach_thermal_ode(dt: f64, t_end: f64) -> (f64, f64) {
    let mesh = generate_structured_rect_mesh_periodic(2, 2, 1.0, 1.0);
    let mut solver = CpuSolver::with_stepping(
        &mesh,
        allmach_thermal_compressible_mms_model().expect("thermal MMS model"),
        Scheme::Upwind,
        TimeScheme::RK4,
        SteppingMode::Explicit,
        CpuBackendConfig::default(),
    )
    .expect("explicit thermal solver");
    let n = mesh.num_cells();
    solver.set_dt(dt as f32);
    solver.set_density(1.0);
    solver.set_viscosity(0.0);
    solver.set_alpha_u(0.7);
    solver.set_field_vec2("U", &vec![(0.0, 0.0); n]).unwrap();
    for (name, value) in [
        ("p", 0.0),
        ("d_p", 0.01),
        ("psi", 0.1),
        ("psi_precond", 0.14),
        ("dt_local", 0.02),
        ("rho", 1.0),
        ("T", 1.0),
        ("rho_t_ref", 1.0),
        ("rho_dT", -1.0),
        ("u_dot_grad_p", 0.0),
        ("rho_floor", 1.0e-12),
        ("t_ref", 1.0),
        ("psi_ref", 0.1),
        ("u_ref", 10.0),
        ("precond_mask", 1.0),
        (ALLMACH_MMS_SOURCE_P_FIELD, 0.5),
        (ALLMACH_MMS_SOURCE_T_FIELD, 4.0),
    ] {
        solver
            .set_field_scalar(name, &vec![value; n])
            .unwrap_or_else(|error| panic!("seed {name}: {error}"));
    }
    solver
        .set_field_vec2(ALLMACH_MMS_SOURCE_U_FIELD, &vec![(0.0, 0.0); n])
        .unwrap();
    solver.initialize_history();
    let steps = (t_end / dt).round() as usize;
    assert!(((steps as f64) * dt - t_end).abs() < 1.0e-12);
    for _ in 0..steps {
        solver.step();
    }
    (
        solver.get_field_scalar("p").unwrap()[0],
        solver.get_field_scalar("T").unwrap()[0],
    )
}

fn reference_uniform_allmach_thermal_ode(t_end: f64) -> (f64, f64) {
    fn rate(p: f64, t: f64) -> (f64, f64) {
        let psi_ref = 0.1;
        let rho_numer = 1.0 + 1.4 * psi_ref * p;
        let rho = rho_numer / t;
        let rho_dt = -rho_numer / (t * t);
        let psi = (psi_ref / t).max(psi_ref);
        // The pressure-row storage coefficient includes the temperature-row
        // Schur offset. Eliminating T leaves the requested target
        // max(psi,1/u_ref^2), rather than the under-scaled (2-gamma)*psi.
        let psi_precond = 0.4 * psi_ref / t + psi.max(0.01);
        let c = -0.4 * psi_ref;
        let det = psi_precond * rho - rho_dt * c;
        (
            (0.5 * rho - 4.0 * rho_dt) / det,
            (4.0 * psi_precond - 0.5 * c) / det,
        )
    }

    let dt = 1.0e-5;
    let steps = (t_end / dt).round() as usize;
    let (mut p, mut t) = (0.0, 1.0);
    for _ in 0..steps {
        let k1 = rate(p, t);
        let k2 = rate(p + 0.5 * dt * k1.0, t + 0.5 * dt * k1.1);
        let k3 = rate(p + 0.5 * dt * k2.0, t + 0.5 * dt * k2.1);
        let k4 = rate(p + dt * k3.0, t + dt * k3.1);
        p += dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        t += dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
    }
    (p, t)
}

#[test]
fn allmach_thermal_local_mass_block_retains_rk4_temporal_order() {
    let t_end = 0.8;
    let reference = reference_uniform_allmach_thermal_ode(t_end);
    let dts = [0.08, 0.04, 0.02];
    let component_errors: Vec<(f64, f64)> = dts
        .iter()
        .map(|&dt| {
            let value = run_uniform_allmach_thermal_ode(dt, t_end);
            ((value.0 - reference.0).abs(), (value.1 - reference.1).abs())
        })
        .collect();
    let errors: Vec<f64> = component_errors.iter().map(|&(p, t)| p.max(t)).collect();
    let orders = observed_orders(&dts, &errors);
    eprintln!(
        "all-Mach thermal RK4 component_errors={component_errors:?}, errors={errors:?}, orders={orders:?}"
    );
    assert!(
        orders.iter().all(|&order| order > 3.5),
        "all-Mach thermal local mass integration lost RK4 order: errors={errors:?}, orders={orders:?}"
    );
}

#[test]
fn compressible_uniform_freestream_is_preserved() {
    let mesh = generate_structured_rect_mesh(4, 4, 1.0, 1.0, BoundarySides::wall());
    let mut solver = CpuSolver::with_stepping(
        &mesh,
        compressible_model().expect("compressible"),
        Scheme::SecondOrderUpwind,
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
            "compressible freestream {name} error={error:e}"
        );
    }
}

#[test]
fn allmach_explicit_rhie_chow_scale_is_independent_of_integrator_dt() {
    fn recovered_d_p(dt: f64, alpha_u: f32) -> Vec<f64> {
        let mesh = generate_structured_rect_mesh(4, 4, 1.0, 1.0, BoundarySides::wall());
        let mut solver = CpuSolver::with_stepping(
            &mesh,
            allmach_pressure_model().expect("allmach"),
            Scheme::QUICKVanLeer,
            TimeScheme::RK4,
            SteppingMode::Explicit,
            CpuBackendConfig::default(),
        )
        .expect("allmach pressure RK4");
        let n = mesh.num_cells();
        solver.set_dt(dt as f32);
        solver.set_density(1.0);
        solver.set_alpha_u(alpha_u);
        solver.set_field_vec2("U", &vec![(0.0, 0.0); n]).unwrap();
        for (name, value) in [
            ("p", 0.0),
            ("rho", 1.0),
            ("psi", 0.1),
            ("psi_precond", 0.1),
            ("dt_local", 0.03),
            ("d_p", 0.0),
        ] {
            solver.set_field_scalar(name, &vec![value; n]).unwrap();
        }
        solver.initialize_history();
        solver.step();
        solver.get_field_scalar("d_p").unwrap()
    }

    let coarse = recovered_d_p(0.02, 0.1);
    let fine = recovered_d_p(0.005, 0.95);
    for (&a, &b) in coarse.iter().zip(&fine) {
        assert!((a - 0.03).abs() < 2.0e-6, "unexpected d_p={a:e}");
        assert!(
            (a - b).abs() < 1.0e-7,
            "explicit d_p changed with RK dt or implicit alpha_u: {a:e} vs {b:e}"
        );
    }
}

#[test]
fn structured_allmach_rhie_chow_damps_the_dense_grid_nyquist_mode() {
    const N: usize = 32;
    fn run(disable_rhie_chow: bool) -> f64 {
        let model = allmach_thermal_structured_model().expect("structured all-Mach");
        let mut solver = StructuredModelSolver::with_config(
            StructuredGrid::new(N, N, 1.0, 1.0),
            &model,
            0.001875,
            1,
            Scheme::SecondOrderUpwindVanLeer,
            TimeScheme::RK4,
        )
        .expect("structured all-Mach RK4");
        solver.set_named_field("U", |_x, _y| 0.0);
        let uy = solver.field_offset("U").unwrap() + 1;
        solver.set_state(uy, |_x, _y| 0.0);
        solver.set_named_field("p", |x, y| {
            let i = (x * N as f64).floor() as usize;
            let j = (y * N as f64).floor() as usize;
            if (i + j) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        });
        let psi = 1.0 / (347.0_f64 * 347.0);
        for (name, value) in [
            ("T", 1.0),
            ("rho", 1.225),
            ("rho_t_ref", 1.225),
            ("rho_dT", -1.225),
            ("u_dot_grad_p", 0.0),
            ("rho_floor", psi * 1.0e-5),
            ("t_ref", 1.0),
            ("psi_ref", psi),
            ("psi", psi),
            ("psi_precond", 0.4 * psi + 0.25),
            ("u_ref", 2.0),
            ("precond_mask", 1.0),
            ("dt_local", if disable_rhie_chow { 0.0 } else { 0.0078125 }),
            (
                "d_p",
                if disable_rhie_chow {
                    0.0
                } else {
                    0.0078125 / 1.225
                },
            ),
            ("mu", 0.0),
            ("ibm_penalty_U", 0.0),
        ] {
            if solver.field_offset(name).is_some() {
                solver.set_named_field(name, move |_x, _y| value);
            }
        }
        solver.set_boundaries(|_edge, _x, _y| {
            (
                3,
                vec![
                    BcComp {
                        kind: 1,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 1,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 0,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 0,
                        value: 0.0,
                    },
                ],
            )
        });
        for _ in 0..16 {
            solver.step();
        }
        let pressure = solver.state_field(solver.field_offset("p").unwrap());
        let (projection, samples) = pressure
            .iter()
            .enumerate()
            .filter_map(|(cell, &value)| {
                let i = cell % N;
                let j = cell / N;
                (i >= 3 && i + 3 < N && j >= 3 && j + 3 < N).then_some(if (i + j) & 1 == 0 {
                    value
                } else {
                    -value
                })
            })
            .fold((0.0, 0usize), |(sum, count), value| {
                (sum + value, count + 1)
            });
        projection.abs() / samples as f64
    }

    let damped = run(false);
    let control = run(true);
    assert!(damped < 0.05, "dense-grid Nyquist mode retained {damped:e}");
    assert!(
        control > 0.90,
        "disabled dense-grid Rhie-Chow control retained only {control:e}"
    );
}

#[test]
fn structured_compressible_uniform_freestream_is_preserved() {
    let model = compressible_structured_model().expect("structured compressible");
    let mut solver = StructuredModelSolver::with_config(
        StructuredGrid::new(4, 4, 1.0, 1.0),
        &model,
        1.0e-3,
        1,
        Scheme::QUICKMinMod,
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
            "structured freestream {name} error={error:e}"
        );
    }
}
