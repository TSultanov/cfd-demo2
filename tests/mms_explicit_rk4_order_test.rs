//! Manufactured-solution gates for the matrix-free explicit RK4 path.
//!
//! These run directly on the CPU kernel interpreter, which is the executable
//! oracle for the typed codegen programs and does not require a GPU adapter.
#![cfg(all(feature = "dev-tests", feature = "cpu"))]

use cfd2::solver::cpu::structured::{BcComp, StructuredGrid, StructuredModelSolver};
use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, BoundarySides, BoundaryType,
};
use cfd2::solver::model::{
    allmach_pressure_model, allmach_thermal_model, allmach_thermal_structured_model,
    compressible_mms_biharmonic_model, compressible_model, compressible_structured_model,
    generic_diffusion_demo_mms_model,
    generic_diffusion_demo_structured_ibm_model, incompressible_momentum_ale_model,
    incompressible_momentum_model, scalar_transport_model, ADVECTING_VELOCITY_FIELD,
    MMS_SOURCE_FIELD, SCALAR_TRANSPORT_FIELD, SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SteppingMode, TimeScheme};

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
            (3, vec![BcComp { kind: 0, value: 0.0 }])
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
        let face_value = |face: u32| {
            exact(mesh.face_cx[face as usize], mesh.face_cy[face as usize]) as f32
        };
        for boundary in [
            GpuBoundaryType::Inlet,
            GpuBoundaryType::Outlet,
            GpuBoundaryType::Wall,
        ] {
            solver
                .set_boundary_values_per_face(
                    boundary,
                    SCALAR_TRANSPORT_FIELD,
                    0,
                    &face_value,
                )
                .expect("boundary values");
        }
        solver.initialize_history();
        solver.step();
        let field = solver
            .get_field_scalar(SCALAR_TRANSPORT_FIELD)
            .expect("read scalar");
        assert!(
            field.iter().all(|v| v.is_finite() && (0.5..1.5).contains(v)),
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
    assert!(ale.validate_explicit_rk4().is_err(), "ALE is intentionally deferred");
    let biharmonic = compressible_mms_biharmonic_model().expect("biharmonic");
    assert!(
        biharmonic.validate_explicit_rk4().is_err(),
        "auxiliary elliptic constraint rows need a dedicated explicit closure"
    );
    // Pressure-based (Rhie–Chow) all-Mach models carry a `d_p` coupling
    // coefficient produced only by the implicit Update phase, which the
    // matrix-free RK stages never run; without an explicit d_p closure the
    // elliptic pressure–velocity coupling would silently vanish, so they are
    // deliberately excluded from the fully explicit path.
    for model in [
        allmach_pressure_model().expect("allmach"),
        allmach_thermal_model().expect("allmach thermal"),
        allmach_thermal_structured_model().expect("structured allmach thermal"),
    ] {
        assert!(
            model.validate_explicit_rk4().is_err(),
            "{} uses Rhie–Chow d_p coupling and must not be presented as a matrix-free explicit ODE",
            model.id
        );
    }
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
    solver.set_field_scalar("rho", &vec![1.0; mesh.num_cells()]).unwrap();
    solver
        .set_field_vec2("rho_u", &vec![(0.0, 0.0); mesh.num_cells()])
        .unwrap();
    solver
        .set_field_scalar("rho_e", &vec![2.5; mesh.num_cells()])
        .unwrap();
    solver.set_field_scalar("p", &vec![1.0; mesh.num_cells()]).unwrap();
    solver.set_field_scalar("T", &vec![1.0; mesh.num_cells()]).unwrap();
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
        assert!(error < 2.0e-6, "compressible freestream {name} error={error:e}");
    }
}

#[test]
fn allmach_pressure_based_models_are_rejected_from_explicit_rk4() {
    // Pressure-based all-Mach models depend on the Rhie–Chow `d_p` coupling that
    // only the implicit Update phase produces; the matrix-free RK path cannot
    // build an RK4 solver for them and must fail loudly at construction rather
    // than silently drop the pressure–velocity coupling.
    let mesh = generate_structured_rect_mesh(4, 4, 1.0, 1.0, BoundarySides::wall());
    let err = CpuSolver::with_stepping(
        &mesh,
        allmach_pressure_model().expect("allmach"),
        Scheme::QUICKVanLeer,
        TimeScheme::RK4,
        SteppingMode::Explicit,
        CpuBackendConfig::default(),
    )
    .err()
    .expect("allmach_pressure must be rejected from the explicit RK4 path");
    assert!(
        err.contains("d_p") || err.to_lowercase().contains("rhie"),
        "unexpected rejection reason for allmach_pressure under RK4: {err}"
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
        (3, vec![BcComp { kind: 0, value: 0.0 }; unknowns])
    });

    solver.step();

    for (name, exact) in [("rho", 1.0), ("rho_e", 2.5), ("p", 1.0), ("T", 1.0)] {
        let values = solver.state_field(solver.field_offset(name).unwrap());
        let error = values
            .into_iter()
            .map(|v| (v - exact).abs())
            .fold(0.0_f64, f64::max);
        assert!(error < 2.0e-6, "structured freestream {name} error={error:e}");
    }
}
