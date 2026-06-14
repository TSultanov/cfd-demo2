//! CPU-backend compressible smoke: a uniform stagnant gas (rho=1, u=0, p=1) with
//! zero manufactured source and matching Dirichlet inlets is an exact steady
//! state. Running the full coupled compressible path on the CPU (KT flux,
//! multi-target gradients, block-CSR assembly, FGMRES+block-Jacobi solve,
//! primitive recovery) must keep it uniform and finite. This surfaces any
//! interpreter intrinsic gaps before the order study.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::compressible_mms_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;

#[test]
fn cpu_compressible_uniform_state_is_steady() {
    let n = 8;
    let mesh = generate_structured_rect_mesh(
        n,
        n,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Inlet,
            bottom: BoundaryType::Inlet,
            top: BoundaryType::Inlet,
        },
    );
    let model = compressible_mms_model().expect("model");
    let cells = mesh.num_cells();

    let gamma = 1.4f64;
    let (rho0, p0) = (1.0f64, 1.0f64);
    let rho_e0 = p0 / (gamma - 1.0); // u = 0 -> no kinetic energy
    let t0 = p0 / rho0; // R = 1

    let mut s = CpuSolver::with_stepping(
        &mesh,
        model,
        Scheme::SecondOrderUpwindVanLeer,
        TimeScheme::BDF2,
        SteppingMode::Implicit { outer_iters: 1 },
        CpuBackendConfig::default(),
    )
    .expect("cpu solver");
    s.set_dt(0.01);
    s.set_dtau(0.0);
    s.set_viscosity(0.05);
    s.set_density(rho0 as f32);
    s.set_outer_iters(1);

    // Uniform initial state (conserved + primitive).
    s.set_field_scalar("rho", &vec![rho0; cells]).unwrap();
    s.set_field_vec2("rho_u", &vec![(0.0, 0.0); cells]).unwrap();
    s.set_field_scalar("rho_e", &vec![rho_e0; cells]).unwrap();
    s.set_field_scalar("p", &vec![p0; cells]).unwrap();
    s.set_field_scalar("T", &vec![t0; cells]).unwrap();
    s.set_field_vec2("u", &vec![(0.0, 0.0); cells]).unwrap();
    // Zero manufactured sources.
    s.set_field_scalar(
        cfd2::solver::model::COMPRESSIBLE_MMS_SOURCE_RHO_FIELD,
        &vec![0.0; cells],
    )
    .unwrap();
    s.set_field_vec2(
        cfd2::solver::model::COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD,
        &vec![(0.0, 0.0); cells],
    )
    .unwrap();
    s.set_field_scalar(
        cfd2::solver::model::COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
        &vec![0.0; cells],
    )
    .unwrap();

    // Matching uniform Dirichlet inlets.
    for (field, val) in [("rho", rho0), ("p", p0), ("T", t0), ("rho_e", rho_e0)] {
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, field, 0, &|_| val as f32)
            .unwrap();
    }
    for c in 0..2u32 {
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "u", c, &|_| 0.0)
            .unwrap();
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_u", c, &|_| 0.0)
            .unwrap();
    }

    s.initialize_history();

    for step in 0..5 {
        s.step();
        let rho = s.get_field_scalar("rho").unwrap();
        let p = s.get_field_scalar("p").unwrap();
        let u = s.get_field_vec2("u").unwrap();
        let max_drho = rho.iter().map(|r| (r - rho0).abs()).fold(0.0, f64::max);
        let max_dp = p.iter().map(|q| (q - p0).abs()).fold(0.0, f64::max);
        let max_u = u.iter().map(|(a, b)| a.abs().max(b.abs())).fold(0.0, f64::max);
        println!("[cpu-compr-smoke] step {step}: max|drho|={max_drho:.3e} max|dp|={max_dp:.3e} max|u|={max_u:.3e}");
        assert!(rho.iter().all(|r| r.is_finite()), "rho went non-finite");
        assert!(
            max_drho < 1e-3 && max_dp < 1e-3 && max_u < 1e-3,
            "uniform state drifted: drho={max_drho:.3e} dp={max_dp:.3e} u={max_u:.3e}"
        );
    }
}
