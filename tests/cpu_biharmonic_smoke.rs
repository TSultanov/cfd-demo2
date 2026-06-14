//! CPU-backend biharmonic-compressible smoke: feasibility check that the CPU runs
//! the biharmonic-dissipation compressible model (the GPU's shipped cure for the
//! interior marginal instability — `lap_<conserved>` unknowns + multi-diffusion
//! assembly + the `+eps4*(lap_neigh-lap_own)` flux term). A uniform stagnant gas
//! is an exact steady state for which the ∇⁴ term is inert (lap==0), so this only
//! exercises that the biharmonic kernels LOWER and RUN on the CPU (block-CSR at
//! the larger stride, primitive recovery, the static-diagonal lap rows). If this
//! passes, the biharmonic path is viable on the CPU and the convergent compressible
//! variant is reachable; if it panics, the biharmonic kernels need CPU support.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::compressible_mms_biharmonic_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;

#[test]
fn cpu_biharmonic_uniform_state_runs() {
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
    let model = compressible_mms_biharmonic_model().expect("biharmonic model");
    let cells = mesh.num_cells();

    let gamma = 1.4f64;
    let (rho0, p0) = (1.0f64, 1.0f64);
    let rho_e0 = p0 / (gamma - 1.0);
    let t0 = p0 / rho0;

    let mut s = CpuSolver::with_stepping(
        &mesh,
        model,
        Scheme::SecondOrderUpwindVanLeer,
        TimeScheme::BDF2,
        SteppingMode::Implicit { outer_iters: 1 },
        CpuBackendConfig::default(),
    )
    .expect("cpu biharmonic solver");
    s.set_dt(0.01);
    s.set_dtau(0.0);
    s.set_viscosity(0.05);
    s.set_density(rho0 as f32);
    s.set_outer_iters(1);

    s.set_field_scalar("rho", &vec![rho0; cells]).unwrap();
    s.set_field_vec2("rho_u", &vec![(0.0, 0.0); cells]).unwrap();
    s.set_field_scalar("rho_e", &vec![rho_e0; cells]).unwrap();
    s.set_field_scalar("p", &vec![p0; cells]).unwrap();
    s.set_field_scalar("T", &vec![t0; cells]).unwrap();
    s.set_field_vec2("u", &vec![(0.0, 0.0); cells]).unwrap();
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
        println!("[cpu-bihar-smoke] step {step}: max|drho|={max_drho:.3e} max|dp|={max_dp:.3e} max|u|={max_u:.3e}");
        assert!(rho.iter().all(|r| r.is_finite()), "rho went non-finite");
        assert!(
            max_drho < 1e-3 && max_dp < 1e-3 && max_u < 1e-3,
            "uniform biharmonic state drifted: drho={max_drho:.3e} dp={max_dp:.3e} u={max_u:.3e}"
        );
    }
}
