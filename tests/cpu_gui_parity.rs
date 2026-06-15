//! GUI ↔ CPU feature parity: exercise the *exact* path the GUI uses when the user
//! ticks "CPU backend" — `UnifiedSolver` selected via `CFD2_BACKEND=cpu`, with the
//! per-model `SolverConfig.stepping` the GUI sets (Coupled for the saddle-point
//! models, Implicit for compressible). This guards two parity fixes:
//!   1. The CPU path must honor `config.stepping` (it used to hardcode Coupled via
//!      `CpuSolver::new`, mis-stepping compressible in the GUI).
//!   2. EOS runtime tuning (`eos.*`) must route to the CPU backend.
//! The four non-compressible families + Ghia already certify the GPU-equivalent
//! solution path; this focuses on the GUI's UnifiedSolver wiring for ALL models.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    compressible_mms_model, incompressible_momentum_mms_model,
    COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, COMPRESSIBLE_MMS_SOURCE_RHO_FIELD,
    COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, INCOMPRESSIBLE_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, TimeScheme, UnifiedSolver};
use std::f64::consts::PI;

fn tg_u(x: f64, y: f64) -> (f64, f64) {
    ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
}
fn tg_source(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = tg_u(x, y);
    (2.0 * PI * PI * ux, 2.0 * PI * PI * uy) // mu = 1
}

/// The GUI's coupled path: incompressible momentum (saddle-point, S=3) via the
/// `UnifiedSolver` CPU backend with `stepping = Coupled`. Converges toward the
/// Taylor-Green MMS, proving the GUI runs coupled models on the CPU (not just
/// scalar transport).
fn run_incompressible_coupled() {
    let n = 16;
    let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
    let model = incompressible_momentum_mms_model().expect("model");
    let config = SolverConfig {
        advection_scheme: Scheme::SecondOrderUpwind,
        time_scheme: TimeScheme::BDF2,
        stepping: SteppingMode::Coupled,
        ..SolverConfig::default()
    };
    let mut solver =
        pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None)).expect("solver");
    assert!(solver.is_cpu(), "expected the CPU backend (CFD2_BACKEND=cpu)");
    solver.set_dt(0.05);
    solver.set_density(1.0);
    solver.set_viscosity(1.0);
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);
    solver.set_outer_iters(25).expect("outer_iters");

    let (fx, fy) = (mesh.face_cx.clone(), mesh.face_cy.clone());
    for c in 0..2u32 {
        let (fx, fy) = (fx.clone(), fy.clone());
        let wall = move |i: u32| {
            let (ux, uy) = tg_u(fx[i as usize], fy[i as usize]);
            (if c == 0 { ux } else { uy }) as f32
        };
        solver
            .set_boundary_values_per_face(GpuBoundaryType::Wall, "U", c, &wall)
            .expect("wall bc");
    }
    let src: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| tg_source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver.set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src).expect("src");
    solver.set_field_vec2("U", &vec![(0.0, 0.0); mesh.num_cells()]).expect("U0");
    solver.set_field_scalar("p", &vec![0.0; mesh.num_cells()]).expect("p0");
    solver.initialize_history();
    for _ in 0..40 {
        solver.step();
    }
    let u = pollster::block_on(solver.get_field_vec2("U")).expect("read U");
    let finite = u.iter().all(|(a, b)| a.is_finite() && b.is_finite());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..mesh.num_cells() {
        let (ex, ey) = tg_u(mesh.cell_cx[i], mesh.cell_cy[i]);
        num += ((u[i].0 - ex).powi(2) + (u[i].1 - ey).powi(2)) * mesh.cell_vol[i];
        den += mesh.cell_vol[i];
    }
    let l2 = (num / den).sqrt();
    println!("[gui-parity][incompressible/Coupled] u_l2={l2:.4e} finite={finite}");
    assert!(finite, "incompressible U went non-finite via the GUI CPU path");
    assert!(l2 < 5e-2, "incompressible coupled CPU solve not converging: {l2:.4e}");
}

/// The GUI's compressible path: `UnifiedSolver` CPU backend with
/// `stepping = Implicit { outer_iters: 1 }` (what the GUI sets for compressible).
/// A uniform stagnant gas with matching inlets + zero sources is an exact steady
/// state; it must stay uniform and finite — proving the CPU path honors the
/// requested Implicit stepping (it used to hardcode Coupled).
fn run_compressible_implicit() {
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
    let (rho0, p0) = (1.0f64, 1.0f64);
    let rho_e0 = p0 / 0.4; // gamma-1 = 0.4, u = 0
    let t0 = p0 / rho0; // R = 1

    let config = SolverConfig {
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        stepping: SteppingMode::Implicit { outer_iters: 1 },
        ..SolverConfig::default()
    };
    let mut solver =
        pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None)).expect("solver");
    assert!(solver.is_cpu(), "expected the CPU backend");
    solver.set_dt(0.01);
    solver.set_viscosity(0.05);
    solver.set_density(rho0 as f32);
    solver.set_outer_iters(1).expect("outer_iters");
    // Pseudo-transient: enables the GUI steady-state auto-pause (should_stop).
    solver.set_dtau(0.01).expect("dtau");
    // EOS runtime tuning routes to the CPU backend (eos.* via set_named_param).
    let eos = solver.model().eos();
    solver
        .set_eos(&eos)
        .expect("set_eos must succeed on the CPU backend");

    solver.set_field_scalar("rho", &vec![rho0; cells]).unwrap();
    solver.set_field_vec2("rho_u", &vec![(0.0, 0.0); cells]).unwrap();
    solver.set_field_scalar("rho_e", &vec![rho_e0; cells]).unwrap();
    solver.set_field_scalar("p", &vec![p0; cells]).unwrap();
    solver.set_field_scalar("T", &vec![t0; cells]).unwrap();
    solver.set_field_vec2("u", &vec![(0.0, 0.0); cells]).unwrap();
    solver.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, &vec![0.0; cells]).unwrap();
    solver.set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &vec![(0.0, 0.0); cells]).unwrap();
    solver.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &vec![0.0; cells]).unwrap();
    for (field, val) in [("rho", rho0), ("p", p0), ("T", t0), ("rho_e", rho_e0)] {
        solver
            .set_boundary_values_per_face(GpuBoundaryType::Inlet, field, 0, &|_| val as f32)
            .unwrap();
    }
    for c in 0..2u32 {
        solver.set_boundary_values_per_face(GpuBoundaryType::Inlet, "u", c, &|_| 0.0).unwrap();
        solver.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_u", c, &|_| 0.0).unwrap();
    }
    solver.initialize_history();
    for _ in 0..5 {
        solver.step();
    }
    let rho = pollster::block_on(solver.get_field_scalar("rho")).unwrap();
    let max_drho = rho.iter().map(|r| (r - rho0).abs()).fold(0.0, f64::max);
    println!("[gui-parity][compressible/Implicit] max|drho|={max_drho:.3e}");
    assert!(rho.iter().all(|r| r.is_finite()), "compressible rho went non-finite");
    assert!(max_drho < 1e-3, "uniform compressible state drifted via the GUI CPU path: {max_drho:.3e}");
    // GUI steady-state auto-pause parity: at a steady state under dtau>0 the CPU
    // must report should_stop=true (so the GUI auto-pauses like the GPU).
    assert_eq!(
        solver.step_stats().should_stop,
        Some(true),
        "CPU should_stop must fire at steady state under dtau>0 (GUI auto-pause parity)"
    );
}

/// Single env-using test (CFD2_BACKEND is process-global): runs both the coupled
/// and the compressible GUI paths on the CPU backend.
#[test]
fn cpu_gui_unified_backend_runs_coupled_and_compressible() {
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", "interpreter");
    run_incompressible_coupled();
    run_compressible_implicit();
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
}

/// EOS runtime tuning routes to the CPU constants the assembly reads (mirrors the
/// GPU `set_eos`). Recognized `eos.*` fields apply; anything else is ignored.
#[test]
fn cpu_set_eos_param_routes() {
    let mesh = generate_structured_rect_mesh(
        4,
        4,
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
    let mut s = CpuSolver::new(
        &mesh,
        model,
        Scheme::SecondOrderUpwindVanLeer,
        TimeScheme::BDF2,
        CpuBackendConfig::default(),
    )
    .expect("cpu solver");
    for f in ["eos.gamma", "eos.gm1", "eos.r", "eos.dp_drho", "eos.p_offset", "eos.theta_ref"] {
        assert!(s.set_eos_param(f, 1.23), "expected {f} to be a recognized EOS field");
    }
    assert!(!s.set_eos_param("eos.bogus", 0.0), "unknown eos field must not be applied");
    assert!(!s.set_eos_param("viscosity", 0.0), "non-eos param must not be routed here");
    let _ = &mesh;
}
