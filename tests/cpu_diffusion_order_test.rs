//! CPU-backend MMS convergence for the diffusion operator
//! (`generic_diffusion_demo_*_mms` models): constructs a `CpuSolver` directly
//! (no GPU adapter) and verifies the manufactured solution converges at design
//! order. This is the first *coupled-path* model on the CPU backend (the
//! recipe-phase-driven driver: prepare/assembly/solve/update, Neumann + per-face
//! Dirichlet BCs, the no-flux path) even though diffusion is `S == 1`.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::{CpuBackendConfig, CpuEngine, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{
    generic_diffusion_demo_mms_dirichlet_model, generic_diffusion_demo_mms_model,
    generic_diffusion_demo_mms_neumann_model, ModelSpec, MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;
use std::f64::consts::PI;

const STEADY_TOL: f64 = 4e-6;
const STEADY_MAX_STEPS: usize = 200;

fn inlet_outlet_wall_sides() -> BoundarySides {
    BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Outlet,
        bottom: BoundaryType::Wall,
        top: BoundaryType::Wall,
    }
}

fn l2_error(mesh: &Mesh, phi: &[f64], exact: impl Fn(f64, f64) -> f64) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..mesh.num_cells() {
        let e = phi[i] - exact(mesh.cell_cx[i], mesh.cell_cy[i]);
        num += e * e * mesh.cell_vol[i];
        den += mesh.cell_vol[i];
    }
    (num / den).sqrt()
}

fn fit_order(hs: &[f64], errs: &[f64]) -> f64 {
    let n = hs.len() as f64;
    let lx: Vec<f64> = hs.iter().map(|h| h.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
    let sx: f64 = lx.iter().sum();
    let sy: f64 = ly.iter().sum();
    let sxx: f64 = lx.iter().map(|x| x * x).sum();
    let sxy: f64 = lx.iter().zip(&ly).map(|(x, y)| x * y).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

fn solve_steady(
    model_fn: fn() -> Result<ModelSpec, String>,
    n: usize,
    config: CpuBackendConfig,
    source: &dyn Fn(f64, f64) -> f64,
    setup_bcs: &dyn Fn(&mut CpuSolver, &Mesh),
) -> (Mesh, Vec<f64>) {
    let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, inlet_outlet_wall_sides());
    let model = model_fn().expect("model");
    let mut solver = CpuSolver::new(&mesh, model, Scheme::Upwind, TimeScheme::Euler, config)
        .expect("cpu solver");
    solver.set_outer_iters(2);
    solver.set_dt(1.0);
    setup_bcs(&mut solver, &mesh);
    let src: Vec<f64> = (0..mesh.num_cells())
        .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver.set_field_scalar(MMS_SOURCE_FIELD, &src).expect("source");
    solver
        .set_field_scalar("phi", &vec![0.0; mesh.num_cells()])
        .expect("init phi");
    solver.initialize_history();

    let mut prev = solver.get_field_scalar("phi").unwrap();
    for _ in 0..STEADY_MAX_STEPS {
        solver.step();
        let cur = solver.get_field_scalar("phi").unwrap();
        let d = cur
            .iter()
            .zip(&prev)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        prev = cur;
        if d < STEADY_TOL {
            break;
        }
    }
    (mesh, prev)
}

/// phi* = sin(pi x) cos(pi y): Dirichlet 0 at left/right, natural zero-gradient
/// at bottom/top. Second-order interior + boundary discretization expected.
#[test]
fn cpu_diffusion_dirichlet_zerograd_second_order() {
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).cos();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).cos();
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [8usize, 16, 32] {
        let (mesh, phi) = solve_steady(
            generic_diffusion_demo_mms_model,
            n,
            CpuBackendConfig::default(),
            &source,
            &|_s, _m| {},
        );
        let e = l2_error(&mesh, &phi, exact);
        println!("[cpu-diff][dirichlet] n={n} l2={e:.3e}");
        hs.push(1.0 / n as f64);
        errs.push(e);
    }
    let order = fit_order(&hs, &errs);
    println!("[cpu-diff][dirichlet] order={order:.3}");
    assert!((1.7..=2.3).contains(&order), "order {order:.3} (expected ~2)");
    assert!(*errs.last().unwrap() < 1e-3, "finest error {:.3e}", errs.last().unwrap());
}

/// phi* = cos(pi x) cos(pi y): per-face Dirichlet on all four boundaries.
#[test]
fn cpu_diffusion_perface_dirichlet_second_order() {
    let exact = |x: f64, y: f64| (PI * x).cos() * (PI * y).cos();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).cos() * (PI * y).cos();
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [8usize, 16, 32] {
        let (mesh, phi) = solve_steady(
            generic_diffusion_demo_mms_dirichlet_model,
            n,
            CpuBackendConfig::default(),
            &source,
            &|solver, mesh| {
                let fx = mesh.face_cx.clone();
                let fy = mesh.face_cy.clone();
                let fv = move |f: u32| {
                    ((PI * fx[f as usize]).cos() * (PI * fy[f as usize]).cos()) as f32
                };
                for b in [
                    GpuBoundaryType::Inlet,
                    GpuBoundaryType::Outlet,
                    GpuBoundaryType::Wall,
                ] {
                    solver
                        .set_boundary_values_per_face(b, "phi", 0, &fv)
                        .expect("dirichlet");
                }
            },
        );
        let e = l2_error(&mesh, &phi, exact);
        println!("[cpu-diff][perface] n={n} l2={e:.3e}");
        hs.push(1.0 / n as f64);
        errs.push(e);
    }
    let order = fit_order(&hs, &errs);
    println!("[cpu-diff][perface] order={order:.3}");
    assert!((1.7..=2.3).contains(&order), "order {order:.3} (expected ~2)");
}

/// phi* = sin(pi x) sin(pi y): Dirichlet 0 at left/right, spatially-varying
/// non-zero Neumann (dphi/dn = -pi sin(pi x)) at the bottom/top walls — the
/// CPU-backend exercise of the Neumann boundary path.
#[test]
fn cpu_diffusion_neumann_mix_second_order() {
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [8usize, 16, 32] {
        let (mesh, phi) = solve_steady(
            generic_diffusion_demo_mms_neumann_model,
            n,
            CpuBackendConfig::default(),
            &source,
            &|solver, mesh| {
                let fx = mesh.face_cx.clone();
                let neumann = move |f: u32| (-PI * (PI * fx[f as usize]).sin()) as f32;
                solver
                    .set_boundary_values_per_face(GpuBoundaryType::Wall, "phi", 0, &neumann)
                    .expect("neumann");
            },
        );
        let e = l2_error(&mesh, &phi, exact);
        println!("[cpu-diff][neumann] n={n} l2={e:.3e}");
        hs.push(1.0 / n as f64);
        errs.push(e);
    }
    let order = fit_order(&hs, &errs);
    println!("[cpu-diff][neumann] order={order:.3}");
    assert!((1.7..=2.3).contains(&order), "order {order:.3} (expected ~2)");
}

/// Both CPU engines (interpreter + transpiled-with-fallback) agree on diffusion.
#[test]
fn cpu_diffusion_engines_agree() {
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).cos();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).cos();
    let n = 24;
    let base = solve_steady(
        generic_diffusion_demo_mms_model,
        n,
        CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false, precision: Default::default(), },
        &source,
        &|_s, _m| {},
    )
    .1;
    for (label, cfg) in [
        ("interp/4t/simd", CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 4, simd: true, precision: Default::default(), }),
        ("transpiled/4t", CpuBackendConfig { engine: CpuEngine::Transpiled, threads: 4, simd: false, precision: Default::default(), }),
    ] {
        let t = solve_steady(generic_diffusion_demo_mms_model, n, cfg, &source, &|_s, _m| {}).1;
        let d = base.iter().zip(&t).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("[cpu-diff][engines] {label} max|diff|={d:.3e}");
        assert!(d < 1e-4, "{label} diverges: {d:.3e}");
    }
    let _ = exact;
}
