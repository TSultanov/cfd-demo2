//! Cross-backend validation: the CPU backend must reproduce the GPU backend's
//! steady scalar-transport solution (they discretize the *same* system; the only
//! differences are the linear solver and f32-vs-f64 arithmetic). Requires both
//! the `cpu` backend and the `dev-tests` (GPU MMS harness) features.
#![cfg(all(feature = "cpu", feature = "dev-tests"))]

use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    scalar_transport_model, ADVECTING_VELOCITY_FIELD, SCALAR_TRANSPORT_FIELD,
    SCALAR_TRANSPORT_KAPPA, SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, TimeScheme, UnifiedSolver};
use std::f64::consts::PI;

const STEADY_TOL: f64 = 4e-6;
const STEADY_MAX_STEPS: usize = 400;

fn unit_square(n: usize) -> Mesh {
    generate_structured_rect_mesh(
        n,
        n,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

fn exact(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).cos()
}
fn source(x: f64, y: f64) -> f64 {
    let (ux, uy) = (4.0, 2.0);
    let k = SCALAR_TRANSPORT_KAPPA;
    let dtdx = PI * (PI * x).cos() * (PI * y).cos();
    let dtdy = -PI * (PI * x).sin() * (PI * y).sin();
    let lap = -2.0 * PI * PI * exact(x, y);
    ux * dtdx + uy * dtdy - k * lap
}

fn run_cpu(mesh: &Mesh, scheme: Scheme) -> Vec<f64> {
    let model = scalar_transport_model().expect("model");
    let mut s = CpuSolver::new(mesh, model, scheme, TimeScheme::Euler, CpuBackendConfig::default())
        .expect("cpu solver");
    s.set_outer_iters(2);
    s.set_dt(0.2);
    let fv = |f: u32| exact(mesh.face_cx[f as usize], mesh.face_cy[f as usize]) as f32;
    for b in [GpuBoundaryType::Inlet, GpuBoundaryType::Outlet, GpuBoundaryType::Wall] {
        s.set_boundary_values_per_face(b, SCALAR_TRANSPORT_FIELD, 0, &fv).unwrap();
    }
    s.set_field_vec2(ADVECTING_VELOCITY_FIELD, &vec![(4.0, 2.0); mesh.num_cells()]).unwrap();
    let src: Vec<f64> = (0..mesh.num_cells()).map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    s.set_field_scalar(SCALAR_TRANSPORT_MMS_SOURCE_FIELD, &src).unwrap();
    s.set_field_scalar(SCALAR_TRANSPORT_FIELD, &vec![0.0; mesh.num_cells()]).unwrap();
    s.initialize_history();

    let mut prev = s.get_field_scalar(SCALAR_TRANSPORT_FIELD).unwrap();
    for _ in 0..STEADY_MAX_STEPS {
        s.step();
        let cur = s.get_field_scalar(SCALAR_TRANSPORT_FIELD).unwrap();
        let d = cur.iter().zip(&prev).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        prev = cur;
        if d < STEADY_TOL {
            break;
        }
    }
    prev
}

fn run_gpu(mesh: &Mesh, scheme: Scheme) -> Vec<f64> {
    let model = scalar_transport_model().expect("model");
    let config = SolverConfig {
        advection_scheme: scheme,
        ..SolverConfig::default()
    };
    let mut s = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None))
        .expect("gpu solver");
    s.set_outer_iters(2).unwrap();
    s.set_dt(0.2);
    let fv = |f: u32| exact(mesh.face_cx[f as usize], mesh.face_cy[f as usize]) as f32;
    for b in [GpuBoundaryType::Inlet, GpuBoundaryType::Outlet, GpuBoundaryType::Wall] {
        s.set_boundary_values_per_face(b, SCALAR_TRANSPORT_FIELD, 0, &fv).unwrap();
    }
    s.set_field_vec2(ADVECTING_VELOCITY_FIELD, &vec![(4.0, 2.0); mesh.num_cells()]).unwrap();
    let src: Vec<f64> = (0..mesh.num_cells()).map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    s.set_field_scalar(SCALAR_TRANSPORT_MMS_SOURCE_FIELD, &src).unwrap();
    s.set_field_scalar(SCALAR_TRANSPORT_FIELD, &vec![0.0; mesh.num_cells()]).unwrap();
    s.initialize_history();

    let mut prev = pollster::block_on(s.get_field_scalar(SCALAR_TRANSPORT_FIELD)).unwrap();
    for _ in 0..STEADY_MAX_STEPS {
        s.step();
        let cur = pollster::block_on(s.get_field_scalar(SCALAR_TRANSPORT_FIELD)).unwrap();
        let d = cur.iter().zip(&prev).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        prev = cur;
        if d < STEADY_TOL {
            break;
        }
    }
    prev
}

#[test]
fn cpu_matches_gpu_scalar_transport_upwind() {
    for &n in &[16usize, 32] {
        let mesh = unit_square(n);
        let cpu = run_cpu(&mesh, Scheme::Upwind);
        let gpu = run_gpu(&mesh, Scheme::Upwind);
        assert_eq!(cpu.len(), gpu.len());
        let max_diff = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        // L2 vs the analytic solution for context.
        let l2 = |t: &[f64]| -> f64 {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..mesh.num_cells() {
                let e = t[i] - exact(mesh.cell_cx[i], mesh.cell_cy[i]);
                num += e * e * mesh.cell_vol[i];
                den += mesh.cell_vol[i];
            }
            (num / den).sqrt()
        };
        println!(
            "[cpu-vs-gpu][upwind] n={n} max|cpu-gpu|={max_diff:.3e} l2_cpu={:.4e} l2_gpu={:.4e}",
            l2(&cpu),
            l2(&gpu)
        );
        // Same discrete system, different linear solver + f32/f64: agree tightly.
        assert!(
            max_diff < 2e-3,
            "CPU and GPU steady fields diverge at n={n}: max|diff|={max_diff:.3e}"
        );
    }
}
