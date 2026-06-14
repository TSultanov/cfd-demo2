//! Exercise the CPU backend through the *real* `UnifiedSolver` API (the same API
//! the existing MMS tests and the GUI use), selected via the `CFD2_BACKEND`
//! environment variables. Validates that the scalar-transport manufactured
//! solution converges on the CPU backend for both engines (interpreter and
//! transpiled) — i.e. the existing harness works on the CPU backend, GPU-free.
#![cfg(feature = "cpu")]

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

fn l2(mesh: &Mesh, t: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..mesh.num_cells() {
        let e = t[i] - exact(mesh.cell_cx[i], mesh.cell_cy[i]);
        num += e * e * mesh.cell_vol[i];
        den += mesh.cell_vol[i];
    }
    (num / den).sqrt()
}

fn solve_steady_unified(n: usize) -> (Mesh, Vec<f64>) {
    let mesh = unit_square(n);
    let model = scalar_transport_model().expect("model");
    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        ..SolverConfig::default()
    };
    // No device/queue: the CPU backend is GPU-free.
    let mut solver = pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))
        .expect("create solver");
    assert!(solver.is_cpu(), "expected the CPU backend (CFD2_BACKEND=cpu)");
    solver.set_outer_iters(2).expect("outer_iters");
    solver.set_dt(0.2);

    let fv = |f: u32| exact(mesh.face_cx[f as usize], mesh.face_cy[f as usize]) as f32;
    for b in [GpuBoundaryType::Inlet, GpuBoundaryType::Outlet, GpuBoundaryType::Wall] {
        solver
            .set_boundary_values_per_face(b, SCALAR_TRANSPORT_FIELD, 0, &fv)
            .expect("bc");
    }
    solver
        .set_field_vec2(ADVECTING_VELOCITY_FIELD, &vec![(4.0, 2.0); mesh.num_cells()])
        .expect("U");
    let src: Vec<f64> = (0..mesh.num_cells())
        .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_scalar(SCALAR_TRANSPORT_MMS_SOURCE_FIELD, &src)
        .expect("src");
    solver
        .set_field_scalar(SCALAR_TRANSPORT_FIELD, &vec![0.0; mesh.num_cells()])
        .expect("init");
    solver.initialize_history();

    let mut prev = pollster::block_on(solver.get_field_scalar(SCALAR_TRANSPORT_FIELD)).unwrap();
    for _ in 0..400 {
        solver.step();
        let cur = pollster::block_on(solver.get_field_scalar(SCALAR_TRANSPORT_FIELD)).unwrap();
        let d = cur.iter().zip(&prev).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        prev = cur;
        if d < 4e-6 {
            break;
        }
    }
    (mesh, prev)
}

#[test]
fn cpu_unified_scalar_transport_converges() {
    // Validate the CPU backend through UnifiedSolver for both engines.
    for engine in ["interpreter", "transpiled"] {
        std::env::set_var("CFD2_BACKEND", "cpu");
        std::env::set_var("CFD2_CPU_ENGINE", engine);

        let levels = [16usize, 32, 64];
        let mut hs = Vec::new();
        let mut errs = Vec::new();
        for &n in &levels {
            let (mesh, t) = solve_steady_unified(n);
            let e = l2(&mesh, &t);
            println!("[cpu-unified][{engine}] n={n} l2={e:.3e}");
            hs.push(1.0 / n as f64);
            errs.push(e);
        }
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "{engine}: errors not decreasing: {errs:?}");
        assert!(*errs.last().unwrap() < 1e-2, "{engine}: finest error too large: {errs:?}");
        let order = fit_order(&hs, &errs);
        println!("[cpu-unified][{engine}] observed order = {order:.3}");
        assert!(
            (0.6..=1.6).contains(&order),
            "{engine}: implausible order {order:.3}"
        );
    }
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
}
