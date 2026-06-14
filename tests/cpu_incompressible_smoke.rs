//! CPU-backend incompressible (Taylor-Green MMS) smoke + order check: the
//! coupled saddle-point path (derived Rhie-Chow flux, S=3 = U,p, model-owned
//! Schur preconditioner). Verifies the CPU builds the model, runs the coupled
//! corrector loop, and converges toward the manufactured solution.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::{CpuBackendConfig, CpuEngine, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, Mesh};
use cfd2::solver::model::{incompressible_momentum_mms_model, INCOMPRESSIBLE_MMS_SOURCE_FIELD};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;
use std::f64::consts::PI;

const MU: f64 = 1.0;
const RHO: f64 = 1.0;

fn exact_u(x: f64, y: f64) -> (f64, f64) {
    ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
}
fn source(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    (2.0 * MU * PI * PI * ux, 2.0 * MU * PI * PI * uy)
}

fn solve(n: usize, steps: usize, cfg: CpuBackendConfig) -> (Mesh, Vec<(f64, f64)>) {
    let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
    let model = incompressible_momentum_mms_model().expect("model");
    let mut s = CpuSolver::with_stepping(
        &mesh,
        model,
        Scheme::SecondOrderUpwind,
        TimeScheme::BDF2,
        SteppingMode::Coupled,
        cfg,
    )
    .expect("cpu solver");
    s.set_dt(0.05);
    s.set_dtau(0.0);
    s.set_density(RHO as f32);
    s.set_viscosity(MU as f32);
    s.set_alpha_u(0.7);
    s.set_alpha_p(0.3);
    s.set_outer_iters(25);

    let (fx, fy) = (mesh.face_cx.clone(), mesh.face_cy.clone());
    for c in 0..2u32 {
        let (fx, fy) = (fx.clone(), fy.clone());
        let wall = move |i: u32| {
            let (ux, uy) = exact_u(fx[i as usize], fy[i as usize]);
            (if c == 0 { ux } else { uy }) as f32
        };
        s.set_boundary_values_per_face(GpuBoundaryType::Wall, "U", c, &wall)
            .expect("wall bc");
    }
    let src: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    s.set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src).expect("src");
    s.set_field_vec2("U", &vec![(0.0, 0.0); mesh.num_cells()]).expect("U0");
    s.set_field_scalar("p", &vec![0.0; mesh.num_cells()]).expect("p0");
    s.initialize_history();

    for _ in 0..steps {
        s.step();
    }
    let u = s.get_field_vec2("U").expect("read U");
    (mesh, u)
}

fn l2_u(mesh: &Mesh, u: &[(f64, f64)]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..mesh.num_cells() {
        let (ex, ey) = exact_u(mesh.cell_cx[i], mesh.cell_cy[i]);
        num += ((u[i].0 - ex).powi(2) + (u[i].1 - ey).powi(2)) * mesh.cell_vol[i];
        den += mesh.cell_vol[i];
    }
    (num / den).sqrt()
}

#[test]
fn cpu_incompressible_taylor_green_converges() {
    let n = 16;
    let (mesh, u) = solve(n, 40, CpuBackendConfig::default());
    let e = l2_u(&mesh, &u);
    let finite = u.iter().all(|(a, b)| a.is_finite() && b.is_finite());
    println!("[cpu-inc] n={n} u_l2={e:.4e} finite={finite}");
    assert!(finite, "U went non-finite");
    assert!(e < 5e-2, "U error too large (saddle-point solve not converging): {e:.4e}");
}

/// The transpiled engine (compiled-Rust kernels, with interpreter fallback for
/// the few unsupported kernels) must agree with the interpreter on the coupled
/// saddle-point path. Short run (cost-bounded); agreement to ~f32 round-off.
#[test]
fn cpu_incompressible_engines_agree() {
    let n = 8;
    let (m, u_i) = solve(n, 10, CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false });
    let (_m, u_t) = solve(n, 10, CpuBackendConfig { engine: CpuEngine::Transpiled, threads: 1, simd: false });
    let d = u_i
        .iter()
        .zip(&u_t)
        .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
        .fold(0.0, f64::max);
    let _ = &m;
    println!("[cpu-inc-engines] n={n} max|interp-transpiled|={d:.3e}");
    assert!(d < 1e-4, "transpiled vs interpreter diverge: {d:.3e}");
}

/// Velocity convergence order through the full coupled saddle-point path
/// (Rhie-Chow flux + CPU Schur preconditioner). Two levels (cost-bounded);
/// SOU convection => ~2nd order velocity.
#[test]
fn cpu_incompressible_taylor_green_order() {
    let mut hs = Vec::new();
    let mut es = Vec::new();
    for n in [8usize, 16] {
        let (mesh, u) = solve(n, 40, CpuBackendConfig::default());
        let e = l2_u(&mesh, &u);
        println!("[cpu-inc-order] n={n} u_l2={e:.4e}");
        hs.push(1.0 / n as f64);
        es.push(e);
    }
    let order = (es[0] / es[1]).log2(); // levels double
    println!("[cpu-inc-order] u order = {order:.3}");
    assert!(order > 1.5, "velocity order {order:.3} too low (expected ~2)");
    assert!(*es.last().unwrap() < 5e-3, "finest u error {:.3e}", es.last().unwrap());
}
