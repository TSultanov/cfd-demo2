//! CPU-backend buoyant (Boussinesq) MMS smoke + order: S=4 coupled saddle-point
//! (U, p, T) with buoyancy coupling and the CPU Schur preconditioner
//! (velocity-block = {U_x, U_y, T}, pressure = p). Mirrors mms_buoyant_order_test
//! but runs entirely on the CPU.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{
    buoyant_incompressible_mms_model, BUOYANT_BETA_G, BUOYANT_K_OVER_CP,
    BUOYANT_MMS_SOURCE_T_FIELD, BUOYANT_MMS_SOURCE_U_FIELD, BUOYANT_T0, BUOYANT_TEMPERATURE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;
use std::f64::consts::PI;

const MU: f64 = 1.0;
const RHO: f64 = 1.0;

fn exact_u(x: f64, y: f64) -> (f64, f64) {
    ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
}
fn exact_t(x: f64, y: f64) -> f64 {
    (PI * x).cos() * (PI * y).cos()
}
fn source_u(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    let visc = 2.0 * MU * PI * PI;
    let f_buoy_y = RHO * BUOYANT_BETA_G * (exact_t(x, y) - BUOYANT_T0);
    (visc * ux, visc * uy - f_buoy_y)
}
fn source_t(x: f64, y: f64) -> f64 {
    let (ux, uy) = exact_u(x, y);
    let dtdx = -PI * (PI * x).sin() * (PI * y).cos();
    let dtdy = -PI * (PI * x).cos() * (PI * y).sin();
    RHO * (ux * dtdx + uy * dtdy) + BUOYANT_K_OVER_CP * 2.0 * PI * PI * exact_t(x, y)
}

fn solve(n: usize, steps: usize) -> (Mesh, Vec<(f64, f64)>, Vec<f64>) {
    let mesh = generate_structured_rect_mesh(
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
    );
    let model = buoyant_incompressible_mms_model().expect("model");
    let mut s = CpuSolver::with_stepping(
        &mesh,
        model,
        Scheme::SecondOrderUpwind,
        TimeScheme::BDF2,
        SteppingMode::Coupled,
        CpuBackendConfig::default(),
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
    for b in [GpuBoundaryType::Inlet, GpuBoundaryType::Outlet, GpuBoundaryType::Wall] {
        for c in 0..2u32 {
            let (fx, fy) = (fx.clone(), fy.clone());
            let uf = move |i: u32| {
                let (ux, uy) = exact_u(fx[i as usize], fy[i as usize]);
                (if c == 0 { ux } else { uy }) as f32
            };
            s.set_boundary_values_per_face(b, "U", c, &uf).expect("U bc");
        }
    }
    for b in [GpuBoundaryType::Inlet, GpuBoundaryType::Outlet] {
        let (fx, fy) = (fx.clone(), fy.clone());
        let tf = move |i: u32| exact_t(fx[i as usize], fy[i as usize]) as f32;
        s.set_boundary_values_per_face(b, BUOYANT_TEMPERATURE_FIELD, 0, &tf)
            .expect("T bc");
    }
    {
        let (fx, fy) = (fx.clone(), fy.clone());
        let pf = move |i: u32| {
            (RHO / 4.0) * ((2.0 * PI * fx[i as usize]).cos() + (2.0 * PI * fy[i as usize]).cos())
        };
        let pf = move |i: u32| pf(i) as f32;
        s.set_boundary_values_per_face(GpuBoundaryType::Outlet, "p", 0, &pf)
            .expect("p bc");
    }

    let su: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| source_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    s.set_field_vec2(BUOYANT_MMS_SOURCE_U_FIELD, &su).expect("S_U");
    let st: Vec<f64> = (0..mesh.num_cells())
        .map(|i| source_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    s.set_field_scalar(BUOYANT_MMS_SOURCE_T_FIELD, &st).expect("S_T");

    s.set_field_vec2("U", &vec![(0.0, 0.0); mesh.num_cells()]).expect("U0");
    s.set_field_scalar("p", &vec![0.0; mesh.num_cells()]).expect("p0");
    s.set_field_scalar(BUOYANT_TEMPERATURE_FIELD, &vec![0.0; mesh.num_cells()])
        .expect("T0");
    s.initialize_history();

    for _ in 0..steps {
        s.step();
    }
    let u = s.get_field_vec2("U").expect("read U");
    let t = s.get_field_scalar(BUOYANT_TEMPERATURE_FIELD).expect("read T");
    (mesh, u, t)
}

fn l2_u(mesh: &Mesh, u: &[(f64, f64)]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..mesh.num_cells() {
        let (ex, ey) = exact_u(mesh.cell_cx[i], mesh.cell_cy[i]);
        num += ((u[i].0 - ex).powi(2) + (u[i].1 - ey).powi(2)) * mesh.cell_vol[i];
        den += mesh.cell_vol[i];
    }
    (num / den).sqrt()
}
fn l2_t(mesh: &Mesh, t: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..mesh.num_cells() {
        let e = t[i] - exact_t(mesh.cell_cx[i], mesh.cell_cy[i]);
        num += e * e * mesh.cell_vol[i];
        den += mesh.cell_vol[i];
    }
    (num / den).sqrt()
}

#[test]
fn cpu_buoyant_taylor_green_converges() {
    let mut hs = Vec::new();
    let mut eu = Vec::new();
    let mut et = Vec::new();
    for n in [8usize, 16] {
        let (mesh, u, t) = solve(n, 60);
        let finite = u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
            && t.iter().all(|v| v.is_finite());
        assert!(finite, "buoyant state went non-finite at n={n}");
        let (e_u, e_t) = (l2_u(&mesh, &u), l2_t(&mesh, &t));
        println!("[cpu-buoyant] n={n} u_l2={e_u:.4e} t_l2={e_t:.4e}");
        hs.push(1.0 / n as f64);
        eu.push(e_u);
        et.push(e_t);
    }
    let ou = (eu[0] / eu[1]).log2();
    let ot = (et[0] / et[1]).log2();
    println!("[cpu-buoyant] orders u={ou:.3} t={ot:.3}");
    assert!(ou > 1.4, "u order {ou:.3} too low");
    assert!(ot > 1.4, "T order {ot:.3} too low");
    assert!(*eu.last().unwrap() < 1e-2 && *et.last().unwrap() < 1e-2, "finest errors too large");
}
