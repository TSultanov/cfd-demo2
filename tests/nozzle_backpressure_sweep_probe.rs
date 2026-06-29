//! EVIDENCE (`#[ignore]`'d): WHY the supersonic-nozzle demo keeps its sub-vacuum
//! back-pressure *target* instead of a positive-absolute one.
//!
//! The vacuum gate (`nozzle_interior_vacuum_probe`) shows the realized field never
//! crosses vacuum; the `-0.045` back-pressure is only a sub-vacuum *specification* (a
//! soft Dirichlet target). The obvious "fix" is to raise the spec to positive-absolute
//! (gauge p_back > -P_REF, i.e. P_abs = P_REF + p_back > 0) by choking the throat harder
//! (higher inlet) so the diverging section accelerates by AREA expansion. This sweep
//! MEASURED that path and shows it is a BAD trade:
//!
//!   inlet 0.09, p_back -0.045 : M_throat 1.04 -> M_exit 1.11   (clean: exit > throat)
//!   inlet 0.11, p_back -0.022 : M_throat 1.19 -> M_exit 1.00   (over-choked, subsonic exit)
//!   inlet 0.13, p_back -0.022 : M_throat 1.31 -> M_exit 1.07   (SHOCK: exit < throat)
//!   inlet 0.15, p_back -0.022 : M_throat 1.44 -> M_exit 1.22   (worse shock)
//!
//! Once the spec is positive-absolute the throat over-chokes and the diverging section
//! DECELERATES (M_throat > M_exit) — an over-expansion shock, not the clean isentropic
//! acceleration the demo teaches (and the GUI-path run drops cooling to T_min 0.85 vs
//! 0.75). So the sub-vacuum soft target is load-bearing; the demo keeps it (decision:
//! "keep the clean demo"). The realized field is positive-absolute regardless — see the
//! gate. Kept as reproducible evidence for that conclusion.
//!
//! Run: cargo test --features "dev-tests ui" --test nozzle_backpressure_sweep_probe -- --ignored --nocapture

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{allmach_thermal_model, ALLMACH_T_REF};
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;

const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80; // area ratio exit/throat = 2.0
const PSI: f64 = 50.0; // effective compressibility of the demo (c = 1/sqrt(psi) ~ 0.1414)

fn nozzle(nx: usize, ny: usize) -> Mesh {
    generate_structured_nozzle_mesh(
        nx, ny, LENGTH, HEIGHT, THROAT_H, THROAT_FRAC, EXIT_H,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

fn build_nozzle(fluid: &Fluid, mesh: &Mesh, inlet_v: f64) -> UnifiedSolver {
    let mut d = gui_defaults_for("allmach_pressure");
    d.inlet_velocity = inlet_v as f32;
    let params = d.to_runtime_params(fluid.density as f32, fluid.viscosity as f32, fluid.eos);
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh, allmach_thermal_model().expect("model"), &params,
        &vec![(inlet_v, 0.0); n], &vec![0.0; n], None, None,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    let rho_ref = fluid.density as f64;
    solver.set_field_scalar("psi", &vec![PSI; n]).unwrap();
    solver.set_field_scalar("psi_precond", &vec![PSI; n]).unwrap();
    solver.set_field_scalar("rho", &vec![rho_ref; n]).unwrap();
    solver.set_field_scalar("rho_t_ref", &vec![rho_ref * ALLMACH_T_REF; n]).unwrap();
    solver.set_field_scalar("T", &vec![ALLMACH_T_REF; n]).unwrap();
    solver
}

fn region_max(mesh: &Mesh, f: &[f64], x0: f64, x1: f64) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for c in 0..mesh.num_cells() {
        let x = mesh.cell_cx[c];
        if x >= x0 && x < x1 {
            m = m.max(f[c]);
        }
    }
    m
}

#[test]
#[ignore]
fn sweep_inlet_backpressure_for_positive_absolute_supersonic() {
    let air = Fluid::presets()[1].clone();
    let mesh = nozzle(96, 32);
    let n = mesh.num_cells();
    let c = 1.0 / PSI.sqrt();
    let rho_ref = air.density;
    let p_ref = rho_ref / PSI; // ~0.0245
    let outlet_cut = 0.92 * LENGTH;
    let steps = 350;

    println!("=== nozzle (inlet, p_back) sweep: P_REF={p_ref:.5}, vacuum at gauge p={:+.5}, c_eff={c:.4} ===", -p_ref);
    println!("target: M_exit>1.0 with p_back > -P_REF (P_abs at outlet > 0)");
    // (inlet, p_back). p_back > -0.0245 are positive-absolute; -0.045 is the over-specified baseline.
    let combos = [
        (0.09, -0.045), // current default (sub-vacuum target) — reference
        (0.11, -0.022),
        (0.13, -0.022),
        (0.15, -0.022),
        (0.13, -0.020),
    ];
    for (inlet_v, p_back) in combos {
        let mut solver = build_nozzle(&air, &mesh, inlet_v);
        solver.set_boundary_scalar(GpuBoundaryType::Outlet, "p", p_back as f32).unwrap();
        let mut diverged = false;
        for _ in 0..steps {
            if solver.step_with_stats().is_err() { diverged = true; break; }
        }
        if diverged {
            println!("inlet={inlet_v:.2} p_back={p_back:+.3} (P_abs_out={:+.4}): DIVERGED", p_ref + p_back);
            continue;
        }
        let u = pollster::block_on(solver.get_field_vec2("U")).unwrap();
        let p = pollster::block_on(solver.get_field_scalar("p")).unwrap();
        let rho = pollster::block_on(solver.get_field_scalar("rho")).unwrap();
        if !u.iter().all(|(a, b)| a.is_finite() && b.is_finite()) {
            println!("inlet={inlet_v:.2} p_back={p_back:+.3}: NON-FINITE U");
            continue;
        }
        let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
        let m_throat = region_max(&mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH);
        let m_exit = region_max(&mesh, &mach, 0.85 * LENGTH, LENGTH);
        let mut min_pabs_int = f64::INFINITY;
        let mut min_pabs_out = f64::INFINITY;
        let mut min_rho = f64::INFINITY;
        for cc in 0..n {
            let pabs = p_ref + p[cc];
            if mesh.cell_cx[cc] > outlet_cut { min_pabs_out = min_pabs_out.min(pabs); }
            else { min_pabs_int = min_pabs_int.min(pabs); }
            min_rho = min_rho.min(rho[cc]);
        }
        let flag = if m_exit > 1.0 && min_pabs_int > 0.0 && min_pabs_out > 0.0 { "  <== SUPERSONIC + NO VACUUM" } else { "" };
        println!(
            "inlet={inlet_v:.2} p_back={p_back:+.3} (P_abs_out_spec={:+.4}): \
             M_throat={m_throat:.3} M_exit={m_exit:.3} | min_P_abs int={min_pabs_int:+.5} out={min_pabs_out:+.5} min_rho={min_rho:.4}{flag}",
            p_ref + p_back
        );
    }
}
