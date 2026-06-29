//! PROBE (linchpin experiment): does the CD nozzle reach STABLE supersonic at
//! REAL Air thermodynamics, instead of the artificial psi=50?
//!
//! Real Air: c = sqrt(gamma*R*T) = sqrt(1.4*287*300) ~ 347 m/s => psi = 1/c^2 ~ 8.3e-6.
//! The near-vacuum floor is at gauge p = -rho_ref/psi = -P_REF ~ -1.48e5 (vs the
//! artificial model's tiny -0.0245), so real units should LIFT the M~1.07 cap toward
//! M~2. This probe sweeps real inlet velocities (as Mach fractions of real c) x real
//! back-pressures (as fractions of P_REF) and reports throat/exit Mach + stability.
//! Read-only experiment; not a gate (no asserts beyond finiteness reporting).

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{allmach_thermal_model, ALLMACH_T_REF};
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;

const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80;

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

struct Run {
    m_throat: f64,
    m_exit: f64,
    rho_min: f64,
    finite: bool,
}

fn min_cell_h(mesh: &Mesh) -> f64 {
    mesh.cell_vol
        .iter()
        .map(|&v| v.sqrt())
        .fold(f64::INFINITY, f64::min)
}

#[allow(clippy::too_many_arguments)]
fn run(air: &Fluid, mesh: &Mesh, psi: f64, inlet_v: f64, p_back: f64, steps: usize) -> Run {
    let c = 1.0 / psi.sqrt();
    let mut d = gui_defaults_for("allmach_pressure");
    d.inlet_velocity = inlet_v as f32;
    // At REAL velocities (~100s m/s) the default adaptive-dt seed (0.02) is CFL~90 on
    // step 1 -> blow-up. Use ADAPTIVE dt (shrinks as the diverging section accelerates,
    // preventing the downstream CFL runaway) with a small velocity-scaled seed so step
    // 1 is already in-bounds, and a conservative target CFL.
    let h = min_cell_h(mesh);
    d.adaptive_dt = true;
    d.target_cfl = 0.4;
    d.timestep = 0.2 * h / inlet_v;
    let params = d.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        allmach_thermal_model().expect("model"),
        &params,
        &vec![(inlet_v, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    let rho_ref = air.density;
    // REAL compressibility, overriding the GUI default psi=50.
    solver.set_field_scalar("psi", &vec![psi; n]).unwrap();
    solver.set_field_scalar("rho", &vec![rho_ref; n]).unwrap();
    solver
        .set_field_scalar("rho_t_ref", &vec![rho_ref * ALLMACH_T_REF; n])
        .unwrap();
    solver.set_field_scalar("T", &vec![ALLMACH_T_REF; n]).unwrap();
    solver
        .set_boundary_scalar(GpuBoundaryType::Outlet, "p", p_back as f32)
        .unwrap();

    for _ in 0..steps {
        if solver.step_with_stats().is_err() {
            return Run { m_throat: f64::NAN, m_exit: f64::NAN, rho_min: f64::NAN, finite: false };
        }
    }
    let u = pollster::block_on(solver.get_field_vec2("U")).unwrap();
    let rho = pollster::block_on(solver.get_field_scalar("rho")).unwrap();
    let finite = u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
        && rho.iter().all(|v| v.is_finite());
    let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
    Run {
        m_throat: region_max(mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH),
        m_exit: region_max(mesh, &mach, 0.85 * LENGTH, LENGTH),
        rho_min: rho.iter().cloned().fold(f64::INFINITY, f64::min),
        finite,
    }
}

/// EVIDENCE (not a gate — `#[ignore]`'d; intentionally DIVERGES): documents WHY the
/// supersonic-nozzle demo keeps an exaggerated (low effective sound speed) regime
/// instead of real Air units. At real `c≈347`, even with adaptive dt (velocity
/// bounded ~Mach 0.7) the gauge-pressure field undershoots through the vacuum floor
/// into NEGATIVE density (`rho_min` ~ -33) — on top of being turbulent (Re≈4.5e7).
/// So real-units strong-supersonic is not a viable laminar default; the EOS-derived
/// `psi=1/c^2` is exaggerated back to the stable effective psi≈50 regime.
/// Run with `cargo test ... -- --ignored --nocapture` to reproduce.
#[test]
#[ignore]
fn real_units_nozzle_supersonic_probe() {
    let air = Fluid::presets()[1].clone(); // Air
    let c_real = air.sound_speed(); // sqrt(gamma R T) ~ 347
    let psi = 1.0 / (c_real * c_real); // EOS-derived ~ 8.3e-6
    let p_ref = air.density / psi; // near-vacuum floor magnitude ~ 1.48e5
    let mesh = nozzle(96, 32);
    println!(
        "[real-nozzle] Air c={c_real:.1} m/s  psi=1/c^2={psi:.3e}  P_REF=rho/psi={p_ref:.3e} Pa"
    );

    // A single moderate config: even the mildest back-pressure goes negative-density.
    let inlet_machs = [0.45_f64];
    let pb_fracs = [0.10_f64];
    for &m_in in &inlet_machs {
        let inlet_v = m_in * c_real;
        for &frac in &pb_fracs {
            let p_back = -frac * p_ref;
            let r = run(&air, &mesh, psi, inlet_v, p_back, 1200);
            if r.finite {
                println!(
                    "  Min={m_in:.2} (U={inlet_v:6.1})  p_back=-{frac:.2}*P_REF={p_back:9.0}  \
                     M_throat={:.3}  M_exit={:.3}  rho_min={:.4}",
                    r.m_throat, r.m_exit, r.rho_min
                );
            } else {
                println!(
                    "  Min={m_in:.2} (U={inlet_v:6.1})  p_back=-{frac:.2}*P_REF={p_back:9.0}  [NON-FINITE]"
                );
            }
        }
    }
}
