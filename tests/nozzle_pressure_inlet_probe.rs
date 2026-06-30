//! EVIDENCE (`#[ignore]`'d): the kill-or-confirm experiment behind the SHIPPED
//! pressure-inlet nozzle (`ALLMACH_THERMAL_NOZZLE.pressure_inlet`,
//! `apply_pressure_inlet_nozzle_bcs`). Documents that the all-Mach CD nozzle CAN run on a
//! PRESSURE INLET + SUPERSONIC (fully-extrapolated) OUTLET — stable, vacuum-free — and
//! WHY it over-expands (throat over-chokes, diverging section diffuses: M_throat > M_exit)
//! in this artificial-compressibility model. The DIR (axial-direction-constrained) inlet
//! is the shipped choice; the p_inlet sweep here motivated the shipped inlet_pressure.
//! The production path is gated by `tests/gui_supersonic_nozzle_demo.rs`.
//!
//! Physics goal (the physically-correct nozzle): pin a static gauge pressure at the
//! inlet (Dirichlet p there = the gauge anchor), let the velocity DEVELOP from the
//! pressure drop, and extrapolate EVERYTHING at the outlet (no back-pressure). The
//! gauge stays anchored at the inlet so the outlet can float without the pressure level
//! drifting (the same one-Dirichlet-pin contract the current model satisfies at the
//! outlet; we just move the single pin upstream).
//!
//! This proves the PHYSICS before any production wiring: BC *kind* is baked at build
//! time, so instead of adding a model variant we MUTATE `allmach_thermal_model()`'s
//! `boundaries` in place, then drive it by inlet pressure. If it chokes + reaches a
//! supersonic exit + stays bounded + positive-absolute, we promote it to a real variant
//! (model + driver + GUI). If it rings / backflows / hits vacuum, we take the off-ramp
//! and keep the velocity-inlet demo.
//!
//! Two inlet flavors are tried:
//!   DIR = axial-direction-constrained (U_x extrapolate, U_y = 0 Dirichlet) — the correct
//!         subsonic-inflow count (3 pinned: p, U_y, T; 1 extrapolated: U_x); blocks
//!         transverse backflow.
//!   EXT = fully extrapolated U (both components ZeroGradient) — the naive version; if it
//!         rings where DIR is stable, that confirms the backflow risk.
//!
//! Run: cargo test --features "dev-tests ui" --test nozzle_pressure_inlet_probe -- --ignored --nocapture

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{
    allmach_thermal_model, BoundaryCondition, ALLMACH_TEMPERATURE_FIELD, ALLMACH_T_REF,
};
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;
use cfd2_ir::dimensions::{DivDim, InvTime, Length, Pressure, Temperature, Velocity};

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

#[derive(Clone, Copy)]
enum InletKind {
    Dir, // axial direction-constrained: U_x extrapolate, U_y = 0
    Ext, // fully extrapolated U
}

/// Build the nozzle with the all-Mach thermal model whose Inlet/Outlet BCs have been
/// flipped in place to pressure-inlet + supersonic-outlet.
fn run(air: &Fluid, mesh: &Mesh, inlet: InletKind, p_inlet: f64, steps: usize) -> Option<Report> {
    let c = 1.0 / PSI.sqrt();
    let n = mesh.num_cells();

    // ---- mutate the baked-at-build BoundarySpec ----
    let mut model = allmach_thermal_model().expect("allmach_thermal model");
    {
        // U-Inlet: extrapolate (Ext) or axial-direction-constrained (Dir).
        let u = model.boundaries.fields.get_mut("U").expect("U field BCs");
        let u_inlet = match inlet {
            InletKind::Ext => vec![
                BoundaryCondition::zero_gradient_dim::<InvTime>(),
                BoundaryCondition::zero_gradient_dim::<InvTime>(),
            ],
            InletKind::Dir => vec![
                BoundaryCondition::zero_gradient_dim::<InvTime>(), // U_x develops with the drop
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0), // U_y pinned axial
            ],
        };
        u.by_boundary.insert(GpuBoundaryType::Inlet, u_inlet);

        // p-Inlet: Dirichlet (the NEW gauge anchor; runtime value set below).
        // p-Outlet: ZeroGradient (supersonic, no back-pressure).
        let p = model.boundaries.fields.get_mut("p").expect("p field BCs");
        p.by_boundary.insert(
            GpuBoundaryType::Inlet,
            vec![BoundaryCondition::dirichlet_dim::<Pressure>(0.0)],
        );
        p.by_boundary.insert(
            GpuBoundaryType::Outlet,
            vec![BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>()],
        );

        // T-Outlet: extrapolate (a supersonic outlet must let the gas cool, not pin T_ref).
        let t = model
            .boundaries
            .fields
            .get_mut(ALLMACH_TEMPERATURE_FIELD)
            .expect("T field BCs");
        t.by_boundary.insert(
            GpuBoundaryType::Outlet,
            vec![BoundaryCondition::zero_gradient_dim::<DivDim<Temperature, Length>>()],
        );
    }

    // ---- params: mirror the supersonic test; inlet_velocity is only the precond/CFL scale now ----
    let mut d = gui_defaults_for("allmach_pressure");
    d.inlet_velocity = 0.09;
    let params = d.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);

    // IC: uniform axial throughflow (NOT rest) + a linear pressure ramp p_inlet -> 0 so step 0
    // already carries the driving gradient (cuts the start-up acoustic transient).
    let initial_u = vec![(0.09_f64, 0.0); n];
    let ramp: Vec<f64> = (0..n)
        .map(|cc| p_inlet * (1.0 - mesh.cell_cx[cc] / LENGTH))
        .collect();

    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh, model, &params, &initial_u, &ramp, None, None,
    ))
    .expect("driver build");
    driver.apply_params(&params); // sets Outlet "p" back-pressure -> inert (Outlet is ZeroGradient now)
    let mut solver = driver.into_solver();

    let rho_ref = air.density;
    solver.set_field_scalar("psi", &vec![PSI; n]).unwrap();
    solver.set_field_scalar("psi_precond", &vec![PSI; n]).unwrap();
    solver.set_field_scalar("rho", &vec![rho_ref; n]).unwrap();
    solver
        .set_field_scalar("rho_t_ref", &vec![rho_ref * ALLMACH_T_REF; n])
        .unwrap();
    solver.set_field_scalar("T", &vec![ALLMACH_T_REF; n]).unwrap();
    solver.set_field_scalar("p", &ramp).unwrap(); // re-assert the ramp IC after seeding

    // The pressure INLET: pin the inlet gauge pressure (the new gauge anchor).
    solver
        .set_boundary_scalar(GpuBoundaryType::Inlet, "p", p_inlet as f32)
        .expect("set inlet pressure");

    for s in 0..steps {
        if solver.step_with_stats().is_err() {
            println!("  [diverged at step {s}]");
            return None;
        }
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).unwrap();
    let p = pollster::block_on(solver.get_field_scalar("p")).unwrap();
    let rho = pollster::block_on(solver.get_field_scalar("rho")).unwrap();
    if !u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
        || !p.iter().all(|v| v.is_finite())
    {
        println!("  [non-finite]");
        return None;
    }
    let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
    let p_ref = rho_ref / PSI;
    let outlet_cut = 0.92 * LENGTH;
    let mut min_pabs_int = f64::INFINITY;
    let mut min_pabs_out = f64::INFINITY;
    let mut min_rho = f64::INFINITY;
    let mut backflow = 0usize;
    for cc in 0..n {
        let pabs = p_ref + p[cc];
        if mesh.cell_cx[cc] > outlet_cut {
            min_pabs_out = min_pabs_out.min(pabs);
        } else {
            min_pabs_int = min_pabs_int.min(pabs);
        }
        min_rho = min_rho.min(rho[cc]);
        // count inlet-region cells with reversed (outflow-at-inlet) axial velocity
        if mesh.cell_cx[cc] < 0.05 * LENGTH && u[cc].0 < 0.0 {
            backflow += 1;
        }
    }
    Some(Report {
        m_throat: region_max(mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH),
        m_exit: region_max(mesh, &mach, 0.85 * LENGTH, LENGTH),
        u_inlet_max: region_max(
            mesh,
            &u.iter().map(|(a, _)| *a).collect::<Vec<_>>(),
            0.0,
            0.05 * LENGTH,
        ),
        min_pabs_int,
        min_pabs_out,
        min_rho,
        backflow,
    })
}

struct Report {
    m_throat: f64,
    m_exit: f64,
    u_inlet_max: f64,
    min_pabs_int: f64,
    min_pabs_out: f64,
    min_rho: f64,
    backflow: usize,
}

#[test]
#[ignore]
fn pressure_inlet_supersonic_outlet_kill_or_confirm() {
    let air = Fluid::presets()[1].clone();
    let mesh = nozzle(96, 32);
    let p_ref = air.density / PSI;
    let steps = 300;
    println!(
        "=== pressure-inlet / supersonic-outlet nozzle: P_REF={p_ref:.5}, vacuum at gauge p={:+.5}, c_eff={:.4} ===",
        -p_ref,
        1.0 / PSI.sqrt()
    );
    println!("PASS = finite + M_throat in [0.9,1.2] + M_exit>1 + min_P_abs>0 + no backflow");

    let combos = [
        (InletKind::Dir, 0.03),
        (InletKind::Dir, 0.045),
        (InletKind::Dir, 0.06),
        (InletKind::Ext, 0.045), // does naive full-extrapolation ring where Dir is stable?
    ];
    for (kind, p_inlet) in combos {
        let tag = match kind {
            InletKind::Dir => "DIR",
            InletKind::Ext => "EXT",
        };
        println!("--- {tag}  p_inlet={p_inlet:+.3} (P_abs_inlet={:+.4}) ---", p_ref + p_inlet);
        match run(&air, &mesh, kind, p_inlet, steps) {
            None => println!("  RESULT: UNSTABLE / diverged"),
            Some(r) => {
                let pass = r.m_throat > 0.9
                    && r.m_throat < 1.2
                    && r.m_exit > 1.0
                    && r.min_pabs_int > 0.0
                    && r.min_pabs_out > 0.0
                    && r.backflow == 0;
                println!(
                    "  M_throat={:.3} M_exit={:.3} | U_inlet_max={:.4} backflow_cells={} | \
                     min_P_abs int={:+.5} out={:+.5} min_rho={:.4}  => {}",
                    r.m_throat,
                    r.m_exit,
                    r.u_inlet_max,
                    r.backflow,
                    r.min_pabs_int,
                    r.min_pabs_out,
                    r.min_rho,
                    if pass { "PASS" } else { "fail" }
                );
            }
        }
    }
}
