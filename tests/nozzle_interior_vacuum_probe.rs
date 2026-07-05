//! GATE: the shipping CD-nozzle demo never reaches true vacuum, anywhere.
//!
//! The all-Mach gauge-pressure EOS has absolute pressure `P_abs = P_REF + p`, with
//! `P_REF = rho_ref / psi`. VACUUM is `P_abs = 0`, i.e. gauge `p = -P_REF`. The nozzle
//! is driven by a PRESSURE INLET + SUPERSONIC (extrapolated) OUTLET
//! (`ALLMACH_THERMAL_NOZZLE`): the inlet gauge pressure is pinned POSITIVE and the
//! outlet FLOATS — no specified back-pressure. This gate asserts the realized absolute
//! pressure stays strictly POSITIVE everywhere (interior AND outlet region) and the
//! density never approaches the EOS floor (`rho ≥ psi·1e-5`).

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::allmach_thermal_model;
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::ALLMACH_THERMAL_NOZZLE;

fn air() -> Fluid {
    Fluid::presets()[1].clone()
}

const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80; // area ratio exit/throat = 2.0

fn nozzle(nx: usize, ny: usize) -> Mesh {
    generate_structured_nozzle_mesh(
        nx,
        ny,
        LENGTH,
        HEIGHT,
        THROAT_H,
        THROAT_FRAC,
        EXIT_H,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

/// Build the nozzle with the `ALLMACH_THERMAL_NOZZLE` GUI defaults: the driver derives
/// the effective `psi` from the fluid's real EOS times the demo exaggeration, and
/// `apply_params` pins the inlet pressure. Returns the derived `psi`.
fn build_nozzle(fluid: &Fluid, mesh: &Mesh) -> (UnifiedSolver, f64) {
    let d = ALLMACH_THERMAL_NOZZLE;
    let params = d.to_runtime_params(fluid.density as f32, fluid.viscosity as f32, fluid.eos);
    let psi = params.compressibility_psi.max(0.0) as f64;
    let n = mesh.num_cells();
    // Develop FROM REST (rest velocity + flat gauge p=0); the driver seeds the all-Mach
    // EOS fields (psi / psi_precond / rho / rho_t_ref / T) and pins the inlet pressure.
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        allmach_thermal_model().expect("allmach_thermal model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params); // flips BCs, pins the 1 MPa inlet gauge pressure
    (driver.into_solver(), psi)
}

#[test]
fn nozzle_interior_vacuum_probe() {
    let air = air();
    let mesh = nozzle(96, 32);
    let n = mesh.num_cells();
    let inlet_pressure = ALLMACH_THERMAL_NOZZLE.inlet_pressure as f64; // pressure-inlet drive
    let (mut solver, psi) = build_nozzle(&air, &mesh);

    // From rest the flow develops slowly; run long enough to reach the choked,
    // supersonic CD profile.
    let steps = 1000;
    for s in 0..steps {
        if solver.step_with_stats().is_err() {
            panic!("nozzle diverged at step {s}");
        }
    }

    let p = pollster::block_on(solver.get_field_scalar("p")).expect("p");
    let rho = pollster::block_on(solver.get_field_scalar("rho")).expect("rho");
    assert!(p.iter().all(|v| v.is_finite()), "p went non-finite");
    assert!(rho.iter().all(|v| v.is_finite()), "rho went non-finite");

    // P_REF = rho_ref / psi, recovered from the field (rho = psi*P_abs barotropic /
    // rho = rho_t_ref/T + psi*p thermal; at the seeded reference T=T_ref, rho_ref
    // corresponds to P_REF). Use the nominal rho_ref/psi for the absolute reference.
    let rho_ref = air.density; // 1.225
    let p_ref = rho_ref / psi; // ≈ 0.0245 at psi≈50
    let p_vacuum = -p_ref; // gauge pressure at true vacuum (P_abs = 0)

    // Helper over a cell predicate: returns (min_gauge_p, max_gauge_p, min_P_abs,
    // argmin_x of P_abs, min_rho).
    let scan = |pred: &dyn Fn(usize) -> bool| {
        let mut min_p = f64::INFINITY;
        let mut max_p = f64::NEG_INFINITY;
        let mut min_pabs = f64::INFINITY;
        let mut argmin_x = f64::NAN;
        let mut min_rho = f64::INFINITY;
        let mut count = 0usize;
        for c in 0..n {
            if !pred(c) {
                continue;
            }
            count += 1;
            min_p = min_p.min(p[c]);
            max_p = max_p.max(p[c]);
            let pabs = p_ref + p[c];
            if pabs < min_pabs {
                min_pabs = pabs;
                argmin_x = mesh.cell_cx[c];
            }
            min_rho = min_rho.min(rho[c]);
        }
        (min_p, max_p, min_pabs, argmin_x, min_rho, count)
    };

    // Outlet region cut: cells with cx > 0.92*Length are "outlet region".
    let outlet_cut = 0.92 * LENGTH;
    let all = scan(&|_c| true);
    let interior = scan(&|c| mesh.cell_cx[c] <= outlet_cut);
    let outlet = scan(&|c| mesh.cell_cx[c] > outlet_cut);

    println!("=== NOZZLE INTERIOR VACUUM PROBE (allmach_thermal, GUI default) ===");
    println!(
        "psi={psi:.4}  rho_ref={rho_ref:.4}  P_REF=rho_ref/psi={p_ref:.6}  \
         inlet_pressure(gauge)={inlet_pressure:+.6}  p_vacuum(gauge)={p_vacuum:+.6}"
    );
    println!("steps={steps}  cells={n}  outlet_cut_x={outlet_cut:.4} (cx>cut = outlet region)");
    println!(
        "ALL      cells={:>4}  p:[{:+.6}, {:+.6}]  min_P_abs={:+.6} @x={:.4}  min_rho={:.6}",
        all.5, all.0, all.1, all.2, all.3, all.4
    );
    println!(
        "INTERIOR cells={:>4}  p:[{:+.6}, {:+.6}]  min_P_abs={:+.6} @x={:.4}  min_rho={:.6}",
        interior.5, interior.0, interior.1, interior.2, interior.3, interior.4
    );
    println!(
        "OUTLET   cells={:>4}  p:[{:+.6}, {:+.6}]  min_P_abs={:+.6} @x={:.4}  min_rho={:.6}",
        outlet.5, outlet.0, outlet.1, outlet.2, outlet.3, outlet.4
    );

    let interior_min_pabs = interior.2;
    let outlet_min_pabs = outlet.2;
    println!(
        "VERDICT: INTERIOR min P_abs = {interior_min_pabs:+.6} (vacuum if <0); \
         OUTLET region min P_abs = {outlet_min_pabs:+.6}"
    );
    println!(
        "VERDICT: realized field stays ABOVE vacuum; pressure-inlet drive (p_in={inlet_pressure:+.3}) \
         pins a POSITIVE inlet gauge and the outlet floats — no sub-vacuum spec anywhere."
    );

    // GATE. Absolute pressure must be strictly positive over the whole field — both the
    // interior AND the outlet region.
    assert!(all.5 == n, "scan should cover all cells");
    assert!(
        interior_min_pabs > 0.0,
        "interior crossed vacuum: min P_abs = {interior_min_pabs:+.6} (the flow itself, \
         not just the BC, would need sub-vacuum pressure)"
    );
    assert!(
        outlet_min_pabs > 0.0,
        "outlet region realized sub-vacuum: min P_abs = {outlet_min_pabs:+.6} (the soft \
         Dirichlet target became a hard sub-vacuum pin)"
    );
    // The EOS density floor (= psi * absolute-pressure floor) must stay orders of
    // magnitude below the realized density — an independent guarantee that the flow is
    // nowhere near the vacuum limit.
    let floor = psi * 1.0e-5;
    assert!(
        all.4 > 1.0e6 * floor,
        "density approached the EOS floor: min_rho = {:.6e}, floor = {floor:.3e}",
        all.4
    );
}
