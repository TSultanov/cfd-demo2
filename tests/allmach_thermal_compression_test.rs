//! Compression-heating SIGN certification for `allmach_thermal`.
//!
//! Setup: a CLOSED "filling box". One velocity inlet on the left; every other
//! side (including the re-tagged right boundary) is a solid no-slip wall, so the
//! domain has NO outlet — injected mass accumulates and the gauge pressure rises
//! (compression). Crucially:
//!   * the inlet injects gas at exactly `T_ref` (Dirichlet T inlet), and
//!   * the walls are adiabatic (zero-gradient T).
//! The bulk temperature is lifted above `T_ref` by the two positive energy
//! sources now in the model: the compression-heating term `-(1/cp)*Dp/Dt` and the
//! viscous-dissipation term `Phi = tau:grad(U)` (always >= 0). Compression is the
//! dominant mechanism in this filling box (the near-quiescent gas has small shear
//! but a large `dp/dt`), so a WRONG compression sign — which would push T *below*
//! `T_ref` — is not masked by the small, always-positive `Phi`.
//!
//! Therefore `max(T) > T_ref` (a clear local rise) and `mean(T) > T_ref` (net
//! warming) certify that the combined heating is positive and, given `Phi >= 0`,
//! that the compression sign is correct: a positive `dp/dt` drives `dT/dt > 0`.
//! The per-term SIGN/magnitude of each source is order-verified independently by
//! the compressible MMS suite (steady: `Phi` + `U.grad(p)`; transient: `dp/dt`),
//! and `Phi` alone by the viscous-dissipation isolation gate.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{allmach_thermal_model, ALLMACH_T_REF};
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;

fn air() -> Fluid {
    Fluid::presets()[1].clone()
}

/// Inlet on the left; walls on the other three sides AND the (re-tagged) right
/// side, giving a closed, mass-accumulating domain.
fn closed_box_mesh(nx: usize, ny: usize, length: f64, height: f64) -> Mesh {
    let mut mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet, // re-tagged to Wall just below
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    for f in 0..mesh.num_faces() {
        if mesh.face_boundary[f] == Some(BoundaryType::Outlet) {
            mesh.face_boundary[f] = Some(BoundaryType::Wall);
        }
    }
    mesh
}

fn build_closed_box(fluid: &Fluid, mesh: &Mesh, psi: f64, inlet_v: f64) -> UnifiedSolver {
    let mut d = gui_defaults_for("allmach_pressure");
    d.inlet_velocity = inlet_v as f32;
    let params = d.to_runtime_params(fluid.density as f32, fluid.viscosity as f32, fluid.eos);
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        allmach_thermal_model().expect("allmach_thermal model"),
        &params,
        &vec![(inlet_v, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    let rho_ref = fluid.density as f64;
    solver.set_field_scalar("psi", &vec![psi; n]).expect("psi");
    // The reference compressibility drives the EOS coefficients (rho recovery, 1/cp);
    // this raw-solver test pins the sound speed, so seed psi_ref = psi too.
    solver
        .set_field_scalar("psi_ref", &vec![psi; n])
        .expect("psi_ref");
    // Pressure-row ddt reads the decoupled `psi_precond` (preconditioning is a driver-only
    // transient device); this raw-solver test pins the compressibility, so seed it = psi.
    solver
        .set_field_scalar("psi_precond", &vec![psi; n])
        .expect("psi_precond");
    solver.set_field_scalar("rho", &vec![rho_ref; n]).expect("rho");
    solver
        .set_field_scalar("rho_t_ref", &vec![rho_ref * ALLMACH_T_REF; n])
        .expect("rho_t_ref");
    solver
        .set_field_scalar("T", &vec![ALLMACH_T_REF; n])
        .expect("T");
    solver
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

#[test]
fn compression_heating_raises_temperature() {
    let air = air();
    // psi=50 => sound speed c=1/sqrt(psi)~0.14; inlet 0.1 => inlet Mach ~0.7, so
    // the accumulation drives a sizeable Dp/Dt and a clearly-detectable T rise.
    let mesh = closed_box_mesh(24, 24, 1.0, 1.0);
    let n = mesh.num_cells();
    let psi = 50.0;
    let inlet_v = 0.1;

    let mut solver = build_closed_box(&air, &mesh, psi, inlet_v);

    let mut steps = 0;
    let mut diverged = false;
    for _ in 0..60 {
        if solver.step_with_stats().is_err() {
            diverged = true;
            break;
        }
        steps += 1;
    }
    assert!(!diverged, "closed-box compression solve diverged after {steps} steps");

    let p = pollster::block_on(solver.get_field_scalar("p")).expect("p");
    let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    let rho = pollster::block_on(solver.get_field_scalar("rho")).expect("rho");

    assert!(p.iter().all(|v| v.is_finite()), "p non-finite");
    assert!(t.iter().all(|v| v.is_finite() && *v > 0.0), "T non-finite/non-positive");
    assert!(rho.iter().all(|v| v.is_finite() && *v > 0.0), "rho non-finite/non-positive");

    let mean_p = mean(&p);
    let mean_t = mean(&t);
    let max_t = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_t = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_rho = mean(&rho);

    println!(
        "[compression] cells={n} steps={steps} psi={psi} inlet_v={inlet_v} \
         mean_p={mean_p:.4e} mean_rho={mean_rho:.4} T: mean={mean_t:.5} max={max_t:.5} \
         min={min_t:.5} (T_ref={ALLMACH_T_REF})"
    );

    // (a) Compression actually happened: gauge pressure (and density) rose.
    assert!(
        mean_p > 1e-4,
        "expected compression (mean gauge p > 0), got mean_p={mean_p:.3e}"
    );
    assert!(
        mean_rho > air.density,
        "expected mass accumulation (mean rho > rho_ref), got {mean_rho:.4}"
    );

    // (b) Compression heating, CORRECT SIGN. With an isothermal (T_ref) inlet,
    // adiabatic walls and no other heat source, the bulk can ONLY warm via the
    // -(1/cp)*Dp/Dt term. A clear local rise + net warming certify presence + sign.
    assert!(
        max_t > ALLMACH_T_REF * 1.05,
        "compression heating absent/too weak: max T = {max_t:.5} (T_ref={ALLMACH_T_REF})"
    );
    assert!(
        mean_t > ALLMACH_T_REF,
        "net temperature did NOT rise under compression — sign likely wrong: \
         mean T = {mean_t:.5} (T_ref={ALLMACH_T_REF})"
    );
}

/// Isolation gate for the viscous-dissipation term `Phi = tau:grad(U)`.
///
/// Same closed filling box, run TWICE at different viscosities. Compression
/// heating `-(1/cp)*Dp/Dt` is INDEPENDENT of `mu`; viscous dissipation
/// `Phi = mu * 2[(du/dx)^2 + ... ] >= 0` scales with `mu`. The inlet velocity is a
/// fixed Dirichlet condition, so the inlet shear layer (hence `grad U`) is anchored
/// regardless of `mu` — a larger `mu` therefore injects strictly more `Phi` heat.
/// `max T(high mu) > max T(base mu)` isolates `Phi`'s PRESENCE and positive
/// (heating) SIGN from the mu-independent compression heating.
#[test]
fn viscous_dissipation_raises_temperature_with_viscosity() {
    let air = air();
    let mut air_hi = air.clone();
    air_hi.viscosity = air.viscosity * 100.0;

    let mesh = closed_box_mesh(24, 24, 1.0, 1.0);
    let psi = 50.0;
    let inlet_v = 0.1;

    let run = |fluid: &Fluid| -> (f64, f64) {
        let mut solver = build_closed_box(fluid, &mesh, psi, inlet_v);
        let mut diverged = false;
        for _ in 0..60 {
            if solver.step_with_stats().is_err() {
                diverged = true;
                break;
            }
        }
        assert!(!diverged, "closed-box viscous solve diverged");
        let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
        assert!(
            t.iter().all(|v| v.is_finite() && *v > 0.0),
            "T non-finite/non-positive"
        );
        let max_t = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (max_t, mean(&t))
    };

    let (max_base, mean_base) = run(&air);
    let (max_hi, mean_hi) = run(&air_hi);

    println!(
        "[viscous-iso] base(mu={:.3e}): maxT={max_base:.6} meanT={mean_base:.6}  \
         hi(mu={:.3e}): maxT={max_hi:.6} meanT={mean_hi:.6}  d(maxT)={:.3e}",
        air.viscosity,
        air_hi.viscosity,
        max_hi - max_base
    );

    // Phi ∝ mu and Phi >= 0, compression heating is mu-independent => a 100x
    // viscosity increase must add heat at the peak (the anchored inlet shear).
    assert!(
        max_hi > max_base + 1e-5,
        "viscous dissipation added no heat: max T {max_base:.6} -> {max_hi:.6} under 100x mu \
         (Phi missing or wrong sign?)"
    );
    assert!(
        mean_hi > mean_base,
        "mean T did not rise with viscosity: {mean_base:.6} -> {mean_hi:.6}"
    );
}
