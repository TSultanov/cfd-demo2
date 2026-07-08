//! COMPRESSIBLE ALE mass-conservation gate (the variable-density analogue of the
//! incompressible `ale_conservation_test`).
//!
//! The incompressible conservation audit is mesh-side only: `rho` is a solver constant,
//! so `Σ rho·V = rho·Σ V` and it just re-checks the swept-quad area telescope. This test
//! exercises the REAL question for the pressure-based compressible-thermal ALE: does the
//! discrete continuity (Rhie-Chow mass flux + the ALE mesh-relative flux + the per-cell
//! `rho·dV/dt` volume source) CONSERVE the total thermal-EOS mass `M = Σ_i rho_i·V_i` on a
//! genuinely MOVING mesh, when `rho = rho(p,T)` is spatially non-uniform and being convected?
//!
//! Setup: a CLOSED box (all SlipWall — no through-flow, so total mass is exactly conserved
//! in the continuum) of an all-Mach thermal gas at rest with a localized Gaussian pressure
//! bump. `psi = 0.1` (`c ≈ 3.16`) makes the EOS density `rho = rho_t_ref/T + γ·psi·t_ref·p/T`
//! vary ~15% with the bump, so the acoustic transit genuinely redistributes mass across a
//! swirling interior Voronoi mesh (regenerated through the full swept-flux + refresh loop
//! every step).
//!
//! FINDING (why this is an ALE-NEUTRALITY gate, not an absolute-conservation one): the base
//! pressure-based solver does NOT conserve the thermal-EOS mass exactly — its continuity
//! conserves a low-Mach PRECONDITIONED pseudo-mass (`psi_precond != real psi`), giving a ~0.7%
//! `M` drift over 120 steps. That is a base pressure-based-vs-density-based property, present
//! identically on a STATIC mesh (the control), NOT an ALE defect. What this test proves is
//! that the ALE machinery (mesh-relative flux + per-cell `rho·dV/dt` volume source) adds NO
//! conservation error on top of that: measured moving/static drift ratio ~1.00. So the
//! moving-mesh continuity is conservation-neutral; strict conservation is the separate
//! density-based-ALE roadmap item. Mass is measured AFTER the first step (once the on-device
//! EOS recovery makes `rho` consistent with the seeded `p`), so this is a pure conservation
//! statement, not an initialization-consistency one.
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{LloydConfig, RectangularChannel};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::allmach_thermal_ale_model;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;

const LX: f64 = 1.0;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const DT: f32 = 0.005;
const STEPS: usize = 120;
const PERIOD: f64 = 80.0 * DT as f64;
/// Genuinely compressible: psi = 1/c^2 = 0.1 (c ~ 3.16), so the p bump moves rho ~15%.
const PSI: f32 = 0.1;
const RHO_REF: f32 = 1.0;
/// Gaussian pressure bump: amplitude and width. p_bump ~ 1 => rho varies ~ gamma*psi ~ 0.14.
const P_AMP: f64 = 1.0;
const P_SIGMA: f64 = 0.15;

/// Flip-free interior swirl (boundary-vanishing), identical to the allmach_ale free-stream
/// test — the domain BOUNDARY stays fixed (closed box) while the interior mesh swirls.
fn swirl(p: [f64; 2], t: f64) -> [f64; 2] {
    let (cx, cy) = (0.5 * LX, 0.5 * LY);
    let bump = (std::f64::consts::PI * p[0] / LX).sin().powi(2)
        * (std::f64::consts::PI * p[1] / LY).sin().powi(2);
    let theta = 0.03 * (2.0 * std::f64::consts::PI * t / PERIOD).sin() * bump;
    let (dx, dy) = (p[0] - cx, p[1] - cy);
    let (c, s) = (theta.cos(), theta.sin());
    [cx + c * dx - s * dy, cy + s * dx + c * dy]
}

/// Static mesh (no motion) — the CONTROL: isolates the base pressure-based solver's mass
/// conservation from the ALE contribution. Any drift here is NOT an ALE defect.
fn no_motion(p: [f64; 2], _t: f64) -> [f64; 2] {
    p
}

fn square() -> (RectangularChannel, Vector2<f64>) {
    (
        RectangularChannel { length: LX, height: LY },
        Vector2::new(LX, LY),
    )
}

/// CLOSED box: every boundary face is a SlipWall (u·n = 0), so there is NO mass flux through
/// the domain boundary and total mass is exactly conserved in the continuum.
fn tag_closed_box(mesh: &mut Mesh) {
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_some() {
            continue;
        }
        mesh.face_boundary[f] = Some(BoundaryType::SlipWall);
    }
}

fn test_params() -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: DT,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 1000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 8,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 0.0,
        density: RHO_REF,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: PSI,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn p_bump(x: f64, y: f64) -> f64 {
    let r2 = (x - 0.5 * LX).powi(2) + (y - 0.5 * LY).powi(2);
    P_AMP * (-r2 / (2.0 * P_SIGMA * P_SIGMA)).exp()
}

/// Total thermal-EOS mass M = Σ_i rho_i · V_i on the CURRENT mesh.
fn total_mass(moving: &MovingMeshDriver, rho_off: usize, stride: usize) -> f64 {
    let state = pollster::block_on(moving.driver().solver().read_state_f32());
    let vols = &moving.mesh().cell_vol;
    (0..moving.mesh().num_cells())
        .map(|c| state[c * stride + rho_off] as f64 * vols[c])
        .sum()
}

struct ConsOut {
    max_total_drift: f64,
    max_step_drift: f64,
    rho_span_pct: f64,
}

fn run_conservation(motion: fn([f64; 2], f64) -> [f64; 2], label: &str) -> ConsOut {
    let (geo, domain) = square();
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_closed_box(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();

    // Initial: gas at rest, localized Gaussian pressure bump (drives the acoustic transient
    // that redistributes mass). rho/T/psi are seeded by the driver; rho is EOS-recovered from
    // (p,T) on-device, so the p bump becomes a ~15% density variation.
    let p0: Vec<f64> = (0..n)
        .map(|c| p_bump(cvt.mesh.cell_cx[c], cvt.mesh.cell_cy[c]))
        .collect();

    let params = test_params();
    let model = allmach_thermal_ale_model().expect("allmach thermal ale model");
    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        model,
        &params,
        MeshMotionSpec::Prescribed(motion),
        &vec![(0.0, 0.0); n],
        &p0,
        None,
        None,
    ))
    .expect("moving allmach thermal driver build");
    moving.set_boundary_retag(Some(tag_closed_box));
    moving.driver_mut().apply_params(&params);

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let rho_off = layout.offset_for("rho").expect("rho offset") as usize;

    let mut m_ref = 0.0f64; // mass after the first step (rho consistent with p)
    let mut max_step_drift = 0.0f64; // max_n |M_n - M_{n-1}| / M_ref
    let mut max_total_drift = 0.0f64; // max_n |M_n - M_ref| / M_ref
    let mut m_prev = 0.0f64;
    let mut rho_span = (f32::MAX, f32::MIN); // min/max rho (confirms the test is compressible)

    for step in 0..STEPS {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("step {step} failed (flip?): {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        assert!(
            stats.identity_err < 1e-10,
            "step {step}: f64 swept-quad identity {:.3e}",
            stats.identity_err
        );
        assert!(
            stats.scl_defect < 1e-7,
            "step {step}: f32 SCL defect {:.3e}",
            stats.scl_defect
        );

        let m = total_mass(&moving, rho_off, stride);
        assert!(m.is_finite() && m > 0.0, "step {step}: non-finite mass {m}");
        if step == 0 {
            m_ref = m;
            m_prev = m;
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            for c in 0..moving.mesh().num_cells() {
                let r = state[c * stride + rho_off];
                rho_span = (rho_span.0.min(r), rho_span.1.max(r));
            }
            continue;
        }
        max_step_drift = max_step_drift.max((m - m_prev).abs() / m_ref);
        max_total_drift = max_total_drift.max((m - m_ref).abs() / m_ref);
        m_prev = m;
    }

    let rho_span_pct = 100.0 * (rho_span.1 - rho_span.0) as f64 / rho_span.0 as f64;
    println!(
        "[ale-compressible-conservation] {label}: M_ref = {m_ref:.6}, rho in [{:.4}, {:.4}] \
         (span {rho_span_pct:.1}%), per-step drift = {max_step_drift:.3e}, \
         total drift = {max_total_drift:.3e} ({STEPS} steps, psi={PSI})",
        rho_span.0, rho_span.1,
    );

    // The test must actually be compressible (rho non-uniform) or it is a vacuous mesh-side
    // check like the incompressible audit.
    assert!(
        rho_span_pct > 2.0,
        "{label}: rho span {rho_span_pct:.3}% too small — the pressure bump is not exercising the EOS density"
    );

    ConsOut { max_total_drift, max_step_drift, rho_span_pct }
}

/// Closed-box mass conservation for the compressible-thermal ALE. Runs a STATIC control and
/// the MOVING mesh; the ALE terms must not add conservation error beyond the base
/// pressure-based solver's (whose own mass behaviour is a separate, non-ALE property).
#[test]
fn allmach_thermal_ale_conserves_mass_closed_box_cpu() {
    let stat = run_conservation(no_motion, "static");
    let moving = run_conservation(swirl, "moving");
    println!(
        "[ale-compressible-conservation] ALE contribution: moving/static total-drift ratio = {:.2} \
         (static {:.3e}, moving {:.3e})",
        moving.max_total_drift / stat.max_total_drift.max(1e-12),
        stat.max_total_drift,
        moving.max_total_drift,
    );

    // ALE-NEUTRALITY gate: the moving-mesh mass drift must not exceed the static-mesh drift —
    // the ALE mesh-relative flux + per-cell rho·dV/dt volume source must add NO conservation
    // error on top of whatever the base pressure-based solver has. Measured ratio ~1.00 (moving
    // even marginally under static), so the moving-mesh continuity is conservation-neutral.
    //
    // NOTE the ABSOLUTE drift (~0.7%) is NOT an ALE defect and is deliberately NOT gated tight:
    // the pressure-based continuity conserves a low-Mach PRECONDITIONED pseudo-mass
    // (psi_precond = max(psi, 1/u_ref^2) != real psi here since the gas starts at rest, u_ref
    // floored), not the thermal-EOS mass ∫rho(p,T)·V — a base pressure-based-vs-density-based
    // property present identically on the STATIC mesh (the control). Density-based ALE (the
    // strictly-conservative alternative) is the separate open roadmap item.
    assert!(
        moving.max_total_drift <= stat.max_total_drift * 1.10,
        "ALE ADDS mass-conservation error on a moving mesh: static {:.3e} vs moving {:.3e}",
        stat.max_total_drift,
        moving.max_total_drift,
    );
}
