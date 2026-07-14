//! End-to-end gate for the KEP flux + selective filter on the GUI structured
//! obstacle — the exact configuration where Mach-scaled Kurganov dissipation
//! previously failed (grid-Nyquist u_x sawtooth swallowed the inlet drive and
//! the channel pressurized like a closed vessel, ~540 Pa/s).
//!
//! The gate runs the real GUI path (structured Brinkman IBM obstacle at cell
//! 0.005, Air, fixed dt) under `Scheme::Kep` with filter sigma 0.2, and pins
//! the three failure signatures against a same-config central-upwind
//! (vanLeer) reference run:
//!
//!  * fields stay finite,
//!  * the through-flow survives (mean fluid u_x comparable to the reference,
//!    not collapsed by an odd-even mode — the historical failure dropped it
//!    ~20x),
//!  * the mean gauge pressure stays at the acoustic piston level (no
//!    closed-vessel pressurization),
//!  * the interior u_x field carries no significant grid-Nyquist sawtooth
//!    (second-difference roughness stays a small fraction of the drive).
#![cfg(all(
    feature = "dev-tests",
    feature = "cpu",
    feature = "meshgen",
    feature = "ui"
))]

use cfd2::solver::scheme::Scheme;
use cfd2::ui::app::{
    gui_explicit_rk4_gpu_available, gui_explicit_rk4_smoke, GuiExplicitRk4Case,
    GuiExplicitRk4Smoke,
};

const CELL_SIZE: f64 = 0.005;
const DT: f32 = 8.0e-6;
const STEPS: usize = 8000;

fn run_obstacle(scheme: Option<Scheme>, filter_sigma: Option<f32>) -> GuiExplicitRk4Smoke {
    gui_explicit_rk4_smoke(GuiExplicitRk4Case {
        model_id: "compressible_structured",
        fluid: "Air",
        geometry: "obstacle",
        mesh_kind: "structured",
        backend: "gpu",
        adaptive: false,
        presentation: "plot",
        moving_mesh: false,
        cell_size: CELL_SIZE,
        steps: STEPS,
        requested_dt: Some(DT),
        advection_scheme: scheme,
        inlet_velocity: None,
        inlet_pressure: None,
        filter_sigma,
    })
    .unwrap_or_else(|error| panic!("obstacle/compressible: {error}"))
}

struct ObstacleMetrics {
    mean_ux: f64,
    mean_gauge_p: f64,
    /// Mean |second difference|/4 of u_x along x over interior fluid cells
    /// with 4-cell solid/boundary clearance — the grid-Nyquist sawtooth
    /// content (a pure (-1)^i mode of amplitude A measures exactly A).
    nyquist_roughness: f64,
}

fn metrics(smoke: &GuiExplicitRk4Smoke) -> ObstacleMetrics {
    // Recover the structured grid dims from the row-major cell centers.
    let nx = smoke
        .cell_centers
        .windows(2)
        .take_while(|w| w[1].0 > w[0].0)
        .count()
        + 1;
    let ny = smoke.cells / nx;
    assert_eq!(nx * ny, smoke.cells, "row-major grid recovery failed");

    let idx = |i: usize, j: usize| j * nx + i;
    let mut sum_ux = 0.0;
    let mut sum_p = 0.0;
    let mut fluid = 0usize;
    for c in 0..smoke.cells {
        if smoke.cell_solid[c] {
            continue;
        }
        let (ux, uy) = smoke.velocity[c];
        let p = smoke.pressure[c];
        assert!(
            ux.is_finite() && uy.is_finite() && p.is_finite(),
            "non-finite field at cell {c}"
        );
        sum_ux += ux as f64;
        sum_p += p as f64;
        fluid += 1;
    }

    let mut rough = 0.0;
    let mut rough_n = 0usize;
    for j in 0..ny {
        for i in 4..nx.saturating_sub(4) {
            let clear = (i - 4..=i + 4).all(|ii| !smoke.cell_solid[idx(ii, j)]);
            if !clear {
                continue;
            }
            let um = smoke.velocity[idx(i - 1, j)].0 as f64;
            let u0 = smoke.velocity[idx(i, j)].0 as f64;
            let up = smoke.velocity[idx(i + 1, j)].0 as f64;
            rough += (um - 2.0 * u0 + up).abs() / 4.0;
            rough_n += 1;
        }
    }

    ObstacleMetrics {
        mean_ux: sum_ux / fluid.max(1) as f64,
        mean_gauge_p: sum_p / fluid.max(1) as f64,
        nyquist_roughness: rough / rough_n.max(1) as f64,
    }
}

#[test]
fn kep_with_filter_holds_the_gui_obstacle_channel() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping: no GPU adapter ({error})");
        return;
    }

    // The un-overridden case resolves through the GUI's structured-explicit
    // compressible DEFAULT — which is now the KEP flux + selective filter
    // (sigma 0.2) — so this run gates both the numerics AND the default
    // resolution. The reference pins the historical central-upwind numerics.
    let reference = run_obstacle(Some(Scheme::SecondOrderUpwindVanLeer), Some(0.0));
    let kep = run_obstacle(None, None);
    let m_ref = metrics(&reference);
    let m_kep = metrics(&kep);
    eprintln!(
        "obstacle t={:.4}s: ref  ux={:.5} p={:.3} rough={:.3e}",
        reference.final_time, m_ref.mean_ux, m_ref.mean_gauge_p, m_ref.nyquist_roughness
    );
    eprintln!(
        "obstacle t={:.4}s: kep  ux={:.5} p={:.3} rough={:.3e}",
        kep.final_time, m_kep.mean_ux, m_kep.mean_gauge_p, m_kep.nyquist_roughness
    );

    // Through-flow survives: the historical Nyquist failure collapsed the
    // mean drive ~20x. The dissipation-free flux may legitimately differ
    // from the smeared reference by tens of percent, not by an order.
    assert!(
        m_kep.mean_ux > 0.3 * m_ref.mean_ux && m_kep.mean_ux < 3.0 * m_ref.mean_ux,
        "KEP through-flow {:.5} not comparable to reference {:.5}",
        m_kep.mean_ux,
        m_ref.mean_ux
    );

    // No closed-vessel pressurization: the failure signature accumulates
    // ~540 Pa/s * t (= ~35 Pa here); the healthy piston level is ~5 Pa.
    let p_bound = (3.0 * m_ref.mean_gauge_p.abs()).max(15.0);
    assert!(
        m_kep.mean_gauge_p.abs() < p_bound,
        "KEP mean gauge pressure {:.2} Pa exceeds bound {:.2} Pa (closed-vessel signature)",
        m_kep.mean_gauge_p,
        p_bound
    );

    // No grid-Nyquist sawtooth: roughness stays a small fraction of the
    // inlet drive (0.011 m/s). The failure mode had sawtooth amplitude of
    // the order of the drive itself.
    assert!(
        m_kep.nyquist_roughness < 0.25 * 0.011,
        "KEP interior u_x Nyquist roughness {:.3e} approaches the inlet drive",
        m_kep.nyquist_roughness
    );
}
