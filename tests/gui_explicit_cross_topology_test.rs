//! Cross-topology equivalence gate for the explicit (RK4) GPU solvers.
//!
//! For every default GUI case that both mesh modes can run, the unstructured
//! solution is the GROUND TRUTH and the structured (dense-Cartesian + Brinkman
//! IBM) solver must reproduce it: both sides advance 500 steps at the same
//! fixed dt from the same GUI seeds/BCs, the unstructured fields are
//! interpolated onto the structured cell centers (nearest fluid centroid), and
//! any significant field discrepancy fails the gate.
//!
//! The GUI's default explicit cases are the three geometries with each
//! explicit-capable model pair: the nozzle dropdown forces the all-Mach
//! thermal model, so it contributes one pair instead of two.
//!
//! Discretization differences are EXPECTED (cut-cell walls vs a rasterized
//! Brinkman mask, different flux stencils), so the thresholds are calibrated
//! envelopes — 2x the observed discrepancy at the time the gate was written —
//! not machine-epsilon parity. Regressions that de-synchronize the physics
//! (dead inlet drive, wrong gauge, mis-scaled dt) blow far past them.
#![cfg(all(
    feature = "dev-tests",
    feature = "cpu",
    feature = "meshgen",
    feature = "ui"
))]

use cfd2::ui::app::{
    gui_explicit_rk4_gpu_available, gui_explicit_rk4_smoke, GuiExplicitRk4Case,
    GuiExplicitRk4Smoke,
};

/// GUI default cell size (the sizing sliders' initial value).
const CELL_SIZE: f64 = 0.025;
const STEPS: usize = 500;

fn run_case(
    model_id: &str,
    geometry: &str,
    mesh_kind: &str,
    dt: f32,
) -> GuiExplicitRk4Smoke {
    gui_explicit_rk4_smoke(GuiExplicitRk4Case {
        model_id,
        fluid: "Air",
        geometry,
        mesh_kind,
        backend: "gpu",
        adaptive: false,
        presentation: "plot",
        moving_mesh: false,
        cell_size: CELL_SIZE,
        steps: STEPS,
        requested_dt: Some(dt),
        advection_scheme: None,
        inlet_velocity: None,
    })
    .unwrap_or_else(|error| panic!("{geometry}/{model_id}: {error}"))
}

/// Uniform spatial hash over scattered points for nearest-neighbour lookups.
struct BucketGrid {
    origin: (f64, f64),
    inv_h: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<usize>>,
}

impl BucketGrid {
    fn build(points: &[(f64, f64)], h: f64) -> Self {
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(x, y) in points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        let inv_h = 1.0 / h;
        let nx = (((max_x - min_x) * inv_h).ceil() as usize + 2).max(1);
        let ny = (((max_y - min_y) * inv_h).ceil() as usize + 2).max(1);
        let mut buckets = vec![Vec::new(); nx * ny];
        for (index, &(x, y)) in points.iter().enumerate() {
            let bx = (((x - min_x) * inv_h) as usize).min(nx - 1);
            let by = (((y - min_y) * inv_h) as usize).min(ny - 1);
            buckets[by * nx + bx].push(index);
        }
        Self {
            origin: (min_x, min_y),
            inv_h,
            nx,
            ny,
            buckets,
        }
    }

    /// Index of the nearest point within `radius` of `(x, y)`, if any.
    fn nearest(
        &self,
        points: &[(f64, f64)],
        x: f64,
        y: f64,
        radius: f64,
    ) -> Option<usize> {
        let bx = ((x - self.origin.0) * self.inv_h).floor() as isize;
        let by = ((y - self.origin.1) * self.inv_h).floor() as isize;
        let reach = (radius * self.inv_h).ceil() as isize;
        let mut best: Option<(f64, usize)> = None;
        for jy in (by - reach)..=(by + reach) {
            for jx in (bx - reach)..=(bx + reach) {
                if jx < 0 || jy < 0 || jx as usize >= self.nx || jy as usize >= self.ny {
                    continue;
                }
                for &index in &self.buckets[jy as usize * self.nx + jx as usize] {
                    let (px, py) = points[index];
                    let d = (px - x).hypot(py - y);
                    if d <= radius && best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, index));
                    }
                }
            }
        }
        best.map(|(_, index)| index)
    }
}

struct FieldMetric {
    name: &'static str,
    rel_l2: f64,
    scale: f64,
    rms_reference: f64,
    rms_candidate: f64,
}

/// Interpolate the unstructured (ground-truth) fields onto the structured
/// fluid cell centers and measure per-field discrepancies.
fn compare(
    reference: &GuiExplicitRk4Smoke,
    candidate: &GuiExplicitRk4Smoke,
) -> Vec<FieldMetric> {
    assert_eq!(reference.topology, "unstructured");
    assert_eq!(candidate.topology, "structured");
    assert!(
        (reference.final_time - candidate.final_time).abs() < 1.0e-9,
        "the two topologies diverged in simulated time: unstructured t={} vs structured t={}",
        reference.final_time,
        candidate.final_time
    );

    let reference_lookup = BucketGrid::build(&reference.cell_centers, CELL_SIZE);
    let solid_centers: Vec<(f64, f64)> = candidate
        .cell_centers
        .iter()
        .zip(&candidate.cell_solid)
        .filter(|(_, &solid)| solid)
        .map(|(&center, _)| center)
        .collect();
    let solid_lookup = (!solid_centers.is_empty())
        .then(|| BucketGrid::build(&solid_centers, CELL_SIZE));

    // Pair each structured fluid sample with its nearest ground-truth cell.
    // Skip a band of ~3 cells around the Brinkman solid: the mask wall and the
    // cut-cell wall legitimately differ there at O(h). Skip points with no
    // ground-truth cell within 1.5 cells (outside the fluid coverage, e.g. the
    // rasterized corners of the fitted nozzle bounding box).
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut uncovered = 0usize;
    for (candidate_cell, &(x, y)) in candidate.cell_centers.iter().enumerate() {
        if candidate.cell_solid[candidate_cell] {
            continue;
        }
        if let Some(lookup) = &solid_lookup {
            if lookup.nearest(&solid_centers, x, y, 3.0 * CELL_SIZE).is_some() {
                continue;
            }
        }
        match reference_lookup.nearest(&reference.cell_centers, x, y, 1.5 * CELL_SIZE) {
            Some(reference_cell) => pairs.push((candidate_cell, reference_cell)),
            None => uncovered += 1,
        }
    }
    let fluid_samples = pairs.len() + uncovered;
    assert!(
        fluid_samples > 0 && pairs.len() * 10 >= fluid_samples * 6,
        "topologies barely overlap: {} of {fluid_samples} interior fluid samples matched a \
         ground-truth cell — geometry alignment is broken",
        pairs.len()
    );

    let metric = |name: &'static str,
                  reference_of: &dyn Fn(usize) -> f64,
                  candidate_of: &dyn Fn(usize) -> f64| {
        let n = pairs.len() as f64;
        let mut sum_sq_difference = 0.0;
        let mut sum_sq_reference = 0.0;
        let mut sum_sq_candidate = 0.0;
        for &(candidate_cell, reference_cell) in &pairs {
            let a = reference_of(reference_cell);
            let b = candidate_of(candidate_cell);
            sum_sq_difference += (a - b) * (a - b);
            sum_sq_reference += a * a;
            sum_sq_candidate += b * b;
        }
        let rms_reference = (sum_sq_reference / n).sqrt();
        let rms_candidate = (sum_sq_candidate / n).sqrt();
        let scale = rms_reference.max(rms_candidate);
        FieldMetric {
            name,
            rel_l2: if scale > 0.0 {
                (sum_sq_difference / n).sqrt() / scale
            } else {
                0.0
            },
            scale,
            rms_reference,
            rms_candidate,
        }
    };

    let mut metrics = vec![
        metric(
            "u_x",
            &|cell| f64::from(reference.velocity[cell].0),
            &|cell| f64::from(candidate.velocity[cell].0),
        ),
        metric(
            "u_y",
            &|cell| f64::from(reference.velocity[cell].1),
            &|cell| f64::from(candidate.velocity[cell].1),
        ),
        metric(
            "p",
            &|cell| f64::from(reference.pressure[cell]),
            &|cell| f64::from(candidate.pressure[cell]),
        ),
    ];
    if let (Some(reference_rho), Some(candidate_rho)) =
        (&reference.density, &candidate.density)
    {
        metrics.push(metric(
            "rho",
            &|cell| f64::from(reference_rho[cell]),
            &|cell| f64::from(candidate_rho[cell]),
        ));
    }
    if let (Some(reference_t), Some(candidate_t)) =
        (&reference.temperature, &candidate.temperature)
    {
        metrics.push(metric(
            "T",
            &|cell| f64::from(reference_t[cell]),
            &|cell| f64::from(candidate_t[cell]),
        ));
    }
    metrics
}

fn run_pair(
    geometry: &str,
    mesh_kind: &str,
    unstructured_model: &str,
    structured_model: &str,
    dt: f32,
) -> Option<Vec<FieldMetric>> {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping cross-topology {geometry}/{unstructured_model}: {error}");
        return None;
    }
    let reference = run_case(unstructured_model, geometry, mesh_kind, dt);
    let candidate = run_case(structured_model, geometry, "structured", dt);
    let metrics = compare(&reference, &candidate);
    for metric in &metrics {
        println!(
            "[cross-topology] {geometry}/{unstructured_model} {}: rel_l2={:.4} \
             (rms unstructured={:.4e} structured={:.4e})",
            metric.name, metric.rel_l2, metric.rms_reference, metric.rms_candidate
        );
    }
    Some(metrics)
}

/// One default GUI case with a developed signal: run both topologies, compare
/// on the structured fluid samples, and gate every listed field.
///
/// `thresholds` are (field name, max relative L2) envelope pairs.
fn gate(
    geometry: &str,
    mesh_kind: &str,
    unstructured_model: &str,
    structured_model: &str,
    dt: f32,
    thresholds: &[(&str, f64)],
) {
    let Some(metrics) = run_pair(geometry, mesh_kind, unstructured_model, structured_model, dt)
    else {
        return;
    };
    for &(name, max_rel_l2) in thresholds {
        let metric = metrics
            .iter()
            .find(|metric| metric.name == name)
            .unwrap_or_else(|| panic!("case exposes no '{name}' field"));
        // Both sides must agree on the field's energy level (a dead inlet or a
        // wrong gauge shows up here long before the pointwise comparison).
        let ratio = metric.rms_candidate / metric.rms_reference.max(1.0e-300);
        assert!(
            (0.5..=2.0).contains(&ratio),
            "{geometry}/{unstructured_model} {name}: structured/unstructured rms ratio \
             {ratio:.3} (rms {:.4e} vs {:.4e}) — one topology's physics is desynchronized",
            metric.rms_candidate,
            metric.rms_reference
        );
        assert!(
            metric.rel_l2 <= max_rel_l2,
            "{geometry}/{unstructured_model} {name}: relative L2 discrepancy {:.4} exceeds \
             the {max_rel_l2} envelope",
            metric.rel_l2
        );
    }
}

/// One default GUI case whose GROUND-TRUTH solution is still quiescent inside
/// the test window (the all-Mach channel cases: the GUI's derived inlet
/// soft-start ramp is ~0.2 s, so 500 explicit steps sit at ~2% of the ramp and
/// the true fields are ~0). The gate asserts BOTH topologies agree on that
/// quiescence — phantom waves, startup noise, or an instability on either
/// side blows the floors by orders of magnitude.
fn gate_quiescent(
    geometry: &str,
    mesh_kind: &str,
    unstructured_model: &str,
    structured_model: &str,
    dt: f32,
    floors: &[(&str, f64)],
) {
    let Some(metrics) = run_pair(geometry, mesh_kind, unstructured_model, structured_model, dt)
    else {
        return;
    };
    for &(name, rms_floor) in floors {
        let metric = metrics
            .iter()
            .find(|metric| metric.name == name)
            .unwrap_or_else(|| panic!("case exposes no '{name}' field"));
        assert!(
            metric.rms_reference <= rms_floor && metric.rms_candidate <= rms_floor,
            "{geometry}/{unstructured_model} {name}: expected both topologies quiescent \
             (rms <= {rms_floor:.1e}) inside the inlet soft-start, got unstructured \
             {:.4e} / structured {:.4e}",
            metric.rms_reference,
            metric.rms_candidate
        );
    }
}

// Per-case fixed dt: below each pair's tightest adaptive equilibrium so the
// fixed-dt run is comfortably stable on BOTH meshes (the fitted nozzle throat
// cells are ~2.5x finer vertically than the Cartesian grid).
const CHANNEL_DT: f32 = 8.0e-6;
const NOZZLE_DT: f32 = 4.0e-6;

// Envelopes are ~2x the discrepancy observed when the gate was written
// (2026-07-13, wgpu 29): genuine cut-cell-vs-Brinkman discretization
// differences on the acoustic/pseudo-acoustic startup transient sit at
// rel_l2 ~ 0.1-0.4; a physics desync sits at ~1.

#[test]
fn cross_topology_obstacle_compressible() {
    gate(
        "obstacle",
        "cutcell",
        "compressible",
        "compressible_structured",
        CHANNEL_DT,
        &[("u_x", 0.15), ("u_y", 0.8), ("p", 0.15), ("rho", 0.15)],
    );
}

#[test]
fn cross_topology_backstep_compressible() {
    gate(
        "backstep",
        "cutcell",
        "compressible",
        "compressible_structured",
        CHANNEL_DT,
        &[("u_x", 0.3), ("u_y", 0.6), ("p", 0.3), ("rho", 0.3)],
    );
}

#[test]
fn cross_topology_nozzle_allmach_thermal() {
    // The GUI's nozzle dropdown forces the all-Mach thermal model, and its
    // default mesh is the body-fitted grid (the validated configuration). The
    // from-rest pressure-driven transient reaches ~Mach 1, so every field
    // carries a strong signal.
    gate(
        "nozzle",
        "fitted",
        "allmach_thermal",
        "allmach_thermal_structured",
        NOZZLE_DT,
        &[("u_x", 0.35), ("u_y", 0.8), ("p", 0.45), ("rho", 0.25), ("T", 0.4)],
    );
}

#[test]
fn cross_topology_obstacle_allmach_thermal_quiescent() {
    gate_quiescent(
        "obstacle",
        "cutcell",
        "allmach_thermal",
        "allmach_thermal_structured",
        CHANNEL_DT,
        &[("u_x", 1.0e-6), ("u_y", 1.0e-6), ("p", 1.0e-4)],
    );
}

#[test]
fn cross_topology_backstep_allmach_thermal_quiescent() {
    gate_quiescent(
        "backstep",
        "cutcell",
        "allmach_thermal",
        "allmach_thermal_structured",
        CHANNEL_DT,
        &[("u_x", 1.0e-6), ("u_y", 1.0e-6), ("p", 1.0e-4)],
    );
}

/// Long-horizon stability of the structured GUI obstacle case: with the
/// outlet gauge-pressure anchor and the kind-1 BC contract (see
/// `setup_structured_bcs`), the fluid pressure stays at the ~1 Pa acoustic
/// level indefinitely. The pre-fix code let the boundary closures feed
/// absolute values into gradient-interpreted kind-2 channels (a spurious
/// ~kappa*area*T heat source per boundary face) with no pressure anchor
/// anywhere — the mean pressure then grew without bound (~3.2e3 Pa by step
/// 6000, ~4e6 Pa by step 12000, health-check halt at t~0.25 s in the GUI).
#[test]
fn structured_obstacle_long_run_pressure_stays_acoustic() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping structured long-run gate: {error}");
        return;
    }
    let smoke = gui_explicit_rk4_smoke(GuiExplicitRk4Case {
        model_id: "compressible_structured",
        fluid: "Air",
        geometry: "obstacle",
        mesh_kind: "structured",
        backend: "gpu",
        adaptive: true,
        presentation: "plot",
        moving_mesh: false,
        cell_size: CELL_SIZE,
        steps: 6000,
        requested_dt: None,
        advection_scheme: None,
        inlet_velocity: None,
    })
    .expect("structured GUI obstacle long run");
    let mut max_gauge_p = 0.0_f64;
    let mut max_speed = 0.0_f64;
    for (cell, &solid) in smoke.cell_solid.iter().enumerate() {
        if solid {
            continue;
        }
        max_gauge_p = max_gauge_p.max(f64::from(smoke.pressure[cell]).abs());
        let (ux, uy) = smoke.velocity[cell];
        max_speed = max_speed.max(f64::from(ux).hypot(f64::from(uy)));
    }
    println!(
        "[long-run] structured obstacle t={:.4e}: max|p'|={max_gauge_p:.4e} max|u|={max_speed:.4e}",
        smoke.final_time
    );
    assert!(
        max_gauge_p < 50.0,
        "structured obstacle mean-pressure runaway is back: max|p'| = {max_gauge_p:.4e} Pa"
    );
    assert!(
        max_speed < 0.05,
        "structured obstacle velocity runaway: max|u| = {max_speed:.4e} m/s at inlet 0.011"
    );
}

/// The compressible pressure-driven nozzle (the GUI's `Compressible` +
/// `Nozzle` combination): both topologies must DEVELOP through-flow from rest
/// under the prescribed 5e4 Pa gauge inlet (the committed bc_expr closures'
/// pressure-inlet branch), stay bounded, and agree on the interpolated fields
/// within the calibrated envelopes.
#[test]
fn cross_topology_nozzle_compressible_pressure_inlet() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping compressible nozzle gate: {error}");
        return;
    }
    // The inlet acoustic front needs ~3.5 ms to reach the throat (x = 1.2 at
    // c ~ 347); 1000 steps at 6e-6 give the through-flow 6 ms to establish.
    let nozzle_case = |model_id, mesh_kind| {
        gui_explicit_rk4_smoke(GuiExplicitRk4Case {
            model_id,
            fluid: "Air",
            geometry: "nozzle",
            mesh_kind,
            backend: "gpu",
            adaptive: false,
            presentation: "plot",
            moving_mesh: false,
            cell_size: CELL_SIZE,
            steps: 1000,
            requested_dt: Some(6.0e-6),
            advection_scheme: None,
            inlet_velocity: None,
        })
        .unwrap_or_else(|error| panic!("nozzle/{model_id}: {error}"))
    };
    let reference = nozzle_case("compressible", "fitted");
    let candidate = nozzle_case("compressible_structured", "structured");
    // Both sides must actually develop: mean axial velocity at the throat
    // station (x ~ 1.2) well above zero. A dead pressure inlet (missing
    // runtime constant, missing table value) reads ~0.
    for (label, smoke) in [("unstructured", &reference), ("structured", &candidate)] {
        let mut throat = (0.0_f64, 0usize);
        for (cell, &(x, _y)) in smoke.cell_centers.iter().enumerate() {
            if !smoke.cell_solid[cell] && (x - 1.2).abs() < 0.05 {
                throat.0 += f64::from(smoke.velocity[cell].0);
                throat.1 += 1;
            }
        }
        let mean = throat.0 / throat.1.max(1) as f64;
        println!("[cross-topology] nozzle/compressible {label} throat u_x = {mean:.2} m/s");
        assert!(
            mean > 20.0,
            "{label} compressible nozzle did not develop (throat u_x = {mean:.3})"
        );
    }
    let metrics = compare(&reference, &candidate);
    for metric in &metrics {
        println!(
            "[cross-topology] nozzle/compressible {}: rel_l2={:.4} \
             (rms unstructured={:.4e} structured={:.4e})",
            metric.name, metric.rel_l2, metric.rms_reference, metric.rms_candidate
        );
    }
    // Envelopes: ~2x the observed fitted-vs-rasterized-IBM discrepancy at the
    // time the gate was written (u_x 0.38, u_y 0.54, p 0.33, rho 0.28, T 0.03).
    for &(name, max_rel_l2) in &[
        ("u_x", 0.8_f64),
        ("u_y", 1.0),
        ("p", 0.7),
        ("rho", 0.6),
        ("T", 0.1),
    ] {
        let metric = metrics
            .iter()
            .find(|metric| metric.name == name)
            .unwrap_or_else(|| panic!("case exposes no '{name}' field"));
        assert!(
            metric.rel_l2 <= max_rel_l2,
            "nozzle/compressible {name}: relative L2 discrepancy {:.4} exceeds \
             the {max_rel_l2} envelope",
            metric.rel_l2
        );
    }
}
