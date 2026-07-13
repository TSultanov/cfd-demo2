//! Diagnostic probe for the structured explicit obstacle blow-up: march the
//! GUI compressible obstacle case in batches and report the pressure/density
//! extrema split into solid cells, the near-interface fluid band, and the far
//! field — plus the unstructured twin's near-wall tangential velocity profile
//! (slip vs no-slip discrimination).

use cfd2::ui::app::{gui_explicit_rk4_smoke, GuiExplicitRk4Case};

fn run(model_id: &str, mesh_kind: &str, steps: usize, adaptive: bool) {
    run_with_inlet(model_id, mesh_kind, steps, adaptive, None);
}

fn run_with_inlet(
    model_id: &str,
    mesh_kind: &str,
    steps: usize,
    adaptive: bool,
    inlet_velocity: Option<f32>,
) {
    let smoke = match gui_explicit_rk4_smoke(GuiExplicitRk4Case {
        model_id,
        fluid: "Air",
        geometry: "obstacle",
        mesh_kind,
        backend: "gpu",
        adaptive,
        presentation: "plot",
        moving_mesh: false,
        cell_size: 0.025,
        steps,
        requested_dt: None,
        advection_scheme: None,
        inlet_velocity,
    }) {
        Ok(smoke) => smoke,
        Err(error) => {
            println!("{model_id} {steps} steps: FAILED: {error}");
            return;
        }
    };
    let (cx, cy, r) = (1.0, 0.51, 0.1);
    let mut classes = [
        ("solid", f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        ("band", f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        ("far", f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
    ];
    let mut max_speed = 0.0_f64;
    for (cell, &(x, y)) in smoke.cell_centers.iter().enumerate() {
        let p = f64::from(smoke.pressure[cell]);
        let rho = smoke
            .density
            .as_ref()
            .map_or(0.0, |rho| f64::from(rho[cell]));
        let d = (x - cx).hypot(y - cy);
        let class = if smoke.cell_solid[cell] {
            0
        } else if d < r + 3.0 * 0.025 {
            1
        } else {
            2
        };
        let entry = &mut classes[class];
        entry.1 = entry.1.min(p);
        entry.2 = entry.2.max(p);
        entry.3 = entry.3.min(rho);
        entry.4 = entry.4.max(rho);
        let (ux, uy) = smoke.velocity[cell];
        max_speed = max_speed.max(f64::from(ux).hypot(f64::from(uy)));
    }
    println!(
        "{model_id} {steps} steps t={:.4e} dt=[{:.3e},{:.3e}] max|u|={max_speed:.4e}",
        smoke.final_time, smoke.min_dt, smoke.max_dt
    );
    for (name, p_min, p_max, rho_min, rho_max) in classes {
        println!("  {name:>5}: p'=[{p_min:.4e},{p_max:.4e}] rho'=[{rho_min:.4e},{rho_max:.4e}]");
    }

    // Near-wall slip check: mean |u_x| in the first interior row along the
    // bottom wall vs the channel mid-height, in the inlet-side stretch where
    // the flow is undisturbed by the obstacle (x in [0.2, 0.7]).
    let mut wall = (0.0, 0usize);
    let mut mid = (0.0, 0usize);
    for (cell, &(x, y)) in smoke.cell_centers.iter().enumerate() {
        if smoke.cell_solid[cell] || !(0.2..=0.7).contains(&x) {
            continue;
        }
        let speed = f64::from(smoke.velocity[cell].0);
        if y < 0.05 {
            wall.0 += speed;
            wall.1 += 1;
        } else if (y - 0.5).abs() < 0.025 {
            mid.0 += speed;
            mid.1 += 1;
        }
    }
    if wall.1 > 0 && mid.1 > 0 {
        println!(
            "  inlet-side u_x: first wall rows {:.4e} (n={}) vs mid-height {:.4e} (n={})",
            wall.0 / wall.1 as f64,
            wall.1,
            mid.0 / mid.1 as f64,
            mid.1
        );
    }
}

fn main() {
    let mach01 = std::env::var_os("PROBE_MACH01").is_some();
    if mach01 {
        // Candidate demo regime: inlet Mach ~0.1 (Re ~ 4.6e5). Watch for a
        // developing asymmetric wake and bounded fields over ~0.4 s.
        for steps in [6000usize, 20000] {
            run_with_inlet("compressible", "cutcell", steps, true, Some(34.7));
            run_with_inlet("compressible_structured", "structured", steps, true, Some(34.7));
        }
        return;
    }
    for steps in [500usize, 2000, 6000, 12000] {
        run("compressible_structured", "structured", steps, true);
    }
    run("compressible", "cutcell", 6000, true);
}
