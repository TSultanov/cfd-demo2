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
        inlet_pressure: None,
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

fn run_nozzle(model_id: &str, mesh_kind: &str, steps: usize) {
    run_nozzle_backend(model_id, mesh_kind, steps, "gpu");
}

/// User-reported crash config: cell 0.005, gauge inlet pressure 1e5 Pa.
fn run_nozzle_fine(model_id: &str, mesh_kind: &str, steps: usize, p_in: f32, cell: f64) {
    run_nozzle_fine_dt(model_id, mesh_kind, steps, p_in, cell, None);
}

fn run_nozzle_fine_dt(
    model_id: &str,
    mesh_kind: &str,
    steps: usize,
    p_in: f32,
    cell: f64,
    fixed_dt: Option<f32>,
) {
    let smoke = match gui_explicit_rk4_smoke(GuiExplicitRk4Case {
        model_id,
        fluid: "Air",
        geometry: "nozzle",
        mesh_kind,
        backend: "gpu",
        adaptive: fixed_dt.is_none(),
        presentation: "plot",
        moving_mesh: false,
        cell_size: cell,
        steps,
        requested_dt: fixed_dt,
        advection_scheme: None,
        inlet_velocity: None,
        inlet_pressure: Some(p_in),
    }) {
        Ok(smoke) => smoke,
        Err(error) => {
            println!(
                "nozzle-fine {model_id} cell={cell} p_in={p_in} dt={fixed_dt:?} {steps} steps: FAILED: {error}"
            );
            return;
        }
    };
    let mut max_speed = 0.0_f64;
    let (mut p_min, mut p_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut rho_min = f64::INFINITY;
    for (cell_idx, _) in smoke.cell_centers.iter().enumerate() {
        if smoke.cell_solid[cell_idx] {
            continue;
        }
        let (ux, uy) = smoke.velocity[cell_idx];
        max_speed = max_speed.max(f64::from(ux).hypot(f64::from(uy)));
        p_min = p_min.min(f64::from(smoke.pressure[cell_idx]));
        p_max = p_max.max(f64::from(smoke.pressure[cell_idx]));
        if let Some(rho) = &smoke.density {
            rho_min = rho_min.min(f64::from(rho[cell_idx]));
        }
    }
    let mut throat = (0.0_f64, 0usize);
    let mut exit = (0.0_f64, 0.0_f64, 0usize);
    for (cell_idx, &(x, _y)) in smoke.cell_centers.iter().enumerate() {
        if smoke.cell_solid[cell_idx] {
            continue;
        }
        let (ux, uy) = smoke.velocity[cell_idx];
        let speed = f64::from(ux).hypot(f64::from(uy));
        if (x - 1.2).abs() < 0.05 {
            throat.0 = throat.0.max(speed);
            throat.1 += 1;
        }
        if x > 2.9 {
            exit.0 = exit.0.max(speed);
            exit.1 += f64::from(smoke.pressure[cell_idx]);
            exit.2 += 1;
        }
    }
    println!(
        "nozzle-fine {model_id} cell={cell} p_in={p_in} dt={fixed_dt:?} {steps} steps t={:.4e} dt=[{:.3e},{:.3e}]: max|u|={max_speed:.1} throat_max|u|={:.1} exit_max|u|={:.1} exit_mean_p'={:.3e} p'=[{p_min:.3e},{p_max:.3e}] rho'_min={rho_min:.3e}",
        smoke.final_time,
        smoke.min_dt,
        smoke.max_dt,
        throat.0,
        exit.0,
        exit.1 / exit.2.max(1) as f64,
    );
}

fn run_nozzle_backend(model_id: &str, mesh_kind: &str, steps: usize, backend: &str) {
    let smoke = match gui_explicit_rk4_smoke(GuiExplicitRk4Case {
        model_id,
        fluid: "Air",
        geometry: "nozzle",
        mesh_kind,
        backend,
        adaptive: true,
        presentation: "plot",
        moving_mesh: false,
        cell_size: 0.025,
        steps,
        requested_dt: None,
        advection_scheme: None,
        inlet_velocity: None,
        inlet_pressure: None,
    }) {
        Ok(smoke) => smoke,
        Err(error) => {
            println!("nozzle {model_id} {steps} steps: FAILED: {error}");
            return;
        }
    };
    // Throat station x ~ 1.2 (throat_frac 0.4 * length 3).
    let mut max_speed = 0.0_f64;
    let mut throat_u = (0.0_f64, 0usize);
    let mut exit_u = (0.0_f64, 0usize);
    let (mut p_min, mut p_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for (cell, &(x, _y)) in smoke.cell_centers.iter().enumerate() {
        if smoke.cell_solid[cell] {
            continue;
        }
        let (ux, uy) = smoke.velocity[cell];
        let speed = f64::from(ux).hypot(f64::from(uy));
        max_speed = max_speed.max(speed);
        let p = f64::from(smoke.pressure[cell]);
        p_min = p_min.min(p);
        p_max = p_max.max(p);
        if (x - 1.2).abs() < 0.05 {
            throat_u.0 += f64::from(ux);
            throat_u.1 += 1;
        }
        if x > 2.9 {
            exit_u.0 += f64::from(ux);
            exit_u.1 += 1;
        }
    }
    println!(
        "nozzle {model_id}/{backend} {steps} steps t={:.4e} dt=[{:.3e},{:.3e}]: max|u|={max_speed:.1} \
         throat u_x={:.1} (n={}) exit u_x={:.1} (n={}) p'=[{p_min:.3e},{p_max:.3e}]",
        smoke.final_time,
        smoke.min_dt,
        smoke.max_dt,
        throat_u.0 / throat_u.1.max(1) as f64,
        throat_u.1,
        exit_u.0 / exit_u.1.max(1) as f64,
        exit_u.1,
    );
}

fn main() {
    if std::env::var_os("PROBE_NOZZLE").is_some() {
        for steps in [500usize, 3000, 10000] {
            run_nozzle("compressible", "fitted", steps);
            run_nozzle("compressible_structured", "structured", steps);
        }
        return;
    }
    if std::env::var_os("PROBE_MACH3_LOCATE").is_some() {
        // Sample just before the step-3541 failure and locate the cell with
        // the minimum internal energy / density.
        let smoke = gui_explicit_rk4_smoke(GuiExplicitRk4Case {
            model_id: "compressible",
            fluid: "Air",
            geometry: "obstacle",
            mesh_kind: "cutcell",
            backend: "gpu",
            adaptive: true,
            presentation: "plot",
            moving_mesh: false,
            cell_size: 0.005,
            steps: 3500,
            requested_dt: None,
            advection_scheme: None,
            inlet_velocity: Some(1000.0),
            inlet_pressure: None,
        })
        .expect("pre-failure state");
        let rho_field = smoke.density.as_ref().expect("rho");
        let mut worst = (f64::INFINITY, 0usize);
        for (cell, _) in smoke.cell_centers.iter().enumerate() {
            let rho = f64::from(rho_field[cell]);
            if rho < worst.0 {
                worst = (rho, cell);
            }
        }
        let (x, y) = smoke.cell_centers[worst.1];
        let d_cyl = (x - 1.0_f64).hypot(y - 0.51) - 0.1;
        println!(
            "mach3-locate t={:.4e}: min rho' {:.4e} at cell {} (x={x:.4}, y={y:.4}, wall_dist={d_cyl:.4})",
            smoke.final_time, worst.0, worst.1
        );
        let (ux, uy) = smoke.velocity[worst.1];
        println!(
            "  cell state: u=({ux:.1},{uy:.1}) p'={:.4e} T={:.1}",
            smoke.pressure[worst.1],
            smoke.temperature.as_ref().map_or(0.0, |t| t[worst.1]),
        );
        return;
    }
    if std::env::var_os("PROBE_MACH3").is_some() {
        // Stress test: obstacle, cell 0.005, inlet 1000 m/s (Mach ~2.9 bow
        // shock). GUI halts with 1 invalid cell at t ~ 1e-3.
        for steps in [1000usize, 3000, 6000] {
            let smoke = match gui_explicit_rk4_smoke(GuiExplicitRk4Case {
                model_id: "compressible",
                fluid: "Air",
                geometry: "obstacle",
                mesh_kind: "cutcell",
                backend: "gpu",
                adaptive: true,
                presentation: "plot",
                moving_mesh: false,
                cell_size: 0.005,
                steps,
                requested_dt: None,
                advection_scheme: None,
                inlet_velocity: Some(1000.0),
                inlet_pressure: None,
            }) {
                Ok(smoke) => smoke,
                Err(error) => {
                    println!("mach3 {steps} steps: FAILED: {error}");
                    continue;
                }
            };
            let (cx, cy, r) = (1.0, 0.51, 0.1);
            let mut wake = (f64::INFINITY, f64::INFINITY);
            let mut global = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY);
            let mut max_speed = 0.0_f64;
            for (cell, &(x, y)) in smoke.cell_centers.iter().enumerate() {
                let p_val = f64::from(smoke.pressure[cell]);
                let rho = smoke.density.as_ref().map_or(0.0, |r| f64::from(r[cell]));
                global.0 = global.0.min(p_val);
                global.1 = global.1.max(p_val);
                global.2 = global.2.min(rho);
                let (ux, uy) = smoke.velocity[cell];
                max_speed = max_speed.max(f64::from(ux).hypot(f64::from(uy)));
                let d = (x - cx).hypot(y - cy);
                if d < r + 0.1 && x > cx {
                    wake.0 = wake.0.min(p_val);
                    wake.1 = wake.1.min(rho);
                }
            }
            println!(
                "mach3 {steps} steps t={:.4e} dt=[{:.3e},{:.3e}]: max|u|={max_speed:.1} p'=[{:.3e},{:.3e}] rho'_min={:.3e} wake_p'_min={:.3e} wake_rho'_min={:.3e}",
                smoke.final_time, smoke.min_dt, smoke.max_dt,
                global.0, global.1, global.2, wake.0, wake.1
            );
        }
        return;
    }
    if std::env::var_os("PROBE_NOZZLE_REGIMES").is_some() {
        // Nozzle start / expansion regimes vs reservoir pressure (area ratio 2,
        // exit-Mach ~2.2 branch: p_exit/p0 ~ 0.094; shock-at-exit back-pressure
        // bound ~0.51 p0; underexpanded exit needs p0 > ~1.08 MPa abs).
        if std::env::var_os("PROBE_STRUCTURED_ONLY").is_some() {
            run_nozzle_fine_dt("compressible_structured", "structured", 20000, 2.5e5, 0.005, None);
            return;
        }
        for p_in in [1.0e5_f32, 2.5e5, 1.2e6] {
            run_nozzle_fine_dt("compressible", "fitted", 20000, p_in, 0.005, None);
        }
        run_nozzle_fine_dt("compressible_structured", "structured", 20000, 2.5e5, 0.005, None);
        return;
    }
    if std::env::var_os("PROBE_NOZZLE_LOCAL").is_some() {
        // Localize the t~2.07ms blow-up: sample the field just before failure
        // and report per-x-band velocity/pressure extrema.
        for steps in [200usize, 450, 650] {
            let smoke = gui_explicit_rk4_smoke(GuiExplicitRk4Case {
                model_id: "compressible",
                fluid: "Air",
                geometry: "nozzle",
                mesh_kind: "fitted",
                backend: "gpu",
                adaptive: false,
                presentation: "plot",
                moving_mesh: false,
                cell_size: 0.005,
                steps,
                requested_dt: Some(3.0e-6),
                advection_scheme: None,
                inlet_velocity: None,
                inlet_pressure: Some(1.0e5),
            })
            .expect("pre-failure sample");
            println!("t={:.4e} ({} steps):", smoke.final_time, steps);
            for band in 0..10 {
                let (x_lo, x_hi) = (band as f64 * 0.3, (band + 1) as f64 * 0.3);
                let mut u_max = 0.0_f64;
                let (mut p_lo, mut p_hi) = (f64::INFINITY, f64::NEG_INFINITY);
                for (cell, &(x, _y)) in smoke.cell_centers.iter().enumerate() {
                    if x < x_lo || x >= x_hi || smoke.cell_solid[cell] {
                        continue;
                    }
                    let (ux, uy) = smoke.velocity[cell];
                    u_max = u_max.max(f64::from(ux).hypot(f64::from(uy)));
                    let p = f64::from(smoke.pressure[cell]);
                    p_lo = p_lo.min(p);
                    p_hi = p_hi.max(p);
                }
                println!(
                    "  x=[{x_lo:.1},{x_hi:.1}): max|u|={u_max:8.1} p'=[{p_lo:11.3e},{p_hi:11.3e}]"
                );
            }
        }
        return;
    }
    if std::env::var_os("PROBE_NOZZLE_DT").is_some() {
        // Temporal-vs-spatial discriminator for the 1e5 Pa fine-mesh blow-up:
        // fixed dt sweep through the t~5 ms failure window.
        run_nozzle_fine_dt("compressible", "fitted", 2700, 1.0e5, 0.005, Some(3.0e-6));
        run_nozzle_fine_dt("compressible", "fitted", 5400, 1.0e5, 0.005, Some(1.5e-6));
        run_nozzle_fine_dt("compressible", "fitted", 10700, 1.0e5, 0.005, Some(7.5e-7));
        return;
    }
    if std::env::var_os("PROBE_NOZZLE_FINE").is_some() {
        // NOTE: the harness has no inlet-pressure override; COMPRESSIBLE_NOZZLE
        // defaults 5e4. Crash repro uses the env-tunable below via defaults? No —
        // run at the default drive first, then rely on the GUI-report config via
        // the CFD2 env override if needed.
        for steps in [2000usize, 8000, 20000] {
            run_nozzle_fine("compressible", "fitted", steps, 1.0e5, 0.005);
            run_nozzle_fine("compressible_structured", "structured", steps, 1.0e5, 0.005);
        }
        return;
    }
    if std::env::var_os("PROBE_NOZZLE_CPU").is_some() {
        run_nozzle_backend("compressible", "fitted", 300, "cpu-transpiled");
        run_nozzle_backend("compressible", "cutcell", 300, "cpu-transpiled");
        run_nozzle_backend("compressible", "fitted", 300, "gpu");
        return;
    }
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
