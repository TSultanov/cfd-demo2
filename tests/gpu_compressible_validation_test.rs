use cfd2::solver::gpu::GpuCompressibleSolver;
use cfd2::solver::mesh::geometry::RectangularChannel;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Mesh};
use nalgebra::Vector2;
use std::env;

fn build_channel_mesh(cell: f64) -> Mesh {
    let length = 1.0;
    let height = 0.2;
    let domain_size = Vector2::new(length, height);
    let geo = RectangularChannel { length, height };
    generate_cut_cell_mesh(&geo, cell, cell, 1.0, domain_size)
}

#[test]
fn compressible_shock_tube_relaxes_discontinuity() {
    let mesh = build_channel_mesh(0.05);
    let n = mesh.num_cells();
    let mid = 0.5;
    let gamma = 1.4;

    let mut rho = vec![0.0f32; n];
    let mut p = vec![0.0f32; n];
    let u = vec![[0.0f32, 0.0f32]; n];

    for i in 0..n {
        if mesh.cell_cx[i] < mid {
            rho[i] = 1.0;
            p[i] = 1.0;
        } else {
            rho[i] = 0.125;
            p[i] = 0.1;
        }
    }

    let mut initial_mass = 0.0;
    let mut initial_energy = 0.0;
    for i in 0..n {
        let vol = mesh.cell_vol[i];
        initial_mass += rho[i] as f64 * vol;
        initial_energy += (p[i] as f64 / (gamma - 1.0)) * vol;
    }

    let mut solver = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));
    solver.set_outer_iters(3);
    solver.set_dt(0.002);
    solver.set_time_scheme(0);
    solver.set_viscosity(0.0);
    solver.set_inlet_velocity(0.0);
    solver.set_state_fields(&rho, &u, &p);
    solver.initialize_history();

    for _ in 0..20 {
        solver.step();
    }

    let rho_out = pollster::block_on(solver.get_rho());
    let u_out = pollster::block_on(solver.get_u());
    let p_out = pollster::block_on(solver.get_p());
    let rho_min = 0.125;
    let rho_max = 1.0;
    let mut mixed = false;
    let mut final_mass = 0.0;
    let mut final_energy = 0.0;

    for (i, &val) in rho_out.iter().enumerate() {
        assert!(val.is_finite());
        assert!(val > 0.0);
        if (mesh.cell_cx[i] - mid).abs() < 0.15 && val > rho_min + 0.02 && val < rho_max - 0.02 {
            mixed = true;
        }
        let vol = mesh.cell_vol[i];
        final_mass += val * vol;
        let p_val = p_out[i];
        let u_val = u_out[i];
        let ke = 0.5 * val * (u_val.0 * u_val.0 + u_val.1 * u_val.1);
        final_energy += (p_val / (gamma - 1.0) + ke) * vol;
    }

    assert!(mixed, "shock tube did not develop mixed states near interface");
    let mass_rel = (final_mass - initial_mass).abs() / initial_mass.max(1e-9);
    let energy_rel = (final_energy - initial_energy).abs() / initial_energy.max(1e-9);
    assert!(mass_rel < 0.02, "mass drift {:.3}", mass_rel);
    assert!(energy_rel < 0.05, "energy drift {:.3}", energy_rel);
}

#[test]
fn compressible_acoustic_pulse_propagates() {
    let mesh = build_channel_mesh(0.05);
    let n = mesh.num_cells();
    let mid = 0.5;
    let sigma = 0.08;
    let gamma: f64 = 1.4;
    let base_p: f64 = 1.0;
    let base_rho: f64 = 1.0;

    let mut rho = vec![0.0f32; n];
    let mut p = vec![0.0f32; n];
    let mut u = vec![[0.0f32, 0.0f32]; n];
    let amp = 0.1;
    let c0: f64 = (gamma * base_p / base_rho).sqrt();

    for i in 0..n {
        let dx = mesh.cell_cx[i] - mid;
        let bump = (-(dx * dx) / (2.0 * sigma * sigma)).exp() as f32;
        let p_perturb = amp * bump;
        let rho_perturb = p_perturb * (base_rho / (gamma * base_p)) as f32;
        p[i] = base_p as f32 + p_perturb;
        rho[i] = base_rho as f32 + rho_perturb;
        u[i][0] = (p_perturb as f64 / (base_rho * c0)) as f32;
    }

    let p_initial = p.clone();
    let mut solver = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));
    solver.set_outer_iters(3);
    solver.set_dt(0.005);
    solver.set_time_scheme(0);
    solver.set_viscosity(0.0);
    solver.set_inlet_velocity(0.0);
    solver.set_state_fields(&rho, &u, &p);
    solver.initialize_history();

    for _ in 0..40 {
        solver.step();
    }

    let p_out = pollster::block_on(solver.get_p());
    let u_out = pollster::block_on(solver.get_u());
    let mut mean_abs_diff = 0.0f64;
    for (after, before) in p_out.iter().zip(p_initial.iter()) {
        assert!(after.is_finite());
        assert!(*after > 0.0);
        mean_abs_diff += (after - *before as f64).abs();
    }
    mean_abs_diff /= p_out.len().max(1) as f64;

    assert!(mean_abs_diff > 5e-4, "acoustic pulse did not evolve");

    let t = solver.constants.dt as f64 * 40.0;
    let expected = c0 * t;
    let mut weight = 0.0;
    let mut weighted_x = 0.0;
    for (i, p_val) in p_out.iter().enumerate() {
        let dp = (p_val - base_p).max(0.0);
        weight += dp;
        weighted_x += dp * mesh.cell_cx[i];
    }
    let mean_x = if weight > 0.0 {
        weighted_x / weight
    } else {
        mid
    };
    let (peak_idx, peak_dp) = p_out
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, val)| (idx, *val - base_p))
        .unwrap_or((0, 0.0));
    let peak_x = mesh.cell_cx[peak_idx];
    let peak_shift = peak_x - mid;
    if env::var("CFD2_DEBUG_ACOUSTIC").ok().as_deref() == Some("1") {
        let mid_idx = mesh
            .cell_cx
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - mid)
                    .abs()
                    .partial_cmp(&(*b - mid).abs())
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        println!(
            "acoustic_debug: mean_x={:.3}, peak_x={:.3}, expected={:.3}, mid_delta={:.3e}",
            mean_x,
            peak_x,
            expected,
            (p_out[mid_idx] - base_p).abs()
        );
        println!("peak dp={:.3e}", peak_dp);
    }
    assert!(
        peak_shift > 0.6 * expected,
        "pressure pulse did not advect far enough (peak shift {:.3}, expected {:.3})",
        peak_shift,
        expected
    );

    let mut max_u: f64 = 0.0;
    for (ux, uy) in u_out {
        max_u = max_u.max((ux * ux + uy * uy).sqrt());
    }
    let max_p = p_out
        .iter()
        .map(|val| (val - base_p).abs())
        .fold(0.0, f64::max);
    if max_p > 0.0 {
        let impedance = max_p / (base_rho * c0);
        assert!(
            max_u > 0.3 * impedance,
            "velocity response too weak relative to acoustic estimate"
        );
    }
}
