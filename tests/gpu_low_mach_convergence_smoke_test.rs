use cfd2::solver::gpu::GpuCompressibleSolver;
use cfd2::solver::mesh::geometry::ChannelWithObstacle;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Mesh};
use nalgebra::{Point2, Vector2};
use std::env;

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn neighbors(mesh: &Mesh, cell: usize) -> Vec<usize> {
    let start = mesh.cell_face_offsets[cell];
    let end = mesh.cell_face_offsets[cell + 1];
    let mut out = Vec::with_capacity(end - start);
    for idx in start..end {
        let face = mesh.cell_faces[idx];
        let owner = mesh.face_owner[face];
        let neigh = mesh.face_neighbor[face];
        let other = if owner == cell { neigh } else { Some(owner) };
        if let Some(n) = other {
            out.push(n);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn checkerboard_metric(mesh: &Mesh, values: &[f64], norm: f64) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for cell in 0..mesh.num_cells() {
        let neighs = neighbors(mesh, cell);
        if neighs.is_empty() {
            continue;
        }
        let mut avg = 0.0f64;
        for &n in &neighs {
            avg += values[n];
        }
        avg /= neighs.len() as f64;
        let diff = values[cell] - avg;
        sum += diff * diff;
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    let rms = (sum / count as f64).sqrt();
    rms / norm.max(1e-12)
}

fn build_mesh(cell: f64) -> Mesh {
    let length = 3.0;
    let height = 1.0;
    let domain_size = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    generate_cut_cell_mesh(&geo, cell, cell, 1.0, domain_size)
}

#[test]
#[ignore]
fn compressible_low_mach_convergence_smoke() {
    let steps = env_usize("CFD2_COMP_CONV_STEPS", 60);
    let dt = env_f64("CFD2_COMP_CONV_DT", 0.01);
    let dtau = env_f64("CFD2_COMP_CONV_DTAU", 5e-5);
    let alpha_u = env_f64("CFD2_COMP_CONV_ALPHA_U", 0.3);
    let precond_model = env_usize("CFD2_COMP_CONV_PRECOND_MODEL", 1) as u32;
    let precond_theta_floor = env_f64("CFD2_COMP_CONV_THETA_FLOOR", 1e-6);
    let comp_iters = env_usize("CFD2_COMP_CONV_ITERS", 8);
    let checker_max = env_f64("CFD2_COMP_CONV_CB_MAX", 5.0);
    let cell = env_f64("CFD2_COMP_CONV_CELL", 0.1);
    let u_in = env_f64("CFD2_COMP_CONV_UIN", 1.0) as f32;
    let density = 1.0f32;
    let nu = 0.001f32;
    let base_pressure = env_f64("CFD2_COMP_CONV_BASE_P", 25.0);

    let mesh = build_mesh(cell);

    let mut comp = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));
    comp.set_dt(dt as f32);
    comp.set_dtau(dtau as f32);
    comp.set_time_scheme(1);
    comp.set_density(density);
    comp.set_viscosity(nu);
    comp.set_inlet_velocity(u_in);
    comp.set_scheme(2);
    comp.set_alpha_u(alpha_u as f32);
    comp.set_precond_model(precond_model);
    comp.set_precond_theta_floor(precond_theta_floor as f32);
    comp.set_outer_iters(comp_iters);

    let rho_init = vec![density; mesh.num_cells()];
    let p_init = vec![base_pressure as f32; mesh.num_cells()];
    let u_init = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    comp.set_state_fields(&rho_init, &u_init, &p_init);
    comp.initialize_history();

    let mut saw_diverged = false;
    for _ in 0..steps {
        let stats = comp.step_with_stats();
        for stat in stats {
            if stat.diverged || !stat.residual.is_finite() {
                saw_diverged = true;
                break;
            }
        }
        if saw_diverged {
            break;
        }
    }

    let p_comp = pollster::block_on(comp.get_p());
    let dyn_pressure = 0.5 * density as f64 * (u_in as f64).powi(2);
    let checker_comp = checkerboard_metric(&mesh, &p_comp, dyn_pressure);

    assert!(!saw_diverged, "FGMRES diverged during the run");
    assert!(
        checker_comp < checker_max,
        "checkerboarding metric {:.3} exceeds {:.3}",
        checker_comp,
        checker_max
    );
}
