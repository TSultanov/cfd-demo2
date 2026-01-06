use cfd2::solver::gpu::GpuCompressibleSolver;
use cfd2::solver::mesh::geometry::ChannelWithObstacle;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Mesh};
use nalgebra::{Point2, Vector2};
use std::env;
use std::fs;
use std::path::PathBuf;

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

#[test]
#[ignore]
fn low_mach_debug_fast() {
    let steps = env_usize("CFD2_FAST_STEPS", 60);
    let dt = env_f64("CFD2_FAST_DT", 0.01);
    let dtau = env_f64("CFD2_FAST_DTAU", 5e-5);
    let alpha_u = env_f64("CFD2_FAST_ALPHA_U", 0.3);
    let nonconv_relax = env_f64("CFD2_FAST_NONCONV_RELAX", 0.5);
    let precond_model = env_usize("CFD2_FAST_PRECOND_MODEL", 1) as u32;
    let precond_theta_floor = env_f64("CFD2_FAST_PRECOND_THETA_FLOOR", 1e-6);
    let pc_alpha = env_f64("CFD2_FAST_PC_ALPHA", 0.0);
    let comp_iters = env_usize("CFD2_FAST_COMP_ITERS", 8);
    let cell = env_f64("CFD2_FAST_CELL", 0.1);
    let u_in = env_f64("CFD2_FAST_UIN", 1.0) as f32;
    let density = 1.0f32;
    let nu = 0.001f32;
    let base_pressure = env_f64("CFD2_FAST_BASE_P", 25.0);
    let conv_frac = env_f64("CFD2_FAST_CONV_FRAC", 0.7);
    let checker_max = env_f64("CFD2_FAST_CB_MAX", 2.0);
    let std_min = env_f64("CFD2_FAST_STD_MIN", 1e-5);

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
    comp.set_pressure_coupling_alpha(pc_alpha as f32);
    comp.set_nonconverged_relax(nonconv_relax as f32);
    comp.set_outer_iters(comp_iters);

    let rho_init = vec![density; mesh.num_cells()];
    let p_init = vec![base_pressure as f32; mesh.num_cells()];
    let u_init = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    comp.set_state_fields(&rho_init, &u_init, &p_init);
    comp.initialize_history();

    let mut conv_hits = 0usize;
    let mut conv_total = 0usize;
    let mut probe_history = Vec::new();
    let probe_idx = mesh
        .cell_cx
        .iter()
        .zip(mesh.cell_cy.iter())
        .enumerate()
        .min_by(|a, b| {
            let da = (a.1 .0 - 1.6).powi(2) + (a.1 .1 - 0.5).powi(2);
            let db = (b.1 .0 - 1.6).powi(2) + (b.1 .1 - 0.5).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    for _ in 0..steps {
        let stats = comp.step_with_stats();
        for stat in &stats {
            conv_total += 1;
            if stat.converged {
                conv_hits += 1;
            }
        }
        let u = pollster::block_on(comp.get_u());
        probe_history.push(u[probe_idx].1);
    }

    let p_comp = pollster::block_on(comp.get_p());
    let dyn_pressure = 0.5 * density as f64 * (u_in as f64).powi(2);
    let checker_comp = checkerboard_metric(&mesh, &p_comp, dyn_pressure);
    let mean = probe_history.iter().sum::<f64>() / probe_history.len() as f64;
    let std = (probe_history
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / probe_history.len() as f64)
        .sqrt();

    let conv_frac_val = if conv_total > 0 {
        conv_hits as f64 / conv_total as f64
    } else {
        0.0
    };

    let mut summary = String::new();
    summary.push_str(&format!(
        "conv_frac={:.3}\nchecker_comp={:.3}\nprobe_std={:.3e}\n",
        conv_frac_val, checker_comp, std
    ));
    let out_dir = PathBuf::from("target/test_plots/low_mach");
    let _ = fs::create_dir_all(&out_dir);
    let _ = fs::write(out_dir.join("low_mach_fast_debug.txt"), summary);

    assert!(
        conv_frac_val >= conv_frac,
        "converged fraction {:.3} below {:.3}",
        conv_frac_val,
        conv_frac
    );
    assert!(
        checker_comp < checker_max,
        "checkerboarding metric {:.3} exceeds {:.3}",
        checker_comp,
        checker_max
    );
    assert!(std > std_min, "probe std {:.3e} below {:.3e}", std, std_min);
}
