use cfd2::solver::mesh::geometry::ChannelWithObstacle;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Mesh};
use cfd2::solver::model::compressible_model;
use cfd2::solver::options::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};
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
    let precond_model = match env_usize("CFD2_COMP_CONV_PRECOND_MODEL", 1) as u32 {
        0 => GpuLowMachPrecondModel::Legacy,
        1 => GpuLowMachPrecondModel::WeissSmith,
        _ => GpuLowMachPrecondModel::Off,
    };
    let precond_theta_floor = env_f64("CFD2_COMP_CONV_THETA_FLOOR", 1e-6);
    let comp_iters = env_usize("CFD2_COMP_CONV_ITERS", 8);
    let nonconv_relax = env_f64("CFD2_COMP_CONV_NONCONV_RELAX", 0.5);
    let checker_max = env_f64("CFD2_COMP_CONV_CB_MAX", 5.0);
    let resid_drop = env_f64("CFD2_COMP_CONV_RESID_DROP", 0.9);
    let resid_drop_frac = env_f64("CFD2_COMP_CONV_RESID_DROP_FRAC", 0.0);
    let converged_frac = env_f64("CFD2_COMP_CONV_CONVERGED_FRAC", 0.7);
    let log_stats = env::var("CFD2_COMP_CONV_LOG").ok().as_deref() == Some("1");
    let cell = env_f64("CFD2_COMP_CONV_CELL", 0.1);
    let u_in = env_f64("CFD2_COMP_CONV_UIN", 1.0) as f32;
    let density = 1.0f32;
    let nu = 0.001f32;
    let base_pressure = env_f64("CFD2_COMP_CONV_BASE_P", 25.0);

    let mesh = build_mesh(cell);

    let mut comp = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model(),
        SolverConfig {
            advection_scheme: Scheme::QUICK,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
        },
        None,
        None,
    ))
    .expect("solver init");
    comp.set_dt(dt as f32);
    comp.set_dtau(dtau as f32);
    comp.set_viscosity(nu);
    comp.set_inlet_velocity(u_in);
    comp.set_alpha_u(alpha_u as f32);
    comp.set_precond_model(precond_model)
        .expect("precond model");
    comp.set_precond_theta_floor(precond_theta_floor as f32)
        .expect("theta floor");
    comp.set_nonconverged_relax(nonconv_relax as f32)
        .expect("nonconverged relax");
    comp.set_outer_iters(comp_iters);

    let rho_init = vec![density; mesh.num_cells()];
    let p_init = vec![base_pressure as f32; mesh.num_cells()];
    let u_init = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    comp.set_state_fields(&rho_init, &u_init, &p_init);
    comp.initialize_history();

    let mut saw_diverged = false;
    let mut drop_hits = 0usize;
    let mut drop_total = 0usize;
    let mut converged_hits = 0usize;
    let mut converged_total = 0usize;
    let mut min_residual = f32::INFINITY;
    let mut max_residual = 0.0f32;
    for _ in 0..steps {
        let stats = comp.step_with_stats().expect("step stats");
        for stat in &stats {
            min_residual = min_residual.min(stat.residual);
            max_residual = max_residual.max(stat.residual);
            if stat.diverged || !stat.residual.is_finite() {
                saw_diverged = true;
                break;
            }
            converged_total += 1;
            if stat.converged {
                converged_hits += 1;
            }
        }
        if let (Some(first), Some(last)) = (stats.first(), stats.last()) {
            if first.residual.is_finite() && last.residual.is_finite() && first.residual > 0.0 {
                drop_total += 1;
                if last.residual <= first.residual * resid_drop as f32 {
                    drop_hits += 1;
                }
                if log_stats {
                    eprintln!(
                        "step residual: start={:.3e} end={:.3e} iters={}",
                        first.residual,
                        last.residual,
                        stats.len()
                    );
                }
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
    if drop_total > 0 && resid_drop_frac > 0.0 {
        let drop_frac = drop_hits as f64 / drop_total as f64;
        assert!(
            drop_frac >= resid_drop_frac,
            "residual drop fraction {:.3} below {:.3} (min={:.3e}, max={:.3e})",
            drop_frac,
            resid_drop_frac,
            min_residual,
            max_residual
        );
    }
    if converged_total > 0 {
        let conv_frac = converged_hits as f64 / converged_total as f64;
        assert!(
            conv_frac >= converged_frac,
            "converged fraction {:.3} below {:.3}",
            conv_frac,
            converged_frac
        );
    }
    assert!(
        checker_comp < checker_max,
        "checkerboarding metric {:.3} exceeds {:.3}",
        checker_comp,
        checker_max
    );
}
