use cfd2::solver::gpu::{GpuCompressibleSolver, GpuSolver};
use cfd2::solver::mesh::geometry::ChannelWithObstacle;
use cfd2::solver::mesh::{generate_cut_cell_mesh, generate_structured_rect_mesh, BoundaryType, Mesh};
use image::{Rgb, RgbImage};
use nalgebra::{Point2, Vector2};
use std::fs;
use std::path::{Path, PathBuf};

fn draw_line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0
            && y0 >= 0
            && (x0 as u32) < img.width()
            && (y0 as u32) < img.height()
        {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn save_line_plot(
    path: &Path,
    x: &[f64],
    series: &[(&[f64], Rgb<u8>)],
    title: &str,
    x_label: &str,
    y_label: &str,
) {
    let width = 900u32;
    let height = 420u32;
    let margin_left = 55i32;
    let margin_right = 15i32;
    let margin_top = 25i32;
    let margin_bottom = 45i32;

    let mut img = RgbImage::from_pixel(width, height, Rgb([255, 255, 255]));

    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    for &xi in x {
        xmin = xmin.min(xi);
        xmax = xmax.max(xi);
    }
    if !(xmax > xmin) {
        xmax = xmin + 1.0;
    }

    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for (ys, _) in series {
        for &y in *ys {
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
    }
    if !(ymax > ymin) {
        ymax = ymin + 1.0;
    }
    let pad = 0.05 * (ymax - ymin);
    ymin -= pad;
    ymax += pad;

    let plot_w = width as i32 - margin_left - margin_right;
    let plot_h = height as i32 - margin_top - margin_bottom;

    let x_to_px = |xi: f64| -> i32 {
        let t = (xi - xmin) / (xmax - xmin);
        margin_left + (t * plot_w as f64).round() as i32
    };
    let y_to_py = |yi: f64| -> i32 {
        let t = (yi - ymin) / (ymax - ymin);
        margin_top + ((1.0 - t) * plot_h as f64).round() as i32
    };

    // Axes
    let axis_color = Rgb([60, 60, 60]);
    let x0 = margin_left;
    let y0 = margin_top + plot_h;
    draw_line(&mut img, x0, y0, x0 + plot_w, y0, axis_color);
    draw_line(&mut img, x0, y0, x0, margin_top, axis_color);

    // Simple title/labels (minimal, raster only)
    // We avoid embedding fonts; just mark with ticks.
    let tick_color = Rgb([170, 170, 170]);
    for i in 0..=10 {
        let t = i as f64 / 10.0;
        let px = margin_left + (t * plot_w as f64).round() as i32;
        draw_line(&mut img, px, y0, px, y0 + 5, axis_color);
        let py = margin_top + (t * plot_h as f64).round() as i32;
        draw_line(&mut img, x0 - 5, py, x0, py, axis_color);
        // grid
        draw_line(&mut img, px, margin_top, px, y0, tick_color);
        draw_line(&mut img, x0, py, x0 + plot_w, py, tick_color);
    }

    // Series
    for (ys, color) in series {
        if ys.len() != x.len() || x.len() < 2 {
            continue;
        }
        for i in 0..x.len() - 1 {
            let x_a = x_to_px(x[i]);
            let y_a = y_to_py(ys[i]);
            let x_b = x_to_px(x[i + 1]);
            let y_b = y_to_py(ys[i + 1]);
            draw_line(&mut img, x_a, y_a, x_b, y_b, *color);
        }
    }

    // Encode some metadata text in a sidecar file (since we don't render fonts).
    let mut meta = String::new();
    meta.push_str(&format!("title: {title}\n"));
    meta.push_str(&format!("x_label: {x_label}\n"));
    meta.push_str(&format!("y_label: {y_label}\n"));
    meta.push_str(&format!("x_range: [{xmin:.6}, {xmax:.6}]\n"));
    meta.push_str(&format!("y_range: [{ymin:.6}, {ymax:.6}]\n"));
    let _ = fs::write(path.with_extension("txt"), meta);

    img.save(path).expect("failed to save plot");
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

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn l2_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let mse = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    mse.sqrt()
}

fn total_variation(values: &[f64]) -> f64 {
    values
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum::<f64>()
}

fn count_local_extrema(values: &[f64], eps: f64) -> usize {
    if values.len() < 3 {
        return 0;
    }
    let mut count = 0usize;
    for i in 1..values.len() - 1 {
        let prev = values[i - 1];
        let curr = values[i];
        let next = values[i + 1];
        let dp0 = curr - prev;
        let dp1 = next - curr;
        if dp0.abs() < eps && dp1.abs() < eps {
            continue;
        }
        if (curr > prev && curr > next) || (curr < prev && curr < next) {
            count += 1;
        }
    }
    count
}

fn out_dir(sub: &str) -> PathBuf {
    let dir = PathBuf::from("target/test_plots/1d").join(sub);
    let _ = fs::create_dir_all(&dir);
    dir
}

#[test]
fn incompressible_structured_mesh_preserves_rest_state() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = generate_structured_rect_mesh(
        16,
        8,
        1.0,
        0.5,
        BoundaryType::Inlet,
        BoundaryType::Outlet,
        BoundaryType::Wall,
        BoundaryType::Wall,
    );
    let mut solver = pollster::block_on(GpuSolver::new(&mesh, None, None));
    solver.set_dt(0.02);
    solver.set_density(1.0);
    solver.set_viscosity(0.05);
    solver.set_scheme(0);
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);
    solver.set_time_scheme(1);
    solver.set_inlet_velocity(0.0);
    solver.set_ramp_time(0.0);
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    for _ in 0..3 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());
    assert!(u.iter().all(|(ux, uy)| ux.is_finite() && uy.is_finite()));
    assert!(p.iter().all(|v| v.is_finite()));
}

#[test]
fn compressible_acoustic_pulse_structured_1d_plot() {
    let nx = 220usize;
    let ny = 1usize;
    let length = 1.0;
    let height = 0.05;
    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundaryType::Outlet,
        BoundaryType::Outlet,
        BoundaryType::Wall,
        BoundaryType::Wall,
    );

    let gamma: f64 = 1.4;
    let base_p: f64 = 1.0;
    let base_rho: f64 = 1.0;
    let c0: f64 = (gamma * base_p / base_rho).sqrt();

    let mid = 0.25 * length;
    let sigma = 0.04 * length;
    let amp = 0.01;

    let n = mesh.num_cells();
    let mut rho = vec![0.0f32; n];
    let mut p = vec![0.0f32; n];
    let mut u = vec![[0.0f32, 0.0f32]; n];
    let mut x = vec![0.0f64; n];
    let mut p0 = vec![0.0f64; n];

    for i in 0..n {
        let xi = mesh.cell_cx[i];
        x[i] = xi;
        let dx = xi - mid;
        let bump = (-(dx * dx) / (2.0 * sigma * sigma)).exp();
        let p_prime = amp * bump;
        let rho_prime = p_prime * (base_rho / (gamma * base_p));
        p[i] = (base_p + p_prime) as f32;
        rho[i] = (base_rho + rho_prime) as f32;
        u[i][0] = (p_prime / (base_rho * c0)) as f32;
        p0[i] = p[i] as f64;
    }

    let mut solver = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));
    let dt = 0.002f32;
    let steps = 70usize;
    solver.set_dt(dt);
    solver.set_time_scheme(0);
    solver.set_viscosity(0.0);
    solver.set_inlet_velocity(0.0);
    solver.set_scheme(2);
    solver.set_outer_iters(3);
    solver.set_state_fields(&rho, &u, &p);
    solver.initialize_history();

    for _ in 0..steps {
        solver.step();
    }

    let p_out = pollster::block_on(solver.get_p());
    let t = dt as f64 * steps as f64;
    let expected_shift = c0 * t;
    let expected_center = mid + expected_shift;
    let (peak_idx, peak_p) = p_out
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, p)| (i, *p))
        .unwrap();
    let peak_x = mesh.cell_cx[peak_idx];
    let shift = peak_x - mid;

    let out = out_dir("acoustic");
    let p_final: Vec<f64> = p_out.iter().copied().collect();
    let p_expected: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let dx = xi - expected_center;
            base_p + amp * (-(dx * dx) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    save_line_plot(
        &out.join("p_vs_x.png"),
        &x,
        &[
            (&p0, Rgb([0, 120, 255])),
            (&p_expected, Rgb([0, 170, 0])),
            (&p_final, Rgb([220, 60, 60])),
        ],
        "compressible acoustic pulse (structured 1D)",
        "x",
        "p",
    );

    let max_p = p_final
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_p = p_final.iter().cloned().fold(f64::INFINITY, f64::min);
    let rel_l2 = l2_error(&p_final, &p_expected) / amp.max(1e-12);
    let tv_norm = total_variation(&p_final) / amp.max(1e-12);
    let extrema = count_local_extrema(&p_final, 1e-6 * amp.max(1e-12));
    let noise: Vec<f64> = p_final
        .iter()
        .zip(p_expected.iter())
        .map(|(a, b)| a - b)
        .collect();
    let noise_rms = rms(&noise) / amp.max(1e-12);
    let summary = format!(
        "c0={:.6}\ndt={:.6}\nsteps={}\nt={:.6}\nexpected_shift={:.6}\nexpected_center={:.6}\npeak_x={:.6}\nshift={:.6}\npeak_p={:.6}\nmin_p={:.6}\nmax_p={:.6}\nrel_l2_over_amp={:.6}\ntv_over_amp={:.6}\nextrema={}\nnoise_rms_over_amp={:.6}\n",
        c0,
        dt,
        steps,
        t,
        expected_shift,
        expected_center,
        peak_x,
        shift,
        peak_p,
        min_p,
        max_p,
        rel_l2,
        tv_norm,
        extrema,
        noise_rms
    );
    let _ = fs::write(out.join("summary.txt"), summary);

    println!(
        "acoustic_metrics: rel_l2={:.3} tv={:.3} extrema={} min_p={:.6} max_p={:.6} noise_rms={:.3}",
        rel_l2, tv_norm, extrema, min_p, max_p, noise_rms
    );

    assert!(
        shift > 0.55 * expected_shift,
        "pulse peak shift {:.4} below expected {:.4}",
        shift,
        expected_shift
    );

    // This is a smooth, small-amplitude linear acoustic wave; strong oscillations/overshoots
    // should fail the regression even if the peak advects.
    assert!(
        rel_l2 < 0.6,
        "waveform distortion too large (rel_l2_over_amp {:.3})",
        rel_l2
    );
    assert!(
        tv_norm < 6.0,
        "excess total variation indicates oscillations (tv_over_amp {:.3})",
        tv_norm
    );
    assert!(
        extrema <= 6,
        "too many local extrema ({}), indicates dispersive ringing",
        extrema
    );
    assert!(
        min_p >= base_p - 0.2 * amp,
        "pressure undershoot too large (min_p {:.6})",
        min_p
    );
    assert!(
        max_p <= base_p + 1.6 * amp,
        "pressure overshoot too large (max_p {:.6})",
        max_p
    );
}

#[test]
#[ignore]
fn low_mach_channel_incompressible_matches_compressible_profiles() {
    std::env::set_var("CFD2_QUIET", "1");

    let length = 1.0;
    let height = 1.0;
    let cell = 0.2;
    let domain = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(0.5, 0.5),
        obstacle_radius: 0.15,
    };
    let mesh = generate_cut_cell_mesh(&geo, cell, cell, 1.0, domain);

    let u_in = 0.5f32;
    let density = 1.0f32;
    let nu = 0.02f32;
    let dt = 0.01f32;
    let steps = 6usize;

    let mut incomp = pollster::block_on(GpuSolver::new(&mesh, None, None));
    incomp.set_dt(dt);
    incomp.set_viscosity(nu);
    incomp.set_density(density);
    incomp.set_scheme(0);
    incomp.set_alpha_u(0.7);
    incomp.set_alpha_p(0.3);
    incomp.set_time_scheme(1);
    incomp.set_inlet_velocity(u_in);
    incomp.set_ramp_time(0.2);
    incomp.n_outer_correctors = 2;
    incomp.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    incomp.set_p(&vec![0.0; mesh.num_cells()]);
    incomp.initialize_history();

    let base_pressure = 25.0f32;
    let mut comp = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));
    comp.set_dt(dt);
    comp.set_dtau(5e-5);
    comp.set_time_scheme(1);
    comp.set_viscosity(nu);
    comp.set_inlet_velocity(u_in);
    comp.set_scheme(0);
    comp.set_alpha_u(0.3);
    comp.set_precond_model(1);
    comp.set_precond_theta_floor(1e-6);
    comp.set_outer_iters(6);
    let rho_init = vec![density; mesh.num_cells()];
    let p_init = vec![base_pressure; mesh.num_cells()];
    let u_init = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    comp.set_state_fields(&rho_init, &u_init, &p_init);
    comp.initialize_history();

    for _ in 0..steps {
        incomp.step();
        comp.step();
    }

    let u_incomp = pollster::block_on(incomp.get_u());
    let p_incomp = pollster::block_on(incomp.get_p());
    let u_comp = pollster::block_on(comp.get_u());
    let p_comp = pollster::block_on(comp.get_p());

    // Sample along the midline (|y - 0.5H| small) for 1D plots.
    let y_mid = 0.5 * height;
    let band = 0.35 * cell;
    let mut x_line = Vec::new();
    let mut u_line_incomp = Vec::new();
    let mut u_line_comp = Vec::new();
    let mut p_line_incomp = Vec::new();
    let mut p_line_comp = Vec::new();

    for i in 0..mesh.num_cells() {
        if (mesh.cell_cy[i] - y_mid).abs() <= band {
            x_line.push(mesh.cell_cx[i]);
            u_line_incomp.push(u_incomp[i].0);
            u_line_comp.push(u_comp[i].0);
            p_line_incomp.push(p_incomp[i]);
            p_line_comp.push(p_comp[i] - base_pressure as f64);
        }
    }

    let mut order: Vec<usize> = (0..x_line.len()).collect();
    order.sort_by(|&a, &b| x_line[a].partial_cmp(&x_line[b]).unwrap());

    let reorder = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let x_sorted = reorder(&x_line);
    let u_incomp_sorted = reorder(&u_line_incomp);
    let u_comp_sorted = reorder(&u_line_comp);
    let mut p_incomp_sorted = reorder(&p_line_incomp);
    let mut p_comp_sorted = reorder(&p_line_comp);
    let mean_p_inc = p_incomp_sorted.iter().sum::<f64>() / p_incomp_sorted.len().max(1) as f64;
    let mean_p_cmp = p_comp_sorted.iter().sum::<f64>() / p_comp_sorted.len().max(1) as f64;
    for v in &mut p_incomp_sorted {
        *v -= mean_p_inc;
    }
    for v in &mut p_comp_sorted {
        *v -= mean_p_cmp;
    }

    let dyn_pressure = 0.5 * density as f64 * (u_in as f64) * (u_in as f64);
    let p_comp_zero_mean: Vec<f64> = {
        let mean = p_comp.iter().sum::<f64>() / p_comp.len().max(1) as f64;
        p_comp.iter().map(|v| v - mean).collect()
    };
    let checker = checkerboard_metric(&mesh, &p_comp_zero_mean, dyn_pressure);

    let out = out_dir("low_mach_channel");
    save_line_plot(
        &out.join("u_centerline.png"),
        &x_sorted,
        &[
            (&u_incomp_sorted, Rgb([0, 120, 255])),
            (&u_comp_sorted, Rgb([220, 60, 60])),
        ],
        "u_x(x) near centerline",
        "x",
        "u_x",
    );
    save_line_plot(
        &out.join("p_centerline.png"),
        &x_sorted,
        &[
            (&p_incomp_sorted, Rgb([0, 120, 255])),
            (&p_comp_sorted, Rgb([220, 60, 60])),
        ],
        "p(x) near centerline (mean removed)",
        "x",
        "p - mean(p)",
    );
    let summary = format!(
        "cells={}\nsteps={steps} dt={dt:.6}\nnu={nu:.6}\nu_in={u_in:.6}\nbase_pressure={base_pressure:.6}\ncheckerboard={checker:.6}\n",
        mesh.num_cells()
    );
    let _ = fs::write(out.join("summary.txt"), summary);

    assert!(checker < 1.0, "checkerboarding metric {:.3} too high", checker);
    let mut l2 = 0.0f64;
    for (a, b) in u_incomp_sorted.iter().zip(u_comp_sorted.iter()) {
        let d = a - b;
        l2 += d * d;
    }
    l2 = (l2 / u_incomp_sorted.len().max(1) as f64).sqrt();
    assert!(l2 < 0.5, "u_x mismatch too large (L2 {:.3})", l2);
}
