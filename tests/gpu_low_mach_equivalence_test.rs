use cfd2::solver::gpu::{GpuCompressibleSolver, GpuSolver};
use cfd2::solver::mesh::geometry::ChannelWithObstacle;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Mesh};
use image::{Rgb, RgbImage};
use nalgebra::{Point2, Vector2};
use std::env;
use std::fs;
use std::path::PathBuf;

fn build_mesh(cell: f64, smooth_alpha: f64, smooth_iters: usize) -> Mesh {
    let length = 3.0;
    let height = 1.0;
    let domain_size = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, cell, cell, 1.0, domain_size);
    if smooth_iters > 0 {
        mesh.smooth(&geo, smooth_alpha, smooth_iters);
    }
    mesh
}

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

fn env_bool(name: &str) -> bool {
    env::var(name).map(|v| v == "1").unwrap_or(false)
}

fn find_probe_index(mesh: &Mesh, x: f64, y: f64) -> usize {
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for i in 0..mesh.num_cells() {
        let dx = mesh.cell_cx[i] - x;
        let dy = mesh.cell_cy[i] - y;
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    best
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

fn gradients(mesh: &Mesh, values: &[f64]) -> Vec<(f64, f64)> {
    let mut grads = vec![(0.0, 0.0); mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        let cx = mesh.cell_cx[i];
        let cy = mesh.cell_cy[i];
        let mut a11 = 0.0;
        let mut a12 = 0.0;
        let mut a22 = 0.0;
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        for n in neighbors(mesh, i) {
            let dx = mesh.cell_cx[n] - cx;
            let dy = mesh.cell_cy[n] - cy;
            let dv = values[n] - values[i];
            a11 += dx * dx;
            a12 += dx * dy;
            a22 += dy * dy;
            b1 += dx * dv;
            b2 += dy * dv;
        }
        let det = a11 * a22 - a12 * a12;
        if det.abs() > 1e-12 {
            let gx = (a22 * b1 - a12 * b2) / det;
            let gy = (-a12 * b1 + a11 * b2) / det;
            grads[i] = (gx, gy);
        }
    }
    grads
}

fn vorticity(mesh: &Mesh, u: &[(f64, f64)]) -> Vec<f64> {
    let u_x: Vec<f64> = u.iter().map(|val| val.0).collect();
    let u_y: Vec<f64> = u.iter().map(|val| val.1).collect();
    let grad_x = gradients(mesh, &u_x);
    let grad_y = gradients(mesh, &u_y);
    grad_x
        .iter()
        .zip(grad_y.iter())
        .map(|(gx, gy)| gy.0 - gx.1)
        .collect()
}

fn mesh_bounds(mesh: &Mesh) -> (f64, f64, f64, f64) {
    let min_x = mesh.vx.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = mesh.vx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = mesh.vy.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = mesh.vy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min_x, max_x, min_y, max_y)
}

fn cell_polygon(mesh: &Mesh, cell: usize) -> Vec<[f64; 2]> {
    let start = mesh.cell_vertex_offsets[cell];
    let end = mesh.cell_vertex_offsets[cell + 1];
    let mut points = Vec::with_capacity(end - start);
    for idx in start..end {
        let v = mesh.cell_vertices[idx];
        points.push([mesh.vx[v], mesh.vy[v]]);
    }
    points
}

fn point_in_poly(px: f64, py: f64, poly: &[[f64; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let xi = poly[i][0];
        let yi = poly[i][1];
        let xj = poly[j][0];
        let yj = poly[j][1];
        let crosses = (yi > py) != (yj > py)
            && px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi;
        if crosses {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn colormap_sequential(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.0, s, 1.0 - s)
    } else {
        let s = (t - 0.5) * 2.0;
        (s, 1.0 - s, 0.0)
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

fn save_scalar_image(
    path: &PathBuf,
    mesh: &Mesh,
    values: &[f64],
    width: usize,
    height: usize,
) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &val in values {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    let (min_x, max_x, min_y, max_y) = mesh_bounds(mesh);
    let scale_x = (width as f64 - 1.0) / (max_x - min_x).max(1e-12);
    let scale_y = (height as f64 - 1.0) / (max_y - min_y).max(1e-12);

    let mut sum = vec![0.0; width * height];
    let mut count = vec![0u32; width * height];

    for cell in 0..mesh.num_cells() {
        let poly = cell_polygon(mesh, cell);
        if poly.len() < 3 {
            continue;
        }
        let mut poly_px = Vec::with_capacity(poly.len());
        let mut min_px = f64::INFINITY;
        let mut max_px = f64::NEG_INFINITY;
        let mut min_py = f64::INFINITY;
        let mut max_py = f64::NEG_INFINITY;
        for p in &poly {
            let x = (p[0] - min_x) * scale_x;
            let y = (max_y - p[1]) * scale_y;
            min_px = min_px.min(x);
            max_px = max_px.max(x);
            min_py = min_py.min(y);
            max_py = max_py.max(y);
            poly_px.push([x, y]);
        }
        let x0 = min_px.floor().max(0.0) as i32;
        let x1 = max_px.ceil().min(width as f64 - 1.0) as i32;
        let y0 = min_py.floor().max(0.0) as i32;
        let y1 = max_py.ceil().min(height as f64 - 1.0) as i32;
        for yi in y0..=y1 {
            for xi in x0..=x1 {
                let px = xi as f64 + 0.5;
                let py = yi as f64 + 0.5;
                if point_in_poly(px, py, &poly_px) {
                    let idx = yi as usize * width + xi as usize;
                    sum[idx] += values[cell];
                    count[idx] += 1;
                }
            }
        }
    }

    let mut img = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let color = if count[idx] == 0 {
                Rgb([10, 10, 10])
            } else {
                let val = sum[idx] / count[idx] as f64;
                let t = (val - min_val) / (max_val - min_val).max(1e-12);
                colormap_sequential(t)
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    let _ = img.save(path);
}

fn std_dev(values: &[f64]) -> f64 {
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

#[test]
#[ignore]
fn low_mach_equivalence_vortex_street() {
    let save_plots = env_bool("CFD2_SAVE_PLOTS");
    let steps = env_usize("CFD2_LOW_MACH_STEPS", 1200);
    let dt = env_f64("CFD2_LOW_MACH_DT", 0.001);
    let cell = env_f64("CFD2_LOW_MACH_CELL", 0.05);
    let smooth_alpha = env_f64("CFD2_LOW_MACH_SMOOTH_ALPHA", 0.3);
    let smooth_iters = env_usize("CFD2_LOW_MACH_SMOOTH_ITERS", 40);
    let base_pressure = env_f64("CFD2_LOW_MACH_BASE_P", 25.0);
    let perturb_amp = env_f64("CFD2_LOW_MACH_PERTURB", 5e-3);
    let probe_stride = env_usize("CFD2_LOW_MACH_PROBE_STRIDE", 10);
    let plot_width = env_usize("CFD2_PLOT_WIDTH", 480);
    let plot_height = env_usize("CFD2_PLOT_HEIGHT", 160);

    let mesh = build_mesh(cell, smooth_alpha, smooth_iters);
    let probe_idx = find_probe_index(&mesh, 1.6, 0.5);

    let u_in = env_f64("CFD2_LOW_MACH_UIN", 1.0) as f32;
    let density = 1.0f32;
    let nu = 0.001f32;

    let mut incomp = pollster::block_on(GpuSolver::new(&mesh, None, None));
    incomp.set_dt(dt as f32);
    incomp.set_viscosity(nu);
    incomp.set_density(density);
    incomp.set_scheme(2);
    incomp.set_inlet_velocity(u_in);
    incomp.set_ramp_time(0.1);
    incomp.n_outer_correctors = 20;
    let mut u_seed = vec![(0.0f64, 0.0f64); mesh.num_cells()];
    let length = 3.0;
    for i in 0..mesh.num_cells() {
        let phase = mesh.cell_cx[i] / length * std::f64::consts::TAU;
        u_seed[i].1 = perturb_amp * phase.sin();
    }
    incomp.set_u(&u_seed);
    incomp.set_p(&vec![0.0f64; mesh.num_cells()]);
    incomp.initialize_history();

    let mut comp = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));
    comp.set_dt(dt as f32);
    comp.set_time_scheme(1);
    comp.set_density(density);
    comp.set_viscosity(nu);
    comp.set_inlet_velocity(u_in);
    let rho_init = vec![density; mesh.num_cells()];
    let p_init = vec![base_pressure as f32; mesh.num_cells()];
    let mut u_init = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        let phase = mesh.cell_cx[i] / length * std::f64::consts::TAU;
        u_init[i][1] = (perturb_amp * phase.sin()) as f32;
    }
    comp.set_state_fields(&rho_init, &u_init, &p_init);
    comp.initialize_history();

    let mut probe_incomp = Vec::new();
    let mut probe_comp = Vec::new();

    for step in 0..steps {
        incomp.step();
        comp.step();
        if step % probe_stride == 0 {
            let u_incomp = pollster::block_on(incomp.get_u());
            let u_comp = pollster::block_on(comp.get_u());
            probe_incomp.push(u_incomp[probe_idx].1);
            probe_comp.push(u_comp[probe_idx].1);
        }
    }

    let u_incomp = pollster::block_on(incomp.get_u());
    let _p_incomp = pollster::block_on(incomp.get_p());
    let u_comp = pollster::block_on(comp.get_u());
    let _p_comp = pollster::block_on(comp.get_p());

    let rms_num = u_incomp
        .iter()
        .zip(u_comp.iter())
        .map(|(a, b)| {
            let dx = a.0 - b.0;
            let dy = a.1 - b.1;
            dx * dx + dy * dy
        })
        .sum::<f64>()
        / u_incomp.len().max(1) as f64;
    let rms_den = u_incomp
        .iter()
        .map(|val| val.0 * val.0 + val.1 * val.1)
        .sum::<f64>()
        / u_incomp.len().max(1) as f64;
    let rel_rms = (rms_num.sqrt()) / rms_den.sqrt().max(1e-6);

    let std_incomp = std_dev(&probe_incomp);
    let std_comp = std_dev(&probe_comp);

    if save_plots {
        let out_dir = PathBuf::from("target/test_plots/low_mach");
        let _ = fs::create_dir_all(&out_dir);
        let vort_incomp = vorticity(&mesh, &u_incomp);
        let vort_comp = vorticity(&mesh, &u_comp);
        let speed_incomp: Vec<f64> = u_incomp
            .iter()
            .map(|val| (val.0 * val.0 + val.1 * val.1).sqrt())
            .collect();
        let speed_comp: Vec<f64> = u_comp
            .iter()
            .map(|val| (val.0 * val.0 + val.1 * val.1).sqrt())
            .collect();
        save_scalar_image(
            &out_dir.join("low_mach_incomp_vort.png"),
            &mesh,
            &vort_incomp,
            plot_width,
            plot_height,
        );
        save_scalar_image(
            &out_dir.join("low_mach_comp_vort.png"),
            &mesh,
            &vort_comp,
            plot_width,
            plot_height,
        );
        save_scalar_image(
            &out_dir.join("low_mach_incomp_speed.png"),
            &mesh,
            &speed_incomp,
            plot_width,
            plot_height,
        );
        save_scalar_image(
            &out_dir.join("low_mach_comp_speed.png"),
            &mesh,
            &speed_comp,
            plot_width,
            plot_height,
        );

        let mut summary = String::new();
        summary.push_str(&format!(
            "probe_std_incomp={:.3e}\nprobe_std_comp={:.3e}\nrel_rms={:.3}\n",
            std_incomp, std_comp, rel_rms
        ));
        let _ = fs::write(out_dir.join("low_mach_summary.txt"), summary);
    }

    assert!(std_incomp > 1e-4, "incompressible probe std too low");
    assert!(std_comp > 1e-5, "compressible probe std too low");
    assert!(rel_rms < 0.6, "relative RMS difference {:.3} too large", rel_rms);
}
