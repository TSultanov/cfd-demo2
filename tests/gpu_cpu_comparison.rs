use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::piso::PisoSolver;
use nalgebra::Vector2;

#[test]
fn test_gpu_cpu_comparison() {
    let domain_size = Vector2::new(2.0, 1.0);
    let geo = BackwardsStep {
        length: 2.0,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    // Coarse mesh for speed and easier debugging
    let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.2, domain_size);
    println!("Mesh generated with {} cells", mesh.num_cells());

    // DEBUG: Check CPU mesh closure
    for i in 0..mesh.num_cells() {
        let mut sum_na = Vector2::new(0.0, 0.0);
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i+1];
        for k in start..end {
            let face_idx = mesh.cell_faces[k];
            let nx = mesh.face_nx[face_idx];
            let ny = mesh.face_ny[face_idx];
            let area = mesh.face_area[face_idx];
            let owner = mesh.face_owner[face_idx];
            
            let mut normal = Vector2::new(nx, ny);
            if owner != i {
                normal = -normal;
            }
            sum_na += normal * area;
        }
        if sum_na.norm() > 1e-6 {
            println!("Cell {} is not closed! Sum NA: {:?}", i, sum_na);
        }
    }

    let mut cpu_solver = PisoSolver::new(mesh.clone());
    let mut gpu_solver = pollster::block_on(GpuSolver::new(&mesh));

    let dt = 0.001;
    cpu_solver.dt = dt;
    gpu_solver.set_dt(dt as f32);

    // Initialize fields
    let num_cells = mesh.num_cells();
    let mut u_init = vec![(0.0, 0.0); num_cells];

    for i in 0..num_cells {
        let cx = mesh.cell_cx[i];
        let cy = mesh.cell_cy[i];
        if cx < 0.1 && cy > 0.5 {
            u_init[i] = (1.0, 0.0);
            cpu_solver.u.vx[i] = 1.0;
            cpu_solver.u.vy[i] = 0.0;
        }
    }
    gpu_solver.set_u(&u_init);

    // Run 1 step
    println!("Step 0");
    cpu_solver.step();
    gpu_solver.step();

    // Inspect GPU internals
    let gpu_u_vec = pollster::block_on(gpu_solver.get_u());
    let gpu_fluxes = pollster::block_on(gpu_solver.get_fluxes());
    let gpu_p = pollster::block_on(gpu_solver.get_p());
    let gpu_d_p = pollster::block_on(gpu_solver.get_d_p());
    
    println!("GPU U[0..5]: {:?}", &gpu_u_vec[0..5.min(gpu_u_vec.len())]);
    println!("GPU Fluxes[0..10]: {:?}", &gpu_fluxes[0..10.min(gpu_fluxes.len())]);
    println!("GPU D_P[0..5]: {:?}", &gpu_d_p[0..5.min(gpu_d_p.len())]);

    // CPU P stats
    let mut max_cpu_p = 0.0;
    let cpu_p = &cpu_solver.p.values;
    for &p in cpu_p {
        if p.abs() > max_cpu_p {
            max_cpu_p = p.abs();
        }
    }
    println!("Max CPU P: {}", max_cpu_p);

    // GPU P stats
    let mut max_gpu_p = 0.0;
    for &p in &gpu_p {
        if p.abs() > max_gpu_p {
            max_gpu_p = p.abs();
        }
    }
    println!("Max GPU P: {}", max_gpu_p);

    // Check Flux Divergence
    let mut cell_divergence = vec![0.0; mesh.num_cells()];
    for face_idx in 0..mesh.num_faces() {
        let flux = gpu_fluxes[face_idx];
        let owner = mesh.face_owner[face_idx];
        
        cell_divergence[owner] += flux; 
        if let Some(neighbor) = mesh.face_neighbor[face_idx] {
            cell_divergence[neighbor] -= flux;
        }
    }
    
    let mut max_div = 0.0;
    let mut total_div = 0.0;
    for div in &cell_divergence {
        if div.abs() > max_div {
            max_div = div.abs();
        }
        total_div += div;
    }
    println!("Max Flux Divergence: {}", max_div);
    println!("Total Flux Divergence: {}", total_div);

    // Compare results
    let gpu_u = gpu_u_vec;

    let mut max_diff_u = 0.0;
    let mut max_diff_p = 0.0;
    let mut max_diff_p = 0.0;

    for i in 0..num_cells {
        let cpu_vx = cpu_solver.u.vx[i];
        let cpu_vy = cpu_solver.u.vy[i];
        let cpu_p = cpu_solver.p.values[i];

        let (gpu_vx, gpu_vy) = gpu_u[i];
        let gpu_p_val = gpu_p[i];

        let diff_vx = (cpu_vx - gpu_vx).abs();
        let diff_vy = (cpu_vy - gpu_vy).abs();
        let diff_p = (cpu_p - gpu_p_val).abs();

        if diff_vx > max_diff_u {
            max_diff_u = diff_vx;
        }
        if diff_vy > max_diff_u {
            max_diff_u = diff_vy;
        }
        if diff_p > max_diff_p {
            max_diff_p = diff_p;
        }
    }

    println!("Max diff U: {}", max_diff_u);
    println!("Max diff P: {}", max_diff_p);

    // Assert with some tolerance - likely to fail if unstable
    assert!(max_diff_u < 1e-2, "Velocity divergence too high");
    assert!(max_diff_p < 5.0, "Pressure divergence too high");
}
