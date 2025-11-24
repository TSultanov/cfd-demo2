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
    let density = 1.0;
    let viscosity = 0.01;

    cpu_solver.dt = dt;
    cpu_solver.density = density;
    cpu_solver.viscosity = viscosity;

    gpu_solver.set_dt(dt as f32);
    gpu_solver.set_density(density as f32);
    gpu_solver.set_viscosity(viscosity as f32);

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

    // Run multiple steps
    for step in 0..20 {
        cpu_solver.step();
        gpu_solver.step();

        // Inspect GPU internals
        let gpu_u_vec = pollster::block_on(gpu_solver.get_u());
        let gpu_p = pollster::block_on(gpu_solver.get_p());
        
        // Compare results
        let mut max_diff_u = 0.0;
        let mut max_diff_p = 0.0;

        for i in 0..num_cells {
            let cpu_vx = cpu_solver.u.vx[i];
            let cpu_vy = cpu_solver.u.vy[i];
            let cpu_p_val = cpu_solver.p.values[i];

            let (gpu_vx, gpu_vy) = gpu_u_vec[i];
            let gpu_p_val = gpu_p[i];

            let diff_vx = (cpu_vx - gpu_vx).abs();
            let diff_vy = (cpu_vy - gpu_vy).abs();
            let diff_p = (cpu_p_val - gpu_p_val).abs();

            if diff_vx > max_diff_u { max_diff_u = diff_vx; }
            if diff_vy > max_diff_u { max_diff_u = diff_vy; }
            if diff_p > max_diff_p { max_diff_p = diff_p; }
        }

        if max_diff_u > 1.0 || max_diff_p > 100.0 {
             println!("Divergence detected at step {}", step);
             break;
        }
    }
}
