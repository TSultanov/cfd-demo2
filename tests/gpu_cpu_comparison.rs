use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle};
use cfd2::solver::piso::PisoSolver;
use nalgebra::{Vector2, Point2};

#[test]
fn test_gpu_cpu_comparison() {
    let domain_size = Vector2::new(3.0, 1.0);
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.5),
        obstacle_radius: 0.2,
    };
    // Use a coarser mesh for test speed, but fine enough to capture geometry
    let cell_size = 0.05; 
    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, domain_size);
    mesh.smooth(&geo, 0.3, 50);
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
    cpu_solver.dt = 0.0001;
    let mut gpu_solver = pollster::block_on(GpuSolver::new(&mesh));
    gpu_solver.constants.dt = 0.0001;
    // gpu_solver.update_constants(); // Private, so we rely on default or need to expose it.
    // Actually, we can't update it easily if method is private.
    // But wait, GpuSolver::new calls update_constants() at the end?
    // No, it initializes buffer with initial constants.
    // If we change constants.dt after new(), buffer is not updated.
    // We need to update the buffer.


    let dt = 0.0001;
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
        if cx < cell_size {
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
        
        // Stats
        let mut max_cpu_p = 0.0;
        for p in &cpu_solver.p.values { if p.abs() > max_cpu_p { max_cpu_p = p.abs(); } }
        let mut max_gpu_p = 0.0;
        for p in &gpu_p { if p.abs() > max_gpu_p { max_gpu_p = p.abs(); } }
        
        println!("  CPU Max P: {:.4}, GPU Max P: {:.4}", max_cpu_p, max_gpu_p);

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

        println!("  Max diff U: {:.4}, Max diff P: {:.4}", max_diff_u, max_diff_p);

        if max_diff_u > 1.0 || max_diff_p > 100.0 {
             println!("Divergence detected at step {}", step);
             break;
        }
    }
}
