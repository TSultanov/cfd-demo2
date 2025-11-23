use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::piso::PisoSolver;
use cfd2::solver::gpu::GpuSolver;
use nalgebra::Vector2;

fn main() {
    pollster::block_on(async {
        let domain_size = Vector2::new(2.0, 1.0);
        let geo = BackwardsStep {
            length: 2.0,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        };
        let mesh = generate_cut_cell_mesh(&geo, 0.05, 0.1, domain_size);
        println!("Mesh generated with {} cells and {} faces", mesh.num_cells(), mesh.num_faces());

        // CPU Solver
        let mut cpu_solver = PisoSolver::new(mesh.clone());
        cpu_solver.dt = 0.001;

        // GPU Solver
        let mut gpu_solver = GpuSolver::new(&mesh).await;
        // gpu_solver.dt = 0.001; // Need to expose dt setting

        // Initialize field
        let mut init_u = Vec::new();
        for i in 0..mesh.num_cells() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];
            let (ux, uy) = if cx < 0.1 && cy > 0.5 {
                (1.0, 0.0)
            } else {
                (0.0, 0.0)
            };
            
            cpu_solver.u.vx[i] = ux;
            cpu_solver.u.vy[i] = uy;
            init_u.push((ux, uy));
        }
        gpu_solver.set_u(&init_u);

        println!("Running 1 step...");
        
        // CPU Step
        cpu_solver.step();
        
        // GPU Step
        gpu_solver.step();
        
        // Compare U
        let gpu_u = gpu_solver.get_u().await;
        let mut max_diff_u = 0.0;
        for i in 0..mesh.num_cells() {
            let cpu_ux = cpu_solver.u.vx[i];
            let cpu_uy = cpu_solver.u.vy[i];
            let (gpu_ux, gpu_uy) = gpu_u[i];
            
            let diff_x = (cpu_ux - gpu_ux).abs();
            let diff_y = (cpu_uy - gpu_uy).abs();
            max_diff_u = max_diff_u.max(diff_x).max(diff_y);
        }
        println!("Max diff U: {}", max_diff_u);
        
        // Compare P
        let gpu_p = gpu_solver.get_p().await;
        let mut max_diff_p = 0.0;
        for i in 0..mesh.num_cells() {
            let cpu_p = cpu_solver.p[i];
            let gpu_p_val = gpu_p[i];
            
            let diff = (cpu_p - gpu_p_val).abs();
            max_diff_p = max_diff_p.max(diff);
        }
        println!("Max diff P: {}", max_diff_p);
        
        if max_diff_u < 1e-3 && max_diff_p < 1e-3 {
            println!("SUCCESS: CPU and GPU results match!");
        } else {
            println!("FAILURE: Results diverge.");
        }
    });
}
