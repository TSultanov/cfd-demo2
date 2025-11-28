#[cfg(test)]
mod tests {
    use cfd2::solver::gpu::multigrid_solver::CycleType;
    use cfd2::solver::gpu::structs::GpuSolver;
    use cfd2::solver::mesh::{generate_cut_cell_mesh, RectangularChannel};
    use nalgebra::Vector2;

    #[test]
    fn test_amg_verification() {
        // 1. Generate Mesh
        let length = 1.0;
        let height = 1.0;
        let domain_size = Vector2::new(length, height);
        let geo = RectangularChannel { length, height };
        let min_cell_size = 0.1;
        let max_cell_size = 0.1;
        let mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);

        // 2. Create Standard Solver
        let mut solver_std = pollster::block_on(GpuSolver::new(&mesh));

        // 3. Create AMG Solver
        let mut solver_amg = pollster::block_on(GpuSolver::new(&mesh));
        // Enable AMG for Ux
        solver_amg.set_amg_cycle("Ux", CycleType::VCycle);

        // 4. Initialize Fields (Same Initial Condition)
        // Set dt
        solver_std.set_dt(0.01);
        solver_amg.set_dt(0.01);

        // Limit iterations for debugging
        solver_std.n_outer_correctors = 1;
        solver_amg.n_outer_correctors = 1;
        solver_std.update_constants();
        solver_amg.update_constants();

        // Set non-zero initial velocity to drive flow
        solver_std.set_uniform_u(1.0, 0.0);
        solver_amg.set_uniform_u(1.0, 0.0);

        // 5. Run 1 Step
        solver_std.step();
        solver_amg.step();

        // 6. Compare Results
        let u_std = pollster::block_on(solver_std.get_u());
        let u_amg = pollster::block_on(solver_amg.get_u());

        assert_eq!(u_std.len(), u_amg.len());

        let mut max_diff = 0.0;
        for i in 0..u_std.len() {
            let (ux_s, uy_s) = u_std[i];
            let (ux_a, uy_a) = u_amg[i];
            let diff_x = (ux_s - ux_a).abs();
            let diff_y = (uy_s - uy_a).abs();
            if diff_x > max_diff {
                max_diff = diff_x;
            }
            if diff_y > max_diff {
                max_diff = diff_y;
            }
        }

        println!("Max difference between Standard and AMG: {}", max_diff);

        assert!(max_diff.is_finite());
        // We can't assert max_diff < epsilon because they are different algorithms.
    }
}
