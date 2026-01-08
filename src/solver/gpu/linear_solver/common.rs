use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};

impl GpuSolver {
    pub fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) {
        assert_eq!(
            matrix_values.len(),
            self.num_nonzeros as usize,
            "matrix_values length mismatch"
        );
        assert_eq!(
            rhs.len(),
            self.num_cells as usize,
            "rhs length mismatch"
        );
        let values = self.linear_port_space.buffer(self.linear_ports.values);
        let b_rhs = self.linear_port_space.buffer(self.linear_ports.rhs);
        self.common
            .context
            .queue
            .write_buffer(values, 0, bytemuck::cast_slice(matrix_values));
        self.common
            .context
            .queue
            .write_buffer(b_rhs, 0, bytemuck::cast_slice(rhs));
    }

    pub async fn get_linear_solution(&self) -> Vec<f32> {
        let b_x = self.linear_port_space.buffer(self.linear_ports.x);
        let data = self
            .read_buffer(b_x, (self.num_cells as u64) * 4)
            .await;
        bytemuck::cast_slice(&data).to_vec()
    }

    pub fn solve_linear_system_cg(&self, max_iters: u32, tol: f32) -> LinearSolverStats {
        self.solve_linear_system_cg_with_size(self.num_cells, max_iters, tol)
    }

    pub fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> LinearSolverStats {
        self.scalar_cg.solve(&self.common.context, n, max_iters, tol)
    }
}
