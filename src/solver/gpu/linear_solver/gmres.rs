// FGMRES (Flexible GMRES) Solver for Coupled Systems
//
// FGMRES is preferred over standard GMRES when the preconditioner changes
// between iterations (as with AMG which rebuilds hierarchies).
//
// For saddle-point systems like coupled momentum-pressure:
// [A   G] [u]   [f]
// [D   C] [p] = [g]
//
// We use block preconditioning based on the Schur complement.

use crate::solver::gpu::structs::GpuSolver;

/// GMRES/FGMRES state for the coupled solver
pub struct GmresState {
    /// Krylov basis vectors V_0, V_1, ..., V_m (each of size n)
    pub basis_vectors: Vec<wgpu::Buffer>,
    /// Preconditioned vectors Z_0, Z_1, ..., Z_m-1 (for FGMRES)
    pub z_vectors: Vec<wgpu::Buffer>,
    /// Upper Hessenberg matrix H (m+1 x m), stored column-major
    pub h_matrix: Vec<f32>,
    /// Givens rotation cosines
    pub cs: Vec<f32>,
    /// Givens rotation sines
    pub sn: Vec<f32>,
    /// Right-hand side of least squares problem (after Givens rotations)
    pub g: Vec<f32>,
    /// Solution coefficients y
    pub y: Vec<f32>,
    /// Current basis size
    pub basis_size: usize,
    /// Maximum restart dimension
    pub max_restart: usize,
    /// Problem size (total DOFs)
    pub n: u32,
}

impl GmresState {
    pub fn new(device: &wgpu::Device, n: u32, max_restart: usize) -> Self {
        let mut basis_vectors = Vec::with_capacity(max_restart + 1);
        let mut z_vectors = Vec::with_capacity(max_restart);

        // Create basis vectors V_0 to V_m
        for i in 0..=max_restart {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("GMRES V_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            basis_vectors.push(buffer);
        }

        // Create Z vectors for FGMRES (preconditioned search directions)
        for i in 0..max_restart {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("GMRES Z_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            z_vectors.push(buffer);
        }

        Self {
            basis_vectors,
            z_vectors,
            h_matrix: vec![0.0; (max_restart + 1) * max_restart],
            cs: vec![0.0; max_restart],
            sn: vec![0.0; max_restart],
            g: vec![0.0; max_restart + 1],
            y: vec![0.0; max_restart],
            basis_size: 0,
            max_restart,
            n,
        }
    }

    /// Reset state for a new solve or restart
    pub fn reset(&mut self) {
        self.basis_size = 0;
        self.h_matrix.fill(0.0);
        self.cs.fill(0.0);
        self.sn.fill(0.0);
        self.g.fill(0.0);
        self.y.fill(0.0);
    }



    /// Solve upper triangular system H * y = g for the current basis size
    pub fn solve_triangular(&mut self) {
        let m = self.max_restart;
        let k = self.basis_size;

        // Back substitution
        for i in (0..k).rev() {
            let mut sum = self.g[i];
            for j in (i + 1)..k {
                sum -= self.h_matrix[j * (m + 1) + i] * self.y[j];
            }
            let h_ii = self.h_matrix[i * (m + 1) + i];
            if h_ii.abs() > 1e-14 {
                self.y[i] = sum / h_ii;
            } else {
                self.y[i] = 0.0;
            }
        }
    }

}

impl GpuSolver {
    /// Read a GPU buffer to CPU (for Hessenberg matrix construction)
    pub async fn read_buffer_f32_async(&self, buffer: &wgpu::Buffer, count: u32) -> Vec<f32> {
        let size = (count as u64) * 4;
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GMRES Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.profiling_stats
            .record_gpu_alloc("gmres:staging", size);

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        self.profiling_stats
            .record_cpu_alloc("gmres:cpu_copy", size);
        drop(data);
        staging.unmap();
        result
    }



    /// Scale a GPU vector: v = alpha * v
    pub fn scale_vector(&self, v: &wgpu::Buffer, alpha: f32, n: u32) {
        // Read, scale on CPU, write back
        // (For production, use a GPU kernel)
        let data = pollster::block_on(self.read_buffer_f32_async(v, n));
        let scaled: Vec<f32> = data.iter().map(|x| x * alpha).collect();
        self.context
            .queue
            .write_buffer(v, 0, bytemuck::cast_slice(&scaled));
    }

    /// AXPY: y = alpha * x + y
    pub fn axpy(&self, alpha: f32, x: &wgpu::Buffer, y: &wgpu::Buffer, n: u32) {
        let x_data = pollster::block_on(self.read_buffer_f32_async(x, n));
        let mut y_data = pollster::block_on(self.read_buffer_f32_async(y, n));
        for i in 0..n as usize {
            y_data[i] += alpha * x_data[i];
        }
        self.context
            .queue
            .write_buffer(y, 0, bytemuck::cast_slice(&y_data));
    }

    /// Copy buffer
    pub fn copy_buffer(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
        self.context.queue.submit(Some(encoder.finish()));
    }
}
