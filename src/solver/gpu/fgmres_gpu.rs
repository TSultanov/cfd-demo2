// GPU-based FGMRES Solver with Schur Complement Preconditioning
//
// This implementation keeps vectors on the GPU and only transfers scalar values
// (dot products, norms) to the CPU for control flow decisions.
//
// For the saddle-point system:
// [A   G] [u]   [b_u]
// [D   C] [p] = [b_p]

use super::structs::{GpuSolver, LinearSolverStats};

/// GPU Resources for FGMRES solver
pub struct FgmresGpuResources {
    // Krylov basis vectors V_0..V_m on GPU
    pub v_buffers: Vec<wgpu::Buffer>,
    // Preconditioned vectors Z_0..Z_{m-1} on GPU  
    pub z_buffers: Vec<wgpu::Buffer>,
    // Work vector w for SpMV result
    pub b_w: wgpu::Buffer,
    // Temporary vectors for preconditioner
    pub b_temp1: wgpu::Buffer,
    pub b_temp2: wgpu::Buffer,
    // Dot product partial sums (one per workgroup)
    pub b_dot_partial: wgpu::Buffer,
    // Scalar staging buffer for readback
    pub b_staging: wgpu::Buffer,
    
    // Pipelines
    pub pipeline_spmv: wgpu::ComputePipeline,
    pub pipeline_axpy: wgpu::ComputePipeline,
    pub pipeline_dot: wgpu::ComputePipeline,
    pub pipeline_scale_copy: wgpu::ComputePipeline,
    pub pipeline_precond_velocity: wgpu::ComputePipeline,
    pub pipeline_precond_pressure: wgpu::ComputePipeline,
    pub pipeline_precond_correct: wgpu::ComputePipeline,
    
    // Bind group layouts
    pub bgl_vectors: wgpu::BindGroupLayout,
    pub bgl_matrix: wgpu::BindGroupLayout,
    pub bgl_params: wgpu::BindGroupLayout,
    
    // Parameters
    pub max_restart: usize,
    pub n: u32,
    pub num_cells: u32,
    pub num_workgroups: u32,
}

impl GpuSolver {
    /// Create GPU resources for FGMRES
    pub fn create_fgmres_gpu_resources(&self, max_restart: usize) -> FgmresGpuResources {
        let device = &self.context.device;
        let n = self.num_cells * 3;
        let workgroup_size = 64u32;
        let num_workgroups = n.div_ceil(workgroup_size);
        
        // Create Krylov basis vectors
        let mut v_buffers = Vec::with_capacity(max_restart + 1);
        for i in 0..=max_restart {
            v_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("FGMRES_GPU V_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        
        // Create preconditioned vectors
        let mut z_buffers = Vec::with_capacity(max_restart);
        for i in 0..max_restart {
            z_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("FGMRES_GPU Z_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        
        // Work buffers
        let b_w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES_GPU w"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let b_temp1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES_GPU temp1"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let b_temp2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES_GPU temp2"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Dot product partial sums
        let b_dot_partial = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES_GPU dot_partial"),
            size: (num_workgroups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Staging buffer for scalar readback
        let b_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES_GPU staging"),
            size: (num_workgroups as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Bind group layouts
        let bgl_vectors = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES vectors layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bgl_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES matrix layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bgl_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES params layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Load shader
        let shader_source = include_str!("shaders/fgmres_kernels.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FGMRES GPU shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FGMRES pipeline layout"),
            bind_group_layouts: &[&bgl_vectors, &bgl_matrix, &bgl_params],
            push_constant_ranges: &[],
        });
        
        // Create pipelines
        let pipeline_spmv = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FGMRES spmv"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "spmv",
        });
        
        let pipeline_axpy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FGMRES axpy"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "axpy",
        });
        
        let pipeline_dot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FGMRES dot"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "dot_product",
        });
        
        let pipeline_scale_copy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FGMRES scale_copy"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "scale_copy",
        });
        
        // Preconditioner pipelines - use simpler layout for now
        let precond_shader_source = include_str!("shaders/schur_precond_gpu.wgsl");
        let precond_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Schur precond GPU shader"),
            source: wgpu::ShaderSource::Wgsl(precond_shader_source.into()),
        });
        
        let pipeline_precond_velocity = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Precond velocity"),
            layout: Some(&pipeline_layout),
            module: &precond_shader,
            entry_point: "precond_velocity_sweep",
        });
        
        let pipeline_precond_pressure = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Precond pressure"),
            layout: Some(&pipeline_layout),
            module: &precond_shader,
            entry_point: "precond_pressure_sweep",
        });
        
        let pipeline_precond_correct = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Precond correct"),
            layout: Some(&pipeline_layout),
            module: &precond_shader,
            entry_point: "precond_velocity_correct",
        });
        
        FgmresGpuResources {
            v_buffers,
            z_buffers,
            b_w,
            b_temp1,
            b_temp2,
            b_dot_partial,
            b_staging,
            pipeline_spmv,
            pipeline_axpy,
            pipeline_dot,
            pipeline_scale_copy,
            pipeline_precond_velocity,
            pipeline_precond_pressure,
            pipeline_precond_correct,
            bgl_vectors,
            bgl_matrix,
            bgl_params,
            max_restart,
            n,
            num_cells: self.num_cells,
            num_workgroups,
        }
    }
    
    /// Solve coupled system using GPU-based FGMRES
    /// 
    /// This keeps all vectors on GPU, only reading scalar values for control flow.
    pub fn solve_coupled_fgmres_gpu(&mut self, resources: &FgmresGpuResources) -> LinearSolverStats {
        let n = resources.n as usize;
        let num_cells = resources.num_cells as usize;
        
        // FGMRES parameters
        let max_restart = resources.max_restart;
        let max_outer = 20;
        let tol = 1e-4f32;
        
        let res = match &self.coupled_resources {
            Some(r) => r,
            None => {
                println!("Coupled resources not initialized!");
                return LinearSolverStats::default();
            }
        };
        
        // Compute initial residual norm ||b||
        let rhs_norm = self.compute_norm_gpu(&res.b_rhs, n as u32);
        
        if rhs_norm < 1e-14 || rhs_norm.is_nan() {
            println!("FGMRES_GPU: RHS norm is {} - trivial or invalid", rhs_norm);
            return LinearSolverStats {
                iterations: 0,
                residual: rhs_norm,
                converged: rhs_norm < 1e-14,
                diverged: rhs_norm.is_nan(),
                time: std::time::Duration::default(),
            };
        }
        
        // Initialize: x = 0, r = b (copy b to r buffer)
        self.zero_buffer(&res.b_x, (n as u64) * 4);
        self.copy_buffer_internal(&res.b_rhs, &res.b_r, (n as u64) * 4);
        
        let mut total_iters = 0u32;
        let mut final_resid = rhs_norm;
        let mut converged = false;
        
        // Hessenberg matrix and Givens rotations (kept on CPU - O(m^2) with small m)
        let mut h_matrix = vec![vec![0.0f32; max_restart]; max_restart + 1];
        let mut cs = vec![0.0f32; max_restart];
        let mut sn = vec![0.0f32; max_restart];
        let mut g = vec![0.0f32; max_restart + 1];
        
        println!("FGMRES_GPU: Initial residual = {:.2e}", rhs_norm);
        
        // FGMRES outer loop (restarts)
        for outer in 0..max_outer {
            // Compute r_norm
            let r_norm = self.compute_norm_gpu(&res.b_r, n as u32);
            
            if r_norm < tol * rhs_norm || r_norm < 1e-10 {
                converged = true;
                final_resid = r_norm;
                println!("FGMRES_GPU converged: iter {}, residual = {:.2e}", total_iters, r_norm);
                break;
            }
            
            // Initialize: v_0 = r / ||r||
            self.scale_copy_gpu(&res.b_r, &resources.v_buffers[0], 1.0 / r_norm, n as u32);
            
            // Reset Hessenberg matrix
            for row in &mut h_matrix {
                row.fill(0.0);
            }
            cs.fill(0.0);
            sn.fill(0.0);
            g.fill(0.0);
            g[0] = r_norm;
            
            let mut k = 0;
            
            // Arnoldi iteration
            for j in 0..max_restart {
                k = j + 1;
                total_iters += 1;
                
                // z_j = M^{-1} * v_j (preconditioner on GPU)
                self.apply_preconditioner_gpu(
                    resources,
                    &resources.v_buffers[j],
                    &resources.z_buffers[j],
                );
                
                // w = A * z_j (SpMV on GPU)
                self.spmv_gpu(res, &resources.z_buffers[j], &resources.b_w, n as u32);
                
                // Modified Gram-Schmidt orthogonalization
                for i in 0..=j {
                    // h_ij = <w, v_i>
                    let h_ij = self.dot_gpu(&resources.b_w, &resources.v_buffers[i], n as u32);
                    h_matrix[i][j] = h_ij;
                    
                    // w = w - h_ij * v_i
                    self.axpy_gpu(-h_ij, &resources.v_buffers[i], &resources.b_w, n as u32);
                }
                
                // w_norm = ||w||
                let w_norm = self.compute_norm_gpu(&resources.b_w, n as u32);
                h_matrix[j + 1][j] = w_norm;
                
                if w_norm < 1e-14 {
                    println!("FGMRES_GPU: Breakdown at iteration {}", total_iters);
                    break;
                }
                
                // v_{j+1} = w / ||w||
                self.scale_copy_gpu(&resources.b_w, &resources.v_buffers[j + 1], 1.0 / w_norm, n as u32);
                
                // Apply previous Givens rotations to new column
                for i in 0..j {
                    let h_i = h_matrix[i][j];
                    let h_i1 = h_matrix[i + 1][j];
                    h_matrix[i][j] = cs[i] * h_i + sn[i] * h_i1;
                    h_matrix[i + 1][j] = -sn[i] * h_i + cs[i] * h_i1;
                }
                
                // Compute new Givens rotation
                let h_jj = h_matrix[j][j];
                let h_j1j = h_matrix[j + 1][j];
                let rho = (h_jj * h_jj + h_j1j * h_j1j).sqrt();
                if rho.abs() < 1e-14 {
                    cs[j] = 1.0;
                    sn[j] = 0.0;
                } else {
                    cs[j] = h_jj / rho;
                    sn[j] = h_j1j / rho;
                }
                
                h_matrix[j][j] = rho;
                h_matrix[j + 1][j] = 0.0;
                
                // Apply to g
                let g_j = g[j];
                let g_j1 = g[j + 1];
                g[j] = cs[j] * g_j + sn[j] * g_j1;
                g[j + 1] = -sn[j] * g_j + cs[j] * g_j1;
                
                let resid_est = g[j + 1].abs();
                
                if total_iters % 10 == 0 || resid_est < tol * rhs_norm {
                    println!("FGMRES_GPU iter {}: residual = {:.2e} (target: {:.2e})", 
                             total_iters, resid_est, tol * rhs_norm);
                }
                
                if resid_est < tol * rhs_norm || resid_est < 1e-10 {
                    converged = true;
                    final_resid = resid_est;
                    println!("FGMRES_GPU converged at iter {}: residual = {:.2e}", total_iters, resid_est);
                    break;
                }
            }
            
            // Solve H * y = g (triangular solve on CPU)
            let mut y = vec![0.0f32; k];
            for i in (0..k).rev() {
                let mut sum = g[i];
                for jj in (i + 1)..k {
                    sum -= h_matrix[i][jj] * y[jj];
                }
                if h_matrix[i][i].abs() > 1e-14 {
                    y[i] = sum / h_matrix[i][i];
                }
            }
            
            // Update solution: x = x + sum(y_j * z_j) on GPU
            for jj in 0..k {
                self.axpy_gpu(y[jj], &resources.z_buffers[jj], &res.b_x, n as u32);
            }
            
            if converged {
                break;
            }
            
            // Compute new residual: r = b - A*x
            self.spmv_gpu(res, &res.b_x, &res.b_r, n as u32);  // r = A*x
            // r = b - r
            self.compute_residual_gpu(res, n as u32);
            
            let true_resid = self.compute_norm_gpu(&res.b_r, n as u32);
            final_resid = true_resid;
            
            if true_resid < tol * rhs_norm {
                converged = true;
                println!("FGMRES_GPU restart {}: true residual = {:.2e} (CONVERGED)", outer + 1, true_resid);
                break;
            }
            
            println!("FGMRES_GPU restart {}: true residual = {:.2e} (target: {:.2e})", 
                     outer + 1, true_resid, tol * rhs_norm);
        }
        
        println!("FGMRES_GPU finished: {} iters, residual = {:.2e}, converged = {}", 
                 total_iters, final_resid, converged);
        
        LinearSolverStats {
            iterations: total_iters,
            residual: final_resid,
            converged,
            diverged: final_resid.is_nan(),
            time: std::time::Duration::default(),
        }
    }
    
    // GPU helper functions
    
    fn compute_norm_gpu(&self, buffer: &wgpu::Buffer, n: u32) -> f32 {
        // Read buffer and compute on CPU for now
        // TODO: Use GPU reduction
        let data = pollster::block_on(self.read_buffer_f32(buffer, n));
        data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
    
    fn scale_copy_gpu(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, alpha: f32, n: u32) {
        let data = pollster::block_on(self.read_buffer_f32(src, n));
        let scaled: Vec<f32> = data.iter().map(|x| x * alpha).collect();
        self.context.queue.write_buffer(dst, 0, bytemuck::cast_slice(&scaled));
    }
    
    fn axpy_gpu(&self, alpha: f32, x: &wgpu::Buffer, y: &wgpu::Buffer, n: u32) {
        let x_data = pollster::block_on(self.read_buffer_f32(x, n));
        let mut y_data = pollster::block_on(self.read_buffer_f32(y, n));
        for i in 0..n as usize {
            y_data[i] += alpha * x_data[i];
        }
        self.context.queue.write_buffer(y, 0, bytemuck::cast_slice(&y_data));
    }
    
    fn dot_gpu(&self, a: &wgpu::Buffer, b: &wgpu::Buffer, n: u32) -> f32 {
        let a_data = pollster::block_on(self.read_buffer_f32(a, n));
        let b_data = pollster::block_on(self.read_buffer_f32(b, n));
        a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum()
    }
    
    fn spmv_gpu(&self, res: &super::structs::CoupledSolverResources, x: &wgpu::Buffer, y: &wgpu::Buffer, n: u32) {
        // Read matrix and vectors, compute on CPU for now
        // TODO: Use GPU SpMV kernel
        let row_offsets = pollster::block_on(self.read_buffer_u32(&res.b_row_offsets, n + 1));
        let col_indices = pollster::block_on(self.read_buffer_u32(&res.b_col_indices, res.num_nonzeros));
        let values = pollster::block_on(self.read_buffer_f32(&res.b_matrix_values, res.num_nonzeros));
        let x_data = pollster::block_on(self.read_buffer_f32(x, n));
        
        let mut y_data = vec![0.0f32; n as usize];
        for row in 0..n as usize {
            let start = row_offsets[row] as usize;
            let end = row_offsets[row + 1] as usize;
            let mut sum = 0.0f32;
            for k in start..end {
                sum += values[k] * x_data[col_indices[k] as usize];
            }
            y_data[row] = sum;
        }
        
        self.context.queue.write_buffer(y, 0, bytemuck::cast_slice(&y_data));
    }
    
    fn compute_residual_gpu(&self, res: &super::structs::CoupledSolverResources, n: u32) {
        // r = b - r (where r currently contains A*x)
        let rhs = pollster::block_on(self.read_buffer_f32(&res.b_rhs, n));
        let ax = pollster::block_on(self.read_buffer_f32(&res.b_r, n));
        let r: Vec<f32> = rhs.iter().zip(ax.iter()).map(|(b, ax)| b - ax).collect();
        self.context.queue.write_buffer(&res.b_r, 0, bytemuck::cast_slice(&r));
    }
    
    fn copy_buffer_internal(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
        self.context.queue.submit(Some(encoder.finish()));
    }
    
    fn apply_preconditioner_gpu(
        &self, 
        _resources: &FgmresGpuResources,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        // For now, use CPU-based preconditioner
        // This will be converted to GPU kernels
        let res = self.coupled_resources.as_ref().unwrap();
        let n = self.num_cells as usize * 3;
        let num_cells = self.num_cells as usize;
        
        // Read data
        let row_offsets = pollster::block_on(self.read_buffer_u32(&res.b_row_offsets, (n + 1) as u32));
        let col_indices = pollster::block_on(self.read_buffer_u32(&res.b_col_indices, res.num_nonzeros));
        let values = pollster::block_on(self.read_buffer_f32(&res.b_matrix_values, res.num_nonzeros));
        let v = pollster::block_on(self.read_buffer_f32(input, n as u32));
        
        // Extract diagonals
        let mut diag_u = vec![0.0f32; num_cells];
        let mut diag_v = vec![0.0f32; num_cells];
        let mut diag_p = vec![0.0f32; num_cells];
        
        for cell in 0..num_cells {
            let row_u = 3 * cell;
            let start = row_offsets[row_u] as usize;
            let end = row_offsets[row_u + 1] as usize;
            for k in start..end {
                if col_indices[k] as usize == row_u {
                    diag_u[cell] = values[k];
                    break;
                }
            }
            
            let row_v = 3 * cell + 1;
            let start = row_offsets[row_v] as usize;
            let end = row_offsets[row_v + 1] as usize;
            for k in start..end {
                if col_indices[k] as usize == row_v {
                    diag_v[cell] = values[k];
                    break;
                }
            }
            
            let row_p = 3 * cell + 2;
            let start = row_offsets[row_p] as usize;
            let end = row_offsets[row_p + 1] as usize;
            for k in start..end {
                if col_indices[k] as usize == row_p {
                    diag_p[cell] = values[k];
                    break;
                }
            }
        }
        
        // Apply block preconditioner (same as before)
        let z = self.apply_block_preconditioner_internal(
            &v, &row_offsets, &col_indices, &values,
            &diag_u, &diag_v, &diag_p, num_cells
        );
        
        self.context.queue.write_buffer(output, 0, bytemuck::cast_slice(&z));
    }
    
    fn apply_block_preconditioner_internal(
        &self,
        v: &[f32],
        row_offsets: &[u32],
        col_indices: &[u32],
        values: &[f32],
        diag_u: &[f32],
        diag_v: &[f32],
        diag_p: &[f32],
        num_cells: usize,
    ) -> Vec<f32> {
        let n = num_cells * 3;
        let mut z = vec![0.0f32; n];
        
        // Step 1: SGS velocity solve
        for cell in 0..num_cells {
            let base = 3 * cell;
            if diag_u[cell].abs() > 1e-14 {
                z[base] = v[base] / diag_u[cell];
            }
            if diag_v[cell].abs() > 1e-14 {
                z[base + 1] = v[base + 1] / diag_v[cell];
            }
        }
        
        let num_vel_sweeps = 3;
        for _sweep in 0..num_vel_sweeps {
            for cell in 0..num_cells {
                let row_u = 3 * cell;
                let row_v = 3 * cell + 1;
                
                let mut res_u = v[row_u];
                let mut res_v = v[row_v];
                
                let start_u = row_offsets[row_u] as usize;
                let end_u = row_offsets[row_u + 1] as usize;
                for k in start_u..end_u {
                    let col = col_indices[k] as usize;
                    let col_type = col % 3;
                    if col != row_u && (col_type == 0 || col_type == 1) {
                        res_u -= values[k] * z[col];
                    }
                }
                
                let start_v = row_offsets[row_v] as usize;
                let end_v = row_offsets[row_v + 1] as usize;
                for k in start_v..end_v {
                    let col = col_indices[k] as usize;
                    let col_type = col % 3;
                    if col != row_v && (col_type == 0 || col_type == 1) {
                        res_v -= values[k] * z[col];
                    }
                }
                
                if diag_u[cell].abs() > 1e-14 {
                    z[row_u] = res_u / diag_u[cell];
                }
                if diag_v[cell].abs() > 1e-14 {
                    z[row_v] = res_v / diag_v[cell];
                }
            }
            
            for cell in (0..num_cells).rev() {
                let row_u = 3 * cell;
                let row_v = 3 * cell + 1;
                
                let mut res_u = v[row_u];
                let mut res_v = v[row_v];
                
                let start_u = row_offsets[row_u] as usize;
                let end_u = row_offsets[row_u + 1] as usize;
                for k in start_u..end_u {
                    let col = col_indices[k] as usize;
                    let col_type = col % 3;
                    if col != row_u && (col_type == 0 || col_type == 1) {
                        res_u -= values[k] * z[col];
                    }
                }
                
                let start_v = row_offsets[row_v] as usize;
                let end_v = row_offsets[row_v + 1] as usize;
                for k in start_v..end_v {
                    let col = col_indices[k] as usize;
                    let col_type = col % 3;
                    if col != row_v && (col_type == 0 || col_type == 1) {
                        res_v -= values[k] * z[col];
                    }
                }
                
                if diag_u[cell].abs() > 1e-14 {
                    z[row_u] = res_u / diag_u[cell];
                }
                if diag_v[cell].abs() > 1e-14 {
                    z[row_v] = res_v / diag_v[cell];
                }
            }
        }
        
        // Step 2: Modified pressure RHS
        let mut v_p_mod = vec![0.0f32; num_cells];
        for cell in 0..num_cells {
            let row_p = 3 * cell + 2;
            let start = row_offsets[row_p] as usize;
            let end = row_offsets[row_p + 1] as usize;
            
            let mut v_p_cell = v[row_p];
            for k in start..end {
                let col = col_indices[k] as usize;
                let col_type = col % 3;
                if col_type == 0 || col_type == 1 {
                    v_p_cell -= values[k] * z[col];
                }
            }
            v_p_mod[cell] = v_p_cell;
        }
        
        // Step 3: Compute Schur diagonal
        let mut schur_diags = vec![0.0f32; num_cells];
        for cell in 0..num_cells {
            let base = 3 * cell;
            let mut schur_diag = diag_p[cell];
            
            // Extract D and G for this cell
            let row_p = base + 2;
            let start_p = row_offsets[row_p] as usize;
            let end_p = row_offsets[row_p + 1] as usize;
            let mut d_pu = 0.0f32;
            let mut d_pv = 0.0f32;
            for k in start_p..end_p {
                let col = col_indices[k] as usize;
                if col == base { d_pu = values[k]; }
                else if col == base + 1 { d_pv = values[k]; }
            }
            
            let row_u = base;
            let start_u = row_offsets[row_u] as usize;
            let end_u = row_offsets[row_u + 1] as usize;
            let mut g_up = 0.0f32;
            for k in start_u..end_u {
                if col_indices[k] as usize == base + 2 {
                    g_up = values[k];
                    break;
                }
            }
            
            let row_v = base + 1;
            let start_v = row_offsets[row_v] as usize;
            let end_v = row_offsets[row_v + 1] as usize;
            let mut g_vp = 0.0f32;
            for k in start_v..end_v {
                if col_indices[k] as usize == base + 2 {
                    g_vp = values[k];
                    break;
                }
            }
            
            if diag_u[cell].abs() > 1e-14 {
                schur_diag -= d_pu * g_up / diag_u[cell];
            }
            if diag_v[cell].abs() > 1e-14 {
                schur_diag -= d_pv * g_vp / diag_v[cell];
            }
            
            if schur_diag <= 0.0 {
                schur_diag = diag_p[cell].max(1e-14);
            }
            schur_diags[cell] = schur_diag;
        }
        
        // Initialize pressure
        for cell in 0..num_cells {
            if schur_diags[cell].abs() > 1e-14 {
                z[3 * cell + 2] = v_p_mod[cell] / schur_diags[cell];
            }
        }
        
        // SSOR pressure sweeps
        let omega = 1.4;
        let num_sweeps = 6;
        
        for _sweep in 0..num_sweeps {
            for cell in 0..num_cells {
                let row = 3 * cell + 2;
                let schur_diag = schur_diags[cell];
                if schur_diag.abs() < 1e-14 { continue; }
                
                let start = row_offsets[row] as usize;
                let end = row_offsets[row + 1] as usize;
                
                let mut sum = 0.0f32;
                for k in start..end {
                    let col = col_indices[k] as usize;
                    if col != row && col % 3 == 2 {
                        sum += values[k] * z[col];
                    }
                }
                
                let z_new = (v_p_mod[cell] - sum) / schur_diag;
                z[row] = (1.0 - omega) * z[row] + omega * z_new;
            }
            
            for cell in (0..num_cells).rev() {
                let row = 3 * cell + 2;
                let schur_diag = schur_diags[cell];
                if schur_diag.abs() < 1e-14 { continue; }
                
                let start = row_offsets[row] as usize;
                let end = row_offsets[row + 1] as usize;
                
                let mut sum = 0.0f32;
                for k in start..end {
                    let col = col_indices[k] as usize;
                    if col != row && col % 3 == 2 {
                        sum += values[k] * z[col];
                    }
                }
                
                let z_new = (v_p_mod[cell] - sum) / schur_diag;
                z[row] = (1.0 - omega) * z[row] + omega * z_new;
            }
        }
        
        // Step 4: Velocity correction
        for cell in 0..num_cells {
            let base = 3 * cell;
            
            let row_u = base;
            let start_u = row_offsets[row_u] as usize;
            let end_u = row_offsets[row_u + 1] as usize;
            for k in start_u..end_u {
                let col = col_indices[k] as usize;
                if col % 3 == 2 {
                    let p_neighbor = z[col];
                    if diag_u[cell].abs() > 1e-14 {
                        z[base] -= (values[k] * p_neighbor) / diag_u[cell];
                    }
                }
            }
            
            let row_v = base + 1;
            let start_v = row_offsets[row_v] as usize;
            let end_v = row_offsets[row_v + 1] as usize;
            for k in start_v..end_v {
                let col = col_indices[k] as usize;
                if col % 3 == 2 {
                    let p_neighbor = z[col];
                    if diag_v[cell].abs() > 1e-14 {
                        z[base + 1] -= (values[k] * p_neighbor) / diag_v[cell];
                    }
                }
            }
        }
        
        z
    }
}
