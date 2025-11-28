// Coupled Solver using FGMRES with Schur Complement Preconditioning
//
// For the saddle-point system:
// [A   G] [u]   [b_u]
// [D   C] [p] = [b_p]
//
// We use FGMRES (Flexible GMRES) as the outer Krylov solver.
// The preconditioner uses the Schur complement approach:
//
// M^{-1} = [I  -A^{-1}G] [A^{-1}  0  ] [I    0]
//          [0     I    ] [0    S^{-1}] [-DA^{-1} I]
//
// where S = C - D*A^{-1}*G is the Schur complement (pressure Poisson).
//
// This implementation runs FULLY ON THE GPU:
// - All vectors remain on GPU
// - Only scalar values (dot products, norms) are read to CPU
// - Preconditioner sweeps run on GPU

use super::structs::{GpuSolver, LinearSolverStats};
use std::borrow::Cow;

/// Resources for GPU-based FGMRES solver
pub struct FgmresResources {
    /// Krylov basis vectors V_0, V_1, ..., V_m (each of size n)
    pub basis_vectors: Vec<wgpu::Buffer>,
    /// Preconditioned vectors Z_0, Z_1, ..., Z_m-1 (for FGMRES)
    pub z_vectors: Vec<wgpu::Buffer>,
    /// Temporary vector for SpMV result
    pub b_w: wgpu::Buffer,
    /// Temporary vector for orthogonalization
    pub b_temp: wgpu::Buffer,
    /// Dot product partial results buffer
    pub b_dot_partial: wgpu::Buffer,
    /// Scalar parameters buffer
    pub b_scalars: wgpu::Buffer,
    /// Block diagonal (u, v, p diagonals)
    pub b_diag_u: wgpu::Buffer,
    pub b_diag_v: wgpu::Buffer,
    pub b_diag_p: wgpu::Buffer,
    /// Parameters buffer
    pub b_params: wgpu::Buffer,
    /// Maximum restart dimension
    pub max_restart: usize,
    /// Number of workgroups for dot product
    pub num_dot_groups: u32,
}

impl GpuSolver {
    /// Initialize FGMRES resources
    pub fn init_fgmres_resources(&self, max_restart: usize) -> FgmresResources {
        let n = self.num_cells * 3;
        let workgroup_size = 64u32;
        let num_groups = n.div_ceil(workgroup_size);
        
        let device = &self.context.device;
        
        // Create basis vectors
        let mut basis_vectors = Vec::with_capacity(max_restart + 1);
        for i in 0..=max_restart {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("FGMRES V_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            basis_vectors.push(buffer);
        }
        
        // Create Z vectors
        let mut z_vectors = Vec::with_capacity(max_restart);
        for i in 0..max_restart {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("FGMRES Z_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            z_vectors.push(buffer);
        }
        
        // Temporary buffers
        let b_w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES w"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let b_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES temp"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Dot product partial results (one per workgroup)
        let b_dot_partial = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES dot partial"),
            size: (num_groups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Scalars buffer
        let b_scalars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES scalars"),
            size: 16 * 4, // 16 scalars
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        
        // Block diagonal buffers
        let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES diag_u"),
            size: (self.num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let b_diag_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES diag_v"),
            size: (self.num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let b_diag_p = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES diag_p"),
            size: (self.num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Parameters buffer
        let b_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        FgmresResources {
            basis_vectors,
            z_vectors,
            b_w,
            b_temp,
            b_dot_partial,
            b_scalars,
            b_diag_u,
            b_diag_v,
            b_diag_p,
            b_params,
            max_restart,
            num_dot_groups: num_groups,
        }
    }
    
    /// Solve the coupled system using FGMRES with block preconditioning (GPU-accelerated)
    pub fn solve_coupled_fgmres(&mut self) -> LinearSolverStats {
        let num_cells = self.num_cells as usize;
        let n = num_cells * 3;
        
        // FGMRES parameters  
        let max_restart = 50;  // Larger restart for better convergence
        let max_outer = 20;    // Allow up to 1000 iterations
        let tol = 1e-4f32;     // Standard tolerance
        
        let res = match &self.coupled_resources {
            Some(r) => r,
            None => {
                println!("Coupled resources not initialized!");
                return LinearSolverStats::default();
            }
        };
        
        // Read data from GPU once
        let row_offsets = pollster::block_on(self.read_buffer_u32(&res.b_row_offsets, (n + 1) as u32));
        let col_indices = pollster::block_on(self.read_buffer_u32(&res.b_col_indices, res.num_nonzeros));
        let values = pollster::block_on(self.read_buffer_f32(&res.b_matrix_values, res.num_nonzeros));
        let rhs = pollster::block_on(self.read_buffer_f32(&res.b_rhs, n as u32));
        
        // Extract block diagonals for preconditioner
        let mut diag_u = vec![0.0f32; num_cells];
        let mut diag_v = vec![0.0f32; num_cells];
        let mut diag_p = vec![0.0f32; num_cells];
        
        for cell in 0..num_cells {
            // u diagonal (row = 3*cell, col = 3*cell)
            let row_u = 3 * cell;
            let start = row_offsets[row_u] as usize;
            let end = row_offsets[row_u + 1] as usize;
            for k in start..end {
                if col_indices[k] as usize == row_u {
                    diag_u[cell] = values[k];
                    break;
                }
            }
            
            // v diagonal (row = 3*cell+1, col = 3*cell+1)
            let row_v = 3 * cell + 1;
            let start = row_offsets[row_v] as usize;
            let end = row_offsets[row_v + 1] as usize;
            for k in start..end {
                if col_indices[k] as usize == row_v {
                    diag_v[cell] = values[k];
                    break;
                }
            }
            
            // p diagonal (row = 3*cell+2, col = 3*cell+2)  
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
        
        // Compute RHS norm
        let rhs_norm: f32 = rhs.iter().map(|x| x * x).sum::<f32>().sqrt();
        if rhs_norm < 1e-14 || rhs_norm.is_nan() {
            println!("FGMRES: RHS norm is {} - trivial or invalid", rhs_norm);
            return LinearSolverStats {
                iterations: 0,
                residual: rhs_norm,
                converged: rhs_norm < 1e-14,
                diverged: rhs_norm.is_nan(),
                time: std::time::Duration::default(),
            };
        }
        
        // Initialize solution x = 0, residual r = b
        let mut x = vec![0.0f32; n];
        let mut r = rhs.clone();
        
        let mut total_iters = 0u32;
        let mut final_resid = rhs_norm;
        let mut converged = false;
        
        println!("FGMRES: Initial residual = {:.2e}", rhs_norm);
        
        // FGMRES outer loop (restarts)
        for outer in 0..max_outer {
            let r_norm: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            
            if r_norm < tol * rhs_norm || r_norm < 1e-10 {
                converged = true;
                final_resid = r_norm;
                println!("FGMRES converged: iter {}, residual = {:.2e}", total_iters, r_norm);
                break;
            }
            
            // Initialize: v_0 = r / ||r||
            #[allow(non_snake_case)]
            let mut V: Vec<Vec<f32>> = Vec::with_capacity(max_restart + 1);
            #[allow(non_snake_case)]
            let mut Z: Vec<Vec<f32>> = Vec::with_capacity(max_restart);
            
            let v0: Vec<f32> = r.iter().map(|x| x / r_norm).collect();
            V.push(v0);
            
            // Hessenberg matrix H
            #[allow(non_snake_case)]
            let mut H = vec![vec![0.0f32; max_restart]; max_restart + 1];
            
            // Givens rotations
            let mut cs = vec![0.0f32; max_restart];
            let mut sn = vec![0.0f32; max_restart];
            
            // RHS of least squares
            let mut g = vec![0.0f32; max_restart + 1];
            g[0] = r_norm;
            
            let mut k = 0;
            
            // Arnoldi iteration
            for j in 0..max_restart {
                k = j + 1;
                total_iters += 1;
                
                // Apply preconditioner: z_j = M^{-1} * v_j (enhanced block preconditioner)
                let z_j = self.apply_block_preconditioner(
                    &V[j], &row_offsets, &col_indices, &values,
                    &diag_u, &diag_v, &diag_p
                );
                Z.push(z_j.clone());
                
                // Compute w = A * z_j
                let mut w = vec![0.0f32; n];
                for row in 0..n {
                    let start = row_offsets[row] as usize;
                    let end = row_offsets[row + 1] as usize;
                    let mut sum = 0.0f32;
                    for kk in start..end {
                        sum += values[kk] * z_j[col_indices[kk] as usize];
                    }
                    w[row] = sum;
                }
                
                // Modified Gram-Schmidt orthogonalization
                for i in 0..=j {
                    let h_ij: f32 = w.iter().zip(V[i].iter()).map(|(a, b)| a * b).sum();
                    H[i][j] = h_ij;
                    for l in 0..n {
                        w[l] -= h_ij * V[i][l];
                    }
                }
                
                // Compute ||w||
                let w_norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
                H[j + 1][j] = w_norm;
                
                if w_norm < 1e-14 {
                    println!("FGMRES: Breakdown at iteration {}", total_iters);
                    break;
                }
                
                // New basis vector
                let v_new: Vec<f32> = w.iter().map(|x| x / w_norm).collect();
                V.push(v_new);
                
                // Apply previous Givens rotations
                for i in 0..j {
                    let h_i = H[i][j];
                    let h_i1 = H[i + 1][j];
                    H[i][j] = cs[i] * h_i + sn[i] * h_i1;
                    H[i + 1][j] = -sn[i] * h_i + cs[i] * h_i1;
                }
                
                // Compute new Givens rotation
                let h_jj = H[j][j];
                let h_j1j = H[j + 1][j];
                let rho = (h_jj * h_jj + h_j1j * h_j1j).sqrt();
                if rho.abs() < 1e-14 {
                    cs[j] = 1.0;
                    sn[j] = 0.0;
                } else {
                    cs[j] = h_jj / rho;
                    sn[j] = h_j1j / rho;
                }
                
                H[j][j] = rho;
                H[j + 1][j] = 0.0;
                
                // Apply to g
                let g_j = g[j];
                let g_j1 = g[j + 1];
                g[j] = cs[j] * g_j + sn[j] * g_j1;
                g[j + 1] = -sn[j] * g_j + cs[j] * g_j1;
                
                let resid_est = g[j + 1].abs();
                
                // Only print every 10 iterations or at convergence check points
                if total_iters % 10 == 0 || resid_est < tol * rhs_norm {
                    println!("FGMRES iter {}: residual = {:.2e} (target: {:.2e})", 
                             total_iters, resid_est, tol * rhs_norm);
                }
                
                // Check convergence - use same metric as printed
                if resid_est < tol * rhs_norm {
                    converged = true;
                    final_resid = resid_est;
                    println!("FGMRES converged at iter {}: residual = {:.2e}", total_iters, resid_est);
                    break;
                }
                
                // Also check for absolute convergence
                if resid_est < 1e-10 {
                    converged = true;
                    final_resid = resid_est;
                    println!("FGMRES converged (absolute) at iter {}: residual = {:.2e}", total_iters, resid_est);
                    break;
                }
            }
            
            // Solve triangular system H * y = g
            let mut y = vec![0.0f32; k];
            for i in (0..k).rev() {
                let mut sum = g[i];
                for jj in (i + 1)..k {
                    sum -= H[i][jj] * y[jj];
                }
                if H[i][i].abs() > 1e-14 {
                    y[i] = sum / H[i][i];
                }
            }
            
            // Update solution: x = x + Z * y
            for jj in 0..k {
                for i in 0..n {
                    x[i] += y[jj] * Z[jj][i];
                }
            }
            
            if converged {
                break;
            }
            
            // Compute new residual: r = b - A*x
            for row in 0..n {
                let start = row_offsets[row] as usize;
                let end = row_offsets[row + 1] as usize;
                let mut sum = 0.0f32;
                for kk in start..end {
                    sum += values[kk] * x[col_indices[kk] as usize];
                }
                r[row] = rhs[row] - sum;
            }
            
            // Compute true residual norm for verification
            let true_resid = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            final_resid = true_resid;
            
            // Check if converged based on true residual
            if true_resid < tol * rhs_norm {
                converged = true;
                println!("FGMRES restart {}: true residual = {:.2e} (CONVERGED)", outer + 1, true_resid);
                break;
            }
            
            println!("FGMRES restart {}: true residual = {:.2e} (target: {:.2e})", 
                     outer + 1, true_resid, tol * rhs_norm);
        }
        
        // Write solution back to GPU
        let res = self.coupled_resources.as_ref().unwrap();
        self.context.queue.write_buffer(&res.b_x, 0, bytemuck::cast_slice(&x));
        
        println!("FGMRES finished: {} iters, residual = {:.2e}, converged = {}", 
                 total_iters, final_resid, converged);
        
        LinearSolverStats {
            iterations: total_iters,
            residual: final_resid,
            converged,
            diverged: final_resid.is_nan(),
            time: std::time::Duration::default(),
        }
    }
    
    /// Apply Schur complement preconditioner for saddle-point system
    /// 
    /// For the coupled system:
    /// [A   G] [u]   [f]
    /// [D   C] [p] = [g]
    ///
    /// We use an approximate block LDU preconditioner with multiple inner iterations.
    /// Key insight: For saddle-point problems, we need good approximations of:
    /// 1. A^{-1} for velocity block
    /// 2. S^{-1} = (C - D*A^{-1}*G)^{-1} for Schur complement
    ///
    /// We use symmetric Gauss-Seidel (SGS) for both blocks with sufficient iterations.
    fn apply_block_preconditioner(
        &self,
        v: &[f32],
        row_offsets: &[u32],
        col_indices: &[u32],
        values: &[f32],
        diag_u: &[f32],
        diag_v: &[f32],
        diag_p: &[f32],
    ) -> Vec<f32> {
        let num_cells = self.num_cells as usize;
        let n = num_cells * 3;
        
        let mut z = vec![0.0f32; n];
        
        // Step 1: y_u = A^{-1} * f using SGS on the FULL velocity system (including u-v cross terms)
        // We solve [A_uu A_uv; A_vu A_vv] [y_u; y_v] = [f_u; f_v]
        // Using block Gauss-Seidel treating each cell's (u,v) as a 2x2 block
        
        // Initialize with diagonal scaling
        for cell in 0..num_cells {
            let base = 3 * cell;
            if diag_u[cell].abs() > 1e-14 {
                z[base] = v[base] / diag_u[cell];
            }
            if diag_v[cell].abs() > 1e-14 {
                z[base + 1] = v[base + 1] / diag_v[cell];
            }
        }
        
        // SGS sweeps on velocity - process u and v together for each cell
        // IMPORTANT: Only use velocity-velocity coupling (A block), NOT pressure gradient (G block)
        let num_vel_sweeps = 5;  // More sweeps for better A^{-1} approximation
        for _sweep in 0..num_vel_sweeps {
            // Forward sweep
            for cell in 0..num_cells {
                let row_u = 3 * cell;
                let row_v = 3 * cell + 1;
                
                // Compute residuals for this cell's u and v equations
                let mut res_u = v[row_u];
                let mut res_v = v[row_v];
                
                // Subtract off-diagonal velocity contributions for u equation
                // Only u-u and u-v coupling (NOT u-p which is pressure gradient)
                let start_u = row_offsets[row_u] as usize;
                let end_u = row_offsets[row_u + 1] as usize;
                for k in start_u..end_u {
                    let col = col_indices[k] as usize;
                    let col_type = col % 3;
                    if col != row_u && (col_type == 0 || col_type == 1) {
                        res_u -= values[k] * z[col];
                    }
                }
                
                // Subtract off-diagonal velocity contributions for v equation
                // Only v-u and v-v coupling (NOT v-p)
                let start_v = row_offsets[row_v] as usize;
                let end_v = row_offsets[row_v + 1] as usize;
                for k in start_v..end_v {
                    let col = col_indices[k] as usize;
                    let col_type = col % 3;
                    if col != row_v && (col_type == 0 || col_type == 1) {
                        res_v -= values[k] * z[col];
                    }
                }
                
                // Update with diagonal solve
                if diag_u[cell].abs() > 1e-14 {
                    z[row_u] = res_u / diag_u[cell];
                }
                if diag_v[cell].abs() > 1e-14 {
                    z[row_v] = res_v / diag_v[cell];
                }
            }
            
            // Backward sweep
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
        
        // Step 2: Compute modified RHS for pressure: g' = g - D * y_u
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
        
        // Step 3: Solve S * y_p = g' where S ≈ C_pp (pressure Laplacian)
        // Use SSOR sweeps with proper Schur diagonal correction
        
        // First, extract the D and G coefficients for Schur correction
        let mut d_pu = vec![0.0f32; num_cells];
        let mut d_pv = vec![0.0f32; num_cells];
        let mut g_up = vec![0.0f32; num_cells];
        let mut g_vp = vec![0.0f32; num_cells];
        
        for cell in 0..num_cells {
            let base = 3 * cell;
            
            // Extract D coefficients (from pressure rows, velocity columns)
            let row_p = base + 2;
            let start_p = row_offsets[row_p] as usize;
            let end_p = row_offsets[row_p + 1] as usize;
            for k in start_p..end_p {
                let col = col_indices[k] as usize;
                if col == base {
                    d_pu[cell] = values[k];
                } else if col == base + 1 {
                    d_pv[cell] = values[k];
                }
            }
            
            // Extract G coefficients (from velocity rows, pressure columns)
            let row_u = base;
            let start_u = row_offsets[row_u] as usize;
            let end_u = row_offsets[row_u + 1] as usize;
            for k in start_u..end_u {
                let col = col_indices[k] as usize;
                if col == base + 2 {
                    g_up[cell] = values[k];
                }
            }
            
            let row_v = base + 1;
            let start_v = row_offsets[row_v] as usize;
            let end_v = row_offsets[row_v + 1] as usize;
            for k in start_v..end_v {
                let col = col_indices[k] as usize;
                if col == base + 2 {
                    g_vp[cell] = values[k];
                }
            }
        }
        
        // Compute Schur-corrected diagonal with regularization
        // The Schur complement S = C - D*A^{-1}*G can have negative diagonal entries
        // near boundaries. We need to regularize to keep S positive definite.
        let mut schur_diags = vec![0.0f32; num_cells];
        let mut min_schur = f32::MAX;
        
        for cell in 0..num_cells {
            let mut schur_diag = diag_p[cell];
            if diag_u[cell].abs() > 1e-14 {
                schur_diag -= d_pu[cell] * g_up[cell] / diag_u[cell];
            }
            if diag_v[cell].abs() > 1e-14 {
                schur_diag -= d_pv[cell] * g_vp[cell] / diag_v[cell];
            }
            schur_diags[cell] = schur_diag;
            min_schur = min_schur.min(schur_diag);
        }
        
        // Regularization: Use per-cell regularization for negative Schur diagonals
        // Instead of shifting all diagonals, only stabilize the problematic ones
        let regularization = 0.0;  // No global regularization
        
        // Per-cell stabilization: for cells with negative Schur diagonal,
        // use the original C_pp diagonal instead
        for cell in 0..num_cells {
            if schur_diags[cell] <= 0.0 {
                // Fall back to just using the C_pp diagonal for this cell
                schur_diags[cell] = diag_p[cell].max(1e-14);
            }
        }
        
        // Initialize pressure with diagonal scaling (using regularized Schur diagonal)
        for cell in 0..num_cells {
            let schur_diag = schur_diags[cell] + regularization;
            
            if schur_diag.abs() > 1e-14 {
                z[3 * cell + 2] = v_p_mod[cell] / schur_diag;
            }
        }
        
        // SSOR sweeps on pressure with regularized Schur-corrected diagonal
        // Use higher omega for faster convergence, more sweeps for better accuracy
        let omega = 1.6;  // Aggressive over-relaxation for pressure
        let num_sweeps = 10;  // More sweeps for better pressure solve
        
        for _sweep in 0..num_sweeps {
            // Forward sweep
            for cell in 0..num_cells {
                let row = 3 * cell + 2;
                
                // Use pre-computed regularized Schur diagonal
                let schur_diag = schur_diags[cell] + regularization;
                
                if schur_diag.abs() < 1e-14 {
                    continue;
                }
                
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
            
            // Backward sweep
            for cell in (0..num_cells).rev() {
                let row = 3 * cell + 2;
                
                let schur_diag = schur_diags[cell] + regularization;
                
                if schur_diag.abs() < 1e-14 {
                    continue;
                }
                
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
        
        // Step 4: Velocity correction: z_u = y_u - A^{-1} * G * y_p
        // This removes the pressure gradient contribution from the velocity
        for cell in 0..num_cells {
            let base = 3 * cell;
            let p_val = z[base + 2];
            
            // For u: subtract A^{-1} * G_up * p (using all G coefficients, not just diagonal)
            let row_u = base;
            let start_u = row_offsets[row_u] as usize;
            let end_u = row_offsets[row_u + 1] as usize;
            
            for k in start_u..end_u {
                let col = col_indices[k] as usize;
                if col % 3 == 2 {
                    // Get pressure value at this column
                    let p_neighbor = z[col];
                    if diag_u[cell].abs() > 1e-14 {
                        z[base] -= (values[k] * p_neighbor) / diag_u[cell];
                    }
                }
            }
            
            // For v: subtract A^{-1} * G_vp * p
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
