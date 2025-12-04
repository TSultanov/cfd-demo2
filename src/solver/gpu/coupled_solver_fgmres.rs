// DEPRECATED: Coupled Solver using FGMRES with Schur Complement Preconditioning
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
// - Preconditioner sweep
use super::async_buffer::AsyncScalarReader;
use super::profiling::ProfileCategory;
use super::structs::{CoupledSolverResources, GpuSolver, LinearSolverStats, PreconditionerParams};
use super::linear_solver::amg::{AmgResources, CsrMatrix};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use std::time::Instant;

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
    /// Temporary buffer for pressure RHS (r_p')
    pub b_temp_p: wgpu::Buffer,
    pub b_p_sol: wgpu::Buffer,
    /// Parameters buffer
    pub b_params: wgpu::Buffer,
    pub b_precond_params: wgpu::Buffer, // New buffer for PreconditionerParams
    /// Maximum restart dimension
    pub max_restart: usize,
    /// Number of workgroups for dot product
    pub num_dot_groups: u32,
    /// Vector bind group layout (x, y, z)
    pub bgl_vectors: wgpu::BindGroupLayout,
    /// Schur Vector bind group layout (r_in, z_out, temp_p, p_sol)
    pub bgl_schur_vectors: wgpu::BindGroupLayout,
    /// Matrix bind group layout (row offsets, indices, values)
    pub bgl_matrix: wgpu::BindGroupLayout,
    /// Preconditioner bind group layout (diag buffers)
    pub bgl_precond: wgpu::BindGroupLayout,
    /// Parameter/scalar bind group layout
    pub bgl_params: wgpu::BindGroupLayout,
    /// Bind group for matrix buffers
    pub bg_matrix: wgpu::BindGroup,
    /// Bind group for preconditioner buffers
    pub bg_precond: wgpu::BindGroup,
    /// Bind group for params/scalars
    pub bg_params: wgpu::BindGroup,
    /// Bind group for pressure matrix (Group 4)
    pub bg_pressure_matrix: wgpu::BindGroup,
    /// Compute pipeline for SpMV
    pub pipeline_spmv: wgpu::ComputePipeline,
    /// Compute pipeline for y = alpha x + y
    pub pipeline_axpy: wgpu::ComputePipeline,
    /// Compute pipeline for z = alpha x + beta y
    pub pipeline_axpby: wgpu::ComputePipeline,
    /// Compute pipeline for y = alpha x
    pub pipeline_scale: wgpu::ComputePipeline,
    /// Compute pipeline for in-place scaling y = alpha y
    pub pipeline_scale_in_place: wgpu::ComputePipeline,
    /// Compute pipeline for y = x
    pub pipeline_copy: wgpu::ComputePipeline,
    /// Compute pipeline for Jacobi preconditioner
    pub pipeline_precond: wgpu::ComputePipeline,
    /// Compute pipeline for dot partial reduction
    pub pipeline_dot_partial: wgpu::ComputePipeline,
    /// Compute pipeline for norm-squared reduction
    pub pipeline_norm_sq: wgpu::ComputePipeline,
    /// Compute pipeline for Gram-Schmidt update
    pub pipeline_orthogonalize: wgpu::ComputePipeline,
    /// Compute pipeline for diagonal extraction
    pub pipeline_extract_diag: wgpu::ComputePipeline,
    /// Schur Preconditioner Pipelines
    pub pipeline_predict_vel: wgpu::ComputePipeline,
    pub pipeline_form_schur: wgpu::ComputePipeline,
    pub pipeline_relax_pressure: wgpu::ComputePipeline,
    pub pipeline_correct_vel: wgpu::ComputePipeline,

    // New resources for GPU-only logic
    pub b_hessenberg: wgpu::Buffer,
    pub b_givens: wgpu::Buffer,
    pub b_g: wgpu::Buffer,
    pub b_y: wgpu::Buffer,
    pub b_iter_params: wgpu::Buffer,

    pub bgl_logic: wgpu::BindGroupLayout,
    pub bgl_logic_params: wgpu::BindGroupLayout,
    pub bg_logic: wgpu::BindGroup,
    pub bg_logic_params: wgpu::BindGroup,

    pub pipeline_reduce_final: wgpu::ComputePipeline,
    pub pipeline_update_hessenberg: wgpu::ComputePipeline,
    pub pipeline_solve_triangular: wgpu::ComputePipeline,
    pub pipeline_copy_scalar: wgpu::ComputePipeline,
    pub pipeline_finish_norm: wgpu::ComputePipeline,
    
    /// Async scalar reader for non-blocking convergence checks (interior mutability)
    pub async_scalar_reader: std::cell::RefCell<AsyncScalarReader>,
    
    /// Reusable staging buffer for scalar reads (avoids creating new buffers)
    pub b_staging_scalar: wgpu::Buffer,

    pub amg_resources: Option<AmgResources>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawFgmresParams {
    n: u32,
    num_cells: u32,
    num_iters: u32,
    omega: f32,
    dispatch_x: u32, // Width of 2D dispatch (in workgroups * 64)
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IterParams {
    current_idx: u32,
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}

impl GpuSolver {
    fn ensure_fgmres_resources(&mut self, max_restart: usize) {
        let rebuild = match &self.fgmres_resources {
            Some(existing) => existing.max_restart < max_restart,
            None => true,
        };

        if rebuild {
            let resources = self.init_fgmres_resources(max_restart);
            self.fgmres_resources = Some(resources);
        }
    }

    pub async fn ensure_amg_resources(&mut self) {
        let needs_init = if let Some(fgmres) = &self.fgmres_resources {
            fgmres.amg_resources.is_none()
        } else {
            false
        };

        if needs_init {
            let row_offsets = self.read_buffer_u32(&self.b_row_offsets, self.num_cells + 1).await;
            let col_indices = self.read_buffer_u32(&self.b_col_indices, self.num_nonzeros).await;
            let values = self.read_buffer_f32(&self.b_matrix_values, self.num_nonzeros).await;
            
            let matrix = CsrMatrix {
                row_offsets,
                col_indices,
                values,
                num_rows: self.num_cells as usize,
                num_cols: self.num_cells as usize,
            };
            
            // Use enough levels to reach a small coarse grid (< 100 cells)
            // For 1M cells with agg factor ~4, we need log4(1M/100) = log4(10000) = 6-7 levels.
            // 20 is a safe upper bound; it will stop early when size < 100.
            let amg = AmgResources::new(&self.context.device, &matrix, 20);
            
            if let Some(fgmres) = &mut self.fgmres_resources {
                fgmres.amg_resources = Some(amg);
            }
        }
    }

    /// Initialize FGMRES resources
    pub fn init_fgmres_resources(&self, max_restart: usize) -> FgmresResources {
        let n = self.num_cells * 3;
        let workgroup_size = 64u32;
        let num_groups = n.div_ceil(workgroup_size);
        let device = &self.context.device;
        let queue = &self.context.queue;
        let coupled = self
            .coupled_resources
            .as_ref()
            .expect("Coupled resources must be initialized before FGMRES");

        // Krylov buffers
        let mut basis_vectors = Vec::with_capacity(max_restart + 1);
        for i in 0..=max_restart {
            basis_vectors.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("FGMRES V_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }

        let mut z_vectors = Vec::with_capacity(max_restart);
        for i in 0..max_restart {
            z_vectors.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("FGMRES Z_{}", i)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }

        let b_w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES w"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES temp"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_dot_partial = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES dot partial"),
            size: (num_groups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_scalars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES scalars"),
            size: 16 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_size = (self.num_cells as u64) * 4;
        let create_diag = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cell_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let b_diag_u = create_diag("FGMRES diag_u");
        let b_diag_v = create_diag("FGMRES diag_v");
        let b_diag_p = create_diag("FGMRES diag_p");
        let b_temp_p = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES temp_p"),
            size: (self.num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_p_sol = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES p_sol"),
            size: (self.num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES params"),
            size: std::mem::size_of::<RawFgmresParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_precond_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES Precond Params"),
            size: std::mem::size_of::<PreconditionerParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let workgroups = self.workgroups_for_size(n);
        let dispatch_x = self.dispatch_x_threads(workgroups);
        let params = RawFgmresParams {
            n,
            num_cells: self.num_cells,
            num_iters: 2,
            omega: 1.0,
            dispatch_x,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&b_params, 0, bytes_of(&params));

        let b_hessenberg = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES Hessenberg"),
            size: ((max_restart + 1) * max_restart * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_givens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES Givens"),
            size: (max_restart * 2 * 4) as u64, // vec2<f32> per restart
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_g = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES RHS g"),
            size: ((max_restart + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES Solution y"),
            size: (max_restart * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_iter_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FGMRES Iter Params"),
            size: std::mem::size_of::<IterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create Pressure Matrix Bind Group Layout
        let bgl_pressure_matrix =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FGMRES Pressure Matrix BGL"),
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

        // Bind group layouts
        let bgl_vectors = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES Vectors BGL"),
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

        let bgl_schur_vectors = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES Schur Vectors BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // temp_p (read/write)
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // p_sol (read/write)
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bgl_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES Matrix BGL"),
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

        // Create Precond/Params Bind Group Layout (Group 2)
        let bgl_precond = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES Precond/Params BGL"),
            entries: &[
                // Diagonals
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Update bgl_params to include iter_params and hessenberg
        let bgl_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES Params BGL"),
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
                // New bindings
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let bg_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FGMRES Matrix BG"),
            layout: &bgl_matrix,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: coupled.b_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coupled.b_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: coupled.b_matrix_values.as_entire_binding(),
                },
            ],
        });

        let bg_precond = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FGMRES Precond/Params BG"),
            layout: &bgl_precond,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_diag_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_diag_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_diag_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_precond_params.as_entire_binding(),
                },
            ],
        });

        let bg_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FGMRES Params BG"),
            layout: &bgl_params,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scalars.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_iter_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_hessenberg.as_entire_binding(),
                },
            ],
        });

        let bg_pressure_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FGMRES Pressure Matrix BG"),
            layout: &bgl_pressure_matrix,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.b_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.b_matrix_values.as_entire_binding(),
                },
            ],
        });

        // New Logic BGL
        let bgl_logic = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES Logic BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        // Logic Params BGL (Group 1 for logic shader)
        // It needs iter_params (Uniform) and scalars (Storage)
        // We can reuse bgl_params if we are careful, but bgl_params has 4 bindings now.
        // The logic shader expects Group 1 to have 2 bindings.
        // So we need a separate layout or modify the shader to match bgl_params.
        // Let's create a separate small layout for logic params to match gmres_logic.wgsl

        let bgl_logic_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FGMRES Logic Params BGL"),
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

        let bg_logic = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FGMRES Logic BG"),
            layout: &bgl_logic,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_hessenberg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_givens.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_y.as_entire_binding(),
                },
            ],
        });

        let bg_logic_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FGMRES Logic Params BG"),
            layout: &bgl_logic_params,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_iter_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scalars.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_gmres =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FGMRES GMRES Pipeline Layout"),
                bind_group_layouts: &[&bgl_vectors, &bgl_matrix, &bgl_precond, &bgl_params],
                push_constant_ranges: &[],
            });

        let pipeline_layout_logic =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FGMRES Logic Pipeline Layout"),
                bind_group_layouts: &[&bgl_logic, &bgl_logic_params],
                push_constant_ranges: &[],
            });

        let pipeline_layout_schur =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FGMRES Schur Pipeline Layout"),
                bind_group_layouts: &[
                    &bgl_schur_vectors,
                    &bgl_matrix,
                    &bgl_precond,
                    &bgl_pressure_matrix,
                ],
                push_constant_ranges: &[],
            });

        let shader_ops = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FGMRES Ops Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gmres_ops.wgsl").into()),
        });

        let shader_schur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Schur Precond Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/schur_precond.wgsl").into()),
        });

        let shader_logic = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FGMRES Logic Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gmres_logic.wgsl").into()),
        });

        let make_pipeline = |label: &str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_gmres),
                module: &shader_ops,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let make_schur_pipeline = |label: &str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_schur),
                module: &shader_schur,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let make_logic_pipeline = |label: &str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_logic),
                module: &shader_logic,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipeline_spmv = make_pipeline("FGMRES SpMV", "spmv");
        let pipeline_axpy = make_pipeline("FGMRES AXPY", "axpy");
        let pipeline_axpby = make_pipeline("FGMRES AXPBY", "axpby");
        let pipeline_scale = make_pipeline("FGMRES Scale", "scale");
        let pipeline_scale_in_place = make_pipeline("FGMRES Scale In Place", "scale_in_place");
        let pipeline_copy = make_pipeline("FGMRES Copy", "copy");
        // let pipeline_precond = make_pipeline("FGMRES Precond", "block_jacobi_precond"); // Replaced by Schur
        let pipeline_precond = make_schur_pipeline("Schur Predict", "predict_velocity"); // Placeholder for struct

        let pipeline_dot_partial = make_pipeline("FGMRES Dot Partial", "dot_product_partial");
        let pipeline_norm_sq = make_pipeline("FGMRES Norm Partial", "norm_sq_partial");
        let pipeline_orthogonalize = make_pipeline("FGMRES Orthogonalize", "orthogonalize");

        let pipeline_predict_vel = make_schur_pipeline("Schur Predict", "predict_velocity");
        let pipeline_form_schur = make_schur_pipeline("Schur Form RHS", "form_schur_rhs");
        let pipeline_relax_pressure = make_schur_pipeline("Schur Relax P", "relax_pressure");
        let pipeline_correct_vel = make_schur_pipeline("Schur Correct Vel", "correct_velocity");
        let pipeline_extract_diag = make_schur_pipeline("Schur Extract Diag", "extract_diagonals");

        let pipeline_reduce_final = make_pipeline("FGMRES Reduce Final", "reduce_final");
        let pipeline_update_hessenberg =
            make_logic_pipeline("FGMRES Update Hessenberg", "update_hessenberg_givens");
        let pipeline_solve_triangular =
            make_logic_pipeline("FGMRES Solve Triangular", "solve_triangular");
        let pipeline_copy_scalar = make_logic_pipeline("FGMRES Copy Scalar", "copy_scalar");
        let pipeline_finish_norm = make_logic_pipeline("FGMRES Finish Norm", "finish_norm");

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
            b_temp_p,
            b_p_sol,
            b_params,
            b_precond_params,
            max_restart,
            num_dot_groups: num_groups,
            bgl_vectors,
            bgl_schur_vectors,
            bgl_matrix,
            bgl_precond,
            bgl_params,
            bg_matrix,
            bg_precond,
            bg_params,
            bg_pressure_matrix,
            pipeline_spmv,
            pipeline_axpy,
            pipeline_axpby,
            pipeline_scale,
            pipeline_scale_in_place,
            pipeline_copy,
            pipeline_precond,
            pipeline_dot_partial,
            pipeline_norm_sq,
            pipeline_orthogonalize,
            pipeline_extract_diag,
            pipeline_predict_vel,
            pipeline_form_schur,
            pipeline_relax_pressure,
            pipeline_correct_vel,
            // New
            b_hessenberg,
            b_givens,
            b_g,
            b_y,
            b_iter_params,
            bgl_logic,
            bgl_logic_params,
            bg_logic,
            bg_logic_params,
            pipeline_reduce_final,
            pipeline_update_hessenberg,
            pipeline_solve_triangular,
            pipeline_copy_scalar,
            pipeline_finish_norm,
            async_scalar_reader: std::cell::RefCell::new(AsyncScalarReader::new(device, 4)),
            b_staging_scalar: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FGMRES Staging Scalar"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            amg_resources: None,
        }
    }

    const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;
    const WORKGROUP_SIZE: u32 = 64;

    fn workgroups_for_size(&self, n: u32) -> u32 {
        n.div_ceil(Self::WORKGROUP_SIZE)
    }

    /// Compute 2D dispatch dimensions for large workgroup counts.
    /// Returns (dispatch_x, dispatch_y) where total workgroups = dispatch_x * dispatch_y
    fn dispatch_2d(&self, workgroups: u32) -> (u32, u32) {
        if workgroups <= Self::MAX_WORKGROUPS_PER_DIMENSION {
            (workgroups, 1)
        } else {
            // Split into 2D grid: use square-ish dimensions
            let dispatch_y = workgroups.div_ceil(Self::MAX_WORKGROUPS_PER_DIMENSION);
            let dispatch_x = workgroups.div_ceil(dispatch_y);
            (dispatch_x, dispatch_y)
        }
    }

    /// Compute the dispatch_x value (in threads, not workgroups) for params
    fn dispatch_x_threads(&self, workgroups: u32) -> u32 {
        let (dispatch_x, _) = self.dispatch_2d(workgroups);
        dispatch_x * Self::WORKGROUP_SIZE
    }

    fn write_scalars(&self, fgmres: &FgmresResources, scalars: &[f32]) {
        let start = Instant::now();
        self.context
            .queue
            .write_buffer(&fgmres.b_scalars, 0, cast_slice(scalars));
        let bytes = (scalars.len() * 4) as u64;
        self.profiling_stats.record_location(
            "write_scalars",
            ProfileCategory::GpuWrite,
            start.elapsed(),
            bytes,
        );
    }

    fn create_vector_bind_group(
        &self,
        fgmres: &FgmresResources,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        z: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        self.context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &fgmres.bgl_vectors,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: z.as_entire_binding(),
                    },
                ],
            })
    }

    fn create_schur_vector_bind_group(
        &self,
        fgmres: &FgmresResources,
        r_in: &wgpu::Buffer,
        z_out: &wgpu::Buffer,
        temp_p: &wgpu::Buffer,
        p_sol: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        self.context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &fgmres.bgl_schur_vectors,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: r_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: z_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: temp_p.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_sol.as_entire_binding(),
                    },
                ],
            })
    }

    fn dispatch_vector_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresResources,
        vector_bg: &wgpu::BindGroup,
        group3_bg: &wgpu::BindGroup,
        workgroups: u32,
        label: &str,
    ) {
        let start = Instant::now();
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

        let (dispatch_x, dispatch_y) = self.dispatch_2d(workgroups);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, vector_bg, &[]);
            pass.set_bind_group(1, &fgmres.bg_matrix, &[]);
            pass.set_bind_group(2, &fgmres.bg_precond, &[]);
            pass.set_bind_group(3, group3_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.profiling_stats.record_location(
            label,
            ProfileCategory::GpuDispatch,
            start.elapsed(),
            0,
        );
    }

    /// Dispatch multiple iterations of a pipeline in a single command buffer submission.
    /// This reduces CPU-GPU synchronization overhead by batching dispatches.
    fn dispatch_vector_pipeline_batched(
        &self,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresResources,
        vector_bg: &wgpu::BindGroup,
        group3_bg: &wgpu::BindGroup,
        workgroups: u32,
        iterations: usize,
        label: &str,
    ) {
        let start = Instant::now();
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

        let (dispatch_x, dispatch_y) = self.dispatch_2d(workgroups);

        // Encode all iterations in a single command buffer
        for _ in 0..iterations {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, vector_bg, &[]);
            pass.set_bind_group(1, &fgmres.bg_matrix, &[]);
            pass.set_bind_group(2, &fgmres.bg_precond, &[]);
            pass.set_bind_group(3, group3_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.profiling_stats.record_location(
            label,
            ProfileCategory::GpuDispatch,
            start.elapsed(),
            0,
        );
    }

    /// Compute norm of a vector using GPU reduction and async read
    /// 
    /// This uses a batched approach: all GPU work (partial reduction, final reduction,
    /// and buffer copy) is submitted in a single command buffer, then we do an async
    /// map and wait. This avoids multiple sync points.
    fn gpu_norm(&self, fgmres: &FgmresResources, x: &wgpu::Buffer, n: u32) -> f32 {
        let start = Instant::now();
        
        let workgroups = self.workgroups_for_size(n);
        let (dispatch_x, dispatch_y) = self.dispatch_2d(workgroups);
        
        // Create bind groups
        let vector_bg = self.create_vector_bind_group(
            fgmres,
            x,
            &fgmres.b_temp,
            &fgmres.b_dot_partial,
            "FGMRES Norm BG",
        );
        
        let reduce_bg = self.create_vector_bind_group(
            fgmres,
            &fgmres.b_dot_partial,
            &fgmres.b_temp,
            &fgmres.b_temp,
            "FGMRES Reduce BG",
        );
        
        // Write params for partial reduction (n = vector size)
        let partial_params = RawFgmresParams {
            n,
            num_cells: self.num_cells,
            num_iters: 2,
            omega: 1.0,
            dispatch_x: self.dispatch_x_threads(workgroups),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        self.context
            .queue
            .write_buffer(&fgmres.b_params, 0, bytes_of(&partial_params));
        
        // Create a single encoder for all GPU work
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Norm Batched"),
        });
        
        // Pass 1: Partial norm-squared reduction
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Norm Partial"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fgmres.pipeline_norm_sq);
            pass.set_bind_group(0, &vector_bg, &[]);
            pass.set_bind_group(1, &fgmres.bg_matrix, &[]);
            pass.set_bind_group(2, &fgmres.bg_precond, &[]);
            pass.set_bind_group(3, &fgmres.bg_params, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        
        // Submit partial reduction first (params need to change)
        self.context.queue.submit(Some(encoder.finish()));
        
        // Write params for final reduction (n = num_dot_groups)
        let reduce_params = RawFgmresParams {
            n: fgmres.num_dot_groups,
            num_cells: 0,
            num_iters: 0,
            omega: 0.0,
            dispatch_x: Self::WORKGROUP_SIZE,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        self.context
            .queue
            .write_buffer(&fgmres.b_params, 0, bytes_of(&reduce_params));
        
        // Create encoder for final reduction + copy
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Norm Final + Copy"),
        });
        
        // Pass 2: Final reduction (sums partials, writes to scalars[0])
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Norm Reduce Final"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fgmres.pipeline_reduce_final);
            pass.set_bind_group(0, &reduce_bg, &[]);
            pass.set_bind_group(1, &fgmres.bg_matrix, &[]);
            pass.set_bind_group(2, &fgmres.bg_precond, &[]);
            pass.set_bind_group(3, &fgmres.bg_params, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        
        // Copy result to staging buffer (in same encoder)
        encoder.copy_buffer_to_buffer(&fgmres.b_scalars, 0, &fgmres.b_staging_scalar, 0, 4);
        
        // Submit final reduction + copy together
        self.context.queue.submit(Some(encoder.finish()));
        
        self.profiling_stats.record_location(
            "gpu_norm:dispatch",
            ProfileCategory::GpuDispatch,
            start.elapsed(),
            0,
        );
        
        // Restore params.n for future operations (non-blocking write)
        let restore_params = RawFgmresParams {
            n,
            num_cells: self.num_cells,
            num_iters: 2,
            omega: 1.0,
            dispatch_x: self.dispatch_x_threads(workgroups),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        self.context
            .queue
            .write_buffer(&fgmres.b_params, 0, bytes_of(&restore_params));
        
        // Async read the scalar
        let read_start = Instant::now();
        let norm_sq = self.read_scalar_async(fgmres);
        self.profiling_stats.record_location(
            "gpu_norm:read_scalar_async",
            ProfileCategory::GpuRead,
            read_start.elapsed(),
            4,
        );

        norm_sq.sqrt()
    }
    
    /// Async scalar read - starts async map immediately after submission
    /// and polls with yielding to allow other work to proceed
    fn read_scalar_async(&self, fgmres: &FgmresResources) -> f32 {
        // Start async map immediately (work is already submitted)
        let slice = fgmres.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        // Poll with a combination of spinning and yielding for efficiency
        let mut spin_count = 0;
        loop {
            // Poll the device to make progress on GPU work
            let _ = self.context.device.poll(wgpu::PollType::Poll);
            
            match rx.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(e)) => panic!("Buffer mapping failed: {:?}", e),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    spin_count += 1;
                    if spin_count < 100 {
                        // Spin briefly for low-latency cases
                        std::hint::spin_loop();
                    } else if spin_count < 1000 {
                        // Yield to other threads after initial spinning
                        std::thread::yield_now();
                    } else {
                        // Sleep briefly if taking too long (shouldn't happen normally)
                        std::thread::sleep(std::time::Duration::from_micros(10));
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("Channel disconnected");
                }
            }
        }
        
        // Read the value
        let data = slice.get_mapped_range();
        let value: f32 = *bytemuck::from_bytes(&data[0..4]);
        drop(data);
        fgmres.b_staging_scalar.unmap();
        
        value
    }

    fn compute_residual_into(
        &self,
        fgmres: &FgmresResources,
        res: &CoupledSolverResources,
        target: &wgpu::Buffer,
        workgroups: u32,
        n: u32,
    ) -> f32 {
        let spmv_bg = self.create_vector_bind_group(
            fgmres,
            &res.b_x,
            &fgmres.b_w,
            &fgmres.b_temp,
            "FGMRES Residual SpMV",
        );
        self.dispatch_vector_pipeline(
            &fgmres.pipeline_spmv,
            fgmres,
            &spmv_bg,
            &fgmres.bg_params,
            workgroups,
            "FGMRES Residual SpMV",
        );

        self.write_scalars(fgmres, &[1.0, -1.0]);
        let residual_bg = self.create_vector_bind_group(
            fgmres,
            &res.b_rhs,
            &fgmres.b_w,
            target,
            "FGMRES Residual Axpby",
        );
        self.dispatch_vector_pipeline(
            &fgmres.pipeline_axpby,
            fgmres,
            &residual_bg,
            &fgmres.bg_params,
            workgroups,
            "FGMRES Residual Axpby",
        );

        self.gpu_norm(fgmres, target, n)
    }

    fn scale_vector_in_place(
        &self,
        fgmres: &FgmresResources,
        buffer: &wgpu::Buffer,
        workgroups: u32,
        label: &str,
    ) {
        let vector_bg = self.create_vector_bind_group(
            fgmres,
            &fgmres.b_temp,
            buffer,
            &fgmres.b_dot_partial,
            label,
        );
        self.dispatch_vector_pipeline(
            &fgmres.pipeline_scale_in_place,
            fgmres,
            &vector_bg,
            &fgmres.bg_params,
            workgroups,
            label,
        );
    }

    /// Solve the coupled system using FGMRES with block preconditioning (GPU-accelerated)
    fn dispatch_logic_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresResources,
        workgroups: u32,
        label: &str,
    ) {
        let start = Instant::now();
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &fgmres.bg_logic, &[]);
            pass.set_bind_group(1, &fgmres.bg_logic_params, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.profiling_stats.record_location(
            label,
            ProfileCategory::GpuDispatch,
            start.elapsed(),
            0,
        );
    }

    /// Solve the coupled system using FGMRES with block preconditioning (GPU-accelerated)
    pub fn solve_coupled_fgmres(&mut self) -> LinearSolverStats {
        let start_time = Instant::now();
        if self.coupled_resources.is_none() {
            println!("Coupled resources not initialized!");
            return LinearSolverStats::default();
        }

        let num_cells = self.num_cells;
        let n = num_cells * 3;
        let max_restart = 50usize;
        let max_outer = 20usize;
        let tol = 1e-5f32;
        let abstol = 1e-7f32;

        self.ensure_fgmres_resources(max_restart);

        if self.constants.precond_type == 1 {
             pollster::block_on(self.ensure_amg_resources());
        }

        let Some(res) = &self.coupled_resources else {
            return LinearSolverStats::default();
        };
        let Some(fgmres) = &self.fgmres_resources else {
            return LinearSolverStats::default();
        };

        let workgroups_dofs = self.workgroups_for_size(n);
        let workgroups_cells = self.workgroups_for_size(num_cells);

        // Initialize IterParams
        let mut iter_params = IterParams {
            current_idx: 0,
            max_restart: max_restart as u32,
            _pad1: 0,
            _pad2: 0,
        };
        self.context
            .queue
            .write_buffer(&fgmres.b_iter_params, 0, bytes_of(&iter_params));

        let precond_params = PreconditionerParams {
            n: 0,
            num_cells: self.num_cells,
            omega: 0.7,
            precond_type: self.constants.precond_type,
        };
        self.context.queue.write_buffer(
            &fgmres.b_precond_params,
            0,
            bytemuck::bytes_of(&precond_params),
        );

        // Refresh block diagonals
        let diag_bg = self.create_schur_vector_bind_group(
            fgmres,
            &res.b_rhs,
            &fgmres.b_temp,
            &fgmres.b_w,
            &fgmres.b_p_sol,
            "FGMRES Extract Diagonal BG",
        );
        self.dispatch_vector_pipeline(
            &fgmres.pipeline_extract_diag,
            fgmres,
            &diag_bg,
            &fgmres.bg_pressure_matrix,
            workgroups_cells,
            "Schur Extract Diag",
        );

        let rhs_norm = self.gpu_norm(fgmres, &res.b_rhs, n);
        if rhs_norm < abstol || !rhs_norm.is_finite() {
            return LinearSolverStats {
                iterations: 0,
                residual: rhs_norm,
                converged: rhs_norm < abstol,
                diverged: !rhs_norm.is_finite(),
                time: start_time.elapsed(),
            };
        }

        let h_idx = |row: usize, col: usize| -> usize { col * (max_restart + 1) + row };

        // Initial residual r = b - A x stored in V_0
        let mut residual_norm =
            self.compute_residual_into(fgmres, res, &fgmres.basis_vectors[0], workgroups_dofs, n);

        let target_resid = (tol * rhs_norm).max(abstol);

        if residual_norm < target_resid {
            println!(
                "FGMRES: Initial guess already converged (||r|| = {:.2e} < {:.2e})",
                residual_norm, target_resid
            );
            return LinearSolverStats {
                iterations: 0,
                residual: residual_norm,
                converged: true,
                diverged: false,
                time: start_time.elapsed(),
            };
        }

        // Normalize V_0
        self.write_scalars(fgmres, &[1.0 / residual_norm]);
        self.scale_vector_in_place(
            fgmres,
            &fgmres.basis_vectors[0],
            workgroups_dofs,
            "FGMRES Normalize V0",
        );

        // Initialize g on GPU
        let mut g_initial = vec![0.0f32; max_restart + 1];
        g_initial[0] = residual_norm;
        self.context
            .queue
            .write_buffer(&fgmres.b_g, 0, cast_slice(&g_initial));

        let mut total_iters = 0u32;
        let mut final_resid = residual_norm;
        let mut converged = false;

        println!("FGMRES: Initial residual = {:.2e}", residual_norm);

        let mut stagnation_count = 0;
        let mut prev_resid_norm = residual_norm;

        'outer: for outer_iter in 0..max_outer {
            let mut basis_size = 0usize;

            for j in 0..max_restart {
                basis_size = j + 1;
                total_iters += 1;

                // 1. Predict Velocity
                // Clear p_sol before starting
                {
                    let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Clear p_sol"),
                    });
                    encoder.clear_buffer(&fgmres.b_p_sol, 0, None);
                    self.context.queue.submit(Some(encoder.finish()));
                }

                let precond_bg = self.create_schur_vector_bind_group(
                    fgmres,
                    &fgmres.basis_vectors[j],
                    &fgmres.z_vectors[j],
                    &fgmres.b_temp_p,
                    &fgmres.b_p_sol,
                    "FGMRES Schur BG",
                );

                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_predict_vel,
                    fgmres,
                    &precond_bg,
                    &fgmres.bg_pressure_matrix,
                    workgroups_cells,
                    "Schur Predict",
                );
                // 2. Form Schur RHS
                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_form_schur,
                    fgmres,
                    &precond_bg,
                    &fgmres.bg_pressure_matrix,
                    workgroups_cells,
                    "Schur Form RHS",
                );

                // 3. Relax Pressure
                if self.constants.precond_type == 1 {
                    if let Some(amg) = &self.fgmres_resources.as_ref().unwrap().amg_resources {
                        let level0 = &amg.levels[0];
                        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("AMG Setup") });
                        
                        encoder.copy_buffer_to_buffer(&fgmres.b_temp_p, 0, &level0.b_b, 0, (self.num_cells as u64) * 4);
                        encoder.copy_buffer_to_buffer(&fgmres.b_p_sol, 0, &level0.b_x, 0, (self.num_cells as u64) * 4);
                        
                        amg.v_cycle(&mut encoder, &self.context.device);
                        
                        encoder.copy_buffer_to_buffer(&level0.b_x, 0, &fgmres.b_p_sol, 0, (self.num_cells as u64) * 4);
                        
                        self.context.queue.submit(Some(encoder.finish()));
                    }
                } else {
                    // Scale iterations with mesh size: Jacobi convergence degrades with larger meshes.
                    // For a mesh of N cells, we need approximately sqrt(N) iterations for good convergence.
                    // Use a minimum of 20 and cap at 200 to balance cost vs. quality.
                    let p_iters = (20 + (num_cells as f32).sqrt() as usize / 2).min(200);
                    self.dispatch_vector_pipeline_batched(
                        &fgmres.pipeline_relax_pressure,
                        fgmres,
                        &precond_bg,
                        &fgmres.bg_pressure_matrix,
                        workgroups_cells,
                        p_iters,
                        "Schur Relax P",
                    );
                }

                // 4. Correct Velocity
                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_correct_vel,
                    fgmres,
                    &precond_bg,
                    &fgmres.bg_pressure_matrix,
                    workgroups_cells,
                    "Schur Correct Vel",
                );

                // w = A * z_j
                let spmv_bg = self.create_vector_bind_group(
                    fgmres,
                    &fgmres.z_vectors[j],
                    &fgmres.b_w,
                    &fgmres.b_temp,
                    "FGMRES SpMV",
                );
                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_spmv,
                    fgmres,
                    &spmv_bg,
                    &fgmres.bg_params,
                    workgroups_dofs,
                    "FGMRES SpMV",
                );

                // Modified Gram-Schmidt on GPU
                for i in 0..=j {
                    // Update iter_params for i
                    iter_params.current_idx = h_idx(i, j) as u32;
                    self.context.queue.write_buffer(
                        &fgmres.b_iter_params,
                        0,
                        bytes_of(&iter_params),
                    );

                    // 1. Dot product partial
                    let dot_bg = self.create_vector_bind_group(
                        fgmres,
                        &fgmres.b_w,
                        &fgmres.basis_vectors[i],
                        &fgmres.b_dot_partial,
                        "FGMRES Dot Partial BG",
                    );
                    self.dispatch_vector_pipeline(
                        &fgmres.pipeline_dot_partial,
                        fgmres,
                        &dot_bg,
                        &fgmres.bg_params,
                        fgmres.num_dot_groups,
                        "FGMRES Dot Partial",
                    );

                    // 2. Reduce Final (sums partials, writes to H[i,j] and scalars[0])
                    let reduce_bg = self.create_vector_bind_group(
                        fgmres,
                        &fgmres.b_dot_partial,
                        &fgmres.b_temp,
                        &fgmres.b_temp,
                        "FGMRES Reduce BG",
                    );

                    let reduce_params = RawFgmresParams {
                        n: fgmres.num_dot_groups,
                        num_cells: 0,
                        num_iters: 0,
                        omega: 0.0,
                        dispatch_x: Self::WORKGROUP_SIZE, // Single workgroup dispatch
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    };
                    self.context
                        .queue
                        .write_buffer(&fgmres.b_params, 0, bytes_of(&reduce_params));

                    self.dispatch_vector_pipeline(
                        &fgmres.pipeline_reduce_final,
                        fgmres,
                        &reduce_bg,
                        &fgmres.bg_params,
                        1,
                        "FGMRES Reduce Final",
                    );

                    // Restore params.n
                    let restore_params = RawFgmresParams {
                        n,
                        num_cells: self.num_cells,
                        num_iters: 2,
                        omega: 1.0,
                        dispatch_x: self.dispatch_x_threads(workgroups_dofs),
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    };
                    self.context
                        .queue
                        .write_buffer(&fgmres.b_params, 0, bytes_of(&restore_params));

                    // 3. Orthogonalize
                    let ortho_bg = self.create_vector_bind_group(
                        fgmres,
                        &fgmres.basis_vectors[i],
                        &fgmres.b_w,
                        &fgmres.b_temp,
                        "FGMRES Orthogonalize",
                    );
                    self.dispatch_vector_pipeline(
                        &fgmres.pipeline_orthogonalize,
                        fgmres,
                        &ortho_bg,
                        &fgmres.bg_params,
                        workgroups_dofs,
                        "FGMRES Orthogonalize",
                    );
                }

                // Compute norm of w (H[j+1, j])
                iter_params.current_idx = h_idx(j + 1, j) as u32;
                self.context
                    .queue
                    .write_buffer(&fgmres.b_iter_params, 0, bytes_of(&iter_params));

                // Norm squared partial
                let norm_bg = self.create_vector_bind_group(
                    fgmres,
                    &fgmres.b_w,
                    &fgmres.b_temp,
                    &fgmres.b_dot_partial,
                    "FGMRES Norm BG",
                );
                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_norm_sq,
                    fgmres,
                    &norm_bg,
                    &fgmres.bg_params,
                    fgmres.num_dot_groups,
                    "FGMRES Norm Partial",
                );

                // Reduce Final (writes to scalars[0])
                let reduce_params = RawFgmresParams {
                    n: fgmres.num_dot_groups,
                    num_cells: 0,
                    num_iters: 0,
                    omega: 0.0,
                    dispatch_x: Self::WORKGROUP_SIZE, // Single workgroup dispatch
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                };
                self.context
                    .queue
                    .write_buffer(&fgmres.b_params, 0, bytes_of(&reduce_params));

                let reduce_bg = self.create_vector_bind_group(
                    fgmres,
                    &fgmres.b_dot_partial,
                    &fgmres.b_temp,
                    &fgmres.b_temp,
                    "FGMRES Reduce BG",
                );

                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_reduce_final,
                    fgmres,
                    &reduce_bg,
                    &fgmres.bg_params,
                    1,
                    "FGMRES Reduce Final",
                );

                // Finish Norm (sqrt, write to H, write 1/norm to scalars[0])
                self.dispatch_logic_pipeline(
                    &fgmres.pipeline_finish_norm,
                    fgmres,
                    1,
                    "FGMRES Finish Norm",
                );

                // Restore params.n
                let restore_params = RawFgmresParams {
                    n,
                    num_cells: self.num_cells,
                    num_iters: 2,
                    omega: 1.0,
                    dispatch_x: self.dispatch_x_threads(workgroups_dofs),
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                };
                self.context
                    .queue
                    .write_buffer(&fgmres.b_params, 0, bytes_of(&restore_params));

                // Normalize w (scale in place using scalars[0])
                self.scale_vector_in_place(
                    fgmres,
                    &fgmres.b_w,
                    workgroups_dofs,
                    "FGMRES Normalize w",
                );

                let copy_bg = self.create_vector_bind_group(
                    fgmres,
                    &fgmres.b_w,
                    &fgmres.basis_vectors[j + 1],
                    &fgmres.b_temp,
                    "FGMRES Copy w",
                );
                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_copy,
                    fgmres,
                    &copy_bg,
                    &fgmres.bg_params,
                    workgroups_dofs,
                    "FGMRES Copy w",
                );

                // Update Hessenberg and Givens (GPU)
                iter_params.current_idx = j as u32;
                let write_start = Instant::now();
                self.context
                    .queue
                    .write_buffer(&fgmres.b_iter_params, 0, bytes_of(&iter_params));
                self.profiling_stats.record_location(
                    "fgmres:write_iter_params",
                    ProfileCategory::GpuWrite,
                    write_start.elapsed(),
                    std::mem::size_of::<IterParams>() as u64,
                );

                self.dispatch_logic_pipeline(
                    &fgmres.pipeline_update_hessenberg,
                    fgmres,
                    1,
                    "FGMRES Update Hessenberg",
                );

                // Async convergence checking - use non-blocking poll
                // Poll the device without waiting (allows GPU work to continue)
                let _ = self.context.device.poll(wgpu::PollType::Poll);
                
                // Check convergence every few iterations to balance early termination vs overhead.
                // Lower values detect convergence earlier but have slightly more overhead.
                // The async buffer now handles back-pressure safely, so any value >= 1 is safe.
                let check_interval = 1;
                let is_last_in_restart = j == max_restart - 1;
                let should_check = (j + 1) % check_interval == 0 || is_last_in_restart;

                if should_check {
                    // Use RefCell to get mutable access to async reader
                    let mut async_reader = fgmres.async_scalar_reader.borrow_mut();
                    
                    // Poll for completed reads from previous iterations
                    async_reader.poll();
                    
                    // Start async read for this iteration's residual
                    let read_start = Instant::now();
                    async_reader.start_read(
                        &self.context.device,
                        &self.context.queue,
                        &fgmres.b_scalars,
                        0,
                    );
                    self.profiling_stats.record_location(
                        "fgmres:convergence_check_start_async",
                        ProfileCategory::Other,
                        read_start.elapsed(),
                        0,
                    );
                    
                    // Check the result from PREVIOUS async read (if available)
                    // This allows GPU work to continue while we wait
                    if let Some(resid_est) = async_reader.get_last_value() {
                        if total_iters % 10 == 0 || resid_est < tol * rhs_norm {
                            println!(
                                "FGMRES iter {}: residual = {:.2e} (target {:.2e})",
                                total_iters,
                                resid_est,
                                tol * rhs_norm
                            );
                        }

                        if resid_est < tol * rhs_norm {
                            converged = true;
                            break;
                        }
                    }
                }
            }

            // Solve upper triangular system (GPU)
            iter_params.current_idx = basis_size as u32;
            self.context
                .queue
                .write_buffer(&fgmres.b_iter_params, 0, bytes_of(&iter_params));

            self.dispatch_logic_pipeline(
                &fgmres.pipeline_solve_triangular,
                fgmres,
                1,
                "FGMRES Solve Triangular",
            );

            // Update solution x = x + sum_j y_j * z_j
            for i in 0..basis_size {
                // Copy y[i] to scalars[0]
                iter_params.current_idx = i as u32;
                self.context
                    .queue
                    .write_buffer(&fgmres.b_iter_params, 0, bytes_of(&iter_params));

                self.dispatch_logic_pipeline(
                    &fgmres.pipeline_copy_scalar,
                    fgmres,
                    1,
                    "FGMRES Copy Scalar",
                );

                let axpy_bg = self.create_vector_bind_group(
                    fgmres,
                    &fgmres.z_vectors[i],
                    &res.b_x,
                    &fgmres.b_temp,
                    "FGMRES Solution Update",
                );
                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_axpy,
                    fgmres,
                    &axpy_bg,
                    &fgmres.bg_params,
                    workgroups_dofs,
                    "FGMRES Solution Update",
                );
            }

            // If already converged from estimated residual, skip true residual computation
            if converged {
                // Read the estimated residual for reporting
                let mut async_reader = fgmres.async_scalar_reader.borrow_mut();
                async_reader.flush(&self.context.device);
                let resid_est = async_reader.get_last_value().unwrap_or(0.0);
                final_resid = resid_est;
                println!(
                    "FGMRES restart {}: estimated residual = {:.2e}",
                    outer_iter + 1,
                    resid_est
                );
                break 'outer;
            }

            // Compute true residual (only when not already converged)
            residual_norm = self.compute_residual_into(
                fgmres,
                res,
                &fgmres.basis_vectors[0],
                workgroups_dofs,
                n,
            );
            final_resid = residual_norm;

            if residual_norm < tol * rhs_norm {
                converged = true;
                println!(
                    "FGMRES restart {}: true residual = {:.2e} (converged)",
                    outer_iter + 1,
                    residual_norm
                );
                break 'outer;
            }

            // Prepare for restart
            // Reset g on GPU
            let mut g_initial = vec![0.0f32; max_restart + 1];
            g_initial[0] = residual_norm;
            self.context
                .queue
                .write_buffer(&fgmres.b_g, 0, cast_slice(&g_initial));

            if residual_norm <= 0.0 {
                println!("FGMRES: residual vanished at restart {}", outer_iter + 1);
                converged = true;
                break;
            }

            self.write_scalars(fgmres, &[1.0 / residual_norm]);
            self.scale_vector_in_place(
                fgmres,
                &fgmres.basis_vectors[0],
                workgroups_dofs,
                "FGMRES Restart Normalize",
            );

            // Stagnation detection
            let improvement = (prev_resid_norm - residual_norm) / prev_resid_norm;
            if improvement < 1e-3 {
                stagnation_count += 1;
                if stagnation_count >= 3 {
                    println!(
                        "FGMRES: Stagnation detected at restart {} (residual {:.2e})",
                        outer_iter + 1,
                        residual_norm
                    );
                    converged = true;
                    break 'outer;
                }
            } else {
                stagnation_count = 0;
            }
            prev_resid_norm = residual_norm;

            println!(
                "FGMRES restart {}: residual = {:.2e} (target {:.2e})",
                outer_iter + 1,
                residual_norm,
                tol * rhs_norm
            );
        }

        println!(
            "FGMRES finished: {} iterations, residual = {:.2e}, converged = {}",
            total_iters, final_resid, converged
        );

        LinearSolverStats {
            iterations: total_iters,
            residual: final_resid,
            converged,
            diverged: final_resid.is_nan(),
            time: start_time.elapsed(),
        }
    }
}
