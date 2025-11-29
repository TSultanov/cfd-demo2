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
// - Preconditioner sweepuse super::context::GpuContext;
use super::structs::{CoupledSolverResources, GpuSolver, LinearSolverStats, PreconditionerParams};
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
    /// Parameters buffer
    pub b_params: wgpu::Buffer,
    pub b_precond_params: wgpu::Buffer, // New buffer for PreconditionerParams
    /// Maximum restart dimension
    pub max_restart: usize,
    /// Number of workgroups for dot product
    pub num_dot_groups: u32,
    /// Vector bind group layout (x, y, z)
    pub bgl_vectors: wgpu::BindGroupLayout,
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
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawFgmresParams {
    n: u32,
    num_cells: u32,
    num_iters: u32,
    omega: f32,
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
        let params = RawFgmresParams {
            n,
            num_cells: self.num_cells,
            num_iters: 2,
            omega: 1.0,
        };
        queue.write_buffer(&b_params, 0, bytes_of(&params));

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
            ],
        });

        let pipeline_layout_gmres =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FGMRES GMRES Pipeline Layout"),
                bind_group_layouts: &[&bgl_vectors, &bgl_matrix, &bgl_precond, &bgl_params],
                push_constant_ranges: &[],
            });

        let pipeline_layout_schur =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FGMRES Schur Pipeline Layout"),
                bind_group_layouts: &[
                    &bgl_vectors,
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

        let make_pipeline = |label: &str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_gmres),
                module: &shader_ops,
                entry_point: entry,
            })
        };

        let make_schur_pipeline = |label: &str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_schur),
                module: &shader_schur,
                entry_point: entry,
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
            b_params,
            b_precond_params,
            max_restart,
            num_dot_groups: num_groups,
            bgl_vectors,
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
        }
    }

    fn workgroups_for_size(&self, n: u32) -> u32 {
        let workgroup_size = 64u32;
        n.div_ceil(workgroup_size)
    }

    fn write_scalars(&self, fgmres: &FgmresResources, scalars: &[f32]) {
        self.context
            .queue
            .write_buffer(&fgmres.b_scalars, 0, cast_slice(scalars));
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

    fn dispatch_vector_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresResources,
        vector_bg: &wgpu::BindGroup,
        group3_bg: &wgpu::BindGroup,
        workgroups: u32,
        label: &str,
    ) {
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
            pass.set_bind_group(0, vector_bg, &[]);
            pass.set_bind_group(1, &fgmres.bg_matrix, &[]);
            pass.set_bind_group(2, &fgmres.bg_precond, &[]);
            pass.set_bind_group(3, group3_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.context.queue.submit(Some(encoder.finish()));
    }

    fn gpu_dot(&self, fgmres: &FgmresResources, x: &wgpu::Buffer, y: &wgpu::Buffer, n: u32) -> f32 {
        let vector_bg =
            self.create_vector_bind_group(fgmres, x, y, &fgmres.b_dot_partial, "FGMRES Dot BG");
        let workgroups = self.workgroups_for_size(n);
        self.dispatch_vector_pipeline(
            &fgmres.pipeline_dot_partial,
            fgmres,
            &vector_bg,
            &fgmres.bg_params,
            workgroups,
            "FGMRES Dot",
        );

        let partial =
            pollster::block_on(self.read_buffer_f32(&fgmres.b_dot_partial, fgmres.num_dot_groups));
        partial.iter().take(workgroups as usize).sum()
    }

    fn gpu_norm(&self, fgmres: &FgmresResources, x: &wgpu::Buffer, n: u32) -> f32 {
        let vector_bg = self.create_vector_bind_group(
            fgmres,
            x,
            &fgmres.b_temp,
            &fgmres.b_dot_partial,
            "FGMRES Norm BG",
        );
        let workgroups = self.workgroups_for_size(n);
        self.dispatch_vector_pipeline(
            &fgmres.pipeline_norm_sq,
            fgmres,
            &vector_bg,
            &fgmres.bg_params,
            workgroups,
            "FGMRES Norm",
        );

        let partial =
            pollster::block_on(self.read_buffer_f32(&fgmres.b_dot_partial, fgmres.num_dot_groups));
        partial.iter().take(workgroups as usize).sum::<f32>().sqrt()
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
        let tol = 1e-6f32;
        let abstol = 1e-8f32;

        self.ensure_fgmres_resources(max_restart);
        let Some(res) = &self.coupled_resources else {
            println!("Coupled resources not initialized!");
            return LinearSolverStats::default();
        };
        let Some(fgmres) = &self.fgmres_resources else {
            println!("FGMRES resources not initialized!");
            return LinearSolverStats::default();
        };

        let workgroups_dofs = self.workgroups_for_size(n);
        let workgroups_cells = self.workgroups_for_size(num_cells);

        let precond_params = PreconditionerParams {
            n: 0, // Updated per iteration if needed, but Schur doesn't use n for now
            num_cells: self.num_cells,
            omega: 0.7, // Pressure relaxation factor
            _pad: 0,
        };
        self.context.queue.write_buffer(
            &fgmres.b_precond_params,
            0,
            bytemuck::bytes_of(&precond_params),
        );

        // Refresh block diagonals for the current coupled matrix
        let diag_bg = self.create_vector_bind_group(
            fgmres,
            &res.b_rhs,
            &fgmres.b_temp,
            &fgmres.b_w,
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
            // println!("FGMRES: RHS norm is {:.2e} - nothing to solve", rhs_norm);
            return LinearSolverStats {
                iterations: 0,
                residual: rhs_norm,
                converged: rhs_norm < abstol,
                diverged: !rhs_norm.is_finite(),
                time: start_time.elapsed(),
            };
        }

        // Storage for Hessenberg matrix and Givens rotations (column-major H)
        let mut h = vec![0.0f32; (max_restart + 1) * max_restart];
        let mut cs = vec![0.0f32; max_restart];
        let mut sn = vec![0.0f32; max_restart];
        let mut g = vec![0.0f32; max_restart + 1];
        let mut y = vec![0.0f32; max_restart];

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

        g.fill(0.0);
        g[0] = residual_norm;
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

                // Schur Complement Preconditioner Steps

                // 1. Predict Velocity: z_u = D_u^{-1} r_u
                let precond_bg = self.create_vector_bind_group(
                    fgmres,
                    &fgmres.basis_vectors[j], // Binding 0: r_in
                    &fgmres.z_vectors[j],     // Binding 1: z_out
                    &fgmres.b_temp_p,         // Binding 2: temp_p
                    "FGMRES Preconditioner BG",
                );

                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_predict_vel,
                    fgmres,
                    &precond_bg,
                    &fgmres.bg_pressure_matrix,
                    workgroups_cells,
                    "Schur Predict",
                );
                // 2. Form Schur RHS: r_p' = r_p - D z_u
                self.dispatch_vector_pipeline(
                    &fgmres.pipeline_form_schur,
                    fgmres,
                    &precond_bg,
                    &fgmres.bg_pressure_matrix,
                    workgroups_cells,
                    "Schur Form RHS",
                );

                // 3. Relax Pressure: Solve S z_p = r_p'
                // Run a few iterations of Jacobi
                let p_iters = 20;
                for _ in 0..p_iters {
                    self.dispatch_vector_pipeline(
                        &fgmres.pipeline_relax_pressure,
                        fgmres,
                        &precond_bg,
                        &fgmres.bg_pressure_matrix,
                        workgroups_cells,
                        "Schur Relax P",
                    );
                }

                // 4. Correct Velocity: z_u = z_u - D_u^{-1} G z_p
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

                // Modified Gram-Schmidt
                for i in 0..=j {
                    let hij = self.gpu_dot(fgmres, &fgmres.b_w, &fgmres.basis_vectors[i], n);
                    h[h_idx(i, j)] = hij;

                    self.write_scalars(fgmres, &[hij]);
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

                let w_norm = self.gpu_norm(fgmres, &fgmres.b_w, n);
                h[h_idx(j + 1, j)] = w_norm;

                if w_norm < 1e-12 {
                    println!("FGMRES: Happy breakdown at iter {}", total_iters);
                    basis_size = j + 1;
                    break;
                }

                // Normalize w and store as next basis vector
                self.write_scalars(fgmres, &[1.0 / w_norm]);
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

                // Apply previous Givens rotations to new column
                for i in 0..j {
                    let h_ij = h[h_idx(i, j)];
                    let h_i1j = h[h_idx(i + 1, j)];
                    h[h_idx(i, j)] = cs[i] * h_ij + sn[i] * h_i1j;
                    h[h_idx(i + 1, j)] = -sn[i] * h_ij + cs[i] * h_i1j;
                }

                // Compute and apply new Givens rotation
                let h_jj = h[h_idx(j, j)];
                let h_j1j = h[h_idx(j + 1, j)];
                let rho = (h_jj * h_jj + h_j1j * h_j1j).sqrt();
                if rho.abs() < 1e-20 {
                    cs[j] = 1.0;
                    sn[j] = 0.0;
                } else {
                    cs[j] = h_jj / rho;
                    sn[j] = h_j1j / rho;
                }
                h[h_idx(j, j)] = rho;
                h[h_idx(j + 1, j)] = 0.0;

                let g_j = g[j];
                let g_j1 = g[j + 1];
                g[j] = cs[j] * g_j + sn[j] * g_j1;
                g[j + 1] = -sn[j] * g_j + cs[j] * g_j1;

                let resid_est = g[j + 1].abs();
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

            // Solve upper triangular system H * y = g
            for i in (0..basis_size).rev() {
                let mut sum = g[i];
                for j in (i + 1)..basis_size {
                    sum -= h[h_idx(i, j)] * y[j];
                }
                let diag = h[h_idx(i, i)];
                if diag.abs() > 1e-12 {
                    y[i] = sum / diag;
                } else {
                    y[i] = 0.0;
                }
            }

            // Update solution x = x + sum_j y_j * z_j
            for i in 0..basis_size {
                self.write_scalars(fgmres, &[y[i]]);
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

            if converged {
                println!(
                    "FGMRES restart {}: estimated residual = {:.2e}",
                    outer_iter + 1,
                    final_resid
                );
                break 'outer;
            }

            // Prepare for restart
            g.fill(0.0);
            g[0] = residual_norm;
            cs.fill(0.0);
            sn.fill(0.0);
            y.fill(0.0);

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
