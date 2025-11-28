use super::structs::{GpuSolver, LinearSolverStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleType {
    None,
    VCycle,
    WCycle,
}

pub struct MultigridLevel {
    pub level_index: usize,
    pub num_cells: u32,
    pub num_nonzeros: u32,

    // Matrix A for this level (CSR)
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub b_matrix_values: wgpu::Buffer,
    pub b_inv_diagonal: wgpu::Buffer, // For smoother

    // Solution and RHS for this level
    pub b_x: wgpu::Buffer,
    pub b_b: wgpu::Buffer,
    pub b_r: wgpu::Buffer, // Residual

    // Restriction Operator R (Fine to Coarse) - CSR
    pub b_r_row_offsets: wgpu::Buffer,
    pub b_r_col_indices: wgpu::Buffer,
    pub b_r_values: wgpu::Buffer,

    // Prolongation Operator P (Coarse to Fine) - CSR
    // Usually P = R^T, but we might store it explicitly for performance
    pub b_p_row_offsets: wgpu::Buffer,
    pub b_p_col_indices: wgpu::Buffer,
    pub b_p_values: wgpu::Buffer,

    // Bind Groups
    pub bg_matrix: wgpu::BindGroup,
    pub bg_state: wgpu::BindGroup, // x, b, r
    pub bg_restriction: wgpu::BindGroup,
    pub bg_prolongation: wgpu::BindGroup,
    pub bg_params: wgpu::BindGroup, // n, omega
    pub b_params: wgpu::Buffer,
}

pub struct MultigridSolver {
    pub levels: Vec<MultigridLevel>,
    pub cycle_type_ux: CycleType,
    pub cycle_type_uy: CycleType,
    pub cycle_type_p: CycleType,

    pub pipeline_residual: wgpu::ComputePipeline,
    pub pipeline_smooth: wgpu::ComputePipeline,
    pub pipeline_restrict: wgpu::ComputePipeline,
    pub pipeline_prolongate: wgpu::ComputePipeline,

    // Coarsening Pipelines
    pub pipeline_init_random: wgpu::ComputePipeline,
    pub pipeline_mis_step: wgpu::ComputePipeline,
    pub pipeline_mis_update: wgpu::ComputePipeline,
    pub pipeline_count_c_points: wgpu::ComputePipeline,
    pub pipeline_assign_coarse_indices: wgpu::ComputePipeline,
}

impl MultigridSolver {
    pub fn new(
        device: &wgpu::Device,
        cycle_type_ux: CycleType,
        cycle_type_uy: CycleType,
        cycle_type_p: CycleType,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AMG Matrix Ops Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/amg_matrix_ops.wgsl").into()),
        });

        // Group 0: Matrix
        let bgl_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Matrix BGL"),
            entries: &[
                // row_offsets
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
                // col_indices
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
                // values
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
                // inv_diagonal
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        // Group 1: State (x, b, r)
        let bgl_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG State BGL"),
            entries: &[
                // x
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
                // b
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
                // r
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

        // Group 2: Params
        let bgl_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_residual = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AMG Residual Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("AMG Residual Layout"),
                    bind_group_layouts: &[&bgl_matrix, &bgl_state, &bgl_params],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "residual",
        });

        let pipeline_smooth = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AMG Smooth Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("AMG Smooth Layout"),
                    bind_group_layouts: &[&bgl_matrix, &bgl_state, &bgl_params],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "smooth_jacobi",
        });

        let pipeline_restrict = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AMG Restrict Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("AMG Restrict Layout"),
                    bind_group_layouts: &[&bgl_matrix, &bgl_state, &bgl_params, &bgl_state],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "restrict_op",
        });

        let pipeline_prolongate =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG Prolongate Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("AMG Prolongate Layout"),
                        bind_group_layouts: &[&bgl_matrix, &bgl_state, &bgl_params, &bgl_state],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: "prolongate_add",
            });

        // Coarsening Pipelines
        let shader_coarsening = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AMG Coarsening Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/amg_coarsening.wgsl").into()),
        });

        // Layouts for Coarsening
        // Group 0: Matrix (Graph) -> bgl_matrix
        // Group 1: State (MIS, random, fine_to_coarse) -> Need new layout?
        // Group 2: Params -> bgl_params
        // Group 3: Counter -> Need new layout?

        let bgl_coarsening_state =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AMG Coarsening State BGL"),
                entries: &[
                    // state (u32)
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
                    // random_values (f32)
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
                    // fine_to_coarse (i32) - Only used in assign_coarse_indices
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

        let bgl_counter = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Counter BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None, // atomic<u32> is 4 bytes
                },
                count: None,
            }],
        });

        let bgl_matrix_coarsening =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AMG Coarsening Matrix BGL"),
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
                ],
            });

        let layout_coarsening = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("AMG Coarsening Layout"),
            bind_group_layouts: &[&bgl_matrix_coarsening, &bgl_coarsening_state, &bgl_params],
            push_constant_ranges: &[],
        });

        let layout_coarsening_counter =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("AMG Coarsening Counter Layout"),
                bind_group_layouts: &[
                    &bgl_matrix_coarsening,
                    &bgl_coarsening_state,
                    &bgl_params,
                    &bgl_counter,
                ],
                push_constant_ranges: &[],
            });

        let pipeline_init_random =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG Init Random Pipeline"),
                layout: Some(&layout_coarsening),
                module: &shader_coarsening,
                entry_point: "init_random",
            });

        let pipeline_mis_step = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AMG MIS Step Pipeline"),
            layout: Some(&layout_coarsening),
            module: &shader_coarsening,
            entry_point: "mis_step",
        });

        let pipeline_mis_update =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG MIS Update Pipeline"),
                layout: Some(&layout_coarsening),
                module: &shader_coarsening,
                entry_point: "mis_update",
            });

        let pipeline_count_c_points =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG Count C Points Pipeline"),
                layout: Some(&layout_coarsening_counter),
                module: &shader_coarsening,
                entry_point: "count_c_points",
            });

        let pipeline_assign_coarse_indices =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG Assign Coarse Indices Pipeline"),
                layout: Some(&layout_coarsening_counter),
                module: &shader_coarsening,
                entry_point: "assign_coarse_indices",
            });

        Self {
            levels: Vec::new(),
            cycle_type_ux,
            cycle_type_uy,
            cycle_type_p,
            pipeline_residual,
            pipeline_smooth,
            pipeline_restrict,
            pipeline_prolongate,
            pipeline_init_random,
            pipeline_mis_step,
            pipeline_mis_update,
            pipeline_count_c_points,
            pipeline_assign_coarse_indices,
        }
    }

    pub fn get_cycle_type(&self, field_name: &str) -> CycleType {
        match field_name {
            "Ux" => self.cycle_type_ux,
            "Uy" => self.cycle_type_uy,
            "p" => self.cycle_type_p,
            _ => CycleType::None,
        }
    }

    pub async fn solve(
        &mut self,
        solver: &GpuSolver,
        field_name: &str,
    ) -> Option<LinearSolverStats> {
        let start_time = std::time::Instant::now();
        let cycle_type = self.get_cycle_type(field_name);

        println!(
            "MultigridSolver::solve for {}, cycle_type: {:?}",
            field_name, cycle_type
        );

        if matches!(cycle_type, CycleType::None) {
            return None;
        }

        if self.levels.is_empty() {
            self.build_amg_hierarchy(solver);
        }

        // Update Level 0 matrix values from the current solver matrix.
        // The matrix A changes every timestep, so we need to refresh it.
        self.update_level0_matrix(solver);

        // Initialize Level 0 (Fine Grid)
        // Copy RHS and Initial Guess from GpuSolver to Level 0
        let level0 = &self.levels[0];
        let size = (level0.num_cells as u64) * 4;

        let mut encoder =
            solver
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("AMG Init Level 0"),
                });

        // Copy b_rhs -> level0.b_b
        encoder.copy_buffer_to_buffer(&solver.b_rhs, 0, &level0.b_b, 0, size);
        // Copy b_x -> level0.b_x
        encoder.copy_buffer_to_buffer(&solver.b_x, 0, &level0.b_x, 0, size);

        solver.context.queue.submit(Some(encoder.finish()));

        // AMG Cycles
        let max_cycles = 20;
        let mut final_resid = 0.0;
        let mut converged = false;
        let mut prev_resid = f32::MAX;

        let tol = 1e-5;
        let stagnation_tolerance = 1e-3;
        let stagnation_factor = 1e-4;

        for i in 0..max_cycles {
            match cycle_type {
                CycleType::VCycle => self.v_cycle(solver, 0),
                CycleType::WCycle => self.w_cycle(solver, 0),
                CycleType::None => break,
            }

            // Compute residual
            self.compute_residual(solver, 0);
            let resid_norm = self.norm(solver, &self.levels[0].b_r);

            println!("AMG Iter {}: Residual = {:.2e}", i, resid_norm);

            if resid_norm < tol {
                final_resid = resid_norm;
                converged = true;
                break;
            }

            // Stagnation check: if residual is not improving significantly
            // but is below tolerance, accept the solution
            if i > 0
                && (resid_norm - prev_resid).abs() < stagnation_factor
                && resid_norm < stagnation_tolerance
            {
                final_resid = resid_norm;
                println!("AMG stagnated at iter {} with residual {:.2e}", i, resid_norm);
                break;
            }

            prev_resid = resid_norm;
            final_resid = resid_norm;
        }

        // Copy solution back
        let mut encoder =
            solver
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("AMG Finalize"),
                });
        encoder.copy_buffer_to_buffer(&level0.b_x, 0, &solver.b_x, 0, size);
        solver.context.queue.submit(Some(encoder.finish()));

        Some(LinearSolverStats {
            iterations: 1, // Placeholder
            residual: final_resid,
            converged,
            diverged: false,
            time: start_time.elapsed(),
        })
    }

    /// Update Level 0 matrix values to match the current solver matrix.
    /// The matrix A changes every timestep in CFD, so we must refresh the
    /// matrix values and recompute the inverse diagonal before each solve.
    fn update_level0_matrix(&mut self, solver: &GpuSolver) {
        if self.levels.is_empty() {
            return;
        }

        let level0 = &self.levels[0];
        let num_cells = level0.num_cells;

        // Copy current matrix values from solver to Level 0
        let mut encoder =
            solver
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Update L0 Matrix"),
                });
        encoder.copy_buffer_to_buffer(
            &solver.b_matrix_values,
            0,
            &level0.b_matrix_values,
            0,
            solver.b_matrix_values.size(),
        );
        solver.context.queue.submit(Some(encoder.finish()));

        // Recompute inverse diagonal for Level 0
        let l0_row_offsets: Vec<u32> =
            self.read_buffer(&solver.context.device, &solver.context.queue, &solver.b_row_offsets);
        let l0_col_indices: Vec<u32> =
            self.read_buffer(&solver.context.device, &solver.context.queue, &solver.b_col_indices);
        let l0_values: Vec<f32> =
            self.read_buffer(&solver.context.device, &solver.context.queue, &solver.b_matrix_values);

        let mut l0_inv_diag = vec![0.0f32; num_cells as usize];
        for i in 0..num_cells as usize {
            let start = l0_row_offsets[i] as usize;
            let end = l0_row_offsets[i + 1] as usize;
            for k in start..end {
                if l0_col_indices[k] == i as u32 {
                    let val = l0_values[k];
                    if val.abs() < 1e-12 {
                        // Handle near-zero diagonal gracefully
                        l0_inv_diag[i] = 1.0;
                    } else {
                        l0_inv_diag[i] = 1.0 / val;
                    }
                    break;
                }
            }
        }
        solver.context.queue.write_buffer(
            &level0.b_inv_diagonal,
            0,
            bytemuck::cast_slice(&l0_inv_diag),
        );
    }

    pub fn build_amg_hierarchy(&mut self, solver: &GpuSolver) {
        let device = &solver.context.device;
        let num_cells = solver.num_cells as u32;

        // 1. Create Level 0 (Fine)
        // Copy matrix data from solver
        let b_row_offsets = self.create_buffer_copy(
            device,
            &solver.context.queue,
            &solver.b_row_offsets,
            (num_cells as u64 + 1) * 4,
            "L0 Row Offsets",
        );
        let b_col_indices = self.create_buffer_copy(
            device,
            &solver.context.queue,
            &solver.b_col_indices,
            solver.b_col_indices.size(),
            "L0 Col Indices",
        );
        let b_matrix_values = self.create_buffer_copy(
            device,
            &solver.context.queue,
            &solver.b_matrix_values,
            solver.b_matrix_values.size(),
            "L0 Matrix Values",
        );
        let b_inv_diagonal =
            self.create_buffer_init(device, (num_cells as u64) * 4, "L0 Inv Diagonal");

        // Compute inv_diagonal for Level 0 on CPU
        {
            let l0_row_offsets: Vec<u32> =
                self.read_buffer(device, &solver.context.queue, &solver.b_row_offsets);
            let l0_col_indices: Vec<u32> =
                self.read_buffer(device, &solver.context.queue, &solver.b_col_indices);
            let l0_values: Vec<f32> =
                self.read_buffer(device, &solver.context.queue, &solver.b_matrix_values);

            let mut l0_inv_diag = vec![0.0f32; num_cells as usize];
            for i in 0..num_cells as usize {
                let start = l0_row_offsets[i] as usize;
                let end = l0_row_offsets[i + 1] as usize;
                for k in start..end {
                    if l0_col_indices[k] == i as u32 {
                        let val = l0_values[k];
                        if val.abs() < 1e-12 {
                            panic!("Zero diagonal in Level 0 at row {}", i);
                        }
                        l0_inv_diag[i] = 1.0 / val;
                        break;
                    }
                }
            }
            solver.context.queue.write_buffer(
                &b_inv_diagonal,
                0,
                bytemuck::cast_slice(&l0_inv_diag),
            );
        }

        // Create state vectors
        let b_x = self.create_buffer_init(device, (num_cells as u64) * 4, "L0 x");
        let b_b = self.create_buffer_init(device, (num_cells as u64) * 4, "L0 b");
        let b_r = self.create_buffer_init(device, (num_cells as u64) * 4, "L0 r");

        // Bind Groups for L0
        // ... (We need to recreate layouts here or store them in self?
        // Ideally, layouts should be stored in self to reuse.
        // For now, I'll recreate them locally or assume we can access them if stored.
        // But they are not stored in self.
        // I should refactor to store layouts in MultigridSolver.
        // For this step, I will just copy the layout creation code or use a helper.
        // Actually, creating layouts every time is fine for initialization.

        // ... (Layout creation code omitted for brevity, will use helper or copy)
        // Wait, I can't easily copy layout creation code inside this method without bloating it.
        // I should have stored layouts in `MultigridSolver`.
        // Let's assume I can recreate them.

        // REFACTOR: Store layouts in MultigridSolver to avoid duplication.
        // But for now, to make progress, I will just implement the loop structure and use placeholders for R/P construction.

        // Let's just create the first level as before, but then enter a loop.

        let mut _current_n = num_cells;

        // Bind Groups for L0
        // We need to define layouts here to create bind groups.
        // Ideally, layouts should be stored in `MultigridSolver`.
        // For now, I'll redefine them here (duplication, but works).

        // Group 0: Matrix
        let bgl_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Matrix BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let bg_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L0 Matrix BG"),
            layout: &bgl_matrix,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_matrix_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_inv_diagonal.as_entire_binding(),
                },
            ],
        });

        // Group 1: State
        let bgl_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG State BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // b is read-write
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

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L0 State BG"),
            layout: &bgl_state,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_r.as_entire_binding(),
                },
            ],
        });

        // Group 2: Params
        let bgl_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let b_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L0 Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bytes = [
            num_cells.to_ne_bytes(),
            0.7f32.to_ne_bytes(),
            [0u8; 4],
            [0u8; 4],
        ]
        .concat();
        solver
            .context
            .queue
            .write_buffer(&b_params, 0, &params_bytes);

        let bg_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L0 Params BG"),
            layout: &bgl_params,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b_params.as_entire_binding(),
            }],
        });

        // Placeholder for restriction/prolongation BGs
        // We can't create valid ones yet as we don't have R/P matrices.
        // We'll reuse bg_matrix as a placeholder (unsafe but satisfies type).
        // For L0, these are used to go to L1.
        // We will update them later or create dummy ones.

        let level0 = MultigridLevel {
            level_index: 0,
            num_cells,
            num_nonzeros: solver.num_nonzeros,
            b_row_offsets,
            b_col_indices,
            b_matrix_values,
            b_inv_diagonal,
            b_x,
            b_b,
            b_r,
            b_r_row_offsets: self.create_buffer_init(device, 4, "Dummy"),
            b_r_col_indices: self.create_buffer_init(device, 4, "Dummy"),
            b_r_values: self.create_buffer_init(device, 4, "Dummy"),
            b_p_row_offsets: self.create_buffer_init(device, 4, "Dummy"),
            b_p_col_indices: self.create_buffer_init(device, 4, "Dummy"),
            b_p_values: self.create_buffer_init(device, 4, "Dummy"),
            bg_matrix,
            bg_state,
            bg_restriction: self.create_dummy_bind_group(device, &bgl_matrix), // Placeholder
            bg_prolongation: self.create_dummy_bind_group(device, &bgl_matrix), // Placeholder
            bg_params,
            b_params,
        };

        self.levels.push(level0);

        // Coarsening Loop (Max 3 levels for now)
        for level_idx in 0..3 {
            // ... (L0 creation logic is already done for level 0, but we need to generalize)
            // Actually, the previous code created L0. We should start loop from 0 if we want to be generic,
            // but L0 is special (comes from solver).
            // Let's assume L0 is already pushed.

            // If we are at the last level, stop.
            if self.levels.len() > 2 {
                break;
            } // Limit depth

            let fine_level_idx = self.levels.len() - 1;
            let fine_n = self.levels[fine_level_idx].num_cells;

            if fine_n < 10 {
                break;
            } // Stop if grid is small

            println!("Coarsening Level {}: {} cells", fine_level_idx, fine_n);

            // 1. Allocate Coarsening Buffers
            // state: u32 (0: Undecided, 1: C, 2: F)
            let b_state = self.create_buffer_init(device, (fine_n as u64) * 4, "MIS State");
            // random: f32
            let b_random = self.create_buffer_init(device, (fine_n as u64) * 4, "MIS Random");
            // fine_to_coarse: i32
            let b_fine_to_coarse =
                self.create_buffer_init(device, (fine_n as u64) * 4, "Fine to Coarse");
            // counter: atomic u32
            let b_counter = self.create_buffer_init(device, 4, "MIS Counter"); // Init to 0

            // Bind Groups for Coarsening
            // We need to create them here.
            // Group 0: Matrix (Graph) - Use fine level's matrix BG?
            // No, fine level's matrix BG has 4 bindings. Coarsening shader expects 2 (row_offsets, col_indices).
            // We need a specific BG for coarsening graph.

            let bgl_matrix_coarsening =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("AMG Coarsening Matrix BGL"),
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
                    ],
                });

            let bg_matrix_coarsening = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Coarsening Matrix BG"),
                layout: &bgl_matrix_coarsening,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.levels[fine_level_idx]
                            .b_row_offsets
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.levels[fine_level_idx]
                            .b_col_indices
                            .as_entire_binding(),
                    },
                ],
            });

            // Group 1: State
            let bgl_coarsening_state =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("AMG Coarsening State BGL"),
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
                    ],
                });

            let bg_coarsening_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Coarsening State BG"),
                layout: &bgl_coarsening_state,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_random.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_fine_to_coarse.as_entire_binding(),
                    },
                ],
            });

            // Group 2: Params (reuse fine level params)
            let bg_params = &self.levels[fine_level_idx].bg_params;

            // Group 3: Counter
            let bgl_counter = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AMG Counter BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            let bg_counter = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Coarsening Counter BG"),
                layout: &bgl_counter,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_counter.as_entire_binding(),
                }],
            });

            // 2. Run MIS Algorithm
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MIS Encoder"),
            });

            let workgroups = (fine_n + 63) / 64;

            // Init Random
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Init Random Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_init_random);
                cpass.set_bind_group(0, &bg_matrix_coarsening, &[]);
                cpass.set_bind_group(1, &bg_coarsening_state, &[]);
                cpass.set_bind_group(2, bg_params, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // MIS Loop (Fixed iterations for now, e.g., 10)
            for _ in 0..10 {
                // Step
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("MIS Step Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_mis_step);
                    cpass.set_bind_group(0, &bg_matrix_coarsening, &[]);
                    cpass.set_bind_group(1, &bg_coarsening_state, &[]);
                    cpass.set_bind_group(2, bg_params, &[]);
                    cpass.dispatch_workgroups(workgroups, 1, 1);
                }
                // Update
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("MIS Update Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_mis_update);
                    cpass.set_bind_group(0, &bg_matrix_coarsening, &[]);
                    cpass.set_bind_group(1, &bg_coarsening_state, &[]);
                    cpass.set_bind_group(2, bg_params, &[]);
                    cpass.dispatch_workgroups(workgroups, 1, 1);
                }
            }

            // Count C-points
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Count C Points Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_count_c_points);
                cpass.set_bind_group(0, &bg_matrix_coarsening, &[]);
                cpass.set_bind_group(1, &bg_coarsening_state, &[]);
                cpass.set_bind_group(2, bg_params, &[]);
                cpass.set_bind_group(3, &bg_counter, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            solver.context.queue.submit(Some(encoder.finish()));

            // Read back counter
            let b_counter_read = self.create_buffer_copy(
                device,
                &solver.context.queue,
                &b_counter,
                4,
                "Read Counter",
            );

            // We need to map and read. This is async.
            // For simplicity in this "build" phase, we can block or use a staging buffer.
            // `create_buffer_copy` creates a STORAGE | COPY_SRC buffer.
            // We need a MAP_READ buffer.

            let b_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Counter"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Counter to Staging"),
            });
            encoder.copy_buffer_to_buffer(&b_counter, 0, &b_staging, 0, 4);
            solver.context.queue.submit(Some(encoder.finish()));

            let slice = b_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let coarse_n = u32::from_ne_bytes(data[0..4].try_into().unwrap());
            drop(data);
            b_staging.unmap();

            println!("  Coarse Grid Size: {}", coarse_n);

            if coarse_n == 0 || coarse_n >= fine_n {
                println!(
                    "  Coarsening failed or converged (n={}). Stopping.",
                    coarse_n
                );
                break;
            }

            // Reset counter for index assignment
            // We need to zero it out.
            solver
                .context
                .queue
                .write_buffer(&b_counter, 0, &[0, 0, 0, 0]);

            // Assign Coarse Indices
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Assign Indices Encoder"),
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Assign Indices Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_assign_coarse_indices);
                cpass.set_bind_group(0, &bg_matrix_coarsening, &[]);
                cpass.set_bind_group(1, &bg_coarsening_state, &[]);
                cpass.set_bind_group(2, bg_params, &[]);
                cpass.set_bind_group(3, &bg_counter, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
            solver.context.queue.submit(Some(encoder.finish()));

            // ... (MIS and Assign Indices done)

            // 3. Read back data for CPU processing
            let fine_to_coarse_gpu: Vec<i32> =
                self.read_buffer(device, &solver.context.queue, &b_fine_to_coarse);

            // Read fine matrix A
            let fine_level = &self.levels[fine_level_idx];
            let a_row_offsets: Vec<u32> =
                self.read_buffer(device, &solver.context.queue, &fine_level.b_row_offsets);
            let a_col_indices: Vec<u32> =
                self.read_buffer(device, &solver.context.queue, &fine_level.b_col_indices);
            let a_values: Vec<f32> =
                self.read_buffer(device, &solver.context.queue, &fine_level.b_matrix_values);

            // 4. CPU: Form Aggregates and Build Operators
            let mut fine_to_coarse = fine_to_coarse_gpu.clone();
            let mut coarse_n = coarse_n as usize; // From GPU counter

            // Assign F-points to C-points
            for i in 0..fine_n as usize {
                if fine_to_coarse[i] == -1 {
                    let start = a_row_offsets[i] as usize;
                    let end = a_row_offsets[i + 1] as usize;
                    let mut best_c = -1;

                    // Simple heuristic: pick first C-neighbor
                    // Better: pick strongest connection (largest -A[i,j])
                    // Since Laplacian has negative off-diagonals, we look for min value (most negative)
                    // or max magnitude.
                    // Let's use max magnitude of negative values.
                    let mut max_strength = -1.0;

                    for k in start..end {
                        let neighbor = a_col_indices[k] as usize;
                        let val = a_values[k];
                        let c_idx = fine_to_coarse[neighbor];

                        if c_idx != -1 && neighbor != i {
                            // It's a C-point (or already assigned F-point? No, we iterate linearly, so maybe assigned F-point)
                            // But we want to attach to a C-point.
                            // fine_to_coarse contains coarse index for C-points.
                            // For F-points, it might be -1 or already assigned.
                            // We prefer original C-points.
                            // But we can't distinguish easily from this array.
                            // However, if we iterate 0..n, and we only assigned C-points on GPU,
                            // then any non-(-1) is a C-point (or previously visited F-point).
                            // Let's assume we attach to any assigned neighbor.

                            let strength = val.abs(); // Simple strength
                            if strength > max_strength {
                                max_strength = strength;
                                best_c = c_idx;
                            }
                        }
                    }

                    if best_c != -1 {
                        fine_to_coarse[i] = best_c;
                    } else {
                        // Orphan. Make it a new C-point?
                        // Or attach to self (singleton aggregate).
                        fine_to_coarse[i] = coarse_n as i32;
                        coarse_n += 1;
                    }
                }
            }

            let coarse_n = coarse_n as u32; // Final coarse size

            // Build P (CSR)
            // P is fine_n x coarse_n.
            // P[i, J] = 1 if fine_to_coarse[i] == J.
            let mut p_row_offsets = Vec::with_capacity(fine_n as usize + 1);
            let mut p_col_indices = Vec::with_capacity(fine_n as usize);
            let mut p_values = Vec::with_capacity(fine_n as usize);

            for i in 0..fine_n as usize {
                p_row_offsets.push(p_col_indices.len() as u32);
                let c = fine_to_coarse[i];
                if c != -1 {
                    p_col_indices.push(c as u32);
                    p_values.push(1.0);
                }
            }
            p_row_offsets.push(p_col_indices.len() as u32);

            // Build R (CSR) = P^T
            // R is coarse_n x fine_n.
            // We can compute R row counts first.
            let mut r_row_counts = vec![0u32; coarse_n as usize];
            for &c in &fine_to_coarse {
                if c != -1 {
                    r_row_counts[c as usize] += 1;
                }
            }

            let mut r_row_offsets = Vec::with_capacity(coarse_n as usize + 1);
            r_row_offsets.push(0);
            let mut current_offset = 0;
            for &count in &r_row_counts {
                current_offset += count;
                r_row_offsets.push(current_offset);
            }

            let mut r_col_indices = vec![0u32; current_offset as usize];
            let mut r_values = vec![0.0f32; current_offset as usize];
            let mut r_current_pos = r_row_offsets[0..coarse_n as usize].to_vec(); // Track insert pos

            for i in 0..fine_n as usize {
                let c = fine_to_coarse[i];
                if c != -1 {
                    let pos = r_current_pos[c as usize] as usize;
                    r_col_indices[pos] = i as u32;
                    r_values[pos] = 1.0;
                    r_current_pos[c as usize] += 1;
                }
            }

            // Build A_c (Galerkin) = R * A * P
            // A_c[I, J] = sum_{i in Agg(I), j in Agg(J)} A[i, j]
            // We iterate over non-zeros of A.

            let mut ac_entries: std::collections::HashMap<(u32, u32), f32> =
                std::collections::HashMap::new();

            for i in 0..fine_n as usize {
                let i_coarse = fine_to_coarse[i];
                if i_coarse == -1 {
                    continue;
                }

                let start = a_row_offsets[i] as usize;
                let end = a_row_offsets[i + 1] as usize;

                for k in start..end {
                    let j = a_col_indices[k] as usize;
                    let val = a_values[k];
                    let j_coarse = fine_to_coarse[j];

                    if j_coarse != -1 {
                        *ac_entries
                            .entry((i_coarse as u32, j_coarse as u32))
                            .or_insert(0.0) += val;
                    }
                }
            }

            // Convert HashMap to CSR
            let mut ac_row_offsets = Vec::with_capacity(coarse_n as usize + 1);
            let mut ac_col_indices = Vec::new();
            let mut ac_values = Vec::new();

            ac_row_offsets.push(0);
            for i_coarse in 0..coarse_n {
                let mut row_entries = Vec::new();
                for j_coarse in 0..coarse_n {
                    if let Some(&val) = ac_entries.get(&(i_coarse, j_coarse)) {
                        row_entries.push((j_coarse, val));
                    }
                }
                // Sort by col index
                row_entries.sort_by_key(|e| e.0);

                for (j_coarse, val) in row_entries {
                    ac_col_indices.push(j_coarse);
                    ac_values.push(val);
                }
                ac_row_offsets.push(ac_col_indices.len() as u32);
            }

            // 5. Upload to GPU
            let b_p_row_offsets = self.create_buffer_from_data(
                device,
                bytemuck::cast_slice(&p_row_offsets),
                "P Row Offsets",
            );
            let b_p_col_indices = self.create_buffer_from_data(
                device,
                bytemuck::cast_slice(&p_col_indices),
                "P Col Indices",
            );
            let b_p_values =
                self.create_buffer_from_data(device, bytemuck::cast_slice(&p_values), "P Values");

            let b_r_row_offsets = self.create_buffer_from_data(
                device,
                bytemuck::cast_slice(&r_row_offsets),
                "R Row Offsets",
            );
            let b_r_col_indices = self.create_buffer_from_data(
                device,
                bytemuck::cast_slice(&r_col_indices),
                "R Col Indices",
            );
            let b_r_values =
                self.create_buffer_from_data(device, bytemuck::cast_slice(&r_values), "R Values");

            let b_ac_row_offsets = self.create_buffer_from_data(
                device,
                bytemuck::cast_slice(&ac_row_offsets),
                "Ac Row Offsets",
            );
            let b_ac_col_indices = self.create_buffer_from_data(
                device,
                bytemuck::cast_slice(&ac_col_indices),
                "Ac Col Indices",
            );
            let b_ac_values =
                self.create_buffer_from_data(device, bytemuck::cast_slice(&ac_values), "Ac Values");

            // Compute inv_diagonal on CPU
            let mut ac_inv_diagonal = vec![0.0f32; coarse_n as usize];
            for i_coarse in 0..coarse_n as usize {
                // Find diagonal entry
                let start = ac_row_offsets[i_coarse] as usize;
                let end = ac_row_offsets[i_coarse + 1] as usize;
                for k in start..end {
                    if ac_col_indices[k] == i_coarse as u32 {
                        let val = ac_values[k];
                        if val.abs() < 1e-12 {
                            panic!("Zero diagonal in Ac at row {}", i_coarse);
                        }
                        ac_inv_diagonal[i_coarse] = 1.0 / val;
                        break;
                    }
                }
            }
            let b_ac_inv_diagonal = self.create_buffer_from_data(
                device,
                bytemuck::cast_slice(&ac_inv_diagonal),
                "Ac Inv Diagonal",
            );

            let b_ac_x = self.create_buffer_init(device, (coarse_n as u64) * 4, "Ac x");
            let b_ac_b = self.create_buffer_init(device, (coarse_n as u64) * 4, "Ac b");
            let b_ac_r = self.create_buffer_init(device, (coarse_n as u64) * 4, "Ac r");

            // Create Bind Groups for Coarse Level
            // Matrix BG
            let bg_matrix_ac = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Ac Matrix BG"),
                layout: &bgl_matrix, // Reuse layout
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_ac_row_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_ac_col_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_ac_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_ac_inv_diagonal.as_entire_binding(),
                    },
                ],
            });

            // State BG
            let bg_state_ac = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Ac State BG"),
                layout: &bgl_state, // Reuse layout
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_ac_x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_ac_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_ac_r.as_entire_binding(),
                    },
                ],
            });

            // Params BG
            let b_params_ac = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Ac Params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let params_bytes_ac = [
                coarse_n.to_ne_bytes(),
                0.7f32.to_ne_bytes(),
                [0u8; 4],
                [0u8; 4],
            ]
            .concat();
            solver
                .context
                .queue
                .write_buffer(&b_params_ac, 0, &params_bytes_ac);

            let bg_params_ac = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Ac Params BG"),
                layout: &bgl_params,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_params_ac.as_entire_binding(),
                }],
            });

            // R and P Bind Groups (for Fine Level)
            // We need to update the Fine Level's bg_restriction and bg_prolongation.
            // But Fine Level is already pushed to `self.levels`.
            // We can update it via mutable reference.
            // Wait, `MultigridLevel` stores `bg_restriction`.
            // We need to create it using `b_r_*` buffers.
            // The layout is `bgl_matrix` (4 bindings).
            // R has row_offsets, col_indices, values. 4th binding? Dummy?
            // Yes, R doesn't have diagonal.
            // So we need a dummy buffer for binding 3.
            let b_dummy = self.create_buffer_init(device, 16, "Dummy");

            let bg_restriction = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("R BG"),
                layout: &bgl_matrix,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_r_row_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_r_col_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_r_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_dummy.as_entire_binding(),
                    },
                ],
            });

            let bg_prolongation = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("P BG"),
                layout: &bgl_matrix,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_p_row_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_p_col_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_p_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_dummy.as_entire_binding(),
                    },
                ],
            });

            // Update Fine Level
            // self.levels[fine_level_idx] is immutable?
            // We can access it mutably.
            self.levels[fine_level_idx].b_r_row_offsets = b_r_row_offsets;
            self.levels[fine_level_idx].b_r_col_indices = b_r_col_indices;
            self.levels[fine_level_idx].b_r_values = b_r_values;
            self.levels[fine_level_idx].b_p_row_offsets = b_p_row_offsets;
            self.levels[fine_level_idx].b_p_col_indices = b_p_col_indices;
            self.levels[fine_level_idx].b_p_values = b_p_values;
            self.levels[fine_level_idx].bg_restriction = bg_restriction;
            self.levels[fine_level_idx].bg_prolongation = bg_prolongation;

            // Create Coarse Level
            let level_coarse = MultigridLevel {
                level_index: level_idx + 1,
                num_cells: coarse_n,
                num_nonzeros: ac_values.len() as u32,
                b_row_offsets: b_ac_row_offsets,
                b_col_indices: b_ac_col_indices,
                b_matrix_values: b_ac_values,
                b_inv_diagonal: b_ac_inv_diagonal,
                b_x: b_ac_x,
                b_b: b_ac_b,
                b_r: b_ac_r,
                // Placeholders for next level R/P
                b_r_row_offsets: self.create_buffer_init(device, 4, "Dummy"),
                b_r_col_indices: self.create_buffer_init(device, 4, "Dummy"),
                b_r_values: self.create_buffer_init(device, 4, "Dummy"),
                b_p_row_offsets: self.create_buffer_init(device, 4, "Dummy"),
                b_p_col_indices: self.create_buffer_init(device, 4, "Dummy"),
                b_p_values: self.create_buffer_init(device, 4, "Dummy"),
                bg_matrix: bg_matrix_ac,
                bg_state: bg_state_ac,
                bg_restriction: self.create_dummy_bind_group(device, &bgl_matrix),
                bg_prolongation: self.create_dummy_bind_group(device, &bgl_matrix),
                bg_params: bg_params_ac,
                b_params: b_params_ac,
            };

            self.levels.push(level_coarse);

            _current_n = coarse_n;
        }
    }

    fn create_buffer_copy(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        size: u64,
        label: &str,
    ) -> wgpu::Buffer {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Buffer"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &buf, 0, size);
        queue.submit(Some(encoder.finish()));
        buf
    }

    fn create_buffer_init(&self, device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn v_cycle(&self, solver: &GpuSolver, level_idx: usize) {
        let is_coarsest = level_idx == self.levels.len() - 1;

        // 1. Pre-smoothing
        self.smooth(solver, level_idx, 2); // 2 iterations

        if !is_coarsest {
            // 2. Compute Residual: r = b - Ax
            self.compute_residual(solver, level_idx);

            // 3. Restriction: r_c = R * r
            self.restrict(solver, level_idx);

            // 4. Initialize Coarse Guess: x_c = 0
            self.levels[level_idx + 1].zero_x(solver);

            // 5. Recursion
            self.v_cycle(solver, level_idx + 1);

            // 6. Prolongation and Correction: x += P * x_c
            self.prolongate_and_correct(solver, level_idx);
        } else {
            // Coarsest level solve (more smoothing)
            self.smooth(solver, level_idx, 50);
        }

        // 7. Post-smoothing
        self.smooth(solver, level_idx, 2);
    }

    pub fn w_cycle(&self, solver: &GpuSolver, level_idx: usize) {
        let is_coarsest = level_idx == self.levels.len() - 1;

        // 1. Pre-smoothing
        self.smooth(solver, level_idx, 2);

        if !is_coarsest {
            // 2. Compute Residual: r = b - Ax
            self.compute_residual(solver, level_idx);

            // 3. Restriction: r_c = R * r
            self.restrict(solver, level_idx);

            // 4. Initialize Coarse Guess: x_c = 0
            self.levels[level_idx + 1].zero_x(solver);

            // 5. Recursion 1
            self.w_cycle(solver, level_idx + 1);

            // 6. Recursion 2 (W-cycle specific)
            self.w_cycle(solver, level_idx + 1);

            // 7. Prolongation and Correction: x += P * x_c
            self.prolongate_and_correct(solver, level_idx);
        } else {
            // Coarsest level solve
            self.smooth(solver, level_idx, 50);
        }

        // 8. Post-smoothing
        self.smooth(solver, level_idx, 2);
    }

    fn smooth(&self, solver: &GpuSolver, level_idx: usize, iterations: u32) {
        let level = &self.levels[level_idx];
        let num_groups = level.num_cells.div_ceil(64);

        for _ in 0..iterations {
            let mut encoder =
                solver
                    .context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("AMG Smooth Encoder"),
                    });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("AMG Smooth Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_smooth);
                cpass.set_bind_group(0, &level.bg_matrix, &[]);
                cpass.set_bind_group(1, &level.bg_state, &[]);
                cpass.set_bind_group(2, &level.bg_params, &[]); // Assuming params in bg_params
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }
            solver.context.queue.submit(Some(encoder.finish()));
        }
    }

    fn compute_residual(&self, solver: &GpuSolver, level_idx: usize) {
        let level = &self.levels[level_idx];
        let num_groups = level.num_cells.div_ceil(64);

        let mut encoder =
            solver
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("AMG Residual Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AMG Residual Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_residual);
            cpass.set_bind_group(0, &level.bg_matrix, &[]);
            cpass.set_bind_group(1, &level.bg_state, &[]);
            cpass.set_bind_group(2, &level.bg_params, &[]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }
        solver.context.queue.submit(Some(encoder.finish()));
    }

    fn restrict(&self, solver: &GpuSolver, level_idx: usize) {
        let level_fine = &self.levels[level_idx];
        let level_coarse = &self.levels[level_idx + 1];
        let num_groups = level_coarse.num_cells.div_ceil(64);

        let mut encoder =
            solver
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("AMG Restrict Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AMG Restrict Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_restrict);
            cpass.set_bind_group(0, &level_fine.bg_restriction, &[]); // R matrix
            cpass.set_bind_group(1, &level_fine.bg_state, &[]); // Fine r (input)
            cpass.set_bind_group(2, &level_coarse.bg_params, &[]); // Params (n_coarse)
            cpass.set_bind_group(3, &level_coarse.bg_state, &[]); // Coarse b (output)
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }
        solver.context.queue.submit(Some(encoder.finish()));
    }

    fn prolongate_and_correct(&self, solver: &GpuSolver, level_idx: usize) {
        let level_fine = &self.levels[level_idx];
        let level_coarse = &self.levels[level_idx + 1];
        let num_groups = level_fine.num_cells.div_ceil(64);

        let mut encoder =
            solver
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("AMG Prolongate Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AMG Prolongate Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_prolongate);
            cpass.set_bind_group(0, &level_fine.bg_prolongation, &[]); // P matrix
            cpass.set_bind_group(1, &level_coarse.bg_state, &[]); // Coarse x (input)
            cpass.set_bind_group(2, &level_fine.bg_params, &[]); // Params (n_fine)
            cpass.set_bind_group(3, &level_fine.bg_state, &[]); // Fine x (output)
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }
        solver.context.queue.submit(Some(encoder.finish()));
    }

    fn create_dummy_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let dummy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dummy BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn read_buffer<T: bytemuck::Pod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<T> {
        let size = buffer.size();
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Buffer Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    }

    fn create_buffer_from_data(
        &self,
        device: &wgpu::Device,
        data: &[u8],
        label: &str,
    ) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(data);
        buffer.unmap();
        buffer
    }

    fn norm(&self, solver: &GpuSolver, buffer: &wgpu::Buffer) -> f32 {
        let data: Vec<f32> =
            self.read_buffer(&solver.context.device, &solver.context.queue, buffer);
        data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
}

impl MultigridLevel {
    pub fn zero_x(&self, solver: &GpuSolver) {
        solver.zero_buffer(&self.b_x, (self.num_cells as u64) * 4);
    }
}
