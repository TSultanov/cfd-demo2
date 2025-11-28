use super::structs::{GpuSolver, LinearSolverStats, SolverType};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleType {
    None,
    VCycle,
    WCycle,
}

const MAX_LEVELS: usize = 4;
const MIN_COARSE_SIZE: u32 = 64;
const STRENGTH_THRESHOLD: f32 = 0.25;
const PROLONGATION_SMOOTH_OMEGA: f32 = 0.7;

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
    pub b_x_aux: wgpu::Buffer,
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
    pub bg_smoother_state: wgpu::BindGroup,
    pub bg_restriction: wgpu::BindGroup,
    pub bg_prolongation: wgpu::BindGroup,
    pub bg_params: wgpu::BindGroup, // n, omega
    pub b_params: wgpu::Buffer,
}

struct CpuLevelData {
    row_offsets: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<f32>,
    inv_diag: Vec<f32>,
}

pub struct MultigridSolver {
    pub levels: Vec<MultigridLevel>,
    pub cycle_type_ux: CycleType,
    pub cycle_type_uy: CycleType,
    pub cycle_type_p: CycleType,

    pub bgl_matrix: wgpu::BindGroupLayout,
    pub bgl_state: wgpu::BindGroupLayout,
    pub bgl_smoother_state: wgpu::BindGroupLayout,
    pub bgl_params: wgpu::BindGroupLayout,
    pub bgl_matrix_coarsening: wgpu::BindGroupLayout,
    pub bgl_coarsening_state: wgpu::BindGroupLayout,
    pub bgl_counter: wgpu::BindGroupLayout,

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
    pub pipeline_extract_pressure: wgpu::ComputePipeline,
    pub pipeline_extract_vector: wgpu::ComputePipeline,
    pub pipeline_insert_vector: wgpu::ComputePipeline,
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

        // Group 3: Extended smoother state (x, b, r, x_aux)
        let bgl_smoother_state =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AMG Smoother State BGL"),
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
                    bind_group_layouts: &[&bgl_matrix, &bgl_smoother_state, &bgl_params],
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
                    // random: f32
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
                    // fine_to_coarse: i32 - Only used in assign_coarse_indices
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

        let bgl_extract = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Extract Pressure BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader_extract = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AMG Extract Pressure Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/extract_pressure_matrix.wgsl").into(),
            ),
        });

        let pipeline_extract_pressure =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG Extract Pressure Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("AMG Extract Pressure Layout"),
                        bind_group_layouts: &[&bgl_extract],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader_extract,
                entry_point: "main",
            });

        let bgl_vector_ops = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Vector Ops BGL"),
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
            ],
        });

        let pipeline_extract_vector =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG Extract Vector Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("AMG Extract Vector Layout"),
                        bind_group_layouts: &[&bgl_vector_ops],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader_extract,
                entry_point: "extract_vector",
            });

        let pipeline_insert_vector =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AMG Insert Vector Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("AMG Insert Vector Layout"),
                        bind_group_layouts: &[&bgl_vector_ops],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader_extract,
                entry_point: "insert_vector",
            });

        Self {
            levels: Vec::new(),
            cycle_type_ux,
            cycle_type_uy,
            cycle_type_p,
            bgl_matrix,
            bgl_state,
            bgl_smoother_state,
            bgl_params,
            bgl_matrix_coarsening,
            bgl_coarsening_state,
            bgl_counter,
            pipeline_residual,
            pipeline_smooth,
            pipeline_restrict,
            pipeline_prolongate,
            pipeline_init_random,
            pipeline_mis_step,
            pipeline_mis_update,
            pipeline_count_c_points,
            pipeline_assign_coarse_indices,
            pipeline_extract_pressure,
            pipeline_extract_vector,
            pipeline_insert_vector,
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

        self.build_amg_hierarchy(solver);

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
                println!(
                    "AMG stagnated at iter {} with residual {:.2e}",
                    i, resid_norm
                );
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

    pub fn build_amg_hierarchy(&mut self, solver: &GpuSolver) {
        self.levels.clear();

        let device = &solver.context.device;
        let queue = &solver.context.queue;
        let num_cells = solver.num_cells as u32;

        // 1. Create Level 0 (Fine)

        // Always use scalar connectivity for AMG
        let src_row_offsets = &solver.b_row_offsets;
        let src_col_indices = &solver.b_col_indices;

        let src_matrix_values = if solver.solver_type == SolverType::Coupled {
            &solver
                .coupled_resources
                .as_ref()
                .expect("Coupled resources missing")
                .b_matrix_values
        } else {
            &solver.b_matrix_values
        };

        // Copy matrix data from solver
        let b_row_offsets = self.create_buffer_copy(
            device,
            &solver.context.queue,
            src_row_offsets,
            (num_cells as u64 + 1) * 4,
            "L0 Row Offsets",
        );
        let b_col_indices = self.create_buffer_copy(
            device,
            &solver.context.queue,
            src_col_indices,
            src_col_indices.size(),
            "L0 Col Indices",
        );

        let b_matrix_values = if solver.solver_type == SolverType::Coupled {
            let size = src_col_indices.size(); // Scalar size
            let b_values = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("L0 Matrix Values (Extracted)"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bgl_extract = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AMG Extract Pressure BGL Temp"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let bg_extract = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("AMG Extract Pressure BG"),
                layout: &bgl_extract,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_row_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_matrix_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_values.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("AMG Extract Pressure Encoder"),
            });

            let workgroup_size = 64;
            let num_groups = (num_cells + workgroup_size - 1) / workgroup_size;

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("AMG Extract Pressure Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_extract_pressure);
                cpass.set_bind_group(0, &bg_extract, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            queue.submit(Some(encoder.finish()));

            b_values
        } else {
            self.create_buffer_copy(
                device,
                &solver.context.queue,
                src_matrix_values,
                src_matrix_values.size(),
                "L0 Matrix Values",
            )
        };

        let b_inv_diagonal =
            self.create_buffer_init(device, (num_cells as u64) * 4, "L0 Inv Diagonal");

        // Compute inv_diagonal for Level 0 on CPU
        let l0_row_offsets: Vec<u32> = self.read_buffer(device, queue, &b_row_offsets);
        let l0_col_indices: Vec<u32> = self.read_buffer(device, queue, &b_col_indices);
        let l0_values: Vec<f32> = self.read_buffer(device, queue, &b_matrix_values);

        let mut l0_inv_diag = vec![0.0f32; num_cells as usize];
        for i in 0..num_cells as usize {
            let start = l0_row_offsets[i] as usize;
            let end = l0_row_offsets[i + 1] as usize;
            for k in start..end {
                if l0_col_indices[k] == i as u32 {
                    let val = l0_values[k];
                    if val.abs() < 1e-12 {
                        l0_inv_diag[i] = 1.0;
                    } else {
                        l0_inv_diag[i] = 1.0 / val;
                    }
                    break;
                }
            }
        }
        queue.write_buffer(&b_inv_diagonal, 0, bytemuck::cast_slice(&l0_inv_diag));

        let mut fine_cpu = CpuLevelData {
            row_offsets: l0_row_offsets,
            col_indices: l0_col_indices,
            values: l0_values,
            inv_diag: l0_inv_diag.clone(),
        };

        // Create state vectors
        let b_x = self.create_buffer_init(device, (num_cells as u64) * 4, "L0 x");
        let b_x_aux = self.create_buffer_init(device, (num_cells as u64) * 4, "L0 x aux");
        let b_b = self.create_buffer_init(device, (num_cells as u64) * 4, "L0 b");
        let b_r = self.create_buffer_init(device, (num_cells as u64) * 4, "L0 r");

        let bg_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L0 Matrix BG"),
            layout: &self.bgl_matrix,
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

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L0 State BG"),
            layout: &self.bgl_state,
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

        let bg_smoother_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L0 Smoother BG"),
            layout: &self.bgl_smoother_state,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_x_aux.as_entire_binding(),
                },
            ],
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
            layout: &self.bgl_params,
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
            b_x_aux,
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
            bg_smoother_state,
            bg_restriction: self.create_dummy_bind_group(device, &self.bgl_matrix),
            bg_prolongation: self.create_dummy_bind_group(device, &self.bgl_matrix),
            bg_params,
            b_params,
        };

        self.levels.push(level0);

        while self.levels.len() < MAX_LEVELS {
            let fine_level_idx = self.levels.len() - 1;
            let fine_n = self.levels[fine_level_idx].num_cells;

            if fine_n <= MIN_COARSE_SIZE {
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

            let bg_matrix_coarsening = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Coarsening Matrix BG"),
                layout: &self.bgl_matrix_coarsening,
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

            let bg_coarsening_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Coarsening State BG"),
                layout: &self.bgl_coarsening_state,
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

            let bg_counter = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Coarsening Counter BG"),
                layout: &self.bgl_counter,
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

            // Read back counter by copying into a staging buffer
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
            let fine_to_coarse_gpu: Vec<i32> = self.read_buffer(device, queue, &b_fine_to_coarse);
            let a_row_offsets = &fine_cpu.row_offsets;
            let a_col_indices = &fine_cpu.col_indices;
            let a_values = &fine_cpu.values;

            // 4. CPU: Form Aggregates and Build Operators
            let mut fine_to_coarse = fine_to_coarse_gpu.clone();
            let mut coarse_n = coarse_n as usize;
            let fine_n_usize = fine_n as usize;

            let mut row_max = vec![0.0f32; fine_n_usize];
            for i in 0..fine_n_usize {
                let start = a_row_offsets[i] as usize;
                let end = a_row_offsets[i + 1] as usize;
                let mut max_val = 0.0f32;
                for k in start..end {
                    if a_col_indices[k] == i as u32 {
                        continue;
                    }
                    max_val = max_val.max(a_values[k].abs());
                }
                row_max[i] = max_val;
            }

            for i in 0..fine_n_usize {
                if fine_to_coarse[i] != -1 {
                    continue;
                }
                let start = a_row_offsets[i] as usize;
                let end = a_row_offsets[i + 1] as usize;
                let mut best_c = -1;
                let mut best_strength = 0.0f32;
                for k in start..end {
                    let neighbor = a_col_indices[k] as usize;
                    if neighbor == i {
                        continue;
                    }
                    let c_idx = fine_to_coarse[neighbor];
                    if c_idx == -1 {
                        continue;
                    }
                    let strength = a_values[k].abs();
                    if row_max[i] > 0.0 && strength < STRENGTH_THRESHOLD * row_max[i] {
                        continue;
                    }
                    if strength > best_strength {
                        best_strength = strength;
                        best_c = c_idx;
                    }
                }

                if best_c != -1 {
                    fine_to_coarse[i] = best_c;
                } else {
                    fine_to_coarse[i] = coarse_n as i32;
                    coarse_n += 1;
                }
            }

            let coarse_n = coarse_n as u32; // Final coarse size

            // Build P with Jacobi-smoothed aggregation
            let mut p_row_offsets = Vec::with_capacity(fine_n_usize + 1);
            let mut p_col_indices = Vec::new();
            let mut p_values = Vec::new();
            let mut p_rows_dense: Vec<Vec<(u32, f32)>> = Vec::with_capacity(fine_n_usize);

            for i in 0..fine_n_usize {
                let mut entries: HashMap<u32, f32> = HashMap::new();
                if fine_to_coarse[i] >= 0 {
                    entries.insert(fine_to_coarse[i] as u32, 1.0);
                }

                let inv_d = fine_cpu.inv_diag[i];
                if inv_d != 0.0 {
                    let start = a_row_offsets[i] as usize;
                    let end = a_row_offsets[i + 1] as usize;
                    for k in start..end {
                        let j = a_col_indices[k] as usize;
                        let agg = fine_to_coarse[j];
                        if agg < 0 {
                            continue;
                        }
                        let contrib = -PROLONGATION_SMOOTH_OMEGA * inv_d * a_values[k];
                        *entries.entry(agg as u32).or_insert(0.0) += contrib;
                    }
                }

                let mut row_entries: Vec<(u32, f32)> = entries
                    .into_iter()
                    .filter(|(_, v)| v.abs() > 1e-7)
                    .collect();
                if row_entries.is_empty() && fine_to_coarse[i] >= 0 {
                    row_entries.push((fine_to_coarse[i] as u32, 1.0));
                }
                row_entries.sort_by_key(|e| e.0);
                p_row_offsets.push(p_col_indices.len() as u32);
                for (col, val) in &row_entries {
                    p_col_indices.push(*col);
                    p_values.push(*val);
                }
                p_rows_dense.push(row_entries);
            }
            p_row_offsets.push(p_col_indices.len() as u32);

            // Build R = P^T using dense rows
            let mut r_row_offsets = vec![0u32; coarse_n as usize + 1];
            for row in &p_rows_dense {
                for (col, _) in row {
                    r_row_offsets[*col as usize + 1] += 1;
                }
            }
            for i in 0..coarse_n as usize {
                r_row_offsets[i + 1] += r_row_offsets[i];
            }
            let mut r_col_indices = vec![0u32; *r_row_offsets.last().unwrap() as usize];
            let mut r_values = vec![0.0f32; r_col_indices.len()];
            let mut r_positions = r_row_offsets[..coarse_n as usize].to_vec();
            for (row_idx, entries) in p_rows_dense.iter().enumerate() {
                for (col, val) in entries {
                    let pos = r_positions[*col as usize] as usize;
                    r_col_indices[pos] = row_idx as u32;
                    r_values[pos] = *val;
                    r_positions[*col as usize] += 1;
                }
            }

            let mut ac_entries: HashMap<(u32, u32), f32> = HashMap::new();
            for i in 0..fine_n_usize {
                let row_start = a_row_offsets[i] as usize;
                let row_end = a_row_offsets[i + 1] as usize;
                if row_start == row_end {
                    continue;
                }
                let p_i = &p_rows_dense[i];
                if p_i.is_empty() {
                    continue;
                }
                for k in row_start..row_end {
                    let j = a_col_indices[k] as usize;
                    let val = a_values[k];
                    let p_j = &p_rows_dense[j];
                    if p_j.is_empty() {
                        continue;
                    }
                    for (ci, wi) in p_i.iter() {
                        for (cj, wj) in p_j.iter() {
                            *ac_entries.entry((*ci, *cj)).or_insert(0.0) += wi * val * wj;
                        }
                    }
                }
            }

            let mut rows = vec![Vec::<(u32, f32)>::new(); coarse_n as usize];
            for ((row, col), val) in ac_entries {
                rows[row as usize].push((col, val));
            }

            let mut ac_row_offsets = Vec::with_capacity(coarse_n as usize + 1);
            let mut ac_col_indices = Vec::new();
            let mut ac_values = Vec::new();
            ac_row_offsets.push(0);
            for (idx, row) in rows.iter_mut().enumerate() {
                if !row.iter().any(|(c, _)| *c == idx as u32) {
                    row.push((idx as u32, 1e-6));
                }
                row.sort_by_key(|e| e.0);
                for (col, val) in row.drain(..) {
                    ac_col_indices.push(col);
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
                let start = ac_row_offsets[i_coarse] as usize;
                let end = ac_row_offsets[i_coarse + 1] as usize;
                for k in start..end {
                    if ac_col_indices[k] == i_coarse as u32 {
                        let val = ac_values[k];
                        ac_inv_diagonal[i_coarse] = if val.abs() < 1e-12 { 0.0 } else { 1.0 / val };
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
            let b_ac_x_aux = self.create_buffer_init(device, (coarse_n as u64) * 4, "Ac x aux");
            let b_ac_b = self.create_buffer_init(device, (coarse_n as u64) * 4, "Ac b");
            let b_ac_r = self.create_buffer_init(device, (coarse_n as u64) * 4, "Ac r");

            // Create Bind Groups for Coarse Level
            // Matrix BG
            let bg_matrix_ac = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Ac Matrix BG"),
                layout: &self.bgl_matrix,
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
                layout: &self.bgl_state,
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

            let bg_smoother_state_ac = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Ac Smoother BG"),
                layout: &self.bgl_smoother_state,
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
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_ac_x_aux.as_entire_binding(),
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
                layout: &self.bgl_params,
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
                layout: &self.bgl_matrix,
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
                layout: &self.bgl_matrix,
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
                level_index: self.levels.len(),
                num_cells: coarse_n,
                num_nonzeros: ac_values.len() as u32,
                b_row_offsets: b_ac_row_offsets,
                b_col_indices: b_ac_col_indices,
                b_matrix_values: b_ac_values,
                b_inv_diagonal: b_ac_inv_diagonal,
                b_x: b_ac_x,
                b_x_aux: b_ac_x_aux,
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
                bg_smoother_state: bg_smoother_state_ac,
                bg_restriction: self.create_dummy_bind_group(device, &self.bgl_matrix),
                bg_prolongation: self.create_dummy_bind_group(device, &self.bgl_matrix),
                bg_params: bg_params_ac,
                b_params: b_params_ac,
            };

            self.levels.push(level_coarse);

            fine_cpu = CpuLevelData {
                row_offsets: ac_row_offsets,
                col_indices: ac_col_indices,
                values: ac_values,
                inv_diag: ac_inv_diagonal,
            };
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
                cpass.set_bind_group(1, &level.bg_smoother_state, &[]);
                cpass.set_bind_group(2, &level.bg_params, &[]); // Assuming params in bg_params
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &level.b_x_aux,
                0,
                &level.b_x,
                0,
                (level.num_cells as u64) * 4,
            );
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

    pub fn solve_coupled_pressure(
        &mut self,
        solver: &GpuSolver,
        coupled_rhs: &wgpu::Buffer,
        coupled_solution: &wgpu::Buffer,
    ) {
        if self.levels.is_empty() {
            return;
        }

        let device = &solver.context.device;
        let queue = &solver.context.queue;
        let level0 = &self.levels[0];
        let num_cells = level0.num_cells;

        // 1. Extract RHS: coupled_rhs -> level0.b_b
        // We need a temporary bind group
        let bgl_vector_ops = self.pipeline_extract_vector.get_bind_group_layout(0);
        let bg_extract = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("AMG Extract Vector BG"),
            layout: &bgl_vector_ops,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: coupled_rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: level0.b_b.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("AMG Extract Vector Encoder"),
        });

        let workgroup_size = 64;
        let num_groups = (num_cells + workgroup_size - 1) / workgroup_size;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AMG Extract Vector Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_extract_vector);
            cpass.set_bind_group(0, &bg_extract, &[]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }

        // Zero initial guess
        encoder.clear_buffer(&level0.b_x, 0, None);

        queue.submit(Some(encoder.finish()));

        // 2. Run AMG Cycles
        // Use V-Cycle for pressure
        let max_cycles = 1; // Just 1 cycle for preconditioner
        for _ in 0..max_cycles {
            self.v_cycle(solver, 0);
        }

        // 3. Insert Solution: level0.b_x -> coupled_solution
        let bg_insert = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("AMG Insert Vector BG"),
            layout: &bgl_vector_ops,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: level0.b_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coupled_solution.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("AMG Insert Vector Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AMG Insert Vector Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_insert_vector);
            cpass.set_bind_group(0, &bg_insert, &[]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

impl MultigridLevel {
    pub fn zero_x(&self, solver: &GpuSolver) {
        solver.zero_buffer(&self.b_x, (self.num_cells as u64) * 4);
    }
}
