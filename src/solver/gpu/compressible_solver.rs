use crate::solver::gpu::bindings::compressible_explicit_update as explicit_update;
use crate::solver::gpu::bindings::generated::{
    compressible_apply as generated_apply, compressible_assembly as generated_assembly,
    compressible_flux_kt as generated_flux_kt, compressible_gradients as generated_gradients,
    compressible_update as generated_update,
};
use crate::solver::gpu::compressible_fgmres::CompressibleFgmresResources;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::csr::build_block_csr;
use crate::solver::gpu::execution_plan::{ExecutionPlan, GraphExecMode, GraphNode, PlanNode};
use crate::solver::gpu::init::compressible_fields::{
    create_compressible_field_bind_groups, init_compressible_field_buffers, PackedStateConfig,
};
use crate::solver::gpu::init::linear_solver::matrix;
use crate::solver::gpu::init::mesh;
use crate::solver::gpu::kernel_graph::{ComputeNode, CopyBufferNode, KernelGraph, KernelNode};
use crate::solver::gpu::model_defaults::default_compressible_model;
use crate::solver::gpu::structs::{GpuConstants, LinearSolverStats};
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::{expand_schemes, SchemeRegistry};
use crate::solver::scheme::Scheme;
use bytemuck::cast_slice;
use std::env;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
struct CompressibleOffsets {
    stride: u32,
    rho: u32,
    rho_u: u32,
    rho_e: u32,
    p: u32,
    u: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompressibleProfile {
    enabled: bool,
    stride: usize,
    step: usize,
    accum_steps: usize,
    accum_total: f64,
    accum_fgmres: f64,
    accum_grad: f64,
    accum_assembly: f64,
    accum_apply: f64,
    accum_update: f64,
    accum_iters: u64,
}

impl CompressibleProfile {
    fn new() -> Self {
        let enabled = env_flag("CFD2_COMP_PROFILE", false);
        let stride = env_usize("CFD2_COMP_PROFILE_STRIDE", 25).max(1);
        CompressibleProfile {
            enabled,
            stride,
            ..Default::default()
        }
    }

    fn record(
        &mut self,
        total: f64,
        grad: f64,
        assembly: f64,
        fgmres: f64,
        apply: f64,
        update: f64,
        iters: u64,
    ) {
        if !self.enabled {
            return;
        }
        self.step += 1;
        self.accum_steps += 1;
        self.accum_total += total;
        self.accum_grad += grad;
        self.accum_assembly += assembly;
        self.accum_fgmres += fgmres;
        self.accum_apply += apply;
        self.accum_update += update;
        self.accum_iters += iters;

        if self.accum_steps >= self.stride {
            let steps = self.accum_steps as f64;
            let avg_total = self.accum_total / steps;
            let avg_fgmres = self.accum_fgmres / steps;
            let avg_grad = self.accum_grad / steps;
            let avg_assembly = self.accum_assembly / steps;
            let avg_apply = self.accum_apply / steps;
            let avg_update = self.accum_update / steps;
            let avg_iters = self.accum_iters as f64 / steps;
            let fgmres_pct = if avg_total > 0.0 {
                100.0 * avg_fgmres / avg_total
            } else {
                0.0
            };
            println!(
                "compressible_profile step {}..{} avg_total={:.3}s fgmres={:.3}s ({:.1}%) iters={:.1} grad={:.3}s assembly={:.3}s apply={:.3}s update={:.3}s",
                self.step + 1 - self.accum_steps,
                self.step,
                avg_total,
                avg_fgmres,
                fgmres_pct,
                avg_iters,
                avg_grad,
                avg_assembly,
                avg_apply,
                avg_update
            );
            self.accum_steps = 0;
            self.accum_total = 0.0;
            self.accum_fgmres = 0.0;
            self.accum_grad = 0.0;
            self.accum_assembly = 0.0;
            self.accum_apply = 0.0;
            self.accum_update = 0.0;
            self.accum_iters = 0;
        }
    }
}

pub struct GpuCompressibleSolver {
    pub context: GpuContext,
    pub num_cells: u32,
    pub num_faces: u32,
    pub num_unknowns: u32,
    pub state_step_index: usize,
    pub state_buffers: Vec<wgpu::Buffer>,
    pub bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub bg_fields: wgpu::BindGroup,
    pub bg_apply_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub bg_apply_fields: wgpu::BindGroup,
    pub b_state: wgpu::Buffer,
    pub b_state_old: wgpu::Buffer,
    pub b_state_old_old: wgpu::Buffer,
    pub b_state_iter: wgpu::Buffer,
    pub b_fluxes: wgpu::Buffer,
    pub b_grad_rho: wgpu::Buffer,
    pub b_grad_rho_u_x: wgpu::Buffer,
    pub b_grad_rho_u_y: wgpu::Buffer,
    pub b_grad_rho_e: wgpu::Buffer,
    pub b_constants: wgpu::Buffer,
    pub constants: GpuConstants,
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub b_matrix_values: wgpu::Buffer,
    pub b_rhs: wgpu::Buffer,
    pub b_x: wgpu::Buffer,
    pub b_scalar_row_offsets: wgpu::Buffer,
    pub bg_mesh: wgpu::BindGroup,
    pub bg_solver: wgpu::BindGroup,
    pub bg_apply_solver: wgpu::BindGroup,
    pub pipeline_assembly: wgpu::ComputePipeline,
    pub pipeline_apply: wgpu::ComputePipeline,
    pub pipeline_gradients: wgpu::ComputePipeline,
    pub pipeline_flux: wgpu::ComputePipeline,
    pub pipeline_explicit_update: wgpu::ComputePipeline,
    pub pipeline_update: wgpu::ComputePipeline,
    pub fgmres_resources: Option<CompressibleFgmresResources>,
    pub outer_iters: usize,
    pub nonconverged_relax: f32,
    pub(crate) scalar_row_offsets: Vec<u32>,
    pub(crate) scalar_col_indices: Vec<u32>,
    pub(crate) block_row_offsets: Vec<u32>,
    pub(crate) block_col_indices: Vec<u32>,
    implicit_tol: f32,
    implicit_max_restart: usize,
    implicit_retry_tol: f32,
    implicit_retry_restart: usize,
    implicit_base_alpha_u: f32,
    implicit_last_stats: LinearSolverStats,
    profile: CompressibleProfile,
    offsets: CompressibleOffsets,
    pre_step_graph: KernelGraph<GpuCompressibleSolver>,
    explicit_graph_first_order: KernelGraph<GpuCompressibleSolver>,
    explicit_graph: KernelGraph<GpuCompressibleSolver>,
    implicit_assembly_graph_first_order: KernelGraph<GpuCompressibleSolver>,
    implicit_grad_assembly_graph: KernelGraph<GpuCompressibleSolver>,
    implicit_snapshot_graph: KernelGraph<GpuCompressibleSolver>,
    implicit_apply_graph: KernelGraph<GpuCompressibleSolver>,
    primitive_update_graph: KernelGraph<GpuCompressibleSolver>,
    needs_gradients: bool,
}

impl GpuCompressibleSolver {
    fn context_ref(&self) -> &GpuContext {
        &self.context
    }

    fn state_size_bytes(&self) -> u64 {
        self.num_cells as u64 * self.offsets.stride as u64 * 4
    }

    fn explicit_plan_pre_step_graph(
        solver: &GpuCompressibleSolver,
    ) -> &KernelGraph<GpuCompressibleSolver> {
        &solver.pre_step_graph
    }

    fn explicit_plan_explicit_graph(
        solver: &GpuCompressibleSolver,
    ) -> &KernelGraph<GpuCompressibleSolver> {
        if solver.needs_gradients {
            &solver.explicit_graph
        } else {
            &solver.explicit_graph_first_order
        }
    }

    fn build_explicit_plan() -> ExecutionPlan<GpuCompressibleSolver> {
        ExecutionPlan::new(
            GpuCompressibleSolver::context_ref,
            vec![
                PlanNode::Graph(GraphNode {
                    label: "compressible:pre_step",
                    graph: GpuCompressibleSolver::explicit_plan_pre_step_graph,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Graph(GraphNode {
                    label: "compressible:explicit_graph",
                    graph: GpuCompressibleSolver::explicit_plan_explicit_graph,
                    mode: GraphExecMode::SplitTimed,
                }),
            ],
        )
    }

    fn explicit_plan() -> &'static ExecutionPlan<GpuCompressibleSolver> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<GpuCompressibleSolver>> =
            std::sync::OnceLock::new();
        PLAN.get_or_init(GpuCompressibleSolver::build_explicit_plan)
    }

    fn implicit_iter_plan_grad_assembly_graph(
        solver: &GpuCompressibleSolver,
    ) -> &KernelGraph<GpuCompressibleSolver> {
        if solver.needs_gradients {
            &solver.implicit_grad_assembly_graph
        } else {
            &solver.implicit_assembly_graph_first_order
        }
    }

    fn implicit_iter_plan_snapshot_graph(
        solver: &GpuCompressibleSolver,
    ) -> &KernelGraph<GpuCompressibleSolver> {
        &solver.implicit_snapshot_graph
    }

    fn implicit_iter_plan_apply_graph(
        solver: &GpuCompressibleSolver,
    ) -> &KernelGraph<GpuCompressibleSolver> {
        &solver.implicit_apply_graph
    }

    fn implicit_host_solve_fgmres(solver: &mut GpuCompressibleSolver) {
        let mut iter_stats =
            solver.solve_compressible_fgmres(solver.implicit_max_restart, solver.implicit_tol);
        if !iter_stats.converged {
            let retry_stats = solver.solve_compressible_fgmres(
                solver.implicit_retry_restart,
                solver.implicit_retry_tol,
            );
            if retry_stats.converged || retry_stats.residual < iter_stats.residual {
                iter_stats = retry_stats;
            }
        }
        solver.implicit_last_stats = iter_stats;
    }

    fn implicit_host_set_alpha_for_apply(solver: &mut GpuCompressibleSolver) {
        let apply_alpha = if solver.implicit_last_stats.converged {
            solver.implicit_base_alpha_u
        } else {
            solver.implicit_base_alpha_u * solver.nonconverged_relax
        };
        if (solver.constants.alpha_u - apply_alpha).abs() > 1e-6 {
            solver.constants.alpha_u = apply_alpha;
            solver.update_constants();
        }
    }

    fn implicit_host_restore_alpha(solver: &mut GpuCompressibleSolver) {
        if (solver.constants.alpha_u - solver.implicit_base_alpha_u).abs() > 1e-6 {
            solver.constants.alpha_u = solver.implicit_base_alpha_u;
            solver.update_constants();
        }
    }

    fn build_implicit_iter_plan() -> ExecutionPlan<GpuCompressibleSolver> {
        ExecutionPlan::new(
            GpuCompressibleSolver::context_ref,
            vec![
                PlanNode::Graph(GraphNode {
                    label: "compressible:implicit_grad_assembly",
                    graph: GpuCompressibleSolver::implicit_iter_plan_grad_assembly_graph,
                    mode: GraphExecMode::SplitTimed,
                }),
                PlanNode::Host(crate::solver::gpu::execution_plan::HostNode {
                    label: "compressible:implicit_fgmres",
                    run: GpuCompressibleSolver::implicit_host_solve_fgmres,
                }),
                PlanNode::Graph(GraphNode {
                    label: "compressible:implicit_snapshot",
                    graph: GpuCompressibleSolver::implicit_iter_plan_snapshot_graph,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Host(crate::solver::gpu::execution_plan::HostNode {
                    label: "compressible:implicit_set_alpha",
                    run: GpuCompressibleSolver::implicit_host_set_alpha_for_apply,
                }),
                PlanNode::Graph(GraphNode {
                    label: "compressible:implicit_apply",
                    graph: GpuCompressibleSolver::implicit_iter_plan_apply_graph,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Host(crate::solver::gpu::execution_plan::HostNode {
                    label: "compressible:implicit_restore_alpha",
                    run: GpuCompressibleSolver::implicit_host_restore_alpha,
                }),
            ],
        )
    }

    fn implicit_iter_plan() -> &'static ExecutionPlan<GpuCompressibleSolver> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<GpuCompressibleSolver>> =
            std::sync::OnceLock::new();
        PLAN.get_or_init(GpuCompressibleSolver::build_implicit_iter_plan)
    }

    fn build_pre_step_graph() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![
            KernelNode::CopyBuffer(CopyBufferNode {
                label: "compressible:copy_old_to_state",
                src: |s| &s.b_state_old,
                dst: |s| &s.b_state,
                size_bytes: GpuCompressibleSolver::state_size_bytes,
            }),
            KernelNode::CopyBuffer(CopyBufferNode {
                label: "compressible:copy_state_to_iter",
                src: |s| &s.b_state,
                dst: |s| &s.b_state_iter,
                size_bytes: GpuCompressibleSolver::state_size_bytes,
            }),
        ])
    }

    fn build_explicit_graph() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![
            KernelNode::Compute(ComputeNode {
                label: "compressible:gradients",
                pipeline: |s| &s.pipeline_gradients,
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:flux_kt",
                pipeline: |s| &s.pipeline_flux,
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_faces,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:explicit_update",
                pipeline: |s| &s.pipeline_explicit_update,
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:primitive_update",
                pipeline: |s| &s.pipeline_update,
                bind_groups: compressible_bind_fields_only,
                workgroups: compressible_workgroups_cells,
            }),
        ])
    }

    fn build_explicit_graph_first_order() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![
            KernelNode::Compute(ComputeNode {
                label: "compressible:flux_kt",
                pipeline: |s| &s.pipeline_flux,
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_faces,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:explicit_update",
                pipeline: |s| &s.pipeline_explicit_update,
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:primitive_update",
                pipeline: |s| &s.pipeline_update,
                bind_groups: compressible_bind_fields_only,
                workgroups: compressible_workgroups_cells,
            }),
        ])
    }

    fn build_implicit_grad_assembly_graph() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![
            KernelNode::Compute(ComputeNode {
                label: "compressible:gradients",
                pipeline: |s| &s.pipeline_gradients,
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:assembly",
                pipeline: |s| &s.pipeline_assembly,
                bind_groups: compressible_bind_mesh_fields_solver,
                workgroups: compressible_workgroups_cells,
            }),
        ])
    }

    fn build_implicit_assembly_graph_first_order() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "compressible:assembly",
            pipeline: |s| &s.pipeline_assembly,
            bind_groups: compressible_bind_mesh_fields_solver,
            workgroups: compressible_workgroups_cells,
        })])
    }

    fn update_needs_gradients(&mut self) {
        let scheme = Scheme::from_gpu_id(self.constants.scheme).unwrap_or(Scheme::Upwind);
        let registry = SchemeRegistry::new(scheme);
        self.needs_gradients = expand_schemes(&default_compressible_model().system, &registry)
            .map(|expansion| expansion.needs_gradients())
            .unwrap_or(true);
    }

    fn build_implicit_snapshot_graph() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![KernelNode::CopyBuffer(CopyBufferNode {
            label: "compressible:snapshot_state_to_iter",
            src: |s| &s.b_state,
            dst: |s| &s.b_state_iter,
            size_bytes: GpuCompressibleSolver::state_size_bytes,
        })])
    }

    fn build_implicit_apply_graph() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "compressible:apply",
            pipeline: |s| &s.pipeline_apply,
            bind_groups: compressible_bind_apply_fields_solver,
            workgroups: compressible_workgroups_cells,
        })])
    }

    fn build_primitive_update_graph() -> KernelGraph<GpuCompressibleSolver> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "compressible:primitive_update",
            pipeline: |s| &s.pipeline_update,
            bind_groups: compressible_bind_fields_only,
            workgroups: compressible_workgroups_cells,
        })])
    }

    pub async fn new(
        mesh: &Mesh,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Self {
        let context = GpuContext::new(device, queue).await;
        let num_cells = mesh.cell_cx.len() as u32;
        let num_faces = mesh.face_owner.len() as u32;
        let profile = CompressibleProfile::new();

        let model = default_compressible_model();
        let unknowns_per_cell = model.system.unknowns_per_cell();
        let num_unknowns = num_cells * unknowns_per_cell;
        let layout = &model.state_layout;
        let offsets = CompressibleOffsets {
            stride: layout.stride(),
            rho: layout.offset_for("rho").expect("rho offset missing"),
            rho_u: layout.offset_for("rho_u").expect("rho_u offset missing"),
            rho_e: layout.offset_for("rho_e").expect("rho_e offset missing"),
            p: layout.offset_for("p").expect("p offset missing"),
            u: layout.offset_for("u").expect("u offset missing"),
        };

        let mesh_res = mesh::init_mesh(&context.device, mesh);

        let field_buffers = init_compressible_field_buffers(
            &context.device,
            num_cells,
            num_faces,
            PackedStateConfig {
                state_stride: offsets.stride,
                flux_stride: 4,
            },
        );

        let pipeline_assembly =
            generated_assembly::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_apply =
            generated_apply::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_gradients =
            generated_gradients::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_flux =
            generated_flux_kt::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_explicit_update =
            explicit_update::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_update =
            generated_update::compute::create_main_pipeline_embed_source(&context.device);

        let mesh_layout = context
            .device
            .create_bind_group_layout(&generated_assembly::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
        let bg_mesh = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compressible Mesh Bind Group"),
                layout: &mesh_layout,
                entries: &generated_assembly::WgpuBindGroup0Entries::new(
                    generated_assembly::WgpuBindGroup0EntriesParams {
                        face_owner: mesh_res.b_face_owner.as_entire_buffer_binding(),
                        face_neighbor: mesh_res.b_face_neighbor.as_entire_buffer_binding(),
                        face_areas: mesh_res.b_face_areas.as_entire_buffer_binding(),
                        face_normals: mesh_res.b_face_normals.as_entire_buffer_binding(),
                        cell_centers: mesh_res.b_cell_centers.as_entire_buffer_binding(),
                        cell_vols: mesh_res.b_cell_vols.as_entire_buffer_binding(),
                        cell_face_offsets: mesh_res.b_cell_face_offsets.as_entire_buffer_binding(),
                        cell_faces: mesh_res.b_cell_faces.as_entire_buffer_binding(),
                        cell_face_matrix_indices: mesh_res
                            .b_cell_face_matrix_indices
                            .as_entire_buffer_binding(),
                        diagonal_indices: mesh_res.b_diagonal_indices.as_entire_buffer_binding(),
                        face_boundary: mesh_res.b_face_boundary.as_entire_buffer_binding(),
                        face_centers: mesh_res.b_face_centers.as_entire_buffer_binding(),
                    },
                )
                .into_array(),
            });

        let fields_layout = context
            .device
            .create_bind_group_layout(&generated_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
        let fields_res =
            create_compressible_field_bind_groups(&context.device, field_buffers, &fields_layout);

        let b_scalar_row_offsets =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Compressible Scalar Row Offsets"),
                    contents: cast_slice(&mesh_res.row_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let scalar_row_offsets = mesh_res.row_offsets.clone();
        let scalar_col_indices = mesh_res.col_indices.clone();
        let (row_offsets, col_indices) = build_block_csr(
            &mesh_res.row_offsets,
            &mesh_res.col_indices,
            unknowns_per_cell,
        );
        let block_row_offsets = row_offsets.clone();
        let block_col_indices = col_indices.clone();
        let matrix_res = matrix::init_matrix(&context.device, &row_offsets, &col_indices);
        let b_rhs = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible RHS"),
            size: num_unknowns as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_x = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible Solution"),
            size: num_unknowns as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let solver_layout = context
            .device
            .create_bind_group_layout(&generated_assembly::WgpuBindGroup2::LAYOUT_DESCRIPTOR);
        let bg_solver = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compressible Solver Bind Group"),
                layout: &solver_layout,
                entries: &generated_assembly::WgpuBindGroup2Entries::new(
                    generated_assembly::WgpuBindGroup2EntriesParams {
                        matrix_values: matrix_res.b_matrix_values.as_entire_buffer_binding(),
                        rhs: b_rhs.as_entire_buffer_binding(),
                        scalar_row_offsets: b_scalar_row_offsets.as_entire_buffer_binding(),
                    },
                )
                .into_array(),
            });

        let apply_fields_layout = context
            .device
            .create_bind_group_layout(&generated_apply::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
        let mut bg_apply_fields_ping_pong = Vec::new();
        for i in 0..3 {
            let (idx_state, idx_old, idx_old_old) = match i {
                0 => (0, 1, 2),
                1 => (2, 0, 1),
                2 => (1, 2, 0),
                _ => (0, 1, 2),
            };
            let bg = context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Compressible Apply Fields Bind Group {}", i)),
                    layout: &apply_fields_layout,
                    entries: &generated_apply::WgpuBindGroup0Entries::new(
                        generated_apply::WgpuBindGroup0EntriesParams {
                            state: fields_res.state_buffers[idx_state].as_entire_buffer_binding(),
                            state_old: fields_res.state_buffers[idx_old].as_entire_buffer_binding(),
                            state_old_old: fields_res.state_buffers[idx_old_old]
                                .as_entire_buffer_binding(),
                            state_iter: fields_res.b_state_iter.as_entire_buffer_binding(),
                            fluxes: fields_res.b_fluxes.as_entire_buffer_binding(),
                            constants: fields_res.b_constants.as_entire_buffer_binding(),
                            grad_rho: fields_res.b_grad_rho.as_entire_buffer_binding(),
                            grad_rho_u_x: fields_res.b_grad_rho_u_x.as_entire_buffer_binding(),
                            grad_rho_u_y: fields_res.b_grad_rho_u_y.as_entire_buffer_binding(),
                            grad_rho_e: fields_res.b_grad_rho_e.as_entire_buffer_binding(),
                        },
                    )
                    .into_array(),
                });
            bg_apply_fields_ping_pong.push(bg);
        }
        let bg_apply_fields = bg_apply_fields_ping_pong[0].clone();
        let apply_solver_layout = context
            .device
            .create_bind_group_layout(&generated_apply::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
        let bg_apply_solver = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compressible Apply Solver Bind Group"),
                layout: &apply_solver_layout,
                entries: &generated_apply::WgpuBindGroup1Entries::new(
                    generated_apply::WgpuBindGroup1EntriesParams {
                        solution: b_x.as_entire_buffer_binding(),
                    },
                )
                .into_array(),
            });

        let mut solver = Self {
            context,
            num_cells,
            num_faces,
            num_unknowns,
            state_step_index: 0,
            state_buffers: fields_res.state_buffers,
            bg_fields_ping_pong: fields_res.bg_fields_ping_pong,
            bg_fields: fields_res.bg_fields,
            bg_apply_fields_ping_pong,
            bg_apply_fields,
            b_state: fields_res.b_state,
            b_state_old: fields_res.b_state_old,
            b_state_old_old: fields_res.b_state_old_old,
            b_state_iter: fields_res.b_state_iter,
            b_fluxes: fields_res.b_fluxes,
            b_grad_rho: fields_res.b_grad_rho,
            b_grad_rho_u_x: fields_res.b_grad_rho_u_x,
            b_grad_rho_u_y: fields_res.b_grad_rho_u_y,
            b_grad_rho_e: fields_res.b_grad_rho_e,
            b_constants: fields_res.b_constants,
            constants: fields_res.constants,
            b_row_offsets: matrix_res.b_row_offsets,
            b_col_indices: matrix_res.b_col_indices,
            b_matrix_values: matrix_res.b_matrix_values,
            b_rhs,
            b_x,
            b_scalar_row_offsets,
            bg_mesh,
            bg_solver,
            bg_apply_solver,
            pipeline_assembly,
            pipeline_apply,
            pipeline_gradients,
            pipeline_flux,
            pipeline_explicit_update,
            pipeline_update,
            fgmres_resources: None,
            outer_iters: 1,
            nonconverged_relax: 0.5,
            scalar_row_offsets,
            scalar_col_indices,
            block_row_offsets,
            block_col_indices,
            implicit_tol: 0.0,
            implicit_max_restart: 1,
            implicit_retry_tol: 0.0,
            implicit_retry_restart: 1,
            implicit_base_alpha_u: 0.0,
            implicit_last_stats: LinearSolverStats::default(),
            profile,
            offsets,
            pre_step_graph: GpuCompressibleSolver::build_pre_step_graph(),
            explicit_graph_first_order: GpuCompressibleSolver::build_explicit_graph_first_order(),
            explicit_graph: GpuCompressibleSolver::build_explicit_graph(),
            implicit_assembly_graph_first_order:
                GpuCompressibleSolver::build_implicit_assembly_graph_first_order(),
            implicit_grad_assembly_graph: GpuCompressibleSolver::build_implicit_grad_assembly_graph(
            ),
            implicit_snapshot_graph: GpuCompressibleSolver::build_implicit_snapshot_graph(),
            implicit_apply_graph: GpuCompressibleSolver::build_implicit_apply_graph(),
            primitive_update_graph: GpuCompressibleSolver::build_primitive_update_graph(),
            needs_gradients: false,
        }
        ;
        solver.update_needs_gradients();
        solver
    }

    pub fn initialize_history(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Initialize History Encoder"),
                });
        let state_size = self.num_cells as u64 * self.offsets.stride as u64 * 4;
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old, 0, state_size);
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old_old, 0, state_size);
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub fn set_dt(&mut self, dt: f32) {
        if self.constants.time <= 0.0 {
            self.constants.dt_old = dt;
        } else {
            self.constants.dt_old = self.constants.dt;
        }
        self.constants.dt = dt;
        self.update_constants();
    }

    pub fn set_dtau(&mut self, dtau: f32) {
        self.constants.dtau = dtau;
        self.update_constants();
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        self.constants.viscosity = mu;
        self.update_constants();
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        self.constants.time_scheme = scheme;
        self.update_constants();
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        self.constants.inlet_velocity = velocity;
        self.update_constants();
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        self.constants.scheme = scheme;
        self.update_constants();
        self.update_needs_gradients();
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        self.constants.alpha_u = alpha_u;
        self.update_constants();
    }

    pub fn set_precond_type(&mut self, precond_type: u32) {
        self.constants.precond_type = precond_type;
        self.update_constants();
    }

    pub fn set_precond_model(&mut self, model: u32) {
        self.constants.precond_model = model;
        self.update_constants();
    }

    pub fn set_precond_theta_floor(&mut self, floor: f32) {
        self.constants.precond_theta_floor = floor;
        self.update_constants();
    }

    pub fn set_outer_iters(&mut self, iters: usize) {
        self.outer_iters = iters.max(1);
    }

    pub fn set_nonconverged_relax(&mut self, relax: f32) {
        self.nonconverged_relax = relax.max(0.0);
    }

    pub fn set_uniform_state(&self, rho: f32, u: [f32; 2], p: f32) {
        let gamma = 1.4f32;
        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let rho_e = p / (gamma - 1.0) + ke;

        let mut state = vec![0.0f32; self.num_cells as usize * self.offsets.stride as usize];
        for cell in 0..self.num_cells as usize {
            let base = cell * self.offsets.stride as usize;
            state[base + self.offsets.rho as usize] = rho;
            state[base + self.offsets.rho_u as usize] = rho * u[0];
            state[base + self.offsets.rho_u as usize + 1] = rho * u[1];
            state[base + self.offsets.rho_e as usize] = rho_e;
            state[base + self.offsets.p as usize] = p;
            state[base + self.offsets.u as usize] = u[0];
            state[base + self.offsets.u as usize + 1] = u[1];
        }

        self.write_state_all(&state);
    }

    pub fn set_state_fields(&self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        assert_eq!(rho.len(), self.num_cells as usize);
        assert_eq!(u.len(), self.num_cells as usize);
        assert_eq!(p.len(), self.num_cells as usize);

        let gamma = 1.4f32;
        let mut state = vec![0.0f32; self.num_cells as usize * self.offsets.stride as usize];

        for cell in 0..self.num_cells as usize {
            let base = cell * self.offsets.stride as usize;
            let rho_val = rho[cell];
            let u_val = u[cell];
            let p_val = p[cell];
            let ke = 0.5 * rho_val * (u_val[0] * u_val[0] + u_val[1] * u_val[1]);
            let rho_e = p_val / (gamma - 1.0) + ke;

            state[base + self.offsets.rho as usize] = rho_val;
            state[base + self.offsets.rho_u as usize] = rho_val * u_val[0];
            state[base + self.offsets.rho_u as usize + 1] = rho_val * u_val[1];
            state[base + self.offsets.rho_e as usize] = rho_e;
            state[base + self.offsets.p as usize] = p_val;
            state[base + self.offsets.u as usize] = u_val[0];
            state[base + self.offsets.u as usize + 1] = u_val[1];
        }

        self.write_state_all(&state);
    }

    pub fn step(&mut self) {
        let _ = self.step_with_stats();
    }

    pub fn step_with_stats(&mut self) -> Vec<LinearSolverStats> {
        let step_start = std::time::Instant::now();

        self.state_step_index = (self.state_step_index + 1) % 3;
        self.bg_fields = self.bg_fields_ping_pong[self.state_step_index].clone();
        self.bg_apply_fields = self.bg_apply_fields_ping_pong[self.state_step_index].clone();

        let (idx_state, idx_old, idx_old_old) = ping_pong_indices(self.state_step_index);
        self.b_state = self.state_buffers[idx_state].clone();
        self.b_state_old = self.state_buffers[idx_old].clone();
        self.b_state_old_old = self.state_buffers[idx_old_old].clone();

        self.constants.time += self.constants.dt;
        self.update_constants();

        // Explicit KT update (rhoCentralFoam-like) for transient Euler timestepping.
        // The implicit FGMRES path is retained for BDF2/pseudo-time (dtau) workflows.
        let use_explicit = self.constants.time_scheme == 0 && self.constants.dtau <= 0.0;
        if use_explicit {
            let total_secs = step_start.elapsed().as_secs_f64();
            let plan_timings = GpuCompressibleSolver::explicit_plan().execute(self);
            let detail = plan_timings
                .graph_detail("compressible:explicit_graph")
                .expect("explicit_graph timings missing");
            let grad_secs = detail.seconds_for("compressible:gradients");
            let flux_secs = detail.seconds_for("compressible:flux_kt");
            let explicit_update_secs = detail.seconds_for("compressible:explicit_update");
            let primitive_update_secs = detail.seconds_for("compressible:primitive_update");
            self.profile.record(
                total_secs,
                grad_secs,
                flux_secs,
                0.0,
                0.0,
                explicit_update_secs + primitive_update_secs,
                0,
            );
            self.constants.dt_old = self.constants.dt;
            self.update_constants();
            return Vec::new();
        }

        self.pre_step_graph.execute(&self.context, &*self);

        let base_alpha_u = self.constants.alpha_u;
        let mut stats = Vec::with_capacity(self.outer_iters);
        let tol_base = env_f32("CFD2_COMP_FGMRES_TOL", 1e-8);
        let warm_scale = env_f32("CFD2_COMP_FGMRES_WARM_SCALE", 100.0).max(1.0);
        let warm_iters = env_usize("CFD2_COMP_FGMRES_WARM_ITERS", 4);
        let retry_scale = env_f32("CFD2_COMP_FGMRES_RETRY_SCALE", 0.5).clamp(0.0, 1.0);
        let max_restart = env_usize("CFD2_COMP_FGMRES_MAX_RESTART", 80).max(1);
        let retry_restart = env_usize("CFD2_COMP_FGMRES_RETRY_RESTART", 160).max(1);
        let mut grad_secs = 0.0f64;
        let mut assembly_secs = 0.0f64;
        let mut fgmres_secs = 0.0f64;
        let mut apply_secs = 0.0f64;
        let mut update_secs = 0.0f64;
        let mut fgmres_iters = 0u64;
        self.implicit_base_alpha_u = base_alpha_u;
        for outer_idx in 0..self.outer_iters {
            let tol = if outer_idx < warm_iters {
                tol_base * warm_scale
            } else {
                tol_base
            };
            let retry_tol = (tol * retry_scale).min(tol_base);
            self.implicit_tol = tol;
            self.implicit_retry_tol = retry_tol;
            self.implicit_max_restart = max_restart;
            self.implicit_retry_restart = retry_restart;

            let iter_timings = GpuCompressibleSolver::implicit_iter_plan().execute(self);
            let detail = iter_timings
                .graph_detail("compressible:implicit_grad_assembly")
                .expect("implicit_grad_assembly timings missing");
            grad_secs += detail.seconds_for("compressible:gradients");
            assembly_secs += detail.seconds_for("compressible:assembly");

            fgmres_secs += self.implicit_last_stats.time.as_secs_f64();
            fgmres_iters += self.implicit_last_stats.iterations as u64;
            stats.push(self.implicit_last_stats);

            apply_secs += iter_timings.seconds_for("compressible:implicit_snapshot")
                + iter_timings.seconds_for("compressible:implicit_set_alpha")
                + iter_timings.seconds_for("compressible:implicit_apply")
                + iter_timings.seconds_for("compressible:implicit_restore_alpha");
        }

        let stage_start = std::time::Instant::now();
        self.primitive_update_graph.execute(&self.context, &*self);
        update_secs += stage_start.elapsed().as_secs_f64();
        let total_secs = step_start.elapsed().as_secs_f64();
        self.profile.record(
            total_secs,
            grad_secs,
            assembly_secs,
            fgmres_secs,
            apply_secs,
            update_secs,
            fgmres_iters,
        );
        self.constants.dt_old = self.constants.dt;
        self.update_constants();
        stats
    }

    pub async fn get_rho(&self) -> Vec<f64> {
        let data = self.read_state().await;
        let stride = self.offsets.stride as usize;
        let offset = self.offsets.rho as usize;
        (0..self.num_cells as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let data = self.read_state().await;
        let stride = self.offsets.stride as usize;
        let offset = self.offsets.u as usize;
        (0..self.num_cells as usize)
            .map(|i| {
                let base = i * stride + offset;
                (data[base] as f64, data[base + 1] as f64)
            })
            .collect()
    }

    pub async fn get_p(&self) -> Vec<f64> {
        let data = self.read_state().await;
        let stride = self.offsets.stride as usize;
        let offset = self.offsets.p as usize;
        (0..self.num_cells as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
    }

    fn update_constants(&self) {
        self.context
            .queue
            .write_buffer(&self.b_constants, 0, bytemuck::bytes_of(&self.constants));
    }

    async fn read_state(&self) -> Vec<f32> {
        let byte_count = self.num_cells as u64 * self.offsets.stride as u64 * 4;
        let raw = self.read_buffer(&self.b_state, byte_count).await;
        cast_slice(&raw).to_vec()
    }

    pub(crate) async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Readback Encoder"),
                });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok().and_then(|v| v.ok()).unwrap();
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    pub(crate) async fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let raw = self.read_buffer(buffer, count as u64 * 4).await;
        cast_slice(&raw).to_vec()
    }

    fn write_state_all(&self, state: &[f32]) {
        let bytes = cast_slice(state);
        for buffer in &self.state_buffers {
            self.context.queue.write_buffer(buffer, 0, bytes);
        }
    }
}

fn compressible_bind_mesh_fields(s: &GpuCompressibleSolver, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, &s.bg_mesh, &[]);
    cpass.set_bind_group(1, &s.bg_fields, &[]);
}

fn compressible_bind_mesh_fields_solver(s: &GpuCompressibleSolver, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, &s.bg_mesh, &[]);
    cpass.set_bind_group(1, &s.bg_fields, &[]);
    cpass.set_bind_group(2, &s.bg_solver, &[]);
}

fn compressible_bind_fields_only(s: &GpuCompressibleSolver, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, &s.bg_fields, &[]);
}

fn compressible_bind_apply_fields_solver(s: &GpuCompressibleSolver, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, &s.bg_apply_fields, &[]);
    cpass.set_bind_group(1, &s.bg_apply_solver, &[]);
}

fn compressible_workgroups_cells(s: &GpuCompressibleSolver) -> (u32, u32, u32) {
    let workgroup_size = 64;
    (s.num_cells.div_ceil(workgroup_size), 1, 1)
}

fn compressible_workgroups_faces(s: &GpuCompressibleSolver) -> (u32, u32, u32) {
    let workgroup_size = 64;
    (s.num_faces.div_ceil(workgroup_size), 1, 1)
}

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|val| {
            let val = val.to_ascii_lowercase();
            matches!(val.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn ping_pong_indices(step_index: usize) -> (usize, usize, usize) {
    match step_index {
        0 => (0, 1, 2),
        1 => (2, 0, 1),
        2 => (1, 2, 0),
        _ => (0, 1, 2),
    }
}
