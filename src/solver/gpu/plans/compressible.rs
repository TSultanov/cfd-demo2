use crate::solver::gpu::plans::compressible_fgmres::CompressibleFgmresResources;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::execution_plan::{ExecutionPlan, GraphExecMode, GraphNode, PlanNode};
use crate::solver::gpu::kernel_graph::{ComputeNode, CopyBufferNode, KernelGraph, KernelNode};
use crate::solver::gpu::model_defaults::default_compressible_model;
use crate::solver::gpu::modules::compressible_kernels::CompressibleKernelsModule;
use crate::solver::gpu::init::compressible_fields::create_compressible_field_bind_groups;
use crate::solver::gpu::modules::compressible_lowering::{
    CompressibleLinearPorts, CompressibleLowered, CompressibleMatrixPorts,
};
use crate::solver::gpu::modules::compressible_kernels::{CompressibleBindGroups, CompressiblePipeline};
use crate::solver::gpu::modules::graph::{DispatchKind, ModuleGraph, ModuleNode, RuntimeDims};
use crate::solver::gpu::modules::ports::PortSpace;
use crate::solver::gpu::structs::{GpuConstants, GpuLowMachParams, LinearSolverStats, PreconditionerType};
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::{expand_schemes, SchemeRegistry};
use crate::solver::scheme::Scheme;
use bytemuck::cast_slice;
use std::env;

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

pub(crate) struct CompressiblePlanResources {
    pub context: GpuContext,
    pub num_cells: u32,
    pub num_faces: u32,
    pub num_unknowns: u32,
    pub state_step_index: usize,
    pub state_buffers: Vec<wgpu::Buffer>,
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
    pub b_low_mach_params: wgpu::Buffer,
    pub constants: GpuConstants,
    pub low_mach_params: GpuLowMachParams,
    pub preconditioner: PreconditionerType,
    pub linear_ports: CompressibleLinearPorts,
    pub matrix_ports: CompressibleMatrixPorts,
    pub port_space: PortSpace,
    pub kernels: CompressibleKernelsModule,
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
    pre_step_graph: KernelGraph<CompressiblePlanResources>,
    explicit_graph_first_order: KernelGraph<CompressiblePlanResources>,
    explicit_graph: KernelGraph<CompressiblePlanResources>,
    implicit_assembly_graph_first_order: KernelGraph<CompressiblePlanResources>,
    implicit_grad_assembly_graph: KernelGraph<CompressiblePlanResources>,
    implicit_snapshot_graph: KernelGraph<CompressiblePlanResources>,
    implicit_apply_graph: KernelGraph<CompressiblePlanResources>,
    primitive_update_graph: KernelGraph<CompressiblePlanResources>,
    needs_gradients: bool,
    explicit_module_graph: ModuleGraph<CompressibleKernelsModule>,
    explicit_module_graph_first_order: ModuleGraph<CompressibleKernelsModule>,
}

impl CompressiblePlanResources {
    fn context_ref(&self) -> &GpuContext {
        &self.context
    }

    fn state_size_bytes(&self) -> u64 {
        self.num_cells as u64 * self.offsets.stride as u64 * 4
    }

    fn explicit_plan_pre_step_graph(
        solver: &CompressiblePlanResources,
    ) -> &KernelGraph<CompressiblePlanResources> {
        &solver.pre_step_graph
    }

    fn explicit_plan_explicit_graph(
        solver: &CompressiblePlanResources,
    ) -> &KernelGraph<CompressiblePlanResources> {
        if solver.needs_gradients {
            &solver.explicit_graph
        } else {
            &solver.explicit_graph_first_order
        }
    }

    fn build_explicit_plan() -> ExecutionPlan<CompressiblePlanResources> {
        ExecutionPlan::new(
            CompressiblePlanResources::context_ref,
            vec![
                PlanNode::Graph(GraphNode {
                    label: "compressible:pre_step",
                    graph: CompressiblePlanResources::explicit_plan_pre_step_graph,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Graph(GraphNode {
                    label: "compressible:explicit_graph",
                    graph: CompressiblePlanResources::explicit_plan_explicit_graph,
                    mode: GraphExecMode::SplitTimed,
                }),
            ],
        )
    }

    fn explicit_plan() -> &'static ExecutionPlan<CompressiblePlanResources> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<CompressiblePlanResources>> =
            std::sync::OnceLock::new();
        PLAN.get_or_init(CompressiblePlanResources::build_explicit_plan)
    }

    fn build_explicit_module_graph(include_gradients: bool) -> ModuleGraph<CompressibleKernelsModule> {
        let mut nodes = Vec::new();
        if include_gradients {
            nodes.push(ModuleNode::Compute(crate::solver::gpu::modules::graph::ComputeSpec {
                label: "compressible:gradients",
                pipeline: CompressiblePipeline::Gradients,
                bind: CompressibleBindGroups::MeshFields,
                dispatch: DispatchKind::Cells,
            }));
        }
        nodes.push(ModuleNode::Compute(crate::solver::gpu::modules::graph::ComputeSpec {
            label: "compressible:flux_kt",
            pipeline: CompressiblePipeline::FluxKt,
            bind: CompressibleBindGroups::MeshFields,
            dispatch: DispatchKind::Faces,
        }));
        nodes.push(ModuleNode::Compute(crate::solver::gpu::modules::graph::ComputeSpec {
            label: "compressible:explicit_update",
            pipeline: CompressiblePipeline::ExplicitUpdate,
            bind: CompressibleBindGroups::MeshFields,
            dispatch: DispatchKind::Cells,
        }));
        nodes.push(ModuleNode::Compute(crate::solver::gpu::modules::graph::ComputeSpec {
            label: "compressible:primitive_update",
            pipeline: CompressiblePipeline::PrimitiveUpdate,
            bind: CompressibleBindGroups::FieldsOnly,
            dispatch: DispatchKind::Cells,
        }));
        ModuleGraph::new(nodes)
    }

    fn implicit_iter_plan_grad_assembly_graph(
        solver: &CompressiblePlanResources,
    ) -> &KernelGraph<CompressiblePlanResources> {
        if solver.needs_gradients {
            &solver.implicit_grad_assembly_graph
        } else {
            &solver.implicit_assembly_graph_first_order
        }
    }

    fn implicit_iter_plan_snapshot_graph(
        solver: &CompressiblePlanResources,
    ) -> &KernelGraph<CompressiblePlanResources> {
        &solver.implicit_snapshot_graph
    }

    fn implicit_iter_plan_apply_graph(
        solver: &CompressiblePlanResources,
    ) -> &KernelGraph<CompressiblePlanResources> {
        &solver.implicit_apply_graph
    }

    fn implicit_host_solve_fgmres(solver: &mut CompressiblePlanResources) {
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

    fn implicit_host_set_alpha_for_apply(solver: &mut CompressiblePlanResources) {
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

    fn implicit_host_restore_alpha(solver: &mut CompressiblePlanResources) {
        if (solver.constants.alpha_u - solver.implicit_base_alpha_u).abs() > 1e-6 {
            solver.constants.alpha_u = solver.implicit_base_alpha_u;
            solver.update_constants();
        }
    }

    fn build_implicit_iter_plan() -> ExecutionPlan<CompressiblePlanResources> {
        ExecutionPlan::new(
            CompressiblePlanResources::context_ref,
            vec![
                PlanNode::Graph(GraphNode {
                    label: "compressible:implicit_grad_assembly",
                    graph: CompressiblePlanResources::implicit_iter_plan_grad_assembly_graph,
                    mode: GraphExecMode::SplitTimed,
                }),
                PlanNode::Host(crate::solver::gpu::execution_plan::HostNode {
                    label: "compressible:implicit_fgmres",
                    run: CompressiblePlanResources::implicit_host_solve_fgmres,
                }),
                PlanNode::Graph(GraphNode {
                    label: "compressible:implicit_snapshot",
                    graph: CompressiblePlanResources::implicit_iter_plan_snapshot_graph,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Host(crate::solver::gpu::execution_plan::HostNode {
                    label: "compressible:implicit_set_alpha",
                    run: CompressiblePlanResources::implicit_host_set_alpha_for_apply,
                }),
                PlanNode::Graph(GraphNode {
                    label: "compressible:implicit_apply",
                    graph: CompressiblePlanResources::implicit_iter_plan_apply_graph,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Host(crate::solver::gpu::execution_plan::HostNode {
                    label: "compressible:implicit_restore_alpha",
                    run: CompressiblePlanResources::implicit_host_restore_alpha,
                }),
            ],
        )
    }

    fn implicit_iter_plan() -> &'static ExecutionPlan<CompressiblePlanResources> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<CompressiblePlanResources>> =
            std::sync::OnceLock::new();
        PLAN.get_or_init(CompressiblePlanResources::build_implicit_iter_plan)
    }

    fn build_pre_step_graph() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![
            KernelNode::CopyBuffer(CopyBufferNode {
                label: "compressible:copy_old_to_state",
                src: |s| &s.b_state_old,
                dst: |s| &s.b_state,
                size_bytes: CompressiblePlanResources::state_size_bytes,
            }),
            KernelNode::CopyBuffer(CopyBufferNode {
                label: "compressible:copy_state_to_iter",
                src: |s| &s.b_state,
                dst: |s| &s.b_state_iter,
                size_bytes: CompressiblePlanResources::state_size_bytes,
            }),
        ])
    }

    fn build_explicit_graph() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![
            KernelNode::Compute(ComputeNode {
                label: "compressible:gradients",
                pipeline: |s| s.kernels.pipeline_gradients(),
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:flux_kt",
                pipeline: |s| s.kernels.pipeline_flux(),
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_faces,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:explicit_update",
                pipeline: |s| s.kernels.pipeline_explicit_update(),
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:primitive_update",
                pipeline: |s| s.kernels.pipeline_update(),
                bind_groups: compressible_bind_fields_only,
                workgroups: compressible_workgroups_cells,
            }),
        ])
    }

    fn build_explicit_graph_first_order() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![
            KernelNode::Compute(ComputeNode {
                label: "compressible:flux_kt",
                pipeline: |s| s.kernels.pipeline_flux(),
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_faces,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:explicit_update",
                pipeline: |s| s.kernels.pipeline_explicit_update(),
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:primitive_update",
                pipeline: |s| s.kernels.pipeline_update(),
                bind_groups: compressible_bind_fields_only,
                workgroups: compressible_workgroups_cells,
            }),
        ])
    }

    fn build_implicit_grad_assembly_graph() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![
            KernelNode::Compute(ComputeNode {
                label: "compressible:gradients",
                pipeline: |s| s.kernels.pipeline_gradients(),
                bind_groups: compressible_bind_mesh_fields,
                workgroups: compressible_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "compressible:assembly",
                pipeline: |s| s.kernels.pipeline_assembly(),
                bind_groups: compressible_bind_mesh_fields_solver,
                workgroups: compressible_workgroups_cells,
            }),
        ])
    }

    fn build_implicit_assembly_graph_first_order() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "compressible:assembly",
            pipeline: |s| s.kernels.pipeline_assembly(),
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

    fn build_implicit_snapshot_graph() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![KernelNode::CopyBuffer(CopyBufferNode {
            label: "compressible:snapshot_state_to_iter",
            src: |s| &s.b_state,
            dst: |s| &s.b_state_iter,
            size_bytes: CompressiblePlanResources::state_size_bytes,
        })])
    }

    fn build_implicit_apply_graph() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "compressible:apply",
            pipeline: |s| s.kernels.pipeline_apply(),
            bind_groups: compressible_bind_apply_fields_solver,
            workgroups: compressible_workgroups_cells,
        })])
    }

    fn build_primitive_update_graph() -> KernelGraph<CompressiblePlanResources> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "compressible:primitive_update",
            pipeline: |s| s.kernels.pipeline_update(),
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
        let profile = CompressibleProfile::new();

        let model = default_compressible_model();
        let unknowns_per_cell = model.system.unknowns_per_cell();
        let layout = &model.state_layout;
        let offsets = CompressibleOffsets {
            stride: layout.stride(),
            rho: layout.offset_for("rho").expect("rho offset missing"),
            rho_u: layout.offset_for("rho_u").expect("rho_u offset missing"),
            rho_e: layout.offset_for("rho_e").expect("rho_e offset missing"),
            p: layout.offset_for("p").expect("p offset missing"),
            u: layout.offset_for("u").expect("u offset missing"),
        };

        let lowered = CompressibleLowered::lower(
            &context.device,
            mesh,
            &model.state_layout,
            unknowns_per_cell,
            4,
        );
        let num_cells = lowered.common.num_cells;
        let num_faces = lowered.common.num_faces;
        let num_unknowns = num_cells * unknowns_per_cell;

        let fields_layout = context
            .device
            .create_bind_group_layout(
                &crate::solver::gpu::bindings::generated::compressible_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR,
            );
        let fields_res = create_compressible_field_bind_groups(
            &context.device,
            lowered.fields,
            &fields_layout,
        );

        let matrix_values = lowered.common.ports.clone_buffer(lowered.matrix_ports.values);
        let kernels = CompressibleKernelsModule::new(
            &context.device,
            &lowered.common.mesh,
            &fields_res,
            &matrix_values,
            &lowered.common.ports,
            lowered.ports,
            &lowered.scalar_row_offsets,
        );

        let port_space = lowered.common.ports;

        let mut solver = Self {
            context,
            num_cells,
            num_faces,
            num_unknowns,
            state_step_index: 0,
            state_buffers: fields_res.state_buffers,
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
            b_low_mach_params: fields_res.b_low_mach_params,
            low_mach_params: fields_res.low_mach_params,
            preconditioner: PreconditionerType::Jacobi,
            linear_ports: lowered.ports,
            matrix_ports: lowered.matrix_ports,
            port_space,
            kernels,
            fgmres_resources: None,
            outer_iters: 1,
            nonconverged_relax: 0.5,
            scalar_row_offsets: lowered.scalar_row_offsets,
            scalar_col_indices: lowered.scalar_col_indices,
            block_row_offsets: lowered.block_row_offsets,
            block_col_indices: lowered.block_col_indices,
            implicit_tol: 0.0,
            implicit_max_restart: 1,
            implicit_retry_tol: 0.0,
            implicit_retry_restart: 1,
            implicit_base_alpha_u: 0.0,
            implicit_last_stats: LinearSolverStats::default(),
            profile,
            offsets,
            pre_step_graph: CompressiblePlanResources::build_pre_step_graph(),
            explicit_graph_first_order: CompressiblePlanResources::build_explicit_graph_first_order(),
            explicit_graph: CompressiblePlanResources::build_explicit_graph(),
            implicit_assembly_graph_first_order:
                CompressiblePlanResources::build_implicit_assembly_graph_first_order(),
            implicit_grad_assembly_graph: CompressiblePlanResources::build_implicit_grad_assembly_graph(
            ),
            implicit_snapshot_graph: CompressiblePlanResources::build_implicit_snapshot_graph(),
            implicit_apply_graph: CompressiblePlanResources::build_implicit_apply_graph(),
            primitive_update_graph: CompressiblePlanResources::build_primitive_update_graph(),
            needs_gradients: false,
            explicit_module_graph: CompressiblePlanResources::build_explicit_module_graph(true),
            explicit_module_graph_first_order:
                CompressiblePlanResources::build_explicit_module_graph(false),
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

    pub fn set_precond_type(&mut self, precond_type: PreconditionerType) {
        self.preconditioner = precond_type;
    }

    pub fn set_precond_model(&mut self, model: u32) {
        self.low_mach_params.model = model;
        self.update_low_mach_params();
    }

    pub fn set_precond_theta_floor(&mut self, floor: f32) {
        self.low_mach_params.theta_floor = floor;
        self.update_low_mach_params();
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
        plan::step(self);
    }

    pub fn step_with_stats(&mut self) -> Vec<LinearSolverStats> {
        plan::step_with_stats(self)
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

    fn update_low_mach_params(&self) {
        self.context.queue.write_buffer(
            &self.b_low_mach_params,
            0,
            bytemuck::bytes_of(&self.low_mach_params),
        );
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

pub(crate) mod plan {
    use super::*;

    pub(crate) fn step(solver: &mut CompressiblePlanResources) {
        let _ = step_with_stats(solver);
    }

    pub(crate) fn step_with_stats(solver: &mut CompressiblePlanResources) -> Vec<LinearSolverStats> {
        let step_start = std::time::Instant::now();

        solver.state_step_index = (solver.state_step_index + 1) % 3;
        solver.kernels.set_step_index(solver.state_step_index);

        let (idx_state, idx_old, idx_old_old) = ping_pong_indices(solver.state_step_index);
        solver.b_state = solver.state_buffers[idx_state].clone();
        solver.b_state_old = solver.state_buffers[idx_old].clone();
        solver.b_state_old_old = solver.state_buffers[idx_old_old].clone();

        solver.constants.time += solver.constants.dt;
        solver.update_constants();

        // Explicit KT update (rhoCentralFoam-like) for transient Euler timestepping.
        // The implicit FGMRES path is retained for BDF2/pseudo-time (dtau) workflows.
        let use_explicit = solver.constants.time_scheme == 0 && solver.constants.dtau <= 0.0;
        if use_explicit {
            solver.pre_step_graph.execute(&solver.context, &*solver);

            let runtime = RuntimeDims {
                num_cells: solver.num_cells,
                num_faces: solver.num_faces,
            };
            let graph = if solver.needs_gradients {
                &solver.explicit_module_graph
            } else {
                &solver.explicit_module_graph_first_order
            };
            let timings = graph.execute_split_timed(&solver.context, &solver.kernels, runtime);

            let total_secs = step_start.elapsed().as_secs_f64();
            let grad_secs = timings.seconds_for("compressible:gradients");
            let flux_secs = timings.seconds_for("compressible:flux_kt");
            let explicit_update_secs = timings.seconds_for("compressible:explicit_update");
            let primitive_update_secs = timings.seconds_for("compressible:primitive_update");
            solver.profile.record(
                total_secs,
                grad_secs,
                flux_secs,
                0.0,
                0.0,
                explicit_update_secs + primitive_update_secs,
                0,
            );
            solver.constants.dt_old = solver.constants.dt;
            solver.update_constants();
            return Vec::new();
        }

        solver.pre_step_graph.execute(&solver.context, &*solver);

        let base_alpha_u = solver.constants.alpha_u;
        let mut stats = Vec::with_capacity(solver.outer_iters);
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
        solver.implicit_base_alpha_u = base_alpha_u;
        for outer_idx in 0..solver.outer_iters {
            let tol = if outer_idx < warm_iters {
                tol_base * warm_scale
            } else {
                tol_base
            };
            let retry_tol = (tol * retry_scale).min(tol_base);
            solver.implicit_tol = tol;
            solver.implicit_retry_tol = retry_tol;
            solver.implicit_max_restart = max_restart;
            solver.implicit_retry_restart = retry_restart;

            let iter_timings = CompressiblePlanResources::implicit_iter_plan().execute(solver);
            let detail = iter_timings
                .graph_detail("compressible:implicit_grad_assembly")
                .expect("implicit_grad_assembly timings missing");
            grad_secs += detail.seconds_for("compressible:gradients");
            assembly_secs += detail.seconds_for("compressible:assembly");

            fgmres_secs += solver.implicit_last_stats.time.as_secs_f64();
            fgmres_iters += solver.implicit_last_stats.iterations as u64;
            stats.push(solver.implicit_last_stats);

            apply_secs += iter_timings.seconds_for("compressible:implicit_snapshot")
                + iter_timings.seconds_for("compressible:implicit_set_alpha")
                + iter_timings.seconds_for("compressible:implicit_apply")
                + iter_timings.seconds_for("compressible:implicit_restore_alpha");
        }

        let stage_start = std::time::Instant::now();
        solver
            .primitive_update_graph
            .execute(&solver.context, &*solver);
        update_secs += stage_start.elapsed().as_secs_f64();
        let total_secs = step_start.elapsed().as_secs_f64();
        solver.profile.record(
            total_secs,
            grad_secs,
            assembly_secs,
            fgmres_secs,
            apply_secs,
            update_secs,
            fgmres_iters,
        );
        solver.constants.dt_old = solver.constants.dt;
        solver.update_constants();
        stats
    }
}

fn compressible_bind_mesh_fields(s: &CompressiblePlanResources, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, s.kernels.bg_mesh(), &[]);
    cpass.set_bind_group(1, s.kernels.bg_fields(), &[]);
}

fn compressible_bind_mesh_fields_solver(s: &CompressiblePlanResources, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, s.kernels.bg_mesh(), &[]);
    cpass.set_bind_group(1, s.kernels.bg_fields(), &[]);
    cpass.set_bind_group(2, s.kernels.bg_solver(), &[]);
}

fn compressible_bind_fields_only(s: &CompressiblePlanResources, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, s.kernels.bg_fields(), &[]);
}

fn compressible_bind_apply_fields_solver(s: &CompressiblePlanResources, cpass: &mut wgpu::ComputePass) {
    cpass.set_bind_group(0, s.kernels.bg_apply_fields(), &[]);
    cpass.set_bind_group(1, s.kernels.bg_apply_solver(), &[]);
}

fn compressible_workgroups_cells(s: &CompressiblePlanResources) -> (u32, u32, u32) {
    let workgroup_size = 64;
    (s.num_cells.div_ceil(workgroup_size), 1, 1)
}

fn compressible_workgroups_faces(s: &CompressiblePlanResources) -> (u32, u32, u32) {
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
