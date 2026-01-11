use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::lowering::unified_registry::UnifiedOpRegistryConfig;
use crate::solver::gpu::modules::generic_coupled_kernels::{
    GenericCoupledBindGroups, GenericCoupledKernelsModule, GenericCoupledPipeline,
};
use crate::solver::gpu::modules::graph::{
    ComputeSpec, DispatchKind, ModuleGraph, ModuleNode, RuntimeDims,
};
use crate::solver::gpu::modules::generic_coupled_schur::GenericCoupledSchurPreconditioner;
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_solver::solve_fgmres;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::modules::unified_graph::{
    build_graph_for_phases, build_optional_graph_for_phase,
};
use crate::solver::gpu::plans::plan_instance::{PlanFuture, PlanLinearSystemDebug, PlanParamValue};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::recipe::{KernelPhase, LinearSolverType, SolverRecipe};
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerParams};
use crate::solver::gpu::linear_solver::fgmres::{FgmresPrecondBindings, FgmresWorkspace};
use crate::solver::model::{ModelPreconditionerSpec, ModelSpec};
use bytemuck::bytes_of;

pub(crate) struct GenericCoupledProgramResources {
    runtime: GpuCsrRuntime,
    fields: UnifiedFieldResources,
    kernels: GenericCoupledKernelsModule,
    assembly_graph: ModuleGraph<GenericCoupledKernelsModule>,
    apply_graph: ModuleGraph<GenericCoupledKernelsModule>,
    update_graph: ModuleGraph<GenericCoupledKernelsModule>,
    outer_iters: usize,
    linear_solver: crate::solver::gpu::recipe::LinearSolverSpec,
    schur: Option<GenericCoupledSchurResources>,
    _b_scalar_row_offsets: wgpu::Buffer,
    _b_scalar_col_indices: wgpu::Buffer,
    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
}

struct GenericCoupledSchurResources {
    solver: KrylovSolveModule<GenericCoupledSchurPreconditioner>,
    dispatch: KrylovDispatch,
    _b_diag_u: wgpu::Buffer,
    _b_diag_v: wgpu::Buffer,
    _b_diag_p: wgpu::Buffer,
    _b_precond_params: wgpu::Buffer,
    _b_p_matrix_values: wgpu::Buffer,
}

impl GenericCoupledProgramResources {
    pub(crate) fn new(
        runtime: GpuCsrRuntime,
        fields: UnifiedFieldResources,
        kernels: GenericCoupledKernelsModule,
        model: &ModelSpec,
        recipe: &SolverRecipe,
        b_scalar_row_offsets: wgpu::Buffer,
        b_scalar_col_indices: wgpu::Buffer,
        b_bc_kind: wgpu::Buffer,
        b_bc_value: wgpu::Buffer,
    ) -> Result<Self, String> {
        // Build graphs from recipe using unified graph builder
        let assembly_graph = build_graph_for_phases(
            recipe,
            &[KernelPhase::FluxComputation, KernelPhase::Assembly],
            &kernels,
            "generic_coupled",
        )
        .unwrap_or_else(|_| build_assembly_graph_fallback());

        // Apply and update are optional depending on the stepping mode.
        // (For implicit outer-iteration recipes, update may be executed in the "apply" stage.)
        let apply_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Apply,
            &kernels,
            "generic_coupled",
        )
        .unwrap_or(None)
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        let update_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Update,
            &kernels,
            "generic_coupled",
        )
        .unwrap_or(None)
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        let outer_iters = match recipe.stepping {
            crate::solver::gpu::recipe::SteppingMode::Implicit { outer_iters } => outer_iters,
            _ => 1,
        };

        let linear_solver = recipe.linear_solver.clone();
        let schur = build_generic_schur(
            model,
            recipe,
            &runtime,
            &b_scalar_row_offsets,
            &b_scalar_col_indices,
        )?;

        Ok(Self {
            runtime,
            fields,
            kernels,
            assembly_graph,
            apply_graph,
            update_graph,
            outer_iters,
            linear_solver,
            schur,
            _b_scalar_row_offsets: b_scalar_row_offsets,
            _b_scalar_col_indices: b_scalar_col_indices,
            _b_bc_kind: b_bc_kind,
            _b_bc_value: b_bc_value,
        })
    }
}

impl GenericCoupledProgramResources {
    fn runtime_dims(&self) -> RuntimeDims {
        RuntimeDims {
            num_cells: self.runtime.common.num_cells,
            num_faces: self.runtime.common.num_faces,
        }
    }
}

fn validate_schur_model(
    model: &ModelSpec,
) -> Result<(f32, crate::solver::model::SchurBlockLayout), String> {
    let Some(solver) = model.linear_solver else {
        return Err("model does not define a linear solver spec".into());
    };
    let ModelPreconditionerSpec::Schur { omega, layout } = solver.preconditioner else {
        return Err("model does not request Schur preconditioning".into());
    };

    if model.method != crate::solver::model::method::MethodSpec::GenericCoupled {
        return Err(
            "Schur preconditioner is only wired for the GenericCoupled pipeline".to_string(),
        );
    }
    if model.system.unknowns_per_cell() < 3 {
        return Err(format!(
            "Schur preconditioner requires unknowns_per_cell >= 3 (got {})",
            model.system.unknowns_per_cell()
        ));
    }

    layout.validate(model.system.unknowns_per_cell())?;

    // Validate the layout against the equation targets used to assemble the system.
    //
    // This prevents a model from declaring indices that don't correspond to the actual
    // vector+scalar targets solved by the linear system, without relying on equation ordering.
    let u_set = {
        let mut vals = [layout.u[0], layout.u[1]];
        vals.sort_unstable();
        vals
    };
    let mut has_u = false;
    let mut has_p = false;
    for eq in model.system.equations() {
        let target = eq.target();
        match target.kind() {
            crate::solver::model::backend::ast::FieldKind::Vector2 => {
                let u0 = model
                    .state_layout
                    .component_offset(target.name(), 0)
                    .ok_or_else(|| {
                        format!("missing '{}' component 0 in state layout", target.name())
                    })?;
                let u1 = model
                    .state_layout
                    .component_offset(target.name(), 1)
                    .ok_or_else(|| {
                        format!("missing '{}' component 1 in state layout", target.name())
                    })?;
                let mut offsets = [u0, u1];
                offsets.sort_unstable();
                if offsets == u_set {
                    has_u = true;
                }
            }
            crate::solver::model::backend::ast::FieldKind::Scalar => {
                let p_offset = model
                    .state_layout
                    .offset_for(target.name())
                    .ok_or_else(|| format!("missing '{}' in state layout", target.name()))?;
                if p_offset == layout.p {
                    has_p = true;
                }
            }
        }
    }
    if !has_u {
        return Err(format!(
            "SchurBlockLayout {:?} does not match any Vector2 equation target in the model",
            layout
        ));
    }
    if !has_p {
        return Err(format!(
            "SchurBlockLayout {:?} pressure index does not match any Scalar equation target in the model",
            layout
        ));
    }

    Ok((omega, layout))
}

fn build_generic_schur(
    model: &ModelSpec,
    recipe: &SolverRecipe,
    runtime: &GpuCsrRuntime,
    scalar_row_offsets: &wgpu::Buffer,
    scalar_col_indices: &wgpu::Buffer,
) -> Result<Option<GenericCoupledSchurResources>, String> {
    let Some(spec) = model.linear_solver else {
        return Ok(None);
    };
    match spec.preconditioner {
        ModelPreconditionerSpec::Default => return Ok(None),
        ModelPreconditionerSpec::Schur { .. } => {}
    }

    let (omega, layout) = validate_schur_model(model)?;

    let LinearSolverType::Fgmres { max_restart } = recipe.linear_solver.solver_type else {
        return Err(
            "Schur preconditioner requires LinearSolverType::Fgmres in the recipe".to_string(),
        );
    };

    let device = &runtime.common.context.device;
    let num_cells = runtime.common.num_cells;
    let num_dofs = runtime.num_dofs;
    let scalar_nnz = runtime.common.mesh.col_indices.len() as u64;

    let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur diag_u_inv"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_diag_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur diag_v_inv"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_diag_p = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur diag_p_inv"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_precond_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur precond_params"),
        size: std::mem::size_of::<PreconditionerParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = PreconditionerParams {
        n: num_dofs,
        num_cells,
        omega,
        unknowns_per_cell: model.system.unknowns_per_cell(),
        u0: layout.u[0],
        u1: layout.u[1],
        p: layout.p,
        _pad0: 0,
    };
    runtime
        .common
        .context
        .queue
        .write_buffer(&b_precond_params, 0, bytes_of(&params));

    let b_p_matrix_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur p_matrix_values"),
        size: scalar_nnz * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_setup_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur setup_params"),
        size: (std::mem::size_of::<u32>() * 8) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let setup_pipeline = GenericCoupledSchurPreconditioner::build_setup_pipeline(device);
    let matrix_values = runtime
        .linear_port_space
        .buffer(runtime.linear_ports.values);
    let diagonal_indices = runtime
        .common
        .mesh
        .buffer_for_binding_name("diagonal_indices")
        .ok_or_else(|| "missing diagonal_indices mesh buffer".to_string())?;
    let setup_bg = GenericCoupledSchurPreconditioner::build_setup_bind_group(
        device,
        &setup_pipeline,
        scalar_row_offsets,
        diagonal_indices,
        matrix_values,
        &b_diag_u,
        &b_diag_v,
        &b_diag_p,
        &b_p_matrix_values,
        &b_setup_params,
    );

    let system = LinearSystemView {
        ports: runtime.linear_ports,
        space: &runtime.linear_port_space,
    };

    let precond_bindings = FgmresPrecondBindings::DiagWithParams {
        diag_u: &b_diag_u,
        diag_v: &b_diag_v,
        diag_p: &b_diag_p,
        precond_params: &b_precond_params,
    };
    let fgmres = FgmresWorkspace::new_from_system(
        device,
        num_dofs,
        num_cells,
        max_restart,
        system,
        precond_bindings,
        "generic_coupled",
    );

    let precond = GenericCoupledSchurPreconditioner::new(
        device,
        &fgmres,
        num_cells,
        scalar_row_offsets,
        scalar_col_indices,
        &b_p_matrix_values,
        setup_bg,
        setup_pipeline,
        b_setup_params,
        model.system.unknowns_per_cell(),
        layout.u[0],
        layout.u[1],
        layout.p,
    );

    let dispatch = DispatchGrids::for_sizes(num_dofs, num_cells);

    Ok(Some(GenericCoupledSchurResources {
        solver: KrylovSolveModule::new(fgmres, precond),
        dispatch,
        _b_diag_u: b_diag_u,
        _b_diag_v: b_diag_v,
        _b_diag_p: b_diag_p,
        _b_precond_params: b_precond_params,
        _b_p_matrix_values: b_p_matrix_values,
    }))
}

impl PlanLinearSystemDebug for GenericCoupledProgramResources {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        self.runtime.set_linear_system(matrix_values, rhs)
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        if n != self.runtime.num_dofs {
            return Err(format!(
                "requested solve size {} does not match num_dofs {}",
                n, self.runtime.num_dofs
            ));
        }

        if let Some(schur) = &mut self.schur {
            let system = LinearSystemView {
                ports: self.runtime.linear_ports,
                space: &self.runtime.linear_port_space,
            };

            // Map the debug max_iters into a maximum restart size.
            let max_restart = (max_iters as usize)
                .max(1)
                .min(schur.solver.fgmres.max_restart());

            Ok(solve_fgmres(
                &self.runtime.common.context,
                &mut schur.solver,
                system,
                n,
                self.runtime.common.num_cells,
                schur.dispatch,
                max_restart,
                tol,
                tol * 1e-4,
                "GenericCoupled Schur (debug)",
            ))
        } else {
            Ok(self.runtime.solve_linear_system_cg(max_iters, tol))
        }
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            let raw = self
                .runtime
                .common
                .read_buffer(
                    self.runtime
                        .linear_port_space
                        .buffer(self.runtime.linear_ports.x),
                    (self.runtime.num_dofs as u64) * 4,
                    "GenericCoupled CSR Runtime Staging Buffer (cached)",
                )
                .await;
            Ok(bytemuck::cast_slice(&raw).to_vec())
        })
    }
}

fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    plan.resources
        .get::<GenericCoupledProgramResources>()
        .expect("missing GenericCoupledProgramResources")
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GenericCoupledProgramResources {
    plan.resources
        .get_mut::<GenericCoupledProgramResources>()
        .expect("missing GenericCoupledProgramResources")
}

/// Register ops using the unified registry builder.
/// The recipe's stepping mode determines which ops are registered.
pub(crate) fn register_ops_from_recipe(
    recipe: &SolverRecipe,
    registry: &mut ProgramOpRegistry,
) -> Result<(), String> {
    let config = UnifiedOpRegistryConfig {
        prepare: Some(host_prepare_step),
        finalize: Some(host_finalize_step),
        solve: Some(host_solve_linear_system),
        assembly_graph: Some(assembly_graph_run),
        apply_graph: Some(apply_graph_run),
        update_graph: Some(update_graph_run),
        implicit_update_graph: Some(update_graph_run),
        implicit_outer_iters: Some(count_outer_iters),
        ..Default::default()
    };

    let built =
        crate::solver::gpu::lowering::unified_registry::build_unified_registry(recipe, config)?;

    // Merge built registry into provided registry
    registry.merge(built)?;

    Ok(())
}

pub(crate) fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    res(plan).runtime.common.num_cells
}

pub(crate) fn spec_time(plan: &GpuProgramPlan) -> f32 {
    res(plan).runtime.time_integration.time as f32
}

pub(crate) fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).runtime.time_integration.dt
}

pub(crate) fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    res(plan).fields.current_state()
}

pub(crate) fn spec_write_state_bytes(plan: &GpuProgramPlan, bytes: &[u8]) -> Result<(), String> {
    res(plan)
        .fields
        .write_state_bytes(&plan.context.queue, bytes);
    Ok(())
}

pub(crate) fn host_prepare_step(plan: &mut GpuProgramPlan) {
    let r = res_mut(plan);
    r.fields.advance_step();
    r.runtime.advance_time();
}

pub(crate) fn host_finalize_step(plan: &mut GpuProgramPlan) {
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    r.runtime
        .time_integration
        .finalize_step(&mut r.runtime.constants, &queue);
}

pub(crate) fn host_solve_linear_system(plan: &mut GpuProgramPlan) {
    let context = crate::solver::gpu::context::GpuContext {
        device: plan.context.device.clone(),
        queue: plan.context.queue.clone(),
    };

    let r = res_mut(plan);

    if let Some(schur) = &mut r.schur {
        let system = LinearSystemView {
            ports: r.runtime.linear_ports,
            space: &r.runtime.linear_port_space,
        };

        let max_restart = match r.linear_solver.solver_type {
            LinearSolverType::Fgmres { max_restart } => max_restart,
            _ => 30,
        };

        let stats = solve_fgmres(
            &context,
            &mut schur.solver,
            system,
            r.runtime.num_dofs,
            r.runtime.common.num_cells,
            schur.dispatch,
            max_restart,
            r.linear_solver.tolerance,
            r.linear_solver.tolerance_abs,
            "generic_coupled:schur",
        );
        plan.last_linear_stats = stats;
        return;
    }

    plan.last_linear_stats = r
        .runtime
        .solve_linear_system_cg(r.linear_solver.max_iters, r.linear_solver.tolerance);
}

pub(crate) fn assembly_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(
        &r.assembly_graph,
        context,
        &r.kernels,
        r.runtime_dims(),
        mode,
    )
}

pub(crate) fn update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.update_graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn apply_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.apply_graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn count_outer_iters(plan: &GpuProgramPlan) -> usize {
    res(plan).outer_iters.max(1)
}

pub(crate) fn param_outer_iters(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Usize(iters) = value else {
        return Err("OuterIters expects Usize".to_string());
    };
    res_mut(plan).outer_iters = iters.max(1);
    Ok(())
}

pub(crate) fn param_dt(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(dt) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_dt(dt);
    Ok(())
}

pub(crate) fn param_advection_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Scheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_scheme(scheme.gpu_id());
    Ok(())
}

pub(crate) fn param_time_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::TimeScheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_time_scheme(scheme as u32);
    Ok(())
}

pub(crate) fn param_preconditioner(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    // Generic-coupled currently supports model-owned preconditioners (e.g. Schur).
    // Do not allow config/runtime to override that choice.
    let PlanParamValue::Preconditioner(_preconditioner) = value else {
        return Err("invalid value type".into());
    };
    if let Some(solver) = plan.model.linear_solver {
        if matches!(
            solver.preconditioner,
            crate::solver::model::ModelPreconditionerSpec::Schur { .. }
        ) {
            return Err("preconditioner is model-owned for this model".to_string());
        }
    }
    Ok(())
}

pub(crate) fn param_detailed_profiling(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Bool(enable) = value else {
        return Err("invalid value type".into());
    };
    if enable {
        plan.profiling_stats.enable();
    } else {
        plan.profiling_stats.disable();
    }
    Ok(())
}

pub(crate) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(res_mut(plan) as &mut dyn PlanLinearSystemDebug)
}

/// Fallback when recipe doesn't define assembly phase
fn build_assembly_graph_fallback() -> ModuleGraph<GenericCoupledKernelsModule> {
    ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
        label: "generic_coupled:assembly",
        pipeline: GenericCoupledPipeline::Assembly,
        bind: GenericCoupledBindGroups::Assembly,
        dispatch: DispatchKind::Cells,
    })])
}

/// Fallback when recipe doesn't define update phase
fn build_update_graph_fallback() -> ModuleGraph<GenericCoupledKernelsModule> {
    ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
        label: "generic_coupled:update",
        pipeline: GenericCoupledPipeline::Update,
        bind: GenericCoupledBindGroups::Update,
        dispatch: DispatchKind::Cells,
    })])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::{incompressible_momentum_generic_model, SchurBlockLayout};

    #[test]
    fn schur_rejects_invalid_layout_indices() {
        let mut model = incompressible_momentum_generic_model();
        let Some(spec) = &mut model.linear_solver else {
            panic!("missing linear_solver spec");
        };
        let ModelPreconditionerSpec::Schur { layout, .. } = &mut spec.preconditioner else {
            panic!("expected Schur preconditioner");
        };
        // Distinct/in-range, but doesn't correspond to any Vector2 field in the state layout.
        *layout = SchurBlockLayout { u: [0, 2], p: 1 };

        let err = validate_schur_model(&model).unwrap_err();
        assert!(err.contains("Vector2"), "unexpected error: {err}");
    }
}
