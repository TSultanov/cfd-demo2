use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::lowering::unified_registry::UnifiedOpRegistryConfig;
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::graph::{
    ComputeSpec, DispatchKind, ModuleGraph, ModuleNode, RuntimeDims,
};
use crate::solver::gpu::modules::generic_coupled_schur::GenericCoupledSchurPreconditioner;
use crate::solver::gpu::modules::generic_linear_solver::IdentityPreconditioner;
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_solver::solve_fgmres;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::modules::time_integration::TimeIntegrationModule;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::modules::unified_graph::{
    build_graph_for_phases, build_optional_graph_for_phase, build_optional_graph_for_phases,
};
use crate::solver::gpu::lowering::models::universal::UniversalProgramResources;
use crate::solver::gpu::plans::plan_instance::{PlanFuture, PlanLinearSystemDebug, PlanParamValue};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::recipe::{KernelPhase, LinearSolverType, SolverRecipe};
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::gpu::linear_solver::fgmres::{FgmresPrecondBindings, FgmresWorkspace};
use crate::solver::model::{KernelId, ModelPreconditionerSpec, ModelSpec};
use bytemuck::bytes_of;

pub(crate) struct GenericCoupledProgramResources {
    runtime: GpuCsrRuntime,
    fields: UnifiedFieldResources,
    time_integration: TimeIntegrationModule,
    kernels: GeneratedKernelsModule,
    assembly_graph: ModuleGraph<GeneratedKernelsModule>,
    apply_graph: ModuleGraph<GeneratedKernelsModule>,
    update_graph: ModuleGraph<GeneratedKernelsModule>,
    explicit_graph: ModuleGraph<GeneratedKernelsModule>,
    outer_iters: usize,
    linear_solver: crate::solver::gpu::recipe::LinearSolverSpec,
    schur: Option<GenericCoupledSchurResources>,
    krylov: Option<GenericCoupledKrylovResources>,
    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
}

struct GenericCoupledSchurResources {
    solver: KrylovSolveModule<GenericCoupledSchurPreconditioner>,
    dispatch: KrylovDispatch,
    _b_diag_u: wgpu::Buffer,
    _b_diag_p: wgpu::Buffer,
    _b_precond_params: wgpu::Buffer,
    _b_p_matrix_values: wgpu::Buffer,
}

struct GenericCoupledKrylovResources {
    solver: KrylovSolveModule<IdentityPreconditioner>,
    dispatch: KrylovDispatch,
    _b_diag_u: wgpu::Buffer,
    _b_diag_v: wgpu::Buffer,
    _b_diag_p: wgpu::Buffer,
}

impl GenericCoupledProgramResources {
    pub(crate) fn new(
        runtime: GpuCsrRuntime,
        fields: UnifiedFieldResources,
        kernels: GeneratedKernelsModule,
        model: &ModelSpec,
        recipe: &SolverRecipe,
        b_bc_kind: wgpu::Buffer,
        b_bc_value: wgpu::Buffer,
    ) -> Result<Self, String> {
        // Build graphs from recipe using unified graph builder.
        //
        // Some models (e.g., compressible KT flux) require a gradient stage before flux.
        // Keep gradients optional so diffusion-only models don't fail graph construction.
        let has_gradients = recipe
            .kernels
            .iter()
            .any(|k| k.phase == KernelPhase::Gradients);
        let assembly_graph_result = if has_gradients {
            build_graph_for_phases(
                recipe,
                &[
                    KernelPhase::Gradients,
                    KernelPhase::FluxComputation,
                    KernelPhase::Assembly,
                ],
                &kernels,
                "generic_coupled",
            )
        } else {
            build_graph_for_phases(
                recipe,
                &[KernelPhase::FluxComputation, KernelPhase::Assembly],
                &kernels,
                "generic_coupled",
            )
        };

        let assembly_graph = match assembly_graph_result {
            Ok(graph) => graph,
            Err(e) if e.starts_with("no kernels found for phases") => {
                build_assembly_graph_fallback()
            }
            Err(e) => return Err(e),
        };

        // Apply and update are optional depending on the stepping mode.
        // (For implicit outer-iteration recipes, update may be executed in the "apply" stage.)
        let apply_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Apply,
            &kernels,
            "generic_coupled",
        )?
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        let update_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Update,
            &kernels,
            "generic_coupled",
        )?
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        // Explicit stepping uses a single graph op; build a combined graph covering all
        // compute phases that might be present in explicit recipes.
        let explicit_graph = build_optional_graph_for_phases(
            recipe,
            &[
                KernelPhase::Gradients,
                KernelPhase::FluxComputation,
                KernelPhase::Assembly,
                KernelPhase::Apply,
                KernelPhase::Update,
            ],
            &kernels,
            "generic_coupled",
        )?
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        let outer_iters = match recipe.stepping {
            crate::solver::gpu::recipe::SteppingMode::Implicit { outer_iters } => outer_iters,
            _ => 1,
        };

        let linear_solver = recipe.linear_solver.clone();
        let scalar_row_offsets = &runtime.common.mesh.b_scalar_row_offsets;
        let scalar_col_indices = &runtime.common.mesh.b_scalar_col_indices;
        let schur = build_generic_schur(
            model,
            recipe,
            &runtime,
            scalar_row_offsets,
            scalar_col_indices,
        )?;
        let krylov = if schur.is_some() {
            None
        } else {
            build_generic_krylov(recipe, &runtime)?
        };

        Ok(Self {
            runtime,
            fields,
            time_integration: TimeIntegrationModule::new(),
            kernels,
            assembly_graph,
            apply_graph,
            update_graph,
            explicit_graph,
            outer_iters,
            linear_solver,
            schur,
            krylov,
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

    if !matches!(
        model.method,
        crate::solver::model::method::MethodSpec::GenericCoupled
            | crate::solver::model::method::MethodSpec::CoupledIncompressible
    ) {
        return Err(
            "Schur preconditioner is only wired for the GenericCoupled pipelines".to_string(),
        );
    }
    layout.validate(model.system.unknowns_per_cell())?;

    // Validate the layout against the equation targets used to assemble the system.
    //
    // For the current Schur bridge, the linear system is assumed to consist only of a
    // velocity-like block and a single pressure-like scalar.
    let mut target_indices = std::collections::BTreeSet::new();
    let mut scalar_targets = std::collections::BTreeSet::new();
    for eq in model.system.equations() {
        let target = eq.target();
        match target.kind() {
            crate::solver::model::backend::ast::FieldKind::Scalar => {
                let idx = model
                    .state_layout
                    .offset_for(target.name())
                    .ok_or_else(|| format!("missing '{}' in state layout", target.name()))?;
                target_indices.insert(idx);
                scalar_targets.insert(idx);
            }
            kind => {
                for comp in 0..kind.component_count() {
                    let comp = comp as u32;
                    let idx = model
                        .state_layout
                        .component_offset(target.name(), comp)
                        .ok_or_else(|| {
                            format!("missing '{}' component {} in state layout", target.name(), comp)
                        })?;
                    target_indices.insert(idx);
                }
            }
        }
    }

    if !scalar_targets.contains(&layout.p) {
        return Err(format!(
            "SchurBlockLayout {:?} pressure index does not match any scalar equation target",
            layout
        ));
    }

    let mut layout_indices = std::collections::BTreeSet::new();
    for &u in layout.u_indices() {
        layout_indices.insert(u);
    }
    layout_indices.insert(layout.p);

    if layout_indices != target_indices {
        return Err(format!(
            "SchurBlockLayout {:?} must cover exactly the model equation targets (layout={:?}, targets={:?})",
            layout,
            layout_indices,
            target_indices
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
    let scalar_nnz = runtime.common.mesh.scalar_col_indices.len() as u64;

    let u_len = layout.u_len;
    let mut u0123 = [0u32; 4];
    let mut u4567 = [0u32; 4];
    for (i, &u) in layout.u_indices().iter().enumerate() {
        if i < 4 {
            u0123[i] = u;
        } else {
            u4567[i - 4] = u;
        }
    }

    let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur diag_u_inv"),
        size: (num_cells as u64) * (u_len as u64) * 4,
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
        size: std::mem::size_of::<crate::solver::gpu::bindings::schur_precond_generic::PrecondParams>()
            as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = crate::solver::gpu::bindings::schur_precond_generic::PrecondParams {
        n: num_dofs,
        num_cells,
        omega,
        unknowns_per_cell: model.system.unknowns_per_cell(),
        p: layout.p,
        u_len,
        _pad0: 0,
        _pad1: 0,
        u0123,
        u4567,
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
        size: std::mem::size_of::<crate::solver::gpu::bindings::generic_coupled_schur_setup::SetupParams>()
            as u64,
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
        &b_diag_p,
        &b_p_matrix_values,
        &b_setup_params,
    )?;

    let system = LinearSystemView {
        ports: runtime.linear_ports,
        space: &runtime.linear_port_space,
    };

    let precond_bindings = FgmresPrecondBindings::SchurWithParams {
        diag_u: &b_diag_u,
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
        layout.p,
        u_len,
        u0123,
        u4567,
    );

    let dispatch = DispatchGrids::for_sizes(num_dofs, num_cells);

    Ok(Some(GenericCoupledSchurResources {
        solver: KrylovSolveModule::new(fgmres, precond),
        dispatch,
        _b_diag_u: b_diag_u,
        _b_diag_p: b_diag_p,
        _b_precond_params: b_precond_params,
        _b_p_matrix_values: b_p_matrix_values,
    }))
}

fn build_generic_krylov(
    recipe: &SolverRecipe,
    runtime: &GpuCsrRuntime,
) -> Result<Option<GenericCoupledKrylovResources>, String> {
    let LinearSolverType::Fgmres { max_restart } = recipe.linear_solver.solver_type else {
        return Ok(None);
    };

    let device = &runtime.common.context.device;
    let num_cells = runtime.common.num_cells;
    let n = runtime.num_dofs;

    let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generic_coupled:identity_diag_u"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_diag_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generic_coupled:identity_diag_v"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_diag_p = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generic_coupled:identity_diag_p"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let system = LinearSystemView {
        ports: runtime.linear_ports,
        space: &runtime.linear_port_space,
    };

    let precond_bindings = FgmresPrecondBindings::Diag {
        diag_u: &b_diag_u,
        diag_v: &b_diag_v,
        diag_p: &b_diag_p,
    };

    let fgmres = FgmresWorkspace::new_from_system(
        device,
        n,
        num_cells,
        max_restart.max(1),
        system,
        precond_bindings,
        "generic_coupled",
    );

    let solver = KrylovSolveModule::new(fgmres, IdentityPreconditioner::new());
    let dispatch = DispatchGrids::for_sizes(n, num_cells);

    Ok(Some(GenericCoupledKrylovResources {
        solver,
        dispatch,
        _b_diag_u: b_diag_u,
        _b_diag_v: b_diag_v,
        _b_diag_p: b_diag_p,
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
        } else if let Some(krylov) = &mut self.krylov {
            let system = LinearSystemView {
                ports: self.runtime.linear_ports,
                space: &self.runtime.linear_port_space,
            };

            let max_restart = match self.linear_solver.solver_type {
                LinearSolverType::Fgmres { max_restart } => max_restart,
                _ => 30,
            };
            Ok(solve_fgmres(
                &self.runtime.common.context,
                &mut krylov.solver,
                system,
                n,
                self.runtime.common.num_cells,
                krylov.dispatch,
                max_restart.max(1),
                tol,
                tol * 1e-4,
                "GenericCoupled FGMRES (debug)",
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
        .get::<UniversalProgramResources>()
        .and_then(|u| u.generic_coupled())
        .expect("missing GenericCoupledProgramResources backend")
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GenericCoupledProgramResources {
    plan.resources
        .get_mut::<UniversalProgramResources>()
        .and_then(|u| u.generic_coupled_mut())
        .expect("missing GenericCoupledProgramResources backend")
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
    res(plan).time_integration.time as f32
}

pub(crate) fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).time_integration.dt
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
    let device = plan.context.device.clone();
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    r.fields.advance_step();
    // Seed the writable `state` buffer with the previous state so kernels that
    // read from `state` (e.g. gradient/flux stages during implicit outer
    // iterations) start from a consistent iterate.
    //
    // This mirrors the EI solver's pre-step copy and avoids reading stale data
    // from the rotated ping-pong buffer.
    let size = r.fields.state_size_bytes();
    let src = r.fields.previous_state();
    let dst = r.fields.current_state();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("generic_coupled:pre_step_copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
    queue.submit(Some(encoder.finish()));
    r.time_integration
        .prepare_step(&mut r.fields.constants, &queue);
}

pub(crate) fn host_finalize_step(plan: &mut GpuProgramPlan) {
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    r.time_integration
        .finalize_step(&mut r.fields.constants, &queue);
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

    if let Some(krylov) = &mut r.krylov {
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
            &mut krylov.solver,
            system,
            r.runtime.num_dofs,
            r.runtime.common.num_cells,
            krylov.dispatch,
            max_restart.max(1),
            r.linear_solver.tolerance,
            r.linear_solver.tolerance_abs,
            "generic_coupled:fgmres",
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

pub(crate) fn explicit_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.explicit_graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn apply_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.apply_graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn implicit_snapshot_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    _mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("generic_coupled:implicit_snapshot"),
        });
    r.fields.snapshot_for_iteration(&mut encoder);
    context.queue.submit(Some(encoder.finish()));
    (0.0, None)
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
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    r.time_integration.set_dt(dt, &mut r.fields.constants, &queue);
    Ok(())
}

pub(crate) fn param_dtau(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(dtau) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.dtau = dtau;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_advection_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Scheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.scheme = scheme.gpu_id();
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_time_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::TimeScheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.time_scheme = scheme as u32;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_viscosity(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(mu) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.viscosity = mu;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_density(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(rho) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.density = rho;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_alpha_u(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(alpha) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.alpha_u = alpha;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_alpha_p(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(alpha) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.alpha_p = alpha;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_inlet_velocity(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(velocity) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.inlet_velocity = velocity;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_ramp_time(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(time) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.ramp_time = time;
    }
    r.fields.constants.write(&queue);
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

pub(crate) fn param_low_mach_model(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::LowMachModel(model) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    let Some(_) = r.fields.low_mach_params_buffer() else {
        return Err("model does not allocate low-mach params".to_string());
    };
    r.fields.low_mach_params_mut().model = model as u32;
    r.fields.update_low_mach_params(&queue);
    Ok(())
}

pub(crate) fn param_low_mach_theta_floor(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(theta) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    let Some(_) = r.fields.low_mach_params_buffer() else {
        return Err("model does not allocate low-mach params".to_string());
    };
    r.fields.low_mach_params_mut().theta_floor = theta;
    r.fields.update_low_mach_params(&queue);
    Ok(())
}

pub(crate) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(res_mut(plan) as &mut dyn PlanLinearSystemDebug)
}

/// Fallback when recipe doesn't define assembly phase
fn build_assembly_graph_fallback() -> ModuleGraph<GeneratedKernelsModule> {
    ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
        label: "generic_coupled:assembly",
        pipeline: KernelId::GENERIC_COUPLED_ASSEMBLY,
        bind: KernelId::GENERIC_COUPLED_ASSEMBLY,
        dispatch: DispatchKind::Cells,
    })])
}

/// Fallback when recipe doesn't define update phase
fn build_update_graph_fallback() -> ModuleGraph<GeneratedKernelsModule> {
    ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
        label: "generic_coupled:update",
        pipeline: KernelId::GENERIC_COUPLED_UPDATE,
        bind: KernelId::GENERIC_COUPLED_UPDATE,
        dispatch: DispatchKind::Cells,
    })])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::{
        incompressible_momentum_generic_model, BoundarySpec, ModelGpuSpec, ModelLinearSolverSpec,
        ModelPreconditionerSpec, SchurBlockLayout,
    };
    use crate::solver::model::{eos, primitives};
    use crate::solver::model::backend::ast::{fvm, vol_scalar, vol_vector3, EquationSystem};
    use crate::solver::units::si;

    #[test]
    fn schur_rejects_invalid_layout_indices() {
        let mut model = incompressible_momentum_generic_model();
        let Some(spec) = &mut model.linear_solver else {
            panic!("missing linear_solver spec");
        };
        let ModelPreconditionerSpec::Schur { layout, .. } = &mut spec.preconditioner else {
            panic!("expected Schur preconditioner");
        };
        // Distinct/in-range, but doesn't cover the full set of equation target indices.
        *layout = SchurBlockLayout::from_u_p(&[0], 2).expect("layout build failed");

        let err = validate_schur_model(&model).unwrap_err();
        assert!(err.contains("equation targets"), "unexpected error: {err}");
    }

    #[test]
    fn schur_accepts_vector3_velocity_layout() {
        let u = vol_vector3("U", si::VELOCITY);
        let p = vol_scalar("p", si::PRESSURE);

        let mut system = EquationSystem::new();
        system.add_equation(fvm::ddt(u).eqn(u));
        system.add_equation(fvm::ddt(p).eqn(p));

        let layout = crate::solver::model::backend::StateLayout::new(vec![u, p]);
        assert_eq!(layout.stride(), 4);
        assert_eq!(system.unknowns_per_cell(), 4);

        let model = crate::solver::model::ModelSpec {
            id: "schur_vector3_test",
            method: crate::solver::model::method::MethodSpec::GenericCoupled,
            eos: eos::EosSpec::Constant,
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),

            linear_solver: Some(ModelLinearSolverSpec {
                preconditioner: ModelPreconditionerSpec::Schur {
                    omega: 1.0,
                    layout: SchurBlockLayout::from_u_p(&[0, 1, 2], 3)
                        .expect("layout build failed"),
                },
            }),
            flux_module: None,
            primitives: primitives::PrimitiveDerivations::default(),
            gpu: ModelGpuSpec::default(),
        }
        .with_derived_gpu();

        validate_schur_model(&model).expect("Vector3 velocity Schur layout should validate");
    }
}
