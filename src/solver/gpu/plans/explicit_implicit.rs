#![allow(dead_code)]

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::modules::compressible_lowering::CompressibleLowered;
use crate::solver::gpu::modules::graph::RuntimeDims;
use crate::solver::gpu::modules::linear_system::LinearSystemPorts;

use crate::solver::gpu::explicit_implicit::fgmres::{CompressibleLinearSolver, LinearTopology};
use crate::solver::gpu::explicit_implicit::graphs::CompressibleGraphs;
use crate::solver::gpu::modules::model_kernels::{ModelKernelsInit, ModelKernelsModule};
use crate::solver::gpu::modules::ports::{BufU32, Port, PortSpace};
use crate::solver::gpu::modules::time_integration::TimeIntegrationModule;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::plans::plan_instance::{PlanFuture, PlanLinearSystemDebug};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::runtime_common::GpuRuntimeCommon;
use crate::solver::gpu::structs::{GpuConstants, LinearSolverStats, PreconditionerType};
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::{expand_schemes, SchemeRegistry};
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use bytemuck::cast_slice;

use crate::solver::gpu::env_utils::{env_flag, env_usize};

#[derive(Clone, Copy, Debug)]
struct ExplicitImplicitOffsets {
    stride: u32,
    rho: u32,
    rho_u: u32,
    rho_e: u32,
    p: u32,
    u: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ExplicitImplicitProfile {
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

impl ExplicitImplicitProfile {
    fn new() -> Self {
        let enabled = env_flag("CFD2_COMP_PROFILE", false);
        let stride = env_usize("CFD2_COMP_PROFILE_STRIDE", 25).max(1);
        ExplicitImplicitProfile {
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

pub(crate) struct ExplicitImplicitPlanResources {
    pub common: GpuRuntimeCommon,
    pub num_cells: u32,
    pub num_faces: u32,
    pub model: ModelSpec,
    pub num_unknowns: u32,
    pub fields: UnifiedFieldResources,
    pub preconditioner: PreconditionerType,
    pub system_ports: LinearSystemPorts,
    pub scalar_row_offsets_port: Port<BufU32>,
    pub port_space: PortSpace,
    pub kernels: ModelKernelsModule,
    pub linear_solver: CompressibleLinearSolver,
    pub graphs: CompressibleGraphs,
    pub outer_iters: usize,
    pub nonconverged_relax: f32,
    pub(crate) scalar_row_offsets: Vec<u32>,
    pub(crate) scalar_col_indices: Vec<u32>,
    pub(crate) block_row_offsets: Vec<u32>,
    pub(crate) block_col_indices: Vec<u32>,
    implicit_base_alpha_u: f32,
    profile: ExplicitImplicitProfile,
    offsets: ExplicitImplicitOffsets,
    pub(crate) needs_gradients: bool,
    pub(crate) time_integration: TimeIntegrationModule,
}

impl ExplicitImplicitPlanResources {
    fn context_ref(&self) -> &GpuContext {
        &self.common.context
    }

    fn state_size_bytes(&self) -> u64 {
        self.num_cells as u64 * self.offsets.stride as u64 * 4
    }

    pub(crate) fn pre_step_copy(&self) {
        let size = self.state_size_bytes();
        let mut encoder =
            self.common
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("compressible:pre_step_copy"),
                });
        encoder.copy_buffer_to_buffer(
            self.fields.state.state_old(),
            0,
            self.fields.state.state(),
            0,
            size,
        );
        self.fields.snapshot_for_iteration(&mut encoder);
        self.common.context.queue.submit(Some(encoder.finish()));
    }

    fn update_needs_gradients(&mut self) {
        let scheme =
            Scheme::from_gpu_id(self.fields.constants.values().scheme).unwrap_or(Scheme::Upwind);
        let registry = SchemeRegistry::new(scheme);
        self.needs_gradients = expand_schemes(&self.model.system, &registry)
            .map(|expansion| expansion.needs_gradients())
            .unwrap_or(true);
    }

    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        recipe: SolverRecipe,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        let common = GpuRuntimeCommon::new(mesh, device, queue).await;
        let profile = ExplicitImplicitProfile::new();

        // Use recipe to determine if gradients are needed (computed at recipe creation time)
        let needs_gradients = recipe.needs_gradients();

        // Use recipe stepping mode for outer iterations
        let outer_iters = match recipe.stepping {
            crate::solver::gpu::recipe::SteppingMode::Implicit { outer_iters } => outer_iters,
            _ => 1,
        };

        let unknowns_per_cell = model.system.unknowns_per_cell();
        let layout = &model.state_layout;
        let offsets = ExplicitImplicitOffsets {
            stride: layout.stride(),
            rho: layout
                .offset_for("rho")
                .ok_or_else(|| "compressible model missing state field 'rho'".to_string())?,
            rho_u: layout
                .offset_for("rho_u")
                .ok_or_else(|| "compressible model missing state field 'rho_u'".to_string())?,
            rho_e: layout
                .offset_for("rho_e")
                .ok_or_else(|| "compressible model missing state field 'rho_e'".to_string())?,
            p: layout
                .offset_for("p")
                .ok_or_else(|| "compressible model missing state field 'p'".to_string())?,
            u: layout
                .offset_for("u")
                .ok_or_else(|| "compressible model missing state field 'u'".to_string())?,
        };

        let lowered = CompressibleLowered::lower(
            &common.context.device,
            &common.mesh,
            common.num_cells,
            common.num_faces,
            &model.state_layout,
            unknowns_per_cell,
        );
        let num_cells = common.num_cells;
        let num_faces = common.num_faces;
        let num_unknowns = num_cells * unknowns_per_cell;

        let initial_constants = GpuConstants {
            dt: 0.0001,
            dt_old: 0.0001,
            dtau: 0.0,
            time: 0.0,
            viscosity: 0.0,
            density: 1.0,
            component: 0,
            alpha_p: 1.0,
            scheme: 0,
            alpha_u: 1.0,
            stride_x: 65535 * 64,
            time_scheme: 0,
            inlet_velocity: 0.0,
            ramp_time: 0.0,
        };

        // Allocate all required fields directly from the recipe.
        let fields_res = UnifiedFieldResources::from_recipe(
            &common.context.device,
            &recipe,
            num_cells,
            num_faces,
            offsets.stride,
            initial_constants,
        );

        let kernels = ModelKernelsModule::new_from_recipe(
            &common.context.device,
            &common.mesh,
            model.id,
            &recipe,
            &fields_res,
            fields_res.step_handle(),
            ModelKernelsInit::linear_system(
                &lowered.ports,
                lowered.system_ports,
                lowered.scalar_row_offsets_port,
            ),
        );

        let port_space = lowered.ports;

        // Create graphs before struct construction (needs borrow of kernels)
        let graphs = CompressibleGraphs::from_recipe(&recipe, &kernels)?;

        let solver = Self {
            common,
            num_cells,
            num_faces,
            model,
            num_unknowns,
            fields: fields_res,
            preconditioner: PreconditionerType::Jacobi,
            system_ports: lowered.system_ports,
            scalar_row_offsets_port: lowered.scalar_row_offsets_port,
            port_space,
            kernels,
            linear_solver: CompressibleLinearSolver::new(),
            graphs,
            outer_iters,
            nonconverged_relax: 0.5,
            scalar_row_offsets: lowered.scalar_row_offsets,
            scalar_col_indices: lowered.scalar_col_indices,
            block_row_offsets: lowered.block_row_offsets,
            block_col_indices: lowered.block_col_indices,
            implicit_base_alpha_u: 0.0,
            profile,
            offsets,
            needs_gradients,
            time_integration: TimeIntegrationModule::new(),
        };
        // No longer need to call update_needs_gradients - recipe provides this
        Ok(solver)
    }

    pub fn initialize_history(&self) {
        let mut encoder =
            self.common
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Initialize History Encoder"),
                });
        let state_size = self.num_cells as u64 * self.offsets.stride as u64 * 4;
        encoder.copy_buffer_to_buffer(
            self.fields.state.state(),
            0,
            self.fields.state.state_old(),
            0,
            state_size,
        );
        encoder.copy_buffer_to_buffer(
            self.fields.state.state(),
            0,
            self.fields.state.state_old_old(),
            0,
            state_size,
        );
        self.common.context.queue.submit(Some(encoder.finish()));
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.time_integration
            .set_dt(dt, &mut self.fields.constants, &self.common.context.queue);
    }

    pub fn set_dtau(&mut self, dtau: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.dtau = dtau;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.viscosity = mu;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        {
            let values = self.fields.constants.values_mut();
            values.time_scheme = scheme;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.inlet_velocity = velocity;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        {
            let values = self.fields.constants.values_mut();
            values.scheme = scheme;
        }
        self.fields.constants.write(&self.common.context.queue);
        self.update_needs_gradients();
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.alpha_u = alpha_u;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_precond_type(&mut self, precond_type: PreconditionerType) {
        self.preconditioner = precond_type;
    }

    pub fn set_precond_model(&mut self, model: u32) {
        self.fields.low_mach_params.model = model;
        self.update_low_mach_params();
    }

    pub fn set_precond_theta_floor(&mut self, floor: f32) {
        self.fields.low_mach_params.theta_floor = floor;
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

    pub(crate) fn should_use_explicit(&self) -> bool {
        let constants = self.fields.constants.values();
        constants.time_scheme == 0 && constants.dtau <= 0.0
    }

    pub(crate) fn runtime_dims(&self) -> RuntimeDims {
        RuntimeDims {
            num_cells: self.num_cells,
            num_faces: self.num_faces,
        }
    }

    pub(crate) fn advance_ping_pong_and_time(&mut self) {
        self.fields.state.advance();
        self.time_integration
            .prepare_step(&mut self.fields.constants, &self.common.context.queue);
    }

    pub(crate) fn implicit_set_base_alpha(&mut self) {
        self.implicit_base_alpha_u = self.fields.constants.values().alpha_u;
    }

    pub(crate) fn implicit_set_iteration_params(
        &mut self,
        tol: f32,
        retry_tol: f32,
        max_restart: usize,
        retry_restart: usize,
    ) {
        self.linear_solver.tol = tol;
        self.linear_solver.retry_tol = retry_tol;
        self.linear_solver.max_restart = max_restart.max(1);
        self.linear_solver.retry_restart = retry_restart.max(1);
    }

    pub(crate) fn implicit_solve_fgmres(&mut self) {
        let topology = LinearTopology {
            num_cells: self.num_cells,
            scalar_row_offsets: &self.scalar_row_offsets,
            scalar_col_indices: &self.scalar_col_indices,
            block_row_offsets: &self.block_row_offsets,
            block_col_indices: &self.block_col_indices,
        };

        let _iter_stats = self.linear_solver.solve(
            &self.common.context,
            &self.common.readback_cache,
            &self.common.profiling_stats,
            self.num_unknowns,
            self.num_cells,
            self.system_ports,
            &self.port_space,
            &topology,
            self.preconditioner,
            self.linear_solver.max_restart,
            self.linear_solver.tol,
        );

        if !self.linear_solver.last_stats.converged {
            let _retry_stats = self.linear_solver.solve(
                &self.common.context,
                &self.common.readback_cache,
                &self.common.profiling_stats,
                self.num_unknowns,
                self.num_cells,
                self.system_ports,
                &self.port_space,
                &topology,
                self.preconditioner,
                self.linear_solver.retry_restart,
                self.linear_solver.retry_tol,
            );
        }
    }

    pub(crate) fn implicit_last_stats(&self) -> LinearSolverStats {
        self.linear_solver.last_stats
    }

    pub(crate) fn implicit_snapshot(&self) {
        let mut encoder =
            self.common
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("compressible:implicit_snapshot"),
                });
        self.fields.snapshot_for_iteration(&mut encoder);
        self.common.context.queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn implicit_set_alpha_for_apply(&mut self) {
        let apply_alpha = if self.linear_solver.last_stats.converged {
            self.implicit_base_alpha_u
        } else {
            self.implicit_base_alpha_u * self.nonconverged_relax
        };
        if (self.fields.constants.values().alpha_u - apply_alpha).abs() > 1e-6 {
            {
                let values = self.fields.constants.values_mut();
                values.alpha_u = apply_alpha;
            }
            self.fields.constants.write(&self.common.context.queue);
        }
    }

    pub(crate) fn implicit_restore_alpha(&mut self) {
        if (self.fields.constants.values().alpha_u - self.implicit_base_alpha_u).abs() > 1e-6 {
            {
                let values = self.fields.constants.values_mut();
                values.alpha_u = self.implicit_base_alpha_u;
            }
            self.fields.constants.write(&self.common.context.queue);
        }
    }

    pub(crate) fn finalize_dt_old(&mut self) {
        self.time_integration
            .finalize_step(&mut self.fields.constants, &self.common.context.queue);
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

    fn update_low_mach_params(&mut self) {
        self.fields
            .update_low_mach_params(&self.common.context.queue);
    }

    async fn read_state(&self) -> Vec<f32> {
        let byte_count = self.num_cells as u64 * self.offsets.stride as u64 * 4;
        let raw = self
            .read_buffer(self.fields.state.state(), byte_count)
            .await;
        cast_slice(&raw).to_vec()
    }

    pub(crate) async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        self.common
            .read_buffer(buffer, size, "Compressible Staging Buffer (cached)")
            .await
    }

    pub(crate) async fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let raw = self.read_buffer(buffer, count as u64 * 4).await;
        cast_slice(&raw).to_vec()
    }

    fn write_state_all(&self, state: &[f32]) {
        self.fields
            .state
            .write_all(&self.common.context.queue, cast_slice(state));
    }

    pub(crate) fn write_state_bytes(&self, bytes: &[u8]) {
        self.fields
            .state
            .write_all(&self.common.context.queue, bytes);
    }

    fn solve_fgmres_helper(&mut self, max_restart: usize, tol: f32) -> LinearSolverStats {
        let topology = LinearTopology {
            num_cells: self.num_cells,
            scalar_row_offsets: &self.scalar_row_offsets,
            scalar_col_indices: &self.scalar_col_indices,
            block_row_offsets: &self.block_row_offsets,
            block_col_indices: &self.block_col_indices,
        };
        self.linear_solver.solve(
            &self.common.context,
            &self.common.readback_cache,
            &self.common.profiling_stats,
            self.num_unknowns,
            self.num_cells,
            self.system_ports,
            &self.port_space,
            &topology,
            self.preconditioner,
            max_restart,
            tol,
        )
    }
}

impl PlanLinearSystemDebug for ExplicitImplicitPlanResources {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        if matrix_values.len() != self.block_col_indices.len() {
            return Err(format!(
                "matrix_values length {} does not match num_nonzeros {}",
                matrix_values.len(),
                self.block_col_indices.len()
            ));
        }
        if rhs.len() != self.num_unknowns as usize {
            return Err(format!(
                "rhs length {} does not match num_unknowns {}",
                rhs.len(),
                self.num_unknowns
            ));
        }
        self.common.context.queue.write_buffer(
            self.port_space.buffer(self.system_ports.values),
            0,
            bytemuck::cast_slice(matrix_values),
        );
        self.common.context.queue.write_buffer(
            self.port_space.buffer(self.system_ports.rhs),
            0,
            bytemuck::cast_slice(rhs),
        );
        Ok(())
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        if n != self.num_unknowns {
            return Err(format!(
                "requested solve size {} does not match num_unknowns {}",
                n, self.num_unknowns
            ));
        }
        let max_restart = max_iters.min(64) as usize;
        Ok(self.solve_fgmres_helper(max_restart, tol))
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            let raw = self
                .read_buffer(
                    self.port_space.buffer(self.system_ports.x),
                    (self.num_unknowns as u64) * 4,
                )
                .await;
            Ok(bytemuck::cast_slice(&raw).to_vec())
        })
    }
}
