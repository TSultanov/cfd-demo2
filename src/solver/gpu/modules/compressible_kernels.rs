use crate::solver::gpu::bindings::compressible_explicit_update as explicit_update;
use crate::solver::gpu::bindings::generated::{
    compressible_apply as generated_apply, compressible_assembly as generated_assembly,
    compressible_flux_kt as generated_flux_kt, compressible_gradients as generated_gradients,
    compressible_update as generated_update,
};
use crate::solver::gpu::init::compressible_fields::CompressibleFieldResources;
use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::wgsl_meta;

use super::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use super::linear_system::{LinearSystemPorts, LinearSystemView};
use super::ports::{BufU32, Port, PortSpace};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum CompressiblePipeline {
    Assembly,
    Apply,
    Gradients,
    FluxKt,
    ExplicitUpdate,
    PrimitiveUpdate,
}

#[derive(Clone, Copy, Debug)]
pub enum CompressibleBindGroups {
    MeshFields,
    MeshFieldsSolver,
    FieldsOnly,
    ApplyFieldsSolver,
}

pub struct CompressibleKernelsModule {
    state_step_index: Arc<AtomicUsize>,

    bg_mesh: wgpu::BindGroup,
    bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_apply_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_solver: wgpu::BindGroup,
    bg_apply_solver: wgpu::BindGroup,

    pipeline_assembly: wgpu::ComputePipeline,
    pipeline_apply: wgpu::ComputePipeline,
    pipeline_gradients: wgpu::ComputePipeline,
    pipeline_flux: wgpu::ComputePipeline,
    pipeline_explicit_update: wgpu::ComputePipeline,
    pipeline_update: wgpu::ComputePipeline,
}

impl CompressibleKernelsModule {
    pub fn new(
        device: &wgpu::Device,
        mesh: &MeshResources,
        fields: &CompressibleFieldResources,
        state_step_index: Arc<AtomicUsize>,
        port_space: &PortSpace,
        system_ports: LinearSystemPorts,
        scalar_row_offsets: Port<BufU32>,
    ) -> Self {
        let system = LinearSystemView {
            ports: system_ports,
            space: port_space,
        };
        let b_scalar_row_offsets = port_space.buffer(scalar_row_offsets);

        let pipeline_assembly =
            generated_assembly::compute::create_main_pipeline_embed_source(device);
        let pipeline_apply = generated_apply::compute::create_main_pipeline_embed_source(device);
        let pipeline_gradients =
            generated_gradients::compute::create_main_pipeline_embed_source(device);
        let pipeline_flux = generated_flux_kt::compute::create_main_pipeline_embed_source(device);
        let pipeline_explicit_update =
            explicit_update::compute::create_main_pipeline_embed_source(device);
        let pipeline_update = generated_update::compute::create_main_pipeline_embed_source(device);

        let bg_mesh = {
            let bgl = pipeline_assembly.get_bind_group_layout(0);
            crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                device,
                "Compressible Mesh Bind Group",
                &bgl,
                wgsl_meta::COMPRESSIBLE_ASSEMBLY_BINDINGS,
                0,
                |name| {
                    mesh.buffer_for_binding_name(name)
                        .map(|buf| wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding()))
                },
            )
            .unwrap_or_else(|err| panic!("Compressible mesh bind group build failed: {err}"))
        };

        let bg_fields_ping_pong = fields.bg_fields_ping_pong.clone();

        let bg_solver = {
            let bgl = pipeline_assembly.get_bind_group_layout(2);
            crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                device,
                "Compressible Solver Bind Group",
                &bgl,
                wgsl_meta::COMPRESSIBLE_ASSEMBLY_BINDINGS,
                2,
                |name| match name {
                    "matrix_values" => Some(wgpu::BindingResource::Buffer(
                        system.values().as_entire_buffer_binding(),
                    )),
                    "rhs" => Some(wgpu::BindingResource::Buffer(
                        system.rhs().as_entire_buffer_binding(),
                    )),
                    "scalar_row_offsets" => Some(wgpu::BindingResource::Buffer(
                        b_scalar_row_offsets.as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )
            .unwrap_or_else(|err| panic!("Compressible solver bind group build failed: {err}"))
        };

        let apply_fields_layout = pipeline_apply.get_bind_group_layout(0);
        let mut bg_apply_fields_ping_pong = Vec::new();
        for i in 0..3 {
            let (idx_state, idx_old, idx_old_old) =
                crate::solver::gpu::modules::state::ping_pong_indices(i);
            let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("Compressible Apply Fields Bind Group {}", i),
                &apply_fields_layout,
                wgsl_meta::COMPRESSIBLE_APPLY_BINDINGS,
                0,
                |name| match name {
                    "state" => Some(wgpu::BindingResource::Buffer(
                        fields.state.buffers()[idx_state].as_entire_buffer_binding(),
                    )),
                    "state_old" => Some(wgpu::BindingResource::Buffer(
                        fields.state.buffers()[idx_old].as_entire_buffer_binding(),
                    )),
                    "state_old_old" => Some(wgpu::BindingResource::Buffer(
                        fields.state.buffers()[idx_old_old].as_entire_buffer_binding(),
                    )),
                    "state_iter" => Some(wgpu::BindingResource::Buffer(
                        fields.b_state_iter.as_entire_buffer_binding(),
                    )),
                    "fluxes" => Some(wgpu::BindingResource::Buffer(
                        fields.b_fluxes.as_entire_buffer_binding(),
                    )),
                    "constants" => Some(wgpu::BindingResource::Buffer(
                        fields.constants.buffer().as_entire_buffer_binding(),
                    )),
                    "grad_rho" => Some(wgpu::BindingResource::Buffer(
                        fields.b_grad_rho.as_entire_buffer_binding(),
                    )),
                    "grad_rho_u_x" => Some(wgpu::BindingResource::Buffer(
                        fields.b_grad_rho_u_x.as_entire_buffer_binding(),
                    )),
                    "grad_rho_u_y" => Some(wgpu::BindingResource::Buffer(
                        fields.b_grad_rho_u_y.as_entire_buffer_binding(),
                    )),
                    "grad_rho_e" => Some(wgpu::BindingResource::Buffer(
                        fields.b_grad_rho_e.as_entire_buffer_binding(),
                    )),
                    "low_mach" => Some(wgpu::BindingResource::Buffer(
                        fields.b_low_mach_params.as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )
            .unwrap_or_else(|err| {
                panic!("Compressible apply-fields bind group build failed: {err}")
            });
            bg_apply_fields_ping_pong.push(bg);
        }

        let apply_solver_layout = pipeline_apply.get_bind_group_layout(1);
        let bg_apply_solver = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
            device,
            "Compressible Apply Solver Bind Group",
            &apply_solver_layout,
            wgsl_meta::COMPRESSIBLE_APPLY_BINDINGS,
            1,
            |name| match name {
                "solution" => Some(wgpu::BindingResource::Buffer(
                    system.x().as_entire_buffer_binding(),
                )),
                _ => None,
            },
        )
        .unwrap_or_else(|err| panic!("Compressible apply-solver bind group build failed: {err}"));

        Self {
            state_step_index,
            bg_mesh,
            bg_fields_ping_pong,
            bg_apply_fields_ping_pong,
            bg_solver,
            bg_apply_solver,
            pipeline_assembly,
            pipeline_apply,
            pipeline_gradients,
            pipeline_flux,
            pipeline_explicit_update,
            pipeline_update,
        }
    }

    pub fn set_step_index(&mut self, idx: usize) {
        self.state_step_index.store(idx % 3, Ordering::Relaxed);
    }

    pub fn bg_mesh(&self) -> &wgpu::BindGroup {
        &self.bg_mesh
    }

    pub fn bg_fields(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self.bg_fields_ping_pong[idx]
    }

    pub fn bg_apply_fields(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self.bg_apply_fields_ping_pong[idx]
    }

    pub fn bg_solver(&self) -> &wgpu::BindGroup {
        &self.bg_solver
    }

    pub fn bg_apply_solver(&self) -> &wgpu::BindGroup {
        &self.bg_apply_solver
    }

    pub fn pipeline_assembly(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_assembly
    }

    pub fn pipeline_apply(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_apply
    }

    pub fn pipeline_gradients(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_gradients
    }

    pub fn pipeline_flux(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_flux
    }

    pub fn pipeline_explicit_update(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_explicit_update
    }

    pub fn pipeline_update(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_update
    }

    fn bind_mesh_fields(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, &self.bg_mesh, &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
    }

    fn bind_mesh_fields_solver(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, &self.bg_mesh, &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
        pass.set_bind_group(2, &self.bg_solver, &[]);
    }

    fn bind_fields_only(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_fields(), &[]);
    }

    fn bind_apply_fields_solver(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_apply_fields(), &[]);
        pass.set_bind_group(1, &self.bg_apply_solver, &[]);
    }
}

impl GpuComputeModule for CompressibleKernelsModule {
    type PipelineKey = CompressiblePipeline;
    type BindKey = CompressibleBindGroups;

    fn pipeline(&self, key: Self::PipelineKey) -> &wgpu::ComputePipeline {
        match key {
            CompressiblePipeline::Assembly => self.pipeline_assembly(),
            CompressiblePipeline::Apply => self.pipeline_apply(),
            CompressiblePipeline::Gradients => self.pipeline_gradients(),
            CompressiblePipeline::FluxKt => self.pipeline_flux(),
            CompressiblePipeline::ExplicitUpdate => self.pipeline_explicit_update(),
            CompressiblePipeline::PrimitiveUpdate => self.pipeline_update(),
        }
    }

    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass) {
        match key {
            CompressibleBindGroups::MeshFields => self.bind_mesh_fields(pass),
            CompressibleBindGroups::MeshFieldsSolver => self.bind_mesh_fields_solver(pass),
            CompressibleBindGroups::FieldsOnly => self.bind_fields_only(pass),
            CompressibleBindGroups::ApplyFieldsSolver => self.bind_apply_fields_solver(pass),
        }
    }

    fn dispatch(&self, kind: DispatchKind, runtime: RuntimeDims) -> (u32, u32, u32) {
        match kind {
            DispatchKind::Cells => ((runtime.num_cells + 63) / 64, 1, 1),
            DispatchKind::Faces => ((runtime.num_faces + 63) / 64, 1, 1),
            DispatchKind::Custom { x, y, z } => (x, y, z),
        }
    }
}
