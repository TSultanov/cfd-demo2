use crate::solver::gpu::bindings::generated::{
    coupled_assembly_merged as generated_coupled_assembly,
    prepare_coupled as generated_prepare_coupled,
    update_fields_from_coupled as generated_update_fields,
};
use crate::solver::gpu::init::fields::FieldResources;
use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::structs::CoupledSolverResources;
use crate::solver::gpu::wgsl_meta;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::graph::{DispatchKind, GpuComputeModule, RuntimeDims};

#[derive(Clone, Copy, Debug)]
pub enum IncompressiblePipeline {
    PrepareCoupled,
    CoupledAssemblyMerged,
    UpdateFieldsFromCoupled,
}

#[derive(Clone, Copy, Debug)]
pub enum IncompressibleBindGroups {
    MeshFieldsSolver,
    UpdateFieldsSolution,
}

pub struct IncompressibleKernelsModule {
    state_step_index: Arc<AtomicUsize>,

    bg_mesh: wgpu::BindGroup,
    bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_solver: wgpu::BindGroup,

    bg_update_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_update_solution: wgpu::BindGroup,

    pipeline_prepare_coupled: wgpu::ComputePipeline,
    pipeline_coupled_assembly_merged: wgpu::ComputePipeline,
    pipeline_update_fields_from_coupled: wgpu::ComputePipeline,
}

impl IncompressibleKernelsModule {
    pub fn new(
        device: &wgpu::Device,
        mesh: &MeshResources,
        fields: &FieldResources,
        state_step_index: Arc<AtomicUsize>,
        coupled: &CoupledSolverResources,
    ) -> Self {
        let pipeline_prepare_coupled =
            generated_prepare_coupled::compute::create_main_pipeline_embed_source(device);
        let pipeline_coupled_assembly_merged =
            generated_coupled_assembly::compute::create_main_pipeline_embed_source(device);
        let pipeline_update_fields_from_coupled =
            generated_update_fields::compute::create_main_pipeline_embed_source(device);

        let bg_mesh = {
            let bgl = pipeline_prepare_coupled.get_bind_group_layout(0);
            crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                device,
                "Incompressible: mesh bind group",
                &bgl,
                wgsl_meta::PREPARE_COUPLED_BINDINGS,
                0,
                |name| {
                    mesh.buffer_for_binding_name(name)
                        .map(|buf| wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding()))
                },
            )
            .unwrap_or_else(|err| panic!("Incompressible mesh bind group build failed: {err}"))
        };

        let bg_fields_ping_pong = fields.bg_fields_ping_pong.clone();
        let bg_solver = coupled.bg_solver.clone();

        let bg_update_fields_ping_pong = {
            let bgl = pipeline_update_fields_from_coupled.get_bind_group_layout(0);
            let mut out = Vec::with_capacity(3);
            for i in 0..3 {
                let (idx_state, idx_old, idx_old_old) =
                    crate::solver::gpu::modules::state::ping_pong_indices(i);
                out.push(
                    crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        &format!("Incompressible update fields bind group {i}"),
                        &bgl,
                        wgsl_meta::UPDATE_FIELDS_FROM_COUPLED_BINDINGS,
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
                            "fluxes" => Some(wgpu::BindingResource::Buffer(
                                fields.b_fluxes.as_entire_buffer_binding(),
                            )),
                            "constants" => Some(wgpu::BindingResource::Buffer(
                                fields.constants.buffer().as_entire_buffer_binding(),
                            )),
                            _ => None,
                        },
                    )
                    .unwrap_or_else(|err| {
                        panic!("Incompressible update-fields bind group build failed: {err}")
                    }),
                );
            }
            out
        };

        let bg_update_solution = coupled.bg_coupled_solution.clone();

        Self {
            state_step_index,
            bg_mesh,
            bg_fields_ping_pong,
            bg_solver,
            bg_update_fields_ping_pong,
            bg_update_solution,
            pipeline_prepare_coupled,
            pipeline_coupled_assembly_merged,
            pipeline_update_fields_from_coupled,
        }
    }

    pub fn set_step_index(&mut self, idx: usize) {
        self.state_step_index.store(idx % 3, Ordering::Relaxed);
    }

    fn bg_fields(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self.bg_fields_ping_pong[idx]
    }

    fn bg_update_fields(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self.bg_update_fields_ping_pong[idx]
    }

    fn bind_mesh_fields_solver(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, &self.bg_mesh, &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
        pass.set_bind_group(2, &self.bg_solver, &[]);
    }

    fn bind_update_fields_solution(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_update_fields(), &[]);
        pass.set_bind_group(1, &self.bg_update_solution, &[]);
    }
}

impl GpuComputeModule for IncompressibleKernelsModule {
    type PipelineKey = IncompressiblePipeline;
    type BindKey = IncompressibleBindGroups;

    fn pipeline(&self, key: Self::PipelineKey) -> &wgpu::ComputePipeline {
        match key {
            IncompressiblePipeline::PrepareCoupled => &self.pipeline_prepare_coupled,
            IncompressiblePipeline::CoupledAssemblyMerged => &self.pipeline_coupled_assembly_merged,
            IncompressiblePipeline::UpdateFieldsFromCoupled => {
                &self.pipeline_update_fields_from_coupled
            }
        }
    }

    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass) {
        match key {
            IncompressibleBindGroups::MeshFieldsSolver => self.bind_mesh_fields_solver(pass),
            IncompressibleBindGroups::UpdateFieldsSolution => {
                self.bind_update_fields_solution(pass)
            }
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
