use crate::solver::gpu::bindings::generated::{
    coupled_assembly_merged as generated_coupled_assembly,
    prepare_coupled as generated_prepare_coupled,
    update_fields_from_coupled as generated_update_fields,
};
use crate::solver::gpu::init::fields::FieldResources;
use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::structs::CoupledSolverResources;

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
    state_step_index: usize,

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
        coupled: &CoupledSolverResources,
    ) -> Self {
        let pipeline_prepare_coupled =
            generated_prepare_coupled::compute::create_main_pipeline_embed_source(device);
        let pipeline_coupled_assembly_merged =
            generated_coupled_assembly::compute::create_main_pipeline_embed_source(device);
        let pipeline_update_fields_from_coupled =
            generated_update_fields::compute::create_main_pipeline_embed_source(device);

        let mesh_layout = device.create_bind_group_layout(
            &generated_prepare_coupled::WgpuBindGroup0::LAYOUT_DESCRIPTOR,
        );
        let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Incompressible: mesh bind group"),
            layout: &mesh_layout,
            entries: &generated_prepare_coupled::WgpuBindGroup0Entries::new(
                generated_prepare_coupled::WgpuBindGroup0EntriesParams {
                    face_owner: mesh.b_face_owner.as_entire_buffer_binding(),
                    face_neighbor: mesh.b_face_neighbor.as_entire_buffer_binding(),
                    face_areas: mesh.b_face_areas.as_entire_buffer_binding(),
                    face_normals: mesh.b_face_normals.as_entire_buffer_binding(),
                    face_centers: mesh.b_face_centers.as_entire_buffer_binding(),
                    cell_centers: mesh.b_cell_centers.as_entire_buffer_binding(),
                    cell_vols: mesh.b_cell_vols.as_entire_buffer_binding(),
                    cell_face_offsets: mesh.b_cell_face_offsets.as_entire_buffer_binding(),
                    cell_faces: mesh.b_cell_faces.as_entire_buffer_binding(),
                    cell_face_matrix_indices: mesh
                        .b_cell_face_matrix_indices
                        .as_entire_buffer_binding(),
                    diagonal_indices: mesh.b_diagonal_indices.as_entire_buffer_binding(),
                    face_boundary: mesh.b_face_boundary.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });

        let bg_fields_ping_pong = fields.bg_fields_ping_pong.clone();
        let bg_solver = coupled.bg_solver.clone();

        let bg_update_fields_ping_pong = {
            let bgl = device.create_bind_group_layout(
                &generated_update_fields::WgpuBindGroup0::LAYOUT_DESCRIPTOR,
            );
            let mut out = Vec::with_capacity(3);
            for i in 0..3 {
                let (idx_state, idx_old, idx_old_old) = match i {
                    0 => (0, 1, 2),
                    1 => (2, 0, 1),
                    2 => (1, 2, 0),
                    _ => (0, 1, 2),
                };
                out.push(
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Incompressible update fields bind group {i}")),
                        layout: &bgl,
                        entries: &generated_update_fields::WgpuBindGroup0Entries::new(
                            generated_update_fields::WgpuBindGroup0EntriesParams {
                                state: fields.state_buffers[idx_state].as_entire_buffer_binding(),
                                state_old: fields.state_buffers[idx_old].as_entire_buffer_binding(),
                                state_old_old: fields.state_buffers[idx_old_old]
                                    .as_entire_buffer_binding(),
                                fluxes: fields.b_fluxes.as_entire_buffer_binding(),
                                constants: fields.b_constants.as_entire_buffer_binding(),
                            },
                        )
                        .into_array(),
                    }),
                );
            }
            out
        };

        let bg_update_solution = coupled.bg_coupled_solution.clone();

        Self {
            state_step_index: 0,
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
        self.state_step_index = idx % 3;
    }

    fn bg_fields(&self) -> &wgpu::BindGroup {
        &self.bg_fields_ping_pong[self.state_step_index]
    }

    fn bg_update_fields(&self) -> &wgpu::BindGroup {
        &self.bg_update_fields_ping_pong[self.state_step_index]
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
