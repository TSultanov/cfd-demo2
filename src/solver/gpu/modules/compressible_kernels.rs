use crate::solver::gpu::bindings::compressible_explicit_update as explicit_update;
use crate::solver::gpu::bindings::generated::{
    compressible_apply as generated_apply, compressible_assembly as generated_assembly,
    compressible_flux_kt as generated_flux_kt, compressible_gradients as generated_gradients,
    compressible_update as generated_update,
};
use crate::solver::gpu::init::compressible_fields::CompressibleFieldResources;
use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::init::linear_solver::matrix::MatrixResources;

use super::compressible_lowering::CompressibleLinearPorts;
use super::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use super::ports::PortSpace;

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
    state_step_index: usize,

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
        matrix: &MatrixResources,
        port_space: &PortSpace,
        ports: CompressibleLinearPorts,
        _scalar_row_offsets: &[u32],
    ) -> Self {
        let b_rhs = port_space.buffer(ports.rhs);
        let b_x = port_space.buffer(ports.x);
        let b_scalar_row_offsets = port_space.buffer(ports.scalar_row_offsets);

        let pipeline_assembly =
            generated_assembly::compute::create_main_pipeline_embed_source(device);
        let pipeline_apply = generated_apply::compute::create_main_pipeline_embed_source(device);
        let pipeline_gradients =
            generated_gradients::compute::create_main_pipeline_embed_source(device);
        let pipeline_flux = generated_flux_kt::compute::create_main_pipeline_embed_source(device);
        let pipeline_explicit_update =
            explicit_update::compute::create_main_pipeline_embed_source(device);
        let pipeline_update = generated_update::compute::create_main_pipeline_embed_source(device);

        let mesh_layout =
            device.create_bind_group_layout(&generated_assembly::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
        let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible Mesh Bind Group"),
            layout: &mesh_layout,
            entries: &generated_assembly::WgpuBindGroup0Entries::new(
                generated_assembly::WgpuBindGroup0EntriesParams {
                    face_owner: mesh.b_face_owner.as_entire_buffer_binding(),
                    face_neighbor: mesh.b_face_neighbor.as_entire_buffer_binding(),
                    face_areas: mesh.b_face_areas.as_entire_buffer_binding(),
                    face_normals: mesh.b_face_normals.as_entire_buffer_binding(),
                    cell_centers: mesh.b_cell_centers.as_entire_buffer_binding(),
                    cell_vols: mesh.b_cell_vols.as_entire_buffer_binding(),
                    cell_face_offsets: mesh.b_cell_face_offsets.as_entire_buffer_binding(),
                    cell_faces: mesh.b_cell_faces.as_entire_buffer_binding(),
                    cell_face_matrix_indices: mesh.b_cell_face_matrix_indices.as_entire_buffer_binding(),
                    diagonal_indices: mesh.b_diagonal_indices.as_entire_buffer_binding(),
                    face_boundary: mesh.b_face_boundary.as_entire_buffer_binding(),
                    face_centers: mesh.b_face_centers.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });

        let bg_fields_ping_pong = fields.bg_fields_ping_pong.clone();

        let solver_layout =
            device.create_bind_group_layout(&generated_assembly::WgpuBindGroup2::LAYOUT_DESCRIPTOR);
        let bg_solver = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible Solver Bind Group"),
            layout: &solver_layout,
            entries: &generated_assembly::WgpuBindGroup2Entries::new(
                generated_assembly::WgpuBindGroup2EntriesParams {
                    matrix_values: matrix.b_matrix_values.as_entire_buffer_binding(),
                    rhs: b_rhs.as_entire_buffer_binding(),
                    scalar_row_offsets: b_scalar_row_offsets.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });

        let apply_fields_layout =
            device.create_bind_group_layout(&generated_apply::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
        let mut bg_apply_fields_ping_pong = Vec::new();
        for i in 0..3 {
            let (idx_state, idx_old, idx_old_old) = match i {
                0 => (0, 1, 2),
                1 => (2, 0, 1),
                2 => (1, 2, 0),
                _ => (0, 1, 2),
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Compressible Apply Fields Bind Group {}", i)),
                layout: &apply_fields_layout,
                entries: &generated_apply::WgpuBindGroup0Entries::new(
                    generated_apply::WgpuBindGroup0EntriesParams {
                        state: fields.state_buffers[idx_state].as_entire_buffer_binding(),
                        state_old: fields.state_buffers[idx_old].as_entire_buffer_binding(),
                        state_old_old: fields.state_buffers[idx_old_old].as_entire_buffer_binding(),
                        state_iter: fields.b_state_iter.as_entire_buffer_binding(),
                        fluxes: fields.b_fluxes.as_entire_buffer_binding(),
                        constants: fields.b_constants.as_entire_buffer_binding(),
                        grad_rho: fields.b_grad_rho.as_entire_buffer_binding(),
                        grad_rho_u_x: fields.b_grad_rho_u_x.as_entire_buffer_binding(),
                        grad_rho_u_y: fields.b_grad_rho_u_y.as_entire_buffer_binding(),
                        grad_rho_e: fields.b_grad_rho_e.as_entire_buffer_binding(),
                        low_mach: fields.b_low_mach_params.as_entire_buffer_binding(),
                    },
                )
                .into_array(),
            });
            bg_apply_fields_ping_pong.push(bg);
        }

        let apply_solver_layout =
            device.create_bind_group_layout(&generated_apply::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
        let bg_apply_solver = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible Apply Solver Bind Group"),
            layout: &apply_solver_layout,
            entries: &generated_apply::WgpuBindGroup1Entries::new(
                generated_apply::WgpuBindGroup1EntriesParams {
                    solution: b_x.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });

        Self {
            state_step_index: 0,
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
        self.state_step_index = idx % 3;
    }

    pub fn bg_mesh(&self) -> &wgpu::BindGroup {
        &self.bg_mesh
    }

    pub fn bg_fields(&self) -> &wgpu::BindGroup {
        &self.bg_fields_ping_pong[self.state_step_index]
    }

    pub fn bg_apply_fields(&self) -> &wgpu::BindGroup {
        &self.bg_apply_fields_ping_pong[self.state_step_index]
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
