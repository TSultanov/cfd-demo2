use crate::solver::gpu::bindings::compressible_explicit_update as explicit_update;
use crate::solver::gpu::bindings::generated::{
    coupled_assembly_merged as generated_coupled_assembly,
    prepare_coupled as generated_prepare_coupled, update_fields_from_coupled as generated_update_fields,
};
use crate::solver::gpu::init::compressible_fields::CompressibleFieldResources;
use crate::solver::gpu::init::fields::FieldResources;
use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use crate::solver::gpu::modules::linear_system::{LinearSystemPorts, LinearSystemView};
use crate::solver::gpu::modules::ports::{BufU32, Port, PortSpace};
use crate::solver::gpu::structs::CoupledSolverResources;
use crate::solver::gpu::wgsl_meta;
use crate::solver::model::KernelKind;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelPipeline {
    Kernel(KernelKind),
    CompressibleExplicitUpdate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelBindGroups {
    MeshFields,
    MeshFieldsSolver,
    FieldsOnly,
    ApplyFieldsSolver,
    UpdateFieldsSolution,
}

pub struct ModelKernelsModule {
    state_step_index: Arc<AtomicUsize>,
    pipelines: HashMap<KernelPipeline, wgpu::ComputePipeline>,

    bg_mesh: Option<wgpu::BindGroup>,
    bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_solver: Option<wgpu::BindGroup>,

    bg_fields_group0_ping_pong: Option<Vec<wgpu::BindGroup>>,

    bg_apply_fields_ping_pong: Option<Vec<wgpu::BindGroup>>,
    bg_apply_solver: Option<wgpu::BindGroup>,

    bg_update_fields_ping_pong: Option<Vec<wgpu::BindGroup>>,
    bg_update_solution: Option<wgpu::BindGroup>,
}

impl ModelKernelsModule {
    pub fn new_compressible(
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

        let mut pipelines = HashMap::new();
        for kind in [
            KernelKind::CompressibleAssembly,
            KernelKind::CompressibleApply,
            KernelKind::CompressibleGradients,
            KernelKind::CompressibleFluxKt,
            KernelKind::CompressibleUpdate,
        ] {
            let source = kernel_registry::kernel_source("compressible", kind)
                .unwrap_or_else(|err| panic!("missing kernel source for {kind:?}: {err}"));
            pipelines.insert(
                KernelPipeline::Kernel(kind),
                (source.create_pipeline)(device),
            );
        }
        pipelines.insert(
            KernelPipeline::CompressibleExplicitUpdate,
            explicit_update::compute::create_main_pipeline_embed_source(device),
        );

        let pipeline_assembly =
            &pipelines[&KernelPipeline::Kernel(KernelKind::CompressibleAssembly)];
        let pipeline_apply = &pipelines[&KernelPipeline::Kernel(KernelKind::CompressibleApply)];
        let pipeline_update = &pipelines[&KernelPipeline::Kernel(KernelKind::CompressibleUpdate)];

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

        let bg_apply_fields_ping_pong = {
            let bgl = pipeline_apply.get_bind_group_layout(0);
            let mut out = Vec::new();
            for i in 0..3 {
                let (idx_state, idx_old, idx_old_old) =
                    crate::solver::gpu::modules::state::ping_pong_indices(i);
                let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("Compressible Apply Fields Bind Group {}", i),
                    &bgl,
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
                out.push(bg);
            }
            out
        };

        let bg_apply_solver = {
            let bgl = pipeline_apply.get_bind_group_layout(1);
            crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                device,
                "Compressible Apply Solver Bind Group",
                &bgl,
                wgsl_meta::COMPRESSIBLE_APPLY_BINDINGS,
                1,
                |name| match name {
                    "solution" => Some(wgpu::BindingResource::Buffer(
                        system.x().as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )
            .unwrap_or_else(|err| {
                panic!("Compressible apply-solver bind group build failed: {err}")
            })
        };

        let bg_fields_group0_ping_pong = {
            let bgl = pipeline_update.get_bind_group_layout(0);
            let mut out = Vec::new();
            for i in 0..3 {
                let (idx_state, idx_old, idx_old_old) =
                    crate::solver::gpu::modules::state::ping_pong_indices(i);
                let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("Compressible Update Fields Bind Group {}", i),
                    &bgl,
                    wgsl_meta::COMPRESSIBLE_UPDATE_BINDINGS,
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
                    panic!("Compressible update-fields bind group build failed: {err}")
                });
                out.push(bg);
            }
            out
        };

        Self {
            state_step_index,
            pipelines,
            bg_mesh: Some(bg_mesh),
            bg_fields_ping_pong,
            bg_solver: Some(bg_solver),
            bg_fields_group0_ping_pong: Some(bg_fields_group0_ping_pong),
            bg_apply_fields_ping_pong: Some(bg_apply_fields_ping_pong),
            bg_apply_solver: Some(bg_apply_solver),
            bg_update_fields_ping_pong: None,
            bg_update_solution: None,
        }
    }

    pub fn new_incompressible(
        device: &wgpu::Device,
        mesh: &MeshResources,
        fields: &FieldResources,
        state_step_index: Arc<AtomicUsize>,
        coupled: &CoupledSolverResources,
    ) -> Self {
        let mut pipelines = HashMap::new();
        pipelines.insert(
            KernelPipeline::Kernel(KernelKind::PrepareCoupled),
            generated_prepare_coupled::compute::create_main_pipeline_embed_source(device),
        );
        pipelines.insert(
            KernelPipeline::Kernel(KernelKind::CoupledAssembly),
            generated_coupled_assembly::compute::create_main_pipeline_embed_source(device),
        );
        pipelines.insert(
            KernelPipeline::Kernel(KernelKind::UpdateFieldsFromCoupled),
            generated_update_fields::compute::create_main_pipeline_embed_source(device),
        );

        let pipeline_prepare = &pipelines[&KernelPipeline::Kernel(KernelKind::PrepareCoupled)];
        let pipeline_update =
            &pipelines[&KernelPipeline::Kernel(KernelKind::UpdateFieldsFromCoupled)];

        let bg_mesh = {
            let bgl = pipeline_prepare.get_bind_group_layout(0);
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

        let bg_update_fields_ping_pong = {
            let bgl = pipeline_update.get_bind_group_layout(0);
            let mut out = Vec::with_capacity(3);
            for i in 0..3 {
                let (idx_state, idx_old, idx_old_old) =
                    crate::solver::gpu::modules::state::ping_pong_indices(i);
                let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
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
                });
                out.push(bg);
            }
            out
        };

        Self {
            state_step_index,
            pipelines,
            bg_mesh: Some(bg_mesh),
            bg_fields_ping_pong,
            bg_solver: Some(coupled.bg_solver.clone()),
            bg_fields_group0_ping_pong: None,
            bg_apply_fields_ping_pong: None,
            bg_apply_solver: None,
            bg_update_fields_ping_pong: Some(bg_update_fields_ping_pong),
            bg_update_solution: Some(coupled.bg_coupled_solution.clone()),
        }
    }

    pub fn set_step_index(&mut self, idx: usize) {
        self.state_step_index.store(idx % 3, Ordering::Relaxed);
    }

    fn step_index(&self) -> usize {
        self.state_step_index.load(Ordering::Relaxed) % 3
    }

    fn bg_mesh(&self) -> &wgpu::BindGroup {
        self.bg_mesh.as_ref().expect("missing mesh bind group")
    }

    fn bg_fields(&self) -> &wgpu::BindGroup {
        let idx = self.step_index();
        &self.bg_fields_ping_pong[idx]
    }

    fn bg_solver(&self) -> &wgpu::BindGroup {
        self.bg_solver.as_ref().expect("missing solver bind group")
    }

    fn bg_fields_group0(&self) -> &wgpu::BindGroup {
        let idx = self.step_index();
        &self
            .bg_fields_group0_ping_pong
            .as_ref()
            .expect("missing fields-only bind groups")[idx]
    }

    fn bg_apply_fields(&self) -> &wgpu::BindGroup {
        let idx = self.step_index();
        &self
            .bg_apply_fields_ping_pong
            .as_ref()
            .expect("missing apply fields bind groups")[idx]
    }

    fn bg_apply_solver(&self) -> &wgpu::BindGroup {
        self.bg_apply_solver
            .as_ref()
            .expect("missing apply solver bind group")
    }

    fn bg_update_fields(&self) -> &wgpu::BindGroup {
        let idx = self.step_index();
        &self
            .bg_update_fields_ping_pong
            .as_ref()
            .expect("missing update fields bind groups")[idx]
    }

    fn bg_update_solution(&self) -> &wgpu::BindGroup {
        self.bg_update_solution
            .as_ref()
            .expect("missing update solution bind group")
    }

    fn bind_mesh_fields(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh(), &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
    }

    fn bind_mesh_fields_solver(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh(), &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
        pass.set_bind_group(2, self.bg_solver(), &[]);
    }

    fn bind_fields_only(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_fields_group0(), &[]);
    }

    fn bind_apply_fields_solver(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_apply_fields(), &[]);
        pass.set_bind_group(1, self.bg_apply_solver(), &[]);
    }

    fn bind_update_fields_solution(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_update_fields(), &[]);
        pass.set_bind_group(1, self.bg_update_solution(), &[]);
    }
}

impl GpuComputeModule for ModelKernelsModule {
    type PipelineKey = KernelPipeline;
    type BindKey = KernelBindGroups;

    fn pipeline(&self, key: Self::PipelineKey) -> &wgpu::ComputePipeline {
        self.pipelines
            .get(&key)
            .unwrap_or_else(|| panic!("missing pipeline {key:?}"))
    }

    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass) {
        match key {
            KernelBindGroups::MeshFields => self.bind_mesh_fields(pass),
            KernelBindGroups::MeshFieldsSolver => self.bind_mesh_fields_solver(pass),
            KernelBindGroups::FieldsOnly => self.bind_fields_only(pass),
            KernelBindGroups::ApplyFieldsSolver => self.bind_apply_fields_solver(pass),
            KernelBindGroups::UpdateFieldsSolution => self.bind_update_fields_solution(pass),
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
