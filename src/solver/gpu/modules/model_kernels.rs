use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::field_provider::FieldProvider;
use crate::solver::gpu::modules::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use crate::solver::gpu::modules::linear_system::{LinearSystemPorts, LinearSystemView};
use crate::solver::gpu::modules::ports::{BufU32, Port, PortSpace};
use crate::solver::gpu::modules::unified_graph::UnifiedGraphModule;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::structs::CoupledSolverResources;
use crate::solver::gpu::wgsl_meta;
use crate::solver::model::KernelId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Helper to get a binding resource from a FieldProvider.
fn field_binding<'a, F: FieldProvider>(
    fields: &'a F,
    name: &str,
    ping_pong_phase: usize,
) -> Option<wgpu::BindingResource<'a>> {
    fields
        .buffer_for_binding(name, ping_pong_phase)
        .map(|buf| wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelPipeline {
    Kernel(KernelId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelBindGroups {
    MeshFields,
    MeshFieldsSolver,
    FieldsOnly,
    ApplyFieldsSolver,
    UpdateFieldsSolution,
}

pub struct ModelKernelsInit<'a> {
    pub port_space: Option<&'a PortSpace>,
    pub system_ports: Option<LinearSystemPorts>,
    pub scalar_row_offsets: Option<Port<BufU32>>,
    pub coupled: Option<&'a CoupledSolverResources>,
}

impl<'a> ModelKernelsInit<'a> {
    pub fn linear_system(
        port_space: &'a PortSpace,
        system_ports: LinearSystemPorts,
        scalar_row_offsets: Port<BufU32>,
    ) -> Self {
        Self {
            port_space: Some(port_space),
            system_ports: Some(system_ports),
            scalar_row_offsets: Some(scalar_row_offsets),
            coupled: None,
        }
    }

    pub fn coupled(coupled: &'a CoupledSolverResources) -> Self {
        Self {
            port_space: None,
            system_ports: None,
            scalar_row_offsets: None,
            coupled: Some(coupled),
        }
    }
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
    pub fn new_from_recipe<F: FieldProvider>(
        device: &wgpu::Device,
        mesh: &MeshResources,
        model_id: &str,
        recipe: &SolverRecipe,
        fields: &F,
        state_step_index: Arc<AtomicUsize>,
        init: ModelKernelsInit<'_>,
    ) -> Self {
        let mut pipelines = HashMap::new();
        for k in &recipe.kernels {
            let id = k.id;
            let source = kernel_registry::kernel_source_by_id(model_id, id)
                .unwrap_or_else(|err| panic!("missing kernel source for {id:?}: {err}"));
            pipelines.insert(KernelPipeline::Kernel(id), (source.create_pipeline)(device));
        }

        let system = match (init.port_space, init.system_ports) {
            (Some(space), Some(ports)) => Some(LinearSystemView { ports, space }),
            _ => None,
        };

        // Base mesh+field bind groups (shared by most physics kernels).
        // Choose the anchor kernel that defines the layout.
        let anchor_kernel = if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::EI_ASSEMBLY)) {
            KernelId::EI_ASSEMBLY
        } else if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::PREPARE_COUPLED)) {
            KernelId::PREPARE_COUPLED
        } else if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::COUPLED_ASSEMBLY)) {
            KernelId::COUPLED_ASSEMBLY
        } else {
            panic!("ModelKernelsModule: no anchor kernel available for mesh/fields bind groups");
        };

        let (bg_mesh, bg_fields_ping_pong, bg_solver) = match anchor_kernel {
            KernelId::EI_ASSEMBLY => {
                let Some(system) = system else {
                    panic!("ModelKernelsModule: compressible kernels require linear system ports");
                };
                let Some(space) = init.port_space else {
                    panic!("ModelKernelsModule: missing port space for linear system ports");
                };
                let scalar_row_offsets_port = init
                    .scalar_row_offsets
                    .expect("ModelKernelsModule: missing scalar_row_offsets port");
                let b_scalar_row_offsets = space.buffer(scalar_row_offsets_port);

                let pipeline_assembly = &pipelines[&KernelPipeline::Kernel(KernelId::EI_ASSEMBLY)];

                let bg_mesh = {
                    let bgl = pipeline_assembly.get_bind_group_layout(0);
                    crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        "ModelKernels: mesh bind group",
                        &bgl,
                        wgsl_meta::EI_ASSEMBLY_BINDINGS,
                        0,
                        |name| {
                            mesh.buffer_for_binding_name(name).map(|buf| {
                                wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding())
                            })
                        },
                    )
                    .unwrap_or_else(|err| panic!("ModelKernels mesh bind group build failed: {err}"))
                };

                let bg_fields_ping_pong = {
                    let bgl = pipeline_assembly.get_bind_group_layout(1);
                    let mut out = Vec::with_capacity(3);
                    for i in 0..3 {
                        let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                            device,
                            &format!("ModelKernels: fields bind group {i}"),
                            &bgl,
                            wgsl_meta::EI_ASSEMBLY_BINDINGS,
                            1,
                            |name| field_binding(fields, name, i),
                        )
                        .unwrap_or_else(|err| {
                            panic!("ModelKernels fields bind group build failed: {err}")
                        });
                        out.push(bg);
                    }
                    out
                };

                let bg_solver = {
                    let bgl = pipeline_assembly.get_bind_group_layout(2);
                    crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        "ModelKernels: solver bind group",
                        &bgl,
                        wgsl_meta::EI_ASSEMBLY_BINDINGS,
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
                    .unwrap_or_else(|err| {
                        panic!("ModelKernels solver bind group build failed: {err}")
                    })
                };

                (Some(bg_mesh), bg_fields_ping_pong, Some(bg_solver))
            }

            KernelId::PREPARE_COUPLED | KernelId::COUPLED_ASSEMBLY => {
                let coupled = init
                    .coupled
                    .expect("ModelKernelsModule: coupled kernels require CoupledSolverResources");
                let pipeline_prepare =
                    &pipelines[&KernelPipeline::Kernel(KernelId::PREPARE_COUPLED)];

                let bg_mesh = {
                    let bgl = pipeline_prepare.get_bind_group_layout(0);
                    crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        "ModelKernels: mesh bind group",
                        &bgl,
                        wgsl_meta::PREPARE_COUPLED_BINDINGS,
                        0,
                        |name| {
                            mesh.buffer_for_binding_name(name).map(|buf| {
                                wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding())
                            })
                        },
                    )
                    .unwrap_or_else(|err| panic!("ModelKernels mesh bind group build failed: {err}"))
                };

                let bg_fields_ping_pong = {
                    let bgl = pipeline_prepare.get_bind_group_layout(1);
                    let mut out = Vec::with_capacity(3);
                    for i in 0..3 {
                        let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                            device,
                            &format!("ModelKernels: fields bind group {i}"),
                            &bgl,
                            wgsl_meta::PREPARE_COUPLED_BINDINGS,
                            1,
                            |name| field_binding(fields, name, i),
                        )
                        .unwrap_or_else(|err| {
                            panic!("ModelKernels fields bind group build failed: {err}")
                        });
                        out.push(bg);
                    }
                    out
                };

                (
                    Some(bg_mesh),
                    bg_fields_ping_pong,
                    Some(coupled.bg_solver.clone()),
                )
            }

            _ => unreachable!("anchor kernel must be one of the known anchors"),
        };

        let bg_apply_fields_ping_pong =
            if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::EI_APPLY)) {
                let pipeline_apply = &pipelines[&KernelPipeline::Kernel(KernelId::EI_APPLY)];
            let bgl = pipeline_apply.get_bind_group_layout(0);
            let mut out = Vec::with_capacity(3);
            for i in 0..3 {
                let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("ModelKernels: apply fields bind group {i}"),
                    &bgl,
                    wgsl_meta::EI_APPLY_BINDINGS,
                    0,
                    |name| field_binding(fields, name, i),
                )
                .unwrap_or_else(|err| panic!("ModelKernels apply-fields bind group build failed: {err}"));
                out.push(bg);
            }
            Some(out)
        } else {
            None
        };

        let bg_apply_solver = if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::EI_APPLY)) {
            let Some(system) = system else {
                panic!("ModelKernelsModule: apply kernel requires linear system ports");
            };
            let pipeline_apply = &pipelines[&KernelPipeline::Kernel(KernelId::EI_APPLY)];
            let bgl = pipeline_apply.get_bind_group_layout(1);
            Some(
                crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    "ModelKernels: apply solver bind group",
                    &bgl,
                    wgsl_meta::EI_APPLY_BINDINGS,
                    1,
                    |name| match name {
                        "solution" => Some(wgpu::BindingResource::Buffer(
                            system.x().as_entire_buffer_binding(),
                        )),
                        _ => None,
                    },
                )
                .unwrap_or_else(|err| {
                    panic!("ModelKernels apply-solver bind group build failed: {err}")
                }),
            )
        } else {
            None
        };

        let bg_fields_group0_ping_pong =
            if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::EI_UPDATE)) {
                let pipeline_update = &pipelines[&KernelPipeline::Kernel(KernelId::EI_UPDATE)];
            let bgl = pipeline_update.get_bind_group_layout(0);
            let mut out = Vec::with_capacity(3);
            for i in 0..3 {
                let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("ModelKernels: fields-only bind group {i}"),
                    &bgl,
                    wgsl_meta::EI_UPDATE_BINDINGS,
                    0,
                    |name| field_binding(fields, name, i),
                )
                .unwrap_or_else(|err| {
                    panic!("ModelKernels fields-only bind group build failed: {err}")
                });
                out.push(bg);
            }
            Some(out)
        } else {
            None
        };

        let bg_update_fields_ping_pong = if pipelines
            .contains_key(&KernelPipeline::Kernel(KernelId::UPDATE_FIELDS_FROM_COUPLED))
        {
            let pipeline_update =
                &pipelines[&KernelPipeline::Kernel(KernelId::UPDATE_FIELDS_FROM_COUPLED)];
            let bgl = pipeline_update.get_bind_group_layout(0);
            let mut out = Vec::with_capacity(3);
            for i in 0..3 {
                let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("ModelKernels: update fields bind group {i}"),
                    &bgl,
                    wgsl_meta::UPDATE_FIELDS_FROM_COUPLED_BINDINGS,
                    0,
                    |name| field_binding(fields, name, i),
                )
                .unwrap_or_else(|err| {
                    panic!("ModelKernels update-fields bind group build failed: {err}")
                });
                out.push(bg);
            }
            Some(out)
        } else {
            None
        };

        let bg_update_solution = if pipelines
            .contains_key(&KernelPipeline::Kernel(KernelId::UPDATE_FIELDS_FROM_COUPLED))
        {
            let coupled = init
                .coupled
                .expect("ModelKernelsModule: update-from-coupled requires CoupledSolverResources");
            Some(coupled.bg_coupled_solution.clone())
        } else {
            None
        };

        Self {
            state_step_index,
            pipelines,
            bg_mesh,
            bg_fields_ping_pong,
            bg_solver,
            bg_fields_group0_ping_pong,
            bg_apply_fields_ping_pong,
            bg_apply_solver,
            bg_update_fields_ping_pong,
            bg_update_solution,
        }
    }

    #[allow(dead_code)]
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

impl UnifiedGraphModule for ModelKernelsModule {
    fn pipeline_for_kernel(&self, id: KernelId) -> Option<Self::PipelineKey> {
        // Only return Some if we have the pipeline registered
        if self.pipelines.contains_key(&KernelPipeline::Kernel(id)) {
            Some(KernelPipeline::Kernel(id))
        } else {
            None
        }
    }

    fn bind_for_kernel(&self, id: KernelId) -> Option<Self::BindKey> {
        // Map kernel kinds to their bind group requirements
        match id {
            // EI kernels
            KernelId::EI_ASSEMBLY => Some(KernelBindGroups::MeshFieldsSolver),
            KernelId::EI_APPLY => Some(KernelBindGroups::ApplyFieldsSolver),
            KernelId::EI_EXPLICIT_UPDATE => Some(KernelBindGroups::MeshFields),
            KernelId::EI_GRADIENTS => Some(KernelBindGroups::MeshFields),
            KernelId::EI_FLUX_KT => Some(KernelBindGroups::MeshFields),
            KernelId::EI_UPDATE => Some(KernelBindGroups::FieldsOnly),
            // Incompressible kernels
            KernelId::PREPARE_COUPLED => Some(KernelBindGroups::MeshFieldsSolver),
            KernelId::COUPLED_ASSEMBLY => Some(KernelBindGroups::MeshFieldsSolver),
            KernelId::UPDATE_FIELDS_FROM_COUPLED => Some(KernelBindGroups::UpdateFieldsSolution),
            // Shared kernels
            KernelId::FLUX_RHIE_CHOW => Some(KernelBindGroups::MeshFields),
            // Unknown or unsupported kernels
            _ => None,
        }
    }
}
