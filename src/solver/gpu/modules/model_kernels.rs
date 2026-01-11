use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::field_provider::FieldProvider;
use crate::solver::gpu::modules::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use crate::solver::gpu::modules::linear_system::LinearSystemPorts;
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
    UpdateFieldsSolution,

    // Legacy compressible kernels (EI family) bind group layouts.
    ConservativeMeshFields,
    ConservativeMeshFieldsBc,
    ConservativeMeshFieldsSolverBc,
    ConservativeFieldsOnly,
    ConservativeFieldsSolution,
}

pub struct ModelKernelsInit<'a> {
    pub port_space: Option<&'a PortSpace>,
    pub system_ports: Option<LinearSystemPorts>,
    pub scalar_row_offsets: Option<Port<BufU32>>,
    pub coupled: Option<&'a CoupledSolverResources>,
    pub bc_kind: Option<&'a wgpu::Buffer>,
    pub bc_value: Option<&'a wgpu::Buffer>,
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
            bc_kind: None,
            bc_value: None,
        }
    }

    pub fn linear_system_with_bc(
        port_space: &'a PortSpace,
        system_ports: LinearSystemPorts,
        scalar_row_offsets: Port<BufU32>,
        bc_kind: &'a wgpu::Buffer,
        bc_value: &'a wgpu::Buffer,
    ) -> Self {
        Self {
            port_space: Some(port_space),
            system_ports: Some(system_ports),
            scalar_row_offsets: Some(scalar_row_offsets),
            coupled: None,
            bc_kind: Some(bc_kind),
            bc_value: Some(bc_value),
        }
    }

    pub fn coupled(coupled: &'a CoupledSolverResources) -> Self {
        Self {
            port_space: None,
            system_ports: None,
            scalar_row_offsets: None,
            coupled: Some(coupled),
            bc_kind: None,
            bc_value: None,
        }
    }
}

pub struct ModelKernelsModule {
    state_step_index: Arc<AtomicUsize>,
    pipelines: HashMap<KernelPipeline, wgpu::ComputePipeline>,

    bg_mesh: Option<wgpu::BindGroup>,
    bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_solver: Option<wgpu::BindGroup>,

    // Optional extras for legacy compressible kernels.
    bg_bc: Option<wgpu::BindGroup>,
    bg_bc_group2: Option<wgpu::BindGroup>,
    bg_fields0_ping_pong: Option<Vec<wgpu::BindGroup>>,
    bg_solution: Option<wgpu::BindGroup>,

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

        // --- Coupled (incompressible) family ---
        if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::PREPARE_COUPLED)) {
            let coupled = init
                .coupled
                .expect("ModelKernelsModule: coupled kernels require CoupledSolverResources");
            let pipeline_prepare = &pipelines[&KernelPipeline::Kernel(KernelId::PREPARE_COUPLED)];

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
                Some(coupled.bg_coupled_solution.clone())
            } else {
                None
            };

            return Self {
                state_step_index,
                pipelines,
                bg_mesh: Some(bg_mesh),
                bg_fields_ping_pong,
                bg_solver: Some(coupled.bg_solver.clone()),
                bg_bc: None,
                bg_bc_group2: None,
                bg_fields0_ping_pong: None,
                bg_solution: None,
                bg_update_fields_ping_pong,
                bg_update_solution,
            };
        }

        // --- Conservative (legacy compressible / EI) family ---
        if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::CONSERVATIVE_ASSEMBLY)) {
            let port_space = init
                .port_space
                .expect("ModelKernelsModule: conservative kernels require PortSpace");
            let system_ports = init
                .system_ports
                .expect("ModelKernelsModule: conservative kernels require LinearSystemPorts");
            let scalar_row_offsets = init
                .scalar_row_offsets
                .expect("ModelKernelsModule: conservative kernels require scalar_row_offsets port");
            let bc_kind = init
                .bc_kind
                .expect("ModelKernelsModule: conservative kernels require bc_kind");
            let bc_value = init
                .bc_value
                .expect("ModelKernelsModule: conservative kernels require bc_value");

            let anchor_mesh_fields = [
                KernelId::CONSERVATIVE_ASSEMBLY,
                KernelId::CONSERVATIVE_GRADIENTS,
                KernelId::CONSERVATIVE_FLUX_KT,
                KernelId::CONSERVATIVE_EXPLICIT_UPDATE,
            ]
            .into_iter()
            .find(|id| pipelines.contains_key(&KernelPipeline::Kernel(*id)))
            .unwrap_or(KernelId::CONSERVATIVE_ASSEMBLY);

            let anchor_source = kernel_registry::kernel_source_by_id(model_id, anchor_mesh_fields)
                .unwrap_or_else(|err| {
                    panic!("missing kernel source for {anchor_mesh_fields:?}: {err}")
                });
            let anchor_pipeline = &pipelines[&KernelPipeline::Kernel(anchor_mesh_fields)];

            let bg_mesh = {
                let bgl = anchor_pipeline.get_bind_group_layout(0);
                crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    "ModelKernels: conservative mesh bind group",
                    &bgl,
                    anchor_source.bindings,
                    0,
                    |name| {
                        mesh.buffer_for_binding_name(name).map(|buf| {
                            wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding())
                        })
                    },
                )
                .unwrap_or_else(|err| {
                    panic!("ModelKernels conservative mesh bind group build failed: {err}")
                })
            };

            let bg_fields_ping_pong = {
                let bgl = anchor_pipeline.get_bind_group_layout(1);
                let mut out = Vec::with_capacity(3);
                for i in 0..3 {
                    let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        &format!("ModelKernels: conservative fields (g1) bind group {i}"),
                        &bgl,
                        anchor_source.bindings,
                        1,
                        |name| field_binding(fields, name, i),
                    )
                    .unwrap_or_else(|err| {
                        panic!("ModelKernels conservative fields(g1) bind group build failed: {err}")
                    });
                    out.push(bg);
                }
                out
            };

            let assembly_source =
                kernel_registry::kernel_source_by_id(model_id, KernelId::CONSERVATIVE_ASSEMBLY)
                    .unwrap_or_else(|err| {
                        panic!("missing kernel source for conservative assembly: {err}")
                    });
            let pipeline_assembly =
                &pipelines[&KernelPipeline::Kernel(KernelId::CONSERVATIVE_ASSEMBLY)];

            let bg_solver = {
                let bgl = pipeline_assembly.get_bind_group_layout(2);
                crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    "ModelKernels: conservative solver bind group",
                    &bgl,
                    assembly_source.bindings,
                    2,
                    |name| match name {
                        "matrix_values" => Some(wgpu::BindingResource::Buffer(
                            port_space
                                .buffer(system_ports.values)
                                .as_entire_buffer_binding(),
                        )),
                        "rhs" => Some(wgpu::BindingResource::Buffer(
                            port_space.buffer(system_ports.rhs).as_entire_buffer_binding(),
                        )),
                        "scalar_row_offsets" => Some(wgpu::BindingResource::Buffer(
                            port_space
                                .buffer(scalar_row_offsets)
                                .as_entire_buffer_binding(),
                        )),
                        _ => None,
                    },
                )
                .unwrap_or_else(|err| {
                    panic!("ModelKernels conservative solver bind group build failed: {err}")
                })
            };

            // Some conservative kernels (notably KT flux) use a separate BC bind group at group=2.
            let bg_bc_group2 = if pipelines.contains_key(&KernelPipeline::Kernel(
                KernelId::CONSERVATIVE_FLUX_KT,
            )) {
                let flux_source =
                    kernel_registry::kernel_source_by_id(model_id, KernelId::CONSERVATIVE_FLUX_KT)
                        .unwrap_or_else(|err| {
                            panic!("missing kernel source for conservative flux_kt: {err}")
                        });
                let pipeline_flux =
                    &pipelines[&KernelPipeline::Kernel(KernelId::CONSERVATIVE_FLUX_KT)];
                let bgl = pipeline_flux.get_bind_group_layout(2);
                Some(
                    crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        "ModelKernels: conservative bc bind group (group2)",
                        &bgl,
                        flux_source.bindings,
                        2,
                        |name| match name {
                            "bc_kind" => Some(wgpu::BindingResource::Buffer(
                                bc_kind.as_entire_buffer_binding(),
                            )),
                            "bc_value" => Some(wgpu::BindingResource::Buffer(
                                bc_value.as_entire_buffer_binding(),
                            )),
                            _ => None,
                        },
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "ModelKernels conservative bc bind group(group2) build failed: {err}"
                        )
                    }),
                )
            } else {
                None
            };

            let bg_bc = {
                let bgl = pipeline_assembly.get_bind_group_layout(3);
                crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    "ModelKernels: conservative bc bind group",
                    &bgl,
                    assembly_source.bindings,
                    3,
                    |name| match name {
                        "bc_kind" => Some(wgpu::BindingResource::Buffer(
                            bc_kind.as_entire_buffer_binding(),
                        )),
                        "bc_value" => Some(wgpu::BindingResource::Buffer(
                            bc_value.as_entire_buffer_binding(),
                        )),
                        _ => None,
                    },
                )
                .unwrap_or_else(|err| {
                    panic!("ModelKernels conservative bc bind group build failed: {err}")
                })
            };

            let fields0_anchor = [KernelId::CONSERVATIVE_UPDATE, KernelId::CONSERVATIVE_APPLY]
                .into_iter()
                .find(|id| pipelines.contains_key(&KernelPipeline::Kernel(*id)))
                .unwrap_or(KernelId::CONSERVATIVE_UPDATE);

            let fields0_source =
                kernel_registry::kernel_source_by_id(model_id, fields0_anchor)
                    .unwrap_or_else(|err| panic!("missing kernel source for {fields0_anchor:?}: {err}"));
            let fields0_pipeline = &pipelines[&KernelPipeline::Kernel(fields0_anchor)];

            let bg_fields0_ping_pong = {
                let bgl = fields0_pipeline.get_bind_group_layout(0);
                let mut out = Vec::with_capacity(3);
                for i in 0..3 {
                    let bg = crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        &format!("ModelKernels: conservative fields (g0) bind group {i}"),
                        &bgl,
                        fields0_source.bindings,
                        0,
                        |name| field_binding(fields, name, i),
                    )
                    .unwrap_or_else(|err| {
                        panic!("ModelKernels conservative fields(g0) bind group build failed: {err}")
                    });
                    out.push(bg);
                }
                out
            };

            let bg_solution = if pipelines.contains_key(&KernelPipeline::Kernel(KernelId::CONSERVATIVE_APPLY))
            {
                let apply_source =
                    kernel_registry::kernel_source_by_id(model_id, KernelId::CONSERVATIVE_APPLY)
                        .unwrap_or_else(|err| {
                            panic!("missing kernel source for conservative apply: {err}")
                        });
                let pipeline_apply =
                    &pipelines[&KernelPipeline::Kernel(KernelId::CONSERVATIVE_APPLY)];
                let bgl = pipeline_apply.get_bind_group_layout(1);
                Some(
                    crate::solver::gpu::wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        "ModelKernels: conservative apply solution bind group",
                        &bgl,
                        apply_source.bindings,
                        1,
                        |name| match name {
                            "solution" => Some(wgpu::BindingResource::Buffer(
                                port_space
                                    .buffer(system_ports.x)
                                    .as_entire_buffer_binding(),
                            )),
                            _ => None,
                        },
                    )
                    .unwrap_or_else(|err| {
                        panic!("ModelKernels conservative apply solution bind group build failed: {err}")
                    }),
                )
            } else {
                None
            };

            return Self {
                state_step_index,
                pipelines,
                bg_mesh: Some(bg_mesh),
                bg_fields_ping_pong,
                bg_solver: Some(bg_solver),
                bg_bc: Some(bg_bc),
                bg_bc_group2,
                bg_fields0_ping_pong: Some(bg_fields0_ping_pong),
                bg_solution,
                bg_update_fields_ping_pong: None,
                bg_update_solution: None,
            };
        }

        panic!("ModelKernelsModule: unsupported kernel set (no coupled or conservative anchor kernel found)");
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

    fn bg_fields0(&self) -> &wgpu::BindGroup {
        let idx = self.step_index();
        &self
            .bg_fields0_ping_pong
            .as_ref()
            .expect("missing conservative fields0 bind groups")[idx]
    }

    fn bg_bc(&self) -> &wgpu::BindGroup {
        self.bg_bc.as_ref().expect("missing bc bind group")
    }

    fn bg_bc_group2(&self) -> &wgpu::BindGroup {
        self.bg_bc_group2
            .as_ref()
            .expect("missing bc bind group (group2)")
    }

    fn bg_solution(&self) -> &wgpu::BindGroup {
        self.bg_solution
            .as_ref()
            .expect("missing solution bind group")
    }

    fn bg_solver(&self) -> &wgpu::BindGroup {
        self.bg_solver.as_ref().expect("missing solver bind group")
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

    fn bind_conservative_mesh_fields(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh(), &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
    }

    fn bind_conservative_mesh_fields_bc(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh(), &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
        pass.set_bind_group(2, self.bg_bc_group2(), &[]);
    }

    fn bind_conservative_mesh_fields_solver_bc(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh(), &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
        pass.set_bind_group(2, self.bg_solver(), &[]);
        pass.set_bind_group(3, self.bg_bc(), &[]);
    }

    fn bind_conservative_fields_only(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_fields0(), &[]);
    }

    fn bind_conservative_fields_solution(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_fields0(), &[]);
        pass.set_bind_group(1, self.bg_solution(), &[]);
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
            KernelBindGroups::UpdateFieldsSolution => self.bind_update_fields_solution(pass),

            KernelBindGroups::ConservativeMeshFields => self.bind_conservative_mesh_fields(pass),
            KernelBindGroups::ConservativeMeshFieldsBc => {
                self.bind_conservative_mesh_fields_bc(pass)
            }
            KernelBindGroups::ConservativeMeshFieldsSolverBc => {
                self.bind_conservative_mesh_fields_solver_bc(pass)
            }
            KernelBindGroups::ConservativeFieldsOnly => self.bind_conservative_fields_only(pass),
            KernelBindGroups::ConservativeFieldsSolution => {
                self.bind_conservative_fields_solution(pass)
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
            // Incompressible kernels
            KernelId::PREPARE_COUPLED => Some(KernelBindGroups::MeshFieldsSolver),
            KernelId::COUPLED_ASSEMBLY => Some(KernelBindGroups::MeshFieldsSolver),
            KernelId::UPDATE_FIELDS_FROM_COUPLED => Some(KernelBindGroups::UpdateFieldsSolution),
            // Shared kernels
            KernelId::FLUX_RHIE_CHOW => Some(KernelBindGroups::MeshFields),

            // Legacy compressible (EI) kernels
            KernelId::CONSERVATIVE_GRADIENTS
            | KernelId::CONSERVATIVE_EXPLICIT_UPDATE => Some(KernelBindGroups::ConservativeMeshFields),

            KernelId::CONSERVATIVE_FLUX_KT => Some(KernelBindGroups::ConservativeMeshFieldsBc),

            KernelId::CONSERVATIVE_ASSEMBLY => {
                Some(KernelBindGroups::ConservativeMeshFieldsSolverBc)
            }
            KernelId::CONSERVATIVE_APPLY => Some(KernelBindGroups::ConservativeFieldsSolution),
            KernelId::CONSERVATIVE_UPDATE => Some(KernelBindGroups::ConservativeFieldsOnly),

            // Unknown or unsupported kernels
            _ => None,
        }
    }
}
