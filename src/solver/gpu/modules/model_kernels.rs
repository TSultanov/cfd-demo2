use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::field_provider::FieldProvider;
use crate::solver::gpu::modules::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
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
}

pub struct ModelKernelsInit<'a> {
    pub coupled: Option<&'a CoupledSolverResources>,
}

impl<'a> ModelKernelsInit<'a> {
    pub fn coupled(coupled: &'a CoupledSolverResources) -> Self {
        Self {
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
                bg_update_fields_ping_pong,
                bg_update_solution,
            };
        }

        panic!("ModelKernelsModule: unsupported kernel set (no coupled anchor kernel found)");
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

            // Unknown or unsupported kernels
            _ => None,
        }
    }
}
