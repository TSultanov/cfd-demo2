use super::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use super::unified_graph::UnifiedGraphModule;
use crate::solver::model::KernelId;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum GenericCoupledPipeline {
    FluxRhieChow,
    KtGradients,
    FluxKt,
    Assembly,
    Update,
}

#[derive(Clone, Copy, Debug)]
pub enum GenericCoupledBindGroups {
    FluxRhieChow,
    KtGradients,
    FluxKt,
    Assembly,
    Update,
}

pub struct GenericCoupledKernelsModule {
    state_step_index: Arc<AtomicUsize>,

    bg_mesh_flux: Option<wgpu::BindGroup>,
    bg_fields_flux_ping_pong: Option<Vec<wgpu::BindGroup>>,

    bg_mesh_kt: Option<wgpu::BindGroup>,
    bg_fields_kt_ping_pong: Option<Vec<wgpu::BindGroup>>,
    bg_bc_kt: Option<wgpu::BindGroup>,

    bg_mesh: wgpu::BindGroup,
    bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_solver: wgpu::BindGroup,
    bg_bc: wgpu::BindGroup,

    bg_update_state_ping_pong: Vec<wgpu::BindGroup>,
    bg_update_solution: wgpu::BindGroup,

    pipeline_flux: Option<wgpu::ComputePipeline>,
    pipeline_kt_gradients: Option<wgpu::ComputePipeline>,
    pipeline_flux_kt: Option<wgpu::ComputePipeline>,
    pipeline_assembly: wgpu::ComputePipeline,
    pipeline_update: wgpu::ComputePipeline,
}

impl GenericCoupledKernelsModule {
    pub fn new(
        state_step_index: Arc<AtomicUsize>,
        bg_mesh_flux: Option<wgpu::BindGroup>,
        bg_fields_flux_ping_pong: Option<Vec<wgpu::BindGroup>>,
        bg_mesh_kt: Option<wgpu::BindGroup>,
        bg_fields_kt_ping_pong: Option<Vec<wgpu::BindGroup>>,
        bg_bc_kt: Option<wgpu::BindGroup>,
        bg_mesh: wgpu::BindGroup,
        bg_fields_ping_pong: Vec<wgpu::BindGroup>,
        bg_solver: wgpu::BindGroup,
        bg_bc: wgpu::BindGroup,
        bg_update_state_ping_pong: Vec<wgpu::BindGroup>,
        bg_update_solution: wgpu::BindGroup,
        pipeline_flux: Option<wgpu::ComputePipeline>,
        pipeline_kt_gradients: Option<wgpu::ComputePipeline>,
        pipeline_flux_kt: Option<wgpu::ComputePipeline>,
        pipeline_assembly: wgpu::ComputePipeline,
        pipeline_update: wgpu::ComputePipeline,
    ) -> Self {
        Self {
            state_step_index,
            bg_mesh_flux,
            bg_fields_flux_ping_pong,
            bg_mesh_kt,
            bg_fields_kt_ping_pong,
            bg_bc_kt,
            bg_mesh,
            bg_fields_ping_pong,
            bg_solver,
            bg_bc,
            bg_update_state_ping_pong,
            bg_update_solution,
            pipeline_flux,
            pipeline_kt_gradients,
            pipeline_flux_kt,
            pipeline_assembly,
            pipeline_update,
        }
    }

    pub fn step_index(&self) -> usize {
        self.state_step_index.load(Ordering::Relaxed) % 3
    }

    pub fn set_step_index(&mut self, idx: usize) {
        self.state_step_index.store(idx % 3, Ordering::Relaxed);
    }

    fn bg_fields(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self.bg_fields_ping_pong[idx]
    }

    fn bg_mesh_flux(&self) -> &wgpu::BindGroup {
        self.bg_mesh_flux
            .as_ref()
            .expect("missing flux mesh bind group")
    }

    fn bg_fields_flux(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self
            .bg_fields_flux_ping_pong
            .as_ref()
            .expect("missing flux fields bind groups")[idx]
    }

    fn bg_mesh_kt(&self) -> &wgpu::BindGroup {
        self.bg_mesh_kt.as_ref().expect("missing KT mesh bind group")
    }

    fn bg_fields_kt(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self
            .bg_fields_kt_ping_pong
            .as_ref()
            .expect("missing KT fields bind groups")[idx]
    }

    fn bg_bc_kt(&self) -> &wgpu::BindGroup {
        self.bg_bc_kt.as_ref().expect("missing KT BC bind group")
    }

    fn bg_update_state(&self) -> &wgpu::BindGroup {
        let idx = self.state_step_index.load(Ordering::Relaxed) % 3;
        &self.bg_update_state_ping_pong[idx]
    }

    fn bind_assembly(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, &self.bg_mesh, &[]);
        pass.set_bind_group(1, self.bg_fields(), &[]);
        pass.set_bind_group(2, &self.bg_solver, &[]);
        pass.set_bind_group(3, &self.bg_bc, &[]);
    }

    fn bind_flux_rhie_chow(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh_flux(), &[]);
        pass.set_bind_group(1, self.bg_fields_flux(), &[]);
    }

    fn bind_kt_gradients(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh_kt(), &[]);
        pass.set_bind_group(1, self.bg_fields_kt(), &[]);
    }

    fn bind_flux_kt(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_mesh_kt(), &[]);
        pass.set_bind_group(1, self.bg_fields_kt(), &[]);
        pass.set_bind_group(2, self.bg_bc_kt(), &[]);
    }

    fn bind_update(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, self.bg_update_state(), &[]);
        pass.set_bind_group(1, &self.bg_update_solution, &[]);
    }
}

impl GpuComputeModule for GenericCoupledKernelsModule {
    type PipelineKey = GenericCoupledPipeline;
    type BindKey = GenericCoupledBindGroups;

    fn pipeline(&self, key: Self::PipelineKey) -> &wgpu::ComputePipeline {
        match key {
            GenericCoupledPipeline::FluxRhieChow => self
                .pipeline_flux
                .as_ref()
                .expect("missing flux pipeline"),
            GenericCoupledPipeline::KtGradients => self
                .pipeline_kt_gradients
                .as_ref()
                .expect("missing KT gradients pipeline"),
            GenericCoupledPipeline::FluxKt => self
                .pipeline_flux_kt
                .as_ref()
                .expect("missing KT flux pipeline"),
            GenericCoupledPipeline::Assembly => &self.pipeline_assembly,
            GenericCoupledPipeline::Update => &self.pipeline_update,
        }
    }

    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass) {
        match key {
            GenericCoupledBindGroups::FluxRhieChow => self.bind_flux_rhie_chow(pass),
            GenericCoupledBindGroups::KtGradients => self.bind_kt_gradients(pass),
            GenericCoupledBindGroups::FluxKt => self.bind_flux_kt(pass),
            GenericCoupledBindGroups::Assembly => self.bind_assembly(pass),
            GenericCoupledBindGroups::Update => self.bind_update(pass),
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

impl UnifiedGraphModule for GenericCoupledKernelsModule {
    fn pipeline_for_kernel(&self, id: KernelId) -> Option<Self::PipelineKey> {
        match id {
            KernelId::FLUX_RHIE_CHOW => self
                .pipeline_flux
                .as_ref()
                .map(|_| GenericCoupledPipeline::FluxRhieChow),
            KernelId::KT_GRADIENTS => self
                .pipeline_kt_gradients
                .as_ref()
                .map(|_| GenericCoupledPipeline::KtGradients),
            KernelId::FLUX_KT => self
                .pipeline_flux_kt
                .as_ref()
                .map(|_| GenericCoupledPipeline::FluxKt),
            KernelId::GENERIC_COUPLED_ASSEMBLY => Some(GenericCoupledPipeline::Assembly),
            KernelId::GENERIC_COUPLED_UPDATE => Some(GenericCoupledPipeline::Update),
            _ => None,
        }
    }

    fn bind_for_kernel(&self, id: KernelId) -> Option<Self::BindKey> {
        match id {
            KernelId::FLUX_RHIE_CHOW => self
                .pipeline_flux
                .as_ref()
                .map(|_| GenericCoupledBindGroups::FluxRhieChow),
            KernelId::KT_GRADIENTS => self
                .pipeline_kt_gradients
                .as_ref()
                .map(|_| GenericCoupledBindGroups::KtGradients),
            KernelId::FLUX_KT => self
                .pipeline_flux_kt
                .as_ref()
                .map(|_| GenericCoupledBindGroups::FluxKt),
            KernelId::GENERIC_COUPLED_ASSEMBLY => Some(GenericCoupledBindGroups::Assembly),
            KernelId::GENERIC_COUPLED_UPDATE => Some(GenericCoupledBindGroups::Update),
            _ => None,
        }
    }
}
