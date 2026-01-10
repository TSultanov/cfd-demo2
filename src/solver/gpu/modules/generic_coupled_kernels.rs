use super::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use super::unified_graph::UnifiedGraphModule;
use crate::solver::model::KernelId;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum GenericCoupledPipeline {
    Assembly,
    Update,
}

#[derive(Clone, Copy, Debug)]
pub enum GenericCoupledBindGroups {
    Assembly,
    Update,
}

pub struct GenericCoupledKernelsModule {
    state_step_index: Arc<AtomicUsize>,

    bg_mesh: wgpu::BindGroup,
    bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_solver: wgpu::BindGroup,
    bg_bc: wgpu::BindGroup,

    bg_update_state_ping_pong: Vec<wgpu::BindGroup>,
    bg_update_solution: wgpu::BindGroup,

    pipeline_assembly: wgpu::ComputePipeline,
    pipeline_update: wgpu::ComputePipeline,
}

impl GenericCoupledKernelsModule {
    pub fn new(
        state_step_index: Arc<AtomicUsize>,
        bg_mesh: wgpu::BindGroup,
        bg_fields_ping_pong: Vec<wgpu::BindGroup>,
        bg_solver: wgpu::BindGroup,
        bg_bc: wgpu::BindGroup,
        bg_update_state_ping_pong: Vec<wgpu::BindGroup>,
        bg_update_solution: wgpu::BindGroup,
        pipeline_assembly: wgpu::ComputePipeline,
        pipeline_update: wgpu::ComputePipeline,
    ) -> Self {
        Self {
            state_step_index,
            bg_mesh,
            bg_fields_ping_pong,
            bg_solver,
            bg_bc,
            bg_update_state_ping_pong,
            bg_update_solution,
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
            GenericCoupledPipeline::Assembly => &self.pipeline_assembly,
            GenericCoupledPipeline::Update => &self.pipeline_update,
        }
    }

    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass) {
        match key {
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
            KernelId::GENERIC_COUPLED_ASSEMBLY => Some(GenericCoupledPipeline::Assembly),
            KernelId::GENERIC_COUPLED_UPDATE => Some(GenericCoupledPipeline::Update),
            _ => None,
        }
    }

    fn bind_for_kernel(&self, id: KernelId) -> Option<Self::BindKey> {
        match id {
            KernelId::GENERIC_COUPLED_ASSEMBLY => Some(GenericCoupledBindGroups::Assembly),
            KernelId::GENERIC_COUPLED_UPDATE => Some(GenericCoupledBindGroups::Update),
            _ => None,
        }
    }
}
