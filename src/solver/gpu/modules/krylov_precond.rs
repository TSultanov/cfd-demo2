use crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace;

/// Dispatch grid dimensions in workgroups.
#[derive(Clone, Copy, Debug)]
pub struct DispatchGrids {
    pub dofs: (u32, u32),
    pub cells: (u32, u32),
}

/// A preconditioner module that can be plugged into the GPU FGMRES loop.
///
/// The module owns any GPU resources it needs (buffers/pipelines/bind-groups) and
/// records work into the caller-provided encoder. The only cross-module coupling
/// should be via the public `FgmresWorkspace` interface.
pub trait FgmresPreconditionerModule {
    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        input: wgpu::BindingResource<'_>,
        output: &wgpu::Buffer,
        dispatch: DispatchGrids,
    );
}

