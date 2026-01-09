use crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace;
use crate::solver::gpu::linear_solver::fgmres::{
    dispatch_2d, dispatch_x_threads, workgroups_for_size,
};

/// Dispatch grid dimensions in workgroups.
#[derive(Clone, Copy, Debug)]
pub struct DispatchGrids {
    pub dofs: (u32, u32),
    pub cells: (u32, u32),
}

#[derive(Clone, Copy, Debug)]
pub struct KrylovDispatch {
    pub grids: DispatchGrids,
    pub dofs_dispatch_x_threads: u32,
    pub dof_groups: u32,
    pub cell_groups: u32,
}

impl DispatchGrids {
    pub fn for_sizes(num_dofs: u32, num_cells: u32) -> KrylovDispatch {
        let dof_groups = workgroups_for_size(num_dofs);
        let cell_groups = workgroups_for_size(num_cells);
        KrylovDispatch {
            grids: DispatchGrids {
                dofs: dispatch_2d(dof_groups),
                cells: dispatch_2d(cell_groups),
            },
            dofs_dispatch_x_threads: dispatch_x_threads(dof_groups),
            dof_groups,
            cell_groups,
        }
    }
}

/// A preconditioner module that can be plugged into the GPU FGMRES loop.
///
/// The module owns any GPU resources it needs (buffers/pipelines/bind-groups) and
/// records work into the caller-provided encoder. The only cross-module coupling
/// should be via the public `FgmresWorkspace` interface.
pub trait FgmresPreconditionerModule {
    fn prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _fgmres: &FgmresWorkspace,
        _rhs: wgpu::BindingResource<'_>,
        _dispatch: DispatchGrids,
    ) {
    }

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
