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
    fn encode_prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        rhs: wgpu::BindingResource<'_>,
        dispatch: DispatchGrids,
    ) {
        self.prepare(device, queue, fgmres, rhs, dispatch);
    }

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
        output: wgpu::BindingResource<'_>,
        dispatch: DispatchGrids,
    );
}

// ---------------------------------------------------------------------------
// Solver-agnostic preconditioner abstraction (ARCH_FIX_6)
// ---------------------------------------------------------------------------

/// Solver-agnostic context passed to preconditioners.
///
/// Bundles the shared GPU resources that any linear solver (FGMRES, CG,
/// BiCGSTAB, …) would provide to its preconditioner.  This decouples
/// preconditioner implementations from [`FgmresWorkspace`] internals.
pub struct PrecondContext<'a> {
    /// CSR matrix bind group (group 1 in the standard layout).
    pub matrix_bg: &'a wgpu::BindGroup,
    /// Diagonal preconditioner data bind group (group 2).
    pub precond_bg: &'a wgpu::BindGroup,
    /// Solver params bind group (group 3).
    pub params_bg: &'a wgpu::BindGroup,
    /// Indirect dispatch buffer for variable-size dispatches.
    pub indirect_args: &'a wgpu::Buffer,
    /// Solver control scalars buffer (convergence flags, residual estimates, etc.).
    pub scalars_buffer: &'a wgpu::Buffer,
    /// Scratch buffer A (at least `num_dofs * 4` bytes; STORAGE | COPY_DST | COPY_SRC).
    ///
    /// In the FGMRES workspace this corresponds to the `w` buffer.
    pub scratch_a: &'a wgpu::Buffer,
    /// Scratch buffer B (at least `num_dofs * 4` bytes; STORAGE | COPY_DST | COPY_SRC).
    ///
    /// In the FGMRES workspace this corresponds to the `temp` buffer.
    pub scratch_b: &'a wgpu::Buffer,
    /// Scratch binding C — a buffer binding sub-range of at least `num_dofs * 4` bytes.
    ///
    /// In the FGMRES workspace this corresponds to `z_binding(0)`.
    pub scratch_c: wgpu::BindingResource<'a>,
    /// Dispatch grid dimensions.
    pub dispatch: DispatchGrids,
    /// Total number of DOFs in the linear system.
    pub num_dofs: u32,
}

impl PrecondContext<'_> {
    /// Byte offset within [`Self::indirect_args`] for a DOF-count indirect dispatch.
    pub const INDIRECT_DISPATCH_DOFS_OFFSET: u64 = 0;
    /// Byte offset within [`Self::indirect_args`] for a cell-count indirect dispatch.
    pub const INDIRECT_DISPATCH_CELLS_OFFSET: u64 = 16;
}

/// Solver-agnostic preconditioner trait.
///
/// Unlike [`FgmresPreconditionerModule`], this trait does not reference
/// [`FgmresWorkspace`] — it operates on the generic [`PrecondContext`] instead,
/// making preconditioner implementations reusable across different Krylov
/// solvers (FGMRES, CG, BiCGSTAB, etc.).
pub trait PreconditionerModule {
    /// Encode any per-solve preparation work (e.g. diagonal extraction, AMG setup).
    ///
    /// Called once before the iterative solve loop begins.  The default
    /// implementation is a no-op.
    fn encode_prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _ctx: &PrecondContext<'_>,
        _rhs: wgpu::BindingResource<'_>,
    ) {
        // Default: no-op.
    }

    /// Encode the preconditioner application: `output ← M⁻¹ · input`.
    ///
    /// Called once per Krylov iteration.  `input` and `output` are buffer
    /// binding sub-ranges of at least `ctx.num_dofs * 4` bytes.
    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PrecondContext<'_>,
        input: wgpu::BindingResource<'_>,
        output: wgpu::BindingResource<'_>,
    );
}

/// Adapter that wraps a [`PreconditionerModule`] so it can be used where
/// [`FgmresPreconditionerModule`] is expected.
///
/// This enables incremental migration: new preconditioners implement the
/// solver-agnostic [`PreconditionerModule`] trait, and this adapter bridges
/// them into the existing FGMRES infrastructure without requiring changes
/// to [`KrylovSolveModule`] or calling code.
pub struct PrecondAdapter<P>(pub P);

impl<P: PreconditionerModule> FgmresPreconditionerModule for PrecondAdapter<P> {
    fn encode_prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        rhs: wgpu::BindingResource<'_>,
        dispatch: DispatchGrids,
    ) {
        let ctx = fgmres.precond_context(dispatch);
        self.0.encode_prepare(device, queue, encoder, &ctx, rhs);
    }

    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        input: wgpu::BindingResource<'_>,
        output: wgpu::BindingResource<'_>,
        dispatch: DispatchGrids,
    ) {
        let ctx = fgmres.precond_context(dispatch);
        self.0.encode_apply(device, encoder, &ctx, input, output);
    }
}
