use crate::solver::gpu::linear_solver::fgmres::{
    dispatch_2d, dispatch_x_threads, workgroups_for_size,
};
use crate::solver::gpu::wgsl_reflect::{self, WgslBindingDesc};

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
    /// Bind group layout for group-0 vector bindings (`vec_x`, `vec_y`, `vec_z`).
    pub vectors_layout: &'a wgpu::BindGroupLayout,
    /// Binding descriptors for group-0 vector bindings.
    pub(crate) vector_bindings: &'static [WgslBindingDesc],
    /// Dispatch grid dimensions.
    pub dispatch: DispatchGrids,
    /// Total number of DOFs in the linear system.
    pub num_dofs: u32,
}

impl<'a> PrecondContext<'a> {
    /// Byte offset within [`Self::indirect_args`] for a DOF-count indirect dispatch.
    pub const INDIRECT_DISPATCH_DOFS_OFFSET: u64 = 0;
    /// Byte offset within [`Self::indirect_args`] for a cell-count indirect dispatch.
    pub const INDIRECT_DISPATCH_CELLS_OFFSET: u64 = 16;

    /// Create a group-0 vector bind group binding `vec_x`, `vec_y`, `vec_z`.
    ///
    /// This is the solver-agnostic replacement for
    /// `FgmresWorkspace::create_vector_bind_group`.
    pub fn create_vector_bind_group(
        &self,
        device: &wgpu::Device,
        x: wgpu::BindingResource<'a>,
        y: wgpu::BindingResource<'a>,
        z: wgpu::BindingResource<'a>,
        label: &str,
    ) -> wgpu::BindGroup {
        wgsl_reflect::create_bind_group_from_bindings(
            device,
            label,
            self.vectors_layout,
            self.vector_bindings,
            0,
            |name| match name {
                "vec_x" => Some(x.clone()),
                "vec_y" => Some(y.clone()),
                "vec_z" => Some(z.clone()),
                _ => None,
            },
        )
        .unwrap_or_else(|err| panic!("{label} creation failed: {err}"))
    }
}

/// Solver-agnostic preconditioner trait.
///
/// This trait does not reference the FGMRES workspace — it operates on the
/// generic [`PrecondContext`] instead, making preconditioner implementations
/// reusable across different Krylov solvers (FGMRES, CG, BiCGSTAB, etc.).
///
/// # Contract
///
/// **Inputs / outputs.** The solver passes `input` and `output` as
/// [`wgpu::BindingResource::Buffer`] sub-ranges of at least
/// `ctx.num_dofs * 4` bytes.  The preconditioner must read from `input`
/// and write the result `M⁻¹ · input` into `output`.  The solver
/// guarantees that `input` and `output` do not alias (they point to
/// different buffer regions).
///
/// **Scratch buffers.** The preconditioner may use `ctx.scratch_a`,
/// `ctx.scratch_b`, and `ctx.scratch_c` as temporary storage.  These are
/// each at least `ctx.num_dofs * 4` bytes.  Their contents are undefined
/// on entry and may be clobbered freely.  The preconditioner must **not**
/// read from or write to the solver's internal buffers (basis vectors,
/// Hessenberg matrix, Givens rotations, etc.).
///
/// **Bind groups.** The context provides `matrix_bg` (group 1),
/// `precond_bg` (group 2), and `params_bg` (group 3) for convenience.
/// These correspond to the standard FGMRES shader layout.  A
/// preconditioner may use them directly, or create its own pipelines
/// and bind groups from the raw buffers accessible via the context.
///
/// **Idempotency.** `encode_apply` may be called multiple times per solve
/// (once per Krylov iteration).  Implementations must not accumulate
/// state across calls unless explicitly reset in `encode_prepare`.
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
