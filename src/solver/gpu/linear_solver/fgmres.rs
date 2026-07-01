use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, PrecondContext};
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::linear_solver::FgmresSolutionUpdateStrategy;
use crate::solver::model::KernelId;
use bytemuck::{bytes_of, Pod, Zeroable};

/// Default workgroup size used by GPU linear solver kernels.
///
/// Can be overridden at construction time via [`FgmresWorkspace::new_from_system`]
/// or [`ScalarCgModule`] to match device-specific optimal sizes.
pub const DEFAULT_WORKGROUP_SIZE: u32 = 64;
pub const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;

pub(crate) const FGMRES_SCALAR_COUNT: usize = 24;
pub(crate) const FGMRES_SCALAR_STOP: usize = 8;
pub(crate) const FGMRES_SCALAR_CONVERGED: usize = 9;
const FGMRES_SCALAR_ITERS_USED: usize = 10;
pub(crate) const FGMRES_SCALAR_RESIDUAL_EST: usize = 11;
const FGMRES_SCALAR_TOL_REL_RHS: usize = 12;
const FGMRES_SCALAR_TOL_ABS: usize = 13;
const FGMRES_SCALAR_RHS_NORM: usize = 14;
const FGMRES_SCALAR_SKIP_UPDATE: usize = 15;
// Restart-boundary monotonicity guard slots (see gmres_logic/restart_guard).
// 16: best true residual so far (0.0 = unset); 17: guard action flag
// (0 none / 1 snapshot / 2 restore).
#[allow(dead_code)]
const FGMRES_SCALAR_BEST_RESID: usize = 16;
#[allow(dead_code)]
const FGMRES_SCALAR_GUARD_FLAG: usize = 17;
// Stall-stop slots (see the stall branch in gmres_logic/restart_guard).
// 18: residual at the previous guard checkpoint; 19: host-written level
// factor (stop only when residual <= STALL_REL * ||b||; 0.0 disables);
// 20: consecutive no-improvement checkpoint count.
#[allow(dead_code)]
const FGMRES_SCALAR_PREV_RESID: usize = 18;
const FGMRES_SCALAR_STALL_REL: usize = 19;
#[allow(dead_code)]
const FGMRES_SCALAR_STALL_COUNT: usize = 20;
// Mid-cycle stall slots (see the stall branch in
// gmres_logic/update_hessenberg_givens): 21 = previous Givens residual
// estimate; 22 = consecutive low-improvement iteration count.
#[allow(dead_code)]
const FGMRES_SCALAR_PREV_EST: usize = 21;
#[allow(dead_code)]
const FGMRES_SCALAR_STALL_COUNT_ITER: usize = 22;
// 23: actual Arnoldi iterations executed this solve (incremented once per
// update_hessenberg_givens past the STOP early-out; zeroed by the
// non-preserve init). Feeds the adaptive iteration budget.
pub(crate) const FGMRES_SCALAR_TOTAL_ITERS: usize = 23;
// STOP flag readback for the adaptive budget (stopped-early detection).
pub(crate) const FGMRES_SCALAR_STOP_PUB: usize = FGMRES_SCALAR_STOP;

const FGMRES_INDIRECT_DISPATCH_COUNT: usize = 3;
const FGMRES_INDIRECT_ENTRY_STRIDE_BYTES: u64 = 16;
const FGMRES_INDIRECT_DOFS_OFFSET: u64 = 0;
const FGMRES_INDIRECT_CELLS_OFFSET: u64 = FGMRES_INDIRECT_ENTRY_STRIDE_BYTES;
const FGMRES_INDIRECT_SCALAR_OFFSET: u64 = FGMRES_INDIRECT_ENTRY_STRIDE_BYTES * 2;

const FGMRES_PARAMS_STRIDE_BYTES: u64 = std::mem::size_of::<RawFgmresParams>() as u64;
const FGMRES_ITER_PARAMS_STRIDE_BYTES: u64 = std::mem::size_of::<IterParams>() as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RawFgmresParams {
    pub n: u32,
    pub num_cells: u32,
    pub num_iters: u32,
    pub omega: f32,
    /// Width of 2D dispatch (in threads, i.e. workgroups_x * 64).
    pub dispatch_x: u32,
    pub max_restart: u32,
    pub column_offset: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct IterParams {
    pub current_idx: u32,
    pub max_restart: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub struct FgmresCore<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,

    pub n: u32,
    pub num_cells: u32,
    pub max_restart: usize,
    pub num_dot_groups: u32,
    pub basis_stride: u64,
    pub z_stride: u64,
    pub solution_update_strategy: FgmresSolutionUpdateStrategy,

    pub b_basis: &'a wgpu::Buffer,
    pub b_z_storage: &'a wgpu::Buffer,
    pub b_w: &'a wgpu::Buffer,
    pub b_temp: &'a wgpu::Buffer,
    pub b_dot_partial: &'a wgpu::Buffer,
    pub b_scalars: &'a wgpu::Buffer,
    pub b_indirect_args: &'a wgpu::Buffer,
    pub b_params: &'a wgpu::Buffer,
    pub b_iter_params: &'a wgpu::Buffer,
    /// Dedicated Reduce-Final params/iter-params (see `bg_params_reduce`).
    pub b_params_reduce: &'a wgpu::Buffer,
    pub b_iter_params_hess: &'a wgpu::Buffer,
    pub b_params_table_iter: &'a wgpu::Buffer,
    pub b_params_table_reduce: &'a wgpu::Buffer,
    pub b_iter_table_j: &'a wgpu::Buffer,
    pub b_iter_table_hessenberg: &'a wgpu::Buffer,
    pub b_hessenberg: &'a wgpu::Buffer,
    pub b_givens: &'a wgpu::Buffer,
    pub b_g: &'a wgpu::Buffer,
    pub b_y: &'a wgpu::Buffer,
    pub b_staging_scalar: &'a wgpu::Buffer,
    /// Tiny scratch buffer (4 bytes) for intra-`b_scalars` copies that cannot use
    /// `copy_buffer_to_buffer` with source == destination (WebGPU forbids same-buffer copies).
    pub b_scalar_copy_staging: &'a wgpu::Buffer,
    /// Best-so-far solution snapshot for the restart-boundary monotonicity guard.
    pub b_x_snapshot: &'a wgpu::Buffer,

    pub bg_matrix: &'a wgpu::BindGroup,
    pub bg_precond: &'a wgpu::BindGroup,
    pub bg_params: &'a wgpu::BindGroup,
    /// Reduce-Final group-3 bind group (dedicated reduce/hessenberg buffers).
    pub bg_params_reduce: &'a wgpu::BindGroup,
    pub bg_logic: &'a wgpu::BindGroup,
    pub bg_logic_params: &'a wgpu::BindGroup,
    pub bg_cgs: &'a wgpu::BindGroup,

    pub bgl_vectors: &'a wgpu::BindGroupLayout,
    vector_bindings: &'static [wgsl_reflect::WgslBindingDesc],

    /// Precomputed per-Arnoldi-column vector bind groups (see [`VectorBgCache`]).
    /// The inner restart loop indexes these by column `j` instead of rebuilding
    /// one bind group per iteration — the CPU hot spot of the coupled solve.
    spmv_bgs: &'a [wgpu::BindGroup],
    scale_bgs: &'a [wgpu::BindGroup],
    norm_bg: &'a wgpu::BindGroup,
    reduce_bg: &'a wgpu::BindGroup,

    pub pipeline_spmv: &'a wgpu::ComputePipeline,
    pub pipeline_axpby: &'a wgpu::ComputePipeline,
    pub pipeline_scale: &'a wgpu::ComputePipeline,
    pub pipeline_scale_in_place: &'a wgpu::ComputePipeline,
    pub pipeline_norm_sq: &'a wgpu::ComputePipeline,
    pub pipeline_reduce_final_and_finish_norm: &'a wgpu::ComputePipeline,
    pub pipeline_update_hessenberg: &'a wgpu::ComputePipeline,
    pub pipeline_solve_triangular: &'a wgpu::ComputePipeline,
    pub pipeline_axpy_fused_from_y: &'a wgpu::ComputePipeline,
    pub pipeline_calc_dots_cgs: &'a wgpu::ComputePipeline,
    pub pipeline_reduce_dots_cgs: &'a wgpu::ComputePipeline,
    pub pipeline_update_w_cgs: &'a wgpu::ComputePipeline,
    pub pipeline_reduce_dots_cgs_reortho: &'a wgpu::ComputePipeline,
    pub pipeline_update_w_cgs_reortho: &'a wgpu::ComputePipeline,
    pub pipeline_restart_guard: &'a wgpu::ComputePipeline,
    pub pipeline_guard_copy: &'a wgpu::ComputePipeline,
    pub pipeline_clamp_rel_scale: &'a wgpu::ComputePipeline,
}

pub struct FgmresWorkspace {
    max_restart: usize,
    n: u32,
    num_cells: u32,
    num_dot_groups: u32,
    basis_stride: u64,
    z_stride: u64,
    solution_update_strategy: FgmresSolutionUpdateStrategy,

    b_basis: wgpu::Buffer,
    b_z_storage: wgpu::Buffer,
    b_w: wgpu::Buffer,
    b_temp: wgpu::Buffer,
    /// Best-so-far solution snapshot for the restart-boundary monotonicity
    /// guard (see `snapshot_x` / `restore_x`).
    b_x_snapshot: wgpu::Buffer,
    b_dot_partial: wgpu::Buffer,
    b_scalars: wgpu::Buffer,
    b_indirect_args: wgpu::Buffer,
    b_params: wgpu::Buffer,
    b_iter_params: wgpu::Buffer,
    /// Dedicated params/iter-params for the Reduce-Final pass, so the inner loop
    /// never swaps `b_params`/`b_iter_params` mid-iteration (see `bg_params_reduce`).
    b_params_reduce: wgpu::Buffer,
    b_iter_params_hess: wgpu::Buffer,
    b_params_table_iter: wgpu::Buffer,
    b_params_table_reduce: wgpu::Buffer,
    b_iter_table_j: wgpu::Buffer,
    b_iter_table_hessenberg: wgpu::Buffer,
    b_hessenberg: wgpu::Buffer,
    b_givens: wgpu::Buffer,
    b_g: wgpu::Buffer,
    b_y: wgpu::Buffer,
    b_staging_scalar: wgpu::Buffer,
    /// Tiny scratch buffer (4 bytes) for intra-`b_scalars` copies via two-hop
    /// (WebGPU forbids `copy_buffer_to_buffer` with same source and destination).
    b_scalar_copy_staging: wgpu::Buffer,

    bgl_vectors: wgpu::BindGroupLayout,
    vector_bindings: &'static [wgsl_reflect::WgslBindingDesc],
    /// Per-Arnoldi-column vector bind groups, built once at construction and
    /// reused by every solve (see [`VectorBgCache`]).
    vector_bg_cache: VectorBgCache,
    bgl_matrix: wgpu::BindGroupLayout,
    bgl_precond: wgpu::BindGroupLayout,
    bgl_params: wgpu::BindGroupLayout,

    bg_matrix: wgpu::BindGroup,
    bg_precond: wgpu::BindGroup,
    bg_params: wgpu::BindGroup,
    /// Group-3 bind group for the Reduce-Final pass, binding the dedicated
    /// `b_params_reduce` / `b_iter_params_hess` (same layout as `bg_params`).
    bg_params_reduce: wgpu::BindGroup,
    bg_logic: wgpu::BindGroup,
    bg_logic_params: wgpu::BindGroup,
    bg_cgs: wgpu::BindGroup,

    pipeline_spmv: wgpu::ComputePipeline,
    pipeline_axpby: wgpu::ComputePipeline,
    pipeline_scale: wgpu::ComputePipeline,
    pipeline_scale_in_place: wgpu::ComputePipeline,
    pipeline_copy: wgpu::ComputePipeline,
    pipeline_norm_sq: wgpu::ComputePipeline,
    pipeline_reduce_final: wgpu::ComputePipeline,
    pipeline_reduce_final_and_finish_norm: wgpu::ComputePipeline,
    pipeline_update_hessenberg: wgpu::ComputePipeline,
    pipeline_solve_triangular: wgpu::ComputePipeline,
    pipeline_calc_dots_cgs: wgpu::ComputePipeline,
    pipeline_reduce_dots_cgs: wgpu::ComputePipeline,
    pipeline_update_w_cgs: wgpu::ComputePipeline,
    pipeline_reduce_dots_cgs_reortho: wgpu::ComputePipeline,
    pipeline_update_w_cgs_reortho: wgpu::ComputePipeline,
    pipeline_axpy_fused_from_y: wgpu::ComputePipeline,
    pipeline_restart_guard: wgpu::ComputePipeline,
    pipeline_guard_copy: wgpu::ComputePipeline,
    pipeline_clamp_rel_scale: wgpu::ComputePipeline,
}

impl FgmresWorkspace {
    /// Build a preconditioner bind group (group 2) compatible with the FGMRES
    /// ops shader layout.
    ///
    /// This is a convenience helper so callers don't need to look up the shader
    /// bindings themselves.  The `resolve` callback maps WGSL binding names
    /// (e.g. `"diag_u"`, `"diag_v"`, `"diag_p"`) to buffer bindings.
    ///
    /// # Example
    /// ```ignore
    /// let bg = FgmresWorkspace::build_precond_bind_group(device, "my solver", |name| {
    ///     match name {
    ///         "diag_u" => Some(buf_u.as_entire_binding()),
    ///         "diag_v" => Some(buf_v.as_entire_binding()),
    ///         "diag_p" => Some(buf_p.as_entire_binding()),
    ///         _ => None,
    ///     }
    /// });
    /// ```
    pub fn build_precond_bind_group<'a>(
        device: &wgpu::Device,
        label: &str,
        resolve: impl FnMut(&str) -> Option<wgpu::BindingResource<'a>>,
    ) -> Result<wgpu::BindGroup, String> {
        let ops_src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_SPMV)
            .map_err(|e| format!("gmres_ops/spmv shader missing: {e}"))?;
        let pipeline = (ops_src.create_pipeline)(device);
        let bgl_precond = pipeline.get_bind_group_layout(2);
        wgsl_reflect::create_bind_group_from_bindings(
            device,
            label,
            &bgl_precond,
            ops_src.bindings,
            2,
            resolve,
        )
        .map_err(|e| format!("{label} creation failed: {e}"))
    }

    /// Create a new FGMRES workspace.
    ///
    /// `precond_bind_group` is the caller-built bind group for group 2
    /// (the preconditioner diagonal/parameter buffers).  This keeps the
    /// FGMRES solver agnostic of which physics-specific buffers are
    /// bound — the caller decides the layout.
    pub fn new_from_system(
        device: &wgpu::Device,
        n: u32,
        num_cells: u32,
        max_restart: usize,
        solution_update_strategy: FgmresSolutionUpdateStrategy,
        system: LinearSystemView<'_>,
        precond_bind_group: wgpu::BindGroup,
        label_prefix: &str,
    ) -> Result<Self, String> {
        let matrix_row_offsets = system.row_offsets();
        let matrix_col_indices = system.col_indices();
        let matrix_values = system.values();

        let num_dot_groups = workgroups_for_size(n);

        let min_alignment = 256u64;
        let basis_stride_unaligned = (n as u64) * 4;
        let basis_stride = (basis_stride_unaligned + min_alignment - 1) & !(min_alignment - 1);
        let basis_size = basis_stride * (max_restart as u64 + 1);
        let z_stride_unaligned = (n as u64) * 4;
        let z_stride = (z_stride_unaligned + min_alignment - 1) & !(min_alignment - 1);

        let b_basis = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES basis")),
            size: basis_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_z_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES Z storage")),
            size: z_stride * (max_restart as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES w")),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES temp")),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_x_snapshot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES x snapshot")),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_dot_partial = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES dot partial")),
            size: (num_dot_groups as u64) * ((max_restart + 1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_scalars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES scalars")),
            size: (FGMRES_SCALAR_COUNT as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_indirect_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES indirect args")),
            size: (FGMRES_INDIRECT_DISPATCH_COUNT as u64) * FGMRES_INDIRECT_ENTRY_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES params")),
            size: std::mem::size_of::<RawFgmresParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_iter_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES iter params")),
            size: std::mem::size_of::<IterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dedicated per-iteration param buffers for the Reduce-Final pass, so the
        // inner loop never has to swap b_params (iter <-> reduce) or b_iter_params
        // (j <-> hessenberg-index) mid-iteration. Holding each distinct value in
        // its own buffer lets all four table-select copies batch at the iteration
        // top (one blit section instead of three) — see the inner loop and the
        // `bg_params_reduce` bind group below.
        let b_params_reduce = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES params (reduce)")),
            size: std::mem::size_of::<RawFgmresParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_iter_params_hess = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES iter params (hessenberg)")),
            size: std::mem::size_of::<IterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let table_capacity = max_restart.max(1) as u64;
        let b_params_table_iter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES params table iter")),
            size: table_capacity * FGMRES_PARAMS_STRIDE_BYTES,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_params_table_reduce = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES params table reduce")),
            size: table_capacity * FGMRES_PARAMS_STRIDE_BYTES,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_iter_table_j = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES iter table j")),
            size: table_capacity * FGMRES_ITER_PARAMS_STRIDE_BYTES,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_iter_table_hessenberg = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES iter table hessenberg")),
            size: table_capacity * FGMRES_ITER_PARAMS_STRIDE_BYTES,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hessenberg_len = (max_restart + 1) * max_restart;
        let b_hessenberg = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES hessenberg")),
            size: (hessenberg_len as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_givens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES givens")),
            size: (max_restart as u64) * 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_g = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES g")),
            size: ((max_restart + 1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES y")),
            size: (max_restart as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_staging_scalar = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES staging scalar")),
            size: (FGMRES_SCALAR_COUNT as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_scalar_copy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES scalar copy staging")),
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ops_spmv_src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_SPMV)
            .map_err(|e| format!("gmres_ops/spmv shader missing: {e}"))?;
        let ops_bindings = ops_spmv_src.bindings;

        let pipeline_spmv = (ops_spmv_src.create_pipeline)(device);
        let pipeline_axpy_fused_from_y = {
            let src = kernel_registry::kernel_source_by_id(
                "",
                KernelId("gmres_update_fused/accumulate_solution"),
            )
            .map_err(|e| format!("gmres_update_fused/accumulate_solution shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_axpby = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_AXPBY)
                .map_err(|e| format!("gmres_ops/axpby shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_scale = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_SCALE)
                .map_err(|e| format!("gmres_ops/scale shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_scale_in_place = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_SCALE_IN_PLACE)
                .map_err(|e| format!("gmres_ops/scale_in_place shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_copy = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_COPY)
                .map_err(|e| format!("gmres_ops/copy shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_norm_sq = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_NORM_SQ_PARTIAL)
                .map_err(|e| format!("gmres_ops/norm_sq_partial shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_reduce_final = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_REDUCE_FINAL)
                .map_err(|e| format!("gmres_ops/reduce_final shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_reduce_final_and_finish_norm = {
            let src = kernel_registry::kernel_source_by_id(
                "",
                KernelId::GMRES_OPS_REDUCE_FINAL_AND_FINISH_NORM,
            )
            .map_err(|e| format!("gmres_ops/reduce_final_and_finish_norm shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_guard_copy = {
            let src = kernel_registry::kernel_source_by_id("", KernelId("gmres_ops/guard_copy"))
                .map_err(|e| format!("gmres_ops/guard_copy shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };

        let bgl_vectors = pipeline_spmv.get_bind_group_layout(0);
        let bgl_matrix = pipeline_spmv.get_bind_group_layout(1);
        let bgl_precond = pipeline_spmv.get_bind_group_layout(2);
        let bgl_params = pipeline_spmv.get_bind_group_layout(3);

        let bg_matrix = {
            let registry = ResourceRegistry::new()
                .with_buffer("row_offsets", matrix_row_offsets)
                .with_buffer("col_indices", matrix_col_indices)
                .with_buffer("matrix_values", matrix_values);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES matrix BG"),
                &bgl_matrix,
                ops_bindings,
                1,
                |name| registry.resolve(name),
            )
            .map_err(|e| format!("FGMRES matrix BG creation failed: {e}"))?
        };

        let bg_precond = precond_bind_group;

        let bg_params = {
            let registry = ResourceRegistry::new()
                .with_buffer("params", &b_params)
                .with_buffer("scalars", &b_scalars)
                .with_buffer("iter_params", &b_iter_params)
                .with_buffer("hessenberg", &b_hessenberg)
                .with_buffer("y_sol", &b_y);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES params BG"),
                &bgl_params,
                ops_bindings,
                3,
                |name| registry.resolve(name),
            )
            .map_err(|e| format!("FGMRES params BG creation failed: {e}"))?
        };

        // Same group-3 layout as `bg_params`, but with the params/iter_params
        // uniforms bound to the dedicated reduce/hessenberg buffers. Only the
        // Reduce-Final pass uses it (see the inner loop); scalars/hessenberg/y_sol
        // are bound to the identical buffers so it still writes hessenberg and reads
        // scalars exactly as `bg_params` does.
        let bg_params_reduce = {
            let registry = ResourceRegistry::new()
                .with_buffer("params", &b_params_reduce)
                .with_buffer("scalars", &b_scalars)
                .with_buffer("iter_params", &b_iter_params_hess)
                .with_buffer("hessenberg", &b_hessenberg)
                .with_buffer("y_sol", &b_y);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES params BG (reduce)"),
                &bgl_params,
                ops_bindings,
                3,
                |name| registry.resolve(name),
            )
            .map_err(|e| format!("FGMRES params (reduce) BG creation failed: {e}"))?
        };

        let logic_update_src = kernel_registry::kernel_source_by_id(
            "",
            KernelId::GMRES_LOGIC_UPDATE_HESSENBERG_GIVENS,
        )
        .map_err(|e| format!("gmres_logic/update_hessenberg_givens shader missing: {e}"))?;
        let pipeline_update_hessenberg = (logic_update_src.create_pipeline)(device);
        let pipeline_solve_triangular = {
            let src =
                kernel_registry::kernel_source_by_id("", KernelId::GMRES_LOGIC_SOLVE_TRIANGULAR)
                    .map_err(|e| format!("gmres_logic/solve_triangular shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_restart_guard = {
            let src =
                kernel_registry::kernel_source_by_id("", KernelId("gmres_logic/restart_guard"))
                    .map_err(|e| format!("gmres_logic/restart_guard shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_clamp_rel_scale = {
            let src =
                kernel_registry::kernel_source_by_id("", KernelId("gmres_logic/clamp_rel_scale"))
                    .map_err(|e| format!("gmres_logic/clamp_rel_scale shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };

        let bgl_logic = pipeline_update_hessenberg.get_bind_group_layout(0);
        let bgl_logic_params = pipeline_update_hessenberg.get_bind_group_layout(1);

        let bg_logic = {
            let registry = ResourceRegistry::new()
                .with_buffer("hessenberg", &b_hessenberg)
                .with_buffer("givens", &b_givens)
                .with_buffer("g_rhs", &b_g)
                .with_buffer("y_sol", &b_y);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES logic BG"),
                &bgl_logic,
                logic_update_src.bindings,
                0,
                |name| registry.resolve(name),
            )
            .map_err(|e| format!("FGMRES logic BG creation failed: {e}"))?
        };

        let bg_logic_params = {
            let registry = ResourceRegistry::new()
                .with_buffer("iter_params", &b_iter_params)
                .with_buffer("scalars", &b_scalars)
                .with_buffer("indirect_args", &b_indirect_args);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES logic params BG"),
                &bgl_logic_params,
                logic_update_src.bindings,
                1,
                |name| registry.resolve(name),
            )
            .map_err(|e| format!("FGMRES logic params BG creation failed: {e}"))?
        };

        let cgs_calc_src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_CGS_CALC_DOTS)
            .map_err(|e| format!("gmres_cgs/calc_dots_cgs shader missing: {e}"))?;
        let pipeline_calc_dots_cgs = (cgs_calc_src.create_pipeline)(device);
        let pipeline_reduce_dots_cgs = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_CGS_REDUCE_DOTS)
                .map_err(|e| format!("gmres_cgs/reduce_dots_cgs shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_update_w_cgs = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_CGS_UPDATE_W)
                .map_err(|e| format!("gmres_cgs/update_w_cgs shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_reduce_dots_cgs_reortho = {
            let src =
                kernel_registry::kernel_source_by_id("", KernelId("gmres_cgs/reduce_dots_cgs_reortho"))
                    .map_err(|e| format!("gmres_cgs/reduce_dots_cgs_reortho shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };
        let pipeline_update_w_cgs_reortho = {
            let src =
                kernel_registry::kernel_source_by_id("", KernelId("gmres_cgs/update_w_cgs_reortho"))
                    .map_err(|e| format!("gmres_cgs/update_w_cgs_reortho shader missing: {e}"))?;
            (src.create_pipeline)(device)
        };

        let bgl_cgs = pipeline_calc_dots_cgs.get_bind_group_layout(0);
        let bg_cgs = {
            let registry = ResourceRegistry::new()
                .with_buffer("params", &b_params)
                .with_buffer("b_basis", &b_basis)
                .with_buffer("b_w", &b_w)
                .with_buffer("b_dot_partial", &b_dot_partial)
                .with_buffer("b_hessenberg", &b_hessenberg)
                .with_buffer("scalars", &b_scalars);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES cgs BG"),
                &bgl_cgs,
                cgs_calc_src.bindings,
                0,
                |name| registry.resolve(name),
            )
            .map_err(|e| format!("FGMRES cgs BG creation failed: {e}"))?
        };

        // Precompute the per-column vector bind groups once (see `VectorBgCache`),
        // so the inner restart loop never rebuilds them.
        let vector_bg_cache = build_vector_bg_cache(
            device,
            &bgl_vectors,
            ops_bindings,
            &b_basis,
            &b_z_storage,
            &b_w,
            &b_temp,
            &b_dot_partial,
            basis_stride,
            z_stride,
            n,
            max_restart,
        );

        Ok(Self {
            max_restart,
            n,
            num_cells,
            num_dot_groups,
            basis_stride,
            z_stride,
            solution_update_strategy,
            b_basis,
            b_z_storage,
            b_w,
            b_temp,
            b_dot_partial,
            b_scalars,
            b_indirect_args,
            b_params,
            b_iter_params,
            b_params_reduce,
            b_iter_params_hess,
            b_params_table_iter,
            b_params_table_reduce,
            b_iter_table_j,
            b_iter_table_hessenberg,
            b_hessenberg,
            b_givens,
            b_g,
            b_y,
            b_staging_scalar,
            b_scalar_copy_staging,
            b_x_snapshot,
            bgl_vectors,
            vector_bindings: ops_bindings,
            vector_bg_cache,
            bgl_matrix,
            bgl_precond,
            bgl_params,
            bg_matrix,
            bg_precond,
            bg_params,
            bg_params_reduce,
            bg_logic,
            bg_logic_params,
            bg_cgs,
            pipeline_spmv,
            pipeline_axpy_fused_from_y,
            pipeline_axpby,
            pipeline_scale,
            pipeline_scale_in_place,
            pipeline_copy,
            pipeline_norm_sq,
            pipeline_reduce_final,
            pipeline_reduce_final_and_finish_norm,
            pipeline_update_hessenberg,
            pipeline_solve_triangular,
            pipeline_calc_dots_cgs,
            pipeline_reduce_dots_cgs,
            pipeline_update_w_cgs,
            pipeline_reduce_dots_cgs_reortho,
            pipeline_update_w_cgs_reortho,
            pipeline_restart_guard,
            pipeline_guard_copy,
            pipeline_clamp_rel_scale,
        })
    }

    pub fn core<'a>(&'a self, device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> FgmresCore<'a> {
        FgmresCore {
            device,
            queue,
            n: self.n,
            num_cells: self.num_cells,
            max_restart: self.max_restart,
            num_dot_groups: self.num_dot_groups,
            basis_stride: self.basis_stride,
            z_stride: self.z_stride,
            solution_update_strategy: self.solution_update_strategy,
            b_basis: &self.b_basis,
            b_z_storage: &self.b_z_storage,
            b_w: &self.b_w,
            b_temp: &self.b_temp,
            b_dot_partial: &self.b_dot_partial,
            b_scalars: &self.b_scalars,
            b_indirect_args: &self.b_indirect_args,
            b_params: &self.b_params,
            b_iter_params: &self.b_iter_params,
            b_params_reduce: &self.b_params_reduce,
            b_iter_params_hess: &self.b_iter_params_hess,
            b_params_table_iter: &self.b_params_table_iter,
            b_params_table_reduce: &self.b_params_table_reduce,
            b_iter_table_j: &self.b_iter_table_j,
            b_iter_table_hessenberg: &self.b_iter_table_hessenberg,
            b_hessenberg: &self.b_hessenberg,
            b_givens: &self.b_givens,
            b_g: &self.b_g,
            b_y: &self.b_y,
            b_staging_scalar: &self.b_staging_scalar,
            b_scalar_copy_staging: &self.b_scalar_copy_staging,
            b_x_snapshot: &self.b_x_snapshot,
            bg_matrix: &self.bg_matrix,
            bg_precond: &self.bg_precond,
            bg_params: &self.bg_params,
            bg_params_reduce: &self.bg_params_reduce,
            bg_logic: &self.bg_logic,
            bg_logic_params: &self.bg_logic_params,
            bg_cgs: &self.bg_cgs,
            bgl_vectors: &self.bgl_vectors,
            vector_bindings: self.vector_bindings,
            spmv_bgs: &self.vector_bg_cache.spmv,
            scale_bgs: &self.vector_bg_cache.scale,
            norm_bg: &self.vector_bg_cache.norm,
            reduce_bg: &self.vector_bg_cache.reduce,
            pipeline_spmv: &self.pipeline_spmv,
            pipeline_axpby: &self.pipeline_axpby,
            pipeline_scale: &self.pipeline_scale,
            pipeline_scale_in_place: &self.pipeline_scale_in_place,
            pipeline_norm_sq: &self.pipeline_norm_sq,
            pipeline_reduce_final_and_finish_norm: &self.pipeline_reduce_final_and_finish_norm,
            pipeline_update_hessenberg: &self.pipeline_update_hessenberg,
            pipeline_solve_triangular: &self.pipeline_solve_triangular,
            pipeline_axpy_fused_from_y: &self.pipeline_axpy_fused_from_y,
            pipeline_calc_dots_cgs: &self.pipeline_calc_dots_cgs,
            pipeline_reduce_dots_cgs: &self.pipeline_reduce_dots_cgs,
            pipeline_update_w_cgs: &self.pipeline_update_w_cgs,
            pipeline_reduce_dots_cgs_reortho: &self.pipeline_reduce_dots_cgs_reortho,
            pipeline_update_w_cgs_reortho: &self.pipeline_update_w_cgs_reortho,
            pipeline_restart_guard: &self.pipeline_restart_guard,
            pipeline_guard_copy: &self.pipeline_guard_copy,
            pipeline_clamp_rel_scale: &self.pipeline_clamp_rel_scale,
        }
    }

    pub fn clear_restart_aux(&self, core: &FgmresCore<'_>) {
        write_zeros(core, self.hessenberg_buffer());
        write_zeros(core, self.givens_buffer());
        write_zeros(core, self.y_buffer());
    }

    pub fn write_g0(&self, queue: &wgpu::Queue, g0: f32) {
        let mut g = vec![0.0_f32; self.max_restart + 1];
        g[0] = g0;
        queue.write_buffer(self.g_buffer(), 0, bytemuck::cast_slice(&g));
    }

    pub fn init_basis0_from_vector_normalized<'src>(
        &self,
        core: &FgmresCore<'_>,
        src: wgpu::BindingResource<'src>,
        inv_norm: f32,
        label_prefix: &str,
    ) {
        let workgroups = workgroups_for_size(self.n);
        let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);

        let basis0 = self.basis_binding(0);
        let copy_bg = self.create_vector_bind_group(
            core.device,
            src,
            basis0,
            self.temp_buffer().as_entire_binding(),
            &format!("{label_prefix} basis0 copy BG"),
        );
        dispatch_vector_pipeline(
            core,
            self.pipeline_copy(),
            &copy_bg,
            dispatch_x,
            dispatch_y,
            &format!("{label_prefix} basis0 copy"),
        );

        self.scale_in_place(
            core,
            self.basis_binding(0),
            inv_norm,
            &format!("{label_prefix} basis0 normalize"),
        );
    }

    pub fn scale_in_place<'a>(
        &'a self,
        core: &FgmresCore<'a>,
        y: wgpu::BindingResource<'a>,
        scalar: f32,
        label: &str,
    ) {
        let workgroups = workgroups_for_size(self.n);
        let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);

        write_scalars(core, &[scalar]);
        let bg = self.create_vector_bind_group(
            core.device,
            self.w_buffer().as_entire_binding(),
            y,
            self.temp_buffer().as_entire_binding(),
            &format!("{label} BG"),
        );
        dispatch_vector_pipeline(
            core,
            self.pipeline_scale_in_place(),
            &bg,
            dispatch_x,
            dispatch_y,
            label,
        );
    }

    /// Copy the current solution `x` into the internal snapshot buffer.
    ///
    /// Together with [`Self::restore_x`] this implements the restart-boundary
    /// monotonicity guard: f32 Arnoldi can lose orthogonality on hard
    /// preconditioned systems, and a restart cycle may then APPLY a solution
    /// update that increases the true residual (observed June 2026 on the
    /// coupled incompressible system: residual growing across restarts by
    /// orders of magnitude, ending in NaN). The host restart loop snapshots
    /// the best-so-far `x` and restores it when a cycle made things worse.
    pub fn snapshot_x<'a>(&'a self, core: &FgmresCore<'a>, x: &'a wgpu::Buffer, label: &str) {
        let workgroups = workgroups_for_size(self.n);
        let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
        let bg = self.create_vector_bind_group(
            core.device,
            x.as_entire_binding(),
            self.b_x_snapshot.as_entire_binding(),
            self.temp_buffer().as_entire_binding(),
            &format!("{label} BG"),
        );
        dispatch_vector_pipeline(core, self.pipeline_copy(), &bg, dispatch_x, dispatch_y, label);
    }

    /// Restore the solution `x` from the snapshot taken by [`Self::snapshot_x`].
    pub fn restore_x<'a>(&'a self, core: &FgmresCore<'a>, x: &'a wgpu::Buffer, label: &str) {
        let workgroups = workgroups_for_size(self.n);
        let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
        let bg = self.create_vector_bind_group(
            core.device,
            self.b_x_snapshot.as_entire_binding(),
            x.as_entire_binding(),
            self.temp_buffer().as_entire_binding(),
            &format!("{label} BG"),
        );
        dispatch_vector_pipeline(core, self.pipeline_copy(), &bg, dispatch_x, dispatch_y, label);
    }

    pub fn compute_residual_norm_into<'a>(
        &'a self,
        core: &FgmresCore<'a>,
        system: LinearSystemView<'a>,
        target: wgpu::BindingResource<'a>,
        label_prefix: &str,
    ) -> f32 {
        debug_assert_eq!(core.n, self.n, "FGMRES residual expects n == self.n");

        let workgroups = workgroups_for_size(self.n);
        let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);

        // w = A * x
        let spmv_bg = self.create_vector_bind_group(
            core.device,
            system.x().as_entire_binding(),
            self.w_buffer().as_entire_binding(),
            self.temp_buffer().as_entire_binding(),
            &format!("{label_prefix} residual spmv BG"),
        );
        dispatch_vector_pipeline(
            core,
            self.pipeline_spmv(),
            &spmv_bg,
            dispatch_x,
            dispatch_y,
            &format!("{label_prefix} residual spmv"),
        );

        // target = rhs - w
        write_scalars(core, &[1.0, -1.0]);
        let residual_bg = self.create_vector_bind_group(
            core.device,
            system.rhs().as_entire_binding(),
            self.w_buffer().as_entire_binding(),
            target.clone(),
            &format!("{label_prefix} residual axpby BG"),
        );
        dispatch_vector_pipeline(
            core,
            self.pipeline_axpby(),
            &residual_bg,
            dispatch_x,
            dispatch_y,
            &format!("{label_prefix} residual axpby"),
        );

        self.gpu_norm(core.device, core.queue, target, self.n)
    }

    pub fn max_restart(&self) -> usize {
        self.max_restart
    }

    pub fn n(&self) -> u32 {
        self.n
    }

    pub fn num_dot_groups(&self) -> u32 {
        self.num_dot_groups
    }

    pub fn basis_stride(&self) -> u64 {
        self.basis_stride
    }

    pub fn vector_bytes(&self) -> u64 {
        (self.n as u64) * 4
    }

    pub fn basis_binding(&self, idx: usize) -> wgpu::BindingResource<'_> {
        basis_binding(&self.b_basis, self.basis_stride, self.vector_bytes(), idx)
    }

    pub fn basis_buffer(&self) -> &wgpu::Buffer {
        &self.b_basis
    }

    pub fn z_stride(&self) -> u64 {
        self.z_stride
    }

    pub fn z_storage_buffer(&self) -> &wgpu::Buffer {
        &self.b_z_storage
    }

    pub fn z_binding(&self, idx: usize) -> wgpu::BindingResource<'_> {
        z_storage_binding(&self.b_z_storage, self.z_stride, self.vector_bytes(), idx)
    }

    pub fn solution_update_strategy(&self) -> FgmresSolutionUpdateStrategy {
        self.solution_update_strategy
    }

    pub fn set_solution_update_strategy(&mut self, strategy: FgmresSolutionUpdateStrategy) {
        self.solution_update_strategy = strategy;
    }

    pub fn w_buffer(&self) -> &wgpu::Buffer {
        &self.b_w
    }

    pub fn temp_buffer(&self) -> &wgpu::Buffer {
        &self.b_temp
    }

    pub fn dot_partial_buffer(&self) -> &wgpu::Buffer {
        &self.b_dot_partial
    }

    pub fn scalars_buffer(&self) -> &wgpu::Buffer {
        &self.b_scalars
    }

    pub fn indirect_args_buffer(&self) -> &wgpu::Buffer {
        &self.b_indirect_args
    }

    pub const fn indirect_dispatch_dofs_offset() -> u64 {
        FGMRES_INDIRECT_DOFS_OFFSET
    }

    pub const fn indirect_dispatch_cells_offset() -> u64 {
        FGMRES_INDIRECT_CELLS_OFFSET
    }

    pub const fn indirect_dispatch_scalar_offset() -> u64 {
        FGMRES_INDIRECT_SCALAR_OFFSET
    }

    pub fn params_buffer(&self) -> &wgpu::Buffer {
        &self.b_params
    }

    pub fn iter_params_buffer(&self) -> &wgpu::Buffer {
        &self.b_iter_params
    }

    pub fn hessenberg_buffer(&self) -> &wgpu::Buffer {
        &self.b_hessenberg
    }

    pub fn givens_buffer(&self) -> &wgpu::Buffer {
        &self.b_givens
    }

    pub fn g_buffer(&self) -> &wgpu::Buffer {
        &self.b_g
    }

    pub fn y_buffer(&self) -> &wgpu::Buffer {
        &self.b_y
    }

    pub fn staging_scalar_buffer(&self) -> &wgpu::Buffer {
        &self.b_staging_scalar
    }

    pub fn vectors_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl_vectors
    }

    pub fn matrix_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl_matrix
    }

    pub fn precond_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl_precond
    }

    pub fn params_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl_params
    }

    pub fn matrix_bg(&self) -> &wgpu::BindGroup {
        &self.bg_matrix
    }

    pub fn precond_bg(&self) -> &wgpu::BindGroup {
        &self.bg_precond
    }

    pub fn params_bg(&self) -> &wgpu::BindGroup {
        &self.bg_params
    }

    /// Build a solver-agnostic [`PrecondContext`] from this workspace.
    ///
    /// This bundles the shared GPU resources (matrix/precond/params bind groups,
    /// indirect dispatch buffer, scratch buffers) into a struct that
    /// preconditioners can use without depending on `FgmresWorkspace` internals.
    pub fn precond_context(&self, dispatch: DispatchGrids) -> PrecondContext<'_> {
        PrecondContext {
            matrix_bg: &self.bg_matrix,
            precond_bg: &self.bg_precond,
            params_bg: &self.bg_params,
            indirect_args: &self.b_indirect_args,
            scalars_buffer: &self.b_scalars,
            scratch_a: &self.b_w,
            scratch_b: &self.b_temp,
            scratch_c: self.z_binding(0),
            vectors_layout: &self.bgl_vectors,
            vector_bindings: self.vector_bindings,
            dispatch,
            num_dofs: self.n,
        }
    }

    pub fn logic_bg(&self) -> &wgpu::BindGroup {
        &self.bg_logic
    }

    pub fn logic_params_bg(&self) -> &wgpu::BindGroup {
        &self.bg_logic_params
    }

    pub fn cgs_bg(&self) -> &wgpu::BindGroup {
        &self.bg_cgs
    }

    pub fn pipeline_spmv(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_spmv
    }

    pub fn pipeline_axpy_fused_from_y(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_axpy_fused_from_y
    }

    pub fn pipeline_axpby(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_axpby
    }

    pub fn pipeline_scale(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_scale
    }

    pub fn pipeline_scale_in_place(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_scale_in_place
    }

    pub fn pipeline_copy(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_copy
    }

    pub fn pipeline_norm_sq(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_norm_sq
    }

    pub fn pipeline_reduce_final(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_reduce_final
    }

    pub fn pipeline_reduce_final_and_finish_norm(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_reduce_final_and_finish_norm
    }

    pub fn pipeline_calc_dots_cgs(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_calc_dots_cgs
    }

    pub fn pipeline_reduce_dots_cgs(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_reduce_dots_cgs
    }

    pub fn pipeline_update_w_cgs(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_update_w_cgs
    }

    pub fn create_vector_bind_group<'a>(
        &self,
        device: &wgpu::Device,
        x: wgpu::BindingResource<'a>,
        y: wgpu::BindingResource<'a>,
        z: wgpu::BindingResource<'a>,
        label: &str,
    ) -> wgpu::BindGroup {
        create_vector_bind_group(
            device,
            &self.bgl_vectors,
            self.vector_bindings,
            x,
            y,
            z,
            label,
        )
    }

    pub fn gpu_norm<'a>(
        &'a self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        x: wgpu::BindingResource<'a>,
        n: u32,
    ) -> f32 {
        debug_assert_eq!(n, self.n, "FgmresWorkspace::gpu_norm expects n == self.n");

        let core = self.core(device, queue);
        let workgroups = workgroups_for_size(n);
        let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
        let dispatch_x_threads = dispatch_x_threads(workgroups);

        let vector_bg = create_vector_bind_group(
            device,
            self.vectors_layout(),
            self.vector_bindings,
            x,
            self.temp_buffer().as_entire_binding(),
            self.dot_partial_buffer().as_entire_binding(),
            "FGMRES norm_sq vector BG",
        );

        let reduce_bg = create_vector_bind_group(
            device,
            self.vectors_layout(),
            self.vector_bindings,
            self.dot_partial_buffer().as_entire_binding(),
            self.temp_buffer().as_entire_binding(),
            self.temp_buffer().as_entire_binding(),
            "FGMRES norm_sq reduce BG",
        );

        // Pass 1: partial reduction
        let partial_params = RawFgmresParams {
            n,
            num_cells: self.num_cells,
            num_iters: 0,
            omega: 1.0,
            dispatch_x: dispatch_x_threads,
            max_restart: self.max_restart as u32,
            column_offset: 0,
            _pad3: 0,
        };
        write_params(&core, &partial_params);

        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FGMRES norm_sq partial"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES norm_sq partial"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.pipeline_norm_sq());
                pass.set_bind_group(0, &vector_bg, &[]);
                pass.set_bind_group(1, self.matrix_bg(), &[]);
                pass.set_bind_group(2, self.precond_bg(), &[]);
                pass.set_bind_group(3, self.params_bg(), &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            queue.submit(Some(encoder.finish()));
            crate::count_submission!("FGMRES", "norm_sq_partial");
        }
        crate::count_dispatch!("FGMRES", "norm_sq_partial");

        // Pass 2: final reduction + staging copy
        let reduce_params = RawFgmresParams {
            n: self.num_dot_groups(),
            num_cells: 0,
            num_iters: 0,
            omega: 0.0,
            dispatch_x: DEFAULT_WORKGROUP_SIZE,
            max_restart: 0,
            column_offset: 0,
            _pad3: 0,
        };
        write_params(&core, &reduce_params);

        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FGMRES norm_sq reduce_final"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES norm_sq reduce_final"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.pipeline_reduce_final());
                pass.set_bind_group(0, &reduce_bg, &[]);
                pass.set_bind_group(1, self.matrix_bg(), &[]);
                pass.set_bind_group(2, self.precond_bg(), &[]);
                pass.set_bind_group(3, self.params_bg(), &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                self.scalars_buffer(),
                0,
                self.staging_scalar_buffer(),
                0,
                4,
            );
            queue.submit(Some(encoder.finish()));
            crate::count_submission!("FGMRES", "norm_sq_reduce_final");
        }

        // Restore params for subsequent vector ops, just in case.
        write_params(&core, &partial_params);

        // Read scalar via async map + polling loop (avoids blocking the whole device).
        let slice = self.staging_scalar_buffer().slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        loop {
            let _ = device.poll(wgpu::PollType::Poll);
            match rx.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(e)) => panic!("buffer mapping failed: {e:?}"),
                Err(std::sync::mpsc::TryRecvError::Empty) => std::thread::yield_now(),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("map_async channel disconnected")
                }
            }
        }

        let data = slice.get_mapped_range();
        let norm_sq: f32 = *bytemuck::from_bytes(&data[0..4]);
        drop(data);
        self.staging_scalar_buffer().unmap();
        norm_sq.sqrt()
    }
}

#[derive(Clone, Copy)]
pub struct FgmresSolveOnceConfig {
    pub tol_rel: f32,
    pub tol_abs: f32,
    pub reset_x_before_update: bool,
    /// Stall-stop level factor for the GPU-side guard (encoded path):
    /// freeze the solve when the restart-boundary true residual improves
    /// <2% for two consecutive checkpoints AND is already below
    /// `stall_level_rel * ||b||`. 0.0 disables. Mirrors the host-loop
    /// stall logic in `solve_fgmres` — keep the two in sync.
    pub stall_level_rel: f32,
    /// Enable CGS2 re-orthogonalization: a second classical Gram-Schmidt
    /// projection pass per Arnoldi iteration ("twice is enough"). Mitigates
    /// f32 orthogonality loss at the root (the restart guard treats the
    /// symptom). Costs one extra calc/reduce/update_w triple per iteration.
    pub enable_cgs2: bool,
    /// Enable the GPU-side restart-boundary monotonicity guard
    /// (gmres_logic/restart_guard + gmres_ops/guard_copy). Only valid on the
    /// encoded-seed path, where the seed writes the true residual norm into
    /// hessenberg[0]; host-seeded paths must keep this false (they have a
    /// host-side guard in `solve_fgmres` instead).
    pub enable_restart_guard: bool,
}

pub struct FgmresSolveOnceResult {
    pub basis_size: usize,
    pub residual_est: f32,
    pub converged: bool,
}

pub struct FgmresEncodeSolveOnceResult {
    pub max_restart: usize,
}

pub fn workgroups_for_size(n: u32) -> u32 {
    workgroups_for_size_ws(n, DEFAULT_WORKGROUP_SIZE)
}

/// Compute the number of workgroups needed to cover `n` elements with the
/// given `workgroup_size`.
pub fn workgroups_for_size_ws(n: u32, workgroup_size: u32) -> u32 {
    n.div_ceil(workgroup_size)
}

pub fn dispatch_2d(workgroups: u32) -> (u32, u32) {
    if workgroups <= MAX_WORKGROUPS_PER_DIMENSION {
        (workgroups, 1)
    } else {
        let dispatch_y = workgroups.div_ceil(MAX_WORKGROUPS_PER_DIMENSION);
        let dispatch_x = workgroups.div_ceil(dispatch_y);
        (dispatch_x, dispatch_y)
    }
}

pub fn dispatch_x_threads(workgroups: u32) -> u32 {
    dispatch_x_threads_ws(workgroups, DEFAULT_WORKGROUP_SIZE)
}

/// Compute the total number of threads in the X dimension for a 2D dispatch
/// layout using the given `workgroup_size`.
pub fn dispatch_x_threads_ws(workgroups: u32, workgroup_size: u32) -> u32 {
    let (dispatch_x, _) = dispatch_2d(workgroups);
    dispatch_x * workgroup_size
}

pub fn basis_binding<'a>(
    b_basis: &'a wgpu::Buffer,
    basis_stride: u64,
    vector_bytes: u64,
    idx: usize,
) -> wgpu::BindingResource<'a> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: b_basis,
        offset: (idx as u64) * basis_stride,
        size: std::num::NonZeroU64::new(vector_bytes),
    })
}

pub fn z_storage_binding<'a>(
    b_z_storage: &'a wgpu::Buffer,
    z_stride: u64,
    vector_bytes: u64,
    idx: usize,
) -> wgpu::BindingResource<'a> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: b_z_storage,
        offset: (idx as u64) * z_stride,
        size: std::num::NonZeroU64::new(vector_bytes),
    })
}

fn create_vector_bind_group<'a>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bindings: &[wgsl_reflect::WgslBindingDesc],
    x: wgpu::BindingResource<'a>,
    y: wgpu::BindingResource<'a>,
    z: wgpu::BindingResource<'a>,
    label: &str,
) -> wgpu::BindGroup {
    wgsl_reflect::create_bind_group_from_bindings(device, label, layout, bindings, 0, |name| {
        match name {
            "vec_x" => Some(x.clone()),
            "vec_y" => Some(y.clone()),
            "vec_z" => Some(z.clone()),
            _ => None,
        }
    })
    .unwrap_or_else(|e| panic!("{label} creation failed: {e}"))
}

/// Per-iteration vector bind groups, precomputed once at workspace construction
/// and reused across every FGMRES solve.
///
/// The inner restart loop needs, for each Arnoldi column `j`, four `bgl_vectors`
/// bind groups (SpMV input, norm-partial, norm-reduce, basis-normalize). Every one
/// binds only workspace-owned buffers (`basis`, `z_storage`, `w`, `temp`,
/// `dot_partial`) at fixed offsets, so they are identical on every solve. Building
/// them per iteration was the dominant CPU cost of the coupled solve (~4 reflection
/// `device.create_bind_group` calls × ~hundreds of iterations per step). Because a
/// `wgpu::BindGroup` is an owned Arc handle (not a Rust borrow) these can live on
/// the workspace and be indexed by `j` in the hot loop instead.
struct VectorBgCache {
    /// SpMV input `(vec_x=z_storage[j], vec_y=w, vec_z=temp)`, one per column `j`.
    spmv: Vec<wgpu::BindGroup>,
    /// Basis normalize/copy `(vec_x=w, vec_y=basis[j+1], vec_z=temp)`, per column `j`.
    scale: Vec<wgpu::BindGroup>,
    /// Norm-partial (column-independent): `(vec_x=w, vec_y=temp, vec_z=dot_partial)`.
    norm: wgpu::BindGroup,
    /// Norm-reduce (column-independent): `(vec_x=dot_partial, vec_y=temp, vec_z=temp)`.
    reduce: wgpu::BindGroup,
}

/// Build the [`VectorBgCache`]. The `(vec_x, vec_y, vec_z)` operand order for each
/// bind group MUST match the inner loop in
/// [`encode_fgmres_solve_once_with_preconditioner`] exactly, so the cached bind
/// groups are byte-for-byte substitutes for the ones that loop used to build.
#[allow(clippy::too_many_arguments)]
fn build_vector_bg_cache(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bindings: &'static [wgsl_reflect::WgslBindingDesc],
    b_basis: &wgpu::Buffer,
    b_z_storage: &wgpu::Buffer,
    b_w: &wgpu::Buffer,
    b_temp: &wgpu::Buffer,
    b_dot_partial: &wgpu::Buffer,
    basis_stride: u64,
    z_stride: u64,
    n: u32,
    max_restart: usize,
) -> VectorBgCache {
    let vector_bytes = (n as u64) * 4;
    let mut spmv = Vec::with_capacity(max_restart);
    let mut scale = Vec::with_capacity(max_restart);
    for j in 0..max_restart {
        spmv.push(create_vector_bind_group(
            device,
            layout,
            bindings,
            z_storage_binding(b_z_storage, z_stride, vector_bytes, j),
            b_w.as_entire_binding(),
            b_temp.as_entire_binding(),
            "FGMRES SpMV BG (cached)",
        ));
        scale.push(create_vector_bind_group(
            device,
            layout,
            bindings,
            b_w.as_entire_binding(),
            basis_binding(b_basis, basis_stride, vector_bytes, j + 1),
            b_temp.as_entire_binding(),
            "FGMRES Normalize Basis BG (cached)",
        ));
    }
    let norm = create_vector_bind_group(
        device,
        layout,
        bindings,
        b_w.as_entire_binding(),
        b_temp.as_entire_binding(),
        b_dot_partial.as_entire_binding(),
        "FGMRES Norm BG (cached)",
    );
    let reduce = create_vector_bind_group(
        device,
        layout,
        bindings,
        b_dot_partial.as_entire_binding(),
        b_temp.as_entire_binding(),
        b_temp.as_entire_binding(),
        "FGMRES Reduce BG (cached)",
    );
    VectorBgCache {
        spmv,
        scale,
        norm,
        reduce,
    }
}

pub fn dispatch_vector_pipeline(
    core: &FgmresCore<'_>,
    pipeline: &wgpu::ComputePipeline,
    vector_bg: &wgpu::BindGroup,
    dispatch_x: u32,
    dispatch_y: u32,
    label: &str,
) {
    let mut encoder = core
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, vector_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    core.queue.submit(Some(encoder.finish()));
    crate::count_submission!("FGMRES", label);
    crate::count_dispatch!("FGMRES", label);
}

pub fn dispatch_logic_pipeline(
    core: &FgmresCore<'_>,
    pipeline: &wgpu::ComputePipeline,
    workgroups: u32,
    label: &str,
) {
    let mut encoder = core
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, core.bg_logic, &[]);
        pass.set_bind_group(1, core.bg_logic_params, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    core.queue.submit(Some(encoder.finish()));
    crate::count_submission!("FGMRES", label);
    crate::count_dispatch!("FGMRES", label);
}

pub fn write_params(core: &FgmresCore<'_>, params: &RawFgmresParams) {
    core.queue.write_buffer(core.b_params, 0, bytes_of(params));
}

pub fn write_iter_params(core: &FgmresCore<'_>, iter_params: &IterParams) {
    core.queue
        .write_buffer(core.b_iter_params, 0, bytes_of(iter_params));
}

pub fn write_scalars(core: &FgmresCore<'_>, scalars: &[f32]) {
    core.queue
        .write_buffer(core.b_scalars, 0, bytemuck::cast_slice(scalars));
}

pub fn write_zeros(core: &FgmresCore<'_>, buffer: &wgpu::Buffer) {
    let size = buffer.size();
    core.queue
        .write_buffer(buffer, 0, &vec![0u8; size as usize]);
}

fn encode_write_buffer_from_bytes(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    dst: &wgpu::Buffer,
    offset: u64,
    bytes: &[u8],
    label: &'static str,
) {
    let size = bytes.len() as u64;
    if size == 0 {
        return;
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let mut mapped = staging.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(bytes);
    }
    staging.unmap();
    encoder.copy_buffer_to_buffer(&staging, 0, dst, offset, size);
}

pub fn encode_write_params(
    core: &FgmresCore<'_>,
    encoder: &mut wgpu::CommandEncoder,
    params: &RawFgmresParams,
) {
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params,
        0,
        bytes_of(params),
        "FGMRES params (encoded)",
    );
}

pub fn encode_fgmres_seed_basis0_from_system<'a>(
    core: &FgmresCore<'a>,
    encoder: &mut wgpu::CommandEncoder,
    system: LinearSystemView<'a>,
    max_restart: u32,
    preserve_convergence_state: bool,
) {
    let n = core.n;
    let vector_bytes = (n as u64) * 4;
    let workgroups = workgroups_for_size(n);
    let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
    let dispatch_x_threads = dispatch_x_threads(workgroups);

    // Vector-op params for SpMV/AXPBY/Norm/Scale passes.
    let vector_params = RawFgmresParams {
        n,
        num_cells: core.num_cells,
        num_iters: 0,
        omega: 1.0,
        dispatch_x: dispatch_x_threads,
        max_restart,
        column_offset: (core.z_stride / 4) as u32,
        _pad3: 0,
    };
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params,
        0,
        bytes_of(&vector_params),
        "FGMRES residual seed params",
    );

    // Reset restart-local aux buffers so this path behaves like the host-seeded variant.
    encoder.clear_buffer(core.b_hessenberg, 0, None);
    encoder.clear_buffer(core.b_givens, 0, None);
    encoder.clear_buffer(core.b_y, 0, None);

    let basis0 = basis_binding(core.b_basis, core.basis_stride, vector_bytes, 0);

    // basis0 = rhs - A*x
    let spmv_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        system.x().as_entire_binding(),
        core.b_w.as_entire_binding(),
        core.b_temp.as_entire_binding(),
        "FGMRES residual seed spmv BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES residual seed spmv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_spmv);
        pass.set_bind_group(0, &spmv_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    if preserve_convergence_state {
        // Only write scratch slots needed by AXPBY; preserve STOP/CONVERGED/RHS_NORM.
        let scratch_scalars: [f32; 2] = [1.0, -1.0]; // alpha, beta for AXPBY
        encode_write_buffer_from_bytes(
            core.device,
            encoder,
            core.b_scalars,
            0,
            bytemuck::cast_slice(&scratch_scalars),
            "FGMRES residual seed scalars (scratch only)",
        );
    } else {
        let mut axpby_scalars = [0.0_f32; FGMRES_SCALAR_COUNT];
        axpby_scalars[FGMRES_SCALAR_STOP] = 0.0;
        axpby_scalars[FGMRES_SCALAR_CONVERGED] = 0.0;
        axpby_scalars[0] = 1.0;
        axpby_scalars[1] = -1.0;
        encode_write_buffer_from_bytes(
            core.device,
            encoder,
            core.b_scalars,
            0,
            bytemuck::cast_slice(&axpby_scalars),
            "FGMRES residual seed scalars",
        );
    }
    let residual_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        system.rhs().as_entire_binding(),
        core.b_w.as_entire_binding(),
        basis0.clone(),
        "FGMRES residual seed axpby BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES residual seed axpby"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_axpby);
        pass.set_bind_group(0, &residual_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Compute ||basis0|| and write 1/||basis0|| to scalars[0].
    let norm_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        basis0.clone(),
        core.b_temp.as_entire_binding(),
        core.b_dot_partial.as_entire_binding(),
        "FGMRES residual seed norm BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES residual seed norm partial"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_norm_sq);
        pass.set_bind_group(0, &norm_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    let reduce_params = RawFgmresParams {
        n: core.num_dot_groups,
        num_cells: 0,
        num_iters: 0,
        omega: 0.0,
        dispatch_x: DEFAULT_WORKGROUP_SIZE,
        max_restart: 0,
        column_offset: 0,
        _pad3: 0,
    };
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params,
        0,
        bytes_of(&reduce_params),
        "FGMRES residual seed reduce params",
    );
    let reduce_iter_params = IterParams {
        current_idx: 0,
        max_restart,
        _pad1: 0,
        _pad2: 0,
    };
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_iter_params,
        0,
        bytes_of(&reduce_iter_params),
        "FGMRES residual seed iter params",
    );

    let reduce_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        core.b_dot_partial.as_entire_binding(),
        core.b_temp.as_entire_binding(),
        core.b_temp.as_entire_binding(),
        "FGMRES residual seed reduce BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES residual seed reduce final"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_reduce_final_and_finish_norm);
        pass.set_bind_group(0, &reduce_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // Restore vector-op params for basis normalization.
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params,
        0,
        bytes_of(&vector_params),
        "FGMRES residual seed params (restore)",
    );

    let scale_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        core.b_w.as_entire_binding(),
        basis0,
        core.b_temp.as_entire_binding(),
        "FGMRES residual seed normalize BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES residual seed normalize"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_scale_in_place);
        pass.set_bind_group(0, &scale_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Initialize g_rhs[0] from the residual norm produced above.
    let g_bytes = ((core.max_restart + 1) as u64) * 4;
    encoder.clear_buffer(core.b_g, 0, Some(g_bytes));
    encoder.copy_buffer_to_buffer(core.b_hessenberg, 0, core.b_g, 0, 4);
}

/// Encode a GPU-side computation of `||rhs||` and store the result in
/// `scalars[FGMRES_SCALAR_RHS_NORM]`.
///
/// Uses the same two-pass norm-reduction pattern as the basis0 seeding path:
///   1. `pipeline_norm_sq` — workgroup-level partial sum-of-squares → `b_dot_partial`
///   2. `pipeline_reduce_final_and_finish_norm` — final sum, sqrt, writes norm to
///      `hessenberg[0]` and `1/norm` to `scalars[0]`.
///   3. `copy_buffer_to_buffer` to propagate `hessenberg[0]` (= `||rhs||`) into
///      `scalars[FGMRES_SCALAR_RHS_NORM]`.
///
/// After this function returns, subsequent GPU kernels can read the RHS norm from
/// `scalars[14]` without a host readback.
pub fn encode_rhs_norm_into_scalars<'a>(
    core: &FgmresCore<'a>,
    encoder: &mut wgpu::CommandEncoder,
    system: LinearSystemView<'a>,
) {
    let n = core.n;
    let workgroups = workgroups_for_size(n);
    let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
    let dispatch_x_threads = dispatch_x_threads(workgroups);

    // Set up vector-op params for the norm reduction.
    let norm_params = RawFgmresParams {
        n,
        num_cells: core.num_cells,
        num_iters: 0,
        omega: 1.0,
        dispatch_x: dispatch_x_threads,
        max_restart: 0,
        column_offset: (core.z_stride / 4) as u32,
        _pad3: 0,
    };
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params,
        0,
        bytes_of(&norm_params),
        "FGMRES rhs_norm params",
    );

    // Pass 1: partial norm-squared of the RHS vector.
    let norm_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        system.rhs().as_entire_binding(),
        core.b_temp.as_entire_binding(),
        core.b_dot_partial.as_entire_binding(),
        "FGMRES rhs_norm partial BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES rhs_norm partial"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_norm_sq);
        pass.set_bind_group(0, &norm_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 2: final reduction → writes norm to hessenberg[0], 1/norm to scalars[0].
    // Temporarily switch params.n to num_dot_groups for the reduction kernel.
    let reduce_params = RawFgmresParams {
        n: core.num_dot_groups,
        num_cells: 0,
        num_iters: 0,
        omega: 0.0,
        dispatch_x: DEFAULT_WORKGROUP_SIZE,
        max_restart: 0,
        column_offset: 0,
        _pad3: 0,
    };
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params,
        0,
        bytes_of(&reduce_params),
        "FGMRES rhs_norm reduce params",
    );
    // iter_params.current_idx = 0 so the norm lands in hessenberg[0].
    let reduce_iter_params = IterParams {
        current_idx: 0,
        max_restart: 0,
        _pad1: 0,
        _pad2: 0,
    };
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_iter_params,
        0,
        bytes_of(&reduce_iter_params),
        "FGMRES rhs_norm reduce iter params",
    );

    // We need SCALAR_STOP = 0 so reduce_final_and_finish_norm doesn't early-exit.
    // Only clear the STOP slot to avoid clobbering other scalars that may already
    // be initialised (e.g. tolerance slots written by encode_fgmres_solve_once).
    let stop_zero: [f32; 1] = [0.0];
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_scalars,
        (FGMRES_SCALAR_STOP * 4) as u64,
        bytemuck::cast_slice(&stop_zero),
        "FGMRES rhs_norm scalars (clear stop)",
    );

    let reduce_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        core.b_dot_partial.as_entire_binding(),
        core.b_temp.as_entire_binding(),
        core.b_temp.as_entire_binding(),
        "FGMRES rhs_norm reduce BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES rhs_norm reduce final"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_reduce_final_and_finish_norm);
        pass.set_bind_group(0, &reduce_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // Propagate ||rhs|| from hessenberg[0] to scalars[FGMRES_SCALAR_RHS_NORM].
    encoder.copy_buffer_to_buffer(
        core.b_hessenberg,
        0,
        core.b_scalars,
        (FGMRES_SCALAR_RHS_NORM * 4) as u64,
        4,
    );
}

pub fn read_scalar(core: &FgmresCore<'_>) -> f32 {
    let mut encoder = core
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FGMRES read scalar"),
        });
    encoder.copy_buffer_to_buffer(core.b_scalars, 0, core.b_staging_scalar, 0, 4);
    let submission_index = core.queue.submit(Some(encoder.finish()));
    crate::count_submission!("FGMRES", "read_scalar");
    read_scalar_after_submit(core, submission_index)
}

pub fn read_scalar_after_submit(
    core: &FgmresCore<'_>,
    submission_index: wgpu::SubmissionIndex,
) -> f32 {
    let slice = core.b_staging_scalar.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    let _ = core.device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission_index),
        timeout: None,
    });
    rx.recv()
        .map_err(|e| format!("FGMRES staging readback recv failed: {e}"))
        .and_then(|r| r.map_err(|e| format!("FGMRES buffer mapping failed: {e:?}")))
        .expect("FGMRES staging readback failed");

    let data = slice.get_mapped_range();
    let value: f32 = *bytemuck::from_bytes(&data[0..4]);
    drop(data);
    core.b_staging_scalar.unmap();
    value
}

pub(crate) fn read_solver_scalars_after_submit(
    core: &FgmresCore<'_>,
    submission_index: wgpu::SubmissionIndex,
) -> [f32; FGMRES_SCALAR_COUNT] {
    let slice = core.b_staging_scalar.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    let _ = core.device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission_index),
        timeout: None,
    });
    rx.recv()
        .map_err(|e| format!("FGMRES scalar readback recv failed: {e}"))
        .and_then(|r| r.map_err(|e| format!("FGMRES buffer mapping failed: {e:?}")))
        .expect("FGMRES scalar readback failed");

    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    let mut out = [0.0_f32; FGMRES_SCALAR_COUNT];
    let count = values.len().min(FGMRES_SCALAR_COUNT);
    out[..count].copy_from_slice(&values[..count]);
    drop(data);
    core.b_staging_scalar.unmap();
    out
}

pub fn encode_fgmres_solve_once_with_preconditioner<'a>(
    core: &FgmresCore<'a>,
    encoder: &mut wgpu::CommandEncoder,
    x: &'a wgpu::Buffer,
    rhs_norm: f32,
    mut params: RawFgmresParams,
    iter_params: IterParams,
    config: FgmresSolveOnceConfig,
    capture_solver_scalars: bool,
    preserve_convergence_state: bool,
    // When `Some`, compute `||rhs||` on the GPU and store in `scalars[RHS_NORM]`
    // after the scalars buffer has been initialised.  Must only be used on the
    // first restart chunk (non-preserve path) where `seed_basis0` has already
    // written beta = ||r0|| into `hessenberg[0]`.  The function saves/restores
    // `hessenberg[0]` around the norm computation.
    rhs_norm_system: Option<LinearSystemView<'a>>,
    mut precondition: impl FnMut(
        usize,
        &mut wgpu::CommandEncoder,
        wgpu::BindingResource<'a>,
        wgpu::BindingResource<'a>,
    ),
) -> FgmresEncodeSolveOnceResult {
    let n = core.n;
    let vector_bytes = (n as u64) * 4;
    let workgroups = workgroups_for_size(n);
    let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
    let cell_workgroups = workgroups_for_size(core.num_cells);
    let (cell_dispatch_x, cell_dispatch_y) = dispatch_2d(cell_workgroups);
    let dispatch_x_threads = dispatch_x_threads(workgroups);

    let max_restart = iter_params.max_restart.max(1).min(core.max_restart as u32) as usize;

    let tol_abs = config.tol_abs;
    let tol_rel_rhs = config.tol_rel * rhs_norm;

    // Ensure vector ops see correct dispatch width and problem size.
    params.n = n;
    params.dispatch_x = dispatch_x_threads;
    params.max_restart = max_restart as u32;
    params.column_offset = (core.z_stride / 4) as u32;
    encode_write_params(core, encoder, &params);

    if preserve_convergence_state {
        // Snapshot SCALAR_STOP → SCALAR_SKIP_UPDATE before the inner loop starts.
        // This lets solve_triangular and accumulate_solution skip when a prior chunk
        // already converged, while still running correctly if convergence happens
        // during THIS chunk (SKIP_UPDATE stays 0 because STOP was 0 at snapshot time).
        //
        // WebGPU forbids copy_buffer_to_buffer with the same source and destination
        // buffer, so we use a two-hop copy through a tiny scratch buffer.
        encoder.copy_buffer_to_buffer(
            core.b_scalars,
            (FGMRES_SCALAR_STOP * 4) as u64,
            core.b_scalar_copy_staging,
            0,
            4,
        );
        encoder.copy_buffer_to_buffer(
            core.b_scalar_copy_staging,
            0,
            core.b_scalars,
            (FGMRES_SCALAR_SKIP_UPDATE * 4) as u64,
            4,
        );

        // Only update tolerance and default slots without resetting STOP/CONVERGED/indirect.
        // This preserves early-termination state from a previous restart chunk so that
        // converged chunks cause all subsequent chunks to be no-ops.
        let tol_scalars: [f32; 2] = [
            tol_rel_rhs, // SCALAR_TOL_REL_RHS (12)
            tol_abs,     // SCALAR_TOL_ABS (13)
        ];
        encode_write_buffer_from_bytes(
            core.device,
            encoder,
            core.b_scalars,
            (FGMRES_SCALAR_TOL_REL_RHS * 4) as u64,
            bytemuck::cast_slice(&tol_scalars),
            "FGMRES solver scalars (tolerance only)",
        );
        // Update ITERS_USED default for this chunk (used by accumulate_solution).
        let iters_default: [f32; 1] = [max_restart as f32];
        encode_write_buffer_from_bytes(
            core.device,
            encoder,
            core.b_scalars,
            (FGMRES_SCALAR_ITERS_USED * 4) as u64,
            bytemuck::cast_slice(&iters_default),
            "FGMRES solver scalars (iters_used default)",
        );
    } else {
        let mut solver_scalars = [0.0_f32; FGMRES_SCALAR_COUNT];
        solver_scalars[FGMRES_SCALAR_STOP] = 0.0;
        solver_scalars[FGMRES_SCALAR_CONVERGED] = 0.0;
        solver_scalars[FGMRES_SCALAR_ITERS_USED] = max_restart as f32;
        solver_scalars[FGMRES_SCALAR_RESIDUAL_EST] = f32::INFINITY;
        solver_scalars[FGMRES_SCALAR_TOL_REL_RHS] = tol_rel_rhs;
        solver_scalars[FGMRES_SCALAR_TOL_ABS] = tol_abs;
        solver_scalars[FGMRES_SCALAR_RHS_NORM] = 1.0; // shader multiplies TOL_REL_RHS * RHS_NORM
        solver_scalars[FGMRES_SCALAR_SKIP_UPDATE] = 0.0;
        solver_scalars[FGMRES_SCALAR_STALL_REL] = config.stall_level_rel;
        encode_write_buffer_from_bytes(
            core.device,
            encoder,
            core.b_scalars,
            0,
            bytemuck::cast_slice(&solver_scalars),
            "FGMRES solver scalars",
        );

        let indirect_args: [u32; FGMRES_INDIRECT_DISPATCH_COUNT * 4] = [
            dispatch_x,
            dispatch_y,
            1,
            0,
            cell_dispatch_x,
            cell_dispatch_y,
            1,
            0,
            max_restart as u32,
            1,
            1,
            0,
        ];
        encode_write_buffer_from_bytes(
            core.device,
            encoder,
            core.b_indirect_args,
            0,
            bytemuck::cast_slice(&indirect_args),
            "FGMRES indirect args",
        );
    }

    // ── GPU-side ||rhs|| computation ────────────────────────────────────
    // Must happen after the scalars init (so our write to scalars[RHS_NORM]
    // is not clobbered) and after seed_basis0 (so basis0 is ready).
    //
    // encode_rhs_norm_into_scalars reuses reduce_final_and_finish_norm which
    // writes the norm into hessenberg[0].  However hessenberg[0] already
    // holds beta = ||r0|| written by seed_basis0 and needed by the inner
    // loop (update_hessenberg_givens).  We save/restore hessenberg[0] via
    // b_y[0] which is cleared by seed_basis0 and unused until
    // solve_triangular at the end.
    if let Some(system) = rhs_norm_system {
        // Save hessenberg[0] (beta) → b_y[0].
        encoder.copy_buffer_to_buffer(core.b_hessenberg, 0, core.b_y, 0, 4);
        // Compute ||rhs|| → scalars[RHS_NORM] (clobbers hessenberg[0]).
        encode_rhs_norm_into_scalars(core, encoder, system);
        // Restore hessenberg[0] (beta) ← b_y[0].
        encoder.copy_buffer_to_buffer(core.b_y, 0, core.b_hessenberg, 0, 4);
        // Align with the host loop's rel_scale = min(||b||, ||r0||): clamp
        // scalars[RHS_NORM] (= ||b||) by beta (= ||r0||, restored above).
        // Without this, a warm start with ||r0|| << ||b|| would declare
        // convergence against ||b|| alone — the two paths diverge exactly
        // when the tolerance is reachable.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES clamp rel scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_clamp_rel_scale);
            pass.set_bind_group(0, core.bg_logic, &[]);
            pass.set_bind_group(1, core.bg_logic_params, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    // ── Restart-boundary monotonicity guard ─────────────────────────────
    // Must run after the scalars init (so BEST_RESID/GUARD_FLAG survive)
    // and after seed_basis0 wrote beta = ||b - A*x|| into hessenberg[0].
    if config.enable_restart_guard {
        encode_restart_guard(core, encoder, x, &params);
    }

    let max_restart_u32 = max_restart as u32;
    let params_iter_table: Vec<RawFgmresParams> = (0..max_restart)
        .map(|j| RawFgmresParams {
            num_iters: j as u32,
            ..params
        })
        .collect();

    let params_reduce_base = RawFgmresParams {
        n: core.num_dot_groups,
        dispatch_x: DEFAULT_WORKGROUP_SIZE,
        ..params
    };
    let params_reduce_table: Vec<RawFgmresParams> = (0..max_restart)
        .map(|j| RawFgmresParams {
            num_iters: j as u32,
            ..params_reduce_base
        })
        .collect();

    let iter_table_j: Vec<IterParams> = (0..max_restart)
        .map(|j| IterParams {
            current_idx: j as u32,
            max_restart: max_restart_u32,
            _pad1: 0,
            _pad2: 0,
        })
        .collect();
    let iter_table_hessenberg: Vec<IterParams> = (0..max_restart)
        .map(|j| IterParams {
            current_idx: (j as u32) * (max_restart_u32 + 1) + (j as u32 + 1),
            max_restart: max_restart_u32,
            _pad1: 0,
            _pad2: 0,
        })
        .collect();

    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params_table_iter,
        0,
        bytemuck::cast_slice(&params_iter_table),
        "FGMRES params table iter",
    );
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params_table_reduce,
        0,
        bytemuck::cast_slice(&params_reduce_table),
        "FGMRES params table reduce",
    );
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_iter_table_j,
        0,
        bytemuck::cast_slice(&iter_table_j),
        "FGMRES iter table j",
    );
    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_iter_table_hessenberg,
        0,
        bytemuck::cast_slice(&iter_table_hessenberg),
        "FGMRES iter table hessenberg",
    );

    for j in 0..max_restart {
        let params_offset = (j as u64) * FGMRES_PARAMS_STRIDE_BYTES;
        let iter_offset = (j as u64) * FGMRES_ITER_PARAMS_STRIDE_BYTES;

        // Select this column's params into their dedicated buffers in ONE batched
        // blit section at the iteration top. Because each distinct concurrently-live
        // value now has its own buffer (b_params=iter, b_params_reduce=reduce,
        // b_iter_params=j, b_iter_params_hess=hessenberg-index), nothing has to be
        // swapped or restored mid-iteration, so all the compute passes below run as
        // one uninterrupted compute-encoder run (3 blit sections -> 1). Byte-identical:
        // every pass reads the same table row it read before (only the reduce-final
        // pass rebinds to the dedicated buffers via `bg_params_reduce`).
        encoder.copy_buffer_to_buffer(
            core.b_params_table_iter,
            params_offset,
            core.b_params,
            0,
            FGMRES_PARAMS_STRIDE_BYTES,
        );
        encoder.copy_buffer_to_buffer(
            core.b_params_table_reduce,
            params_offset,
            core.b_params_reduce,
            0,
            FGMRES_PARAMS_STRIDE_BYTES,
        );
        encoder.copy_buffer_to_buffer(
            core.b_iter_table_j,
            iter_offset,
            core.b_iter_params,
            0,
            FGMRES_ITER_PARAMS_STRIDE_BYTES,
        );
        encoder.copy_buffer_to_buffer(
            core.b_iter_table_hessenberg,
            iter_offset,
            core.b_iter_params_hess,
            0,
            FGMRES_ITER_PARAMS_STRIDE_BYTES,
        );

        let z_buf = z_storage_binding(core.b_z_storage, core.z_stride, vector_bytes, j);
        let vj = basis_binding(core.b_basis, core.basis_stride, vector_bytes, j);

        precondition(j, encoder, vj, z_buf.clone());

        // Cached per-column bind group: identical to
        // `(vec_x=z_storage[j], vec_y=w, vec_z=temp)` built inline before.
        let spmv_bg = &core.spmv_bgs[j];
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES SpMV"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_spmv);
            pass.set_bind_group(0, spmv_bg, &[]);
            pass.set_bind_group(1, core.bg_matrix, &[]);
            pass.set_bind_group(2, core.bg_precond, &[]);
            pass.set_bind_group(3, core.bg_params, &[]);
            pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_DOFS_OFFSET);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES CGS Calc"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_calc_dots_cgs);
            pass.set_bind_group(0, core.bg_cgs, &[]);
            pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_DOFS_OFFSET);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES CGS Reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_reduce_dots_cgs);
            pass.set_bind_group(0, core.bg_cgs, &[]);
            pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_SCALAR_OFFSET);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES CGS Update W"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_update_w_cgs);
            pass.set_bind_group(0, core.bg_cgs, &[]);
            pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_DOFS_OFFSET);
        }
        if config.enable_cgs2 {
            // CGS2: re-project the corrected w against the basis. The pass-2
            // reduce ACCUMULATES into the Hessenberg entries (H = d1 + d2)
            // and stashes the pass-2 coefficients in b_dot_partial, which
            // the pass-2 update_w reads (subtracting H again would
            // double-project). b_params still holds params_table_iter[j].
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES CGS2 Calc"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_calc_dots_cgs);
                pass.set_bind_group(0, core.bg_cgs, &[]);
                pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_DOFS_OFFSET);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES CGS2 Reduce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_reduce_dots_cgs_reortho);
                pass.set_bind_group(0, core.bg_cgs, &[]);
                pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_SCALAR_OFFSET);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES CGS2 Update W"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_update_w_cgs_reortho);
                pass.set_bind_group(0, core.bg_cgs, &[]);
                pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_DOFS_OFFSET);
            }
        }

        // Cached column-independent bind group `(vec_x=w, vec_y=temp, vec_z=dot_partial)`.
        let norm_bg = core.norm_bg;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES Norm Partial"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_norm_sq);
            pass.set_bind_group(0, norm_bg, &[]);
            pass.set_bind_group(1, core.bg_matrix, &[]);
            pass.set_bind_group(2, core.bg_precond, &[]);
            pass.set_bind_group(3, core.bg_params, &[]);
            pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_DOFS_OFFSET);
        }

        // Cached column-independent bind group `(vec_x=dot_partial, vec_y=temp, vec_z=temp)`.
        let reduce_bg = core.reduce_bg;

        // Reduce-Final reads the *reduce* params (n=num_dot_groups) and the
        // hessenberg-index iter_params — supplied by `bg_params_reduce`, which binds
        // the dedicated `b_params_reduce` / `b_iter_params_hess` buffers written at
        // the loop top. (Was: two mid-iteration copies clobbering b_params/b_iter_params.)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES Reduce Final & Finish Norm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_reduce_final_and_finish_norm);
            pass.set_bind_group(0, reduce_bg, &[]);
            pass.set_bind_group(1, core.bg_matrix, &[]);
            pass.set_bind_group(2, core.bg_precond, &[]);
            pass.set_bind_group(3, core.bg_params_reduce, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Cached per-column bind group: identical to
        // `(vec_x=w, vec_y=basis[j+1], vec_z=temp)` built inline before.
        let scale_bg = &core.scale_bgs[j];

        // Normalize reads b_params.n = n (iter value) — still live from the loop-top
        // copy (b_params was never clobbered because the reduce pass used its own
        // buffer), and b_iter_params = j (also from the loop top). Both mid-iteration
        // "restore" copies that used to sit here are now gone.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES Normalize & Copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_scale);
            pass.set_bind_group(0, scale_bg, &[]);
            pass.set_bind_group(1, core.bg_matrix, &[]);
            pass.set_bind_group(2, core.bg_precond, &[]);
            pass.set_bind_group(3, core.bg_params, &[]);
            pass.dispatch_workgroups_indirect(core.b_indirect_args, FGMRES_INDIRECT_DOFS_OFFSET);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FGMRES Update Hessenberg"),
                timestamp_writes: None,
            });
            pass.set_pipeline(core.pipeline_update_hessenberg);
            pass.set_bind_group(0, core.bg_logic, &[]);
            pass.set_bind_group(1, core.bg_logic_params, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    if max_restart > 0 {
        encoder.copy_buffer_to_buffer(
            core.b_iter_table_j,
            0,
            core.b_iter_params,
            0,
            FGMRES_ITER_PARAMS_STRIDE_BYTES,
        );
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES Solve Triangular"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_solve_triangular);
        pass.set_bind_group(0, core.bg_logic, &[]);
        pass.set_bind_group(1, core.bg_logic_params, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    if config.reset_x_before_update {
        encoder.clear_buffer(x, 0, Some(vector_bytes));
    }

    let fused_bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        core.b_z_storage.as_entire_binding(),
        x.as_entire_binding(),
        core.b_temp.as_entire_binding(),
        "FGMRES Solution Update Fused BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES Solution Update Fused"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_axpy_fused_from_y);
        pass.set_bind_group(0, &fused_bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    if capture_solver_scalars {
        encoder.copy_buffer_to_buffer(
            core.b_scalars,
            0,
            core.b_staging_scalar,
            0,
            (FGMRES_SCALAR_COUNT as u64) * 4,
        );
    }

    FgmresEncodeSolveOnceResult { max_restart }
}

/// Encode the restart-boundary monotonicity guard: a 1-thread decision
/// kernel (gmres_logic/restart_guard) followed by a conditional
/// snapshot/restore of the solution vector (gmres_ops/guard_copy).
///
/// Preconditions: the encoded seed has written beta = ||b - A*x|| into
/// hessenberg[0] and the solver scalars are initialized. Overwrites
/// b_params with `params` (vector-op params) for the copy dispatch; the
/// restart body re-writes per-iteration params from its tables, and any
/// caller after the body must not rely on b_params contents.
pub fn encode_restart_guard<'a>(
    core: &FgmresCore<'a>,
    encoder: &mut wgpu::CommandEncoder,
    x: &'a wgpu::Buffer,
    params: &RawFgmresParams,
) {
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES restart guard"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_restart_guard);
        pass.set_bind_group(0, core.bg_logic, &[]);
        pass.set_bind_group(1, core.bg_logic_params, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encode_write_buffer_from_bytes(
        core.device,
        encoder,
        core.b_params,
        0,
        bytes_of(params),
        "FGMRES restart guard params",
    );

    let workgroups = workgroups_for_size(core.n);
    let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
    // vec_x unused (read-only slot), x as vec_y (read_write), snapshot as vec_z.
    let bg = create_vector_bind_group(
        core.device,
        core.bgl_vectors,
        core.vector_bindings,
        core.b_temp.as_entire_binding(),
        x.as_entire_binding(),
        core.b_x_snapshot.as_entire_binding(),
        "FGMRES restart guard copy BG",
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FGMRES restart guard copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(core.pipeline_guard_copy);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_bind_group(1, core.bg_matrix, &[]);
        pass.set_bind_group(2, core.bg_precond, &[]);
        pass.set_bind_group(3, core.bg_params, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

pub fn submit_fgmres_encoded_pass(
    core: &FgmresCore<'_>,
    encoder: wgpu::CommandEncoder,
    label: &'static str,
) -> wgpu::SubmissionIndex {
    let submission_index = core.queue.submit(Some(encoder.finish()));
    crate::count_submission!("FGMRES", label);
    submission_index
}

pub fn solve_once_from_encoded_status(
    core: &FgmresCore<'_>,
    status_submission_index: wgpu::SubmissionIndex,
    max_restart: usize,
) -> FgmresSolveOnceResult {
    let solver_scalars = read_solver_scalars_after_submit(core, status_submission_index);
    let mut basis_size = max_restart;

    let residual_est = solver_scalars[FGMRES_SCALAR_RESIDUAL_EST];
    let converged = solver_scalars[FGMRES_SCALAR_CONVERGED] > 0.5;

    let reported_basis = solver_scalars[FGMRES_SCALAR_ITERS_USED];
    if reported_basis.is_finite() {
        let reported_basis = reported_basis.round().clamp(1.0, max_restart as f32) as usize;
        basis_size = reported_basis;
    }

    FgmresSolveOnceResult {
        basis_size,
        residual_est,
        converged,
    }
}
