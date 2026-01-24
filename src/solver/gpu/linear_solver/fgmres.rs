use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;
use bytemuck::{bytes_of, Pod, Zeroable};

pub const WORKGROUP_SIZE: u32 = 64;
pub const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;

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

    pub b_basis: &'a wgpu::Buffer,
    pub z_vectors: &'a [wgpu::Buffer],
    pub b_w: &'a wgpu::Buffer,
    pub b_temp: &'a wgpu::Buffer,
    pub b_dot_partial: &'a wgpu::Buffer,
    pub b_scalars: &'a wgpu::Buffer,
    pub b_params: &'a wgpu::Buffer,
    pub b_iter_params: &'a wgpu::Buffer,
    pub b_staging_scalar: &'a wgpu::Buffer,

    pub bg_matrix: &'a wgpu::BindGroup,
    pub bg_precond: &'a wgpu::BindGroup,
    pub bg_params: &'a wgpu::BindGroup,
    pub bg_logic: &'a wgpu::BindGroup,
    pub bg_logic_params: &'a wgpu::BindGroup,
    pub bg_cgs: &'a wgpu::BindGroup,

    pub bgl_vectors: &'a wgpu::BindGroupLayout,
    vector_bindings: &'static [wgsl_reflect::WgslBindingDesc],

    pub pipeline_spmv: &'a wgpu::ComputePipeline,
    pub pipeline_scale: &'a wgpu::ComputePipeline,
    pub pipeline_norm_sq: &'a wgpu::ComputePipeline,
    pub pipeline_reduce_final_and_finish_norm: &'a wgpu::ComputePipeline,
    pub pipeline_update_hessenberg: &'a wgpu::ComputePipeline,
    pub pipeline_solve_triangular: &'a wgpu::ComputePipeline,
    pub pipeline_axpy_from_y: &'a wgpu::ComputePipeline,
    pub pipeline_calc_dots_cgs: &'a wgpu::ComputePipeline,
    pub pipeline_reduce_dots_cgs: &'a wgpu::ComputePipeline,
    pub pipeline_update_w_cgs: &'a wgpu::ComputePipeline,
}

pub enum FgmresPrecondBindings<'a> {
    Diag {
        diag_u: &'a wgpu::Buffer,
        diag_v: &'a wgpu::Buffer,
        diag_p: &'a wgpu::Buffer,
    },
    DiagWithParams {
        diag_u: &'a wgpu::Buffer,
        diag_v: &'a wgpu::Buffer,
        diag_p: &'a wgpu::Buffer,
        precond_params: &'a wgpu::Buffer,
    },

    /// Schur complement preconditioner inputs.
    ///
    /// `diag_u` stores per-cell diagonal inverses for all velocity-like components,
    /// packed as `[cell0_u0, cell0_u1, ..., cell0_u_{u_len-1}, cell1_u0, ...]`.
    SchurWithParams {
        diag_u: &'a wgpu::Buffer,
        diag_p: &'a wgpu::Buffer,
        precond_params: &'a wgpu::Buffer,
    },
}

pub struct FgmresWorkspace {
    max_restart: usize,
    n: u32,
    num_cells: u32,
    num_dot_groups: u32,
    basis_stride: u64,

    b_basis: wgpu::Buffer,
    z_vectors: Vec<wgpu::Buffer>,
    b_w: wgpu::Buffer,
    b_temp: wgpu::Buffer,
    b_dot_partial: wgpu::Buffer,
    b_scalars: wgpu::Buffer,
    b_params: wgpu::Buffer,
    b_iter_params: wgpu::Buffer,
    b_hessenberg: wgpu::Buffer,
    b_givens: wgpu::Buffer,
    b_g: wgpu::Buffer,
    b_y: wgpu::Buffer,
    b_staging_scalar: wgpu::Buffer,

    bgl_vectors: wgpu::BindGroupLayout,
    vector_bindings: &'static [wgsl_reflect::WgslBindingDesc],
    bgl_matrix: wgpu::BindGroupLayout,
    bgl_precond: wgpu::BindGroupLayout,
    bgl_params: wgpu::BindGroupLayout,

    bg_matrix: wgpu::BindGroup,
    bg_precond: wgpu::BindGroup,
    bg_params: wgpu::BindGroup,
    bg_logic: wgpu::BindGroup,
    bg_logic_params: wgpu::BindGroup,
    bg_cgs: wgpu::BindGroup,

    pipeline_spmv: wgpu::ComputePipeline,
    pipeline_axpy: wgpu::ComputePipeline,
    pipeline_axpy_from_y: wgpu::ComputePipeline,
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
}

impl FgmresWorkspace {
    pub fn new(
        device: &wgpu::Device,
        n: u32,
        num_cells: u32,
        max_restart: usize,
        matrix_row_offsets: &wgpu::Buffer,
        matrix_col_indices: &wgpu::Buffer,
        matrix_values: &wgpu::Buffer,
        precond: FgmresPrecondBindings<'_>,
        label_prefix: &str,
    ) -> Self {
        let num_dot_groups = workgroups_for_size(n);

        let min_alignment = 256u64;
        let basis_stride_unaligned = (n as u64) * 4;
        let basis_stride = (basis_stride_unaligned + min_alignment - 1) & !(min_alignment - 1);
        let basis_size = basis_stride * (max_restart as u64 + 1);

        let b_basis = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix} FGMRES basis")),
            size: basis_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut z_vectors = Vec::with_capacity(max_restart);
        for i in 0..max_restart {
            z_vectors.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label_prefix} FGMRES Z {i}")),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }

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
            size: 16 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ops_spmv_src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_SPMV)
            .expect("gmres_ops/spmv shader missing from kernel registry");
        let ops_bindings = ops_spmv_src.bindings;

        let pipeline_spmv = (ops_spmv_src.create_pipeline)(device);
        let pipeline_axpy = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_AXPY)
                .expect("gmres_ops/axpy shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_axpy_from_y = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_AXPY_FROM_Y)
                .expect("gmres_ops/axpy_from_y shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_axpby = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_AXPBY)
                .expect("gmres_ops/axpby shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_scale = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_SCALE)
                .expect("gmres_ops/scale shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_scale_in_place = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_SCALE_IN_PLACE)
                .expect("gmres_ops/scale_in_place shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_copy = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_COPY)
                .expect("gmres_ops/copy shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_norm_sq = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_NORM_SQ_PARTIAL)
                .expect("gmres_ops/norm_sq_partial shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_reduce_final = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_OPS_REDUCE_FINAL)
                .expect("gmres_ops/reduce_final shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_reduce_final_and_finish_norm = {
            let src = kernel_registry::kernel_source_by_id(
                "",
                KernelId::GMRES_OPS_REDUCE_FINAL_AND_FINISH_NORM,
            )
            .expect("gmres_ops/reduce_final_and_finish_norm shader missing from kernel registry");
            (src.create_pipeline)(device)
        };

        let bgl_vectors = pipeline_spmv.get_bind_group_layout(0);
        let bgl_matrix = pipeline_spmv.get_bind_group_layout(1);
        let bgl_precond = pipeline_spmv.get_bind_group_layout(2);
        let bgl_params = pipeline_spmv.get_bind_group_layout(3);

        let (diag_u, diag_v, diag_p) = match &precond {
            FgmresPrecondBindings::Diag {
                diag_u,
                diag_v,
                diag_p,
            } => (*diag_u, *diag_v, *diag_p),
            FgmresPrecondBindings::DiagWithParams {
                diag_u,
                diag_v,
                diag_p,
                ..
            } => (*diag_u, *diag_v, *diag_p),
            FgmresPrecondBindings::SchurWithParams { diag_u, diag_p, .. } => {
                (*diag_u, *diag_p, *diag_p)
            }
        };

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
            .unwrap_or_else(|err| panic!("FGMRES matrix BG creation failed: {err}"))
        };

        let bg_precond = {
            let registry = ResourceRegistry::new()
                .with_buffer("diag_u", diag_u)
                .with_buffer("diag_v", diag_v)
                .with_buffer("diag_p", diag_p);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES precond BG"),
                &bgl_precond,
                ops_bindings,
                2,
                |name| registry.resolve(name),
            )
            .unwrap_or_else(|err| panic!("FGMRES precond BG creation failed: {err}"))
        };

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
            .unwrap_or_else(|err| panic!("FGMRES params BG creation failed: {err}"))
        };

        let logic_update_src = kernel_registry::kernel_source_by_id(
            "",
            KernelId::GMRES_LOGIC_UPDATE_HESSENBERG_GIVENS,
        )
        .expect("gmres_logic/update_hessenberg_givens shader missing from kernel registry");
        let pipeline_update_hessenberg = (logic_update_src.create_pipeline)(device);
        let pipeline_solve_triangular = {
            let src =
                kernel_registry::kernel_source_by_id("", KernelId::GMRES_LOGIC_SOLVE_TRIANGULAR)
                    .expect("gmres_logic/solve_triangular shader missing from kernel registry");
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
            .unwrap_or_else(|err| panic!("FGMRES logic BG creation failed: {err}"))
        };

        let bg_logic_params = {
            let registry = ResourceRegistry::new()
                .with_buffer("iter_params", &b_iter_params)
                .with_buffer("scalars", &b_scalars);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES logic params BG"),
                &bgl_logic_params,
                logic_update_src.bindings,
                1,
                |name| registry.resolve(name),
            )
            .unwrap_or_else(|err| panic!("FGMRES logic params BG creation failed: {err}"))
        };

        let cgs_calc_src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_CGS_CALC_DOTS)
            .expect("gmres_cgs/calc_dots_cgs shader missing from kernel registry");
        let pipeline_calc_dots_cgs = (cgs_calc_src.create_pipeline)(device);
        let pipeline_reduce_dots_cgs = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_CGS_REDUCE_DOTS)
                .expect("gmres_cgs/reduce_dots_cgs shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_update_w_cgs = {
            let src = kernel_registry::kernel_source_by_id("", KernelId::GMRES_CGS_UPDATE_W)
                .expect("gmres_cgs/update_w_cgs shader missing from kernel registry");
            (src.create_pipeline)(device)
        };

        let bgl_cgs = pipeline_calc_dots_cgs.get_bind_group_layout(0);
        let bg_cgs = {
            let registry = ResourceRegistry::new()
                .with_buffer("params", &b_params)
                .with_buffer("b_basis", &b_basis)
                .with_buffer("b_w", &b_w)
                .with_buffer("b_dot_partial", &b_dot_partial)
                .with_buffer("b_hessenberg", &b_hessenberg);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                &format!("{label_prefix} FGMRES cgs BG"),
                &bgl_cgs,
                cgs_calc_src.bindings,
                0,
                |name| registry.resolve(name),
            )
            .unwrap_or_else(|err| panic!("FGMRES cgs BG creation failed: {err}"))
        };

        Self {
            max_restart,
            n,
            num_cells,
            num_dot_groups,
            basis_stride,
            b_basis,
            z_vectors,
            b_w,
            b_temp,
            b_dot_partial,
            b_scalars,
            b_params,
            b_iter_params,
            b_hessenberg,
            b_givens,
            b_g,
            b_y,
            b_staging_scalar,
            bgl_vectors,
            vector_bindings: ops_bindings,
            bgl_matrix,
            bgl_precond,
            bgl_params,
            bg_matrix,
            bg_precond,
            bg_params,
            bg_logic,
            bg_logic_params,
            bg_cgs,
            pipeline_spmv,
            pipeline_axpy,
            pipeline_axpy_from_y,
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
        }
    }

    pub fn new_from_system(
        device: &wgpu::Device,
        n: u32,
        num_cells: u32,
        max_restart: usize,
        system: LinearSystemView<'_>,
        precond: FgmresPrecondBindings<'_>,
        label_prefix: &str,
    ) -> Self {
        Self::new(
            device,
            n,
            num_cells,
            max_restart,
            system.row_offsets(),
            system.col_indices(),
            system.values(),
            precond,
            label_prefix,
        )
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
            b_basis: &self.b_basis,
            z_vectors: &self.z_vectors,
            b_w: &self.b_w,
            b_temp: &self.b_temp,
            b_dot_partial: &self.b_dot_partial,
            b_scalars: &self.b_scalars,
            b_params: &self.b_params,
            b_iter_params: &self.b_iter_params,
            b_staging_scalar: &self.b_staging_scalar,
            bg_matrix: &self.bg_matrix,
            bg_precond: &self.bg_precond,
            bg_params: &self.bg_params,
            bg_logic: &self.bg_logic,
            bg_logic_params: &self.bg_logic_params,
            bg_cgs: &self.bg_cgs,
            bgl_vectors: &self.bgl_vectors,
            vector_bindings: self.vector_bindings,
            pipeline_spmv: &self.pipeline_spmv,
            pipeline_scale: &self.pipeline_scale,
            pipeline_norm_sq: &self.pipeline_norm_sq,
            pipeline_reduce_final_and_finish_norm: &self.pipeline_reduce_final_and_finish_norm,
            pipeline_update_hessenberg: &self.pipeline_update_hessenberg,
            pipeline_solve_triangular: &self.pipeline_solve_triangular,
            pipeline_axpy_from_y: &self.pipeline_axpy_from_y,
            pipeline_calc_dots_cgs: &self.pipeline_calc_dots_cgs,
            pipeline_reduce_dots_cgs: &self.pipeline_reduce_dots_cgs,
            pipeline_update_w_cgs: &self.pipeline_update_w_cgs,
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

    pub fn z_vectors(&self) -> &[wgpu::Buffer] {
        &self.z_vectors
    }

    pub fn z_buffer(&self, idx: usize) -> &wgpu::Buffer {
        &self.z_vectors[idx]
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

    pub fn pipeline_axpy(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_axpy
    }

    pub fn pipeline_axpy_from_y(&self) -> &wgpu::ComputePipeline {
        &self.pipeline_axpy_from_y
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
        }

        // Pass 2: final reduction + staging copy
        let reduce_params = RawFgmresParams {
            n: self.num_dot_groups(),
            num_cells: 0,
            num_iters: 0,
            omega: 0.0,
            dispatch_x: WORKGROUP_SIZE,
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
}

pub struct FgmresSolveOnceResult {
    pub basis_size: usize,
    pub residual_est: f32,
    pub converged: bool,
}

pub fn workgroups_for_size(n: u32) -> u32 {
    n.div_ceil(WORKGROUP_SIZE)
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
    let (dispatch_x, _) = dispatch_2d(workgroups);
    dispatch_x * WORKGROUP_SIZE
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
    .unwrap_or_else(|err| panic!("{label} creation failed: {err}"))
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

pub fn read_scalar(core: &FgmresCore<'_>) -> f32 {
    let mut encoder = core
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FGMRES read scalar"),
        });
    encoder.copy_buffer_to_buffer(core.b_scalars, 0, core.b_staging_scalar, 0, 4);
    let submission_index = core.queue.submit(Some(encoder.finish()));
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
    rx.recv().ok().and_then(|v| v.ok()).unwrap();

    let data = slice.get_mapped_range();
    let value: f32 = *bytemuck::from_bytes(&data[0..4]);
    drop(data);
    core.b_staging_scalar.unmap();
    value
}

pub fn fgmres_solve_once_with_preconditioner<'a>(
    core: &FgmresCore<'a>,
    x: &'a wgpu::Buffer,
    rhs_norm: f32,
    mut params: RawFgmresParams,
    mut iter_params: IterParams,
    config: FgmresSolveOnceConfig,
    mut precondition: impl FnMut(
        usize,
        &mut wgpu::CommandEncoder,
        wgpu::BindingResource<'a>,
        &'a wgpu::Buffer,
    ),
) -> FgmresSolveOnceResult {
    let n = core.n;
    let vector_bytes = (n as u64) * 4;
    let workgroups = workgroups_for_size(n);
    let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
    let dispatch_x_threads = dispatch_x_threads(workgroups);

    let max_restart = iter_params.max_restart.max(1).min(core.max_restart as u32) as usize;
    iter_params.max_restart = max_restart as u32;

    let tol_abs = config.tol_abs;
    let tol_rel_rhs = config.tol_rel * rhs_norm;

    let mut basis_size = 0usize;
    let mut residual_est = f32::INFINITY;
    let mut converged = false;

    // Ensure vector ops see correct dispatch width and problem size.
    params.n = n;
    params.dispatch_x = dispatch_x_threads;
    params.max_restart = max_restart as u32;
    write_params(core, &params);

    for j in 0..max_restart {
        basis_size = j + 1;

        let z_buf = &core.z_vectors[j];
        let vj = basis_binding(core.b_basis, core.basis_stride, vector_bytes, j);

        // Update solver params for this iteration.
        params.num_iters = j as u32;
        write_params(core, &params);

        // (1) Precondition, then w = A * z_j, then CGS, then norm partial reduction.
        {
            let mut encoder = core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("FGMRES Iter: SpMV+CGS+NormPartial"),
                });

            precondition(j, &mut encoder, vj, z_buf);

            // w = A * z_j
            let spmv_bg = create_vector_bind_group(
                core.device,
                core.bgl_vectors,
                core.vector_bindings,
                z_buf.as_entire_binding(),
                core.b_w.as_entire_binding(),
                core.b_temp.as_entire_binding(),
                "FGMRES SpMV BG",
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES SpMV"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_spmv);
                pass.set_bind_group(0, &spmv_bg, &[]);
                pass.set_bind_group(1, core.bg_matrix, &[]);
                pass.set_bind_group(2, core.bg_precond, &[]);
                pass.set_bind_group(3, core.bg_params, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // CGS (writes H[i,j] for i=0..j)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES CGS Calc"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_calc_dots_cgs);
                pass.set_bind_group(0, core.bg_cgs, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES CGS Reduce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_reduce_dots_cgs);
                pass.set_bind_group(0, core.bg_cgs, &[]);
                pass.dispatch_workgroups((j + 1) as u32, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES CGS Update W"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_update_w_cgs);
                pass.set_bind_group(0, core.bg_cgs, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // Partial reduction of ||w||^2 into dot_partial.
            let norm_bg = create_vector_bind_group(
                core.device,
                core.bgl_vectors,
                core.vector_bindings,
                core.b_w.as_entire_binding(),
                core.b_temp.as_entire_binding(),
                core.b_dot_partial.as_entire_binding(),
                "FGMRES Norm BG",
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES Norm Partial"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_norm_sq);
                pass.set_bind_group(0, &norm_bg, &[]);
                pass.set_bind_group(1, core.bg_matrix, &[]);
                pass.set_bind_group(2, core.bg_precond, &[]);
                pass.set_bind_group(3, core.bg_params, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            core.queue.submit(Some(encoder.finish()));
        }

        // Compute norm of w and write H[j+1,j] plus scalars[0]=1/norm.
        let h_idx = (j as u32) * (max_restart as u32 + 1) + (j as u32 + 1);
        iter_params.current_idx = h_idx;
        write_iter_params(core, &iter_params);

        // Reduce + finish norm (single thread); hack params.n=num_dot_groups for this dispatch.
        let reduce_params = RawFgmresParams {
            n: core.num_dot_groups,
            dispatch_x: WORKGROUP_SIZE,
            ..params
        };
        write_params(core, &reduce_params);

        let reduce_bg = create_vector_bind_group(
            core.device,
            core.bgl_vectors,
            core.vector_bindings,
            core.b_dot_partial.as_entire_binding(),
            core.b_temp.as_entire_binding(),
            core.b_temp.as_entire_binding(),
            "FGMRES Reduce BG",
        );
        dispatch_vector_pipeline(
            core,
            core.pipeline_reduce_final_and_finish_norm,
            &reduce_bg,
            1,
            1,
            "FGMRES Reduce Final & Finish Norm",
        );

        // Restore params for vector ops
        write_params(core, &params);

        // (3) v_{j+1} = (1/||w||) * w, update Hessenberg/Givens.
        iter_params.current_idx = j as u32;
        write_iter_params(core, &iter_params);

        {
            let submission_index = {
                let mut encoder =
                    core.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("FGMRES Iter: Scale+Update"),
                        });

                // v_{j+1} = (1/||w||) * w
                let v_next = basis_binding(core.b_basis, core.basis_stride, vector_bytes, j + 1);
                let scale_bg = create_vector_bind_group(
                    core.device,
                    core.bgl_vectors,
                    core.vector_bindings,
                    core.b_w.as_entire_binding(),
                    v_next,
                    core.b_temp.as_entire_binding(),
                    "FGMRES Normalize Basis BG",
                );
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("FGMRES Normalize & Copy"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(core.pipeline_scale);
                    pass.set_bind_group(0, &scale_bg, &[]);
                    pass.set_bind_group(1, core.bg_matrix, &[]);
                    pass.set_bind_group(2, core.bg_precond, &[]);
                    pass.set_bind_group(3, core.bg_params, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                // Update Hessenberg/Givens and residual estimate.
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

                // IMPORTANT: `update_hessenberg_givens` writes the current residual estimate to
                // `scalars[0]`. We read it back *every iteration* to support early exit.
                //
                // Do not remove this readback without providing an alternative early-exit
                // mechanism (or a benchmarked check interval). Only checking at the end forces
                // `max_restart` iterations even when GMRES converges early, which is a major
                // performance regression for many flows.
                encoder.copy_buffer_to_buffer(core.b_scalars, 0, core.b_staging_scalar, 0, 4);
                core.queue.submit(Some(encoder.finish()))
            };

            residual_est = read_scalar_after_submit(core, submission_index);
        }

        if residual_est <= tol_rel_rhs || residual_est <= tol_abs {
            converged = true;
            break;
        }
    }

    // Solve upper triangular system for y (size=basis_size)
    iter_params.current_idx = basis_size as u32;
    write_iter_params(core, &iter_params);
    dispatch_logic_pipeline(
        core,
        core.pipeline_solve_triangular,
        1,
        "FGMRES Solve Triangular",
    );

    if config.reset_x_before_update {
        let size = x.size() as usize;
        core.queue.write_buffer(x, 0, &vec![0u8; size]);
    }

    // x = x + sum_i y_i * z_i
    for i in 0..basis_size {
        iter_params.current_idx = i as u32;
        write_iter_params(core, &iter_params);
        let axpy_bg = create_vector_bind_group(
            core.device,
            core.bgl_vectors,
            core.vector_bindings,
            core.z_vectors[i].as_entire_binding(),
            x.as_entire_binding(),
            core.b_temp.as_entire_binding(),
            "FGMRES Solution Update BG",
        );
        dispatch_vector_pipeline(
            core,
            core.pipeline_axpy_from_y,
            &axpy_bg,
            dispatch_x,
            dispatch_y,
            "FGMRES Solution Update",
        );
    }

    FgmresSolveOnceResult {
        basis_size,
        residual_est,
        converged,
    }
}
