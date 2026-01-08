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

pub fn create_vector_bind_group<'a>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    x: wgpu::BindingResource<'a>,
    y: wgpu::BindingResource<'a>,
    z: wgpu::BindingResource<'a>,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x },
            wgpu::BindGroupEntry { binding: 1, resource: y },
            wgpu::BindGroupEntry { binding: 2, resource: z },
        ],
    })
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
    core.queue.submit(Some(encoder.finish()));

    let slice = core.b_staging_scalar.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    let _ = core.device.poll(wgpu::PollType::wait_indefinitely());
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
    mut precondition: impl FnMut(usize, wgpu::BindingResource<'a>, &'a wgpu::Buffer),
) -> FgmresSolveOnceResult {
    let n = core.n;
    let vector_bytes = (n as u64) * 4;
    let workgroups = workgroups_for_size(n);
    let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
    let dispatch_x_threads = dispatch_x_threads(workgroups);

    let tol_abs = config.tol_abs;

    let mut basis_size = 0usize;
    let mut final_residual = f32::INFINITY;
    let mut converged = false;

    // Ensure vector ops see correct dispatch width and problem size.
    params.n = n;
    params.dispatch_x = dispatch_x_threads;
    params.max_restart = core.max_restart as u32;
    write_params(core, &params);

    for j in 0..core.max_restart {
        basis_size = j + 1;

        let z_buf = &core.z_vectors[j];
        let vj = basis_binding(core.b_basis, core.basis_stride, vector_bytes, j);
        precondition(j, vj, z_buf);

        // w = A * z_j
        let spmv_bg = create_vector_bind_group(
            core.device,
            core.bgl_vectors,
            z_buf.as_entire_binding(),
            core.b_w.as_entire_binding(),
            core.b_temp.as_entire_binding(),
            "FGMRES SpMV BG",
        );
        dispatch_vector_pipeline(
            core,
            core.pipeline_spmv,
            &spmv_bg,
            dispatch_x,
            dispatch_y,
            "FGMRES SpMV",
        );

        // CGS (writes H[i,j] for i=0..j)
        params.num_iters = j as u32;
        params.dispatch_x = core.num_dot_groups;
        write_params(core, &params);
        {
            let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FGMRES CGS"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FGMRES CGS Calc"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(core.pipeline_calc_dots_cgs);
                pass.set_bind_group(0, core.bg_cgs, &[]);
                pass.dispatch_workgroups(core.num_dot_groups, 1, 1);
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
                pass.dispatch_workgroups(core.num_dot_groups, 1, 1);
            }
            core.queue.submit(Some(encoder.finish()));
        }

        // Restore vector params
        params.dispatch_x = dispatch_x_threads;
        write_params(core, &params);

        // Compute norm of w and write H[j+1,j] plus scalars[0]=1/norm.
        let h_idx = (j as u32) * (core.max_restart as u32 + 1) + (j as u32 + 1);
        iter_params.current_idx = h_idx;
        write_iter_params(core, &iter_params);

        let norm_bg = create_vector_bind_group(
            core.device,
            core.bgl_vectors,
            core.b_w.as_entire_binding(),
            core.b_temp.as_entire_binding(),
            core.b_dot_partial.as_entire_binding(),
            "FGMRES Norm BG",
        );
        dispatch_vector_pipeline(
            core,
            core.pipeline_norm_sq,
            &norm_bg,
            dispatch_x,
            dispatch_y,
            "FGMRES Norm Partial",
        );

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

        // v_{j+1} = (1/||w||) * w
        let v_next = basis_binding(core.b_basis, core.basis_stride, vector_bytes, j + 1);
        let scale_bg = create_vector_bind_group(
            core.device,
            core.bgl_vectors,
            core.b_w.as_entire_binding(),
            v_next,
            core.b_temp.as_entire_binding(),
            "FGMRES Normalize Basis BG",
        );
        dispatch_vector_pipeline(
            core,
            core.pipeline_scale,
            &scale_bg,
            dispatch_x,
            dispatch_y,
            "FGMRES Normalize & Copy",
        );

        // Update Hessenberg/Givens and residual estimate.
        iter_params.current_idx = j as u32;
        write_iter_params(core, &iter_params);
        dispatch_logic_pipeline(core, core.pipeline_update_hessenberg, 1, "FGMRES Update Hessenberg");

        final_residual = read_scalar(core);
        if final_residual <= config.tol_rel * rhs_norm || final_residual <= tol_abs {
            converged = true;
            break;
        }
    }

    // Solve upper triangular system for y (size=basis_size)
    iter_params.current_idx = basis_size as u32;
    write_iter_params(core, &iter_params);
    dispatch_logic_pipeline(core, core.pipeline_solve_triangular, 1, "FGMRES Solve Triangular");

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
        residual_est: final_residual,
        converged,
    }
}

