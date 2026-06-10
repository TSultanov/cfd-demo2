//! Isolated FGMRES parity test: compare `encode_solve_fgmres_fixed_iterations`
//! (GPU-encoded, no host convergence readbacks) against `solve_fgmres` (host-driven
//! with convergence checking) on a small synthetic linear system.
//!
//! The goal is to verify that the encoded path produces numerically equivalent
//! results to the host path, isolating FGMRES-level differences from the
//! higher-level coupled-solver loop.

use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::linear_solver::fgmres::FgmresWorkspace;
use cfd2::solver::gpu::modules::generic_linear_solver::IdentityPreconditioner;
use cfd2::solver::gpu::modules::krylov_precond::DispatchGrids;
use cfd2::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use cfd2::solver::gpu::modules::linear_solver::{
    encode_solve_fgmres_fixed_iterations, solve_fgmres, SolveFgmresArgs,
};
use cfd2::solver::gpu::modules::linear_system::{LinearSystemPorts, LinearSystemView};
use cfd2::solver::gpu::modules::ports::{BufF32, BufU32, PortSpace};
use cfd2::solver::model::linear_solver::FgmresSolutionUpdateStrategy;
use wgpu::util::DeviceExt;

/// Build CSR data for A = tridiag(-1, 3, -1) with b = [1; n].
///
/// The matrix is symmetric positive definite and well-conditioned (condition
/// number < n for the tridiagonal Toeplitz structure), so FGMRES should converge
/// quickly even without preconditioning.
fn build_tridiag_csr(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    let mut offset = 0u32;
    for i in 0..n {
        row_offsets.push(offset);
        if i > 0 {
            col_indices.push((i - 1) as u32);
            values.push(-1.0f32);
            offset += 1;
        }
        col_indices.push(i as u32);
        values.push(3.0f32);
        offset += 1;
        if i + 1 < n {
            col_indices.push((i + 1) as u32);
            values.push(-1.0f32);
            offset += 1;
        }
    }
    row_offsets.push(offset);

    let rhs = vec![1.0f32; n];
    (row_offsets, col_indices, values, rhs)
}

fn device_buffer_f32(device: &wgpu::Device, data: &[f32], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}

fn device_buffer_u32(device: &wgpu::Device, data: &[u32], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}

fn readback_buffer_f32(ctx: &GpuContext, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, (len * 4) as u64);
    let sub_idx = ctx.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(sub_idx),
        timeout: None,
    });
    rx.recv().unwrap().expect("buffer map failed");
    let view = staging.slice(..).get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    result
}

fn create_gpu_linear_system(
    device: &wgpu::Device,
    row_offsets: &[u32],
    col_indices: &[u32],
    values: &[f32],
    rhs: &[f32],
    x: &[f32],
) -> (PortSpace, LinearSystemPorts) {
    let mut port_space = PortSpace::new();

    let p_row_offsets = port_space.port::<BufU32>("linear:row_offsets");
    port_space.insert(
        p_row_offsets,
        device_buffer_u32(device, row_offsets, "row_offsets"),
    );

    let p_col_indices = port_space.port::<BufU32>("linear:col_indices");
    port_space.insert(
        p_col_indices,
        device_buffer_u32(device, col_indices, "col_indices"),
    );

    let p_values = port_space.port::<BufF32>("linear:values");
    port_space.insert(p_values, device_buffer_f32(device, values, "values"));

    let p_rhs = port_space.port::<BufF32>("linear:rhs");
    port_space.insert(p_rhs, device_buffer_f32(device, rhs, "rhs"));

    let p_x = port_space.port::<BufF32>("linear:x");
    port_space.insert(p_x, device_buffer_f32(device, x, "x"));

    let ports = LinearSystemPorts {
        row_offsets: p_row_offsets,
        col_indices: p_col_indices,
        values: p_values,
        rhs: p_rhs,
        x: p_x,
    };

    (port_space, ports)
}

/// Compare `solve_fgmres` (host-driven) vs `encode_solve_fgmres_fixed_iterations`
/// (GPU-encoded, no host readback) on a 16×16 tridiagonal SPD system.
///
/// Both paths should produce the same solution vector within tight tolerance
/// (the matrix is well-conditioned and small, so floating-point differences
/// between the two code paths should be negligible).
#[test]
fn encoded_fgmres_matches_host_fgmres_on_small_system() {
    std::env::set_var("CFD2_QUIET", "1");
    // Clear any tuning knobs that might affect chunk scheduling.
    std::env::remove_var("CFD2_ONE_SUBMISSION_CHUNKS");
    std::env::remove_var("CFD2_ONE_SUBMISSION_RESTART_BUDGET");
    std::env::remove_var("CFD2_ONE_SUBMISSION_TOTAL_ITERS");
    std::env::remove_var("CFD2_ONE_SUBMISSION_MIN_TAIL");

    let n = 16u32;
    let (row_offsets, col_indices, values, rhs) = build_tridiag_csr(n as usize);
    let x_init = vec![0.0f32; n as usize];

    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("gpu context");

    // --- Build the linear system on GPU ---
    let (port_space, ports) = create_gpu_linear_system(
        &ctx.device,
        &row_offsets,
        &col_indices,
        &values,
        &rhs,
        &x_init,
    );
    let system = LinearSystemView {
        ports,
        space: &port_space,
    };

    // --- Diagonal preconditioner buffers (diag(A) = 3, so 1/diag = 1/3) ---
    // Even with IdentityPreconditioner, FgmresWorkspace needs these bound.
    let diag_inv = vec![1.0f32 / 3.0; n as usize];
    let b_diag_u = device_buffer_f32(&ctx.device, &diag_inv, "diag_u");
    let b_diag_v = device_buffer_f32(&ctx.device, &diag_inv, "diag_v");
    let b_diag_p = device_buffer_f32(&ctx.device, &diag_inv, "diag_p");
    let precond_bg = FgmresWorkspace::build_precond_bind_group(
        &ctx.device,
        "test FGMRES precond BG",
        |name| match name {
            "diag_u" => Some(b_diag_u.as_entire_binding()),
            "diag_v" => Some(b_diag_v.as_entire_binding()),
            "diag_p" => Some(b_diag_p.as_entire_binding()),
            _ => None,
        },
    );

    // --- Create FGMRES workspace ---
    let max_restart = n as usize; // Big enough to converge in one restart cycle.
    let fgmres = FgmresWorkspace::new_from_system(
        &ctx.device,
        n,
        n, // num_cells = n (for dispatch sizing)
        max_restart,
        FgmresSolutionUpdateStrategy::FusedContiguous,
        system,
        precond_bg.expect("precond bind group"),
        "test",
    );

    let mut krylov = KrylovSolveModule::new(fgmres.expect("fgmres workspace"), IdentityPreconditioner::new());
    let dispatch = DispatchGrids::for_sizes(n, n);

    // ====== Host-driven path ======
    let stats_host = solve_fgmres(
        &mut krylov,
        SolveFgmresArgs {
            context: &ctx,
            system,
            n,
            num_cells: n,
            dispatch,
            max_restart,
            max_iters: n * 2,
            tol: 1e-10,
            tol_abs: 1e-10,
            precond_label: "test:host",
            use_encoded_seed_basis0: false,
        },
    );
    let x_host = readback_buffer_f32(&ctx, system.x(), n as usize);
    eprintln!(
        "[fgmres_parity] host: iters={} residual={:.3e} converged={}",
        stats_host.iterations, stats_host.residual, stats_host.converged
    );

    // ====== Reset x to zeros ======
    ctx.queue
        .write_buffer(system.x(), 0, bytemuck::cast_slice(&x_init));

    // ====== Encoded path ======
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoded fgmres"),
        });
    let stats_enc = encode_solve_fgmres_fixed_iterations(
        &mut krylov,
        SolveFgmresArgs {
            context: &ctx,
            system,
            n,
            num_cells: n,
            dispatch,
            max_restart,
            max_iters: n * 2,
            tol: 1e-10,
            tol_abs: 1e-10,
            precond_label: "test:encoded",
            use_encoded_seed_basis0: true,
        },
        &mut encoder,
    );
    ctx.queue.submit(Some(encoder.finish()));
    let x_encoded = readback_buffer_f32(&ctx, system.x(), n as usize);
    eprintln!(
        "[fgmres_parity] encoded: iters={} residual={:.3e}",
        stats_enc.iterations, stats_enc.residual
    );

    // ====== Compare element-wise ======
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for i in 0..n as usize {
        let h = x_host[i] as f64;
        let e = x_encoded[i] as f64;
        let abs_diff = (h - e).abs();
        let scale = h.abs().max(e.abs()).max(1e-15);
        let rel = abs_diff / scale;
        if i < 8 || rel > 1e-4 {
            eprintln!(
                "  x[{i:2}]: host={h:+.8e}  encoded={e:+.8e}  abs={abs_diff:.4e}  rel={rel:.4e}"
            );
        }
        max_abs = max_abs.max(abs_diff);
        max_rel = max_rel.max(rel);
    }
    eprintln!("[fgmres_parity] max_abs={max_abs:.4e}  max_rel={max_rel:.4e}");

    // The two paths should agree within tight tolerance on this small
    // well-conditioned system.
    assert!(
        max_rel < 1e-3,
        "encoded vs host FGMRES: max relative error {max_rel:.4e} exceeds 1e-3"
    );
}
