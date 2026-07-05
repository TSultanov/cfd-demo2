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
        &ctx.pipeline_cache,
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
        &ctx.pipeline_cache,
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
        tight_budget: false,
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
        tight_budget: false,
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

/// Solve tridiag(-1, 3, -1) x = rhs exactly on the host (Thomas algorithm, f64).
fn solve_tridiag_host(rhs: &[f32]) -> Vec<f64> {
    let n = rhs.len();
    let mut c_prime = vec![0.0f64; n];
    let mut d_prime = vec![0.0f64; n];
    c_prime[0] = -1.0 / 3.0;
    d_prime[0] = rhs[0] as f64 / 3.0;
    for i in 1..n {
        let m = 3.0 - (-1.0) * c_prime[i - 1];
        c_prime[i] = -1.0 / m;
        d_prime[i] = (rhs[i] as f64 - (-1.0) * d_prime[i - 1]) / m;
    }
    let mut x = vec![0.0f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

/// Residual norm ||rhs - A x|| for A = tridiag(-1, 3, -1), computed in f64.
fn tridiag_residual_norm(rhs: &[f32], x: &[f64]) -> f64 {
    let n = rhs.len();
    let mut sum = 0.0f64;
    for i in 0..n {
        let mut ax = 3.0 * x[i];
        if i > 0 {
            ax -= x[i - 1];
        }
        if i + 1 < n {
            ax -= x[i + 1];
        }
        let r = rhs[i] as f64 - ax;
        sum += r * r;
    }
    sum.sqrt()
}

/// Warm-start convergence-scale parity: with a REACHABLE relative tolerance
/// and a warm start where ||r0|| << ||b||, both paths must declare
/// convergence against rel_scale = min(||b||, ||r0||).
///
/// The host loop uses min(||b||, ||r0||); the encoded path computes
/// RHS_NORM = ||b|| on the GPU and (without the clamp_rel_scale kernel) would
/// accept a residual that only beat tol * ||b|| — orders of magnitude looser
/// than the host on near-converged warm starts. This test fails without the
/// gmres_logic/clamp_rel_scale dispatch.
#[test]
fn encoded_fgmres_warm_start_uses_min_b_r0_scale() {
    std::env::set_var("CFD2_QUIET", "1");
    std::env::remove_var("CFD2_ONE_SUBMISSION_CHUNKS");
    std::env::remove_var("CFD2_ONE_SUBMISSION_RESTART_BUDGET");
    std::env::remove_var("CFD2_ONE_SUBMISSION_TOTAL_ITERS");
    std::env::remove_var("CFD2_ONE_SUBMISSION_MIN_TAIL");
    std::env::remove_var("CFD2_FGMRES_STALL_REL");
    std::env::remove_var("CFD2_FGMRES_CGS2");

    let n = 64u32;
    let (row_offsets, col_indices, values, rhs) = build_tridiag_csr(n as usize);

    // Warm start: exact solution plus a small smooth perturbation, sized so
    // that ||r0|| ≈ 1e-3 * ||b||. With tol = 1e-2 the unclamped criterion
    // tol * ||b|| is ABOVE ||r0|| (instant fake convergence); the aligned
    // criterion tol * ||r0|| demands another ~100x reduction.
    let x_exact = solve_tridiag_host(&rhs);
    let rhs_norm = (rhs.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sqrt();
    let perturb = 1.0e-3;
    let x_warm_f64: Vec<f64> = x_exact
        .iter()
        .enumerate()
        .map(|(i, &v)| v + perturb * ((i as f64 * 0.37).sin() + 1.5))
        .collect();
    let x_warm: Vec<f32> = x_warm_f64.iter().map(|&v| v as f32).collect();
    let r0_norm = tridiag_residual_norm(&rhs, &x_warm.iter().map(|&v| v as f64).collect::<Vec<_>>());

    let tol = 1.0e-2f32;
    let aligned_threshold = tol as f64 * r0_norm.min(rhs_norm);
    let unclamped_threshold = tol as f64 * rhs_norm;
    eprintln!(
        "[fgmres_warm] ||b||={rhs_norm:.4e} ||r0||={r0_norm:.4e} aligned_thr={aligned_threshold:.4e} unclamped_thr={unclamped_threshold:.4e}"
    );
    // Construction sanity: the warm start must sit between the two criteria,
    // otherwise the test cannot discriminate.
    assert!(
        r0_norm < unclamped_threshold && r0_norm > aligned_threshold * 2.0,
        "warm start does not discriminate the two scales"
    );

    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("gpu context");
    let (port_space, ports) = create_gpu_linear_system(
        &ctx.device,
        &row_offsets,
        &col_indices,
        &values,
        &rhs,
        &x_warm,
    );
    let system = LinearSystemView {
        ports,
        space: &port_space,
    };

    let diag_inv = vec![1.0f32 / 3.0; n as usize];
    let b_diag_u = device_buffer_f32(&ctx.device, &diag_inv, "diag_u");
    let b_diag_v = device_buffer_f32(&ctx.device, &diag_inv, "diag_v");
    let b_diag_p = device_buffer_f32(&ctx.device, &diag_inv, "diag_p");
    let precond_bg = FgmresWorkspace::build_precond_bind_group(
        &ctx.device,
        &ctx.pipeline_cache,
        "test FGMRES precond BG",
        |name| match name {
            "diag_u" => Some(b_diag_u.as_entire_binding()),
            "diag_v" => Some(b_diag_v.as_entire_binding()),
            "diag_p" => Some(b_diag_p.as_entire_binding()),
            _ => None,
        },
    );

    let max_restart = 32usize;
    let fgmres = FgmresWorkspace::new_from_system(
        &ctx.device,
        &ctx.pipeline_cache,
        n,
        n,
        max_restart,
        FgmresSolutionUpdateStrategy::FusedContiguous,
        system,
        precond_bg.expect("precond bind group"),
        "test",
    );
    let mut krylov =
        KrylovSolveModule::new(fgmres.expect("fgmres workspace"), IdentityPreconditioner::new());
    let dispatch = DispatchGrids::for_sizes(n, n);

    // ====== Encoded path from the warm start ======
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoded fgmres warm"),
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
            tol,
            tol_abs: 1e-12,
            precond_label: "test:encoded-warm",
            use_encoded_seed_basis0: true,
        tight_budget: false,
        },
        &mut encoder,
    );
    ctx.queue.submit(Some(encoder.finish()));
    let x_enc = readback_buffer_f32(&ctx, system.x(), n as usize);
    let enc_true_residual =
        tridiag_residual_norm(&rhs, &x_enc.iter().map(|&v| v as f64).collect::<Vec<_>>());
    eprintln!(
        "[fgmres_warm] encoded: iters={} est={:.3e} true_resid={:.3e}",
        stats_enc.iterations, stats_enc.residual, enc_true_residual
    );

    // ====== Host path from the same warm start ======
    ctx.queue
        .write_buffer(system.x(), 0, bytemuck::cast_slice(&x_warm));
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
            tol,
            tol_abs: 1e-12,
            precond_label: "test:host-warm",
            use_encoded_seed_basis0: false,
        tight_budget: false,
        },
    );
    let x_host = readback_buffer_f32(&ctx, system.x(), n as usize);
    let host_true_residual =
        tridiag_residual_norm(&rhs, &x_host.iter().map(|&v| v as f64).collect::<Vec<_>>());
    eprintln!(
        "[fgmres_warm] host: iters={} est={:.3e} true_resid={:.3e} converged={}",
        stats_host.iterations, stats_host.residual, host_true_residual, stats_host.converged
    );

    // Allow slack for the f32 Givens estimate vs the f64 true residual.
    let slack = 1.5f64;
    assert!(
        enc_true_residual <= aligned_threshold * slack,
        "encoded path stopped at true residual {enc_true_residual:.4e} > aligned threshold {aligned_threshold:.4e} — rel-scale clamp not applied (criterion was tol*||b||?)"
    );
    assert!(
        host_true_residual <= aligned_threshold * slack,
        "host path stopped at true residual {host_true_residual:.4e} > aligned threshold {aligned_threshold:.4e}"
    );
    assert!(stats_host.converged, "host path failed to converge");
}
