// Preconditioner Shaders for Coupled Solver
// Implements a 3x3 block-Jacobi preconditioner per control volume to stabilize
// the coupled momentum-continuity system.

// Group 0: Vectors (same layout as linear_solver)
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<storage, read_write> t: array<f32>;

// Group 1: Matrix & Params
@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

struct GpuScalars {
    rho_old: f32,
    rho_new: f32,
    alpha: f32,
    beta: f32,
    omega: f32,
    r0_v: f32,
    t_s: f32,
    t_t: f32,
    r_r: f32,
}
@group(1) @binding(3) var<storage, read_write> scalars: GpuScalars;

struct SolverParams {
    n: u32,
}
@group(1) @binding(4) var<uniform> params: SolverParams;

// Group 2: Preconditioner data
// block_inv stores 3x3 inverse blocks per cell in row-major order (9 floats)
@group(2) @binding(0) var<storage, read_write> block_inv: array<f32>;
@group(2) @binding(1) var<storage, read_write> p_hat: array<f32>;  // M^{-1} * p
@group(2) @binding(2) var<storage, read_write> s_hat: array<f32>;  // M^{-1} * s

fn get_matrix_value(row: u32, col: u32) -> f32 {
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    var value = 0.0;
    for (var k = start; k < end; k++) {
        if (col_indices[k] == col) {
            value = matrix_values[k];
            break;
        }
    }
    return value;
}

fn safe_inverse(val: f32) -> f32 {
    let abs_val = abs(val);
    if (abs_val > 1e-12) {
        return 1.0 / val;
    }
    if (abs_val > 0.0) {
        return sign(val) * 1.0e10;
    }
    return 0.0;
}

// Build block-diagonal inverse per cell
// For coupled momentum-pressure system, use simple diagonal scaling
// which is more robust for saddle-point systems
@compute @workgroup_size(64)
fn extract_diagonal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_unknowns = params.n;
    if (total_unknowns < 3u) {
        return;
    }
    let num_cells = total_unknowns / 3u;
    let cell = global_id.x;
    if (cell >= num_cells) {
        return;
    }

    let row_u = 3u * cell + 0u;
    let row_v = row_u + 1u;
    let row_p = row_u + 2u;
    let col_u = row_u;
    let col_v = row_v;
    let col_p = row_p;

    let m00 = get_matrix_value(row_u, col_u);
    let m11 = get_matrix_value(row_v, col_v);
    let m22 = get_matrix_value(row_p, col_p);

    let offset = cell * 9u;
    
    // Simple diagonal (Jacobi) scaling - more robust for saddle-point systems
    // The momentum diagonal is well-conditioned, use its inverse directly
    let inv_diag_u = safe_inverse(m00);
    let inv_diag_v = safe_inverse(m11);
    
    // For pressure, the diagonal may be very small due to saddle-point structure
    // Scale it to have similar magnitude as momentum inverse
    // This acts like a pressure scaling/weighting
    let momentum_scale = 0.5 * (abs(inv_diag_u) + abs(inv_diag_v));
    let raw_inv_p = safe_inverse(m22);
    // Limit pressure inverse to avoid blowup - use geometric mean scaling
    let inv_diag_p = sign(raw_inv_p) * min(abs(raw_inv_p), 10.0 * momentum_scale + 1e-6);
    
    // Store as pure diagonal preconditioner (no off-diagonal coupling)
    block_inv[offset + 0u] = inv_diag_u;
    block_inv[offset + 1u] = 0.0;
    block_inv[offset + 2u] = 0.0;
    block_inv[offset + 3u] = 0.0;
    block_inv[offset + 4u] = inv_diag_v;
    block_inv[offset + 5u] = 0.0;
    block_inv[offset + 6u] = 0.0;
    block_inv[offset + 7u] = 0.0;
    block_inv[offset + 8u] = inv_diag_p;
}

// Apply block-Jacobi preconditioner: p_hat = M^{-1} * p
@compute @workgroup_size(64)
fn apply_precond_p(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_unknowns = params.n;
    if (total_unknowns < 3u) {
        return;
    }
    let num_cells = total_unknowns / 3u;
    let cell = global_id.x;
    if (cell >= num_cells) {
        return;
    }

    let base = cell * 3u;
    let src0 = p[base + 0u];
    let src1 = p[base + 1u];
    let src2 = p[base + 2u];

    let offset = cell * 9u;
    let dst0 = block_inv[offset + 0u] * src0 + block_inv[offset + 1u] * src1 + block_inv[offset + 2u] * src2;
    let dst1 = block_inv[offset + 3u] * src0 + block_inv[offset + 4u] * src1 + block_inv[offset + 5u] * src2;
    let dst2 = block_inv[offset + 6u] * src0 + block_inv[offset + 7u] * src1 + block_inv[offset + 8u] * src2;

    p_hat[base + 0u] = dst0;
    p_hat[base + 1u] = dst1;
    p_hat[base + 2u] = dst2;
}

// Apply block-Jacobi preconditioner: s_hat = M^{-1} * s
@compute @workgroup_size(64)
fn apply_precond_s(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_unknowns = params.n;
    if (total_unknowns < 3u) {
        return;
    }
    let num_cells = total_unknowns / 3u;
    let cell = global_id.x;
    if (cell >= num_cells) {
        return;
    }

    let base = cell * 3u;
    let src0 = s[base + 0u];
    let src1 = s[base + 1u];
    let src2 = s[base + 2u];

    let offset = cell * 9u;
    let dst0 = block_inv[offset + 0u] * src0 + block_inv[offset + 1u] * src1 + block_inv[offset + 2u] * src2;
    let dst1 = block_inv[offset + 3u] * src0 + block_inv[offset + 4u] * src1 + block_inv[offset + 5u] * src2;
    let dst2 = block_inv[offset + 6u] * src0 + block_inv[offset + 7u] * src1 + block_inv[offset + 8u] * src2;

    s_hat[base + 0u] = dst0;
    s_hat[base + 1u] = dst1;
    s_hat[base + 2u] = dst2;
}

// Preconditioned SpMV: v = A * p_hat (where p_hat = M^{-1} * p)
@compute @workgroup_size(64)
fn spmv_phat_v(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.n) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * p_hat[col];
    }
    
    v[row] = sum;
}

// Preconditioned SpMV: t = A * s_hat (where s_hat = M^{-1} * s)
@compute @workgroup_size(64)
fn spmv_shat_t(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.n) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * s_hat[col];
    }
    
    t[row] = sum;
}

// Preconditioned BiCGStab Update X: x = x + alpha * p_hat + omega * s_hat
// (using preconditioned search directions)
@compute @workgroup_size(64)
fn bicgstab_precond_update_x_r(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    var omega = 0.0;
    if (abs(scalars.t_t) >= 1e-20) {
        omega = scalars.t_s / scalars.t_t;
    }

    if (global_id.x == 0u) {
        scalars.omega = omega;
        scalars.rho_old = scalars.rho_new;
    }

    if (idx >= params.n) {
        return;
    }

    let alpha = scalars.alpha;
    // Use preconditioned directions p_hat and s_hat for solution update
    x[idx] += alpha * p_hat[idx] + omega * s_hat[idx];
    r[idx] = s[idx] - omega * t[idx];
}
