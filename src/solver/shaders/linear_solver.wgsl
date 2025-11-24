// Linear Solver (CG/BiCGStab)

// Group 0: Mesh (Not used here)

// Group 1: Vectors
@group(1) @binding(0) var<storage, read_write> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> r: array<f32>;
@group(1) @binding(2) var<storage, read_write> p: array<f32>;
@group(1) @binding(3) var<storage, read_write> v: array<f32>;
@group(1) @binding(4) var<storage, read_write> s: array<f32>;
@group(1) @binding(5) var<storage, read_write> t: array<f32>;

// Group 2: Matrix & Params
@group(2) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(2) @binding(1) var<storage, read> col_indices: array<u32>;
@group(2) @binding(2) var<storage, read> matrix_values: array<f32>;

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
@group(2) @binding(3) var<storage, read_write> scalars: GpuScalars;

struct SolverParams {
    n: u32,
}
@group(2) @binding(4) var<uniform> params: SolverParams;

// SpMV: v = A * p (or t = A * s)
// We need to specify which vectors to use.
// But bindings are fixed.
// We can use different pipelines or just swap bindings in bind group?
// Swapping bindings is expensive (creating new bind groups).
// Better to have separate kernels or use different bindings.
// But SpMV is generic: y = A * x.
// The current SpMV uses p as input and q (v) as output.
// We need: v = A * p AND t = A * s.
// So we need two SpMV kernels or one generic one where we can select vectors?
// WGSL doesn't support pointers to buffers.
// So we need two kernels: spmv_p_v and spmv_s_t.

// SpMV: v = A * p
@compute @workgroup_size(64)
fn spmv_p_v(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
        sum += val * p[col];
    }
    
    v[row] = sum;
}

// SpMV: t = A * s
@compute @workgroup_size(64)
fn spmv_s_t(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
        sum += val * s[col];
    }
    
    t[row] = sum;
}

// BiCGStab Update P: p = r + beta * (p - omega * v)
@compute @workgroup_size(64)
fn bicgstab_update_p(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    var beta = 0.0;
    if (abs(scalars.omega) >= 1e-20) {
        beta = (scalars.rho_new / scalars.rho_old) * (scalars.alpha / scalars.omega);
    }

    if (global_id.x == 0u) {
        scalars.beta = beta;
    }

    if (idx >= params.n) {
        return;
    }

    p[idx] = r[idx] + beta * (p[idx] - scalars.omega * v[idx]);
}

// BiCGStab Update S: s = r - alpha * v
@compute @workgroup_size(64)
fn bicgstab_update_s(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    var alpha = 0.0;
    if (abs(scalars.r0_v) >= 1e-20) {
        alpha = scalars.rho_new / scalars.r0_v;
    }

    if (global_id.x == 0u) {
        scalars.alpha = alpha;
    }

    if (idx >= params.n) {
        return;
    }

    s[idx] = r[idx] - alpha * v[idx];
}

// BiCGStab Update X and R: x = x + alpha * p + omega * s, r = s - omega * t
@compute @workgroup_size(64)
fn bicgstab_update_x_r(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    x[idx] += alpha * p[idx] + omega * s[idx];
    r[idx] = s[idx] - omega * t[idx];
}
