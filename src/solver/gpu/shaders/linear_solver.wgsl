// Linear Solver (CG)

// Group 0: Mesh (Not used here)

// Group 1: Vectors
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;

// Group 2: Matrix & Params
@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

struct GpuScalars {
    rho_old: f32,
    rho_new: f32,
    alpha: f32,
    beta: f32,
    r0_v: f32,
    r_r: f32,
}
@group(1) @binding(3) var<storage, read_write> scalars: GpuScalars;

struct SolverParams {
    n: u32,
}
@group(1) @binding(4) var<uniform> params: SolverParams;

const WORKGROUP_SIZE: u32 = 64u;

fn global_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * (num_workgroups.x * WORKGROUP_SIZE) + global_id.x;
}

// SpMV: v = A * p
@compute @workgroup_size(64)
fn spmv_p_v(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let row = global_index(global_id, num_workgroups);
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

// CG Update X and R: x = x + alpha * p, r = r - alpha * v
@compute @workgroup_size(64)
fn cg_update_x_r(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);

    var alpha = 0.0;
    if (abs(scalars.r0_v) >= 1e-20) {
        alpha = scalars.rho_old / scalars.r0_v; // r0_v holds p.v
    }

    if (idx == 0u) {
        scalars.alpha = alpha;
    }

    if (idx >= params.n) {
        return;
    }

    x[idx] += alpha * p[idx];
    r[idx] -= alpha * v[idx];
}

// CG Update P: p = r + beta * p
@compute @workgroup_size(64)
fn cg_update_p(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);

    var beta = 0.0;
    if (abs(scalars.rho_old) >= 1e-20) {
        beta = scalars.rho_new / scalars.rho_old;
    }

    if (idx == 0u) {
        scalars.beta = beta;
        // Update rho_old for next iteration
        scalars.rho_old = scalars.rho_new;
    }

    if (idx >= params.n) {
        return;
    }

    p[idx] = r[idx] + beta * p[idx];
}
