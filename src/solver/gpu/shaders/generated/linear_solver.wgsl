// Linear Solver (CG)

@group(0) @binding(0) 
var<storage, read_write> x: array<f32>;

@group(0) @binding(1) 
var<storage, read_write> r: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> p: array<f32>;

@group(0) @binding(3) 
var<storage, read_write> v: array<f32>;

@group(1) @binding(0) 
var<storage, read> row_offsets: array<u32>;

@group(1) @binding(1) 
var<storage, read> col_indices: array<u32>;

@group(1) @binding(2) 
var<storage, read> matrix_values: array<f32>;

struct GpuScalars {
    rho_old: f32,
    rho_new: f32,
    alpha: f32,
    beta: f32,
    r0_v: f32,
    r_r: f32,
    stop: f32,
}

@group(1) @binding(3) 
var<storage, read_write> scalars: GpuScalars;

struct SolverParams {
    n: u32,
}

@group(1) @binding(4) 
var<uniform> params: SolverParams;

fn global_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * (num_workgroups.x * 64u) + global_id.x;
}

@compute
@workgroup_size(64)
fn spmv_p_v(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (scalars.stop > 0.5) {
        return;
    }
    let row = global_index(global_id, num_workgroups);
    if (row >= params.n) {
        return;
    }
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * p[col];
    }
    v[row] = sum;
}

@compute
@workgroup_size(64)
fn cg_update_x_r(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (scalars.stop > 0.5) {
        return;
    }
    let idx = global_index(global_id, num_workgroups);
    var alpha = 0.0;
    if (abs(scalars.r0_v) >= 0.00000000000000000001) {
        alpha = scalars.rho_old / scalars.r0_v;
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

@compute
@workgroup_size(64)
fn cg_update_p(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (scalars.stop > 0.5) {
        return;
    }
    let idx = global_index(global_id, num_workgroups);
    var beta = 0.0;
    if (abs(scalars.rho_old) >= 0.00000000000000000001) {
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
