// FGMRES GPU Kernels
//
// Core operations for GPU-based FGMRES solver

struct FgmresParams {
    n: u32,
    num_cells: u32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> vec_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> vec_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> vec_z: array<f32>;

@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

@group(2) @binding(0) var<uniform> params: FgmresParams;
@group(2) @binding(1) var<storage, read_write> scalars: array<f32>;

// SpMV: y = A * x
@compute @workgroup_size(64)
fn spmv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.n) { return; }
    
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        sum += matrix_values[k] * vec_x[col_indices[k]];
    }
    
    vec_y[row] = sum;
}

// AXPY: y = alpha * x + y
@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    
    vec_y[idx] = params.alpha * vec_x[idx] + vec_y[idx];
}

// Scale and copy: y = alpha * x
@compute @workgroup_size(64)
fn scale_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    
    vec_y[idx] = params.alpha * vec_x[idx];
}

// Dot product with workgroup reduction
var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn dot_product(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let idx = gid.x;
    let local_idx = lid.x;
    
    // Each thread computes its contribution
    var local_sum = 0.0;
    if (idx < params.n) {
        local_sum = vec_x[idx] * vec_y[idx];
    }
    
    partial_sums[local_idx] = local_sum;
    workgroupBarrier();
    
    // Reduce within workgroup
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (local_idx < stride) {
            partial_sums[local_idx] += partial_sums[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // First thread writes workgroup result
    if (local_idx == 0u) {
        vec_z[wg_id.x] = partial_sums[0];
    }
}

// Norm squared with workgroup reduction
@compute @workgroup_size(64)
fn norm_squared(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let idx = gid.x;
    let local_idx = lid.x;
    
    var local_sum = 0.0;
    if (idx < params.n) {
        let val = vec_x[idx];
        local_sum = val * val;
    }
    
    partial_sums[local_idx] = local_sum;
    workgroupBarrier();
    
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (local_idx < stride) {
            partial_sums[local_idx] += partial_sums[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    if (local_idx == 0u) {
        vec_z[wg_id.x] = partial_sums[0];
    }
}

// Residual: y = x - y (where x is RHS, y contains A*solution)
@compute @workgroup_size(64)
fn compute_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    
    vec_y[idx] = vec_x[idx] - vec_y[idx];
}
