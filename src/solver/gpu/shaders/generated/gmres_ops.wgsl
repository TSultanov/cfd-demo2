// GMRES/FGMRES GPU Operations

struct GmresParams {
    n: u32,
    num_cells: u32,
    num_iters: u32,
    omega: f32,
    dispatch_x: u32,
    max_restart: u32,
    column_offset: u32,
    _pad3: u32,
}

struct IterParams {
    current_idx: u32,
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64u;

const SCALAR_STOP: u32 = 8u;

const SCALAR_GUARD_FLAG: u32 = 17u;

fn global_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * (num_workgroups.x * WORKGROUP_SIZE) + global_id.x;
}

fn workgroup_index(group_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return group_id.y * num_workgroups.x + group_id.x;
}

fn safe_inverse(val: f32) -> f32 {
    let abs_val = abs(val);
    if (abs_val > 0.000000000001) {
        return 1.0 / val;
    }
    if (abs_val > 0.0) {
        return sign(val) * 1000000000000.0;
    }
    return 0.0;
}

@group(0) @binding(0) 
var<storage, read> vec_x: array<f32>;

@group(0) @binding(1) 
var<storage, read_write> vec_y: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> vec_z: array<f32>;

@group(1) @binding(0) 
var<storage, read> row_offsets: array<u32>;

@group(1) @binding(1) 
var<storage, read> col_indices: array<u32>;

@group(1) @binding(2) 
var<storage, read> matrix_values: array<f32>;

@group(2) @binding(0) 
var<storage, read_write> diag_u: array<f32>;

@group(2) @binding(1) 
var<storage, read_write> diag_v: array<f32>;

@group(2) @binding(2) 
var<storage, read_write> diag_p: array<f32>;

@group(3) @binding(0) 
var<uniform> params: GmresParams;

@group(3) @binding(1) 
var<storage, read_write> scalars: array<f32>;

@group(3) @binding(2) 
var<uniform> iter_params: IterParams;

@group(3) @binding(3) 
var<storage, read_write> hessenberg: array<f32>;

@group(3) @binding(4) 
var<storage, read> y_sol: array<f32>;

var<workgroup> partial_sums: array<f32, 64>;

@compute
@workgroup_size(64)
fn spmv(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
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
        sum += val * vec_x[col];
    }
    vec_y[row] = sum;
}

@compute
@workgroup_size(64)
fn axpy(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    let alpha = scalars[0u];
    vec_y[idx] = alpha * vec_x[idx] + vec_y[idx];
}

@compute
@workgroup_size(64)
fn axpy_from_y(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    let alpha = y_sol[iter_params.current_idx];
    vec_y[idx] = alpha * vec_x[idx] + vec_y[idx];
}

@compute
@workgroup_size(64)
fn axpby(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    let alpha = scalars[0u];
    let beta = scalars[1u];
    vec_z[idx] = alpha * vec_x[idx] + beta * vec_y[idx];
}

@compute
@workgroup_size(64)
fn scale(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    let alpha = scalars[0u];
    vec_y[idx] = alpha * vec_x[idx];
}

@compute
@workgroup_size(64)
fn scale_in_place(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    let alpha = scalars[0u];
    vec_y[idx] = alpha * vec_y[idx];
}

@compute
@workgroup_size(64)
fn copy(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    vec_y[idx] = vec_x[idx];
}

@compute
@workgroup_size(64)
fn dot_product_partial(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    let lid = local_id.x;
    var local_sum = 0.0;
    if (idx < params.n) {
        local_sum = vec_x[idx] * vec_y[idx];
    }
    partial_sums[lid] = local_sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial_sums[lid] += partial_sums[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let wg_idx = workgroup_index(group_id, num_workgroups);
        let num_groups_n = (params.n + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE;
        if (wg_idx < num_groups_n) {
            vec_z[wg_idx] = partial_sums[0u];
        }
    }
}

@compute
@workgroup_size(64)
fn norm_sq_partial(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    let lid = local_id.x;
    var local_sum = 0.0;
    if (idx < params.n) {
        let val = vec_x[idx];
        local_sum = val * val;
    }
    partial_sums[lid] = local_sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial_sums[lid] += partial_sums[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let wg_idx = workgroup_index(group_id, num_workgroups);
        let num_groups_n = (params.n + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE;
        if (wg_idx < num_groups_n) {
            vec_z[wg_idx] = partial_sums[0u];
        }
    }
}

@compute
@workgroup_size(64)
fn orthogonalize(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    let h = scalars[0u];
    vec_y[idx] = vec_y[idx] - h * vec_x[idx];
}

@compute
@workgroup_size(1)
fn reduce_final(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var total_sum = 0.0;
    let num_partials = params.n;
    for (var i = 0u; i < num_partials; i++) {
        total_sum += vec_x[i];
    }
    scalars[0u] = total_sum;
    hessenberg[iter_params.current_idx] = total_sum;
}

@compute
@workgroup_size(1)
fn reduce_final_and_finish_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        return;
    }
    var total_sum = 0.0;
    let num_partials = params.n;
    for (var i = 0u; i < num_partials; i++) {
        total_sum += vec_x[i];
    }
    let norm = sqrt(total_sum);
    hessenberg[iter_params.current_idx] = norm;
    if (norm > 0.00000000000000000001) {
        scalars[0u] = 1.0 / norm;
    } else {
        scalars[0u] = 0.0;
    }
}

@compute
@workgroup_size(64)
fn extract_diag_inv(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let row = global_index(global_id, num_workgroups);
    if (row >= params.n) {
        return;
    }
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    var diag = 1.0;
    for (var k = start; k < end; k = k + 1u) {
        if (col_indices[k] == row) {
            diag = matrix_values[k];
            break;
        }
    }
    let inv = safe_inverse(diag);
    diag_u[row] = inv;
    diag_v[row] = inv;
    diag_p[row] = inv;
}

@compute
@workgroup_size(64)
fn apply_diag_inv(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    vec_y[idx] = diag_u[idx] * vec_x[idx];
}

@compute
@workgroup_size(64)
fn guard_copy(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    let flag = scalars[SCALAR_GUARD_FLAG];
    if (flag == 1.0) {
        vec_z[idx] = vec_y[idx];
    } else {
        if (flag == 2.0) {
            vec_y[idx] = vec_z[idx];
        }
    }
}
