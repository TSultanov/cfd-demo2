struct AmgParams {
    n: u32,
    omega: f32,
    padding: vec2<u32>,
}

@group(0) @binding(0) 
var<storage, read> row_offsets: array<u32>;

@group(0) @binding(1) 
var<storage, read> col_indices: array<u32>;

@group(0) @binding(2) 
var<storage, read> values: array<f32>;

@group(1) @binding(0) 
var<storage, read_write> x: array<f32>;

@group(1) @binding(1) 
var<storage, read_write> b: array<f32>;

@group(1) @binding(2) 
var<uniform> params: AmgParams;

@group(2) @binding(0) 
var<storage, read> op_row_offsets: array<u32>;

@group(2) @binding(1) 
var<storage, read> op_col_indices: array<u32>;

@group(2) @binding(2) 
var<storage, read> op_values: array<f32>;

@group(3) @binding(0) 
var<storage, read_write> coarse_vec: array<f32>;

@group(3) @binding(1) 
var<storage, read> scalars: array<f32>;

fn amg_should_stop() -> bool {
    return scalars[8u] > 0.5;
}

@compute
@workgroup_size(64)
fn smooth_op(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (amg_should_stop()) {
        return;
    }
    let stride_x = num_workgroups.x * 64u;
    let i = global_id.y * stride_x + global_id.x;
    if (i >= params.n) {
        return;
    }
    let start = row_offsets[i];
    let end = row_offsets[i + 1u];
    var sigma = 0.0;
    var diag = 1.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = values[k];
        if (col == i) {
            diag = val;
        } else {
            sigma += val * x[col];
        }
    }
    if (abs(diag) < 0.00000000000001) {
        diag = 1.0;
    }
    let x_new = (b[i] - sigma) / diag;
    x[i] = mix(x[i], x_new, params.omega);
}

@compute
@workgroup_size(64)
fn prolongate_op(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (amg_should_stop()) {
        return;
    }
    let stride_x = num_workgroups.x * 64u;
    let i = global_id.y * stride_x + global_id.x;
    if (i >= params.n) {
        return;
    }
    let start = op_row_offsets[i];
    let end = op_row_offsets[i + 1u];
    var correction = 0.0;
    for (var k = start; k < end; k++) {
        let coarse_idx = op_col_indices[k];
        let val = op_values[k];
        correction += val * coarse_vec[coarse_idx];
    }
    x[i] += correction;
}

@compute
@workgroup_size(64)
fn restrict_residual(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (amg_should_stop()) {
        return;
    }
    let stride_x = num_workgroups.x * 64u;
    let i = global_id.y * stride_x + global_id.x;
    if (i >= params.n) {
        return;
    }
    let start = op_row_offsets[i];
    let end = op_row_offsets[i + 1u];
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let fine_idx = op_col_indices[k];
        let r_val = op_values[k];
        let a_start = row_offsets[fine_idx];
        let a_end = row_offsets[fine_idx + 1u];
        var ax = 0.0;
        for (var j = a_start; j < a_end; j++) {
            ax += values[j] * x[col_indices[j]];
        }
        let fine_r = b[fine_idx] - ax;
        sum += r_val * fine_r;
    }
    coarse_vec[i] = sum;
}

@compute
@workgroup_size(64)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (amg_should_stop()) {
        return;
    }
    let stride_x = num_workgroups.x * 64u;
    let i = global_id.y * stride_x + global_id.x;
    if (i >= params.n) {
        return;
    }
    x[i] = 0.0;
}
