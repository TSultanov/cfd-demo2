// GMRES Classical Gram-Schmidt (CGS)

struct Params {
    n: u32,
    num_cells: u32,
    num_iters: u32,
    omega: f32,
    dispatch_x: u32,
    max_restart: u32,
    column_offset: u32,
    pad3: u32,
}

const WORKGROUP_SIZE: u32 = 64u;

const SCALAR_STOP: u32 = 8u;

@group(0) @binding(0) 
var<uniform> params: Params;

@group(0) @binding(1) 
var<storage, read> b_basis: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> b_w: array<f32>;

@group(0) @binding(3) 
var<storage, read_write> b_dot_partial: array<f32>;

@group(0) @binding(4) 
var<storage, read_write> b_hessenberg: array<f32>;

@group(0) @binding(5) 
var<storage, read> scalars: array<f32>;

var<workgroup> sdata: array<f32, 64>;

var<workgroup> sdata_vec4: array<vec4<f32>, 64>;

@compute
@workgroup_size(64)
fn calc_dots_cgs(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        return;
    }
    let j = params.num_iters;
    let n = params.n;
    let num_groups_n = (n + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE;
    let stride_x = num_workgroups.x * WORKGROUP_SIZE;
    let idx = global_id.y * stride_x + global_id.x;
    let group_flat = group_id.y * num_workgroups.x + group_id.x;
    if (group_flat >= num_groups_n) {
        return;
    }
    let stride_bytes = n * 4u + 255u & 4294967040u;
    let stride_words = stride_bytes / 4u;
    var w_val = 0.0;
    if (idx < n) {
        w_val = b_w[idx];
    }
    for (var i: u32 = 0u; i <= j; i += 4u) {
        var v = vec4<f32>(0.0);
        if (idx < n) {
            if (i <= j) {
                v.x = b_basis[i * stride_words + idx];
            }
            if (i + 1u <= j) {
                v.y = b_basis[(i + 1u) * stride_words + idx];
            }
            if (i + 2u <= j) {
                v.z = b_basis[(i + 2u) * stride_words + idx];
            }
            if (i + 3u <= j) {
                v.w = b_basis[(i + 3u) * stride_words + idx];
            }
        }
        let prod = v * w_val;
        sdata_vec4[local_id.x] = prod;
        workgroupBarrier();
        if (local_id.x < 32u) {
            sdata_vec4[local_id.x] += sdata_vec4[local_id.x + 32u];
        }
        workgroupBarrier();
        if (local_id.x < 16u) {
            sdata_vec4[local_id.x] += sdata_vec4[local_id.x + 16u];
        }
        workgroupBarrier();
        if (local_id.x < 8u) {
            sdata_vec4[local_id.x] += sdata_vec4[local_id.x + 8u];
        }
        workgroupBarrier();
        if (local_id.x < 4u) {
            sdata_vec4[local_id.x] += sdata_vec4[local_id.x + 4u];
        }
        workgroupBarrier();
        if (local_id.x < 2u) {
            sdata_vec4[local_id.x] += sdata_vec4[local_id.x + 2u];
        }
        workgroupBarrier();
        if (local_id.x < 1u) {
            sdata_vec4[local_id.x] += sdata_vec4[local_id.x + 1u];
            let sum = sdata_vec4[0u];
            if (i <= j) {
                b_dot_partial[i * num_groups_n + group_flat] = sum.x;
            }
            if (i + 1u <= j) {
                b_dot_partial[(i + 1u) * num_groups_n + group_flat] = sum.y;
            }
            if (i + 2u <= j) {
                b_dot_partial[(i + 2u) * num_groups_n + group_flat] = sum.z;
            }
            if (i + 3u <= j) {
                b_dot_partial[(i + 3u) * num_groups_n + group_flat] = sum.w;
            }
        }
        workgroupBarrier();
    }
}

@compute
@workgroup_size(64)
fn reduce_dots_cgs(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        return;
    }
    let i = group_id.x;
    let j = params.num_iters;
    if (i > j || i >= params.max_restart) {
        return;
    }
    let num_groups_n = (params.n + 63u) / 64u;
    var sum = 0.0;
    for (var k: u32 = local_id.x; k < num_groups_n; k += 64u) {
        sum += b_dot_partial[i * num_groups_n + k];
    }
    sdata[local_id.x] = sum;
    workgroupBarrier();
    if (local_id.x < 32u) {
        sdata[local_id.x] += sdata[local_id.x + 32u];
    }
    workgroupBarrier();
    if (local_id.x < 16u) {
        sdata[local_id.x] += sdata[local_id.x + 16u];
    }
    workgroupBarrier();
    if (local_id.x < 8u) {
        sdata[local_id.x] += sdata[local_id.x + 8u];
    }
    workgroupBarrier();
    if (local_id.x < 4u) {
        sdata[local_id.x] += sdata[local_id.x + 4u];
    }
    workgroupBarrier();
    if (local_id.x < 2u) {
        sdata[local_id.x] += sdata[local_id.x + 2u];
    }
    workgroupBarrier();
    if (local_id.x < 1u) {
        sdata[local_id.x] += sdata[local_id.x + 1u];
        let max_restart = params.max_restart;
        let h_idx = j * (max_restart + 1u) + i;
        b_hessenberg[h_idx] = sdata[0u];
    }
}

@compute
@workgroup_size(64)
fn update_w_cgs(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        return;
    }
    let stride_x = num_workgroups.x * WORKGROUP_SIZE;
    let idx = global_id.y * stride_x + global_id.x;
    let j = params.num_iters;
    let n = params.n;
    let max_restart = params.max_restart;
    let stride_bytes = n * 4u + 255u & 4294967040u;
    let stride_words = stride_bytes / 4u;
    if (idx >= n) {
        return;
    }
    var correction = 0.0;
    for (var i: u32 = 0u; i <= j; i += 4u) {
        if (i <= j) {
            let h_val = b_hessenberg[j * (max_restart + 1u) + i];
            let v_val = b_basis[i * stride_words + idx];
            correction += h_val * v_val;
        }
        if (i + 1u <= j) {
            let h_val = b_hessenberg[j * (max_restart + 1u) + (i + 1u)];
            let v_val = b_basis[(i + 1u) * stride_words + idx];
            correction += h_val * v_val;
        }
        if (i + 2u <= j) {
            let h_val = b_hessenberg[j * (max_restart + 1u) + (i + 2u)];
            let v_val = b_basis[(i + 2u) * stride_words + idx];
            correction += h_val * v_val;
        }
        if (i + 3u <= j) {
            let h_val = b_hessenberg[j * (max_restart + 1u) + (i + 3u)];
            let v_val = b_basis[(i + 3u) * stride_words + idx];
            correction += h_val * v_val;
        }
    }
    b_w[idx] = b_w[idx] - correction;
}

@compute
@workgroup_size(64)
fn reduce_dots_cgs_reortho(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        return;
    }
    let i = group_id.x;
    let j = params.num_iters;
    if (i > j || i >= params.max_restart) {
        return;
    }
    let num_groups_n = (params.n + 63u) / 64u;
    var sum = 0.0;
    for (var k: u32 = local_id.x; k < num_groups_n; k += 64u) {
        sum += b_dot_partial[i * num_groups_n + k];
    }
    sdata[local_id.x] = sum;
    workgroupBarrier();
    if (local_id.x < 32u) {
        sdata[local_id.x] += sdata[local_id.x + 32u];
    }
    workgroupBarrier();
    if (local_id.x < 16u) {
        sdata[local_id.x] += sdata[local_id.x + 16u];
    }
    workgroupBarrier();
    if (local_id.x < 8u) {
        sdata[local_id.x] += sdata[local_id.x + 8u];
    }
    workgroupBarrier();
    if (local_id.x < 4u) {
        sdata[local_id.x] += sdata[local_id.x + 4u];
    }
    workgroupBarrier();
    if (local_id.x < 2u) {
        sdata[local_id.x] += sdata[local_id.x + 2u];
    }
    workgroupBarrier();
    if (local_id.x < 1u) {
        sdata[local_id.x] += sdata[local_id.x + 1u];
        let max_restart = params.max_restart;
        let h_idx = j * (max_restart + 1u) + i;
        let d = sdata[0u];
        b_hessenberg[h_idx] += d;
        b_dot_partial[i * num_groups_n] = d;
    }
}

@compute
@workgroup_size(64)
fn update_w_cgs_reortho(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        return;
    }
    let stride_x = num_workgroups.x * WORKGROUP_SIZE;
    let idx = global_id.y * stride_x + global_id.x;
    let j = params.num_iters;
    let n = params.n;
    let max_restart = params.max_restart;
    let stride_bytes = n * 4u + 255u & 4294967040u;
    let stride_words = stride_bytes / 4u;
    let num_groups_n = (n + 63u) / 64u;
    if (idx >= n) {
        return;
    }
    var correction = 0.0;
    for (var i: u32 = 0u; i <= j; i += 4u) {
        if (i <= j) {
            let h_val = b_dot_partial[i * num_groups_n];
            let v_val = b_basis[i * stride_words + idx];
            correction += h_val * v_val;
        }
        if (i + 1u <= j) {
            let h_val = b_dot_partial[(i + 1u) * num_groups_n];
            let v_val = b_basis[(i + 1u) * stride_words + idx];
            correction += h_val * v_val;
        }
        if (i + 2u <= j) {
            let h_val = b_dot_partial[(i + 2u) * num_groups_n];
            let v_val = b_basis[(i + 2u) * stride_words + idx];
            correction += h_val * v_val;
        }
        if (i + 3u <= j) {
            let h_val = b_dot_partial[(i + 3u) * num_groups_n];
            let v_val = b_basis[(i + 3u) * stride_words + idx];
            correction += h_val * v_val;
        }
    }
    b_w[idx] = b_w[idx] - correction;
}
