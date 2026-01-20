struct GpuScalars {
    rho_old: f32,
    rho_new: f32,
    alpha: f32,
    beta: f32,
    r0_v: f32,
    r_r: f32,
}

@group(0) @binding(0) var<storage, read_write> scalars: GpuScalars;
@group(0) @binding(1) var<storage, read> dot_result_1: array<f32>;
@group(0) @binding(2) var<storage, read> dot_result_2: array<f32>;

struct ReduceParams {
    n: u32,
    num_groups: u32,
}
@group(0) @binding(3) var<uniform> params: ReduceParams;

var<workgroup> scratch1: array<f32, 64>;
var<workgroup> scratch2: array<f32, 64>;

@compute @workgroup_size(64)
fn reduce_rho_new_r_r(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = params.num_groups;
    let lid = local_id.x;
    
    var sum1 = 0.0;
    var sum2 = 0.0;
    
    for (var i = lid; i < n; i += 64u) {
        sum1 += dot_result_1[i];
        sum2 += dot_result_2[i];
    }
    
    scratch1[lid] = sum1;
    scratch2[lid] = sum2;
    workgroupBarrier();
    
    for (var i = 32u; i > 0u; i >>= 1u) {
        if (lid < i) {
            scratch1[lid] += scratch1[lid + i];
            scratch2[lid] += scratch2[lid + i];
        }
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        scalars.rho_new = scratch1[0];
        scalars.r_r = scratch2[0];
    }
}

@compute @workgroup_size(64)
fn reduce_r0_v(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = params.num_groups;
    let lid = local_id.x;
    
    var sum = 0.0;
    for (var i = lid; i < n; i += 64u) {
        sum += dot_result_1[i];
    }
    
    scratch1[lid] = sum;
    workgroupBarrier();
    
    for (var i = 32u; i > 0u; i >>= 1u) {
        if (lid < i) {
            scratch1[lid] += scratch1[lid + i];
        }
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        scalars.r0_v = scratch1[0];
    }
}

@compute @workgroup_size(64)
fn init_cg_scalars(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = params.num_groups;
    let lid = local_id.x;
    
    var sum = 0.0;
    for (var i = lid; i < n; i += 64u) {
        sum += dot_result_1[i];
    }
    
    scratch1[lid] = sum;
    workgroupBarrier();
    
    for (var i = 32u; i > 0u; i >>= 1u) {
        if (lid < i) {
            scratch1[lid] += scratch1[lid + i];
        }
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        scalars.rho_old = scratch1[0];
        scalars.alpha = 0.0;
        scalars.beta = 0.0;
    }
}
