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
fn reduce_t_s_t_t(@builtin(local_invocation_id) local_id: vec3<u32>) {
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
        scalars.t_s = scratch1[0];
        scalars.t_t = scratch2[0];
    }
}

@compute @workgroup_size(1)
fn update_beta() {
    if (abs(scalars.omega) < 1e-20) {
        scalars.beta = 0.0;
    } else {
        scalars.beta = (scalars.rho_new / scalars.rho_old) * (scalars.alpha / scalars.omega);
    }
}

@compute @workgroup_size(1)
fn update_alpha() {
    if (abs(scalars.r0_v) < 1e-20) {
        scalars.alpha = 0.0;
    } else {
        scalars.alpha = scalars.rho_new / scalars.r0_v;
    }
}

@compute @workgroup_size(1)
fn update_omega() {
    if (abs(scalars.t_t) < 1e-20) {
        scalars.omega = 0.0;
    } else {
        scalars.omega = scalars.t_s / scalars.t_t;
    }
}

@compute @workgroup_size(1)
fn init_scalars() {
    scalars.rho_old = 1.0;
    scalars.alpha = 1.0;
    scalars.omega = 1.0;
    scalars.beta = 0.0;
}

@compute @workgroup_size(1)
fn update_rho_old() {
    scalars.rho_old = scalars.rho_new;
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

@compute @workgroup_size(1)
fn update_cg_alpha() {
    if (abs(scalars.r0_v) < 1e-20) {
        scalars.alpha = 0.0;
    } else {
        scalars.alpha = scalars.rho_old / scalars.r0_v;
    }
}

@compute @workgroup_size(1)
fn update_cg_beta() {
    if (abs(scalars.rho_old) < 1e-20) {
        scalars.beta = 0.0;
    } else {
        scalars.beta = scalars.rho_new / scalars.rho_old;
    }
}
