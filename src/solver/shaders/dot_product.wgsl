struct SolverParams {
    n: u32,
}
@group(0) @binding(0) var<uniform> params: SolverParams;

@group(1) @binding(0) var<storage, read_write> dot_result: array<f32>;
@group(1) @binding(1) var<storage, read> dot_a: array<f32>;
@group(1) @binding(2) var<storage, read> dot_b: array<f32>;

var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>) {
    let idx = global_id.x;
    let lid = local_id.x;
    
    var val = 0.0;
    if (idx < params.n) {
        val = dot_a[idx] * dot_b[idx]; 
    }
    
    scratch[lid] = val;
    workgroupBarrier();
    
    // Reduction in shared memory
    for (var i = 32u; i > 0u; i >>= 1u) {
        if (lid < i) {
            scratch[lid] += scratch[lid + i];
        }
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        dot_result[group_id.x] = scratch[0];
    }
}
