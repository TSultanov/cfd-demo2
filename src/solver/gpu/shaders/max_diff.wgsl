// Max Absolute Difference Shader for Convergence Checking
//
// Computes max|a - b| for two buffers, using workgroup reduction.
// Used to check convergence without reading full fields back to CPU.

struct MaxDiffParams {
    n: u32,           // Number of elements to compare
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// Group 0: Input vectors
@group(0) @binding(0) var<storage, read> vec_a: array<f32>;
@group(0) @binding(1) var<storage, read> vec_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_max: array<f32>;

// Group 1: Parameters
@group(1) @binding(0) var<uniform> params: MaxDiffParams;
@group(1) @binding(1) var<storage, read_write> result: array<f32>; // Output scalar(s)

var<workgroup> shared_max: array<f32, 64>;

// Compute max|a - b| with workgroup reduction
@compute @workgroup_size(64)
fn max_abs_diff_partial(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) wg_id: vec3<u32>) {
    let idx = global_id.x;
    let lid = local_id.x;
    
    // Compute local max diff
    var local_max = 0.0;
    if (idx < params.n) {
        local_max = abs(vec_a[idx] - vec_b[idx]);
    }
    
    shared_max[lid] = local_max;
    workgroupBarrier();
    
    // Reduce within workgroup (max operation)
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        workgroupBarrier();
    }
    
    // First thread writes workgroup result
    if (lid == 0u) {
        partial_max[wg_id.x] = shared_max[0];
    }
}

// Final reduction of partial maxes
// Note: For the reduce step, we bind partial results to vec_a (binding 0)
// and write to result (binding 1 of group 1)
@compute @workgroup_size(1)
fn max_reduce_final(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // params.n now holds the number of partial results
    // Partial results are in vec_a (binding 0) for the reduce step
    var total_max = 0.0;
    let num_partials = params.n;
    
    for (var i = 0u; i < num_partials; i++) {
        total_max = max(total_max, vec_a[i]);
    }
    
    result[0] = total_max;
}

// Variant for vec2 (e.g., velocity with u, v components)
// Reads 2 floats per element and returns max of both component diffs
@compute @workgroup_size(64)
fn max_abs_diff_vec2_partial(@builtin(global_invocation_id) global_id: vec3<u32>,
                              @builtin(local_invocation_id) local_id: vec3<u32>,
                              @builtin(workgroup_id) wg_id: vec3<u32>) {
    let cell = global_id.x;
    let lid = local_id.x;
    let num_cells = params.n;
    
    // Compute local max diff for both components
    var local_max = 0.0;
    if (cell < num_cells) {
        let base = cell * 2u;
        let diff_u = abs(vec_a[base] - vec_b[base]);
        let diff_v = abs(vec_a[base + 1u] - vec_b[base + 1u]);
        local_max = max(diff_u, diff_v);
    }
    
    shared_max[lid] = local_max;
    workgroupBarrier();
    
    // Reduce within workgroup
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        workgroupBarrier();
    }
    
    // First thread writes workgroup result
    if (lid == 0u) {
        partial_max[wg_id.x] = shared_max[0];
    }
}

// Combined shader: compute max diff for U (vec2) and P (scalar) in one pass
// Writes two results: result[0] = max_diff_u, result[1] = max_diff_p
// Layout: vec_a = [u0, v0, u1, v1, ...], vec_b = old values
//         partial_max used for U, result buffer used for intermediate P storage
@compute @workgroup_size(64)
fn max_abs_diff_u_p_partial(@builtin(global_invocation_id) global_id: vec3<u32>,
                            @builtin(local_invocation_id) local_id: vec3<u32>,
                            @builtin(workgroup_id) wg_id: vec3<u32>) {
    // This variant expects:
    // vec_a[0..2*num_cells] = current U (interleaved u,v)
    // vec_b[0..2*num_cells] = old U
    // vec_a[2*num_cells..3*num_cells] = current P
    // vec_b[2*num_cells..3*num_cells] = old P
    //
    // Actually, simpler to have separate buffers and dispatches
    // This is just a fallback combining approach
    
    let cell = global_id.x;
    let lid = local_id.x;
    let num_cells = params.n;
    
    var local_max_u = 0.0;
    var local_max_p = 0.0;
    
    if (cell < num_cells) {
        // U is stored as vec2
        let u_base = cell * 2u;
        let diff_u = abs(vec_a[u_base] - vec_b[u_base]);
        let diff_v = abs(vec_a[u_base + 1u] - vec_b[u_base + 1u]);
        local_max_u = max(diff_u, diff_v);
        
        // P offset after all U data
        let p_offset = num_cells * 2u;
        local_max_p = abs(vec_a[p_offset + cell] - vec_b[p_offset + cell]);
    }
    
    // Use shared memory for U reduction
    shared_max[lid] = local_max_u;
    workgroupBarrier();
    
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        workgroupBarrier();
    }
    
    // Write U partial max
    if (lid == 0u) {
        partial_max[wg_id.x * 2u] = shared_max[0];
    }
    
    workgroupBarrier();
    
    // Reuse shared memory for P reduction
    shared_max[lid] = local_max_p;
    workgroupBarrier();
    
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        workgroupBarrier();
    }
    
    // Write P partial max
    if (lid == 0u) {
        partial_max[wg_id.x * 2u + 1u] = shared_max[0];
    }
}

// Final reduction for combined U and P
@compute @workgroup_size(1)
fn max_reduce_u_p_final(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // params.n = number of workgroups
    // partial_max has interleaved U and P maxes
    var max_u = 0.0;
    var max_p = 0.0;
    let num_groups = params.n;
    
    for (var i = 0u; i < num_groups; i++) {
        max_u = max(max_u, partial_max[i * 2u]);
        max_p = max(max_p, partial_max[i * 2u + 1u]);
    }
    
    result[0] = max_u;
    result[1] = max_p;
}
