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

// Final reduction of partial maxes for both velocity (vec_a) and pressure (vec_b)
// Both reductions now occur in a single dispatch to minimize GPU submissions.
@compute @workgroup_size(1)
fn max_reduce_final(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_partials = params.n;
    var total_max_u = 0.0;
    var total_max_p = 0.0;

    for (var i = 0u; i < num_partials; i++) {
        total_max_u = max(total_max_u, vec_a[i]);
        total_max_p = max(total_max_p, vec_b[i]);
    }

    result[0] = total_max_u;
    result[1] = total_max_p;
}

