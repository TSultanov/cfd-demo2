struct Params {
    n: u32,
    num_cells: u32,
    num_iters: u32, // 'j'
    omega: f32,
    dispatch_x: u32,
    max_restart: u32,
    column_offset: u32,
    pad3: u32,
}

// Group 0: Vectors
@group(0) @binding(0) var<storage, read> vec_z: array<f32>; // Input z_j
@group(0) @binding(1) var<storage, read_write> vec_w: array<f32>; // Output w

// Group 1: Matrix
@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

// Group 2: CGS Basis
@group(2) @binding(0) var<uniform> params: Params;
@group(2) @binding(1) var<storage, read> b_basis: array<f32>;
@group(2) @binding(2) var<storage, read_write> b_dot_partial: array<f32>;

var<workgroup> sdata: array<f32, 3264>; // 64 * 51 floats = 3264 floats (13KB)

const WORKGROUP_SIZE: u32 = 64u;

// Compute linear index from potentially 2D dispatch
fn get_global_index(global_id: vec3<u32>) -> u32 {
    return global_id.x + global_id.y * params.dispatch_x;
}

@compute @workgroup_size(64)
fn spmv_cgs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let row = get_global_index(global_id);
    let lid = local_id.x;
    let j = params.num_iters;
    let n = params.n;
    
    // 1. SpMV
    var val_w = 0.0;
    if (row < n) {
        let start = row_offsets[row];
        let end = row_offsets[row + 1u];
        
        for (var k = start; k < end; k++) {
            let col = col_indices[k];
            let val = matrix_values[k];
            val_w += val * vec_z[col];
        }
        
        // Write w
        vec_w[row] = val_w;
    }

    // 2. CGS Partial Dots
    // We need to compute dot(w, V_k) for k in 0..=j
    
    // Calculate stride (aligned to 256 bytes)
    // (n * 4 + 255) & !255 / 4
    let stride_bytes = (n * 4u + 255u) & 4294967040u;
    let stride_words = stride_bytes / 4u;

    // Each thread computes contribution to j+1 dot products
    // We store them in shared memory.
    // sdata layout: [lid * 51 + k]
    // Max j is 50. So 51 values per thread.
    
    if (row < n) {
        for (var k = 0u; k <= j; k++) {
            let val_v = b_basis[k * stride_words + row];
            sdata[lid * 51u + k] = val_w * val_v;
        }
    } else {
        for (var k = 0u; k <= j; k++) {
            sdata[lid * 51u + k] = 0.0;
        }
    }
    
    workgroupBarrier();

    // Reduction in shared memory
    // We reduce 64 threads into 1.
    // We do this for all k in parallel?
    // No, we can loop k inside the reduction loop?
    // Or loop reduction steps?
    
    // Standard reduction:
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (lid < s) {
            for (var k = 0u; k <= j; k++) {
                sdata[lid * 51u + k] += sdata[(lid + s) * 51u + k];
            }
        }
        workgroupBarrier();
    }
    
    // Thread 0 writes results to global memory
    if (lid == 0u) {
        // We need to write to b_dot_partial
        // Layout: [k * num_groups + group_id]
        // Note: group_id needs to be linear index if dispatch is 2D
        let dispatch_wg_x = params.dispatch_x / 64u;
        let gid = group_id.x + group_id.y * dispatch_wg_x;
        
        // Total groups from params (we can't get total groups easily from builtin if 2D)
        // But we know b_dot_partial size is based on total groups.
        // We need to know total groups to index correctly?
        // Wait, b_dot_partial is [num_groups * (max_restart+1)].
        // Indexing: k * num_groups + gid.
        // We need num_groups.
        // We can pass it in params? params.dispatch_x is width in threads.
        // params.n is vector size.
        // num_groups = ceil(n / 64).
        let total_groups = (n + 63u) / 64u;
        
        for (var k = 0u; k <= j; k++) {
            b_dot_partial[k * total_groups + gid] = sdata[k];
        }
    }
}
