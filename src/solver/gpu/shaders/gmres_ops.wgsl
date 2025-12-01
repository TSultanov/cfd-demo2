// GMRES/FGMRES GPU Operations
//
// Operations needed for FGMRES:
// 1. SpMV (matrix-vector product) - reuse from linear_solver.wgsl
// 2. Dot product - reuse from dot_product.wgsl
// 3. AXPY: y = alpha*x + y
// 4. Scale: x = alpha*x
// 5. Block preconditioner application (Jacobi smoothing)
// 6. Extract diagonal scaling from matrix

struct GmresParams {
    n: u32,              // Problem size (3 * num_cells)
    num_cells: u32,      // Number of cells
    num_iters: u32,      // Jacobi iterations for preconditioner
    omega: f32,          // Relaxation factor
    dispatch_x: u32,     // Width of 2D dispatch (in threads, i.e. workgroups * 64)
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// Compute linear index from potentially 2D dispatch
fn get_global_index(global_id: vec3<u32>) -> u32 {
    return global_id.x + global_id.y * params.dispatch_x;
}

// Compute linear workgroup ID from potentially 2D dispatch
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    let dispatch_wg_x = params.dispatch_x / 64u;
    return wg_id.x + wg_id.y * dispatch_wg_x;
}

// Group 0: Vectors (for AXPY, scale, copy operations)
@group(0) @binding(0) var<storage, read> vec_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> vec_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> vec_z: array<f32>;

// Group 1: Matrix (CSR format)
@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

// Group 2: Preconditioner data
@group(2) @binding(0) var<storage, read_write> diag_u: array<f32>;
@group(2) @binding(1) var<storage, read_write> diag_v: array<f32>;
@group(2) @binding(2) var<storage, read_write> diag_p: array<f32>;

// Group 3: Parameters and scalars
@group(3) @binding(0) var<uniform> params: GmresParams;
@group(3) @binding(1) var<storage, read_write> scalars: array<f32>;

// SpMV: y = A * x
@compute @workgroup_size(64)
fn spmv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = get_global_index(global_id);
    if (row >= params.n) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * vec_x[col];
    }
    
    vec_y[row] = sum;
}

// AXPY: y = alpha*x + y
@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    if (idx >= params.n) {
        return;
    }
    
    let alpha = scalars[0];
    vec_y[idx] = alpha * vec_x[idx] + vec_y[idx];
}

// AXPBY: z = alpha*x + beta*y
@compute @workgroup_size(64)
fn axpby(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    if (idx >= params.n) {
        return;
    }
    
    let alpha = scalars[0];
    let beta = scalars[1];
    vec_z[idx] = alpha * vec_x[idx] + beta * vec_y[idx];
}

// Scale: y = alpha*x (copy with scaling)
@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    if (idx >= params.n) {
        return;
    }
    
    let alpha = scalars[0];
    vec_y[idx] = alpha * vec_x[idx];
}

// Scale in place: y = alpha * y
@compute @workgroup_size(64)
fn scale_in_place(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    if (idx >= params.n) {
        return;
    }

    let alpha = scalars[0];
    vec_y[idx] = alpha * vec_y[idx];
}

// Copy: y = x
@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    if (idx >= params.n) {
        return;
    }
    
    vec_y[idx] = vec_x[idx];
}

// Block-diagonal preconditioner: z = M^{-1} * x
// Uses diagonal Jacobi for each block (u, v, p)
@compute @workgroup_size(64)
fn block_jacobi_precond(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = get_global_index(global_id);
    if (cell >= params.num_cells) {
        return;
    }
    
    let base = cell * 3u;
    
    // u component: z_u = r_u / diag_u
    let d_u = diag_u[cell];
    if (abs(d_u) > 1e-14) {
        vec_z[base + 0u] = vec_x[base + 0u] / d_u;
    } else {
        vec_z[base + 0u] = 0.0;
    }
    
    // v component: z_v = r_v / diag_v
    let d_v = diag_v[cell];
    if (abs(d_v) > 1e-14) {
        vec_z[base + 1u] = vec_x[base + 1u] / d_v;
    } else {
        vec_z[base + 1u] = 0.0;
    }
    
    // p component: z_p = r_p / diag_p
    let d_p = diag_p[cell];
    if (abs(d_p) > 1e-14) {
        vec_z[base + 2u] = vec_x[base + 2u] / d_p;
    } else {
        vec_z[base + 2u] = 0.0;
    }
}

fn read_diagonal(row: u32) -> f32 {
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    for (var k = start; k < end; k++) {
        if (col_indices[k] == row) {
            return matrix_values[k];
        }
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn extract_block_diagonal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = get_global_index(global_id);
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 3u;
    let row_u = base;
    let row_v = base + 1u;
    let row_p = base + 2u;

    diag_u[cell] = read_diagonal(row_u);
    diag_v[cell] = read_diagonal(row_v);
    diag_p[cell] = read_diagonal(row_p);
}

// Compute squared norm (partial reduction per workgroup)
var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn dot_product_partial(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) wg_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    let lid = local_id.x;
    let wg_idx = get_workgroup_index(wg_id);
    
    // Each thread computes its contribution
    var local_sum = 0.0;
    if (idx < params.n) {
        local_sum = vec_x[idx] * vec_y[idx];
    }
    
    partial_sums[lid] = local_sum;
    workgroupBarrier();
    
    // Reduce within workgroup
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial_sums[lid] += partial_sums[lid + stride];
        }
        workgroupBarrier();
    }
    
    // First thread writes workgroup result
    if (lid == 0u) {
        // Atomic add to global result (stored in scalars[0])
        // Note: WGSL doesn't have atomicAdd for f32 directly, need to use atomicAdd with i32
        // For simplicity, write to per-workgroup output and do final reduction on CPU
        vec_z[wg_idx] = partial_sums[0];
    }
}

// Norm squared: compute ||x||^2 (partial reduction)
@compute @workgroup_size(64)
fn norm_sq_partial(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) wg_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    let lid = local_id.x;
    let wg_idx = get_workgroup_index(wg_id);
    
    var local_sum = 0.0;
    if (idx < params.n) {
        let val = vec_x[idx];
        local_sum = val * val;
    }
    
    partial_sums[lid] = local_sum;
    workgroupBarrier();
    
    // Reduce within workgroup
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial_sums[lid] += partial_sums[lid + stride];
        }
        workgroupBarrier();
    }
    
    // First thread writes workgroup result
    if (lid == 0u) {
        vec_z[wg_idx] = partial_sums[0];
    }
}

// Gram-Schmidt orthogonalization step: w = w - h*v
// where h = <w, v> (computed separately)
@compute @workgroup_size(64) 
fn orthogonalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = get_global_index(global_id);
    if (idx >= params.n) {
        return;
    }
    
    // scalars[0] contains the dot product h = <w, v>
    let h = scalars[0];
    // vec_y is w (to be modified), vec_x is v (orthogonal vector)
    vec_y[idx] = vec_y[idx] - h * vec_x[idx];
}

struct IterParams {
    current_idx: u32, // Target index in Hessenberg matrix
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(3) @binding(2) var<uniform> iter_params: IterParams;
@group(3) @binding(3) var<storage, read_write> hessenberg: array<f32>;

// Final reduction: sums partial results from workgroups and writes to H and scalars
@compute @workgroup_size(1)
fn reduce_final(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Sum up all partial sums
    // We assume the number of workgroups is small enough (e.g. < 1024) 
    // that a single thread can sum them up efficiently.
    // The partial_sums buffer is reused here as input (bound as vec_x or similar, 
    // but actually we need to bind the partial sums buffer specifically).
    
    // Wait, we need to bind the partial sums buffer. 
    // In the Rust code, we will bind `b_dot_partial` to binding 0 (vec_x) of Group 0.
    // So we can read from `vec_x`.
    
    // We need to know how many partial sums there are.
    // This can be passed in `params.n` (reused) or `iter_params`.
    // Let's assume `params.n` holds the number of workgroups for this dispatch.
    
    var total_sum = 0.0;
    let num_partials = params.n; // Hack: we set n to num_dot_groups for this dispatch
    
    for (var i = 0u; i < num_partials; i++) {
        total_sum += vec_x[i];
    }
    
    // Write to scalars[0] for orthogonalize
    scalars[0] = total_sum;
    
    // Write to Hessenberg matrix
    // iter_params.current_idx is the flat index in H
    hessenberg[iter_params.current_idx] = total_sum;
}
