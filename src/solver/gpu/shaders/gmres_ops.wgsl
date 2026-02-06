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
    max_restart: u32,
    column_offset: u32,
    _pad3: u32,
}

const WORKGROUP_SIZE: u32 = 64u;
const SCALAR_STOP: u32 = 8u;

fn global_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * (num_workgroups.x * WORKGROUP_SIZE) + global_id.x;
}

fn workgroup_index(group_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return group_id.y * num_workgroups.x + group_id.x;
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

struct IterParams {
    current_idx: u32,
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(3) @binding(2) var<uniform> iter_params: IterParams;
@group(3) @binding(3) var<storage, read_write> hessenberg: array<f32>;
@group(3) @binding(4) var<storage, read> y_sol: array<f32>;

// SpMV: y = A * x
@compute @workgroup_size(64)
fn spmv(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let row = global_index(global_id, num_workgroups);
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
fn axpy(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    
    let alpha = scalars[0];
    vec_y[idx] = alpha * vec_x[idx] + vec_y[idx];
}

// AXPY from Y: y = y_sol[current_idx] * x + y
@compute @workgroup_size(64)
fn axpy_from_y(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    
    let alpha = y_sol[iter_params.current_idx];
    vec_y[idx] = alpha * vec_x[idx] + vec_y[idx];
}

// AXPBY: z = alpha*x + beta*y
@compute @workgroup_size(64)
fn axpby(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    
    let alpha = scalars[0];
    let beta = scalars[1];
    vec_z[idx] = alpha * vec_x[idx] + beta * vec_y[idx];
}

// Scale: y = alpha*x (copy with scaling)
@compute @workgroup_size(64)
fn scale(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    
    let alpha = scalars[0];
    vec_y[idx] = alpha * vec_x[idx];
}

// Scale in place: y = alpha * y
@compute @workgroup_size(64)
fn scale_in_place(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }

    let alpha = scalars[0];
    vec_y[idx] = alpha * vec_y[idx];
}

// Copy: y = x
@compute @workgroup_size(64)
fn copy(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    
    vec_y[idx] = vec_x[idx];
}


// Compute squared norm (partial reduction per workgroup)
var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn dot_product_partial(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    let lid = local_id.x;
    
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
        let wg_idx = workgroup_index(group_id, num_workgroups);
        let num_groups_n = (params.n + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE;
        if (wg_idx < num_groups_n) {
            vec_z[wg_idx] = partial_sums[0];
        }
    }
}

// Norm squared: compute ||x||^2 (partial reduction)
@compute @workgroup_size(64)
fn norm_sq_partial(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    let lid = local_id.x;
    
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
        let wg_idx = workgroup_index(group_id, num_workgroups);
        let num_groups_n = (params.n + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE;
        if (wg_idx < num_groups_n) {
            vec_z[wg_idx] = partial_sums[0];
        }
    }
}

// Gram-Schmidt orthogonalization step: w = w - h*v
// where h = <w, v> (computed separately)
@compute @workgroup_size(64) 
fn orthogonalize(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }
    
    // scalars[0] contains the dot product h = <w, v>
    let h = scalars[0];
    // vec_y is w (to be modified), vec_x is v (orthogonal vector)
    vec_y[idx] = vec_y[idx] - h * vec_x[idx];
}

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

// Merged Reduce Final + Finish Norm
@compute @workgroup_size(1)
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
    
    // Store in H
    hessenberg[iter_params.current_idx] = norm;
    
    // Store 1/norm for scaling
    if (norm > 1e-20) {
        scalars[0] = 1.0 / norm;
    } else {
        scalars[0] = 0.0;
    }
}

fn safe_inverse(val: f32) -> f32 {
    let abs_val = abs(val);
    if (abs_val > 1e-12) {
        return 1.0 / val;
    }
    if (abs_val > 0.0) {
        return sign(val) * 1.0e12;
    }
    return 0.0;
}

// Extract inverse diagonal values into diag_u (and mirrors into diag_v/diag_p).
//
// This implements a generic Jacobi preconditioner that can be used by model-agnostic
// solver paths without assuming any particular block structure.
@compute @workgroup_size(64)
fn extract_diag_inv(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
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

// Apply Jacobi scaling: y = diag_u * x.
@compute @workgroup_size(64)
fn apply_diag_inv(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }

    vec_y[idx] = diag_u[idx] * vec_x[idx];
}
