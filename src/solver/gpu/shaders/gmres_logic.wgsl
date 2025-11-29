// GMRES Logic Shaders (Small system operations)

struct IterParams {
    current_idx: u32, // j (current column index)
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}

// Group 0: Hessenberg and Givens data
@group(0) @binding(0) var<storage, read_write> hessenberg: array<f32>; // Column-major (max_restart+1) * max_restart
@group(0) @binding(1) var<storage, read_write> givens: array<vec2<f32>>; // pairs of (c, s)
@group(0) @binding(2) var<storage, read_write> g_rhs: array<f32>;       // RHS vector g
@group(0) @binding(3) var<storage, read_write> y_sol: array<f32>;       // Solution vector y

// Group 1: Parameters
@group(1) @binding(0) var<uniform> iter_params: IterParams;
@group(1) @binding(1) var<storage, read_write> scalars: array<f32>;     // For outputting residual

fn h_idx(row: u32, col: u32) -> u32 {
    return col * (iter_params.max_restart + 1u) + row;
}

@compute @workgroup_size(1)
fn update_hessenberg_givens(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = iter_params.current_idx;
    
    // 1. Apply previous Givens rotations to the new column H[:, j]
    for (var i = 0u; i < j; i++) {
        let idx_i = h_idx(i, j);
        let idx_i1 = h_idx(i + 1u, j);
        
        let h_ij = hessenberg[idx_i];
        let h_i1j = hessenberg[idx_i1];
        
        let cs = givens[i];
        let c = cs.x;
        let s = cs.y;
        
        hessenberg[idx_i] = c * h_ij + s * h_i1j;
        hessenberg[idx_i1] = -s * h_ij + c * h_i1j;
    }
    
    // 2. Compute new Givens rotation for H[j, j] and H[j+1, j]
    let idx_jj = h_idx(j, j);
    let idx_j1j = h_idx(j + 1u, j);
    
    let h_jj = hessenberg[idx_jj];
    let h_j1j = hessenberg[idx_j1j];
    
    var c = 1.0;
    var s = 0.0;
    var rho = sqrt(h_jj * h_jj + h_j1j * h_j1j);
    
    if (abs(rho) > 1e-20) {
        c = h_jj / rho;
        s = h_j1j / rho;
    }
    
    // Store rotation
    givens[j] = vec2<f32>(c, s);
    
    // Apply rotation to H
    hessenberg[idx_jj] = rho;
    hessenberg[idx_j1j] = 0.0;
    
    // 3. Apply rotation to RHS vector g
    let g_j = g_rhs[j];
    let g_j1 = g_rhs[j + 1u]; // Should be 0 initially
    
    g_rhs[j] = c * g_j + s * g_j1;
    g_rhs[j + 1u] = -s * g_j + c * g_j1;
    
    // 4. Write residual norm to scalars[0] for CPU readback
    scalars[0] = abs(g_rhs[j + 1u]);
}

@compute @workgroup_size(1)
fn solve_triangular(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Solve H * y = g for upper triangular H (size k x k)
    let k = iter_params.current_idx; // This is the basis size (j+1) passed in
    
    // Backward substitution
    // Note: WGSL loops must be bounded by constant or uniform. 
    // We iterate backwards. Since we can't easily do a dynamic reverse loop,
    // we'll iterate i from 0 to k-1, and access index (k - 1 - i).
    
    for (var loop_i = 0u; loop_i < k; loop_i++) {
        let i = k - 1u - loop_i;
        
        var sum = g_rhs[i];
        
        for (var j = i + 1u; j < k; j++) {
            sum -= hessenberg[h_idx(i, j)] * y_sol[j];
        }
        
        let diag = hessenberg[h_idx(i, i)];
        if (abs(diag) > 1e-12) {
            y_sol[i] = sum / diag;
        } else {
            y_sol[i] = 0.0;
        }
    }
}

@compute @workgroup_size(1)
fn copy_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Copy y_sol[current_idx] to scalars[0] for use in AXPY
    let idx = iter_params.current_idx;
    scalars[0] = y_sol[idx];
}

@compute @workgroup_size(1)
fn finish_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // scalars[0] contains norm^2 (from reduce_final)
    // We need to:
    // 1. Compute norm = sqrt(norm^2)
    // 2. Store norm in H[current_idx] (which is H[j+1, j])
    // 3. Store 1/norm in scalars[0] (for scaling w)
    
    let norm_sq = scalars[0];
    let norm = sqrt(norm_sq);
    
    // Store in H
    hessenberg[iter_params.current_idx] = norm;
    
    // Store 1/norm for scaling
    if (norm > 1e-20) {
        scalars[0] = 1.0 / norm;
    } else {
        scalars[0] = 0.0;
    }
}
