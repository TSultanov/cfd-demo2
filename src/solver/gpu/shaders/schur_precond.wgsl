// Schur Complement Preconditioner for Coupled Solver
//
// Implements the SIMPLE-like preconditioner:
// [A   G] [u]   [f]
// [D   C] [p] = [g]
//
// Preconditioner steps:
// 1. Predict Velocity: z_u = D_u^{-1} r_u
// 2. Form Schur RHS:   r_p' = r_p - D z_u
// 3. Solve Pressure:   A_p z_p = r_p'  (Using separate Pressure Matrix)
// 4. Correct Velocity: z_u = z_u - D_u^{-1} G z_p

struct PrecondParams {
    n: u32,
    num_cells: u32,
    omega: f32, // Relaxation factor for pressure
}

// Group 0: Vectors
@group(0) @binding(0) var<storage, read> r_in: array<f32>;       // Input residual
@group(0) @binding(1) var<storage, read_write> z_out: array<f32>; // Output preconditioned vector
@group(0) @binding(2) var<storage, read_write> temp_p: array<f32>; // Temporary pressure vector

// Group 1: Coupled Matrix (CSR) - for form_schur_rhs and correct_velocity
@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

// Group 2: Diagonals (Inverse) + Params
@group(2) @binding(0) var<storage, read_write> diag_u_inv: array<f32>;
@group(2) @binding(1) var<storage, read_write> diag_v_inv: array<f32>;
@group(2) @binding(2) var<storage, read_write> diag_p_inv: array<f32>;
@group(2) @binding(3) var<uniform> params: PrecondParams;

// Group 3: Pressure Matrix (CSR) - for relax_pressure
@group(3) @binding(0) var<storage, read> p_row_offsets: array<u32>;
@group(3) @binding(1) var<storage, read> p_col_indices: array<u32>;
@group(3) @binding(2) var<storage, read> p_matrix_values: array<f32>;

fn get_matrix_value(row: u32, col: u32) -> f32 {
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    for (var k = start; k < end; k++) {
        if (col_indices[k] == col) {
            return matrix_values[k];
        }
    }
    return 0.0;
}

fn safe_inverse(val: f32) -> f32 {
    if (abs(val) > 1e-14) {
        return 1.0 / val;
    }
    return 0.0;
}

// Kernel 1: Extract Diagonals
@compute @workgroup_size(64)
fn extract_diagonals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let row_u = cell * 3u;
    let row_v = row_u + 1u;
    
    let d_u = get_matrix_value(row_u, row_u);
    let d_v = get_matrix_value(row_v, row_v);
    
    diag_u_inv[cell] = safe_inverse(d_u);
    diag_v_inv[cell] = safe_inverse(d_v);
    
    // For pressure, we use the Pressure Matrix (Scalar Laplacian, N x N)
    // The row index in the pressure matrix corresponds directly to the cell index
    let start = p_row_offsets[cell];
    let end = p_row_offsets[cell + 1u];
    var d_p = 0.0;
    for (var k = start; k < end; k++) {
        if (p_col_indices[k] == cell) {
            d_p = p_matrix_values[k];
            break;
        }
    }
    diag_p_inv[cell] = safe_inverse(d_p);
}

// Kernel 2: Predict Velocity
@compute @workgroup_size(64)
fn predict_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 3u;
    
    let r_u = r_in[base + 0u];
    let r_v = r_in[base + 1u];

    z_out[base + 0u] = diag_u_inv[cell] * r_u;
    z_out[base + 1u] = diag_v_inv[cell] * r_v;
    z_out[base + 2u] = 0.0; 
}

// Kernel 3: Form Schur RHS
@compute @workgroup_size(64)
fn form_schur_rhs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 3u;
    let row_p = base + 2u;
    
    var rhs_p = r_in[row_p];

    // Subtract A_{row_p} * z_out (using Coupled Matrix)
    let start = row_offsets[row_p];
    let end = row_offsets[row_p + 1u];

    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        rhs_p -= matrix_values[k] * z_out[col];
    }

    temp_p[cell] = rhs_p;
}

// Kernel 4: Relax Pressure (Jacobi on Scalar Pressure Matrix)
@compute @workgroup_size(64)
fn relax_pressure(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 3u;
    let z_p_old = z_out[base + 2u];

    // Compute A_p * z_p (using Scalar Pressure Matrix)
    // Row index is 'cell'
    let start = p_row_offsets[cell];
    let end = p_row_offsets[cell + 1u];
    
    var sigma = 0.0;
    for (var k = start; k < end; k++) {
        let col_cell = p_col_indices[k]; // This is a cell index, not a coupled index
        
        if (col_cell != cell) {
            // Map cell index to pressure DOF index: col_cell * 3 + 2
            let z_p_neighbor = z_out[col_cell * 3u + 2u];
            sigma += p_matrix_values[k] * z_p_neighbor;
        }
    }

    let d_inv = diag_p_inv[cell];
    let rhs = temp_p[cell]; 

    let z_p_new = d_inv * (rhs - sigma);
    z_out[base + 2u] = mix(z_p_old, z_p_new, params.omega);
}

// Kernel 5: Correct Velocity
@compute @workgroup_size(64)
fn correct_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 3u;
    let row_u = base + 0u;
    let row_v = base + 1u;

    // Correct U (using Coupled Matrix)
    let start_u = row_offsets[row_u];
    let end_u = row_offsets[row_u + 1u];
    var correction_u = 0.0;
    for (var k = start_u; k < end_u; k++) {
        let col = col_indices[k];
        if (col % 3u == 2u) { 
            correction_u += matrix_values[k] * z_out[col];
        }
    }
    z_out[row_u] -= diag_u_inv[cell] * correction_u;

    // Correct V
    let start_v = row_offsets[row_v];
    let end_v = row_offsets[row_v + 1u];
    var correction_v = 0.0;
    for (var k = start_v; k < end_v; k++) {
        let col = col_indices[k];
        if (col % 3u == 2u) { 
            correction_v += matrix_values[k] * z_out[col];
        }
    }
    z_out[row_v] -= diag_v_inv[cell] * correction_v;
}
