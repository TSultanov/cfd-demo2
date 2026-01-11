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
    u0: u32,
    u1: u32,
    p: u32,
    _pad0: u32,
    _pad1: u32,
}

// Group 0: Vectors
@group(0) @binding(0) var<storage, read> r_in: array<f32>;       // Input residual
@group(0) @binding(1) var<storage, read_write> z_out: array<f32>; // Output preconditioned vector
@group(0) @binding(2) var<storage, read_write> temp_p: array<f32>; // Temporary pressure vector
@group(0) @binding(3) var<storage, read_write> p_sol: array<f32>; // Pressure solution (contiguous)
@group(0) @binding(4) var<storage, read_write> p_prev: array<f32>; // Previous pressure (Chebyshev)

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

fn safe_inverse(val: f32) -> f32 {
    if (abs(val) > 1e-14) {
        return 1.0 / val;
    }
    return 0.0;
}

// Kernel 4: Relax Pressure (Chebyshev / SOR)
// Reads p_sol (x_k) and p_prev (x_{k-1})
// Writes x_{k+1} to p_prev (which becomes the new current/next state)
@compute @workgroup_size(64)
fn relax_pressure(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    // p_sol is binding 3 (Current, x_k)
    // p_prev is binding 4 (Previous, x_{k-1}) -> Output Target
    
    // In ping-pong:
    // BG A: B3=Sol, B4=Prev.  (Read Sol/Prev, Write Prev).  Prev becomes x_{k+1}.
    // BG B: B3=Prev, B4=Sol.  (Read Prev/Sol, Write Sol).  Sol becomes x_{k+2}.
    
    // Compute Sigma using p_sol (Current Neighbors)
    let start = p_row_offsets[cell];
    let end = p_row_offsets[cell + 1u];
    
    var sigma = 0.0;
    for (var k = start; k < end; k++) {
        let col_cell = p_col_indices[k];
        if (col_cell != cell) {
            sigma += p_matrix_values[k] * p_sol[col_cell];
        }
    }

    let d_inv = diag_p_inv[cell];
    let rhs = temp_p[cell]; 
    
    let hat_x = d_inv * (rhs - sigma); // Jacobi Prediction
    let x_prev = p_prev[cell];
    
    // Chebyshev / SOR update
    // x_{k+1} = (1 - omega) * x_{prev} + omega * hat_x
    // If omega=1, x_{k+1} = hat_x (Jacobi)
    
    let x_new = mix(x_prev, hat_x, params.omega);
    p_prev[cell] = x_new;
}

// Kernel 5: Correct Velocity
@compute @workgroup_size(64)
fn correct_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 3u;
    let row_u = base + params.u0;
    let row_v = base + params.u1;

    // Use p_sol (binding 3) as the pressure field.
    // Ensure the final result of relaxation is in the buffer bound to Binding 3.
    let p_val = p_sol[cell];

    // Correct U (using Coupled Matrix)
    let start_u = row_offsets[row_u];
    let end_u = row_offsets[row_u + 1u];
    var correction_u = 0.0;
    for (var k = start_u; k < end_u; k++) {
        let col = col_indices[k];
        if (col % 3u == params.p) { 
            let p_cell = col / 3u;
            // correction_u += matrix_values[k] * p_sol[p_cell]; 
            // We need random access to p_sol. 
            // Note: p_sol is binding 3.
            correction_u += matrix_values[k] * p_sol[p_cell];
        }
    }
    z_out[row_u] -= diag_u_inv[cell] * correction_u;

    // Correct V
    let start_v = row_offsets[row_v];
    let end_v = row_offsets[row_v + 1u];
    var correction_v = 0.0;
    for (var k = start_v; k < end_v; k++) {
        let col = col_indices[k];
        if (col % 3u == params.p) { 
            let p_cell = col / 3u;
            correction_v += matrix_values[k] * p_sol[p_cell];
        }
    }
    z_out[row_v] -= diag_v_inv[cell] * correction_v;
    
    // Update z_out pressure component
    z_out[base + params.p] = p_val;
}

// Kernel 6: Merged Predict Velocity + Form Schur RHS
@compute @workgroup_size(64)
fn predict_and_form_schur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    // Part 1: Predict Velocity (Local)
    let base = cell * 3u;
    let row_u = base + params.u0;
    let row_v = base + params.u1;
    let row_p = base + params.p;

    let r_u = r_in[row_u];
    let r_v = r_in[row_v];

    z_out[row_u] = diag_u_inv[cell] * r_u;
    z_out[row_v] = diag_v_inv[cell] * r_v;
    z_out[row_p] = 0.0;

    // Part 2: Form Schur RHS
    var rhs_p = r_in[row_p];

    let start = row_offsets[row_p];
    let end = row_offsets[row_p + 1u];

    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let rem = col % 3u;
        
        var z_val = 0.0;
        if (rem == params.u0) {
            let c = col / 3u;
            z_val = r_in[col] * diag_u_inv[c];
        } else if (rem == params.u1) {
            let c = col / 3u;
            z_val = r_in[col] * diag_v_inv[c];
        }

        rhs_p -= matrix_values[k] * z_val;
    }

    temp_p[cell] = rhs_p;
    
    // Initialize p_sol with first Jacobi step
    p_sol[cell] = diag_p_inv[cell] * rhs_p;
    
    // Initialize p_prev (Binding 4) to 0.0 for Chebyshev start
    p_prev[cell] = 0.0;
}
