// Preconditioner Shaders for Coupled Solver
// Implements a 3x3 block-Jacobi preconditioner per control volume to stabilize
// the coupled momentum-continuity system.

// Group 0: Vectors (same layout as linear_solver)
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<storage, read_write> t: array<f32>;

// Group 1: Matrix & Params
@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

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
@group(1) @binding(3) var<storage, read_write> scalars: GpuScalars;

struct SolverParams {
    n: u32,
}
@group(1) @binding(4) var<uniform> params: SolverParams;

// Group 2: Preconditioner data
// block_inv stores 3x3 inverse blocks per cell in row-major order (9 floats)
@group(2) @binding(0) var<storage, read_write> block_inv: array<f32>;
@group(2) @binding(1) var<storage, read_write> p_hat: array<f32>;  // M^{-1} * p
@group(2) @binding(2) var<storage, read_write> s_hat: array<f32>;  // M^{-1} * s
@group(2) @binding(3) var<storage, read_write> precond_rhs: array<f32>;

struct PrecondParams {
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(2) @binding(4) var<uniform> precond_params: PrecondParams;

fn use_s_mode() -> bool {
    return precond_params.mode == 1u;
}

fn read_search_vector(idx: u32) -> f32 {
    if (use_s_mode()) {
        return s[idx];
    }
    return p[idx];
}

fn write_hat(idx: u32, value: f32) {
    if (use_s_mode()) {
        s_hat[idx] = value;
    } else {
        p_hat[idx] = value;
    }
}

fn read_hat(idx: u32) -> f32 {
    if (use_s_mode()) {
        return s_hat[idx];
    }
    return p_hat[idx];
}

fn write_rhs(idx: u32, value: f32) {
    precond_rhs[idx] = value;
}

fn get_matrix_value(row: u32, col: u32) -> f32 {
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    var value = 0.0;
    for (var k = start; k < end; k++) {
        if (col_indices[k] == col) {
            value = matrix_values[k];
            break;
        }
    }
    return value;
}

fn safe_inverse(val: f32) -> f32 {
    let abs_val = abs(val);
    if (abs_val > 1e-12) {
        return 1.0 / val;
    }
    if (abs_val > 0.0) {
        return sign(val) * 1.0e10;
    }
    return 0.0;
}


// Stage 2: build Schur RHS g' = g - D * y_u for current mode
// Stage 2: build Schur RHS g' = g - D * y_u for current mode
@compute @workgroup_size(64)
fn build_schur_rhs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_unknowns = params.n;
    if (total_unknowns < 4u) {
        return;
    }
    let num_cells = total_unknowns / 4u;
    let cell = global_id.x;
    if (cell >= num_cells) {
        return;
    }

    let base = cell * 4u;
    let row = base + 3u; // P row index is 3
    var rhs = read_search_vector(row);

    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        if (col % 4u != 3u) { // If not P col (subtract contribution from U, V, E)
            rhs -= matrix_values[k] * read_hat(col);
        }
    }

    write_hat(base + 0u, 0.0); // Dummy writes? No, these are likely unused or cleared.
    write_hat(base + 1u, 0.0);
    write_hat(base + 2u, 0.0);
    write_rhs(base + 3u, rhs); // Write to precond_rhs[P_idx]
}

// Stage 3: apply velocity correction y_u - A^{-1} * G * y_p after AMG solve
@compute @workgroup_size(64)
fn finalize_precond(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_unknowns = params.n;
    if (total_unknowns < 4u) {
        return;
    }
    let num_cells = total_unknowns / 4u;
    let cell = global_id.x;
    if (cell >= num_cells) {
        return;
    }

    let base = cell * 4u;
    let offset = cell * 16u; // 4x4 block

    let row_u = base + 0u;
    let row_v = base + 1u;
    let row_e = base + 2u;

    var vel_u = read_hat(row_u);
    var vel_v = read_hat(row_v);
    var val_e = read_hat(row_e);

    // Diagonal inverses
    let inv_u = block_inv[offset + 0u]; // (0,0)
    let inv_v = block_inv[offset + 5u]; // (1,1) -> 1*4+1=5
    let inv_e = block_inv[offset + 10u]; // (2,2) -> 2*4+2=10

    // Correct U
    let start_u = row_offsets[row_u];
    let end_u = row_offsets[row_u + 1u];
    for (var k = start_u; k < end_u; k++) {
        let col = col_indices[k];
        if (col % 4u == 3u) { // P col
            vel_u -= inv_u * matrix_values[k] * read_hat(col);
        }
    }

    // Correct V
    let start_v = row_offsets[row_v];
    let end_v = row_offsets[row_v + 1u];
    for (var k = start_v; k < end_v; k++) {
        let col = col_indices[k];
        if (col % 4u == 3u) { // P col
            vel_v -= inv_v * matrix_values[k] * read_hat(col);
        }
    }
    
    // Correct E
    let start_e = row_offsets[row_e];
    let end_e = row_offsets[row_e + 1u];
    for (var k = start_e; k < end_e; k++) {
        let col = col_indices[k];
        if (col % 4u == 3u) { // P col
            val_e -= inv_e * matrix_values[k] * read_hat(col);
        }
    }

    write_hat(row_u, vel_u);
    write_hat(row_v, vel_v);
    write_hat(row_e, val_e);
}

// Preconditioned SpMV: v = A * p_hat (where p_hat = M^{-1} * p)
@compute @workgroup_size(64)
fn spmv_phat_v(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.n) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * p_hat[col];
    }
    
    v[row] = sum;
}

// Preconditioned SpMV: t = A * s_hat (where s_hat = M^{-1} * s)
@compute @workgroup_size(64)
fn spmv_shat_t(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.n) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * s_hat[col];
    }
    
    t[row] = sum;
}

