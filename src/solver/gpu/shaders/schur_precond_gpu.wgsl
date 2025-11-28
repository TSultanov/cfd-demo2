// Schur Complement Preconditioner GPU Kernels
//
// Block preconditioner for saddle-point system:
// [A   G] [u]   [f]
// [D   C] [p] = [g]

struct PrecondParams {
    n: u32,
    num_cells: u32,
    omega: f32,
    sweep: u32,
}

@group(0) @binding(0) var<storage, read> vec_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> vec_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> vec_temp: array<f32>;

@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

@group(2) @binding(0) var<uniform> params: PrecondParams;
@group(2) @binding(1) var<storage, read_write> diagonals: array<f32>;  // [diag_u, diag_v, diag_p, schur_diag] per cell

// Velocity diagonal preconditioner initialization
@compute @workgroup_size(64)
fn precond_velocity_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) { return; }
    
    let base = cell * 3u;
    let diag_u = diagonals[cell * 4u + 0u];
    let diag_v = diagonals[cell * 4u + 1u];
    
    if (abs(diag_u) > 1e-14) {
        vec_out[base + 0u] = vec_in[base + 0u] / diag_u;
    } else {
        vec_out[base + 0u] = 0.0;
    }
    
    if (abs(diag_v) > 1e-14) {
        vec_out[base + 1u] = vec_in[base + 1u] / diag_v;
    } else {
        vec_out[base + 1u] = 0.0;
    }
}

// Single Gauss-Seidel sweep for velocity (forward or backward based on params.sweep)
// Note: This is inherently sequential, so we use red-black ordering for parallelism
@compute @workgroup_size(64)
fn precond_velocity_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) { return; }
    
    // Red-black ordering for parallel GS
    let is_red = (cell % 2u) == (params.sweep % 2u);
    if (!is_red) { return; }
    
    let base = cell * 3u;
    let row_u = base;
    let row_v = base + 1u;
    
    let diag_u = diagonals[cell * 4u + 0u];
    let diag_v = diagonals[cell * 4u + 1u];
    
    // Compute residuals for u equation
    var res_u = vec_in[row_u];
    let start_u = row_offsets[row_u];
    let end_u = row_offsets[row_u + 1u];
    for (var k = start_u; k < end_u; k++) {
        let col = col_indices[k];
        let col_type = col % 3u;
        if (col != row_u && (col_type == 0u || col_type == 1u)) {
            res_u -= matrix_values[k] * vec_out[col];
        }
    }
    
    // Compute residuals for v equation
    var res_v = vec_in[row_v];
    let start_v = row_offsets[row_v];
    let end_v = row_offsets[row_v + 1u];
    for (var k = start_v; k < end_v; k++) {
        let col = col_indices[k];
        let col_type = col % 3u;
        if (col != row_v && (col_type == 0u || col_type == 1u)) {
            res_v -= matrix_values[k] * vec_out[col];
        }
    }
    
    if (abs(diag_u) > 1e-14) {
        vec_out[row_u] = res_u / diag_u;
    }
    if (abs(diag_v) > 1e-14) {
        vec_out[row_v] = res_v / diag_v;
    }
}

// Compute modified pressure RHS: g' = g - D * z_u
@compute @workgroup_size(64)
fn precond_pressure_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) { return; }
    
    let row_p = cell * 3u + 2u;
    var v_p = vec_in[row_p];
    
    let start = row_offsets[row_p];
    let end = row_offsets[row_p + 1u];
    
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let col_type = col % 3u;
        if (col_type == 0u || col_type == 1u) {
            v_p -= matrix_values[k] * vec_out[col];
        }
    }
    
    vec_temp[cell] = v_p;  // Store modified RHS
}

// Pressure SSOR sweep (red-black parallel)
@compute @workgroup_size(64)
fn precond_pressure_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) { return; }
    
    let is_red = (cell % 2u) == (params.sweep % 2u);
    if (!is_red) { return; }
    
    let row = cell * 3u + 2u;
    let schur_diag = diagonals[cell * 4u + 3u];
    
    if (abs(schur_diag) < 1e-14) { return; }
    
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        if (col != row && col % 3u == 2u) {
            sum += matrix_values[k] * vec_out[col];
        }
    }
    
    let z_new = (vec_temp[cell] - sum) / schur_diag;
    vec_out[row] = (1.0 - params.omega) * vec_out[row] + params.omega * z_new;
}

// Velocity correction: z_u = y_u - A^{-1} * G * z_p
@compute @workgroup_size(64)
fn precond_velocity_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) { return; }
    
    let base = cell * 3u;
    let diag_u = diagonals[cell * 4u + 0u];
    let diag_v = diagonals[cell * 4u + 1u];
    
    // Correct u
    let row_u = base;
    let start_u = row_offsets[row_u];
    let end_u = row_offsets[row_u + 1u];
    
    for (var k = start_u; k < end_u; k++) {
        let col = col_indices[k];
        if (col % 3u == 2u) {
            let p_neighbor = vec_out[col];
            if (abs(diag_u) > 1e-14) {
                vec_out[base] -= (matrix_values[k] * p_neighbor) / diag_u;
            }
        }
    }
    
    // Correct v
    let row_v = base + 1u;
    let start_v = row_offsets[row_v];
    let end_v = row_offsets[row_v + 1u];
    
    for (var k = start_v; k < end_v; k++) {
        let col = col_indices[k];
        if (col % 3u == 2u) {
            let p_neighbor = vec_out[col];
            if (abs(diag_v) > 1e-14) {
                vec_out[base + 1u] -= (matrix_values[k] * p_neighbor) / diag_v;
            }
        }
    }
}
