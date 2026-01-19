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

const WORKGROUP_SIZE: u32 = 64u;

fn global_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * (num_workgroups.x * WORKGROUP_SIZE) + global_id.x;
}

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
@compute @workgroup_size(64)
fn build_schur_rhs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total_unknowns = params.n;
    if (total_unknowns < 3u) {
        return;
    }
    let num_cells = total_unknowns / 3u;
    let cell = global_index(global_id, num_workgroups);
    if (cell >= num_cells) {
        return;
    }

    let base = cell * 3u;
    let row = base + 2u;
    var rhs = read_search_vector(row);

    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        if (col % 3u != 2u) {
            rhs -= matrix_values[k] * read_hat(col);
        }
    }

    write_rhs(base + 0u, 0.0);
    write_rhs(base + 1u, 0.0);
    write_rhs(base + 2u, rhs);
}

// Stage 3: apply velocity correction y_u - A^{-1} * G * y_p after AMG solve
@compute @workgroup_size(64)
fn finalize_precond(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total_unknowns = params.n;
    if (total_unknowns < 3u) {
        return;
    }
    let num_cells = total_unknowns / 3u;
    let cell = global_index(global_id, num_workgroups);
    if (cell >= num_cells) {
        return;
    }

    let base = cell * 3u;
    let offset = cell * 9u;

    let row_u = base + 0u;
    let row_v = base + 1u;

    var vel_u = read_hat(row_u);
    var vel_v = read_hat(row_v);

    let inv_u = block_inv[offset + 0u];
    let inv_v = block_inv[offset + 4u];

    let start_u = row_offsets[row_u];
    let end_u = row_offsets[row_u + 1u];
    for (var k = start_u; k < end_u; k++) {
        let col = col_indices[k];
        if (col % 3u == 2u) {
            vel_u -= inv_u * matrix_values[k] * read_hat(col);
        }
    }

    let start_v = row_offsets[row_v];
    let end_v = row_offsets[row_v + 1u];
    for (var k = start_v; k < end_v; k++) {
        let col = col_indices[k];
        if (col % 3u == 2u) {
            vel_v -= inv_v * matrix_values[k] * read_hat(col);
        }
    }

    write_hat(row_u, vel_u);
    write_hat(row_v, vel_v);
}

// Preconditioned SpMV: v = A * p_hat (where p_hat = M^{-1} * p)
@compute @workgroup_size(64)
fn spmv_phat_v(
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
        sum += val * p_hat[col];
    }
    
    v[row] = sum;
}

// Preconditioned SpMV: t = A * s_hat (where s_hat = M^{-1} * s)
@compute @workgroup_size(64)
fn spmv_shat_t(
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
        sum += val * s_hat[col];
    }
    
    t[row] = sum;
}
