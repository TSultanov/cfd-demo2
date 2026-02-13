// Generic Schur Complement Preconditioner for Coupled Solver

struct PrecondParams {
    n: u32,
    num_cells: u32,
    omega: f32,
    unknowns_per_cell: u32,
    p: u32,
    u_len: u32,
    _pad0: u32,
    _pad1: u32,
    u0123: vec4<u32>,
    u4567: vec4<u32>,
}

@group(0) @binding(0) 
var<storage, read> r_in: array<f32>;

@group(0) @binding(1) 
var<storage, read_write> z_out: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> temp_p: array<f32>;

@group(0) @binding(3) 
var<storage, read_write> p_sol: array<f32>;

@group(0) @binding(4) 
var<storage, read_write> p_prev: array<f32>;

@group(1) @binding(0) 
var<storage, read> row_offsets: array<u32>;

@group(1) @binding(1) 
var<storage, read> col_indices: array<u32>;

@group(1) @binding(2) 
var<storage, read> matrix_values: array<f32>;

@group(2) @binding(0) 
var<storage, read_write> diag_u_inv: array<f32>;

@group(2) @binding(1) 
var<storage, read_write> diag_p_inv: array<f32>;

@group(2) @binding(2) 
var<uniform> params: PrecondParams;

@group(3) @binding(0) 
var<storage, read> p_row_offsets: array<u32>;

@group(3) @binding(1) 
var<storage, read> p_col_indices: array<u32>;

@group(3) @binding(2) 
var<storage, read> p_matrix_values: array<f32>;

fn safe_inverse(val: f32) -> f32 {
    if (abs(val) > 0.00000000000001) {
        return 1.0 / val;
    }
    return 0.0;
}

fn u_index(i: u32) -> u32 {
    if (i < 4u) {
        return params.u0123[i];
    }
    return params.u4567[i - 4u];
}

const WORKGROUP_SIZE: u32 = 64u;

fn global_cell(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * num_workgroups.x * WORKGROUP_SIZE + global_id.x;
}

@compute
@workgroup_size(64)
fn relax_pressure(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let cell = global_cell(global_id, num_workgroups);
    if (cell >= params.num_cells) {
        return;
    }
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
    let hat_x = d_inv * (rhs - sigma);
    let x_prev = p_prev[cell];
    let x_new = mix(x_prev, hat_x, params.omega);
    p_prev[cell] = x_new;
}

@compute
@workgroup_size(64)
fn correct_velocity(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let cell = global_cell(global_id, num_workgroups);
    if (cell >= params.num_cells) {
        return;
    }
    let base = cell * params.unknowns_per_cell;
    let p_val = p_sol[cell];
    for (var i: u32 = 0u; i < params.u_len; i++) {
        let u = u_index(i);
        let row_u = base + u;
        let start_u = row_offsets[row_u];
        let end_u = row_offsets[row_u + 1u];
        var correction_u = 0.0;
        for (var k = start_u; k < end_u; k++) {
            let col = col_indices[k];
            if (col % params.unknowns_per_cell == params.p) {
                let p_cell = col / params.unknowns_per_cell;
                correction_u += matrix_values[k] * p_sol[p_cell];
            }
        }
        z_out[row_u] -= diag_u_inv[cell * params.u_len + i] * correction_u;
    }
    z_out[base + params.p] = p_val;
}

@compute
@workgroup_size(64)
fn predict_and_form_schur(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let cell = global_cell(global_id, num_workgroups);
    if (cell >= params.num_cells) {
        return;
    }
    // Part 1: Predict Velocity (Local)
    let base = cell * params.unknowns_per_cell;
    let row_p = base + params.p;
    for (var c: u32 = 0u; c < params.unknowns_per_cell; c++) {
        z_out[base + c] = r_in[base + c];
    }
    for (var i: u32 = 0u; i < params.u_len; i++) {
        let u = u_index(i);
        let row_u = base + u;
        z_out[row_u] = diag_u_inv[cell * params.u_len + i] * r_in[row_u];
    }
    z_out[row_p] = 0.0;
    // Part 2: Form Schur RHS
    var rhs_p = r_in[row_p];
    let start = row_offsets[row_p];
    let end = row_offsets[row_p + 1u];
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let rem = col % params.unknowns_per_cell;
        var z_val = 0.0;
        for (var i: u32 = 0u; i < params.u_len; i++) {
            let u = u_index(i);
            if (rem == u) {
                let c = col / params.unknowns_per_cell;
                z_val = r_in[col] * diag_u_inv[c * params.u_len + i];
                break;
            }
        }
        rhs_p -= matrix_values[k] * z_val;
    }
    temp_p[cell] = rhs_p;
    p_sol[cell] = diag_p_inv[cell] * rhs_p;
    p_prev[cell] = 0.0;
}
