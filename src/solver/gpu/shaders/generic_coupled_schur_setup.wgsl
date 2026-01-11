// Generic coupled Schur preconditioner setup.
//
// Builds per-cell diagonal inverses (u/v/p) and extracts the pressure block
// into a scalar CSR values buffer for the Schur relax step.

@group(0) @binding(0) var<storage, read> scalar_row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> diagonal_indices: array<u32>;
@group(0) @binding(2) var<storage, read> matrix_values: array<f32>;

@group(0) @binding(3) var<storage, read_write> diag_u_inv: array<f32>;
@group(0) @binding(4) var<storage, read_write> diag_v_inv: array<f32>;
@group(0) @binding(5) var<storage, read_write> diag_p_inv: array<f32>;
@group(0) @binding(6) var<storage, read_write> p_matrix_values: array<f32>;

struct SetupParams {
    dispatch_x: u32,
    num_cells: u32,
    unknowns_per_cell: u32,
    u0: u32,
    u1: u32,
    p: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(7) var<uniform> params: SetupParams;

fn safe_inverse(val: f32) -> f32 {
    if (abs(val) > 1e-14) {
        return 1.0 / val;
    }
    return 0.0;
}

@compute
@workgroup_size(64)
fn build_diag_and_pressure(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.y * params.dispatch_x + global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let scalar_offset = scalar_row_offsets[cell];
    let scalar_end = scalar_row_offsets[cell + 1u];
    let num_neighbors = scalar_end - scalar_offset;
    let diag_rank = diagonal_indices[cell] - scalar_offset;

    let block_stride = params.unknowns_per_cell * params.unknowns_per_cell;
    let start_row_0 = scalar_offset * block_stride;
    let row_stride = num_neighbors * params.unknowns_per_cell;
    let start_row_u = start_row_0 + params.u0 * row_stride;
    let start_row_v = start_row_0 + params.u1 * row_stride;
    let start_row_p = start_row_0 + params.p * row_stride;

    let diag_u = matrix_values[start_row_u + diag_rank * params.unknowns_per_cell + params.u0];
    let diag_v = matrix_values[start_row_v + diag_rank * params.unknowns_per_cell + params.u1];
    let diag_p = matrix_values[start_row_p + diag_rank * params.unknowns_per_cell + params.p];

    diag_u_inv[cell] = safe_inverse(diag_u);
    diag_v_inv[cell] = safe_inverse(diag_v);
    diag_p_inv[cell] = safe_inverse(diag_p);

    for (var rank = 0u; rank < num_neighbors; rank++) {
        p_matrix_values[scalar_offset + rank] =
            matrix_values[start_row_p + rank * params.unknowns_per_cell + params.p];
    }
}
