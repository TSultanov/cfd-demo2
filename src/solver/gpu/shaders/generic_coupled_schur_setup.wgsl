// Generic coupled Schur preconditioner setup.
//
// Builds per-cell diagonal inverses (u-block + p) and extracts the pressure block
// into a scalar CSR values buffer for the Schur relax step.

@group(0) @binding(0) var<storage, read> scalar_row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> diagonal_indices: array<u32>;
@group(0) @binding(2) var<storage, read> matrix_values: array<f32>;

// Packed as diag_u_inv[cell * u_len + i]
@group(0) @binding(3) var<storage, read_write> diag_u_inv: array<f32>;
@group(0) @binding(4) var<storage, read_write> diag_p_inv: array<f32>;
@group(0) @binding(5) var<storage, read_write> p_matrix_values: array<f32>;

struct SetupParams {
    num_cells: u32,
    unknowns_per_cell: u32,
    p: u32,
    u_len: u32,
    u0123: vec4<u32>,
    u4567: vec4<u32>,
}
@group(0) @binding(6) var<uniform> params: SetupParams;

fn u_index(i: u32) -> u32 {
    if (i < 4u) {
        return params.u0123[i];
    }
    return params.u4567[i - 4u];
}

fn safe_inverse(val: f32) -> f32 {
    if (abs(val) > 1e-14) {
        return 1.0 / val;
    }
    return 0.0;
}

@compute
@workgroup_size(64)
fn build_diag_and_pressure(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let cell = global_id.y * (num_workgroups.x * 64u) + global_id.x;
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
    let start_row_p = start_row_0 + params.p * row_stride;

    let diag_p = matrix_values[start_row_p + diag_rank * params.unknowns_per_cell + params.p];

    for (var i = 0u; i < params.u_len; i++) {
        let u = u_index(i);
        let start_row_u = start_row_0 + u * row_stride;
        let diag_u = matrix_values[start_row_u + diag_rank * params.unknowns_per_cell + u];
        let inv_u = safe_inverse(diag_u);
        diag_u_inv[cell * params.u_len + i] = inv_u;
    }
    diag_p_inv[cell] = safe_inverse(diag_p);

    for (var rank = 0u; rank < num_neighbors; rank++) {
        p_matrix_values[scalar_offset + rank] =
            matrix_values[start_row_p + rank * params.unknowns_per_cell + params.p];
    }
}
