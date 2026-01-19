// Cell-block Jacobi preconditioner for generic-coupled FGMRES.
//
// The block size is derived from the runtime linear system shape:
//   unknowns_per_cell = params.n / params.num_cells
//
// The solver is model-agnostic: unknown ordering is model-defined, but the CSR is built
// with per-cell blocks contiguous in DOF space so we can invert diagonal blocks per cell.

struct GmresParams {
    n: u32,
    num_cells: u32,
    num_iters: u32,
    omega: f32,
    dispatch_x: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct IterParams {
    current_idx: u32,
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}

// Group 0: vectors
@group(0) @binding(0) var<storage, read> vec_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> vec_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> vec_z: array<f32>;

// Group 1: matrix (CSR)
@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

// Group 2: block inverse
@group(2) @binding(0) var<storage, read_write> block_inv: array<f32>;

// Group 3: params (layout matches gmres_ops)
@group(3) @binding(0) var<uniform> params: GmresParams;
@group(3) @binding(1) var<storage, read_write> scalars: array<f32>;
@group(3) @binding(2) var<uniform> iter_params: IterParams;
@group(3) @binding(3) var<storage, read_write> hessenberg: array<f32>;
@group(3) @binding(4) var<storage, read> y_sol: array<f32>;

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

const MAX_BLOCK: u32 = 16u;

fn swap_rows(
    a: ptr<function, array<array<f32, MAX_BLOCK>, MAX_BLOCK>>,
    b: ptr<function, array<array<f32, MAX_BLOCK>, MAX_BLOCK>>,
    r0: u32,
    r1: u32,
    n: u32,
) {
    if (r0 == r1) {
        return;
    }
    for (var c = 0u; c < n; c = c + 1u) {
        let tmp = (*a)[r0][c];
        (*a)[r0][c] = (*a)[r1][c];
        (*a)[r1][c] = tmp;

        let tmp_b = (*b)[r0][c];
        (*b)[r0][c] = (*b)[r1][c];
        (*b)[r1][c] = tmp_b;
    }
}

@compute @workgroup_size(64)
fn build_block_inv(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let stride_x = num_workgroups.x * 64u;
    let cell = global_id.y * stride_x + global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    if (params.num_cells == 0u) {
        return;
    }

    let b = params.n / params.num_cells;
    if (b == 0u || b > MAX_BLOCK) {
        return;
    }

    let base = cell * b;
    var a: array<array<f32, MAX_BLOCK>, MAX_BLOCK>;
    var inv: array<array<f32, MAX_BLOCK>, MAX_BLOCK>;
    var diag_orig: array<f32, MAX_BLOCK>;

    for (var r = 0u; r < b; r = r + 1u) {
        for (var c = 0u; c < b; c = c + 1u) {
            a[r][c] = 0.0;
            inv[r][c] = 0.0;
        }
        let row = base + r;
        let start = row_offsets[row];
        let end = row_offsets[row + 1u];
        for (var k = start; k < end; k = k + 1u) {
            let col = col_indices[k];
            if (col >= base && col < base + b) {
                let local = col - base;
                a[r][local] = matrix_values[k];
            }
        }
        inv[r][r] = 1.0;
        diag_orig[r] = a[r][r];
    }

    var singular = false;
    for (var i = 0u; i < b; i = i + 1u) {
        var pivot = i;
        var pivot_val = abs(a[i][i]);
        for (var r = i + 1u; r < b; r = r + 1u) {
            let val = abs(a[r][i]);
            if (val > pivot_val) {
                pivot_val = val;
                pivot = r;
            }
        }
        if (pivot_val < 1e-12) {
            singular = true;
        }
        swap_rows(&a, &inv, i, pivot, b);
        var piv = a[i][i];
        if (abs(piv) < 1e-12) {
            piv = select(1e-12, -1e-12, piv < 0.0);
        }
        let inv_piv = 1.0 / piv;
        for (var c = 0u; c < b; c = c + 1u) {
            a[i][c] = a[i][c] * inv_piv;
            inv[i][c] = inv[i][c] * inv_piv;
        }
        for (var r = 0u; r < b; r = r + 1u) {
            if (r == i) {
                continue;
            }
            let factor = a[r][i];
            for (var c = 0u; c < b; c = c + 1u) {
                a[r][c] = a[r][c] - factor * a[i][c];
                inv[r][c] = inv[r][c] - factor * inv[i][c];
            }
        }
    }

    if (singular) {
        for (var r = 0u; r < b; r = r + 1u) {
            for (var c = 0u; c < b; c = c + 1u) {
                inv[r][c] = 0.0;
            }
            inv[r][r] = safe_inverse(diag_orig[r]);
        }
    }

    let offset = cell * (b * b);
    for (var r = 0u; r < b; r = r + 1u) {
        for (var c = 0u; c < b; c = c + 1u) {
            block_inv[offset + r * b + c] = inv[r][c];
        }
    }
}

@compute @workgroup_size(64)
fn apply_block_precond(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let stride_x = num_workgroups.x * 64u;
    let cell = global_id.y * stride_x + global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    if (params.num_cells == 0u) {
        return;
    }

    let b = params.n / params.num_cells;
    if (b == 0u || b > MAX_BLOCK) {
        return;
    }

    let base = cell * b;
    let offset = cell * (b * b);

    for (var r = 0u; r < b; r = r + 1u) {
        var sum = 0.0;
        for (var c = 0u; c < b; c = c + 1u) {
            sum += block_inv[offset + r * b + c] * vec_x[base + c];
        }
        vec_y[base + r] = sum;
    }
}
