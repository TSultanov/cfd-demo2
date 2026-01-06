// Block-Jacobi preconditioner for compressible FGMRES (4x4 per cell).

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

fn swap_rows(a: ptr<function, array<array<f32, 4>, 4>>, b: ptr<function, array<array<f32, 4>, 4>>, r0: u32, r1: u32) {
    if (r0 == r1) {
        return;
    }
    for (var c = 0u; c < 4u; c = c + 1u) {
        let tmp = (*a)[r0][c];
        (*a)[r0][c] = (*a)[r1][c];
        (*a)[r1][c] = tmp;

        let tmp_b = (*b)[r0][c];
        (*b)[r0][c] = (*b)[r1][c];
        (*b)[r1][c] = tmp_b;
    }
}

@compute @workgroup_size(64)
fn build_block_inv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 4u;
    var a: array<array<f32, 4>, 4>;
    var inv: array<array<f32, 4>, 4>;
    var diag_orig: array<f32, 4>;

    for (var r = 0u; r < 4u; r = r + 1u) {
        for (var c = 0u; c < 4u; c = c + 1u) {
            a[r][c] = 0.0;
            inv[r][c] = 0.0;
        }
        let row = base + r;
        let start = row_offsets[row];
        let end = row_offsets[row + 1u];
        for (var k = start; k < end; k = k + 1u) {
            let col = col_indices[k];
            if (col >= base && col < base + 4u) {
                let local = col - base;
                a[r][local] = matrix_values[k];
            }
        }
        inv[r][r] = 1.0;
        diag_orig[r] = a[r][r];
    }

    var singular = false;
    for (var i = 0u; i < 4u; i = i + 1u) {
        var pivot = i;
        var pivot_val = abs(a[i][i]);
        for (var r = i + 1u; r < 4u; r = r + 1u) {
            let val = abs(a[r][i]);
            if (val > pivot_val) {
                pivot_val = val;
                pivot = r;
            }
        }
        if (pivot_val < 1e-12) {
            singular = true;
        }
        swap_rows(&a, &inv, i, pivot);
        var piv = a[i][i];
        if (abs(piv) < 1e-12) {
            piv = select(1e-12, -1e-12, piv < 0.0);
        }
        let inv_piv = 1.0 / piv;
        for (var c = 0u; c < 4u; c = c + 1u) {
            a[i][c] = a[i][c] * inv_piv;
            inv[i][c] = inv[i][c] * inv_piv;
        }
        for (var r = 0u; r < 4u; r = r + 1u) {
            if (r == i) {
                continue;
            }
            let factor = a[r][i];
            for (var c = 0u; c < 4u; c = c + 1u) {
                a[r][c] = a[r][c] - factor * a[i][c];
                inv[r][c] = inv[r][c] - factor * inv[i][c];
            }
        }
    }

    if (singular) {
        for (var r = 0u; r < 4u; r = r + 1u) {
            for (var c = 0u; c < 4u; c = c + 1u) {
                inv[r][c] = 0.0;
            }
            inv[r][r] = safe_inverse(diag_orig[r]);
        }
    }

    let offset = cell * 16u;
    for (var r = 0u; r < 4u; r = r + 1u) {
        for (var c = 0u; c < 4u; c = c + 1u) {
            block_inv[offset + r * 4u + c] = inv[r][c];
        }
    }
}

@compute @workgroup_size(64)
fn apply_block_precond(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell = global_id.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * 4u;
    let offset = cell * 16u;

    let x0 = vec_x[base + 0u];
    let x1 = vec_x[base + 1u];
    let x2 = vec_x[base + 2u];
    let x3 = vec_x[base + 3u];

    let y0 = block_inv[offset + 0u] * x0
        + block_inv[offset + 1u] * x1
        + block_inv[offset + 2u] * x2
        + block_inv[offset + 3u] * x3;
    let y1 = block_inv[offset + 4u] * x0
        + block_inv[offset + 5u] * x1
        + block_inv[offset + 6u] * x2
        + block_inv[offset + 7u] * x3;
    let y2 = block_inv[offset + 8u] * x0
        + block_inv[offset + 9u] * x1
        + block_inv[offset + 10u] * x2
        + block_inv[offset + 11u] * x3;
    let y3 = block_inv[offset + 12u] * x0
        + block_inv[offset + 13u] * x1
        + block_inv[offset + 14u] * x2
        + block_inv[offset + 15u] * x3;

    vec_y[base + 0u] = y0;
    vec_y[base + 1u] = y1;
    vec_y[base + 2u] = y2;
    vec_y[base + 3u] = y3;
}
