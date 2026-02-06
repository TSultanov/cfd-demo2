struct GmresParams {
    n: u32,
    num_cells: u32,
    num_iters: u32,
    omega: f32,
    dispatch_x: u32,
    max_restart: u32,
    column_offset: u32,
    _pad3: u32,
}

struct IterParams {
    current_idx: u32,
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64u;
const SCALAR_ITERS_USED: u32 = 10u;

fn global_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * (num_workgroups.x * WORKGROUP_SIZE) + global_id.x;
}

@group(0) @binding(0) var<storage, read> vec_x: array<f32>;        // packed z vectors
@group(0) @binding(1) var<storage, read_write> vec_y: array<f32>;   // x solution vector
@group(0) @binding(2) var<storage, read_write> vec_z: array<f32>;   // unused placeholder

@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> col_indices: array<u32>;
@group(1) @binding(2) var<storage, read> matrix_values: array<f32>;

@group(2) @binding(0) var<storage, read_write> diag_u: array<f32>;
@group(2) @binding(1) var<storage, read_write> diag_v: array<f32>;
@group(2) @binding(2) var<storage, read_write> diag_p: array<f32>;

@group(3) @binding(0) var<uniform> params: GmresParams;
@group(3) @binding(1) var<storage, read_write> scalars: array<f32>;
@group(3) @binding(2) var<uniform> iter_params: IterParams;
@group(3) @binding(3) var<storage, read_write> hessenberg: array<f32>;
@group(3) @binding(4) var<storage, read> y_sol: array<f32>;

@compute @workgroup_size(64)
fn accumulate_solution(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let idx = global_index(global_id, num_workgroups);
    if (idx >= params.n) {
        return;
    }

    let k = u32(clamp(
        round(scalars[SCALAR_ITERS_USED]),
        1.0,
        f32(iter_params.max_restart),
    ));
    let z_stride = max(params.column_offset, params.n);

    var acc = vec_y[idx];
    for (var i = 0u; i < k; i++) {
        acc = y_sol[i] * vec_x[i * z_stride + idx] + acc;
    }

    vec_y[idx] = acc;
}
