struct SolverParams {
    n: u32,
}

@group(0) @binding(0) var<uniform> params: SolverParams;

@group(1) @binding(0) var<storage, read_write> dot_result_a: array<f32>;
@group(1) @binding(1) var<storage, read_write> dot_result_b: array<f32>;
@group(1) @binding(2) var<storage, read> dot_a0: array<f32>;
@group(1) @binding(3) var<storage, read> dot_b0: array<f32>;
@group(1) @binding(4) var<storage, read> dot_a1: array<f32>;
@group(1) @binding(5) var<storage, read> dot_b1: array<f32>;

var<workgroup> scratch_a: array<f32, 64>;
var<workgroup> scratch_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let idx = global_id.x;
    let lid = local_id.x;

    var val0 = 0.0;
    var val1 = 0.0;

    if (idx < params.n) {
        val0 = dot_a0[idx] * dot_b0[idx];
        val1 = dot_a1[idx] * dot_b1[idx];
    }

    scratch_a[lid] = val0;
    scratch_b[lid] = val1;
    workgroupBarrier();

    var offset = 32u;
    loop {
        if (lid < offset) {
            scratch_a[lid] += scratch_a[lid + offset];
            scratch_b[lid] += scratch_b[lid + offset];
        }
        workgroupBarrier();

        if (offset == 1u) {
            break;
        }
        offset = offset >> 1u;
    }

    if (lid == 0u) {
        dot_result_a[group_id.x] = scratch_a[0];
        dot_result_b[group_id.x] = scratch_b[0];
    }
}
