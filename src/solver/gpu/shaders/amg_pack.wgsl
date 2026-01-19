struct PackParams {
    num_cells: u32,
    component: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

@group(1) @binding(0) var<uniform> params: PackParams;

@compute @workgroup_size(64)
fn pack_component(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let stride_x = num_workgroups.x * 64u;
    let idx = global_id.y * stride_x + global_id.x;
    if (idx >= params.num_cells) {
        return;
    }
    let base = idx * 4u + params.component;
    output_buf[idx] = input_buf[base];
}

@compute @workgroup_size(64)
fn unpack_component(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let stride_x = num_workgroups.x * 64u;
    let idx = global_id.y * stride_x + global_id.x;
    if (idx >= params.num_cells) {
        return;
    }
    let base = idx * 4u + params.component;
    output_buf[base] = input_buf[idx];
}
