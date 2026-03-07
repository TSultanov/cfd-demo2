@group(0) @binding(0) 
var<storage, read> break_status: array<u32>;

@group(0) @binding(1) 
var<storage, read_write> scalars: array<f32>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }
    scalars[8u] = f32(break_status[0u]);
}
