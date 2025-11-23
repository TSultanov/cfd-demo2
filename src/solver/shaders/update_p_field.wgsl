struct Constants {
    dt: f32,
    time: f32,
    viscosity: f32,
    density: f32,
    component: u32,
    alpha_p: f32,
}

@group(1) @binding(1) var<storage, read_write> p: array<f32>;
@group(1) @binding(3) var<uniform> constants: Constants;
@group(2) @binding(0) var<storage, read> x: array<f32>; // p_prime

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&p)) {
        return;
    }
    p[idx] += constants.alpha_p * x[idx];
}
