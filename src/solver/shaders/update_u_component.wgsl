struct Vector2 {
    x: f32,
    y: f32,
}

struct Constants {
    dt: f32,
    time: f32,
    viscosity: f32,
    density: f32,
    component: u32,
}

@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(3) var<uniform> constants: Constants;
@group(2) @binding(2) var<storage, read> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&u)) {
        return;
    }
    if (constants.component == 0u) {
        u[idx].x = x[idx];
    } else {
        u[idx].y = x[idx];
    }
}
