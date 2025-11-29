struct Vector2 {
    x: f32,
    y: f32,
}

struct Constants {
    dt: f32,
    dt_old: f32,
    time: f32,
    viscosity: f32,
    density: f32,
    component: u32,
    alpha_p: f32,
    time_scheme: u32,
    inlet_velocity: f32,
    ramp_time: f32,
    alpha_u: f32,
    stride_x: u32,
    padding: u32,
}

@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(3) var<uniform> constants: Constants;
@group(1) @binding(7) var<storage, read> u_old: array<Vector2>;
@group(2) @binding(0) var<storage, read> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&u)) {
        return;
    }
    
    let alpha = constants.alpha_u;
    let u_new = x[idx];
    
    if (constants.component == 0u) {
        let u_old_val = u_old[idx].x;
        u[idx].x = alpha * u_new + (1.0 - alpha) * u_old_val;
    } else {
        let u_old_val = u_old[idx].y;
        u[idx].y = alpha * u_new + (1.0 - alpha) * u_old_val;
    }
}
