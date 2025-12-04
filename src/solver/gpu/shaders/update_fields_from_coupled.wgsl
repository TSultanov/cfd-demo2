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
    scheme: u32,
    alpha_u: f32,
    stride_x: u32,
    time_scheme: u32,
    inlet_velocity: f32,
    ramp_time: f32,
}

// Constants are at group(1) binding(3) in the bg_fields layout
@group(1) @binding(3) var<uniform> constants: Constants;

@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(1) var<storage, read_write> p: array<f32>;

@group(2) @binding(0) var<storage, read> x: array<f32>;
@group(2) @binding(1) var<storage, read_write> u_snapshot: array<Vector2>;
@group(2) @binding(2) var<storage, read_write> p_snapshot: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&p)) {
        return;
    }
    
    let u_new = x[3u * idx + 0u];
    let v_new = x[3u * idx + 1u];
    let p_new = x[3u * idx + 2u];
    
    // Under-relaxation for stability
    // U_updated = U_old + alpha * (U_new - U_old) = (1-alpha)*U_old + alpha*U_new
    let u_old = u[idx];
    let p_old = p[idx];
    
    // Snapshot current values (U_old) for convergence check in next iteration
    u_snapshot[idx] = u_old;
    p_snapshot[idx] = p_old;
    
    // Use relaxation factors (typically 0.7 for velocity, 0.3 for pressure in coupled solvers)
    let alpha_u = constants.alpha_u;
    let alpha_p = constants.alpha_p;
    
    u[idx] = Vector2(
        u_old.x + alpha_u * (u_new - u_old.x),
        u_old.y + alpha_u * (v_new - u_old.y)
    );
    p[idx] = p_old + alpha_p * (p_new - p_old);
}
