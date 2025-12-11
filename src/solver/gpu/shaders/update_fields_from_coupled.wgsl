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

// Consolidated FluidState struct per cell (32 bytes, aligned)
struct FluidState {
    u: vec2<f32>,           // velocity (8 bytes)
    p: f32,                 // pressure (4 bytes)
    d_p: f32,               // pressure correction coefficient (4 bytes)
    grad_p: vec2<f32>,      // pressure gradient (8 bytes)
    grad_component: vec2<f32>, // velocity gradient component (8 bytes)
}

// Group 0: Fields (consolidated FluidState buffers)
@group(0) @binding(0) var<storage, read_write> state: array<FluidState>;
@group(0) @binding(1) var<storage, read> state_old: array<FluidState>;
@group(0) @binding(2) var<storage, read> state_old_old: array<FluidState>;
@group(0) @binding(3) var<storage, read_write> fluxes: array<f32>;
@group(0) @binding(4) var<uniform> constants: Constants;

@group(1) @binding(0) var<storage, read> x: array<f32>;
// Replaced snapshots with partial max diff outputs
@group(1) @binding(1) var<storage, read_write> max_diff_result: array<atomic<u32>>;

var<workgroup> shared_max_u: array<f32, 64>;
var<workgroup> shared_max_p: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {
    let idx = global_id.x;
    let lid = local_id.x;
    
    var diff_u = 0.0;
    var diff_p = 0.0;

    if (idx < arrayLength(&state)) {
        let u_new_val = x[3u * idx + 0u];
        let v_new_val = x[3u * idx + 1u];
        let p_new_val = x[3u * idx + 2u];
        
        // Under-relaxation for stability
        // U_updated = U_old + alpha * (U_new - U_old) = (1-alpha)*U_old + alpha*U_new
        let u_old_val = state[idx].u;
        let p_old_val = state[idx].p;
        
        // Use relaxation factors
        let alpha_u = constants.alpha_u;
        let alpha_p = constants.alpha_p;
        
        let u_updated_x = u_old_val.x + alpha_u * (u_new_val - u_old_val.x);
        let u_updated_y = u_old_val.y + alpha_u * (v_new_val - u_old_val.y);
        let p_updated = p_old_val + alpha_p * (p_new_val - p_old_val);

        state[idx].u = vec2<f32>(u_updated_x, u_updated_y);
        state[idx].p = p_updated;

        // Compute diffs for convergence check
        diff_u = max(abs(u_updated_x - u_old_val.x), abs(u_updated_y - u_old_val.y));
        diff_p = abs(p_updated - p_old_val);
    }
    
    // Workgroup Reduction
    shared_max_u[lid] = diff_u;
    shared_max_p[lid] = diff_p;
    workgroupBarrier();
    
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_max_u[lid] = max(shared_max_u[lid], shared_max_u[lid + stride]);
            shared_max_p[lid] = max(shared_max_p[lid], shared_max_p[lid + stride]);
        }
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        atomicMax(&max_diff_result[0], bitcast<u32>(shared_max_u[0]));
        atomicMax(&max_diff_result[1], bitcast<u32>(shared_max_p[0]));
    }
}
