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
// Replaced snapshots with partial max diff outputs
@group(2) @binding(1) var<storage, read_write> max_diff_result: array<atomic<u32>>;

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

    if (idx < arrayLength(&p)) {
        let u_new_val = x[3u * idx + 0u];
        let v_new_val = x[3u * idx + 1u];
        let p_new_val = x[3u * idx + 2u];
        
        // Under-relaxation for stability
        // U_updated = U_old + alpha * (U_new - U_old) = (1-alpha)*U_old + alpha*U_new
        let u_old = u[idx];
        let p_old = p[idx];
        
        // Use relaxation factors
        let alpha_u = constants.alpha_u;
        let alpha_p = constants.alpha_p;
        
        let u_updated_x = u_old.x + alpha_u * (u_new_val - u_old.x);
        let u_updated_y = u_old.y + alpha_u * (v_new_val - u_old.y);
        let p_updated = p_old + alpha_p * (p_new_val - p_old);

        u[idx] = Vector2(u_updated_x, u_updated_y);
        p[idx] = p_updated;

        // Compute diffs for convergence check
        diff_u = max(abs(u_updated_x - u_old.x), abs(u_updated_y - u_old.y));
        diff_p = abs(p_updated - p_old);
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
