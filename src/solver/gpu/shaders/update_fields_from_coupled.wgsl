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
    gamma: f32,
    r_gas: f32,
    is_compressible: u32,
    gravity_x: f32,
    gravity_y: f32,
    pad0: f32,
    pad1: f32,
}

// Constants are at group(1) binding(3) in the bg_fields layout
@group(0) @binding(3) var<uniform> constants: Constants;

@group(0) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(0) @binding(1) var<storage, read_write> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> fluxes: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_p: array<Vector2>;
@group(0) @binding(5) var<storage, read_write> d_p: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_component: array<Vector2>;
@group(0) @binding(7) var<storage, read> u_old: array<Vector2>;
@group(0) @binding(8) var<storage, read> u_old_old: array<Vector2>;
@group(0) @binding(9) var<storage, read_write> temperature: array<f32>;
@group(0) @binding(10) var<storage, read_write> energy: array<f32>;
@group(0) @binding(11) var<storage, read_write> density: array<f32>;
@group(0) @binding(12) var<storage, read_write> grad_e: array<Vector2>;

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
    var diff_e = 0.0;
    var diff_rho = 0.0;

    // Dynamic stride based on compressibility
    let stride = select(3u, 4u, constants.is_compressible == 1u);

    if (idx < arrayLength(&p)) {
        let u_new_val = x[stride * idx + 0u];
        let v_new_val = x[stride * idx + 1u];
        let p_new_val = x[stride * idx + 2u];
        
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

        // --- Compressible Flow Updates ---
        if (constants.is_compressible == 1u) {
            let e_new_val = x[stride * idx + 3u];
            let e_old = energy[idx];
            // Reuse alpha_u for energy relaxation for now
            let e_updated = e_old + alpha_u * (e_new_val - e_old);
            energy[idx] = e_updated;
            diff_e = abs(e_updated - e_old);

            // Update Temperature: T = E * (gamma - 1) / R
            // Ensure R and gamma are valid to avoid division by zero
            let gm1 = constants.gamma - 1.0;
            let temp_val = e_updated * gm1 / constants.r_gas;
            temperature[idx] = temp_val;

            // Update Density: rho = p / (R * T)
            let rho_old = density[idx];
            var rho_new = rho_old;
            if (temp_val > 1e-4) {
                 rho_new = p_updated / (constants.r_gas * temp_val);
            }
            // Clamp density to reasonable positive values
            rho_new = max(rho_new, 1e-4);
            
            // Simple under-relaxation for density? Or direct update?
            // Direct update consistent with EOS
            density[idx] = rho_new;
            diff_rho = abs(rho_new - rho_old);
        }
    }
    
    // Workgroup Reduction
    shared_max_u[lid] = diff_u;
    shared_max_p[lid] = diff_p;
    // We can reuse shared memory if we synchronize or use separate arrays?
    // Let's add shared_max_e and shared_max_rho. But we need to declare them.
    // Since I cannot change global scope declarations easily in one chunk if they are scattered, 
    // I will use atomicMax directly on global memory if I can't add shared memory? 
    // No, that's slow. I should add shared memory declarations.
    // But this tool call only replaces the main function body. 
    // I will assume I can insert shared var declarations inside main or use a separate tool call for declarations.
    // Actually, WGSL requires shared vars to be module-scope 'var<workgroup>'.
    
    // For now, let's just reduce u and p. I will add e and rho reduction in a separate step or just atomic update per thread for now (slow but safe).
    // Or, I can misuse shared_max_u/p after barrier for e/rho? No, that's racey if not careful.
    
    // I'll skip e/rho reduction optimization for this specific step and just do atomic update for them if is_compressible.
    // Wait, 64 is small workgroup. Atomic contention might be acceptable for now? 
    // Let's stick to correctness.

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
    
    // Reduce Energy and Density (Slow path for now: atomic per thread)
    // Actually, I can use the same shared memory arrays!
    // After u/p reduction is done (lid==0 wrote to global), I can reuse the shared memory.
    // But I need a barrier after the write? No, barrier before reuse.
    
    if (constants.is_compressible == 1u) {
        workgroupBarrier(); // Finish u/p reduction
        
        shared_max_u[lid] = diff_e;  // Reuse u array for e
        shared_max_p[lid] = diff_rho; // Reuse p array for rho
        workgroupBarrier();
        
        for (var stride = 32u; stride > 0u; stride >>= 1u) {
            if (lid < stride) {
                shared_max_u[lid] = max(shared_max_u[lid], shared_max_u[lid + stride]);
                shared_max_p[lid] = max(shared_max_p[lid], shared_max_p[lid + stride]);
            }
            workgroupBarrier();
        }
        
        if (lid == 0u) {
             atomicMax(&max_diff_result[2], bitcast<u32>(shared_max_u[0]));
             atomicMax(&max_diff_result[3], bitcast<u32>(shared_max_p[0]));
        }
    }
}
