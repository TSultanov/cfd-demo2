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

// Group 0: Mesh
@group(0) @binding(0) var<storage, read> face_owner: array<u32>;
@group(0) @binding(1) var<storage, read> face_neighbor: array<i32>;
@group(0) @binding(2) var<storage, read> face_areas: array<f32>;
@group(0) @binding(3) var<storage, read> face_normals: array<Vector2>;
@group(0) @binding(4) var<storage, read> cell_centers: array<Vector2>;
@group(0) @binding(5) var<storage, read> cell_vols: array<f32>;
@group(0) @binding(12) var<storage, read> face_boundary: array<u32>;
@group(0) @binding(13) var<storage, read> face_centers: array<Vector2>;

// Group 1: Fields
@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(1) var<storage, read_write> p: array<f32>;
@group(1) @binding(2) var<storage, read_write> fluxes: array<f32>;
@group(1) @binding(3) var<uniform> constants: Constants;
@group(1) @binding(4) var<storage, read_write> grad_p: array<Vector2>;
@group(1) @binding(5) var<storage, read_write> d_p: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * constants.stride_x + global_id.x;
    if (idx >= arrayLength(&face_areas)) {
        return;
    }
    
    let owner = face_owner[idx];
    let neighbor = face_neighbor[idx];
    let area = face_areas[idx];
    let boundary_type = face_boundary[idx];
    let face_center = face_centers[idx];
    
    // Ensure normal points out of owner
    var normal = face_normals[idx];
    let c_owner = cell_centers[owner];
    let dx_vec = face_center.x - c_owner.x;
    let dy_vec = face_center.y - c_owner.y;
    if (dx_vec * normal.x + dy_vec * normal.y < 0.0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
    }

    var u_face = u[owner];
    var d_p_face = d_p[owner];
    var grad_p_avg = grad_p[owner];
    
    if (neighbor != -1) {
        let neigh_idx = u32(neighbor);
        let u_neigh = u[neigh_idx];
        let d_p_neigh = d_p[neigh_idx];
        let grad_p_neigh = grad_p[neigh_idx];
        
        let c_neigh = cell_centers[neigh_idx];
        
        let d_own = distance(vec2<f32>(c_owner.x, c_owner.y), vec2<f32>(face_center.x, face_center.y));
        let d_neigh = distance(vec2<f32>(c_neigh.x, c_neigh.y), vec2<f32>(face_center.x, face_center.y));
        
        let total_dist = d_own + d_neigh;
        var lambda = 0.5;
        if (total_dist > 1e-6) {
            lambda = d_neigh / total_dist;
        }
        
        // Calculate Central Difference Velocity first to determine direction
        var u_central = u_face;
        u_central.x = lambda * u_face.x + (1.0 - lambda) * u_neigh.x;
        u_central.y = lambda * u_face.y + (1.0 - lambda) * u_neigh.y;
        
        // Use Central Difference for Mass Flux (Standard Rhie-Chow)
        // Upwind is too diffusive and suppresses vortices
        u_face = u_central;
        
        // d_p interpolation: use distance weighting to match pressure assembly
        d_p_face = lambda * d_p_face + (1.0 - lambda) * d_p_neigh;
        
        grad_p_avg.x = lambda * grad_p_avg.x + (1.0 - lambda) * grad_p_neigh.x;
        grad_p_avg.y = lambda * grad_p_avg.y + (1.0 - lambda) * grad_p_neigh.y;
        
        let dx = c_neigh.x - c_owner.x;
        let dy = c_neigh.y - c_owner.y;
        let dist = sqrt(dx*dx + dy*dy);
        
        let p_own = p[owner];
        let p_neigh = p[neigh_idx];
        
        let grad_p_n = grad_p_avg.x * normal.x + grad_p_avg.y * normal.y;
        let p_grad_f = (p_neigh - p_own) / dist;
        
        let rc_term = d_p_face * area * (grad_p_n - p_grad_f);
        
        let u_n = u_face.x * normal.x + u_face.y * normal.y;
        fluxes[idx] = constants.density * (u_n * area + rc_term);
        
    } else {
        // Boundary
        if (boundary_type == 1u) { // Inlet
             // Fixed U with Ramp
             let ramp = smoothstep(0.0, constants.ramp_time, constants.time);
             let u_bc = Vector2(constants.inlet_velocity * ramp, 0.0);
             fluxes[idx] = constants.density * (u_bc.x * normal.x + u_bc.y * normal.y) * area;
        } else if (boundary_type == 3u) { // Wall
             fluxes[idx] = 0.0;
        } else if (boundary_type == 2u) { // Outlet
             let u_n = u_face.x * normal.x + u_face.y * normal.y;
             var rc_term = 0.0;
             let dist_face = distance(
                 vec2<f32>(c_owner.x, c_owner.y),
                 vec2<f32>(face_center.x, face_center.y),
             );
             if (dist_face > 1e-6) {
                 // Disable Rhie-Chow at outlet to prevent instability
                 rc_term = 0.0;
             }
             let raw_flux = constants.density * (u_n * area + rc_term);
             // Prevent backflow (inflow) at outlet for stability
             fluxes[idx] = max(0.0, raw_flux);
        } else {
             // Symmetry / Empty / Undefined
             fluxes[idx] = 0.0;
        }
    }
}
