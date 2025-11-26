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
    alpha_p: f32,
    scheme: u32,
    stride_x: u32,
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
    let c_own = cell_centers[owner];
    let dx_vec = face_center.x - c_own.x;
    let dy_vec = face_center.y - c_own.y;
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
        
        let c_own = cell_centers[owner];
        let c_neigh = cell_centers[neigh_idx];
        
        let d_own = distance(vec2<f32>(c_own.x, c_own.y), vec2<f32>(face_center.x, face_center.y));
        let d_neigh = distance(vec2<f32>(c_neigh.x, c_neigh.y), vec2<f32>(face_center.x, face_center.y));
        
        let total_dist = d_own + d_neigh;
        var lambda = 0.5;
        if (total_dist > 1e-6) {
            lambda = d_neigh / total_dist;
        }
        
        // Linear interpolation to face (CPU uses: val_owner + f * (val_neigh - val_owner) where f = d_own/(d_own+d_neigh))
        // Here lambda = d_neigh / total = weight for owner, so (1-lambda) = d_own/total = weight for neighbor
        // u_face = lambda * u_own + (1-lambda) * u_neigh  
        // This is equivalent to u_own + (1-lambda) * (u_neigh - u_own) = u_own + f * (u_neigh - u_own)
        u_face.x = lambda * u_face.x + (1.0 - lambda) * u_neigh.x;
        u_face.y = lambda * u_face.y + (1.0 - lambda) * u_neigh.y;
        
        // d_p interpolation: use simple average to match CPU flux correction
        d_p_face = 0.5 * (d_p_face + d_p_neigh);
        
        grad_p_avg.x = lambda * grad_p_avg.x + (1.0 - lambda) * grad_p_neigh.x;
        grad_p_avg.y = lambda * grad_p_avg.y + (1.0 - lambda) * grad_p_neigh.y;
        
        let dx = c_neigh.x - c_own.x;
        let dy = c_neigh.y - c_own.y;
        let dist = sqrt(dx*dx + dy*dy);
        
        let p_own = p[owner];
        let p_neigh = p[neigh_idx];
        
        let grad_p_n = grad_p_avg.x * normal.x + grad_p_avg.y * normal.y;
        let p_grad_f = (p_neigh - p_own) / dist;
        
        let rc_term = d_p_face * area * (grad_p_n - p_grad_f);
        
        let u_n = u_face.x * normal.x + u_face.y * normal.y;
        fluxes[idx] = u_n * area + rc_term;
        
    } else {
        // Boundary
        if (boundary_type == 1u) { // Inlet
             // Fixed U
             let u_bc = Vector2(1.0, 0.0);
             fluxes[idx] = (u_bc.x * normal.x + u_bc.y * normal.y) * area;
        } else if (boundary_type == 3u) { // Wall
             fluxes[idx] = 0.0;
        } else { // Outlet
             // Use owner velocity
             fluxes[idx] = (u_face.x * normal.x + u_face.y * normal.y) * area;
        }
    }
}
