struct Vector2 {
    x: f32,
    y: f32,
}

// Group 0: Mesh
@group(0) @binding(0) var<storage, read> face_owner: array<u32>;
@group(0) @binding(1) var<storage, read> face_neighbor: array<i32>;
@group(0) @binding(2) var<storage, read> face_areas: array<f32>;
@group(0) @binding(4) var<storage, read> cell_centers: array<Vector2>;
@group(0) @binding(12) var<storage, read> face_boundary: array<u32>;
@group(0) @binding(13) var<storage, read> face_centers: array<Vector2>;

// Group 1: Fields
@group(1) @binding(2) var<storage, read_write> fluxes: array<f32>;
@group(1) @binding(5) var<storage, read_write> d_p: array<f32>;

// Group 2: Solver (Solution x is p_prime)
@group(2) @binding(2) var<storage, read> x: array<f32>; // p_prime

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&fluxes)) {
        return;
    }
    
    let owner = face_owner[idx];
    let neighbor = face_neighbor[idx];
    let area = face_areas[idx];
    let center = cell_centers[owner];
    
    var d_p_face = d_p[owner];
    var p_prime_own = x[owner];
    
    if (neighbor != -1) {
        let neigh_idx = u32(neighbor);
        let c_neigh = cell_centers[neigh_idx];
        let f_center = face_centers[idx];
        
        let d_own = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
        let d_neigh = distance(vec2<f32>(c_neigh.x, c_neigh.y), vec2<f32>(f_center.x, f_center.y));
        
        let total_dist = d_own + d_neigh;
        var lambda = 0.5;
        if (total_dist > 1e-6) {
            lambda = d_neigh / total_dist;
        }
        
        d_p_face = lambda * d_p_face + (1.0 - lambda) * d_p[neigh_idx];
        
        let p_prime_neigh = x[neigh_idx];
        
        let dx = c_neigh.x - center.x;
        let dy = c_neigh.y - center.y;
        let dist = sqrt(dx*dx + dy*dy);
        
        let grad_p_f = (p_prime_neigh - p_prime_own) / dist;
        
        fluxes[idx] -= d_p_face * area * grad_p_f;
    } else {
        let boundary_type = face_boundary[idx];
        if (boundary_type == 2u) { // Outlet
             let f_center = face_centers[idx];
             let dx = f_center.x - center.x;
             let dy = f_center.y - center.y;
             let dist = sqrt(dx*dx + dy*dy);
             
             let grad_p_f = (0.0 - p_prime_own) / dist;
             fluxes[idx] -= d_p_face * area * grad_p_f;
        }
    }
}
