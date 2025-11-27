struct Vector2 {
    x: f32,
    y: f32,
}

// Group 0: Mesh
@group(0) @binding(0) var<storage, read> face_owner: array<u32>;
@group(0) @binding(1) var<storage, read> face_neighbor: array<i32>;
@group(0) @binding(2) var<storage, read> face_areas: array<f32>;
@group(0) @binding(3) var<storage, read> face_normals: array<Vector2>;
@group(0) @binding(4) var<storage, read> cell_centers: array<Vector2>;
@group(0) @binding(13) var<storage, read> face_centers: array<Vector2>;

// Group 1: Fields
@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(2) var<storage, read_write> fluxes: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&face_areas)) {
        return;
    }
    
    let owner = face_owner[idx];
    let neighbor = face_neighbor[idx];
    let normal = face_normals[idx];
    let area = face_areas[idx];
    let face_center = face_centers[idx];
    
    // Interpolate U to face
    var u_face = u[owner];
    
    if (neighbor != -1) {
        let u_neigh = u[u32(neighbor)];
        let center_own = cell_centers[owner];
        let center_neigh = cell_centers[u32(neighbor)];
        
        let d_own = distance(vec2<f32>(center_own.x, center_own.y), vec2<f32>(face_center.x, face_center.y));
        let d_neigh = distance(vec2<f32>(center_neigh.x, center_neigh.y), vec2<f32>(face_center.x, face_center.y));
        
        let total_dist = d_own + d_neigh;
        if (total_dist > 1e-6) {
            let lambda = d_neigh / total_dist; // Weight for owner
            u_face.x = lambda * u_face.x + (1.0 - lambda) * u_neigh.x;
            u_face.y = lambda * u_face.y + (1.0 - lambda) * u_neigh.y;
        } else {
            u_face.x = (u_face.x + u_neigh.x) * 0.5;
            u_face.y = (u_face.y + u_neigh.y) * 0.5;
        }
    }
    
    fluxes[idx] = (u_face.x * normal.x + u_face.y * normal.y) * area;
}
