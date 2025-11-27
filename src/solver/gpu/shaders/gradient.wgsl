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
@group(0) @binding(5) var<storage, read> cell_vols: array<f32>;
@group(0) @binding(6) var<storage, read> cell_face_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> cell_faces: array<u32>;
@group(0) @binding(12) var<storage, read> face_boundary: array<u32>;
@group(0) @binding(13) var<storage, read> face_centers: array<Vector2>;

struct Constants {
    dt: f32,
    dt_old: f32,
    time: f32,
    viscosity: f32,
    density: f32,
    component: u32, // 0: x, 1: y, 2: p
    alpha_p: f32,
    scheme: u32,
    alpha_u: f32,
    stride_x: u32,
    padding: u32,
}

// Group 1: Fields
@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(1) var<storage, read_write> p: array<f32>;
@group(1) @binding(3) var<uniform> constants: Constants;
@group(1) @binding(4) var<storage, read_write> grad_p: array<Vector2>;
@group(1) @binding(6) var<storage, read_write> grad_component: array<Vector2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&p)) {
        return;
    }
    
    // Green-Gauss Gradient
    // grad(phi) = (1/V) * sum(phi_f * n * A)
    
    let start = cell_face_offsets[idx];
    let end = cell_face_offsets[idx + 1];
    let vol = cell_vols[idx];
    let center = cell_centers[idx];
    
    var val_c = 0.0;
    if (constants.component == 2u) {
        val_c = p[idx];
    } else {
        let u_val = u[idx];
        val_c = select(u_val.x, u_val.y, constants.component == 1u);
    }
    
    var grad = Vector2(0.0, 0.0);
    
    for (var k = start; k < end; k++) {
        let face_idx = cell_faces[k];
        let owner = face_owner[face_idx];
        let neigh_idx = face_neighbor[face_idx];
        
        var normal = face_normals[face_idx];
        let area = face_areas[face_idx];
        let face_center = face_centers[face_idx];
        
        // If we are neighbor, flip normal
        if (owner != idx) {
            normal.x = -normal.x;
            normal.y = -normal.y;
        }
        
        var val_f = 0.0;
        
        if (neigh_idx != -1) {
            // Internal Face: Distance Weighted Interpolation
            var other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }
            
            var val_other = 0.0;
            if (constants.component == 2u) {
                val_other = p[other_idx];
            } else {
                let u_other = u[other_idx];
                val_other = select(u_other.x, u_other.y, constants.component == 1u);
            }
            
            let center_other = cell_centers[other_idx];
            
            let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(face_center.x, face_center.y));
            let d_o = distance(vec2<f32>(center_other.x, center_other.y), vec2<f32>(face_center.x, face_center.y));
            
            // Inverse distance weighting
            // w_c = 1/d_c, w_o = 1/d_o
            // val_f = (w_c * val_c + w_o * val_other) / (w_c + w_o)
            // equivalent to linear interpolation:
            // lambda = d_o / (d_c + d_o) -> weight for val_c
            // val_f = lambda * val_c + (1-lambda) * val_other
            
            let total_dist = d_c + d_o;
            if (total_dist > 1e-6) {
                let lambda = d_o / total_dist;
                val_f = lambda * val_c + (1.0 - lambda) * val_other;
            } else {
                val_f = 0.5 * (val_c + val_other);
            }
        } else {
            // Boundary
            let boundary_type = face_boundary[face_idx];
            if (constants.component == 2u) {
                if (boundary_type == 2u) { // Outlet (Dirichlet p=0)
                    val_f = 0.0;
                } else {
                    // Inlet/Wall (Neumann dp/dn=0)
                    val_f = val_c;
                }
            } else {
                // Velocity Boundary
                if (boundary_type == 1u) { // Inlet
                    let u_bc_x = 1.0;
                    let u_bc_y = 0.0;
                    val_f = select(u_bc_x, u_bc_y, constants.component == 1u);
                } else if (boundary_type == 3u) { // Wall
                    val_f = 0.0;
                } else { // Outlet
                    val_f = val_c; // Zero Gradient
                }
            }
        }
        
        grad.x += val_f * normal.x * area;
        grad.y += val_f * normal.y * area;
    }
    
    grad.x /= vol;
    grad.y /= vol;
    
    if (constants.component == 2u) {
        grad_p[idx] = grad;
    } else {
        grad_component[idx] = grad;
    }
}
