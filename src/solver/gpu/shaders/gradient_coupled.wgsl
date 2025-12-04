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
    component: u32, // Unused here
    alpha_p: f32,
    time_scheme: u32,
    inlet_velocity: f32,
    ramp_time: f32,
    alpha_u: f32,
    stride_x: u32,
    padding: u32,
}

// Group 1: Fields
@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(1) var<storage, read_write> p: array<f32>;
@group(1) @binding(3) var<uniform> constants: Constants;

// Group 2: Coupled Solver Resources (Output)
// We only use bindings 3 and 4
@group(2) @binding(3) var<storage, read_write> grad_u: array<Vector2>;
@group(2) @binding(4) var<storage, read_write> grad_v: array<Vector2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&p)) {
        return;
    }
    
    let start = cell_face_offsets[idx];
    let end = cell_face_offsets[idx + 1];
    let vol = cell_vols[idx];
    let center = cell_centers[idx];
    
    let u_val = u[idx];
    let val_c_u = u_val.x;
    let val_c_v = u_val.y;
    
    var g_u = Vector2(0.0, 0.0);
    var g_v = Vector2(0.0, 0.0);
    
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
        
        var val_f_u = 0.0;
        var val_f_v = 0.0;
        
        if (neigh_idx != -1) {
            // Internal Face
            var other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }
            
            let u_other = u[other_idx];
            let val_other_u = u_other.x;
            let val_other_v = u_other.y;
            
            let center_other = cell_centers[other_idx];
            
            let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(face_center.x, face_center.y));
            let d_o = distance(vec2<f32>(center_other.x, center_other.y), vec2<f32>(face_center.x, face_center.y));
            
            let total_dist = d_c + d_o;
            if (total_dist > 1e-6) {
                let lambda = d_o / total_dist;
                val_f_u = lambda * val_c_u + (1.0 - lambda) * val_other_u;
                val_f_v = lambda * val_c_v + (1.0 - lambda) * val_other_v;
            } else {
                val_f_u = 0.5 * (val_c_u + val_other_u);
                val_f_v = 0.5 * (val_c_v + val_other_v);
            }
        } else {
            // Boundary
            let boundary_type = face_boundary[face_idx];
            
            // Velocity Boundary
            if (boundary_type == 1u) { // Inlet
                let ramp = smoothstep(0.0, constants.ramp_time, constants.time);
                val_f_u = constants.inlet_velocity * ramp;
                val_f_v = 0.0;
            } else if (boundary_type == 3u) { // Wall
                val_f_u = 0.0;
                val_f_v = 0.0;
            } else { // Outlet
                val_f_u = val_c_u; // Zero Gradient
                val_f_v = val_c_v;
            }
        }
        
        g_u.x += val_f_u * normal.x * area;
        g_u.y += val_f_u * normal.y * area;
        
        g_v.x += val_f_v * normal.x * area;
        g_v.y += val_f_v * normal.y * area;
    }
    
    g_u.x /= vol;
    g_u.y /= vol;
    
    g_v.x /= vol;
    g_v.y /= vol;
    
    grad_u[idx] = g_u;
    grad_v[idx] = g_v;
}
