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
@group(0) @binding(10) var<storage, read> cell_face_matrix_indices: array<u32>;
@group(0) @binding(11) var<storage, read> diagonal_indices: array<u32>;

// Group 1: Fields
@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(2) var<storage, read_write> fluxes: array<f32>;

// Group 2: Solver
@group(2) @binding(0) var<storage, read_write> matrix_values: array<f32>;
@group(2) @binding(1) var<storage, read_write> rhs: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&cell_vols)) {
        return;
    }
    
    let start = cell_face_offsets[idx];
    let end = cell_face_offsets[idx + 1];
    let center = cell_centers[idx];
    
    var diag_coeff: f32 = 0.0;
    var rhs_val: f32 = 0.0;
    
    for (var k = start; k < end; k++) {
        let face_idx = cell_faces[k];
        let owner = face_owner[face_idx];
        let neigh_idx = face_neighbor[face_idx];
        
        var normal = face_normals[face_idx];
        let area = face_areas[face_idx];
        
        var normal_sign: f32 = 1.0;
        if (owner != idx) {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal_sign = -1.0;
        }
        
        // RHS: - div(U*)
        let flux = fluxes[face_idx];
        rhs_val -= flux * normal_sign;
        
        var other_center: Vector2;
        var is_boundary = false;
        
        if (neigh_idx != -1) {
            var other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }
            other_center = cell_centers[other_idx];
        } else {
            is_boundary = true;
        }
        
        if (!is_boundary) {
            let dx = center.x - other_center.x;
            let dy = center.y - other_center.y;
            let dist = sqrt(dx*dx + dy*dy);
            
            let coeff = area / dist;
            
            let mat_idx = cell_face_matrix_indices[k];
            if (mat_idx != 4294967295u) {
                matrix_values[mat_idx] = -coeff;
            }
            
            diag_coeff += coeff;
        }
    }
    
    let diag_idx = diagonal_indices[idx];
    matrix_values[diag_idx] = diag_coeff;
    
    rhs[idx] = rhs_val;
}
