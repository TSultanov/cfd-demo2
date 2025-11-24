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
    padding1: u32,
    padding2: u32,
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
@group(0) @binding(12) var<storage, read> face_boundary: array<u32>;
@group(0) @binding(13) var<storage, read> face_centers: array<Vector2>;

// Group 1: Fields
@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(1) var<storage, read_write> p: array<f32>;
@group(1) @binding(2) var<storage, read_write> fluxes: array<f32>;
@group(1) @binding(3) var<uniform> constants: Constants;
@group(1) @binding(4) var<storage, read_write> grad_p: array<Vector2>;
@group(1) @binding(5) var<storage, read_write> d_p: array<f32>;

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
    let vol = cell_vols[idx];
    
    var diag_coeff: f32 = 0.0;
    var rhs_val: f32 = 0.0;
    
    for (var k = start; k < end; k++) {
        let face_idx = cell_faces[k];
        let owner = face_owner[face_idx];
        let neigh_idx = face_neighbor[face_idx];
        let boundary_type = face_boundary[face_idx];
        
        var normal = face_normals[face_idx];
        let area = face_areas[face_idx];
        let f_center = face_centers[face_idx];
        
        // Ensure normal points out of owner
        let c_owner = cell_centers[owner];
        let dx_vec = f_center.x - c_owner.x;
        let dy_vec = f_center.y - c_owner.y;
        if (dx_vec * normal.x + dy_vec * normal.y < 0.0) {
            normal.x = -normal.x;
            normal.y = -normal.y;
        }

        var normal_sign: f32 = 1.0;
        if (owner != idx) {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal_sign = -1.0;
        }
        
        // RHS contribution: - sum(flux)
        // Fluxes are already computed with Rhie-Chow
        let flux = fluxes[face_idx] * normal_sign;
        rhs_val -= flux;
        
        var other_center: Vector2;
        var is_boundary = false;
        var d_p_neigh: f32 = 0.0;
        
        if (neigh_idx != -1) {
            var other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }
            other_center = cell_centers[other_idx];
            d_p_neigh = d_p[other_idx];
        } else {
            is_boundary = true;
            other_center = f_center;
            d_p_neigh = d_p[idx]; // Placeholder
        }
        
        if (!is_boundary) {
            let dx = center.x - other_center.x;
            let dy = center.y - other_center.y;
            let dist = sqrt(dx*dx + dy*dy);
            
            // Calculate lambda for interpolation (match flux_rhie_chow)
            let d_own = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
            let d_neigh = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
            let total_dist = d_own + d_neigh;
            var lambda = 0.5;
            if (total_dist > 1e-6) {
                lambda = d_neigh / total_dist;
            }

            let d_p_own = d_p[idx];
            let d_p_face = lambda * d_p_own + (1.0 - lambda) * d_p_neigh;
            
            let coeff = d_p_face * area / dist;
            
            let mat_idx = cell_face_matrix_indices[k];
            if (mat_idx != 4294967295u) {
                matrix_values[mat_idx] = -coeff;
            }
            
            diag_coeff += coeff;
        } else {
            // Boundary Conditions
            if (boundary_type == 2u) { // Outlet (Dirichlet p=0)
                let dx = center.x - f_center.x;
                let dy = center.y - f_center.y;
                let dist = sqrt(dx*dx + dy*dy);
                
                let d_p_own = d_p[idx];
                let coeff = d_p_own * area / dist;
                
                diag_coeff += coeff;
                // rhs += coeff * 0.0
            }
            // Inlet/Wall are Neumann (flux=0)
        }
    }
    
    let diag_idx = diagonal_indices[idx];
    matrix_values[diag_idx] = diag_coeff;
    
    rhs[idx] = rhs_val;
}
