// Combined Pressure Assembly with Inline Gradient Computation
// This merges gradient.wgsl (for component==2) and pressure_assembly.wgsl
// into a single kernel to reduce dispatch overhead.

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
    alpha_u: f32,
    stride_x: u32,
    padding: u32,
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
    let p_c = p[idx];
    
    var diag_coeff: f32 = 0.0;
    var rhs_val: f32 = 0.0;
    
    // Gradient accumulator (Green-Gauss)
    var grad = Vector2(0.0, 0.0);
    
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
        var p_other: f32 = 0.0;
        var other_idx: u32 = 0u;
        
        if (neigh_idx != -1) {
            other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }
            other_center = cell_centers[other_idx];
            d_p_neigh = d_p[other_idx];
            p_other = p[other_idx];
        } else {
            is_boundary = true;
            other_center = f_center;
            d_p_neigh = d_p[idx];
        }
        
        // Compute face value for gradient (distance-weighted interpolation)
        var p_f: f32 = 0.0;
        if (!is_boundary) {
            let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
            let d_o = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
            let total_dist = d_c + d_o;
            if (total_dist > 1e-6) {
                let lambda = d_o / total_dist;
                p_f = lambda * p_c + (1.0 - lambda) * p_other;
            } else {
                p_f = 0.5 * (p_c + p_other);
            }
        } else {
            // Boundary
            if (boundary_type == 2u) { // Outlet (Dirichlet p=0)
                p_f = 0.0;
            } else {
                // Inlet/Wall (Neumann dp/dn=0)
                p_f = p_c;
            }
        }
        
        // Accumulate gradient contribution
        // Using outward normal for this cell
        grad.x += p_f * normal.x * area;
        grad.y += p_f * normal.y * area;
        
        if (!is_boundary) {
            // Vector from this cell to other cell
            let d_vec_x = other_center.x - center.x;
            let d_vec_y = other_center.y - center.y;
            let dist = sqrt(d_vec_x*d_vec_x + d_vec_y*d_vec_y);
            
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
            
            let coeff = constants.density * d_p_face * area / dist;
            
            let mat_idx = cell_face_matrix_indices[k];
            if (mat_idx != 4294967295u) {
                matrix_values[mat_idx] = -coeff;
            }
            
            diag_coeff += coeff;
            
            // Non-orthogonal correction for pressure Laplacian
            // S = normal * area (face area vector pointing outward from this cell)
            let s_x = normal.x * area;
            let s_y = normal.y * area;
            
            // k_vec = S - d_vec * (area / dist)
            let k_x_raw = s_x - d_vec_x * (area / dist);
            let k_y_raw = s_y - d_vec_y * (area / dist);
            
            // Limit non-orthogonality vector magnitude for stability
            // |k| <= 0.5 * area
            let k_mag = sqrt(k_x_raw * k_x_raw + k_y_raw * k_y_raw);
            let k_limit = 0.5 * area;
            var k_scale = 1.0;
            if (k_mag > k_limit) {
                k_scale = k_limit / k_mag;
            }
            let k_x = k_x_raw * k_scale;
            let k_y = k_y_raw * k_scale;
            
            // Interpolate pressure gradient to face
            // Note: We use the OLD grad_p values here (from previous iteration)
            // This is the deferred correction approach - acceptable for PISO
            let grad_p_own = grad_p[idx];
            let grad_p_neigh = grad_p[other_idx];
            
            // Distance-weighted interpolation
            var interp_f = 0.5;
            if (total_dist > 1e-6) {
                interp_f = d_own / total_dist;  // Weight for neighbor
            }
            
            let grad_p_f_x = grad_p_own.x + interp_f * (grad_p_neigh.x - grad_p_own.x);
            let grad_p_f_y = grad_p_own.y + interp_f * (grad_p_neigh.y - grad_p_own.y);
            
            // Correction flux = rho * d_p_face * dot(grad_p_f, k_vec)
            // Under-relaxed (0.5)
            let correction_flux = 0.5 * constants.density * d_p_face * (grad_p_f_x * k_x + grad_p_f_y * k_y);
            
            // Subtract from RHS (pressure equation is Laplacian)
            rhs_val -= correction_flux;
        } else {
            // Boundary Conditions
            if (boundary_type == 2u) { // Outlet (Dirichlet p=0)
                let dx = center.x - f_center.x;
                let dy = center.y - f_center.y;
                let dist = sqrt(dx*dx + dy*dy);
                
                let d_p_own = d_p[idx];
                let coeff = constants.density * d_p_own * area / dist;
                
                diag_coeff += coeff;
                // rhs += coeff * 0.0
            }
            // Inlet/Wall are Neumann (flux=0)
        }
    }
    
    // Finalize gradient
    grad.x /= vol;
    grad.y /= vol;
    grad_p[idx] = grad;
    
    // Write matrix and RHS
    let diag_idx = diagonal_indices[idx];
    matrix_values[diag_idx] = diag_coeff;
    rhs[idx] = rhs_val;
}
