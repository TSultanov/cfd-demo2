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
@group(1) @binding(7) var<storage, read> u_old: array<Vector2>;
@group(1) @binding(9) var<storage, read> u_old_old: array<Vector2>;

// Group 2: Solver (Coupled)
@group(2) @binding(0) var<storage, read_write> matrix_values: array<f32>;
@group(2) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(2) @binding(2) var<storage, read> scalar_row_offsets: array<u32>;
@group(2) @binding(3) var<storage, read> grad_u: array<Vector2>;
@group(2) @binding(4) var<storage, read> grad_v: array<Vector2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&cell_vols)) {
        return;
    }
    
    let center = cell_centers[idx];
    let vol = cell_vols[idx];
    let start = cell_face_offsets[idx];
    let end = cell_face_offsets[idx + 1];
    
    // Calculate base indices for coupled matrix
    let scalar_offset = scalar_row_offsets[idx];
    let num_neighbors = scalar_row_offsets[idx + 1] - scalar_offset;
    let start_row_0 = 9u * scalar_offset;
    let start_row_1 = start_row_0 + 3u * num_neighbors;
    let start_row_2 = start_row_0 + 6u * num_neighbors;
    
    // Initialize diagonal coefficients
    var diag_u: f32 = 0.0;
    var diag_v: f32 = 0.0;
    var diag_p: f32 = 0.0;
    
    var rhs_u: f32 = 0.0;
    var rhs_v: f32 = 0.0;
    var rhs_p: f32 = 0.0;
    
    // Time derivative
    let u_n = u_old[idx];
    var coeff_time = vol * constants.density / constants.dt;
    var rhs_time_u = coeff_time * u_n.x;
    var rhs_time_v = coeff_time * u_n.y;

    if (constants.time_scheme == 1u) {
        // BDF2
        let dt = constants.dt;
        let dt_old = constants.dt_old;
        let r = dt / dt_old;
        let u_nm1 = u_old_old[idx];
        
        coeff_time = vol * constants.density / dt * (1.0 + 2.0 * r) / (1.0 + r);
        let factor_n = (1.0 + r);
        let factor_nm1 = (r * r) / (1.0 + r);
        
        rhs_time_u = (vol * constants.density / dt) * (factor_n * u_n.x - factor_nm1 * u_nm1.x);
        rhs_time_v = (vol * constants.density / dt) * (factor_n * u_n.y - factor_nm1 * u_nm1.y);
    }

    diag_u += coeff_time;
    diag_v += coeff_time;
    rhs_u += rhs_time_u;
    rhs_v += rhs_time_v;
    
    // Strong pressure regularization
    let p_regularization = 0.01 * coeff_time;
    diag_p -= p_regularization;
    
    // Loop over faces
    for (var k = start; k < end; k++) {
        let face_idx = cell_faces[k];
        let owner = face_owner[face_idx];
        let neigh_idx = face_neighbor[face_idx];
        let boundary_type = face_boundary[face_idx];
        
        var normal = face_normals[face_idx];
        let area = face_areas[face_idx];
        let f_center = face_centers[face_idx];
        
        var normal_sign: f32 = 1.0;
        if (owner != idx) {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal_sign = -1.0;
        }
        
        // Flux is mass flux: rho * u * A
        let flux = fluxes[face_idx] * normal_sign;
        
        var other_center: Vector2;
        var is_boundary = false;
        var other_idx = 0u;
        
        if (neigh_idx != -1) {
            other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }
            other_center = cell_centers[other_idx];
        } else {
            is_boundary = true;
            other_center = f_center;
        }
        
        let d_vec_x = other_center.x - center.x;
        let d_vec_y = other_center.y - center.y;
        let dist = sqrt(d_vec_x*d_vec_x + d_vec_y*d_vec_y);
        
        // Diffusion: mu * A / dist
        let mu = constants.viscosity;
        let diff_coeff = mu * area / dist;
        
        // Convection: Upwind
        var conv_coeff_diag: f32 = 0.0;
        var conv_coeff_off: f32 = 0.0;
        
        if (flux > 0.0) {
            conv_coeff_diag = flux;
        } else {
            conv_coeff_off = flux;
        }
        
        // Matrix indices
        let scalar_mat_idx = cell_face_matrix_indices[k];
        var neighbor_rank = 0u;
        if (scalar_mat_idx != 4294967295u) {
            neighbor_rank = scalar_mat_idx - scalar_offset;
        } else {
            neighbor_rank = scalar_mat_idx - scalar_offset;
        }
        
        let idx_0_0 = start_row_0 + 3u * neighbor_rank + 0u;
        let idx_1_1 = start_row_1 + 3u * neighbor_rank + 1u;
        
        let idx_0_2 = start_row_0 + 3u * neighbor_rank + 2u;
        let idx_1_2 = start_row_1 + 3u * neighbor_rank + 2u;
        
        let idx_2_0 = start_row_2 + 3u * neighbor_rank + 0u;
        let idx_2_1 = start_row_2 + 3u * neighbor_rank + 1u;
        let idx_2_2 = start_row_2 + 3u * neighbor_rank + 2u;
        
        if (!is_boundary) {
            // --- Momentum Equations ---
            let coeff = -diff_coeff + conv_coeff_off;
            
            // Off-diagonal U-U and V-V
            matrix_values[idx_0_0] = coeff; // A_uu
            matrix_values[idx_1_1] = coeff; // A_vv
            
            diag_u += diff_coeff + conv_coeff_diag;
            diag_v += diff_coeff + conv_coeff_diag;
            
            // Deferred Correction for Higher Order Schemes
            if (constants.scheme != 0u) {
                // Get old values (from previous iteration/timestep)
                let u_old_own = u_old[idx];
                let u_old_neigh = u_old[other_idx];
                
                var phi_upwind_u = u_old_own.x;
                var phi_upwind_v = u_old_own.y;
                
                if (flux < 0.0) {
                    phi_upwind_u = u_old_neigh.x;
                    phi_upwind_v = u_old_neigh.y;
                }
                
                var phi_ho_u = phi_upwind_u;
                var phi_ho_v = phi_upwind_v;
                
                if (constants.scheme == 1u) { // Second Order Upwind
                    if (flux > 0.0) {
                        let grad_u_own = grad_u[idx];
                        let grad_v_own = grad_v[idx];
                        let r_x = f_center.x - center.x;
                        let r_y = f_center.y - center.y;
                        phi_ho_u = u_old_own.x + (grad_u_own.x * r_x + grad_u_own.y * r_y);
                        phi_ho_v = u_old_own.y + (grad_v_own.x * r_x + grad_v_own.y * r_y);
                    } else {
                        let grad_u_neigh = grad_u[other_idx];
                        let grad_v_neigh = grad_v[other_idx];
                        let r_x = f_center.x - other_center.x;
                        let r_y = f_center.y - other_center.y;
                        phi_ho_u = u_old_neigh.x + (grad_u_neigh.x * r_x + grad_u_neigh.y * r_y);
                        phi_ho_v = u_old_neigh.y + (grad_v_neigh.x * r_x + grad_v_neigh.y * r_y);
                    }
                } else if (constants.scheme == 2u) { // QUICK
                    if (flux > 0.0) {
                        let grad_u_own = grad_u[idx];
                        let grad_v_own = grad_v[idx];
                        let d_cd_x = other_center.x - center.x;
                        let d_cd_y = other_center.y - center.y;
                        
                        let grad_term_u = grad_u_own.x * d_cd_x + grad_u_own.y * d_cd_y;
                        let grad_term_v = grad_v_own.x * d_cd_x + grad_v_own.y * d_cd_y;
                        
                        phi_ho_u = 0.625 * u_old_own.x + 0.375 * u_old_neigh.x + 0.125 * grad_term_u;
                        phi_ho_v = 0.625 * u_old_own.y + 0.375 * u_old_neigh.y + 0.125 * grad_term_v;
                    } else {
                        let grad_u_neigh = grad_u[other_idx];
                        let grad_v_neigh = grad_v[other_idx];
                        let d_cd_x = center.x - other_center.x;
                        let d_cd_y = center.y - other_center.y;
                        
                        let grad_term_u = grad_u_neigh.x * d_cd_x + grad_u_neigh.y * d_cd_y;
                        let grad_term_v = grad_v_neigh.x * d_cd_x + grad_v_neigh.y * d_cd_y;
                        
                        phi_ho_u = 0.625 * u_old_neigh.x + 0.375 * u_old_own.x + 0.125 * grad_term_u;
                        phi_ho_v = 0.625 * u_old_neigh.y + 0.375 * u_old_own.y + 0.125 * grad_term_v;
                    }
                }
                
                let correction_u = flux * (phi_ho_u - phi_upwind_u);
                let correction_v = flux * (phi_ho_v - phi_upwind_v);
                
                rhs_u -= correction_u;
                rhs_v -= correction_v;
            }
            
            // Pressure Gradient (Cell to Face)
            let pg_coeff_x = area * normal.x / dist;
            let pg_coeff_y = area * normal.y / dist;
            
            // Off-diagonal U-P and V-P (Neighbor P)
            matrix_values[idx_0_2] = -pg_coeff_x; // A_up
            matrix_values[idx_1_2] = -pg_coeff_y; // A_vp
            
            // Diagonal U-P and V-P (Own P) - accumulated
            let scalar_diag_idx = diagonal_indices[idx];
            let diag_rank = scalar_diag_idx - scalar_offset;
            
            let diag_0_2 = start_row_0 + 3u * diag_rank + 2u;
            let diag_1_2 = start_row_1 + 3u * diag_rank + 2u;
            
            matrix_values[diag_0_2] += pg_coeff_x;
            matrix_values[diag_1_2] += pg_coeff_y;
            
            // --- Continuity Equation ---
            let div_coeff_x = 0.5 * normal.x * area;
            let div_coeff_y = 0.5 * normal.y * area;
            
            // Off-diagonal P-U and P-V (Neighbor U, V)
            matrix_values[idx_2_0] = div_coeff_x; // A_pu
            matrix_values[idx_2_1] = div_coeff_y; // A_pv
            
            // Diagonal P-U and P-V (Own U, V)
            let diag_2_0 = start_row_2 + 3u * diag_rank + 0u;
            let diag_2_1 = start_row_2 + 3u * diag_rank + 1u;
            
            matrix_values[diag_2_0] += div_coeff_x;
            matrix_values[diag_2_1] += div_coeff_y;
            
            // Rhie-Chow Pressure Laplacian
            let dp_f = 0.5 * (d_p[idx] + d_p[other_idx]);
            let lapl_coeff = dp_f * area / dist;
            
            // Off-diagonal P-P (Neighbor P)
            matrix_values[idx_2_2] = lapl_coeff; // A_pp
            
            // Diagonal P-P (Own P)
            let diag_2_2 = start_row_2 + 3u * diag_rank + 2u;
            matrix_values[diag_2_2] -= lapl_coeff;
            
        } else {
            // Boundary Conditions
            if (boundary_type == 1u) { // Inlet
                let ramp = smoothstep(0.0, constants.ramp_time, constants.time);
                let u_bc_x = constants.inlet_velocity * ramp;
                let u_bc_y = 0.0;
                
                diag_u += diff_coeff;
                diag_v += diff_coeff;
                
                rhs_u += diff_coeff * u_bc_x;
                rhs_v += diff_coeff * u_bc_y;
                
                if (flux > 0.0) {
                    diag_u += flux;
                    diag_v += flux;
                } else {
                    rhs_u -= flux * u_bc_x;
                    rhs_v -= flux * u_bc_y;
                }
                
                // Continuity at inlet:
                let flux_bc = (u_bc_x * normal.x + u_bc_y * normal.y) * area;
                rhs_p -= flux_bc;
                
            } else if (boundary_type == 3u) { // Wall
                // No slip: u = 0
                diag_u += diff_coeff;
                diag_v += diff_coeff;
                
            } else if (boundary_type == 2u) { // Outlet
                if (flux > 0.0) {
                    diag_u += flux;
                    diag_v += flux;
                }
                
                // Fixed Pressure p = 0
                let pg_coeff_x = area * normal.x / dist;
                let pg_coeff_y = area * normal.y / dist;
                
                let scalar_diag_idx = diagonal_indices[idx];
                let diag_rank = scalar_diag_idx - scalar_offset;
                let diag_0_2 = start_row_0 + 3u * diag_rank + 2u;
                let diag_1_2 = start_row_1 + 3u * diag_rank + 2u;
                
                matrix_values[diag_0_2] += pg_coeff_x;
                matrix_values[diag_1_2] += pg_coeff_y;
                
                // Continuity:
                let div_coeff_x = normal.x * area;
                let div_coeff_y = normal.y * area;
                
                let diag_2_0 = start_row_2 + 3u * diag_rank + 0u;
                let diag_2_1 = start_row_2 + 3u * diag_rank + 1u;
                
                matrix_values[diag_2_0] += div_coeff_x;
                matrix_values[diag_2_1] += div_coeff_y;
                
                // Rhie-Chow at outlet (fixed p)
                let dp_f = d_p[idx];
                let lapl_coeff = dp_f * area / dist;
                let diag_2_2 = start_row_2 + 3u * diag_rank + 2u;
                matrix_values[diag_2_2] -= lapl_coeff;
            }
        }
    }
    
    // Write diagonal coefficients
    let scalar_diag_idx = diagonal_indices[idx];
    let diag_rank = scalar_diag_idx - scalar_offset;
    
    let idx_0_0 = start_row_0 + 3u * diag_rank + 0u;
    let idx_1_1 = start_row_1 + 3u * diag_rank + 1u;
    let idx_2_2 = start_row_2 + 3u * diag_rank + 2u;
    
    matrix_values[idx_0_0] += diag_u;
    matrix_values[idx_1_1] += diag_v;
    matrix_values[idx_2_2] += diag_p;
    
    // Write RHS
    rhs[3u * idx + 0u] = rhs_u;
    rhs[3u * idx + 1u] = rhs_v;
    rhs[3u * idx + 2u] = rhs_p;
}
