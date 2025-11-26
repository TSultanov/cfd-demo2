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
@group(1) @binding(6) var<storage, read_write> grad_component: array<Vector2>;

// Group 2: Solver
@group(2) @binding(0) var<storage, read_write> matrix_values: array<f32>;
@group(2) @binding(1) var<storage, read_write> rhs: array<f32>;

// override component: u32; // 0 for x, 1 for y

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
    
    var diag_coeff: f32 = 0.0;
    var rhs_val: f32 = 0.0;
    
    // Time derivative: V/dt * u_old
    let u_old = u[idx];
    let val_old = select(u_old.x, u_old.y, constants.component == 1u);
    
    // Variables for gradient calculation (only used if component == 0)
    let val_c_p = p[idx];
    var grad_p_accum = Vector2(0.0, 0.0);

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
        
        // Vector from this cell to other cell (d_vec)
        let d_vec_x = other_center.x - center.x;
        let d_vec_y = other_center.y - center.y;
        let dist = sqrt(d_vec_x*d_vec_x + d_vec_y*d_vec_y);
        
        // Diffusion: mu * A / dist (mu = viscosity)
        let mu = constants.viscosity;
        let diff_coeff = mu * area / dist;
        
        // Convection: Upwind
        var conv_coeff_diag: f32 = 0.0;
        var conv_coeff_off: f32 = 0.0;
        
        if (flux > 0.0) {
            conv_coeff_diag = flux;
        } else {
            conv_coeff_off = flux; // Negative
        }
        
        if (!is_boundary) {
            let coeff = -diff_coeff + conv_coeff_off;
            
            let mat_idx = cell_face_matrix_indices[k];
            if (mat_idx != 4294967295u) {
                matrix_values[mat_idx] = coeff;
            }
            
            diag_coeff += diff_coeff + conv_coeff_diag;
            
            // Non-orthogonal diffusion correction
            // S = normal * area (face area vector pointing outward from this cell)
            let s_x = normal.x * area;
            let s_y = normal.y * area;
            
            // k_vec = S - d_vec * (area / dist)
            // This is the component of S perpendicular to d_vec
            let k_x_raw = s_x - d_vec_x * (area / dist);
            let k_y_raw = s_y - d_vec_y * (area / dist);
            
            // Limit non-orthogonality vector magnitude for stability
            // |k| <= 0.5 * area (approx 26.5 degrees max effective angle)
            let k_mag = sqrt(k_x_raw * k_x_raw + k_y_raw * k_y_raw);
            let k_limit = 0.5 * area;
            var k_scale = 1.0;
            if (k_mag > k_limit) {
                k_scale = k_limit / k_mag;
            }
            let k_x = k_x_raw * k_scale;
            let k_y = k_y_raw * k_scale;
            
            // Interpolate gradient to face
            let grad_own = grad_component[idx];
            let grad_neigh = grad_component[other_idx];
            
            // Distance-weighted interpolation
            let d_own = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
            let d_neigh = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
            let total_d = d_own + d_neigh;
            var interp_f = 0.5;
            if (total_d > 1e-6) {
                interp_f = d_own / total_d;  // Weight for neighbor
            }
            
            let grad_f_x = grad_own.x + interp_f * (grad_neigh.x - grad_own.x);
            let grad_f_y = grad_own.y + interp_f * (grad_neigh.y - grad_own.y);
            
            // Correction flux = mu * dot(grad_f, k_vec)
            // Under-relaxed for stability (0.5)
            let correction_flux = 0.5 * mu * (grad_f_x * k_x + grad_f_y * k_y);
            
            // Subtract from RHS (since diffusion is -div(mu * grad(phi)))
            rhs_val -= correction_flux;
        } else {
            // Boundary Conditions
            if (boundary_type == 1u) { // Inlet
                let u_bc_x = 1.0;
                let u_bc_y = 0.0;
                let val_bc = select(u_bc_x, u_bc_y, constants.component == 1u);
                
                diag_coeff += diff_coeff;
                rhs_val += diff_coeff * val_bc;
                
                if (flux > 0.0) {
                    diag_coeff += flux;
                } else {
                    rhs_val -= flux * val_bc;
                }
                
            } else if (boundary_type == 3u) { // Wall
                let val_bc = 0.0;
                
                diag_coeff += diff_coeff;
                
                if (flux > 0.0) {
                    diag_coeff += flux;
                }
            } else { // Outlet or Parallel
                // Zero gradient (Neumann) boundary condition for velocity
                // No diffusion contribution (du/dn = 0)
                // Convection: outflow only
                if (flux > 0.0) {
                    diag_coeff += flux;
                }
                // For backflow (flux < 0), we ignore it to prevent instability
                // This is a common practice for outlet BCs
            }
        }

        // Gradient P calculation (fused)
        if (constants.component == 0u) {
             var val_f_p = 0.0;
             if (!is_boundary) {
                 // Internal Face
                 let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
                 let d_o = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
                 let total_dist_p = d_c + d_o;
                 
                 var lambda_p = 0.5;
                 if (total_dist_p > 1e-6) {
                     lambda_p = d_o / total_dist_p;
                 }
                 
                 let val_other_p = p[other_idx];
                 val_f_p = lambda_p * val_c_p + (1.0 - lambda_p) * val_other_p;
             } else {
                 // Boundary
                 if (boundary_type == 2u) { // Outlet
                     val_f_p = 0.0;
                 } else {
                     val_f_p = val_c_p;
                 }
             }
             
             grad_p_accum.x += val_f_p * normal.x * area;
             grad_p_accum.y += val_f_p * normal.y * area;
        }
        
        // Deferred Correction for Higher Order Schemes
        if (constants.scheme != 0u && !is_boundary) {
            let u_other = u[other_idx];
            let val_other = select(u_other.x, u_other.y, constants.component == 1u);
            
            var phi_upwind = val_old;
            if (flux < 0.0) {
                phi_upwind = val_other;
            }
            
            var phi_ho = phi_upwind;
            
            let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
            let d_o = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
            
            if (constants.scheme == 1u) { // Second Order Upwind (Linear Upwind)
                if (flux > 0.0) {
                    // From Owner
                    let grad = grad_component[idx];
                    let r_x = f_center.x - center.x;
                    let r_y = f_center.y - center.y;
                    phi_ho = val_old + (grad.x * r_x + grad.y * r_y);
                } else {
                    // From Neighbor
                    let grad = grad_component[other_idx];
                    let r_x = f_center.x - other_center.x;
                    let r_y = f_center.y - other_center.y;
                    phi_ho = val_other + (grad.x * r_x + grad.y * r_y);
                }
            } else if (constants.scheme == 2u) { // QUICK
                // phi_f = 5/8 phi_C + 3/8 phi_D + 1/8 (grad_C . d_CD)
                // where C is upwind, D is downwind
                
                if (flux > 0.0) {
                    // C = Owner, D = Neighbor
                    let grad = grad_component[idx];
                    let d_cd_x = other_center.x - center.x;
                    let d_cd_y = other_center.y - center.y;
                    
                    let grad_term = grad.x * d_cd_x + grad.y * d_cd_y;
                    phi_ho = 0.625 * val_old + 0.375 * val_other + 0.125 * grad_term;
                } else {
                    // C = Neighbor, D = Owner
                    let grad = grad_component[other_idx];
                    let d_cd_x = center.x - other_center.x;
                    let d_cd_y = center.y - other_center.y;
                    
                    let grad_term = grad.x * d_cd_x + grad.y * d_cd_y;
                    phi_ho = 0.625 * val_other + 0.375 * val_old + 0.125 * grad_term;
                }
            }
            
            let correction = flux * (phi_ho - phi_upwind);
            rhs_val -= correction;
        }
    }
    
    // Pressure Gradient Source: -grad(p) * V
    if (constants.component == 0u) {
        grad_p_accum.x /= vol;
        grad_p_accum.y /= vol;
        grad_p[idx] = grad_p_accum;
        
        rhs_val -= grad_p_accum.x * vol;
    } else {
        let gp = grad_p[idx];
        rhs_val -= gp.y * vol;
    }

    // Time term
    let time_coeff = vol * constants.density / constants.dt;
    diag_coeff += time_coeff;
    rhs_val += time_coeff * val_old;

    let diag_idx = diagonal_indices[idx];
    matrix_values[diag_idx] = diag_coeff;

    if (constants.component == 0u) {
        if (abs(diag_coeff) > 1e-20) {
            d_p[idx] = vol / diag_coeff;
        } else {
            d_p[idx] = 0.0;
        }
    }
    
    rhs[idx] = rhs_val;
}
