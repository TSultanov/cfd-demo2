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
    gamma: f32,
    r_gas: f32,
    is_compressible: u32,
    gravity_x: f32,
    gravity_y: f32,
    pad0: f32,
    pad1: f32,
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
@group(1) @binding(7) var<storage, read> u_old: array<Vector2>;
@group(1) @binding(8) var<storage, read> u_old_old: array<Vector2>;
@group(1) @binding(9) var<storage, read_write> temperature: array<f32>;
@group(1) @binding(10) var<storage, read_write> energy: array<f32>;
@group(1) @binding(11) var<storage, read_write> density: array<f32>;
@group(1) @binding(12) var<storage, read_write> grad_e: array<Vector2>;

// Group 2: Coupled Solver Resources
@group(2) @binding(0) var<storage, read_write> matrix_values: array<f32>;
@group(2) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(2) @binding(2) var<storage, read> scalar_row_offsets: array<u32>;
@group(2) @binding(3) var<storage, read_write> grad_u: array<Vector2>;
@group(2) @binding(4) var<storage, read_write> grad_v: array<Vector2>;
@group(2) @binding(5) var<storage, read_write> scalar_matrix_values: array<f32>;
@group(2) @binding(6) var<storage, read_write> diag_u_inv: array<f32>;
@group(2) @binding(7) var<storage, read_write> diag_v_inv: array<f32>;
@group(2) @binding(8) var<storage, read_write> diag_p_inv: array<f32>;

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
    
    // --- Part 1: Flux and D_P Calculation ---
    
    var diag_coeff: f32 = 0.0;
    
    // Time derivative
    let u_n = u_old[idx];
    
    var time_coeff = vol * constants.density / constants.dt;
    if (constants.time_scheme == 1u) {
        let dt = constants.dt;
        let dt_old = constants.dt_old;
        let r = dt / dt_old;
        time_coeff = vol * constants.density / dt * (1.0 + 2.0 * r) / (1.0 + r);
    }
    diag_coeff += time_coeff;

    // Variables for gradient calculation (fused)
    let val_c_p = p[idx];
    var grad_p_accum = Vector2(0.0, 0.0);

    // Variables for velocity gradient calculation
    let u_val = u[idx];
    let val_c_u = u_val.x;
    let val_c_v = u_val.y;
    
    var g_u = Vector2(0.0, 0.0);
    var g_v = Vector2(0.0, 0.0);
    
    // Variables for energy gradient
    let val_c_e = energy[idx];
    var g_e = Vector2(0.0, 0.0);

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
        
        // --- Compute Flux (Rhie-Chow) ---
        
        let c_owner = cell_centers[owner];
        // Ensure normal points out of owner (for flux calculation convention)
        var normal_flux = face_normals[face_idx];
        let dx_vec = f_center.x - c_owner.x;
        let dy_vec = f_center.y - c_owner.y;
        if (dx_vec * normal_flux.x + dy_vec * normal_flux.y < 0.0) {
            normal_flux.x = -normal_flux.x;
            normal_flux.y = -normal_flux.y;
        }

        var flux: f32 = 0.0;
        
        if (neigh_idx != -1) {
            let n_idx = u32(neigh_idx);
            let c_neigh = cell_centers[n_idx];
            
            let u_own = u[owner];
            let u_ngh = u[n_idx];
            let dp_own = d_p[owner];
            let dp_ngh = d_p[n_idx];
            let gp_own = grad_p[owner];
            let gp_ngh = grad_p[n_idx];
            let p_own = p[owner];
            let p_ngh = p[n_idx];
            
            let d_own = distance(vec2<f32>(c_owner.x, c_owner.y), vec2<f32>(f_center.x, f_center.y));
            let d_ngh = distance(vec2<f32>(c_neigh.x, c_neigh.y), vec2<f32>(f_center.x, f_center.y));
            let total_dist = d_own + d_ngh;
            
            var lambda = 0.5;
            if (total_dist > 1e-6) {
                lambda = d_ngh / total_dist;
            }
            
            // Interpolate U
            let u_face_x = lambda * u_own.x + (1.0 - lambda) * u_ngh.x;
            let u_face_y = lambda * u_own.y + (1.0 - lambda) * u_ngh.y;
            
            // Interpolate d_p
            let dp_face = lambda * dp_own + (1.0 - lambda) * dp_ngh;
            
            // Interpolate grad_p
            let gp_face_x = lambda * gp_own.x + (1.0 - lambda) * gp_ngh.x;
            let gp_face_y = lambda * gp_own.y + (1.0 - lambda) * gp_ngh.y;
            
            // Pressure gradient at face
            let dx = c_neigh.x - c_owner.x;
            let dy = c_neigh.y - c_owner.y;
            let dist_proj = abs(dx * normal_flux.x + dy * normal_flux.y);
            let dist = max(dist_proj, 1e-6);
            
            let grad_p_n = gp_face_x * normal_flux.x + gp_face_y * normal_flux.y;
            let p_grad_f = (p_ngh - p_own) / dist;
            
            let rc_term = dp_face * area * (grad_p_n - p_grad_f);
            let u_n = u_face_x * normal_flux.x + u_face_y * normal_flux.y;
            
            flux = constants.density * (u_n * area + rc_term);
            
        } else {
            // Boundary
            if (boundary_type == 1u) { // Inlet
                 let ramp = smoothstep(0.0, constants.ramp_time, constants.time);
                 let u_bc = Vector2(constants.inlet_velocity * ramp, 0.0);
                 flux = constants.density * (u_bc.x * normal_flux.x + u_bc.y * normal_flux.y) * area;
            } else if (boundary_type == 3u) { // Wall
                 flux = 0.0;
            } else if (boundary_type == 2u) { // Outlet
                 let u_own = u[owner];
                 let u_n = u_own.x * normal_flux.x + u_own.y * normal_flux.y;
                 let raw_flux = constants.density * u_n * area;
                 flux = max(0.0, raw_flux);
            }
        }
        
        // Write flux if we are owner
        if (owner == idx) {
            fluxes[face_idx] = flux;
        }
        
        // --- Compute D_P Contribution ---
        
        var flux_out = flux;
        if (owner != idx) {
            flux_out = -flux;
        }
        
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
        
        let mu = constants.viscosity;
        let diff_coeff = mu * area / dist;
        
        var conv_coeff_diag: f32 = 0.0;
        if (flux_out > 0.0) {
            conv_coeff_diag = flux_out;
        }
        
        if (!is_boundary) {
            diag_coeff += diff_coeff + conv_coeff_diag;
        } else {
            if (boundary_type == 1u) { // Inlet
                diag_coeff += diff_coeff;
                if (flux_out > 0.0) {
                    diag_coeff += flux_out;
                }
            } else if (boundary_type == 3u) { // Wall
                diag_coeff += diff_coeff;
                if (flux_out > 0.0) {
                    diag_coeff += flux_out;
                }
            } else if (boundary_type == 2u) { // Outlet
                if (flux_out > 0.0) {
                    diag_coeff += flux_out;
                }
            }
        }
        
        // Gradient P calculation (fused)
        if (!is_boundary) {
             let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
             let d_o = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
             let total_dist_p = d_c + d_o;
             
             var lambda_p = 0.5;
             if (total_dist_p > 1e-6) {
                 lambda_p = d_o / total_dist_p;
             }
             
             let val_other_p = p[other_idx];
             let val_f_p = lambda_p * val_c_p + (1.0 - lambda_p) * val_other_p;
             
             grad_p_accum.x += val_f_p * normal.x * area;
             grad_p_accum.y += val_f_p * normal.y * area;
        } else {
             var val_f_p = val_c_p;
             if (boundary_type == 2u) { // Outlet
                 val_f_p = 0.0; 
             }
             grad_p_accum.x += val_f_p * normal.x * area;
             grad_p_accum.y += val_f_p * normal.y * area;
        }

        // --- Part 2: Velocity Gradient Calculation ---
        
        var val_f_u = 0.0;
        var val_f_v = 0.0;
        
        if (!is_boundary) {
            // Internal Face
            let u_other = u[other_idx];
            let val_other_u = u_other.x;
            let val_other_v = u_other.y;
            
            let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
            let d_o = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
            
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
        
        // --- Energy Gradient Calculation ---
        var val_f_e = 0.0;
        if (!is_boundary) {
             let val_other_e = energy[other_idx];
             let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
             let d_o = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
             let total_dist = d_c + d_o;
             if (total_dist > 1e-6) {
                 let lambda = d_o / total_dist;
                 val_f_e = lambda * val_c_e + (1.0 - lambda) * val_other_e;
             } else {
                 val_f_e = 0.5 * (val_c_e + val_other_e);
             }
        } else {
             if (boundary_type == 1u) { // Inlet
                 let e_bc = 220000.0; // Consistent placeholder
                 val_f_e = e_bc;
             } else { // Wall, Outlet (Zero Gradient)
                 val_f_e = val_c_e; 
             }
        }
        g_e.x += val_f_e * normal.x * area;
        g_e.y += val_f_e * normal.y * area;
    }
    
    // Write d_p
    if (abs(diag_coeff) > 1e-20) {
        d_p[idx] = vol / diag_coeff;
    } else {
        d_p[idx] = 0.0;
    }
    
    // Write grad_p
    grad_p_accum.x /= vol;
    grad_p_accum.y /= vol;
    grad_p[idx] = grad_p_accum;

    // Write grad_u, grad_v
    g_u.x /= vol;
    g_u.y /= vol;
    
    g_v.x /= vol;
    g_v.y /= vol;
    
    grad_u[idx] = g_u;
    grad_v[idx] = g_v;
    
    g_e.x /= vol;
    g_e.y /= vol;
    grad_e[idx] = g_e;
}
