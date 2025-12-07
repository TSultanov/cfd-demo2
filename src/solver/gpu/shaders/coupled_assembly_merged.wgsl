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

// Group 2: Solver (Coupled)
@group(2) @binding(0) var<storage, read_write> matrix_values: array<f32>;
@group(2) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(2) @binding(2) var<storage, read> scalar_row_offsets: array<u32>;
@group(2) @binding(3) var<storage, read_write> grad_u: array<Vector2>;
@group(2) @binding(4) var<storage, read_write> grad_v: array<Vector2>;
@group(2) @binding(5) var<storage, read_write> scalar_matrix_values: array<f32>;
@group(2) @binding(6) var<storage, read_write> diag_u_inv: array<f32>;
@group(2) @binding(7) var<storage, read_write> diag_v_inv: array<f32>;
@group(2) @binding(8) var<storage, read_write> diag_p_inv: array<f32>;

fn safe_inverse(val: f32) -> f32 {
    if (abs(val) > 1e-14) {
        return 1.0 / val;
    }
    return 0.0;
}

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
    
    // Calculate base indices for coupled matrix (4x4 blocks)
    let scalar_offset = scalar_row_offsets[idx];
    let num_neighbors = scalar_row_offsets[idx + 1] - scalar_offset;
    // 4x4 = 16 elements per block
    let start_row_0 = 16u * scalar_offset; // U
    let start_row_1 = start_row_0 + 4u * num_neighbors; // V
    let start_row_2 = start_row_0 + 8u * num_neighbors; // E
    let start_row_3 = start_row_0 + 12u * num_neighbors; // P
    
    // Initialize diagonal coefficients
    var diag_u: f32 = 0.0;
    var diag_v: f32 = 0.0;
    var diag_e: f32 = 0.0;
    var diag_p: f32 = 0.0;
    
    // Diagonal accumulators for coupled terms
    var sum_diag_up: f32 = 0.0;
    var sum_diag_vp: f32 = 0.0;
    var sum_diag_ep: f32 = 0.0; // Energy-Pressure coupling? (Usually analytical or zero)
    
    var sum_diag_pu: f32 = 0.0;
    var sum_diag_pv: f32 = 0.0;
    var sum_diag_pe: f32 = 0.0; // Pressure-Energy coupling? (EOS related?)
    
    var sum_diag_pp: f32 = 0.0;
    
    var rhs_u: f32 = 0.0;
    var rhs_v: f32 = 0.0;
    var rhs_e: f32 = 0.0;
    var rhs_p: f32 = 0.0;

    // Scalar pressure diagonal
    var scalar_diag_p: f32 = 0.0;
    
    // Time derivative
    let u_n = u_old[idx];
    let rho = density[idx];
    var coeff_time = vol * rho / constants.dt;
    var rhs_time_u = coeff_time * u_n.x;
    var rhs_time_v = coeff_time * u_n.y;
    
    // Energy Transient (Euler)
    let e_n = energy[idx];
    var rhs_time_e = coeff_time * e_n;
    let diag_e_time = coeff_time;

    if (constants.time_scheme == 1u) {
        // BDF2 (Only for Momentum, as we lack old energy buffers)
        let dt = constants.dt;
        let dt_old = constants.dt_old;
        let r = dt / dt_old;
        let u_nm1 = u_old_old[idx];
        
        coeff_time = vol * rho / dt * (1.0 + 2.0 * r) / (1.0 + r);
        let factor_n = (1.0 + r);
        let factor_nm1 = (r * r) / (1.0 + r);
        
        rhs_time_u = (vol * rho / dt) * (factor_n * u_n.x - factor_nm1 * u_nm1.x);
        rhs_time_v = (vol * rho / dt) * (factor_n * u_n.y - factor_nm1 * u_nm1.y);
    }

    diag_u += coeff_time;
    diag_v += coeff_time;
    diag_e += diag_e_time;
    rhs_u += rhs_time_u;
    rhs_v += rhs_time_v;
    rhs_e += rhs_time_e;
    
    // Buoyancy Source Term (Gravity)
    // F_buoy = vol * rho * g
    let g_x = constants.gravity_x;
    let g_y = constants.gravity_y;
    let buoy_u = vol * rho * g_x;
    let buoy_v = vol * rho * g_y;
    
    rhs_u += buoy_u;
    rhs_v += buoy_v;

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
        var d_p_neigh: f32 = 0.0;
        var rho_neigh: f32 = 0.0;
        
        if (neigh_idx != -1) {
            other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }
            other_center = cell_centers[other_idx];
            d_p_neigh = d_p[other_idx];
            rho_neigh = density[other_idx];
        } else {
            is_boundary = true;
            other_center = f_center;
            d_p_neigh = d_p[idx]; // Placeholder
            rho_neigh = rho; // Neumann for density at boundary (except Inlet/Outlet specific handling)
        }
        
        let d_vec_x = other_center.x - center.x;
        let d_vec_y = other_center.y - center.y;
        
        // Use projected distance for diffusion to handle non-orthogonality
        // dist_proj = |d . n|
        let dist_proj = abs(d_vec_x * normal.x + d_vec_y * normal.y);
        let dist = max(dist_proj, 1e-6); // Avoid division by zero
        
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
        
        let idx_0_0 = start_row_0 + 4u * neighbor_rank + 0u;
        let idx_0_1 = start_row_0 + 4u * neighbor_rank + 1u;
        let idx_0_2 = start_row_0 + 4u * neighbor_rank + 2u;
        let idx_0_3 = start_row_0 + 4u * neighbor_rank + 3u;

        let idx_1_0 = start_row_1 + 4u * neighbor_rank + 0u;
        let idx_1_1 = start_row_1 + 4u * neighbor_rank + 1u;
        let idx_1_2 = start_row_1 + 4u * neighbor_rank + 2u;
        let idx_1_3 = start_row_1 + 4u * neighbor_rank + 3u;
        
        // E-Row
        let idx_2_0 = start_row_2 + 4u * neighbor_rank + 0u;
        let idx_2_1 = start_row_2 + 4u * neighbor_rank + 1u;
        let idx_2_2 = start_row_2 + 4u * neighbor_rank + 2u;
        let idx_2_3 = start_row_2 + 4u * neighbor_rank + 3u;
        
        // P-Row (Continuity)
        let idx_3_0 = start_row_3 + 4u * neighbor_rank + 0u;
        let idx_3_1 = start_row_3 + 4u * neighbor_rank + 1u;
        let idx_3_2 = start_row_3 + 4u * neighbor_rank + 2u;
        let idx_3_3 = start_row_3 + 4u * neighbor_rank + 3u;
        
        if (!is_boundary) {
            // --- Momentum Equations (Rows 0, 1) ---
            let coeff = -diff_coeff + conv_coeff_off;
            
            // Row 0: U equation
            matrix_values[idx_0_0] = coeff; // A_uu
            matrix_values[idx_0_1] = 0.0;   // A_uv
            matrix_values[idx_0_2] = 0.0;   // A_ue
            // matrix_values[idx_0_3] set later (P-grad)
            
            // Row 1: V equation
            matrix_values[idx_1_0] = 0.0;   // A_vu
            matrix_values[idx_1_1] = coeff; // A_vv
            matrix_values[idx_1_2] = 0.0;   // A_ve
            // matrix_values[idx_1_3] set later (P-grad)
            
            diag_u += diff_coeff + conv_coeff_diag;
            diag_v += diff_coeff + conv_coeff_diag;
            
            // Row 2: Energy Equation (assuming Pr=1 -> same coeff)
            // If Pr != 1, diff_coeff_e = diff_coeff / Pr
            matrix_values[idx_2_0] = 0.0;   // A_eu
            matrix_values[idx_2_1] = 0.0;   // A_ev
            matrix_values[idx_2_2] = coeff; // A_ee
            matrix_values[idx_2_3] = 0.0;   // A_ep
            
            diag_e += diff_coeff + conv_coeff_diag;
            
            // Deferred Correction for Higher Order Schemes (Momentum)
            if (constants.scheme != 0u) {
                // Get current values (from latest iteration) for deferred correction
                let u_own = u[idx];
                let u_neigh = u[other_idx];
                
                var phi_upwind_u = u_own.x;
                var phi_upwind_v = u_own.y;
                
                if (flux < 0.0) {
                    phi_upwind_u = u_neigh.x;
                    phi_upwind_v = u_neigh.y;
                }
                
                var phi_ho_u = phi_upwind_u;
                var phi_ho_v = phi_upwind_v;
                
                if (constants.scheme == 1u) { // Second Order Upwind
                    if (flux > 0.0) {
                        let grad_u_own = grad_u[idx];
                        let grad_v_own = grad_v[idx];
                        let r_x = f_center.x - center.x;
                        let r_y = f_center.y - center.y;
                        phi_ho_u = u_own.x + (grad_u_own.x * r_x + grad_u_own.y * r_y);
                        phi_ho_v = u_own.y + (grad_v_own.x * r_x + grad_v_own.y * r_y);
                    } else {
                        let grad_u_neigh = grad_u[other_idx];
                        let grad_v_neigh = grad_v[other_idx];
                        let r_x = f_center.x - other_center.x;
                        let r_y = f_center.y - other_center.y;
                        phi_ho_u = u_neigh.x + (grad_u_neigh.x * r_x + grad_u_neigh.y * r_y);
                        phi_ho_v = u_neigh.y + (grad_v_neigh.x * r_x + grad_v_neigh.y * r_y);
                    }
                } else if (constants.scheme == 2u) { // QUICK
                    if (flux > 0.0) {
                        let grad_u_own = grad_u[idx];
                        let grad_v_own = grad_v[idx];
                        let d_cd_x = other_center.x - center.x;
                        let d_cd_y = other_center.y - center.y;
                        
                        let grad_term_u = grad_u_own.x * d_cd_x + grad_u_own.y * d_cd_y;
                        let grad_term_v = grad_v_own.x * d_cd_x + grad_v_own.y * d_cd_y;
                        
                        phi_ho_u = 0.625 * u_own.x + 0.375 * u_neigh.x + 0.125 * grad_term_u;
                        phi_ho_v = 0.625 * u_own.y + 0.375 * u_neigh.y + 0.125 * grad_term_v;
                    } else {
                        let grad_u_neigh = grad_u[other_idx];
                        let grad_v_neigh = grad_v[other_idx];
                        let d_cd_x = center.x - other_center.x;
                        let d_cd_y = center.y - other_center.y;
                        
                        let grad_term_u = grad_u_neigh.x * d_cd_x + grad_u_neigh.y * d_cd_y;
                        let grad_term_v = grad_v_neigh.x * d_cd_x + grad_v_neigh.y * d_cd_y;
                        
                        phi_ho_u = 0.625 * u_neigh.x + 0.375 * u_own.x + 0.125 * grad_term_u;
                        phi_ho_v = 0.625 * u_neigh.y + 0.375 * u_own.y + 0.125 * grad_term_v;
                    }
                }
                
                let correction_u = flux * (phi_ho_u - phi_upwind_u);
                let correction_v = flux * (phi_ho_v - phi_upwind_v);
                
                rhs_u -= correction_u;
                rhs_v -= correction_v;
                
                // Deferred Correction for Energy
                let e_own = energy[idx];
                let e_neigh = energy[other_idx];
                
                var phi_upwind_e = e_own;
                if (flux < 0.0) {
                    phi_upwind_e = e_neigh;
                }
                
                var phi_ho_e = phi_upwind_e;
                
                if (constants.scheme == 1u) { // Second Order Upwind
                    if (flux > 0.0) {
                        let grad_e_own = grad_e[idx];
                        let r_x = f_center.x - center.x;
                        let r_y = f_center.y - center.y;
                        phi_ho_e = e_own + (grad_e_own.x * r_x + grad_e_own.y * r_y);
                    } else {
                        let grad_e_neigh = grad_e[other_idx];
                        let r_x = f_center.x - other_center.x;
                        let r_y = f_center.y - other_center.y;
                        phi_ho_e = e_neigh + (grad_e_neigh.x * r_x + grad_e_neigh.y * r_y);
                    }
                } else if (constants.scheme == 2u) { // QUICK
                    if (flux > 0.0) {
                        let grad_e_own = grad_e[idx];
                        let d_cd_x = other_center.x - center.x;
                        let d_cd_y = other_center.y - center.y;
                        let grad_term_e = grad_e_own.x * d_cd_x + grad_e_own.y * d_cd_y;
                        phi_ho_e = 0.625 * e_own + 0.375 * e_neigh + 0.125 * grad_term_e;
                    } else {
                        let grad_e_neigh = grad_e[other_idx];
                        let d_cd_x = center.x - other_center.x;
                        let d_cd_y = center.y - other_center.y;
                        let grad_term_e = grad_e_neigh.x * d_cd_x + grad_e_neigh.y * d_cd_y;
                        phi_ho_e = 0.625 * e_neigh + 0.375 * e_own + 0.125 * grad_term_e;
                    }
                }
                
                let correction_e = flux * (phi_ho_e - phi_upwind_e);
                rhs_e -= correction_e;
            }
            
            // Pressure Gradient (Cell to Face)
            let d_own = distance(vec2<f32>(center.x, center.y), vec2<f32>(f_center.x, f_center.y));
            let d_neigh = distance(vec2<f32>(other_center.x, other_center.y), vec2<f32>(f_center.x, f_center.y));
            let total_dist = d_own + d_neigh;
            
            var lambda = 0.5;
            if (total_dist > 1e-6) {
                lambda = d_neigh / total_dist;
            }

            // Interpolate density to face
            let rho_face = lambda * rho + (1.0 - lambda) * rho_neigh;
            
            let pg_force_x = area * normal.x;
            let pg_force_y = area * normal.y;
            
            // Off-diagonal U-P and V-P (Neighbor P) -> Col 3
            matrix_values[idx_0_3] = (1.0 - lambda) * pg_force_x; // A_up
            matrix_values[idx_1_3] = (1.0 - lambda) * pg_force_y; // A_vp
            
            // Diagonal U-P and V-P (Own P) - accumulated
            sum_diag_up += lambda * pg_force_x;
            sum_diag_vp += lambda * pg_force_y;
            
            // --- Continuity Equation (Row 3) ---
            let div_coeff_x = normal.x * area;
            let div_coeff_y = normal.y * area;
            
            // Off-diagonal P-U and P-V (Neighbor U, V)
            matrix_values[idx_3_0] = (1.0 - lambda) * div_coeff_x; // A_pu
            matrix_values[idx_3_1] = (1.0 - lambda) * div_coeff_y; // A_pv
            matrix_values[idx_3_2] = 0.0; // A_pe
            
            // Diagonal P-U and P-V (Own U, V)
            sum_diag_pu += lambda * div_coeff_x;
            sum_diag_pv += lambda * div_coeff_y;
            
            // Rhie-Chow Pressure Laplacian
            let dp_f = lambda * d_p[idx] + (1.0 - lambda) * d_p[other_idx];
            let lapl_coeff = dp_f * area / dist;
            
            // Off-diagonal P-P (Neighbor P)
            matrix_values[idx_3_3] = -lapl_coeff; // A_pp (Negative for diffusion)
            
            // Diagonal P-P (Own P)
            sum_diag_pp += lapl_coeff; // Positive for diffusion

            // --- Scalar Pressure Matrix Assembly ---
            // Calculate scalar pressure matrix coefficient
            // Use same distance and lambda as above
            let d_p_own = d_p[idx];
            let d_p_face = lambda * d_p_own + (1.0 - lambda) * d_p_neigh;
            
            let scalar_coeff = rho_face * d_p_face * area / dist;
            
            if (scalar_mat_idx != 4294967295u) {
                scalar_matrix_values[scalar_mat_idx] = -scalar_coeff;
            }
            
            scalar_diag_p += scalar_coeff;
            
        } else {
            // Boundary Conditions
            if (boundary_type == 1u) { // Inlet
                let ramp = smoothstep(0.0, constants.ramp_time, constants.time);
                let u_bc_x = constants.inlet_velocity * ramp;
                let u_bc_y = 0.0;
                let e_bc = 220000.0; // Placeholder for Inlet Energy (Approx T=300K, Cv=718)
                
                diag_u += diff_coeff;
                diag_v += diff_coeff;
                diag_e += diff_coeff; // Dirichlet for Energy
                
                rhs_u += diff_coeff * u_bc_x;
                rhs_v += diff_coeff * u_bc_y;
                rhs_e += diff_coeff * e_bc;
                
                if (flux > 0.0) {
                    diag_u += flux;
                    diag_v += flux;
                    diag_e += flux;
                } else {
                    rhs_u -= flux * u_bc_x;
                    rhs_v -= flux * u_bc_y;
                    rhs_e -= flux * e_bc;
                }
                
                // Pressure Gradient: Zero Gradient (p_f = p_P)
                let pg_force_x = area * normal.x;
                let pg_force_y = area * normal.y;
                
                sum_diag_up += pg_force_x;
                sum_diag_vp += pg_force_y;
                
                // Continuity at inlet:
                let flux_bc = (u_bc_x * normal.x + u_bc_y * normal.y) * area;
                rhs_p -= flux_bc;
                
            } else if (boundary_type == 3u) { // Wall
                // No slip: u = 0
                diag_u += diff_coeff;
                diag_v += diff_coeff;
                // Adiabatic Wall: No diffusion flux for Energy (Nu=0) -> Do nothing for diag_e
                
                // Pressure Gradient: Zero Gradient (p_f = p_P)
                let pg_force_x = area * normal.x;
                let pg_force_y = area * normal.y;
                
                sum_diag_up += pg_force_x;
                sum_diag_vp += pg_force_y;
                
            } else if (boundary_type == 2u) { // Outlet
                if (flux > 0.0) {
                    diag_u += flux;
                    diag_v += flux;
                    diag_e += flux;
                }
                
                // Continuity:
                let div_coeff_x = normal.x * area;
                let div_coeff_y = normal.y * area;
                
                sum_diag_pu += div_coeff_x;
                sum_diag_pv += div_coeff_y;
                
                // Rhie-Chow at outlet
                let dp_f = d_p[idx];
                let lapl_coeff = dp_f * area / dist; // dist here is d_own
                sum_diag_pp += lapl_coeff;

                // Scalar Pressure Matrix at Outlet (Dirichlet p=0)
                let d_p_own = d_p[idx];
                let scalar_coeff = rho * d_p_own * area / dist;
                scalar_diag_p += scalar_coeff;
            }
        }
    }
    
    // Write diagonal coefficients
    let scalar_diag_idx = diagonal_indices[idx];
    let diag_rank = scalar_diag_idx - scalar_offset;
    
    // 4x4 block indices
    let d_0_0 = start_row_0 + 4u * diag_rank + 0u;
    let d_0_1 = start_row_0 + 4u * diag_rank + 1u;
    let d_0_2 = start_row_0 + 4u * diag_rank + 2u; // U-E
    let d_0_3 = start_row_0 + 4u * diag_rank + 3u; // U-P
    
    let d_1_0 = start_row_1 + 4u * diag_rank + 0u;
    let d_1_1 = start_row_1 + 4u * diag_rank + 1u;
    let d_1_2 = start_row_1 + 4u * diag_rank + 2u; // V-E
    let d_1_3 = start_row_1 + 4u * diag_rank + 3u; // V-P
    
    let d_2_0 = start_row_2 + 4u * diag_rank + 0u; // E-U
    let d_2_1 = start_row_2 + 4u * diag_rank + 1u; // E-V
    let d_2_2 = start_row_2 + 4u * diag_rank + 2u; // E-E
    let d_2_3 = start_row_2 + 4u * diag_rank + 3u; // E-P
    
    let d_3_0 = start_row_3 + 4u * diag_rank + 0u; // P-U
    let d_3_1 = start_row_3 + 4u * diag_rank + 1u; // P-V
    let d_3_2 = start_row_3 + 4u * diag_rank + 2u; // P-E
    let d_3_3 = start_row_3 + 4u * diag_rank + 3u; // P-P
    
    // Row 0 (U)
    matrix_values[d_0_0] = diag_u;
    matrix_values[d_0_1] = 0.0;
    matrix_values[d_0_2] = 0.0;
    matrix_values[d_0_3] = sum_diag_up;
    
    // Row 1 (V)
    matrix_values[d_1_0] = 0.0;
    matrix_values[d_1_1] = diag_v;
    matrix_values[d_1_2] = 0.0;
    matrix_values[d_1_3] = sum_diag_vp;
    
    // Row 2 (E)
    matrix_values[d_2_0] = 0.0;
    matrix_values[d_2_1] = 0.0;
    matrix_values[d_2_2] = diag_e;
    matrix_values[d_2_3] = 0.0; // sum_diag_ep if any
    
    // Row 3 (P)
    matrix_values[d_3_0] = sum_diag_pu;
    matrix_values[d_3_1] = sum_diag_pv;
    matrix_values[d_3_2] = 0.0; // sum_diag_pe
    matrix_values[d_3_3] = diag_p + sum_diag_pp;
    
    // Write RHS (4x4)
    rhs[4u * idx + 0u] = rhs_u;
    rhs[4u * idx + 1u] = rhs_v;
    rhs[4u * idx + 2u] = rhs_e;
    rhs[4u * idx + 3u] = rhs_p;

    // Write Scalar Pressure Diagonal
    scalar_matrix_values[scalar_diag_idx] = scalar_diag_p;

    // Write Diagonal Inverses
    // For Block Jacobi, this shader computes diagonal inverses.
    // Assuming 4x4 block inversion is handled separately or we invoke a separate kernel?
    // Current code: computes scalar diag inverses?
    // Lines 461-463: diag_u_inv[idx] = safe_inverse(diag_u).
    // This assumes simple diagonal scaling.
    // For 4x4 coupled, we might need full block inverse.
    // `coupled_assembly_merged.wgsl` only computes DIAGONAL entries for simple Jacobi?
    // Let's verify what `diag_u_inv` is used for.
    // Used in Smoother loop maybe?
    // If we switch to full Coupled, `b_diag_inv` (block inverse) is computed by `preconditioner.wgsl`.
    // `coupled_assembly` writes `matrix_values`.
    // The `diag_u_inv` outputs here seem to be for simple relaxation or scalar stages.
    
    diag_u_inv[idx] = safe_inverse(diag_u);
    diag_v_inv[idx] = safe_inverse(diag_v);
    diag_p_inv[idx] = safe_inverse(scalar_diag_p);
    // Maybe add diag_e_inv?
    // But let's leave as is for now, standard fields.

}
