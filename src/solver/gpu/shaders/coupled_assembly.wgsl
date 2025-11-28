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

// Group 2: Solver (Coupled)
@group(2) @binding(0) var<storage, read_write> matrix_values: array<f32>;
@group(2) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(2) @binding(2) var<storage, read> scalar_row_offsets: array<u32>;

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
    
    // Time derivative - this dominates the momentum diagonal
    let u_n = u_old[idx];
    let coeff_time = vol / constants.dt;
    diag_u += coeff_time;
    diag_v += coeff_time;
    rhs_u += coeff_time * u_n.x;
    rhs_v += coeff_time * u_n.y;
    
    // Strong pressure regularization to avoid saddle-point singularity
    // Scale with momentum coefficient to balance the system
    // This is artificial compressibility approach
    let p_regularization = 0.01 * coeff_time;
    diag_p += p_regularization;
    
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
            // This is a boundary face or something that doesn't have a matrix entry?
            // Wait, boundary faces DO have matrix entries if they are implicit.
            // But here we only store entries for neighbors in the adjacency list.
            // The diagonal entry is handled separately (accumulated).
            // But we need to write the diagonal entry to the matrix too!
            // In scalar matrix, diagonal is stored as a neighbor of itself.
            // So `scalar_mat_idx` should be valid for diagonal too.
            // Let's assume it is valid.
            neighbor_rank = scalar_mat_idx - scalar_offset;
        }
        
        let idx_0_0 = start_row_0 + 3u * neighbor_rank + 0u;
        let idx_0_1 = start_row_0 + 3u * neighbor_rank + 1u;
        let idx_0_2 = start_row_0 + 3u * neighbor_rank + 2u;
        
        let idx_1_0 = start_row_1 + 3u * neighbor_rank + 0u;
        let idx_1_1 = start_row_1 + 3u * neighbor_rank + 1u;
        let idx_1_2 = start_row_1 + 3u * neighbor_rank + 2u;
        
        let idx_2_0 = start_row_2 + 3u * neighbor_rank + 0u;
        let idx_2_1 = start_row_2 + 3u * neighbor_rank + 1u;
        let idx_2_2 = start_row_2 + 3u * neighbor_rank + 2u;
        
        // Initialize block to zero (since we might visit the same neighbor multiple times if multiple faces connect? No, usually 1 face)
        // But we are accumulating into `matrix_values`. We should probably zero it first?
        // No, the matrix is zeroed before assembly.
        
        if (!is_boundary) {
            // --- Momentum Equations ---
            let coeff = -diff_coeff + conv_coeff_off;
            
            // Off-diagonal U-U and V-V
            matrix_values[idx_0_0] = coeff; // A_uu
            matrix_values[idx_1_1] = coeff; // A_vv
            
            diag_u += diff_coeff + conv_coeff_diag;
            diag_v += diff_coeff + conv_coeff_diag;
            
            // Pressure Gradient (Cell to Face)
            // grad p ~ (p_neigh - p_own) / dist * area * normal
            // Contribution to U eq: - (p_neigh - p_own)/dist * area * nx
            // = - (area * nx / dist) * p_neigh + (area * nx / dist) * p_own
            
            let pg_coeff_x = area * normal.x / dist;
            let pg_coeff_y = area * normal.y / dist;
            
            // Off-diagonal U-P and V-P (Neighbor P)
            matrix_values[idx_0_2] = -pg_coeff_x; // A_up
            matrix_values[idx_1_2] = -pg_coeff_y; // A_vp
            
            // Diagonal U-P and V-P (Own P) - accumulated
            // We need to add this to the diagonal block entry.
            // The diagonal block corresponds to `neighbor_rank` where `neigh_idx == idx`.
            // But here we are processing a neighbor face.
            // We can't easily write to the diagonal block from here without searching for it.
            // BUT, we can just add it to `rhs`? No, it's implicit.
            // We need to find the diagonal block index.
            // The diagonal block is always the last one in the sorted list? Or first?
            // In `init_coupled_resources`, we added `i` to the list and sorted.
            // So we need to find the rank of `i`.
            // We can pre-calculate `diagonal_indices` for coupled matrix?
            // We have `diagonal_indices` for scalar matrix!
            // `scalar_diag_idx = diagonal_indices[idx]`.
            // `diag_rank = scalar_diag_idx - scalar_offset`.
            
            let scalar_diag_idx = diagonal_indices[idx];
            let diag_rank = scalar_diag_idx - scalar_offset;
            
            let diag_0_2 = start_row_0 + 3u * diag_rank + 2u;
            let diag_1_2 = start_row_1 + 3u * diag_rank + 2u;
            
            // Atomic add? No, no atomics for f32.
            // But we are the only thread processing this cell.
            // So we can safely write to our own diagonal block.
            // Wait, `matrix_values` is read_write.
            // We can read, add, write.
            
            matrix_values[diag_0_2] += pg_coeff_x;
            matrix_values[diag_1_2] += pg_coeff_y;
            
            // --- Continuity Equation ---
            // div u = 0
            // sum (u_f . n * A) = 0
            // u_f = 0.5*(u_P + u_N) - d_p * (p_N - p_P) ... (Rhie-Chow)
            // Actually, simple interpolation + pressure correction.
            // u_f = interp(u) - D * (grad p - interp(grad p))
            // Here we linearize u_f as:
            // u_f ~ 0.5 * u_P + 0.5 * u_N
            // So contribution to P eq from U, V:
            // 0.5 * nx * A * u_P + 0.5 * nx * A * u_N ...
            
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
            // - D * (p_N - p_P) / dist * area
            // D is usually vol/A_P.
            // We need D_f. Interpolated from D_P and D_N.
            // D_P = vol / a_P_u.
            // We don't have a_P_u yet! It's being built right now (diag_u).
            // This is the problem with coupled assembly. We need coefficients to compute Rhie-Chow.
            // Usually we use coefficients from previous iteration.
            // `d_p` buffer stores the inverse diagonal from previous iteration.
            
            let dp_f = 0.5 * (d_p[idx] + d_p[other_idx]);
            let lapl_coeff = dp_f * area / dist;
            
            // Off-diagonal P-P (Neighbor P)
            matrix_values[idx_2_2] = -lapl_coeff; // A_pp
            
            // Diagonal P-P (Own P)
            let diag_2_2 = start_row_2 + 3u * diag_rank + 2u;
            matrix_values[diag_2_2] += lapl_coeff;
            
        } else {
            // Boundary Conditions
            if (boundary_type == 1u) { // Inlet
                let ramp = smoothstep(0.0, 0.1, constants.time);
                let u_bc_x = 1.0 * ramp;
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
                
                // Pressure gradient at inlet? Usually zero gradient or fixed p.
                // If fixed p, we add contribution.
                // If zero gradient, no contribution.
                
                // Continuity at inlet:
                // Flux is fixed (u_bc). So it goes to RHS.
                let flux_bc = (u_bc_x * normal.x + u_bc_y * normal.y) * area;
                rhs_p -= flux_bc;
                
            } else if (boundary_type == 3u) { // Wall
                // No slip: u = 0
                diag_u += diff_coeff;
                diag_v += diff_coeff;
                
                // Pressure gradient: zero gradient (dp/dn = 0) -> no contribution
                
                // Continuity: zero flux -> no contribution
                
            } else if (boundary_type == 2u) { // Outlet
                // Zero gradient U
                // No contribution to diag or RHS for diffusion?
                // Usually diffusion is zero gradient.
                
                if (flux > 0.0) {
                    diag_u += flux;
                    diag_v += flux;
                }
                
                // Fixed Pressure p = 0
                // Gradient term: - (p_bc - p_own)/dist * area * n
                // = (area * n / dist) * p_own
                let pg_coeff_x = area * normal.x / dist;
                let pg_coeff_y = area * normal.y / dist;
                
                let scalar_diag_idx = diagonal_indices[idx];
                let diag_rank = scalar_diag_idx - scalar_offset;
                let diag_0_2 = start_row_0 + 3u * diag_rank + 2u;
                let diag_1_2 = start_row_1 + 3u * diag_rank + 2u;
                
                matrix_values[diag_0_2] += pg_coeff_x;
                matrix_values[diag_1_2] += pg_coeff_y;
                
                // Continuity:
                // Flux depends on U.
                // 0.5 * u_P ...
                let div_coeff_x = normal.x * area; // One sided?
                let div_coeff_y = normal.y * area;
                
                let diag_2_0 = start_row_2 + 3u * diag_rank + 0u;
                let diag_2_1 = start_row_2 + 3u * diag_rank + 1u;
                
                matrix_values[diag_2_0] += div_coeff_x;
                matrix_values[diag_2_1] += div_coeff_y;
                
                // Rhie-Chow at outlet (fixed p)
                // - D * (p_bc - p_P) / dist
                let dp_f = d_p[idx];
                let lapl_coeff = dp_f * area / dist;
                let diag_2_2 = start_row_2 + 3u * diag_rank + 2u;
                matrix_values[diag_2_2] += lapl_coeff;
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
