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
        
        let dx = center.x - other_center.x;
        let dy = center.y - other_center.y;
        let dist = sqrt(dx*dx + dy*dy);
        
        // Diffusion: nu * A / dist
        let diff_coeff = constants.viscosity * area / dist;
        
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
                if (flux > 0.0) {
                    diag_coeff += flux;
                } else {
                    // Backflow: Treat explicitly to maintain diagonal dominance
                    rhs_val -= flux * val_old;
                }
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
    }
    
    // Pressure Gradient Source: -grad(p) * V / rho
    if (constants.component == 0u) {
        grad_p_accum.x /= vol;
        grad_p_accum.y /= vol;
        grad_p[idx] = grad_p_accum;
        
        rhs_val -= (grad_p_accum.x / constants.density) * vol;
    } else {
        let gp = grad_p[idx];
        rhs_val -= (gp.y / constants.density) * vol;
    }

    // Time term
    let time_coeff = vol / constants.dt;
    diag_coeff += time_coeff;
    rhs_val += time_coeff * val_old;

    let diag_idx = diagonal_indices[idx];
    matrix_values[diag_idx] = diag_coeff;

    if (constants.component == 0u) {
        if (abs(diag_coeff) > 1e-20) {
            d_p[idx] = vol / (diag_coeff * constants.density);
        } else {
            d_p[idx] = 0.0;
        }
    }
    
    rhs[idx] = rhs_val;
}
