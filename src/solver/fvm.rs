use crate::solver::mesh::{Mesh, BoundaryType};
use crate::solver::linear_solver::SparseMatrix;
use nalgebra::Vector2;
use std::collections::HashMap;

#[derive(Clone)]
pub struct ScalarField {
    pub values: Vec<f64>,
}

impl ScalarField {
    pub fn new(n: usize, val: f64) -> Self {
        Self { values: vec![val; n] }
    }
}

#[derive(Clone)]
pub struct VectorField {
    pub values: Vec<Vector2<f64>>,
}

impl VectorField {
    pub fn new(n: usize, val: Vector2<f64>) -> Self {
        Self { values: vec![val; n] }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scheme {
    Upwind,
    Central,
    QUICK,
}

pub struct Fvm {
    // Helper for FVM operations
}

impl Fvm {
    pub fn compute_gradients<F>(
        mesh: &Mesh, 
        field: &ScalarField, 
        boundary_value: F,
        ghost_values: Option<&[f64]>,
        ghost_map: Option<&HashMap<usize, usize>>
    ) -> Vec<Vector2<f64>> 
    where F: Fn(BoundaryType) -> Option<f64>
    {
        // Green-Gauss Gradient
        let mut grads = vec![Vector2::zeros(); mesh.cells.len()];
        
        for (face_idx, face) in mesh.faces.iter().enumerate() {
            let owner = face.owner;
            let neighbor = face.neighbor;
            
            let val_owner = field.values[owner];
            
            let val_face = if let Some(neigh) = neighbor {
                // Linear interpolation for face value
                let val_neigh = field.values[neigh];
                // Simple average or distance weighted?
                // Distance weighted
                let d_own = (face.center - mesh.cells[owner].center).norm();
                let d_neigh = (face.center - mesh.cells[neigh].center).norm();
                // phi_f = phi_P + f * (phi_N - phi_P)
                // f = d_own / (d_own + d_neigh)
                let f = d_own / (d_own + d_neigh);
                
                val_owner + f * (val_neigh - val_owner)
            } else {
                // Boundary
                let mut val = val_owner;
                let mut handled = false;

                if let Some(bt) = face.boundary_type {
                    // Check Parallel Interface
                    if let Some(map) = ghost_map {
                        if let Some(&ghost_idx) = map.get(&face_idx) {
                            if let Some(ghosts) = ghost_values {
                                let local_ghost_idx = ghost_idx - mesh.cells.len();
                                if local_ghost_idx < ghosts.len() {
                                    let val_neigh = ghosts[local_ghost_idx];
                                    // Interpolate
                                    val = 0.5 * (val_owner + val_neigh);
                                    handled = true;
                                }
                            }
                        }
                    }

                    if !handled {
                        if let Some(bv) = boundary_value(bt) {
                            val = bv; // Dirichlet
                        } else {
                            val = val_owner; // Neumann (Zero Gradient)
                        }
                    }
                }
                val
            };
            
            let area_vec = face.normal * face.area;
            
            // Contribution to owner
            grads[owner] += val_face * area_vec;
            
            // Contribution to neighbor
            if let Some(neigh) = neighbor {
                grads[neigh] -= val_face * area_vec;
            }
        }
        
        for (i, cell) in mesh.cells.iter().enumerate() {
            if cell.volume < 1e-12 {
                println!("Warning: Small cell volume for cell {}: {}", i, cell.volume);
            }
            grads[i] /= cell.volume;
        }
        
        grads
    }
    
    // Assemble matrix for scalar transport:
    // d(phi)/dt + div(u phi) - div(gamma grad phi) = source
    pub fn assemble_scalar_transport<F>(
        mesh: &Mesh,
        phi_old: &ScalarField,
        fluxes: &Vec<f64>,
        gamma: f64,
        dt: f64,
        scheme: &Scheme,
        boundary_value: F,
        ghost_map: Option<&HashMap<usize, usize>>,
        ghost_values: Option<&[f64]>,
    ) -> (SparseMatrix, Vec<f64>) 
    where F: Fn(BoundaryType) -> Option<f64>
    {
        let n_cells = mesh.cells.len();
        let mut triplets = Vec::new();
        let mut rhs = vec![0.0; n_cells];
        
        // Compute gradients for higher order schemes
        let grads = if matches!(scheme, Scheme::Central | Scheme::QUICK) {
             Some(Self::compute_gradients(mesh, phi_old, &boundary_value, ghost_values, ghost_map))
        } else {
             None
        };
        
        for (i, cell) in mesh.cells.iter().enumerate() {
            // Unsteady term: (phi - phi_old)/dt * V
            let coeff_unsteady = cell.volume / dt;
            if coeff_unsteady.is_nan() {
                println!("coeff_unsteady is NaN for cell {}. vol={}, dt={}", i, cell.volume, dt);
            }
            triplets.push((i, i, coeff_unsteady));
            rhs[i] += coeff_unsteady * phi_old.values[i];
            
            // Loop over faces
            for &face_idx in &cell.face_indices {
                let face = &mesh.faces[face_idx];
                let is_owner = face.owner == i;
                let neighbor_idx = if is_owner { face.neighbor } else { Some(face.owner) };
                
                // Flux OUT of cell i
                let flux = if is_owner { fluxes[face_idx] } else { -fluxes[face_idx] };
                
                if flux.is_nan() {
                    println!("flux is NaN for face {}", face_idx);
                }
                
                // Convection: div(u phi) -> sum(flux * phi_f)
                // Implicit Part: Upwind scheme (Diagonally Dominant)
                
                let phi_upwind = if flux > 0.0 {
                    // Outflow: phi_f = phi_P
                    triplets.push((i, i, flux));
                    phi_old.values[i]
                } else {
                    // Inflow
                    if let Some(neigh) = neighbor_idx {
                        // phi_f = phi_N
                        triplets.push((i, neigh, flux));
                        phi_old.values[neigh]
                    } else {
                        // Boundary inflow
                        let mut val = 0.0;
                        let mut handled = false;

                        if let Some(bt) = face.boundary_type {
                            // Check for Parallel Interface
                            if let Some(map) = ghost_map {
                                if let Some(&ghost_idx) = map.get(&face_idx) {
                                    // Treat as inflow from ghost cell
                                    triplets.push((i, ghost_idx, flux));
                                    
                                    if let Some(ghosts) = ghost_values {
                                        let local_ghost_idx = ghost_idx - mesh.cells.len();
                                        if local_ghost_idx < ghosts.len() {
                                            val = ghosts[local_ghost_idx];
                                            // Debug print
                                            // if i == 0 { println!("Ghost val: {}", val); }
                                        }
                                    }
                                    handled = true;
                                }
                            }

                            if !handled {
                                if let Some(bv) = boundary_value(bt) {
                                    // Dirichlet: flux * val_b. Move to RHS.
                                    rhs[i] -= flux * bv;
                                    val = bv;
                                } else {
                                    // Neumann (Zero Gradient): phi_b = phi_P
                                    if flux < 0.0 {
                                        // Inflow with Neumann is unstable and can make diagonal negative.
                                        // Treat as Dirichlet with value from previous step (Explicit)
                                        // phi_b = phi_old[i]
                                        rhs[i] -= flux * phi_old.values[i];
                                        val = phi_old.values[i];
                                    } else {
                                        triplets.push((i, i, flux));
                                        val = phi_old.values[i];
                                    }
                                }
                            }
                        } else {
                             triplets.push((i, i, flux));
                             val = phi_old.values[i];
                        }
                        val
                    }
                };
                
                // Explicit Correction (Deferred Correction) for Higher Order Schemes
                if let Some(gradients) = &grads {
                    let mut phi_ho = phi_upwind;
                    
                    if let Some(neigh) = neighbor_idx {
                        // Internal Face
                        let d_own = (face.center - mesh.cells[i].center).norm();
                        let d_neigh = (face.center - mesh.cells[neigh].center).norm();
                        
                        match scheme {
                            Scheme::Central => {
                                // Linear Interpolation (Central Differencing)
                                let f = d_own / (d_own + d_neigh);
                                let val_own = phi_old.values[i];
                                let val_neigh = phi_old.values[neigh];
                                phi_ho = val_own + f * (val_neigh - val_own);
                            },
                            Scheme::QUICK => {
                                // Linear Upwind (Second Order Upwind)
                                if flux > 0.0 {
                                    // From Owner
                                    let grad = gradients[i];
                                    let r = face.center - mesh.cells[i].center;
                                    phi_ho = phi_old.values[i] + grad.dot(&r);
                                } else {
                                    // From Neighbor
                                    let grad = gradients[neigh];
                                    let r = face.center - mesh.cells[neigh].center;
                                    phi_ho = phi_old.values[neigh] + grad.dot(&r);
                                }
                            },
                            _ => {}
                        }
                    } else {
                        // Boundary Face
                        let mut handled = false;
                        if let Some(_) = face.boundary_type {
                            if let Some(map) = ghost_map {
                                if map.contains_key(&face_idx) {
                                    // Parallel Interface
                                    // Retrieve ghost value
                                    let mut phi_ghost = 0.0;
                                    let mut has_ghost = false;
                                    if let Some(&ghost_idx) = map.get(&face_idx) {
                                        if let Some(ghosts) = ghost_values {
                                            let local_ghost_idx = ghost_idx - mesh.cells.len();
                                            if local_ghost_idx < ghosts.len() {
                                                phi_ghost = ghosts[local_ghost_idx];
                                                has_ghost = true;
                                            }
                                        }
                                    }

                                    if has_ghost {
                                        // Use Central Differencing for consistency across partition boundary
                                        // phi_ho = 0.5 * (phi_P + phi_ghost)
                                        phi_ho = 0.5 * (phi_old.values[i] + phi_ghost);
                                        handled = true;
                                    }
                                }
                            }
                        }

                        if !handled {
                            if flux > 0.0 {
                                // Outflow: Use Upwind (Zero Gradient) to prevent reflections.
                                // Extrapolating with gradients at the outlet can be unstable and cause
                                // unphysical reflections if the gradient is not well-behaved.
                                // phi_ho = phi_upwind; // Already set to phi_old.values[i]
                            }
                            // Inflow: phi_ho = phi_upwind (BC value)
                        }
                    }
                    
                    // Correction term: flux * (phi_ho - phi_upwind)
                    // Move to RHS: ... = ... - correction
                    let correction = flux * (phi_ho - phi_upwind);
                    rhs[i] -= correction;
                }
                
                // Diffusion: -div(gamma grad phi) -> -sum(gamma * grad_phi * S)
                // grad_phi * S ~ (phi_N - phi_P) / d * Area
                // d = distance between centers
                
                if let Some(neigh) = neighbor_idx {
                    let d_vec = mesh.cells[neigh].center - cell.center;
                    let d = d_vec.norm();
                    if d < 1e-9 {
                        println!("Warning: Small distance between cells {} and {}: {}", i, neigh, d);
                    }
                    let diff_coeff = gamma * face.area / d;
                    
                    // - (diff_coeff * (phi_N - phi_P))
                    // = - diff_coeff * phi_N + diff_coeff * phi_P
                    // Add to LHS
                    
                    triplets.push((i, i, diff_coeff));
                    triplets.push((i, neigh, -diff_coeff));

                    // Non-orthogonal correction
                    if let Some(gradients) = &grads {
                        // S = n * A
                        // If is_owner, normal points i -> neigh. S is out of i.
                        // If !is_owner, normal points neigh -> i. S is out of i (so -normal).
                        let s_vec = if is_owner { face.normal } else { -face.normal } * face.area;
                        
                        // Over-relaxed correction vector k
                        // k = S - (S . d / |d|^2) * d  <-- Minimum correction
                        // k = S - (|S|/|d|) * d        <-- Over-relaxed correction
                        // We use Over-relaxed to match the implicit coefficient (gamma * A / d)
                        // A = |S|. So implicit term is gamma * |S|/|d|.
                        
                        let k_vec = s_vec - d_vec * (face.area / d);
                        
                        // Interpolate gradient to face
                        let grad_own = gradients[i];
                        let grad_neigh = gradients[neigh];
                        // Linear interpolation based on distance
                        let d_own = (face.center - mesh.cells[i].center).norm();
                        let d_neigh = (face.center - mesh.cells[neigh].center).norm();
                        let f = d_own / (d_own + d_neigh);
                        let grad_f = grad_own + (grad_neigh - grad_own) * f;
                        
                        let correction_flux = gamma * grad_f.dot(&k_vec);
                        
                        // Subtract from RHS (since it's -div(flux))
                        // flux is out of i.
                        rhs[i] -= correction_flux;
                    }
                } else {
                    // Boundary diffusion
                    if let Some(bt) = face.boundary_type {
                        let d = (face.center - cell.center).norm(); // Distance to face center
                        if d < 1e-9 {
                            println!("Warning: Small distance to boundary face {} for cell {}: {}", face_idx, i, d);
                        }
                        
                        // Check Parallel Interface
                        let is_parallel = if let Some(map) = ghost_map {
                            if let Some(&ghost_idx) = map.get(&face_idx) {
                                // Treat as internal face
                                // Approx distance: 2 * d (assuming uniform mesh across boundary)
                                let dist = 2.0 * d;
                                let diff_coeff = gamma * face.area / dist;
                                
                                triplets.push((i, i, diff_coeff));
                                triplets.push((i, ghost_idx, -diff_coeff));
                                true
                            } else { false }
                        } else { false };

                        if !is_parallel {
                            let diff_coeff = gamma * face.area / d;
                            
                            if diff_coeff.is_nan() {
                                println!("diff_coeff is NaN for face {}. gamma={}, area={}, d={}", face_idx, gamma, face.area, d);
                            }

                            if let Some(bv) = boundary_value(bt) {
                                // Dirichlet: - diff_coeff * (phi_b - phi_P)
                                // = - diff_coeff * phi_b + diff_coeff * phi_P
                                // LHS: + diff_coeff * phi_P
                                // RHS: + diff_coeff * phi_b
                                triplets.push((i, i, diff_coeff));
                                rhs[i] += diff_coeff * bv;
                            } else {
                                // Neumann (Zero Gradient): grad_phi = 0 -> flux = 0
                                // No contribution
                            }
                        }
                    }
                }
            }
        }
        
        let n_cols = if let Some(map) = ghost_map {
            n_cells + map.len()
        } else {
            n_cells
        };
        (SparseMatrix::from_triplets(n_cells, n_cols, &triplets), rhs)
    }
    
    pub fn smooth_gradients(mesh: &Mesh, grads: &[Vector2<f64>]) -> Vec<Vector2<f64>> {
        let mut smoothed = grads.to_vec();
        
        // Simple volume-weighted smoothing
        for i in 0..mesh.cells.len() {
            let cell = &mesh.cells[i];
            let mut sum_g = grads[i] * cell.volume;
            let mut sum_vol = cell.volume;
            
            for &face_idx in &cell.face_indices {
                let face = &mesh.faces[face_idx];
                let neighbor = if face.owner == i { face.neighbor } else { Some(face.owner) };
                
                if let Some(n) = neighbor {
                    let n_vol = mesh.cells[n].volume;
                    sum_g += grads[n] * n_vol;
                    sum_vol += n_vol;
                }
            }
            
            smoothed[i] = sum_g / sum_vol;
        }
        
        smoothed
    }
    
    pub fn limit_gradients(grads: &[Vector2<f64>], max_mag: f64) -> Vec<Vector2<f64>> {
        grads.iter().map(|g| {
            let mag = g.norm();
            if mag > max_mag {
                g * (max_mag / mag)
            } else {
                *g
            }
        }).collect()
    }
}
