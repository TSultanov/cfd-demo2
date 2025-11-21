use crate::solver::mesh::{Mesh, BoundaryType};
use crate::solver::linear_solver::SparseMatrix;
use nalgebra::Vector2;

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
    pub fn compute_gradients<F>(mesh: &Mesh, field: &ScalarField, boundary_value: F) -> Vec<Vector2<f64>> 
    where F: Fn(BoundaryType) -> Option<f64>
    {
        // Green-Gauss Gradient
        let mut grads = vec![Vector2::zeros(); mesh.cells.len()];
        
        for face in mesh.faces.iter() {
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
                if let Some(bt) = face.boundary_type {
                    if let Some(bv) = boundary_value(bt) {
                        bv // Dirichlet
                    } else {
                        val_owner // Neumann (Zero Gradient)
                    }
                } else {
                    val_owner
                }
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
        fluxes: &Vec<f64>, // Mass flux at faces
        gamma: f64,
        dt: f64,
        scheme: &Scheme,
        boundary_value: F,
    ) -> (SparseMatrix, Vec<f64>) 
    where F: Fn(BoundaryType) -> Option<f64>
    {
        let n_cells = mesh.cells.len();
        let mut triplets = Vec::new();
        let mut rhs = vec![0.0; n_cells];
        
        // Compute gradients for higher order schemes
        let grads = if matches!(scheme, Scheme::Central | Scheme::QUICK) {
             Some(Self::compute_gradients(mesh, phi_old, &boundary_value))
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
                        if let Some(bt) = face.boundary_type {
                            if let Some(bv) = boundary_value(bt) {
                                // Dirichlet: flux * val_b. Move to RHS.
                                rhs[i] -= flux * bv;
                                bv
                            } else {
                                // Neumann (Zero Gradient): phi_b = phi_P
                                if flux < 0.0 {
                                    // Inflow with Neumann is unstable and can make diagonal negative.
                                    // Treat as Dirichlet with value from previous step (Explicit)
                                    // phi_b = phi_old[i]
                                    rhs[i] -= flux * phi_old.values[i];
                                    phi_old.values[i]
                                } else {
                                    triplets.push((i, i, flux));
                                    phi_old.values[i]
                                }
                            }
                        } else {
                             triplets.push((i, i, flux));
                             phi_old.values[i]
                        }
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
                        if flux > 0.0 {
                            // Outflow: Use Upwind (Zero Gradient) to prevent reflections.
                            // Extrapolating with gradients at the outlet can be unstable and cause
                            // unphysical reflections if the gradient is not well-behaved.
                            // phi_ho = phi_upwind; // Already set to phi_old.values[i]
                        }
                        // Inflow: phi_ho = phi_upwind (BC value)
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
                    let d = (mesh.cells[neigh].center - cell.center).norm();
                    if d < 1e-9 {
                        println!("Warning: Small distance between cells {} and {}: {}", i, neigh, d);
                    }
                    let diff_coeff = gamma * face.area / d;
                    
                    // - (diff_coeff * (phi_N - phi_P))
                    // = - diff_coeff * phi_N + diff_coeff * phi_P
                    // Add to LHS
                    
                    triplets.push((i, i, diff_coeff));
                    triplets.push((i, neigh, -diff_coeff));
                } else {
                    // Boundary diffusion
                    if let Some(bt) = face.boundary_type {
                        let d = (face.center - cell.center).norm(); // Distance to face center
                        if d < 1e-9 {
                            println!("Warning: Small distance to boundary face {} for cell {}: {}", face_idx, i, d);
                        }
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
        
        (SparseMatrix::from_triplets(n_cells, n_cells, &triplets), rhs)
    }
}
