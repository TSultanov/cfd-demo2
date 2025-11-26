use crate::solver::float::Float;
use crate::solver::linear_solver::SparseMatrix;
use crate::solver::mesh::{BoundaryType, Mesh};
use nalgebra::Vector2;
use std::collections::HashMap;

#[derive(Clone)]
pub struct ScalarField<T: Float> {
    pub values: Vec<T>,
}

impl<T: Float> ScalarField<T> {
    pub fn new(n: usize, val: T) -> Self {
        Self {
            values: vec![val; n],
        }
    }
}

#[derive(Clone)]
pub struct VectorField<T: Float> {
    pub vx: Vec<T>,
    pub vy: Vec<T>,
}

impl<T: Float> VectorField<T> {
    pub fn new(n: usize, val: Vector2<T>) -> Self {
        Self {
            vx: vec![val.x; n],
            vy: vec![val.y; n],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scheme {
    Upwind,
    Central,
    QUICK,
}

pub struct Fvm;

impl Fvm {
    pub fn compute_gradients<T: Float, F>(
        mesh: &Mesh,
        field: &ScalarField<T>,
        boundary_value: F,
        ghost_values: Option<&[T]>,
        ghost_map: Option<&HashMap<usize, usize>>,
        ghost_centers: Option<&Vec<Vector2<f64>>>,
    ) -> VectorField<T>
    where
        F: Fn(BoundaryType) -> Option<T>,
    {
        // Green-Gauss Gradient
        let n_cells = mesh.num_cells();
        let mut grad_x = vec![T::zero(); n_cells];
        let mut grad_y = vec![T::zero(); n_cells];

        for face_idx in 0..mesh.num_faces() {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];

            let val_owner = field.values[owner];

            let val_face = if let Some(neigh) = neighbor {
                // Linear interpolation for face value
                let val_neigh = field.values[neigh];

                let c_owner = Vector2::new(mesh.cell_cx[owner], mesh.cell_cy[owner]);
                let c_neigh = Vector2::new(mesh.cell_cx[neigh], mesh.cell_cy[neigh]);
                let f_center = Vector2::new(mesh.face_cx[face_idx], mesh.face_cy[face_idx]);

                let d_own = (f_center - c_owner).norm();
                let d_neigh = (f_center - c_neigh).norm();

                let f = T::val_from_f64(d_own / (d_own + d_neigh));

                val_owner + f * (val_neigh - val_owner)
            } else {
                // Boundary
                let mut val = val_owner;
                let mut handled = false;

                if let Some(bt) = mesh.face_boundary[face_idx] {
                    // Check Parallel Interface
                    if let Some(map) = ghost_map {
                        if let Some(&ghost_idx) = map.get(&face_idx) {
                            if let Some(ghosts) = ghost_values {
                                let local_ghost_idx = ghost_idx - n_cells;
                                if local_ghost_idx < ghosts.len() {
                                    let val_neigh = ghosts[local_ghost_idx];

                                    let f = if let Some(centers) = ghost_centers {
                                        if let Some(gc) = centers.get(local_ghost_idx) {
                                            let c_owner = Vector2::new(
                                                mesh.cell_cx[owner],
                                                mesh.cell_cy[owner],
                                            );
                                            let f_center = Vector2::new(
                                                mesh.face_cx[face_idx],
                                                mesh.face_cy[face_idx],
                                            );
                                            let d_own = (f_center - c_owner).norm();
                                            let dx = gc.x - f_center.x;
                                            let dy = gc.y - f_center.y;
                                            let d_neigh = (dx * dx + dy * dy).sqrt();
                                            T::val_from_f64(d_own / (d_own + d_neigh))
                                        } else {
                                            T::val_from_f64(0.5)
                                        }
                                    } else {
                                        T::val_from_f64(0.5)
                                    };

                                    val = val_owner + f * (val_neigh - val_owner);
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

            let nx = T::val_from_f64(mesh.face_nx[face_idx]);
            let ny = T::val_from_f64(mesh.face_ny[face_idx]);
            let area = T::val_from_f64(mesh.face_area[face_idx]);

            // Contribution to owner
            grad_x[owner] = grad_x[owner] + val_face * nx * area;
            grad_y[owner] = grad_y[owner] + val_face * ny * area;

            // Contribution to neighbor
            if let Some(neigh) = neighbor {
                grad_x[neigh] = grad_x[neigh] - val_face * nx * area;
                grad_y[neigh] = grad_y[neigh] - val_face * ny * area;
            }
        }

        let mut i = 0;
        while i < n_cells {
            let vol = mesh.cell_vol[i];
            if vol < 1e-12 {
                println!("Warning: Small cell volume for cell {}: {}", i, vol);
            }
            let vol_t = T::val_from_f64(vol);
            grad_x[i] = grad_x[i] / vol_t;
            grad_y[i] = grad_y[i] / vol_t;
            i += 1;
        }

        VectorField {
            vx: grad_x,
            vy: grad_y,
        }
    }

    // Assemble matrix for scalar transport:
    // d(phi)/dt + div(u phi) - div(gamma grad phi) = source
    pub fn assemble_scalar_transport<T: Float, F>(
        mesh: &Mesh,
        phi_old: &ScalarField<T>,
        fluxes: &[T],
        gamma: T,
        dt: T,
        scheme: &Scheme,
        boundary_value: F,
        ghost_map: Option<&HashMap<usize, usize>>,
        ghost_values: Option<&[T]>,
        ghost_centers: Option<&Vec<Vector2<f64>>>,
    ) -> (SparseMatrix<T>, Vec<T>)
    where
        F: Fn(BoundaryType) -> Option<T>,
    {
        let n_cells = mesh.num_cells();
        let mut triplets = Vec::with_capacity(n_cells + 2 * mesh.num_faces());
        let mut rhs = vec![T::zero(); n_cells];

        // Compute gradients for higher order schemes
        let grads = if *scheme != Scheme::Upwind {
            Some(Self::compute_gradients(
                mesh,
                phi_old,
                &boundary_value,
                ghost_values,
                ghost_map,
                ghost_centers,
            ))
        } else {
            None
        };

        // Vectorized Unsteady Term for RHS
        let mut i = 0;
        while i < n_cells {
            let vol = T::val_from_f64(mesh.cell_vol[i]);
            let coeff = vol / dt;
            rhs[i] = coeff * phi_old.values[i];
            i += 1;
        }

        for i in 0..n_cells {
            // Unsteady term: (phi - phi_old)/dt * V
            let vol = T::val_from_f64(mesh.cell_vol[i]);
            let coeff_unsteady = vol / dt;
            if coeff_unsteady.is_nan() {
                println!(
                    "coeff_unsteady is NaN for cell {}. vol={:?}, dt={:?}",
                    i, vol, dt
                );
            }
            triplets.push((i, i, coeff_unsteady));
            // rhs[i] += coeff_unsteady * phi_old.values[i]; // Handled above

            // Loop over faces
            let start = mesh.cell_face_offsets[i];
            let end = mesh.cell_face_offsets[i + 1];

            for k in start..end {
                let face_idx = mesh.cell_faces[k];
                let owner = mesh.face_owner[face_idx];
                let neighbor = mesh.face_neighbor[face_idx];
                let is_owner = owner == i;
                let neighbor_idx = if is_owner { neighbor } else { Some(owner) };

                let f_c = Vector2::new(mesh.face_cx[face_idx], mesh.face_cy[face_idx]);
                let f_area = T::val_from_f64(mesh.face_area[face_idx]);
                let f_normal = Vector2::new(
                    T::val_from_f64(mesh.face_nx[face_idx]),
                    T::val_from_f64(mesh.face_ny[face_idx]),
                );
                let c_i = Vector2::new(mesh.cell_cx[i], mesh.cell_cy[i]);

                // Flux OUT of cell i
                let flux_val = fluxes[face_idx];
                let flux = if is_owner { flux_val } else { -flux_val };

                if flux.is_nan() {
                    println!("flux is NaN for face {}", face_idx);
                }

                // Convection: div(u phi) -> sum(flux * phi_f)
                // Implicit Part: Upwind scheme (Diagonally Dominant)

                let phi_upwind = if flux > T::zero() {
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
                        let mut val = T::zero();
                        let mut handled = false;

                        if let Some(bt) = mesh.face_boundary[face_idx] {
                            // Check for Parallel Interface
                            if let Some(map) = ghost_map {
                                if let Some(&ghost_idx) = map.get(&face_idx) {
                                    // Treat as inflow from ghost cell
                                    triplets.push((i, ghost_idx, flux));

                                    if let Some(ghosts) = ghost_values {
                                        let local_ghost_idx = ghost_idx - mesh.num_cells();
                                        if local_ghost_idx < ghosts.len() {
                                            val = ghosts[local_ghost_idx];
                                        }
                                    }
                                    handled = true;
                                }
                            }

                            if !handled {
                                if let Some(bv) = boundary_value(bt) {
                                    // Dirichlet: flux * val_b. Move to RHS.
                                    rhs[i] = rhs[i] - flux * bv;
                                    val = bv;
                                } else {
                                    // Neumann (Zero Gradient): phi_b = phi_P
                                    if flux < T::zero() {
                                        // Inflow with Neumann is unstable and can make diagonal negative.
                                        // Treat as Dirichlet with value from previous step (Explicit)
                                        // phi_b = phi_old[i]
                                        rhs[i] = rhs[i] - flux * phi_old.values[i];
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
                        let d_own = (f_c - c_i).norm();
                        let d_neigh =
                            (f_c - Vector2::new(mesh.cell_cx[neigh], mesh.cell_cy[neigh])).norm();

                        match scheme {
                            Scheme::Central => {
                                // Linear Interpolation (Central Differencing)
                                let f = T::val_from_f64(d_own / (d_own + d_neigh));
                                let val_own = phi_old.values[i];
                                let val_neigh = phi_old.values[neigh];
                                phi_ho = val_own + f * (val_neigh - val_own);
                            }
                            Scheme::QUICK => {
                                // Linear Upwind (Second Order Upwind)
                                if flux > T::zero() {
                                    // From Owner
                                    let grad = Vector2::new(gradients.vx[i], gradients.vy[i]);
                                    let r = Vector2::new(
                                        T::val_from_f64(f_c.x - c_i.x),
                                        T::val_from_f64(f_c.y - c_i.y),
                                    );
                                    phi_ho = phi_old.values[i] + grad.dot(&r);
                                } else {
                                    // From Neighbor
                                    let grad =
                                        Vector2::new(gradients.vx[neigh], gradients.vy[neigh]);
                                    let r = Vector2::new(
                                        T::val_from_f64(f_c.x - mesh.cell_cx[neigh]),
                                        T::val_from_f64(f_c.y - mesh.cell_cy[neigh]),
                                    );
                                    phi_ho = phi_old.values[neigh] + grad.dot(&r);
                                }
                            }
                            _ => {}
                        }
                    } else {
                        // Boundary Face
                        let mut handled = false;
                        if let Some(_) = mesh.face_boundary[face_idx] {
                            if let Some(map) = ghost_map {
                                if map.contains_key(&face_idx) {
                                    // Parallel Interface
                                    // Retrieve ghost value
                                    let mut phi_ghost = T::zero();
                                    let mut has_ghost = false;
                                    if let Some(&ghost_idx) = map.get(&face_idx) {
                                        if let Some(ghosts) = ghost_values {
                                            let local_ghost_idx = ghost_idx - mesh.num_cells();
                                            if local_ghost_idx < ghosts.len() {
                                                phi_ghost = ghosts[local_ghost_idx];
                                                has_ghost = true;
                                            }
                                        }
                                    }

                                    if has_ghost {
                                        // Use Central Differencing for consistency across partition boundary
                                        // phi_ho = 0.5 * (phi_P + phi_ghost)
                                        phi_ho =
                                            T::val_from_f64(0.5) * (phi_old.values[i] + phi_ghost);
                                        handled = true;
                                    }
                                }
                            }
                        }

                        if !handled {
                            if flux > T::zero() {
                                // Outflow: Use Upwind (Zero Gradient) to prevent reflections.
                            }
                            // Inflow: phi_ho = phi_upwind (BC value)
                        }
                    }

                    // Correction term: flux * (phi_ho - phi_upwind)
                    // Move to RHS: ... = ... - correction
                    let correction = flux * (phi_ho - phi_upwind);
                    rhs[i] = rhs[i] - correction;
                }

                // Diffusion: -div(gamma grad phi) -> -sum(gamma * grad_phi * S)

                if let Some(neigh) = neighbor_idx {
                    let d_vec = Vector2::new(mesh.cell_cx[neigh], mesh.cell_cy[neigh]) - c_i;
                    let d = d_vec.norm();
                    if d < 1e-9 {
                        println!(
                            "Warning: Small distance between cells {} and {}: {}",
                            i, neigh, d
                        );
                    }
                    let diff_coeff = gamma * f_area / T::val_from_f64(d);

                    // - (diff_coeff * (phi_N - phi_P))
                    // = - diff_coeff * phi_N + diff_coeff * phi_P
                    // Add to LHS

                    triplets.push((i, i, diff_coeff));
                    triplets.push((i, neigh, -diff_coeff));

                    // Non-orthogonal correction
                    if let Some(gradients) = &grads {
                        // S = n * A
                        let s_vec = if is_owner { f_normal } else { -f_normal } * f_area;

                        let d_vec_t =
                            Vector2::new(T::val_from_f64(d_vec.x), T::val_from_f64(d_vec.y));
                        let d_t = T::val_from_f64(d);

                        let k_vec = s_vec - d_vec_t * (f_area / d_t);

                        // Interpolate gradient to face
                        let grad_own = Vector2::new(gradients.vx[i], gradients.vy[i]);
                        let grad_neigh = Vector2::new(gradients.vx[neigh], gradients.vy[neigh]);
                        // Linear interpolation based on distance
                        let d_own = (f_c - c_i).norm();
                        let d_neigh =
                            (f_c - Vector2::new(mesh.cell_cx[neigh], mesh.cell_cy[neigh])).norm();
                        let f = T::val_from_f64(d_own / (d_own + d_neigh));
                        let grad_f = grad_own + (grad_neigh - grad_own) * f;

                        let correction_flux = gamma * grad_f.dot(&k_vec);

                        // Subtract from RHS (since it's -div(flux))
                        // flux is out of i.
                        rhs[i] = rhs[i] - correction_flux;
                    }
                } else {
                    // Boundary diffusion
                    if let Some(bt) = mesh.face_boundary[face_idx] {
                        let d = (f_c - c_i).norm(); // Distance to face center
                        if d < 1e-9 {
                            println!(
                                "Warning: Small distance to boundary face {} for cell {}: {}",
                                face_idx, i, d
                            );
                        }

                        // Check Parallel Interface
                        let is_parallel = if let Some(map) = ghost_map {
                            if let Some(&ghost_idx) = map.get(&face_idx) {
                                // Treat as internal face
                                let dist = if let Some(centers) = ghost_centers {
                                    let local_ghost_idx = ghost_idx - mesh.num_cells();
                                    if let Some(gc) = centers.get(local_ghost_idx) {
                                        let dx = gc.x - mesh.cell_cx[i];
                                        let dy = gc.y - mesh.cell_cy[i];
                                        (dx * dx + dy * dy).sqrt()
                                    } else {
                                        2.0 * d
                                    }
                                } else {
                                    2.0 * d
                                };
                                let diff_coeff = gamma * f_area / T::val_from_f64(dist);

                                triplets.push((i, i, diff_coeff));
                                triplets.push((i, ghost_idx, -diff_coeff));
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if !is_parallel {
                            let diff_coeff = gamma * f_area / T::val_from_f64(d);

                            if diff_coeff.is_nan() {
                                println!(
                                    "diff_coeff is NaN for face {}. gamma={:?}, area={:?}, d={}",
                                    face_idx, gamma, f_area, d
                                );
                            }

                            if let Some(bv) = boundary_value(bt) {
                                // Dirichlet: - diff_coeff * (phi_b - phi_P)
                                // = - diff_coeff * phi_b + diff_coeff * phi_P
                                // LHS: + diff_coeff * phi_P
                                // RHS: + diff_coeff * phi_b
                                triplets.push((i, i, diff_coeff));
                                rhs[i] = rhs[i] + diff_coeff * bv;
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

    pub fn smooth_gradients<T: Float>(mesh: &Mesh, grads: &VectorField<T>) -> VectorField<T> {
        let mut smoothed_vx = grads.vx.clone();
        let mut smoothed_vy = grads.vy.clone();

        // Simple volume-weighted smoothing
        for i in 0..mesh.num_cells() {
            let vol = T::val_from_f64(mesh.cell_vol[i]);
            let mut sum_gx = grads.vx[i] * vol;
            let mut sum_gy = grads.vy[i] * vol;
            let mut sum_vol = vol;

            let start = mesh.cell_face_offsets[i];
            let end = mesh.cell_face_offsets[i + 1];
            for k in start..end {
                let face_idx = mesh.cell_faces[k];
                let owner = mesh.face_owner[face_idx];
                let neighbor = mesh.face_neighbor[face_idx];
                let n_idx = if owner == i { neighbor } else { Some(owner) };

                if let Some(n) = n_idx {
                    let n_vol = T::val_from_f64(mesh.cell_vol[n]);
                    sum_gx = sum_gx + grads.vx[n] * n_vol;
                    sum_gy = sum_gy + grads.vy[n] * n_vol;
                    sum_vol = sum_vol + n_vol;
                }
            }

            smoothed_vx[i] = sum_gx / sum_vol;
            smoothed_vy[i] = sum_gy / sum_vol;
        }

        VectorField {
            vx: smoothed_vx,
            vy: smoothed_vy,
        }
    }

    pub fn limit_gradients<T: Float>(grads: &VectorField<T>, max_mag: T) -> VectorField<T> {
        let mut limited_vx = grads.vx.clone();
        let mut limited_vy = grads.vy.clone();

        for i in 0..grads.vx.len() {
            let mag = (grads.vx[i].powi(2) + grads.vy[i].powi(2)).sqrt();
            if mag > max_mag {
                let scale = max_mag / mag;
                limited_vx[i] = limited_vx[i] * scale;
                limited_vy[i] = limited_vy[i] * scale;
            }
        }
        VectorField {
            vx: limited_vx,
            vy: limited_vy,
        }
    }

    pub fn limit_gradients_bj<T: Float>(
        mesh: &Mesh,
        field: &ScalarField<T>,
        grads: &mut VectorField<T>,
    ) {
        for i in 0..mesh.num_cells() {
            let val_p = field.values[i];
            let mut phi_max = val_p;
            let mut phi_min = val_p;

            let start = mesh.cell_face_offsets[i];
            let end = mesh.cell_face_offsets[i + 1];

            // Find min/max of neighbors
            for k in start..end {
                let face_idx = mesh.cell_faces[k];
                let owner = mesh.face_owner[face_idx];
                let neighbor = mesh.face_neighbor[face_idx];
                let n_idx = if owner == i { neighbor } else { Some(owner) };

                if let Some(n) = n_idx {
                    let val_n = field.values[n];
                    if val_n > phi_max {
                        phi_max = val_n;
                    }
                    if val_n < phi_min {
                        phi_min = val_n;
                    }
                }
            }

            // Barth-Jespersen Limiter
            let mut alpha = T::one();
            let grad_x = grads.vx[i];
            let grad_y = grads.vy[i];

            for k in start..end {
                let face_idx = mesh.cell_faces[k];
                let f_cx = T::val_from_f64(mesh.face_cx[face_idx]);
                let f_cy = T::val_from_f64(mesh.face_cy[face_idx]);
                let c_cx = T::val_from_f64(mesh.cell_cx[i]);
                let c_cy = T::val_from_f64(mesh.cell_cy[i]);

                let rx = f_cx - c_cx;
                let ry = f_cy - c_cy;

                let phi_face = val_p + grad_x * rx + grad_y * ry;

                if phi_face > phi_max {
                    let diff = phi_max - val_p;
                    let grad_diff = phi_face - val_p;
                    if grad_diff.abs() > T::val_from_f64(1e-12) {
                        alpha = alpha.min(diff / grad_diff);
                    }
                } else if phi_face < phi_min {
                    let diff = phi_min - val_p;
                    let grad_diff = phi_face - val_p;
                    if grad_diff.abs() > T::val_from_f64(1e-12) {
                        alpha = alpha.min(diff / grad_diff);
                    }
                }
            }

            grads.vx[i] = grads.vx[i] * alpha;
            grads.vy[i] = grads.vy[i] * alpha;
        }
    }
}
