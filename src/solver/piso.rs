use crate::solver::fvm::{Fvm, ScalarField, Scheme, VectorField};
use crate::solver::linear_solver::{solve_bicgstab, SerialOps, SparseMatrix, SolverOps};
use crate::solver::mesh::{BoundaryType, Mesh};
use crate::solver::float::{Float, Simd};
use nalgebra::Vector2;
use std::collections::HashMap;

pub struct PisoSolver<T: Float> {
    pub mesh: Mesh,
    pub u: VectorField<T>,
    pub p: ScalarField<T>,
    pub fluxes: Vec<T>, // Mass flux at faces
    pub dt: T,
    pub time: T,
    pub residuals: Vec<(String, T)>,
    pub viscosity: T,
    pub density: T,
    pub scheme: Scheme,
    pub ghost_map: HashMap<usize, usize>, // face_idx -> ghost_col_idx
    pub ghost_centers: Vec<Vector2<f64>>, // Mesh geometry is f64
    pub save_debug_data: bool,
    pub last_pressure_matrix: Option<SparseMatrix<T>>,
    pub last_pressure_rhs: Option<Vec<T>>,
    pub cell_vol: Vec<T>, // Cached cell volumes in T
}

impl<T: Float> PisoSolver<T> {
    pub fn new(mesh: Mesh) -> Self {
        let n_cells = mesh.num_cells();
        let n_faces = mesh.num_faces();
        let cell_vol: Vec<T> = mesh.cell_vol.iter().map(|&v| T::val_from_f64(v)).collect();
        
        let solver = Self {
            mesh,
            u: VectorField::new(n_cells, Vector2::new(T::zero(), T::zero())),
            p: ScalarField::new(n_cells, T::zero()),
            fluxes: vec![T::zero(); n_faces],
            dt: T::val_from_f64(0.001), // Smaller time step
            time: T::zero(),
            residuals: Vec::new(),
            viscosity: T::val_from_f64(0.01), // Kinematic viscosity
            density: T::one(),
            scheme: Scheme::Upwind,
            ghost_map: HashMap::new(),
            ghost_centers: Vec::new(),
            save_debug_data: false,
            last_pressure_matrix: None,
            last_pressure_rhs: None,
            cell_vol,
        };
        solver.check_connectivity();
        solver
    }

    pub fn step(&mut self) {
        self.step_with_ops(&SerialOps)
    }

    pub fn step_with_ops<O: SolverOps<T>>(&mut self, ops: &O) {
        // Clear residuals from previous step
        self.residuals.clear();
        
        // println!("Starting step at time {:.4}", self.time);
        if self.fluxes.iter().any(|f| f.is_nan()) {
            println!("Fluxes contain NaN at start of step!");
        }

        // Boundary Conditions
        let u_bc = |bt: BoundaryType| match bt {
            BoundaryType::Inlet => Some(T::one()),
            BoundaryType::Wall => Some(T::zero()),
            BoundaryType::Outlet => None, // Neumann
            BoundaryType::ParallelInterface(_, _) => None,
        };
        let v_bc = |bt: BoundaryType| match bt {
            BoundaryType::Inlet => Some(T::zero()),
            BoundaryType::Wall => Some(T::zero()),
            BoundaryType::Outlet => None, // Neumann
            BoundaryType::ParallelInterface(_, _) => None,
        };
        let p_bc = |bt: BoundaryType| match bt {
            BoundaryType::Outlet => Some(T::zero()),
            _ => None, // Neumann (Inlet, Wall, ParallelInterface)
        };

        let nu = self.viscosity / self.density;

        // Exchange ghosts
        let u_x_vals: Vec<T> = self.u.vx.clone();
        let u_y_vals: Vec<T> = self.u.vy.clone();
        let p_vals: Vec<T> = self.p.values.clone();

        let u_x_ghosts = ops.exchange_halo(&u_x_vals);
        let u_y_ghosts = ops.exchange_halo(&u_y_vals);
        let p_ghosts = ops.exchange_halo(&p_vals);

        // 1. Momentum Predictor

        // Solve Ux
        let u_x = ScalarField { values: u_x_vals };
        let (mat_u, rhs_u_base) = Fvm::assemble_scalar_transport(
            &self.mesh,
            &u_x,
            &self.fluxes,
            nu,
            self.dt,
            &self.scheme,
            u_bc,
            Some(&self.ghost_map),
            Some(&u_x_ghosts),
            Some(&self.ghost_centers),
        );

        // Add pressure gradient source term to RHS
        // -grad(p) / rho
        let grad_p = Fvm::compute_gradients(
            &self.mesh,
            &self.p,
            p_bc,
            Some(&p_ghosts),
            Some(&self.ghost_map),
            Some(&self.ghost_centers),
        );
        let mut rhs_ux = rhs_u_base.clone();

        // Re-assemble for Uy
        let u_y = ScalarField { values: u_y_vals };
        let (mat_v, mut rhs_uy) = Fvm::assemble_scalar_transport(
            &self.mesh,
            &u_y,
            &self.fluxes,
            nu,
            self.dt,
            &self.scheme,
            v_bc,
            Some(&self.ghost_map),
            Some(&u_y_ghosts),
            Some(&self.ghost_centers),
        );

        let mut i = 0;
        let n_cells = self.mesh.num_cells();
        let lanes = T::Simd::LANES;
        let inv_density = T::one() / self.density;
        let v_inv_rho = T::Simd::splat(inv_density);

        while i + lanes <= n_cells {
            let v_vol = T::Simd::from_slice(&self.cell_vol[i..i+lanes]);
            let v_gp_x = T::Simd::from_slice(&grad_p.vx[i..i+lanes]);
            let v_gp_y = T::Simd::from_slice(&grad_p.vy[i..i+lanes]);
            let v_rhs_x = T::Simd::from_slice(&rhs_ux[i..i+lanes]);
            let v_rhs_y = T::Simd::from_slice(&rhs_uy[i..i+lanes]);

            let res_x = v_rhs_x - (v_gp_x * v_inv_rho) * v_vol;
            let res_y = v_rhs_y - (v_gp_y * v_inv_rho) * v_vol;

            res_x.write_to_slice(&mut rhs_ux[i..i+lanes]);
            res_y.write_to_slice(&mut rhs_uy[i..i+lanes]);
            i += lanes;
        }
        
        while i < n_cells {
            let vol = self.cell_vol[i];
            rhs_ux[i] -= (grad_p.vx[i] * inv_density) * vol;
            rhs_uy[i] -= (grad_p.vy[i] * inv_density) * vol;
            i += 1;
        }

        // Solve Ux
        let mut x_sol = u_x.values.clone();
        let (_iter_ux, _res_ux, init_res_ux) =
            solve_bicgstab(&mat_u, &rhs_ux, &mut x_sol, 1000, T::val_from_f64(1e-6), ops);
        self.u.vx.copy_from_slice(&x_sol);

        // Solve Uy
        let mut y_sol = u_y.values.clone();
        let (_iter_uy, _res_uy, init_res_uy) =
            solve_bicgstab(&mat_v, &rhs_uy, &mut y_sol, 1000, T::val_from_f64(1e-6), ops);
        self.u.vy.copy_from_slice(&y_sol);

        // Exchange U ghosts (updated after predictor)
        let u_x_vals_pred: Vec<T> = self.u.vx.clone();
        let u_y_vals_pred: Vec<T> = self.u.vy.clone();
        let u_x_ghosts_pred = ops.exchange_halo(&u_x_vals_pred);
        let u_y_ghosts_pred = ops.exchange_halo(&u_y_vals_pred);

        // 2. Pressure Corrector

        // Extract diagonal coefficients A_P from momentum matrix.
        let mut a_p = vec![T::zero(); self.mesh.num_cells()];
        for i in 0..self.mesh.num_cells() {
            // Find diagonal element
            for k in mat_u.row_offsets[i]..mat_u.row_offsets[i + 1] {
                if mat_u.col_indices[k] == i {
                    a_p[i] = mat_u.values[k];
                    break;
                }
            }
        }

        // Compute d_p = Vol / (Ap * rho)
        let mut d_p = vec![T::zero(); self.mesh.num_cells()];

        let n_cells = self.mesh.num_cells();
        let mut i = 0;
        while i < n_cells {
            if a_p[i].abs() > T::val_from_f64(1e-20) {
                d_p[i] = T::val_from_f64(self.mesh.cell_vol[i]) / (a_p[i] * self.density);
            }
            i += 1;
        }

        // Exchange d_p ghosts
        let d_p_ghosts = ops.exchange_halo(&d_p);

        // PISO Loop (Corrector steps)
        let mut init_res_p = T::zero();
        for corrector_step in 0..2 {
            // Exchange p ghosts
            let p_ghosts_loop = ops.exchange_halo(&self.p.values);

            // Recompute grad_p for Rhie-Chow interpolation
            let grad_p = Fvm::compute_gradients(
                &self.mesh,
                &self.p,
                p_bc,
                Some(&p_ghosts_loop),
                Some(&self.ghost_map),
                Some(&self.ghost_centers),
            );

            // Exchange grad_p ghosts for Rhie-Chow
            let gp_x_vals: Vec<T> = grad_p.vx.clone();
            let gp_y_vals: Vec<T> = grad_p.vy.clone();
            let gp_x_ghosts = ops.exchange_halo(&gp_x_vals);
            let gp_y_ghosts = ops.exchange_halo(&gp_y_vals);

            // Assemble Pressure Poisson Equation
            let mut p_triplets = Vec::new();
            let mut p_rhs = vec![T::zero(); self.mesh.num_cells()];

            // Store flux_star for final update in this step
            let mut flux_star = vec![T::zero(); self.mesh.num_faces()];

            for i in 0..self.mesh.num_cells() {
                let mut div_u_star = T::zero();
                let c_i = Vector2::new(self.mesh.cell_cx[i], self.mesh.cell_cy[i]);
                
                let start = self.mesh.cell_face_offsets[i];
                let end = self.mesh.cell_face_offsets[i+1];
                
                for k in start..end {
                    let face_idx = self.mesh.cell_faces[k];
                    let owner = self.mesh.face_owner[face_idx];
                    let neighbor = self.mesh.face_neighbor[face_idx];
                    let is_owner = owner == i;
                    let neighbor_idx = if is_owner { neighbor } else { Some(owner) };
                    
                    let f_c = Vector2::new(self.mesh.face_cx[face_idx], self.mesh.face_cy[face_idx]);
                    let f_area = T::val_from_f64(self.mesh.face_area[face_idx]);
                    let f_normal = Vector2::new(T::val_from_f64(self.mesh.face_nx[face_idx]), T::val_from_f64(self.mesh.face_ny[face_idx]));
                    let normal = if is_owner { f_normal } else { -f_normal };
                    
                    // Interpolate U to face
                    let u_own = Vector2::new(self.u.vx[i], self.u.vy[i]);
                    let mut u_face = u_own; // Default
                    let mut d_p_face = d_p[i];
                    
                    let (u_face_dot_n, d_face_val) = if let Some(neigh) = neighbor_idx {
                        let u_neigh = Vector2::new(self.u.vx[neigh], self.u.vy[neigh]);
                        let d_own = (f_c - c_i).norm();
                        let d_neigh = (f_c - Vector2::new(self.mesh.cell_cx[neigh], self.mesh.cell_cy[neigh])).norm();
                        let f = T::val_from_f64(d_own / (d_own + d_neigh));
                        
                        u_face = u_own + (u_neigh - u_own) * f;
                        d_p_face = d_p[i] + f * (d_p[neigh] - d_p[i]);
                        
                        // Rhie-Chow Correction
                        let grad_p_own = Vector2::new(grad_p.vx[i], grad_p.vy[i]);
                        let grad_p_neigh = Vector2::new(grad_p.vx[neigh], grad_p.vy[neigh]);
                        let grad_p_avg = grad_p_own + (grad_p_neigh - grad_p_own) * f;
                        
                        let grad_p_n = grad_p_avg.dot(&normal);
                        
                        let dist = (Vector2::new(self.mesh.cell_cx[neigh], self.mesh.cell_cy[neigh]) - c_i).norm();
                        let p_grad_face = (self.p.values[neigh] - self.p.values[i]) / T::val_from_f64(dist);
                        
                        let rc_term = d_p_face * f_area * (grad_p_n - p_grad_face);
                        
                        (u_face.dot(&normal) * f_area + rc_term, d_p_face)
                    } else {
                        // Boundary
                        if let Some(bt) = self.mesh.face_boundary[face_idx] {
                            let mut u_b = u_own;
                            if let (Some(ub), Some(vb)) = (u_bc(bt), v_bc(bt)) {
                                u_b = Vector2::new(ub, vb);
                            }
                            
                            let mut flux = u_b.dot(&normal) * f_area;
                            let mut d_face = d_p[i];

                            // Check Parallel Interface
                            let mut handled = false;
                            let map = &self.ghost_map;
                            if let Some(&ghost_idx) = map.get(&face_idx) {
                                // Ghost cell
                                let local_ghost_idx = ghost_idx - self.mesh.num_cells();
                                if local_ghost_idx < u_x_ghosts_pred.len() {
                                        let u_g = Vector2::new(u_x_ghosts_pred[local_ghost_idx], u_y_ghosts_pred[local_ghost_idx]);
                                        let d_p_g = d_p_ghosts[local_ghost_idx];
                                        
                                        // Interpolate
                                        u_b = (u_own + u_g) * T::val_from_f64(0.5);
                                        flux = u_b.dot(&normal) * f_area;
                                        d_face = (d_p[i] + d_p_g) * T::val_from_f64(0.5);
                                        
                                        // Rhie-Chow
                                        if local_ghost_idx < gp_x_ghosts.len() {
                                            let gp_g = Vector2::new(gp_x_ghosts[local_ghost_idx], gp_y_ghosts[local_ghost_idx]);
                                            let grad_p_avg = (Vector2::new(grad_p.vx[i], grad_p.vy[i]) + gp_g) * T::val_from_f64(0.5);
                                            let grad_p_n = grad_p_avg.dot(&normal);
                                            
                                            let p_ghost = p_ghosts_loop[local_ghost_idx];
                                            let dist = if let Some(gc) = self.ghost_centers.get(local_ghost_idx) {
                                                let dx = gc.x - self.mesh.cell_cx[i];
                                                let dy = gc.y - self.mesh.cell_cy[i];
                                                (dx * dx + dy * dy).sqrt()
                                            } else { (f_c - c_i).norm() * 2.0 };
                                            
                                            let p_grad_face = (p_ghost - self.p.values[i]) / T::val_from_f64(dist);
                                            let rc_term = d_face * f_area * (grad_p_n - p_grad_face);
                                            flux += rc_term;
                                        }
                                        handled = true;
                                    }
                                }
                            
                            (flux, d_face)
                        } else {
                            (u_own.dot(&normal) * f_area, d_p[i])
                        }
                    };
                    
                    if is_owner {
                        flux_star[face_idx] = u_face_dot_n;
                    }
                    
                    div_u_star += u_face_dot_n;
                    
                    // Diffusion coeff for pressure: d_face * Area / dist
                    if let Some(neigh) = neighbor_idx {
                        let dist = (Vector2::new(self.mesh.cell_cx[neigh], self.mesh.cell_cy[neigh]) - c_i).norm();
                        let d_coeff = d_face_val * f_area / T::val_from_f64(dist);
                        
                        p_triplets.push((i, i, d_coeff));
                        p_triplets.push((i, neigh, -d_coeff));
                    } else {
                        // Boundary
                        if let Some(ghost_idx) = self.ghost_map.get(&face_idx) {
                            let local_ghost_idx = ghost_idx - self.mesh.num_cells();
                            let dist = if let Some(gc) = self.ghost_centers.get(local_ghost_idx) {
                                let dx = gc.x - self.mesh.cell_cx[i];
                                let dy = gc.y - self.mesh.cell_cy[i];
                                (dx * dx + dy * dy).sqrt()
                            } else { (f_c - c_i).norm() * 2.0 };
                            
                            let d_coeff = d_face_val * f_area / T::val_from_f64(dist);
                            p_triplets.push((i, i, d_coeff));
                            p_triplets.push((i, *ghost_idx, -d_coeff));
                        } else if let Some(bt) = self.mesh.face_boundary[face_idx] {
                            if let Some(_) = p_bc(bt) {
                                // Dirichlet
                                let dist = (f_c - c_i).norm();
                                let d_coeff = d_face_val * f_area / T::val_from_f64(dist);
                                p_triplets.push((i, i, d_coeff));
                            }
                        }
                    }
                }
                
                p_rhs[i] -= div_u_star;
            }
            
            // Solve Pressure
            let n_cols = if self.ghost_map.is_empty() { self.mesh.num_cells() } else { self.mesh.num_cells() + self.ghost_map.len() };
            let mat_p = SparseMatrix::from_triplets(self.mesh.num_cells(), n_cols, &p_triplets);
            
            if self.save_debug_data {
                self.last_pressure_matrix = Some(mat_p.clone());
                self.last_pressure_rhs = Some(p_rhs.clone());
            }
            
            let mut p_prime = vec![T::zero(); self.mesh.num_cells()];
            let (_iter_p, _res_p, init_res_p_step) = solve_bicgstab(&mat_p, &p_rhs, &mut p_prime, 1000, T::val_from_f64(1e-6), ops);
            
            if corrector_step == 0 { init_res_p = init_res_p_step; }
            
            // Correct Pressure: p = p + p_prime
            // Exchange p_prime ghosts
            let p_prime_ghosts = ops.exchange_halo(&p_prime);

            let alpha_p = T::one();
            let v_alpha_p = T::Simd::splat(alpha_p);
            let mut i = 0;
            let lanes = T::Simd::LANES;
            while i + lanes <= self.mesh.num_cells() {
                let v_p = T::Simd::from_slice(&self.p.values[i..i+lanes]);
                let v_pp = T::Simd::from_slice(&p_prime[i..i+lanes]);
                let res = v_p + v_alpha_p * v_pp;
                res.write_to_slice(&mut self.p.values[i..i+lanes]);
                i += lanes;
            }
            while i < self.mesh.num_cells() {
                self.p.values[i] += alpha_p * p_prime[i];
                i += 1;
            }
            
            // Correct Velocity: u = u - d_p * grad(p_prime)
            let p_prime_field = ScalarField { values: p_prime.clone() };
            let grad_p_prime = Fvm::compute_gradients(
                &self.mesh,
                &p_prime_field,
                |bt| if matches!(bt, BoundaryType::Outlet) { Some(T::zero()) } else { None },
                Some(&p_prime_ghosts),
                Some(&self.ghost_map),
                Some(&self.ghost_centers),
            );
            
            let mut i = 0;
            while i + lanes <= self.mesh.num_cells() {
                let v_dp = T::Simd::from_slice(&d_p[i..i+lanes]);
                let v_gpx = T::Simd::from_slice(&grad_p_prime.vx[i..i+lanes]);
                let v_gpy = T::Simd::from_slice(&grad_p_prime.vy[i..i+lanes]);
                let v_ux = T::Simd::from_slice(&self.u.vx[i..i+lanes]);
                let v_uy = T::Simd::from_slice(&self.u.vy[i..i+lanes]);
                
                let res_ux = v_ux - v_dp * v_gpx;
                let res_uy = v_uy - v_dp * v_gpy;
                
                res_ux.write_to_slice(&mut self.u.vx[i..i+lanes]);
                res_uy.write_to_slice(&mut self.u.vy[i..i+lanes]);
                i += lanes;
            }
            while i < self.mesh.num_cells() {
                self.u.vx[i] -= d_p[i] * grad_p_prime.vx[i];
                self.u.vy[i] -= d_p[i] * grad_p_prime.vy[i];
                i += 1;
            }
            
            // Correct Fluxes
            for i in 0..self.mesh.num_faces() {
                let owner = self.mesh.face_owner[i];
                let neighbor = self.mesh.face_neighbor[i];
                
                let mut d_face = d_p[owner];
                if let Some(n) = neighbor {
                    d_face = (d_p[owner] + d_p[n]) * T::val_from_f64(0.5);
                } else if let Some(ghost_idx) = self.ghost_map.get(&i) {
                    let local_ghost_idx = ghost_idx - self.mesh.num_cells();
                    if local_ghost_idx < d_p_ghosts.len() {
                        d_face = (d_p[owner] + d_p_ghosts[local_ghost_idx]) * T::val_from_f64(0.5);
                    }
                }
                
                let mut p_grad_n = T::zero();
                let c_own = Vector2::new(self.mesh.cell_cx[owner], self.mesh.cell_cy[owner]);
                let f_c = Vector2::new(self.mesh.face_cx[i], self.mesh.face_cy[i]);
                
                if let Some(n) = neighbor {
                    let c_neigh = Vector2::new(self.mesh.cell_cx[n], self.mesh.cell_cy[n]);
                    let dist = (c_neigh - c_own).norm();
                    p_grad_n = (p_prime[n] - p_prime[owner]) / T::val_from_f64(dist);
                } else {
                    if let Some(ghost_idx) = self.ghost_map.get(&i) {
                        let local_ghost_idx = ghost_idx - self.mesh.num_cells();
                        if local_ghost_idx < p_prime_ghosts.len() {
                            let p_ghost = p_prime_ghosts[local_ghost_idx];
                            let dist = if local_ghost_idx < self.ghost_centers.len() {
                                (self.ghost_centers[local_ghost_idx] - c_own).norm()
                            } else { (f_c - c_own).norm() * 2.0 };
                            p_grad_n = (p_ghost - p_prime[owner]) / T::val_from_f64(dist);
                        }
                    } else if let Some(bt) = self.mesh.face_boundary[i] {
                        if matches!(bt, BoundaryType::Outlet) {
                            let dist = (f_c - c_own).norm();
                            p_grad_n = (T::zero() - p_prime[owner]) / T::val_from_f64(dist);
                        }
                    }
                }
                
                let correction = d_face * T::val_from_f64(self.mesh.face_area[i]) * p_grad_n;
                self.fluxes[i] = flux_star[i] - correction;
            }
        }
        
        self.time += self.dt;
        self.residuals.push(("Ux".to_string(), init_res_ux));
        self.residuals.push(("Uy".to_string(), init_res_uy));
        self.residuals.push(("P".to_string(), init_res_p));
    }
    
    pub fn check_connectivity(&self) {
        let mut visited = vec![false; self.mesh.num_cells()];
        let mut stack = Vec::new();

        // Check face ownership
        for f_idx in 0..self.mesh.num_faces() {
            let owner = self.mesh.face_owner[f_idx];
            if owner >= self.mesh.num_cells() {
                println!("Face {} has invalid owner {}", f_idx, owner);
            }

            // Check if owner references face
            let start = self.mesh.cell_face_offsets[owner];
            let end = self.mesh.cell_face_offsets[owner + 1];
            let mut found = false;
            for k in start..end {
                if self.mesh.cell_faces[k] == f_idx {
                    found = true;
                    break;
                }
            }
            if !found {
                println!("Face {} owner {} does not reference face", f_idx, owner);
            }

            if let Some(neigh) = self.mesh.face_neighbor[f_idx] {
                if neigh >= self.mesh.num_cells() {
                    println!("Face {} has invalid neighbor {}", f_idx, neigh);
                }

                // Check if neighbor references face
                let start = self.mesh.cell_face_offsets[neigh];
                let end = self.mesh.cell_face_offsets[neigh + 1];
                let mut found = false;
                for k in start..end {
                    if self.mesh.cell_faces[k] == f_idx {
                        found = true;
                        break;
                    }
                }
                if !found {
                    println!("Face {} neighbor {} does not reference face", f_idx, neigh);
                }
            }
        }

        // Check cell faces
        for i in 0..self.mesh.num_cells() {
            let start = self.mesh.cell_face_offsets[i];
            let end = self.mesh.cell_face_offsets[i + 1];
            for k in start..end {
                let face_idx = self.mesh.cell_faces[k];
                if face_idx >= self.mesh.num_faces() {
                    println!("Cell {} references invalid face {}", i, face_idx);
                }
                let owner = self.mesh.face_owner[face_idx];
                let neighbor = self.mesh.face_neighbor[face_idx];
                if owner != i && neighbor != Some(i) {
                    println!(
                        "Cell {} references face {} but is neither owner nor neighbor",
                        i, face_idx
                    );
                }
            }
        }

        // BFS for connectivity
        stack.push(0);
        visited[0] = true;
        let mut count = 0;
        while let Some(i) = stack.pop() {
            count += 1;
            let start = self.mesh.cell_face_offsets[i];
            let end = self.mesh.cell_face_offsets[i + 1];
            for k in start..end {
                let face_idx = self.mesh.cell_faces[k];
                let owner = self.mesh.face_owner[face_idx];
                let neighbor = self.mesh.face_neighbor[face_idx];
                let n_idx = if owner == i { neighbor } else { Some(owner) };

                if let Some(n) = n_idx {
                    if !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }
        }

        if count != self.mesh.num_cells() {
            println!(
                "Mesh is not fully connected! Visited {}/{} cells",
                count,
                self.mesh.num_cells()
            );
            for i in 0..self.mesh.num_cells() {
                if !visited[i] {
                    println!(
                        "Cell {} is isolated. Center: ({}, {})",
                        i, self.mesh.cell_cx[i], self.mesh.cell_cy[i]
                    );
                }
            }
        } else {
            println!("Mesh connectivity check passed. {} cells.", count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::mesh::{
        generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle, RectangularChannel,
    };
    use nalgebra::{Point2, Vector2};

    #[test]
    fn test_channel_flow() {
        let domain_size = Vector2::new(2.0, 1.0);
        let geo = RectangularChannel {
            length: 2.0,
            height: 1.0,
        };
        let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, domain_size);

        let mut solver = PisoSolver::new(mesh);
        solver.dt = 0.001;

        // Run for some steps to let flow develop
        for i in 0..10 {
            solver.step();
            println!("Step {}: Res P = {}", i, solver.residuals[2].1);
        }

        // Check velocity at center (1.0, 0.5)
        let _center = Vector2::new(1.0, 0.5);
        let mut max_u = 0.0;
        let mut found_cells = 0;

        println!("Checking velocity in channel center...");
        for i in 0..solver.mesh.num_cells() {
            let cx = solver.mesh.cell_cx[i];
            let cy = solver.mesh.cell_cy[i];
            // Check cells in the middle of the channel
            if (cx - 1.0).abs() < 0.2 && (cy - 0.5).abs() < 0.2 {
                println!(
                    "Cell at ({}, {}) has velocity ({}, {})",
                    cx, cy, solver.u.vx[i], solver.u.vy[i]
                );
                if solver.u.vx[i] > max_u {
                    max_u = solver.u.vx[i];
                }
                found_cells += 1;
            }
        }

        println!("Max velocity in center region: {}", max_u);
        assert!(found_cells > 0, "No cells found in center region");
        assert!(
            max_u > 0.01,
            "Flow velocity is too low in the channel center. Max U: {}",
            max_u
        );
    }

    #[test]
    fn test_solver_convergence() {
        let domain_size = Vector2::new(2.0, 1.0);
        let geo = BackwardsStep {
            length: 2.0,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        };
        let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, domain_size); // Uniform mesh to avoid T-junctions

        let mut solver = PisoSolver::new(mesh);
        solver.dt = 0.001;

        // Run for some steps
        for i in 0..5 {
            solver.step();
            println!("Step {}: Res P = {}", i, solver.residuals[2].1);
        }

        // Check if residuals are small or decreasing
        // assert!(solver.residuals[2].1 < 1.0); // Example assertion
    }

    #[test]
    fn test_backwards_step_flow_physics() {
        let domain_size = Vector2::new(3.0, 1.0);
        let geo = BackwardsStep {
            length: 3.0,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        };
        let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, domain_size);

        let mut solver = PisoSolver::new(mesh);
        solver.dt = 0.005;

        println!("Running Backwards Step Test...");
        for _ in 0..20 {
            solver.step();
        }

        // 1. Check Mass Balance
        let mut inlet_flux: f64 = 0.0;
        let mut outlet_flux: f64 = 0.0;
        for i in 0..solver.mesh.num_faces() {
            if let Some(bt) = solver.mesh.face_boundary[i] {
                match bt {
                    BoundaryType::Inlet => inlet_flux += solver.fluxes[i],
                    BoundaryType::Outlet => outlet_flux += solver.fluxes[i],
                    _ => {}
                }
            }
        }
        println!(
            "Backwards Step Mass Balance: Inlet Flux = {}, Outlet Flux = {}, Sum = {}",
            inlet_flux,
            outlet_flux,
            inlet_flux + outlet_flux
        );

        // Inlet flux should be negative (entering), Outlet positive (leaving).
        // Sum should be close to 0.
        assert!(
            (inlet_flux + outlet_flux).abs() < 1e-3,
            "Mass is not conserved! Sum: {}",
            inlet_flux + outlet_flux
        );
        assert!(inlet_flux.abs() > 1e-5, "No flow at inlet!");

        // 2. Check Velocity in the middle (downstream of step)
        // Step ends at x=0.5. Let's check at x=1.5, y=0.5 (center of outlet channel)
        let probe_p = Point2::new(1.5, 0.5);
        let mut found = false;
        for i in 0..solver.mesh.num_cells() {
            let c = Point2::new(solver.mesh.cell_cx[i], solver.mesh.cell_cy[i]);
            if (c - probe_p).norm() < 0.1 {
                println!(
                    "Probe at {:?}: Velocity = ({}, {})",
                    c, solver.u.vx[i], solver.u.vy[i]
                );
                assert!(solver.u.vx[i] > 0.01, "Velocity too low downstream of step");
                found = true;
            }
        }
        assert!(found, "Probe point not found in mesh");
    }

    #[test]
    fn test_cylinder_flow_physics() {
        let domain_size = Vector2::new(3.0, 1.0);
        let geo = ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.5),
            obstacle_radius: 0.1,
        };
        let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, domain_size);

        let mut solver = PisoSolver::new(mesh);
        solver.dt = 0.005;

        println!("Running Cylinder Flow Test...");
        for _ in 0..20 {
            solver.step();
        }

        // 1. Check Mass Balance
        let mut inlet_flux: f64 = 0.0;
        let mut outlet_flux: f64 = 0.0;
        for i in 0..solver.mesh.num_faces() {
            if let Some(bt) = solver.mesh.face_boundary[i] {
                match bt {
                    BoundaryType::Inlet => inlet_flux += solver.fluxes[i],
                    BoundaryType::Outlet => outlet_flux += solver.fluxes[i],
                    _ => {}
                }
            }
        }
        println!(
            "Cylinder Mass Balance: Inlet Flux = {}, Outlet Flux = {}, Sum = {}",
            inlet_flux,
            outlet_flux,
            inlet_flux + outlet_flux
        );

        assert!(
            (inlet_flux + outlet_flux).abs() < 1e-3,
            "Mass is not conserved! Sum: {}",
            inlet_flux + outlet_flux
        );
        assert!(inlet_flux.abs() > 1e-5, "No flow at inlet!");

        // 2. Check Velocity downstream (wake region might be slow, but further downstream should be fast)
        // Cylinder at x=1.0. Let's check at x=2.0, y=0.5.
        let probe_p = Point2::new(2.0, 0.5);
        let mut found = false;
        for i in 0..solver.mesh.num_cells() {
            let c = Point2::new(solver.mesh.cell_cx[i], solver.mesh.cell_cy[i]);
            if (c - probe_p).norm() < 0.1 {
                println!(
                    "Probe at {:?}: Velocity = ({}, {})",
                    c, solver.u.vx[i], solver.u.vy[i]
                );
                // Velocity might be recovering from wake, should be positive.
                assert!(
                    solver.u.vx[i] > 0.01,
                    "Velocity too low downstream of cylinder"
                );
                found = true;
            }
        }
        assert!(found, "Probe point not found in mesh");
    }

    #[test]
    fn test_refinement_hanging_nodes() {
        let domain_size = Vector2::new(2.0, 1.0);
        // Use a channel with an obstacle to force refinement near the obstacle
        let geo = ChannelWithObstacle {
            length: 2.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.5),
            obstacle_radius: 0.1,
        };

        // Use different min/max to force quadtree refinement
        // This should create hanging nodes if not handled correctly
        let mesh = generate_cut_cell_mesh(&geo, 0.05, 0.2, domain_size);

        println!(
            "Mesh stats: {} cells, {} faces",
            mesh.num_cells(),
            mesh.num_faces()
        );

        let mut solver = PisoSolver::new(mesh);
        solver.dt = 0.005;

        println!("Running Refinement Test...");
        for _ in 0..20 {
            solver.step();
        }

        // Check flow downstream
        let probe_p = Point2::new(1.5, 0.5);
        let mut found = false;
        for i in 0..solver.mesh.num_cells() {
            let c = Point2::new(solver.mesh.cell_cx[i], solver.mesh.cell_cy[i]);
            if (c - probe_p).norm() < 0.1 {
                println!(
                    "Probe at {:?}: Velocity = ({}, {})",
                    c, solver.u.vx[i], solver.u.vy[i]
                );
                // If hanging nodes block flow, this will be near zero
                if solver.u.vx[i] > 0.01 {
                    found = true;
                }
            }
        }

        if !found {
            // Fail if no velocity found (or cell not found, but we assume cell exists)
            panic!("Flow is blocked! Velocity downstream is too low.");
        }
    }

    #[test]
    fn test_channel_flow_obstacle_fine() {
        let geo = ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.5),
            obstacle_radius: 0.2,
        };

        // Domain size 3.0 x 1.0
        let domain_size = Vector2::new(3.0, 1.0);

        // User specified 0.025 cell size
        let mut mesh = generate_cut_cell_mesh(&geo, 0.025, 0.025, domain_size);

        // Smooth the mesh to reduce skewness near the cylinder
        mesh.smooth(&geo, 0.3, 20);

        println!(
            "Mesh generated. Cells: {}, Faces: {}",
            mesh.num_cells(),
            mesh.num_faces()
        );

        let mut max_skew = 0.0;
        for i in 0..mesh.num_faces() {
            if let Some(neigh) = mesh.face_neighbor[i] {
                let owner = mesh.face_owner[i];
                let d = Vector2::new(mesh.cell_cx[neigh], mesh.cell_cy[neigh])
                    - Vector2::new(mesh.cell_cx[owner], mesh.cell_cy[owner]);
                let d_norm = d.normalize();
                let n = Vector2::new(mesh.face_nx[i], mesh.face_ny[i]);

                let skew = 1.0 - d_norm.dot(&n).abs();
                if skew > max_skew {
                    max_skew = skew;
                }
            }
        }
        println!("Max skewness after smoothing: {}", max_skew);
    }
}
