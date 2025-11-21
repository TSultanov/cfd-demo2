use crate::solver::mesh::{Mesh, BoundaryType};
use crate::solver::fvm::{Fvm, ScalarField, VectorField, Scheme};
use crate::solver::linear_solver::{solve_bicgstab, SparseMatrix};
use nalgebra::Vector2;

pub struct PisoSolver {
    pub mesh: Mesh,
    pub u: VectorField,
    pub p: ScalarField,
    pub fluxes: Vec<f64>, // Mass flux at faces
    pub dt: f64,
    pub time: f64,
    pub residuals: Vec<(String, f64)>,
    pub viscosity: f64,
    pub density: f64,
    pub scheme: Scheme,
}

impl PisoSolver {
    pub fn new(mesh: Mesh) -> Self {
        let n_cells = mesh.cells.len();
        let n_faces = mesh.faces.len();
        let solver = Self {
            mesh,
            u: VectorField::new(n_cells, Vector2::zeros()),
            p: ScalarField::new(n_cells, 0.0),
            fluxes: vec![0.0; n_faces],
            dt: 0.001, // Smaller time step
            time: 0.0,
            residuals: Vec::new(),
            viscosity: 0.01, // Re = 100 if L=1, U=1
            density: 1.0,
            scheme: Scheme::Upwind,
        };
        solver.check_connectivity();
        solver
    }

    pub fn step(&mut self) {
        println!("Starting step at time {:.4}", self.time);
        if self.fluxes.iter().any(|f| f.is_nan()) {
             println!("Fluxes contain NaN at start of step!");
        }

        // Boundary Conditions
        let u_bc = |bt: BoundaryType| match bt {
            BoundaryType::Inlet => Some(1.0),
            BoundaryType::Wall => Some(0.0),
            BoundaryType::Outlet => None, // Neumann
        };
        let v_bc = |bt: BoundaryType| match bt {
            BoundaryType::Inlet => Some(0.0),
            BoundaryType::Wall => Some(0.0),
            BoundaryType::Outlet => None, // Neumann
        };
        let p_bc = |bt: BoundaryType| match bt {
            BoundaryType::Outlet => Some(0.0),
            _ => None, // Neumann (Inlet, Wall)
        };

        let nu = self.viscosity / self.density;

        // 1. Momentum Predictor
        
        // Solve Ux
        let u_x = ScalarField { values: self.u.values.iter().map(|v| v.x).collect() };
        let (mat_u, rhs_u_base) = Fvm::assemble_scalar_transport(
            &self.mesh, 
            &u_x, 
            &self.fluxes, 
            nu, 
            self.dt, 
            &self.scheme,
            u_bc
        );
        
        // Add pressure gradient source term to RHS
        // -grad(p) / rho
        let grad_p = Fvm::compute_gradients(&self.mesh, &self.p, p_bc);
        let mut rhs_ux = rhs_u_base.clone();
        
        // Re-assemble for Uy because BC values might be different (though types are same usually)
        // Actually, assemble_scalar_transport uses BC values for Dirichlet.
        // If u_bc and v_bc are different (e.g. Inlet u=1, v=0), we need to re-assemble or just fix RHS.
        // Let's re-assemble to be safe and correct.
        let u_y = ScalarField { values: self.u.values.iter().map(|v| v.y).collect() };
        let (mat_v, mut rhs_uy) = Fvm::assemble_scalar_transport(
            &self.mesh, 
            &u_y, 
            &self.fluxes, 
            nu, 
            self.dt, 
            &self.scheme,
            v_bc
        );

        for i in 0..self.mesh.cells.len() {
            let vol = self.mesh.cells[i].volume;
            rhs_ux[i] -= (grad_p[i].x / self.density) * vol;
            rhs_uy[i] -= (grad_p[i].y / self.density) * vol;
        }
        
        // Solve Ux
        let mut x_sol = u_x.values.clone();
        let (_iter_ux, _res_ux, init_res_ux) = solve_bicgstab(&mat_u, &rhs_ux, &mut x_sol, 1000, 1e-6);
        for i in 0..self.mesh.cells.len() { self.u.values[i].x = x_sol[i]; }
        
        // Solve Uy
        let mut y_sol = u_y.values.clone();
        let (_iter_uy, _res_uy, init_res_uy) = solve_bicgstab(&mat_v, &rhs_uy, &mut y_sol, 1000, 1e-6);
        for i in 0..self.mesh.cells.len() { self.u.values[i].y = y_sol[i]; }
        
        // 2. Pressure Corrector
        
        // Extract diagonal coefficients A_P from momentum matrix.
        let mut a_p = vec![0.0; self.mesh.cells.len()];
        for i in 0..self.mesh.cells.len() {
            // Find diagonal element
            for k in mat_u.row_offsets[i]..mat_u.row_offsets[i+1] {
                if mat_u.col_indices[k] == i {
                    a_p[i] = mat_u.values[k];
                    break;
                }
            }
        }

        // Compute d_p = Vol / (Ap * rho)
        let mut d_p = vec![0.0; self.mesh.cells.len()];
        for i in 0..self.mesh.cells.len() {
            if a_p[i].abs() > 1e-20 {
                d_p[i] = self.mesh.cells[i].volume / (a_p[i] * self.density);
            }
        }
        
        // PISO Loop (Corrector steps)
        let mut init_res_p = 0.0;
        for corrector_step in 0..2 {
            // Recompute grad_p for Rhie-Chow interpolation
            let grad_p = Fvm::compute_gradients(&self.mesh, &self.p, p_bc);
            
            // Assemble Pressure Poisson Equation
            let mut p_triplets = Vec::new();
            let mut p_rhs = vec![0.0; self.mesh.cells.len()];
            
            // Store flux_star for final update in this step
            let mut flux_star = vec![0.0; self.mesh.faces.len()];

            for (i, cell) in self.mesh.cells.iter().enumerate() {
                let mut div_u_star = 0.0;
                
                for &face_idx in &cell.face_indices {
                    let face = &self.mesh.faces[face_idx];
                    let is_owner = face.owner == i;
                    let neighbor = if is_owner { face.neighbor } else { Some(face.owner) };
                    
                    let normal = if is_owner { face.normal } else { -face.normal };
                    
                    // Interpolate u_star to face
                    let u_own = self.u.values[i];
                    let (u_face_dot_n, d_face) = if let Some(n) = neighbor {
                        let u_neigh = self.u.values[n];
                        let u_avg = (u_own + u_neigh) * 0.5;
                        let d_face = (d_p[i] + d_p[n]) * 0.5;
                        
                        // Rhie-Chow Correction
                        let dist = (self.mesh.cells[n].center - cell.center).norm();
                        let grad_p_avg = (grad_p[i] + grad_p[n]) * 0.5;
                        
                        let p_down = self.p.values[n];
                        let p_up = self.p.values[i];
                        
                        let grad_p_compact = (p_down - p_up) / dist;
                        let grad_p_interp = grad_p_avg.dot(&normal);
                        
                        let rc_term = d_face * (grad_p_interp - grad_p_compact) * face.area;
                        
                        (u_avg.dot(&normal) * face.area + rc_term, d_face)
                    } else {
                        // Boundary
                        if let Some(bt) = face.boundary_type {
                            let u_b = Vector2::new(
                                u_bc(bt).unwrap_or(u_own.x),
                                v_bc(bt).unwrap_or(u_own.y)
                            );
                            (u_b.dot(&normal) * face.area, d_p[i])
                        } else {
                            (u_own.dot(&normal) * face.area, d_p[i])
                        }
                    };
                    
                    if is_owner {
                        flux_star[face_idx] = u_face_dot_n; 
                    }

                    div_u_star += u_face_dot_n;
                    
                    // Diffusion coeff for pressure: d_face * Area / dist
                    
                    if let Some(n) = neighbor {
                        let dist = (self.mesh.cells[n].center - cell.center).norm();
                        let d_coeff = d_face * face.area / dist;
                        
                        p_triplets.push((i, i, d_coeff));
                        p_triplets.push((i, n, -d_coeff));
                    } else {
                        // Boundary
                        if let Some(bt) = face.boundary_type {
                            if let Some(_) = p_bc(bt) {
                                // Dirichlet (Outlet)
                                let dist = (face.center - cell.center).norm();
                                let d_coeff = d_face * face.area / dist;
                                
                                p_triplets.push((i, i, d_coeff));
                                // p_prime_b = 0
                            }
                        }
                    }
                }
                
                p_rhs[i] -= div_u_star;
            }
            
            let mat_p = SparseMatrix::from_triplets(self.mesh.cells.len(), self.mesh.cells.len(), &p_triplets);
            let mut p_prime = vec![0.0; self.mesh.cells.len()];
            // Use BiCGStab for pressure as well, just in case
            let (_iter_p, _res_p, init_res_p_step) = solve_bicgstab(&mat_p, &p_rhs, &mut p_prime, 1000, 1e-10);
            
            if corrector_step == 0 {
                init_res_p = init_res_p_step;
            }
            
            // Correct Velocity and Pressure
            for i in 0..self.mesh.cells.len() {
                self.p.values[i] += 0.1 * p_prime[i]; // Under-relaxation for pressure
            }
            
            // Recompute grad(p_prime) to correct velocity
            let p_prime_field = ScalarField { values: p_prime.clone() };
            let p_prime_bc_func = |bt: BoundaryType| match bt {
                BoundaryType::Outlet => Some(0.0),
                _ => None,
            };
            let grad_p_prime = Fvm::compute_gradients(&self.mesh, &p_prime_field, p_prime_bc_func);
            
            for i in 0..self.mesh.cells.len() {
                self.u.values[i] -= d_p[i] * grad_p_prime[i];
            }
            
            // Update fluxes
            for (face_idx, face) in self.mesh.faces.iter().enumerate() {
                let owner = face.owner;
                let neighbor = face.neighbor;
                
                if let Some(neigh) = neighbor {
                    let d_face = (d_p[owner] + d_p[neigh]) * 0.5;
                    let dist = (self.mesh.cells[neigh].center - self.mesh.cells[owner].center).norm();
                    
                    let p_prime_own = p_prime[owner];
                    let p_prime_neigh = p_prime[neigh];
                    
                    let correction = d_face * (p_prime_neigh - p_prime_own) / dist * face.area;
                    
                    self.fluxes[face_idx] = flux_star[face_idx] - correction;
                } else {
                    // Boundary
                    if let Some(bt) = face.boundary_type {
                        if let Some(_) = p_bc(bt) {
                            // Outlet: p_prime_b = 0
                            let dist = (face.center - self.mesh.cells[owner].center).norm();
                            let p_prime_own = p_prime[owner];
                            let correction = d_p[owner] * (0.0 - p_prime_own) / dist * face.area;
                            self.fluxes[face_idx] = flux_star[face_idx] - correction;
                        } else {
                            self.fluxes[face_idx] = flux_star[face_idx];
                        }
                    }
                }
            }
        }
        
        self.time += self.dt;
        self.residuals = vec![
            ("Ux".to_string(), init_res_ux),
            ("Uy".to_string(), init_res_uy),
            ("P".to_string(), init_res_p),
        ];
        println!("Time: {:.4}, Res: Ux={:.6}, Uy={:.6}, P={:.6}", self.time, init_res_ux, init_res_uy, init_res_p);
    }

    pub fn check_connectivity(&self) {
        let mut visited = vec![false; self.mesh.cells.len()];
        let mut queue = std::collections::VecDeque::new();
        
        // Check face consistency
        for (f_idx, face) in self.mesh.faces.iter().enumerate() {
            let owner = face.owner;
            if !self.mesh.cells[owner].face_indices.contains(&f_idx) {
                println!("ERROR: Face {} has owner {}, but cell {} does not contain face!", f_idx, owner, owner);
            }
            
            if let Some(neigh) = face.neighbor {
                if !self.mesh.cells[neigh].face_indices.contains(&f_idx) {
                    println!("ERROR: Face {} has neighbor {}, but cell {} does not contain face!", f_idx, neigh, neigh);
                }
            }
        }
        
        // Start from Outlet cells
        for (i, cell) in self.mesh.cells.iter().enumerate() {
            for &face_idx in &cell.face_indices {
                let face = &self.mesh.faces[face_idx];
                if let Some(bt) = face.boundary_type {
                    if matches!(bt, BoundaryType::Outlet) {
                        visited[i] = true;
                        queue.push_back(i);
                        break;
                    }
                }
            }
        }
        
        // println!("Found {} outlet cells.", queue.len());
        
        while let Some(i) = queue.pop_front() {
            let cell = &self.mesh.cells[i];
            for &face_idx in &cell.face_indices {
                let face = &self.mesh.faces[face_idx];
                let neighbor = if face.owner == i { face.neighbor } else { Some(face.owner) };
                
                if let Some(n) = neighbor {
                    if !visited[n] {
                        visited[n] = true;
                        queue.push_back(n);
                    }
                }
            }
        }
        
        let unvisited_count = visited.iter().filter(|&&v| !v).count();
        if unvisited_count > 0 {
            println!("WARNING: {} cells are NOT connected to Outlet!", unvisited_count);
            // Print first few
            for (i, &v) in visited.iter().enumerate() {
                if !v {
                    println!("Cell {} is isolated. Center: {:?}", i, self.mesh.cells[i].center);
                    if unvisited_count > 10 && i > 10 { break; }
                }
            }
        } else {
            // println!("All cells are connected to Outlet.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, RectangularChannel, ChannelWithObstacle};
    use nalgebra::{Vector2, Point2};

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
        for (i, cell) in solver.mesh.cells.iter().enumerate() {
            // Check cells in the middle of the channel
            if (cell.center.x - 1.0).abs() < 0.2 && (cell.center.y - 0.5).abs() < 0.2 {
                println!("Cell at {:?} has velocity {:?}", cell.center, solver.u.values[i]);
                if solver.u.values[i].x > max_u {
                    max_u = solver.u.values[i].x;
                }
                found_cells += 1;
            }
        }
        
        println!("Max velocity in center region: {}", max_u);
        assert!(found_cells > 0, "No cells found in center region");
        assert!(max_u > 0.01, "Flow velocity is too low in the channel center. Max U: {}", max_u);
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
        let mut inlet_flux = 0.0;
        let mut outlet_flux = 0.0;
        for (i, face) in solver.mesh.faces.iter().enumerate() {
            if let Some(bt) = face.boundary_type {
                match bt {
                    BoundaryType::Inlet => inlet_flux += solver.fluxes[i],
                    BoundaryType::Outlet => outlet_flux += solver.fluxes[i],
                    _ => {}
                }
            }
        }
        println!("Backwards Step Mass Balance: Inlet Flux = {}, Outlet Flux = {}, Sum = {}", inlet_flux, outlet_flux, inlet_flux + outlet_flux);
        
        // Inlet flux should be negative (entering), Outlet positive (leaving).
        // Sum should be close to 0.
        assert!((inlet_flux + outlet_flux).abs() < 1e-4, "Mass is not conserved! Sum: {}", inlet_flux + outlet_flux);
        assert!(inlet_flux.abs() > 1e-5, "No flow at inlet!");

        // 2. Check Velocity in the middle (downstream of step)
        // Step ends at x=0.5. Let's check at x=1.5, y=0.5 (center of outlet channel)
        let probe_p = Point2::new(1.5, 0.5);
        let mut found = false;
        for (i, cell) in solver.mesh.cells.iter().enumerate() {
            if (cell.center - probe_p).norm() < 0.1 {
                println!("Probe at {:?}: Velocity = {:?}", cell.center, solver.u.values[i]);
                assert!(solver.u.values[i].x > 0.01, "Velocity too low downstream of step");
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
        let mut inlet_flux = 0.0;
        let mut outlet_flux = 0.0;
        for (i, face) in solver.mesh.faces.iter().enumerate() {
            if let Some(bt) = face.boundary_type {
                match bt {
                    BoundaryType::Inlet => inlet_flux += solver.fluxes[i],
                    BoundaryType::Outlet => outlet_flux += solver.fluxes[i],
                    _ => {}
                }
            }
        }
        println!("Cylinder Mass Balance: Inlet Flux = {}, Outlet Flux = {}, Sum = {}", inlet_flux, outlet_flux, inlet_flux + outlet_flux);
        
        assert!((inlet_flux + outlet_flux).abs() < 1e-4, "Mass is not conserved! Sum: {}", inlet_flux + outlet_flux);
        assert!(inlet_flux.abs() > 1e-5, "No flow at inlet!");

        // 2. Check Velocity downstream (wake region might be slow, but further downstream should be fast)
        // Cylinder at x=1.0. Let's check at x=2.0, y=0.5.
        let probe_p = Point2::new(2.0, 0.5);
        let mut found = false;
        for (i, cell) in solver.mesh.cells.iter().enumerate() {
            if (cell.center - probe_p).norm() < 0.1 {
                println!("Probe at {:?}: Velocity = {:?}", cell.center, solver.u.values[i]);
                // Velocity might be recovering from wake, should be positive.
                assert!(solver.u.values[i].x > 0.01, "Velocity too low downstream of cylinder");
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
        
        println!("Mesh stats: {} cells, {} faces", mesh.cells.len(), mesh.faces.len());
        
        let mut solver = PisoSolver::new(mesh);
        solver.dt = 0.005;
        
        println!("Running Refinement Test...");
        for _ in 0..20 {
            solver.step();
        }

        // Check flow downstream
        let probe_p = Point2::new(1.5, 0.5);
        let mut found = false;
        for (i, cell) in solver.mesh.cells.iter().enumerate() {
            if (cell.center - probe_p).norm() < 0.1 {
                println!("Probe at {:?}: Velocity = {:?}", cell.center, solver.u.values[i]);
                // If hanging nodes block flow, this will be near zero
                if solver.u.values[i].x > 0.01 {
                    found = true;
                }
            }
        }
        
        if !found {
            // Fail if no velocity found (or cell not found, but we assume cell exists)
             panic!("Flow is blocked! Velocity downstream is too low.");
        }
    }
}
