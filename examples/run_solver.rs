use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::piso::PisoSolver;
use nalgebra::Vector2;

fn main() {
    let domain_size = Vector2::new(2.0, 1.0);
    let geo = BackwardsStep {
        length: 2.0,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let mesh = generate_cut_cell_mesh(&geo, 0.05, 0.1, domain_size);
    println!("Mesh generated with {} cells and {} faces", mesh.cells.len(), mesh.faces.len());

    let mut solver = PisoSolver::new(mesh);
    solver.dt = 0.001;

    // Initialize field
    for (i, cell) in solver.mesh.cells.iter().enumerate() {
        if cell.center.x < 0.1 && cell.center.y > 0.5 {
             solver.u.values[i] = Vector2::new(1.0, 0.0);
        }
    }

    for i in 0..100 {
        solver.step();
        if i % 10 == 0 {
            // println!("Step {}", i);
        }
    }
}
