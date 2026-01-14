use cfd2::solver::mesh::{generate_cut_cell_mesh, BoundaryType, Geometry, Mesh};
use cfd2::solver::model::backend::ast::{fvm, vol_scalar, Coefficient, EquationSystem, TermOp};
use cfd2::solver::model::generic_diffusion_demo_model;
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::units::{si, UnitDim};
use cfd2::solver::{SolverConfig, UnifiedSolver};
use nalgebra::{Point2, Vector2};

struct RectGeometry {
    length: f64,
    height: f64,
}

impl Geometry for RectGeometry {
    fn is_inside(&self, p: &Point2<f64>) -> bool {
        self.sdf(p) < 0.0
    }

    fn sdf(&self, p: &Point2<f64>) -> f64 {
        let dx = (p.x - self.length / 2.0).abs() - self.length / 2.0;
        let dy = (p.y - self.height / 2.0).abs() - self.height / 2.0;
        dx.max(dy).min(0.0) + Vector2::new(dx.max(0.0), dy.max(0.0)).norm()
    }

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();
        let nx = (self.length / spacing).ceil() as usize;
        let ny = (self.height / spacing).ceil() as usize;

        for i in 0..=nx {
            let x = (i as f64 * spacing).min(self.length);
            points.push(Point2::new(x, 0.0));
            points.push(Point2::new(x, self.height));
        }
        for i in 0..=ny {
            let y = (i as f64 * spacing).min(self.height);
            points.push(Point2::new(0.0, y));
            points.push(Point2::new(self.length, y));
        }

        points
    }
}

fn build_rect_mesh(cell_size: f64) -> Mesh {
    let geo = RectGeometry {
        length: 1.0,
        height: 0.2,
    };
    generate_cut_cell_mesh(
        &geo,
        cell_size,
        cell_size,
        1.0,
        Vector2::new(geo.length, geo.height),
    )
}

fn build_laplace_system() -> EquationSystem {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let eqn = fvm::laplacian(Coefficient::constant(1.0), phi).eqn(phi);
    let mut system = EquationSystem::new();
    system.add_equation(eqn);
    system
}

fn build_poisson_system() -> EquationSystem {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let source = vol_scalar("source", UnitDim::new(0, -2, 0));
    let eqn = fvm::laplacian(Coefficient::constant(1.0), phi) + fvm::source(source);
    let mut system = EquationSystem::new();
    system.add_equation(eqn.eqn(phi));
    system
}

fn build_heat_system(alpha: f64) -> EquationSystem {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let eqn = fvm::ddt(phi)
        + fvm::laplacian(
            Coefficient::constant_unit(alpha, UnitDim::new(0, 2, -1)),
            phi,
        );
    let mut system = EquationSystem::new();
    system.add_equation(eqn.eqn(phi));
    system
}

fn build_csr(mesh: &Mesh) -> (Vec<u32>, Vec<u32>) {
    let num_cells = mesh.num_cells();
    let mut row_offsets = vec![0u32; num_cells + 1];
    let mut col_indices = Vec::new();

    let mut adj = vec![Vec::new(); num_cells];
    for (face_idx, &owner) in mesh.face_owner.iter().enumerate() {
        if let Some(neighbor) = mesh.face_neighbor[face_idx] {
            adj[owner].push(neighbor);
            adj[neighbor].push(owner);
        }
    }

    for (i, list) in adj.iter_mut().enumerate() {
        list.push(i);
        list.sort();
        list.dedup();
    }

    let mut offset = 0u32;
    for (row, list) in adj.iter().enumerate() {
        row_offsets[row] = offset;
        for &col in list {
            col_indices.push(col as u32);
        }
        offset += list.len() as u32;
    }
    row_offsets[num_cells] = offset;

    (row_offsets, col_indices)
}

fn diag_indices(row_offsets: &[u32], col_indices: &[u32]) -> Vec<usize> {
    let mut diag = Vec::with_capacity(row_offsets.len() - 1);
    for row in 0..row_offsets.len() - 1 {
        let start = row_offsets[row] as usize;
        let end = row_offsets[row + 1] as usize;
        let slice = &col_indices[start..end];
        let pos = slice
            .binary_search(&(row as u32))
            .expect("missing diagonal entry");
        diag.push(start + pos);
    }
    diag
}

fn row_entry_index(row_offsets: &[u32], col_indices: &[u32], row: usize, col: usize) -> usize {
    let start = row_offsets[row] as usize;
    let end = row_offsets[row + 1] as usize;
    let slice = &col_indices[start..end];
    let pos = slice
        .binary_search(&(col as u32))
        .unwrap_or_else(|_| panic!("missing entry for row {}, col {}", row, col));
    start + pos
}

fn normal_distance(dx: f64, dy: f64, nx: f64, ny: f64) -> f64 {
    let along = (dx * nx + dy * ny).abs();
    if along > 1e-12 {
        along
    } else {
        (dx * dx + dy * dy).sqrt().max(1e-12)
    }
}

fn assemble_scalar_system<F>(
    mesh: &Mesh,
    system: &EquationSystem,
    dt: f64,
    phi_old: Option<&[f64]>,
    source: Option<&[f64]>,
    boundary_value: F,
) -> (Vec<f32>, Vec<f32>)
where
    F: Fn(BoundaryType, f64, f64) -> Option<f64>,
{
        fn eval_coeff_scalar(coeff: &Coefficient) -> f64 {
            match coeff {
                Coefficient::Constant { value, .. } => *value,
                Coefficient::Field(field) => {
                    panic!("coefficient field not supported: {}", field.name())
                }
                Coefficient::MagSqr(field) => {
                    panic!("coefficient mag_sqr not supported: {}", field.name())
                }
                Coefficient::Product(lhs, rhs) => eval_coeff_scalar(lhs) * eval_coeff_scalar(rhs),
            }
        }

    let num_cells = mesh.num_cells();
    let (row_offsets, col_indices) = build_csr(mesh);
    let diag = diag_indices(&row_offsets, &col_indices);
    let mut matrix = vec![0.0_f64; col_indices.len()];
    let mut rhs = vec![0.0_f64; num_cells];

    let eqn = system
        .equations()
        .first()
        .expect("expected a single equation");
    assert_eq!(eqn.target().name(), "phi");

    for term in eqn.terms() {
        match term.op {
            TermOp::Ddt => {
                let phi_old = phi_old.expect("ddt term requires phi_old");
                let coeff = match term.coeff {
                    Some(ref coeff) => eval_coeff_scalar(coeff),
                    None => 1.0,
                };
                for cell in 0..num_cells {
                    let vol = mesh.cell_vol[cell];
                    let diag_value = coeff * vol / dt;
                    matrix[diag[cell]] += diag_value;
                    rhs[cell] += diag_value * phi_old[cell];
                }
            }
            TermOp::Laplacian => {
                let coeff = match term.coeff {
                    Some(ref coeff) => eval_coeff_scalar(coeff),
                    None => 1.0,
                };
                for face in 0..mesh.num_faces() {
                    let owner = mesh.face_owner[face];
                    let neighbor = mesh.face_neighbor[face];
                    let area = mesh.face_area[face];
                    let nx = mesh.face_nx[face];
                    let ny = mesh.face_ny[face];

                    if let Some(neighbor) = neighbor {
                        let dx = mesh.cell_cx[neighbor] - mesh.cell_cx[owner];
                        let dy = mesh.cell_cy[neighbor] - mesh.cell_cy[owner];
                        let dist = normal_distance(dx, dy, nx, ny);
                        let flux_coeff = coeff * area / dist;

                        let owner_diag = diag[owner];
                        let neighbor_diag = diag[neighbor];
                        let owner_off =
                            row_entry_index(&row_offsets, &col_indices, owner, neighbor);
                        let neighbor_off =
                            row_entry_index(&row_offsets, &col_indices, neighbor, owner);

                        matrix[owner_diag] += flux_coeff;
                        matrix[owner_off] -= flux_coeff;
                        matrix[neighbor_diag] += flux_coeff;
                        matrix[neighbor_off] -= flux_coeff;
                    } else if let Some(boundary) = mesh.face_boundary[face] {
                        if let Some(phi_b) =
                            boundary_value(boundary, mesh.face_cx[face], mesh.face_cy[face])
                        {
                            let dx = mesh.face_cx[face] - mesh.cell_cx[owner];
                            let dy = mesh.face_cy[face] - mesh.cell_cy[owner];
                            let dist = normal_distance(dx, dy, nx, ny);
                            let flux_coeff = coeff * area / dist;
                            let owner_diag = diag[owner];
                            matrix[owner_diag] += flux_coeff;
                            rhs[owner] += flux_coeff * phi_b;
                        }
                    }
                }
            }
            TermOp::Source => {
                let source = source.expect("source term requires source values");
                for cell in 0..num_cells {
                    rhs[cell] += source[cell] * mesh.cell_vol[cell];
                }
            }
            _ => panic!("unsupported term {:?}", term.op),
        }
    }

    let matrix_values: Vec<f32> = matrix.iter().map(|v| *v as f32).collect();
    let rhs_values: Vec<f32> = rhs.iter().map(|v| *v as f32).collect();
    (matrix_values, rhs_values)
}

fn solve_system(mesh: &Mesh, matrix: &[f32], rhs: &[f32]) -> (Vec<f64>, f32) {
    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
    };
    let mut solver = pollster::block_on(UnifiedSolver::new(
        mesh,
        generic_diffusion_demo_model(),
        config,
        None,
        None,
    ))
    .expect("solver init");
    solver
        .set_linear_system(matrix, rhs)
        .expect("set linear system");
    let stats = solver
        .solve_linear_system_with_size(mesh.num_cells() as u32, 400, 1e-6)
        .expect("cg solve");
    assert!(
        stats.converged,
        "CG did not converge (iters {}, residual {:.3e})",
        stats.iterations, stats.residual
    );
    let solution = pollster::block_on(solver.get_linear_solution()).expect("solution read");
    let solution = solution.iter().map(|v| *v as f64).collect();
    (solution, stats.residual)
}

fn rms_error(solution: &[f64], expected: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (value, target) in solution.iter().zip(expected.iter()) {
        let diff = value - target;
        sum += diff * diff;
    }
    (sum / solution.len() as f64).sqrt()
}

fn max_error(solution: &[f64], expected: &[f64]) -> f64 {
    solution
        .iter()
        .zip(expected.iter())
        .map(|(value, target)| (value - target).abs())
        .fold(0.0, f64::max)
}

#[test]
fn laplace_linear_solution_matches() {
    let mesh = build_rect_mesh(0.05);
    let system = build_laplace_system();
    let a = 1.2;
    let b = 0.3;

    let boundary_value = |boundary: BoundaryType, x: f64, _y: f64| match boundary {
        BoundaryType::Inlet | BoundaryType::Outlet => Some(a + b * x),
        BoundaryType::Wall | BoundaryType::SlipWall => None,
    };

    let (matrix, rhs) = assemble_scalar_system(&mesh, &system, 1.0, None, None, boundary_value);
    let (solution, _) = solve_system(&mesh, &matrix, &rhs);
    let expected: Vec<f64> = mesh.cell_cx.iter().map(|&x| a + b * x).collect();

    let rms = rms_error(&solution, &expected);
    let max = max_error(&solution, &expected);
    assert!(rms < 5e-3, "rms error {:.3e}", rms);
    assert!(max < 2e-2, "max error {:.3e}", max);
}

#[test]
fn poisson_quadratic_solution_matches() {
    let mesh = build_rect_mesh(0.05);
    let system = build_poisson_system();

    let boundary_value = |boundary: BoundaryType, x: f64, _y: f64| match boundary {
        BoundaryType::Inlet | BoundaryType::Outlet => Some(x * (1.0 - x)),
        BoundaryType::Wall | BoundaryType::SlipWall => None,
    };

    let source: Vec<f64> = mesh.cell_cx.iter().map(|_| 2.0).collect();
    let (matrix, rhs) =
        assemble_scalar_system(&mesh, &system, 1.0, None, Some(&source), boundary_value);
    let (solution, _) = solve_system(&mesh, &matrix, &rhs);
    let expected: Vec<f64> = mesh.cell_cx.iter().map(|&x| x * (1.0 - x)).collect();

    let rms = rms_error(&solution, &expected);
    let max = max_error(&solution, &expected);
    assert!(rms < 2e-2, "rms error {:.3e}", rms);
    assert!(max < 6e-2, "max error {:.3e}", max);
}

#[test]
fn heat_equation_single_step_matches() {
    let mesh = build_rect_mesh(0.05);
    let alpha = 0.1;
    let dt = 1e-2;
    let system = build_heat_system(alpha);

    let boundary_value = |boundary: BoundaryType, x: f64, _y: f64| match boundary {
        BoundaryType::Inlet | BoundaryType::Outlet => Some((std::f64::consts::PI * x).sin()),
        BoundaryType::Wall | BoundaryType::SlipWall => None,
    };

    let phi_old: Vec<f64> = mesh
        .cell_cx
        .iter()
        .map(|&x| (std::f64::consts::PI * x).sin())
        .collect();
    let decay = (-alpha * std::f64::consts::PI * std::f64::consts::PI * dt).exp();
    let expected: Vec<f64> = phi_old.iter().map(|v| v * decay).collect();

    let (matrix, rhs) =
        assemble_scalar_system(&mesh, &system, dt, Some(&phi_old), None, boundary_value);
    let (solution, _) = solve_system(&mesh, &matrix, &rhs);

    let rms = rms_error(&solution, &expected);
    let max = max_error(&solution, &expected);
    assert!(rms < 2e-2, "rms error {:.3e}", rms);
    assert!(max < 6e-2, "max error {:.3e}", max);
}
