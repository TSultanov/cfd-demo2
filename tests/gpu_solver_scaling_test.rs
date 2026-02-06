#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::gpu::csr::build_block_csr;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Geometry, Mesh};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
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

fn build_mesh(cell_size: f64) -> Mesh {
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

fn solve_identity_system(solver: &mut UnifiedSolver, mesh: &Mesh) {
    let (row_offsets, col_indices) = build_csr(mesh);
    let diag = diag_indices(&row_offsets, &col_indices);

    let coupled_unknowns = solver.coupled_unknowns().expect("coupled unknowns");
    let unknowns_per_cell = coupled_unknowns / mesh.num_cells() as u32;
    assert!(unknowns_per_cell > 0);

    let (block_row_offsets, _block_col_indices) =
        build_block_csr(&row_offsets, &col_indices, unknowns_per_cell);
    let num_nonzeros = *block_row_offsets
        .last()
        .expect("block_row_offsets must be non-empty") as usize;
    let mut matrix = vec![0.0_f32; num_nonzeros];
    let block_size = unknowns_per_cell as usize;

    for row in 0..mesh.num_cells() as usize {
        let row_start = row_offsets[row] as usize;
        let diag_pos_in_row = diag[row]
            .checked_sub(row_start)
            .expect("diag indices should be within row slice");
        for block_row in 0..block_size {
            let dof_row = row * block_size + block_row;
            let dof_start = block_row_offsets[dof_row] as usize;
            let diag_entry = dof_start + diag_pos_in_row * block_size + block_row;
            matrix[diag_entry] = 1.0;
        }
    }

    let rhs: Vec<f32> = mesh
        .cell_cx
        .iter()
        .zip(mesh.cell_cy.iter())
        .flat_map(|(&x, &y)| {
            let base = 1.0 + 0.5 * x as f32 + 0.25 * y as f32;
            (0..unknowns_per_cell).map(move |k| base + 0.1 * k as f32)
        })
        .collect();

    solver
        .set_linear_system(&matrix, &rhs)
        .expect("set linear system");
    let stats = solver
        .solve_linear_system_with_size(coupled_unknowns, 200, 1e-6)
        .expect("cg solve");
    assert!(
        stats.converged,
        "CG did not converge (iters {}, residual {:.3e})",
        stats.iterations, stats.residual
    );

    let solution = pollster::block_on(solver.get_linear_solution()).expect("get solution");
    for (value, target) in solution.iter().zip(rhs.iter()) {
        let diff = (value - target).abs();
        assert!(diff < 1e-4, "identity solve diff {:.3e}", diff);
    }
}

fn assert_fgmres_sizing(solver: &mut UnifiedSolver) {
    let fgmres = solver.fgmres_sizing(10).expect("fgmres sizing");
    let expected_unknowns = solver.coupled_unknowns().expect("coupled unknowns");
    let expected_groups = (expected_unknowns + 63) / 64;
    assert_eq!(fgmres.num_unknowns, expected_unknowns);
    assert_eq!(fgmres.num_dot_groups, expected_groups);
}

#[test]
fn gpu_solvers_scale_with_mesh_size() {
    for cell_size in [0.1, 0.05] {
        let mesh = build_mesh(cell_size);
        let config = SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        };
        let mut solver = pollster::block_on(UnifiedSolver::new(
            &mesh,
            incompressible_momentum_model(),
            config,
            None,
            None,
        ))
        .expect("solver init");
        solve_identity_system(&mut solver, &mesh);
        assert_fgmres_sizing(&mut solver);
    }
}
