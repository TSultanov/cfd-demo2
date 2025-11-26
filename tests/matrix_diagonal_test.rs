// Test to diagnose duplicate diagonal entries in sparse matrix construction

use cfd2::solver::fvm::{Fvm, ScalarField, Scheme};
use cfd2::solver::linear_solver::SparseMatrix;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BoundaryType, RectangularChannel};
use nalgebra::Vector2;

#[test]
fn test_matrix_diagonals() {
    // Create a simple mesh using the cut-cell method
    let geo = RectangularChannel {
        length: 1.0,
        height: 1.0,
    };
    let mesh = generate_cut_cell_mesh(&geo, 0.25, 0.5, Vector2::new(1.0, 1.0));

    println!(
        "Mesh has {} cells, {} faces",
        mesh.num_cells(),
        mesh.num_faces()
    );

    // Create a simple scalar field
    let phi = ScalarField::new(mesh.num_cells(), 1.0f64);

    // Create uniform fluxes (zero for simplicity)
    let fluxes = vec![0.0f64; mesh.num_faces()];

    let gamma = 0.01f64; // diffusion coefficient
    let dt = 0.01f64;

    // Boundary condition
    let bc = |bt: BoundaryType| -> Option<f64> {
        match bt {
            BoundaryType::Inlet => Some(1.0),
            BoundaryType::Wall => Some(0.0),
            BoundaryType::Outlet => None,
            BoundaryType::ParallelInterface(_, _) => None,
        }
    };

    let (mat, rhs) = Fvm::assemble_scalar_transport(
        &mesh,
        &phi,
        &fluxes,
        gamma,
        dt,
        &Scheme::Upwind,
        bc,
        None,
        None,
        None,
    );

    println!("\nMatrix info:");
    println!("  n_rows: {}, n_cols: {}", mat.n_rows, mat.n_cols);
    println!("  row_offsets: {:?}", mat.row_offsets);

    // Check for duplicate column indices within each row
    for row in 0..mat.n_rows {
        let start = mat.row_offsets[row];
        let end = mat.row_offsets[row + 1];

        println!("\nRow {}: {} entries", row, end - start);

        // Collect all column indices and values for this row
        let mut entries: Vec<(usize, f64)> = Vec::new();
        for k in start..end {
            entries.push((mat.col_indices[k], mat.values[k]));
        }

        // Print all entries
        for (col, val) in &entries {
            println!("  ({}, {}) = {:.6}", row, col, val);
        }

        // Check for duplicates
        let mut col_indices_in_row: Vec<usize> = entries.iter().map(|(c, _)| *c).collect();
        col_indices_in_row.sort();

        let unique_count = col_indices_in_row
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        if unique_count != col_indices_in_row.len() {
            println!("  WARNING: Row {} has duplicate column indices!", row);

            // Find duplicates
            let mut counts = std::collections::HashMap::new();
            for &col in &col_indices_in_row {
                *counts.entry(col).or_insert(0) += 1;
            }
            for (col, count) in counts {
                if count > 1 {
                    println!("    Column {} appears {} times", col, count);
                    // Print all values for this column
                    let vals: Vec<f64> = entries
                        .iter()
                        .filter(|(c, _)| *c == col)
                        .map(|(_, v)| *v)
                        .collect();
                    println!("    Values: {:?}", vals);
                    println!("    Sum: {:.6}", vals.iter().sum::<f64>());
                }
            }
        }
    }

    println!("\nRHS: {:?}", rhs);
}

#[test]
fn test_diagonal_sum_vs_expected() {
    // For a simple case, compute expected diagonal values manually
    let geo = RectangularChannel {
        length: 1.0,
        height: 1.0,
    };
    let mesh = generate_cut_cell_mesh(&geo, 0.4, 0.5, Vector2::new(1.0, 1.0));

    println!("Testing simple mesh");
    println!(
        "Mesh has {} cells, {} faces",
        mesh.num_cells(),
        mesh.num_faces()
    );

    // Print cell info
    for i in 0..mesh.num_cells() {
        println!(
            "\nCell {}: center ({:.3}, {:.3}), vol = {:.6}",
            i, mesh.cell_cx[i], mesh.cell_cy[i], mesh.cell_vol[i]
        );
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        println!("  Faces: {:?}", &mesh.cell_faces[start..end]);
    }

    // Print face info
    println!("\nFaces:");
    for f in 0..mesh.num_faces() {
        println!(
            "Face {}: owner={}, neighbor={:?}, boundary={:?}, area={:.3}",
            f, mesh.face_owner[f], mesh.face_neighbor[f], mesh.face_boundary[f], mesh.face_area[f]
        );
    }

    // Create scalar field and assemble
    let phi = ScalarField::new(mesh.num_cells(), 1.0f64);
    let fluxes = vec![0.0f64; mesh.num_faces()];

    let gamma = 0.01f64;
    let dt = 0.01f64;

    let bc = |bt: BoundaryType| -> Option<f64> {
        match bt {
            BoundaryType::Inlet => Some(1.0),
            BoundaryType::Wall => Some(0.0),
            BoundaryType::Outlet => None,
            BoundaryType::ParallelInterface(_, _) => None,
        }
    };

    let (mat, _) = Fvm::assemble_scalar_transport(
        &mesh,
        &phi,
        &fluxes,
        gamma,
        dt,
        &Scheme::Upwind,
        bc,
        None,
        None,
        None,
    );

    // Extract diagonal values
    println!("\nDiagonal analysis:");
    for row in 0..mat.n_rows {
        let start = mat.row_offsets[row];
        let end = mat.row_offsets[row + 1];

        let mut diag_sum = 0.0;
        let mut diag_count = 0;
        for k in start..end {
            if mat.col_indices[k] == row {
                diag_count += 1;
                diag_sum += mat.values[k];
                println!(
                    "  Row {}: diagonal entry #{} = {:.6}",
                    row, diag_count, mat.values[k]
                );
            }
        }

        // Expected diagonal: V/dt + sum of diffusion coefficients to neighbors
        let vol = mesh.cell_vol[row];
        let expected_unsteady = vol / dt;

        println!(
            "  Row {}: expected V/dt = {:.6}, actual diag_sum = {:.6}, count = {}",
            row, expected_unsteady, diag_sum, diag_count
        );

        if diag_count > 1 {
            println!("  ERROR: Row {} has {} diagonal entries!", row, diag_count);
        }
    }
}

#[test]
fn test_triplet_duplicates() {
    // Test from_triplets with known duplicates to verify behavior
    // After the fix, duplicates should be consolidated by summing
    let triplets: Vec<(usize, usize, f64)> = vec![
        (0, 0, 1.0),
        (0, 1, -0.5),
        (0, 0, 2.0), // Duplicate (0,0)!
        (1, 0, -0.5),
        (1, 1, 1.0),
    ];

    let mat = SparseMatrix::from_triplets(2, 2, &triplets);

    println!("Matrix with duplicate triplets:");
    println!("  row_offsets: {:?}", mat.row_offsets);
    println!("  col_indices: {:?}", mat.col_indices);
    println!("  values: {:?}", mat.values);

    // Verify duplicates are consolidated
    // Row 0 should have 2 entries (not 3): (0,0)=3.0 and (0,1)=-0.5
    assert_eq!(
        mat.row_offsets[1] - mat.row_offsets[0],
        2,
        "Row 0 should have exactly 2 entries after consolidation"
    );

    // Row 1 should have 2 entries: (1,0)=-0.5 and (1,1)=1.0
    assert_eq!(
        mat.row_offsets[2] - mat.row_offsets[1],
        2,
        "Row 1 should have 2 entries"
    );

    // Verify column indices are sorted within each row
    for row in 0..mat.n_rows {
        let start = mat.row_offsets[row];
        let end = mat.row_offsets[row + 1];
        for k in start..end - 1 {
            assert!(
                mat.col_indices[k] < mat.col_indices[k + 1],
                "Column indices should be sorted within row {}",
                row
            );
        }
    }

    // Test matrix-vector multiplication
    let x = vec![1.0, 1.0];
    let mut y = vec![0.0, 0.0];
    mat.mat_vec_mul(&x, &mut y);

    println!("\nMat-vec result: {:?}", y);
    println!("Expected: [2.5, 0.5]");

    // The result should be correct with consolidated entries
    assert!((y[0] - 2.5).abs() < 1e-10, "Row 0 result should be 2.5");
    assert!((y[1] - 0.5).abs() < 1e-10, "Row 1 result should be 0.5");
}
