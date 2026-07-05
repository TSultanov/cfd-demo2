//! Pins that the factored scalar-CSR topology builders in `solver::mesh::csr`
//! produce bit-equal output to reference copies of the original inlined logic,
//! across four mesh families (structured, graded, meshless-Voronoi, cut-cell).
//! The reference copies below are verbatim transcriptions of that inlined logic
//! (GPU: `gpu::init::mesh::init_mesh`; CPU: `cpu::solver::build_csr_topology`),
//! kept in the test so production can delegate without weakening the check.
#![cfg(feature = "meshgen")]

use cfd2::solver::mesh::csr::{build_diag_first_scalar_csr, build_sorted_scalar_csr, ScalarCsr};
use cfd2::solver::mesh::{
    generate_cut_cell_mesh, generate_graded_rect_mesh, generate_meshless_voronoi_mesh,
    generate_structured_rect_mesh, AxisGrading, BoundarySides, BoundaryType, ChannelWithObstacle,
    Mesh, RectangularChannel,
};
use nalgebra::{Point2, Vector2};

// ── Reference builders ────────────────────────────────────────────────────────

/// Reference copy of the GPU sorted-adjacency scalar-CSR build
/// (`gpu::init::mesh::init_mesh`).
fn reference_sorted_scalar_csr(mesh: &Mesh) -> ScalarCsr {
    let num_cells = mesh.cell_cx.len() as u32;

    let mut scalar_row_offsets = vec![0u32; num_cells as usize + 1];
    let mut scalar_col_indices = Vec::new();

    let mut adj = vec![Vec::new(); num_cells as usize];
    for (i, &owner) in mesh.face_owner.iter().enumerate() {
        if let Some(neighbor) = mesh.face_neighbor[i] {
            adj[owner].push(neighbor);
            adj[neighbor].push(owner);
        }
    }

    for (i, list) in adj.iter_mut().enumerate() {
        list.push(i);
        list.sort();
        list.dedup();
    }

    let mut current_offset = 0;
    for (i, list) in adj.iter().enumerate() {
        scalar_row_offsets[i] = current_offset;
        for &neighbor in list {
            scalar_col_indices.push(neighbor as u32);
        }
        current_offset += list.len() as u32;
    }
    scalar_row_offsets[num_cells as usize] = current_offset;

    let mut cell_face_matrix_indices = Vec::new();
    for i in 0..num_cells {
        let start = mesh.cell_face_offsets[i as usize];
        let end = mesh.cell_face_offsets[i as usize + 1];

        for k in start..end {
            let face_idx = mesh.cell_faces[k];
            let owner = mesh.face_owner[face_idx];
            let neighbor_opt = mesh.face_neighbor[face_idx];

            let neighbor = if owner == i as usize {
                neighbor_opt
            } else {
                Some(owner)
            };

            let target_col = match neighbor {
                Some(n) => n as u32,
                None => i,
            };

            let row_start = scalar_row_offsets[i as usize] as usize;
            let row_end = scalar_row_offsets[i as usize + 1] as usize;
            let cols = &scalar_col_indices[row_start..row_end];

            if let Ok(idx) = cols.binary_search(&target_col) {
                cell_face_matrix_indices.push((row_start + idx) as u32);
            } else {
                cell_face_matrix_indices.push(u32::MAX);
            }
        }
    }

    let mut diagonal_indices = Vec::with_capacity(num_cells as usize);
    for i in 0..num_cells {
        let row_start = scalar_row_offsets[i as usize] as usize;
        let row_end = scalar_row_offsets[i as usize + 1] as usize;
        let cols = &scalar_col_indices[row_start..row_end];
        let idx = cols.binary_search(&i).expect("diagonal must be present");
        diagonal_indices.push((row_start + idx) as u32);
    }

    ScalarCsr {
        row_offsets: scalar_row_offsets,
        col_indices: scalar_col_indices,
        diagonal_indices,
        cell_face_matrix_indices,
    }
}

/// Reference copy of the CPU diag-first scalar-CSR build
/// (`cpu::solver::build_csr_topology`).
fn reference_diag_first_scalar_csr(mesh: &Mesh) -> ScalarCsr {
    let n = mesh.num_cells();
    let mut row_offsets = vec![0u32; n + 1];
    for i in 0..n {
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        let interior = (start..end)
            .filter(|&k| mesh.face_neighbor[mesh.cell_faces[k]].is_some())
            .count();
        row_offsets[i + 1] = row_offsets[i] + 1 + interior as u32;
    }
    let nnz = *row_offsets.last().unwrap() as usize;
    let mut col_indices = vec![0u32; nnz];
    let mut diagonal_indices = vec![0u32; n];
    let mut cell_face_matrix_indices = vec![0u32; mesh.cell_faces.len()];

    for i in 0..n {
        let base = row_offsets[i] as usize;
        col_indices[base] = i as u32;
        diagonal_indices[i] = base as u32;
        let mut pos = base + 1;
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        for k in start..end {
            let f = mesh.cell_faces[k];
            match mesh.face_neighbor[f] {
                Some(nb) => {
                    let other = if mesh.face_owner[f] == i { nb } else { mesh.face_owner[f] };
                    col_indices[pos] = other as u32;
                    cell_face_matrix_indices[k] = pos as u32;
                    pos += 1;
                }
                None => {
                    cell_face_matrix_indices[k] = base as u32;
                }
            }
        }
    }
    ScalarCsr {
        row_offsets,
        col_indices,
        diagonal_indices,
        cell_face_matrix_indices,
    }
}

// ── Comparison helper ─────────────────────────────────────────────────────────

fn assert_csr_bit_equal(got: &ScalarCsr, want: &ScalarCsr, label: &str) {
    assert_eq!(
        got.row_offsets, want.row_offsets,
        "[{label}] row_offsets differ"
    );
    assert_eq!(
        got.col_indices, want.col_indices,
        "[{label}] col_indices differ"
    );
    assert_eq!(
        got.diagonal_indices, want.diagonal_indices,
        "[{label}] diagonal_indices differ"
    );
    assert_eq!(
        got.cell_face_matrix_indices, want.cell_face_matrix_indices,
        "[{label}] cell_face_matrix_indices differ"
    );
    println!(
        "[{label}] CSR bit-equal: {} cells, {} nnz, {} cell-faces",
        got.row_offsets.len() - 1,
        got.nnz(),
        got.cell_face_matrix_indices.len()
    );
}

// ── Meshes: four families ─────────────────────────────────────────────────────

fn structured_mesh() -> Mesh {
    generate_structured_rect_mesh(
        24,
        16,
        2.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

fn graded_mesh() -> Mesh {
    let grading = AxisGrading::TwoSided { ratio: 4.0 };
    generate_graded_rect_mesh(
        20,
        20,
        1.0,
        1.0,
        grading,
        grading,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

fn meshless_voronoi_mesh() -> Mesh {
    let geo = RectangularChannel {
        length: 2.0,
        height: 1.0,
    };
    generate_meshless_voronoi_mesh(&geo, 0.06, 0.06, 1.2, Vector2::new(2.0, 1.0))
}

fn cut_cell_mesh() -> Mesh {
    let length = 3.0;
    let geo = ChannelWithObstacle {
        length,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.15,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, 0.05, 0.05, 1.2, Vector2::new(length, 1.0));
    mesh.smooth(&geo, 0.3, 50);
    mesh
}

fn all_meshes() -> Vec<(&'static str, Mesh)> {
    vec![
        ("structured", structured_mesh()),
        ("graded", graded_mesh()),
        ("meshless-voronoi", meshless_voronoi_mesh()),
        ("cut-cell", cut_cell_mesh()),
    ]
}

#[test]
fn sorted_scalar_csr_matches_reference() {
    for (name, mesh) in all_meshes() {
        let got = build_sorted_scalar_csr(&mesh).expect("factored sorted CSR build");
        let want = reference_sorted_scalar_csr(&mesh);
        assert_csr_bit_equal(&got, &want, &format!("sorted/{name}"));
    }
}

#[test]
fn diag_first_scalar_csr_matches_reference() {
    for (name, mesh) in all_meshes() {
        let got = build_diag_first_scalar_csr(&mesh);
        let want = reference_diag_first_scalar_csr(&mesh);
        assert_csr_bit_equal(&got, &want, &format!("diag-first/{name}"));
    }
}
