//! Scalar-CSR topology builders shared by the backend init paths and the
//! `Topology`-level mesh refresh (which must rebuild exactly what init built).
//!
//! The two backends use **different** row layouts and cannot share a builder:
//!
//! - **GPU** ([`build_sorted_scalar_csr`]): rows hold the diagonal plus all
//!   interior-face neighbors in **sorted column order** (adjacency lists are
//!   sorted + deduped; ranks found by binary search). Consumed by
//!   `gpu::init::mesh::init_mesh` and the FGMRES/AMG/Schur stack via the
//!   block expansion (`gpu::csr::build_block_csr`).
//! - **CPU** ([`build_diag_first_scalar_csr`]): rows hold the diagonal
//!   **first** (rank 0), then one entry per interior face in cell-face order.
//!   Consumed by `cpu::solver` (`linalg` block solvers assume rank 0 is the
//!   diagonal).
//!
//! Both builders are deterministic pure functions of the mesh topology
//! (sort/order-based — no hash-map iteration), which makes a no-op topology
//! refresh byte-identical by construction.

use super::structs::Mesh;

/// A scalar (one entry per cell pair) CSR topology plus the two per-mesh index
/// maps the assembly kernels consume. Layout (sorted vs diag-first) depends on
/// the builder that produced it — see the module docs.
pub struct ScalarCsr {
    /// Row start offsets, length `num_cells + 1`.
    pub row_offsets: Vec<u32>,
    /// Column indices, length `nnz` (== `row_offsets[num_cells]`).
    pub col_indices: Vec<u32>,
    /// CSR rank of each cell's diagonal entry, length `num_cells`.
    pub diagonal_indices: Vec<u32>,
    /// CSR rank for every (cell, face) pair, indexed like `mesh.cell_faces`.
    /// Boundary faces map to a harmless in-row rank (GPU: the diagonal; CPU:
    /// the diagonal) — kernels guard with `is_boundary` before use.
    pub cell_face_matrix_indices: Vec<u32>,
}

impl ScalarCsr {
    /// Number of stored entries (scalar nonzeros).
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }
}

/// Build the GPU scalar-CSR layout: sorted-adjacency rows (diagonal at its
/// sorted column position).
///
/// `Err` only if a cell's diagonal is missing from its row — impossible by
/// construction, kept as a hard fail-fast.
pub fn build_sorted_scalar_csr(mesh: &Mesh) -> Result<ScalarCsr, String> {
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

            // Boundary faces map to the diagonal so every (cell, face) pair has a
            // valid CSR rank ("ghost equals owner" convention); kernels guard with
            // is_boundary before use.
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
                // Should not happen: scalar CSR always contains the diagonal and any true neighbor.
                cell_face_matrix_indices.push(u32::MAX);
            }
        }
    }

    let mut diagonal_indices = Vec::with_capacity(num_cells as usize);
    for i in 0..num_cells {
        let row_start = scalar_row_offsets[i as usize] as usize;
        let row_end = scalar_row_offsets[i as usize + 1] as usize;
        let cols = &scalar_col_indices[row_start..row_end];

        if let Ok(idx) = cols.binary_search(&i) {
            diagonal_indices.push((row_start + idx) as u32);
        } else {
            return Err(format!("diagonal not found in CSR cols for cell {i}"));
        }
    }

    Ok(ScalarCsr {
        row_offsets: scalar_row_offsets,
        col_indices: scalar_col_indices,
        diagonal_indices,
        cell_face_matrix_indices,
    })
}

/// Build the CPU scalar-CSR layout: each row holds the diagonal (rank 0)
/// followed by one entry per interior face in cell-face order.
pub fn build_diag_first_scalar_csr(mesh: &Mesh) -> ScalarCsr {
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
                    // Boundary face: no column entry; point at the diagonal so any
                    // stray read is harmless (the kernel guards with is_boundary).
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
