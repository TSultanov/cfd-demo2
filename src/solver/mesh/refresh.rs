//! Mesh-refresh seam: shared geometry casts + topology snapshot.
//!
//! The f64→f32 mesh-geometry cast lives here in one place, used by init
//! (`init_mesh` GPU, `upload_mesh` CPU) and refresh on both backends, so they
//! upload bit-identical f32 arrays by construction.
//!
//! A `Geometry`-level refresh is only valid when the mesh topology (face set,
//! adjacency, boundary classification, cell→face connectivity) is unchanged;
//! [`MeshTopology`] is the build-time snapshot that refresh validates against.

use super::structs::Mesh;

/// Outcome flags a [`MeshRefreshLevel::Topology`] refresh reports back to the
/// caller (a `Geometry` refresh reports the default — nothing was reset).
///
/// The load-bearing field is `bc_overrides_reset`: a topology refresh
/// re-derives the boundary-condition tables from the model spec and the new
/// `face_boundary` classification, so any per-face runtime overrides the caller
/// applied via `set_boundary_values_per_face` (keyed by the OLD face indices)
/// are gone. The caller owns re-applying them against the new face indexing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshRefreshReport {
    /// `true` when the BC tables were rebuilt from the model spec and any
    /// runtime per-face BC overrides were dropped (Topology refresh only).
    pub bc_overrides_reset: bool,
}

/// How much of the mesh changed since the solver was built / last refreshed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRefreshLevel {
    /// Same faces, same owners/neighbors/boundary tags; only positions moved.
    /// Updates face areas/normals/centers (+ wrap shifts) and cell
    /// centers/volumes in place. Bind groups / buffer allocations survive.
    Geometry,
    /// Face set / adjacency / boundary classification changed (cell count must
    /// still be unchanged — seed↔cell identity). Rebuilds CSR + face-indexed
    /// buffers. Not yet implemented.
    Topology,
}

/// The f64→f32 casts of the mesh geometry, exactly as the solver kernels
/// consume them. Vector2 arrays are interleaved `x,y,x,y,...` (the byte layout
/// of both a WGSL `array<Vector2>` storage buffer and the CPU `Store::F32
/// {comps: 2}` buffer).
pub struct GeometryArrays {
    /// Per-face area (f32 cast), length `num_faces`.
    pub face_areas: Vec<f32>,
    /// Per-face unit normal, interleaved, length `num_faces * 2`.
    pub face_normals: Vec<f32>,
    /// Per-face center, interleaved, length `num_faces * 2`.
    pub face_centers: Vec<f32>,
    /// Per-face periodic wrap shift, interleaved, length `num_faces * 2`
    /// (all-zero on non-periodic meshes, where `Mesh::face_wrap_shift` is empty).
    pub face_wrap_shift: Vec<f32>,
    /// Per-cell center, interleaved, length `num_cells * 2`.
    pub cell_centers: Vec<f32>,
    /// Per-cell volume (f32 cast), length `num_cells`.
    pub cell_vols: Vec<f32>,
}

fn interleave_f32(xs: &[f64], ys: &[f64]) -> Vec<f32> {
    xs.iter()
        .zip(ys)
        .flat_map(|(&x, &y)| [x as f32, y as f32])
        .collect()
}

/// Produce the f32 geometry arrays for `mesh`. This is the single source of
/// truth for the f64→f32 geometry cast: `init_mesh` (GPU), `upload_mesh` (CPU)
/// and the `refresh_mesh` paths all consume it, so init and refresh — and the
/// two backends — see bit-identical f32 geometry by construction.
pub fn mesh_geometry_f32(mesh: &Mesh) -> GeometryArrays {
    let face_wrap_shift = if mesh.face_wrap_shift.is_empty() {
        vec![0.0f32; mesh.num_faces() * 2]
    } else {
        mesh.face_wrap_shift
            .iter()
            .flat_map(|&[sx, sy]| [sx as f32, sy as f32])
            .collect()
    };
    GeometryArrays {
        face_areas: mesh.face_area.iter().map(|&a| a as f32).collect(),
        face_normals: interleave_f32(&mesh.face_nx, &mesh.face_ny),
        face_centers: interleave_f32(&mesh.face_cx, &mesh.face_cy),
        face_wrap_shift,
        cell_centers: interleave_f32(&mesh.cell_cx, &mesh.cell_cy),
        cell_vols: mesh.cell_vol.iter().map(|&v| v as f32).collect(),
    }
}

/// Build-time snapshot of the mesh topology, kept by each solver backend so a
/// `Geometry`-level refresh can verify the incoming mesh really is
/// topology-identical (same faces, same adjacency, same boundary tags, same
/// cell→face connectivity) before overwriting geometry in place.
pub struct MeshTopology {
    num_cells: usize,
    num_faces: usize,
    face_owner: Vec<u32>,
    /// `-1` encodes "boundary face" (`face_neighbor == None`).
    face_neighbor: Vec<i32>,
    /// `BoundaryType::bc_table_index()`, `0` for interior/untagged faces.
    face_boundary: Vec<u32>,
    cell_face_offsets: Vec<u32>,
    cell_faces: Vec<u32>,
}

impl MeshTopology {
    /// Face count of the snapshotted topology (logical size of face-indexed
    /// buffers — used for sized bindings over capacity-reserved allocations).
    pub fn num_faces(&self) -> usize {
        self.num_faces
    }

    /// Length of the `cell_faces` array (logical size of `cell_faces` /
    /// `cell_face_matrix_indices` buffers).
    pub fn cell_faces_len(&self) -> usize {
        self.cell_faces.len()
    }

    /// Cell count of the snapshotted topology (the refresh invariant: a
    /// Topology refresh may change faces/adjacency but NEVER the cell count —
    /// seed↔cell identity is what lets cell-indexed state survive untouched).
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub fn from_mesh(mesh: &Mesh) -> Self {
        Self {
            num_cells: mesh.num_cells(),
            num_faces: mesh.num_faces(),
            face_owner: mesh.face_owner.iter().map(|&o| o as u32).collect(),
            face_neighbor: mesh
                .face_neighbor
                .iter()
                .map(|n| n.map(|v| v as i32).unwrap_or(-1))
                .collect(),
            face_boundary: mesh
                .face_boundary
                .iter()
                .map(|b| b.map(|t| t.bc_table_index() as u32).unwrap_or(0))
                .collect(),
            cell_face_offsets: mesh.cell_face_offsets.iter().map(|&o| o as u32).collect(),
            cell_faces: mesh.cell_faces.iter().map(|&f| f as u32).collect(),
        }
    }

    /// Validate that `mesh` has exactly this topology, `Err` otherwise.
    ///
    /// Full O(num_cells + num_faces) array comparison; acceptable because it
    /// only runs on explicit `refresh_mesh` calls, never in the step loop.
    pub fn validate_matches(&self, mesh: &Mesh) -> Result<(), String> {
        if mesh.num_cells() != self.num_cells {
            return Err(format!(
                "geometry refresh requires identical topology: num_cells changed ({} -> {})",
                self.num_cells,
                mesh.num_cells()
            ));
        }
        if mesh.num_faces() != self.num_faces {
            return Err(format!(
                "geometry refresh requires identical topology: num_faces changed ({} -> {})",
                self.num_faces,
                mesh.num_faces()
            ));
        }
        let owner_ok = mesh
            .face_owner
            .iter()
            .zip(&self.face_owner)
            .all(|(&m, &s)| m as u32 == s);
        if !owner_ok {
            return Err("geometry refresh requires identical topology: face_owner differs".into());
        }
        let neighbor_ok = mesh
            .face_neighbor
            .iter()
            .zip(&self.face_neighbor)
            .all(|(m, &s)| m.map(|v| v as i32).unwrap_or(-1) == s);
        if !neighbor_ok {
            return Err(
                "geometry refresh requires identical topology: face_neighbor differs".into(),
            );
        }
        let boundary_ok = mesh
            .face_boundary
            .iter()
            .zip(&self.face_boundary)
            .all(|(m, &s)| m.map(|t| t.bc_table_index() as u32).unwrap_or(0) == s);
        if !boundary_ok {
            return Err(
                "geometry refresh requires identical topology: face_boundary differs".into(),
            );
        }
        if mesh.cell_face_offsets.len() != self.cell_face_offsets.len()
            || !mesh
                .cell_face_offsets
                .iter()
                .zip(&self.cell_face_offsets)
                .all(|(&m, &s)| m as u32 == s)
        {
            return Err(
                "geometry refresh requires identical topology: cell_face_offsets differ".into(),
            );
        }
        if mesh.cell_faces.len() != self.cell_faces.len()
            || !mesh
                .cell_faces
                .iter()
                .zip(&self.cell_faces)
                .all(|(&m, &s)| m as u32 == s)
        {
            return Err("geometry refresh requires identical topology: cell_faces differ".into());
        }
        Ok(())
    }
}
