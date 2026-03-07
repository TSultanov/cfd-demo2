use crate::solver::mesh::{BoundaryType, Mesh};
use nalgebra::{Point2, Vector2};

/// A vertex handle returned by [`MeshBuilder::add_vertex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexId(pub usize);

/// A cell handle returned by [`MeshBuilder::add_cell`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub usize);

/// A face handle returned by [`MeshBuilder::add_face`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceId(pub usize);

struct CellData {
    vertices: Vec<VertexId>,
    faces: Vec<FaceId>,
}

struct FaceData {
    v1: VertexId,
    v2: VertexId,
    owner: CellId,
    neighbor: Option<CellId>,
    boundary: Option<BoundaryType>,
}

/// CPU-side mesh builder that encapsulates the error-prone parallel-array
/// bookkeeping of [`Mesh`].
///
/// Usage:
/// 1. Add vertices with [`add_vertex`].
/// 2. Add cells with [`add_cell`].
/// 3. Add faces with [`add_face`], linking them to owner/neighbor cells.
/// 4. Call [`build`] to flatten into a [`Mesh`] with correct geometry.
pub struct MeshBuilder {
    vx: Vec<f64>,
    vy: Vec<f64>,
    v_fixed: Vec<bool>,
    cells: Vec<CellData>,
    faces: Vec<FaceData>,
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            vx: Vec::new(),
            vy: Vec::new(),
            v_fixed: Vec::new(),
            cells: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn with_capacity(n_verts: usize, n_cells: usize, n_faces: usize) -> Self {
        Self {
            vx: Vec::with_capacity(n_verts),
            vy: Vec::with_capacity(n_verts),
            v_fixed: Vec::with_capacity(n_verts),
            cells: Vec::with_capacity(n_cells),
            faces: Vec::with_capacity(n_faces),
        }
    }

    /// Add a vertex at position (x, y). If `fixed` is true, the vertex is
    /// on a geometric boundary and should not be moved by smoothing.
    pub fn add_vertex(&mut self, x: f64, y: f64, fixed: bool) -> VertexId {
        let id = VertexId(self.vx.len());
        self.vx.push(x);
        self.vy.push(y);
        self.v_fixed.push(fixed);
        id
    }

    /// Add a cell defined by the given ordered vertex ring.
    pub fn add_cell(&mut self, vertices: &[VertexId]) -> CellId {
        let id = CellId(self.cells.len());
        self.cells.push(CellData {
            vertices: vertices.to_vec(),
            faces: Vec::new(),
        });
        id
    }

    /// Add a face between two vertices, owned by `owner` with an optional
    /// `neighbor` cell on the other side.
    pub fn add_face(
        &mut self,
        v1: VertexId,
        v2: VertexId,
        owner: CellId,
        neighbor: Option<CellId>,
        boundary: Option<BoundaryType>,
    ) -> FaceId {
        let id = FaceId(self.faces.len());
        self.faces.push(FaceData {
            v1,
            v2,
            owner,
            neighbor,
            boundary,
        });
        self.cells[owner.0].faces.push(id);
        if let Some(n) = neighbor {
            self.cells[n.0].faces.push(id);
        }
        id
    }

    /// Update the neighbor of an existing face and register the face
    /// with the neighbor cell. Clears the boundary type.
    pub fn set_face_neighbor(&mut self, face: FaceId, neighbor: CellId) {
        let f = &mut self.faces[face.0];
        f.neighbor = Some(neighbor);
        f.boundary = None;
        self.cells[neighbor.0].faces.push(face);
    }

    /// Return the position of a vertex.
    #[inline]
    pub fn vertex_pos(&self, v: VertexId) -> (f64, f64) {
        (self.vx[v.0], self.vy[v.0])
    }

    /// Return the position of a vertex as a `Point2`.
    #[inline]
    pub fn vertex_point(&self, v: VertexId) -> Point2<f64> {
        Point2::new(self.vx[v.0], self.vy[v.0])
    }

    /// Current number of cells.
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Current number of vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vx.len()
    }

    /// Current number of faces.
    #[inline]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Flatten into the SoA [`Mesh`] struct, computing all geometry
    /// (face centers, normals, areas, cell centroids, volumes).
    pub fn build(self) -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vx = self.vx;
        mesh.vy = self.vy;
        mesh.v_fixed = self.v_fixed;

        // Flatten faces
        for f in &self.faces {
            mesh.face_v1.push(f.v1.0);
            mesh.face_v2.push(f.v2.0);
            mesh.face_owner.push(f.owner.0);
            mesh.face_neighbor.push(f.neighbor.map(|c| c.0));
            mesh.face_boundary.push(f.boundary);
            // Placeholders — recalculate_geometry fills these
            mesh.face_cx.push(0.0);
            mesh.face_cy.push(0.0);
            mesh.face_nx.push(0.0);
            mesh.face_ny.push(0.0);
            mesh.face_area.push(0.0);
        }

        // Flatten cells
        mesh.cell_face_offsets.push(0);
        mesh.cell_vertex_offsets.push(0);
        for cell in &self.cells {
            mesh.cell_cx.push(0.0);
            mesh.cell_cy.push(0.0);
            mesh.cell_vol.push(0.0);
            for &fid in &cell.faces {
                mesh.cell_faces.push(fid.0);
            }
            mesh.cell_face_offsets.push(mesh.cell_faces.len());
            for &vid in &cell.vertices {
                mesh.cell_vertices.push(vid.0);
            }
            mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
        }

        // First pass: compute geometry with arbitrary normal orientation
        mesh.recalculate_geometry();

        // Fix normal orientation: normals must point from owner to neighbor
        // (i.e., away from the owner cell center).
        for i in 0..mesh.num_faces() {
            let owner = mesh.face_owner[i];
            let c_owner = Point2::new(mesh.cell_cx[owner], mesh.cell_cy[owner]);
            let f_center = Point2::new(mesh.face_cx[i], mesh.face_cy[i]);
            let normal = Vector2::new(mesh.face_nx[i], mesh.face_ny[i]);
            if (f_center - c_owner).dot(&normal) < 0.0 {
                mesh.face_nx[i] = -normal.x;
                mesh.face_ny[i] = -normal.y;
            }
        }

        // Second pass: recalculate with correctly-oriented normal hints
        mesh.recalculate_geometry();

        mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_single_quad() {
        let mut b = MeshBuilder::new();
        let v0 = b.add_vertex(0.0, 0.0, false);
        let v1 = b.add_vertex(1.0, 0.0, false);
        let v2 = b.add_vertex(1.0, 1.0, false);
        let v3 = b.add_vertex(0.0, 1.0, false);

        let c = b.add_cell(&[v0, v1, v2, v3]);

        b.add_face(v0, v1, c, None, Some(BoundaryType::Wall));
        b.add_face(v1, v2, c, None, Some(BoundaryType::Outlet));
        b.add_face(v2, v3, c, None, Some(BoundaryType::Wall));
        b.add_face(v3, v0, c, None, Some(BoundaryType::Inlet));

        let mesh = b.build();
        assert_eq!(mesh.num_cells(), 1);
        assert_eq!(mesh.num_faces(), 4);
        assert_eq!(mesh.num_vertices(), 4);
        assert!((mesh.cell_vol[0] - 1.0).abs() < 1e-12);
        assert!((mesh.cell_cx[0] - 0.5).abs() < 1e-12);
        assert!((mesh.cell_cy[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn build_two_triangles_shared_face() {
        let mut b = MeshBuilder::new();
        let v0 = b.add_vertex(0.0, 0.0, false);
        let v1 = b.add_vertex(1.0, 0.0, false);
        let v2 = b.add_vertex(1.0, 1.0, false);
        let v3 = b.add_vertex(0.0, 1.0, false);

        let c0 = b.add_cell(&[v0, v1, v2]);
        let c1 = b.add_cell(&[v0, v2, v3]);

        b.add_face(v0, v1, c0, None, Some(BoundaryType::Wall));
        b.add_face(v1, v2, c0, None, Some(BoundaryType::Outlet));
        b.add_face(v0, v2, c0, Some(c1), None); // shared internal face
        b.add_face(v2, v3, c1, None, Some(BoundaryType::Wall));
        b.add_face(v3, v0, c1, None, Some(BoundaryType::Inlet));

        let mesh = b.build();
        assert_eq!(mesh.num_cells(), 2);
        assert_eq!(mesh.num_faces(), 5);
        // Total area = 1.0
        let total_vol: f64 = mesh.cell_vol.iter().sum();
        assert!((total_vol - 1.0).abs() < 1e-12);
        // Each triangle = 0.5
        assert!((mesh.cell_vol[0] - 0.5).abs() < 1e-12);
        assert!((mesh.cell_vol[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn set_face_neighbor_updates_correctly() {
        let mut b = MeshBuilder::new();
        let v0 = b.add_vertex(0.0, 0.0, false);
        let v1 = b.add_vertex(1.0, 0.0, false);
        let v2 = b.add_vertex(1.0, 1.0, false);
        let v3 = b.add_vertex(0.0, 1.0, false);

        let c0 = b.add_cell(&[v0, v1, v2]);
        // Initially add the diagonal face as boundary
        let f_diag = b.add_face(v0, v2, c0, None, Some(BoundaryType::Wall));

        // Later add the second cell and update the face
        let c1 = b.add_cell(&[v0, v2, v3]);
        b.set_face_neighbor(f_diag, c1);

        let mesh = b.build();
        // The diagonal face should now be internal
        let diag_idx = f_diag.0;
        assert!(mesh.face_neighbor[diag_idx].is_some());
        assert!(mesh.face_boundary[diag_idx].is_none());
    }

    #[test]
    fn normals_point_away_from_owner() {
        let mut b = MeshBuilder::new();
        let v0 = b.add_vertex(0.0, 0.0, false);
        let v1 = b.add_vertex(2.0, 0.0, false);
        let v2 = b.add_vertex(2.0, 1.0, false);
        let v3 = b.add_vertex(0.0, 1.0, false);

        let c = b.add_cell(&[v0, v1, v2, v3]);
        b.add_face(v0, v1, c, None, Some(BoundaryType::Wall));
        b.add_face(v1, v2, c, None, Some(BoundaryType::Outlet));
        b.add_face(v2, v3, c, None, Some(BoundaryType::Wall));
        b.add_face(v3, v0, c, None, Some(BoundaryType::Inlet));

        let mesh = b.build();
        let cx = mesh.cell_cx[0];
        let cy = mesh.cell_cy[0];

        for i in 0..mesh.num_faces() {
            let fx = mesh.face_cx[i];
            let fy = mesh.face_cy[i];
            let nx = mesh.face_nx[i];
            let ny = mesh.face_ny[i];
            let dot = (fx - cx) * nx + (fy - cy) * ny;
            assert!(
                dot >= 0.0,
                "Face {i} normal should point away from owner: dot={dot}"
            );
        }
    }
}
