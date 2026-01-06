use super::geometry::Geometry;
use super::structs::{BoundaryType, Mesh};
use nalgebra::{Point2, Vector2};

pub fn compute_normal(geo: &(impl Geometry + ?Sized), p: Point2<f64>) -> Vector2<f64> {
    let eps = 1e-6;
    let d_x = geo.sdf(&Point2::new(p.x + eps, p.y)) - geo.sdf(&Point2::new(p.x - eps, p.y));
    let d_y = geo.sdf(&Point2::new(p.x, p.y + eps)) - geo.sdf(&Point2::new(p.x, p.y - eps));
    Vector2::new(d_x, d_y).normalize()
}

pub fn intersect_lines(
    p1: Point2<f64>,
    n1: Vector2<f64>,
    p2: Point2<f64>,
    n2: Vector2<f64>,
) -> Option<Point2<f64>> {
    let det = n1.x * n2.y - n1.y * n2.x;
    if det.abs() < 1e-6 {
        return None;
    }

    let d1 = p1.coords.dot(&n1);
    let d2 = p2.coords.dot(&n2);

    let x = (d1 * n2.y - d2 * n1.y) / det;
    let y = (d2 * n1.x - d1 * n2.x) / det;

    Some(Point2::new(x, y))
}

pub fn generate_structured_rect_mesh(
    nx: usize,
    ny: usize,
    length: f64,
    height: f64,
    left: BoundaryType,
    right: BoundaryType,
    bottom: BoundaryType,
    top: BoundaryType,
) -> Mesh {
    assert!(nx > 0, "nx must be > 0");
    assert!(ny > 0, "ny must be > 0");
    assert!(length > 0.0, "length must be > 0");
    assert!(height > 0.0, "height must be > 0");

    let dx = length / nx as f64;
    let dy = height / ny as f64;

    let mut mesh = Mesh::new();

    // --- Vertices ---
    let num_vertices = (nx + 1) * (ny + 1);
    mesh.vx = vec![0.0; num_vertices];
    mesh.vy = vec![0.0; num_vertices];
    mesh.v_fixed = vec![false; num_vertices];

    let vid = |i: usize, j: usize| -> usize { j * (nx + 1) + i };
    for j in 0..=ny {
        for i in 0..=nx {
            let v = vid(i, j);
            mesh.vx[v] = i as f64 * dx;
            mesh.vy[v] = j as f64 * dy;
        }
    }

    // --- Faces (edges) ---
    // We build a face list for all vertical and horizontal grid edges and a mapping to cell faces.
    let mut face_v1: Vec<usize> = Vec::new();
    let mut face_v2: Vec<usize> = Vec::new();
    let mut face_owner: Vec<usize> = Vec::new();
    let mut face_neighbor: Vec<Option<usize>> = Vec::new();
    let mut face_boundary: Vec<Option<BoundaryType>> = Vec::new();
    let mut face_nx: Vec<f64> = Vec::new();
    let mut face_ny: Vec<f64> = Vec::new();

    let cell_id = |i: usize, j: usize| -> usize { j * nx + i };

    // Maps (vertical edge i,j) -> face index, where i in 0..=nx, j in 0..ny-1
    let mut vert_face = vec![usize::MAX; (nx + 1) * ny];
    let vfid = |i: usize, j: usize| -> usize { j * (nx + 1) + i };

    for j in 0..ny {
        for i in 0..=nx {
            let v0 = vid(i, j);
            let v1 = vid(i, j + 1);
            let is_left = i == 0;
            let is_right = i == nx;

            let (owner, neighbor, bc, nx_out) = if is_left {
                (cell_id(0, j), None, Some(left), -1.0)
            } else if is_right {
                (cell_id(nx - 1, j), None, Some(right), 1.0)
            } else {
                (cell_id(i - 1, j), Some(cell_id(i, j)), None, 1.0)
            };

            let idx = face_v1.len();
            face_v1.push(v0);
            face_v2.push(v1);
            face_owner.push(owner);
            face_neighbor.push(neighbor);
            face_boundary.push(bc);
            face_nx.push(nx_out);
            face_ny.push(0.0);
            vert_face[vfid(i, j)] = idx;
        }
    }

    // Maps (horizontal edge i,j) -> face index, where i in 0..nx-1, j in 0..=ny
    let mut horiz_face = vec![usize::MAX; nx * (ny + 1)];
    let hfid = |i: usize, j: usize| -> usize { j * nx + i };

    for j in 0..=ny {
        for i in 0..nx {
            let v0 = vid(i, j);
            let v1 = vid(i + 1, j);
            let is_bottom = j == 0;
            let is_top = j == ny;

            let (owner, neighbor, bc, ny_out) = if is_bottom {
                (cell_id(i, 0), None, Some(bottom), -1.0)
            } else if is_top {
                (cell_id(i, ny - 1), None, Some(top), 1.0)
            } else {
                (cell_id(i, j - 1), Some(cell_id(i, j)), None, 1.0)
            };

            let idx = face_v1.len();
            face_v1.push(v0);
            face_v2.push(v1);
            face_owner.push(owner);
            face_neighbor.push(neighbor);
            face_boundary.push(bc);
            face_nx.push(0.0);
            face_ny.push(ny_out);
            horiz_face[hfid(i, j)] = idx;
        }
    }

    mesh.face_v1 = face_v1;
    mesh.face_v2 = face_v2;
    mesh.face_owner = face_owner;
    mesh.face_neighbor = face_neighbor;
    mesh.face_boundary = face_boundary;
    mesh.face_nx = face_nx;
    mesh.face_ny = face_ny;
    mesh.face_area = vec![0.0; mesh.face_v1.len()];
    mesh.face_cx = vec![0.0; mesh.face_v1.len()];
    mesh.face_cy = vec![0.0; mesh.face_v1.len()];

    // --- Cells ---
    let num_cells = nx * ny;
    mesh.cell_cx = vec![0.0; num_cells];
    mesh.cell_cy = vec![0.0; num_cells];
    mesh.cell_vol = vec![0.0; num_cells];

    mesh.cell_faces = Vec::with_capacity(num_cells * 4);
    mesh.cell_face_offsets = Vec::with_capacity(num_cells + 1);
    mesh.cell_vertices = Vec::with_capacity(num_cells * 4);
    mesh.cell_vertex_offsets = Vec::with_capacity(num_cells + 1);

    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    for j in 0..ny {
        for i in 0..nx {
            let left_face = vert_face[vfid(i, j)];
            let right_face = vert_face[vfid(i + 1, j)];
            let bottom_face = horiz_face[hfid(i, j)];
            let top_face = horiz_face[hfid(i, j + 1)];

            mesh.cell_faces.push(left_face);
            mesh.cell_faces.push(right_face);
            mesh.cell_faces.push(bottom_face);
            mesh.cell_faces.push(top_face);
            mesh.cell_face_offsets.push(mesh.cell_faces.len());

            let v00 = vid(i, j);
            let v10 = vid(i + 1, j);
            let v11 = vid(i + 1, j + 1);
            let v01 = vid(i, j + 1);
            mesh.cell_vertices.extend([v00, v10, v11, v01]);
            mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
        }
    }

    mesh.recalculate_geometry();
    mesh
}
