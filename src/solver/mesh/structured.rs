use super::structs::{BoundaryType, Mesh};
use nalgebra::{Point2, Vector2};
use std::collections::HashMap;

/// Bundles the four boundary sides for structured mesh generators.
#[derive(Clone, Copy, Debug)]
pub struct BoundarySides {
    pub left: BoundaryType,
    pub right: BoundaryType,
    pub bottom: BoundaryType,
    pub top: BoundaryType,
}

impl BoundarySides {
    /// Creates a new BoundarySides with all sides set to Wall.
    pub fn wall() -> Self {
        Self {
            left: BoundaryType::Wall,
            right: BoundaryType::Wall,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        }
    }
}

pub fn generate_structured_rect_mesh(
    nx: usize,
    ny: usize,
    length: f64,
    height: f64,
    boundaries: BoundarySides,
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
                (cell_id(0, j), None, Some(boundaries.left), -1.0)
            } else if is_right {
                (cell_id(nx - 1, j), None, Some(boundaries.right), 1.0)
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
                (cell_id(i, 0), None, Some(boundaries.bottom), -1.0)
            } else if is_top {
                (cell_id(i, ny - 1), None, Some(boundaries.top), 1.0)
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

#[derive(Clone, Copy, Debug)]
enum CellEdge {
    Left,
    Right,
    Bottom,
    Top,
}

fn generate_structured_mesh_from_vertex_grid<F>(
    nx: usize,
    ny: usize,
    vx: Vec<f64>,
    vy: Vec<f64>,
    cell_exists: F,
    boundaries: BoundarySides,
) -> Mesh
where
    F: Fn(usize, usize) -> bool,
{
    assert_eq!(vx.len(), (nx + 1) * (ny + 1));
    assert_eq!(vy.len(), (nx + 1) * (ny + 1));

    let mut mesh = Mesh::new();
    mesh.vx = vx;
    mesh.vy = vy;
    mesh.v_fixed = vec![false; mesh.vx.len()];

    let vid = |i: usize, j: usize| -> usize { j * (nx + 1) + i };
    let mut cell_map = vec![None; nx * ny];
    let cell_key = |i: usize, j: usize| -> usize { j * nx + i };

    // Assign contiguous cell indices for existing cells.
    let mut next_cell = 0usize;
    for j in 0..ny {
        for i in 0..nx {
            if cell_exists(i, j) {
                cell_map[cell_key(i, j)] = Some(next_cell);
                next_cell += 1;
            }
        }
    }

    let num_cells = next_cell;
    mesh.cell_cx = vec![0.0; num_cells];
    mesh.cell_cy = vec![0.0; num_cells];
    mesh.cell_vol = vec![0.0; num_cells];

    mesh.cell_faces = Vec::with_capacity(num_cells * 4);
    mesh.cell_face_offsets = Vec::with_capacity(num_cells + 1);
    mesh.cell_vertices = Vec::with_capacity(num_cells * 4);
    mesh.cell_vertex_offsets = Vec::with_capacity(num_cells + 1);
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    let mut face_map: HashMap<(usize, usize), usize> = HashMap::new();

    let boundary_for_new_face = |i: usize, j: usize, edge: CellEdge| -> BoundaryType {
        match edge {
            CellEdge::Left => {
                if i == 0 {
                    boundaries.left
                } else {
                    BoundaryType::Wall
                }
            }
            CellEdge::Right => {
                if i + 1 == nx {
                    boundaries.right
                } else {
                    BoundaryType::Wall
                }
            }
            CellEdge::Bottom => {
                if j == 0 {
                    boundaries.bottom
                } else {
                    BoundaryType::Wall
                }
            }
            CellEdge::Top => {
                if j + 1 == ny {
                    boundaries.top
                } else {
                    BoundaryType::Wall
                }
            }
        }
    };

    for j in 0..ny {
        for i in 0..nx {
            let Some(cell_idx) = cell_map[cell_key(i, j)] else {
                continue;
            };

            let v00 = vid(i, j);
            let v10 = vid(i + 1, j);
            let v11 = vid(i + 1, j + 1);
            let v01 = vid(i, j + 1);

            let verts = [v00, v10, v11, v01];
            mesh.cell_vertices.extend(verts);
            mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());

            let cell_edges: [(usize, usize, CellEdge); 4] = [
                (v00, v10, CellEdge::Bottom),
                (v10, v11, CellEdge::Right),
                (v11, v01, CellEdge::Top),
                (v01, v00, CellEdge::Left),
            ];

            for (v1, v2, edge_kind) in cell_edges {
                let (min_v, max_v) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                let key = (min_v, max_v);

                if let Some(&face_idx) = face_map.get(&key) {
                    mesh.face_neighbor[face_idx] = Some(cell_idx);
                    mesh.face_boundary[face_idx] = None;
                    mesh.cell_faces.push(face_idx);
                    continue;
                }

                let p1 = Point2::new(mesh.vx[v1], mesh.vy[v1]);
                let p2 = Point2::new(mesh.vx[v2], mesh.vy[v2]);
                let edge_vec = p2 - p1;
                let area = edge_vec.norm();
                let center = Point2::from((p1.coords + p2.coords) * 0.5);
                let normal = Vector2::new(edge_vec.y, -edge_vec.x).normalize();

                let boundary_type = boundary_for_new_face(i, j, edge_kind);

                let face_idx = mesh.face_cx.len();
                mesh.face_v1.push(v1);
                mesh.face_v2.push(v2);
                mesh.face_owner.push(cell_idx);
                mesh.face_neighbor.push(None);
                mesh.face_boundary.push(Some(boundary_type));
                mesh.face_nx.push(normal.x);
                mesh.face_ny.push(normal.y);
                mesh.face_area.push(area);
                mesh.face_cx.push(center.x);
                mesh.face_cy.push(center.y);

                face_map.insert(key, face_idx);
                mesh.cell_faces.push(face_idx);
            }

            mesh.cell_face_offsets.push(mesh.cell_faces.len());
        }
    }

    mesh.recalculate_geometry();
    mesh
}

pub fn generate_structured_backwards_step_mesh(
    nx: usize,
    ny: usize,
    length: f64,
    height_outlet: f64,
    height_inlet: f64,
    step_x: f64,
) -> Mesh {
    assert!(height_inlet > 0.0 && height_outlet > height_inlet);
    assert!(step_x > 0.0 && step_x < length);

    let dx = length / nx as f64;
    let dy = height_outlet / ny as f64;
    let nx_step = (step_x / dx).round() as usize;
    let step_h = height_outlet - height_inlet;
    let ny_step = (step_h / dy).round() as usize;

    assert!(
        (nx_step as f64 * dx - step_x).abs() < 1e-12,
        "step_x must align with dx"
    );
    assert!(
        (ny_step as f64 * dy - step_h).abs() < 1e-12,
        "step height must align with dy"
    );

    let num_vertices = (nx + 1) * (ny + 1);
    let mut vx = vec![0.0; num_vertices];
    let mut vy = vec![0.0; num_vertices];
    let vid = |i: usize, j: usize| -> usize { j * (nx + 1) + i };
    for j in 0..=ny {
        for i in 0..=nx {
            let v = vid(i, j);
            vx[v] = i as f64 * dx;
            vy[v] = j as f64 * dy;
        }
    }

    let cell_exists = move |i: usize, j: usize| -> bool { i >= nx_step || j >= ny_step };

    generate_structured_mesh_from_vertex_grid(
        nx,
        ny,
        vx,
        vy,
        cell_exists,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

pub fn generate_structured_trapezoid_mesh(
    nx: usize,
    ny: usize,
    length: f64,
    height: f64,
    ramp_height: f64,
    boundaries: BoundarySides,
) -> Mesh {
    assert!(nx > 0 && ny > 0);
    assert!(length > 0.0 && height > 0.0);
    assert!(ramp_height >= 0.0 && ramp_height < height);

    let num_vertices = (nx + 1) * (ny + 1);
    let mut vx = vec![0.0; num_vertices];
    let mut vy = vec![0.0; num_vertices];
    let vid = |i: usize, j: usize| -> usize { j * (nx + 1) + i };

    for j in 0..=ny {
        let eta = j as f64 / ny as f64;
        for i in 0..=nx {
            let xi = i as f64 / nx as f64;
            let x = xi * length;
            let y_bottom = xi * ramp_height;
            let y = y_bottom + eta * (height - y_bottom);
            let v = vid(i, j);
            vx[v] = x;
            vy[v] = y;
        }
    }

    generate_structured_mesh_from_vertex_grid(
        nx,
        ny,
        vx,
        vy,
        |_i, _j| true,
        boundaries,
    )
}
