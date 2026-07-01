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

/// Per-axis grading for [`generate_graded_rect_mesh`].
#[derive(Clone, Copy, Debug)]
pub enum AxisGrading {
    /// Equal cell sizes.
    Uniform,
    /// Geometric stretching across the whole axis; `ratio` is the last cell's
    /// size over the first's (`> 1` puts the smallest cells at the axis
    /// start, `< 1` at the end) — the OpenFOAM `simpleGrading` convention.
    OneSided { ratio: f64 },
    /// Symmetric geometric refinement toward both ends (boundary-layer
    /// meshes): smallest cells at the walls, largest at the center; `ratio`
    /// is the center cell's size over the wall cell's. Requires an even cell
    /// count.
    TwoSided { ratio: f64 },
}

/// Vertex coordinates for `n` geometrically stretched cells spanning
/// `[0, len]`, with `ratio` = last cell size / first cell size.
fn geometric_axis_coords(n: usize, len: f64, ratio: f64) -> Vec<f64> {
    assert!(ratio.is_finite() && ratio > 0.0, "grading ratio must be > 0");
    let mut coords = Vec::with_capacity(n + 1);
    coords.push(0.0);
    if n == 1 || (ratio - 1.0).abs() < 1e-12 {
        for i in 1..=n {
            coords.push(len * i as f64 / n as f64);
        }
        return coords;
    }
    // Cell sizes h_i = h0 * r^i with r^(n-1) = ratio.
    let r = ratio.powf(1.0 / (n as f64 - 1.0));
    let h0 = len * (r - 1.0) / (r.powi(n as i32) - 1.0);
    let mut x = 0.0;
    let mut h = h0;
    for _ in 0..n {
        x += h;
        coords.push(x);
        h *= r;
    }
    coords[n] = len;
    coords
}

/// Vertex coordinates (`n + 1` values spanning `[0, len]`) for one axis.
fn graded_axis_coords(n: usize, len: f64, grading: AxisGrading) -> Vec<f64> {
    match grading {
        AxisGrading::Uniform => geometric_axis_coords(n, len, 1.0),
        AxisGrading::OneSided { ratio } => geometric_axis_coords(n, len, ratio),
        AxisGrading::TwoSided { ratio } => {
            assert!(n % 2 == 0, "TwoSided grading requires an even cell count");
            let m = n / 2;
            let half = geometric_axis_coords(m, len * 0.5, ratio);
            let mut coords = vec![0.0; n + 1];
            for i in 0..=m {
                coords[i] = half[i];
                coords[n - i] = len - half[i];
            }
            coords
        }
    }
}

/// Like [`generate_structured_rect_mesh`] but with per-axis geometric
/// grading. `AxisGrading::Uniform` on both axes reproduces the uniform
/// generator's vertex grid exactly.
pub fn generate_graded_rect_mesh(
    nx: usize,
    ny: usize,
    length: f64,
    height: f64,
    grading_x: AxisGrading,
    grading_y: AxisGrading,
    boundaries: BoundarySides,
) -> Mesh {
    assert!(nx > 0, "nx must be > 0");
    assert!(ny > 0, "ny must be > 0");
    assert!(length > 0.0, "length must be > 0");
    assert!(height > 0.0, "height must be > 0");

    let xs = graded_axis_coords(nx, length, grading_x);
    let ys = graded_axis_coords(ny, height, grading_y);

    let num_vertices = (nx + 1) * (ny + 1);
    let mut vx = vec![0.0; num_vertices];
    let mut vy = vec![0.0; num_vertices];
    let vid = |i: usize, j: usize| -> usize { j * (nx + 1) + i };
    for j in 0..=ny {
        for i in 0..=nx {
            let v = vid(i, j);
            vx[v] = xs[i];
            vy[v] = ys[j];
        }
    }

    generate_structured_mesh_from_vertex_grid(nx, ny, vx, vy, |_i, _j| true, boundaries)
}

/// Fully periodic rectangular mesh: every face is INTERIOR (no boundary
/// faces). Seam faces wrap to the opposite edge and record a
/// `face_wrap_shift` = the vector to add to the neighbor-side cell center to
/// bring it into the owner's frame across the seam. The boundary-free
/// instrument that isolates the interior discretization from boundary
/// effects (Arc N′ N3). Cells keep their own corner vertices; the wrap is
/// carried purely by face connectivity + the shift, so cell geometry is
/// identical to the uniform mesh.
pub fn generate_structured_rect_mesh_periodic(
    nx: usize,
    ny: usize,
    length: f64,
    height: f64,
) -> Mesh {
    assert!(nx > 0 && ny > 0, "nx, ny must be > 0");
    assert!(length > 0.0 && height > 0.0, "length, height must be > 0");
    let dx = length / nx as f64;
    let dy = height / ny as f64;

    let mut mesh = Mesh::new();

    let nvx = nx + 1;
    let num_vertices = nvx * (ny + 1);
    mesh.vx = vec![0.0; num_vertices];
    mesh.vy = vec![0.0; num_vertices];
    mesh.v_fixed = vec![false; num_vertices];
    let vid = |i: usize, j: usize| -> usize { j * nvx + i };
    for j in 0..=ny {
        for i in 0..=nx {
            let v = vid(i, j);
            mesh.vx[v] = i as f64 * dx;
            mesh.vy[v] = j as f64 * dy;
        }
    }

    let cell_id = |i: usize, j: usize| -> usize { j * nx + i };
    // Vertical face (i,j): owner cell(i,j), placed at its RIGHT edge.
    let vfid = |i: usize, j: usize| -> usize { j * nx + i };
    // Horizontal face (i,j): owner cell(i,j), placed at its TOP edge.
    let n_vert = nx * ny;
    let hfid = |i: usize, j: usize| -> usize { n_vert + j * nx + i };

    let nfaces = 2 * nx * ny;
    let mut face_v1 = Vec::with_capacity(nfaces);
    let mut face_v2 = Vec::with_capacity(nfaces);
    let mut face_owner = Vec::with_capacity(nfaces);
    let mut face_neighbor: Vec<Option<usize>> = Vec::with_capacity(nfaces);
    let mut face_boundary: Vec<Option<BoundaryType>> = Vec::with_capacity(nfaces);
    let mut face_nx_v = Vec::with_capacity(nfaces);
    let mut face_ny_v = Vec::with_capacity(nfaces);
    let mut face_wrap_shift: Vec<[f64; 2]> = Vec::with_capacity(nfaces);

    // Vertical faces (normal +x, owner -> neighbor wraps at the right edge).
    for j in 0..ny {
        for i in 0..nx {
            face_v1.push(vid(i + 1, j));
            face_v2.push(vid(i + 1, j + 1));
            face_owner.push(cell_id(i, j));
            face_neighbor.push(Some(cell_id((i + 1) % nx, j)));
            face_boundary.push(None);
            face_nx_v.push(1.0);
            face_ny_v.push(0.0);
            face_wrap_shift.push(if i + 1 == nx { [length, 0.0] } else { [0.0, 0.0] });
        }
    }
    // Horizontal faces (normal +y, owner -> neighbor wraps at the top edge).
    for j in 0..ny {
        for i in 0..nx {
            face_v1.push(vid(i, j + 1));
            face_v2.push(vid(i + 1, j + 1));
            face_owner.push(cell_id(i, j));
            face_neighbor.push(Some(cell_id(i, (j + 1) % ny)));
            face_boundary.push(None);
            face_nx_v.push(0.0);
            face_ny_v.push(1.0);
            face_wrap_shift.push(if j + 1 == ny { [0.0, height] } else { [0.0, 0.0] });
        }
    }

    mesh.face_v1 = face_v1;
    mesh.face_v2 = face_v2;
    mesh.face_owner = face_owner;
    mesh.face_neighbor = face_neighbor;
    mesh.face_boundary = face_boundary;
    mesh.face_nx = face_nx_v;
    mesh.face_ny = face_ny_v;
    mesh.face_wrap_shift = face_wrap_shift;
    mesh.face_area = vec![0.0; nfaces];
    mesh.face_cx = vec![0.0; nfaces];
    mesh.face_cy = vec![0.0; nfaces];

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
            // left = owner's-left = vertical face of cell (i-1); right = own;
            // bottom = horizontal face of cell (j-1); top = own.
            mesh.cell_faces.push(vfid((i + nx - 1) % nx, j));
            mesh.cell_faces.push(vfid(i, j));
            mesh.cell_faces.push(hfid(i, (j + ny - 1) % ny));
            mesh.cell_faces.push(hfid(i, j));
            mesh.cell_face_offsets.push(mesh.cell_faces.len());

            mesh.cell_vertices.push(vid(i, j));
            mesh.cell_vertices.push(vid(i + 1, j));
            mesh.cell_vertices.push(vid(i + 1, j + 1));
            mesh.cell_vertices.push(vid(i, j + 1));
            mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
        }
    }

    mesh.recalculate_geometry();
    mesh
}

#[cfg(test)]
mod graded_mesh_tests {
    use super::*;

    fn assert_mesh_invariants(mesh: &Mesh, length: f64, height: f64) {
        let total_vol: f64 = mesh.cell_vol.iter().sum();
        assert!(
            (total_vol - length * height).abs() < 1e-12 * length * height,
            "cell volumes must sum to the domain area: {total_vol} vs {}",
            length * height
        );
        for c in 0..mesh.cell_vol.len() {
            assert!(mesh.cell_vol[c] > 0.0, "cell {c} has non-positive volume");
        }
        // Per-cell divergence theorem on a constant field: outward-signed
        // face areas cancel (interior faces store the owner-outward normal).
        for c in 0..mesh.cell_vol.len() {
            let mut sx = 0.0;
            let mut sy = 0.0;
            for k in mesh.cell_face_offsets[c]..mesh.cell_face_offsets[c + 1] {
                let f = mesh.cell_faces[k];
                let sign = if mesh.face_owner[f] == c { 1.0 } else { -1.0 };
                sx += sign * mesh.face_nx[f] * mesh.face_area[f];
                sy += sign * mesh.face_ny[f] * mesh.face_area[f];
            }
            assert!(
                sx.abs() < 1e-12 && sy.abs() < 1e-12,
                "cell {c} face areas do not close: ({sx}, {sy})"
            );
        }
    }

    #[test]
    fn periodic_mesh_has_no_boundary_and_correct_wrap() {
        let (nx, ny, lx, ly) = (6usize, 4usize, 2.0, 1.5);
        let mesh = generate_structured_rect_mesh_periodic(nx, ny, lx, ly);
        assert_eq!(mesh.num_cells(), nx * ny);
        assert_eq!(mesh.num_faces(), 2 * nx * ny);
        // Boundary-free: every face has a neighbor.
        for f in 0..mesh.num_faces() {
            assert!(mesh.face_neighbor[f].is_some(), "face {f} is a boundary face");
            assert!(mesh.face_boundary[f].is_none());
        }
        for c in 0..mesh.num_cells() {
            assert_eq!(
                mesh.cell_face_offsets[c + 1] - mesh.cell_face_offsets[c],
                4,
                "cell {c} must have 4 faces"
            );
        }
        // Volumes sum + per-cell signed face-area closure (the divergence
        // theorem holds with every cell interior).
        assert_mesh_invariants(&mesh, lx, ly);
        // Exactly ny x-seam + nx y-seam faces carry a domain-length shift.
        let mut seam = 0;
        for f in 0..mesh.num_faces() {
            let [sx, sy] = mesh.face_wrap_shift[f];
            if sx != 0.0 || sy != 0.0 {
                seam += 1;
                assert!(
                    (sx.abs() - lx).abs() < 1e-12 || (sy.abs() - ly).abs() < 1e-12,
                    "seam shift must equal a domain length"
                );
                // The shifted neighbor center is adjacent to the seam face.
                let n = mesh.face_neighbor[f].unwrap();
                let d = ((mesh.cell_cx[n] + sx - mesh.face_cx[f]).powi(2)
                    + (mesh.cell_cy[n] + sy - mesh.face_cy[f]).powi(2))
                .sqrt();
                let hmax = (lx / nx as f64).max(ly / ny as f64);
                assert!(d < 0.5 * hmax + 1e-9, "wrapped neighbor not adjacent: d={d}");
            }
        }
        assert_eq!(seam, ny + nx, "expected ny x-seam + nx y-seam faces");
    }

    #[test]
    fn uniform_grading_matches_uniform_generator() {
        let graded = generate_graded_rect_mesh(
            8,
            6,
            2.0,
            1.5,
            AxisGrading::Uniform,
            AxisGrading::Uniform,
            BoundarySides::wall(),
        );
        let uniform = generate_structured_rect_mesh(8, 6, 2.0, 1.5, BoundarySides::wall());
        assert_eq!(graded.vx, uniform.vx);
        assert_eq!(graded.vy, uniform.vy);
        assert_eq!(graded.cell_vol.len(), uniform.cell_vol.len());
        assert_mesh_invariants(&graded, 2.0, 1.5);
    }

    #[test]
    fn one_sided_grading_hits_requested_ratio() {
        let n = 16;
        let ratio = 4.0;
        let coords = graded_axis_coords(n, 1.0, AxisGrading::OneSided { ratio });
        assert_eq!(coords.len(), n + 1);
        assert_eq!(coords[0], 0.0);
        assert_eq!(coords[n], 1.0);
        let h_first = coords[1] - coords[0];
        let h_last = coords[n] - coords[n - 1];
        assert!(
            (h_last / h_first - ratio).abs() < 1e-9,
            "last/first = {} vs requested {ratio}",
            h_last / h_first
        );
        for w in coords.windows(2) {
            assert!(w[1] > w[0], "coords must be strictly increasing");
        }
    }

    #[test]
    fn two_sided_grading_is_symmetric_wall_refined() {
        let n = 32;
        let ratio = 8.0;
        let coords = graded_axis_coords(n, 1.0, AxisGrading::TwoSided { ratio });
        assert_eq!(coords[0], 0.0);
        assert_eq!(coords[n], 1.0);
        assert_eq!(coords[n / 2], 0.5);
        for i in 0..=n {
            assert!(
                (coords[i] - (1.0 - coords[n - i])).abs() < 1e-15,
                "two-sided coords must be mirror-symmetric at i={i}"
            );
        }
        let h_wall = coords[1] - coords[0];
        let h_center = coords[n / 2] - coords[n / 2 - 1];
        assert!(
            (h_center / h_wall - ratio).abs() < 1e-9,
            "center/wall = {} vs requested {ratio}",
            h_center / h_wall
        );

        let mesh = generate_graded_rect_mesh(
            n,
            n,
            1.0,
            1.0,
            AxisGrading::TwoSided { ratio },
            AxisGrading::TwoSided { ratio },
            BoundarySides::wall(),
        );
        assert_eq!(mesh.cell_vol.len(), n * n);
        assert_mesh_invariants(&mesh, 1.0, 1.0);
    }
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

    generate_structured_mesh_from_vertex_grid(nx, ny, vx, vy, |_i, _j| true, boundaries)
}

/// Smooth converging–diverging channel height profile at normalized position
/// `xi in [0,1]`: `h_in` at the inlet, the minimum `h_throat` at `throat_frac`,
/// then `h_exit` at the outlet. Cosine blends give zero wall slope at the throat
/// (no spurious corner shock). A diverging exit (`h_exit > h_throat`) is what lets
/// the flow keep accelerating past M=1.
pub fn nozzle_height(xi: f64, h_in: f64, h_throat: f64, throat_frac: f64, h_exit: f64) -> f64 {
    use std::f64::consts::PI;
    if xi <= throat_frac {
        let t = if throat_frac > 0.0 { xi / throat_frac } else { 1.0 };
        // cos(0)=1 -> h_in ; cos(PI)=-1 -> h_throat
        h_throat + (h_in - h_throat) * 0.5 * (1.0 + (PI * t).cos())
    } else {
        let t = (xi - throat_frac) / (1.0 - throat_frac).max(1e-12);
        // t=0 -> h_throat ; t=1 -> h_exit
        h_throat + (h_exit - h_throat) * 0.5 * (1.0 - (PI * t).cos())
    }
}

/// Converging–diverging nozzle channel: flat bottom (y=0), shaped top wall at
/// `y = nozzle_height(x)`. The channel narrows from `height` at the inlet to
/// `throat_height` at `throat_frac*length`, then widens to `exit_height` — the
/// classic CD-nozzle geometry for accelerating a subsonic inflow through a sonic
/// throat into supersonic flow in the diverging section.
#[allow(clippy::too_many_arguments)]
pub fn generate_structured_nozzle_mesh(
    nx: usize,
    ny: usize,
    length: f64,
    height: f64,
    throat_height: f64,
    throat_frac: f64,
    exit_height: f64,
    boundaries: BoundarySides,
) -> Mesh {
    assert!(nx > 0 && ny > 0);
    assert!(length > 0.0 && height > 0.0);
    assert!(throat_height > 0.0 && throat_height <= height);
    assert!(exit_height > 0.0 && exit_height <= height);
    assert!(throat_frac > 0.0 && throat_frac < 1.0);

    let num_vertices = (nx + 1) * (ny + 1);
    let mut vx = vec![0.0; num_vertices];
    let mut vy = vec![0.0; num_vertices];
    let vid = |i: usize, j: usize| -> usize { j * (nx + 1) + i };

    for j in 0..=ny {
        let eta = j as f64 / ny as f64;
        for i in 0..=nx {
            let xi = i as f64 / nx as f64;
            let x = xi * length;
            let h = nozzle_height(xi, height, throat_height, throat_frac, exit_height);
            let v = vid(i, j);
            vx[v] = x;
            vy[v] = eta * h;
        }
    }

    generate_structured_mesh_from_vertex_grid(nx, ny, vx, vy, |_i, _j| true, boundaries)
}
