use nalgebra::{Point2, Vector2};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryType {
    Inlet,
    Outlet,
    Wall,
    SlipWall,
    MovingWall,
}

#[derive(Default, Clone)]
pub struct Mesh {
    // Vertices
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub v_fixed: Vec<bool>,

    // Faces
    pub face_v1: Vec<usize>,
    pub face_v2: Vec<usize>,
    pub face_owner: Vec<usize>,
    pub face_neighbor: Vec<Option<usize>>,
    pub face_boundary: Vec<Option<BoundaryType>>,
    pub face_nx: Vec<f64>,
    pub face_ny: Vec<f64>,
    pub face_area: Vec<f64>,
    pub face_cx: Vec<f64>,
    pub face_cy: Vec<f64>,

    // Cells
    pub cell_cx: Vec<f64>,
    pub cell_cy: Vec<f64>,
    pub cell_vol: Vec<f64>,

    // Connectivity
    pub cell_faces: Vec<usize>,
    pub cell_face_offsets: Vec<usize>, // cell_face_offsets[i] .. cell_face_offsets[i+1]

    pub cell_vertices: Vec<usize>,
    pub cell_vertex_offsets: Vec<usize>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_cells(&self) -> usize {
        self.cell_cx.len()
    }

    pub fn num_faces(&self) -> usize {
        self.face_cx.len()
    }

    pub fn num_vertices(&self) -> usize {
        self.vx.len()
    }

    pub fn recalculate_geometry(&mut self) {
        // 1) Faces
        let vx = &self.vx;
        let vy = &self.vy;
        let face_v1 = &self.face_v1;
        let face_v2 = &self.face_v2;

        let face_cx = &mut self.face_cx;
        let face_cy = &mut self.face_cy;
        let face_area = &mut self.face_area;
        let face_nx = &mut self.face_nx;
        let face_ny = &mut self.face_ny;

        for i in 0..face_cx.len() {
            let v0_idx = face_v1[i];
            let v1_idx = face_v2[i];

            let v0 = Point2::new(vx[v0_idx], vy[v0_idx]);
            let v1 = Point2::new(vx[v1_idx], vy[v1_idx]);

            let center = Point2::from((v0.coords + v1.coords) * 0.5);
            face_cx[i] = center.x;
            face_cy[i] = center.y;

            let edge_vec = v1 - v0;
            face_area[i] = edge_vec.norm();

            // Preserve normal orientation.
            let tangent = edge_vec.normalize();
            let mut normal = Vector2::new(tangent.y, -tangent.x);

            let current_normal = Vector2::new(face_nx[i], face_ny[i]);
            if normal.dot(&current_normal) < 0.0 {
                normal = -normal;
            }
            face_nx[i] = normal.x;
            face_ny[i] = normal.y;
        }

        // 2) Cells
        let cell_vertex_offsets = &self.cell_vertex_offsets;
        let cell_vertices = &self.cell_vertices;

        for i in 0..self.cell_cx.len() {
            let start = cell_vertex_offsets[i];
            let end = cell_vertex_offsets[i + 1];
            let n = end - start;

            // Polygon area and centroid.
            let mut signed_area = 0.0;
            let mut c_x = 0.0;
            let mut c_y = 0.0;

            for k in 0..n {
                let idx0 = cell_vertices[start + k];
                let idx1 = cell_vertices[start + (k + 1) % n];

                let p0_x = vx[idx0];
                let p0_y = vy[idx0];
                let p1_x = vx[idx1];
                let p1_y = vy[idx1];

                let cross = p0_x * p1_y - p1_x * p0_y;
                signed_area += cross;
                c_x += (p0_x + p1_x) * cross;
                c_y += (p0_y + p1_y) * cross;
            }

            signed_area *= 0.5;
            let area = signed_area.abs();

            let center = if area > 1e-12 {
                Vector2::new(c_x / (6.0 * signed_area), c_y / (6.0 * signed_area))
            } else {
                // Fallback to average.
                let mut avg = Vector2::zeros();
                for k in 0..n {
                    let idx = cell_vertices[start + k];
                    avg.x += vx[idx];
                    avg.y += vy[idx];
                }
                avg / n as f64
            };

            self.cell_cx[i] = center.x;
            self.cell_cy[i] = center.y;
            self.cell_vol[i] = area;
        }
    }
}
