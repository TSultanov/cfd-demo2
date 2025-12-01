use nalgebra::{Point2, Vector2};
use std::collections::HashMap;

use super::geometry::Geometry;
use super::quadtree::generate_base_polygons;
use super::structs::{BoundaryType, Mesh};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    pub v1: usize,
    pub v2: usize,
}

impl Edge {
    pub fn new(v1: usize, v2: usize) -> Self {
        if v1 < v2 {
            Self { v1, v2 }
        } else {
            Self { v1: v2, v2: v1 }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v1: usize,
    pub v2: usize,
    pub v3: usize,
    pub circumcenter: Point2<f64>,
    pub r_sq: f64,
}

impl Triangle {
    pub fn new(
        v1: usize,
        v2: usize,
        v3: usize,
        p1: Point2<f64>,
        p2: Point2<f64>,
        p3: Point2<f64>,
    ) -> Self {
        let (circumcenter, r_sq) = Self::calculate_circumcircle(p1, p2, p3);
        Self {
            v1,
            v2,
            v3,
            circumcenter,
            r_sq,
        }
    }

    pub fn calculate_circumcircle(
        a: Point2<f64>,
        b: Point2<f64>,
        c: Point2<f64>,
    ) -> (Point2<f64>, f64) {
        let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
        if d.abs() < 1e-12 {
            // Collinear or degenerate, return something safe but invalid
            return (Point2::origin(), f64::MAX);
        }
        let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
            + (b.x * b.x + b.y * b.y) * (c.y - a.y)
            + (c.x * c.x + c.y * c.y) * (a.y - b.y))
            / d;
        let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
            + (b.x * b.x + b.y * b.y) * (a.x - c.x)
            + (c.x * c.x + c.y * c.y) * (b.x - a.x))
            / d;
        let center = Point2::new(ux, uy);
        let r_sq = (center - a).norm_squared();
        (center, r_sq)
    }

    pub fn in_circumcircle(&self, p: Point2<f64>) -> bool {
        // Use determinant method for robustness
        // | ax ay ax^2+ay^2 1 |
        // | bx by ...       1 |
        // | cx cy ...       1 |
        // | px py ...       1 |
        // > 0 means inside (assuming ccw)

        // We don't have the vertices stored in Triangle, only indices and circumcenter.
        // But we need vertices for robust check.
        // The current architecture stores points in the main function, not in Triangle.
        // So we cannot implement the determinant method here without passing points.

        // Fallback: Use the existing method but with a tolerance?
        // Or better: Pass points to this function?
        // Changing signature requires changing call sites.

        let dist_sq = (p - self.circumcenter).norm_squared();
        dist_sq <= self.r_sq * (1.0 + 1e-9) // Add epsilon tolerance
    }
}

pub fn triangulate(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> (Vec<Point2<f64>>, Vec<Triangle>, Vec<bool>) {
    // 1. Generate Points using robust polygon generation
    let all_polys =
        generate_base_polygons(geo, min_cell_size, max_cell_size, growth_rate, domain_size);

    let mut points = Vec::new();
    let mut fixed_nodes = Vec::new();
    let mut unique_map = HashMap::new();
    let quantize = |x: f64, y: f64| ((x * 100000.0).round() as i64, (y * 100000.0).round() as i64);

    for poly in all_polys {
        for (p, fixed) in poly {
            let key = quantize(p.x, p.y);
            if !unique_map.contains_key(&key) {
                unique_map.insert(key, points.len());
                points.push(p);
                fixed_nodes.push(fixed);
            } else if fixed {
                // If we encounter the point again and it's fixed, ensure it's marked fixed
                let idx = unique_map[&key];
                fixed_nodes[idx] = true;
            }
        }
    }

    // Jitter internal points to improve mesh quality (avoid right-angled triangles from grid)
    // and break cocircularity degeneracies.
    let jitter_scale = 0.25 * min_cell_size;
    for (i, p) in points.iter_mut().enumerate() {
        if !fixed_nodes[i] {
            // Simple pseudo-random based on position
            let s = (p.x * 12.9898 + p.y * 78.233).sin() * 43758.5453;
            let noise_x = s.fract();
            let s2 = (p.x * 39.7867 + p.y * 27.123).sin() * 23412.1234;
            let noise_y = s2.fract();

            let dx = (noise_x - 0.5) * jitter_scale;
            let dy = (noise_y - 0.5) * jitter_scale;

            let p_new = Point2::new(p.x + dx, p.y + dy);

            // Only apply if still inside geometry
            if geo.is_inside(&p_new) {
                *p = p_new;
            }
        }
    }

    // 2. Delaunay Triangulation (Bowyer-Watson)
    let mut triangles: Vec<Triangle> = Vec::new();

    // Super-triangle
    // Use non-integer coordinates to avoid co-circularity with grid points
    let margin = 1.2345;
    let p1 = Point2::new(-margin, -margin);
    let p2 = Point2::new(2.0 * domain_size.x + margin, -margin);
    let p3 = Point2::new(-margin, 2.0 * domain_size.y + margin);

    // Add super-triangle vertices to points list (temporarily)
    let n_points = points.len();
    points.push(p1);
    points.push(p2);
    points.push(p3);

    triangles.push(Triangle::new(
        n_points,
        n_points + 1,
        n_points + 2,
        p1,
        p2,
        p3,
    ));

    for (i, &p) in points.iter().enumerate().take(n_points) {
        let mut bad_triangles = Vec::new();
        for (t_idx, t) in triangles.iter().enumerate() {
            if t.in_circumcircle(p) {
                bad_triangles.push(t_idx);
            }
        }

        let mut polygon = Vec::new();
        for &t_idx in &bad_triangles {
            let t = triangles[t_idx];
            let edges = [
                Edge::new(t.v1, t.v2),
                Edge::new(t.v2, t.v3),
                Edge::new(t.v3, t.v1),
            ];

            for &edge in &edges {
                let mut shared = false;
                for &other_t_idx in &bad_triangles {
                    if t_idx == other_t_idx {
                        continue;
                    }
                    let other_t = triangles[other_t_idx];
                    let other_edges = [
                        Edge::new(other_t.v1, other_t.v2),
                        Edge::new(other_t.v2, other_t.v3),
                        Edge::new(other_t.v3, other_t.v1),
                    ];
                    if other_edges.contains(&edge) {
                        shared = true;
                        break;
                    }
                }
                if !shared {
                    polygon.push(edge);
                }
            }
        }

        // Remove bad triangles
        // Sort indices descending to remove efficiently
        bad_triangles.sort_unstable_by(|a, b| b.cmp(a));

        for idx in bad_triangles {
            triangles.swap_remove(idx);
        }

        // Re-triangulate
        for edge in polygon {
            triangles.push(Triangle::new(
                edge.v1,
                edge.v2,
                i,
                points[edge.v1],
                points[edge.v2],
                points[i],
            ));
        }
    }

    // Remove triangles connected to super-triangle
    triangles.retain(|t| t.v1 < n_points && t.v2 < n_points && t.v3 < n_points);
    // Remove super-triangle vertices
    points.truncate(n_points);

    // Filter triangles that are outside the geometry (e.g. inside obstacles)
    triangles.retain(|t| {
        // If the triangle has any internal vertex, keep it.
        // Internal vertices are generated inside the fluid, so triangles connecting them are valid.
        // Only triangles formed purely by boundary vertices (chords) need to be checked via centroid.
        if !fixed_nodes[t.v1] || !fixed_nodes[t.v2] || !fixed_nodes[t.v3] {
            return true;
        }

        let p1 = points[t.v1];
        let p2 = points[t.v2];
        let p3 = points[t.v3];
        let centroid = Point2::new((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0);
        geo.is_inside(&centroid)
    });

    (points, triangles, fixed_nodes)
}

pub fn generate_delaunay_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Mesh {
    let mut mesh = Mesh::new();
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    let (points, triangles, fixed_nodes) =
        triangulate(geo, min_cell_size, max_cell_size, growth_rate, domain_size);

    // 3. Convert to Mesh
    mesh.vx = points.iter().map(|p| p.x).collect();
    mesh.vy = points.iter().map(|p| p.y).collect();
    mesh.v_fixed = fixed_nodes;

    // Build faces and cells
    // Map edge -> face_index
    let mut edge_face_map: HashMap<Edge, usize> = HashMap::new();

    for t in triangles {
        let cell_idx = mesh.cell_cx.len();

        // Calculate centroid and area
        let p1 = points[t.v1];
        let p2 = points[t.v2];
        let p3 = points[t.v3];

        let center = Point2::new((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0);
        let area = 0.5 * ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)).abs();

        mesh.cell_cx.push(center.x);
        mesh.cell_cy.push(center.y);
        mesh.cell_vol.push(area);

        let edges = [
            Edge::new(t.v1, t.v2),
            Edge::new(t.v2, t.v3),
            Edge::new(t.v3, t.v1),
        ];

        for edge in edges {
            if let Some(&face_idx) = edge_face_map.get(&edge) {
                mesh.face_neighbor[face_idx] = Some(cell_idx);
                mesh.face_boundary[face_idx] = None; // Internal
                mesh.cell_faces.push(face_idx);
            } else {
                let face_idx = mesh.face_cx.len();
                let p_a = points[edge.v1];
                let p_b = points[edge.v2];
                let f_center = Point2::new((p_a.x + p_b.x) / 2.0, (p_a.y + p_b.y) / 2.0);
                let f_len = (p_a - p_b).norm();
                let normal = Vector2::new(p_b.y - p_a.y, p_a.x - p_b.x).normalize();

                mesh.face_v1.push(edge.v1);
                mesh.face_v2.push(edge.v2);
                mesh.face_owner.push(cell_idx);
                mesh.face_neighbor.push(None);

                // Determine boundary
                // If edge vertices are both fixed, it MIGHT be a boundary face.
                // But internal edges can also connect fixed vertices (e.g. corner to corner).
                // Better check: is the face center on boundary?
                let is_boundary = mesh.v_fixed[edge.v1] && mesh.v_fixed[edge.v2];
                let boundary_type = if is_boundary {
                    // Check position
                    if f_center.x < 1e-6 {
                        Some(BoundaryType::Inlet)
                    } else if (f_center.x - domain_size.x).abs() < 1e-6 {
                        Some(BoundaryType::Outlet)
                    } else {
                        Some(BoundaryType::Wall)
                    }
                } else {
                    None
                };

                mesh.face_boundary.push(boundary_type);
                mesh.face_nx.push(normal.x);
                mesh.face_ny.push(normal.y);
                mesh.face_area.push(f_len);
                mesh.face_cx.push(f_center.x);
                mesh.face_cy.push(f_center.y);

                edge_face_map.insert(edge, face_idx);
                mesh.cell_faces.push(face_idx);
            }
        }

        mesh.cell_face_offsets.push(mesh.cell_faces.len());

        // Vertices
        mesh.cell_vertices.push(t.v1);
        mesh.cell_vertices.push(t.v2);
        mesh.cell_vertices.push(t.v3);
        mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
    }

    // Fix normals (must point from owner to neighbor)
    // And handle single-neighbor faces (boundary)
    for i in 0..mesh.face_cx.len() {
        let owner = mesh.face_owner[i];
        let c_owner = Point2::new(mesh.cell_cx[owner], mesh.cell_cy[owner]);
        let f_center = Point2::new(mesh.face_cx[i], mesh.face_cy[i]);
        let normal = Vector2::new(mesh.face_nx[i], mesh.face_ny[i]);

        if (f_center - c_owner).dot(&normal) < 0.0 {
            mesh.face_nx[i] = -normal.x;
            mesh.face_ny[i] = -normal.y;
        }
    }

    mesh
}
