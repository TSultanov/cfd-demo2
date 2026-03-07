use crate::solver::mesh::BoundaryType;
use nalgebra::Vector2;

/// Tolerances for mesh generation, scaled to the problem geometry.
///
/// All tolerances are derived from two inputs:
/// - `min_cell_size`: the smallest cell dimension in the mesh
/// - `domain_size`: the bounding box of the computational domain
#[derive(Debug, Clone)]
pub struct MeshgenTolerances {
    /// Quantization inverse: `1 / vertex_merge_distance`.
    /// Used to map continuous coordinates to integer grid keys.
    pub quantize_inv: f64,

    /// Degeneracy guard for determinants and denominators.
    /// `min_cell_size^2 * 1e-12`.
    pub determinant_eps: f64,

    /// Parametric tolerance for line-segment intersection.
    /// Dimensionless: a point within `t_eps` of an endpoint (t=0 or t=1)
    /// is treated as coincident with the endpoint.
    pub t_eps: f64,

    /// Geometric proximity tolerance for "is this point on the boundary?"
    /// `domain_bbox_diag * 1e-6`.
    pub boundary_eps: f64,

    /// Finite difference step for SDF gradient computation.
    /// `min_cell_size * 1e-4`.
    pub sdf_grad_eps: f64,

    /// Edge collapse guard for smoothing: edges shorter than this squared
    /// distance trigger move rejection. `(min_cell_size * 1e-3)^2`.
    pub edge_collapse_sq: f64,

    /// Circumcircle / in-circle predicate tolerance.
    /// `min_cell_size^2 * 1e-10`.
    pub circumcircle_eps: f64,

    /// Cross product tolerance for convexity / orientation checks.
    /// `min_cell_size^2 * 1e-10`.
    pub cross_eps: f64,

    /// Cell area degeneracy threshold.
    /// `min_cell_size^2 * 1e-6`.
    pub area_eps: f64,

    /// Edge length degeneracy threshold.
    /// `min_cell_size * 1e-6`.
    pub edge_len_eps: f64,

    /// SDF root-finding convergence tolerance.
    /// `min_cell_size * 1e-9`.
    pub sdf_root_eps: f64,

    /// Squared distance threshold for "is this the same point?"
    /// Used for SIMD and scalar endpoint/projection checks.
    /// `(min_cell_size * 1e-4)^2`.
    pub point_coincidence_sq: f64,

    /// Segment degeneracy: squared length below which a segment is skipped.
    /// `(min_cell_size * 1e-5)^2`.
    pub segment_degenerate_sq: f64,

    /// Smoothing weight denominator clamp (avoids division by zero in
    /// radius-weighted Laplacian smoothing). `min_cell_size * 1e-6`.
    pub smoothing_denom_eps: f64,

    /// Denominator threshold for line-intersection (meshgen_utils).
    /// Same as `determinant_eps` but 1D. `min_cell_size * 1e-6`.
    pub line_det_eps: f64,

    /// Normal degeneracy threshold for skewness computation.
    /// `min_cell_size^2 * 1e-12`.
    pub normal_degenerate_sq: f64,
}

impl MeshgenTolerances {
    pub fn from_geometry(min_cell_size: f64, domain_size: Vector2<f64>) -> Self {
        let diag = (domain_size.x * domain_size.x + domain_size.y * domain_size.y).sqrt();
        let mcs = min_cell_size;
        let vertex_merge = mcs * 1e-6;
        Self {
            quantize_inv: 1.0 / vertex_merge,
            determinant_eps: mcs * mcs * 1e-12,
            t_eps: 1e-6,
            boundary_eps: diag * 1e-6,
            sdf_grad_eps: mcs * 1e-4,
            edge_collapse_sq: (mcs * 1e-3) * (mcs * 1e-3),
            circumcircle_eps: mcs * mcs * 1e-10,
            cross_eps: mcs * mcs * 1e-10,
            area_eps: mcs * mcs * 1e-6,
            edge_len_eps: mcs * 1e-6,
            sdf_root_eps: mcs * 1e-9,
            point_coincidence_sq: (mcs * 1e-4) * (mcs * 1e-4),
            segment_degenerate_sq: (mcs * 1e-5) * (mcs * 1e-5),
            smoothing_denom_eps: mcs * 1e-6,
            line_det_eps: mcs * 1e-6,
            normal_degenerate_sq: mcs * mcs * 1e-12,
        }
    }

    /// Quantize a coordinate to an integer grid key for vertex deduplication.
    #[inline]
    pub fn quantize(&self, v: f64) -> i64 {
        (v * self.quantize_inv).round() as i64
    }

    /// Quantize a 2D point to a grid key.
    #[inline]
    pub fn quantize_point(&self, x: f64, y: f64) -> (i64, i64) {
        (self.quantize(x), self.quantize(y))
    }

    /// Determine boundary type from face center position.
    /// Assumes rectangular domain `[0, domain_x] × [0, domain_y]`.
    pub fn classify_boundary(
        &self,
        face_center_x: f64,
        face_center_y: f64,
        domain_x: f64,
        domain_y: f64,
    ) -> Option<BoundaryType> {
        if face_center_x < self.boundary_eps {
            Some(BoundaryType::Inlet)
        } else if (face_center_x - domain_x).abs() < self.boundary_eps {
            Some(BoundaryType::Outlet)
        } else if face_center_y < self.boundary_eps || (face_center_y - domain_y).abs() < self.boundary_eps {
            Some(BoundaryType::Wall)
        } else {
            None
        }
    }
}
