mod cut_cell;
mod delaunay;
mod geometry;
pub(crate) mod mesh_builder;
mod meshgen_ext;
mod meshgen_utils;
pub mod meshless;
mod quadtree;
pub(crate) mod tolerances;
mod voronoi;

pub use cut_cell::generate_cut_cell_mesh;
// `triangulate` is re-exported as the seed seam for the meshless-equivalence
// gates: it yields the incumbent Voronoi generator's exact post-smoothing
// point set (deterministic, fixed RNG), which tests feed to both pipelines.
pub use delaunay::{generate_delaunay_mesh, triangulate, Edge, Triangle};
pub use geometry::{BackwardsStep, ChannelWithObstacle, Geometry, Nozzle, RectangularChannel};
pub use mesh_builder::{CellId, FaceId, MeshBuilder, VertexId};
pub use meshless::{generate_cvt_mesh, generate_meshless_voronoi_mesh, LloydConfig};
pub use tolerances::MeshgenTolerances;
pub use voronoi::generate_voronoi_mesh;

#[cfg(test)]
mod tests;
