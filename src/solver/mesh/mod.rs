pub mod structs;
pub mod structured;

#[cfg(feature = "meshgen")]
pub mod cut_cell;
#[cfg(feature = "meshgen")]
pub mod delaunay;
#[cfg(feature = "meshgen")]
pub mod geometry;
#[cfg(feature = "meshgen")]
pub mod quadtree;
#[cfg(feature = "meshgen")]
mod meshgen_utils;
#[cfg(feature = "meshgen")]
pub mod voronoi;

#[cfg(all(test, feature = "meshgen"))]
mod tests;

pub use structs::{BoundaryType, Mesh};
pub use structured::{
    generate_structured_backwards_step_mesh, generate_structured_rect_mesh,
    generate_structured_trapezoid_mesh,
};

#[cfg(feature = "meshgen")]
pub use cut_cell::generate_cut_cell_mesh;
#[cfg(feature = "meshgen")]
pub use delaunay::{generate_delaunay_mesh, Edge, Triangle};
#[cfg(feature = "meshgen")]
pub use geometry::{BackwardsStep, ChannelWithObstacle, Geometry};
#[cfg(feature = "meshgen")]
pub use voronoi::generate_voronoi_mesh;
