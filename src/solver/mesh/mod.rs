pub mod structs;
pub mod structured;

pub use structs::{BoundaryType, Mesh};
pub use structured::{
    generate_structured_backwards_step_mesh, generate_structured_rect_mesh,
    generate_structured_trapezoid_mesh, BoundarySides,
};

#[cfg(feature = "meshgen")]
pub use crate::meshgen::{
    generate_cut_cell_mesh, generate_delaunay_mesh, generate_voronoi_mesh, BackwardsStep,
    ChannelWithObstacle, Edge, Geometry, RectangularChannel, Triangle,
};
