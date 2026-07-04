pub mod ale;
pub mod csr;
pub mod ordering;
pub mod refresh;
pub mod structs;
pub mod structured;

pub use ale::{swept_mesh_fluxes_closed, SweptMeshFluxes};
pub use csr::{build_diag_first_scalar_csr, build_sorted_scalar_csr, ScalarCsr};
pub use refresh::MeshRefreshLevel;
pub use structs::{BoundaryType, Mesh};
pub use structured::{
    generate_graded_rect_mesh, generate_structured_backwards_step_mesh,
    generate_structured_nozzle_mesh, generate_structured_rect_mesh,
    generate_structured_rect_mesh_periodic, generate_structured_trapezoid_mesh, AxisGrading,
    BoundarySides,
};

#[cfg(feature = "meshgen")]
pub use crate::meshgen::{
    generate_cut_cell_mesh, generate_cvt_mesh, generate_delaunay_mesh,
    generate_meshless_voronoi_mesh, generate_voronoi_mesh, BackwardsStep, ChannelWithObstacle,
    Edge, Geometry, LloydConfig, Nozzle, RectangularChannel, Triangle,
};
