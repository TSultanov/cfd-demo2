use cfd2::solver::gpu::recipe::SolverRecipe;
use cfd2::solver::model::compressible_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SteppingMode, TimeScheme};

/// Fully explicit stepping is only meaningful for the matrix-free RK4 path.
/// Requesting `SteppingMode::Explicit` with a non-RK4 time scheme (here Euler)
/// must be rejected up front — before any kernel schedule is built — rather than
/// silently falling through to an implicit assembly that binds the solution
/// buffer `x`. (The `binds_solution_x` guard in `from_model` remains as
/// defence-in-depth for a model that passed the RK4 capability gate yet still
/// scheduled an `x`-binding kernel; no shipped model reaches it.)
#[test]
fn explicit_stepping_requires_rk4() {
    let model = compressible_model().expect("model");
    let err = SolverRecipe::from_model(
        &model,
        Scheme::Upwind,
        TimeScheme::Euler,
        PreconditionerType::Jacobi,
        SteppingMode::Explicit,
    )
    .expect_err("expected explicit stepping with a non-RK4 time scheme to be rejected");

    assert!(
        err.contains("RK4"),
        "unexpected error (expected the explicit-requires-RK4 rejection): {err}"
    );
}
