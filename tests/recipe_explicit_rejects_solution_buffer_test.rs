use cfd2::solver::gpu::recipe::SolverRecipe;
use cfd2::solver::model::compressible_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SteppingMode, TimeScheme};

#[test]
fn explicit_recipes_reject_models_that_bind_solution_x() {
    let model = compressible_model();
    let err = SolverRecipe::from_model(
        &model,
        Scheme::Upwind,
        TimeScheme::Euler,
        PreconditionerType::Jacobi,
        SteppingMode::Explicit,
    )
    .expect_err("expected explicit stepping to be rejected for models that bind x");

    assert!(
        err.contains("SteppingMode::Explicit"),
        "unexpected error: {err}"
    );
    assert!(
        err.contains("solution buffer 'x'"),
        "unexpected error: {err}"
    );
}
