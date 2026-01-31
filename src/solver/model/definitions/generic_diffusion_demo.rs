use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{vol_scalar, EquationSystem};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::backend::typed_ast::{typed_fvm, Scalar, TypedCoeff, TypedFieldRef};
use crate::solver::units::si;
use cfd2_ir::solver::dimensions::{
    Dimensionless, DivDim, Area, Time, Volume,
};

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

/// Type alias for the integrated unit of the diffusion equation:
/// ddt(phi) has unit: Volume * Dimensionless / Time = Volume / Time
/// laplacian(kappa, phi) has unit: (Area/Time) * Dimensionless * Area / Length = Volume / Time
pub type DiffusionIntegratedUnit = DivDim<Volume, Time>;

pub fn generic_diffusion_demo_model() -> ModelSpec {
    // Build typed field reference for phi (dimensionless scalar)
    let phi_typed = TypedFieldRef::<Dimensionless, Scalar>::new("phi");

    // Build coefficient kappa with unit Area/Time
    let kappa_typed: TypedCoeff<DivDim<Area, Time>> = TypedCoeff::constant(1.0);

    // Build equation terms using typed FVM constructors
    // ddt(phi): integrated unit is Volume/Time (Dimensionless * Volume / Time)
    let ddt_term = typed_fvm::ddt(phi_typed);

    // laplacian(kappa, phi): integrated unit is (Area/Time) * Dimensionless * Area / Length
    // = Area^2 / (Time * Length) = Volume / Time (since Area = Length^2, Volume = Length^3)
    let laplacian_term = typed_fvm::laplacian(kappa_typed, phi_typed);

    // Need to cast terms to a common dimension type for addition
    // The actual integrated unit for both is Volume/Time
    let ddt_cast = ddt_term.cast_to::<DiffusionIntegratedUnit>();
    let laplacian_cast = laplacian_term.cast_to::<DiffusionIntegratedUnit>();

    // Now we can add them (both have the same type after casting)
    let eqn = (ddt_cast + laplacian_cast).eqn(phi_typed);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

    // Validate units to ensure the system is consistent (debug builds only)
    #[cfg(debug_assertions)]
    {
        system
            .validate_units()
            .expect("generic diffusion demo system failed unit validation");
    }

    // Use untyped field for StateLayout/boundaries (unchanged behavior)
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let layout = StateLayout::new(vec![phi]);
    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "phi",
        FieldBoundarySpec::new()
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::DIMENSIONLESS),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::DIMENSIONLESS),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::DIMENSIONLESS / si::LENGTH),
            ),
    );
    let model = ModelSpec {
        id: "generic_diffusion_demo",
        system,
        state_layout: layout,
        boundaries,

        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            crate::solver::model::modules::generic_coupled::generic_coupled_module(
                crate::solver::model::method::MethodSpec::Coupled(
                    crate::solver::model::method::CoupledCapabilities::default(),
                ),
            ),
        ],
        linear_solver: None,
        primitives: crate::solver::model::primitives::PrimitiveDerivations::default(),
    };

    model
}

pub fn generic_diffusion_demo_neumann_model() -> ModelSpec {
    // Build typed field reference for phi (dimensionless scalar)
    let phi_typed = TypedFieldRef::<Dimensionless, Scalar>::new("phi");

    // Build coefficient kappa with unit Area/Time
    let kappa_typed: TypedCoeff<DivDim<Area, Time>> = TypedCoeff::constant(1.0);

    // Build equation terms using typed FVM constructors
    let ddt_term = typed_fvm::ddt(phi_typed);
    let laplacian_term = typed_fvm::laplacian(kappa_typed, phi_typed);

    // Cast to common dimension type for addition
    let ddt_cast = ddt_term.cast_to::<DiffusionIntegratedUnit>();
    let laplacian_cast = laplacian_term.cast_to::<DiffusionIntegratedUnit>();

    // Build equation
    let eqn = (ddt_cast + laplacian_cast).eqn(phi_typed);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

    // Validate units to ensure the system is consistent (debug builds only)
    #[cfg(debug_assertions)]
    {
        system
            .validate_units()
            .expect("generic diffusion demo neumann system failed unit validation");
    }

    // Use untyped field for StateLayout/boundaries (unchanged behavior)
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let layout = StateLayout::new(vec![phi]);
    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "phi",
        FieldBoundarySpec::new()
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::neumann(0.0, si::DIMENSIONLESS / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::neumann(0.0, si::DIMENSIONLESS / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::DIMENSIONLESS / si::LENGTH),
            ),
    );
    let model = ModelSpec {
        id: "generic_diffusion_demo_neumann",
        system,
        state_layout: layout,
        boundaries,

        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            crate::solver::model::modules::generic_coupled::generic_coupled_module(
                crate::solver::model::method::MethodSpec::Coupled(
                    crate::solver::model::method::CoupledCapabilities::default(),
                ),
            ),
        ],
        linear_solver: None,
        primitives: crate::solver::model::primitives::PrimitiveDerivations::default(),
    };

    model
}
