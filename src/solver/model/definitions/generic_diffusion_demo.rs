use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    fvm, vol_scalar, Coefficient, EquationSystem,
};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::gpu_spec::ModelGpuSpec;
use crate::solver::units::si;

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

pub fn generic_diffusion_demo_model() -> ModelSpec {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let kappa = Coefficient::constant_unit(1.0, si::AREA / si::TIME);
    let eqn = (fvm::ddt(phi) + fvm::laplacian(kappa, phi)).eqn(phi);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

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
    ModelSpec {
        id: "generic_diffusion_demo",
        method: crate::solver::model::method::MethodSpec::GenericCoupled,
        eos: crate::solver::model::eos::EosSpec::Constant,
        system,
        state_layout: layout,
        boundaries,

        extra_kernels: Vec::new(),
        linear_solver: None,
        flux_module: None,
        primitives: crate::solver::model::primitives::PrimitiveDerivations::default(),
        generated_kernels: Vec::new(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}

pub fn generic_diffusion_demo_neumann_model() -> ModelSpec {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let kappa = Coefficient::constant_unit(1.0, si::AREA / si::TIME);
    let eqn = (fvm::ddt(phi) + fvm::laplacian(kappa, phi)).eqn(phi);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

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
    ModelSpec {
        id: "generic_diffusion_demo_neumann",
        method: crate::solver::model::method::MethodSpec::GenericCoupled,
        eos: crate::solver::model::eos::EosSpec::Constant,
        system,
        state_layout: layout,
        boundaries,

        extra_kernels: Vec::new(),
        linear_solver: None,
        flux_module: None,
        primitives: crate::solver::model::primitives::PrimitiveDerivations::default(),
        generated_kernels: Vec::new(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}
