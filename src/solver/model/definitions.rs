use super::backend::ast::{
    fvc, fvm, surface_scalar, surface_vector, vol_scalar, vol_vector, Coefficient, EquationSystem,
    FieldRef, FluxRef,
};
use super::backend::state_layout::StateLayout;
use super::kernel::{KernelKind, KernelPlan};
use crate::solver::units::si;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub system: EquationSystem,
    pub state_layout: StateLayout,
    pub fields: ModelFields,
}

#[derive(Debug, Clone)]
pub enum ModelFields {
    Incompressible(IncompressibleMomentumFields),
    Compressible(CompressibleFields),
    GenericCoupled(GenericCoupledFields),
}

impl ModelSpec {
    pub fn kernel_plan(&self) -> KernelPlan {
        match self.fields {
            ModelFields::Incompressible(_) => KernelPlan::new(vec![
                KernelKind::PrepareCoupled,
                KernelKind::CoupledAssembly,
                KernelKind::PressureAssembly,
                KernelKind::UpdateFieldsFromCoupled,
                KernelKind::FluxRhieChow,
            ]),
            ModelFields::Compressible(_) => KernelPlan::new(vec![
                KernelKind::CompressibleFluxKt,
                KernelKind::CompressibleGradients,
                KernelKind::CompressibleAssembly,
                KernelKind::CompressibleApply,
                KernelKind::CompressibleUpdate,
            ]),
            ModelFields::GenericCoupled(_) => KernelPlan::new(vec![
                KernelKind::GenericCoupledAssembly,
                KernelKind::GenericCoupledApply,
                KernelKind::GenericCoupledUpdate,
            ]),
        }
    }
}

impl ModelFields {
    pub fn incompressible(&self) -> Option<&IncompressibleMomentumFields> {
        match self {
            ModelFields::Incompressible(fields) => Some(fields),
            _ => None,
        }
    }

    pub fn compressible(&self) -> Option<&CompressibleFields> {
        match self {
            ModelFields::Compressible(fields) => Some(fields),
            _ => None,
        }
    }

    pub fn generic_coupled(&self) -> Option<&GenericCoupledFields> {
        match self {
            ModelFields::GenericCoupled(fields) => Some(fields),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenericCoupledFields {
    pub state: Vec<FieldRef>,
}

impl GenericCoupledFields {
    pub fn new(state: Vec<FieldRef>) -> Self {
        Self { state }
    }
}

#[derive(Debug, Clone)]
pub struct IncompressibleMomentumFields {
    pub u: FieldRef,
    pub p: FieldRef,
    pub phi: FluxRef,
    pub mu: FieldRef,
    pub rho: FieldRef,
    pub d_p: FieldRef,
    pub grad_p: FieldRef,
    pub grad_component: FieldRef,
}

impl IncompressibleMomentumFields {
    pub fn new() -> Self {
        Self {
            u: vol_vector("U", si::VELOCITY),
            p: vol_scalar("p", si::PRESSURE),
            phi: surface_scalar("phi", si::MASS_FLUX),
            mu: vol_scalar("mu", si::DYNAMIC_VISCOSITY),
            rho: vol_scalar("rho", si::DENSITY),
            d_p: vol_scalar("d_p", si::D_P),
            grad_p: vol_vector("grad_p", si::PRESSURE_GRADIENT),
            grad_component: vol_vector("grad_component", si::INV_TIME),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressibleFields {
    pub rho: FieldRef,
    pub rho_u: FieldRef,
    pub rho_e: FieldRef,
    pub p: FieldRef,
    pub u: FieldRef,
    pub mu: FieldRef,
    pub phi_rho: FluxRef,
    pub phi_rho_u: FluxRef,
    pub phi_rho_e: FluxRef,
}

impl CompressibleFields {
    pub fn new() -> Self {
        Self {
            rho: vol_scalar("rho", si::DENSITY),
            rho_u: vol_vector("rho_u", si::MOMENTUM_DENSITY),
            rho_e: vol_scalar("rho_e", si::ENERGY_DENSITY),
            p: vol_scalar("p", si::PRESSURE),
            u: vol_vector("u", si::VELOCITY),
            mu: vol_scalar("mu", si::DYNAMIC_VISCOSITY),
            phi_rho: surface_scalar("phi_rho", si::MASS_FLUX),
            phi_rho_u: surface_vector("phi_rho_u", si::FORCE),
            phi_rho_e: surface_scalar("phi_rho_e", si::POWER),
        }
    }
}

fn build_incompressible_momentum_system(fields: &IncompressibleMomentumFields) -> EquationSystem {
    let momentum = (fvm::ddt_coeff(
        Coefficient::field(fields.rho).expect("rho must be scalar"),
        fields.u,
    )
        + fvm::div(fields.phi, fields.u)
        + fvm::laplacian(
            Coefficient::field(fields.mu).expect("mu must be scalar"),
            fields.u,
        )
        + fvc::grad(fields.p))
    .eqn(fields.u);

    let pressure = fvm::laplacian(
        Coefficient::product(
            Coefficient::field(fields.rho).expect("rho must be scalar"),
            Coefficient::field(fields.d_p).expect("d_p must be scalar"),
        )
        .expect("pressure coefficient must be scalar"),
        fields.p,
    )
    .eqn(fields.p);

    let mut system = EquationSystem::new();
    system.add_equation(momentum);
    system.add_equation(pressure);
    system
}

fn build_compressible_system(fields: &CompressibleFields) -> EquationSystem {
    let rho_eqn =
        (fvm::ddt(fields.rho) + fvm::div_flux(fields.phi_rho, fields.rho)).eqn(fields.rho);
    let rho_u_eqn =
        (fvm::ddt(fields.rho_u)
            + fvm::div_flux(fields.phi_rho_u, fields.rho_u))
            .eqn(fields.rho_u);
    let rho_e_eqn =
        (fvm::ddt(fields.rho_e)
            + fvm::div_flux(fields.phi_rho_e, fields.rho_e))
            .eqn(fields.rho_e);

    let mut system = EquationSystem::new();
    system.add_equation(rho_eqn);
    system.add_equation(rho_u_eqn);
    system.add_equation(rho_e_eqn);
    system
}

pub fn incompressible_momentum_system() -> EquationSystem {
    let fields = IncompressibleMomentumFields::new();
    build_incompressible_momentum_system(&fields)
}

pub fn compressible_system() -> EquationSystem {
    let fields = CompressibleFields::new();
    build_compressible_system(&fields)
}

pub fn incompressible_momentum_model() -> ModelSpec {
    let fields = IncompressibleMomentumFields::new();
    let system = build_incompressible_momentum_system(&fields);
    let layout = StateLayout::new(vec![
        fields.u,
        fields.p,
        fields.d_p,
        fields.grad_p,
        fields.grad_component,
    ]);
    ModelSpec {
        system,
        state_layout: layout,
        fields: ModelFields::Incompressible(fields),
    }
}

pub fn compressible_model() -> ModelSpec {
    let fields = CompressibleFields::new();
    let system = build_compressible_system(&fields);
    let layout = StateLayout::new(vec![fields.rho, fields.rho_u, fields.rho_e, fields.p, fields.u]);
    ModelSpec {
        system,
        state_layout: layout,
        fields: ModelFields::Compressible(fields),
    }
}

pub fn generic_diffusion_demo_model() -> ModelSpec {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let kappa = Coefficient::constant_unit(1.0, si::AREA / si::TIME);
    let eqn = (fvm::ddt(phi) + fvm::laplacian(kappa, phi)).eqn(phi);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

    let layout = StateLayout::new(vec![phi]);
    ModelSpec {
        system,
        state_layout: layout,
        fields: ModelFields::GenericCoupled(GenericCoupledFields::new(vec![phi])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::TermOp;

    #[test]
    fn incompressible_momentum_system_contains_expected_terms() {
        let system = incompressible_momentum_system();
        assert_eq!(system.equations().len(), 2);
        let momentum = &system.equations()[0];
        assert_eq!(momentum.target().name(), "U");
        assert_eq!(momentum.terms().len(), 4);
        assert_eq!(momentum.terms()[0].op, TermOp::Ddt);
        match &momentum.terms()[0].coeff {
            Some(Coefficient::Field(field)) => assert_eq!(field.name(), "rho"),
            other => panic!("expected rho coefficient, got {:?}", other),
        }
        assert_eq!(momentum.terms()[1].op, TermOp::Div);
        assert_eq!(momentum.terms()[2].op, TermOp::Laplacian);
        assert_eq!(momentum.terms()[3].op, TermOp::Grad);

        let pressure = &system.equations()[1];
        assert_eq!(pressure.target().name(), "p");
        assert_eq!(pressure.terms().len(), 1);
        assert_eq!(pressure.terms()[0].op, TermOp::Laplacian);
        match &pressure.terms()[0].coeff {
            Some(Coefficient::Product(lhs, rhs)) => {
                assert!(matches!(**lhs, Coefficient::Field(_)));
                assert!(matches!(**rhs, Coefficient::Field(_)));
            }
            other => panic!("expected coefficient product, got {:?}", other),
        }
    }

    #[test]
    fn incompressible_momentum_model_includes_state_layout() {
        let model = incompressible_momentum_model();
        assert_eq!(model.state_layout.offset_for("U"), Some(0));
        assert_eq!(model.state_layout.offset_for("p"), Some(2));
        assert_eq!(model.state_layout.stride(), 8);
        assert_eq!(model.system.equations().len(), 2);
        assert!(model.kernel_plan().contains(KernelKind::CoupledAssembly));
        assert!(matches!(model.fields, ModelFields::Incompressible(_)));
    }

    #[test]
    fn compressible_model_defines_conservative_equations() {
        let model = compressible_model();
        assert_eq!(model.system.equations().len(), 3);
        assert_eq!(model.system.equations()[1].terms().len(), 2);
        assert_eq!(model.system.equations()[2].terms().len(), 2);
        assert!(model.kernel_plan().contains(KernelKind::CompressibleFluxKt));
        assert!(matches!(model.fields, ModelFields::Compressible(_)));
    }
}
