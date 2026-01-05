use super::ast::{
    fvc, fvm, surface_scalar, vol_scalar, vol_vector, Coefficient, EquationSystem, FieldRef,
    FluxRef,
};
use super::state_layout::StateLayout;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub system: EquationSystem,
    pub state_layout: StateLayout,
}

#[derive(Debug, Clone)]
struct IncompressibleMomentumFields {
    u: FieldRef,
    p: FieldRef,
    phi: FluxRef,
    nu: FieldRef,
    rho: FieldRef,
    d_p: FieldRef,
    grad_p: FieldRef,
    grad_component: FieldRef,
}

impl IncompressibleMomentumFields {
    fn new() -> Self {
        Self {
            u: vol_vector("U"),
            p: vol_scalar("p"),
            phi: surface_scalar("phi"),
            nu: vol_scalar("nu"),
            rho: vol_scalar("rho"),
            d_p: vol_scalar("d_p"),
            grad_p: vol_vector("grad_p"),
            grad_component: vol_vector("grad_component"),
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
            Coefficient::field(fields.nu).expect("nu must be scalar"),
            fields.u,
        )
        + fvc::grad(fields.p))
    .eqn(fields.u);

    let pressure = fvm::laplacian(
        Coefficient::field(fields.d_p).expect("d_p must be scalar"),
        fields.p,
    )
    .eqn(fields.p);

    let mut system = EquationSystem::new();
    system.add_equation(momentum);
    system.add_equation(pressure);
    system
}

pub fn incompressible_momentum_system() -> EquationSystem {
    let fields = IncompressibleMomentumFields::new();
    build_incompressible_momentum_system(&fields)
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::ast::TermOp;

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
    }

    #[test]
    fn incompressible_momentum_model_includes_state_layout() {
        let model = incompressible_momentum_model();
        assert_eq!(model.state_layout.offset_for("U"), Some(0));
        assert_eq!(model.state_layout.offset_for("p"), Some(2));
        assert_eq!(model.state_layout.stride(), 8);
        assert_eq!(model.system.equations().len(), 2);
    }
}
