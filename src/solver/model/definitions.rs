use super::ast::{
    fvc, fvm, surface_scalar, vol_scalar, vol_vector, Coefficient, EquationSystem,
};
use super::state_layout::StateLayout;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub system: EquationSystem,
    pub state_layout: StateLayout,
}

pub fn incompressible_momentum_system() -> EquationSystem {
    let u = vol_vector("U");
    let p = vol_scalar("p");
    let phi = surface_scalar("phi");
    let nu = vol_scalar("nu");
    let d_p = vol_scalar("d_p");

    let momentum = (fvm::ddt(u.clone())
        + fvm::div(phi, u.clone())
        + fvm::laplacian(Coefficient::field(nu).expect("nu must be scalar"), u.clone())
        + fvc::grad(p.clone()))
    .eqn(u);

    let pressure = fvm::laplacian(
        Coefficient::field(d_p).expect("d_p must be scalar"),
        p.clone(),
    )
    .eqn(p);

    let mut system = EquationSystem::new();
    system.add_equation(momentum);
    system.add_equation(pressure);
    system
}

pub fn incompressible_momentum_model() -> ModelSpec {
    let system = incompressible_momentum_system();
    let layout = StateLayout::new(vec![
        vol_vector("U"),
        vol_scalar("p"),
        vol_scalar("d_p"),
        vol_vector("grad_p"),
        vol_vector("grad_component"),
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
