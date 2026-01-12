use cfd2_ir::solver::ir::{EquationSystem, FieldKind, StateLayout};
use cfd2_ir::solver::units::si;

/// Minimal state/system fixture for KT codegen tests.
///
/// This intentionally does not depend on `ModelSpec` to preserve the mechanical IR boundary.
pub(crate) fn compressible_state_layout_and_system() -> (StateLayout, EquationSystem) {
    let rho = cfd2_ir::solver::ir::vol_scalar("rho", si::kg() / si::m().powi(3));
    let rho_u = cfd2_ir::solver::ir::vol_vector("rho_u", si::kg() / si::m().powi(2) / si::s());
    let rho_e = cfd2_ir::solver::ir::vol_scalar(
        "rho_e",
        si::kg() / si::m().powi(1) / si::s().powi(2),
    );

    let layout = StateLayout::new(vec![
        cfd2_ir::solver::ir::StateField::new("rho", FieldKind::Scalar, rho.unit()),
        cfd2_ir::solver::ir::StateField::new("rho_u", FieldKind::Vector2, rho_u.unit()),
        cfd2_ir::solver::ir::StateField::new("rho_e", FieldKind::Scalar, rho_e.unit()),
    ]);

    let mut system = EquationSystem::new();
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho));
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho_u));
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho_e));

    (layout, system)
}

