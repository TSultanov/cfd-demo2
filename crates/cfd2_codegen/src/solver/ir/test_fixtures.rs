use cfd2_ir::solver::ir::{EquationSystem, StateLayout};
use cfd2_ir::solver::units::si;

/// Minimal state/system fixture for KT codegen tests.
///
/// This intentionally does not depend on `ModelSpec` to preserve the mechanical IR boundary.
pub(crate) fn compressible_state_layout_and_system() -> (StateLayout, EquationSystem) {
    let rho = cfd2_ir::solver::ir::vol_scalar("rho", si::DENSITY);
    let rho_u = cfd2_ir::solver::ir::vol_vector("rho_u", si::MOMENTUM_DENSITY);
    let rho_e = cfd2_ir::solver::ir::vol_scalar("rho_e", si::ENERGY_DENSITY);

    let layout = StateLayout::new(vec![rho, rho_u, rho_e]);

    let mut system = EquationSystem::new();
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho));
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho_u));
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho_e));

    (layout, system)
}
