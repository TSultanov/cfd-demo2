use cfd2_ir::solver::dimensions::{Density, EnergyDensity, MomentumDensity};
use cfd2_ir::solver::ir::{EquationSystem, StateLayout};

/// Minimal state/system fixture for KT codegen tests.
///
/// This intentionally does not depend on `ModelSpec` to preserve the mechanical IR boundary.
pub(crate) fn compressible_state_layout_and_system() -> (StateLayout, EquationSystem) {
    let rho = cfd2_ir::solver::ir::vol_scalar_dim::<Density>("rho");
    let rho_u = cfd2_ir::solver::ir::vol_vector_dim::<MomentumDensity>("rho_u");
    let rho_e = cfd2_ir::solver::ir::vol_scalar_dim::<EnergyDensity>("rho_e");

    let layout = StateLayout::new(vec![rho, rho_u, rho_e]);

    let mut system = EquationSystem::new();
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho));
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho_u));
    system.add_equation(cfd2_ir::solver::ir::Equation::new(rho_e));

    (layout, system)
}
