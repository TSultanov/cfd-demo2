//! Passing test: Both ModulePorts and PortSet derives on the same struct.
//!
//! This tests that a struct can implement both traits simultaneously,
//! enabling end-to-end port declaration for solver modules.

use cfd2::solver::model::backend::ast::vol_scalar;
use cfd2::solver::model::backend::state_layout::StateLayout;
use cfd2::solver::model::ports::{
    Dimensionless, FieldPort, ModulePortsTrait, ParamPort, Port, PortRegistry, PortSetTrait, Scalar,
    Time, F32,
};
use cfd2::solver::units::si;
use cfd2_macros::{ModulePorts, PortSet};

/// A complete module ports definition using both derives.
/// This pattern is used by solver modules to declare their port requirements.
#[derive(PortSet, ModulePorts)]
#[port(module = "test_module")]
pub struct TestModulePorts {
    /// Parameter port: time step
    #[param(name = "dt", wgsl = "dt")]
    pub dt: ParamPort<F32, Time>,

    /// Field port: pressure field
    #[field(name = "p")]
    pub pressure: FieldPort<Dimensionless, Scalar>,
}

fn main() {
    // Create a test state layout with a pressure field
    let layout = StateLayout::new(vec![vol_scalar("p", si::PRESSURE)]);
    let mut registry = PortRegistry::new(layout);

    // Test PortSetTrait::from_registry - creates ports from registry
    let ports = TestModulePorts::from_registry(&mut registry).expect("should create ports");

    // Verify parameter port
    assert_eq!(ports.dt.key(), "dt");

    // Verify field port
    assert_eq!(ports.pressure.name(), "p");

    // Test ModulePortsTrait methods
    // module_name() returns the module name from #[port(module = "...")]
    assert_eq!(ports.module_name(), "test_module");

    // port_set() returns &Self since Self is the PortSet
    let port_set_ref: &TestModulePorts = ports.port_set();
    assert_eq!(port_set_ref.pressure.name(), "p");

    // Test that MODULE_NAME constant is available
    assert_eq!(TestModulePorts::MODULE_NAME, "test_module");

    // Test port_manifest() from PortSet trait
    let manifest = TestModulePorts::port_manifest();
    assert_eq!(manifest.params.len(), 1);
    assert_eq!(manifest.params[0].key, "dt");
    assert_eq!(manifest.fields.len(), 1);
    assert_eq!(manifest.fields[0].name, "p");
}
