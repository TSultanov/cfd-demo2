//! Passing test: PortSet derive with valid attributes.

use cfd2::solver::model::backend::ast::vol_scalar;
use cfd2::solver::model::backend::state_layout::StateLayout;
use cfd2::solver::model::ports::{
    Dimensionless, FieldPort, ParamPort, Port, PortRegistry, PortSetTrait, Scalar, Time, F32,
};
use cfd2::solver::units::si;
use cfd2_macros::PortSet;

#[derive(PortSet)]
pub struct TestParams {
    #[param(name = "dt", wgsl = "dt")]
    pub dt: ParamPort<F32, Time>,
}

#[derive(PortSet)]
pub struct TestFields {
    #[field(name = "p")]
    pub pressure: FieldPort<Dimensionless, Scalar>,
}

fn main() {
    // Create a test state layout
    let layout = StateLayout::new(vec![vol_scalar("p", si::PRESSURE)]);
    let mut registry = PortRegistry::new(layout);

    // Test from_registry
    let params = TestParams::from_registry(&mut registry).expect("should create params");
    assert_eq!(params.dt.key(), "dt");

    let fields = TestFields::from_registry(&mut registry).expect("should create fields");
    assert_eq!(fields.pressure.name(), "p");
}
