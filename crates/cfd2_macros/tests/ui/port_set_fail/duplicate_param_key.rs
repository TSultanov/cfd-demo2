//! Failing test: Duplicate parameter key.

use cfd2::solver::model::ports::{Dimensionless, ParamPort, Time, F32};
use cfd2_macros::PortSet;

#[derive(PortSet)]
pub struct TestParams {
    #[param(name = "dt", wgsl = "dt")]
    pub dt: ParamPort<F32, Time>,

    #[param(name = "dt", wgsl = "delta_t")] // Duplicate key!
    pub dt2: ParamPort<F32, Time>,
}

fn main() {}
