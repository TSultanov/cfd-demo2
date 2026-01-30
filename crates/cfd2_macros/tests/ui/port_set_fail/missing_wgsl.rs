//! Failing test: Missing wgsl attribute on param field.

use cfd2::solver::model::ports::{ParamPort, Time, F32};
use cfd2_macros::PortSet;

#[derive(PortSet)]
pub struct TestParams {
    #[param(name = "dt")] // Missing wgsl attribute
    pub dt: ParamPort<F32, Time>,
}

fn main() {}
