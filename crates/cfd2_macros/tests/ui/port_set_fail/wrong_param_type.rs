//! Failing test: Wrong field type for param attribute.

use cfd2::solver::model::ports::{Time, F32};
use cfd2_macros::PortSet;

// Wrong type - should be ParamPort
#[derive(PortSet)]
pub struct TestParams {
    #[param(name = "dt", wgsl = "dt")]
    pub dt: F32, // Wrong type!
}

fn main() {}
