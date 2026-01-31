// Prelude module for port authoring.
//
// This module provides commonly used types and traits for declaring port sets
// and implementing model modules. Use `use crate::solver::model::ports::prelude::*;`
// for convenient access to essential port functionality.
//
// # Example
//
// ```rust,ignore
// use crate::solver::model::ports::prelude::*;
//
// #[derive(PortSet)]
// pub struct MyPorts {
//     #[field(name = "p")]
//     pub pressure: FieldPort<Pressure, Scalar>,
//
//     #[param(name = "dt", wgsl = "dt")]
//     pub dt: ParamPort<F32, Time>,
// }
// ```

// Derive macros for port sets and module ports (only available in runtime, not in build script)
#[cfg(cfd2_build_script)]
pub use cfd2_macros::{ModulePorts, PortSet};

// Core port types and field kinds
pub use super::{
    BufferPort, FieldPort, FieldKind, ParamPort, Scalar, Vector2, Vector3,
};

// Access modes for buffer ports
pub use super::{ReadOnly, ReadWrite};

// Parameter types
pub use super::{F32, F64, I32, U32};

// Port registry and errors
pub use super::{PortRegistry, PortRegistryError, PortValidationError};

// Core dimension trait and common dimension aliases
pub use super::{
    Acceleration, Area, Density, Dimensionless, DynamicViscosity, Energy, EnergyDensity, Force,
    InvTime, KinematicViscosity, Length, Mass, MassFlux, MomentumDensity, Pressure,
    PressureGradient, Temperature, Time, UnitDimension, Velocity, Volume, D_P,
};
