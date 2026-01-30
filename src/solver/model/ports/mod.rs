//! Type-safe port system for model fields, parameters, and GPU resources.
//!
//! This module provides statically-typed port abstractions that replace string-based
//! field name lookups with compile-time verified references. Ports track physical
//! dimensions, field kinds, and access modes to prevent errors at compile time.
//!
//! # Overview
//!
//! - [`FieldPort`]: References to fields in the state layout with physical dimensions
//! - [`ParamPort`]: Type-safe named parameters with units
//! - [`BufferPort`]: WGSL buffer binding specifications
//! - [`PortRegistry`]: Runtime storage and lookup for ports
//!
//! # Example
//!
//! ```rust,ignore
//! // Instead of string lookups:
//! let offset = model.state_layout.offset_for("d_p").unwrap();
//!
//! // Use typed ports:
//! let dp_port: FieldPort<DPDim, Scalar> = model.ports.scalar_field("d_p")?;
//! let expr = dp_port.access("idx"); // Generates: state[idx * stride + offset]
//! ```

pub mod buffer;
pub mod dimensions;
pub mod field;
pub mod params;
pub mod registry;
pub mod traits;

#[cfg(test)]
mod tests;

pub use buffer::{
    bind_groups, AccessMode, BufferBindingGroup, BufferBindingGroupBuilder, BufferF32, BufferI32,
    BufferPort, BufferType, BufferU32, BufferVec2F32, BufferVec3F32, ReadOnly, ReadWrite,
};
pub use dimensions::{
    Acceleration, Area, Density, DimCompatible, Dimensionless, DivDim, DynamicViscosity, Energy,
    EnergyDensity, Force, InvTime, KinematicViscosity, Length, Mass, MassFlux, MomentumDensity,
    MulDim, PowDim, Pressure, PressureGradient, SqrtDim, Temperature, Time, UnitDimension,
    Velocity, Volume, D_P,
};
pub use field::{
    ComponentOffset, FieldKind, FieldPort, FieldPortError, FieldPortProvider, Scalar, Vector2,
    Vector3,
};
pub use params::{
    ParamPort, ParamPortError, ParamPortProvider, ParamPortSet, ParamPortSetBuilder, ParamType,
    F32, F64, I32, U32,
};
pub use registry::{PortRegistry, PortRegistryError, TypedPortRegistry};

// Re-export derive macros from cfd2_macros
pub use cfd2_macros::{ModulePorts, PortSet};

// Re-export traits
pub use traits::{ModulePorts as ModulePortsTrait, PortSet as PortSetTrait, PortValidationError};

/// A unique identifier for a port within a registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortId(u32);

impl PortId {
    /// Create a new port ID. This should only be called by PortRegistry.
    pub(crate) fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the underlying ID value.
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Core trait for all port types.
///
/// Ports provide a type-safe way to reference runtime resources with compile-time
/// verification of properties like physical dimensions and data types.
pub trait Port: Clone + Copy + PartialEq + Eq {
    /// The type of resource this port references.
    type Resource;

    /// Get the port's unique identifier.
    fn id(&self) -> PortId;

    /// Get the human-readable name of this port.
    fn name(&self) -> &'static str;
}

/// Trait for ports that can generate WGSL code.
///
/// This trait allows ports to emit the appropriate WGSL syntax for accessing
/// the resource they represent.
pub trait WgslPort: Port {
    /// Generate a WGSL expression to access this resource.
    ///
    /// For field ports, this generates array access expressions.
    /// For parameter ports, this generates uniform buffer field access.
    fn wgsl_access(&self, index: Option<&str>) -> String;

    /// Generate the WGSL type for this port's resource.
    fn wgsl_type(&self) -> &'static str;

    /// Generate the WGSL binding declaration for this port (if applicable).
    fn wgsl_binding(&self) -> Option<String> {
        None
    }
}

/// Trait for ports with associated physical dimensions.
///
/// This enables compile-time verification of dimensional consistency
/// in arithmetic operations.
pub trait DimensionalPort: Port {
    /// The physical dimension of values accessed through this port.
    type Dimension: UnitDimension;

    /// Get the physical dimension at runtime (for error messages).
    fn dimension(&self) -> crate::solver::units::UnitDim;
}

/// Compile-time assertion that two dimensions are compatible.
///
/// This macro generates a compile error if the dimensions don't match.
#[macro_export]
macro_rules! assert_dim_compatible {
    ($a:ty, $b:ty) => {
        const _: () = assert!(
            <$a as $crate::solver::model::ports::UnitDimension>::IS_COMPATIBLE_WITH::<$b>(),
            "Dimension mismatch: expected compatible dimensions"
        );
    };
}
