//! Parameter ports for type-safe named parameter access.
//!
//! Parameter ports replace string-based named parameter lookups with
//! compile-time type and dimension verification.

use super::{DimensionalPort, Port, PortId, WgslPort};
use crate::solver::model::ports::dimensions::UnitDimension;
use crate::solver::units::UnitDim;
use std::marker::PhantomData;

/// Trait for parameter value types.
///
/// This trait is implemented by marker types representing valid parameter types.
pub trait ParamType: 'static + Copy + Send + Sync + Eq + PartialEq + std::fmt::Debug {
    /// The WGSL type name for this parameter.
    fn wgsl_type() -> &'static str;

    /// Size in bytes of this parameter type.
    fn size_bytes() -> usize;
}

/// 32-bit floating point parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32;

impl ParamType for F32 {
    fn wgsl_type() -> &'static str {
        "f32"
    }

    fn size_bytes() -> usize {
        4
    }
}

/// 64-bit floating point parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F64;

impl ParamType for F64 {
    fn wgsl_type() -> &'static str {
        "f64"
    }

    fn size_bytes() -> usize {
        8
    }
}

/// 32-bit signed integer parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I32;

impl ParamType for I32 {
    fn wgsl_type() -> &'static str {
        "i32"
    }

    fn size_bytes() -> usize {
        4
    }
}

/// 32-bit unsigned integer parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U32;

impl ParamType for U32 {
    fn wgsl_type() -> &'static str {
        "u32"
    }

    fn size_bytes() -> usize {
        4
    }
}

/// A type-safe reference to a named parameter.
///
/// `ParamPort` provides compile-time verification of:
/// - Parameter type (f32, u32, etc.)
/// - Physical dimension
/// - Mapping to WGSL uniform buffer fields
///
/// # Type Parameters
///
/// - `T`: The parameter type (e.g., `F32`, `U32`)
/// - `D`: The physical dimension (e.g., `Time`, `Dimensionless`)
///
/// # Example
///
/// ```rust,ignore
/// // Create a port for the time step parameter
/// let dt_port: ParamPort<F32, Time> = registry.param("dt")?;
///
/// // Access generates: constants.dt
/// let dt_expr = dt_port.wgsl_access(None);
///
/// // Physical dimension is tracked at compile time
/// fn compute_velocity(dx: ParamPort<F32, Length>, dt: ParamPort<F32, Time>) -> Expr {
///     dx.wgsl_access(None) / dt.wgsl_access(None)  // Returns Length/Time = Velocity
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamPort<T: ParamType, D: UnitDimension> {
    id: PortId,
    key: &'static str,
    wgsl_field_name: &'static str,
    runtime_dim: UnitDim,
    _ty: PhantomData<T>,
    _dim: PhantomData<D>,
}

impl<T: ParamType, D: UnitDimension> ParamPort<T, D> {
    /// Create a new parameter port.
    ///
    /// This should be called by PortRegistry during parameter registration.
    pub fn new(
        id: PortId,
        key: &'static str,
        wgsl_field_name: &'static str,
        runtime_dim: UnitDim,
    ) -> Self {
        Self {
            id,
            key,
            wgsl_field_name,
            runtime_dim,
            _ty: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Get the parameter key (human-readable name).
    pub fn key(&self) -> &'static str {
        self.key
    }

    /// Get the WGSL field name in the constants struct.
    pub fn wgsl_field_name(&self) -> &'static str {
        self.wgsl_field_name
    }

    /// Get the runtime physical dimension.
    pub fn runtime_dimension(&self) -> UnitDim {
        self.runtime_dim
    }
}

impl<T: ParamType, D: UnitDimension> Port for ParamPort<T, D> {
    type Resource = ();

    fn id(&self) -> PortId {
        self.id
    }

    fn name(&self) -> &'static str {
        self.key
    }
}

impl<T: ParamType, D: UnitDimension> DimensionalPort for ParamPort<T, D> {
    type Dimension = D;

    fn dimension(&self) -> UnitDim {
        self.runtime_dim
    }
}

impl<T: ParamType, D: UnitDimension> WgslPort for ParamPort<T, D> {
    fn wgsl_access(&self, _index: Option<&str>) -> String {
        format!("constants.{}", self.wgsl_field_name)
    }

    fn wgsl_type(&self) -> &'static str {
        T::wgsl_type()
    }
}

/// A set of related parameter ports.
///
/// This is typically used to group parameters by module or functionality.
#[derive(Debug, Clone)]
pub struct ParamPortSet {
    name: &'static str,
    ports: Vec<ParamPortEntry>,
}

/// Entry in a parameter port set.
#[derive(Debug, Clone)]
pub struct ParamPortEntry {
    pub key: &'static str,
    pub wgsl_field: &'static str,
    pub runtime_dim: UnitDim,
    pub param_type: ParamTypeKind,
}

/// Enum representing the type of a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamTypeKind {
    F32,
    F64,
    I32,
    U32,
}

impl ParamTypeKind {
    pub fn wgsl_type(self) -> &'static str {
        match self {
            ParamTypeKind::F32 => "f32",
            ParamTypeKind::F64 => "f64",
            ParamTypeKind::I32 => "i32",
            ParamTypeKind::U32 => "u32",
        }
    }

    pub fn size_bytes(self) -> usize {
        match self {
            ParamTypeKind::F32 => 4,
            ParamTypeKind::F64 => 8,
            ParamTypeKind::I32 => 4,
            ParamTypeKind::U32 => 4,
        }
    }
}

/// Builder for parameter port sets.
#[derive(Debug, Default)]
pub struct ParamPortSetBuilder {
    name: &'static str,
    entries: Vec<ParamPortEntry>,
}

impl ParamPortSetBuilder {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            entries: Vec::new(),
        }
    }

    pub fn add_param<T: ParamType + IntoParamTypeKind, D: UnitDimension>(
        mut self,
        key: &'static str,
        wgsl_field: &'static str,
    ) -> Self {
        self.entries.push(ParamPortEntry {
            key,
            wgsl_field,
            runtime_dim: D::to_runtime(),
            param_type: T::into_kind(),
        });
        self
    }

    pub fn build(self) -> ParamPortSet {
        ParamPortSet {
            name: self.name,
            ports: self.entries,
        }
    }
}

/// Trait to convert ParamType marker types to ParamTypeKind.
pub trait IntoParamTypeKind: ParamType {
    fn into_kind() -> ParamTypeKind;
}

impl IntoParamTypeKind for F32 {
    fn into_kind() -> ParamTypeKind {
        ParamTypeKind::F32
    }
}

impl IntoParamTypeKind for F64 {
    fn into_kind() -> ParamTypeKind {
        ParamTypeKind::F64
    }
}

impl IntoParamTypeKind for I32 {
    fn into_kind() -> ParamTypeKind {
        ParamTypeKind::I32
    }
}

impl IntoParamTypeKind for U32 {
    fn into_kind() -> ParamTypeKind {
        ParamTypeKind::U32
    }
}

/// Trait for types that can provide parameter ports.
pub trait ParamPortProvider {
    /// Get a parameter port by key.
    fn param<T: ParamType, D: UnitDimension>(
        &self,
        key: &str,
    ) -> Result<ParamPort<T, D>, ParamPortError>;
}

/// Error type for parameter port operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamPortError {
    /// Parameter not found.
    ParamNotFound { key: String },
    /// Parameter type mismatch.
    TypeMismatch {
        key: String,
        expected: ParamTypeKind,
        found: ParamTypeKind,
    },
    /// Physical dimension mismatch.
    DimensionMismatch {
        key: String,
        expected: UnitDim,
        found: UnitDim,
    },
}

impl std::fmt::Display for ParamPortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamPortError::ParamNotFound { key } => {
                write!(f, "Parameter '{}' not found", key)
            }
            ParamPortError::TypeMismatch {
                key,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Parameter '{}' type mismatch: expected {:?}, found {:?}",
                    key, expected, found
                )
            }
            ParamPortError::DimensionMismatch {
                key,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Parameter '{}' dimension mismatch: expected {}, found {}",
                    key, expected, found
                )
            }
        }
    }
}

impl std::error::Error for ParamPortError {}

/// Macro for defining parameter port sets.
///
/// # Example
///
/// ```rust,ignore
/// define_param_set! {
///     GenericCoupledParams {
///         dt: (F32, Time, "dt"),
///         viscosity: (F32, DynamicViscosity, "viscosity"),
///         density: (F32, Density, "density"),
///         alpha_u: (F32, Dimensionless, "alpha_u"),
///         alpha_p: (F32, Dimensionless, "alpha_p"),
///     }
/// }
/// ```
#[macro_export]
macro_rules! define_param_set {
    (
        $name:ident {
            $(
                $field:ident: ($type:ty, $dim:ty, $wgsl:literal)
            ),* $(,)?
        }
    ) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            $(
                pub $field: $crate::solver::model::ports::ParamPort<$type, $dim>,
            )*
        }

        impl $name {
            /// Create the parameter set from a registry.
            pub fn from_registry(registry: &dyn $crate::solver::model::ports::ParamPortProvider) -> Result<Self, $crate::solver::model::ports::ParamPortError> {
                Ok(Self {
                    $(
                        $field: registry.param::<$type, $dim>(stringify!($field))?,
                    )*
                })
            }

            /// Generate the WGSL struct definition for these parameters.
            pub fn wgsl_struct_definition() -> String {
                let mut fields = Vec::new();
                $(
                    fields.push(format!("    {}: {},", $wgsl, <$type as $crate::solver::model::ports::ParamType>::wgsl_type()));
                )*
                format!("struct Constants {{\n{}\n}}", fields.join("\n"))
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::ports::dimensions::{Dimensionless, Length, Time, Velocity};

    #[test]
    fn param_type_properties() {
        assert_eq!(F32::wgsl_type(), "f32");
        assert_eq!(F32::size_bytes(), 4);

        assert_eq!(U32::wgsl_type(), "u32");
        assert_eq!(U32::size_bytes(), 4);

        assert_eq!(F64::wgsl_type(), "f64");
        assert_eq!(F64::size_bytes(), 8);
    }

    #[test]
    fn param_port_wgsl_access() {
        let dt = ParamPort::<F32, Time>::new(PortId::new(1), "dt", "dt", UnitDim::new(0, 0, 1));

        assert_eq!(dt.wgsl_access(None), "constants.dt");
        assert_eq!(dt.key(), "dt");
        assert_eq!(dt.wgsl_field_name(), "dt");
        assert_eq!(dt.wgsl_type(), "f32");
    }

    #[test]
    fn param_type_kind_conversion() {
        assert_eq!(F32::into_kind(), ParamTypeKind::F32);
        assert_eq!(F64::into_kind(), ParamTypeKind::F64);
        assert_eq!(I32::into_kind(), ParamTypeKind::I32);
        assert_eq!(U32::into_kind(), ParamTypeKind::U32);
    }

    #[test]
    fn param_port_set_builder() {
        let set = ParamPortSetBuilder::new("test_params")
            .add_param::<F32, Time>("dt", "dt")
            .add_param::<F32, Dimensionless>("alpha", "alpha_u")
            .build();

        assert_eq!(set.name, "test_params");
        assert_eq!(set.ports.len(), 2);
        assert_eq!(set.ports[0].key, "dt");
        assert_eq!(set.ports[1].key, "alpha");
    }
}
