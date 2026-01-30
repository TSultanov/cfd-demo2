//! Field ports for type-safe state field access.
//!
//! Field ports provide compile-time verified access to fields in the state layout,
//! tracking both the physical dimension and the field kind (scalar, vector2, vector3).

use super::{DimensionalPort, Port, PortId, WgslPort};
use crate::solver::model::ports::dimensions::UnitDimension;
use crate::solver::units::UnitDim;
use std::marker::PhantomData;

/// Trait for field kinds (scalar, vector types).
///
/// This trait is implemented by marker types that represent the shape of field data.
pub trait FieldKind: 'static + Copy + Send + Sync + Eq + PartialEq + std::fmt::Debug {
    /// Number of components in this field kind (1 for scalar, 2 for Vector2, etc.)
    const COMPONENT_COUNT: u32;

    /// The WGSL type name for this field kind.
    fn wgsl_type() -> &'static str;

    /// Get the component at the given index.
    ///
    /// Returns the component name ("x", "y", "z") for the given index.
    fn component_name(index: u32) -> Option<&'static str> {
        match index {
            0 => Some("x"),
            1 if Self::COMPONENT_COUNT > 1 => Some("y"),
            2 if Self::COMPONENT_COUNT > 2 => Some("z"),
            _ => None,
        }
    }
}

/// Scalar field (single f32 value).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scalar;

impl FieldKind for Scalar {
    const COMPONENT_COUNT: u32 = 1;

    fn wgsl_type() -> &'static str {
        "f32"
    }
}

/// 2D vector field (vec2<f32>).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vector2;

impl FieldKind for Vector2 {
    const COMPONENT_COUNT: u32 = 2;

    fn wgsl_type() -> &'static str {
        "vec2<f32>"
    }
}

/// 3D vector field (vec3<f32>).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vector3;

impl FieldKind for Vector3 {
    const COMPONENT_COUNT: u32 = 3;

    fn wgsl_type() -> &'static str {
        "vec3<f32>"
    }
}

/// A type-safe reference to a field in the state layout.
///
/// `FieldPort` combines compile-time knowledge of:
/// - Physical dimension `D`
/// - Field kind `K` (scalar, vector type)
/// - Runtime offset within the state array
///
/// # Type Parameters
///
/// - `D`: The physical dimension (e.g., `Velocity`, `Pressure`)
/// - `K`: The field kind (e.g., `Scalar`, `Vector2`)
///
/// # Example
///
/// ```rust,ignore
/// // Create a port for the velocity field
/// let u_port: FieldPort<Velocity, Vector2> = registry.vector_field("U")?;
///
/// // Access generates: state[idx * stride + offset]
/// let u_expr = u_port.access("idx");
///
/// // Access specific component
/// let u_x = u_port.component(0)?.access("idx"); // state[idx * stride + offset + 0]
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldPort<D: UnitDimension, K: FieldKind> {
    id: PortId,
    name: &'static str,
    offset: u32,
    stride: u32,
    runtime_dim: UnitDim,
    _dim: PhantomData<D>,
    _kind: PhantomData<K>,
}

impl<D: UnitDimension, K: FieldKind> FieldPort<D, K> {
    /// Create a new field port.
    ///
    /// This should be called by PortRegistry during field registration.
    pub fn new(
        id: PortId,
        name: &'static str,
        offset: u32,
        stride: u32,
        runtime_dim: UnitDim,
    ) -> Self {
        Self {
            id,
            name,
            offset,
            stride,
            runtime_dim,
            _dim: PhantomData,
            _kind: PhantomData,
        }
    }

    /// Get the offset of this field within each cell's state.
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Get the stride between cells in the state array.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// Get the runtime physical dimension (for error messages).
    pub fn runtime_dimension(&self) -> UnitDim {
        self.runtime_dim
    }

    /// Access this field as a component port.
    ///
    /// Returns a `ComponentPort` that can access individual components
    /// for vector fields, or the scalar value itself.
    pub fn component(&self, index: u32) -> Option<ComponentOffset<D, K>> {
        if index >= K::COMPONENT_COUNT {
            return None;
        }
        Some(ComponentOffset {
            base_offset: self.offset,
            component_offset: index,
            stride: self.stride,
            _dim: PhantomData,
            _kind: PhantomData,
        })
    }

    /// Generate a linear index expression for this field.
    ///
    /// Returns an expression like `idx * stride + offset`.
    pub fn linear_index(&self, index_var: &str) -> String {
        if self.stride == 1 {
            format!("{} + {}u", index_var, self.offset)
        } else {
            format!("{} * {}u + {}u", index_var, self.stride, self.offset)
        }
    }
}

impl<D: UnitDimension, K: FieldKind> Port for FieldPort<D, K> {
    type Resource = ();

    fn id(&self) -> PortId {
        self.id
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

impl<D: UnitDimension, K: FieldKind> DimensionalPort for FieldPort<D, K> {
    type Dimension = D;

    fn dimension(&self) -> UnitDim {
        self.runtime_dim
    }
}

impl<D: UnitDimension, K: FieldKind> WgslPort for FieldPort<D, K> {
    fn wgsl_access(&self, index: Option<&str>) -> String {
        let idx = index.unwrap_or("idx");
        format!("state[{}]", self.linear_index(idx))
    }

    fn wgsl_type(&self) -> &'static str {
        K::wgsl_type()
    }
}

/// Offset information for accessing a specific component of a field.
///
/// This is returned by `FieldPort::component()` and provides access to
/// individual components of vector fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentOffset<D: UnitDimension, K: FieldKind> {
    base_offset: u32,
    component_offset: u32,
    stride: u32,
    _dim: PhantomData<D>,
    _kind: PhantomData<K>,
}

impl<D: UnitDimension, K: FieldKind> ComponentOffset<D, K> {
    /// Get the full offset (base + component).
    pub fn full_offset(&self) -> u32 {
        self.base_offset + self.component_offset
    }

    /// Generate WGSL array access for this component.
    pub fn wgsl_access(&self, index_var: &str) -> String {
        let full_offset = self.full_offset();
        if self.stride == 1 {
            format!("state[{} + {}u]", index_var, full_offset)
        } else {
            format!("state[{} * {}u + {}u]", index_var, self.stride, full_offset)
        }
    }
}

/// Trait for types that can provide field ports.
///
/// This is implemented by model components that expose fields.
pub trait FieldPortProvider {
    /// Get a scalar field port by name.
    fn scalar_field<D: UnitDimension>(
        &self,
        name: &str,
    ) -> Result<FieldPort<D, Scalar>, FieldPortError>;

    /// Get a Vector2 field port by name.
    fn vector2_field<D: UnitDimension>(
        &self,
        name: &str,
    ) -> Result<FieldPort<D, Vector2>, FieldPortError>;

    /// Get a Vector3 field port by name.
    fn vector3_field<D: UnitDimension>(
        &self,
        name: &str,
    ) -> Result<FieldPort<D, Vector3>, FieldPortError>;
}

/// Error type for field port operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldPortError {
    /// Field not found.
    FieldNotFound { name: String },
    /// Field kind mismatch (e.g., expected Vector2 but found Scalar).
    KindMismatch {
        name: String,
        expected: &'static str,
        found: &'static str,
    },
    /// Physical dimension mismatch.
    DimensionMismatch {
        name: String,
        expected: UnitDim,
        found: UnitDim,
    },
}

impl std::fmt::Display for FieldPortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldPortError::FieldNotFound { name } => {
                write!(f, "Field '{}' not found in state layout", name)
            }
            FieldPortError::KindMismatch {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Field '{}' kind mismatch: expected {}, found {}",
                    name, expected, found
                )
            }
            FieldPortError::DimensionMismatch {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Field '{}' dimension mismatch: expected {}, found {}",
                    name, expected, found
                )
            }
        }
    }
}

impl std::error::Error for FieldPortError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::ports::dimensions::{Length, Time, Velocity};

    #[test]
    fn scalar_field_kind_properties() {
        assert_eq!(Scalar::COMPONENT_COUNT, 1);
        assert_eq!(Scalar::wgsl_type(), "f32");
        assert_eq!(Scalar::component_name(0), Some("x"));
        assert_eq!(Scalar::component_name(1), None);
    }

    #[test]
    fn vector2_field_kind_properties() {
        assert_eq!(Vector2::COMPONENT_COUNT, 2);
        assert_eq!(Vector2::wgsl_type(), "vec2<f32>");
        assert_eq!(Vector2::component_name(0), Some("x"));
        assert_eq!(Vector2::component_name(1), Some("y"));
        assert_eq!(Vector2::component_name(2), None);
    }

    #[test]
    fn field_port_linear_index() {
        let port =
            FieldPort::<Velocity, Vector2>::new(PortId::new(0), "U", 0, 3, UnitDim::new(0, 1, -1));

        assert_eq!(port.linear_index("idx"), "idx * 3u + 0u");
        assert_eq!(port.linear_index("cell_idx"), "cell_idx * 3u + 0u");
    }

    #[test]
    fn field_port_linear_index_unit_stride() {
        let port =
            FieldPort::<Length, Scalar>::new(PortId::new(0), "x", 2, 1, UnitDim::new(0, 1, 0));

        assert_eq!(port.linear_index("idx"), "idx + 2u");
    }

    #[test]
    fn field_port_component_access() {
        let port =
            FieldPort::<Velocity, Vector2>::new(PortId::new(0), "U", 0, 3, UnitDim::new(0, 1, -1));

        let comp0 = port.component(0).unwrap();
        assert_eq!(comp0.full_offset(), 0);
        assert_eq!(comp0.wgsl_access("idx"), "state[idx * 3u + 0u]");

        let comp1 = port.component(1).unwrap();
        assert_eq!(comp1.full_offset(), 1);
        assert_eq!(comp1.wgsl_access("idx"), "state[idx * 3u + 1u]");

        assert!(port.component(2).is_none());
    }

    #[test]
    fn field_port_wgsl_generation() {
        let port =
            FieldPort::<Length, Scalar>::new(PortId::new(0), "p", 2, 5, UnitDim::new(0, 1, 0));

        assert_eq!(port.wgsl_type(), "f32");
        assert_eq!(
            port.wgsl_access(Some("cell_idx")),
            "state[cell_idx * 5u + 2u]"
        );
    }

    #[test]
    fn field_port_implements_port_trait() {
        let port =
            FieldPort::<Time, Scalar>::new(PortId::new(42), "dt", 0, 1, UnitDim::new(0, 0, 1));

        assert_eq!(port.id().as_u32(), 42);
        assert_eq!(port.name(), "dt");
    }
}
