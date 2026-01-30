//! Port registry for runtime management of ports.
//!
//! The port registry manages the creation and storage of field, parameter,
//! and buffer ports, mapping them to their underlying resources.

use super::{
    AccessMode, BufferPort, BufferType, FieldKind, FieldPort, ParamPort, ParamType, PortId,
};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::ports::dimensions::UnitDimension;
use crate::solver::units::UnitDim;
use std::collections::HashMap;

/// Central registry for all ports in a model.
///
/// The `PortRegistry` manages the creation of field, parameter, and buffer ports,
/// ensuring unique IDs and proper mapping to underlying resources.
///
/// # Example
///
/// ```rust,ignore
/// let mut registry = PortRegistry::new(&model.state_layout);
///
/// // Register a field port
/// let u_port = registry.register_field::<Velocity, Vector2>("U")?;
///
/// // Register a parameter port
/// let dt_port = registry.register_param::<F32, Time>("dt", "dt")?;
///
/// // Register a buffer port
/// let state_buffer = registry.register_buffer::<BufferF32, ReadWrite>(
///     "state", 1, 0
/// );
/// ```
#[derive(Debug)]
pub struct PortRegistry {
    next_id: u32,
    state_layout: StateLayout,
    field_ports: HashMap<PortId, FieldPortEntry>,
    param_ports: HashMap<PortId, ParamPortEntry>,
    buffer_ports: HashMap<PortId, BufferPortEntry>,
}

/// Storage for field port metadata.
#[derive(Debug, Clone)]
struct FieldPortEntry {
    name: String,
    offset: u32,
    stride: u32,
    component_count: u32,
    runtime_dim: UnitDim,
}

/// Storage for parameter port metadata.
#[derive(Debug, Clone)]
struct ParamPortEntry {
    key: String,
    wgsl_field: String,
    runtime_dim: UnitDim,
}

/// Storage for buffer port metadata.
#[derive(Debug, Clone)]
struct BufferPortEntry {
    name: String,
    group: u32,
    binding: u32,
}

impl PortRegistry {
    /// Create a new port registry for the given state layout.
    pub fn new(state_layout: StateLayout) -> Self {
        Self {
            next_id: 0,
            state_layout,
            field_ports: HashMap::new(),
            param_ports: HashMap::new(),
            buffer_ports: HashMap::new(),
        }
    }

    /// Get the state layout.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Allocate a new unique port ID.
    fn allocate_id(&mut self) -> PortId {
        let id = PortId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Register a scalar field port.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is not found in the state layout
    /// or if its kind doesn't match Scalar.
    pub fn register_scalar_field<D: UnitDimension>(
        &mut self,
        name: &'static str,
    ) -> Result<FieldPort<D, super::Scalar>, PortRegistryError> {
        self.register_field::<D, super::Scalar>(name)
    }

    /// Register a Vector2 field port.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is not found in the state layout
    /// or if its kind doesn't match Vector2.
    pub fn register_vector2_field<D: UnitDimension>(
        &mut self,
        name: &'static str,
    ) -> Result<FieldPort<D, super::Vector2>, PortRegistryError> {
        self.register_field::<D, super::Vector2>(name)
    }

    /// Register a Vector3 field port.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is not found in the state layout
    /// or if its kind doesn't match Vector3.
    pub fn register_vector3_field<D: UnitDimension>(
        &mut self,
        name: &'static str,
    ) -> Result<FieldPort<D, super::Vector3>, PortRegistryError> {
        self.register_field::<D, super::Vector3>(name)
    }

    /// Register a field port with arbitrary kind.
    ///
    /// # Type Parameters
    ///
    /// - `D`: The expected physical dimension
    /// - `K`: The expected field kind
    pub fn register_field<D: UnitDimension, K: FieldKind>(
        &mut self,
        name: &'static str,
    ) -> Result<FieldPort<D, K>, PortRegistryError> {
        let field =
            self.state_layout
                .field(name)
                .ok_or_else(|| PortRegistryError::FieldNotFound {
                    name: name.to_string(),
                })?;

        // Verify field kind matches
        let expected_components = K::COMPONENT_COUNT;
        let actual_components = field.component_count();
        if expected_components != actual_components {
            return Err(PortRegistryError::FieldKindMismatch {
                name: name.to_string(),
                expected: expected_components,
                found: actual_components,
            });
        }

        // Copy all data we need before mutable borrow
        let offset = field.offset();
        let stride = self.state_layout.stride();
        let runtime_dim = field.unit();

        let id = self.allocate_id();

        self.field_ports.insert(
            id,
            FieldPortEntry {
                name: name.to_string(),
                offset,
                stride,
                component_count: actual_components,
                runtime_dim,
            },
        );

        Ok(FieldPort::new(id, name, offset, stride, runtime_dim))
    }

    /// Register a parameter port.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The parameter type
    /// - `D`: The physical dimension
    pub fn register_param<T: ParamType, D: UnitDimension>(
        &mut self,
        key: &'static str,
        wgsl_field_name: &'static str,
    ) -> ParamPort<T, D> {
        let id = self.allocate_id();
        let runtime_dim = D::to_runtime();

        self.param_ports.insert(
            id,
            ParamPortEntry {
                key: key.to_string(),
                wgsl_field: wgsl_field_name.to_string(),
                runtime_dim,
            },
        );

        ParamPort::new(id, key, wgsl_field_name, runtime_dim)
    }

    /// Register a buffer port.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The buffer element type
    /// - `A`: The access mode
    pub fn register_buffer<T: BufferType, A: AccessMode>(
        &mut self,
        name: &'static str,
        group: u32,
        binding: u32,
    ) -> BufferPort<T, A> {
        let id = self.allocate_id();

        self.buffer_ports.insert(
            id,
            BufferPortEntry {
                name: name.to_string(),
                group,
                binding,
            },
        );

        BufferPort::new(id, name, group, binding)
    }

    /// Get the number of registered field ports.
    pub fn field_port_count(&self) -> usize {
        self.field_ports.len()
    }

    /// Get the number of registered parameter ports.
    pub fn param_port_count(&self) -> usize {
        self.param_ports.len()
    }

    /// Get the number of registered buffer ports.
    pub fn buffer_port_count(&self) -> usize {
        self.buffer_ports.len()
    }
}

/// Typed wrapper around PortRegistry with type-safe lookup methods.
///
/// This provides a more ergonomic API for looking up ports with proper
/// type checking.
pub struct TypedPortRegistry<'a> {
    registry: &'a PortRegistry,
}

impl<'a> TypedPortRegistry<'a> {
    /// Create a new typed registry wrapper.
    pub fn new(registry: &'a PortRegistry) -> Self {
        Self { registry }
    }

    /// Look up a scalar field port by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is not found or has the wrong kind.
    pub fn scalar_field<D: UnitDimension>(
        &self,
        name: &str,
    ) -> Result<FieldPort<D, super::Scalar>, PortRegistryError> {
        self.field::<D, super::Scalar>(name)
    }

    /// Look up a Vector2 field port by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is not found or has the wrong kind.
    pub fn vector2_field<D: UnitDimension>(
        &self,
        name: &str,
    ) -> Result<FieldPort<D, super::Vector2>, PortRegistryError> {
        self.field::<D, super::Vector2>(name)
    }

    /// Look up a Vector3 field port by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is not found or has the wrong kind.
    pub fn vector3_field<D: UnitDimension>(
        &self,
        name: &str,
    ) -> Result<FieldPort<D, super::Vector3>, PortRegistryError> {
        self.field::<D, super::Vector3>(name)
    }

    /// Look up a field port by name with specific dimension and kind.
    fn field<D: UnitDimension, K: FieldKind>(
        &self,
        name: &str,
    ) -> Result<FieldPort<D, K>, PortRegistryError> {
        let field = self.registry.state_layout().field(name).ok_or_else(|| {
            PortRegistryError::FieldNotFound {
                name: name.to_string(),
            }
        })?;

        let expected_components = K::COMPONENT_COUNT;
        let actual_components = field.component_count();
        if expected_components != actual_components {
            return Err(PortRegistryError::FieldKindMismatch {
                name: name.to_string(),
                expected: expected_components,
                found: actual_components,
            });
        }

        let offset = field.offset();
        let stride = self.registry.state_layout().stride();
        let runtime_dim = field.unit();

        // We use a temporary ID since this is a lookup, not registration
        Ok(FieldPort::new(
            PortId::new(0),
            "",
            offset,
            stride,
            runtime_dim,
        ))
    }
}

/// Error type for port registry operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PortRegistryError {
    /// Field not found in state layout.
    FieldNotFound { name: String },
    /// Field kind mismatch.
    FieldKindMismatch {
        name: String,
        expected: u32,
        found: u32,
    },
}

impl std::fmt::Display for PortRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortRegistryError::FieldNotFound { name } => {
                write!(f, "Field '{}' not found in state layout", name)
            }
            PortRegistryError::FieldKindMismatch {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Field '{}' has {} component(s), expected {}",
                    name, found, expected
                )
            }
        }
    }
}

impl std::error::Error for PortRegistryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{vol_scalar, vol_vector};
    use crate::solver::model::ports::dimensions::{Length, Pressure, Time, Velocity};
    use crate::solver::model::ports::{BufferF32, ReadOnly, ReadWrite, Scalar, Vector2, F32};
    use crate::solver::model::ports::{Port, WgslPort};
    use crate::solver::units::si;

    fn create_test_layout() -> StateLayout {
        StateLayout::new(vec![
            vol_vector("U", si::VELOCITY),
            vol_scalar("p", si::PRESSURE),
            vol_scalar("d_p", si::D_P),
        ])
    }

    #[test]
    fn register_scalar_field_success() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let p = registry
            .register_scalar_field::<Pressure>("p")
            .expect("should register p");

        assert_eq!(p.name(), "p");
        assert_eq!(p.offset(), 2); // After U (2 components)
        assert_eq!(p.stride(), 4); // U(2) + p(1) + d_p(1)
    }

    #[test]
    fn register_vector2_field_success() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let u = registry
            .register_vector2_field::<Velocity>("U")
            .expect("should register U");

        assert_eq!(u.name(), "U");
        assert_eq!(u.offset(), 0);
        assert_eq!(u.stride(), 4);
    }

    #[test]
    fn register_field_wrong_kind() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let err = registry
            .register_scalar_field::<Pressure>("U")
            .expect_err("should fail");

        match err {
            PortRegistryError::FieldKindMismatch {
                name,
                expected,
                found,
            } => {
                assert_eq!(name, "U");
                assert_eq!(expected, 1);
                assert_eq!(found, 2);
            }
            _ => panic!("expected FieldKindMismatch"),
        }
    }

    #[test]
    fn register_field_not_found() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let err = registry
            .register_scalar_field::<Pressure>("nonexistent")
            .expect_err("should fail");

        match err {
            PortRegistryError::FieldNotFound { name } => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("expected FieldNotFound"),
        }
    }

    #[test]
    fn register_param_port() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let dt = registry.register_param::<F32, Time>("dt", "dt");

        assert_eq!(dt.key(), "dt");
        assert_eq!(dt.wgsl_field_name(), "dt");
        assert_eq!(registry.param_port_count(), 1);
    }

    #[test]
    fn register_buffer_port() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let state = registry.register_buffer::<BufferF32, ReadWrite>("state", 1, 0);

        assert_eq!(state.name(), "state");
        assert_eq!(state.group(), 1);
        assert_eq!(state.binding(), 0);
        assert!(state.allows_write());
        assert_eq!(registry.buffer_port_count(), 1);
    }

    #[test]
    fn multiple_ports_get_unique_ids() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let p = registry.register_scalar_field::<Pressure>("p").unwrap();
        let dt = registry.register_param::<F32, Time>("dt", "dt");
        let state = registry.register_buffer::<BufferF32, ReadWrite>("state", 1, 0);

        assert_ne!(p.id(), dt.id());
        assert_ne!(dt.id(), state.id());
        assert_ne!(p.id(), state.id());
    }

    #[test]
    fn field_port_wgsl_generation() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let u = registry.register_vector2_field::<Velocity>("U").unwrap();

        assert_eq!(u.wgsl_type(), "vec2<f32>");
        assert_eq!(u.linear_index("idx"), "idx * 4u + 0u");
        assert_eq!(u.wgsl_access(Some("cell_idx")), "state[cell_idx * 4u + 0u]");
    }
}
