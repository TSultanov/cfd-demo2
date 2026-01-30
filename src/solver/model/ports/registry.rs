//! Port registry for runtime management of ports.
//!
//! The port registry manages the creation and storage of field, parameter,
//! and buffer ports, mapping them to their underlying resources.

use super::{
    AccessMode, BufferPort, BufferType, FieldKind, FieldPort, ParamPort, ParamType, PortId,
};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::ports::dimensions::UnitDimension;
use crate::solver::model::ports::params::ParamTypeKind;
use crate::solver::units::UnitDim;
use std::collections::HashMap;

/// Central registry for all ports in a model.
///
/// The `PortRegistry` manages the creation of field, parameter, and buffer ports,
/// ensuring unique IDs and proper mapping to underlying resources.
///
/// Registration is idempotent: registering the same port multiple times with the
/// same spec returns the existing port. Registering with a conflicting spec returns
/// an error.
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
#[derive(Debug, Clone)]
pub struct PortRegistry {
    next_id: u32,
    state_layout: StateLayout,
    field_ports: HashMap<PortId, FieldPortEntry>,
    param_ports: HashMap<PortId, ParamPortEntry>,
    buffer_ports: HashMap<PortId, BufferPortEntry>,
    // Idempotency indexes: name/key -> PortId
    field_name_to_id: HashMap<String, PortId>,
    param_key_to_id: HashMap<String, PortId>,
    buffer_key_to_id: HashMap<BufferKey, PortId>,
}

/// Key for buffer lookup: (name, group, binding)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferKey {
    name: String,
    group: u32,
    binding: u32,
}

/// Storage for field port metadata.
#[derive(Debug, Clone)]
struct FieldPortEntry {
    id: PortId,
    name: String,
    offset: u32,
    stride: u32,
    component_count: u32,
    runtime_dim: UnitDim,
}

/// Storage for parameter port metadata.
#[derive(Debug, Clone)]
struct ParamPortEntry {
    id: PortId,
    key: String,
    wgsl_field: String,
    runtime_dim: UnitDim,
    param_type: ParamTypeKind,
}

/// Storage for buffer port metadata.
#[derive(Debug, Clone)]
struct BufferPortEntry {
    id: PortId,
    name: String,
    group: u32,
    binding: u32,
    buffer_type: BufferTypeKind,
    access_mode: AccessModeKind,
}

/// Enum representing buffer element types for storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferTypeKind {
    F32,
    U32,
    I32,
    Vec2F32,
    Vec3F32,
}

impl BufferTypeKind {
    fn from_type<T: BufferType>() -> Self {
        use super::{BufferF32, BufferI32, BufferU32, BufferVec2F32, BufferVec3F32};
        // This is a bit hacky but works for our marker types
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<BufferF32>() {
            BufferTypeKind::F32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<BufferU32>() {
            BufferTypeKind::U32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<BufferI32>() {
            BufferTypeKind::I32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<BufferVec2F32>() {
            BufferTypeKind::Vec2F32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<BufferVec3F32>() {
            BufferTypeKind::Vec3F32
        } else {
            panic!("Unknown buffer type")
        }
    }

    /// Parse a WGSL element type string into a BufferTypeKind.
    pub fn from_wgsl_type(elem_type: &str) -> Option<Self> {
        match elem_type {
            "f32" => Some(BufferTypeKind::F32),
            "u32" => Some(BufferTypeKind::U32),
            "i32" => Some(BufferTypeKind::I32),
            "vec2<f32>" => Some(BufferTypeKind::Vec2F32),
            "vec3<f32>" => Some(BufferTypeKind::Vec3F32),
            _ => None,
        }
    }
}

/// Enum representing access modes for storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessModeKind {
    ReadOnly,
    ReadWrite,
}

impl AccessModeKind {
    fn from_mode<A: AccessMode>() -> Self {
        use super::{ReadOnly, ReadWrite};
        if std::any::TypeId::of::<A>() == std::any::TypeId::of::<ReadOnly>() {
            AccessModeKind::ReadOnly
        } else {
            AccessModeKind::ReadWrite
        }
    }
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
            field_name_to_id: HashMap::new(),
            param_key_to_id: HashMap::new(),
            buffer_key_to_id: HashMap::new(),
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
    /// Idempotent: if a field with this name is already registered with the same
    /// spec, returns the existing port. If registered with a conflicting spec,
    /// returns an error.
    ///
    /// # Type Parameters
    ///
    /// - `D`: The expected physical dimension
    /// - `K`: The expected field kind
    pub fn register_field<D: UnitDimension, K: FieldKind>(
        &mut self,
        name: &'static str,
    ) -> Result<FieldPort<D, K>, PortRegistryError> {
        // Check if already registered
        if let Some(&existing_id) = self.field_name_to_id.get(name) {
            let entry = self.field_ports.get(&existing_id).expect("entry exists");
            // Verify kind matches
            let expected_components = K::COMPONENT_COUNT;
            if entry.component_count != expected_components {
                return Err(PortRegistryError::FieldSpecConflict {
                    name: name.to_string(),
                    expected_kind: expected_components,
                    registered_kind: entry.component_count,
                });
            }
            // Return existing port
            return Ok(FieldPort::new(
                existing_id,
                name,
                entry.offset,
                entry.stride,
                entry.runtime_dim,
            ));
        }

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
                id,
                name: name.to_string(),
                offset,
                stride,
                component_count: actual_components,
                runtime_dim,
            },
        );
        self.field_name_to_id.insert(name.to_string(), id);

        Ok(FieldPort::new(id, name, offset, stride, runtime_dim))
    }

    /// Register a parameter port.
    ///
    /// Idempotent: if a parameter with this key is already registered with the same
    /// spec, returns the existing port. If registered with a conflicting spec,
    /// returns an error.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The parameter type
    /// - `D`: The physical dimension
    pub fn register_param<T: ParamType, D: UnitDimension>(
        &mut self,
        key: &'static str,
        wgsl_field_name: &'static str,
    ) -> Result<ParamPort<T, D>, PortRegistryError> {
        // Check if already registered
        if let Some(&existing_id) = self.param_key_to_id.get(key) {
            let entry = self.param_ports.get(&existing_id).expect("entry exists");
            // Verify wgsl_field matches
            if entry.wgsl_field != wgsl_field_name {
                return Err(PortRegistryError::ParamSpecConflict {
                    key: key.to_string(),
                    expected_wgsl: wgsl_field_name.to_string(),
                    registered_wgsl: entry.wgsl_field.clone(),
                });
            }
            // Verify type matches
            let expected_type = ParamTypeKind::from_type::<T>();
            if entry.param_type != expected_type {
                return Err(PortRegistryError::ParamTypeConflict {
                    key: key.to_string(),
                    expected_type,
                    registered_type: entry.param_type,
                });
            }
            // Return existing port
            return Ok(ParamPort::new(
                existing_id,
                key,
                wgsl_field_name,
                entry.runtime_dim,
            ));
        }

        let id = self.allocate_id();
        let runtime_dim = D::to_runtime();

        self.param_ports.insert(
            id,
            ParamPortEntry {
                id,
                key: key.to_string(),
                wgsl_field: wgsl_field_name.to_string(),
                runtime_dim,
                param_type: ParamTypeKind::from_type::<T>(),
            },
        );
        self.param_key_to_id.insert(key.to_string(), id);

        Ok(ParamPort::new(id, key, wgsl_field_name, runtime_dim))
    }

    /// Register a buffer port.
    ///
    /// Idempotent: if a buffer with this name/group/binding is already registered
    /// with the same spec, returns the existing port. If registered with a
    /// conflicting spec, returns an error.
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
    ) -> Result<BufferPort<T, A>, PortRegistryError> {
        let buffer_key = BufferKey {
            name: name.to_string(),
            group,
            binding,
        };

        // Check if already registered
        if let Some(&existing_id) = self.buffer_key_to_id.get(&buffer_key) {
            let entry = self.buffer_ports.get(&existing_id).expect("entry exists");
            // Verify type matches
            let expected_type = BufferTypeKind::from_type::<T>();
            if entry.buffer_type != expected_type {
                return Err(PortRegistryError::BufferTypeConflict {
                    name: name.to_string(),
                    group,
                    binding,
                    expected_type,
                    registered_type: entry.buffer_type,
                });
            }
            // Verify access mode matches
            let expected_mode = AccessModeKind::from_mode::<A>();
            if entry.access_mode != expected_mode {
                return Err(PortRegistryError::BufferAccessConflict {
                    name: name.to_string(),
                    group,
                    binding,
                    expected_mode,
                    registered_mode: entry.access_mode,
                });
            }
            // Return existing port
            return Ok(BufferPort::new(existing_id, name, group, binding));
        }

        let id = self.allocate_id();

        self.buffer_ports.insert(
            id,
            BufferPortEntry {
                id,
                name: name.to_string(),
                group,
                binding,
                buffer_type: BufferTypeKind::from_type::<T>(),
                access_mode: AccessModeKind::from_mode::<A>(),
            },
        );
        self.buffer_key_to_id.insert(buffer_key, id);

        Ok(BufferPort::new(id, name, group, binding))
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

    /// Look up a field port by name.
    ///
    /// Returns `Some(port_id)` if a field with this name has been registered.
    pub fn lookup_field(&self, name: &str) -> Option<PortId> {
        self.field_name_to_id.get(name).copied()
    }

    /// Look up a parameter port by key.
    ///
    /// Returns `Some(port_id)` if a parameter with this key has been registered.
    pub fn lookup_param(&self, key: &str) -> Option<PortId> {
        self.param_key_to_id.get(key).copied()
    }

    /// Look up a buffer port by name/group/binding.
    ///
    /// Returns `Some(port_id)` if a buffer with this key has been registered.
    pub fn lookup_buffer(&self, name: &str, group: u32, binding: u32) -> Option<PortId> {
        let key = BufferKey {
            name: name.to_string(),
            group,
            binding,
        };
        self.buffer_key_to_id.get(&key).copied()
    }

    /// Get a field port entry by ID.
    pub fn get_field_entry(&self, id: PortId) -> Option<&FieldPortEntry> {
        self.field_ports.get(&id)
    }

    /// Get a parameter port entry by ID.
    pub fn get_param_entry(&self, id: PortId) -> Option<&ParamPortEntry> {
        self.param_ports.get(&id)
    }

    /// Get a buffer port entry by ID.
    pub fn get_buffer_entry(&self, id: PortId) -> Option<&BufferPortEntry> {
        self.buffer_ports.get(&id)
    }

    /// Register all ports from a manifest.
    ///
    /// This method iterates through all params, fields, and buffers in the manifest
    /// and registers them idempotently. Field validation is performed against the
    /// state layout.
    ///
    /// # Arguments
    ///
    /// * `module_name` - The name of the module providing this manifest (for error context)
    /// * `manifest` - The port manifest to register
    ///
    /// # Errors
    ///
    /// Returns an error if any port cannot be registered or if validation fails.
    pub fn register_manifest(
        &mut self,
        module_name: &str,
        manifest: &crate::solver::ir::ports::PortManifest,
    ) -> Result<(), PortRegistryError> {
        // Register params
        for param in &manifest.params {
            self.register_param_from_spec(module_name, param)?;
        }

        // Register fields
        for field in &manifest.fields {
            self.register_field_from_spec(module_name, field)?;
        }

        // Register buffers
        for buffer in &manifest.buffers {
            self.register_buffer_from_spec(module_name, buffer)?;
        }

        Ok(())
    }

    /// Register a parameter from a ParamSpec.
    fn register_param_from_spec(
        &mut self,
        module_name: &str,
        param: &crate::solver::ir::ports::ParamSpec,
    ) -> Result<(), PortRegistryError> {
        let param_type = ParamTypeKind::from_wgsl_type(param.wgsl_type).ok_or_else(|| {
            PortRegistryError::InvalidParamType {
                module: module_name.to_string(),
                key: param.key.to_string(),
                wgsl_type: param.wgsl_type.to_string(),
            }
        })?;

        // Check if already registered
        if let Some(&existing_id) = self.param_key_to_id.get(param.key) {
            let entry = self.param_ports.get(&existing_id).expect("entry exists");
            // Verify wgsl_field matches
            if entry.wgsl_field != param.wgsl_field {
                return Err(PortRegistryError::ParamSpecConflict {
                    key: format!("{} (from module '{}')", param.key, module_name),
                    expected_wgsl: param.wgsl_field.to_string(),
                    registered_wgsl: entry.wgsl_field.clone(),
                });
            }
            // Verify type matches
            if entry.param_type != param_type {
                return Err(PortRegistryError::ParamTypeConflict {
                    key: format!("{} (from module '{}')", param.key, module_name),
                    expected_type: param_type,
                    registered_type: entry.param_type,
                });
            }
            // Verify unit matches (if we want strict unit checking)
            if entry.runtime_dim != param.unit {
                // For now, just warn - we may want to make this an error later
                // This allows for "dynamic dimensions" where the same param key
                // can have different units in different contexts
            }
            return Ok(());
        }

        let id = self.allocate_id();

        self.param_ports.insert(
            id,
            ParamPortEntry {
                id,
                key: param.key.to_string(),
                wgsl_field: param.wgsl_field.to_string(),
                runtime_dim: param.unit,
                param_type,
            },
        );
        self.param_key_to_id.insert(param.key.to_string(), id);

        Ok(())
    }

    /// Register a field from a FieldSpec.
    fn register_field_from_spec(
        &mut self,
        module_name: &str,
        field: &crate::solver::ir::ports::FieldSpec,
    ) -> Result<(), PortRegistryError> {
        // Check if already registered
        if let Some(&existing_id) = self.field_name_to_id.get(field.name) {
            let entry = self.field_ports.get(&existing_id).expect("entry exists");
            // Verify kind matches
            let expected_components = field.kind.component_count();
            if entry.component_count != expected_components {
                return Err(PortRegistryError::FieldSpecConflict {
                    name: format!("{} (from module '{}')", field.name, module_name),
                    expected_kind: expected_components,
                    registered_kind: entry.component_count,
                });
            }
            // Unit validation - warn on mismatch for now
            if entry.runtime_dim != field.unit {
                // Allow dynamic dimensions for now
            }
            return Ok(());
        }

        // Validate field exists in state layout
        let (offset, stride, actual_components, runtime_dim) = {
            let layout_field = self.state_layout.field(field.name).ok_or_else(|| {
                PortRegistryError::FieldNotFound {
                    name: format!("{} (required by module '{}')", field.name, module_name),
                }
            })?;

            // Validate field kind matches
            let expected_components = field.kind.component_count();
            let actual_components = layout_field.component_count();
            if expected_components != actual_components {
                return Err(PortRegistryError::FieldKindMismatch {
                    name: format!("{} (from module '{}')", field.name, module_name),
                    expected: expected_components,
                    found: actual_components,
                });
            }

            // Validate unit matches (optional for now)
            let runtime_dim = layout_field.unit();
            if runtime_dim != field.unit {
                // For now, allow unit mismatches - this supports "dynamic dimensions"
                // where the same field name can have different units in different models
            }

            (
                layout_field.offset(),
                self.state_layout.stride(),
                actual_components,
                runtime_dim,
            )
        };

        let id = self.allocate_id();

        self.field_ports.insert(
            id,
            FieldPortEntry {
                id,
                name: field.name.to_string(),
                offset,
                stride,
                component_count: actual_components,
                runtime_dim,
            },
        );
        self.field_name_to_id.insert(field.name.to_string(), id);

        Ok(())
    }

    /// Register a buffer from a BufferSpec.
    fn register_buffer_from_spec(
        &mut self,
        module_name: &str,
        buffer: &crate::solver::ir::ports::BufferSpec,
    ) -> Result<(), PortRegistryError> {
        let buffer_type =
            BufferTypeKind::from_wgsl_type(buffer.elem_wgsl_type).ok_or_else(|| {
                PortRegistryError::InvalidBufferType {
                    module: module_name.to_string(),
                    name: buffer.name.to_string(),
                    elem_type: buffer.elem_wgsl_type.to_string(),
                }
            })?;

        let access_mode = match buffer.access {
            crate::solver::ir::ports::BufferAccess::ReadOnly => AccessModeKind::ReadOnly,
            crate::solver::ir::ports::BufferAccess::ReadWrite => AccessModeKind::ReadWrite,
        };

        let buffer_key = BufferKey {
            name: buffer.name.to_string(),
            group: buffer.group,
            binding: buffer.binding,
        };

        // Check if already registered
        if let Some(&existing_id) = self.buffer_key_to_id.get(&buffer_key) {
            let entry = self.buffer_ports.get(&existing_id).expect("entry exists");
            // Verify type matches
            if entry.buffer_type != buffer_type {
                return Err(PortRegistryError::BufferTypeConflict {
                    name: format!("{} (from module '{}')", buffer.name, module_name),
                    group: buffer.group,
                    binding: buffer.binding,
                    expected_type: buffer_type,
                    registered_type: entry.buffer_type,
                });
            }
            // Verify access mode matches
            if entry.access_mode != access_mode {
                return Err(PortRegistryError::BufferAccessConflict {
                    name: format!("{} (from module '{}')", buffer.name, module_name),
                    group: buffer.group,
                    binding: buffer.binding,
                    expected_mode: access_mode,
                    registered_mode: entry.access_mode,
                });
            }
            return Ok(());
        }

        let id = self.allocate_id();

        self.buffer_ports.insert(
            id,
            BufferPortEntry {
                id,
                name: buffer.name.to_string(),
                group: buffer.group,
                binding: buffer.binding,
                buffer_type,
                access_mode,
            },
        );
        self.buffer_key_to_id.insert(buffer_key, id);

        Ok(())
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
    /// Field spec conflict (re-registered with different kind).
    FieldSpecConflict {
        name: String,
        expected_kind: u32,
        registered_kind: u32,
    },
    /// Parameter spec conflict (re-registered with different wgsl field).
    ParamSpecConflict {
        key: String,
        expected_wgsl: String,
        registered_wgsl: String,
    },
    /// Parameter type conflict (re-registered with different type).
    ParamTypeConflict {
        key: String,
        expected_type: ParamTypeKind,
        registered_type: ParamTypeKind,
    },
    /// Buffer type conflict (re-registered with different type).
    BufferTypeConflict {
        name: String,
        group: u32,
        binding: u32,
        expected_type: BufferTypeKind,
        registered_type: BufferTypeKind,
    },
    /// Buffer access mode conflict (re-registered with different access).
    BufferAccessConflict {
        name: String,
        group: u32,
        binding: u32,
        expected_mode: AccessModeKind,
        registered_mode: AccessModeKind,
    },
    /// Invalid parameter type in manifest.
    InvalidParamType {
        module: String,
        key: String,
        wgsl_type: String,
    },
    /// Invalid buffer element type in manifest.
    InvalidBufferType {
        module: String,
        name: String,
        elem_type: String,
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
            PortRegistryError::FieldSpecConflict {
                name,
                expected_kind,
                registered_kind,
            } => {
                write!(
                    f,
                    "Field '{}' already registered with {} component(s), expected {}",
                    name, registered_kind, expected_kind
                )
            }
            PortRegistryError::ParamSpecConflict {
                key,
                expected_wgsl,
                registered_wgsl,
            } => {
                write!(
                    f,
                    "Parameter '{}' already registered with wgsl field '{}', expected '{}'",
                    key, registered_wgsl, expected_wgsl
                )
            }
            PortRegistryError::ParamTypeConflict {
                key,
                expected_type,
                registered_type,
            } => {
                write!(
                    f,
                    "Parameter '{}' already registered as {:?}, expected {:?}",
                    key, registered_type, expected_type
                )
            }
            PortRegistryError::BufferTypeConflict {
                name,
                group,
                binding,
                expected_type,
                registered_type,
            } => {
                write!(
                    f,
                    "Buffer '{}' (group={}, binding={}) already registered as {:?}, expected {:?}",
                    name, group, binding, registered_type, expected_type
                )
            }
            PortRegistryError::BufferAccessConflict {
                name,
                group,
                binding,
                expected_mode,
                registered_mode,
            } => {
                write!(
                    f,
                    "Buffer '{}' (group={}, binding={}) already registered with {:?} access, expected {:?}",
                    name, group, binding, registered_mode, expected_mode
                )
            }
            PortRegistryError::InvalidParamType {
                module,
                key,
                wgsl_type,
            } => {
                write!(
                    f,
                    "Module '{}' defines parameter '{}' with unsupported WGSL type '{}'",
                    module, key, wgsl_type
                )
            }
            PortRegistryError::InvalidBufferType {
                module,
                name,
                elem_type,
            } => {
                write!(
                    f,
                    "Module '{}' defines buffer '{}' with unsupported element type '{}'",
                    module, name, elem_type
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
    use crate::solver::model::ports::{
        BufferF32, BufferU32, ReadOnly, ReadWrite, Scalar, Vector2, F32,
    };
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

        let dt = registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("should register dt");

        assert_eq!(dt.key(), "dt");
        assert_eq!(dt.wgsl_field_name(), "dt");
        assert_eq!(registry.param_port_count(), 1);
    }

    #[test]
    fn register_buffer_port() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let state = registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state");

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
        let dt = registry.register_param::<F32, Time>("dt", "dt").unwrap();
        let state = registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .unwrap();

        assert_ne!(p.id(), dt.id());
        assert_ne!(dt.id(), state.id());
        assert_ne!(p.id(), state.id());
    }

    #[test]
    fn register_field_idempotent() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // First registration
        let p1 = registry
            .register_scalar_field::<Pressure>("p")
            .expect("should register p");

        // Second registration with same spec returns same port
        let p2 = registry
            .register_scalar_field::<Pressure>("p")
            .expect("should return existing p");

        assert_eq!(p1.id(), p2.id());
        assert_eq!(registry.field_port_count(), 1);
    }

    #[test]
    fn register_field_conflict() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // Register as scalar
        registry
            .register_scalar_field::<Pressure>("p")
            .expect("should register p as scalar");

        // Try to register same field as vector - should fail
        let err = registry
            .register_vector2_field::<Pressure>("p")
            .expect_err("should fail with conflict");

        match err {
            PortRegistryError::FieldSpecConflict {
                name,
                expected_kind,
                registered_kind,
            } => {
                assert_eq!(name, "p");
                assert_eq!(expected_kind, 2); // Vector2
                assert_eq!(registered_kind, 1); // Scalar
            }
            _ => panic!("expected FieldSpecConflict, got {:?}", err),
        }
    }

    #[test]
    fn register_param_idempotent() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // First registration
        let dt1 = registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("should register dt");

        // Second registration with same spec returns same port
        let dt2 = registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("should return existing dt");

        assert_eq!(dt1.id(), dt2.id());
        assert_eq!(registry.param_port_count(), 1);
    }

    #[test]
    fn register_param_conflict() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // Register with wgsl field "dt"
        registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("should register dt");

        // Try to register same key with different wgsl field - should fail
        let err = registry
            .register_param::<F32, Time>("dt", "delta_t")
            .expect_err("should fail with conflict");

        match err {
            PortRegistryError::ParamSpecConflict {
                key,
                expected_wgsl,
                registered_wgsl,
            } => {
                assert_eq!(key, "dt");
                assert_eq!(expected_wgsl, "delta_t");
                assert_eq!(registered_wgsl, "dt");
            }
            _ => panic!("expected ParamSpecConflict, got {:?}", err),
        }
    }

    #[test]
    fn register_buffer_idempotent() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // First registration
        let state1 = registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state");

        // Second registration with same spec returns same port
        let state2 = registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should return existing state");

        assert_eq!(state1.id(), state2.id());
        assert_eq!(registry.buffer_port_count(), 1);
    }

    #[test]
    fn register_buffer_type_conflict() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // Register as f32
        registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state as f32");

        // Try to register same buffer as u32 - should fail
        let err = registry
            .register_buffer::<BufferU32, ReadWrite>("state", 1, 0)
            .expect_err("should fail with conflict");

        match err {
            PortRegistryError::BufferTypeConflict {
                name,
                group,
                binding,
                expected_type: _,
                registered_type: _,
            } => {
                assert_eq!(name, "state");
                assert_eq!(group, 1);
                assert_eq!(binding, 0);
            }
            _ => panic!("expected BufferTypeConflict, got {:?}", err),
        }
    }

    #[test]
    fn register_buffer_access_conflict() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // Register as read-write
        registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state as read-write");

        // Try to register same buffer as read-only - should fail
        let err = registry
            .register_buffer::<BufferF32, ReadOnly>("state", 1, 0)
            .expect_err("should fail with conflict");

        match err {
            PortRegistryError::BufferAccessConflict {
                name,
                group,
                binding,
                expected_mode: _,
                registered_mode: _,
            } => {
                assert_eq!(name, "state");
                assert_eq!(group, 1);
                assert_eq!(binding, 0);
            }
            _ => panic!("expected BufferAccessConflict, got {:?}", err),
        }
    }

    #[test]
    fn lookup_ports() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let p = registry
            .register_scalar_field::<Pressure>("p")
            .expect("should register p");
        let dt = registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("should register dt");
        let state = registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state");

        // Lookup by name/key
        assert_eq!(registry.lookup_field("p"), Some(p.id()));
        assert_eq!(registry.lookup_param("dt"), Some(dt.id()));
        assert_eq!(registry.lookup_buffer("state", 1, 0), Some(state.id()));

        // Lookup non-existent
        assert_eq!(registry.lookup_field("nonexistent"), None);
        assert_eq!(registry.lookup_param("nonexistent"), None);
        assert_eq!(registry.lookup_buffer("nonexistent", 1, 0), None);
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

    #[test]
    fn register_manifest_success() {
        use crate::solver::ir::ports::{
            BufferAccess, BufferSpec, FieldSpec, ParamSpec, PortFieldKind, PortManifest,
        };

        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let manifest = PortManifest {
            params: vec![ParamSpec {
                key: "test.dt",
                wgsl_field: "dt",
                wgsl_type: "f32",
                unit: si::TIME,
            }],
            fields: vec![FieldSpec {
                name: "p",
                kind: PortFieldKind::Scalar,
                unit: si::PRESSURE,
            }],
            buffers: vec![BufferSpec {
                name: "test_buffer",
                group: 1,
                binding: 0,
                elem_wgsl_type: "f32",
                access: BufferAccess::ReadWrite,
            }],
        };

        registry
            .register_manifest("test_module", &manifest)
            .expect("should register manifest");

        assert_eq!(registry.param_port_count(), 1);
        assert_eq!(registry.field_port_count(), 1);
        assert_eq!(registry.buffer_port_count(), 1);

        // Verify params
        let dt_id = registry
            .lookup_param("test.dt")
            .expect("should find dt param");
        let dt_entry = registry.get_param_entry(dt_id).unwrap();
        assert_eq!(dt_entry.key, "test.dt");
        assert_eq!(dt_entry.wgsl_field, "dt");

        // Verify fields
        let p_id = registry.lookup_field("p").expect("should find p field");
        let p_entry = registry.get_field_entry(p_id).unwrap();
        assert_eq!(p_entry.name, "p");
        assert_eq!(p_entry.component_count, 1);

        // Verify buffers
        let buf_id = registry
            .lookup_buffer("test_buffer", 1, 0)
            .expect("should find buffer");
        let buf_entry = registry.get_buffer_entry(buf_id).unwrap();
        assert_eq!(buf_entry.name, "test_buffer");
        assert_eq!(buf_entry.group, 1);
        assert_eq!(buf_entry.binding, 0);
    }

    #[test]
    fn register_manifest_idempotent() {
        use crate::solver::ir::ports::{ParamSpec, PortManifest};

        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let manifest = PortManifest {
            params: vec![ParamSpec {
                key: "test.param",
                wgsl_field: "param",
                wgsl_type: "f32",
                unit: si::DIMENSIONLESS,
            }],
            fields: vec![],
            buffers: vec![],
        };

        // First registration
        registry
            .register_manifest("module1", &manifest)
            .expect("should register first time");
        assert_eq!(registry.param_port_count(), 1);

        // Second registration with same spec should succeed (idempotent)
        registry
            .register_manifest("module2", &manifest)
            .expect("should be idempotent");
        assert_eq!(registry.param_port_count(), 1);
    }

    #[test]
    fn register_manifest_field_not_found() {
        use crate::solver::ir::ports::{FieldSpec, PortFieldKind, PortManifest};

        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let manifest = PortManifest {
            params: vec![],
            fields: vec![FieldSpec {
                name: "nonexistent_field",
                kind: PortFieldKind::Scalar,
                unit: si::PRESSURE,
            }],
            buffers: vec![],
        };

        let err = registry
            .register_manifest("test_module", &manifest)
            .expect_err("should fail for missing field");

        match err {
            PortRegistryError::FieldNotFound { name } => {
                assert!(name.contains("nonexistent_field"));
                assert!(name.contains("test_module"));
            }
            _ => panic!("expected FieldNotFound, got {:?}", err),
        }
    }

    #[test]
    fn register_manifest_invalid_param_type() {
        use crate::solver::ir::ports::{ParamSpec, PortManifest};

        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let manifest = PortManifest {
            params: vec![ParamSpec {
                key: "test.param",
                wgsl_field: "param",
                wgsl_type: "vec2<f32>", // Invalid for params
                unit: si::DIMENSIONLESS,
            }],
            fields: vec![],
            buffers: vec![],
        };

        let err = registry
            .register_manifest("test_module", &manifest)
            .expect_err("should fail for invalid param type");

        match err {
            PortRegistryError::InvalidParamType {
                module,
                key,
                wgsl_type,
            } => {
                assert_eq!(module, "test_module");
                assert_eq!(key, "test.param");
                assert_eq!(wgsl_type, "vec2<f32>");
            }
            _ => panic!("expected InvalidParamType, got {:?}", err),
        }
    }

    #[test]
    fn eos_port_manifest_units() {
        use crate::solver::model::modules::eos_ports::eos_uniform_port_manifest;
        use crate::solver::units::si;

        let manifest = eos_uniform_port_manifest();

        assert_eq!(manifest.params.len(), 6);

        // Find each param and verify its unit
        let gamma = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.gamma")
            .unwrap();
        assert_eq!(gamma.unit, si::DIMENSIONLESS);

        let gm1 = manifest.params.iter().find(|p| p.key == "eos.gm1").unwrap();
        assert_eq!(gm1.unit, si::DIMENSIONLESS);

        let r = manifest.params.iter().find(|p| p.key == "eos.r").unwrap();
        // R = P/(rho*T) = (ML⁻¹T⁻²)/(ML⁻³·K) = L²T⁻²K⁻¹
        let expected_r = si::PRESSURE.div_dim(si::DENSITY).div_dim(si::TEMPERATURE);
        assert_eq!(r.unit, expected_r);

        let dp_drho = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.dp_drho")
            .unwrap();
        // dp/drho = P/rho = (ML⁻¹T⁻²)/(ML⁻³) = L²T⁻²
        let expected_dp_drho = si::PRESSURE.div_dim(si::DENSITY);
        assert_eq!(dp_drho.unit, expected_dp_drho);

        let p_offset = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.p_offset")
            .unwrap();
        assert_eq!(p_offset.unit, si::PRESSURE);

        let theta_ref = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.theta_ref")
            .unwrap();
        // theta = P/rho has units L²/T² (specific energy)
        assert_eq!(theta_ref.unit, expected_dp_drho);
    }
}
