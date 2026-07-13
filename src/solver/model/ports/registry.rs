// Port registry: creates and stores field, parameter, and buffer ports.

use super::{BufferPort, FieldKind, FieldPort, ParamPort, ParamType, PortId, PortValidationError};

use super::buffer::{AccessModeKind, BufferTypeKind, IntoAccessModeKind, IntoBufferTypeKind};

use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::ports::dimensions::{AnyDimension, UnitDimension};
use crate::solver::model::ports::params::ParamTypeKind;
use crate::solver::units::UnitDim;
use std::any::TypeId;
use std::collections::HashMap;

/// Check if the dimension type is `AnyDimension` (the escape hatch).
///
/// Returns `true` if `D` is `AnyDimension`, which signals that unit checks
/// should be skipped for this registration.
fn is_any_dimension<D: UnitDimension>() -> bool {
    TypeId::of::<D>() == TypeId::of::<AnyDimension>()
}

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
pub struct FieldPortEntry {
    id: PortId,
    name: String,
    offset: u32,
    stride: u32,
    component_count: u32,
    runtime_dim: UnitDim,
}

impl FieldPortEntry {
    /// Get the offset of this field within each cell's state.
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Get the stride between cells in the state array.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// Get the component count for this field.
    pub fn component_count(&self) -> u32 {
        self.component_count
    }

    /// Get the runtime physical dimension.
    pub fn runtime_dimension(&self) -> UnitDim {
        self.runtime_dim
    }

    /// Get the field name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the port ID.
    pub fn id(&self) -> PortId {
        self.id
    }
}

/// Storage for parameter port metadata.
#[derive(Debug, Clone)]
struct ParamPortEntry {
    wgsl_field: String,
    runtime_dim: UnitDim,
    param_type: ParamTypeKind,
}

/// Storage for buffer port metadata.
#[derive(Debug, Clone)]
struct BufferPortEntry {
    buffer_type: BufferTypeKind,
    access_mode: AccessModeKind,
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

    /// Create a new port registry from a list of field references.
    ///
    /// This is the preferred constructor: the `StateLayout` is built internally,
    /// so there is no separate layout object that can diverge from the registry.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let registry = PortRegistry::from_fields(vec![
    ///     vol_vector_dim::<Velocity>("U"),
    ///     vol_scalar_dim::<Pressure>("p"),
    /// ]);
    /// ```
    pub fn from_fields(fields: Vec<crate::solver::model::backend::ast::FieldRef>) -> Self {
        let state_layout = StateLayout::new(fields);
        Self::new(state_layout)
    }

    /// Consume the registry and return the underlying `StateLayout`.
    ///
    /// This is useful when transitioning from `PortRegistry::from_fields()` to
    /// sites that still require an owned `StateLayout` (e.g., `ModelSpec`).
    pub fn into_state_layout(self) -> StateLayout {
        self.state_layout
    }

    /// Get the state layout.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Validate that a scalar field exists with the expected kind and dimension.
    pub fn validate_scalar_field<D: UnitDimension>(
        &self,
        module: &'static str,
        name: &str,
    ) -> Result<(), PortValidationError> {
        self.validate_field::<D, super::Scalar>(module, name)
    }

    /// Validate that a Vector2 field exists with the expected kind and dimension.
    pub fn validate_vector2_field<D: UnitDimension>(
        &self,
        module: &'static str,
        name: &str,
    ) -> Result<(), PortValidationError> {
        self.validate_field::<D, super::Vector2>(module, name)
    }

    /// Validate that a Vector3 field exists with the expected kind and dimension.
    pub fn validate_vector3_field<D: UnitDimension>(
        &self,
        module: &'static str,
        name: &str,
    ) -> Result<(), PortValidationError> {
        self.validate_field::<D, super::Vector3>(module, name)
    }

    /// Validate that a parameter exists with the expected type.
    pub fn validate_param<T: ParamType>(
        &self,
        module: &'static str,
        key: &str,
    ) -> Result<(), PortValidationError> {
        let Some(&id) = self.param_key_to_id.get(key) else {
            return Err(PortValidationError::MissingParameter {
                module,
                param: key.to_string(),
            });
        };

        let entry = self.param_ports.get(&id).ok_or_else(|| {
            PortValidationError::InternalInconsistency {
                detail: format!("param port entry for key '{key}' (id={id:?}) not found after index lookup"),
            }
        })?;
        let expected = ParamTypeKind::from_type::<T>();
        if entry.param_type != expected {
            return Err(PortValidationError::ParameterTypeMismatch {
                param: key.to_string(),
                expected: expected.wgsl_type().to_string(),
                found: entry.param_type.wgsl_type().to_string(),
            });
        }

        Ok(())
    }

    fn validate_field<D: UnitDimension, K: FieldKind>(
        &self,
        module: &'static str,
        name: &str,
    ) -> Result<(), PortValidationError> {
        let field =
            self.state_layout
                .field(name)
                .ok_or_else(|| PortValidationError::MissingField {
                    module,
                    field: name.to_string(),
                })?;

        let expected_components = K::COMPONENT_COUNT;
        let actual_components = field.component_count();
        if expected_components != actual_components {
            return Err(PortValidationError::FieldKindMismatch {
                field: name.to_string(),
                expected: Self::field_kind_label(expected_components),
                found: Self::field_kind_label(actual_components),
            });
        }

        // Skip dimension check if using AnyDimension escape hatch
        if !is_any_dimension::<D>() {
            let expected_dim = D::to_runtime();
            let found_dim = field.unit();
            if expected_dim != found_dim {
                return Err(PortValidationError::DimensionMismatch {
                    field: name.to_string(),
                    expected: expected_dim.to_string(),
                    found: found_dim.to_string(),
                });
            }
        }

        Ok(())
    }

    fn field_kind_label(components: u32) -> String {
        match components {
            1 => "Scalar".to_string(),
            2 => "Vector2".to_string(),
            3 => "Vector3".to_string(),
            n => format!("Vector{n}"),
        }
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
        name: &str,
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
        name: &str,
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
        name: &str,
    ) -> Result<FieldPort<D, super::Vector3>, PortRegistryError> {
        self.register_field::<D, super::Vector3>(name)
    }

    /// Register a field from the StateLayout without returning a typed port.
    ///
    /// This is useful for bulk-registration of all state fields where the specific
    /// port type is not needed (e.g., during recipe initialization).
    ///
    /// # Errors
    ///
    /// Returns an error if the field is not found in the state layout.
    pub fn register_state_field(&mut self, name: &str) -> Result<(), PortRegistryError> {
        if self.field_name_to_id.contains_key(name) {
            return Ok(());
        }

        let field =
            self.state_layout
                .field(name)
                .ok_or_else(|| PortRegistryError::FieldNotFound {
                    name: name.to_string(),
                })?;

        // Copy all data we need before mutable borrow
        let offset = field.offset();
        let stride = self.state_layout.stride();
        let runtime_dim = field.unit();
        let component_count = field.component_count();

        let id = self.allocate_id();

        self.field_ports.insert(
            id,
            FieldPortEntry {
                id,
                name: name.to_string(),
                offset,
                stride,
                component_count,
                runtime_dim,
            },
        );
        self.field_name_to_id.insert(name.to_string(), id);

        Ok(())
    }

    /// Register a field port with arbitrary kind.
    ///
    /// Idempotent: if a field with this name is already registered with the same
    /// spec, returns the existing port. If registered with a conflicting spec,
    /// returns an error.
    ///
    /// The name is interned to obtain a `&'static str` for storage.
    ///
    /// # Type Parameters
    ///
    /// - `D`: The expected physical dimension
    /// - `K`: The expected field kind
    pub fn register_field<D: UnitDimension, K: FieldKind>(
        &mut self,
        name: &str,
    ) -> Result<FieldPort<D, K>, PortRegistryError> {
        let name = super::intern(name);

        if let Some(&existing_id) = self.field_name_to_id.get(name) {
            let entry = self.field_ports.get(&existing_id).ok_or_else(|| {
                PortRegistryError::InternalInconsistency {
                    detail: format!("field port entry for '{name}' (id={existing_id:?}) not found after index lookup"),
                }
            })?;
            let expected_components = K::COMPONENT_COUNT;
            if entry.component_count != expected_components {
                return Err(PortRegistryError::FieldSpecConflict {
                    name: name.to_string(),
                    expected_kind: expected_components,
                    registered_kind: entry.component_count,
                });
            }
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

        let expected_components = K::COMPONENT_COUNT;
        let actual_components = field.component_count();
        if expected_components != actual_components {
            return Err(PortRegistryError::FieldKindMismatch {
                name: name.to_string(),
                expected: expected_components,
                found: actual_components,
            });
        }

        // Verify unit dimension matches (unless using AnyDimension escape hatch)
        let runtime_dim = field.unit();
        if !is_any_dimension::<D>() {
            let expected_dim = D::to_runtime();
            if expected_dim != runtime_dim {
                return Err(PortRegistryError::FieldUnitMismatch {
                    name: name.to_string(),
                    expected: expected_dim,
                    found: runtime_dim,
                });
            }
        }

        // Copy all data we need before mutable borrow
        let offset = field.offset();
        let stride = self.state_layout.stride();

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
        if let Some(&existing_id) = self.param_key_to_id.get(key) {
            let entry = self
                .param_ports
                .get_mut(&existing_id)
                .ok_or_else(|| PortRegistryError::InternalInconsistency {
                    detail: format!("param port entry for key '{key}' (id={existing_id:?}) not found after index lookup"),
                })?;
            if entry.wgsl_field != wgsl_field_name {
                return Err(PortRegistryError::ParamSpecConflict {
                    key: key.to_string(),
                    expected_wgsl: wgsl_field_name.to_string(),
                    registered_wgsl: entry.wgsl_field.clone(),
                });
            }
            let expected_type = ParamTypeKind::from_type::<T>();
            if entry.param_type != expected_type {
                return Err(PortRegistryError::ParamTypeConflict {
                    key: key.to_string(),
                    expected_type,
                    registered_type: entry.param_type,
                });
            }

            // Skipped for AnyDimension; if the stored dim is AnyDimension, a later
            // concrete call resolves it.
            if !is_any_dimension::<D>() {
                let expected_dim = D::to_runtime();
                if entry.runtime_dim != expected_dim {
                    if entry.runtime_dim == AnyDimension::to_runtime() {
                        entry.runtime_dim = expected_dim;
                    } else {
                        return Err(PortRegistryError::ParamUnitMismatch {
                            key: key.to_string(),
                            expected: expected_dim,
                            found: entry.runtime_dim,
                        });
                    }
                }
            }
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
    pub fn register_buffer<T: IntoBufferTypeKind, A: IntoAccessModeKind>(
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

        if let Some(&existing_id) = self.buffer_key_to_id.get(&buffer_key) {
            let entry = self.buffer_ports.get(&existing_id).ok_or_else(|| {
                PortRegistryError::InternalInconsistency {
                    detail: format!("buffer port entry for '{name}' (id={existing_id:?}) not found after index lookup"),
                }
            })?;
            let expected_type = T::into_kind();
            if entry.buffer_type != expected_type {
                return Err(PortRegistryError::BufferTypeConflict {
                    name: name.to_string(),
                    group,
                    binding,
                    expected_type,
                    registered_type: entry.buffer_type,
                });
            }
            let expected_mode = A::into_kind();
            if entry.access_mode != expected_mode {
                return Err(PortRegistryError::BufferAccessConflict {
                    name: name.to_string(),
                    group,
                    binding,
                    expected_mode,
                    registered_mode: entry.access_mode,
                });
            }
            return Ok(BufferPort::new(existing_id, name, group, binding));
        }

        let id = self.allocate_id();

        self.buffer_ports.insert(
            id,
            BufferPortEntry {
                buffer_type: T::into_kind(),
                access_mode: A::into_kind(),
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

    /// Get a field port entry by name.
    ///
    /// Returns `Some(&FieldPortEntry)` if a field with this name has been registered.
    pub fn get_field_entry_by_name(&self, name: &str) -> Option<&FieldPortEntry> {
        self.field_name_to_id
            .get(name)
            .and_then(|id| self.field_ports.get(id))
    }

    /// Build a [`ResolvedStateSlotsSpec`] covering *all* fields in the underlying
    /// [`StateLayout`].
    ///
    /// This is the preferred way to obtain an IR-safe snapshot of the full state
    /// layout for codegen, replacing the pattern of manually walking `StateLayout`
    /// in a separate resolver pass.
    pub fn to_resolved_state_slots(
        &self,
    ) -> crate::solver::ir::ports::ResolvedStateSlotsSpec {
        use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec};

        let mut slots: Vec<ResolvedStateSlotSpec> = self
            .state_layout
            .fields()
            .iter()
            .map(|f| {
                let kind = match f.component_count() {
                    1 => PortFieldKind::Scalar,
                    2 => PortFieldKind::Vector2,
                    3 => PortFieldKind::Vector3,
                    n => panic!("unsupported component count {n} for field '{}'", f.name()),
                };
                ResolvedStateSlotSpec {
                    name: f.name().to_string(),
                    kind,
                    unit: f.unit(),
                    base_offset: f.offset(),
                }
            })
            .collect();
        slots.sort_by(|a, b| a.name.cmp(&b.name));

        crate::solver::ir::ports::ResolvedStateSlotsSpec {
            stride: self.state_layout.stride(),
            slots,
        }
    }

    /// Build a [`ResolvedStateSlotsSpec`] covering only the named fields.
    ///
    /// Returns an error if any requested field is not present in the state layout.
    pub fn to_resolved_state_slots_for(
        &self,
        field_names: &std::collections::HashSet<String>,
    ) -> Result<crate::solver::ir::ports::ResolvedStateSlotsSpec, String> {
        use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec};

        let mut slots = Vec::new();
        for name in field_names {
            let f = self
                .state_layout
                .field(name)
                .ok_or_else(|| format!("state field '{name}' not found in state layout"))?;
            let kind = match f.component_count() {
                1 => PortFieldKind::Scalar,
                2 => PortFieldKind::Vector2,
                3 => PortFieldKind::Vector3,
                n => {
                    return Err(format!(
                        "unsupported component count {n} for field '{name}'"
                    ))
                }
            };
            slots.push(ResolvedStateSlotSpec {
                name: name.clone(),
                kind,
                unit: f.unit(),
                base_offset: f.offset(),
            });
        }
        slots.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(crate::solver::ir::ports::ResolvedStateSlotsSpec {
            stride: self.state_layout.stride(),
            slots,
        })
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

        if let Some(&existing_id) = self.param_key_to_id.get(param.key) {
            let entry = self
                .param_ports
                .get_mut(&existing_id)
                .ok_or_else(|| PortRegistryError::InternalInconsistency {
                    detail: format!("param port entry for key '{}' (id={existing_id:?}) not found after index lookup", param.key),
                })?;
            if entry.wgsl_field != param.wgsl_field {
                return Err(PortRegistryError::ParamSpecConflict {
                    key: format!("{} (from module '{}')", param.key, module_name),
                    expected_wgsl: param.wgsl_field.to_string(),
                    registered_wgsl: entry.wgsl_field.clone(),
                });
            }
            if entry.param_type != param_type {
                return Err(PortRegistryError::ParamTypeConflict {
                    key: format!("{} (from module '{}')", param.key, module_name),
                    expected_type: param_type,
                    registered_type: entry.param_type,
                });
            }
            if entry.runtime_dim != param.unit {
                if entry.runtime_dim == AnyDimension::to_runtime() {
                    entry.runtime_dim = param.unit;
                } else {
                    return Err(PortRegistryError::ParamUnitMismatch {
                        key: format!("{} (from module '{}')", param.key, module_name),
                        expected: entry.runtime_dim,
                        found: param.unit,
                    });
                }
            }
            return Ok(());
        }

        let id = self.allocate_id();

        self.param_ports.insert(
            id,
            ParamPortEntry {
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
        if let Some(&existing_id) = self.field_name_to_id.get(field.name) {
            let entry = self.field_ports.get(&existing_id).ok_or_else(|| {
                PortRegistryError::InternalInconsistency {
                    detail: format!("field port entry for '{}' (id={existing_id:?}) not found after index lookup", field.name),
                }
            })?;
            let expected_components = field.kind.component_count();
            if entry.component_count != expected_components {
                return Err(PortRegistryError::FieldSpecConflict {
                    name: format!("{} (from module '{}')", field.name, module_name),
                    expected_kind: expected_components,
                    registered_kind: entry.component_count,
                });
            }
            // Unit validation (skip if field expects ANY_DIMENSION sentinel)
            if !crate::solver::ir::ports::is_any_dimension(field.unit)
                && entry.runtime_dim != field.unit
            {
                return Err(PortRegistryError::FieldUnitMismatch {
                    name: format!("{} (from module '{}')", field.name, module_name),
                    expected: entry.runtime_dim,
                    found: field.unit,
                });
            }
            return Ok(());
        }

        let (offset, stride, actual_components, runtime_dim) = {
            let layout_field = self.state_layout.field(field.name).ok_or_else(|| {
                PortRegistryError::FieldNotFound {
                    name: format!("{} (required by module '{}')", field.name, module_name),
                }
            })?;

            let expected_components = field.kind.component_count();
            let actual_components = layout_field.component_count();
            if expected_components != actual_components {
                return Err(PortRegistryError::FieldKindMismatch {
                    name: format!("{} (from module '{}')", field.name, module_name),
                    expected: expected_components,
                    found: actual_components,
                });
            }

            // Validate unit matches (skip if field expects ANY_DIMENSION sentinel)
            let runtime_dim = layout_field.unit();
            if !crate::solver::ir::ports::is_any_dimension(field.unit) && runtime_dim != field.unit
            {
                return Err(PortRegistryError::FieldUnitMismatch {
                    name: format!("{} (from module '{}')", field.name, module_name),
                    expected: field.unit,
                    found: runtime_dim,
                });
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

        if let Some(&existing_id) = self.buffer_key_to_id.get(&buffer_key) {
            let entry = self.buffer_ports.get(&existing_id).ok_or_else(|| {
                PortRegistryError::InternalInconsistency {
                    detail: format!("buffer port entry for '{}' (id={existing_id:?}) not found after index lookup", buffer.name),
                }
            })?;
            if entry.buffer_type != buffer_type {
                return Err(PortRegistryError::BufferTypeConflict {
                    name: format!("{} (from module '{}')", buffer.name, module_name),
                    group: buffer.group,
                    binding: buffer.binding,
                    expected_type: buffer_type,
                    registered_type: entry.buffer_type,
                });
            }
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
                buffer_type,
                access_mode,
            },
        );
        self.buffer_key_to_id.insert(buffer_key, id);

        Ok(())
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
    /// Field unit dimension mismatch.
    FieldUnitMismatch {
        name: String,
        expected: UnitDim,
        found: UnitDim,
    },
    /// Parameter unit dimension mismatch.
    ParamUnitMismatch {
        key: String,
        expected: UnitDim,
        found: UnitDim,
    },
    /// Internal inconsistency: a lookup by ID failed despite the ID being
    /// present in the name→ID index.
    InternalInconsistency { detail: String },
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
            PortRegistryError::FieldUnitMismatch {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Field '{}' has unit {}, expected {}",
                    name, found, expected
                )
            }
            PortRegistryError::ParamUnitMismatch {
                key,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Parameter '{}' has unit {}, expected {}",
                    key, found, expected
                )
            }
            PortRegistryError::InternalInconsistency { detail } => {
                write!(f, "Internal inconsistency: {}", detail)
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
        BufferF32, BufferU32, Port, ReadOnly, ReadWrite, WgslPort, F32, U32,
    };
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
    fn validate_field_success() {
        let layout = create_test_layout();
        let registry = PortRegistry::new(layout);

        registry
            .validate_scalar_field::<Pressure>("test_module", "p")
            .expect("should validate p");
    }

    #[test]
    fn validate_field_missing() {
        let layout = create_test_layout();
        let registry = PortRegistry::new(layout);

        let err = registry
            .validate_scalar_field::<Pressure>("test_module", "does_not_exist")
            .expect_err("should fail");

        match err {
            PortValidationError::MissingField { module, field } => {
                assert_eq!(module, "test_module");
                assert_eq!(field, "does_not_exist");
            }
            _ => panic!("expected MissingField"),
        }
    }

    #[test]
    fn validate_field_kind_mismatch() {
        let layout = create_test_layout();
        let registry = PortRegistry::new(layout);

        let err = registry
            .validate_vector2_field::<Velocity>("test_module", "p")
            .expect_err("should fail");

        match err {
            PortValidationError::FieldKindMismatch {
                field,
                expected,
                found,
            } => {
                assert_eq!(field, "p");
                assert_eq!(expected, "Vector2");
                assert_eq!(found, "Scalar");
            }
            _ => panic!("expected FieldKindMismatch"),
        }
    }

    #[test]
    fn validate_field_dimension_mismatch() {
        let layout = create_test_layout();
        let registry = PortRegistry::new(layout);

        let err = registry
            .validate_scalar_field::<Length>("test_module", "p")
            .expect_err("should fail");

        match err {
            PortValidationError::DimensionMismatch {
                field,
                expected,
                found,
            } => {
                assert_eq!(field, "p");
                assert_eq!(expected, Length::to_runtime().to_string());
                assert_eq!(found, si::PRESSURE.to_string());
            }
            _ => panic!("expected DimensionMismatch"),
        }
    }

    #[test]
    fn validate_param_missing_and_type_mismatch() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let err = registry
            .validate_param::<F32>("test_module", "dt")
            .expect_err("should fail");
        match err {
            PortValidationError::MissingParameter { module, param } => {
                assert_eq!(module, "test_module");
                assert_eq!(param, "dt");
            }
            _ => panic!("expected MissingParameter"),
        }

        registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("register dt");

        let err = registry
            .validate_param::<U32>("test_module", "dt")
            .expect_err("should fail");
        match err {
            PortValidationError::ParameterTypeMismatch {
                param,
                expected,
                found,
            } => {
                assert_eq!(param, "dt");
                assert_eq!(expected, "u32");
                assert_eq!(found, "f32");
            }
            _ => panic!("expected ParameterTypeMismatch"),
        }

        registry
            .validate_param::<F32>("test_module", "dt")
            .expect("should validate dt");
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

        let p1 = registry
            .register_scalar_field::<Pressure>("p")
            .expect("should register p");

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

        registry
            .register_scalar_field::<Pressure>("p")
            .expect("should register p as scalar");

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

        let dt1 = registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("should register dt");

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

        registry
            .register_param::<F32, Time>("dt", "dt")
            .expect("should register dt");

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

        let state1 = registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state");

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

        registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state as f32");

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

        registry
            .register_buffer::<BufferF32, ReadWrite>("state", 1, 0)
            .expect("should register state as read-write");

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

        assert_eq!(registry.lookup_field("p"), Some(p.id()));
        assert_eq!(registry.lookup_param("dt"), Some(dt.id()));
        assert_eq!(registry.lookup_buffer("state", 1, 0), Some(state.id()));

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
            gradient_targets: vec![],
            resolved_state_slots: None,
        };

        registry
            .register_manifest("test_module", &manifest)
            .expect("should register manifest");

        assert_eq!(registry.param_port_count(), 1);
        assert_eq!(registry.field_port_count(), 1);
        assert_eq!(registry.buffer_port_count(), 1);

        assert!(
            registry.lookup_param("test.dt").is_some(),
            "should find dt param"
        );

        let p_id = registry.lookup_field("p").expect("should find p field");
        let p_entry = registry.get_field_entry(p_id).unwrap();
        assert_eq!(p_entry.name(), "p");
        assert_eq!(p_entry.component_count(), 1);

        assert!(
            registry.lookup_buffer("test_buffer", 1, 0).is_some(),
            "should find buffer"
        );
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
            gradient_targets: vec![],
            resolved_state_slots: None,
        };

        registry
            .register_manifest("module1", &manifest)
            .expect("should register first time");
        assert_eq!(registry.param_port_count(), 1);

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
            gradient_targets: vec![],
            resolved_state_slots: None,
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
            gradient_targets: vec![],
            resolved_state_slots: None,
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

        assert_eq!(manifest.params.len(), 12);

        let gamma = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.gamma")
            .unwrap();
        assert_eq!(gamma.unit, si::DIMENSIONLESS);
        assert_eq!(gamma.wgsl_field, "eos_gamma");

        let gm1 = manifest.params.iter().find(|p| p.key == "eos.gm1").unwrap();
        assert_eq!(gm1.unit, si::DIMENSIONLESS);
        assert_eq!(gm1.wgsl_field, "eos_gm1");

        let r = manifest.params.iter().find(|p| p.key == "eos.r").unwrap();
        // R = P/(rho*T) = (ML⁻¹T⁻²)/(ML⁻³·K) = L²T⁻²K⁻¹
        let expected_r = si::PRESSURE.div_dim(si::DENSITY).div_dim(si::TEMPERATURE);
        assert_eq!(r.unit, expected_r);
        assert_eq!(r.wgsl_field, "eos_r");

        let dp_drho = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.dp_drho")
            .unwrap();
        // dp/drho = P/rho = (ML⁻¹T⁻²)/(ML⁻³) = L²T⁻²
        let expected_dp_drho = si::PRESSURE.div_dim(si::DENSITY);
        assert_eq!(dp_drho.unit, expected_dp_drho);
        assert_eq!(dp_drho.wgsl_field, "eos_dp_drho");

        let p_ref = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.p_ref")
            .unwrap();
        assert_eq!(p_ref.unit, si::PRESSURE);
        assert_eq!(p_ref.wgsl_field, "eos_p_ref");

        let theta_ref = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.theta_ref")
            .unwrap();
        // theta = P/rho has units L²/T² (specific energy)
        assert_eq!(theta_ref.unit, expected_dp_drho);
        assert_eq!(theta_ref.wgsl_field, "eos_theta_ref");

        let rho_ref = manifest
            .params
            .iter()
            .find(|p| p.key == "eos.rho_ref")
            .unwrap();
        assert_eq!(rho_ref.unit, si::DENSITY);
        assert_eq!(rho_ref.wgsl_field, "eos_rho_ref");
    }

    #[test]
    fn register_field_with_owned_string_name() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        // Owned String simulates a derived name like format!("grad_{}", p)
        let owned_name = String::from("p");
        let p = registry
            .register_scalar_field::<Pressure>(&owned_name)
            .expect("should register field with owned String name");

        assert_eq!(p.name(), "p");
        assert_eq!(p.offset(), 2);
    }

    #[test]
    fn register_field_idempotent_with_different_allocations() {
        let layout = create_test_layout();
        let mut registry = PortRegistry::new(layout);

        let name1 = String::from("p");
        let p1 = registry
            .register_scalar_field::<Pressure>(&name1)
            .expect("first registration");

        let name2 = String::from("p");
        let p2 = registry
            .register_scalar_field::<Pressure>(&name2)
            .expect("second registration");

        assert_eq!(p1.id(), p2.id(), "same field should have same PortId");
        assert_eq!(p1.name(), p2.name());
    }

    #[test]
    fn register_field_with_derived_name_pattern() {
        // Test the pattern used in rhie_chow: format!("grad_{}", pressure.name())
        let layout = StateLayout::new(vec![
            vol_scalar("p", si::PRESSURE),
            vol_vector("grad_p", si::PRESSURE / si::LENGTH),
        ]);
        let mut registry = PortRegistry::new(layout);

        let pressure_name = "p";
        let grad_name = format!("grad_{}", pressure_name);

        let grad_p = registry
            .register_vector2_field::<super::super::dimensions::PressureGradient>(&grad_name)
            .expect("should register grad_p with derived name");

        assert_eq!(grad_p.name(), "grad_p");

        let grad_name2 = format!("grad_{}", pressure_name);
        let grad_p2 = registry
            .register_vector2_field::<super::super::dimensions::PressureGradient>(&grad_name2)
            .expect("second registration with derived name");

        assert_eq!(grad_p.id(), grad_p2.id());
    }

    #[test]
    fn to_resolved_state_slots_covers_all_fields() {
        let layout = StateLayout::new(vec![
            vol_vector("U", si::VELOCITY),
            vol_scalar("p", si::PRESSURE),
            vol_scalar("d_p", si::D_P),
        ]);
        let registry = PortRegistry::new(layout);

        let resolved = registry.to_resolved_state_slots();
        assert_eq!(resolved.stride, 4);
        assert_eq!(resolved.slots.len(), 3);

        let u = resolved.slots.iter().find(|s| s.name == "U").unwrap();
        assert_eq!(u.base_offset, 0);
        assert_eq!(
            u.kind,
            crate::solver::ir::ports::PortFieldKind::Vector2
        );

        let p = resolved.slots.iter().find(|s| s.name == "p").unwrap();
        assert_eq!(p.base_offset, 2);
        assert_eq!(
            p.kind,
            crate::solver::ir::ports::PortFieldKind::Scalar
        );

        let dp = resolved.slots.iter().find(|s| s.name == "d_p").unwrap();
        assert_eq!(dp.base_offset, 3);
    }

    #[test]
    fn to_resolved_state_slots_for_subset() {
        let layout = StateLayout::new(vec![
            vol_vector("U", si::VELOCITY),
            vol_scalar("p", si::PRESSURE),
            vol_scalar("d_p", si::D_P),
        ]);
        let registry = PortRegistry::new(layout);

        let names: std::collections::HashSet<String> =
            ["U", "p"].iter().map(|s| s.to_string()).collect();
        let resolved = registry.to_resolved_state_slots_for(&names).unwrap();
        assert_eq!(resolved.stride, 4);
        assert_eq!(resolved.slots.len(), 2);
        assert!(resolved.slots.iter().any(|s| s.name == "U"));
        assert!(resolved.slots.iter().any(|s| s.name == "p"));
    }

    #[test]
    fn to_resolved_state_slots_for_missing_field_errors() {
        let layout = StateLayout::new(vec![
            vol_scalar("p", si::PRESSURE),
        ]);
        let registry = PortRegistry::new(layout);

        let names: std::collections::HashSet<String> =
            ["p", "nonexistent"].iter().map(|s| s.to_string()).collect();
        let err = registry.to_resolved_state_slots_for(&names).unwrap_err();
        assert!(err.contains("nonexistent"), "expected error to mention missing field: {err}");
    }

    #[test]
    fn from_fields_constructs_equivalent_registry() {
        let u = vol_vector("U", si::VELOCITY);
        let p = vol_scalar("p", si::PRESSURE);

        let registry_new = PortRegistry::from_fields(vec![u, p]);
        let layout = StateLayout::new(vec![u, p]);
        let registry_old = PortRegistry::new(layout);

        let resolved_new = registry_new.to_resolved_state_slots();
        let resolved_old = registry_old.to_resolved_state_slots();
        assert_eq!(resolved_new, resolved_old);
    }
}
