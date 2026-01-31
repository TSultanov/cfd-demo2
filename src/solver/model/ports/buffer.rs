// Buffer ports for WGSL storage buffer bindings.
//
// Buffer ports provide type-safe abstractions for GPU buffer bindings,
// tracking access modes (read-only, read-write) and data types.

use super::{Port, PortId, WgslPort};
use std::marker::PhantomData;

/// Trait for buffer data types.
///
/// This trait is implemented by marker types representing valid buffer element types.
pub trait BufferType: 'static + Copy + Send + Sync + Eq + PartialEq + std::fmt::Debug {
    /// The WGSL type name for elements in this buffer.
    fn wgsl_element_type() -> &'static str;

    /// Size in bytes of each element.
    fn element_size() -> usize;

    /// Whether this type requires alignment padding in arrays.
    fn requires_alignment() -> bool {
        false
    }

    /// Get the alignment requirement in bytes.
    fn alignment() -> usize {
        Self::element_size()
    }
}

/// f32 array buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferF32;

impl BufferType for BufferF32 {
    fn wgsl_element_type() -> &'static str {
        "f32"
    }

    fn element_size() -> usize {
        4
    }
}

/// u32 array buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferU32;

impl BufferType for BufferU32 {
    fn wgsl_element_type() -> &'static str {
        "u32"
    }

    fn element_size() -> usize {
        4
    }
}

/// i32 array buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferI32;

impl BufferType for BufferI32 {
    fn wgsl_element_type() -> &'static str {
        "i32"
    }

    fn element_size() -> usize {
        4
    }
}

/// vec2<f32> array buffer (8-byte aligned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferVec2F32;

impl BufferType for BufferVec2F32 {
    fn wgsl_element_type() -> &'static str {
        "vec2<f32>"
    }

    fn element_size() -> usize {
        8
    }

    fn alignment() -> usize {
        8
    }
}

/// vec3<f32> array buffer (16-byte aligned due to WGSL rules).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferVec3F32;

impl BufferType for BufferVec3F32 {
    fn wgsl_element_type() -> &'static str {
        "vec3<f32>"
    }

    fn element_size() -> usize {
        12
    }

    fn requires_alignment() -> bool {
        true
    }

    fn alignment() -> usize {
        16 // WGSL aligns vec3 to 16 bytes
    }
}

/// Trait for buffer access modes.
///
/// This trait is implemented by marker types representing read-only or read-write access.
pub trait AccessMode: 'static + Copy + Send + Sync + Eq + PartialEq + std::fmt::Debug {
    /// The WGSL access mode keyword.
    fn wgsl_access_mode() -> &'static str;

    /// Whether this mode allows writing.
    fn allows_write() -> bool;

    /// Whether this mode allows reading.
    fn allows_read() -> bool;
}

/// Read-only access mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadOnly;

impl AccessMode for ReadOnly {
    fn wgsl_access_mode() -> &'static str {
        "read"
    }

    fn allows_write() -> bool {
        false
    }

    fn allows_read() -> bool {
        true
    }
}

/// Read-write access mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadWrite;

impl AccessMode for ReadWrite {
    fn wgsl_access_mode() -> &'static str {
        "read_write"
    }

    fn allows_write() -> bool {
        true
    }

    fn allows_read() -> bool {
        true
    }
}

/// A type-safe reference to a GPU buffer binding.
///
/// `BufferPort` provides compile-time verification of:
/// - Buffer element type
/// - Access mode (read-only or read-write)
/// - Binding group and index
///
/// # Type Parameters
///
/// - `T`: The buffer element type (e.g., `BufferF32`, `BufferVec2F32`)
/// - `A`: The access mode (`ReadOnly` or `ReadWrite`)
///
/// # Example
///
/// ```rust,ignore
/// // Create a port for the state buffer
/// let state: BufferPort<BufferF32, ReadWrite> = registry.buffer("state", 1, 0)?;
///
/// // Generates WGSL binding:
/// // @group(1) @binding(0) var<storage, read_write> state: array<f32>;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferPort<T: BufferType, A: AccessMode> {
    id: PortId,
    name: &'static str,
    group: u32,
    binding: u32,
    _ty: PhantomData<T>,
    _access: PhantomData<A>,
}

impl<T: BufferType, A: AccessMode> BufferPort<T, A> {
    /// Create a new buffer port.
    ///
    /// This should be called by PortRegistry during buffer registration.
    pub fn new(id: PortId, name: &'static str, group: u32, binding: u32) -> Self {
        Self {
            id,
            name,
            group,
            binding,
            _ty: PhantomData,
            _access: PhantomData,
        }
    }

    /// Get the binding group index.
    pub fn group(&self) -> u32 {
        self.group
    }

    /// Get the binding index within the group.
    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Check if this buffer allows write access.
    pub fn allows_write(&self) -> bool {
        A::allows_write()
    }

    /// Check if this buffer allows read access.
    pub fn allows_read(&self) -> bool {
        A::allows_read()
    }

    /// Generate the WGSL binding declaration.
    ///
    /// Returns text like:
    /// `@group(0) @binding(1) var<storage, read_write> state: array<f32>;`
    pub fn wgsl_binding_decl(&self) -> String {
        format!(
            "@group({}) @binding({})\nvar<storage, {}> {}: array<{}>;",
            self.group,
            self.binding,
            A::wgsl_access_mode(),
            self.name,
            T::wgsl_element_type()
        )
    }

    /// Generate WGSL code to access an element in this buffer.
    ///
    /// Returns text like `state[index]`.
    pub fn wgsl_element_access(&self, index_expr: &str) -> String {
        format!("{}[{}]", self.name, index_expr)
    }
}

impl<T: BufferType, A: AccessMode> Port for BufferPort<T, A> {
    type Resource = ();

    fn id(&self) -> PortId {
        self.id
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

impl<T: BufferType, A: AccessMode> WgslPort for BufferPort<T, A> {
    fn wgsl_access(&self, index: Option<&str>) -> String {
        let idx = index.unwrap_or("idx");
        self.wgsl_element_access(idx)
    }

    fn wgsl_type(&self) -> &'static str {
        T::wgsl_element_type()
    }

    fn wgsl_binding(&self) -> Option<String> {
        Some(self.wgsl_binding_decl())
    }
}

/// A group of related buffer bindings.
///
/// This is typically used to represent all bindings in a WGSL bind group.
#[derive(Debug, Clone)]
pub struct BufferBindingGroup {
    group_index: u32,
    bindings: Vec<BufferBindingEntry>,
}

/// Entry in a buffer binding group.
#[derive(Debug, Clone)]
pub struct BufferBindingEntry {
    pub binding: u32,
    pub name: &'static str,
    pub buffer_type: BufferTypeKind,
    pub access_mode: AccessModeKind,
}

/// Enum representing buffer element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferTypeKind {
    F32,
    U32,
    I32,
    Vec2F32,
    Vec3F32,
}

impl BufferTypeKind {
    pub fn wgsl_element_type(self) -> &'static str {
        match self {
            BufferTypeKind::F32 => "f32",
            BufferTypeKind::U32 => "u32",
            BufferTypeKind::I32 => "i32",
            BufferTypeKind::Vec2F32 => "vec2<f32>",
            BufferTypeKind::Vec3F32 => "vec3<f32>",
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            BufferTypeKind::F32 => 4,
            BufferTypeKind::U32 => 4,
            BufferTypeKind::I32 => 4,
            BufferTypeKind::Vec2F32 => 8,
            BufferTypeKind::Vec3F32 => 12,
        }
    }

    pub fn alignment(self) -> usize {
        match self {
            BufferTypeKind::F32 => 4,
            BufferTypeKind::U32 => 4,
            BufferTypeKind::I32 => 4,
            BufferTypeKind::Vec2F32 => 8,
            BufferTypeKind::Vec3F32 => 16,
        }
    }
}

/// Enum representing access modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessModeKind {
    ReadOnly,
    ReadWrite,
}

impl AccessModeKind {
    pub fn wgsl_access_mode(self) -> &'static str {
        match self {
            AccessModeKind::ReadOnly => "read",
            AccessModeKind::ReadWrite => "read_write",
        }
    }

    pub fn allows_write(self) -> bool {
        matches!(self, AccessModeKind::ReadWrite)
    }
}

/// Builder for buffer binding groups.
#[derive(Debug)]
pub struct BufferBindingGroupBuilder {
    group_index: u32,
    next_binding: u32,
    entries: Vec<BufferBindingEntry>,
}

impl BufferBindingGroupBuilder {
    pub fn new(group_index: u32) -> Self {
        Self {
            group_index,
            next_binding: 0,
            entries: Vec::new(),
        }
    }

    pub fn add_buffer<T: BufferType + IntoBufferTypeKind, A: AccessMode + IntoAccessModeKind>(
        mut self,
        name: &'static str,
    ) -> Self {
        self.entries.push(BufferBindingEntry {
            binding: self.next_binding,
            name,
            buffer_type: T::into_kind(),
            access_mode: A::into_kind(),
        });
        self.next_binding += 1;
        self
    }

    pub fn build(self) -> BufferBindingGroup {
        BufferBindingGroup {
            group_index: self.group_index,
            bindings: self.entries,
        }
    }
}

/// Trait to convert BufferType marker types to BufferTypeKind.
pub trait IntoBufferTypeKind: BufferType {
    fn into_kind() -> BufferTypeKind;
}

impl IntoBufferTypeKind for BufferF32 {
    fn into_kind() -> BufferTypeKind {
        BufferTypeKind::F32
    }
}

impl IntoBufferTypeKind for BufferU32 {
    fn into_kind() -> BufferTypeKind {
        BufferTypeKind::U32
    }
}

impl IntoBufferTypeKind for BufferI32 {
    fn into_kind() -> BufferTypeKind {
        BufferTypeKind::I32
    }
}

impl IntoBufferTypeKind for BufferVec2F32 {
    fn into_kind() -> BufferTypeKind {
        BufferTypeKind::Vec2F32
    }
}

impl IntoBufferTypeKind for BufferVec3F32 {
    fn into_kind() -> BufferTypeKind {
        BufferTypeKind::Vec3F32
    }
}

/// Trait to convert AccessMode marker types to AccessModeKind.
pub trait IntoAccessModeKind: AccessMode {
    fn into_kind() -> AccessModeKind;
}

impl IntoAccessModeKind for ReadOnly {
    fn into_kind() -> AccessModeKind {
        AccessModeKind::ReadOnly
    }
}

impl IntoAccessModeKind for ReadWrite {
    fn into_kind() -> AccessModeKind {
        AccessModeKind::ReadWrite
    }
}

/// Standard binding group indices used throughout the solver.
pub mod bind_groups {
    /// Group 0: Mesh topology and geometry.
    pub const MESH: u32 = 0;

    /// Group 1: Field data (state, state_old, fluxes, etc.).
    pub const FIELDS: u32 = 1;

    /// Group 2: Linear system data (matrix, RHS, etc.).
    pub const LINEAR_SYSTEM: u32 = 2;

    /// Group 3: Boundary conditions.
    pub const BOUNDARY: u32 = 3;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_type_properties() {
        assert_eq!(BufferF32::wgsl_element_type(), "f32");
        assert_eq!(BufferF32::element_size(), 4);

        assert_eq!(BufferVec2F32::wgsl_element_type(), "vec2<f32>");
        assert_eq!(BufferVec2F32::element_size(), 8);
        assert_eq!(BufferVec2F32::alignment(), 8);

        assert_eq!(BufferVec3F32::wgsl_element_type(), "vec3<f32>");
        assert_eq!(BufferVec3F32::element_size(), 12);
        assert_eq!(BufferVec3F32::alignment(), 16);
        assert!(BufferVec3F32::requires_alignment());
    }

    #[test]
    fn access_mode_properties() {
        assert_eq!(ReadOnly::wgsl_access_mode(), "read");
        assert!(!ReadOnly::allows_write());
        assert!(ReadOnly::allows_read());

        assert_eq!(ReadWrite::wgsl_access_mode(), "read_write");
        assert!(ReadWrite::allows_write());
        assert!(ReadWrite::allows_read());
    }

    #[test]
    fn buffer_port_wgsl_binding() {
        let state = BufferPort::<BufferF32, ReadWrite>::new(PortId::new(0), "state", 1, 0);

        let expected = "@group(1) @binding(0)\nvar<storage, read_write> state: array<f32>;";
        assert_eq!(state.wgsl_binding_decl(), expected);
        assert!(state.allows_write());
    }

    #[test]
    fn buffer_port_readonly() {
        let cell_vols = BufferPort::<BufferF32, ReadOnly>::new(PortId::new(1), "cell_vols", 0, 5);

        assert!(!cell_vols.allows_write());
        assert_eq!(cell_vols.group(), 0);
        assert_eq!(cell_vols.binding(), 5);
    }

    #[test]
    fn buffer_binding_group_builder() {
        let group = BufferBindingGroupBuilder::new(bind_groups::FIELDS)
            .add_buffer::<BufferF32, ReadWrite>("state")
            .add_buffer::<BufferF32, ReadOnly>("state_old")
            .build();

        assert_eq!(group.group_index, 1);
        assert_eq!(group.bindings.len(), 2);
        assert_eq!(group.bindings[0].name, "state");
        assert_eq!(group.bindings[0].binding, 0);
        assert_eq!(group.bindings[1].name, "state_old");
        assert_eq!(group.bindings[1].binding, 1);
    }

    #[test]
    fn buffer_port_element_access() {
        let state = BufferPort::<BufferF32, ReadWrite>::new(PortId::new(0), "state", 1, 0);

        assert_eq!(state.wgsl_element_access("idx"), "state[idx]");
        assert_eq!(
            state.wgsl_element_access("global_id.x"),
            "state[global_id.x]"
        );
    }
}
