/// IR-safe port manifest types.
///
/// These types provide a pure-data representation of port metadata that can cross
/// the IR/codegen boundary without depending on runtime port types.
use crate::solver::units::UnitDim;
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PortManifest {
    /// Parameter ports (uniform buffer entries).
    pub params: Vec<ParamSpec>,
    /// Field ports (state layout entries).
    pub fields: Vec<FieldSpec>,
    /// Buffer ports (storage buffer bindings).
    pub buffers: Vec<BufferSpec>,
    /// Resolved gradient targets for flux module gradients kernel.
    /// When gradients are enabled, this contains pre-resolved offsets and metadata
    /// for each gradient computation target.
    pub gradient_targets: Vec<ResolvedGradientTargetSpec>,
}

/// Specification for a parameter port.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpec {
    /// The parameter key (human-readable identifier).
    pub key: &'static str,
    /// The WGSL field name in the constants struct.
    pub wgsl_field: &'static str,
    /// The WGSL type (e.g., "f32", "u32").
    pub wgsl_type: &'static str,
    /// The physical unit dimension.
    pub unit: UnitDim,
}

/// Specification for a field port.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSpec {
    /// The field name in the state layout.
    pub name: &'static str,
    /// The field kind (scalar, vector2, vector3).
    pub kind: PortFieldKind,
    /// The expected physical unit dimension (from type-level D).
    pub unit: UnitDim,
}

/// Specification for a buffer port.
#[derive(Debug, Clone, PartialEq)]
pub struct BufferSpec {
    /// The buffer name (WGSL variable name).
    pub name: &'static str,
    /// The binding group index.
    pub group: u32,
    /// The binding index within the group.
    pub binding: u32,
    /// The WGSL element type (e.g., "f32", "vec2<f32>").
    pub elem_wgsl_type: &'static str,
    /// The access mode.
    pub access: BufferAccess,
}

/// Field kind enumeration (IR-safe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortFieldKind {
    /// Scalar field (single f32).
    Scalar,
    /// 2D vector field (vec2<f32>).
    Vector2,
    /// 3D vector field (vec3<f32>).
    Vector3,
}

impl PortFieldKind {
    /// Number of components for this field kind.
    pub const fn component_count(self) -> u32 {
        match self {
            PortFieldKind::Scalar => 1,
            PortFieldKind::Vector2 => 2,
            PortFieldKind::Vector3 => 3,
        }
    }

    /// WGSL type name for this field kind.
    pub const fn wgsl_type(self) -> &'static str {
        match self {
            PortFieldKind::Scalar => "f32",
            PortFieldKind::Vector2 => "vec2<f32>",
            PortFieldKind::Vector3 => "vec3<f32>",
        }
    }
}

/// Buffer access mode enumeration (IR-safe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// Read-only access.
    ReadOnly,
    /// Read-write access.
    ReadWrite,
}

impl BufferAccess {
    /// WGSL access mode keyword.
    pub const fn wgsl_keyword(self) -> &'static str {
        match self {
            BufferAccess::ReadOnly => "read",
            BufferAccess::ReadWrite => "read_write",
        }
    }

    /// Whether this access mode allows writing.
    pub const fn allows_write(self) -> bool {
        matches!(self, BufferAccess::ReadWrite)
    }
}

/// IR-safe specification for a resolved gradient target.
///
/// This record holds pre-resolved offsets and metadata for a single gradient computation target,
/// eliminating the need to scan StateLayout during WGSL generation. It is stored in the
/// PortManifest so codegen can access it without depending on runtime types.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGradientTargetSpec {
    /// The component key used in BC/flux tables (e.g., "rho_u_x")
    pub component: String,
    /// Base field name (e.g., "rho_u" for component "rho_u_x")
    pub base_field: String,
    /// Component index within the base field (0/1/2)
    pub base_component: u32,
    /// Offset of the base field/component in state array
    pub base_offset: u32,
    /// Offset of gradient x-component in state array
    pub grad_x_offset: u32,
    /// Offset of gradient y-component in state array
    pub grad_y_offset: u32,
    /// Offset in flux layout for BC lookup (if applicable)
    pub bc_unknown_offset: Option<u32>,
    /// SlipWall: x-offset of full vec2 field (for velocity fields)
    pub slip_vec2_x_offset: Option<u32>,
    /// SlipWall: y-offset of full vec2 field (for velocity fields)
    pub slip_vec2_y_offset: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_kind_properties() {
        assert_eq!(PortFieldKind::Scalar.component_count(), 1);
        assert_eq!(PortFieldKind::Vector2.component_count(), 2);
        assert_eq!(PortFieldKind::Vector3.component_count(), 3);

        assert_eq!(PortFieldKind::Scalar.wgsl_type(), "f32");
        assert_eq!(PortFieldKind::Vector2.wgsl_type(), "vec2<f32>");
        assert_eq!(PortFieldKind::Vector3.wgsl_type(), "vec3<f32>");
    }

    #[test]
    fn buffer_access_properties() {
        assert_eq!(BufferAccess::ReadOnly.wgsl_keyword(), "read");
        assert_eq!(BufferAccess::ReadWrite.wgsl_keyword(), "read_write");

        assert!(!BufferAccess::ReadOnly.allows_write());
        assert!(BufferAccess::ReadWrite.allows_write());
    }

    #[test]
    fn port_manifest_default() {
        let manifest = PortManifest::default();
        assert!(manifest.params.is_empty());
        assert!(manifest.fields.is_empty());
        assert!(manifest.buffers.is_empty());
        assert!(manifest.gradient_targets.is_empty());
    }
}
