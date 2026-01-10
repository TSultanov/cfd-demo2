use crate::solver::model::backend::FieldRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientStorage {
    /// No gradient buffers are allocated/bound.
    None,
    /// Gradients are stored per field component (e.g. `grad_rho_u_x`).
    PerFieldComponents,
    /// Gradients are stored for the packed state vector (e.g. `grad_state`).
    PackedState,
}

impl Default for GradientStorage {
    fn default() -> Self {
        Self::None
    }
}

/// Specification for a face-based flux buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FluxSpec {
    /// Floats per face.
    pub stride: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ModelGpuSpec {
    /// Optional face-based flux storage requirements.
    pub flux: Option<FluxSpec>,

    /// Whether the model requires a low-mach params uniform buffer.
    pub requires_low_mach_params: bool,

    /// How gradient buffers are exposed to shaders.
    pub gradient_storage: GradientStorage,

    /// Additional gradient buffers that must exist regardless of scheme expansion.
    ///
    /// Names must match the reflection binding convention used by kernels:
    /// - `PerFieldComponents`: names like `rho`, `rho_u_x`, `rho_u_y` (bound as `grad_<name>`)
    /// - `PackedState`: the single name `state` (bound as `grad_state`)
    pub required_gradient_fields: Vec<String>,
}

pub fn expand_field_components(field: FieldRef) -> Vec<String> {
    match field.kind() {
        crate::solver::model::backend::FieldKind::Scalar => vec![field.name().to_string()],
        crate::solver::model::backend::FieldKind::Vector2 => vec![
            format!("{}_x", field.name()),
            format!("{}_y", field.name()),
        ],
    }
}
