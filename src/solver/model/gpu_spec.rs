use crate::solver::model::backend::FieldRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GradientStorage {
    /// No gradient buffers are allocated/bound.
    #[default]
    None,
    /// Gradients are stored per field name (e.g. `grad_U`, `grad_p`).
    ///
    /// This is the legacy convention used by incompressible/coupled kernels.
    PerFieldName,
    /// Gradients are stored per field component (e.g. `grad_rho_u_x`).
    PerFieldComponents,
    /// Gradients are stored for the packed state vector (e.g. `grad_state`).
    PackedState,
}

pub fn expand_field_components(field: FieldRef) -> Vec<String> {
    match field.kind() {
        crate::solver::model::backend::FieldKind::Scalar => vec![field.name().to_string()],
        crate::solver::model::backend::FieldKind::Vector2 => {
            vec![format!("{}_x", field.name()), format!("{}_y", field.name())]
        }
        crate::solver::model::backend::FieldKind::Vector3 => vec![
            format!("{}_x", field.name()),
            format!("{}_y", field.name()),
            format!("{}_z", field.name()),
        ],
    }
}
