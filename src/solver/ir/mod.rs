// Internal IR facade.
//
// Incremental boundary: codegen must depend on types from this module rather than reaching into
// `crate::solver::model::backend` directly.

#[allow(unused_imports)]
pub(crate) use crate::solver::model::backend::{
    fvc, fvm, expand_schemes, Coefficient, Discretization, Equation, EquationSystem, FieldKind,
    FieldRef, FluxRef, SchemeExpansion, SchemeRegistry, StateField, StateLayout, Term, TermKey,
    TermOp,
};

#[allow(unused_imports)]
pub(crate) use crate::solver::model::backend::ast::{
    surface_scalar, surface_vector3, vol_scalar, vol_vector, vol_vector3, CodegenError,
    UnitValidationError,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EosSpec {
    IdealGas { gamma: f32 },
    Constant,
}

impl EosSpec {
    pub fn ideal_gas_gamma(&self) -> Option<f32> {
        match self {
            EosSpec::IdealGas { gamma } => Some(*gamma),
            EosSpec::Constant => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FluxComponent {
    pub name: String,
    pub offset: u32,
}

/// Named-component description of a packed face-flux buffer.
///
/// This intentionally does NOT reuse `FluxKind`. The purpose is to provide a stable,
/// model-derived mapping from semantic flux components to packed offsets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FluxLayout {
    pub stride: u32,
    pub components: Vec<FluxComponent>,
}

impl FluxLayout {
    pub fn from_system(system: &EquationSystem) -> Self {
        let mut components = Vec::new();
        let mut offset: u32 = 0;

        for eq in system.equations() {
            let field = eq.target();
            match field.kind() {
                FieldKind::Scalar => {
                    components.push(FluxComponent {
                        name: field.name().to_string(),
                        offset,
                    });
                    offset += 1;
                }
                FieldKind::Vector2 => {
                    // Match the existing coupled ordering: x then y.
                    components.push(FluxComponent {
                        name: format!("{}_x", field.name()),
                        offset,
                    });
                    offset += 1;
                    components.push(FluxComponent {
                        name: format!("{}_y", field.name()),
                        offset,
                    });
                    offset += 1;
                }
                FieldKind::Vector3 => {
                    // 3D ordering: x, y, z.
                    components.push(FluxComponent {
                        name: format!("{}_x", field.name()),
                        offset,
                    });
                    offset += 1;
                    components.push(FluxComponent {
                        name: format!("{}_y", field.name()),
                        offset,
                    });
                    offset += 1;
                    components.push(FluxComponent {
                        name: format!("{}_z", field.name()),
                        offset,
                    });
                    offset += 1;
                }
            }
        }

        FluxLayout {
            stride: offset,
            components,
        }
    }

    pub fn offset_for(&self, name: &str) -> Option<u32> {
        self.components
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.offset)
    }

    pub fn offset_for_field_component(&self, field: FieldRef, component: u32) -> Option<u32> {
        match field.kind() {
            FieldKind::Scalar => {
                if component == 0 {
                    self.offset_for(field.name())
                } else {
                    None
                }
            }
            FieldKind::Vector2 => {
                let suffix = match component {
                    0 => "x",
                    1 => "y",
                    _ => return None,
                };
                self.offset_for(&format!("{}_{}", field.name(), suffix))
            }
            FieldKind::Vector3 => {
                let suffix = match component {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => return None,
                };
                self.offset_for(&format!("{}_{}", field.name(), suffix))
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    pub(crate) use crate::solver::model::compressible_model;
}
