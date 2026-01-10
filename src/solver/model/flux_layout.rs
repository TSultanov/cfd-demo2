use crate::solver::model::backend::ast::{EquationSystem, FieldKind, FieldRef};

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
        }
    }
}
