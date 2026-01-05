use std::collections::HashMap;

use super::ast::{FieldKind, FieldRef};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateField {
    name: String,
    kind: FieldKind,
    offset: u32,
}

impl StateField {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> FieldKind {
        self.kind
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn component_count(&self) -> u32 {
        match self.kind {
            FieldKind::Scalar => 1,
            FieldKind::Vector2 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateLayout {
    fields: Vec<StateField>,
    offsets: HashMap<String, usize>,
    stride: u32,
}

impl StateLayout {
    pub fn new(fields: Vec<FieldRef>) -> Self {
        let mut offsets = HashMap::new();
        let mut layout_fields = Vec::new();
        let mut offset = 0u32;

        for field in fields {
            if offsets.contains_key(field.name()) {
                continue;
            }
            let entry = StateField {
                name: field.name().to_string(),
                kind: field.kind(),
                offset,
            };
            offsets.insert(entry.name.clone(), layout_fields.len());
            offset += entry.component_count();
            layout_fields.push(entry);
        }

        Self {
            fields: layout_fields,
            offsets,
            stride: offset,
        }
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    pub fn fields(&self) -> &[StateField] {
        &self.fields
    }

    pub fn offset_for(&self, name: &str) -> Option<u32> {
        self.offsets.get(name).map(|&idx| self.fields[idx].offset)
    }

    pub fn component_offset(&self, name: &str, component: u32) -> Option<u32> {
        let field = self.field(name)?;
        if component >= field.component_count() {
            return None;
        }
        Some(field.offset + component)
    }

    pub fn field(&self, name: &str) -> Option<&StateField> {
        self.offsets.get(name).map(|&idx| &self.fields[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::ast::{vol_scalar, vol_vector};

    #[test]
    fn state_layout_assigns_offsets_and_stride() {
        let u = vol_vector("U");
        let p = vol_scalar("p");
        let d_p = vol_scalar("d_p");
        let grad_p = vol_vector("grad_p");
        let grad_comp = vol_vector("grad_component");

        let layout = StateLayout::new(vec![u, p, d_p, grad_p, grad_comp]);
        assert_eq!(layout.stride(), 8);
        assert_eq!(layout.offset_for("U"), Some(0));
        assert_eq!(layout.offset_for("p"), Some(2));
        assert_eq!(layout.offset_for("d_p"), Some(3));
        assert_eq!(layout.offset_for("grad_p"), Some(4));
        assert_eq!(layout.offset_for("grad_component"), Some(6));
        assert_eq!(layout.component_offset("U", 1), Some(1));
        assert_eq!(layout.component_offset("U", 2), None);
    }
}
