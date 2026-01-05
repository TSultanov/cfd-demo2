use crate::solver::model::backend::StateLayout;

pub fn state_component_expr(
    layout: &StateLayout,
    buffer: &str,
    idx: &str,
    field: &str,
    component: u32,
) -> String {
    let offset = layout
        .component_offset(field, component)
        .unwrap_or_else(|| {
            panic!(
                "missing field '{}' component {} in state layout",
                field, component
            )
        });
    let stride = layout.stride();
    format!("{}[{} * {}u + {}u]", buffer, idx, stride, offset)
}

pub fn state_scalar_expr(layout: &StateLayout, buffer: &str, idx: &str, field: &str) -> String {
    state_component_expr(layout, buffer, idx, field, 0)
}

pub fn state_vec2_expr(layout: &StateLayout, buffer: &str, idx: &str, field: &str) -> String {
    let x_expr = state_component_expr(layout, buffer, idx, field, 0);
    let y_expr = state_component_expr(layout, buffer, idx, field, 1);
    format!("vec2<f32>({}, {})", x_expr, y_expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{vol_scalar, vol_vector};

    #[test]
    fn state_access_builds_scalar_and_vector_exprs() {
        let layout = StateLayout::new(vec![vol_vector("U"), vol_scalar("p")]);
        let scalar = state_scalar_expr(&layout, "state", "idx", "p");
        assert_eq!(scalar, "state[idx * 3u + 2u]");

        let component = state_component_expr(&layout, "state", "idx", "U", 1);
        assert_eq!(component, "state[idx * 3u + 1u]");

        let vec2 = state_vec2_expr(&layout, "state", "idx", "U");
        assert_eq!(
            vec2,
            "vec2<f32>(state[idx * 3u + 0u], state[idx * 3u + 1u])"
        );
    }
}
