use crate::solver::model::backend::StateLayout;
use crate::solver::codegen::wgsl_ast::{BinaryOp, Expr};

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

pub fn state_component(
    layout: &StateLayout,
    buffer: &str,
    idx: &str,
    field: &str,
    component: u32,
) -> Expr {
    let offset = layout
        .component_offset(field, component)
        .unwrap_or_else(|| {
            panic!(
                "missing field '{}' component {} in state layout",
                field, component
            )
        });
    let stride = layout.stride();
    let idx_expr = Expr::ident(idx);
    let offset_expr = Expr::lit_u32(offset);
    let stride_expr = Expr::lit_u32(stride);
    let index_expr = Expr::binary(
        Expr::binary(idx_expr, BinaryOp::Mul, stride_expr),
        BinaryOp::Add,
        offset_expr,
    );
    Expr::ident(buffer).index(index_expr)
}

pub fn state_scalar(layout: &StateLayout, buffer: &str, idx: &str, field: &str) -> Expr {
    state_component(layout, buffer, idx, field, 0)
}

pub fn state_vec2(layout: &StateLayout, buffer: &str, idx: &str, field: &str) -> Expr {
    let x_expr = state_component(layout, buffer, idx, field, 0);
    let y_expr = state_component(layout, buffer, idx, field, 1);
    Expr::call_named("vec2<f32>", vec![x_expr, y_expr])
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

    #[test]
    fn state_access_builds_expr_ast() {
        let layout = StateLayout::new(vec![vol_vector("U"), vol_scalar("p")]);
        let scalar = state_scalar(&layout, "state", "idx", "p");
        assert_eq!(scalar.to_string(), "state[idx * 3u + 2u]");

        let component = state_component(&layout, "state", "idx", "U", 1);
        assert_eq!(component.to_string(), "state[idx * 3u + 1u]");

        let vec2 = state_vec2(&layout, "state", "idx", "U");
        assert_eq!(
            vec2.to_string(),
            "vec2<f32>(state[idx * 3u + 0u], state[idx * 3u + 1u])"
        );
    }
}
