use super::wgsl_ast::{AssignOp, Block, Expr, ForInit, ForStep, Stmt, Type};

pub fn vec2_f32(x: Expr, y: Expr) -> Expr {
    Expr::call_named("vec2<f32>", vec![x, y])
}

pub fn vec2_f32_xy_fields(name: &str) -> Expr {
    let v = Expr::ident(name);
    vec2_f32(v.clone().field("x"), v.field("y"))
}

pub fn vec2_f32_from_xy_fields(value: Expr) -> Expr {
    vec2_f32(value.clone().field("x"), value.field("y"))
}

pub fn dot_expr(lhs: Expr, rhs: Expr) -> Expr {
    Expr::call_named("dot", vec![lhs, rhs])
}

pub fn min_expr(lhs: Expr, rhs: Expr) -> Expr {
    Expr::call_named("min", vec![lhs, rhs])
}

pub fn max_expr(lhs: Expr, rhs: Expr) -> Expr {
    Expr::call_named("max", vec![lhs, rhs])
}

pub fn array_access(array: &str, index: Expr) -> Expr {
    Expr::ident(array).index(index)
}

pub fn linear_index(idx: Expr, stride: u32, offset: u32) -> Expr {
    idx * Expr::lit_u32(stride) + Expr::lit_u32(offset)
}

pub fn array_access_linear(array: &str, idx: Expr, stride: u32, offset: u32) -> Expr {
    array_access(array, linear_index(idx, stride, offset))
}

pub fn block(stmts: Vec<Stmt>) -> Block {
    Block::new(stmts)
}

pub fn comment(text: &str) -> Stmt {
    Stmt::Comment(text.to_string())
}

pub fn let_expr(name: &str, expr: Expr) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty: None,
        expr,
    }
}

pub fn let_typed_expr(name: &str, ty: Type, expr: Expr) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty: Some(ty),
        expr,
    }
}

pub fn var_expr(name: &str, expr: Expr) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: None,
        expr: Some(expr),
    }
}

pub fn var_typed_expr(name: &str, ty: Type, expr: Option<Expr>) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr,
    }
}

pub fn assign_expr(target: Expr, value: Expr) -> Stmt {
    Stmt::Assign { target, value }
}

pub fn assign_array_access(array: &str, index: Expr, value: Expr) -> Stmt {
    assign_expr(array_access(array, index), value)
}

pub fn assign_array_access_linear(
    array: &str,
    idx: Expr,
    stride: u32,
    offset: u32,
    value: Expr,
) -> Stmt {
    assign_expr(array_access_linear(array, idx, stride, offset), value)
}

pub fn assign_op_expr(op: AssignOp, target: Expr, value: Expr) -> Stmt {
    Stmt::AssignOp { target, op, value }
}

pub fn increment_expr(expr: Expr) -> Stmt {
    Stmt::Increment(expr)
}

pub fn call_stmt_expr(expr: Expr) -> Stmt {
    Stmt::Call(expr)
}

pub fn if_block_expr(cond: Expr, then_block: Block, else_block: Option<Block>) -> Stmt {
    Stmt::If {
        cond,
        then_block,
        else_block,
    }
}

pub fn for_loop_expr(init: ForInit, cond: Expr, step: ForStep, body: Block) -> Stmt {
    Stmt::For {
        init,
        cond,
        step,
        body,
    }
}

pub fn for_init_var_expr(name: &str, expr: Expr) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: None,
        expr,
    }
}

pub fn for_init_var_typed_expr(name: &str, ty: Type, expr: Expr) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr,
    }
}

pub fn for_init_assign_expr(target: Expr, value: Expr) -> ForInit {
    ForInit::Assign { target, value }
}

pub fn for_step_increment_expr(expr: Expr) -> ForStep {
    ForStep::Increment(expr)
}

pub fn for_step_assign_expr(target: Expr, value: Expr) -> ForStep {
    ForStep::Assign { target, value }
}

pub fn for_step_assign_op_expr(op: AssignOp, target: Expr, value: Expr) -> ForStep {
    ForStep::AssignOp { target, op, value }
}

pub fn for_each_xy<F>(mut f: F) -> Vec<Stmt>
where
    F: FnMut(&str) -> Stmt,
{
    vec![f("x"), f("y")]
}

pub fn for_each_xy_block<F>(mut f: F) -> Vec<Stmt>
where
    F: FnMut(&str) -> Vec<Stmt>,
{
    let mut out = Vec::new();
    out.extend(f("x"));
    out.extend(f("y"));
    out
}

pub fn for_each_mat_entry<F>(n: usize, mut f: F) -> Vec<Stmt>
where
    F: FnMut(usize, usize) -> Stmt,
{
    let mut out = Vec::new();
    for row in 0..n {
        for col in 0..n {
            out.push(f(row, col));
        }
    }
    out
}

pub fn for_each_mat_entry_block<F>(n: usize, mut f: F) -> Vec<Stmt>
where
    F: FnMut(usize, usize) -> Vec<Stmt>,
{
    let mut out = Vec::new();
    for row in 0..n {
        for col in 0..n {
            out.extend(f(row, col));
        }
    }
    out
}

pub fn assign_op_matrix_from_prefix_scaled_expr(
    op: AssignOp,
    dest_prefix: &str,
    src_prefix: &str,
    n: usize,
    scale: Option<Expr>,
) -> Vec<Stmt> {
    for_each_mat_entry(n, |row, col| {
        let target = Expr::ident(format!("{dest_prefix}_{row}{col}"));
        let mut value = Expr::ident(format!("{src_prefix}_{row}{col}"));
        if let Some(scale) = scale {
            value = value * scale;
        }
        assign_op_expr(op, target, value)
    })
}

pub fn assign_matrix_array_from_prefix_scaled_expr(
    matrix_array: &str,
    base_prefix: &str,
    src_prefix: &str,
    n: usize,
    scale: Option<Expr>,
) -> Vec<Stmt> {
    for_each_mat_entry(n, |row, col| {
        let base = Expr::ident(format!("{base_prefix}_{row}"));
        let index = base + Expr::lit_u32(col as u32);
        let target = Expr::ident(matrix_array).index(index);
        let mut value = Expr::ident(format!("{src_prefix}_{row}{col}"));
        if let Some(scale) = scale {
            value = value * scale;
        }
        assign_expr(target, value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::wgsl_ast::{AssignOp, Type};

    #[test]
    fn dsl_builds_basic_statements() {
        let stmt = let_expr("idx", Expr::ident("global_id").field("x"));
        assert!(matches!(stmt, Stmt::Let { .. }));

        let stmt = var_typed_expr("normal", Type::Custom("Vector2".to_string()), None);
        assert!(matches!(stmt, Stmt::Var { .. }));

        let stmt = assign_op_expr(AssignOp::Add, Expr::ident("diag_u"), Expr::ident("coeff_time"));
        assert!(matches!(stmt, Stmt::AssignOp { .. }));
    }

    #[test]
    fn dsl_builds_matrix_helpers() {
        let stmts = assign_op_matrix_from_prefix_scaled_expr(
            AssignOp::Add,
            "diag",
            "jac",
            2,
            Some(Expr::ident("area")),
        );
        assert_eq!(stmts.len(), 4);
        assert!(stmts
            .iter()
            .all(|stmt| matches!(stmt, Stmt::AssignOp { .. })));

        let stmts = assign_matrix_array_from_prefix_scaled_expr(
            "matrix_values",
            "base",
            "jac",
            2,
            Some(Expr::ident("area")),
        );
        assert_eq!(stmts.len(), 4);
        assert!(stmts.iter().all(|stmt| matches!(stmt, Stmt::Assign { .. })));
    }

    #[test]
    fn dsl_builds_linear_array_access() {
        let expr = array_access_linear("rhs", Expr::ident("idx"), 3, 2);
        assert_eq!(expr.to_string(), "rhs[idx * 3u + 2u]");

        let stmt = assign_array_access_linear("rhs", Expr::ident("idx"), 3, 2, Expr::ident("x"));
        match stmt {
            Stmt::Assign { target, value } => {
                assert_eq!(target.to_string(), "rhs[idx * 3u + 2u]");
                assert_eq!(value.to_string(), "x");
            }
            _ => panic!("expected assign stmt"),
        }
    }
}
