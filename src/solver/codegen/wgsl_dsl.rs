use super::wgsl_ast::{AssignOp, BinaryOp, Block, Expr, ForInit, ForStep, ParseError, Stmt, Type};

pub fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("invalid WGSL expression")
}

pub fn array_access(array: &str, index: Expr) -> Expr {
    Expr::ident(array).index(index)
}

pub fn linear_index(idx: Expr, stride: u32, offset: u32) -> Expr {
    Expr::binary(
        Expr::binary(idx, BinaryOp::Mul, Expr::lit_u32(stride)),
        BinaryOp::Add,
        Expr::lit_u32(offset),
    )
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

pub fn let_(name: &str, value: &str) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty: None,
        expr: expr(value),
    }
}

pub fn let_expr(name: &str, expr: Expr) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty: None,
        expr,
    }
}

pub fn let_typed(name: &str, ty: Type, value: &str) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty: Some(ty),
        expr: expr(value),
    }
}

pub fn let_typed_expr(name: &str, ty: Type, expr: Expr) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty: Some(ty),
        expr,
    }
}

pub fn var(name: &str, value: &str) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: None,
        expr: Some(expr(value)),
    }
}

pub fn var_expr(name: &str, expr: Expr) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: None,
        expr: Some(expr),
    }
}

pub fn var_typed(name: &str, ty: Type, value: Option<&str>) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr: value.map(expr),
    }
}

pub fn var_typed_expr(name: &str, ty: Type, expr: Option<Expr>) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr,
    }
}

pub fn assign(target: &str, value: &str) -> Stmt {
    Stmt::Assign {
        target: expr(target),
        value: expr(value),
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

pub fn assign_op(op: AssignOp, target: &str, value: &str) -> Stmt {
    Stmt::AssignOp {
        target: expr(target),
        op,
        value: expr(value),
    }
}

pub fn assign_op_expr(op: AssignOp, target: Expr, value: Expr) -> Stmt {
    Stmt::AssignOp { target, op, value }
}

pub fn increment(expr_str: &str) -> Stmt {
    Stmt::Increment(expr(expr_str))
}

pub fn increment_expr(expr: Expr) -> Stmt {
    Stmt::Increment(expr)
}

pub fn call_stmt(expr_str: &str) -> Stmt {
    Stmt::Call(expr(expr_str))
}

pub fn call_stmt_expr(expr: Expr) -> Stmt {
    Stmt::Call(expr)
}

pub fn if_block(cond: &str, then_block: Block, else_block: Option<Block>) -> Stmt {
    Stmt::If {
        cond: expr(cond),
        then_block,
        else_block,
    }
}

pub fn if_block_expr(cond: Expr, then_block: Block, else_block: Option<Block>) -> Stmt {
    Stmt::If {
        cond,
        then_block,
        else_block,
    }
}

pub fn for_loop(init: ForInit, cond: &str, step: ForStep, body: Block) -> Stmt {
    Stmt::For {
        init,
        cond: expr(cond),
        step,
        body,
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

pub fn for_init_var(name: &str, value: &str) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: None,
        expr: expr(value),
    }
}

pub fn for_init_var_expr(name: &str, expr: Expr) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: None,
        expr,
    }
}

pub fn for_init_var_typed(name: &str, ty: Type, value: &str) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr: expr(value),
    }
}

pub fn for_init_var_typed_expr(name: &str, ty: Type, expr: Expr) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr,
    }
}

pub fn for_init_assign(target: &str, value: &str) -> ForInit {
    ForInit::Assign {
        target: expr(target),
        value: expr(value),
    }
}

pub fn for_init_assign_expr(target: Expr, value: Expr) -> ForInit {
    ForInit::Assign { target, value }
}

pub fn for_step_increment(expr_str: &str) -> ForStep {
    ForStep::Increment(expr(expr_str))
}

pub fn for_step_increment_expr(expr: Expr) -> ForStep {
    ForStep::Increment(expr)
}

pub fn for_step_assign(target: &str, value: &str) -> ForStep {
    ForStep::Assign {
        target: expr(target),
        value: expr(value),
    }
}

pub fn for_step_assign_expr(target: Expr, value: Expr) -> ForStep {
    ForStep::Assign { target, value }
}

pub fn for_step_assign_op(op: AssignOp, target: &str, value: &str) -> ForStep {
    ForStep::AssignOp {
        target: expr(target),
        op,
        value: expr(value),
    }
}

pub fn for_step_assign_op_expr(op: AssignOp, target: Expr, value: Expr) -> ForStep {
    ForStep::AssignOp { target, op, value }
}

pub fn parse_expr(input: &str) -> Result<Expr, ParseError> {
    Expr::parse(input)
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

pub fn assign_xy(target: &str, x_value: &str, y_value: &str) -> Vec<Stmt> {
    vec![
        assign(&format!("{target}.x"), x_value),
        assign(&format!("{target}.y"), y_value),
    ]
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

pub fn var_matrix(prefix: &str, n: usize, init: &str) -> Vec<Stmt> {
    for_each_mat_entry(n, |row, col| var(&format!("{prefix}_{row}{col}"), init))
}

pub fn var_matrix_expr<F>(prefix: &str, n: usize, mut value: F) -> Vec<Stmt>
where
    F: FnMut(usize, usize) -> String,
{
    for_each_mat_entry(n, |row, col| {
        let expr = value(row, col);
        var(&format!("{prefix}_{row}{col}"), &expr)
    })
}

pub fn assign_op_matrix_from_prefix_scaled(
    op: AssignOp,
    dest_prefix: &str,
    src_prefix: &str,
    n: usize,
    scale: Option<&str>,
) -> Vec<Stmt> {
    for_each_mat_entry(n, |row, col| {
        let target = format!("{dest_prefix}_{row}{col}");
        let mut value = format!("{src_prefix}_{row}{col}");
        if let Some(scale) = scale {
            value.push_str(" * ");
            value.push_str(scale);
        }
        assign_op(op, &target, &value)
    })
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
        if let Some(scale) = scale.clone() {
            value = Expr::binary(value, BinaryOp::Mul, scale);
        }
        assign_op_expr(op, target, value)
    })
}

pub fn assign_op_matrix_diag(op: AssignOp, prefix: &str, n: usize, value: &str) -> Vec<Stmt> {
    let mut out = Vec::new();
    for idx in 0..n {
        out.push(assign_op(op, &format!("{prefix}_{idx}{idx}"), value));
    }
    out
}

pub fn assign_matrix_array_from_prefix_scaled(
    matrix_array: &str,
    base_prefix: &str,
    src_prefix: &str,
    n: usize,
    scale: Option<&str>,
) -> Vec<Stmt> {
    for_each_mat_entry(n, |row, col| {
        let base = format!("{base_prefix}_{row}");
        let target = format!("{matrix_array}[{base} + {col}u]");
        let mut value = format!("{src_prefix}_{row}{col}");
        if let Some(scale) = scale {
            value.push_str(" * ");
            value.push_str(scale);
        }
        assign(&target, &value)
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
        let index = Expr::binary(base, BinaryOp::Add, Expr::lit_u32(col as u32));
        let target = Expr::ident(matrix_array).index(index);
        let mut value = Expr::ident(format!("{src_prefix}_{row}{col}"));
        if let Some(scale) = scale.clone() {
            value = Expr::binary(value, BinaryOp::Mul, scale);
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
        let stmt = let_("idx", "global_id.x");
        assert!(matches!(stmt, Stmt::Let { .. }));

        let stmt = var_typed("normal", Type::Custom("Vector2".to_string()), None);
        assert!(matches!(stmt, Stmt::Var { .. }));

        let stmt = assign_op(AssignOp::Add, "diag_u", "coeff_time");
        assert!(matches!(stmt, Stmt::AssignOp { .. }));
    }

    #[test]
    fn dsl_parse_expr_reports_errors() {
        let err = parse_expr("1 +").unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEof));
    }

    #[test]
    fn dsl_builds_matrix_helpers() {
        let stmts = var_matrix("diag", 2, "0.0");
        assert_eq!(stmts.len(), 4);
        assert!(stmts.iter().all(|stmt| matches!(stmt, Stmt::Var { .. })));
        let stmts =
            assign_op_matrix_from_prefix_scaled(AssignOp::Add, "diag", "jac", 2, Some("area"));
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
