use super::wgsl_ast::{
    AssignOp, Block, Expr, ForInit, ForStep, ParseError, Stmt, Type,
};

pub fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("invalid WGSL expression")
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

pub fn let_typed(name: &str, ty: Type, value: &str) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty: Some(ty),
        expr: expr(value),
    }
}

pub fn var(name: &str, value: &str) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: None,
        expr: Some(expr(value)),
    }
}

pub fn var_typed(name: &str, ty: Type, value: Option<&str>) -> Stmt {
    Stmt::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr: value.map(expr),
    }
}

pub fn assign(target: &str, value: &str) -> Stmt {
    Stmt::Assign {
        target: expr(target),
        value: expr(value),
    }
}

pub fn assign_op(op: AssignOp, target: &str, value: &str) -> Stmt {
    Stmt::AssignOp {
        target: expr(target),
        op,
        value: expr(value),
    }
}

pub fn increment(expr_str: &str) -> Stmt {
    Stmt::Increment(expr(expr_str))
}

pub fn call_stmt(expr_str: &str) -> Stmt {
    Stmt::Call(expr(expr_str))
}

pub fn if_block(cond: &str, then_block: Block, else_block: Option<Block>) -> Stmt {
    Stmt::If {
        cond: expr(cond),
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

pub fn for_init_var(name: &str, value: &str) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: None,
        expr: expr(value),
    }
}

pub fn for_init_var_typed(name: &str, ty: Type, value: &str) -> ForInit {
    ForInit::Var {
        name: name.to_string(),
        ty: Some(ty),
        expr: expr(value),
    }
}

pub fn for_init_assign(target: &str, value: &str) -> ForInit {
    ForInit::Assign {
        target: expr(target),
        value: expr(value),
    }
}

pub fn for_step_increment(expr_str: &str) -> ForStep {
    ForStep::Increment(expr(expr_str))
}

pub fn for_step_assign(target: &str, value: &str) -> ForStep {
    ForStep::Assign {
        target: expr(target),
        value: expr(value),
    }
}

pub fn for_step_assign_op(op: AssignOp, target: &str, value: &str) -> ForStep {
    ForStep::AssignOp {
        target: expr(target),
        op,
        value: expr(value),
    }
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
        let stmts = assign_op_matrix_from_prefix_scaled(AssignOp::Add, "diag", "jac", 2, Some("area"));
        assert_eq!(stmts.len(), 4);
        assert!(stmts.iter().all(|stmt| matches!(stmt, Stmt::AssignOp { .. })));
    }
}
