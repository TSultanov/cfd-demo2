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
}
