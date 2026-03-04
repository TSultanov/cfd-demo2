use std::fmt;

use super::expr::Expr;
use super::types::Type;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

impl Block {
    pub fn new(stmts: Vec<Stmt>) -> Self {
        Self { stmts }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Comment(String),
    Let {
        name: String,
        ty: Option<Type>,
        expr: Expr,
    },
    Var {
        name: String,
        ty: Option<Type>,
        expr: Option<Expr>,
    },
    Assign {
        target: Expr,
        value: Expr,
    },
    AssignOp {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    If {
        cond: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    For {
        init: ForInit,
        cond: Expr,
        step: ForStep,
        body: Block,
    },
    /// WGSL `loop { ... }` — an infinite loop exited by `break`.
    Loop {
        body: Block,
    },
    /// WGSL `while (cond) { ... }` — a pre-condition loop.
    While {
        cond: Expr,
        body: Block,
    },
    Break,
    Continue,
    Return(Option<Expr>),
    Call(Expr),
    Increment(Expr),
    Decrement(Expr),
}

impl Stmt {
    /// Render this statement to WGSL source lines using the provided render context.
    pub fn render(&self, ctx: &mut super::types::RenderContext<'_>) {
        match self {
            Stmt::Comment(text) => ctx.line(&format!("// {}", text)),
            Stmt::Let { name, ty, expr } => {
                if let Some(ty) = ty {
                    ctx.line(&format!("let {}: {} = {};", name, ty, expr));
                } else {
                    ctx.line(&format!("let {} = {};", name, expr));
                }
            }
            Stmt::Var { name, ty, expr } => match (ty, expr) {
                (Some(ty), Some(expr)) => {
                    ctx.line(&format!("var {}: {} = {};", name, ty, expr));
                }
                (Some(ty), None) => {
                    ctx.line(&format!("var {}: {};", name, ty));
                }
                (None, Some(expr)) => {
                    ctx.line(&format!("var {} = {};", name, expr));
                }
                (None, None) => {
                    ctx.line(&format!("var {};", name));
                }
            },
            Stmt::Assign { target, value } => {
                ctx.line(&format!("{} = {};", target, value));
            }
            Stmt::AssignOp { target, op, value } => {
                ctx.line(&format!("{} {}= {};", target, op, value));
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
            } => {
                ctx.line(&format!("if ({}) {{", cond));
                ctx.indent();
                for stmt in &then_block.stmts {
                    stmt.render(ctx);
                }
                ctx.dedent();
                if let Some(else_block) = else_block {
                    ctx.line("} else {");
                    ctx.indent();
                    for stmt in &else_block.stmts {
                        stmt.render(ctx);
                    }
                    ctx.dedent();
                    ctx.line("}");
                } else {
                    ctx.line("}");
                }
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => {
                ctx.line(&format!(
                    "for ({}; {}; {}) {{",
                    init.to_wgsl(),
                    cond,
                    step.to_wgsl()
                ));
                ctx.indent();
                for stmt in &body.stmts {
                    stmt.render(ctx);
                }
                ctx.dedent();
                ctx.line("}");
            }
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    ctx.line(&format!("return {};", expr));
                } else {
                    ctx.line("return;");
                }
            }
            Stmt::Loop { body } => {
                ctx.line("loop {");
                ctx.indent();
                for stmt in &body.stmts {
                    stmt.render(ctx);
                }
                ctx.dedent();
                ctx.line("}");
            }
            Stmt::While { cond, body } => {
                ctx.line(&format!("while ({}) {{", cond));
                ctx.indent();
                for stmt in &body.stmts {
                    stmt.render(ctx);
                }
                ctx.dedent();
                ctx.line("}");
            }
            Stmt::Break => ctx.line("break;"),
            Stmt::Continue => ctx.line("continue;"),
            Stmt::Call(expr) => ctx.line(&format!("{};", expr)),
            Stmt::Increment(expr) => ctx.line(&format!("{}++;", expr)),
            Stmt::Decrement(expr) => ctx.line(&format!("{}--;", expr)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignOp {
    Add,
    Sub,
    Mul,
    Div,
    Modulo,
    ShiftRight,
    ShiftLeft,
    BitwiseAnd,
    BitwiseOr,
}

impl fmt::Display for AssignOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssignOp::Add => write!(f, "+"),
            AssignOp::Sub => write!(f, "-"),
            AssignOp::Mul => write!(f, "*"),
            AssignOp::Div => write!(f, "/"),
            AssignOp::Modulo => write!(f, "%"),
            AssignOp::ShiftRight => write!(f, ">>"),
            AssignOp::ShiftLeft => write!(f, "<<"),
            AssignOp::BitwiseAnd => write!(f, "&"),
            AssignOp::BitwiseOr => write!(f, "|"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForInit {
    Let {
        name: String,
        ty: Option<Type>,
        expr: Expr,
    },
    Var {
        name: String,
        ty: Option<Type>,
        expr: Expr,
    },
    Assign {
        target: Expr,
        value: Expr,
    },
}

impl ForInit {
    pub fn to_wgsl(&self) -> String {
        match self {
            ForInit::Let { name, ty, expr } => {
                if let Some(ty) = ty {
                    format!("let {}: {} = {}", name, ty, expr)
                } else {
                    format!("let {} = {}", name, expr)
                }
            }
            ForInit::Var { name, ty, expr } => {
                if let Some(ty) = ty {
                    format!("var {}: {} = {}", name, ty, expr)
                } else {
                    format!("var {} = {}", name, expr)
                }
            }
            ForInit::Assign { target, value } => format!("{} = {}", target, value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForStep {
    Increment(Expr),
    Decrement(Expr),
    Assign {
        target: Expr,
        value: Expr,
    },
    AssignOp {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
}

impl ForStep {
    pub fn to_wgsl(&self) -> String {
        match self {
            ForStep::Increment(expr) => format!("{}++", expr),
            ForStep::Decrement(expr) => format!("{}--", expr),
            ForStep::Assign { target, value } => format!("{} = {}", target, value),
            ForStep::AssignOp { target, op, value } => format!("{} {}= {}", target, op, value),
        }
    }
}

/// Render a statement block into WGSL source lines without function/module wrappers.
pub fn render_block_lines(block: &Block) -> Vec<String> {
    let mut out = String::new();
    {
        let mut ctx = super::types::RenderContext::new(&mut out);
        for stmt in &block.stmts {
            stmt.render(&mut ctx);
        }
    }
    out.lines().map(|line| line.to_string()).collect()
}

/// Render a statement slice into WGSL source lines.
pub fn render_stmt_lines(stmts: &[Stmt]) -> Vec<String> {
    render_block_lines(&Block::new(stmts.to_vec()))
}

/// Collect local symbol declarations (`let`/`var` and loop init symbols)
/// from a statement slice in deterministic first-seen order.
pub fn collect_local_symbols(stmts: &[Stmt]) -> Vec<String> {
    fn push_unique_symbol(out: &mut Vec<String>, name: &str) {
        if !out.iter().any(|existing| existing == name) {
            out.push(name.to_string());
        }
    }

    fn visit_stmt(stmt: &Stmt, out: &mut Vec<String>) {
        match stmt {
            Stmt::Let { name, .. } | Stmt::Var { name, .. } => push_unique_symbol(out, name),
            Stmt::If {
                then_block,
                else_block,
                ..
            } => {
                for inner in &then_block.stmts {
                    visit_stmt(inner, out);
                }
                if let Some(else_block) = else_block {
                    for inner in &else_block.stmts {
                        visit_stmt(inner, out);
                    }
                }
            }
            Stmt::For { init, body, .. } => {
                match init {
                    ForInit::Let { name, .. } | ForInit::Var { name, .. } => {
                        push_unique_symbol(out, name)
                    }
                    ForInit::Assign { .. } => {}
                }
                for inner in &body.stmts {
                    visit_stmt(inner, out);
                }
            }
            Stmt::Loop { body } | Stmt::While { body, .. } => {
                for inner in &body.stmts {
                    visit_stmt(inner, out);
                }
            }
            Stmt::Comment(_)
            | Stmt::Assign { .. }
            | Stmt::AssignOp { .. }
            | Stmt::Return(_)
            | Stmt::Call(_)
            | Stmt::Increment(_)
            | Stmt::Decrement(_)
            | Stmt::Break
            | Stmt::Continue => {}
        }
    }

    let mut out = Vec::new();
    for stmt in stmts {
        visit_stmt(stmt, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_local_symbols_walks_nested_blocks_in_order() {
        let stmts = vec![
            Stmt::Let {
                name: "a".to_string(),
                ty: None,
                expr: 1.0.into(),
            },
            Stmt::For {
                init: ForInit::Var {
                    name: "k".to_string(),
                    ty: Some(Type::U32),
                    expr: 0u32.into(),
                },
                cond: Expr::ident("k").lt(4u32),
                step: ForStep::Increment(Expr::ident("k")),
                body: Block::new(vec![Stmt::If {
                    cond: Expr::ident("k").eq(0u32),
                    then_block: Block::new(vec![Stmt::Var {
                        name: "inner".to_string(),
                        ty: Some(Type::F32),
                        expr: Some(0.0.into()),
                    }]),
                    else_block: Some(Block::new(vec![Stmt::Let {
                        name: "alt".to_string(),
                        ty: None,
                        expr: 2.0.into(),
                    }])),
                }]),
            },
            Stmt::Var {
                name: "a".to_string(),
                ty: Some(Type::F32),
                expr: Some(3.0.into()),
            },
        ];

        assert_eq!(
            collect_local_symbols(&stmts),
            vec!["a", "k", "inner", "alt"]
        );
    }

    #[test]
    fn render_stmt_lines_produces_expected_output() {
        let stmts = vec![
            Stmt::Let {
                name: "x".to_string(),
                ty: None,
                expr: Expr::ident("a") + Expr::ident("b"),
            },
            Stmt::Assign {
                target: Expr::ident("y"),
                value: Expr::ident("x"),
            },
        ];
        let lines = render_stmt_lines(&stmts);
        assert_eq!(lines, vec!["let x = a + b;", "y = x;"]);
    }

    #[test]
    fn stmt_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Stmt>();
        assert_send_sync::<Block>();
        assert_send_sync::<ForInit>();
        assert_send_sync::<ForStep>();
    }
}
