//! Phase-3 IR→Rust transpiler (emitter core).
//!
//! This is the *second renderer* over the same `cfd2_ir::ast` that the WGSL
//! emitter (`lower_kernel_program_to_wgsl`) consumes — the basis for a
//! native-speed CPU path. Where the interpreter walks the AST at runtime, the
//! transpiler renders each kernel to Rust source once; compiled, it runs at
//! native speed and lets the optimiser auto-vectorise the regular interior.
//!
//! This module is the emitter (AST → Rust text) plus unit tests. The generated
//! code targets a small prelude:
//!
//! ```ignore
//! #[derive(Clone, Copy)] struct Vec2 { x: f32, y: f32 }
//! impl Vec2 { fn new(x: f32, y: f32) -> Self { Self { x, y } } }
//! // + Add/Sub/Mul(scalar) and `fn dot(a: Vec2, b: Vec2) -> f32`, plus typed
//! //   buffer accessors `buf_f32(name) -> &[f32]` etc. and a `Constants` struct.
//! ```
//!
//! Wiring the emitted modules through `build.rs` and selecting them at runtime is
//! the remaining productionisation step; the interpreter already provides the
//! functional, validated CPU backend.

use cfd2_ir::ast::{BinaryOp, Block, Expr, ExprNode, ForInit, ForStep, Literal, Stmt, UnaryOp};

/// Render an expression to a Rust expression string. Binary subexpressions are
/// fully parenthesised — verbose but always correctly grouped for generated code.
pub fn emit_expr(expr: &Expr) -> String {
    match expr.node() {
        ExprNode::Literal(lit) => emit_literal(lit),
        ExprNode::Ident(name) => name.clone(),
        ExprNode::Field { base, field } => {
            // `vec.x` / `constants.dt` map 1:1 onto Rust field access.
            format!("{}.{}", emit_expr(base), field)
        }
        ExprNode::Index { base, index } => {
            // WGSL indices are u32/i32; Rust needs usize.
            format!("{}[({}) as usize]", emit_expr(base), emit_expr(index))
        }
        ExprNode::Unary { op, expr: inner } => match op {
            UnaryOp::Negate => format!("(-{})", emit_expr(inner)),
            UnaryOp::Not => format!("(!{})", emit_expr(inner)),
            // Pointer ops only appear inside intrinsics (arrayLength/atomics),
            // which are handled in `emit_call`; a bare one is a bug.
            UnaryOp::AddressOf => format!("(&{})", emit_expr(inner)),
            UnaryOp::Deref => format!("(*{})", emit_expr(inner)),
        },
        ExprNode::Binary { left, op, right } => {
            format!("({} {} {})", emit_expr(left), emit_binop(*op), emit_expr(right))
        }
        ExprNode::Call { callee, args } => emit_call(callee, args),
    }
}

fn emit_literal(lit: &Literal) -> String {
    match lit {
        Literal::Bool(b) => b.to_string(),
        Literal::Int(v) => format!("{v}i32"),
        Literal::Uint(v) => format!("{v}u32"),
        Literal::Float(s) => match s.as_str() {
            "nan()" => "f32::NAN".to_string(),
            "inf()" => "f32::INFINITY".to_string(),
            "-inf()" => "f32::NEG_INFINITY".to_string(),
            other => format!("{other}f32"),
        },
    }
}

fn emit_binop(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Modulo => "%",
        BinaryOp::Less => "<",
        BinaryOp::LessEq => "<=",
        BinaryOp::Greater => ">",
        BinaryOp::GreaterEq => ">=",
        BinaryOp::Equal => "==",
        BinaryOp::NotEqual => "!=",
        BinaryOp::And => "&&",
        BinaryOp::Or => "||",
        BinaryOp::ShiftRight => ">>",
        BinaryOp::ShiftLeft => "<<",
        BinaryOp::BitwiseAnd => "&",
        BinaryOp::BitwiseOr => "|",
    }
}

fn emit_assign_op(op: cfd2_ir::ast::AssignOp) -> &'static str {
    use cfd2_ir::ast::AssignOp::*;
    match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        Div => "/",
        Modulo => "%",
        ShiftRight => ">>",
        ShiftLeft => "<<",
        BitwiseAnd => "&",
        BitwiseOr => "|",
    }
}

fn ident_name(expr: &Expr) -> Option<&str> {
    match expr.node() {
        ExprNode::Ident(n) => Some(n),
        _ => None,
    }
}

fn emit_call(callee: &Expr, args: &[Expr]) -> String {
    let name = ident_name(callee).unwrap_or_else(|| panic!("call of non-ident callee"));
    let a: Vec<String> = args.iter().map(emit_expr).collect();
    match name {
        // Method-style scalar math.
        "sqrt" | "abs" | "floor" | "ceil" | "round" | "exp2" | "log2" | "signum" => {
            format!("({}).{name}()", a[0])
        }
        "sign" => format!("({}).signum()", a[0]),
        "min" => format!("({}).min({})", a[0], a[1]),
        "max" => format!("({}).max({})", a[0], a[1]),
        "pow" => format!("({}).powf({})", a[0], a[1]),
        "fma" => format!("({}).mul_add({}, {})", a[0], a[1], a[2]),
        "clamp" => format!("({}).clamp({}, {})", a[0], a[1], a[2]),
        // WGSL `select(false_value, true_value, condition)`.
        "select" => format!("(if {} {{ {} }} else {{ {} }})", a[2], a[1], a[0]),
        // Prelude helpers.
        "dot" => format!("dot({}, {})", a[0], a[1]),
        "length" => format!("length({})", a[0]),
        "distance" => format!("distance({}, {})", a[0], a[1]),
        "mix" => format!("mix({}, {}, {})", a[0], a[1], a[2]),
        "smoothstep" => format!("smoothstep({}, {}, {})", a[0], a[1], a[2]),
        "vec2<f32>" if args.len() == 1 => format!("Vec2::splat({})", a[0]),
        "vec2<f32>" => format!("Vec2::new({}, {})", a[0], a[1]),
        "vec3<f32>" if args.len() == 1 => format!("Vec3::splat({})", a[0]),
        "vec3<f32>" => format!("Vec3::new({}, {}, {})", a[0], a[1], a[2]),
        "vec4<f32>" if args.len() == 1 => format!("Vec4::splat({})", a[0]),
        "vec4<f32>" => format!("Vec4::new({}, {}, {}, {})", a[0], a[1], a[2], a[3]),
        // Casts.
        "f32" => format!("(({}) as f32)", a[0]),
        "u32" => format!("(({}) as u32)", a[0]),
        "i32" => format!("(({}) as i32)", a[0]),
        "bool" => format!("(({}) != 0)", a[0]),
        // GPU-only → CPU.
        "arrayLength" => {
            // arg is `&buffer`; the prelude exposes a `.len()`-able accessor.
            let inner = match args[0].node() {
                ExprNode::Unary { op: UnaryOp::AddressOf, expr } => emit_expr(expr),
                _ => emit_expr(&args[0]),
            };
            format!("({}.len() as u32)", inner)
        }
        other if other.starts_with("bitcast<f32>") => format!("f32::from_bits(({}) as u32)", a[0]),
        other if other.starts_with("bitcast<u32>") => format!("(({}).to_bits())", a[0]),
        // Unknown calls pass through (user prelude fn).
        other => format!("{other}({})", a.join(", ")),
    }
}

/// Render a statement to Rust source lines (indented by `indent` levels).
pub fn emit_stmt(stmt: &Stmt, indent: usize, out: &mut String) {
    let pad = "    ".repeat(indent);
    match stmt {
        Stmt::Comment(text) => out.push_str(&format!("{pad}// {text}\n")),
        Stmt::Let { name, expr, .. } => {
            out.push_str(&format!("{pad}let {name} = {};\n", emit_expr(expr)));
        }
        Stmt::Var { name, expr, .. } => match expr {
            Some(e) => out.push_str(&format!("{pad}let mut {name} = {};\n", emit_expr(e))),
            None => out.push_str(&format!("{pad}let mut {name} = Default::default();\n")),
        },
        Stmt::Assign { target, value } => {
            out.push_str(&format!("{pad}{} = {};\n", emit_expr(target), emit_expr(value)));
        }
        Stmt::AssignOp { target, op, value } => {
            out.push_str(&format!(
                "{pad}{} {}= {};\n",
                emit_expr(target),
                emit_assign_op(*op),
                emit_expr(value)
            ));
        }
        Stmt::If { cond, then_block, else_block } => {
            out.push_str(&format!("{pad}if {} {{\n", emit_expr(cond)));
            emit_block(then_block, indent + 1, out);
            if let Some(eb) = else_block {
                out.push_str(&format!("{pad}}} else {{\n"));
                emit_block(eb, indent + 1, out);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::For { init, cond, step, body } => {
            // WGSL `for (init; cond; step) {}` → Rust `init; while cond { body; step }`.
            out.push_str(&format!("{pad}{{\n"));
            emit_for_init(init, indent + 1, out);
            let ipad = "    ".repeat(indent + 1);
            out.push_str(&format!("{ipad}while {} {{\n", emit_expr(cond)));
            emit_block(body, indent + 2, out);
            emit_for_step(step, indent + 2, out);
            out.push_str(&format!("{ipad}}}\n"));
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::While { cond, body } => {
            out.push_str(&format!("{pad}while {} {{\n", emit_expr(cond)));
            emit_block(body, indent + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Loop { body } => {
            out.push_str(&format!("{pad}loop {{\n"));
            emit_block(body, indent + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Break => out.push_str(&format!("{pad}break;\n")),
        Stmt::Continue => out.push_str(&format!("{pad}continue;\n")),
        Stmt::Return(e) => match e {
            Some(e) => out.push_str(&format!("{pad}return {};\n", emit_expr(e))),
            None => out.push_str(&format!("{pad}return;\n")),
        },
        Stmt::Call(expr) => out.push_str(&format!("{pad}{};\n", emit_expr(expr))),
        Stmt::Increment(e) => out.push_str(&format!("{pad}{} += 1;\n", emit_expr(e))),
        Stmt::Decrement(e) => out.push_str(&format!("{pad}{} -= 1;\n", emit_expr(e))),
    }
}

fn emit_block(block: &Block, indent: usize, out: &mut String) {
    for stmt in &block.stmts {
        emit_stmt(stmt, indent, out);
    }
}

fn emit_for_init(init: &ForInit, indent: usize, out: &mut String) {
    let pad = "    ".repeat(indent);
    match init {
        ForInit::Let { name, expr, .. } | ForInit::Var { name, expr, .. } => {
            out.push_str(&format!("{pad}let mut {name} = {};\n", emit_expr(expr)));
        }
        ForInit::Assign { target, value } => {
            out.push_str(&format!("{pad}{} = {};\n", emit_expr(target), emit_expr(value)));
        }
    }
}

fn emit_for_step(step: &ForStep, indent: usize, out: &mut String) {
    let pad = "    ".repeat(indent);
    match step {
        ForStep::Increment(e) => out.push_str(&format!("{pad}{} += 1;\n", emit_expr(e))),
        ForStep::Decrement(e) => out.push_str(&format!("{pad}{} -= 1;\n", emit_expr(e))),
        ForStep::Assign { target, value } => {
            out.push_str(&format!("{pad}{} = {};\n", emit_expr(target), emit_expr(value)));
        }
        ForStep::AssignOp { target, op, value } => {
            out.push_str(&format!(
                "{pad}{} {}= {};\n",
                emit_expr(target),
                emit_assign_op(*op),
                emit_expr(value)
            ));
        }
    }
}

/// Render a kernel body (statement slice) to a Rust function body string.
pub fn emit_kernel_body(stmts: &[Stmt]) -> String {
    let mut out = String::new();
    for stmt in stmts {
        emit_stmt(stmt, 1, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd2_ir::ast::{AssignOp, Expr, Type};

    #[test]
    fn emits_arithmetic_and_intrinsics() {
        let e = Expr::ident("a") + Expr::ident("b") * Expr::ident("c");
        assert_eq!(emit_expr(&e), "(a + (b * c))");

        let s = Expr::call_named("sqrt", vec![Expr::call_named("abs", vec![Expr::ident("x")])]);
        assert_eq!(emit_expr(&s), "((x).abs()).sqrt()");

        let sel = Expr::call_named("select", vec![1.0.into(), 2.0.into(), Expr::ident("c")]);
        assert_eq!(emit_expr(&sel), "(if c { 2.0f32 } else { 1.0f32 })");

        let mn = Expr::call_named("min", vec![Expr::ident("a"), 0.0.into()]);
        assert_eq!(emit_expr(&mn), "(a).min(0.0f32)");
    }

    #[test]
    fn emits_buffer_index_and_casts() {
        let e = Expr::ident("state").index(Expr::ident("idx") * 4u32 + 1u32);
        // The index subexpression is itself fully parenthesised (harmless).
        assert_eq!(emit_expr(&e), "state[(((idx * 4u32) + 1u32)) as usize]");

        let c = Expr::call_named("f32", vec![Expr::ident("k")]);
        assert_eq!(emit_expr(&c), "((k) as f32)");

        let al = Expr::call_named("arrayLength", vec![Expr::ident("cell_vols").addr_of()]);
        assert_eq!(emit_expr(&al), "(cell_vols.len() as u32)");
    }

    #[test]
    fn emits_vec_and_field_access() {
        let v = Expr::call_named("vec2<f32>", vec![Expr::ident("a"), Expr::ident("b")]);
        assert_eq!(emit_expr(&v), "Vec2::new(a, b)");
        assert_eq!(emit_expr(&Expr::ident("n").field("x")), "n.x");
        assert_eq!(emit_expr(&Expr::ident("constants").field("dt")), "constants.dt");
        let d = Expr::call_named("dot", vec![v, Expr::ident("normal")]);
        assert_eq!(emit_expr(&d), "dot(Vec2::new(a, b), normal)");
    }

    #[test]
    fn emits_control_flow() {
        let stmts = vec![
            Stmt::Var { name: "sum".into(), ty: None, expr: Some(0.0.into()) },
            Stmt::For {
                init: ForInit::Var { name: "k".into(), ty: Some(Type::U32), expr: 0u32.into() },
                cond: Expr::ident("k").lt(Expr::ident("n")),
                step: ForStep::Increment(Expr::ident("k")),
                body: Block::new(vec![Stmt::AssignOp {
                    target: Expr::ident("sum"),
                    op: AssignOp::Add,
                    value: Expr::ident("k"),
                }]),
            },
        ];
        let src = emit_kernel_body(&stmts);
        assert!(src.contains("let mut sum = 0.0f32;"));
        assert!(src.contains("let mut k = 0u32;"));
        assert!(src.contains("while (k < n) {"));
        assert!(src.contains("sum += k;"));
        assert!(src.contains("k += 1;"));
    }

    #[test]
    fn emits_if_else_and_assign() {
        let s = Stmt::If {
            cond: Expr::ident("is_boundary"),
            then_block: Block::new(vec![Stmt::Assign {
                target: Expr::ident("rhs").index(Expr::ident("idx")),
                value: Expr::ident("v"),
            }]),
            else_block: Some(Block::new(vec![Stmt::Assign {
                target: Expr::ident("diag"),
                value: Expr::ident("diag") + 1.0,
            }])),
        };
        let mut out = String::new();
        emit_stmt(&s, 0, &mut out);
        assert!(out.contains("if is_boundary {"));
        assert!(out.contains("rhs[(idx) as usize] = v;"));
        assert!(out.contains("} else {"));
        assert!(out.contains("diag = (diag + 1.0f32);"));
    }
}
