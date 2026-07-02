//! IR → Rust transpiler for the CPU backend (Phase 3).
//!
//! The *second renderer* over the same `KernelProgram` IR the WGSL emitter
//! (`fusion::lower_kernel_program_to_wgsl`) consumes. It renders each kernel to a
//! standalone Rust function that the host crate compiles (via `build.rs`); run,
//! it executes at native speed and lets the optimiser auto-vectorise the regular
//! interior — the performance counterpart to the interpreter.
//!
//! Output is pure text and depends only on `cfd2_ir`, so it is callable from both
//! `build.rs` and the runtime crate. Generated functions have the signature
//!
//! ```ignore
//! pub fn <fn>(bufs: &Buffers, idx: u32, constants: &GpuConstants) { … }
//! ```
//!
//! and target the host prelude (`solver::cpu::transpile_rt`): `Vec2/Vec3/Vec4`,
//! `dot`/`length`/`mix`/…, and typed atomic load/store helpers
//! (`ldf/stf/ld2/st2/st2c/ldu/ldi/stu/sti`). Buffer handles (`&[AtomicU32]`) are
//! resolved once per kernel via `bufs.atom("name")`.

use std::collections::BTreeMap;

use cfd2_ir::ast::{AssignOp, BinaryOp, Block, Expr, ExprNode, ForInit, ForStep, Literal, Stmt, Type, UnaryOp};
use crate::solver::ir::KernelProgram;

#[derive(Clone, Copy, PartialEq)]
enum BufKind {
    F32,
    U32,
    I32,
    Vec2,
}

/// Buffer name → element kind, derived from the kernel's bindings.
struct Tx {
    bufs: BTreeMap<String, BufKind>,
}

impl Tx {
    fn from_program(p: &KernelProgram) -> Self {
        let mut bufs = BTreeMap::new();
        for b in &p.bindings {
            let kind = match b.wgsl_type.as_str() {
                "array<f32>" => BufKind::F32,
                "array<u32>" => BufKind::U32,
                "array<i32>" => BufKind::I32,
                "array<Vector2>" => BufKind::Vec2,
                _ => continue, // uniforms (Constants) and unsupported types
            };
            bufs.insert(b.name.clone(), kind);
        }
        Tx { bufs }
    }

    fn kind_of(&self, expr: &Expr) -> Option<BufKind> {
        match expr.node() {
            ExprNode::Ident(name) => self.bufs.get(name).copied(),
            _ => None,
        }
    }
}

/// Emit a complete Rust function for `program`.
pub fn emit_kernel_fn(fn_name: &str, program: &KernelProgram) -> String {
    let tx = Tx::from_program(program);
    let mut s = String::new();
    s.push_str("#[allow(unused_variables, unused_mut, unused_parens, clippy::all)]\n");
    s.push_str(&format!(
        "pub fn {fn_name}(bufs: &Buffers, start: u32, end: u32, constants: &GpuConstants) {{\n"
    ));
    // Resolve each buffer handle once PER CHUNK, not per index: `bufs.atom` is
    // a HashMap<String, _> lookup, and the per-index entry point measured
    // ~25-30% of the assembly phase in name lookups alone (the grad_state
    // kernel resolves 23 handles per cell). The dispatch loop lives inside
    // the function so the optimiser can also keep the handles in registers.
    for name in tx.bufs.keys() {
        s.push_str(&format!("    let {name} = bufs.atom(\"{name}\");\n"));
    }
    s.push_str("    for idx in start..end {\n");
    for stmt in program
        .indexing
        .iter()
        .chain(&program.preamble)
        .chain(&program.body)
    {
        emit_stmt(stmt, 2, &tx, &mut s);
    }
    s.push_str("    }\n");
    s.push_str("}\n");
    s
}

fn rust_type(ty: &Type) -> String {
    match ty {
        Type::F32 => "f32".into(),
        Type::U32 => "u32".into(),
        Type::I32 => "i32".into(),
        Type::Bool => "bool".into(),
        Type::Vec2(_) => "Vec2".into(),
        Type::Vec3(_) => "Vec3".into(),
        Type::Vec4(_) => "Vec4".into(),
        Type::Custom(n) if n == "Vector2" => "Vec2".into(),
        Type::Custom(n) if n == "Vector3" => "Vec3".into(),
        Type::Custom(n) if n == "Vector4" => "Vec4".into(),
        other => panic!("rust_type: unsupported type {other}"),
    }
}

fn ld_fn(kind: BufKind) -> &'static str {
    match kind {
        BufKind::F32 => "ldf",
        BufKind::U32 => "ldu",
        BufKind::I32 => "ldi",
        BufKind::Vec2 => "ld2",
    }
}

fn emit_expr(expr: &Expr, tx: &Tx, out: &mut String) {
    match expr.node() {
        ExprNode::Literal(lit) => out.push_str(&emit_literal(lit)),
        ExprNode::Ident(name) => out.push_str(name),
        ExprNode::Field { base, field } => {
            // `constants.dt` and vector component access (`v.x`, `ld2(..).x`).
            emit_expr(base, tx, out);
            out.push('.');
            out.push_str(field);
        }
        ExprNode::Index { base, index } => {
            if let Some(kind) = tx.kind_of(base) {
                // Buffer element load through the typed helper.
                out.push_str(ld_fn(kind));
                out.push('(');
                emit_expr(base, tx, out);
                out.push_str(", (");
                emit_expr(index, tx, out);
                out.push_str(") as usize)");
            } else {
                // Non-buffer index (not expected); fall back to slice indexing.
                emit_expr(base, tx, out);
                out.push_str("[(");
                emit_expr(index, tx, out);
                out.push_str(") as usize]");
            }
        }
        ExprNode::Unary { op, expr: inner } => {
            let sym = match op {
                UnaryOp::Negate => "-",
                UnaryOp::Not => "!",
                UnaryOp::AddressOf => "&",
                UnaryOp::Deref => "*",
            };
            out.push('(');
            out.push_str(sym);
            emit_expr(inner, tx, out);
            out.push(')');
        }
        ExprNode::Binary { left, op, right } => {
            out.push('(');
            emit_expr(left, tx, out);
            out.push(' ');
            out.push_str(emit_binop(*op));
            out.push(' ');
            emit_expr(right, tx, out);
            out.push(')');
        }
        ExprNode::Call { callee, args } => emit_call(callee, args, tx, out),
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

fn s_expr(expr: &Expr, tx: &Tx) -> String {
    let mut s = String::new();
    emit_expr(expr, tx, &mut s);
    s
}

fn emit_call(callee: &Expr, args: &[Expr], tx: &Tx, out: &mut String) {
    let name = match callee.node() {
        ExprNode::Ident(n) => n.as_str(),
        _ => panic!("call of non-ident callee"),
    };
    let a: Vec<String> = args.iter().map(|e| s_expr(e, tx)).collect();
    let emitted = match name {
        "sqrt" | "abs" | "floor" | "ceil" | "round" | "exp2" | "log2" => {
            format!("({}).{name}()", a[0])
        }
        "sign" => format!("({}).signum()", a[0]),
        "min" => format!("({}).min({})", a[0], a[1]),
        "max" => format!("({}).max({})", a[0], a[1]),
        "pow" => format!("({}).powf({})", a[0], a[1]),
        "fma" => format!("({}).mul_add({}, {})", a[0], a[1], a[2]),
        "clamp" => format!("({}).clamp({}, {})", a[0], a[1], a[2]),
        "select" => format!("(if {} {{ {} }} else {{ {} }})", a[2], a[1], a[0]),
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
        "f32" => format!("(({}) as f32)", a[0]),
        "u32" => format!("(({}) as u32)", a[0]),
        "i32" => format!("(({}) as i32)", a[0]),
        "bool" => format!("(({}) != 0)", a[0]),
        "arrayLength" => {
            // arg is `&buffer`; the handle is `&[AtomicU32]` (element count for
            // scalar buffers, which is where arrayLength is used).
            let inner = match args[0].node() {
                ExprNode::Unary { op: UnaryOp::AddressOf, expr } => s_expr(expr, tx),
                _ => a[0].clone(),
            };
            format!("({inner}.len() as u32)")
        }
        other if other.starts_with("bitcast<f32>") => format!("f32::from_bits(({}) as u32)", a[0]),
        other if other.starts_with("bitcast<u32>") => format!("(({}).to_bits())", a[0]),
        other => format!("{other}({})", a.join(", ")),
    };
    out.push_str(&emitted);
}

/// Emit an lvalue store: `target = value`-style, dispatching buffer element
/// stores to the typed helpers.
fn emit_store(target: &Expr, tx: &Tx, indent: usize, out: &mut String, rhs: &str) {
    let pad = "    ".repeat(indent);
    match target.node() {
        ExprNode::Index { base, index } => match tx.kind_of(base) {
            Some(kind) => {
                let st = match kind {
                    BufKind::F32 => "stf",
                    BufKind::U32 => "stu",
                    BufKind::I32 => "sti",
                    BufKind::Vec2 => "st2",
                };
                out.push_str(&format!(
                    "{pad}{st}({}, ({}) as usize, {rhs});\n",
                    s_expr(base, tx),
                    s_expr(index, tx)
                ));
            }
            None => out.push_str(&format!(
                "{pad}{}[({}) as usize] = {rhs};\n",
                s_expr(base, tx),
                s_expr(index, tx)
            )),
        },
        ExprNode::Field { base, field } => {
            // Component store: into a buffer Vector2 element, or a local vector.
            let comp = match field.as_str() {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => panic!("unsupported component store .{field}"),
            };
            if let ExprNode::Index { base: bbase, index } = base.node() {
                if tx.kind_of(bbase).is_some() {
                    out.push_str(&format!(
                        "{pad}st2c({}, ({}) as usize, {comp}, {rhs});\n",
                        s_expr(bbase, tx),
                        s_expr(index, tx)
                    ));
                    return;
                }
            }
            // Local vector component.
            out.push_str(&format!("{pad}{}.{field} = {rhs};\n", s_expr(base, tx)));
        }
        ExprNode::Ident(name) => out.push_str(&format!("{pad}{name} = {rhs};\n")),
        other => panic!("unsupported store target {other:?}"),
    }
}

fn emit_stmt(stmt: &Stmt, indent: usize, tx: &Tx, out: &mut String) {
    let pad = "    ".repeat(indent);
    match stmt {
        Stmt::Comment(t) => out.push_str(&format!("{pad}// {t}\n")),
        Stmt::Let { name, expr, .. } => {
            out.push_str(&format!("{pad}let {name} = {};\n", s_expr(expr, tx)));
        }
        Stmt::Var { name, ty, expr } => match (ty, expr) {
            (_, Some(e)) => out.push_str(&format!("{pad}let mut {name} = {};\n", s_expr(e, tx))),
            (Some(ty), None) => out.push_str(&format!(
                "{pad}let mut {name}: {} = Default::default();\n",
                rust_type(ty)
            )),
            (None, None) => {
                out.push_str(&format!("{pad}let mut {name} = Default::default();\n"))
            }
        },
        Stmt::Assign { target, value } => {
            let rhs = s_expr(value, tx);
            emit_store(target, tx, indent, out, &rhs);
        }
        Stmt::AssignOp { target, op, value } => {
            // Read-modify-write. For buffer targets this lowers through the typed
            // store helpers with an explicit load of the current value.
            let cur = s_expr(target, tx);
            let rhs = format!("({} {} ({}))", cur, emit_binop(binop_of(*op)), s_expr(value, tx));
            emit_store(target, tx, indent, out, &rhs);
        }
        Stmt::If { cond, then_block, else_block } => {
            out.push_str(&format!("{pad}if {} {{\n", s_expr(cond, tx)));
            emit_block(then_block, indent + 1, tx, out);
            if let Some(eb) = else_block {
                out.push_str(&format!("{pad}}} else {{\n"));
                emit_block(eb, indent + 1, tx, out);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::For { init, cond, step, body } => {
            out.push_str(&format!("{pad}{{\n"));
            emit_for_init(init, indent + 1, tx, out);
            let ipad = "    ".repeat(indent + 1);
            out.push_str(&format!("{ipad}while {} {{\n", s_expr(cond, tx)));
            emit_block(body, indent + 2, tx, out);
            emit_for_step(step, indent + 2, tx, out);
            out.push_str(&format!("{ipad}}}\n"));
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::While { cond, body } => {
            out.push_str(&format!("{pad}while {} {{\n", s_expr(cond, tx)));
            emit_block(body, indent + 1, tx, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Loop { body } => {
            out.push_str(&format!("{pad}loop {{\n"));
            emit_block(body, indent + 1, tx, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Break => out.push_str(&format!("{pad}break;\n")),
        Stmt::Continue => out.push_str(&format!("{pad}continue;\n")),
        Stmt::Return(e) => match e {
            Some(e) => out.push_str(&format!("{pad}return {};\n", s_expr(e, tx))),
            None => out.push_str(&format!("{pad}return;\n")),
        },
        Stmt::Call(expr) => out.push_str(&format!("{pad}{};\n", s_expr(expr, tx))),
        Stmt::Increment(e) => {
            let rhs = format!("({} + 1)", s_expr(e, tx));
            emit_store(e, tx, indent, out, &rhs);
        }
        Stmt::Decrement(e) => {
            let rhs = format!("({} - 1)", s_expr(e, tx));
            emit_store(e, tx, indent, out, &rhs);
        }
    }
}

fn binop_of(op: AssignOp) -> BinaryOp {
    match op {
        AssignOp::Add => BinaryOp::Add,
        AssignOp::Sub => BinaryOp::Sub,
        AssignOp::Mul => BinaryOp::Mul,
        AssignOp::Div => BinaryOp::Div,
        AssignOp::Modulo => BinaryOp::Modulo,
        AssignOp::ShiftRight => BinaryOp::ShiftRight,
        AssignOp::ShiftLeft => BinaryOp::ShiftLeft,
        AssignOp::BitwiseAnd => BinaryOp::BitwiseAnd,
        AssignOp::BitwiseOr => BinaryOp::BitwiseOr,
    }
}

fn emit_block(block: &Block, indent: usize, tx: &Tx, out: &mut String) {
    for stmt in &block.stmts {
        emit_stmt(stmt, indent, tx, out);
    }
}

fn emit_for_init(init: &ForInit, indent: usize, tx: &Tx, out: &mut String) {
    let pad = "    ".repeat(indent);
    match init {
        ForInit::Let { name, expr, .. } | ForInit::Var { name, expr, .. } => {
            out.push_str(&format!("{pad}let mut {name} = {};\n", s_expr(expr, tx)));
        }
        ForInit::Assign { target, value } => {
            let rhs = s_expr(value, tx);
            emit_store(target, tx, indent, out, &rhs);
        }
    }
}

fn emit_for_step(step: &ForStep, indent: usize, tx: &Tx, out: &mut String) {
    match step {
        ForStep::Increment(e) => {
            let rhs = format!("({} + 1)", s_expr(e, tx));
            emit_store(e, tx, indent, out, &rhs);
        }
        ForStep::Decrement(e) => {
            let rhs = format!("({} - 1)", s_expr(e, tx));
            emit_store(e, tx, indent, out, &rhs);
        }
        ForStep::Assign { target, value } => {
            let rhs = s_expr(value, tx);
            emit_store(target, tx, indent, out, &rhs);
        }
        ForStep::AssignOp { target, op, value } => {
            let cur = s_expr(target, tx);
            let rhs = format!("({} {} ({}))", cur, emit_binop(binop_of(*op)), s_expr(value, tx));
            emit_store(target, tx, indent, out, &rhs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd2_ir::kernel::{DispatchDomain, KernelBinding, BindingAccess, KernelProgram, LaunchSemantics};

    fn prog_with(bindings: Vec<KernelBinding>, body: Vec<Stmt>) -> KernelProgram {
        let mut p = KernelProgram::new(
            "k/test",
            DispatchDomain::Cells,
            LaunchSemantics::new([64, 1, 1], "idx", Some("idx >= n")),
            bindings,
        );
        p.body = body;
        p
    }

    #[test]
    fn emits_buffer_loads_and_stores() {
        let bindings = vec![
            KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
            KernelBinding::new(0, 1, "rhs", "array<f32>", BindingAccess::ReadWriteStorage),
            KernelBinding::new(0, 2, "grad", "array<Vector2>", BindingAccess::ReadWriteStorage),
        ];
        // rhs[idx] = state[idx*4+0] + 1.0;
        let body = vec![Stmt::Assign {
            target: Expr::ident("rhs").index(Expr::ident("idx")),
            value: Expr::ident("state").index(Expr::ident("idx") * 4u32 + 0u32) + 1.0,
        }];
        let src = emit_kernel_fn("k", &prog_with(bindings, body));
        assert!(src.contains("let state = bufs.atom(\"state\");"));
        assert!(src.contains("let rhs = bufs.atom(\"rhs\");"));
        assert!(src.contains("stf(rhs, (idx) as usize,"), "got:\n{src}");
        assert!(src.contains("ldf(state,") && src.contains("1.0f32"), "got:\n{src}");
    }

    #[test]
    fn emits_vec2_component_store_and_constants() {
        let bindings = vec![
            KernelBinding::new(0, 0, "grad", "array<Vector2>", BindingAccess::ReadWriteStorage),
            KernelBinding::new(1, 0, "constants", "Constants", BindingAccess::Uniform),
        ];
        // grad[idx].x = constants.dt;
        let body = vec![Stmt::Assign {
            target: Expr::ident("grad").index(Expr::ident("idx")).field("x"),
            value: Expr::ident("constants").field("dt"),
        }];
        let src = emit_kernel_fn("k", &prog_with(bindings, body));
        assert!(src.contains("st2c(grad, (idx) as usize, 0, constants.dt);"), "got:\n{src}");
        // `constants` is a uniform, not a buffer handle.
        assert!(!src.contains("bufs.atom(\"constants\")"));
    }

    #[test]
    fn emits_for_loop_and_intrinsics() {
        let bindings = vec![KernelBinding::new(
            0, 0, "cell_vols", "array<f32>", BindingAccess::ReadOnlyStorage,
        )];
        let body = vec![Stmt::For {
            init: ForInit::Var { name: "k".into(), ty: Some(Type::U32), expr: 0u32.into() },
            cond: Expr::ident("k").lt(Expr::call_named("arrayLength", vec![Expr::ident("cell_vols").addr_of()])),
            step: ForStep::Increment(Expr::ident("k")),
            body: Block::new(vec![Stmt::Let {
                name: "a".into(),
                ty: None,
                expr: Expr::call_named("sqrt", vec![Expr::ident("cell_vols").index(Expr::ident("k"))]),
            }]),
        }];
        let src = emit_kernel_fn("k", &prog_with(bindings, body));
        assert!(src.contains("while (k < (cell_vols.len() as u32)) {"), "got:\n{src}");
        assert!(src.contains("let a = (ldf(cell_vols, (k) as usize)).sqrt();"), "got:\n{src}");
        assert!(src.contains("k = (k + 1);"));
    }
}
