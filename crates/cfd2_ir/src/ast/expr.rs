use std::fmt;
use std::sync::Arc;

/// A self-contained expression node (no arena, no thread-local).
///
/// Wraps `Arc<ExprNode>` for cheap cloning and structural sharing.
#[derive(Debug, Clone)]
pub struct Expr(Arc<ExprNode>);

impl Expr {
    fn alloc(node: ExprNode) -> Self {
        Expr(Arc::new(node))
    }

    pub fn alloc_node(node: ExprNode) -> Self {
        Self::alloc(node)
    }

    #[inline]
    pub fn node(&self) -> &ExprNode {
        &self.0
    }

    pub fn ident(name: impl Into<String>) -> Self {
        Expr::alloc(ExprNode::Ident(name.into()))
    }

    pub fn lit_bool(value: bool) -> Self {
        Expr::alloc(ExprNode::Literal(Literal::Bool(value)))
    }

    pub fn lit_i32(value: i32) -> Self {
        Expr::alloc(ExprNode::Literal(Literal::Int(value)))
    }

    pub fn lit_u32(value: u32) -> Self {
        Expr::alloc(ExprNode::Literal(Literal::Uint(value)))
    }

    pub fn lit_f32(value: f32) -> Self {
        Expr::alloc(ExprNode::Literal(Literal::Float(Self::format_f32_literal(
            value,
        ))))
    }

    pub fn field(self, field: impl Into<String>) -> Self {
        Expr::alloc(ExprNode::Field {
            base: self,
            field: field.into(),
        })
    }

    pub fn index(self, index: impl Into<Expr>) -> Self {
        Expr::alloc(ExprNode::Index {
            base: self,
            index: index.into(),
        })
    }

    pub fn call(callee: Expr, args: Vec<Expr>) -> Self {
        Expr::alloc(ExprNode::Call { callee, args })
    }

    pub fn call_named(name: &str, args: Vec<Expr>) -> Self {
        Expr::call(Expr::ident(name), args)
    }

    pub fn sqrt(self) -> Self {
        Expr::call_named("sqrt", vec![self])
    }

    pub fn addr_of(self) -> Self {
        Expr::unary(UnaryOp::AddressOf, self)
    }

    pub fn deref(self) -> Self {
        Expr::unary(UnaryOp::Deref, self)
    }

    pub fn lt(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::Less, rhs.into())
    }

    pub fn le(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::LessEq, rhs.into())
    }

    pub fn gt(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::Greater, rhs.into())
    }

    pub fn ge(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::GreaterEq, rhs.into())
    }

    pub fn eq(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::Equal, rhs.into())
    }

    pub fn ne(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::NotEqual, rhs.into())
    }

    pub fn shr(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::ShiftRight, rhs.into())
    }

    pub fn shl(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::ShiftLeft, rhs.into())
    }

    pub fn modulo(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::Modulo, rhs.into())
    }

    pub fn bitwise_and(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::BitwiseAnd, rhs.into())
    }

    pub fn bitwise_or(self, rhs: impl Into<Expr>) -> Self {
        Expr::binary(self, BinaryOp::BitwiseOr, rhs.into())
    }

    pub fn try_call_named(&self, name: &str) -> Option<Vec<Expr>> {
        match self.node() {
            ExprNode::Call { callee, args } => match callee.node() {
                ExprNode::Ident(callee_name) if callee_name == name => Some(args.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn try_f32_literal(&self) -> Option<f32> {
        match self.node() {
            ExprNode::Literal(Literal::Float(value)) => value.parse::<f32>().ok(),
            _ => None,
        }
    }

    fn unary(op: UnaryOp, expr: Expr) -> Self {
        Expr::alloc(ExprNode::Unary { op, expr })
    }

    pub fn binary(left: Expr, op: BinaryOp, right: Expr) -> Self {
        Expr::alloc(ExprNode::Binary { left, op, right })
    }

    fn format_f32_literal(value: f32) -> String {
        let mut out = if value.is_finite() {
            format!("{value}")
        } else if value.is_nan() {
            "nan()".to_string()
        } else if value.is_sign_positive() {
            "inf()".to_string()
        } else {
            "-inf()".to_string()
        };
        if !out.contains('.') && !out.contains('e') && !out.contains('E') && !out.ends_with(')') {
            out.push_str(".0");
        }
        out
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: same Arc pointer means same expression.
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for Expr {}

impl std::hash::Hash for Expr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        render_expr(self, f, Precedence::Lowest)
    }
}

impl From<&str> for Expr {
    fn from(value: &str) -> Self {
        Expr::ident(value)
    }
}

impl From<String> for Expr {
    fn from(value: String) -> Self {
        Expr::ident(value)
    }
}

impl From<bool> for Expr {
    fn from(value: bool) -> Self {
        Expr::lit_bool(value)
    }
}

impl From<i32> for Expr {
    fn from(value: i32) -> Self {
        Expr::lit_i32(value)
    }
}

impl From<u32> for Expr {
    fn from(value: u32) -> Self {
        Expr::lit_u32(value)
    }
}

impl From<usize> for Expr {
    fn from(value: usize) -> Self {
        let value = u32::try_from(value).expect("usize literal does not fit in u32");
        Expr::lit_u32(value)
    }
}

impl From<f32> for Expr {
    fn from(value: f32) -> Self {
        Expr::lit_f32(value)
    }
}

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Expr::lit_f32(value as f32)
    }
}

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Add, rhs)
    }
}

impl std::ops::Add<u32> for Expr {
    type Output = Expr;
    fn add(self, rhs: u32) -> Self::Output {
        self + Expr::from(rhs)
    }
}

impl std::ops::Add<f32> for Expr {
    type Output = Expr;
    fn add(self, rhs: f32) -> Self::Output {
        self + Expr::from(rhs)
    }
}

impl std::ops::Add<&str> for Expr {
    type Output = Expr;
    fn add(self, rhs: &str) -> Self::Output {
        self + Expr::from(rhs)
    }
}

impl std::ops::Add<String> for Expr {
    type Output = Expr;
    fn add(self, rhs: String) -> Self::Output {
        self + Expr::from(rhs)
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Sub, rhs)
    }
}

impl std::ops::Sub<u32> for Expr {
    type Output = Expr;
    fn sub(self, rhs: u32) -> Self::Output {
        self - Expr::from(rhs)
    }
}

impl std::ops::Sub<f32> for Expr {
    type Output = Expr;
    fn sub(self, rhs: f32) -> Self::Output {
        self - Expr::from(rhs)
    }
}

impl std::ops::Sub<&str> for Expr {
    type Output = Expr;
    fn sub(self, rhs: &str) -> Self::Output {
        self - Expr::from(rhs)
    }
}

impl std::ops::Sub<String> for Expr {
    type Output = Expr;
    fn sub(self, rhs: String) -> Self::Output {
        self - Expr::from(rhs)
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Mul, rhs)
    }
}

impl std::ops::Mul<u32> for Expr {
    type Output = Expr;
    fn mul(self, rhs: u32) -> Self::Output {
        self * Expr::from(rhs)
    }
}

impl std::ops::Mul<f32> for Expr {
    type Output = Expr;
    fn mul(self, rhs: f32) -> Self::Output {
        self * Expr::from(rhs)
    }
}

impl std::ops::Mul<&str> for Expr {
    type Output = Expr;
    fn mul(self, rhs: &str) -> Self::Output {
        self * Expr::from(rhs)
    }
}

impl std::ops::Mul<String> for Expr {
    type Output = Expr;
    fn mul(self, rhs: String) -> Self::Output {
        self * Expr::from(rhs)
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Div, rhs)
    }
}

impl std::ops::Div<u32> for Expr {
    type Output = Expr;
    fn div(self, rhs: u32) -> Self::Output {
        self / Expr::from(rhs)
    }
}

impl std::ops::Div<f32> for Expr {
    type Output = Expr;
    fn div(self, rhs: f32) -> Self::Output {
        self / Expr::from(rhs)
    }
}

impl std::ops::Div<&str> for Expr {
    type Output = Expr;
    fn div(self, rhs: &str) -> Self::Output {
        self / Expr::from(rhs)
    }
}

impl std::ops::Div<String> for Expr {
    type Output = Expr;
    fn div(self, rhs: String) -> Self::Output {
        self / Expr::from(rhs)
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Self::Output {
        Expr::unary(UnaryOp::Negate, self)
    }
}

impl std::ops::Not for Expr {
    type Output = Expr;
    fn not(self) -> Self::Output {
        Expr::unary(UnaryOp::Not, self)
    }
}

impl std::ops::BitAnd for Expr {
    type Output = Expr;
    fn bitand(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::And, rhs)
    }
}

impl std::ops::BitAnd<bool> for Expr {
    type Output = Expr;
    fn bitand(self, rhs: bool) -> Self::Output {
        self & Expr::from(rhs)
    }
}

impl std::ops::BitOr for Expr {
    type Output = Expr;
    fn bitor(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Or, rhs)
    }
}

impl std::ops::BitOr<bool> for Expr {
    type Output = Expr;
    fn bitor(self, rhs: bool) -> Self::Output {
        self | Expr::from(rhs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprNode {
    Literal(Literal),
    Ident(String),
    Field {
        base: Expr,
        field: String,
    },
    Index {
        base: Expr,
        index: Expr,
    },
    Unary {
        op: UnaryOp,
        expr: Expr,
    },
    Binary {
        left: Expr,
        op: BinaryOp,
        right: Expr,
    },
    Call {
        callee: Expr,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Bool(bool),
    Int(i32),
    Uint(u32),
    Float(String),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Bool(value) => write!(f, "{}", if *value { "true" } else { "false" }),
            Literal::Int(value) => write!(f, "{}", value),
            Literal::Uint(value) => write!(f, "{}u", value),
            Literal::Float(value) => write!(f, "{}", value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Negate,
    Not,
    AddressOf,
    Deref,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Negate => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::AddressOf => write!(f, "&"),
            UnaryOp::Deref => write!(f, "*"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Modulo,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Equal,
    NotEqual,
    And,
    Or,
    ShiftRight,
    ShiftLeft,
    BitwiseAnd,
    BitwiseOr,
}

impl BinaryOp {
    pub fn precedence(self) -> Precedence {
        match self {
            BinaryOp::Or => Precedence::Or,
            BinaryOp::And => Precedence::And,
            BinaryOp::BitwiseOr => Precedence::BitwiseOr,
            BinaryOp::BitwiseAnd => Precedence::BitwiseAnd,
            BinaryOp::Equal | BinaryOp::NotEqual => Precedence::Equality,
            BinaryOp::Less | BinaryOp::LessEq | BinaryOp::Greater | BinaryOp::GreaterEq => {
                Precedence::Comparison
            }
            BinaryOp::ShiftLeft | BinaryOp::ShiftRight => Precedence::Shift,
            BinaryOp::Add | BinaryOp::Sub => Precedence::Sum,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Modulo => Precedence::Product,
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Modulo => write!(f, "%"),
            BinaryOp::Less => write!(f, "<"),
            BinaryOp::LessEq => write!(f, "<="),
            BinaryOp::Greater => write!(f, ">"),
            BinaryOp::GreaterEq => write!(f, ">="),
            BinaryOp::Equal => write!(f, "=="),
            BinaryOp::NotEqual => write!(f, "!="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
            BinaryOp::ShiftRight => write!(f, ">>"),
            BinaryOp::ShiftLeft => write!(f, "<<"),
            BinaryOp::BitwiseAnd => write!(f, "&"),
            BinaryOp::BitwiseOr => write!(f, "|"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precedence {
    Lowest,
    Or,
    And,
    BitwiseOr,
    BitwiseAnd,
    Equality,
    Comparison,
    Shift,
    Sum,
    Product,
    Prefix,
    Postfix,
}

pub fn next_precedence(prec: Precedence) -> Precedence {
    match prec {
        Precedence::Lowest => Precedence::Or,
        Precedence::Or => Precedence::And,
        Precedence::And => Precedence::BitwiseOr,
        Precedence::BitwiseOr => Precedence::BitwiseAnd,
        Precedence::BitwiseAnd => Precedence::Equality,
        Precedence::Equality => Precedence::Comparison,
        Precedence::Comparison => Precedence::Shift,
        Precedence::Shift => Precedence::Sum,
        Precedence::Sum => Precedence::Product,
        Precedence::Product => Precedence::Prefix,
        Precedence::Prefix | Precedence::Postfix => Precedence::Postfix,
    }
}

fn render_expr(expr: &Expr, f: &mut fmt::Formatter<'_>, parent_prec: Precedence) -> fmt::Result {
    match expr.node() {
        ExprNode::Literal(lit) => write!(f, "{}", lit),
        ExprNode::Ident(name) => write!(f, "{}", name),
        ExprNode::Field { base, field } => {
            if let Some(axis) = axis_from_field(field) {
                if let Some((dim, args)) = try_vec_ctor(base) {
                    if axis < dim {
                        return render_expr(&args[axis], f, parent_prec);
                    }
                }
            }

            let needs_paren = expr_precedence(base) < Precedence::Postfix;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(base, f, Precedence::Postfix)?;
            if needs_paren {
                write!(f, ")")?;
            }
            write!(f, ".{}", field)
        }
        ExprNode::Index { base, index } => {
            let needs_paren = expr_precedence(base) < Precedence::Postfix;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(base, f, Precedence::Postfix)?;
            if needs_paren {
                write!(f, ")")?;
            }
            write!(f, "[")?;
            render_expr(index, f, Precedence::Lowest)?;
            write!(f, "]")
        }
        ExprNode::Unary { op, expr: inner } => {
            let prec = Precedence::Prefix;
            let needs_paren = prec < parent_prec;
            if needs_paren {
                write!(f, "(")?;
            }
            write!(f, "{}", op)?;
            render_expr(inner, f, prec)?;
            if needs_paren {
                write!(f, ")")?;
            }
            Ok(())
        }
        ExprNode::Binary { left, op, right } => {
            if let Some(simplified) = simplify_binary_expr(left, *op, right, f, parent_prec) {
                return simplified;
            }

            let prec = op.precedence();
            let needs_paren = prec < parent_prec;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(left, f, prec)?;
            write!(f, " {} ", op)?;
            // Preserve RHS grouping for every floating-point arithmetic op:
            // `a + (b - c)` re-printed as `a + b - c` would left-associate to
            // `(a + b) - c`, which is a DIFFERENT f32 value. Deliberate
            // right-grouping is load-bearing (e.g. the gauge-storage pressure
            // closure groups `rho + (gauge_rho_ref - rho_ref)` so a
            // gauge-stored liquid density never round-trips through its large
            // absolute value). The CPU Rust emitter is fully parenthesized, so
            // this also keeps CPU/GPU evaluation trees identical.
            let right_prec = match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                    next_precedence(prec)
                }
                _ => prec,
            };
            render_expr(right, f, right_prec)?;
            if needs_paren {
                write!(f, ")")?;
            }
            Ok(())
        }
        ExprNode::Call { callee, args } => {
            if args.len() == 2 && is_ident(callee, "dot") {
                let lhs = &args[0];
                let rhs = &args[1];

                if let Some((axis, sign)) = unit_vector_axis(rhs) {
                    return render_component(lhs, axis, sign, f, parent_prec);
                }
                if let Some((axis, sign)) = unit_vector_axis(lhs) {
                    return render_component(rhs, axis, sign, f, parent_prec);
                }
            }

            let needs_paren = expr_precedence(callee) < Precedence::Postfix;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(callee, f, Precedence::Postfix)?;
            if needs_paren {
                write!(f, ")")?;
            }
            write!(f, "(")?;
            for (idx, arg) in args.iter().enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                render_expr(arg, f, Precedence::Lowest)?;
            }
            write!(f, ")")
        }
    }
}

fn simplify_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    f: &mut fmt::Formatter<'_>,
    parent_prec: Precedence,
) -> Option<fmt::Result> {
    match op {
        BinaryOp::Add => {
            if right.try_f32_literal() == Some(0.0) {
                return Some(render_expr(left, f, parent_prec));
            }
            if left.try_f32_literal() == Some(0.0) {
                return Some(render_expr(right, f, parent_prec));
            }
        }
        BinaryOp::Sub => {
            if right.try_f32_literal() == Some(0.0) {
                return Some(render_expr(left, f, parent_prec));
            }
        }
        BinaryOp::Mul => {
            if let Some(v) = left.try_f32_literal() {
                if v == 1.0 {
                    return Some(render_expr(right, f, parent_prec));
                }
                if v == -1.0 {
                    return Some(render_negated_expr(right, f, parent_prec));
                }
            }
            if let Some(v) = right.try_f32_literal() {
                if v == 1.0 {
                    return Some(render_expr(left, f, parent_prec));
                }
                if v == -1.0 {
                    return Some(render_negated_expr(left, f, parent_prec));
                }
            }
        }
        BinaryOp::Div => {
            if let Some(v) = right.try_f32_literal() {
                if v == 1.0 {
                    return Some(render_expr(left, f, parent_prec));
                }
                if v == -1.0 {
                    return Some(render_negated_expr(left, f, parent_prec));
                }
            }
        }
        _ => {}
    }
    None
}

fn render_negated_expr(
    expr: &Expr,
    f: &mut fmt::Formatter<'_>,
    parent_prec: Precedence,
) -> fmt::Result {
    let prec = Precedence::Prefix;
    let needs_paren = prec < parent_prec;
    if needs_paren {
        write!(f, "(")?;
    }
    write!(f, "-")?;
    render_expr(expr, f, prec)?;
    if needs_paren {
        write!(f, ")")?;
    }
    Ok(())
}

fn axis_from_field(field: &str) -> Option<usize> {
    match field {
        "x" => Some(0),
        "y" => Some(1),
        "z" => Some(2),
        "w" => Some(3),
        _ => None,
    }
}

fn is_ident(expr: &Expr, name: &str) -> bool {
    matches!(expr.node(), ExprNode::Ident(id) if id == name)
}

fn try_vec_ctor(expr: &Expr) -> Option<(usize, Vec<Expr>)> {
    match expr.node() {
        ExprNode::Call { callee, args } => match callee.node() {
            ExprNode::Ident(name) => match name.as_str() {
                "vec2<f32>" if args.len() == 2 => Some((2, args.clone())),
                "vec3<f32>" if args.len() == 3 => Some((3, args.clone())),
                "vec4<f32>" if args.len() == 4 => Some((4, args.clone())),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn unit_vector_axis(expr: &Expr) -> Option<(usize, f32)> {
    let (dim, args) = try_vec_ctor(expr)?;
    let mut axis: Option<usize> = None;
    let mut sign: f32 = 1.0;

    for (idx, arg) in args.iter().enumerate() {
        let v = arg.try_f32_literal()?;
        if v == 0.0 {
            continue;
        }
        if v == 1.0 || v == -1.0 {
            if axis.is_some() {
                return None;
            }
            axis = Some(idx);
            sign = v.signum();
            continue;
        }
        return None;
    }

    axis.and_then(|axis| (axis < dim).then_some((axis, sign)))
}

fn render_component(
    vec: &Expr,
    axis: usize,
    sign: f32,
    f: &mut fmt::Formatter<'_>,
    parent_prec: Precedence,
) -> fmt::Result {
    if sign == -1.0 {
        let prec = Precedence::Prefix;
        let needs_paren = prec < parent_prec;
        if needs_paren {
            write!(f, "(")?;
        }
        write!(f, "-")?;
        render_component(vec, axis, 1.0, f, prec)?;
        if needs_paren {
            write!(f, ")")?;
        }
        return Ok(());
    }

    if let Some((dim, args)) = try_vec_ctor(vec) {
        if axis < dim {
            return render_expr(&args[axis], f, parent_prec);
        }
    }

    let field = match axis {
        0 => "x",
        1 => "y",
        2 => "z",
        3 => "w",
        _ => return render_expr(vec, f, parent_prec),
    };

    let needs_paren = expr_precedence(vec) < Precedence::Postfix;
    if needs_paren {
        write!(f, "(")?;
    }
    render_expr(vec, f, Precedence::Postfix)?;
    if needs_paren {
        write!(f, ")")?;
    }
    write!(f, ".{field}")
}

fn expr_precedence(expr: &Expr) -> Precedence {
    match expr.node() {
        ExprNode::Literal(_) | ExprNode::Ident(_) => Precedence::Postfix,
        ExprNode::Field { .. } | ExprNode::Index { .. } | ExprNode::Call { .. } => {
            Precedence::Postfix
        }
        ExprNode::Unary { .. } => Precedence::Prefix,
        ExprNode::Binary { op, .. } => op.precedence(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_builders_render_expected_wgsl() {
        let expr = Expr::ident("a") + Expr::ident("b") * Expr::ident("c");
        assert_eq!(expr.to_string(), "a + b * c");

        let expr = Expr::call_named("arrayLength", vec![Expr::ident("cell_vols").addr_of()]);
        assert_eq!(expr.to_string(), "arrayLength(&cell_vols)");

        let expr = Expr::call_named("vec2<f32>", vec![1.0.into(), 2.0.into()]);
        assert_eq!(expr.to_string(), "vec2<f32>(1.0, 2.0)");

        let expr: Expr = 3u32.into();
        assert_eq!(expr.to_string(), "3u");

        let expr: Expr = 1.0.into();
        assert_eq!(expr.to_string(), "1.0");

        let expr = Expr::call_named("max", vec![1.0.into(), Expr::ident("x")]);
        assert_eq!(expr.to_string(), "max(1.0, x)");

        let expr = Expr::ident("state")
            .index(Expr::ident("idx"))
            .field("u")
            .field("x");
        assert_eq!(expr.to_string(), "state[idx].u.x");
    }

    #[test]
    fn expr_simplifies_vec_ctor_field_access() {
        let v = Expr::call_named("vec2<f32>", vec![Expr::ident("a"), Expr::ident("b")]);
        assert_eq!(v.clone().field("x").to_string(), "a");
        assert_eq!(v.field("y").to_string(), "b");
    }

    #[test]
    fn expr_simplifies_dot_with_unit_vectors() {
        let v = Expr::call_named("vec2<f32>", vec![Expr::ident("a"), Expr::ident("b")]);
        let ex = Expr::call_named("vec2<f32>", vec![1.0.into(), 0.0.into()]);
        let ey = Expr::call_named("vec2<f32>", vec![0.0.into(), 1.0.into()]);

        assert_eq!(
            Expr::call_named("dot", vec![v.clone(), ex.clone()]).to_string(),
            "a"
        );

        let v2 = Expr::call_named("vec2<f32>", vec![Expr::ident("a"), Expr::ident("b")]);
        assert_eq!(
            Expr::call_named("dot", vec![ey, v2]).to_string(),
            "b"
        );

        let vx = Expr::call_named(
            "vec2<f32>",
            vec![Expr::ident("a") + Expr::ident("b"), Expr::ident("c")],
        );
        let expr = Expr::ident("k") * Expr::call_named("dot", vec![vx, ex]);
        assert_eq!(expr.to_string(), "k * (a + b)");
    }

    #[test]
    fn expr_converts_strings_and_numbers() {
        let expr: Expr = "x".into();
        assert_eq!(expr.to_string(), "x");

        let expr = Expr::ident("x") + 1u32;
        assert_eq!(expr.to_string(), "x + 1u");

        let expr = Expr::ident("x") + 1.0;
        assert_eq!(expr.to_string(), "x + 1.0");

        let expr = Expr::ident("arr").index(0);
        assert_eq!(expr.to_string(), "arr[0]");

        let expr = Expr::ident("neighbor").ne(-1);
        assert_eq!(expr.to_string(), "neighbor != -1");

        let expr = Expr::ident("cond") & true;
        assert_eq!(expr.to_string(), "cond && true");
    }

    #[test]
    fn expr_renders_non_associative_rhs_with_parentheses() {
        let a = Expr::ident("a");
        let b = Expr::ident("b");
        let c = Expr::ident("c");

        assert_eq!((a.clone() - (b.clone() - c.clone())).to_string(), "a - (b - c)");
        assert_eq!((a.clone() / (b.clone() / c.clone())).to_string(), "a / (b / c)");
        assert_eq!((a / (b * c)).to_string(), "a / (b * c)");
    }

    #[test]
    fn expr_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Expr>();
        assert_send_sync::<ExprNode>();
    }

    #[test]
    fn expr_structural_equality() {
        let a1 = Expr::ident("a") + Expr::ident("b");
        let a2 = Expr::ident("a") + Expr::ident("b");
        assert_eq!(a1, a2);

        let a3 = Expr::ident("a") + Expr::ident("c");
        assert_ne!(a1, a3);
    }
}
