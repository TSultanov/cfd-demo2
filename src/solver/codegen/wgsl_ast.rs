use std::cell::RefCell;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    items: Vec<Item>,
}

impl Module {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: Item) {
        self.items.push(item);
    }

    pub fn extend<I: IntoIterator<Item = Item>>(&mut self, items: I) {
        self.items.extend(items);
    }

    pub fn to_wgsl(&self) -> String {
        let mut out = String::new();
        let mut ctx = RenderContext::new(&mut out);
        for (idx, item) in self.items.iter().enumerate() {
            if idx > 0 {
                ctx.blank_line();
            }
            item.render(&mut ctx);
        }
        out
    }
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Comment(String),
    Struct(StructDef),
    GlobalVar(GlobalVar),
    Function(Function),
}

impl Item {
    fn render(&self, ctx: &mut RenderContext<'_>) {
        match self {
            Item::Comment(text) => {
                ctx.line(&format!("// {}", text));
            }
            Item::Struct(def) => def.render(ctx),
            Item::GlobalVar(var) => var.render(ctx),
            Item::Function(function) => function.render(ctx),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<StructField>,
}

impl StructDef {
    pub fn new(name: impl Into<String>, fields: Vec<StructField>) -> Self {
        Self {
            name: name.into(),
            fields,
        }
    }

    fn render(&self, ctx: &mut RenderContext<'_>) {
        ctx.line(&format!("struct {} {{", self.name));
        ctx.indent();
        for field in &self.fields {
            field.render(ctx);
        }
        ctx.dedent();
        ctx.line("}");
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub comment: Option<String>,
}

impl StructField {
    pub fn new(name: impl Into<String>, ty: Type) -> Self {
        Self {
            name: name.into(),
            ty,
            comment: None,
        }
    }

    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    fn render(&self, ctx: &mut RenderContext<'_>) {
        if let Some(comment) = &self.comment {
            ctx.line(&format!("{}: {}, // {}", self.name, self.ty, comment));
        } else {
            ctx.line(&format!("{}: {},", self.name, self.ty));
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlobalVar {
    pub name: String,
    pub ty: Type,
    pub storage: StorageClass,
    pub access: Option<AccessMode>,
    pub attributes: Vec<Attribute>,
}

impl GlobalVar {
    pub fn new(
        name: impl Into<String>,
        ty: Type,
        storage: StorageClass,
        access: Option<AccessMode>,
        attributes: Vec<Attribute>,
    ) -> Self {
        Self {
            name: name.into(),
            ty,
            storage,
            access,
            attributes,
        }
    }

    fn render(&self, ctx: &mut RenderContext<'_>) {
        for attr in &self.attributes {
            ctx.inline_attr(attr);
        }
        if !self.attributes.is_empty() {
            ctx.space();
        }
        match (self.storage, self.access) {
            (StorageClass::Storage, Some(access)) => {
                ctx.line(&format!(
                    "var<storage, {}> {}: {};",
                    access,
                    self.name,
                    self.ty
                ));
            }
            (StorageClass::Uniform, None) => {
                ctx.line(&format!("var<uniform> {}: {};", self.name, self.ty));
            }
            (StorageClass::Storage, None) => {
                ctx.line(&format!("var<storage> {}: {};", self.name, self.ty));
            }
            (StorageClass::Uniform, Some(access)) => {
                ctx.line(&format!(
                    "var<uniform, {}> {}: {};",
                    access,
                    self.name,
                    self.ty
                ));
            }
            (StorageClass::Workgroup, None) => {
                ctx.line(&format!("var<workgroup> {}: {};", self.name, self.ty));
            }
            (StorageClass::Workgroup, Some(access)) => {
                ctx.line(&format!(
                    "var<workgroup, {}> {}: {};",
                    access,
                    self.name,
                    self.ty
                ));
            }
        }
        ctx.flush_line();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Storage,
    Uniform,
    Workgroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    ReadWrite,
}

impl fmt::Display for AccessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccessMode::Read => write!(f, "read"),
            AccessMode::ReadWrite => write!(f, "read_write"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub attributes: Vec<Attribute>,
    pub body: Block,
}

impl Function {
    pub fn new(
        name: impl Into<String>,
        params: Vec<Param>,
        return_type: Option<Type>,
        attributes: Vec<Attribute>,
        body: Block,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            return_type,
            attributes,
            body,
        }
    }

    fn render(&self, ctx: &mut RenderContext<'_>) {
        for attr in &self.attributes {
            ctx.line(&format!("{}", attr));
        }
        let params = self
            .params
            .iter()
            .map(|param| param.to_wgsl())
            .collect::<Vec<_>>()
            .join(", ");
        if let Some(ret) = &self.return_type {
            ctx.line(&format!("fn {}({}) -> {} {{", self.name, params, ret));
        } else {
            ctx.line(&format!("fn {}({}) {{", self.name, params));
        }
        ctx.indent();
        self.body.render(ctx);
        ctx.dedent();
        ctx.line("}");
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub attributes: Vec<Attribute>,
}

impl Param {
    pub fn new(name: impl Into<String>, ty: Type, attributes: Vec<Attribute>) -> Self {
        Self {
            name: name.into(),
            ty,
            attributes,
        }
    }

    fn to_wgsl(&self) -> String {
        let mut out = String::new();
        for attr in &self.attributes {
            out.push_str(&format!("{} ", attr));
        }
        out.push_str(&format!("{}: {}", self.name, self.ty));
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

impl Block {
    pub fn new(stmts: Vec<Stmt>) -> Self {
        Self { stmts }
    }

    fn render(&self, ctx: &mut RenderContext<'_>) {
        for stmt in &self.stmts {
            stmt.render(ctx);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    Return(Option<Expr>),
    Call(Expr),
    Increment(Expr),
}

impl Stmt {
    fn render(&self, ctx: &mut RenderContext<'_>) {
        match self {
            Stmt::Comment(text) => ctx.line(&format!("// {}", text)),
            Stmt::Let { name, ty, expr } => {
                if let Some(ty) = ty {
                    ctx.line(&format!("let {}: {} = {};", name, ty, expr));
                } else {
                    ctx.line(&format!("let {} = {};", name, expr));
                }
            }
            Stmt::Var { name, ty, expr } => {
                match (ty, expr) {
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
                }
            }
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
                then_block.render(ctx);
                ctx.dedent();
                if let Some(else_block) = else_block {
                    ctx.line("} else {");
                    ctx.indent();
                    else_block.render(ctx);
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
                body.render(ctx);
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
            Stmt::Call(expr) => ctx.line(&format!("{};", expr)),
            Stmt::Increment(expr) => ctx.line(&format!("{}++;", expr)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl fmt::Display for AssignOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssignOp::Add => write!(f, "+"),
            AssignOp::Sub => write!(f, "-"),
            AssignOp::Mul => write!(f, "*"),
            AssignOp::Div => write!(f, "/"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    Let { name: String, ty: Option<Type>, expr: Expr },
    Var { name: String, ty: Option<Type>, expr: Expr },
    Assign { target: Expr, value: Expr },
}

impl ForInit {
    fn to_wgsl(&self) -> String {
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

#[derive(Debug, Clone, PartialEq)]
pub enum ForStep {
    Increment(Expr),
    Assign { target: Expr, value: Expr },
    AssignOp { target: Expr, op: AssignOp, value: Expr },
}

impl ForStep {
    fn to_wgsl(&self) -> String {
        match self {
            ForStep::Increment(expr) => format!("{}++", expr),
            ForStep::Assign { target, value } => format!("{} = {}", target, value),
            ForStep::AssignOp { target, op, value } => format!("{} {}= {}", target, op, value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    F32,
    U32,
    I32,
    Bool,
    Vec2(Box<Type>),
    Vec3(Box<Type>),
    Vec4(Box<Type>),
    Array(Box<Type>),
    Custom(String),
}

impl Type {
    pub fn vec2_f32() -> Self {
        Type::Vec2(Box::new(Type::F32))
    }

    pub fn vec3_u32() -> Self {
        Type::Vec3(Box::new(Type::U32))
    }

    pub fn array(inner: Type) -> Self {
        Type::Array(Box::new(inner))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::F32 => write!(f, "f32"),
            Type::U32 => write!(f, "u32"),
            Type::I32 => write!(f, "i32"),
            Type::Bool => write!(f, "bool"),
            Type::Vec2(inner) => write!(f, "vec2<{}>", inner),
            Type::Vec3(inner) => write!(f, "vec3<{}>", inner),
            Type::Vec4(inner) => write!(f, "vec4<{}>", inner),
            Type::Array(inner) => write!(f, "array<{}>", inner),
            Type::Custom(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attribute {
    Group(u32),
    Binding(u32),
    Builtin(String),
    Compute,
    WorkgroupSize(u32),
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attribute::Group(index) => write!(f, "@group({})", index),
            Attribute::Binding(index) => write!(f, "@binding({})", index),
            Attribute::Builtin(name) => write!(f, "@builtin({})", name),
            Attribute::Compute => write!(f, "@compute"),
            Attribute::WorkgroupSize(size) => write!(f, "@workgroup_size({})", size),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Expr(u32);

#[derive(Debug, Clone)]
enum ExprNode {
    Literal(Literal),
    Ident(String),
    Field { base: Expr, field: String },
    Index { base: Expr, index: Expr },
    Unary { op: UnaryOp, expr: Expr },
    Binary { left: Expr, op: BinaryOp, right: Expr },
    Call { callee: Expr, args: Vec<Expr> },
}

thread_local! {
    static EXPR_ARENA: RefCell<Vec<ExprNode>> = RefCell::new(Vec::new());
}

impl Expr {
    fn alloc(node: ExprNode) -> Self {
        EXPR_ARENA.with(|arena| {
            let mut arena = arena.borrow_mut();
            let id = u32::try_from(arena.len()).expect("expression arena overflow");
            arena.push(node);
            Expr(id)
        })
    }

    fn with_node<R>(self, f: impl FnOnce(&ExprNode) -> R) -> R {
        EXPR_ARENA.with(|arena| {
            let arena = arena.borrow();
            let node = arena
                .get(self.0 as usize)
                .unwrap_or_else(|| panic!("invalid Expr id {}", self.0));
            f(node)
        })
    }

    fn unary(op: UnaryOp, expr: Expr) -> Self {
        Expr::alloc(ExprNode::Unary { op, expr })
    }

    fn binary(left: Expr, op: BinaryOp, right: Expr) -> Self {
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
        Expr::alloc(ExprNode::Literal(Literal::Float(Self::format_f32_literal(value))))
    }

    pub fn field(self, field: impl Into<String>) -> Self {
        Expr::alloc(ExprNode::Field {
            base: self,
            field: field.into(),
        })
    }

    pub fn index(self, index: Expr) -> Self {
        Expr::alloc(ExprNode::Index { base: self, index })
    }

    pub fn call(callee: Expr, args: Vec<Expr>) -> Self {
        Expr::alloc(ExprNode::Call { callee, args })
    }

    pub fn call_named(name: &str, args: Vec<Expr>) -> Self {
        Expr::call(Expr::ident(name), args)
    }

    pub fn addr_of(self) -> Self {
        Expr::unary(UnaryOp::AddressOf, self)
    }

    pub fn lt(self, rhs: Expr) -> Self {
        Expr::binary(self, BinaryOp::Less, rhs)
    }

    pub fn le(self, rhs: Expr) -> Self {
        Expr::binary(self, BinaryOp::LessEq, rhs)
    }

    pub fn gt(self, rhs: Expr) -> Self {
        Expr::binary(self, BinaryOp::Greater, rhs)
    }

    pub fn ge(self, rhs: Expr) -> Self {
        Expr::binary(self, BinaryOp::GreaterEq, rhs)
    }

    pub fn eq(self, rhs: Expr) -> Self {
        Expr::binary(self, BinaryOp::Equal, rhs)
    }

    pub fn ne(self, rhs: Expr) -> Self {
        Expr::binary(self, BinaryOp::NotEqual, rhs)
    }

    pub fn try_call_named(self, name: &str) -> Option<Vec<Expr>> {
        self.with_node(|node| match node {
            ExprNode::Call { callee, args } => callee.with_node(|callee_node| match callee_node {
                ExprNode::Ident(callee_name) if callee_name == name => Some(args.clone()),
                _ => None,
            }),
            _ => None,
        })
    }

    pub fn try_f32_literal(self) -> Option<f32> {
        self.with_node(|node| match node {
            ExprNode::Literal(Literal::Float(value)) => value.parse::<f32>().ok(),
            _ => None,
        })
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        render_expr(*self, f, Precedence::Lowest)
    }
}

#[derive(Debug, Clone)]
enum Literal {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryOp {
    Negate,
    Not,
    AddressOf,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Negate => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::AddressOf => write!(f, "&"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Equal,
    NotEqual,
    And,
    Or,
}

impl BinaryOp {
    fn precedence(self) -> Precedence {
        match self {
            BinaryOp::Or => Precedence::Or,
            BinaryOp::And => Precedence::And,
            BinaryOp::Equal | BinaryOp::NotEqual => Precedence::Equality,
            BinaryOp::Less | BinaryOp::LessEq | BinaryOp::Greater | BinaryOp::GreaterEq => {
                Precedence::Comparison
            }
            BinaryOp::Add | BinaryOp::Sub => Precedence::Sum,
            BinaryOp::Mul | BinaryOp::Div => Precedence::Product,
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
            BinaryOp::Less => write!(f, "<"),
            BinaryOp::LessEq => write!(f, "<="),
            BinaryOp::Greater => write!(f, ">"),
            BinaryOp::GreaterEq => write!(f, ">="),
            BinaryOp::Equal => write!(f, "=="),
            BinaryOp::NotEqual => write!(f, "!="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Precedence {
    Lowest,
    Or,
    And,
    Equality,
    Comparison,
    Sum,
    Product,
    Prefix,
    Postfix,
}

fn render_expr(expr: Expr, f: &mut fmt::Formatter<'_>, parent_prec: Precedence) -> fmt::Result {
    expr.with_node(|node| match node {
        ExprNode::Literal(lit) => write!(f, "{}", lit),
        ExprNode::Ident(name) => write!(f, "{}", name),
        ExprNode::Field { base, field } => {
            let needs_paren = expr_precedence(*base) < Precedence::Postfix;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(*base, f, Precedence::Postfix)?;
            if needs_paren {
                write!(f, ")")?;
            }
            write!(f, ".{}", field)
        }
        ExprNode::Index { base, index } => {
            let needs_paren = expr_precedence(*base) < Precedence::Postfix;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(*base, f, Precedence::Postfix)?;
            if needs_paren {
                write!(f, ")")?;
            }
            write!(f, "[")?;
            render_expr(*index, f, Precedence::Lowest)?;
            write!(f, "]")
        }
        ExprNode::Unary { op, expr } => {
            let prec = Precedence::Prefix;
            let needs_paren = prec < parent_prec;
            if needs_paren {
                write!(f, "(")?;
            }
            write!(f, "{}", op)?;
            render_expr(*expr, f, prec)?;
            if needs_paren {
                write!(f, ")")?;
            }
            Ok(())
        }
        ExprNode::Binary { left, op, right } => {
            let prec = op.precedence();
            let needs_paren = prec < parent_prec;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(*left, f, prec)?;
            write!(f, " {} ", op)?;
            render_expr(*right, f, prec)?;
            if needs_paren {
                write!(f, ")")?;
            }
            Ok(())
        }
        ExprNode::Call { callee, args } => {
            let needs_paren = expr_precedence(*callee) < Precedence::Postfix;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(*callee, f, Precedence::Postfix)?;
            if needs_paren {
                write!(f, ")")?;
            }
            write!(f, "(")?;
            for (idx, arg) in args.iter().enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                render_expr(*arg, f, Precedence::Lowest)?;
            }
            write!(f, ")")
        }
    })
}

fn expr_precedence(expr: Expr) -> Precedence {
    expr.with_node(|node| match node {
        ExprNode::Literal(_) | ExprNode::Ident(_) => Precedence::Postfix,
        ExprNode::Field { .. } | ExprNode::Index { .. } | ExprNode::Call { .. } => {
            Precedence::Postfix
        }
        ExprNode::Unary { .. } => Precedence::Prefix,
        ExprNode::Binary { op, .. } => op.precedence(),
    })
}

impl std::ops::Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Add, rhs)
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Sub, rhs)
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Mul, rhs)
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Div, rhs)
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

impl std::ops::BitOr for Expr {
    type Output = Expr;

    fn bitor(self, rhs: Expr) -> Self::Output {
        Expr::binary(self, BinaryOp::Or, rhs)
    }
}

struct RenderContext<'a> {
    output: &'a mut String,
    indent: usize,
    pending_line: String,
}

impl<'a> RenderContext<'a> {
    fn new(output: &'a mut String) -> Self {
        Self {
            output,
            indent: 0,
            pending_line: String::new(),
        }
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    fn line(&mut self, text: &str) {
        if !self.pending_line.is_empty() {
            self.flush_line();
        }
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(text);
        self.output.push('\n');
    }

    fn blank_line(&mut self) {
        if !self.pending_line.is_empty() {
            self.flush_line();
        }
        self.output.push('\n');
    }

    fn inline_attr(&mut self, attr: &Attribute) {
        if !self.pending_line.is_empty() {
            self.pending_line.push(' ');
        }
        self.pending_line.push_str(&attr.to_string());
    }

    fn space(&mut self) {
        if !self.pending_line.is_empty() {
            self.pending_line.push(' ');
        }
    }

    fn flush_line(&mut self) {
        if self.pending_line.is_empty() {
            return;
        }
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(&self.pending_line);
        self.output.push('\n');
        self.pending_line.clear();
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

        let expr = Expr::call_named("vec2<f32>", vec![Expr::lit_f32(1.0), Expr::lit_f32(2.0)]);
        assert_eq!(expr.to_string(), "vec2<f32>(1.0, 2.0)");

        let expr = Expr::lit_u32(3);
        assert_eq!(expr.to_string(), "3u");

        let expr = Expr::lit_f32(1.0);
        assert_eq!(expr.to_string(), "1.0");

        let expr = Expr::call_named("max", vec![Expr::lit_f32(1.0), Expr::ident("x")]);
        assert_eq!(expr.to_string(), "max(1.0, x)");

        let expr = Expr::ident("state")
            .index(Expr::ident("idx"))
            .field("u")
            .field("x");
        assert_eq!(expr.to_string(), "state[idx].u.x");
    }

    #[test]
    fn module_renders_structs_and_functions() {
        let module = Module {
            items: vec![
                Item::Comment("test".to_string()),
                Item::Struct(StructDef::new(
                    "Foo",
                    vec![StructField::new("value", Type::F32)],
                )),
                Item::Function(Function::new(
                    "main",
                    Vec::new(),
                    None,
                    Vec::new(),
                    Block::new(vec![Stmt::Return(None)]),
                )),
            ],
        };
        let output = module.to_wgsl();
        assert!(output.contains("// test"));
        assert!(output.contains("struct Foo"));
        assert!(output.contains("fn main()"));
        assert!(output.contains("return;"));
    }
}
