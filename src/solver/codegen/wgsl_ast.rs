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

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Ident(String),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        render_expr(self, f, Precedence::Lowest)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Bool(bool),
    Int(String),
    Uint(String),
    Float(String),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Bool(value) => write!(f, "{}", if *value { "true" } else { "false" }),
            Literal::Int(value) => write!(f, "{}", value),
            Literal::Uint(value) => write!(f, "{}", value),
            Literal::Float(value) => write!(f, "{}", value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
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
pub enum BinaryOp {
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

fn render_expr(expr: &Expr, f: &mut fmt::Formatter<'_>, parent_prec: Precedence) -> fmt::Result {
    match expr {
        Expr::Literal(lit) => write!(f, "{}", lit),
        Expr::Ident(name) => write!(f, "{}", name),
        Expr::Field(base, field) => {
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
        Expr::Index(base, index) => {
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
        Expr::Unary { op, expr } => {
            let prec = Precedence::Prefix;
            let needs_paren = prec < parent_prec;
            if needs_paren {
                write!(f, "(")?;
            }
            write!(f, "{}", op)?;
            render_expr(expr, f, prec)?;
            if needs_paren {
                write!(f, ")")?;
            }
            Ok(())
        }
        Expr::Binary { left, op, right } => {
            let prec = op.precedence();
            let needs_paren = prec < parent_prec;
            if needs_paren {
                write!(f, "(")?;
            }
            render_expr(left, f, prec)?;
            write!(f, " {} ", op)?;
            render_expr(right, f, prec)?;
            if needs_paren {
                write!(f, ")")?;
            }
            Ok(())
        }
        Expr::Call { callee, args } => {
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

fn expr_precedence(expr: &Expr) -> Precedence {
    match expr {
        Expr::Literal(_) | Expr::Ident(_) => Precedence::Postfix,
        Expr::Field(_, _) | Expr::Index(_, _) | Expr::Call { .. } => Precedence::Postfix,
        Expr::Unary { .. } => Precedence::Prefix,
        Expr::Binary { op, .. } => op.precedence(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken(String),
    UnexpectedEof,
    InvalidExpression(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken(token) => write!(f, "unexpected token '{}'", token),
            ParseError::UnexpectedEof => write!(f, "unexpected end of input"),
            ParseError::InvalidExpression(expr) => write!(f, "invalid expression '{}'", expr),
        }
    }
}

impl std::error::Error for ParseError {}

impl Expr {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        while let Some(token) = lexer.next_token()? {
            tokens.push(token);
        }
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expression(Precedence::Lowest)?;
        if parser.peek().is_some() {
            return Err(ParseError::InvalidExpression(input.to_string()));
        }
        Ok(expr)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Int(String),
    Uint(String),
    Float(String),
    Bool(bool),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Dot,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    EqualEqual,
    NotEqual,
    AndAnd,
    OrOr,
    Bang,
    Amp,
}

struct Lexer<'a> {
    chars: std::str::Chars<'a>,
    lookahead: Option<char>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        let mut chars = input.chars();
        let lookahead = chars.next();
        Self {
            chars,
            lookahead,
        }
    }

    fn next_char(&mut self) -> Option<char> {
        let current = self.lookahead;
        self.lookahead = self.chars.next();
        current
    }

    fn peek_char(&self) -> Option<char> {
        self.lookahead
    }

    fn consume_while<F>(&mut self, mut predicate: F) -> String
    where
        F: FnMut(char) -> bool,
    {
        let mut out = String::new();
        while let Some(ch) = self.peek_char() {
            if !predicate(ch) {
                break;
            }
            out.push(ch);
            self.next_char();
        }
        out
    }

    fn skip_whitespace(&mut self) {
        self.consume_while(|ch| ch.is_whitespace());
    }

    fn next_token(&mut self) -> Result<Option<Token>, ParseError> {
        self.skip_whitespace();
        let ch = match self.peek_char() {
            Some(ch) => ch,
            None => return Ok(None),
        };
        match ch {
            '(' => {
                self.next_char();
                Ok(Some(Token::LParen))
            }
            ')' => {
                self.next_char();
                Ok(Some(Token::RParen))
            }
            '[' => {
                self.next_char();
                Ok(Some(Token::LBracket))
            }
            ']' => {
                self.next_char();
                Ok(Some(Token::RBracket))
            }
            '.' => {
                self.next_char();
                Ok(Some(Token::Dot))
            }
            ',' => {
                self.next_char();
                Ok(Some(Token::Comma))
            }
            '+' => {
                self.next_char();
                Ok(Some(Token::Plus))
            }
            '-' => {
                self.next_char();
                Ok(Some(Token::Minus))
            }
            '*' => {
                self.next_char();
                Ok(Some(Token::Star))
            }
            '/' => {
                self.next_char();
                Ok(Some(Token::Slash))
            }
            '&' => {
                self.next_char();
                if self.peek_char() == Some('&') {
                    self.next_char();
                    Ok(Some(Token::AndAnd))
                } else {
                    Ok(Some(Token::Amp))
                }
            }
            '|' => {
                self.next_char();
                if self.peek_char() == Some('|') {
                    self.next_char();
                    Ok(Some(Token::OrOr))
                } else {
                    Err(ParseError::UnexpectedToken("|".to_string()))
                }
            }
            '!' => {
                self.next_char();
                if self.peek_char() == Some('=') {
                    self.next_char();
                    Ok(Some(Token::NotEqual))
                } else {
                    Ok(Some(Token::Bang))
                }
            }
            '=' => {
                self.next_char();
                if self.peek_char() == Some('=') {
                    self.next_char();
                    Ok(Some(Token::EqualEqual))
                } else {
                    Err(ParseError::UnexpectedToken("=".to_string()))
                }
            }
            '<' => {
                self.next_char();
                if self.peek_char() == Some('=') {
                    self.next_char();
                    Ok(Some(Token::LessEq))
                } else {
                    Ok(Some(Token::Less))
                }
            }
            '>' => {
                self.next_char();
                if self.peek_char() == Some('=') {
                    self.next_char();
                    Ok(Some(Token::GreaterEq))
                } else {
                    Ok(Some(Token::Greater))
                }
            }
            ch if ch.is_ascii_digit() || ch == '.' => {
                let token = self.lex_number()?;
                Ok(Some(token))
            }
            ch if is_ident_start(ch) => {
                let ident = self.consume_while(is_ident_continue);
                let token = match ident.as_str() {
                    "true" => Token::Bool(true),
                    "false" => Token::Bool(false),
                    _ => Token::Ident(ident),
                };
                Ok(Some(token))
            }
            other => Err(ParseError::UnexpectedToken(other.to_string())),
        }
    }

    fn lex_number(&mut self) -> Result<Token, ParseError> {
        let mut value = String::new();
        if let Some(ch) = self.peek_char() {
            if ch == '.' {
                value.push('.');
                self.next_char();
            }
        }
        value.push_str(&self.consume_while(|c| c.is_ascii_digit()));
        if self.peek_char() == Some('.') {
            value.push('.');
            self.next_char();
            value.push_str(&self.consume_while(|c| c.is_ascii_digit()));
        }
        if matches!(self.peek_char(), Some('e') | Some('E')) {
            value.push(self.next_char().unwrap());
            if matches!(self.peek_char(), Some('+') | Some('-')) {
                value.push(self.next_char().unwrap());
            }
            value.push_str(&self.consume_while(|c| c.is_ascii_digit()));
        }
        if self.peek_char() == Some('u') {
            value.push('u');
            self.next_char();
            return Ok(Token::Uint(value));
        }
        if value.contains('.') || value.contains('e') || value.contains('E') {
            Ok(Token::Float(value))
        } else {
            Ok(Token::Int(value))
        }
    }
}

fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '<' || ch == '>'
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        token
    }

    fn parse_expression(&mut self, min_prec: Precedence) -> Result<Expr, ParseError> {
        let mut expr = self.parse_prefix()?;
        loop {
            expr = match self.peek() {
                Some(Token::Dot) => {
                    self.next();
                    let field = match self.next() {
                        Some(Token::Ident(name)) => name,
                        Some(token) => return Err(ParseError::UnexpectedToken(format!("{:?}", token))),
                        None => return Err(ParseError::UnexpectedEof),
                    };
                    Expr::Field(Box::new(expr), field)
                }
                Some(Token::LBracket) => {
                    self.next();
                    let index = self.parse_expression(Precedence::Lowest)?;
                    match self.next() {
                        Some(Token::RBracket) => {}
                        Some(token) => return Err(ParseError::UnexpectedToken(format!("{:?}", token))),
                        None => return Err(ParseError::UnexpectedEof),
                    }
                    Expr::Index(Box::new(expr), Box::new(index))
                }
                Some(Token::LParen) => {
                    let args = self.parse_call_args()?;
                    Expr::Call {
                        callee: Box::new(expr),
                        args,
                    }
                }
                Some(token) => {
                    if let Some((op, prec)) = binary_op(token) {
                        if prec < min_prec {
                            break;
                        }
                        self.next();
                        let right = self.parse_expression(prec.next())?;
                        Expr::Binary {
                            left: Box::new(expr),
                            op,
                            right: Box::new(right),
                        }
                    } else {
                        break;
                    }
                }
                None => break,
            };
        }
        Ok(expr)
    }

    fn parse_prefix(&mut self) -> Result<Expr, ParseError> {
        match self.next() {
            Some(Token::Ident(name)) => Ok(Expr::Ident(name)),
            Some(Token::Bool(value)) => Ok(Expr::Literal(Literal::Bool(value))),
            Some(Token::Int(value)) => Ok(Expr::Literal(Literal::Int(value))),
            Some(Token::Uint(value)) => Ok(Expr::Literal(Literal::Uint(value))),
            Some(Token::Float(value)) => Ok(Expr::Literal(Literal::Float(value))),
            Some(Token::Minus) => {
                let expr = self.parse_expression(Precedence::Prefix)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Negate,
                    expr: Box::new(expr),
                })
            }
            Some(Token::Bang) => {
                let expr = self.parse_expression(Precedence::Prefix)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            Some(Token::Amp) => {
                let expr = self.parse_expression(Precedence::Prefix)?;
                Ok(Expr::Unary {
                    op: UnaryOp::AddressOf,
                    expr: Box::new(expr),
                })
            }
            Some(Token::LParen) => {
                let expr = self.parse_expression(Precedence::Lowest)?;
                match self.next() {
                    Some(Token::RParen) => Ok(expr),
                    Some(token) => Err(ParseError::UnexpectedToken(format!("{:?}", token))),
                    None => Err(ParseError::UnexpectedEof),
                }
            }
            Some(token) => Err(ParseError::UnexpectedToken(format!("{:?}", token))),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        match self.next() {
            Some(Token::LParen) => {}
            Some(token) => return Err(ParseError::UnexpectedToken(format!("{:?}", token))),
            None => return Err(ParseError::UnexpectedEof),
        }
        let mut args = Vec::new();
        if matches!(self.peek(), Some(Token::RParen)) {
            self.next();
            return Ok(args);
        }
        loop {
            args.push(self.parse_expression(Precedence::Lowest)?);
            match self.next() {
                Some(Token::Comma) => continue,
                Some(Token::RParen) => break,
                Some(token) => return Err(ParseError::UnexpectedToken(format!("{:?}", token))),
                None => return Err(ParseError::UnexpectedEof),
            }
        }
        Ok(args)
    }
}

impl Precedence {
    fn next(self) -> Precedence {
        match self {
            Precedence::Lowest => Precedence::Or,
            Precedence::Or => Precedence::And,
            Precedence::And => Precedence::Equality,
            Precedence::Equality => Precedence::Comparison,
            Precedence::Comparison => Precedence::Sum,
            Precedence::Sum => Precedence::Product,
            Precedence::Product => Precedence::Prefix,
            Precedence::Prefix => Precedence::Postfix,
            Precedence::Postfix => Precedence::Postfix,
        }
    }
}

fn binary_op(token: &Token) -> Option<(BinaryOp, Precedence)> {
    match token {
        Token::OrOr => Some((BinaryOp::Or, Precedence::Or)),
        Token::AndAnd => Some((BinaryOp::And, Precedence::And)),
        Token::EqualEqual => Some((BinaryOp::Equal, Precedence::Equality)),
        Token::NotEqual => Some((BinaryOp::NotEqual, Precedence::Equality)),
        Token::Less => Some((BinaryOp::Less, Precedence::Comparison)),
        Token::LessEq => Some((BinaryOp::LessEq, Precedence::Comparison)),
        Token::Greater => Some((BinaryOp::Greater, Precedence::Comparison)),
        Token::GreaterEq => Some((BinaryOp::GreaterEq, Precedence::Comparison)),
        Token::Plus => Some((BinaryOp::Add, Precedence::Sum)),
        Token::Minus => Some((BinaryOp::Sub, Precedence::Sum)),
        Token::Star => Some((BinaryOp::Mul, Precedence::Product)),
        Token::Slash => Some((BinaryOp::Div, Precedence::Product)),
        _ => None,
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
    fn expr_parser_handles_precedence_and_calls() {
        let expr = Expr::parse("a + b * c").unwrap();
        assert_eq!(expr.to_string(), "a + b * c");

        let expr = Expr::parse("arrayLength(&cell_vols)").unwrap();
        assert_eq!(expr.to_string(), "arrayLength(&cell_vols)");

        let expr = Expr::parse("vec2<f32>(1.0, 2.0)").unwrap();
        assert_eq!(expr.to_string(), "vec2<f32>(1.0, 2.0)");
    }

    #[test]
    fn expr_parser_handles_field_and_index() {
        let expr = Expr::parse("state[idx].u.x").unwrap();
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
