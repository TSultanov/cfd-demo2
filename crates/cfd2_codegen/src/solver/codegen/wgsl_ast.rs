//! WGSL AST types for codegen.
//!
//! Core AST types (`Expr`, `ExprNode`, `Stmt`, `Block`, `Type`, etc.) are re-exported
//! from `cfd2_ir::ast`. This module adds codegen-specific helpers: `Module`, `Item`,
//! `Function`, `CseBuilder`, etc.

use std::fmt;

use indexmap::{IndexMap, IndexSet};

// ── Re-export core AST from cfd2_ir ────────────────────────────────────

pub use cfd2_ir::ast::{
    AddressSpace, AssignOp, BinaryOp, Block, Expr, ExprNode, ForInit, ForStep, Literal, Precedence,
    Stmt, Type, UnaryOp,
};

// Re-export rendering helpers
pub use cfd2_ir::ast::stmt::{collect_local_symbols, render_block_lines, render_stmt_lines};

// ── Module-level types (codegen-specific) ──────────────────────────────

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
        let mut ctx = cfd2_ir::ast::types::RenderContext::new(&mut out);
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
    /// Module-level `const NAME: TYPE = EXPR;` declaration.
    Const {
        name: String,
        ty: Type,
        expr: Expr,
    },
    Struct(StructDef),
    GlobalVar(GlobalVar),
    Function(Function),
}

impl Item {
    fn render(&self, ctx: &mut cfd2_ir::ast::types::RenderContext<'_>) {
        match self {
            Item::Comment(text) => {
                ctx.line(&format!("// {}", text));
            }
            Item::Const { name, ty, expr } => {
                ctx.line(&format!("const {}: {} = {};", name, ty, expr));
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

    fn render(&self, ctx: &mut cfd2_ir::ast::types::RenderContext<'_>) {
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

    fn render(&self, ctx: &mut cfd2_ir::ast::types::RenderContext<'_>) {
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

    fn render(&self, ctx: &mut cfd2_ir::ast::types::RenderContext<'_>) {
        for attr in &self.attributes {
            ctx.inline_attr(&attr.to_string());
        }
        if !self.attributes.is_empty() {
            ctx.space();
        }
        match (self.storage, self.access) {
            (StorageClass::Storage, Some(access)) => {
                ctx.line(&format!(
                    "var<storage, {}> {}: {};",
                    access, self.name, self.ty
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
                    access, self.name, self.ty
                ));
            }
            (StorageClass::Workgroup, None) => {
                ctx.line(&format!("var<workgroup> {}: {};", self.name, self.ty));
            }
            (StorageClass::Workgroup, Some(access)) => {
                ctx.line(&format!(
                    "var<workgroup, {}> {}: {};",
                    access, self.name, self.ty
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

    fn render(&self, ctx: &mut cfd2_ir::ast::types::RenderContext<'_>) {
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
        for stmt in &self.body.stmts {
            stmt.render(ctx);
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attribute {
    Group(u32),
    Binding(u32),
    Builtin(String),
    Compute,
    WorkgroupSize(u32),
    /// 3-component workgroup size, e.g. `@workgroup_size(256, 1, 1)`.
    WorkgroupSize3(u32, u32, u32),
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attribute::Group(index) => write!(f, "@group({})", index),
            Attribute::Binding(index) => write!(f, "@binding({})", index),
            Attribute::Builtin(name) => write!(f, "@builtin({})", name),
            Attribute::Compute => write!(f, "@compute"),
            Attribute::WorkgroupSize(size) => write!(f, "@workgroup_size({})", size),
            Attribute::WorkgroupSize3(x, y, z) => {
                write!(f, "@workgroup_size({}, {}, {})", x, y, z)
            }
        }
    }
}



// ── CSE (Common Subexpression Elimination) ─────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct CseConfig {
    pub min_occurrences: usize,
    pub min_nodes: usize,
    pub max_bindings: usize,
}

impl Default for CseConfig {
    fn default() -> Self {
        Self {
            min_occurrences: 2,
            // Avoid flooding with tiny temporaries.
            min_nodes: 6,
            // Safety valve: keep temporary explosions under control.
            max_bindings: 64,
        }
    }
}

/// Common-subexpression elimination (CSE) for a set of WGSL expressions.
///
/// Returns a prelude of `let` statements (ordered by dependency) and rewritten roots that
/// reference those temporaries.
///
/// Note: this is purely a codegen size/compile-time optimization; expressions are assumed to be
/// side-effect free.
pub struct CseBuilder {
    prefix: String,
    next_id: u32,
    config: CseConfig,
}

impl CseBuilder {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self::with_config(prefix, CseConfig::default())
    }

    pub fn with_config(prefix: impl Into<String>, config: CseConfig) -> Self {
        Self {
            prefix: prefix.into(),
            next_id: 0,
            config,
        }
    }

    pub fn eliminate(&mut self, roots: &[Expr]) -> (Vec<Stmt>, Vec<Expr>) {
        if roots.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // ── Canonicalization ────────────────────────────────────────
        // With Arc-based Expr, structural equality works directly.
        // We intern by structural key to deduplicate.

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        enum NodeKey {
            Literal(Literal),
            Ident(String),
            Field {
                base: Box<NodeKey>,
                field: String,
            },
            Index {
                base: Box<NodeKey>,
                index: Box<NodeKey>,
            },
            Unary {
                op: UnaryOp,
                expr: Box<NodeKey>,
            },
            Binary {
                left: Box<NodeKey>,
                op: BinaryOp,
                right: Box<NodeKey>,
            },
            Call {
                callee: Box<NodeKey>,
                args: Vec<NodeKey>,
            },
        }

        fn expr_to_key(expr: &Expr) -> NodeKey {
            match expr.node() {
                ExprNode::Literal(lit) => NodeKey::Literal(lit.clone()),
                ExprNode::Ident(name) => NodeKey::Ident(name.clone()),
                ExprNode::Field { base, field } => NodeKey::Field {
                    base: Box::new(expr_to_key(base)),
                    field: field.clone(),
                },
                ExprNode::Index { base, index } => NodeKey::Index {
                    base: Box::new(expr_to_key(base)),
                    index: Box::new(expr_to_key(index)),
                },
                ExprNode::Unary { op, expr: inner } => NodeKey::Unary {
                    op: *op,
                    expr: Box::new(expr_to_key(inner)),
                },
                ExprNode::Binary { left, op, right } => NodeKey::Binary {
                    left: Box::new(expr_to_key(left)),
                    op: *op,
                    right: Box::new(expr_to_key(right)),
                },
                ExprNode::Call { callee, args } => NodeKey::Call {
                    callee: Box::new(expr_to_key(callee)),
                    args: args.iter().map(expr_to_key).collect(),
                },
            }
        }

        #[derive(Default)]
        struct Canonicalizer {
            intern: IndexMap<NodeKey, Expr>,
        }

        impl Canonicalizer {
            fn canon(&mut self, expr: &Expr) -> Expr {
                let key = expr_to_key(expr);
                if let Some(rep) = self.intern.get(&key) {
                    return rep.clone();
                }
                self.intern.insert(key, expr.clone());
                expr.clone()
            }
        }

        fn expr_is_trivial(expr: &Expr) -> bool {
            matches!(expr.node(), ExprNode::Literal(_) | ExprNode::Ident(_))
        }

        fn expr_size(
            expr: &Expr,
            canon: &mut Canonicalizer,
            memo: &mut IndexMap<NodeKey, usize>,
        ) -> usize {
            let key = expr_to_key(expr);
            if let Some(size) = memo.get(&key).copied() {
                return size;
            }
            let size = match expr.node() {
                ExprNode::Literal(_) | ExprNode::Ident(_) => 1,
                ExprNode::Field { base, .. } => 1 + expr_size(base, canon, memo),
                ExprNode::Index { base, index } => {
                    1 + expr_size(base, canon, memo) + expr_size(index, canon, memo)
                }
                ExprNode::Unary { expr: inner, .. } => 1 + expr_size(inner, canon, memo),
                ExprNode::Binary { left, right, .. } => {
                    1 + expr_size(left, canon, memo) + expr_size(right, canon, memo)
                }
                ExprNode::Call { callee, args } => {
                    1 + expr_size(callee, canon, memo)
                        + args
                            .iter()
                            .map(|arg| expr_size(arg, canon, memo))
                            .sum::<usize>()
                }
            };
            memo.insert(key, size);
            size
        }

        fn count_subexprs(
            expr: &Expr,
            canon: &mut Canonicalizer,
            counts: &mut IndexMap<NodeKey, (Expr, usize)>,
        ) {
            let key = expr_to_key(expr);
            let entry = counts.entry(key).or_insert_with(|| (expr.clone(), 0));
            entry.1 += 1;
            match expr.node() {
                ExprNode::Literal(_) | ExprNode::Ident(_) => {}
                ExprNode::Field { base, .. } => count_subexprs(base, canon, counts),
                ExprNode::Index { base, index } => {
                    count_subexprs(base, canon, counts);
                    count_subexprs(index, canon, counts);
                }
                ExprNode::Unary { expr: inner, .. } => count_subexprs(inner, canon, counts),
                ExprNode::Binary { left, right, .. } => {
                    count_subexprs(left, canon, counts);
                    count_subexprs(right, canon, counts);
                }
                ExprNode::Call { callee, args } => {
                    count_subexprs(callee, canon, counts);
                    for arg in args {
                        count_subexprs(arg, canon, counts);
                    }
                }
            }
        }

        let mut canon = Canonicalizer::default();
        let mut counts = IndexMap::<NodeKey, (Expr, usize)>::new();
        for root in roots {
            count_subexprs(root, &mut canon, &mut counts);
        }

        let mut sizes = IndexMap::<NodeKey, usize>::new();
        let mut candidates = Vec::new();
        for (key, (rep, count)) in &counts {
            if *count < self.config.min_occurrences {
                continue;
            }
            if expr_is_trivial(rep) {
                continue;
            }
            let size = expr_size(rep, &mut canon, &mut sizes);
            if size < self.config.min_nodes {
                continue;
            }

            // Rough benefit: saves `(count - 1)` re-emissions of this subtree.
            let benefit = (count.saturating_sub(1)) * size;
            candidates.push((key.clone(), rep.clone(), benefit));
        }

        // Sort by benefit descending, then by key position for determinism.
        candidates.sort_by(|a, b| b.2.cmp(&a.2));
        candidates.truncate(self.config.max_bindings);

        let extract: IndexSet<NodeKey> = candidates.into_iter().map(|(key, _, _)| key).collect();

        struct Rewriter<'a> {
            extract: &'a IndexSet<NodeKey>,
            names: IndexMap<NodeKey, String>,
            stmts: Vec<Stmt>,
            prefix: &'a str,
            next_id: &'a mut u32,
        }

        impl<'a> Rewriter<'a> {
            fn rewrite(&mut self, expr: &Expr) -> Expr {
                let key = expr_to_key(expr);
                if let Some(name) = self.names.get(&key) {
                    return Expr::ident(name.clone());
                }

                if self.extract.contains(&key) {
                    let name = format!("{}{}", self.prefix, *self.next_id);
                    *self.next_id = self.next_id.saturating_add(1);
                    self.names.insert(key, name.clone());
                    let rhs = self.rebuild(expr);
                    self.stmts.push(Stmt::Let {
                        name: name.clone(),
                        ty: None,
                        expr: rhs,
                    });
                    return Expr::ident(name);
                }

                self.rebuild(expr)
            }

            fn rebuild(&mut self, expr: &Expr) -> Expr {
                match expr.node().clone() {
                    ExprNode::Literal(_) | ExprNode::Ident(_) => expr.clone(),
                    ExprNode::Field { base, field } => {
                        let new_base = self.rewrite(&base);
                        if new_base == base {
                            expr.clone()
                        } else {
                            new_base.field(field)
                        }
                    }
                    ExprNode::Index { base, index } => {
                        let new_base = self.rewrite(&base);
                        let new_index = self.rewrite(&index);
                        if new_base == base && new_index == index {
                            expr.clone()
                        } else {
                            new_base.index(new_index)
                        }
                    }
                    ExprNode::Unary { op, expr: inner } => {
                        let new_inner = self.rewrite(&inner);
                        if new_inner == inner {
                            return expr.clone();
                        }
                        match op {
                            UnaryOp::Negate => -new_inner,
                            UnaryOp::Not => !new_inner,
                            UnaryOp::AddressOf => new_inner.addr_of(),
                            UnaryOp::Deref => new_inner.deref(),
                        }
                    }
                    ExprNode::Binary { left, op, right } => {
                        let new_left = self.rewrite(&left);
                        let new_right = self.rewrite(&right);
                        if new_left == left && new_right == right {
                            return expr.clone();
                        }
                        Expr::binary(new_left, op, new_right)
                    }
                    ExprNode::Call { callee, args } => {
                        let new_callee = self.rewrite(&callee);
                        let mut changed = new_callee != callee;
                        let new_args: Vec<Expr> = args
                            .iter()
                            .map(|arg| {
                                let new_arg = self.rewrite(arg);
                                if new_arg != *arg {
                                    changed = true;
                                }
                                new_arg
                            })
                            .collect();
                        if !changed {
                            expr.clone()
                        } else {
                            Expr::call(new_callee, new_args)
                        }
                    }
                }
            }
        }

        let mut rw = Rewriter {
            extract: &extract,
            names: IndexMap::new(),
            stmts: Vec::new(),
            prefix: &self.prefix,
            next_id: &mut self.next_id,
        };

        let new_roots = roots.iter().map(|r| rw.rewrite(r)).collect();
        (rw.stmts, new_roots)
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
        assert_eq!(Expr::call_named("dot", vec![ey, v2]).to_string(), "b");

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

        assert_eq!(
            (a.clone() - (b.clone() - c.clone())).to_string(),
            "a - (b - c)"
        );
        assert_eq!(
            (a.clone() / (b.clone() / c.clone())).to_string(),
            "a / (b / c)"
        );
        assert_eq!((a / (b * c)).to_string(), "a / (b * c)");
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
}
