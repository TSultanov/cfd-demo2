/// Shared IR and expression types used by both model and codegen.
///
/// This module breaks the circular dependency: model specs can now reference
/// WGSL AST types for primitive expressions and flux module definitions without
/// importing from codegen.

pub mod wgsl_ast;
pub mod wgsl_dsl;
pub mod expr;

pub use wgsl_ast::{Expr, Block, Stmt, Type, Module, Item, Function};
pub use expr::PrimitiveExpr;
