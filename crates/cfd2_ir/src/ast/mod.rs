//! Portable, self-contained WGSL AST types.
//!
//! These types use `Arc<ExprNode>` instead of arena indices, making them
//! `Clone + Send + Sync + PartialEq + Eq` and safe to store in `KernelProgram`.

mod expr;
pub mod stmt;
pub mod types;

pub use expr::{BinaryOp, Expr, ExprNode, Literal, Precedence, UnaryOp};
pub use stmt::{AssignOp, Block, ForInit, ForStep, Stmt};
pub use types::{AddressSpace, Type};
