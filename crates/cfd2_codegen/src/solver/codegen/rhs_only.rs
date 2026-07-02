//! RHS-only assembly variants for matrix freezing across outer iterations.
//!
//! The coupled assembly kernels zero their block-CSR matrix row, accumulate
//! matrix coefficients (`diag_<i>` register accumulators + direct off-diagonal
//! `matrix_values` stores) and produce the RHS in a single pass. When the
//! matrix is FROZEN for an outer iteration (re-linearization skipped: the
//! relaxation lives in the update kernel and the default d_p is closed-form,
//! so the frozen matrix stays consistent), only the RHS needs refreshing —
//! this transform derives that kernel mechanically from the full one by
//! removing every statement that writes the matrix:
//!
//! - assignments (plain or compound) whose target roots at `matrix_values`
//!   (the row zeroing loop, the off-diagonal stores and the diagonal
//!   write-back), and
//! - assignments to the `diag_<N>` register accumulators (their values only
//!   ever flow into the removed write-back).
//!
//! Verified precondition: RHS expressions never READ `diag_<N>` accumulators
//! (bounded-correction sums use their own `bounded_sum_*` locals). The
//! `let diag_rank`/`let diag_bdf2` bindings do NOT match the accumulator
//! pattern and are preserved. Dead `var diag_<N>` declarations (and the
//! emptied zeroing loop) are left in place — backends tolerate unused
//! locals, and keeping the transform purely subtractive keeps it auditable.

use crate::solver::ir::KernelProgram;
use cfd2_ir::ast::{Block, Expr, ExprNode, Stmt};

/// Derive the RHS-only variant of an assembly `KernelProgram` under a new id.
pub fn rhs_only_kernel_program(full: &KernelProgram, id: &str) -> KernelProgram {
    let mut program = full.clone();
    program.id = id.to_string();
    program.body = strip_stmts(program.body);
    program
}

/// `diag_<N>` register accumulator names (NOT `diag_rank` / `diag_bdf2`).
fn is_diag_accumulator(name: &str) -> bool {
    name.strip_prefix("diag_")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()))
}

/// Root identifier of an assignment target (`matrix_values[k]` -> `matrix_values`).
fn target_root(expr: &Expr) -> Option<&str> {
    match expr.node() {
        ExprNode::Ident(name) => Some(name),
        ExprNode::Index { base, .. } | ExprNode::Field { base, .. } => target_root(base),
        _ => None,
    }
}

fn is_matrix_write_target(target: &Expr) -> bool {
    match target_root(target) {
        Some("matrix_values") => true,
        Some(name) => is_diag_accumulator(name),
        None => false,
    }
}

fn strip_block(block: Block) -> Block {
    Block::new(strip_stmts(block.stmts))
}

fn strip_stmts(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts
        .into_iter()
        .filter_map(|stmt| match stmt {
            Stmt::Assign { ref target, .. } | Stmt::AssignOp { ref target, .. }
                if is_matrix_write_target(target) =>
            {
                None
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
            } => Some(Stmt::If {
                cond,
                then_block: strip_block(then_block),
                else_block: else_block.map(strip_block),
            }),
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => Some(Stmt::For {
                init,
                cond,
                step,
                body: strip_block(body),
            }),
            Stmt::Loop { body } => Some(Stmt::Loop {
                body: strip_block(body),
            }),
            Stmt::While { cond, body } => Some(Stmt::While {
                cond,
                body: strip_block(body),
            }),
            other => Some(other),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd2_ir::ast::AssignOp;

    #[test]
    fn strips_matrix_and_diag_writes_but_keeps_rhs() {
        let stmts = vec![
            Stmt::Assign {
                target: Expr::ident("matrix_values").index(Expr::ident("k")),
                value: 0.0.into(),
            },
            Stmt::AssignOp {
                target: Expr::ident("diag_0"),
                op: AssignOp::Add,
                value: Expr::ident("flux_pos"),
            },
            Stmt::Let {
                name: "diag_bdf2".to_string(),
                ty: None,
                expr: 1.0.into(),
            },
            Stmt::AssignOp {
                target: Expr::ident("rhs_0"),
                op: AssignOp::Sub,
                value: Expr::ident("dc"),
            },
            Stmt::If {
                cond: true.into(),
                then_block: Block::new(vec![
                    Stmt::AssignOp {
                        target: Expr::ident("matrix_values").index(Expr::ident("j")),
                        op: AssignOp::Add,
                        value: Expr::ident("flux_neg"),
                    },
                    Stmt::Assign {
                        target: Expr::ident("rhs_1"),
                        value: 2.0.into(),
                    },
                ]),
                else_block: None,
            },
        ];
        let out = strip_stmts(stmts);
        // matrix_values / diag_0 writes gone; let diag_bdf2 + rhs writes kept.
        assert_eq!(out.len(), 3);
        assert!(matches!(&out[0], Stmt::Let { name, .. } if name == "diag_bdf2"));
        assert!(
            matches!(&out[1], Stmt::AssignOp { target, .. } if target_root(target) == Some("rhs_0"))
        );
        match &out[2] {
            Stmt::If { then_block, .. } => {
                assert_eq!(then_block.stmts.len(), 1);
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn diag_accumulator_pattern_is_exact() {
        assert!(is_diag_accumulator("diag_0"));
        assert!(is_diag_accumulator("diag_11"));
        assert!(!is_diag_accumulator("diag_rank"));
        assert!(!is_diag_accumulator("diag_bdf2"));
        assert!(!is_diag_accumulator("diag_"));
        assert!(!is_diag_accumulator("rhs_0"));
    }
}
