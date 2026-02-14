use crate::solver::codegen::wgsl_ast::{AssignOp, Expr, Stmt, Type};
use crate::solver::codegen::wgsl_dsl as dsl;

/// Code-generation helper for the per-cell accumulator variables
/// (`diag_0`, `diag_1`, …, `rhs_0`, `rhs_1`, …) used in coupled
/// assembly kernels.
///
/// This struct does **not** hold mutable state — it is a factory that
/// produces WGSL AST nodes (`Stmt`, `Expr`) based on the coupled stride
/// and consistent naming conventions.
///
/// # Lifecycle in generated WGSL
///
/// 1. **Declare start-row variables**: `let start_row_0 = …; let start_row_1 = …;`
/// 2. **Declare accumulators**: `var diag_0: f32 = 0.0; var rhs_0: f32 = 0.0; …`
/// 3. **Accumulate** via `diag_{u} += …`, `rhs_{u} -= …`, etc.
/// 4. **Optionally assign** (e.g. BDF2 correction): `diag_{u} = …`
/// 5. **Declare phi variables** (convection): `var phi_{u}: f32 = …;`
/// 6. **Write out**: `matrix_diag[i,i] += diag_{i}`, `rhs_buf[idx*stride+i] = rhs_{i}`
pub struct CoupledAccumulators {
    pub coupled_stride: u32,
}

impl CoupledAccumulators {
    pub fn new(coupled_stride: u32) -> Self {
        Self { coupled_stride }
    }

    // ── Name helpers ──────────────────────────────────────────────────

    fn diag_name(index: u32) -> String {
        format!("diag_{index}")
    }

    fn rhs_name(index: u32) -> String {
        format!("rhs_{index}")
    }

    fn phi_name(index: u32) -> String {
        format!("phi_{index}")
    }

    fn start_row_name(row: u32) -> String {
        format!("start_row_{row}")
    }

    // ── Expr accessors ────────────────────────────────────────────────

    /// Returns `Expr::ident("diag_{index}")`.
    pub fn diag(&self, index: impl Into<AccIndex>) -> Expr {
        Expr::ident(Self::diag_name(index.into().resolve()))
    }

    /// Returns `Expr::ident("rhs_{index}")`.
    pub fn rhs(&self, index: impl Into<AccIndex>) -> Expr {
        Expr::ident(Self::rhs_name(index.into().resolve()))
    }

    /// Returns `Expr::ident("phi_{index}")`.
    pub fn phi(&self, index: impl Into<AccIndex>) -> Expr {
        Expr::ident(Self::phi_name(index.into().resolve()))
    }

    /// Returns `Expr::ident("start_row_{row}")`.
    pub fn start_row(&self, row: u32) -> Expr {
        Expr::ident(Self::start_row_name(row))
    }

    // ── Declaration statements ────────────────────────────────────────

    /// Emits `let start_row_0 = base_expr;` followed by
    /// `let start_row_i = start_row_0 + num_neighbors * coupled_stride * i;`
    /// for i in 1..coupled_stride.
    pub fn declare_start_rows(&self, base_expr: Expr, num_neighbors: Expr) -> Vec<Stmt> {
        let block_stride = self.coupled_stride * self.coupled_stride;
        let mut stmts = vec![dsl::let_expr(
            &Self::start_row_name(0),
            base_expr * block_stride,
        )];
        for row in 1..self.coupled_stride {
            stmts.push(dsl::let_expr(
                &Self::start_row_name(row),
                self.start_row(0) + num_neighbors * self.coupled_stride * row,
            ));
        }
        stmts
    }

    /// Emits `var diag_i: f32 = 0.0; var rhs_i: f32 = 0.0;` for
    /// i in 0..coupled_stride.
    pub fn declare(&self) -> Vec<Stmt> {
        let mut stmts = Vec::with_capacity(self.coupled_stride as usize * 2);
        for i in 0..self.coupled_stride {
            stmts.push(dsl::var_typed_expr(
                &Self::diag_name(i),
                Type::F32,
                Some(0.0.into()),
            ));
            stmts.push(dsl::var_typed_expr(
                &Self::rhs_name(i),
                Type::F32,
                Some(0.0.into()),
            ));
        }
        stmts
    }

    /// Emits `var phi_{index}: f32 = init_expr;`.
    pub fn declare_phi(&self, index: impl Into<AccIndex>, init_expr: Expr) -> Stmt {
        dsl::var_typed_expr(
            &Self::phi_name(index.into().resolve()),
            Type::F32,
            Some(init_expr),
        )
    }

    // ── Accumulation statements ───────────────────────────────────────

    /// `diag_{index} += value`
    pub fn add_diag(&self, index: impl Into<AccIndex>, value: impl Into<Expr>) -> Stmt {
        dsl::assign_op_expr(AssignOp::Add, self.diag(index), value)
    }

    /// `diag_{index} -= value`
    pub fn sub_diag(&self, index: impl Into<AccIndex>, value: impl Into<Expr>) -> Stmt {
        dsl::assign_op_expr(AssignOp::Sub, self.diag(index), value)
    }

    /// `diag_{index} = value`
    pub fn set_diag(&self, index: impl Into<AccIndex>, value: impl Into<Expr>) -> Stmt {
        dsl::assign_expr(self.diag(index), value)
    }

    /// `rhs_{index} += value`
    pub fn add_rhs(&self, index: impl Into<AccIndex>, value: impl Into<Expr>) -> Stmt {
        dsl::assign_op_expr(AssignOp::Add, self.rhs(index), value)
    }

    /// `rhs_{index} -= value`
    pub fn sub_rhs(&self, index: impl Into<AccIndex>, value: impl Into<Expr>) -> Stmt {
        dsl::assign_op_expr(AssignOp::Sub, self.rhs(index), value)
    }

    /// `rhs_{index} = value`
    pub fn set_rhs(&self, index: impl Into<AccIndex>, value: impl Into<Expr>) -> Stmt {
        dsl::assign_expr(self.rhs(index), value)
    }

    // ── Write-back ────────────────────────────────────────────────────

    /// Emits the standard write-back loop:
    /// ```wgsl
    /// diag_entry.entry(i,i).expr += diag_i;
    /// rhs[idx * coupled_stride + i] = rhs_i;
    /// ```
    ///
    /// `diag_entry_fn` receives the loop index `i` (0..coupled_stride) and
    /// must return the `Expr` for the diagonal matrix LHS
    /// (e.g. `block_matrix.row_entry(&diag_rank).entry(i, i).expr`).
    pub fn writeback(
        &self,
        diag_entry_fn: impl Fn(u32) -> Expr,
        rhs_array: &str,
        idx_expr: Expr,
    ) -> Vec<Stmt> {
        let mut stmts = Vec::with_capacity(self.coupled_stride as usize * 2);
        for i in 0..self.coupled_stride {
            stmts.push(dsl::assign_op_expr(
                AssignOp::Add,
                diag_entry_fn(i),
                self.diag(i),
            ));
            stmts.push(dsl::assign_expr(
                dsl::array_access_linear(rhs_array, idx_expr, self.coupled_stride, i),
                self.rhs(i),
            ));
        }
        stmts
    }
}

/// Thin wrapper allowing both `u32` literals and pre-computed `u32`
/// expressions to be passed as accumulator indices.
///
/// Most call sites use a bare `u32` (e.g. `acc.diag(u_idx)`); some compute
/// `base_offset + component` inline. Both work thanks to `From` impls.
pub struct AccIndex(u32);

impl AccIndex {
    fn resolve(self) -> u32 {
        self.0
    }
}

impl From<u32> for AccIndex {
    fn from(v: u32) -> Self {
        AccIndex(v)
    }
}

// Allow i32 for convenience (loop counters sometimes use i32)
impl From<i32> for AccIndex {
    fn from(v: i32) -> Self {
        AccIndex(v as u32)
    }
}

// Allow usize for iteration contexts
impl From<usize> for AccIndex {
    fn from(v: usize) -> Self {
        AccIndex(v as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::wgsl_ast::render_stmt_lines;

    fn render(stmts: &[Stmt]) -> Vec<String> {
        render_stmt_lines(stmts)
    }

    fn render_one(stmt: &Stmt) -> String {
        render_stmt_lines(&[stmt.clone()])
            .into_iter()
            .next()
            .unwrap()
    }

    #[test]
    fn diag_rhs_names() {
        let acc = CoupledAccumulators::new(3);
        assert_eq!(acc.diag(0u32).to_string(), "diag_0");
        assert_eq!(acc.diag(2u32).to_string(), "diag_2");
        assert_eq!(acc.rhs(1u32).to_string(), "rhs_1");
    }

    #[test]
    fn declare_creates_var_pairs() {
        let acc = CoupledAccumulators::new(2);
        let stmts = acc.declare();
        assert_eq!(stmts.len(), 4);
        let lines = render(&stmts);
        eprintln!("LINES: {lines:#?}");
        assert!(lines[0].contains("diag_0"));
        assert!(lines[1].contains("rhs_0"));
        assert!(lines[2].contains("diag_1"));
        assert!(lines[3].contains("rhs_1"));
    }

    #[test]
    fn start_row_declaration_stride_2() {
        let acc = CoupledAccumulators::new(2);
        let stmts =
            acc.declare_start_rows(Expr::ident("scalar_offset"), Expr::ident("num_neighbors"));
        assert_eq!(stmts.len(), 2);
        let lines = render(&stmts);
        assert!(lines[0].contains("let start_row_0"));
        assert!(lines[1].contains("let start_row_1"));
    }

    #[test]
    fn add_diag_emits_assign_op() {
        let acc = CoupledAccumulators::new(2);
        let stmt = acc.add_diag(0u32, Expr::ident("coeff"));
        let line = render_one(&stmt);
        assert!(line.contains("diag_0 += coeff"));
    }

    #[test]
    fn sub_rhs_emits_assign_op() {
        let acc = CoupledAccumulators::new(2);
        let stmt = acc.sub_rhs(1u32, Expr::ident("flux"));
        let line = render_one(&stmt);
        assert!(line.contains("rhs_1 -= flux"));
    }

    #[test]
    fn phi_declare_and_access() {
        let acc = CoupledAccumulators::new(3);
        let decl = acc.declare_phi(2u32, Expr::ident("flux_val"));
        let line = render_one(&decl);
        assert!(line.contains("var phi_2: f32 = flux_val"));
        assert_eq!(acc.phi(2u32).to_string(), "phi_2");
    }
}
