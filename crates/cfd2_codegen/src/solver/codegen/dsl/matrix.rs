use std::marker::PhantomData;

use crate::solver::codegen::wgsl_ast::Expr;

use super::expr::DynExpr;
use super::tensor::{BlockCol, BlockRow, CoupledAxis};
use super::types::{ScalarType, Shape};
use super::{DslType, UnitDim};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockShape {
    pub rows: u8,
    pub cols: u8,
}

impl BlockShape {
    pub const fn new(rows: u8, cols: u8) -> Self {
        Self { rows, cols }
    }

    pub const fn entry_count(self) -> u32 {
        self.rows as u32 * self.cols as u32
    }

    pub const fn is_square(self) -> bool {
        self.rows == self.cols
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CsrPattern {
    pub row_offsets: Expr,
    pub col_indices: Expr,
}

impl CsrPattern {
    pub fn new(row_offsets: Expr, col_indices: Expr) -> Self {
        Self {
            row_offsets,
            col_indices,
        }
    }

    pub fn from_idents(row_offsets: &str, col_indices: &str) -> Self {
        Self::new(Expr::ident(row_offsets), Expr::ident(col_indices))
    }

    pub fn row_start(&self, row: &Expr) -> Expr {
        self.row_offsets.index(*row)
    }

    pub fn row_end(&self, row: &Expr) -> Expr {
        let next = *row + 1u32;
        self.row_offsets.index(next)
    }

    pub fn col_at(&self, nnz_index: &Expr) -> Expr {
        self.col_indices.index(*nnz_index)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub pattern: CsrPattern,
    pub values: Expr,
    pub entry_ty: DslType,
    pub entry_unit: UnitDim,
}

impl CsrMatrix {
    pub fn new(pattern: CsrPattern, values: Expr, entry_ty: DslType, entry_unit: UnitDim) -> Self {
        Self {
            pattern,
            values,
            entry_ty,
            entry_unit,
        }
    }

    pub fn scalar_f32(pattern: CsrPattern, values: Expr, unit: UnitDim) -> Self {
        Self::new(pattern, values, DslType::f32(), unit)
    }

    pub fn value_at(&self, nnz_index: &Expr) -> DynExpr {
        DynExpr::new(
            self.values.index(*nnz_index),
            self.entry_ty,
            self.entry_unit,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockCsrMatrix {
    pub pattern: CsrPattern,
    pub values: Expr,
    pub block: BlockShape,
    pub scalar: ScalarType,
    pub entry_unit: UnitDim,
}

impl BlockCsrMatrix {
    pub fn new(
        pattern: CsrPattern,
        values: Expr,
        block: BlockShape,
        scalar: ScalarType,
        entry_unit: UnitDim,
    ) -> Self {
        Self {
            pattern,
            values,
            block,
            scalar,
            entry_unit,
        }
    }

    pub fn entry(&self, nnz_index: &Expr, row: u8, col: u8) -> DynExpr {
        assert!(row < self.block.rows, "block row out of bounds");
        assert!(col < self.block.cols, "block col out of bounds");
        let base = *nnz_index * self.block.entry_count();
        let offset = row as u32 * self.block.cols as u32 + col as u32;
        let index = base + offset;
        let ty = DslType::new(self.scalar, Shape::Scalar);
        DynExpr::new(self.values.index(index), ty, self.entry_unit)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockCsrSoaMatrix {
    pub values: Expr,
    pub start_rows: Vec<Expr>,
    pub block: BlockShape,
    pub scalar: ScalarType,
    pub entry_unit: UnitDim,
}

impl BlockCsrSoaMatrix {
    pub fn new(
        values: Expr,
        start_rows: Vec<Expr>,
        block: BlockShape,
        scalar: ScalarType,
        entry_unit: UnitDim,
    ) -> Self {
        assert!(
            start_rows.len() == block.rows as usize,
            "start_rows must match block row count"
        );
        Self {
            values,
            start_rows,
            block,
            scalar,
            entry_unit,
        }
    }

    pub fn from_start_row_prefix(
        values: &str,
        start_row_prefix: &str,
        block: BlockShape,
        scalar: ScalarType,
        entry_unit: UnitDim,
    ) -> Self {
        let start_rows = (0..block.rows)
            .map(|row| Expr::ident(format!("{start_row_prefix}_{row}")))
            .collect();
        Self::new(Expr::ident(values), start_rows, block, scalar, entry_unit)
    }

    pub fn row_entry(&self, rank: &Expr) -> BlockCsrSoaEntry {
        let cols = self.block.cols as u32;
        let bases = self
            .start_rows
            .iter()
            .cloned()
            .map(|start| start + *rank * cols)
            .collect();
        BlockCsrSoaEntry::new(self.values, bases, self.block, self.scalar, self.entry_unit)
    }

    pub fn entry(&self, rank: &Expr, row: u8, col: u8) -> DynExpr {
        self.row_entry(rank).entry(row, col)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockCsrSoaEntry {
    pub values: Expr,
    pub row_bases: Vec<Expr>,
    pub block: BlockShape,
    pub scalar: ScalarType,
    pub entry_unit: UnitDim,
}

impl BlockCsrSoaEntry {
    pub fn new(
        values: Expr,
        row_bases: Vec<Expr>,
        block: BlockShape,
        scalar: ScalarType,
        entry_unit: UnitDim,
    ) -> Self {
        assert!(
            row_bases.len() == block.rows as usize,
            "row_bases must match block row count"
        );
        Self {
            values,
            row_bases,
            block,
            scalar,
            entry_unit,
        }
    }

    pub fn entry(&self, row: u8, col: u8) -> DynExpr {
        let ty = DslType::new(self.scalar, Shape::Scalar);
        DynExpr::new(self.access_expr(row, col), ty, self.entry_unit)
    }

    pub fn index_expr(&self, row: u8, col: u8) -> Expr {
        assert!(row < self.block.rows, "block row out of bounds");
        assert!(col < self.block.cols, "block col out of bounds");
        let base = self.row_bases[row as usize];
        base + col as u32
    }

    pub fn access_expr(&self, row: u8, col: u8) -> Expr {
        self.values.index(self.index_expr(row, col))
    }
}

// ── Named (phantom-typed) block CSR SOA wrappers ──────────────────────

/// Phantom-typed wrapper around [`BlockCsrSoaMatrix`] that requires
/// [`BlockRow`]/[`BlockCol`] typed indices in `entry()` calls, preventing
/// accidental row/col transposition at compile time.
///
/// The axis type `Ax` describes the coupled unknowns (e.g.,
/// [`IncompressibleAxis2D`](super::tensor::IncompressibleAxis2D)).
/// Since block matrices in this codebase are always square, a single axis
/// type covers both rows and columns.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedBlockCsrSoaMatrix<Ax: CoupledAxis> {
    inner: BlockCsrSoaMatrix,
    _axis: PhantomData<Ax>,
}

impl<Ax: CoupledAxis> NamedBlockCsrSoaMatrix<Ax> {
    /// Wrap an existing `BlockCsrSoaMatrix`, asserting that its shape
    /// matches `Ax::STRIDE`.
    pub fn new(inner: BlockCsrSoaMatrix) -> Self {
        assert_eq!(
            inner.block.rows as u32,
            Ax::STRIDE,
            "NamedBlockCsrSoaMatrix: block rows ({}) do not match axis stride ({})",
            inner.block.rows,
            Ax::STRIDE,
        );
        assert_eq!(
            inner.block.cols as u32,
            Ax::STRIDE,
            "NamedBlockCsrSoaMatrix: block cols ({}) do not match axis stride ({})",
            inner.block.cols,
            Ax::STRIDE,
        );
        Self {
            inner,
            _axis: PhantomData,
        }
    }

    /// Convenience constructor matching [`BlockCsrSoaMatrix::from_start_row_prefix`].
    pub fn from_start_row_prefix(
        values: &str,
        start_row_prefix: &str,
        scalar: ScalarType,
        entry_unit: UnitDim,
    ) -> Self {
        let stride = Ax::STRIDE as u8;
        let block = BlockShape::new(stride, stride);
        Self::new(BlockCsrSoaMatrix::from_start_row_prefix(
            values,
            start_row_prefix,
            block,
            scalar,
            entry_unit,
        ))
    }

    /// Type-safe block entry access.
    pub fn entry(&self, rank: &Expr, row: BlockRow<Ax>, col: BlockCol<Ax>) -> DynExpr {
        self.inner.entry(rank, row.to_u8(), col.to_u8())
    }

    /// Type-safe row entry (pre-computes row bases for a given rank).
    pub fn row_entry(&self, rank: &Expr) -> NamedBlockCsrSoaEntry<Ax> {
        NamedBlockCsrSoaEntry::new(self.inner.row_entry(rank))
    }

    /// Access the underlying untyped matrix (escape hatch for code that
    /// must iterate generically, e.g. zeroing loops).
    pub fn inner(&self) -> &BlockCsrSoaMatrix {
        &self.inner
    }
}

/// Phantom-typed wrapper around [`BlockCsrSoaEntry`] that requires
/// [`BlockRow`]/[`BlockCol`] typed indices.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedBlockCsrSoaEntry<Ax: CoupledAxis> {
    inner: BlockCsrSoaEntry,
    _axis: PhantomData<Ax>,
}

impl<Ax: CoupledAxis> NamedBlockCsrSoaEntry<Ax> {
    pub fn new(inner: BlockCsrSoaEntry) -> Self {
        Self {
            inner,
            _axis: PhantomData,
        }
    }

    /// Type-safe block entry access.
    pub fn entry(&self, row: BlockRow<Ax>, col: BlockCol<Ax>) -> DynExpr {
        self.inner.entry(row.to_u8(), col.to_u8())
    }

    /// Type-safe index expression (the array offset, without the array access).
    pub fn index_expr(&self, row: BlockRow<Ax>, col: BlockCol<Ax>) -> Expr {
        self.inner.index_expr(row.to_u8(), col.to_u8())
    }

    /// Type-safe access expression (the full `values[offset]` expression).
    pub fn access_expr(&self, row: BlockRow<Ax>, col: BlockCol<Ax>) -> Expr {
        self.inner.access_expr(row.to_u8(), col.to_u8())
    }

    /// Access the underlying untyped entry (escape hatch).
    pub fn inner(&self) -> &BlockCsrSoaEntry {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_pattern_builds_row_slices() {
        let pattern = CsrPattern::from_idents("row_offsets", "col_indices");
        let row = Expr::ident("row");
        assert_eq!(pattern.row_start(&row).to_string(), "row_offsets[row]");
        assert_eq!(pattern.row_end(&row).to_string(), "row_offsets[row + 1u]");
    }

    #[test]
    fn block_csr_entry_indexes_block_values() {
        let pattern = CsrPattern::from_idents("row_offsets", "col_indices");
        let block = BlockShape::new(4, 4);
        let mat = BlockCsrMatrix::new(
            pattern,
            Expr::ident("matrix_values"),
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let nnz = Expr::ident("nnz");
        let expr = mat.entry(&nnz, 3, 2).expr.to_string();
        assert_eq!(expr, "matrix_values[nnz * 16u + 14u]");
    }

    #[test]
    fn block_csr_soa_entry_indexes_row_splits() {
        let block = BlockShape::new(4, 4);
        let entry = BlockCsrSoaEntry::new(
            Expr::ident("matrix_values"),
            (0..4).map(|r| Expr::ident(format!("base_{r}"))).collect(),
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let expr = entry.entry(2, 3).expr.to_string();
        assert_eq!(expr, "matrix_values[base_2 + 3u]");
    }

    // ── NamedBlockCsrSoaMatrix tests ──────────────────────────────────

    use crate::solver::codegen::dsl::tensor::{
        block_col, block_row, BlockCol, BlockRow, IncompressibleAxis2D, ScalarAxis,
    };

    #[test]
    fn named_matrix_wraps_matching_stride() {
        let mat = NamedBlockCsrSoaMatrix::<IncompressibleAxis2D>::from_start_row_prefix(
            "vals",
            "start",
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        assert_eq!(mat.inner().block.rows, 3);
        assert_eq!(mat.inner().block.cols, 3);
    }

    #[test]
    #[should_panic(expected = "block rows (4) do not match axis stride (3)")]
    fn named_matrix_panics_on_stride_mismatch() {
        let block = BlockShape::new(4, 4);
        let inner = BlockCsrSoaMatrix::from_start_row_prefix(
            "vals",
            "start",
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        NamedBlockCsrSoaMatrix::<IncompressibleAxis2D>::new(inner);
    }

    #[test]
    fn named_matrix_entry_matches_untyped() {
        let block = BlockShape::new(3, 3);
        let inner = BlockCsrSoaMatrix::from_start_row_prefix(
            "vals",
            "start",
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let named = NamedBlockCsrSoaMatrix::<IncompressibleAxis2D>::new(inner.clone());
        let rank = Expr::ident("rank");

        // Typed entry should produce the same expression as untyped
        let typed = named
            .entry(
                &rank,
                BlockRow::new(IncompressibleAxis2D::Uy),
                BlockCol::new(IncompressibleAxis2D::P),
            )
            .expr
            .to_string();
        let untyped = inner.entry(&rank, 1, 2).expr.to_string();
        assert_eq!(typed, untyped);
    }

    #[test]
    fn named_matrix_entry_via_free_functions() {
        let named = NamedBlockCsrSoaMatrix::<IncompressibleAxis2D>::from_start_row_prefix(
            "vals",
            "start",
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let rank = Expr::ident("rank");

        let expr = named
            .entry(&rank, block_row(0), block_col(2))
            .expr
            .to_string();
        // Should be same as untyped entry(rank, 0, 2)
        let expected = named.inner().entry(&rank, 0, 2).expr.to_string();
        assert_eq!(expr, expected);
    }

    // ── NamedBlockCsrSoaEntry tests ───────────────────────────────────

    #[test]
    fn named_entry_delegates_to_inner() {
        let block = BlockShape::new(3, 3);
        let inner_entry = BlockCsrSoaEntry::new(
            Expr::ident("vals"),
            (0..3).map(|r| Expr::ident(format!("base_{r}"))).collect(),
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let named = NamedBlockCsrSoaEntry::<IncompressibleAxis2D>::new(inner_entry.clone());

        // entry()
        let typed = named
            .entry(
                BlockRow::new(IncompressibleAxis2D::Ux),
                BlockCol::new(IncompressibleAxis2D::P),
            )
            .expr
            .to_string();
        assert_eq!(typed, inner_entry.entry(0, 2).expr.to_string());

        // index_expr()
        let typed_idx = named
            .index_expr(
                BlockRow::new(IncompressibleAxis2D::Uy),
                BlockCol::new(IncompressibleAxis2D::Ux),
            )
            .to_string();
        assert_eq!(typed_idx, inner_entry.index_expr(1, 0).to_string());

        // access_expr()
        let typed_acc = named
            .access_expr(
                BlockRow::new(IncompressibleAxis2D::P),
                BlockCol::new(IncompressibleAxis2D::Uy),
            )
            .to_string();
        assert_eq!(typed_acc, inner_entry.access_expr(2, 1).to_string());
    }

    #[test]
    fn named_entry_from_row_entry() {
        let named_mat = NamedBlockCsrSoaMatrix::<ScalarAxis>::from_start_row_prefix(
            "v",
            "s",
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let rank = Expr::ident("r");
        let named_entry = named_mat.row_entry(&rank);

        let expr = named_entry
            .entry(
                BlockRow::new(ScalarAxis::Phi),
                BlockCol::new(ScalarAxis::Phi),
            )
            .expr
            .to_string();
        // Should be same as inner().row_entry().entry(0, 0)
        let expected = named_mat
            .inner()
            .row_entry(&rank)
            .entry(0, 0)
            .expr
            .to_string();
        assert_eq!(expr, expected);
    }
}
