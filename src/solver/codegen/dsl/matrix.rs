use crate::solver::codegen::wgsl_ast::{BinaryOp, Expr};

use super::expr::TypedExpr;
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
        self.row_offsets.clone().index(row.clone())
    }

    pub fn row_end(&self, row: &Expr) -> Expr {
        let next = Expr::binary(row.clone(), BinaryOp::Add, Expr::lit_u32(1));
        self.row_offsets.clone().index(next)
    }

    pub fn col_at(&self, nnz_index: &Expr) -> Expr {
        self.col_indices.clone().index(nnz_index.clone())
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

    pub fn value_at(&self, nnz_index: &Expr) -> TypedExpr {
        TypedExpr::new(
            self.values.clone().index(nnz_index.clone()),
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

    pub fn entry(&self, nnz_index: &Expr, row: u8, col: u8) -> TypedExpr {
        assert!(row < self.block.rows, "block row out of bounds");
        assert!(col < self.block.cols, "block col out of bounds");
        let block_stride = Expr::lit_u32(self.block.entry_count());
        let base = Expr::binary(nnz_index.clone(), BinaryOp::Mul, block_stride);
        let offset = Expr::lit_u32(row as u32 * self.block.cols as u32 + col as u32);
        let index = Expr::binary(base, BinaryOp::Add, offset);
        let ty = DslType::new(self.scalar, Shape::Scalar);
        TypedExpr::new(self.values.clone().index(index), ty, self.entry_unit)
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
        let cols = Expr::lit_u32(self.block.cols as u32);
        let bases = self
            .start_rows
            .iter()
            .cloned()
            .map(|start| {
                let offset = Expr::binary(cols.clone(), BinaryOp::Mul, rank.clone());
                Expr::binary(start, BinaryOp::Add, offset)
            })
            .collect();
        BlockCsrSoaEntry::new(
            self.values.clone(),
            bases,
            self.block,
            self.scalar,
            self.entry_unit,
        )
    }

    pub fn entry(&self, rank: &Expr, row: u8, col: u8) -> TypedExpr {
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

    pub fn entry(&self, row: u8, col: u8) -> TypedExpr {
        let ty = DslType::new(self.scalar, Shape::Scalar);
        TypedExpr::new(self.access_expr(row, col), ty, self.entry_unit)
    }

    pub fn index_expr(&self, row: u8, col: u8) -> Expr {
        assert!(row < self.block.rows, "block row out of bounds");
        assert!(col < self.block.cols, "block col out of bounds");
        let base = self.row_bases[row as usize].clone();
        Expr::binary(base, BinaryOp::Add, Expr::lit_u32(col as u32))
    }

    pub fn access_expr(&self, row: u8, col: u8) -> Expr {
        self.values.clone().index(self.index_expr(row, col))
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
}
