use crate::solver::codegen::dsl::EnumExpr;
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::codegen::wgsl_dsl as dsl;
use crate::solver::gpu::enums::GpuBcKind;

/// Code-generation helper for the flat `bc_kind[]` / `bc_value[]`
/// boundary-condition lookup tables used in assembly, gradient, and
/// flux-module kernels.
///
/// Layout: `bc_kind[face_idx * stride + unknown_offset]` (u32)
///         `bc_value[face_idx * stride + unknown_offset]` (f32)
///
/// # Motivation
///
/// Every BC lookup site manually computes
/// `face_expr * stride + offset`, then passes the result to
/// `dsl::array_access("bc_kind", …)` and `dsl::array_access("bc_value", …)`.
/// This struct centralises that arithmetic and provides typed accessors
/// that make the intent explicit.
///
/// # Example
///
/// ```ignore
/// let bc = BcTable::new(Expr::ident("face_idx"), coupled_stride);
/// let (kind, value) = bc.lookup(u_idx);
/// // kind: EnumExpr<GpuBcKind>, value: Expr
/// ```
pub struct BcTable {
    face_expr: Expr,
    stride: Expr,
}

impl BcTable {
    /// Create a new BC table accessor.
    ///
    /// * `face_expr` — the WGSL expression for the face index
    ///   (typically `Expr::ident("face_idx")` or `Expr::ident("idx")`).
    /// * `stride` — the number of unknowns per face (coupled stride).
    pub fn new(face_expr: Expr, stride: impl Into<Expr>) -> Self {
        Self {
            face_expr,
            stride: stride.into(),
        }
    }

    /// Returns the index expression: `face_expr * stride + unknown_offset`.
    fn index(&self, unknown_offset: impl Into<Expr>) -> Expr {
        self.face_expr.clone() * self.stride.clone() + unknown_offset.into()
    }

    /// Returns `bc_kind[face * stride + offset]` as a typed `GpuBcKind` enum expression.
    ///
    /// Use this in assembly kernels where BC kind is compared via
    /// `kind.eq(GpuBcKind::Dirichlet)`.
    pub fn kind(&self, unknown_offset: impl Into<Expr>) -> EnumExpr<GpuBcKind> {
        EnumExpr::<GpuBcKind>::from_expr(dsl::array_access("bc_kind", self.index(unknown_offset)))
    }

    /// Returns `bc_kind[face * stride + offset]` as a raw `Expr`.
    ///
    /// Use this in gradient/flux kernels where BC kind is compared via
    /// `kind.eq(Expr::from(1u32))` inside `dsl::select` chains.
    pub fn kind_raw(&self, unknown_offset: impl Into<Expr>) -> Expr {
        dsl::array_access("bc_kind", self.index(unknown_offset))
    }

    /// Returns `bc_value[face * stride + offset]`.
    pub fn value(&self, unknown_offset: impl Into<Expr>) -> Expr {
        dsl::array_access("bc_value", self.index(unknown_offset))
    }

    /// Returns `(kind_typed, value)` — convenience for assembly-style BC
    /// lookups that need both the typed enum and the scalar value.
    pub fn lookup(&self, unknown_offset: impl Into<Expr>) -> (EnumExpr<GpuBcKind>, Expr) {
        let idx = self.index(unknown_offset);
        let kind = EnumExpr::<GpuBcKind>::from_expr(dsl::array_access("bc_kind", idx.clone()));
        let value = dsl::array_access("bc_value", idx);
        (kind, value)
    }

    /// Computes the standard gradient ghost value for a boundary face:
    ///
    /// ```wgsl
    /// select(
    ///   select(cell_val, bc_value, kind == 1u),   // Dirichlet
    ///   cell_val + bc_value * d_own,               // Neumann
    ///   kind == 2u,
    /// )
    /// ```
    ///
    /// This deduplicates the identical nested `select` pattern used in
    /// `packed_state_gradients.rs`, `flux_module_gradients_wgsl.rs`,
    /// and `rhie_chow.rs`.
    pub fn ghost_value(
        &self,
        unknown_offset: impl Into<Expr>,
        cell_val: Expr,
        d_own: Expr,
    ) -> Expr {
        let idx = self.index(unknown_offset);
        let kind = dsl::array_access("bc_kind", idx.clone());
        let value = dsl::array_access("bc_value", idx);
        dsl::select(
            dsl::select(
                cell_val.clone(),
                value.clone(),
                kind.clone().eq(Expr::from(1u32)),
            ),
            cell_val + value * d_own,
            kind.eq(Expr::from(2u32)),
        )
    }
}

/// Host-side helper for the flat BC table layout
/// (`row_index * stride + column_index`).
///
/// Used when constructing the `bc_kind` / `bc_value` vectors on the CPU
/// before uploading to the GPU, and when patching individual BC values
/// at runtime.
pub struct HostBcTable {
    /// Number of unknown components per face/boundary-type row.
    pub stride: usize,
}

/// Number of boundary type categories
/// (None, Inlet, Outlet, Wall, SlipWall, MovingWall).
pub const BOUNDARY_TYPE_COUNT: usize = 6;

impl HostBcTable {
    /// Create a host-side BC table helper with the given coupled stride.
    pub fn new(stride: usize) -> Self {
        Self { stride }
    }

    /// Returns the flat array offset: `row * stride + col`.
    #[inline]
    pub fn offset(&self, row: usize, col: usize) -> usize {
        row * self.stride + col
    }

    /// Returns the byte offset for a single `f32`/`u32` element:
    /// `(row * stride + col) * 4`.
    ///
    /// Useful for `queue.write_buffer()` calls that patch individual entries.
    #[inline]
    pub fn byte_offset(&self, row: usize, col: usize) -> u64 {
        (self.offset(row, col) as u64) * 4
    }

    /// Returns the base offset for the start of a row: `row * stride`.
    ///
    /// Useful for `copy_from_slice` calls that copy an entire row.
    #[inline]
    pub fn row_base(&self, row: usize) -> usize {
        row * self.stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BcTable (codegen DSL) ────────────────────────────────────────

    #[test]
    fn bc_table_kind_produces_typed_enum_expr() {
        let bc = BcTable::new(Expr::ident("face_idx"), Expr::from(3u32));
        let kind = bc.kind(Expr::from(1u32));
        // Should produce bc_kind[face_idx * 3u + 1u] wrapped in EnumExpr
        let eq_dirichlet = kind.eq(GpuBcKind::Dirichlet);
        // The expression should compile without panicking
        let _ = format!("{:?}", eq_dirichlet);
    }

    #[test]
    fn bc_table_kind_raw_produces_raw_expr() {
        let bc = BcTable::new(Expr::ident("face_idx"), Expr::from(3u32));
        let kind = bc.kind_raw(Expr::from(0u32));
        // Raw expr can be compared with Expr::from(1u32) for select-style usage
        let eq_1 = kind.eq(Expr::from(1u32));
        let _ = format!("{:?}", eq_1);
    }

    #[test]
    fn bc_table_value_produces_array_access() {
        let bc = BcTable::new(Expr::ident("face_idx"), Expr::from(3u32));
        let value = bc.value(Expr::from(2u32));
        let _ = format!("{:?}", value);
    }

    #[test]
    fn bc_table_lookup_returns_kind_and_value() {
        let bc = BcTable::new(Expr::ident("face_idx"), Expr::from(3u32));
        let (kind, value) = bc.lookup(Expr::from(1u32));
        // Both should be usable
        let _ = kind.eq(GpuBcKind::Dirichlet);
        let _ = format!("{:?}", value);
    }

    #[test]
    fn bc_table_ghost_value_produces_nested_select() {
        let bc = BcTable::new(Expr::ident("face_idx"), Expr::from(3u32));
        let ghost = bc.ghost_value(
            Expr::from(0u32),
            Expr::ident("cell_val"),
            Expr::ident("d_own"),
        );
        let rendered = ghost.to_string();
        // Should contain "select" calls
        assert!(
            rendered.contains("select"),
            "ghost_value should produce select expressions, got: {rendered}"
        );
    }

    #[test]
    fn bc_table_ghost_value_matches_manual_pattern() {
        // Reproduce the manual pattern from packed_state_gradients.rs
        let face = Expr::ident("face_idx");
        let stride = Expr::from(3u32);
        let component = Expr::from(1u32);
        let cell_val = Expr::ident("cell_val");
        let d_own = Expr::ident("d_own");

        // Manual pattern:
        let manual_idx = face.clone() * stride.clone() + component.clone();
        let manual_kind = dsl::array_access("bc_kind", manual_idx.clone());
        let manual_value = dsl::array_access("bc_value", manual_idx);
        let manual_ghost = dsl::select(
            dsl::select(
                cell_val.clone(),
                manual_value.clone(),
                manual_kind.clone().eq(Expr::from(1u32)),
            ),
            cell_val.clone() + manual_value * d_own.clone(),
            manual_kind.eq(Expr::from(2u32)),
        );

        // BcTable pattern:
        let bc = BcTable::new(Expr::ident("face_idx"), Expr::from(3u32));
        let typed_ghost = bc.ghost_value(Expr::from(1u32), cell_val, d_own);

        // Compare rendered WGSL output (structural equality)
        assert_eq!(
            manual_ghost.to_string(),
            typed_ghost.to_string(),
            "BcTable::ghost_value should produce the same WGSL as the manual pattern"
        );
    }

    // ── HostBcTable ──────────────────────────────────────────────────

    #[test]
    fn host_bc_table_offset() {
        let table = HostBcTable::new(3);
        assert_eq!(table.offset(0, 0), 0);
        assert_eq!(table.offset(0, 2), 2);
        assert_eq!(table.offset(1, 0), 3);
        assert_eq!(table.offset(5, 2), 17);
    }

    #[test]
    fn host_bc_table_byte_offset() {
        let table = HostBcTable::new(3);
        assert_eq!(table.byte_offset(0, 0), 0);
        assert_eq!(table.byte_offset(0, 1), 4);
        assert_eq!(table.byte_offset(1, 0), 12);
        assert_eq!(table.byte_offset(5, 2), 68);
    }

    #[test]
    fn host_bc_table_row_base() {
        let table = HostBcTable::new(4);
        assert_eq!(table.row_base(0), 0);
        assert_eq!(table.row_base(1), 4);
        assert_eq!(table.row_base(3), 12);
    }
}
