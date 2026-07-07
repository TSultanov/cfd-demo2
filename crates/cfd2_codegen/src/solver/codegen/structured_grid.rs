//! Shared emitters for the `TopologyMode::Structured2D` kernel family.
//!
//! Every face-gathering kernel (assembly, gradients, flux module, Rhie–Chow)
//! needs the same thing: for a dense Cartesian cell `idx = gj*nx + gi`, walk its
//! 4 faces and bind the neighbour cell, the outward face normal, the face area,
//! the cell-to-cell spacing and the face/neighbour centres — all by index
//! arithmetic on the `grid` uniform, with NO connectivity buffers.
//!
//! These helpers emit that descriptor under fixed `sfd_*` local names (Structured
//! Face Descriptor); each subsystem then aliases them to whatever names its body
//! already uses (`normal` vs `normal_vec`, `area`, `other_idx`, …), so the
//! subsystem's physics stays byte-for-byte the unstructured code. The direction
//! order is the ascending-column band layout `[S(idx-nx), W(idx-1), diag,
//! E(idx+1), N(idx+nx)]`: `k = 0` South, `1` West, `2` East, `3` North.

use super::wgsl_ast::{AccessMode, Expr, Item, Stmt, Type};
use super::wgsl_bindings::storage_var;
use super::wgsl_dsl as dsl;

fn id(s: &str) -> Expr {
    Expr::ident(s)
}
fn grid(f: &str) -> Expr {
    Expr::ident("grid").field(f)
}

/// Bind the structured CELL geometry for `idx`: `sfd_gi`, `sfd_gj` (i,j),
/// `sfd_cx`, `sfd_cy` (cell-centre coords) and `sfd_vol` (`dx*dy`). Emit once at
/// the top of a cell-dispatched structured kernel.
pub fn structured_cell_geom() -> Vec<Stmt> {
    let f32_of = |s: &str| Expr::call_named("f32", vec![Expr::ident(s)]);
    vec![
        dsl::let_expr("sfd_gi", id("idx").modulo(grid("nx"))),
        dsl::let_expr("sfd_gj", id("idx") / grid("nx")),
        dsl::let_expr("sfd_cx", (f32_of("sfd_gi") + Expr::from(0.5)) * grid("dx")),
        dsl::let_expr("sfd_cy", (f32_of("sfd_gj") + Expr::from(0.5)) * grid("dy")),
        dsl::let_expr("sfd_vol", grid("dx") * grid("dy")),
    ]
}

/// Bind the structured FACE descriptor for direction `k` (0..4). Requires the
/// cell geometry (`structured_cell_geom`) and the loop variable `k` in scope.
/// Binds (all `sfd_*`):
/// - `sfd_axis_is_x` (bool): W/E faces (x-aligned normal)
/// - `sfd_normal_x`, `sfd_normal_y` (f32): unit normal outward from `idx`
/// - `sfd_area` (f32): face area (the OTHER axis' spacing)
/// - `sfd_spacing` (f32): cell-to-cell spacing along the face normal
/// - `sfd_is_boundary` (bool): the face lies on a domain edge
/// - `sfd_neighbor` (u32): neighbour cell index (valid only when interior)
/// - `sfd_other_idx` (u32): `sfd_neighbor` interior, else `idx`
/// - `sfd_face_cx/cy` (f32): face-centre coords
/// - `sfd_other_cx/cy` (f32): neighbour-centre coords (face centre on a boundary)
/// - `sfd_band_rank` (u32): off-diagonal band slot (S=0,W=1,E=3,N=4)
/// - `sfd_face_id` (u32): per-(cell,dir) face id `idx*4 + k` for the BC table
pub fn structured_face_locals() -> Vec<Stmt> {
    let k = || id("k");
    let axis_is_x = k().ge(1u32) & k().le(2u32);
    let k_pos = k().ge(2u32);
    vec![
        dsl::let_expr("sfd_axis_is_x", axis_is_x),
        dsl::let_expr("sfd_sign", dsl::select(-1.0, 1.0, k_pos.clone())),
        dsl::let_expr(
            "sfd_normal_x",
            dsl::select(0.0, id("sfd_sign"), id("sfd_axis_is_x")),
        ),
        dsl::let_expr(
            "sfd_normal_y",
            dsl::select(id("sfd_sign"), 0.0, id("sfd_axis_is_x")),
        ),
        dsl::let_expr("sfd_area", dsl::select(grid("dx"), grid("dy"), id("sfd_axis_is_x"))),
        dsl::let_expr("sfd_spacing", dsl::select(grid("dy"), grid("dx"), id("sfd_axis_is_x"))),
        dsl::let_expr("sfd_half", Expr::from(0.5) * id("sfd_spacing")),
        dsl::let_expr("sfd_coord", dsl::select(id("sfd_gj"), id("sfd_gi"), id("sfd_axis_is_x"))),
        dsl::let_expr("sfd_ext", dsl::select(grid("ny"), grid("nx"), id("sfd_axis_is_x"))),
        dsl::let_expr(
            "sfd_is_boundary",
            dsl::select(id("sfd_coord").eq(id("sfd_ext") - 1u32), id("sfd_coord").eq(0u32), k().lt(2u32)),
        ),
        dsl::let_expr("sfd_off", dsl::select(grid("nx"), 1u32, id("sfd_axis_is_x"))),
        dsl::let_expr(
            "sfd_neighbor",
            dsl::select(id("idx") - id("sfd_off"), id("idx") + id("sfd_off"), k_pos),
        ),
        dsl::let_expr(
            "sfd_other_idx",
            dsl::select(id("sfd_neighbor"), id("idx"), id("sfd_is_boundary")),
        ),
        dsl::let_expr("sfd_face_cx", id("sfd_cx") + id("sfd_half") * id("sfd_normal_x")),
        dsl::let_expr("sfd_face_cy", id("sfd_cy") + id("sfd_half") * id("sfd_normal_y")),
        dsl::let_expr("sfd_mult", dsl::select(id("sfd_spacing"), id("sfd_half"), id("sfd_is_boundary"))),
        dsl::let_expr("sfd_other_cx", id("sfd_cx") + id("sfd_mult") * id("sfd_normal_x")),
        dsl::let_expr("sfd_other_cy", id("sfd_cy") + id("sfd_mult") * id("sfd_normal_y")),
        dsl::let_expr(
            "sfd_band_rank",
            dsl::select(k() + 1u32, k(), k().lt(2u32)),
        ),
        dsl::let_expr("sfd_face_id", id("idx") * 4u32 + k()),
    ]
}

/// The boundary TYPE for the current structured face (`face_boundary[sfd_face_id]`,
/// 0 interior). Kept separate from [`structured_face_locals`] because only the
/// assembly + flux module need it (and its `face_boundary` binding); the gradient
/// kernels do not. Parity with the unstructured `face_boundary[face_idx]`.
pub fn structured_boundary_type() -> Expr {
    dsl::array_access("face_boundary", id("sfd_face_id"))
}

/// A `vec2<f32>` expression from two scalar component idents.
pub fn sfd_vec2(x: &str, y: &str) -> Expr {
    Expr::call_named("vec2<f32>", vec![id(x), id(y)])
}

/// A `Vector2(...)` struct expression from two scalar component idents.
pub fn sfd_vector2(x: &str, y: &str) -> Expr {
    Expr::call_named("Vector2", vec![id(x), id(y)])
}

/// The structured `face_boundary` storage binding (group 0, binding 1): the
/// boundary TYPE per `(cell, dir)` face id (`idx*4 + k`), `0` on interior faces.
/// Emitted alongside the `grid` uniform by every structured face-gathering
/// kernel so `sfd_boundary_type` resolves.
pub fn structured_face_boundary_binding() -> Item {
    storage_var("face_boundary", Type::array(Type::U32), 0, 1, AccessMode::Read)
}

/// The structured bound guard `idx >= grid.nx * grid.ny` (a `Stmt::If` returning).
pub fn structured_bound_guard() -> Stmt {
    dsl::if_block_expr(
        id("idx").ge(grid("nx") * grid("ny")),
        dsl::block(vec![Stmt::Return(None)]),
        None,
    )
}

/// Helper: `let <name>: <ty> = <expr>;` sugar re-exported for subsystem heads.
pub fn let_typed(name: &str, ty: Type, expr: Expr) -> Stmt {
    dsl::let_typed_expr(name, ty, expr)
}
