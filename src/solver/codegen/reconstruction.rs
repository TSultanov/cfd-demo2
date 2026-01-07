use super::wgsl_ast::{Expr, Stmt};
use super::wgsl_dsl as dsl;

#[derive(Debug, Clone)]
pub struct ScalarReconstruction {
    pub phi_upwind: Expr,
    pub phi_ho: Expr,
}

pub fn scalar_reconstruction(
    prefix: &str,
    scheme_id: Expr,
    flux: Expr,
    phi_own: Expr,
    phi_neigh: Expr,
    grad_own: Expr,
    grad_neigh: Expr,
    center: Expr,
    other_center: Expr,
    face_center: Expr,
) -> (Vec<Stmt>, ScalarReconstruction) {
    let phi_upwind = format!("phi_upwind_{prefix}");
    let phi_ho = format!("phi_ho_{prefix}");

    let mut stmts = Vec::new();
    stmts.push(dsl::var_expr(&phi_upwind, phi_own));
    stmts.push(dsl::if_block_expr(
        flux.lt(0.0),
        dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_upwind),
            phi_neigh,
        )]),
        None,
    ));
    stmts.push(dsl::var_expr(&phi_ho, Expr::ident(&phi_upwind)));

    let xy = |point: &Expr| dsl::vec2_f32(point.field("x"), point.field("y"));

    let sou_block = dsl::block(vec![dsl::if_block_expr(
        flux.gt(0.0),
        dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            phi_own
                + dsl::dot(
                    dsl::vec2_f32_from_xy_fields(grad_own),
                    xy(&face_center) - xy(&center),
                ),
        )]),
        Some(dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            phi_neigh
                + dsl::dot(
                    dsl::vec2_f32_from_xy_fields(grad_neigh),
                    xy(&face_center) - xy(&other_center),
                ),
        )])),
    )]);

    let quick_block = dsl::block(vec![dsl::if_block_expr(
        flux.gt(0.0),
        dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            phi_own * 0.625
                + phi_neigh * 0.375
                + dsl::dot(
                    dsl::vec2_f32_from_xy_fields(grad_own),
                    xy(&other_center) - xy(&center),
                ) * 0.125,
        )]),
        Some(dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            phi_neigh * 0.625
                + phi_own * 0.375
                + dsl::dot(
                    dsl::vec2_f32_from_xy_fields(grad_neigh),
                    xy(&center) - xy(&other_center),
                ) * 0.125,
        )])),
    )]);

    stmts.push(dsl::if_block_expr(
        scheme_id.eq(1u32),
        sou_block,
        Some(dsl::block(vec![dsl::if_block_expr(
            scheme_id.eq(2u32),
            quick_block,
            None,
        )])),
    ));

    (
        stmts,
        ScalarReconstruction {
            phi_upwind: Expr::ident(&phi_upwind),
            phi_ho: Expr::ident(&phi_ho),
        },
    )
}

pub fn limited_linear_reconstruct_face(
    prefix: &str,
    side: &str,
    phi_cell: Expr,
    phi_other: Expr,
    grad_cell: Expr,
    r_x: Expr,
    r_y: Expr,
) -> (Vec<Stmt>, Expr) {
    let diff = format!("diff_{prefix}_{side}");
    let min_diff = format!("min_diff_{prefix}_{side}");
    let max_diff = format!("max_diff_{prefix}_{side}");
    let delta = format!("delta_{prefix}_{side}");
    let delta_limited = format!("delta_{prefix}_{side}_limited");
    let phi_face = format!("{prefix}_{side}_face");

    let mut stmts = Vec::new();
    stmts.push(dsl::let_expr(
        &diff,
        phi_other - phi_cell,
    ));
    stmts.push(dsl::let_expr(
        &min_diff,
        dsl::min(Expr::ident(&diff), 0.0),
    ));
    stmts.push(dsl::let_expr(
        &max_diff,
        dsl::max(Expr::ident(&diff), 0.0),
    ));
    stmts.push(dsl::let_expr(
        &delta,
        dsl::dot(dsl::vec2_f32_from_xy_fields(grad_cell), dsl::vec2_f32(r_x, r_y)),
    ));
    stmts.push(dsl::let_expr(
        &delta_limited,
        dsl::min(
            dsl::max(Expr::ident(&delta), Expr::ident(&min_diff)),
            Expr::ident(&max_diff),
        ),
    ));
    stmts.push(dsl::let_expr(
        &phi_face,
        phi_cell + Expr::ident(&delta_limited),
    ));

    (stmts, Expr::ident(&phi_face))
}
