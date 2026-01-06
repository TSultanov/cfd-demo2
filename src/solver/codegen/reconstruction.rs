use super::wgsl_ast::{BinaryOp, Expr, Stmt};
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
    stmts.push(dsl::var_expr(&phi_upwind, phi_own.clone()));
    stmts.push(dsl::if_block_expr(
        Expr::binary(flux.clone(), BinaryOp::Less, Expr::lit_f32(0.0)),
        dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_upwind),
            phi_neigh.clone(),
        )]),
        None,
    ));
    stmts.push(dsl::var_expr(&phi_ho, Expr::ident(&phi_upwind)));

    let xy = |point: &Expr| dsl::vec2_f32(point.clone().field("x"), point.clone().field("y"));

    let sou_block = dsl::block(vec![dsl::if_block_expr(
        Expr::binary(flux.clone(), BinaryOp::Greater, Expr::lit_f32(0.0)),
        dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            Expr::binary(
                phi_own.clone(),
                BinaryOp::Add,
                dsl::dot_expr(
                    dsl::vec2_f32_from_xy_fields(grad_own.clone()),
                    Expr::binary(xy(&face_center), BinaryOp::Sub, xy(&center)),
                ),
            ),
        )]),
        Some(dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            Expr::binary(
                phi_neigh.clone(),
                BinaryOp::Add,
                dsl::dot_expr(
                    dsl::vec2_f32_from_xy_fields(grad_neigh.clone()),
                    Expr::binary(xy(&face_center), BinaryOp::Sub, xy(&other_center)),
                ),
            ),
        )])),
    )]);

    let quick_block = dsl::block(vec![dsl::if_block_expr(
        Expr::binary(flux.clone(), BinaryOp::Greater, Expr::lit_f32(0.0)),
        dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            Expr::binary(
                Expr::binary(
                    Expr::binary(Expr::lit_f32(0.625), BinaryOp::Mul, phi_own.clone()),
                    BinaryOp::Add,
                    Expr::binary(Expr::lit_f32(0.375), BinaryOp::Mul, phi_neigh.clone()),
                ),
                BinaryOp::Add,
                Expr::binary(
                    Expr::lit_f32(0.125),
                    BinaryOp::Mul,
                    dsl::dot_expr(
                        dsl::vec2_f32_from_xy_fields(grad_own.clone()),
                        Expr::binary(xy(&other_center), BinaryOp::Sub, xy(&center)),
                    ),
                ),
            ),
        )]),
        Some(dsl::block(vec![dsl::assign_expr(
            Expr::ident(&phi_ho),
            Expr::binary(
                Expr::binary(
                    Expr::binary(Expr::lit_f32(0.625), BinaryOp::Mul, phi_neigh.clone()),
                    BinaryOp::Add,
                    Expr::binary(Expr::lit_f32(0.375), BinaryOp::Mul, phi_own.clone()),
                ),
                BinaryOp::Add,
                Expr::binary(
                    Expr::lit_f32(0.125),
                    BinaryOp::Mul,
                    dsl::dot_expr(
                        dsl::vec2_f32_from_xy_fields(grad_neigh.clone()),
                        Expr::binary(xy(&center), BinaryOp::Sub, xy(&other_center)),
                    ),
                ),
            ),
        )])),
    )]);

    stmts.push(dsl::if_block_expr(
        Expr::binary(scheme_id.clone(), BinaryOp::Equal, Expr::lit_u32(1)),
        sou_block,
        Some(dsl::block(vec![dsl::if_block_expr(
            Expr::binary(scheme_id, BinaryOp::Equal, Expr::lit_u32(2)),
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
        Expr::binary(phi_other, BinaryOp::Sub, phi_cell.clone()),
    ));
    stmts.push(dsl::let_expr(
        &min_diff,
        dsl::min_expr(Expr::ident(&diff), Expr::lit_f32(0.0)),
    ));
    stmts.push(dsl::let_expr(
        &max_diff,
        dsl::max_expr(Expr::ident(&diff), Expr::lit_f32(0.0)),
    ));
    stmts.push(dsl::let_expr(
        &delta,
        dsl::dot_expr(dsl::vec2_f32_from_xy_fields(grad_cell), dsl::vec2_f32(r_x, r_y)),
    ));
    stmts.push(dsl::let_expr(
        &delta_limited,
        dsl::min_expr(
            dsl::max_expr(Expr::ident(&delta), Expr::ident(&min_diff)),
            Expr::ident(&max_diff),
        ),
    ));
    stmts.push(dsl::let_expr(
        &phi_face,
        Expr::binary(phi_cell, BinaryOp::Add, Expr::ident(&delta_limited)),
    ));

    (stmts, Expr::ident(&phi_face))
}
