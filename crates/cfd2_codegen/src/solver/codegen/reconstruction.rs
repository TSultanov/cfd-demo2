use super::dsl as typed;
use super::dsl::EnumExpr;
use super::wgsl_ast::{Expr, Stmt};
use super::wgsl_dsl as dsl;
use crate::solver::scheme::Scheme;

#[derive(Debug, Clone)]
pub struct ScalarReconstruction {
    pub phi_upwind: Expr,
    pub phi_ho: Expr,
}

pub fn scalar_reconstruction(
    scheme: EnumExpr<Scheme>,
    flux: Expr,
    phi_own: Expr,
    phi_neigh: Expr,
    grad_own: Expr,
    grad_neigh: Expr,
    center: Expr,
    other_center: Expr,
    face_center: Expr,
) -> ScalarReconstruction {
    let xy = |point: &Expr| dsl::vec2_f32(point.field("x"), point.field("y"));

    let phi_upwind = dsl::select(phi_own, phi_neigh, flux.lt(0.0));

    let minmod_limit = |phi_cell: Expr, phi_other: Expr, delta: Expr| -> Expr {
        let diff = phi_other - phi_cell.clone();
        let min_diff = dsl::min(diff.clone(), 0.0);
        let max_diff = dsl::max(diff, 0.0);
        dsl::min(dsl::max(delta, min_diff), max_diff)
    };

    let vanleer_limit = |phi_cell: Expr, phi_other: Expr, delta: Expr| -> Expr {
        let diff = phi_other - phi_cell.clone();
        let abs_diff = dsl::abs(diff.clone());
        let abs_delta = dsl::abs(delta.clone());
        let denom = dsl::max(abs_diff.clone(), abs_delta + 1e-8);
        delta * (abs_diff / denom)
    };

    let delta_sou_pos = dsl::dot(
        dsl::vec2_f32_from_xy_fields(grad_own.clone()),
        xy(&face_center) - xy(&center),
    );
    let delta_sou_neg = dsl::dot(
        dsl::vec2_f32_from_xy_fields(grad_neigh.clone()),
        xy(&face_center) - xy(&other_center),
    );

    let sou_pos = phi_own.clone() + delta_sou_pos.clone();
    let sou_neg = phi_neigh.clone() + delta_sou_neg.clone();
    let phi_sou = dsl::select(sou_neg, sou_pos, flux.gt(0.0));

    let sou_pos_mm = phi_own.clone() + minmod_limit(phi_own.clone(), phi_neigh.clone(), delta_sou_pos.clone());
    let sou_neg_mm = phi_neigh.clone() + minmod_limit(phi_neigh.clone(), phi_own.clone(), delta_sou_neg.clone());
    let phi_sou_mm = dsl::select(sou_neg_mm, sou_pos_mm, flux.gt(0.0));

    let sou_pos_vl = phi_own.clone() + vanleer_limit(phi_own.clone(), phi_neigh.clone(), delta_sou_pos);
    let sou_neg_vl = phi_neigh.clone() + vanleer_limit(phi_neigh.clone(), phi_own.clone(), delta_sou_neg);
    let phi_sou_vl = dsl::select(sou_neg_vl, sou_pos_vl, flux.gt(0.0));

    let quick_pos = phi_own.clone() * 0.625
        + phi_neigh.clone() * 0.375
        + dsl::dot(
            dsl::vec2_f32_from_xy_fields(grad_own),
            xy(&other_center) - xy(&center),
        ) * 0.125;
    let quick_neg = phi_neigh.clone() * 0.625
        + phi_own.clone() * 0.375
        + dsl::dot(
            dsl::vec2_f32_from_xy_fields(grad_neigh),
            xy(&center) - xy(&other_center),
        ) * 0.125;
    let phi_quick = dsl::select(quick_neg, quick_pos, flux.gt(0.0));

    let delta_quick_pos = quick_pos.clone() - phi_own.clone();
    let delta_quick_neg = quick_neg.clone() - phi_neigh.clone();
    let quick_pos_mm = phi_own.clone() + minmod_limit(phi_own.clone(), phi_neigh.clone(), delta_quick_pos.clone());
    let quick_neg_mm = phi_neigh.clone() + minmod_limit(phi_neigh.clone(), phi_own.clone(), delta_quick_neg.clone());
    let phi_quick_mm = dsl::select(quick_neg_mm, quick_pos_mm, flux.gt(0.0));

    let quick_pos_vl = phi_own.clone() + vanleer_limit(phi_own.clone(), phi_neigh.clone(), delta_quick_pos);
    let quick_neg_vl = phi_neigh.clone() + vanleer_limit(phi_neigh.clone(), phi_own.clone(), delta_quick_neg);
    let phi_quick_vl = dsl::select(quick_neg_vl, quick_pos_vl, flux.gt(0.0));

    let phi_ho = dsl::select(phi_upwind, phi_sou, scheme.eq(Scheme::SecondOrderUpwind));
    let phi_ho = dsl::select(phi_ho, phi_quick, scheme.eq(Scheme::QUICK));
    let phi_ho = dsl::select(
        phi_ho,
        phi_sou_mm,
        scheme.eq(Scheme::SecondOrderUpwindMinMod),
    );
    let phi_ho = dsl::select(
        phi_ho,
        phi_sou_vl,
        scheme.eq(Scheme::SecondOrderUpwindVanLeer),
    );
    let phi_ho = dsl::select(phi_ho, phi_quick_mm, scheme.eq(Scheme::QUICKMinMod));
    let phi_ho = dsl::select(phi_ho, phi_quick_vl, scheme.eq(Scheme::QUICKVanLeer));

    ScalarReconstruction { phi_upwind, phi_ho }
}

#[derive(Debug, Clone)]
pub struct Vec2Reconstruction {
    pub phi_upwind: typed::VecExpr<2>,
    pub phi_ho: typed::VecExpr<2>,
}

pub fn vec2_reconstruction_xy(
    scheme: EnumExpr<Scheme>,
    flux: Expr,
    phi_own: typed::VecExpr<2>,
    phi_neigh: typed::VecExpr<2>,
    grad_own: [Expr; 2],
    grad_neigh: [Expr; 2],
    center: Expr,
    other_center: Expr,
    face_center: Expr,
) -> Vec2Reconstruction {
    let rec_x = scalar_reconstruction(
        scheme,
        flux,
        phi_own.component(0),
        phi_neigh.component(0),
        grad_own[0],
        grad_neigh[0],
        center,
        other_center,
        face_center,
    );
    let rec_y = scalar_reconstruction(
        scheme,
        flux,
        phi_own.component(1),
        phi_neigh.component(1),
        grad_own[1],
        grad_neigh[1],
        center,
        other_center,
        face_center,
    );

    Vec2Reconstruction {
        phi_upwind: typed::VecExpr::<2>::from_components([rec_x.phi_upwind, rec_y.phi_upwind]),
        phi_ho: typed::VecExpr::<2>::from_components([rec_x.phi_ho, rec_y.phi_ho]),
    }
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
    stmts.push(dsl::let_expr(&diff, phi_other - phi_cell));
    stmts.push(dsl::let_expr(&min_diff, dsl::min(Expr::ident(&diff), 0.0)));
    stmts.push(dsl::let_expr(&max_diff, dsl::max(Expr::ident(&diff), 0.0)));
    stmts.push(dsl::let_expr(
        &delta,
        dsl::dot(
            dsl::vec2_f32_from_xy_fields(grad_cell),
            dsl::vec2_f32(r_x, r_y),
        ),
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
