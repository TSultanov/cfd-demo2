use super::dsl as typed;
use super::dsl::EnumExpr;
use super::wgsl_ast::{Expr, Stmt};
use super::wgsl_dsl as dsl;
use crate::solver::ir::reconstruction::{limited_linear_face_value, quick_face_value, ReconstructionBuilder};
use crate::solver::ir::LimiterSpec;
use crate::solver::scheme::Scheme;

#[derive(Debug, Clone)]
pub struct ScalarReconstruction {
    pub phi_upwind: Expr,
    pub phi_ho: Expr,
}

struct WgslExprBuilder;

impl ReconstructionBuilder for WgslExprBuilder {
    type Scalar = Expr;
    type Vec2 = Expr;

    fn lit(v: f32) -> Self::Scalar {
        v.into()
    }

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a + b
    }

    fn sub(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a - b
    }

    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a * b
    }

    fn div(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a / b
    }

    fn abs(a: Self::Scalar) -> Self::Scalar {
        dsl::abs(a)
    }

    fn min(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        dsl::min(a, b)
    }

    fn max(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        dsl::max(a, b)
    }

    fn vec2_sub(a: Self::Vec2, b: Self::Vec2) -> Self::Vec2 {
        a - b
    }

    fn vec2_dot(a: Self::Vec2, b: Self::Vec2) -> Self::Scalar {
        dsl::dot(a, b)
    }
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

    let grad_own_vec = dsl::vec2_f32_from_xy_fields(grad_own.clone());
    let grad_neigh_vec = dsl::vec2_f32_from_xy_fields(grad_neigh.clone());
    let r_own = xy(&face_center) - xy(&center);
    let r_neigh = xy(&face_center) - xy(&other_center);

    let sou_pos = limited_linear_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec.clone(),
        r_own.clone(),
        LimiterSpec::None,
    );
    let sou_neg = limited_linear_face_value::<WgslExprBuilder>(
        phi_neigh.clone(),
        phi_own.clone(),
        grad_neigh_vec.clone(),
        r_neigh.clone(),
        LimiterSpec::None,
    );
    let phi_sou = dsl::select(sou_neg, sou_pos, flux.gt(0.0));

    let sou_pos_mm = limited_linear_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec.clone(),
        r_own.clone(),
        LimiterSpec::MinMod,
    );
    let sou_neg_mm = limited_linear_face_value::<WgslExprBuilder>(
        phi_neigh.clone(),
        phi_own.clone(),
        grad_neigh_vec.clone(),
        r_neigh.clone(),
        LimiterSpec::MinMod,
    );
    let phi_sou_mm = dsl::select(sou_neg_mm, sou_pos_mm, flux.gt(0.0));

    let sou_pos_vl = limited_linear_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec.clone(),
        r_own.clone(),
        LimiterSpec::VanLeer,
    );
    let sou_neg_vl = limited_linear_face_value::<WgslExprBuilder>(
        phi_neigh.clone(),
        phi_own.clone(),
        grad_neigh_vec.clone(),
        r_neigh.clone(),
        LimiterSpec::VanLeer,
    );
    let phi_sou_vl = dsl::select(sou_neg_vl, sou_pos_vl, flux.gt(0.0));

    let d_pos = xy(&other_center) - xy(&center);
    let d_neg = xy(&center) - xy(&other_center);

    let quick_pos = quick_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec.clone(),
        d_pos.clone(),
        LimiterSpec::None,
    );
    let quick_neg = quick_face_value::<WgslExprBuilder>(
        phi_neigh.clone(),
        phi_own.clone(),
        grad_neigh_vec.clone(),
        d_neg.clone(),
        LimiterSpec::None,
    );
    let phi_quick = dsl::select(quick_neg, quick_pos, flux.gt(0.0));

    let quick_pos_mm = quick_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec.clone(),
        d_pos.clone(),
        LimiterSpec::MinMod,
    );
    let quick_neg_mm = quick_face_value::<WgslExprBuilder>(
        phi_neigh.clone(),
        phi_own.clone(),
        grad_neigh_vec.clone(),
        d_neg.clone(),
        LimiterSpec::MinMod,
    );
    let phi_quick_mm = dsl::select(quick_neg_mm, quick_pos_mm, flux.gt(0.0));

    let quick_pos_vl = quick_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec,
        d_pos,
        LimiterSpec::VanLeer,
    );
    let quick_neg_vl = quick_face_value::<WgslExprBuilder>(
        phi_neigh.clone(),
        phi_own.clone(),
        grad_neigh_vec,
        d_neg,
        LimiterSpec::VanLeer,
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::wgsl_ast::{
        Block, Function, Item, Module, Param, StructDef, StructField, Type,
    };

    fn wgsl_for_phi_ho(scheme: Scheme) -> String {
        let mut module = Module::new();

        module.push(Item::Struct(StructDef::new(
            "Vector2",
            vec![
                StructField::new("x", Type::F32),
                StructField::new("y", Type::F32),
            ],
        )));

        let rec = scalar_reconstruction(
            typed::EnumExpr::<Scheme>::from_expr(scheme.gpu_id().into()),
            Expr::ident("flux"),
            Expr::ident("phi_own"),
            Expr::ident("phi_neigh"),
            Expr::ident("grad_own"),
            Expr::ident("grad_neigh"),
            Expr::ident("center"),
            Expr::ident("other_center"),
            Expr::ident("face_center"),
        );

        let body = Block::new(vec![Stmt::Return(Some(rec.phi_ho))]);
        module.push(Item::Function(Function::new(
            "test_phi_ho",
            vec![
                Param::new("flux", Type::F32, Vec::new()),
                Param::new("phi_own", Type::F32, Vec::new()),
                Param::new("phi_neigh", Type::F32, Vec::new()),
                Param::new("grad_own", Type::Custom("Vector2".to_string()), Vec::new()),
                Param::new(
                    "grad_neigh",
                    Type::Custom("Vector2".to_string()),
                    Vec::new(),
                ),
                Param::new("center", Type::Custom("Vector2".to_string()), Vec::new()),
                Param::new(
                    "other_center",
                    Type::Custom("Vector2".to_string()),
                    Vec::new(),
                ),
                Param::new(
                    "face_center",
                    Type::Custom("Vector2".to_string()),
                    Vec::new(),
                ),
            ],
            Some(Type::F32),
            Vec::new(),
            body,
        )));

        module.to_wgsl()
    }

    #[test]
    fn vanleer_limiter_guards_opposite_signed_slopes() {
        // Regression: VanLeer-limited schemes must not allow opposite-signed slopes.
        // Ensure the codegen emits a sign guard (max(p,0)/max(abs(p),eps)) for the VanLeer branch.
        for scheme in [Scheme::SecondOrderUpwindVanLeer, Scheme::QUICKVanLeer] {
            let wgsl = wgsl_for_phi_ho(scheme);
            let compact: String = wgsl.chars().filter(|c| !c.is_whitespace()).collect();
            assert!(
                compact.contains("max(abs("),
                "expected {scheme:?} VanLeer limiter to include max(abs(p), eps) in the sign guard"
            );
            assert!(
                compact.contains("max(") && compact.contains("abs(") && compact.contains("0.0"),
                "expected {scheme:?} VanLeer limiter to include abs/max and a 0.0 clamp"
            );
            assert!(
                compact.contains("1e-8") || compact.contains("0.00000001"),
                "expected {scheme:?} VanLeer limiter to include the epsilon literal"
            );
        }
    }
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
