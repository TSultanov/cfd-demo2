use super::dsl as typed;
use super::dsl::EnumExpr;
use super::dsl::XY;
use super::wgsl_ast::{Expr, Stmt};
use super::wgsl_dsl as dsl;
use crate::solver::ir::reconstruction::{
    limited_linear_face_value, quick_face_value, ReconstructionBuilder,
};
use crate::solver::ir::LimiterSpec;
use crate::solver::scheme::Scheme;

#[derive(Debug, Clone)]
pub struct ScalarReconstruction {
    pub phi_upwind: Expr,
    pub phi_ho: Expr,
}

/// Geometry points needed for face reconstruction.
#[derive(Debug, Clone)]
pub struct GeometryPoints {
    pub center: Expr,
    pub other_center: Expr,
    pub face_center: Expr,
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

/// How the advection scheme reaches a reconstruction site.
pub enum SchemeSource {
    /// Term-declared scheme, baked at codegen time: only this variant's
    /// arithmetic is emitted (no runtime dispatch at all).
    Baked(Scheme),
    /// Runtime-selected scheme id expression (`constants.scheme`): emit an
    /// if/else-if chain so each face computes ONLY the active variant. The
    /// branch is uniform on the GPU (one scheme per dispatch) and predictable
    /// on the CPU — unlike the nested-`select` expression form, which
    /// computed all seven reconstructions per face and discarded six.
    Runtime(Expr),
}

/// Statement-emitting variant of [`scalar_reconstruction`]: returns the
/// declarations that compute `phi_ho` (specialized per [`SchemeSource`]) plus
/// the reconstruction exprs, where `phi_ho` is an ident referring to the
/// emitted local. The caller must splice the statements into the enclosing
/// block BEFORE any use of `phi_ho` (e.g. at the head of the interior-face
/// branch, so boundary faces skip the reconstruction entirely).
pub fn scalar_reconstruction_stmts(
    prefix: &str,
    scheme: SchemeSource,
    flux: Expr,
    phi_own: Expr,
    phi_neigh: Expr,
    grad_own: Expr,
    grad_neigh: Expr,
    geom: GeometryPoints,
) -> (Vec<Stmt>, ScalarReconstruction) {
    let xy = |point: &Expr| dsl::vec2_f32(point.clone().field("x"), point.clone().field("y"));

    let phi_upwind = dsl::select(phi_own.clone(), phi_neigh.clone(), flux.clone().lt(0.0));

    let grad_own_vec = dsl::vec2_f32_from_xy_fields(grad_own);
    let grad_neigh_vec = dsl::vec2_f32_from_xy_fields(grad_neigh);
    let r_own = xy(&geom.face_center) - xy(&geom.center);
    let r_neigh = xy(&geom.face_center) - xy(&geom.other_center);
    let d_pos = xy(&geom.other_center) - xy(&geom.center);
    let d_neg = xy(&geom.center) - xy(&geom.other_center);

    // Build ONE scheme variant's face-value expression (only invoked for the
    // variants that actually get emitted).
    let variant = |s: Scheme| -> Expr {
        let (limiter, quick) = match s {
            Scheme::Upwind => return phi_upwind.clone(),
            Scheme::SecondOrderUpwind => (LimiterSpec::None, false),
            Scheme::SecondOrderUpwindMinMod => (LimiterSpec::MinMod, false),
            Scheme::SecondOrderUpwindVanLeer => (LimiterSpec::VanLeer, false),
            Scheme::QUICK => (LimiterSpec::None, true),
            Scheme::QUICKMinMod => (LimiterSpec::MinMod, true),
            Scheme::QUICKVanLeer => (LimiterSpec::VanLeer, true),
            // Flux-family selectors (only the compressible flux modules
            // implement them): the matrix-path reconstruction falls back to
            // the vanLeer-limited MUSCL default.
            Scheme::Kep | Scheme::Slau2 => (LimiterSpec::VanLeer, false),
        };
        let (pos, neg) = if quick {
            (
                quick_face_value::<WgslExprBuilder>(
                    phi_own.clone(),
                    phi_neigh.clone(),
                    grad_own_vec.clone(),
                    d_pos.clone(),
                    limiter,
                ),
                quick_face_value::<WgslExprBuilder>(
                    phi_neigh.clone(),
                    phi_own.clone(),
                    grad_neigh_vec.clone(),
                    d_neg.clone(),
                    limiter,
                ),
            )
        } else {
            (
                limited_linear_face_value::<WgslExprBuilder>(
                    phi_own.clone(),
                    phi_neigh.clone(),
                    grad_own_vec.clone(),
                    r_own.clone(),
                    limiter,
                ),
                limited_linear_face_value::<WgslExprBuilder>(
                    phi_neigh.clone(),
                    phi_own.clone(),
                    grad_neigh_vec.clone(),
                    r_neigh.clone(),
                    limiter,
                ),
            )
        };
        dsl::select(neg, pos, flux.clone().gt(0.0))
    };

    let var_name = format!("{prefix}_phi_ho");
    let stmts = match scheme {
        SchemeSource::Baked(s) => vec![dsl::let_expr(&var_name, variant(s))],
        SchemeSource::Runtime(scheme_expr) => {
            let mut stmts = vec![dsl::var_expr(&var_name, phi_upwind.clone())];
            // Right-folded if/else-if chain over the non-upwind schemes; the
            // default (Upwind, id 0, and any unknown id) is the initializer.
            let arms = [
                Scheme::SecondOrderUpwind,
                Scheme::QUICK,
                Scheme::SecondOrderUpwindMinMod,
                Scheme::SecondOrderUpwindVanLeer,
                Scheme::QUICKMinMod,
                Scheme::QUICKVanLeer,
                // Flux-family selectors: matrix-path reconstruction uses the
                // vanLeer MUSCL fallback (see `variant`).
                Scheme::Kep,
                Scheme::Slau2,
            ];
            let mut chain: Option<Stmt> = None;
            for s in arms.iter().rev() {
                let then = dsl::block(vec![dsl::assign_expr(
                    Expr::ident(&var_name),
                    variant(*s),
                )]);
                let else_block = chain.take().map(|st| dsl::block(vec![st]));
                chain = Some(dsl::if_block_expr(
                    scheme_expr.clone().eq(s.gpu_id()),
                    then,
                    else_block,
                ));
            }
            if let Some(chain) = chain {
                stmts.push(chain);
            }
            stmts
        }
    };

    (
        stmts,
        ScalarReconstruction {
            phi_upwind,
            phi_ho: Expr::ident(&var_name),
        },
    )
}

pub fn scalar_reconstruction(
    scheme: EnumExpr<Scheme>,
    flux: Expr,
    phi_own: Expr,
    phi_neigh: Expr,
    grad_own: Expr,
    grad_neigh: Expr,
    geom: GeometryPoints,
) -> ScalarReconstruction {
    let xy = |point: &Expr| dsl::vec2_f32(point.clone().field("x"), point.clone().field("y"));

    let phi_upwind = dsl::select(phi_own.clone(), phi_neigh.clone(), flux.clone().lt(0.0));

    let grad_own_vec = dsl::vec2_f32_from_xy_fields(grad_own);
    let grad_neigh_vec = dsl::vec2_f32_from_xy_fields(grad_neigh);
    let r_own = xy(&geom.face_center) - xy(&geom.center);
    let r_neigh = xy(&geom.face_center) - xy(&geom.other_center);

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
    let phi_sou = dsl::select(sou_neg, sou_pos, flux.clone().gt(0.0));

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
    let phi_sou_mm = dsl::select(sou_neg_mm, sou_pos_mm, flux.clone().gt(0.0));

    let sou_pos_vl = limited_linear_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec.clone(),
        r_own,
        LimiterSpec::VanLeer,
    );
    let sou_neg_vl = limited_linear_face_value::<WgslExprBuilder>(
        phi_neigh.clone(),
        phi_own.clone(),
        grad_neigh_vec.clone(),
        r_neigh,
        LimiterSpec::VanLeer,
    );
    let phi_sou_vl = dsl::select(sou_neg_vl, sou_pos_vl, flux.clone().gt(0.0));

    let d_pos = xy(&geom.other_center) - xy(&geom.center);
    let d_neg = xy(&geom.center) - xy(&geom.other_center);

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
    let phi_quick = dsl::select(quick_neg, quick_pos, flux.clone().gt(0.0));

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
    let phi_quick_mm = dsl::select(quick_neg_mm, quick_pos_mm, flux.clone().gt(0.0));

    let quick_pos_vl = quick_face_value::<WgslExprBuilder>(
        phi_own.clone(),
        phi_neigh.clone(),
        grad_own_vec,
        d_pos,
        LimiterSpec::VanLeer,
    );
    let quick_neg_vl = quick_face_value::<WgslExprBuilder>(
        phi_neigh,
        phi_own,
        grad_neigh_vec,
        d_neg,
        LimiterSpec::VanLeer,
    );
    let phi_quick_vl = dsl::select(quick_neg_vl, quick_pos_vl, flux.clone().gt(0.0));

    let phi_ho = dsl::select(phi_upwind.clone(), phi_sou, scheme.eq(Scheme::SecondOrderUpwind));
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
            GeometryPoints {
                center: Expr::ident("center"),
                other_center: Expr::ident("other_center"),
                face_center: Expr::ident("face_center"),
            },
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
    geom: GeometryPoints,
) -> Vec2Reconstruction {
    let phi_own = typed::NamedVecExpr::<2, typed::AxisXY>::from_vec(phi_own);
    let phi_neigh = typed::NamedVecExpr::<2, typed::AxisXY>::from_vec(phi_neigh);

    let rec_x = scalar_reconstruction(
        scheme.clone(),
        flux.clone(),
        phi_own.at(XY::X),
        phi_neigh.at(XY::X),
        grad_own[XY::X.to_usize()].clone(),
        grad_neigh[XY::X.to_usize()].clone(),
        geom.clone(),
    );
    let rec_y = scalar_reconstruction(
        scheme,
        flux,
        phi_own.at(XY::Y),
        phi_neigh.at(XY::Y),
        grad_own[XY::Y.to_usize()].clone(),
        grad_neigh[XY::Y.to_usize()].clone(),
        geom,
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

    let stmts = vec![
        dsl::let_expr(&diff, phi_other - phi_cell.clone()),
        dsl::let_expr(&min_diff, dsl::min(Expr::ident(&diff), 0.0)),
        dsl::let_expr(&max_diff, dsl::max(Expr::ident(&diff), 0.0)),
        dsl::let_expr(
            &delta,
            dsl::dot(
                dsl::vec2_f32_from_xy_fields(grad_cell),
                dsl::vec2_f32(r_x, r_y),
            ),
        ),
        dsl::let_expr(
            &delta_limited,
            dsl::min(
                dsl::max(Expr::ident(&delta), Expr::ident(&min_diff)),
                Expr::ident(&max_diff),
            ),
        ),
        dsl::let_expr(&phi_face, phi_cell + Expr::ident(&delta_limited)),
    ];

    (stmts, Expr::ident(&phi_face))
}
