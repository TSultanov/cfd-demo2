use super::{LimiterSpec, VANLEER_EPS};

/// Minimal expression-builder interface for reconstruction formulas that can target both
/// codegen-time IR (`FaceScalarExpr` / `FaceVec2Expr`) and WGSL AST expressions.
pub trait ReconstructionBuilder {
    type Scalar: Clone;
    type Vec2: Clone;

    fn lit(v: f32) -> Self::Scalar;

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn sub(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn div(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    fn abs(a: Self::Scalar) -> Self::Scalar;
    fn min(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn max(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    fn vec2_sub(a: Self::Vec2, b: Self::Vec2) -> Self::Vec2;
    fn vec2_dot(a: Self::Vec2, b: Self::Vec2) -> Self::Scalar;
}

pub fn minmod_delta_limited<B: ReconstructionBuilder>(
    phi_cell: B::Scalar,
    phi_other: B::Scalar,
    delta: B::Scalar,
) -> B::Scalar {
    let diff = B::sub(phi_other, phi_cell.clone());
    let min_diff = B::min(diff.clone(), B::lit(0.0));
    let max_diff = B::max(diff, B::lit(0.0));
    B::min(B::max(delta, min_diff), max_diff)
}

pub fn vanleer_delta_limited<B: ReconstructionBuilder>(diff: B::Scalar, delta: B::Scalar) -> B::Scalar {
    // VanLeer-style slope scaling:
    // - Always scale by |diff| / max(|diff|, |delta| + eps)
    // - Additionally guard against opposite-signed slopes (diff * delta <= 0) to avoid
    //   introducing new extrema.
    //
    // The opposite-slope sign-guard is implemented without comparisons/conditionals:
    //   sign_guard = max(p, 0) / max(abs(p), eps), where p = diff * delta.
    // This is 0 for p < 0 and ~1 for p >> eps.
    let abs_diff = B::abs(diff.clone());
    let abs_delta = B::abs(delta.clone());

    let denom = B::max(abs_diff.clone(), B::add(abs_delta, B::lit(VANLEER_EPS)));
    let scale = B::div(abs_diff, denom);

    let p = B::mul(diff, delta.clone());
    let sign_num = B::max(p.clone(), B::lit(0.0));
    let sign_denom = B::max(B::abs(p), B::lit(VANLEER_EPS));
    let sign_guard = B::div(sign_num, sign_denom);

    let delta_scaled = B::mul(delta, scale);
    B::mul(delta_scaled, sign_guard)
}

pub fn limit_delta<B: ReconstructionBuilder>(
    limiter: LimiterSpec,
    phi_cell: B::Scalar,
    phi_other: B::Scalar,
    delta: B::Scalar,
) -> B::Scalar {
    match limiter {
        LimiterSpec::None => delta,
        LimiterSpec::MinMod => minmod_delta_limited::<B>(phi_cell, phi_other, delta),
        LimiterSpec::VanLeer => {
            let diff = B::sub(phi_other, phi_cell);
            vanleer_delta_limited::<B>(diff, delta)
        }
    }
}

pub fn limited_linear_face_value<B: ReconstructionBuilder>(
    phi_cell: B::Scalar,
    phi_other: B::Scalar,
    grad_cell: B::Vec2,
    cell_to_face: B::Vec2,
    limiter: LimiterSpec,
) -> B::Scalar {
    let delta = B::vec2_dot(grad_cell, cell_to_face);
    let delta_limited = limit_delta::<B>(limiter, phi_cell.clone(), phi_other, delta);
    B::add(phi_cell, delta_limited)
}

pub fn quick_face_value<B: ReconstructionBuilder>(
    phi_cell: B::Scalar,
    phi_other: B::Scalar,
    grad_cell: B::Vec2,
    other_center_minus_center: B::Vec2,
    limiter: LimiterSpec,
) -> B::Scalar {
    let quick = {
        let term0 = B::mul(phi_cell.clone(), B::lit(0.625));
        let term1 = B::mul(phi_other.clone(), B::lit(0.375));
        let term2 = B::mul(
            B::vec2_dot(grad_cell, other_center_minus_center),
            B::lit(0.125),
        );
        B::add(B::add(term0, term1), term2)
    };
    let delta = B::sub(quick, phi_cell.clone());
    let delta_limited = limit_delta::<B>(limiter, phi_cell.clone(), phi_other, delta);
    B::add(phi_cell, delta_limited)
}

/// Builder implementation for the IR expression types.
pub struct FaceExprBuilder;

impl ReconstructionBuilder for FaceExprBuilder {
    type Scalar = super::FaceScalarExpr;
    type Vec2 = super::FaceVec2Expr;

    fn lit(v: f32) -> Self::Scalar {
        super::FaceScalarExpr::lit(v)
    }

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        super::FaceScalarExpr::Add(Box::new(a), Box::new(b))
    }

    fn sub(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        super::FaceScalarExpr::Sub(Box::new(a), Box::new(b))
    }

    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        super::FaceScalarExpr::Mul(Box::new(a), Box::new(b))
    }

    fn div(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        super::FaceScalarExpr::Div(Box::new(a), Box::new(b))
    }

    fn abs(a: Self::Scalar) -> Self::Scalar {
        super::FaceScalarExpr::Abs(Box::new(a))
    }

    fn min(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        super::FaceScalarExpr::Min(Box::new(a), Box::new(b))
    }

    fn max(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        super::FaceScalarExpr::Max(Box::new(a), Box::new(b))
    }

    fn vec2_sub(a: Self::Vec2, b: Self::Vec2) -> Self::Vec2 {
        super::FaceVec2Expr::Sub(Box::new(a), Box::new(b))
    }

    fn vec2_dot(a: Self::Vec2, b: Self::Vec2) -> Self::Scalar {
        super::FaceScalarExpr::Dot(Box::new(a), Box::new(b))
    }
}

