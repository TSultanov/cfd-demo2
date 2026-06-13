// Declarative boundary-value expressions.
//
// A `BoundaryExpr` describes, as math, the value a boundary-face table entry
// (`bc_value`) takes for one unknown component: a function of interior
// (owner-cell) state, other prescribed boundary values, uniform model
// params, and literals. A generic Preparation-phase kernel (the `bc_expr`
// module) evaluates these per boundary face every outer iteration,
// replacing hand-written runtime-BC kernels; the same expressions can be
// evaluated host-side in f64 for seeding helpers.
//
// Compilation contexts: like `algebraic.rs`/`typed_ast.rs`, this file is
// compiled both as `cfd2_ir::equation::boundary` and (via include!) inside
// build.rs as a sibling module. Paths must be `super::ast` /
// `super::algebraic` / `crate::dimensions`; no inner (`//!`) doc comments;
// no inherent impls on `ast` types.
//
// # Evaluation semantics (the contract the generic kernel implements)
//
// - `interior(f, c)` reads the owner cell of the face from the CURRENT
//   state (the state the outer iteration linearizes around).
// - `bc(f, c)` reads the boundary table AS IT WAS WHEN THE KERNEL STARTED
//   (snapshot semantics): all `bc(..)` reads are hoisted before any writes,
//   so expression sets are order-independent and reads always observe
//   host-prescribed values, never values written by sibling expressions in
//   the same refresh.
// - `param(p)` reads a uniform model parameter (`constants.<name>`).
// - Literals are unit-wildcards: unit validation treats them as compatible
//   with anything (epsilon floors like `max(rho, 1e-6)` would otherwise
//   need unit-annotated constants everywhere).

use super::algebraic::ParamRef;
use super::ast::FieldRef;
use crate::dimensions::UnitDim;

/// A scalar boundary-value expression for one unknown component.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryExpr {
    /// Literal constant (unit-wildcard for validation purposes).
    Lit(f64),
    /// Uniform model parameter (`constants.<name>`).
    Param(ParamRef),
    /// Interior (owner-cell) value of a state field component.
    Interior { field: FieldRef, component: u32 },
    /// Boundary-table value of an unknown component, snapshot at kernel
    /// start (host-prescribed values; never sibling writes).
    BcValue { field: FieldRef, component: u32 },
    Add(Box<BoundaryExpr>, Box<BoundaryExpr>),
    Sub(Box<BoundaryExpr>, Box<BoundaryExpr>),
    Mul(Box<BoundaryExpr>, Box<BoundaryExpr>),
    Div(Box<BoundaryExpr>, Box<BoundaryExpr>),
    Neg(Box<BoundaryExpr>),
    /// Square root. Unit-wildcard (the LODI/characteristic closures mix it
    /// into otherwise dimensioned trees; see `Normal`).
    Sqrt(Box<BoundaryExpr>),
    /// A component of the face OUTWARD normal (unit vector). Unit-wildcard.
    /// Per-face geometric atom (the only one): lets boundary closures form
    /// the normal velocity `u·n` for characteristic conditions. Has no
    /// host-evaluation form (seeding helpers must not declare normal-bearing
    /// expressions; the GPU `bc_expr` kernel reads `face_normals`).
    Normal { component: u32 },
    Max(Box<BoundaryExpr>, Box<BoundaryExpr>),
    Min(Box<BoundaryExpr>, Box<BoundaryExpr>),
    /// `if lhs > rhs { on_true } else { on_false }`.
    SelectGt {
        lhs: Box<BoundaryExpr>,
        rhs: Box<BoundaryExpr>,
        on_true: Box<BoundaryExpr>,
        on_false: Box<BoundaryExpr>,
    },
}

impl BoundaryExpr {
    pub fn lit(value: f64) -> Self {
        BoundaryExpr::Lit(value)
    }

    pub fn param(p: ParamRef) -> Self {
        BoundaryExpr::Param(p)
    }

    /// Interior (owner-cell) value of a scalar field.
    pub fn interior(field: FieldRef) -> Self {
        BoundaryExpr::Interior {
            field,
            component: 0,
        }
    }

    /// Interior (owner-cell) value of a field component.
    pub fn interior_comp(field: FieldRef, component: u32) -> Self {
        BoundaryExpr::Interior { field, component }
    }

    /// Prescribed boundary value of a scalar unknown.
    pub fn bc(field: FieldRef) -> Self {
        BoundaryExpr::BcValue {
            field,
            component: 0,
        }
    }

    /// Prescribed boundary value of an unknown component.
    pub fn bc_comp(field: FieldRef, component: u32) -> Self {
        BoundaryExpr::BcValue { field, component }
    }

    /// Square root of this expression.
    pub fn sqrt(self) -> Self {
        BoundaryExpr::Sqrt(Box::new(self))
    }

    /// A component (0 = x, 1 = y) of the face outward normal.
    pub fn normal(component: u32) -> Self {
        BoundaryExpr::Normal { component }
    }

    pub fn max(self, other: BoundaryExpr) -> Self {
        BoundaryExpr::Max(Box::new(self), Box::new(other))
    }

    pub fn min(self, other: BoundaryExpr) -> Self {
        BoundaryExpr::Min(Box::new(self), Box::new(other))
    }

    /// `if self > rhs { on_true } else { on_false }`.
    pub fn select_gt(self, rhs: BoundaryExpr, on_true: BoundaryExpr, on_false: BoundaryExpr) -> Self {
        BoundaryExpr::SelectGt {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            on_true: Box::new(on_true),
            on_false: Box::new(on_false),
        }
    }

    /// Unit of the expression, when determinable.
    ///
    /// Literals are wildcards (`None`); binary ops require operand
    /// agreement where applicable and otherwise propagate the known side.
    /// Returns `Err` on a definite unit conflict.
    pub fn unit(&self) -> Result<Option<UnitDim>, String> {
        match self {
            BoundaryExpr::Lit(_) => Ok(None),
            BoundaryExpr::Param(p) => Ok(Some(p.unit())),
            BoundaryExpr::Interior { field, .. } | BoundaryExpr::BcValue { field, .. } => {
                Ok(Some(field.unit()))
            }
            BoundaryExpr::Add(a, b)
            | BoundaryExpr::Sub(a, b)
            | BoundaryExpr::Max(a, b)
            | BoundaryExpr::Min(a, b) => merge_same_unit(a.unit()?, b.unit()?),
            BoundaryExpr::Mul(a, b) => match (a.unit()?, b.unit()?) {
                (Some(ua), Some(ub)) => Ok(Some(ua * ub)),
                // A wildcard factor makes the product unit unknown.
                _ => Ok(None),
            },
            BoundaryExpr::Div(a, b) => match (a.unit()?, b.unit()?) {
                (Some(ua), Some(ub)) => Ok(Some(ua / ub)),
                _ => Ok(None),
            },
            BoundaryExpr::Neg(a) => a.unit(),
            // Wildcards: these atoms appear inside characteristic closures
            // whose units we do not track through the sqrt/normal algebra
            // (the enclosing Add against a dimensioned term re-pins the unit).
            BoundaryExpr::Sqrt(_) | BoundaryExpr::Normal { .. } => Ok(None),
            BoundaryExpr::SelectGt {
                lhs,
                rhs,
                on_true,
                on_false,
            } => {
                merge_same_unit(lhs.unit()?, rhs.unit()?)?;
                merge_same_unit(on_true.unit()?, on_false.unit()?)
            }
        }
    }

    /// Walk the tree, invoking `visit` on every node (preorder).
    pub fn visit<'a>(&'a self, visit: &mut impl FnMut(&'a BoundaryExpr)) {
        visit(self);
        match self {
            BoundaryExpr::Lit(_)
            | BoundaryExpr::Param(_)
            | BoundaryExpr::Interior { .. }
            | BoundaryExpr::BcValue { .. } => {}
            BoundaryExpr::Add(a, b)
            | BoundaryExpr::Sub(a, b)
            | BoundaryExpr::Mul(a, b)
            | BoundaryExpr::Div(a, b)
            | BoundaryExpr::Max(a, b)
            | BoundaryExpr::Min(a, b) => {
                a.visit(visit);
                b.visit(visit);
            }
            BoundaryExpr::Neg(a) | BoundaryExpr::Sqrt(a) => a.visit(visit),
            BoundaryExpr::Normal { .. } => {}
            BoundaryExpr::SelectGt {
                lhs,
                rhs,
                on_true,
                on_false,
            } => {
                lhs.visit(visit);
                rhs.visit(visit);
                on_true.visit(visit);
                on_false.visit(visit);
            }
        }
    }
}

fn merge_same_unit(
    a: Option<UnitDim>,
    b: Option<UnitDim>,
) -> Result<Option<UnitDim>, String> {
    match (a, b) {
        (Some(ua), Some(ub)) => {
            if ua != ub {
                Err(format!(
                    "boundary expression unit mismatch: {} vs {}",
                    ua, ub
                ))
            } else {
                Ok(Some(ua))
            }
        }
        (Some(u), None) | (None, Some(u)) => Ok(Some(u)),
        (None, None) => Ok(None),
    }
}

impl std::ops::Add for BoundaryExpr {
    type Output = BoundaryExpr;
    fn add(self, rhs: BoundaryExpr) -> BoundaryExpr {
        BoundaryExpr::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Sub for BoundaryExpr {
    type Output = BoundaryExpr;
    fn sub(self, rhs: BoundaryExpr) -> BoundaryExpr {
        BoundaryExpr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for BoundaryExpr {
    type Output = BoundaryExpr;
    fn mul(self, rhs: BoundaryExpr) -> BoundaryExpr {
        BoundaryExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Div for BoundaryExpr {
    type Output = BoundaryExpr;
    fn div(self, rhs: BoundaryExpr) -> BoundaryExpr {
        BoundaryExpr::Div(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Neg for BoundaryExpr {
    type Output = BoundaryExpr;
    fn neg(self) -> BoundaryExpr {
        BoundaryExpr::Neg(Box::new(self))
    }
}

/// Host-side f64 evaluation of a boundary expression.
///
/// `interior` and `bc` resolve `Interior`/`BcValue` atoms; `param` resolves
/// uniform parameters. Use [`eval_boundary_expr_f32`] when the result must
/// match GPU arithmetic bit-for-bit.
pub fn eval_boundary_expr(
    expr: &BoundaryExpr,
    interior: &dyn Fn(&FieldRef, u32) -> Result<f64, String>,
    bc: &dyn Fn(&FieldRef, u32) -> Result<f64, String>,
    param: &dyn Fn(&ParamRef) -> Result<f64, String>,
) -> Result<f64, String> {
    let eval = |e: &BoundaryExpr| eval_boundary_expr(e, interior, bc, param);
    Ok(match expr {
        BoundaryExpr::Lit(v) => *v,
        BoundaryExpr::Param(p) => param(p)?,
        BoundaryExpr::Interior { field, component } => interior(field, *component)?,
        BoundaryExpr::BcValue { field, component } => bc(field, *component)?,
        BoundaryExpr::Add(a, b) => eval(a)? + eval(b)?,
        BoundaryExpr::Sub(a, b) => eval(a)? - eval(b)?,
        BoundaryExpr::Mul(a, b) => eval(a)? * eval(b)?,
        BoundaryExpr::Div(a, b) => eval(a)? / eval(b)?,
        BoundaryExpr::Neg(a) => -eval(a)?,
        BoundaryExpr::Sqrt(a) => eval(a)?.sqrt(),
        BoundaryExpr::Normal { component } => {
            return Err(format!(
                "boundary expression: normal({component}) has no host evaluation \
                 (declare normal-bearing closures only on the GPU bc_expr path)"
            ))
        }
        BoundaryExpr::Max(a, b) => eval(a)?.max(eval(b)?),
        BoundaryExpr::Min(a, b) => eval(a)?.min(eval(b)?),
        BoundaryExpr::SelectGt {
            lhs,
            rhs,
            on_true,
            on_false,
        } => {
            if eval(lhs)? > eval(rhs)? {
                eval(on_true)?
            } else {
                eval(on_false)?
            }
        }
    })
}

/// Host-side f32 evaluation of a boundary expression, rounding after every
/// operation exactly like the generated WGSL kernel does. BC seeding
/// helpers use this so host-seeded values are bit-identical to what the
/// GPU refresh kernel computes for the same inputs.
pub fn eval_boundary_expr_f32(
    expr: &BoundaryExpr,
    interior: &dyn Fn(&FieldRef, u32) -> Result<f32, String>,
    bc: &dyn Fn(&FieldRef, u32) -> Result<f32, String>,
    param: &dyn Fn(&ParamRef) -> Result<f32, String>,
) -> Result<f32, String> {
    let eval = |e: &BoundaryExpr| eval_boundary_expr_f32(e, interior, bc, param);
    Ok(match expr {
        BoundaryExpr::Lit(v) => *v as f32,
        BoundaryExpr::Param(p) => param(p)?,
        BoundaryExpr::Interior { field, component } => interior(field, *component)?,
        BoundaryExpr::BcValue { field, component } => bc(field, *component)?,
        BoundaryExpr::Add(a, b) => eval(a)? + eval(b)?,
        BoundaryExpr::Sub(a, b) => eval(a)? - eval(b)?,
        BoundaryExpr::Mul(a, b) => eval(a)? * eval(b)?,
        BoundaryExpr::Div(a, b) => eval(a)? / eval(b)?,
        BoundaryExpr::Neg(a) => -eval(a)?,
        BoundaryExpr::Sqrt(a) => eval(a)?.sqrt(),
        BoundaryExpr::Normal { component } => {
            return Err(format!(
                "boundary expression: normal({component}) has no host evaluation \
                 (declare normal-bearing closures only on the GPU bc_expr path)"
            ))
        }
        BoundaryExpr::Max(a, b) => eval(a)?.max(eval(b)?),
        BoundaryExpr::Min(a, b) => eval(a)?.min(eval(b)?),
        BoundaryExpr::SelectGt {
            lhs,
            rhs,
            on_true,
            on_false,
        } => {
            if eval(lhs)? > eval(rhs)? {
                eval(on_true)?
            } else {
                eval(on_false)?
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::{Density, Pressure, UnitDimension};
    use crate::equation::ast::{vol_scalar_dim, vol_vector_dim};

    #[test]
    fn unit_propagates_through_ops_and_wildcards() {
        let rho = vol_scalar_dim::<Density>("rho");
        let p = vol_scalar_dim::<Pressure>("p");

        // max(interior(rho), 1e-6): literal is a wildcard, unit = Density.
        let floored = BoundaryExpr::interior(rho).max(BoundaryExpr::lit(1e-6));
        assert_eq!(floored.unit().unwrap(), Some(Density::UNIT));

        // p / rho has a definite combined unit.
        let ratio = BoundaryExpr::interior(p) / BoundaryExpr::interior(rho);
        assert_eq!(
            ratio.unit().unwrap(),
            Some(Pressure::UNIT / Density::UNIT)
        );

        // Adding incompatible units is a definite conflict.
        let bad = BoundaryExpr::interior(p) + BoundaryExpr::interior(rho);
        assert!(bad.unit().is_err());
    }

    #[test]
    fn host_eval_matches_expression_semantics() {
        let rho = vol_scalar_dim::<Density>("rho");
        let u = vol_vector_dim::<crate::dimensions::Velocity>("u");
        let gm1 = ParamRef::new("eos_gm1", crate::dimensions::Dimensionless::UNIT);

        // ke = 0.5 * bc(rho) * (bc(u_x)^2 + bc(u_y)^2)
        let ke = BoundaryExpr::lit(0.5)
            * BoundaryExpr::bc(rho)
            * (BoundaryExpr::bc_comp(u, 0) * BoundaryExpr::bc_comp(u, 0)
                + BoundaryExpr::bc_comp(u, 1) * BoundaryExpr::bc_comp(u, 1));
        // rho_e = select(ke, p/gm1 + ke, gm1 > 0) with p = interior pressure
        let p = vol_scalar_dim::<Pressure>("p");
        let rho_e = BoundaryExpr::param(gm1).select_gt(
            BoundaryExpr::lit(0.0),
            BoundaryExpr::interior(p) / BoundaryExpr::param(gm1) + ke.clone(),
            ke,
        );

        let interior = |f: &FieldRef, _c: u32| -> Result<f64, String> {
            match f.name() {
                "p" => Ok(2.0),
                other => Err(format!("unexpected interior read {other}")),
            }
        };
        let bc = |f: &FieldRef, c: u32| -> Result<f64, String> {
            match (f.name(), c) {
                ("rho", 0) => Ok(1.0),
                ("u", 0) => Ok(3.0),
                ("u", 1) => Ok(4.0),
                other => Err(format!("unexpected bc read {other:?}")),
            }
        };
        let param = |p: &ParamRef| -> Result<f64, String> {
            match p.name() {
                "eos_gm1" => Ok(0.4),
                other => Err(format!("unexpected param read {other}")),
            }
        };

        let ke_expected = 0.5 * 1.0 * (9.0 + 16.0);
        let value = eval_boundary_expr(&rho_e, &interior, &bc, &param).unwrap();
        assert_eq!(value, 2.0 / 0.4 + ke_expected);
    }

    #[test]
    fn sqrt_and_normal_grammar() {
        use crate::dimensions::Velocity;

        let interior = |_: &FieldRef, _: u32| -> Result<f64, String> { Ok(0.0) };
        let bc = |_: &FieldRef, _: u32| -> Result<f64, String> { Ok(0.0) };
        let param = |_: &ParamRef| -> Result<f64, String> { Ok(0.0) };

        // Sqrt evaluates host-side.
        let four = BoundaryExpr::lit(4.0).sqrt();
        assert_eq!(eval_boundary_expr(&four, &interior, &bc, &param).unwrap(), 2.0);

        // Normal is a unit-wildcard and has NO host evaluation (soft error):
        // characteristic closures live only on the GPU bc_expr path.
        let n = BoundaryExpr::normal(0);
        assert_eq!(n.unit().unwrap(), None);
        assert_eq!(BoundaryExpr::lit(1.0).sqrt().unit().unwrap(), None);
        assert!(eval_boundary_expr(&n, &interior, &bc, &param).is_err());

        // A wildcard (normal/sqrt) term added against a dimensioned interior
        // value re-pins the unit (velocity + wildcard = velocity).
        let u = vol_vector_dim::<Velocity>("u");
        let expr = BoundaryExpr::interior_comp(u, 0)
            + BoundaryExpr::lit(1.0).sqrt() * BoundaryExpr::normal(0);
        assert_eq!(expr.unit().unwrap(), Some(Velocity::UNIT));
    }
}
