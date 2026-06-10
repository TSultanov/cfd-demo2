pub mod algebraic;
pub mod ast;
pub mod boundary;
pub mod scheme;
pub mod scheme_expansion;
pub mod state_layout;
pub mod typed_ast;

pub use algebraic::{
    add_algebraic_equation, lower_algebraic_equation, typed_alg, AlgExpr, AlgebraicEquation,
    ParamRef, TypedAlgExpr, TypedParamRef,
};
pub use ast::{
    fvc, fvm, Coefficient, Discretization, Equation, EquationSystem, FieldKind, FieldRef, FluxRef,
    Term, TermOp,
};
pub use boundary::{eval_boundary_expr, BoundaryExpr};
pub use scheme::{SchemeRegistry, TermKey};
pub use scheme_expansion::{expand_schemes, expand_schemes_unchecked, SchemeExpansion};
pub use state_layout::{StateField, StateLayout};
pub use typed_ast::{
    typed_fvc, typed_fvm, Kind, Scalar, TypedCoeff, TypedEquation, TypedEquationSystem,
    TypedFieldRef, TypedFluxRef, TypedTerm, TypedTermSum, Vector2, Vector3,
};
