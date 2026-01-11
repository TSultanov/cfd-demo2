// Internal IR facade.
//
// Incremental boundary: codegen must depend on types from this module rather than reaching into
// `crate::solver::model::backend` directly.

pub(crate) use crate::solver::model::backend::{
    fvc, fvm, expand_schemes, Coefficient, Discretization, Equation, EquationSystem, FieldKind,
    FieldRef, FluxRef, SchemeExpansion, SchemeRegistry, StateField, StateLayout, Term, TermKey,
    TermOp,
};

pub(crate) use crate::solver::model::backend::ast::{
    surface_scalar, vol_scalar, vol_vector, CodegenError, UnitValidationError,
};
