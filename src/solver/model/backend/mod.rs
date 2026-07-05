pub use cfd2_ir::equation::*;

// Explicitly re-export typed_ast items (glob doesn't include these)
pub use cfd2_ir::equation::typed_ast::{
    typed_fvc, typed_fvm, Kind, Scalar, TypedCoeff, TypedEquation, TypedEquationSystem,
    TypedFieldRef, TypedFluxRef, TypedTerm, TypedTermSum, Vector2, Vector3,
};

pub mod ast {
    pub use cfd2_ir::equation::ast::*;
}

pub mod scheme {
    pub use cfd2_ir::equation::scheme::*;
}

pub mod scheme_expansion {
    pub use cfd2_ir::equation::scheme_expansion::*;
}

pub mod state_layout {
    pub use cfd2_ir::equation::state_layout::*;
}

pub mod typed_ast {
    pub use cfd2_ir::equation::typed_ast::*;
}

// Algebraic equations (lowered to coupled source rows); in the build.rs
// context this module is include!'d instead so its typed wrappers take the
// include!'d TypedFieldRef types.
pub mod algebraic {
    pub use cfd2_ir::equation::algebraic::*;
}

// Declarative boundary-value expressions (dual-context like algebraic).
pub mod boundary {
    pub use cfd2_ir::equation::boundary::*;
}
