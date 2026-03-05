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

// Re-export typed_ast module itself
pub mod typed_ast {
    pub use cfd2_ir::equation::typed_ast::*;
}
