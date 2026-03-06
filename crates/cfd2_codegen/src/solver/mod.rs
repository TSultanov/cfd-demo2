pub mod codegen;

pub mod gpu {
    pub mod enums {
        pub use cfd2_ir::gpu_enums::*;
    }
}

pub mod ir {
    pub use cfd2_ir::kernel::*;
    pub mod ports {
        pub use cfd2_ir::ports::*;
    }
}

pub mod scheme {
    pub use cfd2_ir::scheme::*;
}

pub mod units {
    pub use cfd2_ir::units::*;
}

pub mod shared {
    // PrimitiveExpr has been unified into Expr. Re-export Expr for backwards compatibility.
    pub use cfd2_ir::ast::Expr;
}
