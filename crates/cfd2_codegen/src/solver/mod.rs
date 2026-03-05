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
    pub use cfd2_ir::flux::PrimitiveExpr;
}
