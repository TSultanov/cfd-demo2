pub mod accumulators;
pub mod enums;
pub mod expr;
pub mod matrix;
pub mod tensor;
pub mod types;
pub mod units;

pub use accumulators::CoupledAccumulators;
pub use enums::{EnumExpr, WgslEnum};
pub use expr::{DslError, DynExpr, TypedExpr, TypedSqrt};
pub use matrix::{
    BlockCsrMatrix, BlockCsrSoaEntry, BlockCsrSoaMatrix, BlockShape, CsrMatrix, CsrPattern,
    NamedBlockCsrSoaEntry, NamedBlockCsrSoaMatrix,
};
pub use tensor::{
    block_col, block_row, dispatch_by_coupled_stride, Axis, AxisCons, AxisXY, BlockCol, BlockRow,
    CompressibleAxis2D, Cons, CoupledAxis, DispatchByStride, IncompressibleAxis2D,
    IncompressibleAxis3D, MatExpr, NamedMatExpr, NamedVecExpr, ScalarAxis, VecExpr, XY,
};
pub use types::{DslType, ScalarType, Shape};
pub use units::UnitDim;
