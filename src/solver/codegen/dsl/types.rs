use crate::solver::codegen::wgsl_ast::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F32,
    U32,
    I32,
    Bool,
}

impl ScalarType {
    pub fn to_wgsl(self) -> Type {
        match self {
            ScalarType::F32 => Type::F32,
            ScalarType::U32 => Type::U32,
            ScalarType::I32 => Type::I32,
            ScalarType::Bool => Type::Bool,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shape {
    Scalar,
    Vec(u8),
    Mat { rows: u8, cols: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DslType {
    pub scalar: ScalarType,
    pub shape: Shape,
}

impl DslType {
    pub const fn new(scalar: ScalarType, shape: Shape) -> Self {
        Self { scalar, shape }
    }

    pub const fn f32() -> Self {
        Self::new(ScalarType::F32, Shape::Scalar)
    }

    pub const fn vec2_f32() -> Self {
        Self::new(ScalarType::F32, Shape::Vec(2))
    }

    pub const fn vec3_f32() -> Self {
        Self::new(ScalarType::F32, Shape::Vec(3))
    }

    pub const fn mat_f32(rows: u8, cols: u8) -> Self {
        Self::new(ScalarType::F32, Shape::Mat { rows, cols })
    }

    pub fn to_wgsl(self) -> Type {
        match self.shape {
            Shape::Scalar => self.scalar.to_wgsl(),
            Shape::Vec(2) => Type::Vec2(Box::new(self.scalar.to_wgsl())),
            Shape::Vec(3) => Type::Vec3(Box::new(self.scalar.to_wgsl())),
            Shape::Vec(4) => Type::Vec4(Box::new(self.scalar.to_wgsl())),
            Shape::Vec(n) => Type::Custom(format!("vec{}<{}>", n, self.scalar.to_wgsl())),
            Shape::Mat { rows, cols } => {
                // WGSL uses column-major `matCxR<T>` (columns x rows).
                Type::Custom(format!("mat{}x{}<{}>", cols, rows, self.scalar.to_wgsl()))
            }
        }
    }
}
