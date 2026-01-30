// Internal IR facade.
//
// Incremental boundary: codegen must depend on types from this module rather than reaching into
// `crate::solver::model::backend` directly.

#[allow(unused_imports)]
pub use crate::solver::model::backend::{
    expand_schemes, fvc, fvm, Coefficient, Discretization, Equation, EquationSystem, FieldKind,
    FieldRef, FluxRef, SchemeExpansion, SchemeRegistry, StateField, StateLayout, Term, TermKey,
    TermOp,
};

#[allow(unused_imports)]
pub use crate::solver::model::backend::ast::{
    surface_scalar, surface_vector3, vol_scalar, vol_vector, vol_vector3, CodegenError,
    UnitValidationError,
};

use crate::solver::scheme::Scheme;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EosSpec {
    IdealGas { gamma: f32 },
    Constant,
}

impl EosSpec {
    pub fn ideal_gas_gamma(&self) -> Option<f32> {
        match self {
            EosSpec::IdealGas { gamma } => Some(*gamma),
            EosSpec::Constant => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FluxComponent {
    pub name: String,
    pub offset: u32,
}

/// Named-component description of a packed face-flux buffer.
///
/// This intentionally does NOT reuse `FluxKind`. The purpose is to provide a stable,
/// model-derived mapping from semantic flux components to packed offsets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FluxLayout {
    pub stride: u32,
    pub components: Vec<FluxComponent>,
}

impl FluxLayout {
    pub fn from_system(system: &EquationSystem) -> Self {
        let mut components = Vec::new();
        let mut offset: u32 = 0;

        for eq in system.equations() {
            let field = eq.target();
            match field.kind() {
                FieldKind::Scalar => {
                    components.push(FluxComponent {
                        name: field.name().to_string(),
                        offset,
                    });
                    offset += 1;
                }
                FieldKind::Vector2 => {
                    // Match the existing coupled ordering: x then y.
                    components.push(FluxComponent {
                        name: format!("{}_x", field.name()),
                        offset,
                    });
                    offset += 1;
                    components.push(FluxComponent {
                        name: format!("{}_y", field.name()),
                        offset,
                    });
                    offset += 1;
                }
                FieldKind::Vector3 => {
                    // 3D ordering: x, y, z.
                    components.push(FluxComponent {
                        name: format!("{}_x", field.name()),
                        offset,
                    });
                    offset += 1;
                    components.push(FluxComponent {
                        name: format!("{}_y", field.name()),
                        offset,
                    });
                    offset += 1;
                    components.push(FluxComponent {
                        name: format!("{}_z", field.name()),
                        offset,
                    });
                    offset += 1;
                }
            }
        }

        FluxLayout {
            stride: offset,
            components,
        }
    }

    pub fn offset_for(&self, name: &str) -> Option<u32> {
        self.components
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.offset)
    }

    pub fn offset_for_field_component(&self, field: FieldRef, component: u32) -> Option<u32> {
        match field.kind() {
            FieldKind::Scalar => {
                if component == 0 {
                    self.offset_for(field.name())
                } else {
                    None
                }
            }
            FieldKind::Vector2 => {
                let suffix = match component {
                    0 => "x",
                    1 => "y",
                    _ => return None,
                };
                self.offset_for(&format!("{}_{}", field.name(), suffix))
            }
            FieldKind::Vector3 => {
                let suffix = match component {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => return None,
                };
                self.offset_for(&format!("{}_{}", field.name(), suffix))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceSide {
    Owner,
    Neighbor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceScalarBuiltin {
    Area,
    Dist,
    Lambda,
    LambdaOther,
    /// 1.0 for boundary faces, 0.0 for interior faces.
    IsBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LowMachParam {
    /// Low-Mach preconditioning model selector (cast from `u32` in codegen).
    Model,
    ThetaFloor,
    PressureCouplingAlpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceVec2Builtin {
    Normal,
    /// Vector from the given side's cell center to the face center.
    ///
    /// This is geometry-only (PDE-agnostic) and enables IR-driven reconstruction.
    CellToFace {
        side: FaceSide,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FaceVec2Expr {
    Builtin(FaceVec2Builtin),
    Vec2(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    StateVec2 {
        side: FaceSide,
        field: String,
    },
    /// Read a vec2 from the state buffer at the given side's cell index, without applying
    /// boundary conditions (raw cell-centered values).
    CellStateVec2 {
        side: FaceSide,
        field: String,
    },
    Add(Box<FaceVec2Expr>, Box<FaceVec2Expr>),
    Sub(Box<FaceVec2Expr>, Box<FaceVec2Expr>),
    Neg(Box<FaceVec2Expr>),
    MulScalar(Box<FaceVec2Expr>, Box<FaceScalarExpr>),
    Lerp(Box<FaceVec2Expr>, Box<FaceVec2Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FaceScalarExpr {
    Literal(f32),
    Builtin(FaceScalarBuiltin),
    /// Read a scalar from the shared `Constants` uniform buffer (e.g. `dt`, `density`).
    Constant {
        name: String,
    },
    /// Read a scalar from the optional `LowMachParams` uniform buffer.
    ///
    /// Note: `LowMachParam::Model` is stored as `u32` but is exposed to IR as an `f32`
    /// (codegen inserts the cast).
    LowMachParam(LowMachParam),
    State {
        side: FaceSide,
        name: String,
    },
    Primitive {
        side: FaceSide,
        name: String,
    },
    Add(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    Sub(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    Mul(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    Div(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    Neg(Box<FaceScalarExpr>),
    Abs(Box<FaceScalarExpr>),
    Sqrt(Box<FaceScalarExpr>),
    Max(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    Min(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    Lerp(Box<FaceScalarExpr>, Box<FaceScalarExpr>),
    Dot(Box<FaceVec2Expr>, Box<FaceVec2Expr>),
}

impl FaceScalarExpr {
    pub fn lit(v: f32) -> Self {
        FaceScalarExpr::Literal(v)
    }

    pub fn area() -> Self {
        FaceScalarExpr::Builtin(FaceScalarBuiltin::Area)
    }

    pub fn dist() -> Self {
        FaceScalarExpr::Builtin(FaceScalarBuiltin::Dist)
    }

    pub fn lambda() -> Self {
        FaceScalarExpr::Builtin(FaceScalarBuiltin::Lambda)
    }

    pub fn lambda_other() -> Self {
        FaceScalarExpr::Builtin(FaceScalarBuiltin::LambdaOther)
    }

    pub fn is_boundary() -> Self {
        FaceScalarExpr::Builtin(FaceScalarBuiltin::IsBoundary)
    }

    pub fn state(side: FaceSide, name: impl Into<String>) -> Self {
        FaceScalarExpr::State {
            side,
            name: name.into(),
        }
    }

    pub fn constant(name: impl Into<String>) -> Self {
        FaceScalarExpr::Constant { name: name.into() }
    }

    pub fn low_mach_model() -> Self {
        FaceScalarExpr::LowMachParam(LowMachParam::Model)
    }

    pub fn low_mach_theta_floor() -> Self {
        FaceScalarExpr::LowMachParam(LowMachParam::ThetaFloor)
    }

    pub fn low_mach_pressure_coupling_alpha() -> Self {
        FaceScalarExpr::LowMachParam(LowMachParam::PressureCouplingAlpha)
    }

    pub fn primitive(side: FaceSide, name: impl Into<String>) -> Self {
        FaceScalarExpr::Primitive {
            side,
            name: name.into(),
        }
    }
}

impl FaceVec2Expr {
    pub fn normal() -> Self {
        FaceVec2Expr::Builtin(FaceVec2Builtin::Normal)
    }

    pub fn cell_to_face(side: FaceSide) -> Self {
        FaceVec2Expr::Builtin(FaceVec2Builtin::CellToFace { side })
    }

    pub fn vec2(x: FaceScalarExpr, y: FaceScalarExpr) -> Self {
        FaceVec2Expr::Vec2(Box::new(x), Box::new(y))
    }

    pub fn state_vec2(side: FaceSide, field: impl Into<String>) -> Self {
        FaceVec2Expr::StateVec2 {
            side,
            field: field.into(),
        }
    }

    pub fn cell_state_vec2(side: FaceSide, field: impl Into<String>) -> Self {
        FaceVec2Expr::CellStateVec2 {
            side,
            field: field.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FluxModuleKernelSpec {
    /// Compute a scalar face flux and replicate it into all coupled unknown-component slots.
    ScalarReplicated { phi: FaceScalarExpr },

    /// Compute one scalar face flux per coupled unknown-component slot.
    ///
    /// Each entry is written directly into the packed `fluxes[face, component]` table.
    /// Expressions are expected to already include any geometric factors (e.g. multiply by
    /// `area` when returning an integrated face flux).
    ScalarPerComponent {
        /// Coupled unknown-component names in packed order.
        components: Vec<String>,
        /// Integrated face flux per component (same length as `components`).
        flux: Vec<FaceScalarExpr>,
    },

    /// Central-upwind (Kurganov–Tadmor-style) numerical flux for a conservation law.
    ///
    /// All fluxes are expressed in the *face-normal* direction and are written as integrated
    /// face fluxes (i.e., multiplied by face `area` at the end of the kernel).
    CentralUpwind {
        /// Reconstruction scheme for left/right states.
        ///
        /// Use `Scheme::Upwind` for first-order behavior.
        reconstruction: Scheme,
        /// Coupled unknown-component names in packed order.
        components: Vec<String>,
        /// Reconstructed left state U_L (one scalar per component).
        u_left: Vec<FaceScalarExpr>,
        /// Reconstructed right state U_R (one scalar per component).
        u_right: Vec<FaceScalarExpr>,
        /// Physical flux F(U_L)·n (one scalar per component).
        flux_left: Vec<FaceScalarExpr>,
        /// Physical flux F(U_R)·n (one scalar per component).
        flux_right: Vec<FaceScalarExpr>,
        /// Upper wave speed bound (>= 0).
        a_plus: FaceScalarExpr,
        /// Lower wave speed bound (<= 0).
        a_minus: FaceScalarExpr,
    },
}

/// Slope limiter for high-order reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterSpec {
    /// No limiting (can be unstable).
    None,
    /// MinMod limiter.
    MinMod,
    /// Van Leer limiter.
    VanLeer,
}

/// Shared epsilon used by VanLeer-style limiter guards.
///
/// This constant is a *cross-path drift guard*: both unified_assembly reconstruction and
/// flux-module MUSCL reconstruction should use this value (via a shared constant) rather than
/// embedding separate numeric literals.
pub const VANLEER_EPS: f32 = 1e-8;

pub mod ports;
pub mod reconstruction;

// Re-export port types for convenience
pub use ports::{BufferAccess, BufferSpec, FieldSpec, ParamSpec, PortFieldKind, PortManifest};

impl Default for LimiterSpec {
    fn default() -> Self {
        LimiterSpec::None
    }
}

// Intentionally no test fixtures here: `cfd2_ir` must not depend on model definitions.
