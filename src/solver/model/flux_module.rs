/// Pluggable flux computation module specification.
///
/// Models declare *which* flux method to use (KT, Rhie-Chow, etc.) without
/// implementing the WGSL generation. Codegen lowers these specs to kernels.

/// Flux computation module specification.
#[derive(Debug, Clone, PartialEq)]
pub enum FluxModuleSpec {
    /// A model-provided flux kernel specification expressed in IR terms.
    ///
    /// This is the PDE-agnostic boundary: codegen compiles this spec without hardcoded
    /// physics assumptions.
    Kernel {
        /// Optional gradients stage kernel (currently unused by built-in models).
        gradients: Option<crate::solver::ir::FluxModuleKernelSpec>,
        /// Face flux computation kernel.
        kernel: crate::solver::ir::FluxModuleKernelSpec,
    },
}

/// Reconstruction method for face values (used by flux modules).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionSpec {
    /// First-order upwind (cell-centered values)
    FirstOrder,

    /// MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws)
    Muscl {
        /// Slope limiter
        limiter: LimiterSpec,
    },
}

/// Slope limiter for high-order reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterSpec {
    /// Van Leer limiter
    VanLeer,

    /// MinMod limiter
    MinMod,

    /// No limiting (central difference, can be unstable)
    None,
}

impl Default for FluxModuleSpec {
    fn default() -> Self {
        FluxModuleSpec::Kernel {
            gradients: None,
            kernel: crate::solver::ir::FluxModuleKernelSpec::ScalarReplicated {
                phi: crate::solver::ir::FaceScalarExpr::lit(0.0),
            },
        }
    }
}
