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
        /// Optional gradients stage spec.
        ///
        /// Some flux modules (e.g. Rhie–Chow) require precomputed gradients of scalar
        /// unknowns (e.g. `grad_p`) that live in the state layout but are not solved-for
        /// directly.
        gradients: Option<FluxModuleGradientsSpec>,
        /// Face flux computation kernel.
        kernel: crate::solver::ir::FluxModuleKernelSpec,
    },

    /// A named flux scheme that is lowered to an IR kernel during model lowering.
    ///
    /// This keeps model definitions declarative (choose scheme + params) without embedding
    /// flux-formula construction logic in every model definition.
    Scheme {
        /// Optional gradients stage spec.
        gradients: Option<FluxModuleGradientsSpec>,
        /// Scheme selection + parameters.
        scheme: FluxSchemeSpec,
    },
}

/// Named flux schemes (solver-side, model/PDE-aware lowering).
#[derive(Debug, Clone, PartialEq)]
pub enum FluxSchemeSpec {
    /// Central-upwind (KT-style) Euler flux for an ideal gas.
    EulerCentralUpwind,
}

/// Gradients stage spec for flux modules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FluxModuleGradientsSpec {
    /// Compute gradients for all `grad_<name>` fields in the state layout where:
    /// - `<name>` exists as a scalar field in the same layout, and
    /// - `grad_<name>` is a `Vector2`.
    ///
    /// Boundary values are derived from the runtime `bc_kind`/`bc_value` tables when
    /// `<name>` is one of the coupled unknowns; otherwise boundary faces fall back to
    /// zero-gradient extrapolation.
    FromStateLayout,
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
