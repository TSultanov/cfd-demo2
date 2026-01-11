/// Pluggable flux computation module specification.
///
/// Models declare *which* flux method to use (KT, Rhie-Chow, etc.) without
/// implementing the WGSL generation. Codegen lowers these specs to kernels.

/// Flux computation module specification.
#[derive(Debug, Clone, PartialEq)]
pub enum FluxModuleSpec {
    /// Kurganov-Tadmor (central) flux for conservation laws.
    ///
    /// Requires: state → primitives (p, u), EOS (sound speed)
    /// Computes: convective + pressure fluxes for conserved variables
    ///
    /// Used by: compressible Euler, shallow water, etc.
    KurganovTadmor {
        /// Face value reconstruction method
        reconstruction: ReconstructionSpec,
    },

    /// Rhie-Chow interpolation for incompressible momentum-pressure coupling.
    ///
    /// Requires: velocity, pressure, d_p field
    /// Computes: mass flux phi on cell faces
    ///
    /// Used by: incompressible momentum solvers
    RhieChow {
        /// Momentum under-relaxation factor (if applicable)
        alpha_u: Option<f32>,
    },

    /// Generic convective flux: F = phi * field (no pressure/momentum coupling).
    ///
    /// Requires: a pre-computed face flux field (e.g., "phi")
    /// Computes: advective flux for the target field
    Convective {
        /// Name of the face flux field to use (e.g., "phi")
        flux_field: String,
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
        // Safe default: no flux computation (for pure diffusion models)
        FluxModuleSpec::Convective {
            flux_field: "phi".to_string(),
        }
    }
}
