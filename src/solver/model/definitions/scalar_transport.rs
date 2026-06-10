//! Passive scalar transport: the Phase-1 "vertical slice" of the math-driven solver.
//!
//! The entire model is a declaration: equation terms (with an optionally declared
//! per-term scheme), a declarative flux definition (advecting velocity), boundary
//! conditions, and an MMS source field. There are no model-specific modules, no
//! hand-built face expressions, and no hand-written WGSL.

use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{vol_scalar_dim, vol_vector_dim, EquationSystem};
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef,
};
use crate::solver::model::flux_derivation::{derive_flux_module_kernel, FluxExprSpec};
use crate::solver::model::ports::PortRegistry;
use crate::solver::scheme::Scheme;
use cfd2_ir::dimensions::{
    Area, Dimensionless, DivDim, InvTime, Length, Time, Velocity, Volume,
};

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

type DimensionlessGradient = DivDim<Dimensionless, Length>;
type VolumetricFlux = DivDim<Volume, Time>;
/// Integrated unit of every term in the transport equation: Volume / Time.
type TransportIntegratedUnit = DivDim<Volume, Time>;

/// Name of the solved scalar.
pub const SCALAR_FIELD: &str = "T";
/// Name of the (host-written, frozen) advecting velocity field.
pub const ADVECTING_VELOCITY_FIELD: &str = "U_adv";
/// Name of the per-cell manufactured-solution source field.
pub const MMS_SOURCE_FIELD: &str = "mms_src_T";

/// Diffusivity baked into the model declaration (`kappa` in `laplacian(kappa, T)`).
pub const KAPPA: f64 = 1.0;

fn build_scalar_transport_model(
    id: &'static str,
    declared_div_scheme: Option<Scheme>,
) -> Result<ModelSpec, String> {
    let t_typed = TypedFieldRef::<Dimensionless, Scalar>::new(SCALAR_FIELD);
    let phi_typed = TypedFluxRef::<VolumetricFlux, Scalar>::new("phi_adv");
    let kappa_typed: TypedCoeff<DivDim<Area, Time>> = TypedCoeff::constant(KAPPA);
    let mms_src_typed = TypedFieldRef::<InvTime, Scalar>::new(MMS_SOURCE_FIELD);

    // ddt(T) + div(phi_adv, T) - kappa*lap(T) = S
    // (the `laplacian` term assembles as -div(kappa grad T) on the LHS; the explicit
    // source adds +S*V to the RHS — see unified_assembly sign conventions).
    let ddt_term = typed_fvm::ddt(t_typed).cast_to::<TransportIntegratedUnit>();
    let mut div_term = typed_fvm::div(phi_typed, t_typed);
    if let Some(scheme) = declared_div_scheme {
        div_term = div_term.scheme(scheme);
    }
    let div_cast = div_term.cast_to::<TransportIntegratedUnit>();
    let laplacian_term =
        typed_fvm::laplacian(kappa_typed, t_typed).cast_to::<TransportIntegratedUnit>();
    let source_term = typed_fvc::source_coeff(TypedCoeff::from_field(mms_src_typed), t_typed)
        .cast_to::<TransportIntegratedUnit>();

    let eqn = (ddt_term + div_cast + laplacian_term + source_term).eqn(t_typed);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);
    system
        .validate_units()
        .map_err(|e| format!("{id} system failed unit validation: {e:?}"))?;

    // State layout: solved scalar + frozen advecting velocity + MMS source.
    let layout = PortRegistry::from_fields(vec![
        vol_scalar_dim::<Dimensionless>(SCALAR_FIELD),
        vol_vector_dim::<Velocity>(ADVECTING_VELOCITY_FIELD),
        vol_scalar_dim::<InvTime>(MMS_SOURCE_FIELD),
    ])
    .into_state_layout();

    // The flux is declared, not hand-built: volumetric flux of U_adv.
    let flux_kernel = derive_flux_module_kernel(
        &system,
        &layout,
        &FluxExprSpec::AdvectingVelocity {
            velocity: ADVECTING_VELOCITY_FIELD,
            density: None,
        },
    )
    .map_err(|e| format!("{id}: failed to derive advecting flux: {e}"))?;

    // Dirichlet on every boundary type used by the tests (values set per face at
    // runtime); zero-gradient is covered by the diffusion-demo MMS suite.
    let dirichlet = || BoundaryCondition::dirichlet_dim::<Dimensionless>(0.0);
    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        SCALAR_FIELD,
        FieldBoundarySpec::new()
            .set_uniform(GpuBoundaryType::Inlet, 1, dirichlet())
            .set_uniform(GpuBoundaryType::Outlet, 1, dirichlet())
            .set_uniform(GpuBoundaryType::Wall, 1, dirichlet()),
    );

    let method = crate::solver::model::method::MethodSpec::Coupled(
        crate::solver::model::method::CoupledCapabilities {
            apply_relaxation_in_update: false,
            relaxation_requires_dtau: false,
            requires_flux_module: true,
            gradient_storage: crate::solver::model::gpu_spec::GradientStorage::PackedState,
        },
    );
    let primitives = crate::solver::model::primitives::PrimitiveDerivations::identity();
    let flux_module = crate::solver::model::flux_module::FluxModuleSpec::Kernel {
        gradients: None,
        kernel: flux_kernel,
    };
    let flux_module_module = crate::solver::model::modules::flux_module::flux_module_module(
        flux_module,
        &system,
        &layout,
        &primitives,
    )
    .map_err(|e| format!("{id}: failed to build flux_module module: {e}"))?;

    Ok(ModelSpec {
        id,
        system,
        state_layout: layout,
        boundaries,
        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            flux_module_module,
            crate::solver::model::modules::generic_coupled::generic_coupled_module(method),
        ],
        linear_solver: None,
        primitives,
    })
}

/// Scalar transport with the advection scheme left on the runtime knob.
pub fn scalar_transport_model() -> Result<ModelSpec, String> {
    build_scalar_transport_model("scalar_transport", None)
}

/// Scalar transport with second-order upwind *declared* on the convection term:
/// the declared scheme is baked into the generated kernel and is independent of
/// the runtime `advection_scheme` knob.
pub fn scalar_transport_sou_model() -> Result<ModelSpec, String> {
    build_scalar_transport_model("scalar_transport_sou", Some(Scheme::SecondOrderUpwind))
}
