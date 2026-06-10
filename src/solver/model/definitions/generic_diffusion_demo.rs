use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{vol_scalar_dim, EquationSystem};
use crate::solver::model::backend::typed_ast::{typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef};
use crate::solver::model::ports::PortRegistry;
// si module no longer needed for boundary conditions - using type-level dimensions
use cfd2_ir::dimensions::{Area, Dimensionless, DivDim, InvTime, Length, Time, Volume};
// Type alias for dimensionless gradient (used for boundary conditions)
type DimensionlessGradient = DivDim<Dimensionless, Length>;

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

/// Type alias for the integrated unit of the diffusion equation:
/// ddt(phi) has unit: Volume * Dimensionless / Time = Volume / Time
/// laplacian(kappa, phi) has unit: (Area/Time) * Dimensionless * Area / Length = Volume / Time
pub type DiffusionIntegratedUnit = DivDim<Volume, Time>;

/// Name of the per-cell manufactured-solution source field used by the `_mms` model variants.
///
/// The field holds `S = dphi*/dt - kappa * lap(phi*)` evaluated at cell centers; the equation
/// gains an explicit `source_coeff(S, phi)` term, i.e. the manufactured source is just one more
/// term in the math declaration. Host code uploads values via `set_field_scalar`.
pub const MMS_SOURCE_FIELD: &str = "mms_src_phi";

/// Boundary-condition layout for the diffusion demo variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiffusionBcVariant {
    /// Dirichlet at Inlet/Outlet, zero-gradient at Wall (the original demo).
    DirichletInletOutlet,
    /// Zero-valued Neumann at Inlet/Outlet, zero-gradient at Wall (the original Neumann demo).
    NeumannInletOutlet,
    /// Dirichlet at Inlet/Outlet *and* Wall (values intended to be set per face at runtime).
    DirichletAll,
    /// Dirichlet at Inlet/Outlet, non-zero-capable Neumann at Wall (values set per face).
    DirichletWithNeumannWalls,
}

fn diffusion_boundaries(variant: DiffusionBcVariant) -> BoundarySpec {
    let mut boundaries = BoundarySpec::default();
    let dirichlet = || BoundaryCondition::dirichlet_dim::<Dimensionless>(0.0);
    let neumann = || BoundaryCondition::neumann_dim::<DimensionlessGradient>(0.0);
    let zero_gradient = || BoundaryCondition::zero_gradient_dim::<DimensionlessGradient>();

    let (inlet, outlet, wall) = match variant {
        DiffusionBcVariant::DirichletInletOutlet => (dirichlet(), dirichlet(), zero_gradient()),
        DiffusionBcVariant::NeumannInletOutlet => (neumann(), neumann(), zero_gradient()),
        DiffusionBcVariant::DirichletAll => (dirichlet(), dirichlet(), dirichlet()),
        DiffusionBcVariant::DirichletWithNeumannWalls => (dirichlet(), dirichlet(), neumann()),
    };

    boundaries.set_field(
        "phi",
        FieldBoundarySpec::new()
            .set_uniform(GpuBoundaryType::Inlet, 1, inlet)
            .set_uniform(GpuBoundaryType::Outlet, 1, outlet)
            .set_uniform(GpuBoundaryType::Wall, 1, wall),
    );
    boundaries
}

fn build_diffusion_model(
    id: &'static str,
    variant: DiffusionBcVariant,
    with_mms_source: bool,
) -> Result<ModelSpec, String> {
    // Build typed field reference for phi (dimensionless scalar)
    let phi_typed = TypedFieldRef::<Dimensionless, Scalar>::new("phi");

    // Build coefficient kappa with unit Area/Time
    let kappa_typed: TypedCoeff<DivDim<Area, Time>> = TypedCoeff::constant(1.0);

    // Build equation terms using typed FVM constructors
    // ddt(phi): integrated unit is Volume/Time (Dimensionless * Volume / Time)
    let ddt_term = typed_fvm::ddt(phi_typed);

    // laplacian(kappa, phi): integrated unit is (Area/Time) * Dimensionless * Area / Length
    // = Area^2 / (Time * Length) = Volume / Time (since Area = Length^2, Volume = Length^3)
    let laplacian_term = typed_fvm::laplacian(kappa_typed, phi_typed);

    // Need to cast terms to a common dimension type for addition
    // The actual integrated unit for both is Volume/Time
    let ddt_cast = ddt_term.cast_to::<DiffusionIntegratedUnit>();
    let laplacian_cast = laplacian_term.cast_to::<DiffusionIntegratedUnit>();

    let eqn = if with_mms_source {
        // Manufactured-solution source: an explicit per-cell source term whose coefficient is
        // the state-layout field `mms_src_phi` (unit 1/Time so that coeff * Volume matches the
        // equation's integrated unit Volume/Time).
        let mms_src_typed = TypedFieldRef::<InvTime, Scalar>::new(MMS_SOURCE_FIELD);
        let mms_coeff = TypedCoeff::from_field(mms_src_typed);
        let source_cast =
            typed_fvc::source_coeff(mms_coeff, phi_typed).cast_to::<DiffusionIntegratedUnit>();
        (ddt_cast + laplacian_cast + source_cast).eqn(phi_typed)
    } else {
        (ddt_cast + laplacian_cast).eqn(phi_typed)
    };

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

    // Validate units to ensure the system is consistent
    system
        .validate_units()
        .map_err(|e| format!("{id} system failed unit validation: {e:?}"))?;

    // Build state layout via PortRegistry (single source of truth for field offsets)
    let phi = vol_scalar_dim::<Dimensionless>("phi");
    let mut state_fields = vec![phi];
    if with_mms_source {
        state_fields.push(vol_scalar_dim::<InvTime>(MMS_SOURCE_FIELD));
    }
    let layout = PortRegistry::from_fields(state_fields).into_state_layout();

    Ok(ModelSpec {
        id,
        system,
        state_layout: layout,
        boundaries: diffusion_boundaries(variant),

        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            crate::solver::model::modules::generic_coupled::generic_coupled_module(
                crate::solver::model::method::MethodSpec::Coupled(
                    crate::solver::model::method::CoupledCapabilities::default(),
                ),
            ),
        ],
        linear_solver: None,
        primitives: crate::solver::model::primitives::PrimitiveDerivations::default(),
    })
}

pub fn generic_diffusion_demo_model() -> Result<ModelSpec, String> {
    build_diffusion_model(
        "generic_diffusion_demo",
        DiffusionBcVariant::DirichletInletOutlet,
        false,
    )
}

pub fn generic_diffusion_demo_neumann_model() -> Result<ModelSpec, String> {
    build_diffusion_model(
        "generic_diffusion_demo_neumann",
        DiffusionBcVariant::NeumannInletOutlet,
        false,
    )
}

/// MMS variant of the diffusion demo: Dirichlet at Inlet/Outlet, zero-gradient at Wall,
/// plus the manufactured source field/term.
pub fn generic_diffusion_demo_mms_model() -> Result<ModelSpec, String> {
    build_diffusion_model(
        "generic_diffusion_demo_mms",
        DiffusionBcVariant::DirichletInletOutlet,
        true,
    )
}

/// MMS variant with Dirichlet on every boundary type (per-face values set at runtime).
pub fn generic_diffusion_demo_mms_dirichlet_model() -> Result<ModelSpec, String> {
    build_diffusion_model(
        "generic_diffusion_demo_mms_dirichlet",
        DiffusionBcVariant::DirichletAll,
        true,
    )
}

/// MMS variant with Dirichlet at Inlet/Outlet and (non-zero-capable) Neumann at Wall.
pub fn generic_diffusion_demo_mms_neumann_model() -> Result<ModelSpec, String> {
    build_diffusion_model(
        "generic_diffusion_demo_mms_neumann",
        DiffusionBcVariant::DirichletWithNeumannWalls,
        true,
    )
}
