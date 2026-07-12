use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    surface_scalar_dim, vol_scalar_dim, vol_vector_dim, EquationSystem, FieldRef, FluxRef,
};
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef, Vector2,
};
use crate::solver::model::ports::PortRegistry;
use cfd2_codegen::solver::codegen::dsl::XY;
use cfd2_ir::dimensions::{
    Density, DivDim, DynamicViscosity, Force, InvTime, Length, MassFlux, Pressure, Time, Velocity,
};

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

#[derive(Debug, Clone)]
pub struct IncompressibleMomentumFields {
    pub u: FieldRef,
    pub p: FieldRef,
    pub phi: FluxRef,
    pub mu: FieldRef,
    pub rho: FieldRef,
    pub d_p: FieldRef,
    pub grad_p: FieldRef,
    pub grad_p_old: FieldRef,
}

impl IncompressibleMomentumFields {
    pub fn new() -> Self {
        Self {
            u: vol_vector_dim::<Velocity>("U"),
            p: vol_scalar_dim::<Pressure>("p"),
            phi: surface_scalar_dim::<MassFlux>("phi"),
            mu: vol_scalar_dim::<DynamicViscosity>("mu"),
            rho: vol_scalar_dim::<Density>("rho"),
            d_p: vol_scalar_dim::<cfd2_ir::dimensions::D_P>("d_p"),
            grad_p: vol_vector_dim::<DivDim<Pressure, Length>>("grad_p"),
            grad_p_old: vol_vector_dim::<DivDim<Pressure, Length>>("grad_p_old"),
        }
    }
}

impl Default for IncompressibleMomentumFields {
    fn default() -> Self {
        Self::new()
    }
}

/// Name of the manufactured momentum source field on the `_mms` model
/// variant (Vector2, unit Force/Volume; uploaded host-side per cell).
pub const INCOMPRESSIBLE_MMS_SOURCE_FIELD: &str = "mms_src_U";

/// Viscous stress form declared in the momentum equation.
// The non-default arm is the documented revert path, not dead code.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViscousStressForm {
    /// `laplacian(mu, U)` only (OpenFOAM icoFoam-like). DEFAULT.
    LaplacianOnly,
    /// Plus the explicit transpose/deviatoric correction
    /// `div(mu * dev2((grad U)^T))` — the OpenFOAM simpleFoam laminar UEqn
    /// form. Analytically zero for div-free fields but discretely nonzero.
    FullDev2,
}

/// `FullDev2` is the shipped default: adding the reference solver's transpose
/// stress cuts the corner-singular lid/backstep pressure error. MMS orders
/// hold (the term is analytically zero for div-free fields, residual O(h²)).
const VISCOUS_STRESS_FORM: ViscousStressForm = ViscousStressForm::FullDev2;

/// Per-cell Brinkman penalisation field for immersed obstacles in the momentum
/// equation (`Sp`, unit Density/Time). Large NEGATIVE inside the solid drives
/// `U -> 0` (a momentum sink); zero in the fluid recovers the base operator.
/// Host code samples the obstacle SDF at cell centres. NOTE: for a clean
/// (non-porous) body the Rhie–Chow face flux must ALSO see `d_p = 0` on solid
/// cells, else pressure-driven flux leaks through the obstacle.
pub const IBM_MOMENTUM_PENALTY_FIELD: &str = "ibm_penalty_U";

fn build_incompressible_momentum_system(
    _fields: &IncompressibleMomentumFields,
    with_mms_source: bool,
    ale: bool,
) -> EquationSystem {
    build_incompressible_momentum_system_ibm(_fields, with_mms_source, ale, false)
}

fn build_incompressible_momentum_system_ibm(
    _fields: &IncompressibleMomentumFields,
    with_mms_source: bool,
    ale: bool,
    ibm: bool,
) -> EquationSystem {
    // Type-level dimension expressions are not normalized, so semantically
    // equivalent dimensions (e.g. MassFlux*Velocity vs MomentumDensity*Volume/Time)
    // are distinct types; the explicit cast_to::<Force>() calls unify them.
    let u_typed = TypedFieldRef::<Velocity, Vector2>::new("U");
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let phi_typed = TypedFluxRef::<MassFlux, Scalar>::new("phi");
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let mu_typed = TypedFieldRef::<DynamicViscosity, Scalar>::new("mu");
    let d_p_typed = TypedFieldRef::<cfd2_ir::dimensions::D_P, Scalar>::new("d_p");

    let rho_coeff = TypedCoeff::from_field(rho_typed);
    let mu_coeff = TypedCoeff::from_field(mu_typed);
    let rho_dp_coeff =
        TypedCoeff::from_field(rho_typed).multiply(TypedCoeff::from_field(d_p_typed));

    // ddt(rho, U): integrated unit MomentumDensity*Volume/Time = Force
    let ddt_term = typed_fvm::ddt_coeff(rho_coeff, u_typed);

    // div(phi, U): integrated unit MassFlux*Velocity = Force.
    // Declared bounded (OpenFOAM `bounded Gauss`): assembly subtracts the
    // continuity defect (div phi)*U_P from the diagonal, matching the
    // reference convection form while the flux is not exactly div-free.
    let mut div_term = typed_fvm::div(phi_typed, u_typed).bounded();
    if ale {
        // Convection consumes the mesh-relative flux `phi - rho*mesh_fluxes[face]`
        // (see `Term::relative_to_mesh`).
        div_term = div_term.with_mesh_relative();
    }

    // laplacian(mu, U): integrated unit DynamicViscosity*Velocity*Area/Length = Force
    let laplacian_term = typed_fvm::laplacian(mu_coeff, u_typed);

    // grad(p): integrated unit Pressure*Area = Force
    let grad_term = typed_fvc::grad(p_typed);

    let mut momentum_sum = ddt_term.cast_to::<Force>()
        + div_term.cast_to::<Force>()
        + laplacian_term.cast_to::<Force>()
        + grad_term.cast_to::<Force>();
    if VISCOUS_STRESS_FORM == ViscousStressForm::FullDev2 {
        // Explicit transpose/deviatoric viscous correction; same integrated
        // unit as the laplacian term (mu * U * Area / Length = Force).
        let mu_coeff2 = TypedCoeff::from_field(mu_typed);
        momentum_sum = momentum_sum
            + typed_fvc::div_dev2_grad_transpose(mu_coeff2, u_typed).cast_to::<Force>();
    }
    if with_mms_source {
        // Manufactured per-component momentum source (MMS).
        let mms_src_typed =
            TypedFieldRef::<DivDim<Force, cfd2_ir::dimensions::Volume>, Vector2>::new(
                INCOMPRESSIBLE_MMS_SOURCE_FIELD,
            );
        momentum_sum =
            momentum_sum + typed_fvc::source_vector(mms_src_typed, u_typed).cast_to::<Force>();
    }
    if ibm {
        // Immersed-boundary Brinkman penalisation: an implicit per-component sink
        // `source_coeff(Sp, U)` (Sp a Density/Time reaction rate) adds `-Sp*V` to
        // each velocity diagonal, driving `U -> 0` where the mask is large. Sp=0
        // in the fluid is the IEEE-identity recovery of the base momentum operator.
        let penalty_typed =
            TypedFieldRef::<DivDim<Density, Time>, Scalar>::new(IBM_MOMENTUM_PENALTY_FIELD);
        let penalty_coeff = TypedCoeff::from_field(penalty_typed);
        momentum_sum =
            momentum_sum + typed_fvm::source_coeff(penalty_coeff, u_typed).cast_to::<Force>();
    }
    let momentum_eqn = momentum_sum.eqn(u_typed);

    // laplacian(rho*d_p, p): integrated unit is (rho*d_p) * Pressure * Area / Length = MassFlux
    // where rho*d_p has units: Density * (Volume*Time/Mass) = Time (since Volume/Mass = 1/Density)
    // So the unit is: Time * Pressure * Area / Length = Time * (Mass/(Length*Time^2)) * Length
    // = Mass / Time = MassFlux
    let p_laplacian_term = typed_fvm::laplacian(rho_dp_coeff, p_typed);

    // div_flux(phi, p): integrated unit is MassFlux
    let mut p_div_flux_term = typed_fvm::div_flux(phi_typed, p_typed);
    if ale {
        // Continuity on the moving mesh is also mesh-relative. The compensating
        // volume-change source (`+rho*(V^{n+1}-V^n)/dt`, exact by SCL
        // construction) lands with the moving-volume ddt.
        p_div_flux_term = p_div_flux_term.with_mesh_relative();
    }

    let pressure_eqn = (p_laplacian_term.cast_to::<MassFlux>()
        + p_div_flux_term.cast_to::<MassFlux>())
    .eqn(p_typed);

    let mut system = EquationSystem::new();
    system.add_equation(momentum_eqn);
    system.add_equation(pressure_eqn);

    system
        .validate_units()
        .expect("incompressible momentum system failed unit validation");

    system
}

pub fn incompressible_momentum_system() -> EquationSystem {
    let fields = IncompressibleMomentumFields::new();
    build_incompressible_momentum_system(&fields, false, false)
}

pub fn incompressible_momentum_model() -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl(false, false)
}

/// `incompressible_momentum` plus a manufactured per-component momentum
/// source field (`INCOMPRESSIBLE_MMS_SOURCE_FIELD`) for MMS order tests.
pub fn incompressible_momentum_mms_model() -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl(true, false)
}

/// ALE (moving-mesh) variant of `incompressible_momentum`: identical physics
/// except the convection terms — `div(phi, U).bounded()` and `div_flux(phi, p)`
/// — are declared `.with_mesh_relative()`, so assembly consumes
/// `phi_rel = phi - rho * mesh_fluxes[face]`. Its own model id gets its own
/// generated kernels. With `mesh_fluxes` zero-filled and equal volume history
/// this reproduces the static model bitwise (`x - rho*0.0` is an IEEE identity).
pub fn incompressible_momentum_ale_model() -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl(false, true)
}

/// ALE + MMS combined variant (prescribed-motion MMS): the mesh-relative
/// convection of `_ale` plus the manufactured momentum source of `_mms`, under
/// its own model id. The manufactured source is re-evaluated at the moved cell
/// centroids every step (`set_field_vec2_current`, history-preserving).
pub fn incompressible_momentum_ale_mms_model() -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl(true, true)
}

/// STRUCTURED (`TopologyMode::Structured2D`) incompressible momentum: identical
/// Rhie–Chow pressure-velocity physics, lowered to the dense-array Cartesian
/// kernels (the operator expansion, flux module, gradients and Rhie–Chow all
/// emit index arithmetic — no connectivity indirection). Obstacles are immersed
/// via Brinkman penalisation, never cut out of the grid.
pub fn incompressible_momentum_structured_model() -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl_topo(
        false,
        false,
        cfd2_ir::equation::TopologyMode::Structured2D,
    )
}

fn incompressible_momentum_model_impl(
    with_mms_source: bool,
    ale: bool,
) -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl_topo(
        with_mms_source,
        ale,
        cfd2_ir::equation::TopologyMode::Unstructured,
    )
}

fn incompressible_momentum_model_impl_topo(
    with_mms_source: bool,
    ale: bool,
    topology: cfd2_ir::equation::TopologyMode,
) -> Result<ModelSpec, String> {
    let fields = IncompressibleMomentumFields::new();
    // The structured (Cartesian) variant is the immersed-boundary-capable one:
    // obstacles live in a per-cell Brinkman mask, never cut from the dense grid.
    let ibm = topology == cfd2_ir::equation::TopologyMode::Structured2D;
    let mut system = build_incompressible_momentum_system_ibm(&fields, with_mms_source, ale, ibm);
    system.set_topology(topology);
    let mut layout_fields = vec![
        fields.u,
        fields.p,
        fields.d_p,
        fields.grad_p,
        fields.grad_p_old,
    ];
    if with_mms_source {
        layout_fields.push(
            vol_vector_dim::<DivDim<Force, cfd2_ir::dimensions::Volume>>(
                INCOMPRESSIBLE_MMS_SOURCE_FIELD,
            ),
        );
    }
    if ibm {
        layout_fields.push(vol_scalar_dim::<DivDim<Density, Time>>(
            IBM_MOMENTUM_PENALTY_FIELD,
        ));
    }
    let layout = PortRegistry::from_fields(layout_fields).into_state_layout();
    // d_p stays the closed form. Two assembled-matrix alternatives exist:
    //
    // - FromAssembledDiagonal (OpenFOAM rAU, d_p = V/a_P): the coupled outer
    //   loop's gain at this d_p scale is >1 even with near-exact linear solves,
    //   so the outer map is unstable — row equilibration (a pure row scaling)
    //   cannot fix it.
    // - FromAssembledRowSum (SIMPLEC, d_p = V/Σ_row a): stable by construction
    //   and better on inlet-dominated flows, but the boundary-shrunk d_p
    //   intrinsically penalizes wall-bounded recirculating flows (lid), so it
    //   is default-off. Flip this call to
    //   derive_rhie_chow_with_dp(.., FromAssembledRowSum { theta: 0.5 }) to use it.
    let derived_rhie_chow =
        crate::solver::model::flux_derivation::derive_rhie_chow(&system, &layout)
            .map_err(|e| format!("failed to derive Rhie–Chow flux: {e}"))?;

    let (u0, u1, p) = {
        use crate::solver::model::ports::{PortRegistry, Pressure, Velocity};

        let mut registry = PortRegistry::new(layout.clone());

        registry
            .validate_vector2_field::<Velocity>("incompressible_momentum_model", "U")
            .map_err(|e| format!("state layout validation failed: {e}"))?;
        registry
            .validate_scalar_field::<Pressure>("incompressible_momentum_model", "p")
            .map_err(|e| format!("state layout validation failed: {e}"))?;

        let u_port = registry
            .register_vector2_field::<Velocity>("U")
            .map_err(|e| format!("U field registration failed: {e}"))?;
        let p_port = registry
            .register_scalar_field::<Pressure>("p")
            .map_err(|e| format!("p field registration failed: {e}"))?;

        let u0 = u_port
            .component(XY::X.to_usize() as u32)
            .ok_or_else(|| "U component x not found".to_string())?
            .full_offset();
        let u1 = u_port
            .component(XY::Y.to_usize() as u32)
            .ok_or_else(|| "U component y not found".to_string())?
            .full_offset();
        let p = p_port.offset();
        (u0, u1, p)
    };

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "U",
        FieldBoundarySpec::new()
            // Inlet velocity is Dirichlet; value updatable at runtime via
            // `set_boundary_vec2(GpuBoundaryType::Inlet, "U", ...)`.
            .set_uniform(
                GpuBoundaryType::Inlet,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            )
            // Outlet: do not constrain velocity (Neumann/zeroGradient).
            .set_uniform(
                GpuBoundaryType::Outlet,
                2,
                BoundaryCondition::zero_gradient_dim::<InvTime>(),
            )
            // Walls: no-slip.
            .set_uniform(
                GpuBoundaryType::Wall,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                2,
                BoundaryCondition::zero_gradient_dim::<InvTime>(),
            )
            // MovingWall: Dirichlet velocity (value set at runtime via boundary table API).
            .set_uniform(
                GpuBoundaryType::MovingWall,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            ),
    );
    boundaries.set_field(
        "p",
        FieldBoundarySpec::new()
            // Inlet and walls: zero-gradient pressure.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            // MovingWall: zero-gradient pressure (same as regular wall).
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            // Outlet: fix gauge by pinning pressure to 0.
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet_dim::<Pressure>(0.0),
            ),
    );

    let method = crate::solver::model::method::MethodSpec::Coupled(
        crate::solver::model::method::CoupledCapabilities {
            apply_relaxation_in_update: true,
            relaxation_requires_dtau: false,
            requires_flux_module: true,
            gradient_storage: crate::solver::model::gpu_spec::GradientStorage::PackedState,
        },
    );
    let flux_module = crate::solver::model::flux_module::FluxModuleSpec::Kernel {
        gradients: Some(
            crate::solver::model::flux_module::FluxModuleGradientsSpec::FromStateLayout,
        ),
        kernel: derived_rhie_chow.flux_kernel,
    };
    let primitives = crate::solver::model::primitives::PrimitiveDerivations::identity();

    let layout_for_flux = layout.clone();
    let flux_module_module = crate::solver::model::modules::flux_module::flux_module_module(
        flux_module,
        &system,
        &layout_for_flux,
        &primitives,
        None,
    )
    .map_err(|e| format!("failed to build flux_module module: {e}"))?;

    Ok(ModelSpec {
        id: match (with_mms_source, ale, topology) {
            (_, _, cfd2_ir::equation::TopologyMode::Structured2D) => {
                "incompressible_momentum_structured"
            }
            (true, false, _) => "incompressible_momentum_mms",
            (true, true, _) => "incompressible_momentum_ale_mms",
            (false, true, _) => "incompressible_momentum_ale",
            (false, false, _) => "incompressible_momentum",
        },
        system,
        state_layout: layout,
        boundaries,

        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            flux_module_module,
            crate::solver::model::modules::generic_coupled::generic_coupled_module(method),
            derived_rhie_chow.aux_module,
        ],
        // The generic coupled path needs a saddle-point-capable preconditioner.
        linear_solver: Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
            preconditioner: crate::solver::model::linear_solver::ModelPreconditionerSpec::Schur {
                omega: 1.0,
                sweeps_cap: 64,
                layout: crate::solver::model::linear_solver::SchurBlockLayout::from_u_p(
                    &[u0, u1],
                    p,
                )
                .map_err(|e| format!("invalid SchurBlockLayout: {e}"))?,
            },
            ..Default::default()
        }),
        primitives,
        explicit_primitives: None,
        explicit_mass_closure_proof: super::ExplicitMassClosureProof::RuntimePivoted {
            justification: "the momentum mass is the positive runtime density parameter; generated row-scaled pivot guards enforce its per-stage domain",
        },
    })
}
