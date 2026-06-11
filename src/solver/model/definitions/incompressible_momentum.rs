use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    surface_scalar_dim, vol_scalar_dim, vol_vector_dim, EquationSystem, FieldRef, FluxRef,
};
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef, Vector2,
};
use crate::solver::model::ports::PortRegistry;
// si module no longer needed for boundary conditions - using type-level dimensions
use cfd2_codegen::solver::codegen::dsl::XY;
use cfd2_ir::dimensions::{
    Density, DivDim, DynamicViscosity, Force, InvTime, Length, MassFlux, Pressure, Velocity,
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

fn build_incompressible_momentum_system(
    _fields: &IncompressibleMomentumFields,
    with_mms_source: bool,
) -> EquationSystem {
    // NOTE: This model uses typed builder APIs with explicit cast_to() calls to align
    // terms to canonical dimension types. The type-level dimension expressions are not
    // normalized, so semantically equivalent dimensions (e.g., MassFlux * Velocity vs
    // MomentumDensity * Volume / Time) are different types; cast_to::<Force>() unifies them.

    // Build typed field and flux references
    let u_typed = TypedFieldRef::<Velocity, Vector2>::new("U");
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let phi_typed = TypedFluxRef::<MassFlux, Scalar>::new("phi");
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let mu_typed = TypedFieldRef::<DynamicViscosity, Scalar>::new("mu");
    let d_p_typed = TypedFieldRef::<cfd2_ir::dimensions::D_P, Scalar>::new("d_p");

    // Build coefficients
    let rho_coeff = TypedCoeff::from_field(rho_typed);
    let mu_coeff = TypedCoeff::from_field(mu_typed);
    let rho_dp_coeff =
        TypedCoeff::from_field(rho_typed).multiply(TypedCoeff::from_field(d_p_typed));

    // Build momentum equation terms
    // ddt(rho, U): integrated unit is MomentumDensity * Volume / Time = Force
    let ddt_term = typed_fvm::ddt_coeff(rho_coeff, u_typed);

    // div(phi, U): integrated unit is MassFlux * Velocity = Force.
    // Declared bounded (OpenFOAM `bounded Gauss`): assembly subtracts the
    // continuity defect (div phi) * U_P from the diagonal, matching the
    // reference solver's convection form while the flux is not exactly
    // divergence-free.
    let div_term = typed_fvm::div(phi_typed, u_typed).bounded();

    // laplacian(mu, U): integrated unit is DynamicViscosity * Velocity * Area / Length = Force
    let laplacian_term = typed_fvm::laplacian(mu_coeff, u_typed);

    // grad(p): integrated unit is Pressure * Area = Force
    let grad_term = typed_fvc::grad(p_typed);

    // Cast all terms to Force and add
    let mut momentum_sum = ddt_term.cast_to::<Force>()
        + div_term.cast_to::<Force>()
        + laplacian_term.cast_to::<Force>()
        + grad_term.cast_to::<Force>();
    if with_mms_source {
        // Manufactured per-component momentum source (MMS): one more
        // declared equation term, exactly like the scalar MMS variants.
        let mms_src_typed = TypedFieldRef::<DivDim<Force, cfd2_ir::dimensions::Volume>, Vector2>::new(
            INCOMPRESSIBLE_MMS_SOURCE_FIELD,
        );
        momentum_sum =
            momentum_sum + typed_fvc::source_vector(mms_src_typed, u_typed).cast_to::<Force>();
    }
    let momentum_eqn = momentum_sum.eqn(u_typed);

    // Build pressure equation terms
    // laplacian(rho*d_p, p): integrated unit is (rho*d_p) * Pressure * Area / Length = MassFlux
    // where rho*d_p has units: Density * (Volume*Time/Mass) = Time (since Volume/Mass = 1/Density)
    // So the unit is: Time * Pressure * Area / Length = Time * (Mass/(Length*Time^2)) * Length
    // = Mass / Time = MassFlux
    let p_laplacian_term = typed_fvm::laplacian(rho_dp_coeff, p_typed);

    // div_flux(phi, p): integrated unit is MassFlux
    let p_div_flux_term = typed_fvm::div_flux(phi_typed, p_typed);

    // Cast all terms to MassFlux and add
    let pressure_eqn = (p_laplacian_term.cast_to::<MassFlux>()
        + p_div_flux_term.cast_to::<MassFlux>())
    .eqn(p_typed);

    let mut system = EquationSystem::new();
    system.add_equation(momentum_eqn);
    system.add_equation(pressure_eqn);

    // Validate units to ensure the system is consistent (debug builds only)
    system
        .validate_units()
        .expect("incompressible momentum system failed unit validation");

    system
}

pub fn incompressible_momentum_system() -> EquationSystem {
    let fields = IncompressibleMomentumFields::new();
    build_incompressible_momentum_system(&fields, false)
}

pub fn incompressible_momentum_model() -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl(false)
}

/// `incompressible_momentum` plus a manufactured per-component momentum
/// source field (`INCOMPRESSIBLE_MMS_SOURCE_FIELD`) for MMS order tests.
pub fn incompressible_momentum_mms_model() -> Result<ModelSpec, String> {
    incompressible_momentum_model_impl(true)
}

fn incompressible_momentum_model_impl(with_mms_source: bool) -> Result<ModelSpec, String> {
    let fields = IncompressibleMomentumFields::new();
    let system = build_incompressible_momentum_system(&fields, with_mms_source);
    let mut layout_fields = vec![
        fields.u,
        fields.p,
        fields.d_p,
        fields.grad_p,
        fields.grad_p_old,
    ];
    if with_mms_source {
        layout_fields.push(vol_vector_dim::<DivDim<Force, cfd2_ir::dimensions::Volume>>(
            INCOMPRESSIBLE_MMS_SOURCE_FIELD,
        ));
    }
    let layout = PortRegistry::from_fields(layout_fields).into_state_layout();
    // d_p stays the closed form. The OpenFOAM-rAU-style alternative
    // (DpFormulation::FromAssembledDiagonal, June 2026) computes a VERIFIED
    // correct d_p = V/a_P (numerically matches the physical estimate;
    // see tests/dp_diag_probe.rs) but destabilizes the coupled outer loop:
    // the pressure response scales ~1/d_p while the relaxed updates and the
    // d_p-scaled velocity correction are calibrated for the closed-form
    // scale, so the step map amplifies (|u| x3.5/step at gentle settings;
    // NaN within 2-4 steps at momentum-MMS settings where a_P is 10x the
    // ddt-only value; lid reference rel_l2 0.61 vs 0.019). theta-damping of
    // d_p and alpha pairing (0.7/0.3, 1.0/1.0, relaxed/unrelaxed rAU) do
    // not help: the loop gain AT the target d_p is unstable. Re-enabling
    // requires a compensating mechanism (pressure-row equilibration or an
    // update/preconditioner redesign) — tracked in the plan as the Phase A
    // follow-up.
    let derived_rhie_chow =
        crate::solver::model::flux_derivation::derive_rhie_chow(&system, &layout)
            .map_err(|e| format!("failed to derive Rhie–Chow flux: {e}"))?;

    // Port-based validation and offset resolution (replaces ad-hoc StateLayout lookups)
    let (u0, u1, p) = {
        use crate::solver::model::ports::{PortRegistry, Pressure, Velocity};

        let mut registry = PortRegistry::new(layout.clone());

        // Validate required fields with clear errors
        registry
            .validate_vector2_field::<Velocity>("incompressible_momentum_model", "U")
            .map_err(|e| format!("state layout validation failed: {e}"))?;
        registry
            .validate_scalar_field::<Pressure>("incompressible_momentum_model", "p")
            .map_err(|e| format!("state layout validation failed: {e}"))?;

        // Resolve offsets via ports (no StateLayout probing in this function)
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
            // Inlet velocity is Dirichlet; the value can be updated at runtime via the solver's
            // boundary table API (e.g. `set_boundary_vec2(GpuBoundaryType::Inlet, "U", ...)`).
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

    // Clone layout for flux_module_module since we need to move it into ModelSpec
    let layout_for_flux = layout.clone();
    let flux_module_module = crate::solver::model::modules::flux_module::flux_module_module(
        flux_module,
        &system,
        &layout_for_flux,
        &primitives,
    )
    .map_err(|e| format!("failed to build flux_module module: {e}"))?;

    Ok(ModelSpec {
        id: if with_mms_source {
            "incompressible_momentum_mms"
        } else {
            "incompressible_momentum"
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
                layout: crate::solver::model::linear_solver::SchurBlockLayout::from_u_p(
                    &[u0, u1],
                    p,
                )
                .map_err(|e| format!("invalid SchurBlockLayout: {e}"))?,
            },
            ..Default::default()
        }),
        primitives,
    })
}

