//! All-Mach, f32, pressure-based solver model.
//!
//! This is the pressure-based counterpart to the density-based `compressible`
//! model. It is built as an *additive extension* of `incompressible_momentum`:
//! the velocity/pressure unknowns, the Rhie–Chow collocated coupling, the Schur
//! preconditioner and the coupled assembly are all reused verbatim. The only
//! physics added is **compressibility**, expressed as a single time term in the
//! continuity/pressure equation:
//!
//! ```text
//!   momentum :  ddt(rho,U) + div(phi,U) - lap(mu,U) + grad(p) + dev2(mu,(grad U)^T) = 0
//!   pressure :  ddt(psi,p) + lap(rho*d_p,p) + div_flux(phi,p)                        = 0
//! ```
//!
//! `psi = d(rho)/d(p) = 1/c^2` is the compressibility (units Density/Pressure).
//! The `ddt(psi,p)` term is the discrete `d(rho)/dt = (1/c^2) dp/dt` of the
//! continuity equation under a barotropic EOS `rho = rho_ref + psi*(p - p_ref)`.
//!
//! **All-Mach by construction:** as Mach -> 0 (`c -> inf`, `psi -> 0`,
//! `rho -> rho_ref`) every added term vanishes and the system reduces *exactly*
//! to `incompressible_momentum`, so the validated incompressible behaviour is the
//! `psi -> 0` limit. As `psi` grows, acoustic/compressible effects appear.
//!
//! **f32 by construction:** `p` is a gauge/perturbation pressure (O(rho*U^2),
//! centred at 0, pinned to 0 at the outlet — inherited from the incompressible
//! model), so `grad(p)` is resolved at full f32 precision. This is the property
//! the density-based solver lacks (it carries absolute thermodynamic pressure
//! ~1e5, burying the O(1e-4) wake signal below f32 epsilon).
//!
//! `psi` is a runtime state field (set uniformly to `1/c^2`); the density `rho`
//! is likewise a state field. For the first slice `rho` is held at `rho_ref`
//! (correct to O(Mach^2)); a `rho = rho_ref + psi*p` refresh kernel (true
//! barotropic density coupling) is layered on in a later slice.

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
    Volume,
};

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

/// Compressibility `psi = d(rho)/d(p) = 1/c^2`, units Density / Pressure.
type Compressibility = DivDim<Density, Pressure>;

/// Manufactured per-component momentum source (Vector2, Force/Volume).
pub const ALLMACH_MMS_SOURCE_U_FIELD: &str = "mms_src_U";
/// Manufactured continuity/pressure source (Scalar, MassFlux/Volume).
pub const ALLMACH_MMS_SOURCE_P_FIELD: &str = "mms_src_p";

#[derive(Debug, Clone)]
pub struct AllMachPressureFields {
    pub u: FieldRef,
    pub p: FieldRef,
    pub phi: FluxRef,
    pub mu: FieldRef,
    pub rho: FieldRef,
    pub d_p: FieldRef,
    pub grad_p: FieldRef,
    pub grad_p_old: FieldRef,
    /// Compressibility psi = 1/c^2 (runtime field; 0 => incompressible).
    pub psi: FieldRef,
    /// Per-cell local pseudo-time step for steady-state acceleration (Local Time
    /// Stepping). When 0 (default), the assembly falls back to the global
    /// `constants.dt` (time-accurate, byte-identical to a model without this
    /// field); when filled with per-cell stable steps (`cfl*h_c/lambda_c`) each
    /// cell advances toward steady state at its own maximal rate.
    pub dt_local: FieldRef,
}

impl AllMachPressureFields {
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
            psi: vol_scalar_dim::<Compressibility>("psi"),
            dt_local: vol_scalar_dim::<Time>("dt_local"),
        }
    }
}

impl Default for AllMachPressureFields {
    fn default() -> Self {
        Self::new()
    }
}

/// Mirror of the incompressible momentum model's shipped viscous stress form
/// (`FullDev2`): laplacian(mu,U) plus the explicit transpose/deviatoric
/// correction. Kept identical so the incompressible limit (psi -> 0) is
/// byte-comparable to `incompressible_momentum`.
const USE_FULL_DEV2: bool = true;

fn build_allmach_system(_fields: &AllMachPressureFields, with_mms_source: bool) -> EquationSystem {
    let u_typed = TypedFieldRef::<Velocity, Vector2>::new("U");
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let phi_typed = TypedFluxRef::<MassFlux, Scalar>::new("phi");
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let mu_typed = TypedFieldRef::<DynamicViscosity, Scalar>::new("mu");
    let d_p_typed = TypedFieldRef::<cfd2_ir::dimensions::D_P, Scalar>::new("d_p");
    let psi_typed = TypedFieldRef::<Compressibility, Scalar>::new("psi");

    let rho_coeff = TypedCoeff::from_field(rho_typed);
    let mu_coeff = TypedCoeff::from_field(mu_typed);
    let rho_dp_coeff =
        TypedCoeff::from_field(rho_typed).multiply(TypedCoeff::from_field(d_p_typed));
    let psi_coeff = TypedCoeff::from_field(psi_typed);

    // ----- Momentum equation (identical to incompressible_momentum) -----
    let ddt_term = typed_fvm::ddt_coeff(rho_coeff, u_typed);
    let div_term = typed_fvm::div(phi_typed, u_typed).bounded();
    let laplacian_term = typed_fvm::laplacian(mu_coeff, u_typed);
    let grad_term = typed_fvc::grad(p_typed);

    let mut momentum_sum = ddt_term.cast_to::<Force>()
        + div_term.cast_to::<Force>()
        + laplacian_term.cast_to::<Force>()
        + grad_term.cast_to::<Force>();
    if USE_FULL_DEV2 {
        let mu_coeff2 = TypedCoeff::from_field(mu_typed);
        momentum_sum = momentum_sum
            + typed_fvc::div_dev2_grad_transpose(mu_coeff2, u_typed).cast_to::<Force>();
    }
    if with_mms_source {
        let mms_src_typed = TypedFieldRef::<DivDim<Force, Volume>, Vector2>::new(
            ALLMACH_MMS_SOURCE_U_FIELD,
        );
        momentum_sum =
            momentum_sum + typed_fvc::source_vector(mms_src_typed, u_typed).cast_to::<Force>();
    }
    let momentum_eqn = momentum_sum.eqn(u_typed);

    // ----- Continuity/pressure equation: incompressible terms + ddt(psi,p) -----
    // ddt(psi,p) integrates to: psi * p * Vol / Time = (Density/Pressure) * Pressure * Vol/Time
    //                         = Density * Vol / Time = Mass/Time = MassFlux. ✓
    let compressibility_term = typed_fvm::ddt_coeff(psi_coeff, p_typed);
    let p_laplacian_term = typed_fvm::laplacian(rho_dp_coeff, p_typed);
    let p_div_flux_term = typed_fvm::div_flux(phi_typed, p_typed);

    let mut pressure_sum = compressibility_term.cast_to::<MassFlux>()
        + p_laplacian_term.cast_to::<MassFlux>()
        + p_div_flux_term.cast_to::<MassFlux>();
    if with_mms_source {
        // Scalar continuity source: mms_src_p (MassFlux/Volume) * V = MassFlux.
        let mms_src_p_typed =
            TypedFieldRef::<DivDim<MassFlux, Volume>, Scalar>::new(ALLMACH_MMS_SOURCE_P_FIELD);
        let mms_src_p_coeff = TypedCoeff::from_field(mms_src_p_typed);
        pressure_sum =
            pressure_sum + typed_fvc::source_coeff(mms_src_p_coeff, p_typed).cast_to::<MassFlux>();
    }
    let pressure_eqn = pressure_sum.eqn(p_typed);

    let mut system = EquationSystem::new();
    system.add_equation(momentum_eqn);
    system.add_equation(pressure_eqn);

    system
        .validate_units()
        .expect("allmach_pressure system failed unit validation");

    system
}

pub fn allmach_pressure_system() -> EquationSystem {
    let fields = AllMachPressureFields::new();
    build_allmach_system(&fields, false)
}

pub fn allmach_pressure_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(false)
}

/// `allmach_pressure` plus manufactured momentum + continuity source fields for
/// MMS order tests.
pub fn allmach_pressure_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true)
}

fn allmach_pressure_model_impl(with_mms_source: bool) -> Result<ModelSpec, String> {
    let fields = AllMachPressureFields::new();
    let system = build_allmach_system(&fields, with_mms_source);

    // Keep U,p,d_p,grad_p,grad_p_old at the same offsets as incompressible
    // (0,2,3,4,6); append psi (and any MMS sources) after.
    let mut layout_fields = vec![
        fields.u,
        fields.p,
        fields.d_p,
        fields.grad_p,
        fields.grad_p_old,
        fields.psi,
        fields.dt_local,
        // Variable density: when `rho` is a state-layout field, the Rhie–Chow flux
        // deriver and the ddt(rho,U) coefficient read it per-cell (instead of the
        // uniform `constants.density`), giving a genuinely compressible continuity
        // `div(rho*U)=0`. Must be initialised to rho_ref and refreshed from the EOS
        // (`rho = rho_ref + psi*p`) each outer iteration.
        fields.rho,
    ];
    if with_mms_source {
        layout_fields.push(vol_vector_dim::<DivDim<Force, Volume>>(
            ALLMACH_MMS_SOURCE_U_FIELD,
        ));
        layout_fields.push(vol_scalar_dim::<DivDim<MassFlux, Volume>>(
            ALLMACH_MMS_SOURCE_P_FIELD,
        ));
    }
    let layout = PortRegistry::from_fields(layout_fields).into_state_layout();

    let derived_rhie_chow =
        crate::solver::model::flux_derivation::derive_rhie_chow(&system, &layout)
            .map_err(|e| format!("failed to derive Rhie–Chow flux: {e}"))?;

    let (u0, u1, p) = {
        use crate::solver::model::ports::{PortRegistry, Pressure as PortPressure, Velocity as PortVelocity};

        let mut registry = PortRegistry::new(layout.clone());
        registry
            .validate_vector2_field::<PortVelocity>("allmach_pressure_model", "U")
            .map_err(|e| format!("state layout validation failed: {e}"))?;
        registry
            .validate_scalar_field::<PortPressure>("allmach_pressure_model", "p")
            .map_err(|e| format!("state layout validation failed: {e}"))?;

        let u_port = registry
            .register_vector2_field::<PortVelocity>("U")
            .map_err(|e| format!("U field registration failed: {e}"))?;
        let p_port = registry
            .register_scalar_field::<PortPressure>("p")
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
            .set_uniform(
                GpuBoundaryType::Inlet,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                2,
                BoundaryCondition::zero_gradient_dim::<InvTime>(),
            )
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
            .set_uniform(
                GpuBoundaryType::MovingWall,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            ),
    );
    boundaries.set_field(
        "p",
        FieldBoundarySpec::new()
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
        gradients: Some(crate::solver::model::flux_module::FluxModuleGradientsSpec::FromStateLayout),
        kernel: derived_rhie_chow.flux_kernel,
    };
    let primitives = crate::solver::model::primitives::PrimitiveDerivations::identity();

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
            "allmach_pressure_mms"
        } else {
            "allmach_pressure"
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
