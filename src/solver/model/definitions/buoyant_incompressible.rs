// Buoyant incompressible flow (Boussinesq): incompressible momentum + pressure
// as `incompressible_momentum`, plus a temperature transport equation advected
// by the solved Rhie-Chow mass flux and Boussinesq buoyancy fed back into
// momentum as a directional explicit source.
//
// Equations (all terms sum to zero):
//
//   momentum:  ddt(rho U) + div(phi, U)|bounded - lap(mu, U) + grad(p)
//                = -rho beta (T - T0) g_vec        (buoyancy, explicit)
//   pressure:  -lap(rho d_p, p) + divFlux(phi, p) = 0
//   energy/cp: ddt(rho T) + div(phi, T) - lap(k/cp, T) = 0
//
// Buoyancy is two directional sources (gravity along -y, dir [0, -1]):
//   (-beta*g * rho * T) * dir   +   (+beta*g*T0 * rho) * dir
// summing to f_y = rho*beta*g*(T - T0): hot fluid rises.
//
// The temperature equation is declared divided by the (constant) specific heat
// so every term shares unit Density*Temperature*Volume/Time; the conduction
// coefficient is therefore k/cp.

use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{vol_scalar_dim, vol_vector_dim, EquationSystem};
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef, Vector2,
};
use crate::solver::model::ports::PortRegistry;
use cfd2_ir::dimensions::{
    Acceleration, Density, DivDim, DynamicViscosity, Force, Length, MassFlux, MulDim, Pressure,
    Temperature, Time, Velocity, Volume,
};

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

/// beta * |g|: thermal expansion coefficient times gravity magnitude.
/// Default of the runtime param `buoyant.beta_g`; equations read
/// `constants.buoyant_beta_g` at runtime. MMS solutions are derived from these
/// canonical values, so tests must leave the params at their defaults.
pub const BUOYANT_BETA_G: f64 = 0.1;
/// Reference temperature of the Boussinesq linearization.
/// Default of the runtime param `buoyant.t0`.
pub const BUOYANT_T0: f64 = 0.5;
/// Conduction coefficient divided by specific heat (rho * thermal diffusivity).
/// Default of the runtime param `buoyant.k_over_cp`.
pub const BUOYANT_K_OVER_CP: f64 = 1.0;
/// Name of the solved temperature field.
pub const BUOYANT_TEMPERATURE_FIELD: &str = "T";
/// Manufactured momentum source field on the `_mms` variant (Vector2).
pub const BUOYANT_MMS_SOURCE_U_FIELD: &str = "mms_src_U";
/// Manufactured temperature source field on the `_mms` variant (scalar).
pub const BUOYANT_MMS_SOURCE_T_FIELD: &str = "mms_src_T";

// Unit aliases for the temperature equation (declared divided by cp).
type TEquationUnit = DivDim<MulDim<MulDim<Density, Temperature>, Volume>, Time>;
type TSourceUnit = DivDim<MulDim<Density, Temperature>, Time>;
// beta*g has units of acceleration per kelvin.
type BetaG = DivDim<Acceleration, Temperature>;

fn build_buoyant_system(with_mms_sources: bool) -> EquationSystem {
    let u_typed = TypedFieldRef::<Velocity, Vector2>::new("U");
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let t_typed = TypedFieldRef::<Temperature, Scalar>::new(BUOYANT_TEMPERATURE_FIELD);
    let phi_typed = TypedFluxRef::<MassFlux, Scalar>::new("phi");
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let mu_typed = TypedFieldRef::<DynamicViscosity, Scalar>::new("mu");
    let d_p_typed = TypedFieldRef::<cfd2_ir::dimensions::D_P, Scalar>::new("d_p");

    let rho_coeff = TypedCoeff::from_field(rho_typed);
    let mu_coeff = TypedCoeff::from_field(mu_typed);
    let rho_dp_coeff =
        TypedCoeff::from_field(rho_typed).multiply(TypedCoeff::from_field(d_p_typed));

    // ---- Momentum: as incompressible_momentum, plus buoyancy. ----
    let ddt_term = typed_fvm::ddt_coeff(rho_coeff.clone(), u_typed);
    let div_term = typed_fvm::div(phi_typed, u_typed).bounded();
    let laplacian_term = typed_fvm::laplacian(mu_coeff, u_typed);
    let grad_term = typed_fvc::grad(p_typed);

    // Boussinesq buoyancy, direction [0, -1] (gravity along -y):
    //   (-beta_g * rho * T) * dir_c  +  (beta_g * T0 * rho) * dir_c
    // beta_g and T0 are runtime uniform params: the field names below are not
    // state slots, so coefficient lowering falls through to the
    // named-constants table (coeff_named_expr_dyn).
    let beta_g_param =
        TypedCoeff::from_field(TypedFieldRef::<BetaG, Scalar>::new("buoyant_beta_g"));
    let t0_param = TypedCoeff::from_field(TypedFieldRef::<Temperature, Scalar>::new("buoyant_t0"));
    let minus_one: TypedCoeff<cfd2_ir::dimensions::Dimensionless> = TypedCoeff::constant(-1.0);
    let buoy_t_coeff = minus_one
        .multiply(beta_g_param.clone())
        .multiply(rho_coeff.clone())
        .multiply(TypedCoeff::from_field(t_typed));
    let buoy_const_coeff = beta_g_param.multiply(t0_param).multiply(rho_coeff.clone());

    let gravity_dir = [0.0, -1.0];
    let buoy_t_term = typed_fvc::source_directional(buoy_t_coeff, gravity_dir, u_typed);
    let buoy_const_term = typed_fvc::source_directional(buoy_const_coeff, gravity_dir, u_typed);

    let mut momentum_sum = ddt_term.cast_to::<Force>()
        + div_term.cast_to::<Force>()
        + laplacian_term.cast_to::<Force>()
        + grad_term.cast_to::<Force>()
        + buoy_t_term.cast_to::<Force>()
        + buoy_const_term.cast_to::<Force>();
    // Explicit dev2 transpose viscous correction (full-stress form). The MMS
    // velocity is divergence-free, so the MMS sources are unchanged. Declaring
    // the term forces the gradients pipeline on and excludes the
    // gradients+assembly fusion (neighbor grad_state reads).
    {
        let mu_coeff2 = TypedCoeff::from_field(mu_typed);
        momentum_sum = momentum_sum
            + typed_fvc::div_dev2_grad_transpose(mu_coeff2, u_typed).cast_to::<Force>();
    }
    if with_mms_sources {
        let mms_u =
            TypedFieldRef::<DivDim<Force, Volume>, Vector2>::new(BUOYANT_MMS_SOURCE_U_FIELD);
        momentum_sum = momentum_sum + typed_fvc::source_vector(mms_u, u_typed).cast_to::<Force>();
    }
    let momentum_eqn = momentum_sum.eqn(u_typed);

    // ---- Pressure: as incompressible_momentum. ----
    let p_laplacian_term = typed_fvm::laplacian(rho_dp_coeff, p_typed);
    let p_div_flux_term = typed_fvm::div_flux(phi_typed, p_typed);
    let pressure_eqn = (p_laplacian_term.cast_to::<MassFlux>()
        + p_div_flux_term.cast_to::<MassFlux>())
    .eqn(p_typed);

    // ---- Temperature (divided by cp): advected by the SOLVED flux. ----
    let t_ddt = typed_fvm::ddt_coeff(rho_coeff, t_typed);
    let t_div = typed_fvm::div(phi_typed, t_typed);
    // Runtime uniform param (constants.buoyant_k_over_cp).
    let k_over_cp = TypedCoeff::from_field(TypedFieldRef::<
        DivDim<MulDim<Density, Volume>, MulDim<Length, Time>>,
        Scalar,
    >::new("buoyant_k_over_cp"));
    let t_lap = typed_fvm::laplacian(k_over_cp, t_typed);

    let mut t_sum = t_ddt.cast_to::<TEquationUnit>()
        + t_div.cast_to::<TEquationUnit>()
        + t_lap.cast_to::<TEquationUnit>();
    if with_mms_sources {
        let mms_t = TypedCoeff::from_field(TypedFieldRef::<TSourceUnit, Scalar>::new(
            BUOYANT_MMS_SOURCE_T_FIELD,
        ));
        t_sum = t_sum + typed_fvc::source_coeff(mms_t, t_typed).cast_to::<TEquationUnit>();
    }
    let t_eqn = t_sum.eqn(t_typed);

    let mut system = EquationSystem::new();
    system.add_equation(momentum_eqn);
    system.add_equation(pressure_eqn);
    system.add_equation(t_eqn);
    system
        .validate_units()
        .expect("buoyant incompressible system failed unit validation");
    system
}

pub fn buoyant_incompressible_model() -> Result<ModelSpec, String> {
    buoyant_incompressible_model_impl(false)
}

/// `buoyant_incompressible` plus manufactured source fields for MMS.
pub fn buoyant_incompressible_mms_model() -> Result<ModelSpec, String> {
    buoyant_incompressible_model_impl(true)
}

fn buoyant_incompressible_model_impl(with_mms_sources: bool) -> Result<ModelSpec, String> {
    let system = build_buoyant_system(with_mms_sources);

    let mut layout_fields = vec![
        vol_vector_dim::<Velocity>("U"),
        vol_scalar_dim::<Pressure>("p"),
        vol_scalar_dim::<cfd2_ir::dimensions::D_P>("d_p"),
        vol_vector_dim::<DivDim<Pressure, Length>>("grad_p"),
        vol_vector_dim::<DivDim<Pressure, Length>>("grad_p_old"),
        vol_scalar_dim::<Temperature>(BUOYANT_TEMPERATURE_FIELD),
    ];
    if with_mms_sources {
        layout_fields.push(vol_vector_dim::<DivDim<Force, Volume>>(
            BUOYANT_MMS_SOURCE_U_FIELD,
        ));
        layout_fields.push(vol_scalar_dim::<TSourceUnit>(BUOYANT_MMS_SOURCE_T_FIELD));
    }
    let layout = PortRegistry::from_fields(layout_fields).into_state_layout();

    let derived_rhie_chow =
        crate::solver::model::flux_derivation::derive_rhie_chow(&system, &layout)
            .map_err(|e| format!("failed to derive Rhie–Chow flux: {e}"))?;

    // All four mesh sides are solid no-slip walls; the Inlet/Outlet boundary
    // TYPES are repurposed as the hot/cold isothermal walls (values set via
    // the boundary table API), Wall stays adiabatic (zero-gradient T).
    let mut boundaries = BoundarySpec::default();
    let mut u_spec = FieldBoundarySpec::new();
    for boundary in [
        GpuBoundaryType::Inlet,
        GpuBoundaryType::Outlet,
        GpuBoundaryType::Wall,
        GpuBoundaryType::SlipWall,
        GpuBoundaryType::MovingWall,
    ] {
        u_spec = u_spec.set_uniform(
            boundary,
            2,
            BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
        );
    }
    boundaries.set_field("U", u_spec);

    // Pressure: zero-gradient except at the Outlet type, which pins the
    // gauge with a Dirichlet value (same convention as incompressible
    // momentum; the derived Rhie-Chow flux treats outlet faces as
    // pressure-Dirichlet, so a model without any pressure Dirichlet on an
    // outlet-bearing mesh would have an inconsistent singular pressure
    // system).
    let mut p_spec = FieldBoundarySpec::new().set_uniform(
        GpuBoundaryType::Outlet,
        1,
        BoundaryCondition::dirichlet_dim::<Pressure>(0.0),
    );
    for boundary in [
        GpuBoundaryType::Inlet,
        GpuBoundaryType::Wall,
        GpuBoundaryType::SlipWall,
        GpuBoundaryType::MovingWall,
    ] {
        p_spec = p_spec.set_uniform(
            boundary,
            1,
            BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
        );
    }
    boundaries.set_field("p", p_spec);

    boundaries.set_field(
        BUOYANT_TEMPERATURE_FIELD,
        FieldBoundarySpec::new()
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet_dim::<Temperature>(1.0),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet_dim::<Temperature>(0.0),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Temperature, Length>>(),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Temperature, Length>>(),
            )
            // MovingWall is the THERMALLY HELD wall: Dirichlet T (per-face
            // values settable), no-slip U like Wall. Gives sealed-cavity
            // benchmarks (heated cavity: hot/cold = MovingWall, adiabatic =
            // Wall) without an Outlet's pressure pin. Wall/SlipWall stay
            // adiabatic (zero-gradient).
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::dirichlet_dim::<Temperature>(0.0),
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

    // Schur block layout from coupled FluxLayout RANKS (see the comment at
    // the linear_solver field below).
    let schur_layout = {
        let fl = crate::solver::model::FluxLayout::from_system(&system);
        let ux = fl.offset_for("U_x").ok_or("U_x not in coupled layout")?;
        let uy = fl.offset_for("U_y").ok_or("U_y not in coupled layout")?;
        let t = fl
            .offset_for(BUOYANT_TEMPERATURE_FIELD)
            .ok_or("T not in coupled layout")?;
        let p = fl.offset_for("p").ok_or("p not in coupled layout")?;
        crate::solver::model::linear_solver::SchurBlockLayout::from_u_p(&[ux, uy, t], p)
            .map_err(|e| format!("invalid buoyant SchurBlockLayout: {e}"))?
    };

    Ok(ModelSpec {
        id: if with_mms_sources {
            "buoyant_incompressible_mms"
        } else {
            "buoyant_incompressible"
        },
        system,
        state_layout: layout,
        boundaries,
        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            crate::solver::model::modules::buoyant_ports::buoyant_params_module(),
            flux_module_module,
            crate::solver::model::modules::generic_coupled::generic_coupled_module(method),
            derived_rhie_chow.aux_module,
        ],
        // Schur preconditioning with T in the u-block. The Schur kernels are
        // N-generic (u_index tables, u_len-sized buffers); A_pT/A_Tp blocks are
        // structurally present but zero, so T degenerates to exact Jacobi
        // inside the preconditioner — mathematically benign. Kept for
        // saddle-point robustness, not speed.
        //
        // NOTE: layout indices are coupled FluxLayout RANKS (equation order:
        // U_x=0, U_y=1, p=2, T=3), NOT state-layout offsets (T sits at state
        // offset 8) — the known rank-vs-offset latent-bug class.
        linear_solver: Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
            preconditioner: crate::solver::model::linear_solver::ModelPreconditionerSpec::Schur {
                omega: 1.0,
                sweeps_cap: 64,
                layout: schur_layout,
            },
            ..Default::default()
        }),
        primitives,
        explicit_primitives: None,
        explicit_mass_closure_proof: super::ExplicitMassClosureProof::RuntimePivoted {
            justification: "the buoyant momentum mass is the positive runtime density parameter; generated row-scaled pivot guards enforce its per-stage domain",
        },
    })
}
