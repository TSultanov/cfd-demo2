use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    surface_scalar_dim, vol_scalar_dim, vol_vector_dim, EquationSystem, FieldRef, FluxRef,
};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef, Vector2,
};
// si module no longer needed for boundary conditions - using type-level dimensions
use cfd2_ir::solver::dimensions::{
    Density, DivDim, DynamicViscosity, Force, InvTime, Length, MassFlux, Pressure, Velocity,
};

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};
use crate::solver::model::ports::PortRegistry;

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
            d_p: vol_scalar_dim::<cfd2_ir::solver::dimensions::D_P>("d_p"),
            grad_p: vol_vector_dim::<DivDim<Pressure, Length>>("grad_p"),
            grad_p_old: vol_vector_dim::<DivDim<Pressure, Length>>("grad_p_old"),
        }
    }
}

fn build_incompressible_momentum_system(_fields: &IncompressibleMomentumFields) -> EquationSystem {
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
    let d_p_typed = TypedFieldRef::<cfd2_ir::solver::dimensions::D_P, Scalar>::new("d_p");

    // Build coefficients
    let rho_coeff = TypedCoeff::from_field(rho_typed);
    let mu_coeff = TypedCoeff::from_field(mu_typed);
    let rho_dp_coeff = TypedCoeff::from_field(rho_typed).multiply(TypedCoeff::from_field(d_p_typed));

    // Build momentum equation terms
    // ddt(rho, U): integrated unit is MomentumDensity * Volume / Time = Force
    let ddt_term = typed_fvm::ddt_coeff(rho_coeff, u_typed);

    // div(phi, U): integrated unit is MassFlux * Velocity = Force
    let div_term = typed_fvm::div(phi_typed, u_typed);

    // laplacian(mu, U): integrated unit is DynamicViscosity * Velocity * Area / Length = Force
    let laplacian_term = typed_fvm::laplacian(mu_coeff, u_typed);

    // grad(p): integrated unit is Pressure * Area = Force
    let grad_term = typed_fvc::grad(p_typed);

    // Cast all terms to Force and add
    let momentum_eqn = (ddt_term.cast_to::<Force>()
        + div_term.cast_to::<Force>()
        + laplacian_term.cast_to::<Force>()
        + grad_term.cast_to::<Force>())
    .eqn(u_typed);

    // Build pressure equation terms
    // laplacian(rho*d_p, p): integrated unit is (rho*d_p) * Pressure * Area / Length = MassFlux
    // where rho*d_p has units: Density * (Volume*Time/Mass) = Time (since Volume/Mass = 1/Density)
    // So the unit is: Time * Pressure * Area / Length = Time * (Mass/(Length*Time^2)) * Length
    // = Mass / Time = MassFlux
    let p_laplacian_term = typed_fvm::laplacian(rho_dp_coeff, p_typed);

    // div_flux(phi, p): integrated unit is MassFlux
    let p_div_flux_term = typed_fvm::div_flux(phi_typed, p_typed);

    // Cast all terms to MassFlux and add
    let pressure_eqn = (p_laplacian_term.cast_to::<MassFlux>() + p_div_flux_term.cast_to::<MassFlux>()).eqn(p_typed);

    let mut system = EquationSystem::new();
    system.add_equation(momentum_eqn);
    system.add_equation(pressure_eqn);

    // Validate units to ensure the system is consistent (debug builds only)
    #[cfg(debug_assertions)]
    {
        system
            .validate_units()
            .expect("incompressible momentum system failed unit validation");
    }

    system
}

pub fn incompressible_momentum_system() -> EquationSystem {
    let fields = IncompressibleMomentumFields::new();
    build_incompressible_momentum_system(&fields)
}

pub fn incompressible_momentum_model() -> ModelSpec {
    let fields = IncompressibleMomentumFields::new();
    let system = build_incompressible_momentum_system(&fields);
    let layout = StateLayout::new(vec![
        fields.u,
        fields.p,
        fields.d_p,
        fields.grad_p,
        fields.grad_p_old,
    ]);
    let flux_kernel = rhie_chow_flux_module_kernel(&system, &layout)
        .expect("failed to derive Rhie–Chow flux formula from model system/layout");

    // Port-based validation and offset resolution (replaces ad-hoc StateLayout lookups)
    let (u0, u1, p) = {
        use crate::solver::model::ports::{PortRegistry, Pressure, Velocity};

        let mut registry = PortRegistry::new(layout.clone());

        // Validate required fields with clear errors
        registry
            .validate_vector2_field::<Velocity>("incompressible_momentum_model", "U")
            .expect("state layout validation failed");
        registry
            .validate_scalar_field::<Pressure>("incompressible_momentum_model", "p")
            .expect("state layout validation failed");

        // Resolve offsets via ports (no StateLayout probing in this function)
        let u_port = registry
            .register_vector2_field::<Velocity>("U")
            .expect("U field registration failed");
        let p_port = registry
            .register_scalar_field::<Pressure>("p")
            .expect("p field registration failed");

        let u0 = u_port
            .component(0)
            .expect("U component 0")
            .full_offset();
        let u1 = u_port
            .component(1)
            .expect("U component 1")
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
        kernel: flux_kernel,
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
    .expect("failed to build flux_module module");

    // Build rhie_chow module before system is moved into ModelSpec
    let rhie_chow_module =
        crate::solver::model::modules::rhie_chow::rhie_chow_aux_module(&system, "d_p", true, true)
            .expect("failed to create rhie_chow_aux_module");

    let model = ModelSpec {
        id: "incompressible_momentum",
        system,
        state_layout: layout,
        boundaries,

        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            flux_module_module,
            crate::solver::model::modules::generic_coupled::generic_coupled_module(method),
            rhie_chow_module,
        ],
        // The generic coupled path needs a saddle-point-capable preconditioner.
        linear_solver: Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
            preconditioner: crate::solver::model::linear_solver::ModelPreconditionerSpec::Schur {
                omega: 1.0,
                layout: crate::solver::model::linear_solver::SchurBlockLayout::from_u_p(
                    &[u0, u1],
                    p,
                )
                .expect("invalid SchurBlockLayout"),
            },
            ..Default::default()
        }),
        primitives,
    };

    model
}

#[derive(Debug, Clone)]
struct RhieChowFields {
    momentum: String,
    pressure: String,
    d_p: String,
}

fn rhie_chow_flux_module_kernel(
    system: &EquationSystem,
    layout: &StateLayout,
) -> Result<crate::solver::ir::FluxModuleKernelSpec, String> {
    use crate::solver::ir::{
        FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxModuleKernelSpec,
    };
    use crate::solver::model::backend::{Coefficient as BackendCoeff, FieldKind, FieldRef, TermOp};
    use std::collections::{HashMap, HashSet};

    fn collect_coeff_fields(coeff: &BackendCoeff, out: &mut Vec<FieldRef>) {
        match coeff {
            BackendCoeff::Constant { .. } => {}
            BackendCoeff::Field(field) => out.push(*field),
            BackendCoeff::MagSqr(field) => out.push(*field),
            BackendCoeff::Product(lhs, rhs) => {
                collect_coeff_fields(lhs, out);
                collect_coeff_fields(rhs, out);
            }
        }
    }

    // Create a PortRegistry for runtime validation of fields
    let registry = PortRegistry::new(layout.clone());

    fn density_face_expr(registry: &PortRegistry) -> Result<S, String> {
        // Prefer a state-layout density when present (variable-density extension);
        // otherwise fall back to the global constant density uniform.
        // Use PortRegistry validation to check for existence, kind, and dimension.
        match registry.validate_scalar_field::<Density>(
            "rhie_chow_flux_module_kernel",
            "rho",
        ) {
            Ok(()) => Ok(S::Lerp(
                Box::new(S::state(FaceSide::Owner, "rho")),
                Box::new(S::state(FaceSide::Neighbor, "rho")),
            )),
            Err(crate::solver::model::ports::PortValidationError::MissingField { .. }) => {
                // rho is not in state layout; fall back to uniform density
                Ok(S::constant("density"))
            }
            Err(e) => Err(format!(
                "rho field exists but has wrong kind or unit: {}",
                e
            )),
        }
    }

    // Infer (momentum, pressure) coupling from the declared equation system.
    let equations = system.equations();
    let mut eq_by_target: HashMap<FieldRef, usize> = HashMap::new();
    for (idx, eq) in equations.iter().enumerate() {
        eq_by_target.insert(*eq.target(), idx);
    }

    let mut candidates = Vec::new();
    for eq in equations {
        if !matches!(eq.target().kind(), FieldKind::Vector2 | FieldKind::Vector3) {
            continue;
        }

        let has_transport = eq
            .terms()
            .iter()
            .any(|t| matches!(t.op, TermOp::Div | TermOp::Laplacian));
        if !has_transport {
            continue;
        }

        let mut grad_scalars: HashSet<FieldRef> = HashSet::new();
        for term in eq.terms() {
            if term.op == TermOp::Grad && term.field.kind() == FieldKind::Scalar {
                grad_scalars.insert(term.field);
            }
        }

        for pressure in grad_scalars {
            let Some(&p_eq_idx) = eq_by_target.get(&pressure) else {
                continue;
            };
            let p_eq = &equations[p_eq_idx];
            let p_has_laplacian = p_eq.terms().iter().any(|t| t.op == TermOp::Laplacian);
            if p_has_laplacian {
                candidates.push((eq.target().name().to_string(), pressure.name().to_string()));
            }
        }
    }

    let (momentum, pressure) = match candidates.as_slice() {
        [(m, p)] => (m.clone(), p.clone()),
        [] => return Err("no unique momentum-pressure coupling found for Rhie–Chow".to_string()),
        many => {
            return Err(format!(
                "Rhie–Chow requires a unique momentum-pressure coupling, found {} candidates: [{}]",
                many.len(),
                many.iter()
                    .map(|(m, p)| format!("{m}↔{p}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    };

    // Infer d_p from the pressure equation laplacian coefficient: pick the unique scalar field
    // referenced by that coefficient that is present in the model's state layout.
    let pressure_eq = equations
        .iter()
        .find(|eq| eq.target().name() == pressure)
        .ok_or_else(|| {
            format!("missing pressure equation for inferred pressure field '{pressure}'")
        })?;
    let pressure_laplacian = pressure_eq
        .terms()
        .iter()
        .find(|t| t.op == TermOp::Laplacian)
        .ok_or_else(|| {
            format!("pressure equation for '{pressure}' must include a laplacian term")
        })?;
    let Some(coeff) = &pressure_laplacian.coeff else {
        return Err(format!(
            "pressure laplacian coefficient for '{pressure}' is missing"
        ));
    };
    let mut coeff_fields = Vec::new();
    collect_coeff_fields(coeff, &mut coeff_fields);
    // Use PortRegistry validation to filter for scalar fields with D_P dimension
    let mut d_p_candidates: Vec<String> = Vec::new();
    for f in &coeff_fields {
        match registry.validate_scalar_field::<cfd2_ir::solver::dimensions::D_P>(
            "rhie_chow_flux_module_kernel",
            f.name(),
        ) {
            Ok(()) => d_p_candidates.push(f.name().to_string()),
            Err(crate::solver::model::ports::PortValidationError::MissingField { .. }) => {
                // Field is not in state layout; not a candidate
            }
            Err(_) => {
                // Field exists but has wrong kind or unit; not a candidate
            }
        }
    }
    let d_p = match d_p_candidates.as_slice() {
        [only] => only.clone(),
        [] => {
            return Err(format!(
                "pressure laplacian coefficient for '{pressure}' does not reference any state-layout scalar fields with D_P units"
            ));
        }
        many => {
            return Err(format!(
                "pressure laplacian coefficient for '{pressure}' references multiple state-layout fields with D_P units; cannot derive unique d_p: [{}]",
                many.join(", ")
            ));
        }
    };

    let fields = RhieChowFields {
        momentum,
        pressure,
        d_p,
    };

    // Rhie–Chow-style mass flux:
    //   phi = rho * (u_f · n) * area  -  rho * d_p_f * ((p_N - p_O) / dist) * area
    //
    // Notes:
    // - `d_p` is inferred from the pressure equation Laplacian coefficient and is expected to be
    //   updated by the coupled pressure/momentum preconditioner (and seeded by `dp_init`).
    // - The pressure gradient term uses the same face-normal distance projection (`dist`) as the
    //   Laplacian discretization, so the pressure equation's Laplacian term and this correction
    //   term stay numerically consistent.
    //
    // This definition is intentionally "general": it only relies on the model-declared
    // momentum/pressure coupling and the presence of `(d_p, grad_p)` in the state layout.
    let rho_face = density_face_expr(&registry)?;

    let d_p_face = S::Lerp(
        Box::new(S::state(FaceSide::Owner, fields.d_p.clone())),
        Box::new(S::state(FaceSide::Neighbor, fields.d_p.clone())),
    );

    // Rhie–Chow uses the momentum predictor `HbyA` for the "predicted" mass flux on the RHS
    // of the pressure equation, then subtracts an explicit pressure correction flux.
    //
    // Approximate `HbyA` from the current cell-centered velocity and pressure gradient:
    //   HbyA ≈ U + d_p * grad(p)
    let grad_p_field = format!("grad_{}", fields.pressure);
    let u_face = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Owner, fields.momentum.clone())),
        Box::new(V::state_vec2(FaceSide::Neighbor, fields.momentum.clone())),
    );
    let grad_p_face = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Owner, grad_p_field.clone())),
        Box::new(V::state_vec2(FaceSide::Neighbor, grad_p_field)),
    );
    let hby_a_face = V::Add(
        Box::new(u_face),
        Box::new(V::MulScalar(
            Box::new(grad_p_face),
            Box::new(d_p_face.clone()),
        )),
    );
    let u_n = S::Dot(Box::new(hby_a_face), Box::new(V::normal()));
    let phi_pred = S::Mul(
        Box::new(S::Mul(Box::new(rho_face.clone()), Box::new(u_n))),
        Box::new(S::area()),
    );
    let dp = S::Sub(
        Box::new(S::state(FaceSide::Neighbor, fields.pressure.clone())),
        Box::new(S::state(FaceSide::Owner, fields.pressure.clone())),
    );
    let dp_over_dist = S::Div(Box::new(dp), Box::new(S::dist()));
    let phi_p = S::Mul(
        Box::new(S::Mul(
            Box::new(S::Mul(Box::new(rho_face), Box::new(d_p_face))),
            Box::new(dp_over_dist),
        )),
        Box::new(S::area()),
    );
    let phi_corr = S::Sub(Box::new(phi_pred.clone()), Box::new(phi_p));

    // Pressure equation needs the *predicted* mass flux (phi_pred) on the RHS:
    //   -div(rho*d_p*grad(p)) + div(phi_pred) = 0
    //
    // Momentum equation convection uses the corrected mass flux (phi_corr) to reduce
    // pressure–velocity decoupling on collocated grids (Rhie–Chow).
    //
    // The flux buffer is indexed by coupled unknown component, so we can provide different
    // values for `p` vs `U` while still using a single scalar flux module kernel.
    let flux_layout = crate::solver::ir::FluxLayout::from_system(system);
    let components: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();
    let flux: Vec<S> = components
        .iter()
        .map(|name| {
            if name == &fields.pressure {
                phi_pred.clone()
            } else {
                phi_corr.clone()
            }
        })
        .collect();

    Ok(FluxModuleKernelSpec::ScalarPerComponent { components, flux })
}
