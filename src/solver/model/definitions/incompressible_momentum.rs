use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    fvm, surface_scalar, vol_scalar, vol_vector, Coefficient, EquationSystem, FieldRef, FluxRef,
};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::units::si;

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
    pub grad_component: FieldRef,
}

impl IncompressibleMomentumFields {
    pub fn new() -> Self {
        Self {
            u: vol_vector("U", si::VELOCITY),
            p: vol_scalar("p", si::PRESSURE),
            phi: surface_scalar("phi", si::MASS_FLUX),
            mu: vol_scalar("mu", si::DYNAMIC_VISCOSITY),
            rho: vol_scalar("rho", si::DENSITY),
            d_p: vol_scalar("d_p", si::D_P),
            grad_p: vol_vector("grad_p", si::PRESSURE_GRADIENT),
            grad_component: vol_vector("grad_component", si::INV_TIME),
        }
    }
}

fn build_incompressible_momentum_system(fields: &IncompressibleMomentumFields) -> EquationSystem {
    let momentum = (fvm::ddt_coeff(
        Coefficient::field(fields.rho).expect("rho must be scalar"),
        fields.u,
    ) + fvm::div(fields.phi, fields.u)
        + fvm::laplacian(
            Coefficient::field(fields.mu).expect("mu must be scalar"),
            fields.u,
        )
        + fvm::grad(fields.p))
    .eqn(fields.u);

    let pressure = (fvm::laplacian(
        Coefficient::product(
            Coefficient::field(fields.rho).expect("rho must be scalar"),
            Coefficient::field(fields.d_p).expect("d_p must be scalar"),
        )
        .expect("pressure coefficient must be scalar"),
        fields.p,
    )
    // SIMPLE-style pressure correction source term.
    //
    // Enforce continuity via div(phi) = 0 by solving a Poisson equation whose RHS is the
    // divergence of the current face mass flux.
    + fvm::div_flux(fields.phi, fields.p))
    .eqn(fields.p);

    let mut system = EquationSystem::new();
    system.add_equation(momentum);
    system.add_equation(pressure);
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
        fields.grad_component,
    ]);
    let flux_kernel = rhie_chow_flux_module_kernel(&system, &layout)
        .expect("failed to derive Rhie–Chow flux formula from model system/layout");

    let u0 = layout
        .component_offset("U", 0)
        .ok_or_else(|| "incompressible_momentum_model missing U[0] in state layout".to_string())
        .expect("state layout validation failed");
    let u1 = layout
        .component_offset("U", 1)
        .ok_or_else(|| "incompressible_momentum_model missing U[1] in state layout".to_string())
        .expect("state layout validation failed");
    let p = layout
        .offset_for("p")
        .ok_or_else(|| "incompressible_momentum_model missing p in state layout".to_string())
        .expect("state layout validation failed");

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "U",
        FieldBoundarySpec::new()
            // Inlet velocity is Dirichlet; the value can be updated at runtime via the solver's
            // boundary table API (e.g. `set_boundary_vec2(GpuBoundaryType::Inlet, "U", ...)`).
            .set_uniform(
                GpuBoundaryType::Inlet,
                2,
                BoundaryCondition::dirichlet(0.0, si::VELOCITY),
            )
            // Outlet: do not constrain velocity (Neumann/zeroGradient).
            .set_uniform(
                GpuBoundaryType::Outlet,
                2,
                BoundaryCondition::zero_gradient(si::INV_TIME),
            )
            // Walls: no-slip.
            .set_uniform(
                GpuBoundaryType::Wall,
                2,
                BoundaryCondition::dirichlet(0.0, si::VELOCITY),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                2,
                BoundaryCondition::zero_gradient(si::INV_TIME),
            ),
    );
    boundaries.set_field(
        "p",
        FieldBoundarySpec::new()
            // Inlet and walls: zero-gradient pressure.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::zero_gradient(si::PRESSURE_GRADIENT),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::PRESSURE_GRADIENT),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient(si::PRESSURE_GRADIENT),
            )
            // Outlet: fix gauge by pinning pressure to 0.
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::PRESSURE),
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
        kernel: flux_kernel,
    };

    let model = ModelSpec {
        id: "incompressible_momentum",
        system,
        state_layout: layout,
        boundaries,

        modules: vec![
            crate::solver::model::modules::eos::eos_module(crate::solver::model::eos::EosSpec::Constant),
            crate::solver::model::modules::flux_module::flux_module_module(flux_module)
                .expect("failed to build flux_module module"),
            crate::solver::model::modules::generic_coupled::generic_coupled_module(method),
            crate::solver::model::modules::rhie_chow::rhie_chow_aux_module(
                "d_p",
                true,
                true,
            ),
        ],
        // The generic coupled path needs a saddle-point-capable preconditioner.
        linear_solver: Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
            preconditioner: crate::solver::model::linear_solver::ModelPreconditionerSpec::Schur {
                omega: 1.0,
                layout: crate::solver::model::linear_solver::SchurBlockLayout::from_u_p(&[u0, u1], p)
                    .expect("invalid SchurBlockLayout"),
            },
            ..Default::default()
        }),
        primitives: crate::solver::model::primitives::PrimitiveDerivations::identity(),
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

    fn density_face_expr(layout: &StateLayout) -> S {
        // Prefer a state-layout density when present (variable-density extension);
        // otherwise fall back to the global constant density uniform.
        if let Some(rho) = layout.field("rho") {
            if rho.kind() == FieldKind::Scalar {
                return S::Lerp(
                    Box::new(S::state(FaceSide::Owner, "rho")),
                    Box::new(S::state(FaceSide::Neighbor, "rho")),
                );
            }
        }
        S::constant("density")
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
        .ok_or_else(|| format!("missing pressure equation for inferred pressure field '{pressure}'"))?;
    let pressure_laplacian = pressure_eq
        .terms()
        .iter()
        .find(|t| t.op == TermOp::Laplacian)
        .ok_or_else(|| format!("pressure equation for '{pressure}' must include a laplacian term"))?;
    let Some(coeff) = &pressure_laplacian.coeff else {
        return Err(format!("pressure laplacian coefficient for '{pressure}' is missing"));
    };
    let mut coeff_fields = Vec::new();
    collect_coeff_fields(coeff, &mut coeff_fields);
    let layout_coeff_fields: Vec<_> = coeff_fields
        .into_iter()
        .filter(|f| layout.field(f.name()).is_some())
        .collect();
    let d_p = match layout_coeff_fields.as_slice() {
        [only] => only.name().to_string(),
        [] => {
            return Err(format!(
                "pressure laplacian coefficient for '{pressure}' does not reference any state-layout scalar fields"
            ));
        }
        many => {
            return Err(format!(
                "pressure laplacian coefficient for '{pressure}' references multiple state-layout fields; cannot derive unique d_p: [{}]",
                many.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
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
    let u_face = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Owner, fields.momentum.clone())),
        Box::new(V::state_vec2(FaceSide::Neighbor, fields.momentum.clone())),
    );
    let u_n = S::Dot(Box::new(u_face), Box::new(V::normal()));
    let rho_face = density_face_expr(layout);
    let phi_pred = S::Mul(
        Box::new(S::Mul(Box::new(rho_face.clone()), Box::new(u_n))),
        Box::new(S::area()),
    );

    let d_p_face = S::Lerp(
        Box::new(S::state(FaceSide::Owner, fields.d_p.clone())),
        Box::new(S::state(FaceSide::Neighbor, fields.d_p.clone())),
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
    let phi = S::Sub(Box::new(phi_pred), Box::new(phi_p));

    Ok(FluxModuleKernelSpec::ScalarReplicated { phi })
}
