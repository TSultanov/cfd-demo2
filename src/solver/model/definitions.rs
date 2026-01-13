use super::backend::ast::{
    fvm, surface_scalar, surface_vector, vol_scalar, vol_vector, Coefficient, EquationSystem,
    FieldRef, FluxRef,
};
use super::backend::state_layout::StateLayout;
use super::kernel::KernelPlan;
use crate::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use crate::solver::model::gpu_spec::{FluxSpec, GradientStorage, ModelGpuSpec};
use crate::solver::units::{si, UnitDim};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: &'static str,
    pub method: crate::solver::model::method::MethodSpec,
    pub eos: crate::solver::model::eos::EosSpec,
    pub system: EquationSystem,
    pub state_layout: StateLayout,
    pub boundaries: BoundarySpec,

    /// Optional model-owned linear solver configuration.
    ///
    /// When present, this is treated as authoritative by solver families that
    /// support model-driven solver selection.
    pub linear_solver: Option<crate::solver::model::linear_solver::ModelLinearSolverSpec>,

    /// Flux computation module (optional: for pure diffusion models, set to None)
    pub flux_module: Option<crate::solver::model::flux_module::FluxModuleSpec>,

    /// Derived primitive recovery expressions (empty if primitives = conserved state)
    pub primitives: crate::solver::model::primitives::PrimitiveDerivations,

    pub gpu: crate::solver::model::gpu_spec::ModelGpuSpec,
}

impl ModelSpec {
    pub fn with_derived_gpu(mut self) -> Self {
        self.gpu = self.derive_gpu_spec();
        self
    }

    pub fn derive_gpu_spec(&self) -> ModelGpuSpec {
        // Flux storage is model-driven.
        //
        // - If the model has an explicit flux module, store face fluxes for each coupled
        //   unknown component.
        let flux = self.flux_module.as_ref().map(|_| FluxSpec {
            // Flux modules write into a packed per-unknown-component face flux table.
            //
            // Even if the module computes a single scalar face flux (phi), it is replicated
            // into all coupled unknown-component slots so assembly can remain layout-driven.
            stride: self.system.unknowns_per_cell(),
        });

        // Low-Mach parameters are currently only used by compressible models/tests.
        let requires_low_mach_params = matches!(
            self.eos,
            crate::solver::model::eos::EosSpec::IdealGas { .. }
        );

        let gradient_storage = match self.method {
            crate::solver::model::method::MethodSpec::GenericCoupled
            | crate::solver::model::method::MethodSpec::GenericCoupledImplicit { .. }
            | crate::solver::model::method::MethodSpec::CoupledIncompressible => {
                GradientStorage::PackedState
            }
        };

        let required_gradient_fields = Vec::new();

        ModelGpuSpec {
            flux,
            requires_low_mach_params,
            gradient_storage,
            required_gradient_fields,
        }
    }

    pub fn kernel_plan(&self) -> KernelPlan {
        super::kernel::derive_kernel_plan_for_model(self)
    }
}

#[derive(Debug, Clone, Default)]
pub struct BoundarySpec {
    pub fields: HashMap<String, FieldBoundarySpec>,
}

impl BoundarySpec {
    pub fn set_field(&mut self, name: impl Into<String>, spec: FieldBoundarySpec) {
        self.fields.insert(name.into(), spec);
    }

    pub fn field(&self, name: &str) -> Option<&FieldBoundarySpec> {
        self.fields.get(name)
    }

    pub fn to_gpu_tables(&self, system: &EquationSystem) -> Result<(Vec<u32>, Vec<f32>), String> {
        let mut unknowns: Vec<(FieldRef, usize)> = Vec::new();
        for eqn in system.equations() {
            let field = eqn.target();
            for component in 0..field.kind().component_count() {
                unknowns.push((*field, component));
            }
        }

        let coupled_stride = unknowns.len();
        let mut kind = vec![GpuBcKind::ZeroGradient as u32; 4 * coupled_stride];
        let mut value = vec![0.0_f32; 4 * coupled_stride];

        let boundary_types = [
            GpuBoundaryType::None,
            GpuBoundaryType::Inlet,
            GpuBoundaryType::Outlet,
            GpuBoundaryType::Wall,
        ];

        for (b_i, &b) in boundary_types.iter().enumerate() {
            for (u_idx, (field, component)) in unknowns.iter().enumerate() {
                let entry = if let Some(spec) = self.field(field.name()) {
                    if let Some(conditions) = spec.by_boundary.get(&b) {
                        let expected_components = field.kind().component_count();
                        if conditions.len() != expected_components {
                            return Err(format!(
                                "boundary spec for field '{}' on {:?} has {} components, expected {}",
                                field.name(),
                                b,
                                conditions.len(),
                                expected_components
                            ));
                        }
                        conditions.get(*component).cloned()
                    } else {
                        None
                    }
                } else {
                    None
                };

                let expected_unit = match entry.as_ref().map(|c| c.kind) {
                    Some(GpuBcKind::Dirichlet) => field.unit(),
                    Some(GpuBcKind::Neumann) | Some(GpuBcKind::ZeroGradient) | None => {
                        field.unit() / si::LENGTH
                    }
                };

                if let Some(cond) = entry {
                    if cond.unit != expected_unit {
                        return Err(format!(
                            "boundary units mismatch for field '{}': got {}, expected {} for {:?}",
                            field.name(),
                            cond.unit,
                            expected_unit,
                            cond.kind
                        ));
                    }
                    kind[b_i * coupled_stride + u_idx] = cond.kind as u32;
                    value[b_i * coupled_stride + u_idx] = cond.value as f32;
                } else {
                    // Default: ZeroGradient (Neumann=0), with expected unit field.unit()/L.
                    kind[b_i * coupled_stride + u_idx] = GpuBcKind::ZeroGradient as u32;
                    value[b_i * coupled_stride + u_idx] = 0.0;
                }
            }
        }

        Ok((kind, value))
    }
}

#[derive(Debug, Clone)]
pub struct FieldBoundarySpec {
    /// Per boundary type (Inlet/Outlet/Wall), per component (scalar=1, vec2=2).
    pub by_boundary: HashMap<GpuBoundaryType, Vec<BoundaryCondition>>,
}

impl FieldBoundarySpec {
    pub fn new() -> Self {
        Self {
            by_boundary: HashMap::new(),
        }
    }

    pub fn set_uniform(
        mut self,
        boundary: GpuBoundaryType,
        components: usize,
        condition: BoundaryCondition,
    ) -> Self {
        self.by_boundary
            .insert(boundary, vec![condition; components]);
        self
    }
}

#[derive(Debug, Clone)]
pub struct BoundaryCondition {
    pub kind: GpuBcKind,
    pub value: f64,
    pub unit: UnitDim,
}

impl BoundaryCondition {
    pub fn zero_gradient(unit: UnitDim) -> Self {
        Self {
            kind: GpuBcKind::ZeroGradient,
            value: 0.0,
            unit,
        }
    }

    pub fn dirichlet(value: f64, unit: UnitDim) -> Self {
        Self {
            kind: GpuBcKind::Dirichlet,
            value,
            unit,
        }
    }

    /// Value is `dphi/dn` (outward normal gradient).
    pub fn neumann(dphi_dn: f64, unit: UnitDim) -> Self {
        Self {
            kind: GpuBcKind::Neumann,
            value: dphi_dn,
            unit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenericCoupledFields {
    pub state: Vec<FieldRef>,
}

impl GenericCoupledFields {
    pub fn new(state: Vec<FieldRef>) -> Self {
        Self { state }
    }
}

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

#[derive(Debug, Clone)]
pub struct CompressibleFields {
    pub rho: FieldRef,
    pub rho_u: FieldRef,
    pub rho_e: FieldRef,
    pub p: FieldRef,
    pub u: FieldRef,
    pub mu: FieldRef,
    pub phi_rho: FluxRef,
    pub phi_rho_u: FluxRef,
    pub phi_rho_e: FluxRef,
}

impl CompressibleFields {
    pub fn new() -> Self {
        Self {
            rho: vol_scalar("rho", si::DENSITY),
            rho_u: vol_vector("rho_u", si::MOMENTUM_DENSITY),
            rho_e: vol_scalar("rho_e", si::ENERGY_DENSITY),
            p: vol_scalar("p", si::PRESSURE),
            u: vol_vector("u", si::VELOCITY),
            mu: vol_scalar("mu", si::DYNAMIC_VISCOSITY),
            phi_rho: surface_scalar("phi_rho", si::MASS_FLUX),
            phi_rho_u: surface_vector("phi_rho_u", si::FORCE),
            phi_rho_e: surface_scalar("phi_rho_e", si::POWER),
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

fn build_compressible_system(fields: &CompressibleFields) -> EquationSystem {
    let rho_eqn =
        (fvm::ddt(fields.rho) + fvm::div_flux(fields.phi_rho, fields.rho)).eqn(fields.rho);
    let rho_u_eqn =
        (fvm::ddt(fields.rho_u) + fvm::div_flux(fields.phi_rho_u, fields.rho_u)).eqn(fields.rho_u);
    let rho_e_eqn =
        (fvm::ddt(fields.rho_e) + fvm::div_flux(fields.phi_rho_e, fields.rho_e)).eqn(fields.rho_e);

    let mut system = EquationSystem::new();
    system.add_equation(rho_eqn);
    system.add_equation(rho_u_eqn);
    system.add_equation(rho_e_eqn);
    system
}

pub fn incompressible_momentum_system() -> EquationSystem {
    let fields = IncompressibleMomentumFields::new();
    build_incompressible_momentum_system(&fields)
}

pub fn compressible_system() -> EquationSystem {
    let fields = CompressibleFields::new();
    build_compressible_system(&fields)
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
            // Inlet velocity is applied via the runtime `InletVelocity` plan param, which updates
            // the GPU `bc_value` table for `GpuBoundaryType::Inlet` (see `param_inlet_velocity`).
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
            // Outlet: fix gauge by pinning pressure to 0.
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::PRESSURE),
            ),
    );

    ModelSpec {
        id: "incompressible_momentum",
        method: crate::solver::model::method::MethodSpec::CoupledIncompressible,
        eos: crate::solver::model::eos::EosSpec::Constant,
        system,
        state_layout: layout,
        boundaries,

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
        }),
        flux_module: Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel {
            gradients: Some(
                crate::solver::model::flux_module::FluxModuleGradientsSpec::FromStateLayout,
            ),
            kernel: flux_kernel,
        }),
        primitives: crate::solver::model::primitives::PrimitiveDerivations::identity(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}

pub fn incompressible_momentum_generic_model() -> ModelSpec {
    let mut model = incompressible_momentum_model();
    model.id = "incompressible_momentum_generic";

    model.with_derived_gpu()
}

pub fn compressible_model() -> ModelSpec {
    let fields = CompressibleFields::new();
    let system = build_compressible_system(&fields);
    let layout = StateLayout::new(vec![
        fields.rho,
        fields.rho_u,
        fields.rho_e,
        fields.p,
        fields.u,
    ]);

    let gamma = 1.4;
    let flux_kernel = compressible_euler_central_upwind_flux_module_kernel(&system, &fields, gamma)
        .expect("failed to build compressible flux kernel spec");

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "rho",
        FieldBoundarySpec::new()
            // Inlet density is driven via the runtime `Density` plan param, which updates the
            // GPU `bc_value` table for `GpuBoundaryType::Inlet` (see `param_density`).
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(1.0, si::DENSITY),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::zero_gradient(si::DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::DENSITY / si::LENGTH),
            ),
    );
    boundaries.set_field(
        "rho_u",
        FieldBoundarySpec::new()
            // Inlet momentum density is driven via the runtime `InletVelocity` plan param
            // (see `param_inlet_velocity`); initial value is a placeholder.
            .set_uniform(
                GpuBoundaryType::Inlet,
                2,
                BoundaryCondition::dirichlet(0.0, si::MOMENTUM_DENSITY),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                2,
                BoundaryCondition::zero_gradient(si::MOMENTUM_DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                2,
                BoundaryCondition::zero_gradient(si::MOMENTUM_DENSITY / si::LENGTH),
            ),
    );
    boundaries.set_field(
        "rho_e",
        FieldBoundarySpec::new()
            // Inlet energy is updated alongside rho/rho_u when inlet parameters change.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::ENERGY_DENSITY),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::zero_gradient(si::ENERGY_DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::ENERGY_DENSITY / si::LENGTH),
            ),
    );

    ModelSpec {
        id: "compressible",
        // Route compressible through the generic coupled pipeline; KT flux + primitive recovery
        // remain transitional kernels behind model-driven IDs.
        method: crate::solver::model::method::MethodSpec::GenericCoupledImplicit { outer_iters: 1 },
        eos: crate::solver::model::eos::EosSpec::IdealGas { gamma },
        system,
        state_layout: layout,
        boundaries,

        linear_solver: None,
        flux_module: Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel {
            gradients: None,
            kernel: flux_kernel,
        }),
        primitives: crate::solver::model::primitives::PrimitiveDerivations::euler_ideal_gas(gamma),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}

#[derive(Debug, Clone)]
struct RhieChowFields {
    momentum: String,
    pressure: String,
    d_p: String,
    grad_p: String,
}

fn rhie_chow_flux_module_kernel(
    system: &EquationSystem,
    layout: &StateLayout,
) -> Result<crate::solver::ir::FluxModuleKernelSpec, String> {
    use super::backend::{Coefficient as BackendCoeff, FieldKind, FieldRef, TermOp};
    use crate::solver::ir::{
        FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxModuleKernelSpec,
    };
    use std::collections::{HashMap, HashSet};

    fn collect_coeff_fields(coeff: &BackendCoeff, out: &mut Vec<FieldRef>) {
        match coeff {
            BackendCoeff::Constant { .. } => {}
            BackendCoeff::Field(field) => out.push(*field),
            BackendCoeff::Product(lhs, rhs) => {
                collect_coeff_fields(lhs, out);
                collect_coeff_fields(rhs, out);
            }
        }
    }

    fn find_grad_field_for_scalar(
        layout: &StateLayout,
        scalar: &str,
        unit: UnitDim,
    ) -> Result<String, String> {
        let expected = format!("grad_{scalar}");
        if let Some(f) = layout.field(&expected) {
            if f.kind() == FieldKind::Vector2 {
                return Ok(expected);
            }
        }

        let mut candidates = Vec::new();
        for f in layout.fields() {
            if f.kind() != FieldKind::Vector2 {
                continue;
            }
            if f.unit() != unit {
                continue;
            }
            candidates.push(f.name().to_string());
        }

        match candidates.as_slice() {
            [only] => Ok(only.clone()),
            [] => Err(format!(
                "state layout missing required gradient field for '{scalar}' (expected '{expected}' or a unique Vector2 field with unit {unit})"
            )),
            many => Err(format!(
                "state layout has multiple candidate gradient fields for '{scalar}' (unit {unit}); add an explicit '{expected}' field or disambiguate: [{}]",
                many.join(", ")
            )),
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

    let grad_p = find_grad_field_for_scalar(layout, &pressure, si::PRESSURE_GRADIENT)?;

    let fields = RhieChowFields {
        momentum,
        pressure,
        d_p,
        grad_p,
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

fn compressible_euler_central_upwind_flux_module_kernel(
    system: &EquationSystem,
    fields: &CompressibleFields,
    gamma: f32,
) -> Result<crate::solver::ir::FluxModuleKernelSpec, String> {
    use crate::solver::ir::{
        FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec,
    };

    let flux_layout = FluxLayout::from_system(system);
    let components: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();

    let ex = V::vec2(S::lit(1.0), S::lit(0.0));
    let ey = V::vec2(S::lit(0.0), S::lit(1.0));

    let rho_name = fields.rho.name();
    let rho_u_name = fields.rho_u.name();
    let rho_e_name = fields.rho_e.name();

    let rho = |side: FaceSide| S::state(side, rho_name);
    let rho_e = |side: FaceSide| S::state(side, rho_e_name);
    let rho_u = |side: FaceSide| V::state_vec2(side, rho_u_name);
    let rho_u_x = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ex.clone()));
    let rho_u_y = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ey.clone()));

    let inv_rho = |side: FaceSide| S::Div(Box::new(S::lit(1.0)), Box::new(rho(side)));
    let u_vec = |side: FaceSide| V::MulScalar(Box::new(rho_u(side)), Box::new(inv_rho(side)));
    let u_n = |side: FaceSide| S::Dot(Box::new(u_vec(side)), Box::new(V::normal()));

    let p = |side: FaceSide| S::primitive(side, "p");
    let c = |side: FaceSide| {
        S::Sqrt(Box::new(S::Div(
            Box::new(S::Mul(Box::new(S::lit(gamma)), Box::new(p(side)))),
            Box::new(rho(side)),
        )))
    };

    let n_x = S::Dot(Box::new(V::normal()), Box::new(ex.clone()));
    let n_y = S::Dot(Box::new(V::normal()), Box::new(ey.clone()));

    let flux_mass = |side: FaceSide| S::Mul(Box::new(rho(side)), Box::new(u_n(side)));
    let flux_mom_x = |side: FaceSide| {
        S::Add(
            Box::new(S::Mul(Box::new(rho_u_x(side)), Box::new(u_n(side)))),
            Box::new(S::Mul(Box::new(p(side)), Box::new(n_x.clone()))),
        )
    };
    let flux_mom_y = |side: FaceSide| {
        S::Add(
            Box::new(S::Mul(Box::new(rho_u_y(side)), Box::new(u_n(side)))),
            Box::new(S::Mul(Box::new(p(side)), Box::new(n_y.clone()))),
        )
    };
    let flux_energy = |side: FaceSide| {
        S::Mul(
            Box::new(S::Add(Box::new(rho_e(side)), Box::new(p(side)))),
            Box::new(u_n(side)),
        )
    };

    let mut u_left = Vec::new();
    let mut u_right = Vec::new();
    let mut flux_left = Vec::new();
    let mut flux_right = Vec::new();

    for name in &components {
        if name == rho_name {
            u_left.push(rho(FaceSide::Owner));
            u_right.push(rho(FaceSide::Neighbor));
            flux_left.push(flux_mass(FaceSide::Owner));
            flux_right.push(flux_mass(FaceSide::Neighbor));
        } else if name == &format!("{rho_u_name}_x") {
            u_left.push(rho_u_x(FaceSide::Owner));
            u_right.push(rho_u_x(FaceSide::Neighbor));
            flux_left.push(flux_mom_x(FaceSide::Owner));
            flux_right.push(flux_mom_x(FaceSide::Neighbor));
        } else if name == &format!("{rho_u_name}_y") {
            u_left.push(rho_u_y(FaceSide::Owner));
            u_right.push(rho_u_y(FaceSide::Neighbor));
            flux_left.push(flux_mom_y(FaceSide::Owner));
            flux_right.push(flux_mom_y(FaceSide::Neighbor));
        } else if name == rho_e_name {
            u_left.push(rho_e(FaceSide::Owner));
            u_right.push(rho_e(FaceSide::Neighbor));
            flux_left.push(flux_energy(FaceSide::Owner));
            flux_right.push(flux_energy(FaceSide::Neighbor));
        } else {
            return Err(format!(
                "compressible central-upwind kernel builder does not know how to flux component '{name}'"
            ));
        }
    }

    let a_plus = S::Max(
        Box::new(S::lit(0.0)),
        Box::new(S::Max(
            Box::new(S::Add(
                Box::new(u_n(FaceSide::Owner)),
                Box::new(c(FaceSide::Owner)),
            )),
            Box::new(S::Add(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c(FaceSide::Neighbor)),
            )),
        )),
    );
    let a_minus = S::Min(
        Box::new(S::lit(0.0)),
        Box::new(S::Min(
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Owner)),
                Box::new(c(FaceSide::Owner)),
            )),
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c(FaceSide::Neighbor)),
            )),
        )),
    );

    Ok(FluxModuleKernelSpec::CentralUpwind {
        components,
        u_left,
        u_right,
        flux_left,
        flux_right,
        a_plus,
        a_minus,
    })
}

pub fn generic_diffusion_demo_model() -> ModelSpec {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let kappa = Coefficient::constant_unit(1.0, si::AREA / si::TIME);
    let eqn = (fvm::ddt(phi) + fvm::laplacian(kappa, phi)).eqn(phi);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

    let layout = StateLayout::new(vec![phi]);
    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "phi",
        FieldBoundarySpec::new()
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::DIMENSIONLESS),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::DIMENSIONLESS),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::DIMENSIONLESS / si::LENGTH),
            ),
    );
    ModelSpec {
        id: "generic_diffusion_demo",
        method: crate::solver::model::method::MethodSpec::GenericCoupled,
        eos: crate::solver::model::eos::EosSpec::Constant,
        system,
        state_layout: layout,
        boundaries,

        linear_solver: None,
        flux_module: None,
        primitives: crate::solver::model::primitives::PrimitiveDerivations::default(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}

pub fn generic_diffusion_demo_neumann_model() -> ModelSpec {
    let phi = vol_scalar("phi", si::DIMENSIONLESS);
    let kappa = Coefficient::constant_unit(1.0, si::AREA / si::TIME);
    let eqn = (fvm::ddt(phi) + fvm::laplacian(kappa, phi)).eqn(phi);

    let mut system = EquationSystem::new();
    system.add_equation(eqn);

    let layout = StateLayout::new(vec![phi]);
    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "phi",
        FieldBoundarySpec::new()
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::neumann(0.0, si::DIMENSIONLESS / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::neumann(0.0, si::DIMENSIONLESS / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::DIMENSIONLESS / si::LENGTH),
            ),
    );
    ModelSpec {
        id: "generic_diffusion_demo_neumann",
        method: crate::solver::model::method::MethodSpec::GenericCoupled,
        eos: crate::solver::model::eos::EosSpec::Constant,
        system,
        state_layout: layout,
        boundaries,

        linear_solver: None,
        flux_module: None,
        primitives: crate::solver::model::primitives::PrimitiveDerivations::default(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}

/// Build-time model registry.
///
/// This is the single list `build.rs` iterates for WGSL emission and registry generation.
pub fn all_models() -> Vec<ModelSpec> {
    vec![
        incompressible_momentum_model(),
        incompressible_momentum_generic_model(),
        compressible_model(),
        generic_diffusion_demo_model(),
        generic_diffusion_demo_neumann_model(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::TermOp;
    use crate::solver::model::KernelKind;

    #[test]
    fn incompressible_momentum_system_contains_expected_terms() {
        let system = incompressible_momentum_system();
        assert_eq!(system.equations().len(), 2);
        let momentum = &system.equations()[0];
        assert_eq!(momentum.target().name(), "U");
        assert_eq!(momentum.terms().len(), 4);
        assert_eq!(momentum.terms()[0].op, TermOp::Ddt);
        match &momentum.terms()[0].coeff {
            Some(Coefficient::Field(field)) => assert_eq!(field.name(), "rho"),
            other => panic!("expected rho coefficient, got {:?}", other),
        }
        assert_eq!(momentum.terms()[1].op, TermOp::Div);
        assert_eq!(momentum.terms()[2].op, TermOp::Laplacian);
        assert_eq!(momentum.terms()[3].op, TermOp::Grad);

        let pressure = &system.equations()[1];
        assert_eq!(pressure.target().name(), "p");
        assert_eq!(pressure.terms().len(), 2);
        assert_eq!(pressure.terms()[0].op, TermOp::Laplacian);
        match &pressure.terms()[0].coeff {
            Some(Coefficient::Product(lhs, rhs)) => {
                assert!(matches!(**lhs, Coefficient::Field(_)));
                assert!(matches!(**rhs, Coefficient::Field(_)));
            }
            other => panic!("expected coefficient product, got {:?}", other),
        }
        assert_eq!(pressure.terms()[1].op, TermOp::DivFlux);
    }

    #[test]
    fn incompressible_momentum_model_includes_state_layout() {
        let model = incompressible_momentum_model();
        assert_eq!(model.state_layout.offset_for("U"), Some(0));
        assert_eq!(model.state_layout.offset_for("p"), Some(2));
        assert_eq!(model.state_layout.stride(), 8);
        assert_eq!(model.system.equations().len(), 2);
        assert!(model.kernel_plan().contains(KernelKind::FluxModule));
        assert!(model
            .kernel_plan()
            .contains(KernelKind::GenericCoupledAssembly));
        assert!(model
            .kernel_plan()
            .contains(KernelKind::GenericCoupledUpdate));
    }

    #[test]
    fn compressible_model_routes_through_generic_coupled_pipeline() {
        let model = compressible_model();
        assert_eq!(model.system.equations().len(), 3);
        assert_eq!(model.system.equations()[1].terms().len(), 2);
        assert_eq!(model.system.equations()[2].terms().len(), 2);

        // Compressible uses the generic-coupled pipeline with a model-defined flux module stage.
        assert!(model.kernel_plan().contains(KernelKind::FluxModule));
        assert!(model
            .kernel_plan()
            .contains(KernelKind::GenericCoupledAssembly));
        assert!(model
            .kernel_plan()
            .contains(KernelKind::GenericCoupledUpdate));
    }

    #[test]
    fn boundary_spec_can_build_gpu_tables() {
        let model = generic_diffusion_demo_model();
        let (kind, value) = model
            .boundaries
            .to_gpu_tables(&model.system)
            .expect("gpu tables");
        assert_eq!(kind.len(), 4);
        assert_eq!(value.len(), 4);
        assert_eq!(
            kind[GpuBoundaryType::Inlet as usize],
            GpuBcKind::Dirichlet as u32
        );
        assert_eq!(
            kind[GpuBoundaryType::Outlet as usize],
            GpuBcKind::Dirichlet as u32
        );
        assert_eq!(
            kind[GpuBoundaryType::Wall as usize],
            GpuBcKind::ZeroGradient as u32
        );
    }
}
