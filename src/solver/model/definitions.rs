use super::backend::ast::{
    fvc, fvm, surface_scalar, surface_vector, vol_scalar, vol_vector, Coefficient, EquationSystem,
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
        let has_kt_flux = matches!(
            self.flux_module,
            Some(crate::solver::model::flux_module::FluxModuleSpec::KurganovTadmor { .. })
        );

        // Flux storage is model-driven.
        //
        // - If the model has an explicit flux module, store face fluxes for each coupled
        //   unknown component.
        let flux = match &self.flux_module {
            Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow { .. }) => {
                // Rhie–Chow computes a single scalar face flux (phi).
                Some(FluxSpec { stride: 1 })
            }
            Some(crate::solver::model::flux_module::FluxModuleSpec::KurganovTadmor { .. }) => {
                // Conservative KT fluxes are stored per-unknown component.
                Some(FluxSpec {
                    stride: self.system.unknowns_per_cell(),
                })
            }
            Some(crate::solver::model::flux_module::FluxModuleSpec::Convective { .. }) => {
                // Generic convective flux uses a scalar face flux.
                Some(FluxSpec { stride: 1 })
            }
            None => None,
        };

        // Low-Mach parameters are currently required by the KT flux kernel bridge.
        let requires_low_mach_params = has_kt_flux;

        let gradient_storage = if has_kt_flux {
            // Legacy EI kernels expect per-component gradient buffers (e.g. grad_rho_u_x).
            GradientStorage::PerFieldComponents
        } else {
            match self.method {
                crate::solver::model::method::MethodSpec::GenericCoupled
                | crate::solver::model::method::MethodSpec::GenericCoupledImplicit { .. } => {
                    GradientStorage::PackedState
                }
                crate::solver::model::method::MethodSpec::CoupledIncompressible => {
                    GradientStorage::PerFieldName
                }
            }
        };

        let required_gradient_fields = if has_kt_flux {
            use crate::solver::model::gpu_spec::expand_field_components;
            use std::collections::HashSet;

            // KT uses per-component gradients of the coupled unknowns (conserved variables),
            // derived from the equation targets in the model's `EquationSystem`.
            let mut out = Vec::new();
            let mut seen = HashSet::new();
            for eq in self.system.equations() {
                for name in expand_field_components(*eq.target()) {
                    if seen.insert(name.clone()) {
                        out.push(name);
                    }
                }
            }
            out
        } else {
            Vec::new()
        };

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
        + fvc::grad(fields.p))
    .eqn(fields.u);

    let pressure = fvm::laplacian(
        Coefficient::product(
            Coefficient::field(fields.rho).expect("rho must be scalar"),
            Coefficient::field(fields.d_p).expect("d_p must be scalar"),
        )
        .expect("pressure coefficient must be scalar"),
        fields.p,
    )
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

    ModelSpec {
        id: "incompressible_momentum",
        method: crate::solver::model::method::MethodSpec::CoupledIncompressible,
        eos: crate::solver::model::eos::EosSpec::Constant,
        system,
        state_layout: layout,
        boundaries: BoundarySpec::default(),

        linear_solver: None,
        flux_module: Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow {
            alpha_u: None,
        }),
        primitives: crate::solver::model::primitives::PrimitiveDerivations::identity(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}

/// Transitional: incompressible model routed through the generic coupled pipeline.
///
/// This keeps the same state layout (including auxiliary fields like `d_p`, `grad_p`) so
/// existing flux kernels (e.g. Rhie–Chow) remain usable while generic assembly/solve is
/// brought up.
pub fn incompressible_momentum_generic_model() -> ModelSpec {
    let mut model = incompressible_momentum_model();
    model.id = "incompressible_momentum_generic";
    model.method = crate::solver::model::method::MethodSpec::GenericCoupled;

    let u0 = model
        .state_layout
        .component_offset("U", 0)
        .ok_or_else(|| "incompressible_momentum_generic_model missing U[0] in state layout".to_string())
        .expect("state layout validation failed");
    let u1 = model
        .state_layout
        .component_offset("U", 1)
        .ok_or_else(|| "incompressible_momentum_generic_model missing U[1] in state layout".to_string())
        .expect("state layout validation failed");
    let p = model
        .state_layout
        .offset_for("p")
        .ok_or_else(|| "incompressible_momentum_generic_model missing p in state layout".to_string())
        .expect("state layout validation failed");

    // The generic coupled path needs a saddle-point-capable preconditioner.
    // Keep the legacy incompressible model on the specialized coupled solver
    // until this preconditioner is stable.
    model.linear_solver = Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
        preconditioner: crate::solver::model::linear_solver::ModelPreconditionerSpec::Schur {
            omega: 1.0,
            layout: crate::solver::model::linear_solver::SchurBlockLayout::from_u_p(&[u0, u1], p)
                .expect("invalid SchurBlockLayout"),
        },
    });

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

    ModelSpec {
        id: "compressible",
        // Route compressible through the generic coupled pipeline; KT flux + primitive recovery
        // remain transitional kernels behind model-driven IDs.
        method: crate::solver::model::method::MethodSpec::GenericCoupledImplicit { outer_iters: 1 },
        eos: crate::solver::model::eos::EosSpec::IdealGas { gamma },
        system,
        state_layout: layout,
        boundaries: BoundarySpec::default(),

        linear_solver: None,
        flux_module: Some(crate::solver::model::flux_module::FluxModuleSpec::KurganovTadmor {
            reconstruction: crate::solver::model::flux_module::ReconstructionSpec::FirstOrder,
        }),
        primitives: crate::solver::model::primitives::PrimitiveDerivations::euler_ideal_gas(gamma),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
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
        assert_eq!(pressure.terms().len(), 1);
        assert_eq!(pressure.terms()[0].op, TermOp::Laplacian);
        match &pressure.terms()[0].coeff {
            Some(Coefficient::Product(lhs, rhs)) => {
                assert!(matches!(**lhs, Coefficient::Field(_)));
                assert!(matches!(**rhs, Coefficient::Field(_)));
            }
            other => panic!("expected coefficient product, got {:?}", other),
        }
    }

    #[test]
    fn incompressible_momentum_model_includes_state_layout() {
        let model = incompressible_momentum_model();
        assert_eq!(model.state_layout.offset_for("U"), Some(0));
        assert_eq!(model.state_layout.offset_for("p"), Some(2));
        assert_eq!(model.state_layout.stride(), 8);
        assert_eq!(model.system.equations().len(), 2);
        assert!(model.kernel_plan().contains(KernelKind::CoupledAssembly));
    }

    #[test]
    fn compressible_model_routes_through_generic_coupled_pipeline() {
        let model = compressible_model();
        assert_eq!(model.system.equations().len(), 3);
        assert_eq!(model.system.equations()[1].terms().len(), 2);
        assert_eq!(model.system.equations()[2].terms().len(), 2);

        // Compressible uses the generic-coupled pipeline with a KT flux module stage.
        assert!(model.kernel_plan().contains(KernelKind::KtGradients));
        assert!(model.kernel_plan().contains(KernelKind::FluxKt));
        assert!(model.kernel_plan().contains(KernelKind::GenericCoupledAssembly));
        assert!(model.kernel_plan().contains(KernelKind::GenericCoupledUpdate));
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
