/// Model registry + model specs + boundary specs.

use crate::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use crate::solver::model::backend::ast::{EquationSystem, FieldRef};
use crate::solver::model::backend::state_layout::StateLayout;
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

    /// Optional, model-defined kernel schedule extensions.
    ///
    /// This is the first step toward pluggable numerical “modules” (Gap 0 in `CODEGEN_PLAN.md`):
    /// models can inject additional kernels without requiring edits to central kernel registries.
    pub extra_kernels: Vec<crate::solver::model::kernel::ModelKernelSpec>,

    /// Optional model-owned linear solver configuration.
    ///
    /// When present, this is treated as authoritative by solver families that
    /// support model-driven solver selection.
    pub linear_solver: Option<crate::solver::model::linear_solver::ModelLinearSolverSpec>,

    /// Flux computation module (optional: for pure diffusion models, set to None)
    pub flux_module: Option<crate::solver::model::flux_module::FluxModuleSpec>,

    /// Derived primitive recovery expressions (empty if primitives = conserved state)
    pub primitives: crate::solver::model::primitives::PrimitiveDerivations,

    /// Optional, model-defined build-time WGSL generators for extra kernels in `extra_kernels`.
    ///
    /// This allows adding new model/method modules without growing global `match` statements in
    /// `src/solver/model/kernel.rs` or `build.rs`.
    pub generated_kernels: Vec<crate::solver::model::kernel::ModelKernelGeneratorSpec>,

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

        let boundary_types = [
            GpuBoundaryType::None,
            GpuBoundaryType::Inlet,
            GpuBoundaryType::Outlet,
            GpuBoundaryType::Wall,
            GpuBoundaryType::SlipWall,
        ];

        let mut kind = vec![GpuBcKind::ZeroGradient as u32; boundary_types.len() * coupled_stride];
        let mut value = vec![0.0_f32; boundary_types.len() * coupled_stride];

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

#[path = "definitions/compressible.rs"]
mod compressible;
#[path = "definitions/generic_diffusion_demo.rs"]
mod generic_diffusion_demo;
#[path = "definitions/incompressible_momentum.rs"]
mod incompressible_momentum;

pub use compressible::{compressible_model, compressible_system, CompressibleFields};
pub use generic_diffusion_demo::{generic_diffusion_demo_model, generic_diffusion_demo_neumann_model};
pub use incompressible_momentum::{
    incompressible_momentum_generic_model, incompressible_momentum_model,
    incompressible_momentum_system, IncompressibleMomentumFields,
};

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
    use crate::solver::model::backend::ast::Coefficient;
    use crate::solver::model::backend::ast::TermOp;
    use crate::solver::model::kernel::derive_kernel_specs_for_model;
    use crate::solver::model::KernelId;

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

        let kernel_ids: Vec<_> = derive_kernel_specs_for_model(&model)
            .expect("kernel specs")
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert!(kernel_ids.contains(&KernelId::FLUX_MODULE));
        assert!(kernel_ids.contains(&KernelId::GENERIC_COUPLED_ASSEMBLY));
        assert!(kernel_ids.contains(&KernelId::GENERIC_COUPLED_UPDATE));
    }

    #[test]
    fn compressible_model_routes_through_generic_coupled_pipeline() {
        let model = compressible_model();
        assert_eq!(model.system.equations().len(), 3);
        assert_eq!(model.system.equations()[1].terms().len(), 2);
        assert_eq!(model.system.equations()[2].terms().len(), 2);

        // Compressible uses the generic-coupled pipeline with a model-defined flux module stage.
        let kernel_ids: Vec<_> = derive_kernel_specs_for_model(&model)
            .expect("kernel specs")
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert!(kernel_ids.contains(&KernelId::FLUX_MODULE));
        assert!(kernel_ids.contains(&KernelId::GENERIC_COUPLED_ASSEMBLY));
        assert!(kernel_ids.contains(&KernelId::GENERIC_COUPLED_UPDATE));
    }

    #[test]
    fn boundary_spec_can_build_gpu_tables() {
        let model = generic_diffusion_demo_model();
        let (kind, value) = model
            .boundaries
            .to_gpu_tables(&model.system)
            .expect("gpu tables");
        assert_eq!(kind.len(), 5);
        assert_eq!(value.len(), 5);
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
        assert_eq!(
            kind[GpuBoundaryType::SlipWall as usize],
            GpuBcKind::ZeroGradient as u32
        );
    }
}
