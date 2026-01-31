/// Model registry + model specs + boundary specs.
use crate::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use crate::solver::model::backend::ast::{EquationSystem, FieldRef};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::units::{si, UnitDim};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: &'static str,
    pub system: EquationSystem,
    pub state_layout: StateLayout,
    pub boundaries: BoundarySpec,

    /// Model-defined numerical modules.
    ///
    /// Each module contributes kernel passes (schedule) and optional build-time WGSL generators.
    ///
    /// This is the primary mechanism for Gap 0 in `CODEGEN_PLAN.md`: adding a new numerical
    /// module should not require edits to central kernel registries.
    pub modules: Vec<crate::solver::model::module::KernelBundleModule>,

    /// Optional model-owned linear solver configuration.
    ///
    /// When present, this is treated as authoritative by solver families that
    /// support model-driven solver selection.
    pub linear_solver: Option<crate::solver::model::linear_solver::ModelLinearSolverSpec>,

    /// Derived primitive recovery expressions (empty if primitives = conserved state)
    pub primitives: crate::solver::model::primitives::PrimitiveDerivations,
}

impl ModelSpec {
    pub fn named_param_keys(&self) -> Vec<&'static str> {
        let mut out: std::collections::HashSet<&'static str> = std::collections::HashSet::new();

        for module in &self.modules {
            // Include params from named_params (legacy)
            for key in &module.manifest.named_params {
                out.insert(key.as_str());
            }
            // Include params from port_manifest (new)
            if let Some(ref port_manifest) = module.manifest.port_manifest {
                for param in &port_manifest.params {
                    out.insert(param.key);
                }
            }
        }

        let mut v: Vec<&'static str> = out.into_iter().collect();
        v.sort_unstable();
        v
    }

    pub fn eos(&self) -> crate::solver::model::eos::EosSpec {
        self.eos_checked().unwrap_or_default()
    }

    pub fn eos_checked(&self) -> Result<crate::solver::model::eos::EosSpec, String> {
        let mut found: Option<(&'static str, crate::solver::model::eos::EosSpec)> = None;
        for module in &self.modules {
            if let Some(eos) = module.eos {
                match found {
                    None => found = Some((module.name, eos)),
                    Some((prev_name, _)) => {
                        return Err(format!(
                            "model defines EOS in multiple modules ('{prev_name}', '{}')",
                            module.name
                        ));
                    }
                }
            }
        }
        found
            .map(|(_, eos)| eos)
            .ok_or_else(|| "model defines no EOS module".to_string())
    }

    pub fn method(&self) -> Result<crate::solver::model::method::MethodSpec, String> {
        let mut found: Option<(&'static str, crate::solver::model::method::MethodSpec)> = None;
        for module in &self.modules {
            if let Some(method) = module.manifest.method {
                match found {
                    None => found = Some((module.name, method)),
                    Some((prev_name, _)) => {
                        return Err(format!(
                            "model defines method in multiple modules ('{prev_name}', '{}')",
                            module.name
                        ));
                    }
                }
            }
        }
        found
            .map(|(_, m)| m)
            .ok_or_else(|| "model defines no method module".to_string())
    }

    pub fn flux_module(
        &self,
    ) -> Result<Option<&crate::solver::model::flux_module::FluxModuleSpec>, String> {
        let mut found: Option<(
            &'static str,
            &crate::solver::model::flux_module::FluxModuleSpec,
        )> = None;
        for module in &self.modules {
            if let Some(spec) = module.manifest.flux_module.as_ref() {
                match found {
                    None => found = Some((module.name, spec)),
                    Some((prev_name, _)) => {
                        return Err(format!(
                            "model defines flux_module in multiple modules ('{prev_name}', '{}')",
                            module.name
                        ));
                    }
                }
            }
        }
        Ok(found.map(|(_, spec)| spec))
    }

    pub fn validate_module_manifests(&self) -> Result<(), String> {
        let _ = self.eos_checked()?;
        let method = self.method()?;
        let flux = self.flux_module()?;

        let caps = match method {
            crate::solver::model::method::MethodSpec::Coupled(caps) => caps,
        };
        if caps.requires_flux_module && flux.is_none() {
            return Err("Coupled method requires a flux_module-providing module".to_string());
        }

        // Flux-module gradients stages must be explicitly supported by the model.
        //
        // Reconstruction scheme selection is a numerical-method knob driven by the runtime
        // `advection_scheme` parameter (shared with unified_assembly), but some flux modules
        // still require precomputed `grad_*` fields (e.g. Rhie–Chow / pressure-correction).
        if let Some(flux) = flux {
            use crate::solver::model::flux_module::FluxModuleGradientsSpec;

            let gradients = match flux {
                crate::solver::model::flux_module::FluxModuleSpec::Kernel { gradients, .. } => {
                    gradients.as_ref()
                }
                crate::solver::model::flux_module::FluxModuleSpec::Scheme { gradients, .. } => {
                    gradients.as_ref()
                }
            };

            if matches!(gradients, Some(FluxModuleGradientsSpec::FromStateLayout)) {
                // Find the unique module providing flux_module to access its port manifest.
                // This uses the same uniqueness assumptions as ModelSpec::flux_module().
                let flux_module_provider = self.modules.iter().find(|m| {
                    m.manifest.flux_module.is_some()
                });

                let has_gradient_targets = flux_module_provider
                    .and_then(|m| m.manifest.port_manifest.as_ref())
                    .map(|pm| !pm.gradient_targets.is_empty())
                    .unwrap_or(false);

                if !has_gradient_targets {
                    return Err("flux_module_gradients requested but no grad_<field> targets found in state layout".to_string());
                }

                // Gradient targets are pre-resolved at module creation time
                // (in flux_module_module()) and stored in port_manifest.gradient_targets.
                // No additional StateLayout scanning needed here.
            }
        }

        // Validate port_manifest fields against state_layout.
        // This makes PortManifest.fields the single source of truth for required fields.
        for module in &self.modules {
            if let Some(ref port_manifest) = module.manifest.port_manifest {
                for field_spec in &port_manifest.fields {
                    let name = field_spec.name;
                    let Some(layout_field) = self.state_layout.field(name) else {
                        return Err(format!(
                            "module '{}' port_manifest field '{}' not found in state layout",
                            module.name, name
                        ));
                    };

                    // Validate component count (kind) matches
                    let expected_components = field_spec.kind.component_count();
                    let actual_components = layout_field.component_count();
                    if expected_components != actual_components {
                        return Err(format!(
                            "module '{}' port_manifest field '{}' kind mismatch: expected {} component(s), found {}",
                            module.name,
                            name,
                            expected_components,
                            actual_components
                        ));
                    }

                    // Validate unit dimension matches (skip for ANY_DIMENSION sentinel)
                    if field_spec.unit != crate::solver::ir::ports::ANY_DIMENSION
                        && field_spec.unit != layout_field.unit()
                    {
                        return Err(format!(
                            "module '{}' port_manifest field '{}' unit mismatch: expected {}, found {}",
                            module.name,
                            name,
                            field_spec.unit,
                            layout_field.unit()
                        ));
                    }
                }
            }
        }

        // Validate typed invariant requirements declared by modules.
        for module in &self.modules {
            for inv in &module.manifest.invariants {
                use crate::solver::model::backend::FieldKind;
                use crate::solver::model::module::{FieldKindReq, ModuleInvariant};

                let err_prefix = format!("module '{}' invariant failed: ", module.name);

                match *inv {
                    ModuleInvariant::RequireStateField { name, kind } => {
                        let Some(field) = self.state_layout.field(name) else {
                            return Err(format!(
                                "{err_prefix}missing required state field '{name}'"
                            ));
                        };
                        if let Some(req) = kind {
                            let expected = match req {
                                FieldKindReq::Scalar => FieldKind::Scalar,
                                FieldKindReq::Vector2 => FieldKind::Vector2,
                                FieldKindReq::Vector3 => FieldKind::Vector3,
                            };
                            if field.kind() != expected {
                                return Err(format!(
                                    "{err_prefix}state field '{name}' has kind {:?}, expected {:?}",
                                    field.kind(),
                                    expected
                                ));
                            }
                        }
                    }
                    ModuleInvariant::RequireUniqueMomentumPressureCouplingReferencingDp {
                        dp_field,
                        require_vector2_momentum,
                        require_pressure_gradient,
                    } => {
                        let Some(dp) = self.state_layout.field(dp_field) else {
                            return Err(format!(
                                "{err_prefix}requires '{dp_field}' in state layout"
                            ));
                        };
                        if dp.kind() != FieldKind::Scalar {
                            return Err(format!(
                                "{err_prefix}requires '{dp_field}' to be a scalar field"
                            ));
                        }

                        let coupling = crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp(
                            self,
                            dp_field,
                        )
                        .map_err(|e| format!("{err_prefix}{e}"))?;

                        if require_vector2_momentum
                            && coupling.momentum.kind() != FieldKind::Vector2
                        {
                            return Err(format!(
                                "{err_prefix}requires Vector2 momentum field; got {:?} ('{}')",
                                coupling.momentum.kind(),
                                coupling.momentum.name()
                            ));
                        }

                        if require_pressure_gradient {
                            let grad_name = format!("grad_{}", coupling.pressure.name());
                            let Some(grad) = self.state_layout.field(&grad_name) else {
                                return Err(format!(
                                    "{err_prefix}requires '{grad_name}' in state layout"
                                ));
                            };
                            if grad.kind() != FieldKind::Vector2 {
                                return Err(format!(
                                    "{err_prefix}requires '{grad_name}' to be Vector2"
                                ));
                            }

                            // Ensure the component offsets exist (avoids a later codegen error).
                            for c in 0..2 {
                                self.state_layout
                                    .component_offset(&grad_name, c)
                                    .ok_or_else(|| {
                                        format!(
                                        "{err_prefix}requires '{grad_name}[{c}]' in state layout"
                                    )
                                    })?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
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
            GpuBoundaryType::MovingWall,
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

#[allow(unused_imports)]
pub use compressible::{
    compressible_model, compressible_model_with_eos, compressible_system, CompressibleFields,
};
pub use generic_diffusion_demo::{
    generic_diffusion_demo_model, generic_diffusion_demo_neumann_model,
};
pub use incompressible_momentum::{
    incompressible_momentum_model, incompressible_momentum_system, IncompressibleMomentumFields,
};

/// Build-time model registry.
///
/// This is the single list `build.rs` iterates for WGSL emission and registry generation.
pub fn all_models() -> Vec<ModelSpec> {
    vec![
        incompressible_momentum_model(),
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
    use crate::solver::units::si;

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
        assert_eq!(model.system.equations().len(), 6);
        assert_eq!(model.system.equations()[0].target().name(), "rho");
        assert_eq!(model.system.equations()[1].target().name(), "rho_u");
        assert_eq!(model.system.equations()[2].target().name(), "rho_e");
        assert_eq!(model.system.equations()[3].target().name(), "u");
        assert_eq!(model.system.equations()[4].target().name(), "p");
        assert_eq!(model.system.equations()[5].target().name(), "T");

        assert_eq!(model.system.equations()[0].terms().len(), 2);
        assert_eq!(model.system.equations()[1].terms().len(), 3);
        assert_eq!(model.system.equations()[2].terms().len(), 3);
        assert_eq!(model.system.equations()[3].terms().len(), 2);
        assert_eq!(model.system.equations()[4].terms().len(), 5);
        assert_eq!(model.system.equations()[5].terms().len(), 2);

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
    fn flux_module_gradients_stage_validates_grad_field_shape_and_base_field() {
        use crate::solver::model::backend::ast::vol_scalar;
        use crate::solver::model::flux_module::{FluxModuleGradientsSpec, FluxSchemeSpec};
        use crate::solver::model::modules::flux_module::flux_module_module;

        let mut model = compressible_model();

        // Add an invalid grad_* field: wrong shape (scalar instead of Vector2).
        let fields = CompressibleFields::new();
        model.state_layout = StateLayout::new(vec![
            fields.rho,
            fields.rho_u,
            fields.rho_e,
            fields.p,
            fields.t,
            fields.u,
            vol_scalar("grad_rho", si::DENSITY / si::LENGTH),
        ]);

        let with_gradients = crate::solver::model::FluxModuleSpec::Scheme {
            gradients: Some(FluxModuleGradientsSpec::FromStateLayout),
            scheme: FluxSchemeSpec::EulerCentralUpwind,
        };

        // Now that gradient targets are resolved at module creation time,
        // the error should be caught when building the flux module.
        let err = flux_module_module(with_gradients, &model.system, &model.state_layout, &model.primitives)
            .unwrap_err();
        assert!(
            err.contains("no grad_<field> fields found") || err.contains("Vector2"),
            "Expected error about missing/invalid gradient fields, got: {}",
            err
        );
    }

    #[test]
    fn boundary_spec_can_build_gpu_tables() {
        let model = generic_diffusion_demo_model();
        let (kind, value) = model
            .boundaries
            .to_gpu_tables(&model.system)
            .expect("gpu tables");
        assert_eq!(kind.len(), 6);
        assert_eq!(value.len(), 6);
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
        assert_eq!(
            kind[GpuBoundaryType::MovingWall as usize],
            GpuBcKind::ZeroGradient as u32
        );
    }
}
