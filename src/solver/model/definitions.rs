use crate::solver::dimensions::{Length, UnitDimension};
/// Model registry + model specs + boundary specs.
use crate::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use crate::solver::model::backend::ast::{EquationSystem, FieldRef};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::units::UnitDim;
use cfd2_codegen::solver::codegen::bc_table::HostBcTable;
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
    pub modules: Vec<crate::solver::model::module::KernelBundleModule>,

    /// Optional model-owned linear solver configuration.
    ///
    /// When present, this is treated as authoritative by solver families that
    /// support model-driven solver selection.
    pub linear_solver: Option<crate::solver::model::linear_solver::ModelLinearSolverSpec>,

    /// Derived primitive recovery expressions (empty if primitives = conserved state)
    pub primitives: crate::solver::model::primitives::PrimitiveDerivations,

    /// Optional primitive closure used specifically by the explicit
    /// method-of-lines integrator.  This is separate from `primitives` because
    /// some implicit models deliberately represent local algebraic closures as
    /// coupled rows, while RK stages must recover those same values directly
    /// after advancing the conserved variables.
    pub explicit_primitives: Option<crate::solver::model::primitives::PrimitiveDerivations>,
}

impl ModelSpec {
    /// Validate that the model is a static method-of-lines system suitable for
    /// the fully explicit RK4 path.
    ///
    /// Every equation row must either declare an own-variable `ddt` (a
    /// differential row) or be recoverable from the model's primitive
    /// derivations (a local algebraic row).  Saddle-point pressure constraints
    /// therefore fail this gate, while scalar transport/diffusion and the
    /// compressible differential models pass.  ALE is intentionally excluded
    /// until stage-consistent moving geometry is defined.
    pub fn validate_explicit_rk4(&self) -> Result<(), String> {
        if self.system.is_ale() {
            return Err(format!(
                "model '{}' uses ALE/moving-mesh terms; explicit RK4 currently supports static meshes only",
                self.id
            ));
        }
        let primitives = self.explicit_primitives.as_ref().unwrap_or(&self.primitives);
        // A collocated pressure model is eligible only when it declares the
        // matrix-free Rhie--Chow coefficient as part of its explicit algebraic
        // closure. Implicit dp_init/dp_update kernels are deliberately absent
        // from RK recipes.
        if self
            .state_layout
            .fields()
            .iter()
            .any(|field| field.name() == "d_p")
            && !primitives.contains("d_p")
        {
            return Err(format!(
                "model '{}' uses Rhie–Chow pressure coupling but declares no explicit `d_p` closure",
                self.id
            ));
        }
        for equation in self.system.equations() {
            let target = equation.target();
            let has_own_ddt = equation.terms().iter().any(|term| {
                term.op == crate::solver::ir::TermOp::Ddt && term.field == *target
            });
            let recoverable = match target.kind() {
                crate::solver::ir::FieldKind::Scalar => primitives.contains(target.name()),
                crate::solver::ir::FieldKind::Vector2 => ["x", "y"].iter().all(|suffix| {
                    primitives
                        .contains(&format!("{}_{}", target.name(), suffix))
                }),
                crate::solver::ir::FieldKind::Vector3 => ["x", "y", "z"].iter().all(|suffix| {
                    primitives
                        .contains(&format!("{}_{}", target.name(), suffix))
                }),
            };
            if !has_own_ddt && !recoverable {
                return Err(format!(
                    "model '{}' row '{}' has no own-variable ddt and no explicit primitive recovery; it is not a method-of-lines row",
                    self.id,
                    target.name()
                ));
            }
        }
        Ok(())
    }

    /// Get the state stride (number of f32 per cell in the state buffer).
    ///
    /// Convenience accessor — equivalent to `self.state_layout.stride()`.
    pub fn state_stride(&self) -> u32 {
        self.state_layout.stride()
    }

    pub fn named_param_keys(&self) -> Vec<&'static str> {
        let mut out: std::collections::HashSet<&'static str> = std::collections::HashSet::new();

        for module in &self.modules {
            for key in &module.named_params {
                out.insert(*key);
            }
            if let Some(ref port_manifest) = module.port_manifest {
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

    /// Query the model-declared relaxation defaults (if any module provides them).
    pub fn relaxation_defaults(
        &self,
    ) -> Option<crate::solver::model::module::RelaxationDefaults> {
        let mut found: Option<crate::solver::model::module::RelaxationDefaults> = None;
        for module in &self.modules {
            if let Some(defaults) = module.relaxation_defaults {
                // Last-one-wins; typically only one module declares these.
                found = Some(defaults);
            }
        }
        found
    }

    pub fn method(&self) -> Result<crate::solver::model::method::MethodSpec, String> {
        let mut found: Option<(&'static str, crate::solver::model::method::MethodSpec)> = None;
        for module in &self.modules {
            if let Some(method) = module.method {
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
            if let Some(spec) = module.flux_module.as_ref() {
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

        let crate::solver::model::method::MethodSpec::Coupled(caps) = method;
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
                // Relies on the same flux_module uniqueness assumption as ModelSpec::flux_module().
                let flux_module_provider = self.modules.iter().find(|m| m.flux_module.is_some());

                let has_gradient_targets = flux_module_provider
                    .and_then(|m| m.port_manifest.as_ref())
                    .map(|pm| !pm.gradient_targets.is_empty())
                    .unwrap_or(false);

                if !has_gradient_targets {
                    return Err("flux_module_gradients requested but no grad_<field> targets found in state layout".to_string());
                }
            }
        }

        self.validate_module_manifests_with_registry()
            .map_err(|e| e.to_string())
    }

    /// Validate module manifests using PortRegistry.
    ///
    /// This resolves all required state-field metadata once and reuses it across
    /// validation passes, producing structured `PortValidationError` internally.
    fn validate_module_manifests_with_registry(
        &self,
    ) -> Result<(), crate::solver::model::ports::PortValidationError> {
        use crate::solver::model::module::{FieldKindReq, ModuleInvariant};
        use crate::solver::model::ports::{PortRegistry, PortValidationError};

        let mut registry = PortRegistry::new(self.state_layout.clone());
        for field_ref in self.state_layout.fields() {
            let _ = registry.register_state_field(field_ref.name());
        }

        for module in &self.modules {
            if let Some(ref port_manifest) = module.port_manifest {
                for field_spec in &port_manifest.fields {
                    let name = field_spec.name;

                    let Some(entry) = registry.get_field_entry_by_name(name) else {
                        return Err(PortValidationError::MissingField {
                            module: module.name,
                            field: name.to_string(),
                        });
                    };

                    let expected_components = field_spec.kind.component_count();
                    let actual_components = entry.component_count();
                    if expected_components != actual_components {
                        return Err(PortValidationError::FieldKindMismatch {
                            field: name.to_string(),
                            expected: match expected_components {
                                1 => "Scalar".to_string(),
                                2 => "Vector2".to_string(),
                                3 => "Vector3".to_string(),
                                n => format!("Vector{n}"),
                            },
                            found: match actual_components {
                                1 => "Scalar".to_string(),
                                2 => "Vector2".to_string(),
                                3 => "Vector3".to_string(),
                                n => format!("Vector{n}"),
                            },
                        });
                    }

                    // ANY_DIMENSION is a wildcard sentinel: skip the unit check.
                    if field_spec.unit != crate::solver::ir::ports::ANY_DIMENSION
                        && field_spec.unit != entry.runtime_dimension()
                    {
                        return Err(PortValidationError::DimensionMismatch {
                            field: name.to_string(),
                            expected: field_spec.unit.to_string(),
                            found: entry.runtime_dimension().to_string(),
                        });
                    }
                }
            }
        }

        for module in &self.modules {
            for inv in &module.invariants {
                match *inv {
                    ModuleInvariant::RequireStateField { name, kind } => {
                        let Some(entry) = registry.get_field_entry_by_name(name) else {
                            return Err(PortValidationError::MissingField {
                                module: module.name,
                                field: name.to_string(),
                            });
                        };

                        if let Some(req) = kind {
                            let expected_components = match req {
                                FieldKindReq::Scalar => 1,
                                FieldKindReq::Vector2 => 2,
                                FieldKindReq::Vector3 => 3,
                            };
                            let actual_components = entry.component_count();
                            if expected_components != actual_components {
                                return Err(PortValidationError::FieldKindMismatch {
                                    field: name.to_string(),
                                    expected: match expected_components {
                                        1 => "Scalar".to_string(),
                                        2 => "Vector2".to_string(),
                                        3 => "Vector3".to_string(),
                                        n => format!("Vector{n}"),
                                    },
                                    found: match actual_components {
                                        1 => "Scalar".to_string(),
                                        2 => "Vector2".to_string(),
                                        3 => "Vector3".to_string(),
                                        n => format!("Vector{n}"),
                                    },
                                });
                            }
                        }
                    }
                    ModuleInvariant::RequireUniqueMomentumPressureCouplingReferencingDp {
                        dp_field,
                        require_vector2_momentum,
                        require_pressure_gradient,
                    } => {
                        let Some(dp_entry) = registry.get_field_entry_by_name(dp_field) else {
                            return Err(PortValidationError::MissingField {
                                module: module.name,
                                field: dp_field.to_string(),
                            });
                        };
                        if dp_entry.component_count() != 1 {
                            return Err(PortValidationError::FieldKindMismatch {
                                field: dp_field.to_string(),
                                expected: "Scalar".to_string(),
                                found: match dp_entry.component_count() {
                                    1 => "Scalar".to_string(),
                                    2 => "Vector2".to_string(),
                                    3 => "Vector3".to_string(),
                                    n => format!("Vector{n}"),
                                },
                            });
                        }

                        let coupling = crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp(
                            self,
                            dp_field,
                        )
                        .map_err(|e| PortValidationError::MissingField {
                            module: module.name,
                            field: format!("coupling inference failed: {e}"),
                        })?;

                        if require_vector2_momentum {
                            let momentum_entry = registry
                                .get_field_entry_by_name(coupling.momentum.name())
                                .ok_or_else(|| PortValidationError::MissingField {
                                    module: module.name,
                                    field: coupling.momentum.name().to_string(),
                                })?;
                            if momentum_entry.component_count() != 2 {
                                return Err(PortValidationError::FieldKindMismatch {
                                    field: coupling.momentum.name().to_string(),
                                    expected: "Vector2".to_string(),
                                    found: match momentum_entry.component_count() {
                                        1 => "Scalar".to_string(),
                                        2 => "Vector2".to_string(),
                                        3 => "Vector3".to_string(),
                                        n => format!("Vector{n}"),
                                    },
                                });
                            }
                        }

                        if require_pressure_gradient {
                            let grad_name = format!("grad_{}", coupling.pressure.name());
                            let Some(grad_entry) = registry.get_field_entry_by_name(&grad_name)
                            else {
                                return Err(PortValidationError::MissingField {
                                    module: module.name,
                                    field: grad_name.clone(),
                                });
                            };
                            if grad_entry.component_count() != 2 {
                                return Err(PortValidationError::FieldKindMismatch {
                                    field: grad_name.clone(),
                                    expected: "Vector2".to_string(),
                                    found: match grad_entry.component_count() {
                                        1 => "Scalar".to_string(),
                                        2 => "Vector2".to_string(),
                                        3 => "Vector3".to_string(),
                                        n => format!("Vector{n}"),
                                    },
                                });
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

        let table = HostBcTable::new(coupled_stride);

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

                let expected_unit = match entry.as_ref() {
                    // Expression-refreshed entries hold boundary-face STATE values
                    // (in field.unit()) regardless of kind, so they skip the /L gradient unit.
                    Some(c) if c.expr_value().is_some() => field.unit(),
                    Some(c) if c.kind == GpuBcKind::Dirichlet => field.unit(),
                    _ => field.unit() / Length::UNIT,
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
                    kind[table.offset(b_i, u_idx)] = cond.kind as u32;
                    value[table.offset(b_i, u_idx)] = cond.seed_value() as f32;
                } else {
                    kind[table.offset(b_i, u_idx)] = GpuBcKind::ZeroGradient as u32;
                    value[table.offset(b_i, u_idx)] = 0.0;
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

    /// Set distinct per-component conditions (needed when components carry
    /// different expression values, e.g. a vector unknown's x/y entries).
    pub fn set_components(
        mut self,
        boundary: GpuBoundaryType,
        conditions: Vec<BoundaryCondition>,
    ) -> Self {
        self.by_boundary.insert(boundary, conditions);
        self
    }
}

impl Default for FieldBoundarySpec {
    fn default() -> Self {
        Self::new()
    }
}

/// The value of a boundary-table entry: a host-prescribed constant, or a
/// declared expression a generic Preparation-phase kernel refreshes per
/// boundary face every outer iteration (see `backend::boundary`).
#[derive(Debug, Clone)]
pub enum BcValue {
    Const(f64),
    Expr(crate::solver::model::backend::boundary::BoundaryExpr),
}

#[derive(Debug, Clone)]
pub struct BoundaryCondition {
    pub kind: GpuBcKind,
    pub value: BcValue,
    pub unit: UnitDim,
}

impl BoundaryCondition {
    pub fn zero_gradient(unit: UnitDim) -> Self {
        Self {
            kind: GpuBcKind::ZeroGradient,
            value: BcValue::Const(0.0),
            unit,
        }
    }

    pub fn dirichlet(value: f64, unit: UnitDim) -> Self {
        Self {
            kind: GpuBcKind::Dirichlet,
            value: BcValue::Const(value),
            unit,
        }
    }

    /// Value is `dphi/dn` (outward normal gradient).
    pub fn neumann(dphi_dn: f64, unit: UnitDim) -> Self {
        Self {
            kind: GpuBcKind::Neumann,
            value: BcValue::Const(dphi_dn),
            unit,
        }
    }

    /// Typed zero-gradient boundary condition using type-level dimensions.
    pub fn zero_gradient_dim<D: UnitDimension>() -> Self {
        Self::zero_gradient(D::UNIT)
    }

    /// Typed Dirichlet boundary condition using type-level dimensions.
    pub fn dirichlet_dim<D: UnitDimension>(value: f64) -> Self {
        Self::dirichlet(value, D::UNIT)
    }

    /// Typed Neumann boundary condition using type-level dimensions.
    /// Value is `dphi/dn` (outward normal gradient).
    pub fn neumann_dim<D: UnitDimension>(dphi_dn: f64) -> Self {
        Self::neumann(dphi_dn, D::UNIT)
    }

    /// Boundary condition whose table value is refreshed per face from a
    /// declared expression (the static table seeds the constructor-supplied
    /// `seed` until the first refresh). `kind` stays as declared: a
    /// Dirichlet entry constrains assembly with the refreshed value, while
    /// a ZeroGradient entry's refreshed value only feeds consumers that
    /// read boundary-face state (e.g. flux reconstruction).
    pub fn with_expr_value_dim<D: UnitDimension>(
        kind: GpuBcKind,
        expr: crate::solver::model::backend::boundary::BoundaryExpr,
    ) -> Result<Self, String> {
        if let Some(unit) = expr.unit()? {
            if unit != D::UNIT {
                return Err(format!(
                    "boundary expression unit {} does not match declared unit {}",
                    unit,
                    D::UNIT
                ));
            }
        }
        Ok(Self {
            kind,
            value: BcValue::Expr(expr),
            unit: D::UNIT,
        })
    }

    /// The constant table value: `Const` as-is; expression-valued entries
    /// seed 0.0 (the refresh kernel overwrites before first use).
    pub fn seed_value(&self) -> f64 {
        match &self.value {
            BcValue::Const(v) => *v,
            BcValue::Expr(_) => 0.0,
        }
    }

    /// The declared expression, if this entry is expression-valued.
    pub fn expr_value(&self) -> Option<&crate::solver::model::backend::boundary::BoundaryExpr> {
        match &self.value {
            BcValue::Const(_) => None,
            BcValue::Expr(expr) => Some(expr),
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
#[path = "definitions/allmach_pressure.rs"]
mod allmach_pressure;
#[path = "definitions/buoyant_incompressible.rs"]
mod buoyant_incompressible;
#[path = "definitions/scalar_transport.rs"]
mod scalar_transport;

#[allow(unused_imports)]
pub use compressible::{
    compressible_central_upwind_decl, compressible_generalized_wave_speed_sq,
    compressible_mms_biharmonic_model, compressible_mms_model, compressible_model,
    compressible_structured_model,
    compressible_model_with_eos, compressible_system,
    compressible_wave_speed_sq, CompressibleFields, COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
    COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD,
};
#[allow(unused_imports)]
pub use generic_diffusion_demo::{
    generic_diffusion_demo_mms_dirichlet_model, generic_diffusion_demo_mms_model,
    generic_diffusion_demo_mms_neumann_model, generic_diffusion_demo_model,
    generic_diffusion_demo_neumann_model, generic_diffusion_demo_structured_ibm_model,
    generic_diffusion_demo_structured_mms_model, generic_diffusion_demo_structured_model,
    IBM_PENALTY_FIELD, MMS_SOURCE_FIELD,
};
#[allow(unused_imports)]
pub use buoyant_incompressible::{
    buoyant_incompressible_mms_model, buoyant_incompressible_model, BUOYANT_BETA_G,
    BUOYANT_K_OVER_CP, BUOYANT_MMS_SOURCE_T_FIELD, BUOYANT_MMS_SOURCE_U_FIELD,
    BUOYANT_T0, BUOYANT_TEMPERATURE_FIELD,
};
#[allow(unused_imports)]
pub use incompressible_momentum::{
    incompressible_momentum_ale_mms_model, incompressible_momentum_ale_model,
    incompressible_momentum_mms_model, incompressible_momentum_model,
    incompressible_momentum_structured_model, incompressible_momentum_system,
    IncompressibleMomentumFields, INCOMPRESSIBLE_MMS_SOURCE_FIELD,
};
#[allow(unused_imports)]
pub use allmach_pressure::{
    allmach_pressure_ale_mms_model, allmach_pressure_ale_model, allmach_pressure_mms_model,
    allmach_pressure_model, allmach_pressure_system, allmach_thermal_ale_mms_model,
    allmach_thermal_ale_model, allmach_thermal_compressible_mms_ale_model,
    allmach_thermal_compressible_mms_model, allmach_thermal_mms_model, allmach_thermal_model,
    allmach_thermal_structured_model,
    apply_pressure_inlet_nozzle_bcs,
    AllMachPressureFields, ALLMACH_GAMMA, ALLMACH_K_OVER_CP,
    ALLMACH_MMS_SOURCE_P_FIELD, ALLMACH_MMS_SOURCE_T_FIELD, ALLMACH_MMS_SOURCE_U_FIELD,
    ALLMACH_RHO_DT_FIELD, ALLMACH_RHO_T_REF_FIELD, ALLMACH_TEMPERATURE_FIELD, ALLMACH_T_REF,
};
#[allow(unused_imports)]
pub use scalar_transport::{
    scalar_transport_model, scalar_transport_sou_model, ADVECTING_VELOCITY_FIELD,
    KAPPA as SCALAR_TRANSPORT_KAPPA, MMS_SOURCE_FIELD as SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
    SCALAR_FIELD as SCALAR_TRANSPORT_FIELD,
};

/// Build-time model registry.
///
/// This is the single list `build.rs` iterates for WGSL emission and registry generation.
pub fn all_models() -> Result<Vec<ModelSpec>, String> {
    Ok(vec![
        incompressible_momentum_model()?,
        incompressible_momentum_mms_model()?,
        // ALE (moving-mesh) variant: same physics with mesh-relative convection;
        // own id => own generated kernels, so static models stay byte-identical.
        incompressible_momentum_ale_model()?,
        // ALE + manufactured source (prescribed-motion MMS).
        incompressible_momentum_ale_mms_model()?,
        allmach_pressure_model()?,
        allmach_pressure_mms_model()?,
        allmach_thermal_model()?,
        allmach_thermal_mms_model()?,
        // Compressible thermal MMS: kept-physics + manufactured sources (PSI>0 steady
        // order test of the real-EOS/compression/viscous-dissipation operator).
        allmach_thermal_compressible_mms_model()?,
        // ALE variant (mesh-relative convection; byte-identical to static at zero flux).
        allmach_thermal_compressible_mms_ale_model()?,
        // ALE (moving-mesh) variants: same physics with mesh-relative convection;
        // own ids => own generated kernels, so static models stay byte-identical.
        allmach_pressure_ale_model()?,
        allmach_pressure_ale_mms_model()?,
        allmach_thermal_ale_model()?,
        allmach_thermal_ale_mms_model()?,
        buoyant_incompressible_model()?,
        buoyant_incompressible_mms_model()?,
        compressible_model()?,
        compressible_mms_model()?,
        // Distinct id so its (larger, lap-extended) state stride gets its own committed
        // kernel sources instead of reusing compressible_mms's smaller-stride kernels
        // (which would misalign every cell and collapse the solve).
        compressible_mms_biharmonic_model()?,
        generic_diffusion_demo_model()?,
        generic_diffusion_demo_neumann_model()?,
        generic_diffusion_demo_mms_model()?,
        generic_diffusion_demo_mms_dirichlet_model()?,
        generic_diffusion_demo_mms_neumann_model()?,
        generic_diffusion_demo_structured_model()?,
        generic_diffusion_demo_structured_mms_model()?,
        generic_diffusion_demo_structured_ibm_model()?,
        incompressible_momentum_structured_model()?,
        allmach_thermal_structured_model()?,
        compressible_structured_model()?,
        scalar_transport_model()?,
        scalar_transport_sou_model()?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::dimensions::{Density, DivDim, Length, Pressure, Velocity};
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
        // 4 base terms, plus the explicit dev2 transpose viscous term when
        // ViscousStressForm::FullDev2 is the declared default.
        assert!(
            momentum.terms().len() == 4 || momentum.terms().len() == 5,
            "unexpected momentum term count {}",
            momentum.terms().len()
        );
        assert_eq!(momentum.terms()[0].op, TermOp::Ddt);
        match &momentum.terms()[0].coeff {
            Some(Coefficient::Field(field)) => assert_eq!(field.name(), "rho"),
            other => panic!("expected rho coefficient, got {:?}", other),
        }
        assert_eq!(momentum.terms()[1].op, TermOp::Div);
        assert_eq!(momentum.terms()[2].op, TermOp::Laplacian);
        assert_eq!(momentum.terms()[3].op, TermOp::Grad);
        if let Some(dev2) = momentum.terms().get(4) {
            assert_eq!(dev2.op, TermOp::Laplacian);
            assert!(dev2.transpose_dev2, "5th momentum term must be the dev2 form");
        }

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

    /// The ALE variant flags exactly its two convection terms as mesh-relative
    /// (`div(phi,U).bounded()` and `div_flux(phi,p)`), derives `is_ale()`, and —
    /// critically — the static and MMS models stay non-ALE (their generated
    /// kernels must remain byte-identical).
    #[test]
    fn incompressible_momentum_ale_flags_convection_terms_only() {
        let ale = incompressible_momentum_ale_model().expect("ale model");
        assert_eq!(ale.id, "incompressible_momentum_ale");
        assert!(ale.system.is_ale(), "ALE model must derive is_ale()");

        let momentum = &ale.system.equations()[0];
        let div = &momentum.terms()[1];
        assert_eq!(div.op, TermOp::Div);
        assert!(div.bounded, "ALE div term must keep the bounded form");
        assert!(div.relative_to_mesh, "div(phi,U) must be mesh-relative");
        let pressure = &ale.system.equations()[1];
        let div_flux = &pressure.terms()[1];
        assert_eq!(div_flux.op, TermOp::DivFlux);
        assert!(div_flux.relative_to_mesh, "div_flux(phi,p) must be mesh-relative");

        // No other term is flagged.
        let flagged: usize = ale
            .system
            .equations()
            .iter()
            .flat_map(|eq| eq.terms())
            .filter(|t| t.relative_to_mesh)
            .count();
        assert_eq!(flagged, 2, "exactly the two convection terms are flagged");

        // Static + MMS variants are untouched (the do-no-harm invariant).
        for model in [
            incompressible_momentum_model().expect("model"),
            incompressible_momentum_mms_model().expect("mms model"),
        ] {
            assert!(
                !model.system.is_ale(),
                "static model '{}' must not be ALE",
                model.id
            );
        }
    }

    #[test]
    fn incompressible_momentum_model_includes_state_layout() {
        let model = incompressible_momentum_model().expect("model");
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
        let model = compressible_model().expect("model");
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

        let mut model = compressible_model().expect("model");

        // Add an invalid grad_* field: wrong shape (scalar instead of Vector2).
        let fields = CompressibleFields::new();
        model.state_layout = StateLayout::new(vec![
            fields.rho,
            fields.rho_u,
            fields.rho_e,
            fields.p,
            fields.t,
            fields.u,
            vol_scalar("grad_rho", DivDim::<Density, Length>::UNIT),
        ]);

        let with_gradients = crate::solver::model::FluxModuleSpec::Scheme {
            gradients: Some(FluxModuleGradientsSpec::FromStateLayout),
            scheme: FluxSchemeSpec::CentralUpwind(
                compressible::compressible_central_upwind_decl(),
            ),
        };

        // Gradient targets resolve at module creation, so the error surfaces
        // when building the flux module.
        let err = flux_module_module(
            with_gradients,
            &model.system,
            &model.state_layout,
            &model.primitives,
            model.explicit_primitives.as_ref(),
        )
        .unwrap_err();
        assert!(
            err.contains("no grad_<field> fields found") || err.contains("Vector2"),
            "Expected error about missing/invalid gradient fields, got: {}",
            err
        );
    }

    #[test]
    fn boundary_spec_can_build_gpu_tables() {
        let model = generic_diffusion_demo_model().expect("model");
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

    /// Test that validate_module_manifests produces reasonable error messages
    /// for invalid port manifest field specs (using the PortRegistry-based path).
    #[test]
    fn validate_module_manifests_reports_missing_port_manifest_field() {
        use crate::solver::ir::ports::{FieldSpec, PortFieldKind, PortManifest};

        let mut model = incompressible_momentum_model().expect("model");

        // Add a module with a port_manifest referencing a non-existent field
        let bad_module = crate::solver::model::module::KernelBundleModule {
            name: "test_module",
            port_manifest: Some(PortManifest {
                fields: vec![FieldSpec {
                    name: "nonexistent_field",
                    kind: PortFieldKind::Scalar,
                    unit: Pressure::UNIT,
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        model.modules.push(bad_module);

        // Validation should fail with a clear error message
        let err = model
            .validate_module_manifests()
            .expect_err("should fail for missing field");
        assert!(
            err.contains("nonexistent_field") || err.contains("MissingField"),
            "Expected error mentioning missing field, got: {}",
            err
        );
    }

    /// Test that validate_module_manifests reports kind mismatches correctly.
    #[test]
    fn validate_module_manifests_reports_kind_mismatch() {
        use crate::solver::ir::ports::{FieldSpec, PortFieldKind, PortManifest};

        let mut model = incompressible_momentum_model().expect("model");

        // Add a module expecting 'U' to be a Scalar (but it's Vector2 in the model)
        let bad_module = crate::solver::model::module::KernelBundleModule {
            name: "test_module",
            port_manifest: Some(PortManifest {
                fields: vec![FieldSpec {
                    name: "U",                   // U is Vector2 in incompressible_momentum_model
                    kind: PortFieldKind::Scalar, // But we claim it's Scalar
                    unit: Velocity::UNIT,
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        model.modules.push(bad_module);

        // Validation should fail with a kind mismatch error
        let err = model
            .validate_module_manifests()
            .expect_err("should fail for kind mismatch");
        assert!(
            err.contains("U") && (err.contains("kind") || err.contains("FieldKindMismatch")),
            "Expected error mentioning field 'U' and kind mismatch, got: {}",
            err
        );
    }
}
