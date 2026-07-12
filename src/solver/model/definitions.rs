use crate::solver::dimensions::{Length, UnitDimension};
/// Model registry + model specs + boundary specs.
use crate::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use crate::solver::model::backend::ast::{EquationSystem, FieldRef};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::units::UnitDim;
use cfd2_codegen::solver::codegen::bc_table::HostBcTable;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplicitMassClosureProof {
    /// The emitted local mass block must be fully constant. Dynamic f32
    /// arithmetic cannot use exact-real polynomial cancellation as an
    /// invertibility proof because rounding is neither associative nor
    /// distributive.
    ExactSymbolic,
    /// The model deliberately carries nonlinear point-local mass closures.
    /// Generated partial pivoting enforces the domain at every cell/stage and
    /// emits a non-finite rate when the pivoted block falls below the cutoff.
    RuntimePivoted { justification: &'static str },
    /// A model-owned analytic domain proof permits a different power-of-two
    /// equilibration floor than the generic runtime-pivoted contract. The
    /// exponent is part of the mathematical model declaration, not a model-ID
    /// branch in codegen.
    RuntimePivotedScaled {
        justification: &'static str,
        equilibration_floor_power: i32,
    },
}

impl ExplicitMassClosureProof {
    pub(crate) fn equilibration_floor_power(self) -> i32 {
        match self {
            Self::ExactSymbolic | Self::RuntimePivoted { .. } => -19,
            Self::RuntimePivotedScaled {
                equilibration_floor_power,
                ..
            } => equilibration_floor_power,
        }
    }

    pub(crate) fn equilibration_floor(self) -> f32 {
        2.0_f32.powi(self.equilibration_floor_power())
    }
}

impl Default for ExplicitMassClosureProof {
    fn default() -> Self {
        Self::ExactSymbolic
    }
}

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

    /// Proof policy for nonlinear primitive closures used by local DDT mass
    /// coefficients. Exact symbolic proof is the safe default.
    pub explicit_mass_closure_proof: ExplicitMassClosureProof,
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
        let primitives = self
            .explicit_primitives
            .as_ref()
            .unwrap_or(&self.primitives);
        let ordered_primitives = primitives
            .ordered()
            .map_err(|error| format!("model '{}' explicit primitive closure: {error}", self.id))?;
        let mut primitive_identifiers: std::collections::HashSet<String> = ordered_primitives
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let mut primitive_constant_fields: std::collections::HashSet<String> =
            cfd2_codegen::solver::codegen::constants::base_constant_field_names()
                .iter()
                .map(|field| (*field).to_string())
                .collect();
        for module in &self.modules {
            if let Some(manifest) = &module.port_manifest {
                primitive_constant_fields.extend(
                    manifest
                        .params
                        .iter()
                        .map(|parameter| parameter.wgsl_field.to_string()),
                );
            }
        }
        primitive_identifiers.insert("constants".to_string());
        for field in self.state_layout.fields() {
            primitive_identifiers.insert(field.name().to_string());
            let suffixes: &[&str] = match field.kind() {
                crate::solver::ir::FieldKind::Scalar => &[],
                crate::solver::ir::FieldKind::Vector2 => &["x", "y"],
                crate::solver::ir::FieldKind::Vector3 => &["x", "y", "z"],
            };
            for suffix in suffixes {
                primitive_identifiers.insert(format!("{}_{}", field.name(), suffix));
            }
        }
        for (name, expression) in &ordered_primitives {
            validate_point_local_primitive_expr(
                expression,
                &primitive_identifiers,
                &primitive_constant_fields,
            )
            .map_err(|error| {
                format!(
                    "model '{}' explicit primitive '{name}' is not point-local: {error}",
                    self.id
                )
            })?;
        }

        let differential_fields: std::collections::HashSet<&str> = self
            .system
            .equations()
            .iter()
            .filter(|equation| {
                equation.terms().iter().any(|term| {
                    term.op == crate::solver::ir::TermOp::Ddt && term.field == *equation.target()
                })
            })
            .map(|equation| equation.target().name())
            .collect();
        validate_explicit_mass_determinant(
            &self.system,
            &self.state_layout,
            &differential_fields,
            &ordered_primitives,
            self.explicit_mass_closure_proof,
        )
        .map_err(|error| format!("model '{}' {error}", self.id))?;
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
            for source in equation
                .terms()
                .iter()
                .filter(|term| term.op == crate::solver::ir::TermOp::Source)
            {
                validate_explicit_source_term(&self.state_layout, target, source).map_err(
                    |error| {
                        format!(
                            "model '{}' source on row '{}' is not valid for matrix-free explicit assembly: {error}",
                            self.id,
                            target.name()
                        )
                    },
                )?;
            }
            let has_own_ddt = equation
                .terms()
                .iter()
                .any(|term| term.op == crate::solver::ir::TermOp::Ddt && term.field == *target);
            let recoverable = match target.kind() {
                crate::solver::ir::FieldKind::Scalar => primitives
                    .get(target.name())
                    .is_some_and(|expression| {
                        !matches!(
                            expression.node(),
                            cfd2_ir::ast::ExprNode::Ident(name) if name == target.name()
                        )
                    }),
                crate::solver::ir::FieldKind::Vector2 => ["x", "y"].iter().all(|suffix| {
                    let name = format!("{}_{}", target.name(), suffix);
                    primitives.get(&name).is_some_and(|expression| {
                        !matches!(expression.node(), cfd2_ir::ast::ExprNode::Ident(inner) if inner == &name)
                    })
                }),
                crate::solver::ir::FieldKind::Vector3 => ["x", "y", "z"].iter().all(|suffix| {
                    let name = format!("{}_{}", target.name(), suffix);
                    primitives.get(&name).is_some_and(|expression| {
                        !matches!(expression.node(), cfd2_ir::ast::ExprNode::Ident(inner) if inner == &name)
                    })
                }),
            };
            if !has_own_ddt && !recoverable {
                return Err(format!(
                    "model '{}' row '{}' has no own-variable ddt and no explicit primitive recovery; it is not a method-of-lines row",
                    self.id,
                    target.name()
                ));
            }
            if !has_own_ddt {
                if let Some(nonlocal) = equation.terms().iter().find(|term| {
                    term.op != crate::solver::ir::TermOp::Source
                        || term.flux.is_some()
                        || term.bounded
                        || term.transpose_dev2
                        || term.linearize_pressure_flux.is_some()
                        || term.relative_to_mesh
                        || term.viscous_dissipation
                        || term.non_conservative_ale
                }) {
                    return Err(format!(
                        "model '{}' algebraic row '{}' contains nonlocal/nonclosure operator '{}'; explicit primitive recovery may replace only point-local Source rows without gradient/flux/ALE flags (the declared explicit primitive is authoritative for the local closure)",
                        self.id,
                        target.name(),
                        nonlocal.op.as_str()
                    ));
                }
            }
            if has_own_ddt {
                let component_names: Vec<String> = match target.kind() {
                    crate::solver::ir::FieldKind::Scalar => vec![target.name().to_string()],
                    crate::solver::ir::FieldKind::Vector2 => ["x", "y"]
                        .iter()
                        .map(|suffix| format!("{}_{}", target.name(), suffix))
                        .collect(),
                    crate::solver::ir::FieldKind::Vector3 => ["x", "y", "z"]
                        .iter()
                        .map(|suffix| format!("{}_{}", target.name(), suffix))
                        .collect(),
                };
                for component_name in component_names {
                    let Some(expr) = primitives.get(&component_name) else {
                        continue;
                    };
                    let identity = matches!(
                        expr.node(),
                        cfd2_ir::ast::ExprNode::Ident(name) if name == &component_name
                    );
                    if !identity {
                        return Err(format!(
                            "model '{}' explicit primitive '{}' overwrites a differential RK component",
                            self.id, component_name
                        ));
                    }
                }

                for ddt in equation
                    .terms()
                    .iter()
                    .filter(|term| term.op == crate::solver::ir::TermOp::Ddt)
                {
                    if !differential_fields.contains(ddt.field.name()) {
                        return Err(format!(
                            "model '{}' differential row '{}' contains ddt of recoverable algebraic field '{}'; explicit RK needs a declared chain-rule mass coupling or a differential equation for that field",
                            self.id,
                            target.name(),
                            ddt.field.name()
                        ));
                    }
                }
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
    pub fn relaxation_defaults(&self) -> Option<crate::solver::model::module::RelaxationDefaults> {
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

/// Check the preconditions that `unified_assembly` relies on when the
/// method-of-lines lowering turns every source into a direct RHS evaluation.
/// Keeping this beside the explicit capability gate turns declaration errors
/// into deterministic model-validation failures instead of codegen panics.
fn validate_explicit_source_term(
    state_layout: &StateLayout,
    target: &FieldRef,
    term: &crate::solver::ir::Term,
) -> Result<(), String> {
    use crate::solver::ir::{Coefficient, Discretization, FieldKind};

    fn matching_state_field(state_layout: &StateLayout, field: &FieldRef) -> Result<(), String> {
        let Some(state_field) = state_layout.field(field.name()) else {
            return Err(format!(
                "field '{}' is absent from the state layout",
                field.name()
            ));
        };
        if state_field.kind() != field.kind() || state_field.unit() != field.unit() {
            return Err(format!(
                "field '{}' disagrees with the state layout (declared kind={}, unit={}; layout kind={}, unit={})",
                field.name(),
                field.kind().as_str(),
                field.unit(),
                state_field.kind().as_str(),
                state_field.unit()
            ));
        }
        Ok(())
    }

    fn scalar_coefficient(
        state_layout: &StateLayout,
        coefficient: &Coefficient,
    ) -> Result<(), String> {
        match coefficient {
            Coefficient::Constant { value, .. } => {
                if !value.is_finite() {
                    return Err(format!("coefficient constant {value} is not finite"));
                }
            }
            Coefficient::Field(field) => {
                if let Some(state_field) = state_layout.field(field.name()) {
                    if state_field.kind() != FieldKind::Scalar
                        || field.kind() != FieldKind::Scalar
                        || state_field.unit() != field.unit()
                    {
                        return Err(format!(
                            "coefficient field '{}' must match a scalar state field",
                            field.name()
                        ));
                    }
                } else if cfd2_codegen::solver::codegen::coeff_expr::coeff_named_expr_dyn(
                    field.name(),
                )
                .is_none()
                {
                    return Err(format!(
                        "coefficient field '{}' is neither a scalar state field nor a supported uniform coefficient",
                        field.name()
                    ));
                }
            }
            Coefficient::MagSqr(field) => matching_state_field(state_layout, field)?,
            Coefficient::Product(lhs, rhs) => {
                scalar_coefficient(state_layout, lhs)?;
                scalar_coefficient(state_layout, rhs)?;
            }
            Coefficient::Sum(lhs, rhs) => {
                scalar_coefficient(state_layout, lhs)?;
                scalar_coefficient(state_layout, rhs)?;
                if lhs.unit() != rhs.unit() {
                    return Err(format!(
                        "coefficient sum has mismatched units: {} vs {}",
                        lhs.unit(),
                        rhs.unit()
                    ));
                }
            }
        }
        Ok(())
    }

    if term.discretization == Discretization::Implicit {
        matching_state_field(state_layout, &term.field)?;
        if term.field.kind() != target.kind() {
            return Err(format!(
                "implicit reaction field '{}' has kind {}, but target '{}' has kind {}",
                term.field.name(),
                term.field.kind().as_str(),
                target.name(),
                target.kind().as_str()
            ));
        }
        if let Some(coefficient) = &term.coeff {
            scalar_coefficient(state_layout, coefficient)?;
        }
        return Ok(());
    }

    if let Some(direction) = &term.direction {
        if direction.len() != target.kind().component_count() {
            return Err(format!(
                "direction has {} components, but target '{}' has {}",
                direction.len(),
                target.name(),
                target.kind().component_count()
            ));
        }
        if direction.iter().any(|component| !component.is_finite()) {
            return Err("direction contains a non-finite component".to_string());
        }
        if let Some(coefficient) = &term.coeff {
            scalar_coefficient(state_layout, coefficient)?;
        }
        return Ok(());
    }

    if target.kind().component_count() > 1 {
        let Some(Coefficient::Field(source_field)) = &term.coeff else {
            return Err(format!(
                "vector target '{}' requires a matching vector-field coefficient or a direction",
                target.name()
            ));
        };
        matching_state_field(state_layout, source_field)?;
        if source_field.kind() != target.kind() {
            return Err(format!(
                "vector source field '{}' has kind {}, but target '{}' has kind {}",
                source_field.name(),
                source_field.kind().as_str(),
                target.name(),
                target.kind().as_str()
            ));
        }
    } else if let Some(coefficient) = &term.coeff {
        scalar_coefficient(state_layout, coefficient)?;
    }

    Ok(())
}

fn validate_point_local_primitive_expr(
    expression: &cfd2_ir::ast::Expr,
    allowed_identifiers: &std::collections::HashSet<String>,
    allowed_constant_fields: &std::collections::HashSet<String>,
) -> Result<(), String> {
    use cfd2_ir::ast::ExprNode;
    match expression.node() {
        ExprNode::Ident(name) => {
            if allowed_identifiers.contains(name) {
                Ok(())
            } else {
                Err(format!("undeclared identifier '{name}'"))
            }
        }
        ExprNode::Literal(_) => Ok(()),
        ExprNode::Field { base, field } => {
            if matches!(base.node(), ExprNode::Ident(name) if name == "constants") {
                if !allowed_constant_fields.contains(field) {
                    return Err(format!("undeclared constants field '{field}'"));
                }
            } else if !matches!(field.as_str(), "x" | "y" | "z") {
                return Err(format!("unsupported field selection '.{field}'"));
            }
            validate_point_local_primitive_expr(base, allowed_identifiers, allowed_constant_fields)
        }
        ExprNode::Unary { expr: base, .. } => {
            validate_point_local_primitive_expr(base, allowed_identifiers, allowed_constant_fields)
        }
        ExprNode::Binary { left, right, .. } => {
            validate_point_local_primitive_expr(
                left,
                allowed_identifiers,
                allowed_constant_fields,
            )?;
            validate_point_local_primitive_expr(right, allowed_identifiers, allowed_constant_fields)
        }
        ExprNode::Index { .. } => {
            Err("buffer indexing is forbidden in a local closure".to_string())
        }
        ExprNode::Call { callee, args } => {
            let ExprNode::Ident(name) = callee.node() else {
                return Err("closure call target must be a pure builtin identifier".to_string());
            };
            if !matches!(
                name.as_str(),
                "abs"
                    | "acos"
                    | "asin"
                    | "atan"
                    | "ceil"
                    | "clamp"
                    | "cos"
                    | "exp"
                    | "floor"
                    | "log"
                    | "max"
                    | "min"
                    | "pow"
                    | "select"
                    | "sign"
                    | "sin"
                    | "sqrt"
                    | "tan"
            ) {
                return Err(format!("call to non-whitelisted function '{name}'"));
            }
            for argument in args {
                validate_point_local_primitive_expr(
                    argument,
                    allowed_identifiers,
                    allowed_constant_fields,
                )?;
            }
            Ok(())
        }
    }
}

/// A tiny canonical polynomial used only to prove that the declared local ddt
/// block is not *identically* singular. Each `Coefficient` is a product tree,
/// so a sorted symbolic monomial plus its constant factor is sufficient; sums
/// arise only when several ddt terms target the same matrix entry and during
/// determinant expansion.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExplicitMassAtom {
    StateField(String),
    Uniform(String),
    VectorMagnitudeSquared(String),
    /// Canonical point-local primitive expression whose nonlinear operation
    /// is outside the polynomial subset. Equal closures share this atom, so
    /// determinant cancellation still sees aliases such as `a=f(q), b=f(q)`.
    PrimitiveClosure(String),
}

type ExplicitMassMonomial = Vec<ExplicitMassAtom>;

#[derive(Clone, Debug, Default, PartialEq)]
struct ExplicitMassPolynomial(BTreeMap<ExplicitMassMonomial, BigRational>);

const EXPLICIT_MASS_POLYNOMIAL_TERM_BUDGET: usize = 100_000;
const EXPLICIT_MASS_DETERMINANT_WORK_BUDGET: usize = 1_000_000;

impl ExplicitMassPolynomial {
    fn constant(value: f64) -> Self {
        if value == 0.0 {
            Self::default()
        } else {
            Self(BTreeMap::from([(Vec::new(), exact_f64_rational(value))]))
        }
    }

    fn monomial(value: BigRational, mut factors: Vec<ExplicitMassAtom>) -> Self {
        if value.is_zero() {
            return Self::default();
        }
        factors.sort();
        Self(BTreeMap::from([(factors, value)]))
    }

    fn atom(atom: ExplicitMassAtom) -> Self {
        Self::monomial(BigRational::from_integer(BigInt::from(1u8)), vec![atom])
    }

    fn constant_f32(&self) -> Option<f32> {
        if self.0.is_empty() {
            return Some(0.0);
        }
        if self.0.len() != 1 {
            return None;
        }
        let (factors, value) = self.0.first_key_value()?;
        factors.is_empty().then(|| value.to_f32()).flatten()
    }

    /// Matrix-entry construction in WGSL starts from 0f and executes one `+=`
    /// per ddt declaration. Preserve f32 rounding for constant-only sums; exact
    /// rational addition would accept blocks that become singular after codegen.
    fn add_emitted_term(&mut self, rhs: &Self) -> Result<(), String> {
        if let (Some(left), Some(right)) = (self.constant_f32(), rhs.constant_f32()) {
            let sum = left + right;
            if !sum.is_finite() {
                return Err("explicit ddt constant sum is non-finite in f32".to_string());
            }
            *self = Self::monomial(exact_f64_rational(sum as f64), Vec::new());
            return Ok(());
        }
        self.add_assign(rhs, false)?;
        Ok(())
    }

    fn emitted_product(&self, rhs: &Self) -> Result<Self, String> {
        if let (Some(left), Some(right)) = (self.constant_f32(), rhs.constant_f32()) {
            let product = left * right;
            if !product.is_finite() {
                return Err("explicit ddt constant product is non-finite in f32".to_string());
            }
            return Ok(Self::monomial(
                exact_f64_rational(product as f64),
                Vec::new(),
            ));
        }
        self.multiplied(rhs)
    }

    fn emitted_sum(&self, rhs: &Self, negative: bool) -> Result<Self, String> {
        if let (Some(left), Some(right)) = (self.constant_f32(), rhs.constant_f32()) {
            let sum = if negative { left - right } else { left + right };
            if !sum.is_finite() {
                return Err("explicit primitive constant sum is non-finite in f32".to_string());
            }
            return Ok(Self::monomial(exact_f64_rational(sum as f64), Vec::new()));
        }
        let mut result = self.clone();
        result.add_assign(rhs, negative)?;
        Ok(result)
    }

    fn add_assign(&mut self, rhs: &Self, negative: bool) -> Result<(), String> {
        for (monomial, coefficient) in &rhs.0 {
            if !self.0.contains_key(monomial)
                && self.0.len() >= EXPLICIT_MASS_POLYNOMIAL_TERM_BUDGET
            {
                return Err(format!(
                    "explicit mass symbolic polynomial exceeded {} terms",
                    EXPLICIT_MASS_POLYNOMIAL_TERM_BUDGET
                ));
            }
            let value = self
                .0
                .entry(monomial.clone())
                .or_insert_with(BigRational::zero);
            if negative {
                *value -= coefficient;
            } else {
                *value += coefficient;
            }
        }
        self.0.retain(|_, value| !value.is_zero());
        Ok(())
    }

    fn multiplied(&self, rhs: &Self) -> Result<Self, String> {
        let mut result = Self::default();
        for (left_factors, left_value) in &self.0 {
            for (right_factors, right_value) in &rhs.0 {
                let mut factors = left_factors.clone();
                factors.extend(right_factors.iter().cloned());
                factors.sort();
                if !result.0.contains_key(&factors)
                    && result.0.len() >= EXPLICIT_MASS_POLYNOMIAL_TERM_BUDGET
                {
                    return Err(format!(
                        "explicit mass symbolic polynomial exceeded {} terms",
                        EXPLICIT_MASS_POLYNOMIAL_TERM_BUDGET
                    ));
                }
                *result.0.entry(factors).or_insert_with(BigRational::zero) +=
                    left_value * right_value;
            }
        }
        result.0.retain(|_, value| !value.is_zero());
        Ok(result)
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn contains_opaque_closure(&self) -> bool {
        self.0.keys().any(|monomial| {
            monomial
                .iter()
                .any(|atom| matches!(atom, ExplicitMassAtom::PrimitiveClosure(_)))
        })
    }

    fn contains_dynamic_atom(&self) -> bool {
        self.0.keys().any(|monomial| !monomial.is_empty())
    }
}

/// Convert an IEEE-754 `f64` to its exact binary rational. Validation must not
/// use an epsilon: pruning a tiny coefficient before determinant products can
/// turn an exactly singular matrix into an apparently nonsingular one.
fn exact_f64_rational(value: f64) -> BigRational {
    debug_assert!(value.is_finite());
    if value == 0.0 {
        return BigRational::zero();
    }
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mantissa, exponent) = if exponent_bits == 0 {
        (fraction, -1022 - 52)
    } else {
        ((1u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    let mut numerator = BigInt::from(mantissa);
    if negative {
        numerator = -numerator;
    }
    if exponent >= 0 {
        BigRational::from_integer(numerator << exponent as usize)
    } else {
        BigRational::new(numerator, BigInt::from(1u8) << (-exponent) as usize)
    }
}

/// Lower the polynomial subset of ordered explicit primitive closures and
/// canonicalize every remaining point-local expression as an opaque atom.
/// This is deliberately closure-aware: fields recovered from the same
/// expression are not algebraically independent mass coefficients.
fn explicit_primitive_mass_values(
    ordered_primitives: &[(String, cfd2_ir::ast::Expr)],
    state_fields: &HashMap<&str, (crate::solver::ir::FieldKind, UnitDim)>,
) -> Result<HashMap<String, ExplicitMassPolynomial>, String> {
    use cfd2_ir::ast::{BinaryOp, Expr, ExprNode, Literal, UnaryOp};

    fn fingerprint(expr: &Expr, derived: &HashMap<String, String>) -> String {
        match expr.node() {
            ExprNode::Literal(value) => format!("lit:{value}"),
            ExprNode::Ident(name) => derived
                .get(name)
                .cloned()
                .unwrap_or_else(|| format!("id:{name}")),
            ExprNode::Field { base, field } => {
                format!("field:{}.{field}", fingerprint(base, derived))
            }
            ExprNode::Index { base, index } => format!(
                "index:{}[{}]",
                fingerprint(base, derived),
                fingerprint(index, derived)
            ),
            ExprNode::Unary { op, expr } => {
                format!("unary:{op}({})", fingerprint(expr, derived))
            }
            ExprNode::Binary { left, op, right } => {
                let mut left = fingerprint(left, derived);
                let mut right = fingerprint(right, derived);
                if matches!(op, BinaryOp::Add | BinaryOp::Mul) && right < left {
                    std::mem::swap(&mut left, &mut right);
                }
                format!("binary:{op}({left},{right})")
            }
            ExprNode::Call { callee, args } => {
                let args = args
                    .iter()
                    .map(|arg| fingerprint(arg, derived))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("call:{}({args})", fingerprint(callee, derived))
            }
        }
    }

    fn numeric_literal(value: &Literal) -> Option<f32> {
        match value {
            Literal::Float(text) => text.parse::<f32>().ok(),
            Literal::Int(value) => Some(*value as f32),
            Literal::Uint(value) => Some(*value as f32),
            Literal::Bool(_) => None,
        }
    }

    fn lower(
        expr: &Expr,
        derived: &HashMap<String, ExplicitMassPolynomial>,
        state_fields: &HashMap<&str, (crate::solver::ir::FieldKind, UnitDim)>,
    ) -> Result<Option<ExplicitMassPolynomial>, String> {
        match expr.node() {
            ExprNode::Literal(value) => {
                Ok(numeric_literal(value)
                    .map(|value| ExplicitMassPolynomial::constant(value as f64)))
            }
            ExprNode::Ident(name) => {
                if let Some(value) = derived.get(name) {
                    return Ok(Some(value.clone()));
                }
                if name == "constants" {
                    return Ok(None);
                }
                if let Some(&(kind, _)) = state_fields.get(name.as_str()) {
                    if kind != crate::solver::ir::FieldKind::Scalar {
                        return Ok(None);
                    }
                }
                Ok(Some(ExplicitMassPolynomial::atom(
                    ExplicitMassAtom::StateField(name.clone()),
                )))
            }
            ExprNode::Field { base, field } if matches!(base.node(), ExprNode::Ident(name) if name == "constants") => {
                Ok(Some(ExplicitMassPolynomial::atom(
                    ExplicitMassAtom::Uniform(field.clone()),
                )))
            }
            ExprNode::Unary {
                op: UnaryOp::Negate,
                expr,
            } => {
                let Some(value) = lower(expr, derived, state_fields)? else {
                    return Ok(None);
                };
                Ok(Some(value.emitted_product(
                    &ExplicitMassPolynomial::constant(-1.0),
                )?))
            }
            ExprNode::Binary { left, op, right } => {
                let Some(left) = lower(left, derived, state_fields)? else {
                    return Ok(None);
                };
                let Some(right) = lower(right, derived, state_fields)? else {
                    return Ok(None);
                };
                match op {
                    BinaryOp::Add => Ok(Some(left.emitted_sum(&right, false)?)),
                    BinaryOp::Sub => Ok(Some(left.emitted_sum(&right, true)?)),
                    BinaryOp::Mul => Ok(Some(left.emitted_product(&right)?)),
                    BinaryOp::Div => {
                        let Some(divisor) = right.constant_f32() else {
                            return Ok(None);
                        };
                        if divisor == 0.0 || !divisor.is_finite() {
                            return Ok(None);
                        }
                        Ok(Some(left.emitted_product(
                            &ExplicitMassPolynomial::constant((1.0f32 / divisor) as f64),
                        )?))
                    }
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    let mut values = HashMap::new();
    let mut fingerprints = HashMap::new();
    for (name, expression) in ordered_primitives {
        let canonical = fingerprint(expression, &fingerprints);
        let value = lower(expression, &values, state_fields)?.unwrap_or_else(|| {
            ExplicitMassPolynomial::atom(ExplicitMassAtom::PrimitiveClosure(canonical.clone()))
        });
        values.insert(name.clone(), value);
        fingerprints.insert(name.clone(), canonical);
    }
    Ok(values)
}

fn coefficient_mass_polynomial(
    coefficient: Option<&crate::solver::model::backend::ast::Coefficient>,
    state_fields: &HashMap<&str, (crate::solver::ir::FieldKind, UnitDim)>,
    primitive_values: &HashMap<String, ExplicitMassPolynomial>,
) -> Result<ExplicitMassPolynomial, String> {
    use crate::solver::model::backend::ast::Coefficient;

    fn field_polynomial(
        field: &FieldRef,
        state_fields: &HashMap<&str, (crate::solver::ir::FieldKind, UnitDim)>,
        primitive_values: &HashMap<String, ExplicitMassPolynomial>,
    ) -> Result<ExplicitMassPolynomial, String> {
        let name = field.name();
        if let Some(&(slot_kind, slot_unit)) = state_fields.get(name) {
            if slot_kind != field.kind() || slot_unit != field.unit() {
                return Err(format!(
                    "explicit ddt coefficient field '{name}' kind/unit does not match its state slot"
                ));
            }
            if slot_kind != crate::solver::ir::FieldKind::Scalar {
                return Err(format!(
                    "explicit ddt coefficient field '{name}' resolves to a non-scalar state slot"
                ));
            }
            return Ok(primitive_values.get(name).cloned().unwrap_or_else(|| {
                ExplicitMassPolynomial::atom(ExplicitMassAtom::StateField(name.to_string()))
            }));
        }
        let uniform = match name {
            "rho" => "density",
            "mu" | "nu" => "viscosity",
            "eos_gamma" | "eos_gm1" | "eos_r" | "eos_dp_drho" | "eos_p_ref"
            | "eos_theta_ref" | "eos_rho_ref" | "buoyant_beta_g" | "buoyant_t0"
            | "buoyant_k_over_cp" => name,
            // These lower to compound runtime expressions rather than one
            // independently varying scalar atom. Reject instead of inventing
            // a symbolic identity that differs from generated WGSL.
            "inv_dt" | "kappa" => {
                return Err(format!(
                    "explicit ddt coefficient '{name}' has a compound named lowering unsupported by the symbolic mass proof"
                ));
            }
            _ => {
                return Err(format!(
                    "explicit ddt coefficient field '{name}' is neither a state slot nor a known runtime uniform"
                ));
            }
        };
        Ok(ExplicitMassPolynomial::atom(ExplicitMassAtom::Uniform(
            uniform.to_string(),
        )))
    }

    fn emitted_constant(value: f64) -> Result<BigRational, String> {
        let emitted = value as f32;
        if !emitted.is_finite() {
            return Err(format!(
                "explicit ddt coefficient constant {value:?} is non-finite after f32 emission"
            ));
        }
        Ok(exact_f64_rational(emitted as f64))
    }

    fn lower(
        coefficient: &Coefficient,
        state_fields: &HashMap<&str, (crate::solver::ir::FieldKind, UnitDim)>,
        primitive_values: &HashMap<String, ExplicitMassPolynomial>,
    ) -> Result<ExplicitMassPolynomial, String> {
        match coefficient {
            Coefficient::Constant { value, .. } => Ok(ExplicitMassPolynomial::monomial(
                emitted_constant(*value)?,
                Vec::new(),
            )),
            Coefficient::Field(field) => field_polynomial(field, state_fields, primitive_values),
            Coefficient::MagSqr(field) => {
                let Some(&(slot_kind, slot_unit)) = state_fields.get(field.name()) else {
                    return Err(format!(
                        "explicit ddt mag_sqr coefficient '{}' has no state slot",
                        field.name()
                    ));
                };
                if slot_kind != field.kind() || slot_unit != field.unit() {
                    return Err(format!(
                        "explicit ddt mag_sqr field '{}' kind/unit does not match its state slot",
                        field.name()
                    ));
                }
                if slot_kind == crate::solver::ir::FieldKind::Scalar {
                    // Scalar mag_sqr(a) is exactly a*a; use the same canonical
                    // atoms so determinant cancellation can see that identity.
                    let value = field_polynomial(field, state_fields, primitive_values)?;
                    value.multiplied(&value)
                } else {
                    // Vector |U|^2 is a sum of squared components, which the
                    // scalar coefficient tree does not expose separately.
                    Ok(ExplicitMassPolynomial::atom(
                        ExplicitMassAtom::VectorMagnitudeSquared(field.name().to_string()),
                    ))
                }
            }
            Coefficient::Product(left, right) => {
                let left = lower(left, state_fields, primitive_values)?;
                let right = lower(right, state_fields, primitive_values)?;
                left.emitted_product(&right)
            }
            Coefficient::Sum(left, right) => {
                let mut left = lower(left, state_fields, primitive_values)?;
                let right = lower(right, state_fields, primitive_values)?;
                left.add_emitted_term(&right)?;
                Ok(left)
            }
        }
    }

    let Some(coefficient) = coefficient else {
        return Ok(ExplicitMassPolynomial::constant(1.0));
    };
    lower(coefficient, state_fields, primitive_values)
}

fn explicit_mass_determinant(
    matrix: &[Vec<ExplicitMassPolynomial>],
) -> Result<ExplicitMassPolynomial, String> {
    fn recurse(
        matrix: &[Vec<ExplicitMassPolynomial>],
        row: usize,
        used_columns: u64,
        memo: &mut HashMap<(usize, u64), ExplicitMassPolynomial>,
        work: &mut usize,
    ) -> Result<ExplicitMassPolynomial, String> {
        if row == matrix.len() {
            return Ok(ExplicitMassPolynomial::constant(1.0));
        }
        if let Some(cached) = memo.get(&(row, used_columns)) {
            return Ok(cached.clone());
        }
        let mut result = ExplicitMassPolynomial::default();
        for column in 0..matrix.len() {
            let bit = 1u64 << column;
            if used_columns & bit != 0 || matrix[row][column].is_zero() {
                continue;
            }
            let tail = recurse(matrix, row + 1, used_columns | bit, memo, work)?;
            let multiply_pairs = matrix[row][column].0.len().saturating_mul(tail.0.len());
            *work = work.saturating_add(multiply_pairs);
            if *work > EXPLICIT_MASS_DETERMINANT_WORK_BUDGET {
                return Err(format!(
                    "explicit mass determinant exceeded symbolic work budget {}",
                    EXPLICIT_MASS_DETERMINANT_WORK_BUDGET
                ));
            }
            let product = matrix[row][column].multiplied(&tail)?;
            // Appending `column` contributes one inversion for every already
            // chosen column greater than it.
            let inversions = (used_columns >> (column + 1)).count_ones();
            let sign = if inversions % 2 == 0 { 1.0 } else { -1.0 };
            result.add_assign(&product, sign < 0.0)?;
        }
        memo.insert((row, used_columns), result.clone());
        Ok(result)
    }

    recurse(matrix, 0, 0, &mut HashMap::new(), &mut 0usize)
}

fn validate_explicit_mass_determinant<'a>(
    system: &'a EquationSystem,
    state_layout: &'a StateLayout,
    differential_fields: &std::collections::HashSet<&'a str>,
    ordered_primitives: &[(String, cfd2_ir::ast::Expr)],
    closure_proof: ExplicitMassClosureProof,
) -> Result<(), String> {
    let state_fields: HashMap<&str, (crate::solver::ir::FieldKind, UnitDim)> = state_layout
        .fields()
        .iter()
        .map(|field| (field.name(), (field.kind(), field.unit())))
        .collect();
    let primitive_values = explicit_primitive_mass_values(ordered_primitives, &state_fields)?;
    let mut scalar_offsets = HashMap::<&str, usize>::new();
    let mut dimension = 0usize;
    for equation in system.equations() {
        if differential_fields.contains(equation.target().name()) {
            scalar_offsets.insert(equation.target().name(), dimension);
            dimension += equation.target().kind().component_count();
        }
    }
    if dimension == 0 {
        return Err("explicit RK4 declares no differential mass block".to_string());
    }
    if dimension > 8 {
        return Err(format!(
            "explicit RK4 differential mass block has {dimension} scalar rows; exact symbolic invertibility proof is capped at 8 to bound worst-case determinant expansion"
        ));
    }

    let mut matrix = vec![vec![ExplicitMassPolynomial::default(); dimension]; dimension];
    // Preserve runtime-dependence provenance independently of polynomial
    // simplification. Dynamic f32 additions are not associative, so an entry
    // whose exact-real terms cancel can still emit a rounded nonzero value.
    let mut matrix_dynamic = vec![vec![false; dimension]; dimension];
    let mut matrix_opaque = vec![vec![false; dimension]; dimension];
    for equation in system.equations() {
        let Some(&row_base) = scalar_offsets.get(equation.target().name()) else {
            continue;
        };
        let row_components = equation.target().kind().component_count();
        let own_terms: Vec<_> = equation
            .terms()
            .iter()
            .filter(|term| {
                term.op == crate::solver::ir::TermOp::Ddt && term.field == *equation.target()
            })
            .collect();
        if own_terms.is_empty() {
            return Err(format!(
                "differential row '{}' must declare at least one own-variable ddt",
                equation.target().name(),
            ));
        }

        for term in equation
            .terms()
            .iter()
            .filter(|term| term.op == crate::solver::ir::TermOp::Ddt)
        {
            let Some(&column_base) = scalar_offsets.get(term.field.name()) else {
                // The existing capability diagnostic reports this with the
                // field/row names after this structural pass.
                continue;
            };
            let column_components = term.field.kind().component_count();
            if row_components != column_components {
                return Err(format!(
                    "differential row '{}' ddt field '{}' has component mismatch {} vs {}",
                    equation.target().name(),
                    term.field.name(),
                    row_components,
                    column_components
                ));
            }
            let coefficient =
                coefficient_mass_polynomial(term.coeff.as_ref(), &state_fields, &primitive_values)?;
            let coefficient_dynamic = coefficient.contains_dynamic_atom();
            let coefficient_opaque = coefficient.contains_opaque_closure();
            for component in 0..row_components {
                matrix_dynamic[row_base + component][column_base + component] |=
                    coefficient_dynamic;
                matrix_opaque[row_base + component][column_base + component] |=
                    coefficient_opaque;
                matrix[row_base + component][column_base + component]
                    .add_emitted_term(&coefficient)?;
            }
        }
    }

    // Guarded runtime row pivoting handles zero declaration-order pivots, so
    // only the full symbolic determinant must be non-identically-zero here.
    // This is a generic algebraic proof, not a pointwise domain proof for
    // dynamic field coefficients; physical positivity/closure invariants still
    // matter at runtime.
    let mass_uses_dynamic = matrix_dynamic.iter().flatten().any(|&dynamic| dynamic);
    let mass_uses_opaque_closure = matrix_opaque.iter().flatten().any(|&opaque| opaque);
    let determinant = explicit_mass_determinant(&matrix)?;
    if determinant.is_zero() {
        return Err(
            "explicit RK4 local ddt mass block is identically singular after f32 constant emission"
                .to_string(),
        );
    }
    if mass_uses_dynamic {
        match closure_proof {
            ExplicitMassClosureProof::ExactSymbolic => {
                let detail = if mass_uses_opaque_closure {
                    "a nonlinear/opaque primitive closure"
                } else {
                    "dynamic f32 mass arithmetic"
                };
                return Err(format!(
                    "explicit RK4 mass determinant depends on {detail}; exact-real polynomial identities do not prove emitted-f32 invertibility, so declare a model-owned RuntimePivoted mass-domain proof"
                ));
            }
            ExplicitMassClosureProof::RuntimePivoted { justification }
            | ExplicitMassClosureProof::RuntimePivotedScaled {
                justification, ..
            } => {
                if justification.trim().is_empty() {
                    return Err(
                        "explicit RK4 RuntimePivoted mass-domain proof requires a nonempty justification"
                            .to_string(),
                    );
                }
            }
        }
    }
    if let ExplicitMassClosureProof::RuntimePivotedScaled {
        equilibration_floor_power,
        ..
    } = closure_proof
    {
        if !(-30..=-1).contains(&equilibration_floor_power) {
            return Err(format!(
                "explicit RK4 model-owned equilibration floor exponent {equilibration_floor_power} is outside the supported [-30,-1] range"
            ));
        }
    }

    // Generic runtime conditioning remains cheap and auditable only for scalar
    // or 2x2 connected blocks. Shipped all-Mach has two isolated momentum
    // scalars plus one p/T block. Larger blocks are still supported when fully
    // constant (and receive the build-time rcond audit below), but a dynamic
    // model must provide a future structure-specific certificate rather than
    // silently relying on a weak determinant heuristic.
    let mut component_for = vec![usize::MAX; dimension];
    let mut component_sizes = Vec::<usize>::new();
    for seed in 0..dimension {
        if component_for[seed] != usize::MAX {
            continue;
        }
        let component = component_sizes.len();
        let mut stack = vec![seed];
        component_for[seed] = component;
        let mut size = 0usize;
        while let Some(node) = stack.pop() {
            size += 1;
            for other in 0..dimension {
                let connected = !matrix[node][other].is_zero()
                    || !matrix[other][node].is_zero()
                    || matrix_dynamic[node][other]
                    || matrix_dynamic[other][node];
                if component_for[other] == usize::MAX && connected {
                    component_for[other] = component;
                    stack.push(other);
                }
            }
        }
        component_sizes.push(size);
    }
    let mut component_dynamic = vec![false; component_sizes.len()];
    for row in 0..dimension {
        for column in 0..dimension {
            if matrix_dynamic[row][column] {
                component_dynamic[component_for[row]] = true;
            }
        }
    }
    if let Some((component, &size)) = component_sizes
        .iter()
        .enumerate()
        .find(|(component, size)| component_dynamic[*component] && **size > 2)
    {
        return Err(format!(
            "explicit RK4 runtime-dependent mass component {component} has rank {size}; guarded RuntimePivoted blocks are limited to rank 2"
        ));
    }

    // Audit every fully constant connected component independently, even when
    // an unrelated component is runtime-dependent. A dynamic scalar must not
    // disable the rcond certificate for a disconnected constant 3x3 block.
    for component in 0..component_sizes.len() {
        if component_dynamic[component] {
            continue;
        }
        let members: Vec<usize> = component_for
            .iter()
            .enumerate()
            .filter_map(|(rank, &owner)| (owner == component).then_some(rank))
            .collect();
        let size = members.len();
        let mut values: Vec<Vec<f32>> = members
            .iter()
            .map(|&row| {
                members
                    .iter()
                    .map(|&column| {
                        matrix[row][column].constant_f32().ok_or_else(|| {
                            format!(
                                "explicit RK4 mass component {component} was classified constant but entry ({row},{column}) is dynamic"
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        const REL_FLOOR: f32 = 1.0e-6;
        let original_values = values.clone();
        for row in 0..size {
            if size == 1 {
                break;
            }
            let scale = values[row]
                .iter()
                .fold(0.0_f32, |scale, value| scale.max(value.abs()));
            if !(scale > 0.0) || !scale.is_finite() {
                return Err(format!(
                    "explicit RK4 constant mass component {component} row {row} has no finite nonzero normalization"
                ));
            }
            for column in 0..size {
                let raw = values[row][column];
                let normalized = raw / scale;
                if !normalized.is_finite()
                    || (raw != 0.0 && normalized.abs() < f32::MIN_POSITIVE)
                {
                    return Err(format!(
                        "explicit RK4 constant mass component {component} row normalization loses normal f32 precision at row {row}, column {column}"
                    ));
                }
                values[row][column] = normalized;
            }
        }
        let mut column_scales = vec![0.0_f32; size];
        for row in 0..size {
            for column in 0..size {
                column_scales[column] = column_scales[column].max(values[row][column].abs());
            }
        }
        for row in 0..size {
            if size == 1 {
                break;
            }
            for column in 0..size {
                values[row][column] /= column_scales[column];
                if !values[row][column].is_finite() {
                    return Err(format!(
                        "explicit RK4 constant mass component {component} column normalization is non-finite at row {row}, column {column}"
                    ));
                }
            }
        }
        if size > 1 {
            let runtime_quality = if size == 2 {
                let diagonal = values[0][0] * values[1][1];
                let cross = values[0][1] * values[1][0];
                ((diagonal - cross).abs()
                    / diagonal.abs().max(cross.abs()).max(f32::from_bits(1)))
                    .min(1.0)
            } else {
                1.0e-5
            };
            let minimum_column_scale = column_scales
                .iter()
                .copied()
                .fold(f32::MAX, f32::min);
            let conditioning_scale = minimum_column_scale * runtime_quality;
            let equilibration_floor = closure_proof.equilibration_floor();
            if !conditioning_scale.is_finite() || conditioning_scale < equilibration_floor {
                return Err(format!(
                    "explicit RK4 constant mass component {component} f32 equilibration conditioning scale {conditioning_scale:.3e} is below the 2^{} safety floor",
                    closure_proof.equilibration_floor_power(),
                ));
            }
        }
        for pivot in 0..size {
            for row in (pivot + 1)..size {
                if values[row][pivot].abs() > values[pivot][pivot].abs() {
                    values.swap(pivot, row);
                }
            }
            let diagonal = values[pivot][pivot];
            if !diagonal.is_finite()
                || diagonal.abs() < f32::from_bits(1)
                || (size > 1 && diagonal.abs() < REL_FLOOR)
            {
                return Err(format!(
                    "explicit RK4 constant mass component {component} pivot {pivot} is zero/non-finite or below the generated 1e-6 equilibrated cutoff"
                ));
            }
            for row in (pivot + 1)..size {
                let factor = values[row][pivot] / diagonal;
                if !factor.is_finite() {
                    return Err(format!(
                        "explicit RK4 constant mass component {component} elimination factor at pivot {pivot}, row {row} is non-finite"
                    ));
                }
                for column in pivot..size {
                    values[row][column] -= factor * values[pivot][column];
                    if !values[row][column].is_finite() {
                        return Err(format!(
                            "explicit RK4 constant mass component {component} elimination overflow at pivot {pivot}, row {row}, column {column}"
                        ));
                    }
                }
            }
        }
        let reciprocal_condition = constant_mass_reciprocal_condition_inf(&original_values)
            .ok_or_else(|| {
                format!("explicit RK4 constant mass component {component} has no finite inverse")
            })?;
        if reciprocal_condition < 1.0e-5 {
            return Err(format!(
                "explicit RK4 constant mass component {component} reciprocal infinity-norm condition after row/column equilibration {reciprocal_condition:.3e} is below the f32 safety floor 1e-5"
            ));
        }
    }
    Ok(())
}

/// Row/column-equilibrated reciprocal infinity-norm condition estimate of the
/// exact f32 matrix values, evaluated in f64 at model-validation time. The
/// diagonal scaling removes arbitrary variable-unit magnitudes. Constant local
/// blocks pay no runtime estimator cost, yet cannot enter generated f32
/// elimination with a condition number that makes a finite grossly wrong rate
/// plausible.
fn constant_mass_reciprocal_condition_inf(matrix: &[Vec<f32>]) -> Option<f64> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let row_scales: Vec<f64> = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| (*value as f64).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    if row_scales.iter().any(|scale| !(*scale > 0.0) || !scale.is_finite()) {
        return None;
    }
    let mut column_scales = vec![0.0_f64; n];
    for row in 0..n {
        for column in 0..n {
            column_scales[column] = column_scales[column]
                .max((matrix[row][column] as f64).abs() / row_scales[row]);
        }
    }
    if column_scales
        .iter()
        .any(|scale| !(*scale > 0.0) || !scale.is_finite())
    {
        return None;
    }
    let equilibrated: Vec<Vec<f64>> = (0..n)
        .map(|row| {
            (0..n)
                .map(|column| {
                    matrix[row][column] as f64 / row_scales[row] / column_scales[column]
                })
                .collect()
        })
        .collect();
    let norm = equilibrated
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    if !(norm > 0.0) || !norm.is_finite() {
        return None;
    }

    let mut augmented = vec![vec![0.0_f64; 2 * n]; n];
    for row in 0..n {
        for column in 0..n {
            augmented[row][column] = equilibrated[row][column];
        }
        augmented[row][n + row] = 1.0;
    }
    for pivot in 0..n {
        let best = (pivot..n).max_by(|left, right| {
            augmented[*left][pivot]
                .abs()
                .total_cmp(&augmented[*right][pivot].abs())
        })?;
        if augmented[best][pivot] == 0.0 || !augmented[best][pivot].is_finite() {
            return None;
        }
        augmented.swap(pivot, best);
        let diagonal = augmented[pivot][pivot];
        for column in 0..(2 * n) {
            augmented[pivot][column] /= diagonal;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for column in 0..(2 * n) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    let inverse_norm = augmented
        .iter()
        .map(|row| row[n..].iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let product = norm * inverse_norm;
    (product > 0.0 && product.is_finite()).then_some(1.0 / product)
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

#[path = "definitions/allmach_pressure.rs"]
mod allmach_pressure;
#[path = "definitions/buoyant_incompressible.rs"]
mod buoyant_incompressible;
#[path = "definitions/compressible.rs"]
mod compressible;
#[path = "definitions/generic_diffusion_demo.rs"]
mod generic_diffusion_demo;
#[path = "definitions/incompressible_momentum.rs"]
mod incompressible_momentum;
#[path = "definitions/scalar_transport.rs"]
mod scalar_transport;

#[allow(unused_imports)]
pub use allmach_pressure::{
    allmach_pressure_ale_mms_model, allmach_pressure_ale_model, allmach_pressure_mms_model,
    allmach_pressure_model, allmach_pressure_system, allmach_thermal_ale_mms_model,
    allmach_thermal_ale_model, allmach_thermal_compressible_mms_ale_model,
    allmach_thermal_compressible_mms_model, allmach_thermal_mms_model, allmach_thermal_model,
    allmach_thermal_structured_model, apply_pressure_inlet_nozzle_bcs, AllMachPressureFields,
    ALLMACH_GAMMA, ALLMACH_K_OVER_CP, ALLMACH_MMS_SOURCE_P_FIELD, ALLMACH_MMS_SOURCE_T_FIELD,
    ALLMACH_MMS_SOURCE_U_FIELD, ALLMACH_RHO_DT_FIELD, ALLMACH_RHO_T_REF_FIELD,
    ALLMACH_TEMPERATURE_FIELD, ALLMACH_T_REF,
};
#[allow(unused_imports)]
pub use buoyant_incompressible::{
    buoyant_incompressible_mms_model, buoyant_incompressible_model, BUOYANT_BETA_G,
    BUOYANT_K_OVER_CP, BUOYANT_MMS_SOURCE_T_FIELD, BUOYANT_MMS_SOURCE_U_FIELD, BUOYANT_T0,
    BUOYANT_TEMPERATURE_FIELD,
};
#[allow(unused_imports)]
pub use compressible::{
    compressible_central_upwind_decl, compressible_generalized_wave_speed_sq,
    compressible_mms_biharmonic_model, compressible_mms_model, compressible_model,
    compressible_model_with_eos, compressible_structured_model, compressible_system,
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
pub use incompressible_momentum::{
    incompressible_momentum_ale_mms_model, incompressible_momentum_ale_model,
    incompressible_momentum_mms_model, incompressible_momentum_model,
    incompressible_momentum_structured_model, incompressible_momentum_system,
    IncompressibleMomentumFields, INCOMPRESSIBLE_MMS_SOURCE_FIELD,
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

    fn explicit_test_model(
        system: crate::solver::ir::EquationSystem,
        fields: Vec<crate::solver::ir::FieldRef>,
    ) -> ModelSpec {
        let mut model = generic_diffusion_demo_model().expect("base model");
        model.system = system;
        model.state_layout = crate::solver::ir::StateLayout::new(fields);
        model.primitives = crate::solver::model::PrimitiveDerivations::default();
        model.explicit_primitives = None;
        model
    }

    #[test]
    fn explicit_capability_rejects_unlowered_algebraic_time_derivatives() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        q_equation.add_term(crate::solver::ir::fvm::ddt(algebraic));
        let algebraic_equation = crate::solver::ir::Equation::new(algebraic);
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(algebraic_equation);

        let mut model = generic_diffusion_demo_model().expect("base model");
        model.system = system;
        model.state_layout = crate::solver::ir::StateLayout::new(vec![q, algebraic]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::ident("q"))]
                .into_iter()
                .collect(),
        });

        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("ddt") && error.contains("algebraic field 'a'"),
            "unexpected cross-ddt rejection: {error}"
        );
    }

    #[test]
    fn explicit_capability_rejects_closure_overwrite_of_differential_state() {
        let mut model = generic_diffusion_demo_model().expect("base model");
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("phi".to_string(), cfd2_ir::ast::Expr::lit_f32(0.0))]
                .into_iter()
                .collect(),
        });
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("overwrites a differential RK component"),
            "unexpected primitive overwrite rejection: {error}"
        );
    }

    #[test]
    fn explicit_capability_rejects_spatial_algebraic_rows() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        let mut algebraic_equation = crate::solver::ir::Equation::new(algebraic);
        algebraic_equation.add_term(crate::solver::ir::fvm::laplacian(
            Coefficient::constant(1.0),
            algebraic,
        ));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(algebraic_equation);

        let mut model = generic_diffusion_demo_model().expect("base model");
        model.system = system;
        model.state_layout = crate::solver::ir::StateLayout::new(vec![q, algebraic]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::ident("q"))]
                .into_iter()
                .collect(),
        });

        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("algebraic row 'a'") && error.contains("laplacian"),
            "unexpected spatial-algebraic rejection: {error}"
        );
    }

    #[test]
    fn explicit_capability_requires_a_point_local_nonidentity_algebraic_closure() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        let mut algebraic_equation = crate::solver::ir::Equation::new(algebraic);
        algebraic_equation.add_term(crate::solver::ir::fvm::source(algebraic));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(algebraic_equation);

        let mut identity = explicit_test_model(system.clone(), vec![q, algebraic]);
        identity.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::ident("a"))]
                .into_iter()
                .collect(),
        });
        let identity_error = identity.validate_explicit_rk4().unwrap_err();
        assert!(identity_error.contains("no explicit primitive recovery"));

        let mut indexed = explicit_test_model(system.clone(), vec![q, algebraic]);
        indexed.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [(
                "a".to_string(),
                cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::lit_u32(0)),
            )]
            .into_iter()
            .collect(),
        });
        let indexed_error = indexed.validate_explicit_rk4().unwrap_err();
        assert!(indexed_error.contains("buffer indexing is forbidden"));

        let mut typo = explicit_test_model(system, vec![q, algebraic]);
        typo.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::ident("typo_q"))]
                .into_iter()
                .collect(),
        });
        let typo_error = typo.validate_explicit_rk4().unwrap_err();
        assert!(typo_error.contains("undeclared identifier 'typo_q'"));
    }

    #[test]
    fn explicit_capability_rejects_nonlocal_flags_on_algebraic_source_rows() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        let mut algebraic_equation = crate::solver::ir::Equation::new(algebraic);
        algebraic_equation
            .add_term(crate::solver::ir::fvc::source(algebraic).with_viscous_dissipation());
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(algebraic_equation);
        let mut model = explicit_test_model(system, vec![q, algebraic]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::ident("q"))]
                .into_iter()
                .collect(),
        });
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(error.contains("without gradient/flux/ALE flags"));
    }

    #[test]
    fn explicit_capability_rejects_missing_algebraic_reaction_field() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let missing = crate::solver::ir::vol_scalar_dim::<Dimensionless>("missing");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        let mut algebraic_equation = crate::solver::ir::Equation::new(algebraic);
        algebraic_equation.add_term(crate::solver::ir::fvm::source(missing));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(algebraic_equation);

        let mut model = explicit_test_model(system, vec![q, algebraic]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::ident("q"))]
                .into_iter()
                .collect(),
        });
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("missing") && error.contains("absent from the state layout"),
            "unexpected missing-source rejection: {error}"
        );
    }

    #[test]
    fn explicit_capability_rejects_bad_vector_source_direction() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_vector_dim::<Dimensionless>("a");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        let mut algebraic_equation = crate::solver::ir::Equation::new(algebraic);
        algebraic_equation
            .add_term(crate::solver::ir::fvc::source(algebraic).with_direction(vec![1.0]));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(algebraic_equation);

        let mut model = explicit_test_model(system, vec![q, algebraic]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [
                ("a_x".to_string(), cfd2_ir::ast::Expr::ident("q")),
                ("a_y".to_string(), cfd2_ir::ast::Expr::ident("q")),
            ]
            .into_iter()
            .collect(),
        });
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("direction has 1 components") && error.contains("target 'a' has 2"),
            "unexpected vector-direction rejection: {error}"
        );
    }

    #[test]
    fn explicit_capability_rejects_missing_explicit_scalar_coefficient() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let missing = crate::solver::ir::vol_scalar_dim::<Dimensionless>("missing_coeff");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        let mut algebraic_equation = crate::solver::ir::Equation::new(algebraic);
        algebraic_equation.add_term(crate::solver::ir::fvc::source_coeff(
            Coefficient::Field(missing),
            algebraic,
        ));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(algebraic_equation);

        let mut model = explicit_test_model(system, vec![q, algebraic]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::ident("q"))]
                .into_iter()
                .collect(),
        });
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("missing_coeff") && error.contains("supported uniform coefficient"),
            "unexpected missing-coefficient rejection: {error}"
        );
    }

    #[test]
    fn explicit_capability_rejects_symbolically_singular_mass_block() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");
        let mut q_equation = crate::solver::ir::Equation::new(q);
        q_equation.add_term(crate::solver::ir::fvm::ddt(q));
        q_equation.add_term(crate::solver::ir::fvm::ddt(r));
        let mut r_equation = crate::solver::ir::Equation::new(r);
        r_equation.add_term(crate::solver::ir::fvm::ddt(q));
        r_equation.add_term(crate::solver::ir::fvm::ddt(r));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_equation);
        system.add_equation(r_equation);

        let mut model = generic_diffusion_demo_model().expect("base model");
        model.system = system;
        model.state_layout = crate::solver::ir::StateLayout::new(vec![q, r]);
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("identically singular"),
            "unexpected singular-mass rejection: {error}"
        );
    }

    #[test]
    fn explicit_capability_rejects_zero_own_ddt_coefficient() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let mut equation = crate::solver::ir::Equation::new(q);
        equation.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::constant(0.0),
            q,
        ));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(equation);

        let mut model = generic_diffusion_demo_model().expect("base model");
        model.system = system;
        model.state_layout = crate::solver::ir::StateLayout::new(vec![q]);
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("identically singular"),
            "unexpected zero-ddt rejection: {error}"
        );
    }

    #[test]
    fn explicit_mass_guard_accepts_invertible_block_requiring_runtime_row_pivot() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");
        let s = crate::solver::ir::vol_scalar_dim::<Dimensionless>("s");
        let mut q_eq = crate::solver::ir::Equation::new(q);
        q_eq.add_term(crate::solver::ir::fvm::ddt(q));
        q_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut r_eq = crate::solver::ir::Equation::new(r);
        r_eq.add_term(crate::solver::ir::fvm::ddt(q));
        r_eq.add_term(crate::solver::ir::fvm::ddt(r));
        r_eq.add_term(crate::solver::ir::fvm::ddt(s));
        let mut s_eq = crate::solver::ir::Equation::new(s);
        s_eq.add_term(crate::solver::ir::fvm::ddt(r));
        s_eq.add_term(crate::solver::ir::fvm::ddt(s));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(r_eq);
        system.add_equation(s_eq);
        explicit_test_model(system, vec![q, r, s])
            .validate_explicit_rk4()
            .expect("det=-1 block is invertible with guarded row pivoting");
    }

    #[test]
    fn explicit_mass_guard_uses_full_partial_pivoting_above_floor() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");
        let mut q_eq = crate::solver::ir::Equation::new(q);
        q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::constant(1.0e-19),
            q,
        ));
        q_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut r_eq = crate::solver::ir::Equation::new(r);
        r_eq.add_term(crate::solver::ir::fvm::ddt(q));
        r_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(r_eq);
        explicit_test_model(system, vec![q, r])
            .validate_explicit_rk4()
            .expect("well-conditioned block must pivot on the unit lower entry");
    }

    #[test]
    fn explicit_mass_guard_substitutes_algebraic_primitive_closures() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let algebraic = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");

        // Runtime closure a:=1 makes [[a,1],[1,1]] exactly singular. Treating
        // `a` as an independent symbolic atom is an unsound capability proof.
        let mut q_eq = crate::solver::ir::Equation::new(q);
        q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::Field(algebraic),
            q,
        ));
        q_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut algebraic_eq = crate::solver::ir::Equation::new(algebraic);
        algebraic_eq.add_term(crate::solver::ir::fvm::source(algebraic));
        let mut r_eq = crate::solver::ir::Equation::new(r);
        r_eq.add_term(crate::solver::ir::fvm::ddt(q));
        r_eq.add_term(crate::solver::ir::fvm::ddt(r));

        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(algebraic_eq);
        system.add_equation(r_eq);
        let mut model = explicit_test_model(system, vec![q, algebraic, r]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [("a".to_string(), cfd2_ir::ast::Expr::lit_f32(1.0))]
                .into_iter()
                .collect(),
        });
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(
            error.contains("identically singular"),
            "closure-forced singular block was accepted: {error}"
        );
    }

    #[test]
    fn explicit_mass_guard_canonicalizes_equal_primitive_aliases() {
        use cfd2_ir::dimensions::Dimensionless;

        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let a = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");
        let b = crate::solver::ir::vol_scalar_dim::<Dimensionless>("b");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");
        let mut q_eq = crate::solver::ir::Equation::new(q);
        q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(Coefficient::Field(a), q));
        q_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut a_eq = crate::solver::ir::Equation::new(a);
        a_eq.add_term(crate::solver::ir::fvm::source(a));
        let mut b_eq = crate::solver::ir::Equation::new(b);
        b_eq.add_term(crate::solver::ir::fvm::source(b));
        let mut r_eq = crate::solver::ir::Equation::new(r);
        r_eq.add_term(crate::solver::ir::fvm::ddt_coeff(Coefficient::Field(b), q));
        r_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(a_eq);
        system.add_equation(b_eq);
        system.add_equation(r_eq);
        let mut model = explicit_test_model(system, vec![q, a, b, r]);
        model.explicit_primitives = Some(crate::solver::model::PrimitiveDerivations {
            derivations: [
                ("a".to_string(), cfd2_ir::ast::Expr::ident("q")),
                ("b".to_string(), cfd2_ir::ast::Expr::ident("q")),
            ]
            .into_iter()
            .collect(),
        });
        let error = model.validate_explicit_rk4().unwrap_err();
        assert!(error.contains("identically singular"));
    }

    #[test]
    fn explicit_mass_guard_matches_f32_constant_emission_and_accumulation() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");

        let build = |q_diag: Vec<f64>, product_diag: Option<(f64, f64)>, lower_cross: f64| {
            let mut q_eq = crate::solver::ir::Equation::new(q);
            if let Some((left, right)) = product_diag {
                let product =
                    Coefficient::product(Coefficient::constant(left), Coefficient::constant(right))
                        .unwrap();
                q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(product, q));
            } else {
                for value in &q_diag {
                    q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
                        Coefficient::constant(*value),
                        q,
                    ));
                }
            }
            q_eq.add_term(crate::solver::ir::fvm::ddt(r));
            let mut r_eq = crate::solver::ir::Equation::new(r);
            r_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
                Coefficient::constant(lower_cross),
                q,
            ));
            r_eq.add_term(crate::solver::ir::fvm::ddt(r));
            let mut system = crate::solver::ir::EquationSystem::new();
            system.add_equation(q_eq);
            system.add_equation(r_eq);
            explicit_test_model(system, vec![q, r])
        };

        let leaf_rounding = build(vec![1.0 + 2f64.powi(-25)], None, 1.0)
            .validate_explicit_rk4()
            .unwrap_err();
        assert!(leaf_rounding.contains("identically singular"));

        let sum_rounding = build(vec![1.0, 2f64.powi(-25)], None, 1.0)
            .validate_explicit_rk4()
            .unwrap_err();
        assert!(sum_rounding.contains("identically singular"));

        let a = 1.0 + 2f64.powi(-23);
        let product_rounding = build(Vec::new(), Some((a, a)), 1.0 + 2f64.powi(-22))
            .validate_explicit_rk4()
            .unwrap_err();
        assert!(product_rounding.contains("identically singular"));
    }

    #[test]
    fn explicit_mass_guard_canonicalizes_scalar_mag_sqr_and_uniform_aliases() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");
        let a = crate::solver::ir::vol_scalar_dim::<Dimensionless>("a");

        let mut q_eq = crate::solver::ir::Equation::new(q);
        q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::mag_sqr(a),
            q,
        ));
        q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::field(a).unwrap(),
            r,
        ));
        let mut r_eq = crate::solver::ir::Equation::new(r);
        r_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::field(a).unwrap(),
            q,
        ));
        r_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(r_eq);
        let mag_error = explicit_test_model(system, vec![q, r, a])
            .validate_explicit_rk4()
            .unwrap_err();
        assert!(mag_error.contains("identically singular"));

        let mu = crate::solver::ir::vol_scalar_dim::<Dimensionless>("mu");
        let nu = crate::solver::ir::vol_scalar_dim::<Dimensionless>("nu");
        let mut q_eq = crate::solver::ir::Equation::new(q);
        q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::field(mu).unwrap(),
            q,
        ));
        q_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut r_eq = crate::solver::ir::Equation::new(r);
        r_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::field(nu).unwrap(),
            q,
        ));
        r_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(r_eq);
        let alias_error = explicit_test_model(system, vec![q, r])
            .validate_explicit_rk4()
            .unwrap_err();
        assert!(alias_error.contains("identically singular"));
    }

    #[test]
    fn explicit_mass_guard_accepts_well_scaled_nonzero_scalar_mass() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let mut equation = crate::solver::ir::Equation::new(q);
        equation.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::constant(2f64.powi(-100)),
            q,
        ));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(equation);
        explicit_test_model(system, vec![q])
            .validate_explicit_rk4()
            .expect("an isolated nonzero scalar mass has condition number one regardless of units");
    }

    #[test]
    fn explicit_mass_guard_resolves_coefficient_kind_from_state_layout() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let real_u = crate::solver::ir::vol_vector_dim::<Dimensionless>("U");
        let fake_scalar_u = crate::solver::ir::vol_scalar_dim::<Dimensionless>("U");
        let mut equation = crate::solver::ir::Equation::new(q);
        equation.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::mag_sqr(fake_scalar_u),
            q,
        ));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(equation);
        let error = explicit_test_model(system, vec![q, real_u])
            .validate_explicit_rk4()
            .unwrap_err();
        assert!(error.contains("kind/unit does not match its state slot"));
    }

    #[test]
    fn explicit_mass_guard_rejects_unrepresentable_constant_scaling() {
        use cfd2_ir::dimensions::Dimensionless;
        let q = crate::solver::ir::vol_scalar_dim::<Dimensionless>("q");
        let r = crate::solver::ir::vol_scalar_dim::<Dimensionless>("r");
        let s = crate::solver::ir::vol_scalar_dim::<Dimensionless>("s");
        let mut q_eq = crate::solver::ir::Equation::new(q);
        q_eq.add_term(crate::solver::ir::fvm::ddt(q));
        q_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::constant(1.0e10),
            s,
        ));
        let mut r_eq = crate::solver::ir::Equation::new(r);
        r_eq.add_term(crate::solver::ir::fvm::ddt_coeff(
            Coefficient::constant(1.0e30),
            q,
        ));
        r_eq.add_term(crate::solver::ir::fvm::ddt(r));
        let mut s_eq = crate::solver::ir::Equation::new(s);
        s_eq.add_term(crate::solver::ir::fvm::ddt(s));
        let mut system = crate::solver::ir::EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(r_eq);
        system.add_equation(s_eq);
        let error = explicit_test_model(system, vec![q, r, s])
            .validate_explicit_rk4()
            .unwrap_err();
        assert!(
            error.contains("equilibration conditioning scale")
                || error.contains("elimination overflow")
                || error.contains("below the generated"),
            "unexpected ill-conditioned constant-block rejection: {error}"
        );
    }

    #[test]
    fn shipped_coupled_explicit_mass_blocks_are_symbolically_nonsingular() {
        for model in [
            compressible_model().expect("compressible"),
            compressible_structured_model().expect("structured compressible"),
            allmach_thermal_model().expect("all-Mach thermal"),
            allmach_thermal_structured_model().expect("structured all-Mach thermal"),
        ] {
            model
                .validate_explicit_rk4()
                .unwrap_or_else(|error| panic!("{} explicit mass rejected: {error}", model.id));
        }
    }

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
            assert!(
                dev2.transpose_dev2,
                "5th momentum term must be the dev2 form"
            );
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
        assert!(
            div_flux.relative_to_mesh,
            "div_flux(phi,p) must be mesh-relative"
        );

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
        // p row: target + rho_e + two implicit rho terms + the four explicit
        // gauge/reference constants of the state-form closure (all zero-valued
        // at zero gauge references).
        assert_eq!(model.system.equations()[4].terms().len(), 8);
        // T row: merged target + p + the explicit -gauge_p_ref constant.
        assert_eq!(model.system.equations()[5].terms().len(), 3);

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
            scheme: FluxSchemeSpec::CentralUpwind(compressible::compressible_central_upwind_decl()),
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
