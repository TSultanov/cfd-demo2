use std::sync::Arc;

use cfd2_codegen::solver::codegen::KernelWgsl;
use cfd2_ir::solver::ir::ports::{
    ParamSpec, PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec,
};
use cfd2_ir::solver::ir::StateLayout;

/// Stable identifier for a compute kernel.
///
/// This is used by the unified solver orchestration to decouple scheduling and lookup
/// from handwritten enums/matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub &'static str);

impl KernelId {
    pub const FLUX_MODULE_GRADIENTS: KernelId = KernelId("flux_module_gradients");
    pub const FLUX_MODULE: KernelId = KernelId("flux_module");

    pub const COMPRESSIBLE_VISCOUS_P_DIV_U: KernelId = KernelId("compressible/viscous_p_div_u");

    pub const GENERIC_COUPLED_ASSEMBLY: KernelId = KernelId("generic_coupled_assembly");
    pub const GENERIC_COUPLED_ASSEMBLY_GRAD_STATE: KernelId =
        KernelId("generic_coupled_assembly_grad_state");
    pub const GENERIC_COUPLED_APPLY: KernelId = KernelId("generic_coupled_apply");
    pub const GENERIC_COUPLED_UPDATE: KernelId = KernelId("generic_coupled_update");

    // Handwritten solver-infrastructure kernels (single-entrypoint compute shaders).
    pub const DOT_PRODUCT: KernelId = KernelId("dot_product");
    pub const DOT_PRODUCT_PAIR: KernelId = KernelId("dot_product_pair");
    pub const OUTER_CONVERGENCE: KernelId = KernelId("outer_convergence");

    pub const SCALARS_INIT_CG: KernelId = KernelId("scalars/init_cg_scalars");
    pub const SCALARS_REDUCE_RHO_NEW_R_R: KernelId = KernelId("scalars/reduce_rho_new_r_r");
    pub const SCALARS_REDUCE_R0_V: KernelId = KernelId("scalars/reduce_r0_v");

    pub const LINEAR_SOLVER_SPMV_P_V: KernelId = KernelId("linear_solver/spmv_p_v");
    pub const LINEAR_SOLVER_CG_UPDATE_X_R: KernelId = KernelId("linear_solver/cg_update_x_r");
    pub const LINEAR_SOLVER_CG_UPDATE_P: KernelId = KernelId("linear_solver/cg_update_p");

    // Handwritten solver-infrastructure kernels (multi-entrypoint compute shaders).
    pub const AMG_SMOOTH_OP: KernelId = KernelId("amg/smooth_op");
    pub const AMG_RESTRICT_RESIDUAL: KernelId = KernelId("amg/restrict_residual");
    pub const AMG_PROLONGATE_OP: KernelId = KernelId("amg/prolongate_op");
    pub const AMG_CLEAR: KernelId = KernelId("amg/clear");

    pub const AMG_PACK_PACK_COMPONENT: KernelId = KernelId("amg_pack/pack_component");
    pub const AMG_PACK_UNPACK_COMPONENT: KernelId = KernelId("amg_pack/unpack_component");

    pub const BLOCK_PRECOND_BUILD_BLOCK_INV: KernelId = KernelId("block_precond/build_block_inv");
    pub const BLOCK_PRECOND_APPLY_BLOCK_PRECOND: KernelId =
        KernelId("block_precond/apply_block_precond");

    pub const SCHUR_PRECOND_PREDICT_AND_FORM: KernelId =
        KernelId("schur_precond/predict_and_form_schur");
    pub const SCHUR_PRECOND_RELAX_PRESSURE: KernelId = KernelId("schur_precond/relax_pressure");
    pub const SCHUR_PRECOND_CORRECT_VELOCITY: KernelId = KernelId("schur_precond/correct_velocity");

    pub const SCHUR_GENERIC_PRECOND_PREDICT_AND_FORM: KernelId =
        KernelId("schur_precond_generic/predict_and_form_schur");
    pub const SCHUR_GENERIC_PRECOND_RELAX_PRESSURE: KernelId =
        KernelId("schur_precond_generic/relax_pressure");
    pub const SCHUR_GENERIC_PRECOND_CORRECT_VELOCITY: KernelId =
        KernelId("schur_precond_generic/correct_velocity");

    pub const GENERIC_COUPLED_SCHUR_SETUP_BUILD_DIAG_AND_PRESSURE: KernelId =
        KernelId("generic_coupled_schur_setup/build_diag_and_pressure");

    pub const GMRES_OPS_SPMV: KernelId = KernelId("gmres_ops/spmv");
    pub const GMRES_OPS_AXPY: KernelId = KernelId("gmres_ops/axpy");
    pub const GMRES_OPS_AXPY_FROM_Y: KernelId = KernelId("gmres_ops/axpy_from_y");
    pub const GMRES_OPS_AXPBY: KernelId = KernelId("gmres_ops/axpby");
    pub const GMRES_OPS_SCALE: KernelId = KernelId("gmres_ops/scale");
    pub const GMRES_OPS_SCALE_IN_PLACE: KernelId = KernelId("gmres_ops/scale_in_place");
    pub const GMRES_OPS_COPY: KernelId = KernelId("gmres_ops/copy");
    pub const GMRES_OPS_DOT_PRODUCT_PARTIAL: KernelId = KernelId("gmres_ops/dot_product_partial");
    pub const GMRES_OPS_NORM_SQ_PARTIAL: KernelId = KernelId("gmres_ops/norm_sq_partial");
    pub const GMRES_OPS_REDUCE_FINAL: KernelId = KernelId("gmres_ops/reduce_final");
    pub const GMRES_OPS_REDUCE_FINAL_AND_FINISH_NORM: KernelId =
        KernelId("gmres_ops/reduce_final_and_finish_norm");

    pub const GMRES_LOGIC_UPDATE_HESSENBERG_GIVENS: KernelId =
        KernelId("gmres_logic/update_hessenberg_givens");
    pub const GMRES_LOGIC_SOLVE_TRIANGULAR: KernelId = KernelId("gmres_logic/solve_triangular");

    pub const GMRES_CGS_CALC_DOTS: KernelId = KernelId("gmres_cgs/calc_dots_cgs");
    pub const GMRES_CGS_REDUCE_DOTS: KernelId = KernelId("gmres_cgs/reduce_dots_cgs");
    pub const GMRES_CGS_UPDATE_W: KernelId = KernelId("gmres_cgs/update_w_cgs");

    pub fn as_str(self) -> &'static str {
        self.0
    }
}

/// Build-time-generated kernels.
///
/// Kernel filenames are derived from `KernelId` by replacing `/` with `_`.
///
/// To support pluggable numerical modules (Gap 0 in `CODEGEN_PLAN.md`), per-model kernel WGSL
/// generators are looked up via the model's module list.
/// Model-owned kernel phase classification (GPU-agnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelPhaseId {
    Preparation,
    Gradients,
    FluxComputation,
    Assembly,
    Apply,
    Update,
}

/// Model-owned dispatch kind (GPU-agnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKindId {
    Cells,
    Faces,
}

/// Kernel inclusion conditions expressed on the model side and evaluated when building a recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelConditionId {
    /// Always include this kernel when the module is present.
    Always,

    /// Include only when the recipe allocates a packed `grad_state` buffer.
    RequiresGradState,

    /// Include only when the recipe does not allocate a packed `grad_state` buffer.
    RequiresNoGradState,

    /// Include only when the solver stepping mode is implicit.
    RequiresImplicitStepping,
}

/// A fully specified kernel pass derived from the model + method selection.
///
/// This is the "model side" of a recipe: phase/dispatch are decided here so the
/// GPU recipe does not need per-kernel `match` arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelKernelSpec {
    pub id: KernelId,
    pub phase: KernelPhaseId,
    pub dispatch: DispatchKindId,
    pub condition: KernelConditionId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KernelFusionPolicy {
    Off,
    Safe,
    Aggressive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelFusionStepping {
    Explicit,
    Implicit,
    Coupled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionGuard {
    RequiresGradState,
    RequiresNoGradState,
    RequiresStepping(KernelFusionStepping),
    RequiresImplicitOrCoupled,
    RequiresModule(&'static str),
    MinPolicy(KernelFusionPolicy),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelPatternAtom {
    pub id: KernelId,
    pub dispatch: Option<DispatchKindId>,
}

impl KernelPatternAtom {
    pub const fn id(id: KernelId) -> Self {
        Self { id, dispatch: None }
    }

    pub const fn with_dispatch(id: KernelId, dispatch: DispatchKindId) -> Self {
        Self {
            id,
            dispatch: Some(dispatch),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelKernelFusionRule {
    pub name: &'static str,
    pub priority: i32,
    pub phase: KernelPhaseId,
    pub pattern: Vec<KernelPatternAtom>,
    pub replacement: ModelKernelSpec,
    pub guards: Vec<FusionGuard>,
}

#[derive(Debug, Clone)]
pub struct AppliedFusionRule {
    pub name: &'static str,
    pub start_index: usize,
    pub pattern_len: usize,
    pub replacement: KernelId,
}

#[derive(Debug, Clone)]
pub struct FusionResult {
    pub kernels: Vec<ModelKernelSpec>,
    pub applied: Vec<AppliedFusionRule>,
}

#[derive(Debug, Clone, Copy)]
pub struct KernelFusionContext<'a> {
    pub policy: KernelFusionPolicy,
    pub stepping: KernelFusionStepping,
    pub has_grad_state: bool,
    pub module_names: &'a [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelKernelArtifact {
    Wgsl(KernelWgsl),
    DslProgram(crate::solver::ir::KernelProgram),
}

impl ModelKernelArtifact {
    pub fn into_wgsl(self) -> Result<KernelWgsl, String> {
        match self {
            ModelKernelArtifact::Wgsl(kernel) => Ok(kernel),
            ModelKernelArtifact::DslProgram(program) => {
                cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl(&program)
            }
        }
    }
}

pub(crate) type ModelKernelArtifactGenerator = Arc<
    dyn Fn(
            &crate::solver::model::ModelSpec,
            &crate::solver::ir::SchemeRegistry,
        ) -> Result<ModelKernelArtifact, String>
        + Send
        + Sync
        + 'static,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelWgslScope {
    PerModel,
    Shared,
}

#[derive(Clone)]
pub struct ModelKernelGeneratorSpec {
    pub id: KernelId,
    pub scope: KernelWgslScope,
    pub generator: ModelKernelArtifactGenerator,
}

impl ModelKernelGeneratorSpec {
    pub fn new(
        id: KernelId,
        generator: impl Fn(
                &crate::solver::model::ModelSpec,
                &crate::solver::ir::SchemeRegistry,
            ) -> Result<KernelWgsl, String>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            id,
            scope: KernelWgslScope::PerModel,
            generator: Arc::new(move |model, schemes| {
                generator(model, schemes).map(ModelKernelArtifact::Wgsl)
            }),
        }
    }

    pub fn new_shared(
        id: KernelId,
        generator: impl Fn(
                &crate::solver::model::ModelSpec,
                &crate::solver::ir::SchemeRegistry,
            ) -> Result<KernelWgsl, String>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            id,
            scope: KernelWgslScope::Shared,
            generator: Arc::new(move |model, schemes| {
                generator(model, schemes).map(ModelKernelArtifact::Wgsl)
            }),
        }
    }

    pub fn new_dsl(
        id: KernelId,
        generator: impl Fn(
                &crate::solver::model::ModelSpec,
                &crate::solver::ir::SchemeRegistry,
            ) -> Result<crate::solver::ir::KernelProgram, String>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            id,
            scope: KernelWgslScope::PerModel,
            generator: Arc::new(move |model, schemes| {
                generator(model, schemes).map(ModelKernelArtifact::DslProgram)
            }),
        }
    }

    pub fn new_shared_dsl(
        id: KernelId,
        generator: impl Fn(
                &crate::solver::model::ModelSpec,
                &crate::solver::ir::SchemeRegistry,
            ) -> Result<crate::solver::ir::KernelProgram, String>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            id,
            scope: KernelWgslScope::Shared,
            generator: Arc::new(move |model, schemes| {
                generator(model, schemes).map(ModelKernelArtifact::DslProgram)
            }),
        }
    }
}

impl std::fmt::Debug for ModelKernelGeneratorSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelKernelGeneratorSpec")
            .field("id", &self.id.as_str())
            .finish_non_exhaustive()
    }
}

pub fn derive_kernel_specs_for_model(
    model: &crate::solver::model::ModelSpec,
) -> Result<Vec<ModelKernelSpec>, String> {
    let mut kernels = Vec::new();

    for module in &model.modules {
        let module: &dyn crate::solver::model::module::ModelModule = module;
        kernels.extend_from_slice(module.kernel_specs());
    }
    Ok(kernels)
}

pub fn derive_kernel_fusion_rules_for_model(
    model: &crate::solver::model::ModelSpec,
) -> Vec<ModelKernelFusionRule> {
    let mut rules = Vec::new();
    for module in &model.modules {
        let module: &dyn crate::solver::model::module::ModelModule = module;
        rules.extend_from_slice(module.fusion_rules());
    }
    rules
}

fn guard_matches(guard: FusionGuard, ctx: &KernelFusionContext<'_>) -> bool {
    match guard {
        FusionGuard::RequiresGradState => ctx.has_grad_state,
        FusionGuard::RequiresNoGradState => !ctx.has_grad_state,
        FusionGuard::RequiresStepping(s) => ctx.stepping == s,
        FusionGuard::RequiresImplicitOrCoupled => matches!(
            ctx.stepping,
            KernelFusionStepping::Implicit | KernelFusionStepping::Coupled
        ),
        FusionGuard::RequiresModule(name) => ctx.module_names.contains(&name),
        FusionGuard::MinPolicy(min) => ctx.policy >= min,
    }
}

fn rule_enabled(rule: &ModelKernelFusionRule, ctx: &KernelFusionContext<'_>) -> bool {
    rule.guards.iter().all(|&g| guard_matches(g, ctx))
}

fn rule_matches_at(
    rule: &ModelKernelFusionRule,
    kernels: &[ModelKernelSpec],
    start: usize,
) -> bool {
    if rule.pattern.is_empty() {
        return false;
    }
    if start + rule.pattern.len() > kernels.len() {
        return false;
    }

    for (offset, atom) in rule.pattern.iter().enumerate() {
        let spec = kernels[start + offset];
        if spec.phase != rule.phase {
            return false;
        }
        if spec.id != atom.id {
            return false;
        }
        if let Some(dispatch) = atom.dispatch {
            if spec.dispatch != dispatch {
                return false;
            }
        }
    }

    true
}

pub fn apply_model_fusion_rules(
    kernels: &[ModelKernelSpec],
    rules: &[ModelKernelFusionRule],
    ctx: &KernelFusionContext<'_>,
) -> FusionResult {
    if ctx.policy == KernelFusionPolicy::Off || rules.is_empty() {
        return FusionResult {
            kernels: kernels.to_vec(),
            applied: Vec::new(),
        };
    }

    let mut ranked_rules: Vec<(usize, &ModelKernelFusionRule)> = rules
        .iter()
        .enumerate()
        .filter(|(_, r)| rule_enabled(r, ctx))
        .collect();
    ranked_rules.sort_by(|(ia, a), (ib, b)| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| b.pattern.len().cmp(&a.pattern.len()))
            .then_with(|| ia.cmp(ib))
    });

    let mut out = Vec::with_capacity(kernels.len());
    let mut applied = Vec::new();
    let mut i = 0usize;

    while i < kernels.len() {
        let mut matched: Option<&ModelKernelFusionRule> = None;
        for (_, rule) in &ranked_rules {
            if rule_matches_at(rule, kernels, i) {
                matched = Some(rule);
                break;
            }
        }

        if let Some(rule) = matched {
            out.push(rule.replacement);
            applied.push(AppliedFusionRule {
                name: rule.name,
                start_index: i,
                pattern_len: rule.pattern.len(),
                replacement: rule.replacement.id,
            });
            i += rule.pattern.len();
        } else {
            out.push(kernels[i]);
            i += 1;
        }
    }

    FusionResult {
        kernels: out,
        applied,
    }
}

pub fn derive_fusion_replacement_kernel_ids_for_model(
    model: &crate::solver::model::ModelSpec,
) -> Vec<KernelId> {
    let mut seen = std::collections::HashSet::<KernelId>::new();
    let mut ids = Vec::new();
    for rule in derive_kernel_fusion_rules_for_model(model) {
        let id = rule.replacement.id;
        if seen.insert(id) {
            ids.push(id);
        }
    }
    ids
}

pub fn kernel_output_name_for_model(model_id: &str, kernel_id: KernelId) -> Result<String, String> {
    let prefix = kernel_id.as_str().replace('/', "_");
    if model_id.is_empty() {
        Ok(format!("{prefix}.wgsl"))
    } else {
        Ok(format!("{prefix}_{model_id}.wgsl"))
    }
}

/// Convert a StateLayout to a ResolvedStateSlotsSpec for use in codegen.
fn resolved_slots_from_layout(layout: &StateLayout) -> ResolvedStateSlotsSpec {
    let mut slots = Vec::new();
    for field in layout.fields() {
        let kind = match field.kind() {
            cfd2_ir::solver::ir::FieldKind::Scalar => PortFieldKind::Scalar,
            cfd2_ir::solver::ir::FieldKind::Vector2 => PortFieldKind::Vector2,
            cfd2_ir::solver::ir::FieldKind::Vector3 => PortFieldKind::Vector3,
        };
        slots.push(ResolvedStateSlotSpec {
            name: field.name().to_string(),
            kind,
            unit: field.unit(),
            base_offset: field.offset(),
        });
    }
    ResolvedStateSlotsSpec {
        stride: layout.stride(),
        slots,
    }
}

/// Extract EOS params for WGSL generation.
///
/// Uses the model's "eos" module port manifest if present, otherwise falls back
/// to the canonical EOS param list to ensure shared kernels generate identical
/// WGSL across all models.
pub(crate) fn extract_eos_params(model: &crate::solver::model::ModelSpec) -> Vec<ParamSpec> {
    model
        .modules
        .iter()
        .find(|m| m.name == "eos")
        .and_then(|m| m.port_manifest.as_ref())
        .map(|p| p.params.clone())
        .unwrap_or_else(|| {
            crate::solver::model::modules::eos_ports::eos_uniform_port_manifest().params
        })
}

pub(crate) fn generate_generic_coupled_assembly_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
    // Use unchecked variant: model.system is already validated during construction.
    let discrete = cfd2_codegen::solver::codegen::lower_system_unchecked(&model.system, schemes);

    // `generic_coupled_assembly` is the no-grad_state variant.
    // The grad_state-binding variant is emitted under `KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE`.
    let needs_gradients = false;
    let flux_stride = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .map(|_| model.system.unknowns_per_cell())
        .unwrap_or(0);
    let slots = resolved_slots_from_layout(&model.state_layout);
    let eos_params = extract_eos_params(model);
    Ok(
        cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_wgsl(
            &discrete,
            &slots,
            flux_stride,
            needs_gradients,
            &eos_params,
        ),
    )
}

pub(crate) fn generate_generic_coupled_assembly_grad_state_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::GradientStorage;

    let method = model.method().map_err(|e| e.to_string())?;
    let gradient_storage = match method {
        crate::solver::model::method::MethodSpec::Coupled(caps) => caps.gradient_storage,
    };

    if gradient_storage != GradientStorage::PackedState {
        return Err(
            "generic_coupled_assembly_grad_state requires MethodSpec::Coupled(CoupledCapabilities { gradient_storage: PackedState })"
                .to_string(),
        );
    }

    // Use unchecked variant: model.system is already validated during construction.
    let discrete = cfd2_codegen::solver::codegen::lower_system_unchecked(&model.system, schemes);

    let needs_gradients = true;
    let flux_stride = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .map(|_| model.system.unknowns_per_cell())
        .unwrap_or(0);
    let slots = resolved_slots_from_layout(&model.state_layout);
    let eos_params = extract_eos_params(model);
    Ok(
        cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_wgsl(
            &discrete,
            &slots,
            flux_stride,
            needs_gradients,
            &eos_params,
        ),
    )
}

pub(crate) fn generate_packed_state_gradients_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
    let eos_params = extract_eos_params(model);
    cfd2_codegen::solver::codegen::generate_packed_state_gradients_wgsl(
        &model.state_layout,
        model.system.unknowns_per_cell(),
        &eos_params,
    )
    .map_err(|e| e.to_string())
}

/// Resolve a state offset by field name, supporting component suffixes (e.g., "rho_u_x").
/// Uses the ResolvedStateSlotsSpec to find the base offset and add component index.
fn resolve_offset_from_slots(slots: &ResolvedStateSlotsSpec, name: &str) -> Option<u32> {
    // Helper to find a slot by field name
    fn find_slot<'a>(
        slots: &'a ResolvedStateSlotsSpec,
        field: &str,
    ) -> Option<&'a ResolvedStateSlotSpec> {
        slots.slots.iter().find(|s| s.name == field)
    }

    // First, try direct field lookup
    if let Some(slot) = find_slot(slots, name) {
        return Some(slot.base_offset);
    }

    // Try to parse component suffix (_x, _y, _z)
    let (base, component) = name.rsplit_once('_')?;
    let component = match component {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => return None,
    };

    let slot = find_slot(slots, base)?;
    if component >= slot.kind.component_count() {
        return None;
    }
    Some(slot.base_offset + component)
}

pub(crate) fn generate_generic_coupled_update_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
    // Use unchecked variant: model.system is already validated during construction.
    let discrete = cfd2_codegen::solver::codegen::lower_system_unchecked(&model.system, schemes);
    let prims = model
        .primitives
        .ordered()
        .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
    let (apply_relaxation, relaxation_requires_dtau) =
        match model.method().map_err(|e| e.to_string())? {
            crate::solver::model::method::MethodSpec::Coupled(caps) => (
                caps.apply_relaxation_in_update,
                caps.relaxation_requires_dtau,
            ),
        };
    let slots = resolved_slots_from_layout(&model.state_layout);

    // Pre-resolve primitive output offsets (skip primitives that cannot be resolved)
    let resolved_prims: Vec<(u32, cfd2_codegen::solver::shared::PrimitiveExpr)> = prims
        .into_iter()
        .filter_map(|(name, expr)| {
            resolve_offset_from_slots(&slots, &name).map(|offset| (offset, expr))
        })
        .collect();

    let eos_params = extract_eos_params(model);
    Ok(cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_update_wgsl(
        &discrete,
        &slots,
        &resolved_prims,
        apply_relaxation,
        relaxation_requires_dtau,
        &eos_params,
    ))
}

fn kernel_generator_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    kernel_id: KernelId,
) -> Option<&ModelKernelGeneratorSpec> {
    for module in &model.modules {
        let module: &dyn crate::solver::model::module::ModelModule = module;
        if let Some(spec) = module
            .kernel_generators()
            .iter()
            .find(|s| s.id == kernel_id)
        {
            return Some(spec);
        }
    }

    None
}

fn generate_kernel_artifact_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kernel_id: KernelId,
) -> Result<ModelKernelArtifact, String> {
    if let Some(generator) = kernel_generator_for_model_by_id(model, kernel_id) {
        return generator.generator.as_ref()(model, schemes);
    }

    Err(format!(
        "KernelId '{}' is not a build-time generated per-model kernel",
        kernel_id.as_str()
    ))
}

pub fn generate_kernel_wgsl_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kernel_id: KernelId,
) -> Result<String, String> {
    let artifact = generate_kernel_artifact_for_model_by_id(model, schemes, kernel_id)?;
    let kernel = artifact.into_wgsl()?;
    Ok(kernel.to_wgsl())
}

pub fn emit_shared_kernels_wgsl(
    base_dir: impl AsRef<std::path::Path>,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    Ok(emit_shared_kernels_wgsl_with_ids(base_dir)?
        .into_iter()
        .map(|(_, path)| path)
        .collect())
}

pub fn emit_shared_kernels_wgsl_with_ids(
    base_dir: impl AsRef<std::path::Path>,
) -> std::io::Result<Vec<(KernelId, std::path::PathBuf)>> {
    emit_shared_kernels_wgsl_with_ids_for_models(
        base_dir,
        &crate::solver::model::all_models(),
        &crate::solver::ir::SchemeRegistry::default(),
    )
}

pub fn emit_shared_kernels_wgsl_with_ids_for_models(
    base_dir: impl AsRef<std::path::Path>,
    models: &[crate::solver::model::ModelSpec],
    schemes: &crate::solver::ir::SchemeRegistry,
) -> std::io::Result<Vec<(KernelId, std::path::PathBuf)>> {
    let base_dir = base_dir.as_ref();
    let mut outputs = Vec::new();

    let mut wgsl_by_id: std::collections::HashMap<KernelId, String> =
        std::collections::HashMap::new();
    for model in models {
        for module in &model.modules {
            let module: &dyn crate::solver::model::module::ModelModule = module;
            for spec in module.kernel_generators() {
                if spec.scope != KernelWgslScope::Shared {
                    continue;
                }

                let artifact =
                    spec.generator.as_ref()(model, schemes).map_err(std::io::Error::other)?;
                let kernel = artifact.into_wgsl().map_err(std::io::Error::other)?;
                let wgsl = kernel.to_wgsl();

                if let Some(prev) = wgsl_by_id.insert(spec.id, wgsl.clone()) {
                    if prev != wgsl {
                        return Err(std::io::Error::other(format!(
                            "shared kernel '{}' generated different WGSL across models (latest from model '{}' module '{}')",
                            spec.id.as_str(),
                            model.id,
                            module.name()
                        )));
                    }
                }
            }
        }
    }

    let mut shared_ids: Vec<KernelId> = wgsl_by_id.keys().copied().collect();
    shared_ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));

    for kernel_id in shared_ids {
        let filename =
            kernel_output_name_for_model("", kernel_id).map_err(std::io::Error::other)?;
        let wgsl = wgsl_by_id
            .get(&kernel_id)
            .expect("shared kernel id disappeared from table");
        let path = cfd2_codegen::compiler::write_generated_wgsl(base_dir, filename, wgsl)?;
        outputs.push((kernel_id, path));
    }

    Ok(outputs)
}

pub fn emit_model_kernels_wgsl(
    base_dir: impl AsRef<std::path::Path>,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    Ok(emit_model_kernels_wgsl_with_ids(base_dir, model, schemes)?
        .into_iter()
        .map(|(_, path)| path)
        .collect())
}

fn generate_kernel_program_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kernel_id: KernelId,
) -> Result<crate::solver::ir::KernelProgram, String> {
    match generate_kernel_artifact_for_model_by_id(model, schemes, kernel_id)? {
        ModelKernelArtifact::DslProgram(program) => Ok(program),
        ModelKernelArtifact::Wgsl(_) => Err(format!(
            "kernel '{}' is WGSL-only and cannot be used as DSL fusion input",
            kernel_id.as_str()
        )),
    }
}

fn synthesize_fusion_replacement_wgsl_for_model(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    replacement_id: KernelId,
) -> Result<Option<String>, String> {
    let mut matching_rules: Vec<ModelKernelFusionRule> = derive_kernel_fusion_rules_for_model(model)
        .into_iter()
        .filter(|rule| rule.replacement.id == replacement_id)
        .collect();
    if matching_rules.is_empty() {
        return Ok(None);
    }
    matching_rules.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then(b.pattern.len().cmp(&a.pattern.len()))
            .then(a.name.cmp(b.name))
    });

    let selected = &matching_rules[0];
    let mut programs = Vec::with_capacity(selected.pattern.len());
    for atom in &selected.pattern {
        programs.push(generate_kernel_program_for_model_by_id(
            model, schemes, atom.id,
        )?);
    }

    let fused_program = cfd2_codegen::solver::codegen::fusion::synthesize_fused_program(
        replacement_id.as_str().to_string(),
        selected.name,
        &programs,
        cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy::Safe,
    )?;
    let wgsl =
        cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl(&fused_program)?;
    Ok(Some(wgsl.to_wgsl()))
}

pub fn emit_model_kernels_wgsl_with_ids(
    base_dir: impl AsRef<std::path::Path>,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> std::io::Result<Vec<(KernelId, std::path::PathBuf)>> {
    let mut outputs = Vec::new();

    let specs = derive_kernel_specs_for_model(model).map_err(std::io::Error::other)?;
    let replacement_ids = derive_fusion_replacement_kernel_ids_for_model(model);
    let replacement_id_set: std::collections::HashSet<KernelId> =
        replacement_ids.iter().copied().collect();

    let mut seen: std::collections::HashSet<KernelId> = std::collections::HashSet::new();
    let mut kernel_ids = Vec::with_capacity(specs.len() + replacement_ids.len());
    kernel_ids.extend(specs.into_iter().map(|s| s.id));
    kernel_ids.extend(replacement_ids);

    for kernel_id in kernel_ids {
        if !seen.insert(kernel_id) {
            continue;
        }

        let Some(generator) = kernel_generator_for_model_by_id(model, kernel_id) else {
            if replacement_id_set.contains(&kernel_id) {
                let Some(wgsl) =
                    synthesize_fusion_replacement_wgsl_for_model(model, schemes, kernel_id)
                        .map_err(std::io::Error::other)?
                else {
                    return Err(std::io::Error::other(format!(
                        "replacement kernel '{}' has no generator and no synthesizeable fusion rule",
                        kernel_id.as_str()
                    )));
                };
                let filename =
                    kernel_output_name_for_model(model.id, kernel_id).map_err(std::io::Error::other)?;
                let path = cfd2_codegen::compiler::write_generated_wgsl(base_dir.as_ref(), filename, &wgsl)?;
                outputs.push((kernel_id, path));
            }
            continue;
        };
        if generator.scope == KernelWgslScope::Shared {
            continue;
        }

        let path = emit_model_kernel_wgsl_by_id(&base_dir, model, schemes, kernel_id)?;
        outputs.push((kernel_id, path));
    }

    Ok(outputs)
}

pub fn emit_model_kernel_wgsl_by_id(
    base_dir: impl AsRef<std::path::Path>,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kernel_id: KernelId,
) -> std::io::Result<std::path::PathBuf> {
    let base_dir = base_dir.as_ref();
    let filename =
        kernel_output_name_for_model(model.id, kernel_id).map_err(std::io::Error::other)?;
    let wgsl = generate_kernel_wgsl_for_model_by_id(model, schemes, kernel_id)
        .map_err(std::io::Error::other)?;
    cfd2_codegen::compiler::write_generated_wgsl(base_dir, filename, &wgsl)
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use crate::solver::model::module::KernelBundleModule;
    use crate::solver::model::ModelSpec;
    use cfd2_ir::solver::ir::{
        BindingAccess, DispatchDomain, KernelBinding, KernelProgram, LaunchSemantics,
    };

    fn contract_kernel_generator(
        _model: &ModelSpec,
        _schemes: &crate::solver::ir::SchemeRegistry,
    ) -> Result<KernelWgsl, String> {
        let mut module = cfd2_codegen::solver::codegen::wgsl_ast::Module::new();
        module.push(cfd2_codegen::solver::codegen::wgsl_ast::Item::Comment(
            "contract: module-defined kernel generator".to_string(),
        ));
        Ok(KernelWgsl::from(module))
    }

    fn contract_dsl_kernel_generator(
        kernel_id: &'static str,
        body_stmt: &'static str,
    ) -> impl Fn(
        &ModelSpec,
        &crate::solver::ir::SchemeRegistry,
    ) -> Result<KernelProgram, String> + Send
           + Sync
           + 'static {
        move |_model, _schemes| {
            let launch = LaunchSemantics::new(
                [64, 1, 1],
                "global_id.y * constants.stride_x + global_id.x",
                Some("idx >= arrayLength(&state)"),
            );
            let mut program = KernelProgram::new(
                kernel_id,
                DispatchDomain::Cells,
                launch,
                vec![
                    KernelBinding::new(
                        0,
                        0,
                        "state",
                        "array<f32>",
                        BindingAccess::ReadWriteStorage,
                    ),
                    KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
                ],
            );
            program.indexing = vec!["let inv = idx;".to_string()];
            program.preamble = vec!["var value: f32 = state[idx];".to_string()];
            program.body = vec![
                body_stmt.to_string(),
                "state[idx] = value;".to_string(),
                "state[inv] = state[idx];".to_string(),
            ];
            program.local_symbols = vec!["value".to_string(), "inv".to_string()];
            Ok(program)
        }
    }

    #[test]
    fn contract_gap0_module_defined_kernel_id_is_module_driven() {
        let contract_id = KernelId("contract/module_defined_kernel");

        let module = KernelBundleModule {
            name: "contract_module_defined_kernel",
            kernels: vec![ModelKernelSpec {
                id: contract_id,
                phase: KernelPhaseId::Preparation,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            }],
            generators: vec![ModelKernelGeneratorSpec::new(
                contract_id,
                contract_kernel_generator,
            )],
            ..Default::default()
        };

        let mut model = crate::solver::model::generic_diffusion_demo_model();
        model.modules.push(module);

        let specs = derive_kernel_specs_for_model(&model).expect("failed to derive kernel specs");
        assert!(
            specs.iter().any(|s| s.id == contract_id),
            "derived kernel specs must include module-defined KernelId"
        );

        let schemes = crate::solver::ir::SchemeRegistry::default();
        let wgsl = generate_kernel_wgsl_for_model_by_id(&model, &schemes, contract_id)
            .expect("module-defined kernel generator was not located by id");
        assert!(
            wgsl.contains("contract: module-defined kernel generator"),
            "generated WGSL must contain the module-defined marker"
        );
    }

    #[test]
    fn contract_fusion_replacement_kernels_are_emitted_for_codegen() {
        let base_id = KernelId("contract/fusion_base");
        let fused_id = KernelId("contract/fusion_fused");

        let module = KernelBundleModule {
            name: "contract_fusion_module",
            kernels: vec![ModelKernelSpec {
                id: base_id,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            }],
            generators: vec![
                ModelKernelGeneratorSpec::new(base_id, contract_kernel_generator),
                ModelKernelGeneratorSpec::new(fused_id, contract_kernel_generator),
            ],
            fusion_rules: vec![ModelKernelFusionRule {
                name: "contract:fuse_base",
                priority: 1,
                phase: KernelPhaseId::Update,
                pattern: vec![KernelPatternAtom::id(base_id)],
                replacement: ModelKernelSpec {
                    id: fused_id,
                    phase: KernelPhaseId::Update,
                    dispatch: DispatchKindId::Cells,
                    condition: KernelConditionId::Always,
                },
                guards: Vec::new(),
            }],
            ..Default::default()
        };

        let mut model = crate::solver::model::generic_diffusion_demo_model();
        model.modules.push(module);

        let replacement_ids = derive_fusion_replacement_kernel_ids_for_model(&model);
        assert!(replacement_ids.contains(&fused_id));

        let schemes = crate::solver::ir::SchemeRegistry::default();
        let mut out_dir = std::env::temp_dir();
        out_dir.push(format!(
            "cfd2_kernel_emit_contract_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&out_dir).expect("create temp output dir");

        let emitted = emit_model_kernels_wgsl_with_ids(&out_dir, &model, &schemes)
            .expect("emit kernels with fusion replacements");
        assert!(emitted.iter().any(|(id, _)| *id == fused_id));

        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn contract_mixed_mode_kernel_generators_are_supported() {
        let wgsl_id = KernelId("contract/mixed_wgsl");
        let dsl_id = KernelId("contract/mixed_dsl");

        let module = KernelBundleModule {
            name: "contract_mixed_mode_module",
            kernels: vec![
                ModelKernelSpec {
                    id: wgsl_id,
                    phase: KernelPhaseId::Preparation,
                    dispatch: DispatchKindId::Cells,
                    condition: KernelConditionId::Always,
                },
                ModelKernelSpec {
                    id: dsl_id,
                    phase: KernelPhaseId::Preparation,
                    dispatch: DispatchKindId::Cells,
                    condition: KernelConditionId::Always,
                },
            ],
            generators: vec![
                ModelKernelGeneratorSpec::new(wgsl_id, contract_kernel_generator),
                ModelKernelGeneratorSpec::new_dsl(
                    dsl_id,
                    contract_dsl_kernel_generator(dsl_id.as_str(), "value = value + 1.0;"),
                ),
            ],
            ..Default::default()
        };

        let mut model = crate::solver::model::generic_diffusion_demo_model();
        model.modules.push(module);
        let schemes = crate::solver::ir::SchemeRegistry::default();

        let wgsl = generate_kernel_wgsl_for_model_by_id(&model, &schemes, wgsl_id)
            .expect("wgsl generator should resolve");
        assert!(
            wgsl.contains("contract: module-defined kernel generator"),
            "expected WGSL artifact content"
        );

        let lowered = generate_kernel_wgsl_for_model_by_id(&model, &schemes, dsl_id)
            .expect("dsl generator should lower to WGSL");
        assert!(
            lowered.contains("GENERATED BY CFD2 DSL FUSION"),
            "expected DSL lowering marker"
        );

        let mut out_dir = std::env::temp_dir();
        out_dir.push(format!(
            "cfd2_kernel_emit_mixed_mode_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&out_dir).expect("create temp output dir");
        let emitted = emit_model_kernels_wgsl_with_ids(&out_dir, &model, &schemes)
            .expect("emit mixed-mode kernels");
        assert!(emitted.iter().any(|(id, _)| *id == wgsl_id));
        assert!(emitted.iter().any(|(id, _)| *id == dsl_id));
        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn contract_fusion_replacement_is_synthesized_from_dsl_inputs() {
        let a_id = KernelId("contract/dsl_a");
        let b_id = KernelId("contract/dsl_b");
        let fused_id = KernelId("contract/dsl_fused");

        let module = KernelBundleModule {
            name: "contract_dsl_fusion_module",
            kernels: vec![
                ModelKernelSpec {
                    id: a_id,
                    phase: KernelPhaseId::Update,
                    dispatch: DispatchKindId::Cells,
                    condition: KernelConditionId::Always,
                },
                ModelKernelSpec {
                    id: b_id,
                    phase: KernelPhaseId::Update,
                    dispatch: DispatchKindId::Cells,
                    condition: KernelConditionId::Always,
                },
            ],
            generators: vec![
                ModelKernelGeneratorSpec::new_dsl(
                    a_id,
                    contract_dsl_kernel_generator(a_id.as_str(), "value = value + 1.0;"),
                ),
                ModelKernelGeneratorSpec::new_dsl(
                    b_id,
                    contract_dsl_kernel_generator(b_id.as_str(), "value = value + 2.0;"),
                ),
            ],
            fusion_rules: vec![ModelKernelFusionRule {
                name: "contract:dsl_fuse_ab",
                priority: 50,
                phase: KernelPhaseId::Update,
                pattern: vec![KernelPatternAtom::id(a_id), KernelPatternAtom::id(b_id)],
                replacement: ModelKernelSpec {
                    id: fused_id,
                    phase: KernelPhaseId::Update,
                    dispatch: DispatchKindId::Cells,
                    condition: KernelConditionId::Always,
                },
                guards: Vec::new(),
            }],
            ..Default::default()
        };

        let mut model = crate::solver::model::generic_diffusion_demo_model();
        model.modules.push(module);
        let schemes = crate::solver::ir::SchemeRegistry::default();

        let mut out_dir = std::env::temp_dir();
        out_dir.push(format!(
            "cfd2_kernel_emit_dsl_fuse_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&out_dir).expect("create temp output dir");

        let emitted = emit_model_kernels_wgsl_with_ids(&out_dir, &model, &schemes)
            .expect("emit kernels with synthesized fusion replacement");
        assert!(
            emitted.iter().any(|(id, _)| *id == fused_id),
            "expected synthesized fused kernel output"
        );

        let fused_path = emitted
            .iter()
            .find_map(|(id, path)| if *id == fused_id { Some(path) } else { None })
            .expect("fused kernel path");
        let src = std::fs::read_to_string(fused_path).expect("read fused wgsl");
        assert!(
            src.contains("synthesized by fusion rule"),
            "missing synthesis marker in fused output"
        );
        assert!(
            src.contains("k1_value"),
            "expected deterministic local-symbol rename for second kernel"
        );

        let _ = std::fs::remove_dir_all(&out_dir);
    }
}

// (intentionally no additional kernel-analysis helpers here; kernel selection is recipe-driven)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::scheme::Scheme;
    use std::collections::HashSet;

    #[test]
    fn fusion_pass_applies_highest_priority_longest_match() {
        let a = ModelKernelSpec {
            id: KernelId("fusion/a"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };
        let b = ModelKernelSpec {
            id: KernelId("fusion/b"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };
        let c = ModelKernelSpec {
            id: KernelId("fusion/c"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };
        let fused_ab = ModelKernelSpec {
            id: KernelId("fusion/ab"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };
        let fused_abc = ModelKernelSpec {
            id: KernelId("fusion/abc"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };

        let rules = vec![
            ModelKernelFusionRule {
                name: "fuse_ab",
                priority: 10,
                phase: KernelPhaseId::Update,
                pattern: vec![KernelPatternAtom::id(a.id), KernelPatternAtom::id(b.id)],
                replacement: fused_ab,
                guards: Vec::new(),
            },
            ModelKernelFusionRule {
                name: "fuse_abc",
                priority: 10,
                phase: KernelPhaseId::Update,
                pattern: vec![
                    KernelPatternAtom::id(a.id),
                    KernelPatternAtom::id(b.id),
                    KernelPatternAtom::id(c.id),
                ],
                replacement: fused_abc,
                guards: Vec::new(),
            },
        ];
        let kernels = vec![a, b, c];
        let module_names = ["fusion_module"];
        let ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Safe,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            module_names: &module_names,
        };

        let out = apply_model_fusion_rules(&kernels, &rules, &ctx);
        assert_eq!(out.kernels.len(), 1);
        assert_eq!(out.kernels[0].id, fused_abc.id);
        assert_eq!(out.applied.len(), 1);
        assert_eq!(out.applied[0].name, "fuse_abc");
    }

    #[test]
    fn fusion_guard_min_policy_gates_rules() {
        let base = ModelKernelSpec {
            id: KernelId("fusion/base"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };
        let fused = ModelKernelSpec {
            id: KernelId("fusion/fused"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };
        let rules = vec![ModelKernelFusionRule {
            name: "fuse_when_aggressive",
            priority: 1,
            phase: KernelPhaseId::Update,
            pattern: vec![KernelPatternAtom::id(base.id)],
            replacement: fused,
            guards: vec![FusionGuard::MinPolicy(KernelFusionPolicy::Aggressive)],
        }];
        let kernels = vec![base];
        let module_names = ["fusion_module"];

        let safe_ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Safe,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            module_names: &module_names,
        };
        let safe_out = apply_model_fusion_rules(&kernels, &rules, &safe_ctx);
        assert_eq!(safe_out.kernels.len(), 1);
        assert_eq!(safe_out.kernels[0].id, base.id);
        assert!(safe_out.applied.is_empty());

        let aggressive_ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Aggressive,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            module_names: &module_names,
        };
        let aggressive_out = apply_model_fusion_rules(&kernels, &rules, &aggressive_ctx);
        assert_eq!(aggressive_out.kernels.len(), 1);
        assert_eq!(aggressive_out.kernels[0].id, fused.id);
        assert_eq!(aggressive_out.applied.len(), 1);
    }

    #[test]
    fn contract_flux_module_uses_runtime_scheme_and_includes_limited_paths() {
        // Contract: the flux-module kernel must not ignore the runtime `advection_scheme` knob.
        // It should branch on `constants.scheme` and include the limited reconstruction paths
        // in WGSL (even though shipped defaults remain Upwind).
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);

        let model = crate::solver::model::compressible_model();
        let wgsl = generate_kernel_wgsl_for_model_by_id(&model, &schemes, KernelId::FLUX_MODULE)
            .expect("failed to generate flux_module WGSL");

        assert!(
            wgsl.contains("constants.scheme"),
            "flux_module WGSL should reference constants.scheme for runtime selection"
        );

        // 1) Ensure runtime branches exist for each limited variant.
        let ids = scheme_ids_from_wgsl(&wgsl);
        for scheme in [
            Scheme::SecondOrderUpwindMinMod,
            Scheme::SecondOrderUpwindVanLeer,
            Scheme::QUICKMinMod,
            Scheme::QUICKVanLeer,
        ] {
            assert!(
                ids.contains(&scheme.gpu_id()),
                "WGSL should branch on constants.scheme == {}u for {scheme:?}",
                scheme.gpu_id(),
            );
        }

        // 2) Ensure limiter implementations are present.
        //    - VanLeer: has a small epsilon literal.
        assert!(
            wgsl.contains("1e-8") || wgsl.contains("0.00000001"),
            "expected VanLeer epsilon literal to appear in flux_module WGSL"
        );
    }

    #[test]
    fn contract_unified_assembly_uses_runtime_scheme_and_includes_limited_paths() {
        // Contract: the generic-coupled assembly kernel should not hard-code advection scheme
        // selection at codegen time. Instead, it should branch on `constants.scheme` and include
        // the limited reconstruction paths in WGSL (even though shipped defaults remain Upwind).
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);

        let model = crate::solver::model::incompressible_momentum_model();
        let wgsl = generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId::GENERIC_COUPLED_ASSEMBLY,
        )
        .expect("failed to generate generic_coupled_assembly WGSL");

        assert!(
            wgsl.contains("constants.scheme"),
            "generic_coupled_assembly WGSL should reference constants.scheme for runtime selection"
        );

        // VanLeer-limited paths include a small epsilon literal; check it is present so the
        // limiter is not silently ignored/hard-coded away.
        assert!(
            wgsl.contains("1e-8") || wgsl.contains("0.00000001"),
            "expected VanLeer epsilon literal to appear in generic_coupled_assembly WGSL"
        );
    }

    fn wgsl_compact(wgsl: &str) -> String {
        wgsl.chars().filter(|c| !c.is_whitespace()).collect()
    }

    fn scheme_ids_from_wgsl(wgsl: &str) -> HashSet<u32> {
        let compact = wgsl_compact(wgsl);
        let mut ids = HashSet::new();

        // Match on the stable runtime selection pattern emitted by codegen.
        let needle = "constants.scheme==";
        let mut start = 0;
        while let Some(idx) = compact[start..].find(needle) {
            let after = &compact[start + idx + needle.len()..];
            let end = after.find('u').unwrap_or(after.len());
            if let Ok(value) = after[..end].parse::<u32>() {
                ids.insert(value);
            }
            start += idx + needle.len();
        }

        ids
    }

    #[test]
    fn contract_unified_assembly_limited_scheme_variants_are_wired() {
        // Regression test: ensure the new limited advection scheme variants remain wired
        // and cannot silently degrade back to unlimited reconstruction.
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);

        let model = crate::solver::model::incompressible_momentum_model();
        let wgsl = generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId::GENERIC_COUPLED_ASSEMBLY,
        )
        .expect("failed to generate generic_coupled_assembly WGSL");

        // 1) Ensure runtime branches exist for each limited variant.
        let ids = scheme_ids_from_wgsl(&wgsl);
        for scheme in [
            Scheme::SecondOrderUpwindMinMod,
            Scheme::SecondOrderUpwindVanLeer,
            Scheme::QUICKMinMod,
            Scheme::QUICKVanLeer,
        ] {
            assert!(
                ids.contains(&scheme.gpu_id()),
                "WGSL should branch on constants.scheme == {}u for {scheme:?}",
                scheme.gpu_id(),
            );
        }

        // 2) Ensure both limiter implementations are present.
        //    - MinMod: has a nested min(max(...)) clamp.
        //    - VanLeer: has a small epsilon literal.
        let compact = wgsl_compact(&wgsl);
        assert!(
            compact.contains("min(max("),
            "expected MinMod limiter clamp (min(max(...))) to appear in WGSL"
        );
        assert!(
            wgsl.contains("1e-8") || wgsl.contains("0.00000001"),
            "expected VanLeer epsilon literal to appear in WGSL"
        );
    }

    #[test]
    fn contract_unified_assembly_variants_have_distinct_grad_state_bindings() {
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);
        let model = crate::solver::model::incompressible_momentum_model();

        let no_grad = generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId::GENERIC_COUPLED_ASSEMBLY,
        )
        .expect("failed to generate generic_coupled_assembly WGSL");
        assert!(
            !no_grad.contains("grad_state"),
            "generic_coupled_assembly must not bind grad_state"
        );

        let with_grad = generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
        )
        .expect("failed to generate generic_coupled_assembly_grad_state WGSL");
        assert!(
            with_grad.contains("grad_state"),
            "generic_coupled_assembly_grad_state must bind grad_state"
        );
    }
}
