use std::sync::Arc;

use cfd2_codegen::solver::codegen::KernelWgsl;
use cfd2_ir::ports::{
    ParamSpec, ResolvedStateSlotSpec, ResolvedStateSlotsSpec,
};
use cfd2_ir::kernel::StateLayout;

/// Stable identifier for a compute kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub &'static str);

impl KernelId {
    pub const FLUX_MODULE_GRADIENTS: KernelId = KernelId("flux_module_gradients");
    pub const FLUX_MODULE: KernelId = KernelId("flux_module");

    /// Rhie-Chow post-solve pressure-gradient refresh.
    /// Schedule logic keys on this (see [`Self::refreshes_grad_p`]): it writes
    /// the SAME state grad_p slots with the SAME Green-Gauss stencil as
    /// [`Self::FLUX_MODULE_GRADIENTS`], which both backends therefore skip on
    /// outer iterations after the first.
    pub const RHIE_CHOW_GRAD_P_UPDATE: KernelId = KernelId("rhie_chow/grad_p_update");
    pub const RHIE_CHOW_GRAD_P_UPDATE_CORRECT_VELOCITY_DELTA_FUSED: KernelId =
        KernelId("rhie_chow/grad_p_update_correct_velocity_delta_fused");
    pub const RHIE_CHOW_STORE_GRAD_P_GRAD_P_UPDATE_FUSED: KernelId =
        KernelId("rhie_chow/store_grad_p_grad_p_update_fused");
    #[allow(clippy::doc_markdown)]
    pub const RHIE_CHOW_DP_INIT_DP_UPDATE_STORE_GRAD_P_GRAD_P_UPDATE_CORRECT_VELOCITY_DELTA_FUSED:
        KernelId =
        KernelId("rhie_chow/dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused");

    /// True when this kernel performs the Rhie-Chow post-solve pressure-gradient
    /// refresh — standalone or as any of its fused variants. Both backends key
    /// the "skip [`Self::FLUX_MODULE_GRADIENTS`] on outer iterations after the
    /// first" schedule optimization on the update group containing one of these.
    pub fn refreshes_grad_p(self) -> bool {
        Self::id_refreshes_grad_p(self.0)
    }

    /// [`Self::refreshes_grad_p`] for contexts that hold the kernel id as a
    /// plain string (the CPU schedule).
    pub fn id_refreshes_grad_p(id: &str) -> bool {
        id == Self::RHIE_CHOW_GRAD_P_UPDATE.0
            || id == Self::RHIE_CHOW_GRAD_P_UPDATE_CORRECT_VELOCITY_DELTA_FUSED.0
            || id == Self::RHIE_CHOW_STORE_GRAD_P_GRAD_P_UPDATE_FUSED.0
            || id == Self::RHIE_CHOW_DP_INIT_DP_UPDATE_STORE_GRAD_P_GRAD_P_UPDATE_CORRECT_VELOCITY_DELTA_FUSED.0
    }

    /// Generic refresh of expression-valued boundary-table entries (`BcValue::Expr`).
    pub const BC_EXPR_UPDATE: KernelId = KernelId("bc_expr_update");
    pub const COMPRESSIBLE_VISCOUS_P_DIV_U: KernelId = KernelId("compressible/viscous_p_div_u");

    pub const GENERIC_COUPLED_ASSEMBLY: KernelId = KernelId("generic_coupled_assembly");
    pub const GENERIC_COUPLED_ASSEMBLY_GRAD_STATE: KernelId =
        KernelId("generic_coupled_assembly_grad_state");
    /// RHS-only variants of the two assembly kernels (matrix writes stripped)
    /// for outer iterations that FREEZE the assembled matrix; scheduled only
    /// when matrix freezing is active (default off).
    pub const GENERIC_COUPLED_ASSEMBLY_RHS_ONLY: KernelId =
        KernelId("generic_coupled_assembly_rhs_only");
    pub const GENERIC_COUPLED_ASSEMBLY_GRAD_STATE_RHS_ONLY: KernelId =
        KernelId("generic_coupled_assembly_grad_state_rhs_only");
    pub const GENERIC_COUPLED_APPLY: KernelId = KernelId("generic_coupled_apply");
    pub const GENERIC_COUPLED_UPDATE: KernelId = KernelId("generic_coupled_update");

    // Handwritten solver-infrastructure kernels (single-entrypoint compute shaders).
    pub const DOT_PRODUCT: KernelId = KernelId("dot_product");
    pub const DOT_PRODUCT_PAIR: KernelId = KernelId("dot_product_pair");
    pub const OUTER_CONVERGENCE: KernelId = KernelId("outer_convergence");
    pub const OUTER_CONVERGENCE_BREAK: KernelId = KernelId("outer_convergence_break");
    pub const OUTER_GATE: KernelId = KernelId("outer_gate");
    pub const OUTER_STOP_INJECT_FGMRES: KernelId = KernelId("outer_stop_inject_fgmres");
    pub const OUTER_STOP_INJECT_CG: KernelId = KernelId("outer_stop_inject_cg");

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

    pub const BLOCK_PRECOND_BUILD_BLOCK_INV: KernelId = KernelId("block_precond/build_block_inv");
    pub const BLOCK_PRECOND_APPLY_BLOCK_PRECOND: KernelId =
        KernelId("block_precond/apply_block_precond");

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

/// Kernel filenames are derived from `KernelId` by replacing `/` with `_`.
/// Model-owned kernel phase classification (GPU-agnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelPhaseId {
    Preparation,
    Gradients,
    FluxComputation,
    Assembly,
    /// RHS-only re-assembly for outer iterations with a FROZEN matrix. Never
    /// part of the normal per-iteration graphs; scheduled only by the
    /// matrix-freeze paths (default off).
    AssemblyRhsOnly,
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
    /// Disable fusion entirely and keep the original ordered kernel list.
    Off,
    /// Allow only strict hazard-checked fusion (default production policy).
    /// This requires matching dispatch/indexing/launch contracts and rejects
    /// programs that use barriers/atomics.
    Safe,
    /// Enable broader fusion behavior for experimentation/perf tuning.
    /// This includes safe matching plus aggressive-only cleanup transforms
    /// during synthesized DSL fusion.
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
    ExactPolicy(KernelFusionPolicy),
    /// Reject when the model declares a term whose ASSEMBLY reads NEIGHBOR
    /// cells' grad_state with numerical effect (transpose_dev2 viscous
    /// terms). Fusing the gradients kernel into the assembly dispatch then
    /// makes neighbor gradient reads racy (fresh-or-stale within the same
    /// dispatch), breaking the Safe policy's bit-identical contract.
    /// (SOU reconstruction's neighbor gradient reads are tolerated as a
    /// lagged correction; dev2 terms make gradient values first-class
    /// operands, so the fusion must not apply.)
    RequiresNoNeighborGradConsumers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelPatternAtom {
    pub id: KernelId,
    pub dispatch: Option<DispatchKindId>,
    /// Override the expected phase for this atom.  When `None` the atom
    /// inherits the rule-level `phase`.
    pub phase: Option<KernelPhaseId>,
}

impl KernelPatternAtom {
    pub const fn id(id: KernelId) -> Self {
        Self {
            id,
            dispatch: None,
            phase: None,
        }
    }

    pub const fn with_dispatch(id: KernelId, dispatch: DispatchKindId) -> Self {
        Self {
            id,
            dispatch: Some(dispatch),
            phase: None,
        }
    }

    pub const fn with_phase(id: KernelId, dispatch: DispatchKindId, phase: KernelPhaseId) -> Self {
        Self {
            id,
            dispatch: Some(dispatch),
            phase: Some(phase),
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
    /// Binding slot remaps applied before fusion merging.
    ///
    /// When the kernels being fused have the same logical buffer at different
    /// `(group, binding)` slots (e.g. boundary conditions at group 2 in one
    /// kernel vs group 3 in another), each entry instructs the synthesizer to
    /// relocate a specific program's binding slot to the target layout.
    pub binding_remaps: Vec<cfd2_codegen::solver::codegen::fusion::BindingRemap>,
    /// Hazard whitelist for aggressive fusion rules.
    ///
    /// Each entry is a `(HazardKind, kernel_id)` pair that the rule author has
    /// audited and confirmed is safe (e.g. because the kernels operate on
    /// disjoint index ranges, or the dependency is a false positive from
    /// conservative side-effect tracking).
    ///
    /// Under `Safe` policy this field is ignored (all hazards reject).
    /// Under `Aggressive` policy, only whitelisted hazards are tolerated;
    /// any non-whitelisted hazard still causes a hard rejection.
    pub expected_hazards: Vec<cfd2_codegen::solver::codegen::fusion::ExpectedHazard>,
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
    /// Model declares transpose_dev2 terms (assembly consumes neighbor
    /// grad_state values with numerical effect; see
    /// `FusionGuard::RequiresNoNeighborGradConsumers`).
    pub has_neighbor_grad_consumers: bool,
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
    // Structured (`TopologyMode::Structured2D`) models run their kernels
    // unfused: the fused variants (e.g. packed_state_gradients + grad-state
    // assembly) mix a structured assembly with an as-yet-unstructured gradients
    // kernel, whose launch bounds differ, and the gradient pipeline is unused by
    // the structured pure-diffusion slice anyway (`RequiresNoGradState`).
    // Keeping fusion off here is byte-identical for every existing (unstructured)
    // model and avoids synthesizing a launch-incompatible fused kernel.
    if model.system.topology() == cfd2_ir::equation::TopologyMode::Structured2D {
        return Vec::new();
    }
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
        FusionGuard::ExactPolicy(policy) => ctx.policy == policy,
        FusionGuard::RequiresNoNeighborGradConsumers => !ctx.has_neighbor_grad_consumers,
    }
}

/// True when the model's system declares any transpose_dev2 term (the
/// assembly then consumes neighbor grad_state values with numerical effect).
pub fn model_has_neighbor_grad_consumers(model: &crate::solver::model::ModelSpec) -> bool {
    model
        .system
        .equations()
        .iter()
        .any(|eq| {
            eq.terms()
                .iter()
                .any(|t| t.transpose_dev2 || t.viscous_dissipation)
        })
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
        let expected_phase = atom.phase.unwrap_or(rule.phase);
        if spec.phase != expected_phase {
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

/// Build a [`ResolvedStateSlotsSpec`] from a [`StateLayout`] for use in codegen.
fn resolved_slots_from_layout(layout: &StateLayout) -> ResolvedStateSlotsSpec {
    let registry = crate::solver::model::ports::PortRegistry::new(layout.clone());
    registry.to_resolved_state_slots()
}

/// Extract Constants-struct extra params for WGSL generation.
///
/// The canonical EOS block always comes first (the model's "eos" module port
/// manifest if present, otherwise the canonical EOS param list) so shared
/// kernels generate identical WGSL across all models. Additional module
/// manifest params (e.g. the buoyant model's runtime params) are appended
/// AFTER the EOS block, skipping specs that alias base constants fields —
/// some modules declare those purely for named-parameter routing.
///
/// LAYOUT CONTRACT: the host `GpuConstants` POD is shared across models and
/// written wholesale; appended params must mirror its tail field order.
pub(crate) fn extract_eos_params(model: &crate::solver::model::ModelSpec) -> Vec<ParamSpec> {
    let mut params = model
        .modules
        .iter()
        .find(|m| m.name == "eos")
        .and_then(|m| m.port_manifest.as_ref())
        .map(|p| p.params.clone())
        .unwrap_or_else(|| {
            crate::solver::model::modules::eos_ports::eos_uniform_port_manifest().params
        });

    let base_names = cfd2_codegen::solver::codegen::constants::base_constant_field_names();
    for module in &model.modules {
        if module.name == "eos" {
            continue;
        }
        let Some(manifest) = &module.port_manifest else {
            continue;
        };
        for spec in &manifest.params {
            if base_names.contains(&spec.wgsl_field) {
                continue;
            }
            if params.iter().any(|p| p.wgsl_field == spec.wgsl_field) {
                continue;
            }
            params.push(spec.clone());
        }
    }
    params
}

fn generate_generic_coupled_assembly_kernel_program_impl(
    kernel_id: &str,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    needs_gradients: bool,
) -> Result<crate::solver::ir::KernelProgram, String> {
    // Use unchecked variant: model.system is already validated during construction.
    let discrete = cfd2_codegen::solver::codegen::lower_system_unchecked(&model.system, schemes);

    let flux_stride = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .map(|_| model.system.unknowns_per_cell())
        .unwrap_or(0);
    let slots = resolved_slots_from_layout(&model.state_layout);
    let eos_params = extract_eos_params(model);
    cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_kernel_program(
        kernel_id,
        &discrete,
        &slots,
        flux_stride,
        needs_gradients,
        &eos_params,
    )
}

pub(crate) fn generate_generic_coupled_assembly_kernel_program(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<crate::solver::ir::KernelProgram, String> {
    generate_generic_coupled_assembly_kernel_program_impl(
        KernelId::GENERIC_COUPLED_ASSEMBLY.as_str(),
        model,
        schemes,
        false,
    )
}

pub(crate) fn generate_generic_coupled_assembly_grad_state_kernel_program(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<crate::solver::ir::KernelProgram, String> {
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

    generate_generic_coupled_assembly_kernel_program_impl(
        KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE.as_str(),
        model,
        schemes,
        true,
    )
}

pub(crate) fn generate_generic_coupled_assembly_rhs_only_kernel_program(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<crate::solver::ir::KernelProgram, String> {
    let full = generate_generic_coupled_assembly_kernel_program(model, schemes)?;
    Ok(cfd2_codegen::solver::codegen::rhs_only::rhs_only_kernel_program(
        &full,
        KernelId::GENERIC_COUPLED_ASSEMBLY_RHS_ONLY.as_str(),
    ))
}

pub(crate) fn generate_generic_coupled_assembly_grad_state_rhs_only_kernel_program(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<crate::solver::ir::KernelProgram, String> {
    let full = generate_generic_coupled_assembly_grad_state_kernel_program(model, schemes)?;
    Ok(cfd2_codegen::solver::codegen::rhs_only::rhs_only_kernel_program(
        &full,
        KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE_RHS_ONLY.as_str(),
    ))
}

pub(crate) fn generate_packed_state_gradients_kernel_program(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<crate::solver::ir::KernelProgram, String> {
    let eos_params = extract_eos_params(model);
    // The gradients are needed UNCONDITIONALLY (the "skip when the runtime
    // knob says Upwind" guard must not be emitted) when either:
    // - a convection term declares a non-upwind scheme (the declared scheme
    //   is baked into the assembly and ignores the knob), or
    // - a transpose_dev2 viscous term is present (its face flux is built
    //   from grad_state regardless of the convection scheme; under the
    //   guard it would silently read zeros — and the Taylor-Green MMS
    //   cannot detect that, the term being analytically zero there).
    let gradients_required_unconditionally = model.system.equations().iter().any(|eq| {
        eq.terms().iter().any(|t| {
            (matches!(
                t.op,
                crate::solver::ir::TermOp::Div | crate::solver::ir::TermOp::DivFlux
            ) && t
                .scheme
                .map_or(false, |s| s != crate::solver::scheme::Scheme::Upwind))
                || t.transpose_dev2
                // Viscous dissipation reads the cell velocity-gradient tensor from
                // grad_state too; under the Upwind guard it would read zeros.
                || t.viscous_dissipation
        })
    });
    // grad_state is keyed by STATE OFFSET (matching the assembly's
    // reconstruction reads), so the generator needs each solved unknown's
    // state offset in equation-declaration (boundary-table rank) order.
    // Unknown rank == state offset only for models whose unknowns are a
    // prefix of the state layout; the buoyant model's temperature sits
    // behind d_p/grad_p aux fields.
    let unknown_state_offsets = model_unknown_state_offsets(model)
        .map_err(|e| format!("packed_state_gradients: {e}"))?;
    cfd2_codegen::solver::codegen::generate_packed_state_gradients_kernel_program(
        "packed_state_gradients",
        &model.state_layout,
        &unknown_state_offsets,
        &eos_params,
        !gradients_required_unconditionally,
        model.system.topology() == cfd2_ir::equation::TopologyMode::Structured2D,
    )
}

/// State offsets of the solved unknowns in equation-declaration (boundary-table
/// rank) order: unknown rank `r` of the coupled system lives at state offset
/// `result[r]`. Unknown rank == state offset only for models whose unknowns are
/// a prefix of the state layout (the buoyant model's temperature sits behind
/// the d_p/grad_p aux fields). Shared by the packed-gradients generator and the
/// CPU backend's outer plateau detector.
pub(crate) fn model_unknown_state_offsets(
    model: &crate::solver::model::ModelSpec,
) -> Result<Vec<u32>, String> {
    let slots = resolved_slots_from_layout(&model.state_layout);
    let mut unknown_state_offsets: Vec<u32> = Vec::new();
    for eq in model.system.equations() {
        let target = eq.target();
        let base = resolve_offset_from_slots(&slots, target.name())
            .ok_or_else(|| format!("no state slot for unknown '{}'", target.name()))?;
        for comp in 0..target.kind().component_count() {
            unknown_state_offsets.push(base + comp as u32);
        }
    }
    Ok(unknown_state_offsets)
}

/// [`model_unknown_state_offsets`] grouped by equation TARGET: one inner `Vec` per
/// solved field, holding that field's component state offsets (`[Ux, Uy]`, `[p]`,
/// `[T]`, …).
///
/// The Picard outer-convergence residual must be measured per FIELD, not per
/// component: a nearly-symmetric flow has `max|Uy| ≈ 0`, so a per-component
/// relative residual `|ΔUy| / max|Uy|` never falls below tolerance and would stall
/// the loop at its iteration cap forever. Both components of `U` share the
/// velocity scale.
pub(crate) fn model_unknown_state_offset_groups(
    model: &crate::solver::model::ModelSpec,
) -> Result<Vec<Vec<u32>>, String> {
    let slots = resolved_slots_from_layout(&model.state_layout);
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for eq in model.system.equations() {
        let target = eq.target();
        let base = resolve_offset_from_slots(&slots, target.name())
            .ok_or_else(|| format!("no state slot for unknown '{}'", target.name()))?;
        groups.push(
            (0..target.kind().component_count())
                .map(|c| base + c as u32)
                .collect(),
        );
    }
    Ok(groups)
}

/// Resolve a state offset by field name, supporting component suffixes (e.g., "rho_u_x").
fn resolve_offset_from_slots(slots: &ResolvedStateSlotsSpec, name: &str) -> Option<u32> {
    fn find_slot<'a>(
        slots: &'a ResolvedStateSlotsSpec,
        field: &str,
    ) -> Option<&'a ResolvedStateSlotSpec> {
        slots.slots.iter().find(|s| s.name == field)
    }

    if let Some(slot) = find_slot(slots, name) {
        return Some(slot.base_offset);
    }

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

pub(crate) fn generate_generic_coupled_update_kernel_program(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<crate::solver::ir::KernelProgram, String> {
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
    let resolved_prims: Vec<(u32, cfd2_ir::ast::Expr)> = prims
        .into_iter()
        .filter_map(|(name, expr)| {
            resolve_offset_from_slots(&slots, &name).map(|offset| (offset, expr))
        })
        .collect();

    let eos_params = extract_eos_params(model);
    cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_update_kernel_program(
        KernelId::GENERIC_COUPLED_UPDATE.as_str(),
        &discrete,
        &slots,
        &resolved_prims,
        apply_relaxation,
        relaxation_requires_dtau,
        &eos_params,
    )
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
        &crate::solver::model::all_models()
            .expect("failed to build model definitions"),
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
    let mut matching_rules: Vec<ModelKernelFusionRule> =
        derive_kernel_fusion_rules_for_model(model)
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
    let synthesis_policy = fusion_safety_policy_for_rule(selected);

    let mut programs = Vec::with_capacity(selected.pattern.len());
    for atom in &selected.pattern {
        programs.push(generate_kernel_program_for_model_by_id(
            model, schemes, atom.id,
        )?);
    }

    let fused_program = cfd2_codegen::solver::codegen::fusion::synthesize_fused_program_remapped_whitelisted(
        replacement_id.as_str().to_string(),
        selected.name,
        &programs,
        synthesis_policy,
        &selected.binding_remaps,
        &selected.expected_hazards,
    )?;
    let wgsl = cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl(&fused_program)?;
    Ok(Some(wgsl.to_wgsl()))
}

fn fusion_safety_policy_for_rule(
    rule: &ModelKernelFusionRule,
) -> cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy {
    rule.guards
        .iter()
        .filter_map(|guard| match guard {
            FusionGuard::MinPolicy(policy) => Some(*policy),
            _ => None,
        })
        .max()
        .map(|policy| match policy {
            KernelFusionPolicy::Off | KernelFusionPolicy::Safe => {
                cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy::Safe
            }
            KernelFusionPolicy::Aggressive => {
                cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy::Aggressive
            }
        })
        .unwrap_or(cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy::Safe)
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
                let filename = kernel_output_name_for_model(model.id, kernel_id)
                    .map_err(std::io::Error::other)?;
                let path = cfd2_codegen::compiler::write_generated_wgsl(
                    base_dir.as_ref(),
                    filename,
                    &wgsl,
                )?;
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
    use cfd2_ir::kernel::{
        BindingAccess, DispatchDomain, KernelBinding,
        KernelProgram, LaunchSemantics,
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
        body_stmt: cfd2_ir::ast::Stmt,
    ) -> impl Fn(&ModelSpec, &crate::solver::ir::SchemeRegistry) -> Result<KernelProgram, String>
           + Send
           + Sync
           + 'static {
        move |_model, _schemes| {
            use cfd2_ir::ast::{Expr, Stmt, Type};
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
            let indexing_stmts = vec![Stmt::Let {
                name: "inv".to_string(),
                ty: None,
                expr: Expr::ident("idx"),
            }];
            let preamble_stmts = vec![Stmt::Var {
                name: "value".to_string(),
                ty: Some(Type::F32),
                expr: Some(Expr::ident("state").index(Expr::ident("idx"))),
            }];
            let body_stmts = vec![
                body_stmt.clone(),
                Stmt::Assign {
                    target: Expr::ident("state").index(Expr::ident("idx")),
                    value: Expr::ident("value"),
                },
                Stmt::Assign {
                    target: Expr::ident("state").index(Expr::ident("inv")),
                    value: Expr::ident("state").index(Expr::ident("idx")),
                },
            ];
            program.indexing = indexing_stmts;
            program.preamble = preamble_stmts;
            program.body = body_stmts;
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

        let mut model = crate::solver::model::generic_diffusion_demo_model().expect("model");
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
                binding_remaps: vec![],
                expected_hazards: vec![],
            }],
            ..Default::default()
        };

        let mut model = crate::solver::model::generic_diffusion_demo_model().expect("model");
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
                    contract_dsl_kernel_generator(dsl_id.as_str(), cfd2_ir::ast::Stmt::Assign {
                        target: cfd2_ir::ast::Expr::ident("value"),
                        value: cfd2_ir::ast::Expr::ident("value") + cfd2_ir::ast::Expr::lit_f32(1.0),
                    }),
                ),
            ],
            ..Default::default()
        };

        let mut model = crate::solver::model::generic_diffusion_demo_model().expect("model");
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
                    contract_dsl_kernel_generator(a_id.as_str(), cfd2_ir::ast::Stmt::Assign {
                        target: cfd2_ir::ast::Expr::ident("value"),
                        value: cfd2_ir::ast::Expr::ident("value") + cfd2_ir::ast::Expr::lit_f32(1.0),
                    }),
                ),
                ModelKernelGeneratorSpec::new_dsl(
                    b_id,
                    contract_dsl_kernel_generator(b_id.as_str(), cfd2_ir::ast::Stmt::Assign {
                        target: cfd2_ir::ast::Expr::ident("value"),
                        value: cfd2_ir::ast::Expr::ident("value") + cfd2_ir::ast::Expr::lit_f32(2.0),
                    }),
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
                binding_remaps: vec![],
                expected_hazards: vec![],
            }],
            ..Default::default()
        };

        let mut model = crate::solver::model::generic_diffusion_demo_model().expect("model");
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
                binding_remaps: vec![],
                expected_hazards: vec![],
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
                binding_remaps: vec![],
                expected_hazards: vec![],
            },
        ];
        let kernels = vec![a, b, c];
        let module_names = ["fusion_module"];
        let ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Safe,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            has_neighbor_grad_consumers: false,
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
            binding_remaps: vec![],
            expected_hazards: vec![],
        }];
        let kernels = vec![base];
        let module_names = ["fusion_module"];

        let safe_ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Safe,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            has_neighbor_grad_consumers: false,
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
            has_neighbor_grad_consumers: false,
            module_names: &module_names,
        };
        let aggressive_out = apply_model_fusion_rules(&kernels, &rules, &aggressive_ctx);
        assert_eq!(aggressive_out.kernels.len(), 1);
        assert_eq!(aggressive_out.kernels[0].id, fused.id);
        assert_eq!(aggressive_out.applied.len(), 1);
    }

    #[test]
    fn fusion_guard_exact_policy_gates_rules() {
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
            name: "fuse_when_safe_only",
            priority: 1,
            phase: KernelPhaseId::Update,
            pattern: vec![KernelPatternAtom::id(base.id)],
            replacement: fused,
            guards: vec![FusionGuard::ExactPolicy(KernelFusionPolicy::Safe)],
            binding_remaps: vec![],
            expected_hazards: vec![],
        }];
        let kernels = vec![base];
        let module_names = ["fusion_module"];

        let off_ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Off,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            has_neighbor_grad_consumers: false,
            module_names: &module_names,
        };
        let off_out = apply_model_fusion_rules(&kernels, &rules, &off_ctx);
        assert_eq!(off_out.kernels[0].id, base.id);
        assert!(off_out.applied.is_empty());

        let safe_ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Safe,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            has_neighbor_grad_consumers: false,
            module_names: &module_names,
        };
        let safe_out = apply_model_fusion_rules(&kernels, &rules, &safe_ctx);
        assert_eq!(safe_out.kernels[0].id, fused.id);
        assert_eq!(safe_out.applied.len(), 1);

        let aggressive_ctx = KernelFusionContext {
            policy: KernelFusionPolicy::Aggressive,
            stepping: KernelFusionStepping::Coupled,
            has_grad_state: false,
            has_neighbor_grad_consumers: false,
            module_names: &module_names,
        };
        let aggressive_out = apply_model_fusion_rules(&kernels, &rules, &aggressive_ctx);
        assert_eq!(aggressive_out.kernels[0].id, base.id);
        assert!(aggressive_out.applied.is_empty());
    }

    #[test]
    fn fusion_synthesis_policy_tracks_min_policy_guard() {
        let replacement = ModelKernelSpec {
            id: KernelId("fusion/fused"),
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        };
        let base_rule = ModelKernelFusionRule {
            name: "rule",
            priority: 1,
            phase: KernelPhaseId::Update,
            pattern: vec![KernelPatternAtom::id(KernelId("fusion/base"))],
            replacement,
            guards: Vec::new(),
            binding_remaps: vec![],
            expected_hazards: vec![],
        };

        let safe = fusion_safety_policy_for_rule(&base_rule);
        assert_eq!(
            safe,
            cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy::Safe
        );

        let mut guarded_safe = base_rule.clone();
        guarded_safe
            .guards
            .push(FusionGuard::MinPolicy(KernelFusionPolicy::Safe));
        let safe_guarded_policy = fusion_safety_policy_for_rule(&guarded_safe);
        assert_eq!(
            safe_guarded_policy,
            cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy::Safe
        );

        let mut guarded_aggressive = base_rule.clone();
        guarded_aggressive
            .guards
            .push(FusionGuard::MinPolicy(KernelFusionPolicy::Aggressive));
        let aggressive_policy = fusion_safety_policy_for_rule(&guarded_aggressive);
        assert_eq!(
            aggressive_policy,
            cfd2_codegen::solver::codegen::fusion::FusionSafetyPolicy::Aggressive
        );
    }

    #[test]
    fn generic_coupled_kernels_are_dsl_artifacts() {
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);
        let model = crate::solver::model::compressible_model().expect("model");

        for kernel_id in [
            KernelId::GENERIC_COUPLED_ASSEMBLY,
            KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
            KernelId::GENERIC_COUPLED_APPLY,
            KernelId::GENERIC_COUPLED_UPDATE,
            KernelId::FLUX_MODULE,
            KernelId::FLUX_MODULE_GRADIENTS,
        ] {
            let artifact = generate_kernel_artifact_for_model_by_id(&model, &schemes, kernel_id)
                .expect("generic coupled assembly generator should resolve");
            assert!(
                matches!(artifact, ModelKernelArtifact::DslProgram(_)),
                "kernel '{}' should be emitted as a DSL artifact",
                kernel_id.as_str()
            );
        }
    }

    #[test]
    fn contract_flux_module_uses_runtime_scheme_and_includes_limited_paths() {
        // Contract: the flux-module kernel must not ignore the runtime `advection_scheme` knob.
        // It should branch on `constants.scheme` and include the limited reconstruction paths
        // in WGSL (even though shipped defaults remain Upwind).
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);

        let model = crate::solver::model::compressible_model().expect("model");
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

        let model = crate::solver::model::incompressible_momentum_model().expect("model");
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

        let model = crate::solver::model::incompressible_momentum_model().expect("model");
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
        let model = crate::solver::model::incompressible_momentum_model().expect("model");

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

    /// ALE mesh-flux face-density INVARIANT (regression gate for the
    /// upwind-rho_f mesh-relative subtraction fix): the face density multiplying
    /// `mesh_fluxes` in the assembly's mesh-relative subtraction must be the
    /// EXACT expression the flux module bakes into the convective mass flux.
    /// For the thermal (`t_ref`, real-EOS) ALE family the flux module UPWINDS
    /// `rho_f` by the MESH-RELATIVE face-normal velocity; the assembly must
    /// therefore (a) hoist the identical upwind blend (`ale_rho_f` /
    /// `ale_upwind_sgn` locals, mesh-relative `- ale_mesh_flux_out / area` in
    /// the sgn argument) and (b) multiply `mesh_fluxes[face_idx]` by THAT local
    /// — never by the central Lerp the pre-fix code used (which removed a
    /// different mass flux than convection added: a spurious source
    /// ∝ (rho jump) × (mesh velocity) wherever grad(rho) != 0 on a moving mesh).
    /// The barotropic ALE family (no `t_ref`) keeps the central Lerp on BOTH
    /// sides — also asserted, so this test pins the whole case split of
    /// `unified_assembly::ale_relative_flux_expr` against
    /// `flux_derivation::density_face_expr`.
    #[test]
    fn contract_ale_mesh_flux_face_density_matches_flux_module() {
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);

        // THERMAL ALE family: upwind rho_f, mesh-relative direction, both sites.
        let model = crate::solver::model::allmach_thermal_ale_model().expect("model");
        let flux = generate_kernel_wgsl_for_model_by_id(&model, &schemes, KernelId::FLUX_MODULE)
            .expect("flux_module WGSL");
        assert!(
            flux.contains("mesh_fluxes[idx] / area"),
            "thermal ALE flux module must upwind rho_f by the MESH-RELATIVE normal velocity \
             (sgn argument `U_f.n - mesh_fluxes[idx]/area`); absolute-velocity upwinding picks \
             the downwind side wherever the mesh outruns the flow"
        );
        assert!(
            flux.contains("0.5 * (s_own_rho + s_neigh_rho)"),
            "thermal ALE flux module must carry the upwind rho_f blend"
        );

        let asm = generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId::GENERIC_COUPLED_ASSEMBLY,
        )
        .expect("generic_coupled_assembly WGSL");
        assert!(
            asm.contains("- ale_mesh_flux_out / area"),
            "thermal ALE assembly's upwind sgn must use the same MESH-RELATIVE velocity \
             as the flux module"
        );
        assert!(
            asm.contains("ale_upwind_sgn * 0.5 * (state["),
            "thermal ALE assembly must hoist the flux module's upwind rho_f blend (ale_rho_f)"
        );
        // Every mesh-relative subtraction must multiply mesh_fluxes by the
        // hoisted upwind blend — the pre-fix central Lerp against mesh_fluxes
        // is the exact defect this pins out.
        assert!(
            asm.contains("- ale_rho_f * mesh_fluxes[face_idx]"),
            "thermal ALE assembly's mesh-relative subtraction must use the upwind ale_rho_f"
        );
        let subtractions = asm.matches("* mesh_fluxes[face_idx]").count();
        let upwind_subtractions = asm.matches("ale_rho_f * mesh_fluxes[face_idx]").count();
        assert!(
            subtractions > 0 && subtractions == upwind_subtractions,
            "every mesh-relative subtraction in the thermal ALE assembly must carry the \
             upwind ale_rho_f face density ({upwind_subtractions}/{subtractions} did)"
        );

        // BAROTROPIC ALE family: central Lerp on both sides, NO upwind locals,
        // NO mesh-relative sgn (there is no sgn at all).
        let baro = crate::solver::model::allmach_pressure_ale_model().expect("model");
        let baro_flux =
            generate_kernel_wgsl_for_model_by_id(&baro, &schemes, KernelId::FLUX_MODULE)
                .expect("flux_module WGSL");
        assert!(
            !baro_flux.contains("mesh_fluxes"),
            "barotropic ALE flux module central-Lerps rho_f (no upwind, no mesh_fluxes binding)"
        );
        let baro_asm = generate_kernel_wgsl_for_model_by_id(
            &baro,
            &schemes,
            KernelId::GENERIC_COUPLED_ASSEMBLY,
        )
        .expect("generic_coupled_assembly WGSL");
        assert!(
            !baro_asm.contains("ale_rho_f"),
            "barotropic ALE assembly must keep the central-Lerp face density \
             (matching its flux module), not the thermal upwind blend"
        );
        assert!(
            baro_asm.contains("* mesh_fluxes[face_idx]"),
            "barotropic ALE assembly still subtracts the mesh flux"
        );
    }

    /// IBM face-d_p seal INVARIANT (regression gate for the interface-leak
    /// fix): on models whose layout carries the Brinkman mask `ibm_penalty_U`,
    /// the flux module and the assembly's pressure-Laplacian coefficient must
    /// apply the SAME impermeable-wall face d_p — `min(d_p_own, d_p_neigh)` —
    /// and the flux module must additionally seal the WHOLE face flux with
    /// `1 - min(|Sp_own|+|Sp_neigh|, 1)` on every component. If either site
    /// regresses to the arithmetic blend, the wall leaks ~half the fluid-side
    /// conductance (measured gross leak 0.39·rho·Uin·D — the interface
    /// speckle); if the flux keeps the blended advective term while the
    /// Laplacian is sealed, solid-cell continuity is forced through the
    /// ~1/|Sp| conductance and solid p blows up ~1/d_p (measured p ptp
    /// 2.3e-4 → 3.8e15). The assembly must ALSO keep the UNSEALED blend on
    /// SOLID rows (the anchor that slaves solid p to the neighbour average —
    /// without it the all-Mach ddt(psi_precond,p) solid block is a pressure
    /// resonator, ring mean|U| ~16x inlet) and gate the deferred correction.
    /// Non-IBM models must keep the pre-fix blend byte-identically — also
    /// pinned, so the whole gate keys on layout field presence exactly.
    #[test]
    fn contract_ibm_face_dp_seal_matches_flux_module() {
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);

        for model in [
            crate::solver::model::incompressible_momentum_structured_model().expect("model"),
            crate::solver::model::allmach_thermal_structured_model().expect("model"),
        ] {
            let flux =
                generate_kernel_wgsl_for_model_by_id(&model, &schemes, KernelId::FLUX_MODULE)
                    .expect("flux_module WGSL");
            assert!(
                flux.contains("min(s_own_d_p, s_neigh_d_p)"),
                "[{}] IBM flux module must use the min-based (impermeable-wall) face d_p",
                model.id
            );
            assert!(
                !flux.contains("s_own_d_p * lambda"),
                "[{}] IBM flux module must not blend d_p arithmetically across the \
                 penalty jump (the interface-leak defect)",
                model.id
            );
            let writes = flux.matches("fluxes[sfd_face_id").count();
            let seals = flux
                .matches("(1.0 - min(abs(s_own_ibm_penalty_U) + abs(s_neigh_ibm_penalty_U), 1.0))")
                .count();
            assert!(
                writes > 0 && writes == seals,
                "[{}] every flux component must carry the whole-face IBM seal \
                 ({seals}/{writes} did) — an unsealed advective HbyA.n forces solid \
                 continuity through the ~1/|Sp| conductance and blows up solid p",
                model.id
            );

            let asm = generate_kernel_wgsl_for_model_by_id(
                &model,
                &schemes,
                KernelId::GENERIC_COUPLED_ASSEMBLY,
            )
            .expect("generic_coupled_assembly WGSL");
            assert!(
                asm.contains(") * min(state["),
                "[{}] IBM assembly's pressure-Laplacian fluid-row coefficient must use \
                 the SAME min-based face d_p as the flux module",
                model.id
            );
            assert!(
                asm.contains(") > 0.0), !is_boundary)"),
                "[{}] IBM assembly's pressure-Laplacian must be row-dependent \
                 (solid rows keep the unsealed blend that anchors solid p)",
                model.id
            );
            assert!(
                asm.contains("select(1.0, 0.0, abs(state["),
                "[{}] IBM assembly must gate the deferred high-order correction \
                 off penalty-adjacent faces",
                model.id
            );
        }

        // NON-IBM control (no `ibm_penalty_U` in the layout): the pre-fix
        // arithmetic blend everywhere, no seal, no penalty reads — pinning
        // that the gate keys on layout field presence and nothing else.
        let base = crate::solver::model::incompressible_momentum_model().expect("model");
        let base_flux = generate_kernel_wgsl_for_model_by_id(&base, &schemes, KernelId::FLUX_MODULE)
            .expect("flux_module WGSL");
        assert!(
            base_flux.contains("s_own_d_p * lambda + s_neigh_d_p * lambda_other"),
            "non-IBM flux module keeps the distance-weighted face d_p blend"
        );
        assert!(
            !base_flux.contains("ibm_penalty_U") && !base_flux.contains("min(s_own_d_p"),
            "non-IBM flux module must not reference the IBM seal"
        );
        let base_asm = generate_kernel_wgsl_for_model_by_id(
            &base,
            &schemes,
            KernelId::GENERIC_COUPLED_ASSEMBLY,
        )
        .expect("generic_coupled_assembly WGSL");
        assert!(
            !base_asm.contains(") * min(state[") && !base_asm.contains("select(1.0, 0.0, abs(state["),
            "non-IBM assembly must keep the pre-fix blend and deferred correction"
        );
    }
}
