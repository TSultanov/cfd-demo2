use std::sync::Arc;

/// Stable identifier for a compute kernel.
///
/// This is used by the unified solver orchestration to decouple scheduling and lookup
/// from handwritten enums/matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub &'static str);

impl KernelId {
    pub const FLUX_MODULE_GRADIENTS: KernelId = KernelId("flux_module_gradients");
    pub const FLUX_MODULE: KernelId = KernelId("flux_module");

    pub const GENERIC_COUPLED_ASSEMBLY: KernelId = KernelId("generic_coupled_assembly");
    pub const GENERIC_COUPLED_ASSEMBLY_GRAD_STATE: KernelId =
        KernelId("generic_coupled_assembly_grad_state");
    pub const GENERIC_COUPLED_APPLY: KernelId = KernelId("generic_coupled_apply");
    pub const GENERIC_COUPLED_UPDATE: KernelId = KernelId("generic_coupled_update");

    // Handwritten solver-infrastructure kernels (single-entrypoint compute shaders).
    pub const DOT_PRODUCT: KernelId = KernelId("dot_product");
    pub const DOT_PRODUCT_PAIR: KernelId = KernelId("dot_product_pair");

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

pub(crate) type ModelKernelWgslGenerator = Arc<
    dyn Fn(&crate::solver::model::ModelSpec, &crate::solver::ir::SchemeRegistry) -> Result<String, String>
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
    pub generator: ModelKernelWgslGenerator,
}

impl ModelKernelGeneratorSpec {
    pub fn new(
        id: KernelId,
        generator: impl Fn(&crate::solver::model::ModelSpec, &crate::solver::ir::SchemeRegistry) -> Result<String, String>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            id,
            scope: KernelWgslScope::PerModel,
            generator: Arc::new(generator),
        }
    }

    pub fn new_shared(
        id: KernelId,
        generator: impl Fn(&crate::solver::model::ModelSpec, &crate::solver::ir::SchemeRegistry) -> Result<String, String>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            id,
            scope: KernelWgslScope::Shared,
            generator: Arc::new(generator),
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

pub fn kernel_output_name_for_model(model_id: &str, kernel_id: KernelId) -> Result<String, String> {
    let prefix = kernel_id.as_str().replace('/', "_");
    if model_id.is_empty() {
        Ok(format!("{prefix}.wgsl"))
    } else {
        Ok(format!("{prefix}_{model_id}.wgsl"))
    }
}

pub(crate) fn generate_flux_module_gradients_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let flux = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "flux_module_gradients requested but model has no flux module".to_string())?;

    match flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel {
            gradients: Some(_),
            ..
        }
        | crate::solver::model::flux_module::FluxModuleSpec::Scheme {
            gradients: Some(_),
            ..
        } => {
            let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
            cfd2_codegen::solver::codegen::generate_flux_module_gradients_wgsl(
                &model.state_layout,
                &flux_layout,
            )
            .map_err(|e| e.to_string())
        }
        _ => Err("flux_module_gradients requested but model has no gradients stage".to_string()),
    }
}

pub(crate) fn generate_flux_module_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let flux_stride = model.system.unknowns_per_cell();
    let prims = model
        .primitives
        .ordered()
        .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;

    let flux = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "flux_module requested but model has no flux module".to_string())?;

    match flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel { kernel, .. } => {
            Ok(cfd2_codegen::solver::codegen::flux_module::generate_flux_module_wgsl(
                &model.state_layout,
                &flux_layout,
                flux_stride,
                &prims,
                kernel,
            ))
        }
        crate::solver::model::flux_module::FluxModuleSpec::Scheme { scheme, .. } => {
            use crate::solver::scheme::Scheme;

            let schemes = [
                Scheme::Upwind,
                Scheme::SecondOrderUpwind,
                Scheme::QUICK,
                Scheme::SecondOrderUpwindMinMod,
                Scheme::SecondOrderUpwindVanLeer,
                Scheme::QUICKMinMod,
                Scheme::QUICKVanLeer,
            ];

            let mut variants = Vec::new();
            for reconstruction in schemes {
                let kernel = crate::solver::model::flux_schemes::lower_flux_scheme(
                    scheme,
                    &model.system,
                    reconstruction,
                )
                .map_err(|e| format!("flux scheme lowering failed: {e}"))?;
                variants.push((reconstruction, kernel));
            }

            Ok(cfd2_codegen::solver::codegen::flux_module::generate_flux_module_wgsl_runtime_scheme(
                &model.state_layout,
                &flux_layout,
                flux_stride,
                &prims,
                &variants,
            ))
        }
    }
}

pub(crate) fn generate_generic_coupled_assembly_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let discrete = cfd2_codegen::solver::codegen::lower_system(&model.system, schemes)
        .map_err(|e| e.to_string())?;

    // `generic_coupled_assembly` is the no-grad_state variant.
    // The grad_state-binding variant is emitted under `KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE`.
    let needs_gradients = false;
    let flux_stride = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .map(|_| model.system.unknowns_per_cell())
        .unwrap_or(0);
    Ok(cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_wgsl(
        &discrete,
        &model.state_layout,
        flux_stride,
        needs_gradients,
    ))
}

pub(crate) fn generate_generic_coupled_assembly_grad_state_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
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

    let discrete = cfd2_codegen::solver::codegen::lower_system(&model.system, schemes)
        .map_err(|e| e.to_string())?;

    let needs_gradients = true;
    let flux_stride = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .map(|_| model.system.unknowns_per_cell())
        .unwrap_or(0);
    Ok(cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_wgsl(
        &discrete,
        &model.state_layout,
        flux_stride,
        needs_gradients,
    ))
}

pub(crate) fn generate_packed_state_gradients_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    cfd2_codegen::solver::codegen::generate_packed_state_gradients_wgsl(
        &model.state_layout,
        model.system.unknowns_per_cell(),
    )
    .map_err(|e| e.to_string())
}

pub(crate) fn generate_generic_coupled_update_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let discrete = cfd2_codegen::solver::codegen::lower_system(&model.system, schemes)
        .map_err(|e| e.to_string())?;
    let prims = model
        .primitives
        .ordered()
        .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
    let (apply_relaxation, relaxation_requires_dtau) = match model.method().map_err(|e| e.to_string())?
    {
        crate::solver::model::method::MethodSpec::Coupled(caps) => {
            (caps.apply_relaxation_in_update, caps.relaxation_requires_dtau)
        }
    };
    Ok(cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_update_wgsl(
        &discrete,
        &model.state_layout,
        &prims,
        apply_relaxation,
        relaxation_requires_dtau,
    ))
}

fn kernel_generator_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    kernel_id: KernelId,
) -> Option<&ModelKernelGeneratorSpec> {
    for module in &model.modules {
        let module: &dyn crate::solver::model::module::ModelModule = module;
        if let Some(spec) = module.kernel_generators().iter().find(|s| s.id == kernel_id) {
            return Some(spec);
        }
    }

    None
}

pub fn generate_kernel_wgsl_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kernel_id: KernelId,
) -> Result<String, String> {
    if let Some(gen) = kernel_generator_for_model_by_id(model, kernel_id) {
        return gen.generator.as_ref()(model, schemes);
    }

    Err(format!(
        "KernelId '{}' is not a build-time generated per-model kernel",
        kernel_id.as_str()
    ))
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

    let mut wgsl_by_id: std::collections::HashMap<KernelId, String> = std::collections::HashMap::new();
    for model in models {
        for module in &model.modules {
            let module: &dyn crate::solver::model::module::ModelModule = module;
            for spec in module.kernel_generators() {
                if spec.scope != KernelWgslScope::Shared {
                    continue;
                }

                let wgsl = spec
                    .generator
                    .as_ref()(model, schemes)
                    .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;

                if let Some(prev) = wgsl_by_id.insert(spec.id, wgsl.clone()) {
                    if prev != wgsl {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "shared kernel '{}' generated different WGSL across models (latest from model '{}' module '{}')",
                                spec.id.as_str(),
                                model.id,
                                module.name()
                            ),
                        ));
                    }
                }
            }
        }
    }

    let mut shared_ids: Vec<KernelId> = wgsl_by_id.keys().copied().collect();
    shared_ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));

    for kernel_id in shared_ids {
        let filename = kernel_output_name_for_model("", kernel_id)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
        let wgsl = wgsl_by_id
            .get(&kernel_id)
            .expect("shared kernel id disappeared from table");
        let path =
            cfd2_codegen::compiler::write_generated_wgsl(base_dir, filename, wgsl)?;
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

pub fn emit_model_kernels_wgsl_with_ids(
    base_dir: impl AsRef<std::path::Path>,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> std::io::Result<Vec<(KernelId, std::path::PathBuf)>> {
    let mut outputs = Vec::new();

    let specs = derive_kernel_specs_for_model(model)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;

    let mut seen: std::collections::HashSet<KernelId> = std::collections::HashSet::new();
    for spec in specs {
        if !seen.insert(spec.id) {
            continue;
        }

        let Some(generator) = kernel_generator_for_model_by_id(model, spec.id) else {
            continue;
        };
        if generator.scope == KernelWgslScope::Shared {
            continue;
        }

        let path = emit_model_kernel_wgsl_by_id(&base_dir, model, schemes, spec.id)?;
        outputs.push((spec.id, path));
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
    let filename = kernel_output_name_for_model(model.id, kernel_id)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    let wgsl = generate_kernel_wgsl_for_model_by_id(model, schemes, kernel_id)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    cfd2_codegen::compiler::write_generated_wgsl(base_dir, filename, &wgsl)
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use crate::solver::model::module::KernelBundleModule;
    use crate::solver::model::ModelSpec;

    fn contract_kernel_generator(
        _model: &ModelSpec,
        _schemes: &crate::solver::ir::SchemeRegistry,
    ) -> Result<String, String> {
        Ok("// contract: module-defined kernel generator\n".to_string())
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
            generators: vec![ModelKernelGeneratorSpec::new(contract_id, contract_kernel_generator)],
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
}

// (intentionally no additional kernel-analysis helpers here; kernel selection is recipe-driven)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::scheme::Scheme;
    use std::collections::HashSet;

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
