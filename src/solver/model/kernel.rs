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
    pub const GENERIC_COUPLED_APPLY: KernelId = KernelId("generic_coupled_apply");
    pub const GENERIC_COUPLED_UPDATE: KernelId = KernelId("generic_coupled_update");

    // Handwritten solver-infrastructure kernels (single-entrypoint compute shaders).
    pub const DOT_PRODUCT: KernelId = KernelId("dot_product");
    pub const DOT_PRODUCT_PAIR: KernelId = KernelId("dot_product_pair");

    pub const SCALARS_INIT: KernelId = KernelId("scalars/init_scalars");
    pub const SCALARS_INIT_CG: KernelId = KernelId("scalars/init_cg_scalars");
    pub const SCALARS_REDUCE_RHO_NEW_R_R: KernelId = KernelId("scalars/reduce_rho_new_r_r");
    pub const SCALARS_REDUCE_R0_V: KernelId = KernelId("scalars/reduce_r0_v");
    pub const SCALARS_REDUCE_T_S_T_T: KernelId = KernelId("scalars/reduce_t_s_t_t");
    pub const SCALARS_UPDATE_CG_ALPHA: KernelId = KernelId("scalars/update_cg_alpha");
    pub const SCALARS_UPDATE_CG_BETA: KernelId = KernelId("scalars/update_cg_beta");
    pub const SCALARS_UPDATE_RHO_OLD: KernelId = KernelId("scalars/update_rho_old");

    pub const LINEAR_SOLVER_SPMV_P_V: KernelId = KernelId("linear_solver/spmv_p_v");
    pub const LINEAR_SOLVER_SPMV_S_T: KernelId = KernelId("linear_solver/spmv_s_t");
    pub const LINEAR_SOLVER_BICGSTAB_UPDATE_X_R: KernelId =
        KernelId("linear_solver/bicgstab_update_x_r");
    pub const LINEAR_SOLVER_BICGSTAB_UPDATE_P: KernelId =
        KernelId("linear_solver/bicgstab_update_p");
    pub const LINEAR_SOLVER_BICGSTAB_UPDATE_S: KernelId =
        KernelId("linear_solver/bicgstab_update_s");
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

    pub const PRECONDITIONER_BUILD_SCHUR_RHS: KernelId = KernelId("preconditioner/build_schur_rhs");
    pub const PRECONDITIONER_FINALIZE_PRECOND: KernelId =
        KernelId("preconditioner/finalize_precond");
    pub const PRECONDITIONER_SPMV_PHAT_V: KernelId = KernelId("preconditioner/spmv_phat_v");
    pub const PRECONDITIONER_SPMV_SHAT_T: KernelId = KernelId("preconditioner/spmv_shat_t");

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

/// A fully specified kernel pass derived from the model + method selection.
///
/// This is the "model side" of a recipe: phase/dispatch are decided here so the
/// GPU recipe does not need per-kernel `match` arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelKernelSpec {
    pub id: KernelId,
    pub phase: KernelPhaseId,
    pub dispatch: DispatchKindId,
}

pub(crate) type ModelKernelWgslGenerator = fn(
    &crate::solver::model::ModelSpec,
    &crate::solver::ir::SchemeRegistry,
) -> Result<String, String>;

#[derive(Clone, Copy)]
pub struct ModelKernelGeneratorSpec {
    pub id: KernelId,
    pub generator: ModelKernelWgslGenerator,
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

pub fn generic_coupled_module(
    method: crate::solver::model::method::MethodSpec,
) -> crate::solver::model::module::KernelBundleModule {
    use crate::solver::model::module::{ModuleManifest, NamedParamKey};

    crate::solver::model::module::KernelBundleModule {
        name: "generic_coupled",
        kernels: vec![
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
            },
        ],
        generators: vec![
            ModelKernelGeneratorSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                generator: generate_generic_coupled_assembly_kernel_wgsl,
            },
            ModelKernelGeneratorSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                generator: generate_generic_coupled_update_kernel_wgsl,
            },
        ],
        manifest: ModuleManifest {
            method: Some(method),
            named_params: vec![
                NamedParamKey::Key("dt"),
                NamedParamKey::Key("dtau"),
                NamedParamKey::Key("advection_scheme"),
                NamedParamKey::Key("time_scheme"),
                NamedParamKey::Key("preconditioner"),
                NamedParamKey::Key("viscosity"),
                NamedParamKey::Key("density"),
                NamedParamKey::Key("alpha_u"),
                NamedParamKey::Key("alpha_p"),
                NamedParamKey::Key("nonconverged_relax"),
                NamedParamKey::Key("outer_iters"),
                NamedParamKey::Key("detailed_profiling_enabled"),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn flux_module_module(
    flux: crate::solver::model::flux_module::FluxModuleSpec,
) -> Result<crate::solver::model::module::KernelBundleModule, String> {
    use crate::solver::model::flux_module::FluxModuleSpec;
    use crate::solver::model::module::ModuleManifest;

    let has_gradients = match &flux {
        FluxModuleSpec::Kernel { gradients, .. } => gradients.is_some(),
        FluxModuleSpec::Scheme { gradients, .. } => gradients.is_some(),
    };

    let mut out = crate::solver::model::module::KernelBundleModule {
        name: "flux_module",
        kernels: Vec::new(),
        generators: Vec::new(),
        manifest: ModuleManifest {
            flux_module: Some(flux),
            ..Default::default()
        },
        ..Default::default()
    };

    if has_gradients {
        out.kernels.push(ModelKernelSpec {
            id: KernelId::FLUX_MODULE_GRADIENTS,
            phase: KernelPhaseId::Gradients,
            dispatch: DispatchKindId::Cells,
        });
        out.generators.push(ModelKernelGeneratorSpec {
            id: KernelId::FLUX_MODULE_GRADIENTS,
            generator: generate_flux_module_gradients_kernel_wgsl,
        });
    }

    out.kernels.push(ModelKernelSpec {
        id: KernelId::FLUX_MODULE,
        phase: KernelPhaseId::FluxComputation,
        dispatch: DispatchKindId::Faces,
    });
    out.generators.push(ModelKernelGeneratorSpec {
        id: KernelId::FLUX_MODULE,
        generator: generate_flux_module_kernel_wgsl,
    });

    Ok(out)
}

pub fn kernel_output_name_for_model(model_id: &str, kernel_id: KernelId) -> Result<String, String> {
    let prefix = kernel_id.as_str().replace('/', "_");
    if model_id.is_empty() {
        Ok(format!("{prefix}.wgsl"))
    } else {
        Ok(format!("{prefix}_{model_id}.wgsl"))
    }
}

pub fn generate_dp_init_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let stride = model.state_layout.stride();
    let Some(d_p) = model.state_layout.field("d_p") else {
        return Err("dp_init requested but model has no 'd_p' field in state layout".to_string());
    };
    if d_p.kind() != crate::solver::model::backend::FieldKind::Scalar {
        return Err("dp_init requires 'd_p' to be a scalar field".to_string());
    }
    let d_p_offset = d_p.offset();
    Ok(cfd2_codegen::solver::codegen::generate_dp_init_wgsl(
        stride, d_p_offset,
    ))
}

pub fn generate_dp_update_from_diag_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    use crate::solver::model::backend::FieldKind;

    let stride = model.state_layout.stride();
    let Some(d_p) = model.state_layout.field("d_p") else {
        return Err(
            "dp_update_from_diag requested but model has no 'd_p' field in state layout"
                .to_string(),
        );
    };
    if d_p.kind() != FieldKind::Scalar {
        return Err("dp_update_from_diag requires 'd_p' to be a scalar field".to_string());
    }
    let d_p_offset = d_p.offset();

    let coupling = crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp(
        model,
        "d_p",
    )
    .map_err(|e| format!("dp_update_from_diag {e}"))?;
    let momentum = coupling.momentum;

    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let mut u_indices = Vec::new();
    for component in 0..momentum.kind().component_count() as u32 {
        let Some(offset) = flux_layout.offset_for_field_component(momentum, component) else {
            return Err(format!(
                "dp_update_from_diag: missing unknown offset for momentum field '{}' component {}",
                momentum.name(),
                component
            ));
        };
        u_indices.push(offset);
    }

    cfd2_codegen::solver::codegen::generate_dp_update_from_diag_wgsl(
        stride,
        d_p_offset,
        model.system.unknowns_per_cell(),
        &u_indices,
    )
}

pub fn generate_rhie_chow_correct_velocity_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    use crate::solver::model::backend::FieldKind;

    let stride = model.state_layout.stride();
    let d_p = model
        .state_layout
        .offset_for("d_p")
        .ok_or_else(|| "rhie_chow/correct_velocity requires 'd_p' in state layout".to_string())?;

    let coupling = crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp(
        model,
        "d_p",
    )
    .map_err(|e| format!("rhie_chow/correct_velocity {e}"))?;
    let momentum = coupling.momentum;
    let pressure = coupling.pressure;

    if momentum.kind() != FieldKind::Vector2 {
        return Err(
            "rhie_chow/correct_velocity currently supports only Vector2 momentum fields"
                .to_string(),
        );
    }

    let u_x = model.state_layout.component_offset(momentum.name(), 0).ok_or_else(|| {
        format!(
            "rhie_chow/correct_velocity requires '{}[0]' in state layout",
            momentum.name()
        )
    })?;
    let u_y = model.state_layout.component_offset(momentum.name(), 1).ok_or_else(|| {
        format!(
            "rhie_chow/correct_velocity requires '{}[1]' in state layout",
            momentum.name()
        )
    })?;

    let grad_name = format!("grad_{}", pressure.name());
    let grad_p_x = model
        .state_layout
        .component_offset(&grad_name, 0)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity requires '{}[0]' in state layout",
                grad_name
            )
        })?;
    let grad_p_y = model
        .state_layout
        .component_offset(&grad_name, 1)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity requires '{}[1]' in state layout",
                grad_name
            )
        })?;

    Ok(cfd2_codegen::solver::codegen::generate_rhie_chow_correct_velocity_wgsl(
        stride, u_x, u_y, d_p, grad_p_x, grad_p_y,
    ))
}

fn generate_flux_module_gradients_kernel_wgsl(
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

fn generate_flux_module_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let flux_stride = model.gpu.flux.map(|f| f.stride).unwrap_or(0);
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
        crate::solver::model::flux_module::FluxModuleSpec::Scheme {
            scheme,
            reconstruction,
            ..
        } => {
            let kernel = crate::solver::model::flux_schemes::lower_flux_scheme(
                scheme,
                &model.system,
                *reconstruction,
            )
            .map_err(|e| format!("flux scheme lowering failed: {e}"))?;
            Ok(cfd2_codegen::solver::codegen::flux_module::generate_flux_module_wgsl(
                &model.state_layout,
                &flux_layout,
                flux_stride,
                &prims,
                &kernel,
            ))
        }
    }
}

fn generate_generic_coupled_assembly_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let discrete = cfd2_codegen::solver::codegen::lower_system(&model.system, schemes)
        .map_err(|e| e.to_string())?;
    let needs_gradients = crate::solver::ir::expand_schemes(&model.system, schemes)
        .map(|e| e.needs_gradients())
        .unwrap_or(false);
    let flux_stride = model.gpu.flux.map(|f| f.stride).unwrap_or(0);
    Ok(cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_wgsl(
        &discrete,
        &model.state_layout,
        flux_stride,
        needs_gradients,
    ))
}

fn generate_generic_coupled_update_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<String, String> {
    let discrete = cfd2_codegen::solver::codegen::lower_system(&model.system, schemes)
        .map_err(|e| e.to_string())?;
    let prims = model
        .primitives
        .ordered()
        .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
    let apply_relaxation = match model.method().map_err(|e| e.to_string())? {
        crate::solver::model::method::MethodSpec::Coupled(caps) => caps.apply_relaxation_in_update,
    };
    Ok(cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_update_wgsl(
        &discrete,
        &model.state_layout,
        &prims,
        apply_relaxation,
    ))
}

fn kernel_generator_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    kernel_id: KernelId,
) -> Option<ModelKernelWgslGenerator> {
    for module in &model.modules {
        let module: &dyn crate::solver::model::module::ModelModule = module;
        if let Some(spec) = module.kernel_generators().iter().find(|s| s.id == kernel_id) {
            return Some(spec.generator);
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
        return gen(model, schemes);
    }

    Err(format!(
        "KernelId '{}' is not a build-time generated per-model kernel",
        kernel_id.as_str()
    ))
}

pub fn generate_shared_kernel_wgsl_by_id(kernel_id: KernelId) -> Result<String, String> {
    if kernel_id != KernelId::GENERIC_COUPLED_APPLY {
        return Err(format!(
            "KernelId '{}' is not a build-time generated shared kernel",
            kernel_id.as_str()
        ));
    }
    Ok(
        cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_apply_wgsl(),
    )
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
    let base_dir = base_dir.as_ref();
    let mut outputs = Vec::new();

    // Shared kernel(s) are emitted once and keyed by kernel id only.
    let kernel_id = KernelId::GENERIC_COUPLED_APPLY;
    let filename = kernel_output_name_for_model("", kernel_id)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    let wgsl = generate_shared_kernel_wgsl_by_id(kernel_id)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    let path = cfd2_codegen::compiler::write_generated_wgsl(
        base_dir, filename, &wgsl,
    )?;
    outputs.push((kernel_id, path));

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

        let has_generator = kernel_generator_for_model_by_id(model, spec.id).is_some();
        if !has_generator {
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
            }],
            generators: vec![ModelKernelGeneratorSpec {
                id: contract_id,
                generator: contract_kernel_generator,
            }],
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
    use crate::solver::model::backend::ast::vol_vector;
    use crate::solver::model::backend::StateLayout;
    use crate::solver::model::flux_module::{FluxModuleGradientsSpec, FluxSchemeSpec};
    use crate::solver::scheme::Scheme;
    use crate::solver::units::si;
    use std::collections::HashSet;

    #[test]
    fn contract_muscl_reconstruction_changes_generated_flux_module_wgsl() {
        // This is intentionally a "contract"-style test: it ensures that
        // FluxReconstructionSpec::Muscl affects generated WGSL (and is not silently ignored).
        //
        // Shipped models remain FirstOrder by default; we only construct a MUSCL variant here.
        let schemes = crate::solver::ir::SchemeRegistry::new(Scheme::Upwind);

        let model_first_order = crate::solver::model::compressible_model();
        let wgsl_first_order = generate_kernel_wgsl_for_model_by_id(
            &model_first_order,
            &schemes,
            KernelId::FLUX_MODULE,
        )
        .expect("failed to generate FirstOrder flux_module WGSL");

        let mut model_muscl = crate::solver::model::compressible_model();

        // Enable the gradients stage + add the gradient fields required by MUSCL validation.
        let fields = crate::solver::model::CompressibleFields::new();
        let mut state_fields =
            vec![fields.rho, fields.rho_u, fields.rho_e, fields.p, fields.t, fields.u];
        state_fields.push(vol_vector("grad_rho", si::DENSITY / si::LENGTH));
        state_fields.push(vol_vector(
            "grad_rho_u_x",
            si::MOMENTUM_DENSITY / si::LENGTH,
        ));
        state_fields.push(vol_vector(
            "grad_rho_u_y",
            si::MOMENTUM_DENSITY / si::LENGTH,
        ));
        state_fields.push(vol_vector("grad_rho_e", si::ENERGY_DENSITY / si::LENGTH));
        state_fields.push(vol_vector("grad_p", si::PRESSURE_GRADIENT));
        model_muscl.state_layout = StateLayout::new(state_fields);

        let flux_spec = crate::solver::model::FluxModuleSpec::Scheme {
            gradients: Some(FluxModuleGradientsSpec::FromStateLayout),
            scheme: FluxSchemeSpec::EulerCentralUpwind,
            reconstruction: crate::solver::ir::FluxReconstructionSpec::Muscl {
                limiter: crate::solver::ir::LimiterSpec::VanLeer,
            },
        };
        let flux_module = flux_module_module(flux_spec).expect("failed to build MUSCL flux_module");

        let mut replaced = false;
        for module in &mut model_muscl.modules {
            if module.name == "flux_module" {
                *module = flux_module.clone();
                replaced = true;
            }
        }
        assert!(replaced, "compressible_model must include a flux_module module");

        // Re-run manifest validation + derive GPU spec after changing modules/state layout.
        model_muscl = model_muscl.with_derived_gpu();

        let wgsl_muscl =
            generate_kernel_wgsl_for_model_by_id(&model_muscl, &schemes, KernelId::FLUX_MODULE)
                .expect("failed to generate MUSCL flux_module WGSL");

        assert_ne!(
            wgsl_first_order, wgsl_muscl,
            "FluxReconstructionSpec::Muscl must change flux_module WGSL"
        );

        // VanLeer limiter uses an epsilon literal in the IR; ensure it reaches WGSL.
        let has_vanleer_epsilon = wgsl_muscl.contains("1e-8") || wgsl_muscl.contains("0.00000001");
        assert!(
            has_vanleer_epsilon,
            "MUSCL(VanLeer) WGSL should contain the VanLeer epsilon literal"
        );
        assert!(
            !wgsl_first_order.contains("1e-8") && !wgsl_first_order.contains("0.00000001"),
            "FirstOrder WGSL should not contain the VanLeer epsilon literal"
        );

        // MUSCL reconstruction relies on CellToFace{Neighbor}, which lowers to face_center - neigh_cell.
        assert!(
            wgsl_muscl.contains("face_center_vec - c_neigh_cell_vec"),
            "MUSCL WGSL should contain neighbor CellToFace geometry"
        );
        assert!(
            !wgsl_first_order.contains("face_center_vec - c_neigh_cell_vec"),
            "FirstOrder WGSL should not contain neighbor CellToFace geometry"
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
}
