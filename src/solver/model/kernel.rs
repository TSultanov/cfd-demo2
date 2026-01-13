#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    /// Flux-module gradients stage (optional).
    ///
    /// Some flux modules (e.g. KT) require precomputed gradients of the coupled unknowns.
    FluxModuleGradients,

    /// Flux computation stage (optional).
    ///
    /// This is a model-selected numerical module (KT, Rhie–Chow, etc.) and is emitted
    /// as a per-model kernel in the registry keyed by `(model_id, KernelId)`.
    FluxModule,

    GenericCoupledAssembly,
    GenericCoupledApply,
    GenericCoupledUpdate,
}

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
    pub const SCHUR_PRECOND_CORRECT_VELOCITY: KernelId =
        KernelId("schur_precond/correct_velocity");

    pub const SCHUR_GENERIC_PRECOND_PREDICT_AND_FORM: KernelId =
        KernelId("schur_precond_generic/predict_and_form_schur");
    pub const SCHUR_GENERIC_PRECOND_RELAX_PRESSURE: KernelId =
        KernelId("schur_precond_generic/relax_pressure");
    pub const SCHUR_GENERIC_PRECOND_CORRECT_VELOCITY: KernelId =
        KernelId("schur_precond_generic/correct_velocity");

    pub const GENERIC_COUPLED_SCHUR_SETUP_BUILD_DIAG_AND_PRESSURE: KernelId =
        KernelId("generic_coupled_schur_setup/build_diag_and_pressure");

    pub const PRECONDITIONER_BUILD_SCHUR_RHS: KernelId =
        KernelId("preconditioner/build_schur_rhs");
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

impl From<KernelKind> for KernelId {
    fn from(kind: KernelKind) -> Self {
        match kind {
            KernelKind::FluxModuleGradients => KernelId::FLUX_MODULE_GRADIENTS,
            KernelKind::FluxModule => KernelId::FLUX_MODULE,

            KernelKind::GenericCoupledAssembly => KernelId::GENERIC_COUPLED_ASSEMBLY,
            KernelKind::GenericCoupledApply => KernelId::GENERIC_COUPLED_APPLY,
            KernelKind::GenericCoupledUpdate => KernelId::GENERIC_COUPLED_UPDATE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    kernels: Vec<KernelKind>,
}

impl KernelPlan {
    pub fn new(kernels: Vec<KernelKind>) -> Self {
        Self { kernels }
    }

    pub fn kernels(&self) -> &[KernelKind] {
        &self.kernels
    }

    pub fn contains(&self, kind: KernelKind) -> bool {
        self.kernels.contains(&kind)
    }
}

pub fn derive_kernel_plan(system: &crate::solver::model::backend::EquationSystem) -> KernelPlan {
    // Back-compat shim: without a `ModelSpec` we cannot select a method.
    // Prefer the generic coupled path as a conservative default.
    let _ = system;
    KernelPlan::new(vec![
        KernelKind::GenericCoupledAssembly,
        KernelKind::GenericCoupledUpdate,
    ])
}

/// Derive a kernel execution ordering in terms of stable `KernelId`s.
///
/// This is the model-structure-level kernel plan used by the unified solver.
///
/// Notes:
/// - This returns only *which* kernels are required and their order.
/// - Phase assignment and dispatch kind are decided by the solver recipe.
/// - `KernelKind` is retained as a legacy/debug bridge.
pub fn derive_kernel_ids(system: &crate::solver::model::backend::EquationSystem) -> Vec<KernelId> {
    // Back-compat shim: without a `ModelSpec` we cannot select a method.
    // Prefer the generic coupled path as a conservative default.
    let _ = system;
    vec![
        KernelId::GENERIC_COUPLED_ASSEMBLY,
        KernelId::GENERIC_COUPLED_UPDATE,
    ]
}

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

/// Model-driven kernel plan: selected from `ModelSpec.method`.
pub fn derive_kernel_plan_for_model(model: &crate::solver::model::ModelSpec) -> KernelPlan {
    use crate::solver::model::MethodSpec;

    match model.method {
        MethodSpec::CoupledIncompressible => {
            let mut kernels = Vec::new();
            if let Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel {
                gradients,
                ..
            }) = &model.flux_module
            {
                if gradients.is_some() {
                    kernels.push(KernelKind::FluxModuleGradients);
                }
                kernels.push(KernelKind::FluxModule);
            }
            kernels.push(KernelKind::GenericCoupledAssembly);
            kernels.push(KernelKind::GenericCoupledUpdate);
            KernelPlan::new(kernels)
        }
        MethodSpec::GenericCoupled => {
            let mut kernels = Vec::new();
            // Optional flux module stage.
            //
            // Emit a generic flux-module stage, plus an optional gradients stage for modules
            // that require it.
            if let Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel {
                gradients,
                ..
            }) = &model.flux_module
            {
                if gradients.is_some() {
                    kernels.push(KernelKind::FluxModuleGradients);
                }
                kernels.push(KernelKind::FluxModule);
            }

            kernels.push(KernelKind::GenericCoupledAssembly);
            kernels.push(KernelKind::GenericCoupledUpdate);
            KernelPlan::new(kernels)
        }
        MethodSpec::GenericCoupledImplicit { .. } => {
            // For implicit outer iterations, update may be executed in the loop body.
            // Keep kernel ordering stable; phases are assigned by the recipe.
            let mut kernels = Vec::new();

            if let Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel {
                gradients,
                ..
            }) = &model.flux_module
            {
                if gradients.is_some() {
                    kernels.push(KernelKind::FluxModuleGradients);
                }
                kernels.push(KernelKind::FluxModule);
            }

            kernels.push(KernelKind::GenericCoupledAssembly);
            kernels.push(KernelKind::GenericCoupledUpdate);
            KernelPlan::new(kernels)
        }
    }
}

pub fn derive_kernel_specs_for_model(
    model: &crate::solver::model::ModelSpec,
) -> Result<Vec<ModelKernelSpec>, String> {
    use crate::solver::model::flux_module::FluxModuleSpec;
    use crate::solver::model::MethodSpec;

    match model.method {
        MethodSpec::CoupledIncompressible => {
            if model.flux_module.is_none() {
                return Err("CoupledIncompressible requires a flux_module in ModelSpec".to_string());
            }

            let mut kernels = Vec::new();
            if let Some(FluxModuleSpec::Kernel { gradients, .. }) = &model.flux_module {
                if gradients.is_some() {
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE_GRADIENTS,
                        phase: KernelPhaseId::Gradients,
                        dispatch: DispatchKindId::Cells,
                    });
                }
                kernels.push(ModelKernelSpec {
                    id: KernelId::FLUX_MODULE,
                    phase: KernelPhaseId::FluxComputation,
                    dispatch: DispatchKindId::Faces,
                });
            }
            kernels.push(ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
            });
            kernels.push(ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
            });

            Ok(kernels)
        }
        MethodSpec::GenericCoupled => {
            let mut kernels = Vec::new();

            match &model.flux_module {
                Some(FluxModuleSpec::Kernel { gradients, .. }) => {
                    if gradients.is_some() {
                        kernels.push(ModelKernelSpec {
                            id: KernelId::FLUX_MODULE_GRADIENTS,
                            phase: KernelPhaseId::Gradients,
                            dispatch: DispatchKindId::Cells,
                        });
                    }
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE,
                        phase: KernelPhaseId::FluxComputation,
                        dispatch: DispatchKindId::Faces,
                    });
                }
                None => {}
            }

            kernels.push(ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
            });
            kernels.push(ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
            });

            Ok(kernels)
        }
        MethodSpec::GenericCoupledImplicit { .. } => {
            let mut kernels = Vec::new();

            match &model.flux_module {
                Some(FluxModuleSpec::Kernel { gradients, .. }) => {
                    if gradients.is_some() {
                        kernels.push(ModelKernelSpec {
                            id: KernelId::FLUX_MODULE_GRADIENTS,
                            phase: KernelPhaseId::Gradients,
                            dispatch: DispatchKindId::Cells,
                        });
                    }
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE,
                        phase: KernelPhaseId::FluxComputation,
                        dispatch: DispatchKindId::Faces,
                    });
                }
                None => {}
            }

            kernels.push(ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
            });

            // For implicit outer iterations, the update kernel is executed in the "apply" stage.
            kernels.push(ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                phase: KernelPhaseId::Apply,
                dispatch: DispatchKindId::Cells,
            });

            Ok(kernels)
        }
    }
}

pub fn kernel_output_name(model_id: &str, kind: KernelKind) -> String {
    match kind {
        KernelKind::FluxModuleGradients => format!("flux_module_gradients_{model_id}.wgsl"),
        KernelKind::FluxModule => format!("flux_module_{model_id}.wgsl"),

        KernelKind::GenericCoupledAssembly => format!("generic_coupled_assembly_{model_id}.wgsl"),
        KernelKind::GenericCoupledApply => "generic_coupled_apply.wgsl".to_string(),
        KernelKind::GenericCoupledUpdate => format!("generic_coupled_update_{model_id}.wgsl"),
    }
}

pub fn generate_kernel_wgsl_for_model(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kind: KernelKind,
) -> Result<String, String> {
    use cfd2_codegen::solver::codegen::{lower_system, DiscreteSystem};

    let discrete: DiscreteSystem = lower_system(&model.system, schemes).map_err(|e| e.to_string())?;

    let wgsl = match kind {
        KernelKind::FluxModuleGradients => match &model.flux_module {
            Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel {
                gradients: Some(_),
                ..
            }) => {
                let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
                cfd2_codegen::solver::codegen::generate_flux_module_gradients_wgsl(
                    &model.state_layout,
                    &flux_layout,
                )?
            }
            _ => {
                return Err(
                    "FluxModuleGradients requested but model has no gradients stage".to_string(),
                );
            }
        },
        KernelKind::FluxModule => match &model.flux_module {
            Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel { kernel, .. }) => {
                let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
                let flux_stride = model.gpu.flux.map(|f| f.stride).unwrap_or(0);
                let prims = model
                    .primitives
                    .ordered()
                    .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
                cfd2_codegen::solver::codegen::flux_module::generate_flux_module_wgsl(
                    &model.state_layout,
                    &flux_layout,
                    flux_stride,
                    &prims,
                    kernel,
                )
            }
            None => {
                return Err("FluxModule requested but model has no flux module".to_string());
            }
        },

        KernelKind::GenericCoupledAssembly => {
            let needs_gradients = crate::solver::ir::expand_schemes(&model.system, schemes)
                .map(|e| e.needs_gradients())
                .unwrap_or(false);
            let flux_stride = model.gpu.flux.map(|f| f.stride).unwrap_or(0);
            cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_wgsl(
                &discrete,
                &model.state_layout,
                flux_stride,
                needs_gradients,
            )
        }
        KernelKind::GenericCoupledApply => {
            cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_apply_wgsl()
        }
        KernelKind::GenericCoupledUpdate => {
            let prims = model
                .primitives
                .ordered()
                .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
            cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_update_wgsl(
                &discrete,
                &model.state_layout,
                &prims,
            )
        }
    };

    Ok(wgsl)
}

pub fn emit_model_kernels_wgsl(
    base_dir: impl AsRef<std::path::Path>,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut outputs = Vec::new();
    let plan = model.kernel_plan();
    for kind in plan.kernels() {
        outputs.push(emit_model_kernel_wgsl(&base_dir, model, schemes, *kind)?);
    }
    Ok(outputs)
}

pub fn emit_model_kernel_wgsl(
    base_dir: impl AsRef<std::path::Path>,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kind: KernelKind,
) -> std::io::Result<std::path::PathBuf> {
    let base_dir = base_dir.as_ref();
    let filename = kernel_output_name(model.id, kind);
    let wgsl = generate_kernel_wgsl_for_model(model, schemes, kind)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    cfd2_codegen::compiler::write_generated_wgsl(base_dir, filename, &wgsl)
}

// (intentionally no additional kernel-analysis helpers here; kernel selection is recipe-driven)
