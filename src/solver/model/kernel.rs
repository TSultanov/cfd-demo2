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
        MethodSpec::CoupledIncompressible => KernelPlan::new(vec![
            KernelKind::FluxModule,
            KernelKind::GenericCoupledAssembly,
            KernelKind::GenericCoupledUpdate,
        ]),
        MethodSpec::GenericCoupled => {
            let mut kernels = Vec::new();
            // Optional flux module stage.
            //
            // Emit a generic flux-module stage, plus an optional gradients stage for modules
            // that require it (e.g. KT).
            if matches!(model.flux_module, Some(crate::solver::model::flux_module::FluxModuleSpec::KurganovTadmor { .. })) {
                kernels.push(KernelKind::FluxModuleGradients);
                kernels.push(KernelKind::FluxModule);
            } else if matches!(model.flux_module, Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow { .. })) {
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

            if matches!(model.flux_module, Some(crate::solver::model::flux_module::FluxModuleSpec::KurganovTadmor { .. })) {
                kernels.push(KernelKind::FluxModuleGradients);
                kernels.push(KernelKind::FluxModule);
            } else if matches!(model.flux_module, Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow { .. })) {
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
            if !matches!(model.flux_module, Some(FluxModuleSpec::RhieChow { .. })) {
                return Err(
                    "CoupledIncompressible requires flux_module = RhieChow in ModelSpec".to_string(),
                );
            }

            Ok(vec![
                ModelKernelSpec {
                    id: KernelId::FLUX_MODULE,
                    phase: KernelPhaseId::FluxComputation,
                    dispatch: DispatchKindId::Faces,
                },
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
            ])
        }
        MethodSpec::GenericCoupled => {
            let mut kernels = Vec::new();

            match &model.flux_module {
                Some(FluxModuleSpec::RhieChow { .. }) => {
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE,
                        phase: KernelPhaseId::FluxComputation,
                        dispatch: DispatchKindId::Faces,
                    });
                }
                Some(FluxModuleSpec::KurganovTadmor { .. }) => {
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE_GRADIENTS,
                        phase: KernelPhaseId::Gradients,
                        dispatch: DispatchKindId::Cells,
                    });
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE,
                        phase: KernelPhaseId::FluxComputation,
                        dispatch: DispatchKindId::Faces,
                    });
                }
                Some(FluxModuleSpec::Convective { .. }) | None => {}
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
                Some(FluxModuleSpec::RhieChow { .. }) => {
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE,
                        phase: KernelPhaseId::FluxComputation,
                        dispatch: DispatchKindId::Faces,
                    });
                }
                Some(FluxModuleSpec::KurganovTadmor { .. }) => {
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE_GRADIENTS,
                        phase: KernelPhaseId::Gradients,
                        dispatch: DispatchKindId::Cells,
                    });
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE,
                        phase: KernelPhaseId::FluxComputation,
                        dispatch: DispatchKindId::Faces,
                    });
                }
                Some(FluxModuleSpec::Convective { .. }) | None => {}
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

fn collect_coefficient_fields(
    coeff: &crate::solver::model::backend::Coefficient,
    out: &mut Vec<crate::solver::model::backend::FieldRef>,
) {
    use crate::solver::model::backend::Coefficient;
    match coeff {
        Coefficient::Constant { .. } => {}
        Coefficient::Field(field) => out.push(*field),
        Coefficient::Product(lhs, rhs) => {
            collect_coefficient_fields(lhs, out);
            collect_coefficient_fields(rhs, out);
        }
    }
}

pub type KernelCodegenFieldMap = std::collections::BTreeMap<String, String>;

/// Derive kernel-specific codegen parameters from the model math + layout.
///
/// This is intentionally *kernel-agnostic plumbing*: it exposes derived values as a
/// string-keyed map so build-time codegen can remain generic.
///
/// Note: legacy kernels may rely on naming conventions for auxiliary fields. Those
/// conventions are validated against `ModelSpec.state_layout` here.
pub fn derive_kernel_codegen_fields_for_model(
    model: &crate::solver::model::ModelSpec,
    kind: KernelKind,
) -> Result<KernelCodegenFieldMap, String> {
    let mut out = KernelCodegenFieldMap::new();
    match kind {
        KernelKind::FluxModule => {
            if matches!(
                model.flux_module,
                Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow { .. })
            ) {
                derive_rhie_chow_fields(model, &mut out)?;
            }
        }
        KernelKind::FluxModuleGradients
        | KernelKind::GenericCoupledAssembly
        | KernelKind::GenericCoupledApply
        | KernelKind::GenericCoupledUpdate => {}
    }
    Ok(out)
}

fn derive_rhie_chow_fields(
    model: &crate::solver::model::ModelSpec,
    out: &mut KernelCodegenFieldMap,
) -> Result<(), String> {
    use crate::solver::model::backend::{FieldKind, TermOp};

    let req = analyze_kernel_requirements(&model.system);
    let coupling = req.pressure_coupling.ok_or_else(|| {
        "missing unique momentum-pressure coupling required for Rhie–Chow kernels".to_string()
    })?;

    if coupling.momentum.kind() != FieldKind::Vector2 {
        return Err(format!(
            "Rhie–Chow kernels require Vector2 momentum, got {} for '{}'",
            coupling.momentum.kind().as_str(),
            coupling.momentum.name()
        ));
    }

    let pressure_eq = model
        .system
        .equations()
        .iter()
        .find(|eq| *eq.target() == coupling.pressure)
        .ok_or_else(|| {
            format!(
                "missing pressure equation for inferred pressure field '{}'",
                coupling.pressure.name()
            )
        })?;

    let pressure_laplacian = pressure_eq
        .terms()
        .iter()
        .find(|t| t.op == TermOp::Laplacian)
        .ok_or_else(|| {
            format!(
                "pressure equation for '{}' must include a laplacian term",
                coupling.pressure.name()
            )
        })?;

    let mut coeff_fields = Vec::new();
    if let Some(coeff) = &pressure_laplacian.coeff {
        collect_coefficient_fields(coeff, &mut coeff_fields);
    } else {
        return Err(format!(
            "pressure laplacian coefficient for '{}' is missing",
            coupling.pressure.name()
        ));
    }

    let layout_coeff_fields: Vec<_> = coeff_fields
        .into_iter()
        .filter(|f| model.state_layout.field(f.name()).is_some())
        .collect();
    let d_p = match layout_coeff_fields.as_slice() {
        [only] => only.name().to_string(),
        [] => {
            return Err(format!(
                "pressure laplacian coefficient for '{}' does not reference any state-layout scalar fields",
                coupling.pressure.name()
            ));
        }
        many => {
            return Err(format!(
                "pressure laplacian coefficient for '{}' references multiple state-layout fields; cannot derive unique d_p: [{}]",
                coupling.pressure.name(),
                many.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
            ));
        }
    };

    let grad_p = format!("grad_{}", coupling.pressure.name());
    if model.state_layout.field(&grad_p).is_none() {
        return Err(format!(
            "state layout missing required pressure-gradient field '{}' (expected by Rhie–Chow)",
            grad_p
        ));
    }

    out.insert("momentum".to_string(), coupling.momentum.name().to_string());
    out.insert("pressure".to_string(), coupling.pressure.name().to_string());
    out.insert("d_p".to_string(), d_p);
    out.insert("grad_p".to_string(), grad_p);

    Ok(())
}

fn required_codegen_field<'a>(map: &'a KernelCodegenFieldMap, key: &str) -> &'a str {
    map.get(key)
        .map(|s| s.as_str())
        .unwrap_or_else(|| panic!("missing required derived kernel codegen field '{key}'"))
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
            Some(crate::solver::model::flux_module::FluxModuleSpec::KurganovTadmor { .. }) => {
                cfd2_codegen::solver::codegen::kt_gradients::generate_kt_gradients_wgsl(
                    &model.state_layout,
                )
            }
            _ => {
                return Err("FluxModuleGradients requested but model has no gradients-based flux module".to_string());
            }
        },
        KernelKind::FluxModule => match &model.flux_module {
            Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow { .. }) => {
                let fields = derive_kernel_codegen_fields_for_model(model, KernelKind::FluxModule)?;
                let flux_stride = model.gpu.flux.map(|f| f.stride).unwrap_or(0);
                cfd2_codegen::solver::codegen::flux_rhie_chow::generate_flux_rhie_chow_wgsl(
                    &discrete,
                    &model.state_layout,
                    flux_stride,
                    required_codegen_field(&fields, "momentum"),
                    required_codegen_field(&fields, "pressure"),
                    required_codegen_field(&fields, "d_p"),
                    required_codegen_field(&fields, "grad_p"),
                )
            }
            Some(crate::solver::model::flux_module::FluxModuleSpec::KurganovTadmor { .. }) => {
                let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
                let eos = ir_eos_from_model(model.eos);
                cfd2_codegen::solver::codegen::flux_kt::generate_flux_kt_wgsl(
                    &model.state_layout,
                    &flux_layout,
                    &eos,
                )
            }
            Some(crate::solver::model::flux_module::FluxModuleSpec::Convective { .. }) | None => {
                return Err("FluxModule requested but model has no flux module requiring a kernel".to_string());
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

fn ir_eos_from_model(eos: crate::solver::model::EosSpec) -> crate::solver::ir::EosSpec {
    match eos {
        crate::solver::model::EosSpec::IdealGas { gamma } => crate::solver::ir::EosSpec::IdealGas { gamma },
        crate::solver::model::EosSpec::Constant => crate::solver::ir::EosSpec::Constant,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PressureCoupling {
    momentum: crate::solver::model::backend::FieldRef,
    pressure: crate::solver::model::backend::FieldRef,
}

#[derive(Debug, Clone, Default)]
struct KernelRequirements {
    pressure_coupling: Option<PressureCoupling>,
}

fn analyze_kernel_requirements(
    system: &crate::solver::model::backend::EquationSystem,
) -> KernelRequirements {
    use crate::solver::model::backend::{FieldKind, FieldRef, TermOp};
    use std::collections::{HashMap, HashSet};

    let equations = system.equations();

    // Build a lookup for equations by target field.
    let mut eq_by_target: HashMap<FieldRef, usize> = HashMap::new();
    for (idx, eq) in equations.iter().enumerate() {
        eq_by_target.insert(*eq.target(), idx);
    }

    // Detect incompressible momentum+pressure coupling structurally:
    // - find a vector-target equation that takes Grad(p) for some scalar p
    // - require some transport operator (Div or Laplacian) on the momentum equation
    // - require a Laplacian term on the pressure equation targeting p
    let mut candidates: Vec<PressureCoupling> = Vec::new();
    for eq in equations {
        if !matches!(eq.target().kind(), FieldKind::Vector2 | FieldKind::Vector3) {
            continue;
        }

        let momentum = *eq.target();
        let has_transport = eq
            .terms()
            .iter()
            .any(|t| matches!(t.op, TermOp::Div | TermOp::Laplacian));
        if !has_transport {
            continue;
        }

        let mut grad_scalars: HashSet<FieldRef> = HashSet::new();
        for term in eq.terms() {
            if term.op == TermOp::Grad && term.field.kind() == FieldKind::Scalar {
                grad_scalars.insert(term.field);
            }
        }

        for pressure in grad_scalars {
            let Some(&p_eq_idx) = eq_by_target.get(&pressure) else {
                continue;
            };
            let p_eq = &equations[p_eq_idx];
            let p_has_laplacian = p_eq.terms().iter().any(|t| t.op == TermOp::Laplacian);
            if p_has_laplacian {
                candidates.push(PressureCoupling { momentum, pressure });
            }
        }
    }

    KernelRequirements {
        pressure_coupling: (candidates.len() == 1).then(|| candidates[0]),
    }
}

fn synthesize_kernel_plan(req: &KernelRequirements) -> KernelPlan {
    if req.pressure_coupling.is_some() {
        return KernelPlan::new(vec![
            KernelKind::FluxModule,
            KernelKind::GenericCoupledAssembly,
            KernelKind::GenericCoupledUpdate,
        ]);
    }

    KernelPlan::new(vec![
        KernelKind::GenericCoupledAssembly,
        KernelKind::GenericCoupledApply,
        KernelKind::GenericCoupledUpdate,
    ])
}

fn synthesize_kernel_ids(req: &KernelRequirements) -> Vec<KernelId> {
    if req.pressure_coupling.is_some() {
        return vec![
            KernelId::FLUX_MODULE,
            KernelId::GENERIC_COUPLED_ASSEMBLY,
            KernelId::GENERIC_COUPLED_UPDATE,
        ];
    }

    vec![
        KernelId::GENERIC_COUPLED_ASSEMBLY,
        KernelId::GENERIC_COUPLED_APPLY,
        KernelId::GENERIC_COUPLED_UPDATE,
    ]
}
