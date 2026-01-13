/// Stable identifier for a compute kernel.
///
/// This is used by the unified solver orchestration to decouple scheduling and lookup
/// from handwritten enums/matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub &'static str);

impl KernelId {
    pub const DP_INIT: KernelId = KernelId("dp_init");
    pub const DP_UPDATE_FROM_DIAG: KernelId = KernelId("dp_update_from_diag");
    pub const RHIE_CHOW_CORRECT_VELOCITY: KernelId = KernelId("rhie_chow/correct_velocity");

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

/// Build-time-generated WGSL kernel templates.
///
/// Each template is emitted either per-model (when it depends on state/system layout)
/// or once globally (shared infrastructure kernel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneratedKernelTemplate {
    DpInit,
    DpUpdateFromDiag,
    RhieChowCorrectVelocity,
    FluxModuleGradients,
    FluxModule,
    GenericCoupledAssembly,
    GenericCoupledUpdate,

    /// Shared kernel (no model id suffix).
    GenericCoupledApply,
}

impl GeneratedKernelTemplate {
    pub fn kernel_id(self) -> KernelId {
        match self {
            GeneratedKernelTemplate::DpInit => KernelId::DP_INIT,
            GeneratedKernelTemplate::DpUpdateFromDiag => KernelId::DP_UPDATE_FROM_DIAG,
            GeneratedKernelTemplate::RhieChowCorrectVelocity => {
                KernelId::RHIE_CHOW_CORRECT_VELOCITY
            }
            GeneratedKernelTemplate::FluxModuleGradients => KernelId::FLUX_MODULE_GRADIENTS,
            GeneratedKernelTemplate::FluxModule => KernelId::FLUX_MODULE,
            GeneratedKernelTemplate::GenericCoupledAssembly => KernelId::GENERIC_COUPLED_ASSEMBLY,
            GeneratedKernelTemplate::GenericCoupledUpdate => KernelId::GENERIC_COUPLED_UPDATE,
            GeneratedKernelTemplate::GenericCoupledApply => KernelId::GENERIC_COUPLED_APPLY,
        }
    }

    pub fn file_prefix(self) -> &'static str {
        match self {
            GeneratedKernelTemplate::DpInit => "dp_init",
            GeneratedKernelTemplate::DpUpdateFromDiag => "dp_update_from_diag",
            GeneratedKernelTemplate::RhieChowCorrectVelocity => "rhie_chow_correct_velocity",
            GeneratedKernelTemplate::FluxModuleGradients => "flux_module_gradients",
            GeneratedKernelTemplate::FluxModule => "flux_module",
            GeneratedKernelTemplate::GenericCoupledAssembly => "generic_coupled_assembly",
            GeneratedKernelTemplate::GenericCoupledUpdate => "generic_coupled_update",
            GeneratedKernelTemplate::GenericCoupledApply => "generic_coupled_apply",
        }
    }

    pub fn is_per_model(self) -> bool {
        !matches!(self, GeneratedKernelTemplate::GenericCoupledApply)
    }
}

pub fn generated_template_for_kernel_id(kernel_id: KernelId) -> Option<GeneratedKernelTemplate> {
    match kernel_id.as_str() {
        "dp_init" => Some(GeneratedKernelTemplate::DpInit),
        "dp_update_from_diag" => Some(GeneratedKernelTemplate::DpUpdateFromDiag),
        "rhie_chow/correct_velocity" => Some(GeneratedKernelTemplate::RhieChowCorrectVelocity),
        "flux_module_gradients" => Some(GeneratedKernelTemplate::FluxModuleGradients),
        "flux_module" => Some(GeneratedKernelTemplate::FluxModule),
        "generic_coupled_assembly" => Some(GeneratedKernelTemplate::GenericCoupledAssembly),
        "generic_coupled_update" => Some(GeneratedKernelTemplate::GenericCoupledUpdate),
        "generic_coupled_apply" => Some(GeneratedKernelTemplate::GenericCoupledApply),
        _ => None,
    }
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
            if model.state_layout.field("d_p").is_some() {
                kernels.push(ModelKernelSpec {
                    id: KernelId::DP_INIT,
                    phase: KernelPhaseId::Preparation,
                    dispatch: DispatchKindId::Cells,
                });
            }
            if let Some(FluxModuleSpec::Kernel { gradients, .. }) = &model.flux_module {
                if gradients.is_some() {
                    kernels.push(ModelKernelSpec {
                        id: KernelId::FLUX_MODULE_GRADIENTS,
                        phase: KernelPhaseId::Gradients,
                        dispatch: DispatchKindId::Cells,
                    });
                }
                if gradients.is_some() && model.state_layout.field("d_p").is_some() {
                    kernels.push(ModelKernelSpec {
                        id: KernelId::RHIE_CHOW_CORRECT_VELOCITY,
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
            if model.state_layout.field("d_p").is_some() {
                kernels.push(ModelKernelSpec {
                    id: KernelId::DP_UPDATE_FROM_DIAG,
                    phase: KernelPhaseId::Update,
                    dispatch: DispatchKindId::Cells,
                });
            }

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
            if model.state_layout.field("d_p").is_some() {
                kernels.push(ModelKernelSpec {
                    id: KernelId::DP_UPDATE_FROM_DIAG,
                    phase: KernelPhaseId::Update,
                    dispatch: DispatchKindId::Cells,
                });
            }

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
            if model.state_layout.field("d_p").is_some() {
                kernels.push(ModelKernelSpec {
                    id: KernelId::DP_UPDATE_FROM_DIAG,
                    phase: KernelPhaseId::Apply,
                    dispatch: DispatchKindId::Cells,
                });
            }

            Ok(kernels)
        }
    }
}

pub fn kernel_output_name_for_model(model_id: &str, kernel_id: KernelId) -> Result<String, String> {
    let Some(template) = generated_template_for_kernel_id(kernel_id) else {
        return Err(format!(
            "KernelId '{}' is not a build-time generated kernel",
            kernel_id.as_str()
        ));
    };

    if template.is_per_model() {
        Ok(format!("{}_{}.wgsl", template.file_prefix(), model_id))
    } else {
        Ok(format!("{}.wgsl", template.file_prefix()))
    }
}

pub fn generate_kernel_wgsl_for_model_by_id(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
    kernel_id: KernelId,
) -> Result<String, String> {
    use cfd2_codegen::solver::codegen::{lower_system, DiscreteSystem};

    let Some(template) = generated_template_for_kernel_id(kernel_id) else {
        return Err(format!(
            "KernelId '{}' is not a build-time generated kernel",
            kernel_id.as_str()
        ));
    };

    // Some kernels don't require a lowered system, but most do; keep this cheap and local.
    let discrete: Option<DiscreteSystem> = match template {
        GeneratedKernelTemplate::GenericCoupledAssembly
        | GeneratedKernelTemplate::GenericCoupledUpdate => {
            Some(lower_system(&model.system, schemes).map_err(|e| e.to_string())?)
        }
        _ => None,
    };

    let wgsl = match template {
        GeneratedKernelTemplate::DpInit => {
            let stride = model.state_layout.stride();
            let Some(d_p) = model.state_layout.field("d_p") else {
                return Err(
                    "dp_init requested but model has no 'd_p' field in state layout".to_string(),
                );
            };
            if d_p.kind() != crate::solver::model::backend::FieldKind::Scalar {
                return Err("dp_init requires 'd_p' to be a scalar field".to_string());
            }
            let d_p_offset = d_p.offset();
            cfd2_codegen::solver::codegen::generate_dp_init_wgsl(stride, d_p_offset)
        }
        GeneratedKernelTemplate::DpUpdateFromDiag => {
            use crate::solver::model::backend::{Coefficient as BackendCoeff, FieldKind, TermOp};

            fn collect_coeff_fields(
                coeff: &BackendCoeff,
                out: &mut Vec<crate::solver::model::backend::FieldRef>,
            ) {
                match coeff {
                    BackendCoeff::Constant { .. } => {}
                    BackendCoeff::Field(field) => out.push(*field),
                    BackendCoeff::Product(lhs, rhs) => {
                        collect_coeff_fields(lhs, out);
                        collect_coeff_fields(rhs, out);
                    }
                }
            }

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

            let equations = model.system.equations();
            let mut eq_by_target = std::collections::HashMap::new();
            for (idx, eq) in equations.iter().enumerate() {
                eq_by_target.insert(*eq.target(), idx);
            }

            // Infer the velocity-like (vector) equation that couples to a pressure-like scalar
            // whose Laplacian coefficient references `d_p`.
            let mut candidates = Vec::new();
            for eq in equations {
                if !matches!(eq.target().kind(), FieldKind::Vector2 | FieldKind::Vector3) {
                    continue;
                }
                for term in eq.terms() {
                    if term.op != TermOp::Grad || term.field.kind() != FieldKind::Scalar {
                        continue;
                    }
                    let Some(&p_eq_idx) = eq_by_target.get(&term.field) else {
                        continue;
                    };
                    let p_eq = &equations[p_eq_idx];
                    let Some(lap) = p_eq.terms().iter().find(|t| t.op == TermOp::Laplacian) else {
                        continue;
                    };
                    let Some(coeff) = &lap.coeff else {
                        continue;
                    };
                    let mut coeff_fields = Vec::new();
                    collect_coeff_fields(coeff, &mut coeff_fields);
                    if coeff_fields.iter().any(|f| f.name() == "d_p") {
                        candidates.push(*eq.target());
                    }
                }
            }

            let momentum = match candidates.as_slice() {
                [only] => *only,
                [] => {
                    return Err(
                        "dp_update_from_diag requires a unique momentum-pressure coupling referencing 'd_p'"
                            .to_string(),
                    )
                }
                many => {
                    return Err(format!(
                        "dp_update_from_diag requires a unique momentum-pressure coupling referencing 'd_p', found {} candidates: [{}]",
                        many.len(),
                        many.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
                    ))
                }
            };

            let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
            let mut u_indices = Vec::new();
            for component in 0..momentum.kind().component_count() as u32 {
                let Some(offset) = flux_layout.offset_for_field_component(momentum, component)
                else {
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
            )?
        }
        GeneratedKernelTemplate::RhieChowCorrectVelocity => {
            use crate::solver::model::backend::{Coefficient as BackendCoeff, FieldKind, TermOp};

            fn collect_coeff_fields(
                coeff: &BackendCoeff,
                out: &mut Vec<crate::solver::model::backend::FieldRef>,
            ) {
                match coeff {
                    BackendCoeff::Constant { .. } => {}
                    BackendCoeff::Field(field) => out.push(*field),
                    BackendCoeff::Product(lhs, rhs) => {
                        collect_coeff_fields(lhs, out);
                        collect_coeff_fields(rhs, out);
                    }
                }
            }

            let stride = model.state_layout.stride();
            let d_p = model
                .state_layout
                .offset_for("d_p")
                .ok_or_else(|| {
                    "rhie_chow/correct_velocity requires 'd_p' in state layout".to_string()
                })?;

            let equations = model.system.equations();
            let mut eq_by_target = std::collections::HashMap::new();
            for (idx, eq) in equations.iter().enumerate() {
                eq_by_target.insert(*eq.target(), idx);
            }

            let mut candidates = Vec::new();
            for eq in equations {
                if !matches!(eq.target().kind(), FieldKind::Vector2 | FieldKind::Vector3) {
                    continue;
                }
                for term in eq.terms() {
                    if term.op != TermOp::Grad || term.field.kind() != FieldKind::Scalar {
                        continue;
                    }
                    let Some(&p_eq_idx) = eq_by_target.get(&term.field) else {
                        continue;
                    };
                    let p_eq = &model.system.equations()[p_eq_idx];
                    let Some(lap) = p_eq.terms().iter().find(|t| t.op == TermOp::Laplacian) else {
                        continue;
                    };
                    let Some(coeff) = &lap.coeff else {
                        continue;
                    };
                    let mut coeff_fields = Vec::new();
                    collect_coeff_fields(coeff, &mut coeff_fields);
                    if coeff_fields.iter().any(|f| f.name() == "d_p") {
                        candidates.push((*eq.target(), term.field));
                    }
                }
            }

            let (momentum, pressure) = match candidates.as_slice() {
                [(m, p)] => (*m, *p),
                [] => {
                    return Err(
                        "rhie_chow/correct_velocity requires a unique momentum-pressure coupling referencing 'd_p'"
                            .to_string(),
                    )
                }
                many => {
                    return Err(format!(
                        "rhie_chow/correct_velocity requires a unique momentum-pressure coupling referencing 'd_p', found {} candidates",
                        many.len()
                    ))
                }
            };

            if momentum.kind() != FieldKind::Vector2 {
                return Err(
                    "rhie_chow/correct_velocity currently supports only Vector2 momentum fields"
                        .to_string(),
                );
            }

            let u_x = model
                .state_layout
                .component_offset(momentum.name(), 0)
                .ok_or_else(|| {
                    format!(
                        "rhie_chow/correct_velocity requires '{}[0]' in state layout",
                        momentum.name()
                    )
                })?;
            let u_y = model
                .state_layout
                .component_offset(momentum.name(), 1)
                .ok_or_else(|| {
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

            cfd2_codegen::solver::codegen::generate_rhie_chow_correct_velocity_wgsl(
                stride, u_x, u_y, d_p, grad_p_x, grad_p_y,
            )
        }
        GeneratedKernelTemplate::FluxModuleGradients => match &model.flux_module {
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
                    "flux_module_gradients requested but model has no gradients stage".to_string(),
                );
            }
        },
        GeneratedKernelTemplate::FluxModule => match &model.flux_module {
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
                return Err("flux_module requested but model has no flux module".to_string());
            }
        },

        GeneratedKernelTemplate::GenericCoupledAssembly => {
            let Some(discrete) = discrete.as_ref() else {
                return Err("missing lowered system for generic_coupled_assembly".to_string());
            };
            let needs_gradients = crate::solver::ir::expand_schemes(&model.system, schemes)
                .map(|e| e.needs_gradients())
                .unwrap_or(false);
            let flux_stride = model.gpu.flux.map(|f| f.stride).unwrap_or(0);
            cfd2_codegen::solver::codegen::unified_assembly::generate_unified_assembly_wgsl(
                discrete,
                &model.state_layout,
                flux_stride,
                needs_gradients,
            )
        }
        GeneratedKernelTemplate::GenericCoupledUpdate => {
            let Some(discrete) = discrete.as_ref() else {
                return Err("missing lowered system for generic_coupled_update".to_string());
            };
            let prims = model
                .primitives
                .ordered()
                .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
            cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_update_wgsl(
                discrete,
                &model.state_layout,
                &prims,
            )
        }

        GeneratedKernelTemplate::GenericCoupledApply => {
            cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_apply_wgsl()
        }
    };

    Ok(wgsl)
}

pub fn generate_shared_kernel_wgsl_by_id(kernel_id: KernelId) -> Result<String, String> {
    let Some(template) = generated_template_for_kernel_id(kernel_id) else {
        return Err(format!(
            "KernelId '{}' is not a build-time generated kernel",
            kernel_id.as_str()
        ));
    };

    if template.is_per_model() {
        return Err(format!(
            "KernelId '{}' is not a shared (global) kernel",
            kernel_id.as_str()
        ));
    }

    match template {
        GeneratedKernelTemplate::GenericCoupledApply => Ok(
            cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_apply_wgsl(),
        ),
        _ => Err(format!(
            "KernelId '{}' does not have a shared-kernel generator",
            kernel_id.as_str()
        )),
    }
}

pub fn emit_shared_kernels_wgsl(
    base_dir: impl AsRef<std::path::Path>,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    let base_dir = base_dir.as_ref();
    let mut outputs = Vec::new();

    // Shared kernel(s) are emitted once and keyed by kernel id only.
    let kernel_id = KernelId::GENERIC_COUPLED_APPLY;
    let filename = kernel_output_name_for_model("", kernel_id)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    let wgsl = generate_shared_kernel_wgsl_by_id(kernel_id)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    outputs.push(cfd2_codegen::compiler::write_generated_wgsl(
        base_dir, filename, &wgsl,
    )?);

    Ok(outputs)
}

pub fn emit_model_kernels_wgsl(
    base_dir: impl AsRef<std::path::Path>,
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut outputs = Vec::new();

    let specs = derive_kernel_specs_for_model(model)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;

    let mut seen: std::collections::HashSet<KernelId> = std::collections::HashSet::new();
    for spec in specs {
        let Some(template) = generated_template_for_kernel_id(spec.id) else {
            continue;
        };
        if !template.is_per_model() {
            continue;
        }
        if !seen.insert(spec.id) {
            continue;
        }
        outputs.push(emit_model_kernel_wgsl_by_id(
            &base_dir, model, schemes, spec.id,
        )?);
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

// (intentionally no additional kernel-analysis helpers here; kernel selection is recipe-driven)
