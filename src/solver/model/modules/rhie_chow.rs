use crate::solver::model::kernel::{KernelConditionId, ModelKernelGeneratorSpec, ModelKernelSpec};
use crate::solver::model::module::{KernelBundleModule, ModuleInvariant, ModuleManifest};
use crate::solver::model::KernelId;
use crate::solver::model::kernel::{DispatchKindId, KernelPhaseId};

use cfd2_codegen::solver::codegen::KernelWgsl;

mod wgsl {
    include!("rhie_chow_wgsl.rs");
}

pub fn rhie_chow_aux_module(
    dp_field: &'static str,
    require_vector2_momentum: bool,
    require_pressure_gradient: bool,
) -> KernelBundleModule {
    let kernel_dp_init = KernelId("dp_init");
    let kernel_dp_update_from_diag = KernelId("dp_update_from_diag");
    let kernel_store_grad_p = KernelId("rhie_chow/store_grad_p");
    let kernel_grad_p_update = KernelId("rhie_chow/grad_p_update");
    let kernel_rhie_chow_correct_velocity_delta = KernelId("rhie_chow/correct_velocity_delta");

    let kernels = vec![
        ModelKernelSpec {
            id: kernel_dp_init,
            phase: KernelPhaseId::Preparation,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        ModelKernelSpec {
            id: kernel_dp_update_from_diag,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        // Snapshot the pressure gradient before it is recomputed so velocity correction can use
        // the change in pressure gradient within the same nonlinear iteration.
        ModelKernelSpec {
            id: kernel_store_grad_p,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        // Recompute `grad(p)` after the pressure update so velocity correction can use the
        // change in pressure gradient within the same nonlinear iteration.
        ModelKernelSpec {
            id: kernel_grad_p_update,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        ModelKernelSpec {
            id: kernel_rhie_chow_correct_velocity_delta,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
    ];

    let generators = vec![
        ModelKernelGeneratorSpec::new(kernel_dp_init, move |model, _schemes| {
            generate_dp_init_kernel_wgsl(model, dp_field)
        }),
        ModelKernelGeneratorSpec::new(kernel_dp_update_from_diag, move |model, _schemes| {
            generate_dp_update_from_diag_kernel_wgsl(model, dp_field)
        }),
        ModelKernelGeneratorSpec::new(kernel_store_grad_p, move |model, _schemes| {
            generate_rhie_chow_store_grad_p_kernel_wgsl(model, dp_field)
        }),
        ModelKernelGeneratorSpec::new(kernel_grad_p_update, move |model, _schemes| {
            generate_rhie_chow_grad_p_update_kernel_wgsl(model)
        }),
        ModelKernelGeneratorSpec::new(kernel_rhie_chow_correct_velocity_delta, move |model, _schemes| {
            generate_rhie_chow_correct_velocity_delta_kernel_wgsl(model, dp_field)
        }),
    ];

    KernelBundleModule {
        name: "rhie_chow_aux",
        kernels,
        generators,
        manifest: ModuleManifest {
            invariants: vec![
                ModuleInvariant::RequireUniqueMomentumPressureCouplingReferencingDp {
                    dp_field,
                    require_vector2_momentum,
                    require_pressure_gradient,
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn generate_dp_init_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::backend::FieldKind;

    let stride = model.state_layout.stride();
    let Some(d_p) = model.state_layout.field(dp_field) else {
        return Err(format!(
            "dp_init requested but model has no '{dp_field}' field in state layout"
        ));
    };
    if d_p.kind() != FieldKind::Scalar {
        return Err(format!("dp_init requires '{dp_field}' to be a scalar field"));
    }
    let d_p_offset = d_p.offset();
    Ok(wgsl::generate_dp_init_wgsl(stride, d_p_offset))
}

fn generate_dp_update_from_diag_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::backend::FieldKind;

    let stride = model.state_layout.stride();
    let Some(d_p) = model.state_layout.field(dp_field) else {
        return Err(format!(
            "dp_update_from_diag requested but model has no '{dp_field}' field in state layout"
        ));
    };
    if d_p.kind() != FieldKind::Scalar {
        return Err(format!("dp_update_from_diag requires '{dp_field}' to be a scalar field"));
    }
    let d_p_offset = d_p.offset();

    let coupling = crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp(
        model,
        dp_field,
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

    wgsl::generate_dp_update_from_diag_wgsl(
        stride,
        d_p_offset,
        model.system.unknowns_per_cell(),
        &u_indices,
    )
}

fn generate_rhie_chow_grad_p_update_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
) -> Result<KernelWgsl, String> {
    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    crate::solver::model::modules::flux_module::generate_flux_module_gradients_wgsl(
        &model.state_layout,
        &flux_layout,
    )
}

fn generate_rhie_chow_store_grad_p_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
) -> Result<KernelWgsl, String> {
    let stride = model.state_layout.stride();

    let coupling = crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp(
        model,
        dp_field,
    )
    .map_err(|e| format!("rhie_chow/store_grad_p {e}"))?;
    let pressure = coupling.pressure;

    let grad_name = format!("grad_{}", pressure.name());
    let grad_old_name = format!("grad_{}_old", pressure.name());

    let grad_p_x = model
        .state_layout
        .component_offset(&grad_name, 0)
        .ok_or_else(|| {
            format!(
                "rhie_chow/store_grad_p requires '{}[0]' in state layout",
                grad_name
            )
        })?;
    let grad_p_y = model
        .state_layout
        .component_offset(&grad_name, 1)
        .ok_or_else(|| {
            format!(
                "rhie_chow/store_grad_p requires '{}[1]' in state layout",
                grad_name
            )
        })?;
    let grad_old_x = model
        .state_layout
        .component_offset(&grad_old_name, 0)
        .ok_or_else(|| {
            format!(
                "rhie_chow/store_grad_p requires '{}[0]' in state layout",
                grad_old_name
            )
        })?;
    let grad_old_y = model
        .state_layout
        .component_offset(&grad_old_name, 1)
        .ok_or_else(|| {
            format!(
                "rhie_chow/store_grad_p requires '{}[1]' in state layout",
                grad_old_name
            )
        })?;

    Ok(wgsl::generate_rhie_chow_store_grad_p_wgsl(
        stride,
        grad_p_x,
        grad_p_y,
        grad_old_x,
        grad_old_y,
    ))
}

fn generate_rhie_chow_correct_velocity_delta_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::backend::FieldKind;

    let stride = model.state_layout.stride();
    let d_p = model
        .state_layout
        .offset_for(dp_field)
        .ok_or_else(|| {
            format!("rhie_chow/correct_velocity_delta requires '{dp_field}' in state layout")
        })?;

    let coupling = crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp(
        model,
        dp_field,
    )
    .map_err(|e| format!("rhie_chow/correct_velocity_delta {e}"))?;
    let momentum = coupling.momentum;
    let pressure = coupling.pressure;

    if momentum.kind() != FieldKind::Vector2 {
        return Err(
            "rhie_chow/correct_velocity_delta currently supports only Vector2 momentum fields"
                .to_string(),
        );
    }

    let u_x = model
        .state_layout
        .component_offset(momentum.name(), 0)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity_delta requires '{}[0]' in state layout",
                momentum.name()
            )
        })?;
    let u_y = model
        .state_layout
        .component_offset(momentum.name(), 1)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity_delta requires '{}[1]' in state layout",
                momentum.name()
            )
        })?;

    let grad_name = format!("grad_{}", pressure.name());
    let grad_old_name = format!("grad_{}_old", pressure.name());

    let grad_p_x = model
        .state_layout
        .component_offset(&grad_name, 0)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity_delta requires '{}[0]' in state layout",
                grad_name
            )
        })?;
    let grad_p_y = model
        .state_layout
        .component_offset(&grad_name, 1)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity_delta requires '{}[1]' in state layout",
                grad_name
            )
        })?;

    let grad_old_x = model
        .state_layout
        .component_offset(&grad_old_name, 0)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity_delta requires '{}[0]' in state layout",
                grad_old_name
            )
        })?;
    let grad_old_y = model
        .state_layout
        .component_offset(&grad_old_name, 1)
        .ok_or_else(|| {
            format!(
                "rhie_chow/correct_velocity_delta requires '{}[1]' in state layout",
                grad_old_name
            )
        })?;

    Ok(wgsl::generate_rhie_chow_correct_velocity_delta_wgsl(
        stride,
        u_x,
        u_y,
        d_p,
        grad_p_x,
        grad_p_y,
        grad_old_x,
        grad_old_y,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{fvm, surface_scalar, vol_scalar, vol_vector, Coefficient, EquationSystem};
    use crate::solver::model::backend::state_layout::StateLayout;
    use crate::solver::model::{BoundarySpec, PrimitiveDerivations};
    use crate::solver::units::si;

    #[test]
    fn contract_rhie_chow_aux_kernel_generators_honor_dp_field_name() {
        let u = vol_vector("U", si::VELOCITY);
        let p = vol_scalar("p", si::PRESSURE);
        let phi = surface_scalar("phi", si::MASS_FLUX);
        let mu = vol_scalar("mu", si::DYNAMIC_VISCOSITY);
        let rho = vol_scalar("rho", si::DENSITY);
        let dp_custom = vol_scalar("dp_custom", si::D_P);
        let grad_p = vol_vector("grad_p", si::PRESSURE_GRADIENT);
        let grad_p_old = vol_vector("grad_p_old", si::PRESSURE_GRADIENT);

        let momentum = (fvm::ddt_coeff(
            Coefficient::field(rho).expect("rho must be scalar"),
            u,
        ) + fvm::div(phi, u)
            + fvm::laplacian(
                Coefficient::field(mu).expect("mu must be scalar"),
                u,
            )
            + fvm::grad(p))
        .eqn(u);

        let pressure = (fvm::laplacian(
            Coefficient::product(
                Coefficient::field(rho).expect("rho must be scalar"),
                Coefficient::field(dp_custom).expect("dp_custom must be scalar"),
            )
            .expect("pressure coefficient must be scalar"),
            p,
        ) + fvm::div_flux(phi, p))
        .eqn(p);

        let mut system = EquationSystem::new();
        system.add_equation(momentum);
        system.add_equation(pressure);

        let layout = StateLayout::new(vec![u, p, dp_custom, grad_p, grad_p_old]);

        let model = crate::solver::model::ModelSpec {
            id: "rhie_chow_dp_custom_test",
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),
            modules: vec![rhie_chow_aux_module("dp_custom", true, true)],
            linear_solver: None,
            primitives: PrimitiveDerivations::identity(),
        };

        let schemes = crate::solver::ir::SchemeRegistry::default();
        for kernel_id in [
            KernelId("dp_init"),
            KernelId("dp_update_from_diag"),
            KernelId("rhie_chow/store_grad_p"),
            KernelId("rhie_chow/grad_p_update"),
            KernelId("rhie_chow/correct_velocity_delta"),
        ] {
            crate::solver::model::kernel::generate_kernel_wgsl_for_model_by_id(&model, &schemes, kernel_id)
                .unwrap_or_else(|e| panic!("generator missing or failed for {}: {e}", kernel_id.as_str()));
        }
    }
}
