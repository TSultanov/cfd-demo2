use crate::solver::model::kernel::{DispatchKindId, KernelPhaseId};
use crate::solver::model::kernel::{KernelConditionId, ModelKernelGeneratorSpec, ModelKernelSpec};
use crate::solver::model::module::{KernelBundleModule, ModuleInvariant};
use crate::solver::model::KernelId;

use cfd2_codegen::solver::codegen::KernelWgsl;

mod wgsl {
    include!("rhie_chow_wgsl.rs");
}

/// Rhie-Chow auxiliary module that manages pressure correction and velocity correction.
///
/// This module provides kernels for:
/// - `dp_init`: Initialize the pressure correction field
/// - `dp_update_from_diag`: Update dp using diagonal coefficients from the linear system
/// - `rhie_chow/store_grad_p`: Store the current pressure gradient for later use
/// - `rhie_chow/grad_p_update`: Recompute the pressure gradient after pressure update
/// - `rhie_chow/correct_velocity_delta`: Apply Rhie-Chow velocity correction
///
/// # Arguments
///
/// * `system` - The equation system (needed to infer momentum-pressure coupling)
/// * `dp_field` - The name of the pressure correction field in the state layout
/// * `require_vector2_momentum` - Whether to require Vector2 momentum field
/// * `require_pressure_gradient` - Whether to require pressure gradient fields
///
/// # Errors
///
/// Returns an error if the momentum-pressure coupling cannot be inferred from the system.
pub fn rhie_chow_aux_module(
    system: &crate::solver::model::backend::ast::EquationSystem,
    dp_field: &'static str,
    require_vector2_momentum: bool,
    require_pressure_gradient: bool,
) -> Result<KernelBundleModule, String> {
    // Infer coupling once at module construction time
    let coupling =
        crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp_system(
            system, dp_field,
        )
        .map_err(|e| format!("rhie_chow_aux_module: {e}"))?;

    let pressure_name = coupling.pressure.name();

    // Precompute derived gradient field names once
    // These are interned/leaked to obtain &'static str for PortManifest
    let grad_p_name: &'static str = Box::leak(format!("grad_{}", pressure_name).into_boxed_str());
    let grad_p_old_name: &'static str =
        Box::leak(format!("grad_{}_old", pressure_name).into_boxed_str());

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

    // Clone coupling data for the generator closures
    let coupling_for_dp_update = coupling;
    let coupling_for_store = coupling;
    let coupling_for_correct = coupling;

    let generators = vec![
        ModelKernelGeneratorSpec::new(kernel_dp_init, move |model, _schemes| {
            generate_dp_init_kernel_wgsl(model, dp_field)
        }),
        ModelKernelGeneratorSpec::new(kernel_dp_update_from_diag, move |model, _schemes| {
            generate_dp_update_from_diag_kernel_wgsl(model, dp_field, coupling_for_dp_update)
        }),
        ModelKernelGeneratorSpec::new(kernel_store_grad_p, move |model, _schemes| {
            generate_rhie_chow_store_grad_p_kernel_wgsl(
                model,
                coupling_for_store,
                grad_p_name,
                grad_p_old_name,
            )
        }),
        ModelKernelGeneratorSpec::new(kernel_grad_p_update, move |model, _schemes| {
            generate_rhie_chow_grad_p_update_kernel_wgsl(model)
        }),
        ModelKernelGeneratorSpec::new(
            kernel_rhie_chow_correct_velocity_delta,
            move |model, _schemes| {
                generate_rhie_chow_correct_velocity_delta_kernel_wgsl(
                    model,
                    dp_field,
                    coupling_for_correct,
                    grad_p_name,
                    grad_p_old_name,
                )
            },
        ),
    ];

    // Build PortManifest with required fields
    use crate::solver::dimensions::{D_P, PressureGradient, UnitDimension};
    use crate::solver::ir::ports::{FieldSpec, PortFieldKind, PortManifest};

    let port_manifest = Some(PortManifest {
        fields: vec![
            // dp field: Scalar with D_P unit
            FieldSpec {
                name: dp_field,
                kind: PortFieldKind::Scalar,
                unit: D_P::UNIT,
            },
            // grad_p field: Vector2 with PRESSURE_GRADIENT unit
            FieldSpec {
                name: grad_p_name,
                kind: PortFieldKind::Vector2,
                unit: PressureGradient::UNIT,
            },
            // grad_p_old field: Vector2 with PRESSURE_GRADIENT unit
            FieldSpec {
                name: grad_p_old_name,
                kind: PortFieldKind::Vector2,
                unit: PressureGradient::UNIT,
            },
            // momentum field: Vector2 with ANY_DIMENSION (dynamic dimension)
            FieldSpec {
                name: coupling.momentum.name(),
                kind: PortFieldKind::Vector2,
                unit: crate::solver::ir::ports::ANY_DIMENSION,
            },
        ],
        ..Default::default()
    });

    Ok(KernelBundleModule {
        name: "rhie_chow_aux",
        kernels,
        generators,
        invariants: vec![
            ModuleInvariant::RequireUniqueMomentumPressureCouplingReferencingDp {
                dp_field,
                require_vector2_momentum,
                require_pressure_gradient,
            },
        ],
        port_manifest,
        ..Default::default()
    })
}

fn generate_dp_init_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::ports::dimensions::D_P;
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    let d_p = registry
        .register_scalar_field::<D_P>(dp_field)
        .map_err(|e| format!("dp_init: {e}"))?;

    let stride = registry.state_layout().stride();
    let d_p_offset = d_p.offset();
    Ok(wgsl::generate_dp_init_wgsl(stride, d_p_offset))
}

fn generate_dp_update_from_diag_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
    coupling: crate::solver::model::invariants::MomentumPressureCoupling,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::ports::dimensions::{AnyDimension, D_P};
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    let d_p = registry
        .register_scalar_field::<D_P>(dp_field)
        .map_err(|e| format!("dp_update_from_diag: {e}"))?;

    let momentum = coupling.momentum;

    // Register momentum field to validate it exists (offset not needed here)
    let _u = registry
        .register_vector2_field::<AnyDimension>(momentum.name())
        .map_err(|e| format!("dp_update_from_diag: {e}"))?;

    let stride = registry.state_layout().stride();
    let d_p_offset = d_p.offset();

    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let mut u_indices = Vec::new();
    for component in 0..2u32 {
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
    // Resolve gradient targets first, then generate WGSL
    let targets =
        crate::solver::model::modules::flux_module::resolve_flux_module_gradients_targets(
            &model.state_layout,
            &flux_layout,
        )?;
    Ok(
        crate::solver::model::modules::flux_module::generate_flux_module_gradients_wgsl(
            model.state_layout.stride(),
            &flux_layout,
            &targets,
        ),
    )
}

fn generate_rhie_chow_store_grad_p_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _coupling: crate::solver::model::invariants::MomentumPressureCoupling,
    grad_p_name: &'static str,
    grad_p_old_name: &'static str,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::ports::dimensions::PressureGradient;
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    // Register gradient fields using pre-computed derived names
    let grad_p = registry
        .register_vector2_field::<PressureGradient>(grad_p_name)
        .map_err(|e| {
            format!(
                "rhie_chow/store_grad_p: missing gradient field '{}': {e}",
                grad_p_name
            )
        })?;

    let grad_old = registry
        .register_vector2_field::<PressureGradient>(grad_p_old_name)
        .map_err(|e| {
            format!(
                "rhie_chow/store_grad_p: missing old gradient field '{}': {e}",
                grad_p_old_name
            )
        })?;

    let stride = registry.state_layout().stride();
    let grad_p_x = grad_p
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 0")?;
    let grad_p_y = grad_p
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 1")?;
    let grad_old_x = grad_old
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 0")?;
    let grad_old_y = grad_old
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 1")?;

    Ok(wgsl::generate_rhie_chow_store_grad_p_wgsl(
        stride, grad_p_x, grad_p_y, grad_old_x, grad_old_y,
    ))
}

fn generate_rhie_chow_correct_velocity_delta_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
    coupling: crate::solver::model::invariants::MomentumPressureCoupling,
    grad_p_name: &'static str,
    grad_p_old_name: &'static str,
) -> Result<KernelWgsl, String> {
    use crate::solver::model::ports::dimensions::{AnyDimension, D_P};
    use crate::solver::model::ports::dimensions::PressureGradient;
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    let d_p = registry
        .register_scalar_field::<D_P>(dp_field)
        .map_err(|e| format!("rhie_chow/correct_velocity_delta: {e}"))?;

    let momentum = coupling.momentum;

    // Register momentum field
    let u = registry
        .register_vector2_field::<AnyDimension>(momentum.name())
        .map_err(|e| format!("rhie_chow/correct_velocity_delta: {e}"))?;

    // Register gradient fields using pre-computed derived names
    let grad_p = registry
        .register_vector2_field::<PressureGradient>(grad_p_name)
        .map_err(|e| {
            format!(
                "rhie_chow/correct_velocity_delta: missing gradient field '{}': {e}",
                grad_p_name
            )
        })?;

    let grad_old = registry
        .register_vector2_field::<PressureGradient>(grad_p_old_name)
        .map_err(|e| {
            format!(
                "rhie_chow/correct_velocity_delta: missing old gradient field '{}': {e}",
                grad_p_old_name
            )
        })?;

    let stride = registry.state_layout().stride();
    let d_p_offset = d_p.offset();
    let u_x = u
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("u component 0")?;
    let u_y = u
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("u component 1")?;
    let grad_p_x = grad_p
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 0")?;
    let grad_p_y = grad_p
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 1")?;
    let grad_old_x = grad_old
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 0")?;
    let grad_old_y = grad_old
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 1")?;

    Ok(wgsl::generate_rhie_chow_correct_velocity_delta_wgsl(
        stride, u_x, u_y, d_p_offset, grad_p_x, grad_p_y, grad_old_x, grad_old_y,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::dimensions::{
        D_P, Density, DynamicViscosity, MassFlux, Pressure, PressureGradient, UnitDimension,
        Velocity,
    };
    use crate::solver::model::backend::ast::{
        fvm, surface_scalar, vol_scalar, vol_vector, Coefficient, EquationSystem,
    };
    use crate::solver::model::backend::state_layout::StateLayout;
    use crate::solver::model::{BoundarySpec, PrimitiveDerivations};

    #[test]
    fn contract_rhie_chow_aux_kernel_generators_honor_dp_field_name() {
        let u = vol_vector("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);
        let phi = surface_scalar("phi", MassFlux::UNIT);
        let mu = vol_scalar("mu", DynamicViscosity::UNIT);
        let rho = vol_scalar("rho", Density::UNIT);
        let dp_custom = vol_scalar("dp_custom", D_P::UNIT);
        let grad_p = vol_vector("grad_p", PressureGradient::UNIT);
        let grad_p_old = vol_vector("grad_p_old", PressureGradient::UNIT);

        let momentum = (fvm::ddt_coeff(Coefficient::field(rho).expect("rho must be scalar"), u)
            + fvm::div(phi, u)
            + fvm::laplacian(Coefficient::field(mu).expect("mu must be scalar"), u)
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

        let module =
            rhie_chow_aux_module(&system, "dp_custom", true, true).expect("module creation failed");

        let model = crate::solver::model::ModelSpec {
            id: "rhie_chow_dp_custom_test",
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),
            modules: vec![module],
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
            crate::solver::model::kernel::generate_kernel_wgsl_for_model_by_id(
                &model, &schemes, kernel_id,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "generator missing or failed for {}: {e}",
                    kernel_id.as_str()
                )
            });
        }
    }

    #[test]
    fn missing_grad_p_old_returns_clear_error() {
        // Regression test: when grad_p_old is missing, the generator should
        // return a clear error containing the missing field name.
        let u = vol_vector("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);
        let phi = surface_scalar("phi", MassFlux::UNIT);
        let mu = vol_scalar("mu", DynamicViscosity::UNIT);
        let rho = vol_scalar("rho", Density::UNIT);
        let dp = vol_scalar("dp", D_P::UNIT);
        let grad_p = vol_vector("grad_p", PressureGradient::UNIT);
        // Note: grad_p_old is intentionally missing

        let momentum = (fvm::ddt_coeff(Coefficient::field(rho).expect("rho must be scalar"), u)
            + fvm::div(phi, u)
            + fvm::laplacian(Coefficient::field(mu).expect("mu must be scalar"), u)
            + fvm::grad(p))
        .eqn(u);

        let pressure = (fvm::laplacian(
            Coefficient::product(
                Coefficient::field(rho).expect("rho must be scalar"),
                Coefficient::field(dp).expect("dp must be scalar"),
            )
            .expect("pressure coefficient must be scalar"),
            p,
        ) + fvm::div_flux(phi, p))
        .eqn(p);

        let mut system = EquationSystem::new();
        system.add_equation(momentum);
        system.add_equation(pressure);

        let layout = StateLayout::new(vec![u, p, dp, grad_p]);

        // Module creation should succeed (no StateLayout validation yet)
        let module =
            rhie_chow_aux_module(&system, "dp", true, true).expect("module creation failed");

        let model = crate::solver::model::ModelSpec {
            id: "rhie_chow_missing_grad_p_old_test",
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),
            modules: vec![module],
            linear_solver: None,
            primitives: PrimitiveDerivations::identity(),
        };

        let schemes = crate::solver::ir::SchemeRegistry::default();

        // Test store_grad_p kernel - should fail with clear error about missing grad_p_old
        let result = crate::solver::model::kernel::generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId("rhie_chow/store_grad_p"),
        );
        let err = result.expect_err("should fail when grad_p_old is missing");
        assert!(
            err.contains("grad_p_old"),
            "error should contain the missing field name 'grad_p_old': {err}"
        );

        // Test correct_velocity_delta kernel - should also fail with clear error
        let result2 = crate::solver::model::kernel::generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId("rhie_chow/correct_velocity_delta"),
        );
        let err2 = result2.expect_err("should fail when grad_p_old is missing");
        assert!(
            err2.contains("grad_p_old"),
            "error should contain the missing field name 'grad_p_old': {err2}"
        );
    }

    #[test]
    fn rhie_chow_module_has_port_manifest() {
        // Verify that rhie_chow_aux_module produces a PortManifest with expected fields.
        let u = vol_vector("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);
        let phi = surface_scalar("phi", MassFlux::UNIT);
        let mu = vol_scalar("mu", DynamicViscosity::UNIT);
        let rho = vol_scalar("rho", Density::UNIT);
        let dp = vol_scalar("dp", D_P::UNIT);
        let grad_p = vol_vector("grad_p", PressureGradient::UNIT);
        let grad_p_old = vol_vector("grad_p_old", PressureGradient::UNIT);

        let momentum = (fvm::ddt_coeff(Coefficient::field(rho).expect("rho must be scalar"), u)
            + fvm::div(phi, u)
            + fvm::laplacian(Coefficient::field(mu).expect("mu must be scalar"), u)
            + fvm::grad(p))
        .eqn(u);

        let pressure = (fvm::laplacian(
            Coefficient::product(
                Coefficient::field(rho).expect("rho must be scalar"),
                Coefficient::field(dp).expect("dp must be scalar"),
            )
            .expect("pressure coefficient must be scalar"),
            p,
        ) + fvm::div_flux(phi, p))
        .eqn(p);

        let mut system = EquationSystem::new();
        system.add_equation(momentum);
        system.add_equation(pressure);

        let _layout = StateLayout::new(vec![u, p, dp, grad_p, grad_p_old]);

        let module = rhie_chow_aux_module(&system, "dp", true, true).expect("module creation");

        // Check that port_manifest is present
        let port_manifest = module
            .port_manifest
            .expect("port_manifest should be present");

        // Should have 4 fields: dp, grad_p, grad_p_old, momentum
        assert_eq!(port_manifest.fields.len(), 4, "expected 4 field specs");

        // Verify field specs
        let dp_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "dp")
            .expect("dp field spec");
        assert_eq!(dp_field.kind, crate::solver::ir::ports::PortFieldKind::Scalar);
        assert_eq!(dp_field.unit, D_P::UNIT);

        let grad_p_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "grad_p")
            .expect("grad_p field spec");
        assert_eq!(
            grad_p_field.kind,
            crate::solver::ir::ports::PortFieldKind::Vector2
        );
        assert_eq!(grad_p_field.unit, PressureGradient::UNIT);

        let grad_p_old_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "grad_p_old")
            .expect("grad_p_old field spec");
        assert_eq!(
            grad_p_old_field.kind,
            crate::solver::ir::ports::PortFieldKind::Vector2
        );
        assert_eq!(grad_p_old_field.unit, PressureGradient::UNIT);

        let momentum_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "U")
            .expect("momentum field spec");
        assert_eq!(
            momentum_field.kind,
            crate::solver::ir::ports::PortFieldKind::Vector2
        );
        assert_eq!(
            momentum_field.unit,
            crate::solver::ir::ports::ANY_DIMENSION,
            "momentum should use ANY_DIMENSION sentinel"
        );
    }
}
