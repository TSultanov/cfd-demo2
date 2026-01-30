use crate::solver::ir::{FieldKind, FluxLayout, StateLayout};
use crate::solver::model::flux_module::FluxModuleSpec;
use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, ModuleManifest};
use crate::solver::model::KernelId;

use cfd2_codegen::solver::codegen::KernelWgsl;

mod wgsl_flux {
    include!("flux_module_wgsl.rs");
}

mod wgsl_gradients {
    include!("flux_module_gradients_wgsl.rs");
}

pub(crate) use wgsl_gradients::generate_flux_module_gradients_wgsl;

/// Resolved gradient target for flux module gradients kernel.
///
/// This record holds pre-resolved offsets and metadata for a single gradient computation target,
/// eliminating the need to scan StateLayout during WGSL generation.
#[derive(Debug, Clone)]
pub struct ResolvedGradientTarget {
    /// The component key used in BC/flux tables (e.g., "rho_u_x")
    pub component: String,
    /// Base field name (e.g., "rho_u" for component "rho_u_x")
    pub base_field: String,
    /// Component index within the base field (0/1/2)
    pub base_component: u32,
    /// Offset of the base field/component in state array
    pub base_offset: u32,
    /// Offset of gradient x-component in state array
    pub grad_x_offset: u32,
    /// Offset of gradient y-component in state array
    pub grad_y_offset: u32,
    /// Offset in flux layout for BC lookup (if applicable)
    pub bc_unknown_offset: Option<u32>,
    /// SlipWall: x-offset of full vec2 field (for velocity fields)
    pub slip_vec2_x_offset: Option<u32>,
    /// SlipWall: y-offset of full vec2 field (for velocity fields)
    pub slip_vec2_y_offset: Option<u32>,
}

/// Resolve gradient targets from state layout and flux layout.
///
/// This function discovers all `grad_*` fields in the layout and resolves their offsets
/// and metadata. The result can be passed directly to the WGSL generator without further
/// StateLayout probing.
pub fn resolve_flux_module_gradients_targets(
    layout: &StateLayout,
    flux_layout: &FluxLayout,
) -> Result<Vec<ResolvedGradientTarget>, String> {
    let gradients = collect_gradient_targets(layout)?;
    build_resolved_targets(layout, flux_layout, &gradients)
}

/// Collect gradient field pairs (base_component_name, grad_field_name) from layout.
fn collect_gradient_targets(layout: &StateLayout) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for field in layout.fields() {
        let name = field.name();
        if !name.starts_with("grad_") {
            continue;
        }
        if field.kind() != FieldKind::Vector2 {
            continue;
        }
        let base = &name["grad_".len()..];
        if base.is_empty() {
            continue;
        }
        // Gradient targets are declared implicitly by naming convention:
        // - `grad_<scalar>` computes gradients for scalar fields.
        // - `grad_<vec>_x` / `grad_<vec>_y` compute gradients for individual components.
        if layout.field(base).is_some() {
            out.push((base.to_string(), name.to_string()));
            continue;
        }

        if let Some((base_field, component)) = base.rsplit_once('_') {
            let comp_ok = matches!(component, "x" | "y" | "z");
            if comp_ok && layout.field(base_field).is_some() {
                out.push((base.to_string(), name.to_string()));
            }
        }
    }

    if out.is_empty() {
        return Err(
            "flux_module_gradients requested but no grad_<field> fields found in state layout"
                .to_string(),
        );
    }

    Ok(out)
}

/// Build resolved gradient targets with offset resolution.
fn build_resolved_targets(
    layout: &StateLayout,
    flux_layout: &FluxLayout,
    gradients: &[(String, String)],
) -> Result<Vec<ResolvedGradientTarget>, String> {
    #[cfg(cfd2_build_script)]
    {
        build_resolved_targets_runtime(layout, flux_layout, gradients)
    }
    #[cfg(not(cfd2_build_script))]
    {
        build_resolved_targets_build_script(layout, flux_layout, gradients)
    }
}

#[cfg(cfd2_build_script)]
fn build_resolved_targets_runtime(
    layout: &StateLayout,
    flux_layout: &FluxLayout,
    gradients: &[(String, String)],
) -> Result<Vec<ResolvedGradientTarget>, String> {
    use crate::solver::model::ports::dimensions::Dimensionless;
    use crate::solver::model::ports::{PortRegistry, Scalar, Vector2, Vector3};

    let mut registry = PortRegistry::new(layout.clone());
    let mut targets = Vec::new();

    for (component, grad) in gradients {
        let (base_field, base_component) = resolve_base_scalar(layout, component)?;

        // Register base field and get offset
        let base_offset = if let Some(field) = layout.field(&base_field) {
            match field.kind() {
                FieldKind::Scalar => {
                    let port = registry
                        .register_scalar_field::<Dimensionless>(base_field.as_str())
                        .map_err(|e| format!("flux_module_gradients: {e}"))?;
                    port.offset()
                }
                FieldKind::Vector2 => {
                    let port = registry
                        .register_vector2_field::<Dimensionless>(base_field.as_str())
                        .map_err(|e| format!("flux_module_gradients: {e}"))?;
                    port.component(base_component)
                        .map(|c| c.full_offset())
                        .ok_or_else(|| format!("flux_module_gradients: invalid component {base_component} for '{base_field}'"))?
                }
                FieldKind::Vector3 => {
                    let port = registry
                        .register_vector3_field::<Dimensionless>(base_field.as_str())
                        .map_err(|e| format!("flux_module_gradients: {e}"))?;
                    port.component(base_component)
                        .map(|c| c.full_offset())
                        .ok_or_else(|| format!("flux_module_gradients: invalid component {base_component} for '{base_field}'"))?
                }
            }
        } else {
            return Err(format!(
                "flux_module_gradients: base field '{base_field}' not found"
            ));
        };

        // Register grad field and get component offsets
        let grad_port = registry
            .register_vector2_field::<Dimensionless>(grad.as_str())
            .map_err(|e| format!("flux_module_gradients: missing gradient field '{grad}': {e}"))?;

        let grad_x_offset = grad_port
            .component(0)
            .map(|c| c.full_offset())
            .ok_or_else(|| format!("flux_module_gradients: missing '{grad}[0]'"))?;
        let grad_y_offset = grad_port
            .component(1)
            .map(|c| c.full_offset())
            .ok_or_else(|| format!("flux_module_gradients: missing '{grad}[1]'"))?;

        let bc_unknown_offset = flux_layout.offset_for(component).map(|v| v as u32);

        // SlipWall offsets for velocity-like vec2 fields
        let (slip_vec2_x_offset, slip_vec2_y_offset) = match base_field.as_str() {
            "u" | "U" | "rho_u" | "rhoU" => {
                let slip_port = registry
                    .register_vector2_field::<Dimensionless>(base_field.as_str())
                    .map_err(|e| format!("flux_module_gradients: {e}"))?;
                (
                    slip_port.component(0).map(|c| c.full_offset()),
                    slip_port.component(1).map(|c| c.full_offset()),
                )
            }
            _ => (None, None),
        };

        targets.push(ResolvedGradientTarget {
            component: component.clone(),
            base_field,
            base_component,
            base_offset,
            grad_x_offset,
            grad_y_offset,
            bc_unknown_offset,
            slip_vec2_x_offset,
            slip_vec2_y_offset,
        });
    }
    Ok(targets)
}

#[cfg(not(cfd2_build_script))]
fn build_resolved_targets_build_script(
    layout: &StateLayout,
    flux_layout: &FluxLayout,
    gradients: &[(String, String)],
) -> Result<Vec<ResolvedGradientTarget>, String> {
    let mut targets = Vec::new();
    for (component, grad) in gradients {
        let (base_field, base_component) = resolve_base_scalar(layout, component)?;
        let base_offset = layout
            .component_offset(&base_field, base_component)
            .ok_or_else(|| {
                format!("state layout missing field '{base_field}[{base_component}]'")
            })?;
        let grad_x_offset = layout
            .component_offset(grad, 0)
            .ok_or_else(|| format!("state layout missing vector field '{grad}[0]'"))?;
        let grad_y_offset = layout
            .component_offset(grad, 1)
            .ok_or_else(|| format!("state layout missing vector field '{grad}[1]'"))?;
        let bc_unknown_offset = flux_layout.offset_for(component).map(|v| v as u32);

        // SlipWall uses a vector constraint u_patch = u_owner - (u_owner·n)n, which cannot be
        // represented by scalar per-component BC tables. For velocity-like vec2 fields, keep the
        // offsets of the full vec2 so the gradients kernel can apply the projection at boundary
        // faces (matching OpenFOAM's patch-field handling in `fvc::grad`).
        let (slip_vec2_x_offset, slip_vec2_y_offset) = match base_field.as_str() {
            "u" | "U" | "rho_u" | "rhoU" => (
                layout.component_offset(&base_field, 0),
                layout.component_offset(&base_field, 1),
            ),
            _ => (None, None),
        };

        targets.push(ResolvedGradientTarget {
            component: component.clone(),
            base_field,
            base_component,
            base_offset,
            grad_x_offset,
            grad_y_offset,
            bc_unknown_offset,
            slip_vec2_x_offset,
            slip_vec2_y_offset,
        });
    }
    Ok(targets)
}

fn resolve_base_scalar(layout: &StateLayout, component: &str) -> Result<(String, u32), String> {
    if let Some(field) = layout.field(component) {
        if field.kind() != FieldKind::Scalar {
            return Err(format!(
                "flux_module_gradients expects '{component}' to be scalar or a <field>_<component> selector"
            ));
        }
        return Ok((component.to_string(), 0));
    }

    let (base, component_name) = component
        .rsplit_once('_')
        .ok_or_else(|| format!("flux_module_gradients: missing base field for '{component}'"))?;
    let component_idx = match component_name {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => {
            return Err(format!(
            "flux_module_gradients: unknown component suffix '{component_name}' in '{component}'"
        ))
        }
    };

    let Some(base_field) = layout.field(base) else {
        return Err(format!(
            "flux_module_gradients: base field '{base}' not found for '{component}'"
        ));
    };

    if component_idx as u32 >= base_field.component_count() {
        return Err(format!(
            "flux_module_gradients: base field '{base}' has {} components, cannot select '{component}'",
            base_field.component_count()
        ));
    }

    Ok((base.to_string(), component_idx))
}

pub fn flux_module_module(
    flux: FluxModuleSpec,
    system: &crate::solver::model::backend::ast::EquationSystem,
    state_layout: &StateLayout,
) -> Result<KernelBundleModule, String> {
    let has_gradients = match &flux {
        FluxModuleSpec::Kernel { gradients, .. } => gradients.is_some(),
        FluxModuleSpec::Scheme { gradients, .. } => gradients.is_some(),
    };

    // Pre-resolve gradient targets and attach to manifest when gradients are enabled
    let port_manifest = if has_gradients {
        let flux_layout = crate::solver::ir::FluxLayout::from_system(system);
        let targets = resolve_flux_module_gradients_targets(state_layout, &flux_layout)?;
        Some(crate::solver::ir::ports::PortManifest {
            gradient_targets: targets
                .into_iter()
                .map(|t| crate::solver::ir::ports::ResolvedGradientTargetSpec {
                    component: t.component,
                    base_field: t.base_field,
                    base_component: t.base_component,
                    base_offset: t.base_offset,
                    grad_x_offset: t.grad_x_offset,
                    grad_y_offset: t.grad_y_offset,
                    bc_unknown_offset: t.bc_unknown_offset,
                    slip_vec2_x_offset: t.slip_vec2_x_offset,
                    slip_vec2_y_offset: t.slip_vec2_y_offset,
                })
                .collect(),
            ..Default::default()
        })
    } else {
        None
    };

    let manifest = ModuleManifest {
        flux_module: Some(flux),
        port_manifest,
        ..Default::default()
    };

    let mut out = KernelBundleModule {
        name: "flux_module",
        kernels: Vec::new(),
        generators: Vec::new(),
        manifest,
        ..Default::default()
    };

    if has_gradients {
        out.kernels.push(ModelKernelSpec {
            id: KernelId::FLUX_MODULE_GRADIENTS,
            phase: KernelPhaseId::Gradients,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        });
        out.generators.push(ModelKernelGeneratorSpec::new(
            KernelId::FLUX_MODULE_GRADIENTS,
            generate_flux_module_gradients_kernel_wgsl,
        ));
    }

    out.kernels.push(ModelKernelSpec {
        id: KernelId::FLUX_MODULE,
        phase: KernelPhaseId::FluxComputation,
        dispatch: DispatchKindId::Faces,
        condition: KernelConditionId::Always,
    });
    out.generators.push(ModelKernelGeneratorSpec::new(
        KernelId::FLUX_MODULE,
        generate_flux_module_kernel_wgsl,
    ));

    Ok(out)
}

fn generate_flux_module_gradients_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
    let flux = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| {
            "flux_module_gradients requested but model has no flux module".to_string()
        })?;

    match flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel {
            gradients: Some(_), ..
        }
        | crate::solver::model::flux_module::FluxModuleSpec::Scheme {
            gradients: Some(_), ..
        } => {
            let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
            // Resolve gradient targets first, then generate WGSL
            let targets = resolve_flux_module_gradients_targets(&model.state_layout, &flux_layout)?;
            Ok(generate_flux_module_gradients_wgsl(
                model.state_layout.stride(),
                &flux_layout,
                &targets,
            ))
        }
        _ => Err("flux_module_gradients requested but model has no gradients stage".to_string()),
    }
}

fn generate_flux_module_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
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
            Ok(wgsl_flux::generate_flux_module_wgsl(
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

            Ok(wgsl_flux::generate_flux_module_wgsl_runtime_scheme(
                &model.state_layout,
                &flux_layout,
                flux_stride,
                &prims,
                &variants,
            ))
        }
    }
}
