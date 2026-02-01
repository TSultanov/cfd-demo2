use crate::solver::ir::{FieldKind, FluxLayout, StateLayout};
use std::collections::HashSet;
use crate::solver::model::flux_module::FluxModuleSpec;
use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::KernelBundleModule;
use crate::solver::model::KernelId;

use cfd2_codegen::solver::codegen::KernelWgsl;

mod wgsl_flux {
    include!("flux_module_wgsl.rs");
}

mod wgsl_gradients {
    include!("flux_module_gradients_wgsl.rs");
}

mod resolver_pass {
    include!("flux_module_resolver_pass.rs");
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
    // Precompute a HashSet of state field names for O(1) membership checks
    let state_field_names: HashSet<&str> = layout.fields().iter().map(|f| f.name()).collect();

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
        if state_field_names.contains(base) {
            out.push((base.to_string(), name.to_string()));
            continue;
        }

        if let Some((base_field, component)) = base.rsplit_once('_') {
            let comp_ok = matches!(component, "x" | "y" | "z");
            if comp_ok && state_field_names.contains(base_field) {
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
    use crate::solver::model::ports::dimensions::AnyDimension;
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(layout.clone());
    let layout_meta = build_layout_metadata(layout);
    let mut targets = Vec::new();

    for (component, grad) in gradients {
        let (base_field, base_component) = resolve_base_scalar(&layout_meta, component)?;

        // Get base field metadata and compute offset
        let base_meta = layout_meta.fields_by_name.get(&base_field).ok_or_else(|| {
            format!("flux_module_gradients: base field '{base_field}' not found")
        })?;

        let base_offset = match base_meta.component_count {
            1 => {
                // Scalar field
                let port = registry
                    .register_scalar_field::<AnyDimension>(base_field.as_str())
                    .map_err(|e| format!("flux_module_gradients: {e}"))?;
                port.offset()
            }
            2 => {
                // Vector2 field
                let port = registry
                    .register_vector2_field::<AnyDimension>(base_field.as_str())
                    .map_err(|e| format!("flux_module_gradients: {e}"))?;
                port.component(base_component)
                    .map(|c| c.full_offset())
                    .ok_or_else(|| format!("flux_module_gradients: invalid component {base_component} for '{base_field}'"))?
            }
            3 => {
                // Vector3 field
                let port = registry
                    .register_vector3_field::<AnyDimension>(base_field.as_str())
                    .map_err(|e| format!("flux_module_gradients: {e}"))?;
                port.component(base_component)
                    .map(|c| c.full_offset())
                    .ok_or_else(|| format!("flux_module_gradients: invalid component {base_component} for '{base_field}'"))?
            }
            n => {
                return Err(format!(
                    "flux_module_gradients: unsupported component count {n} for '{base_field}'"
                ));
            }
        };

        // Register grad field and get component offsets
        let grad_port = registry
            .register_vector2_field::<AnyDimension>(grad.as_str())
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
                    .register_vector2_field::<AnyDimension>(base_field.as_str())
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

/// Precomputed layout metadata for efficient field lookups without StateLayout::field() probing.
struct LayoutMetadata {
    fields_by_name: std::collections::HashMap<String, FieldMetadata>,
}

struct FieldMetadata {
    kind: FieldKind,
    component_count: u32,
}

fn build_layout_metadata(layout: &StateLayout) -> LayoutMetadata {
    let mut fields_by_name = std::collections::HashMap::new();
    for f in layout.fields() {
        fields_by_name.insert(
            f.name().to_string(),
            FieldMetadata {
                kind: f.kind(),
                component_count: f.component_count(),
            },
        );
    }
    LayoutMetadata { fields_by_name }
}

fn resolve_base_scalar(
    layout_meta: &LayoutMetadata,
    component: &str,
) -> Result<(String, u32), String> {
    // Check if component exists as a scalar field
    if let Some(meta) = layout_meta.fields_by_name.get(component) {
        if meta.kind == FieldKind::Scalar {
            return Ok((component.to_string(), 0));
        }
        // Field exists but is not scalar; fall through to component selector logic
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

    let base_meta = layout_meta.fields_by_name.get(base).ok_or_else(|| {
        format!("flux_module_gradients: base field '{base}' not found for '{component}'")
    })?;

    if component_idx as u32 >= base_meta.component_count {
        return Err(format!(
            "flux_module_gradients: base field '{base}' has {} components, cannot select '{component}'",
            base_meta.component_count
        ));
    }

    Ok((base.to_string(), component_idx))
}

pub fn flux_module_module(
    flux: FluxModuleSpec,
    system: &crate::solver::model::backend::ast::EquationSystem,
    state_layout: &StateLayout,
    primitives: &crate::solver::model::primitives::PrimitiveDerivations,
) -> Result<KernelBundleModule, String> {
    let has_gradients = match &flux {
        FluxModuleSpec::Kernel { gradients, .. } => gradients.is_some(),
        FluxModuleSpec::Scheme { gradients, .. } => gradients.is_some(),
    };

    // Pre-resolve gradient targets and attach to manifest when gradients are enabled
    let mut gradient_targets = Vec::new();
    if has_gradients {
        let flux_layout = crate::solver::ir::FluxLayout::from_system(system);
        let targets = resolve_flux_module_gradients_targets(state_layout, &flux_layout)?;
        gradient_targets = targets
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
            .collect();
    }

    // Pre-resolve state field references for flux module WGSL generation
    let ordered_primitives = primitives
        .ordered()
        .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
    let resolved_state_slots =
        resolve_state_slots_for_flux(&flux, system, state_layout, &ordered_primitives)?;

    let mut out = KernelBundleModule {
        name: "flux_module",
        kernels: Vec::new(),
        generators: Vec::new(),
        flux_module: Some(flux),
        port_manifest: Some(crate::solver::ir::ports::PortManifest {
            gradient_targets,
            resolved_state_slots: Some(resolved_state_slots),
            ..Default::default()
        }),
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

    // Check if gradients are enabled for this flux module
    let has_gradients = match &flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel { gradients, .. } => {
            gradients.is_some()
        }
        crate::solver::model::flux_module::FluxModuleSpec::Scheme { gradients, .. } => {
            gradients.is_some()
        }
    };

    if !has_gradients {
        return Err("flux_module_gradients requested but model has no gradients stage".to_string());
    }

    // Fetch pre-resolved gradient targets from the flux_module's port_manifest
    let flux_module = model
        .modules
        .iter()
        .find(|m| m.name == "flux_module")
        .ok_or_else(|| "flux_module_gradients: flux_module not found in model".to_string())?;

    let port_manifest = flux_module
        .port_manifest
        .as_ref()
        .ok_or_else(|| "flux_module_gradients: port_manifest missing".to_string())?;

    if port_manifest.gradient_targets.is_empty() {
        return Err("flux_module_gradients: gradient_targets empty in port_manifest".to_string());
    }

    // Convert IR-safe ResolvedGradientTargetSpec to local ResolvedGradientTarget
    let targets: Vec<ResolvedGradientTarget> = port_manifest
        .gradient_targets
        .iter()
        .map(|spec| ResolvedGradientTarget {
            component: spec.component.clone(),
            base_field: spec.base_field.clone(),
            base_component: spec.base_component,
            base_offset: spec.base_offset,
            grad_x_offset: spec.grad_x_offset,
            grad_y_offset: spec.grad_y_offset,
            bc_unknown_offset: spec.bc_unknown_offset,
            slip_vec2_x_offset: spec.slip_vec2_x_offset,
            slip_vec2_y_offset: spec.slip_vec2_y_offset,
        })
        .collect();

    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    Ok(generate_flux_module_gradients_wgsl(
        model.state_layout.stride(),
        &flux_layout,
        &targets,
    ))
}

/// Resolve state slots for flux module based on the spec type.
fn resolve_state_slots_for_flux(
    flux: &FluxModuleSpec,
    system: &crate::solver::model::backend::ast::EquationSystem,
    state_layout: &StateLayout,
    primitives: &[(String, crate::solver::shared::PrimitiveExpr)],
) -> Result<crate::solver::ir::ports::ResolvedStateSlotsSpec, String> {
    match flux {
        FluxModuleSpec::Kernel { kernel, .. } => {
            resolver_pass::resolve_flux_module_state_slots(kernel, primitives, state_layout)
        }
        FluxModuleSpec::Scheme { scheme, .. } => {
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
                    system,
                    reconstruction,
                )
                .map_err(|e| format!("flux scheme lowering failed: {e}"))?;
                variants.push((reconstruction, kernel));
            }

            resolver_pass::resolve_flux_module_state_slots_runtime_scheme(
                &variants,
                primitives,
                state_layout,
            )
        }
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

    // Get resolved state slots from port_manifest (populated during module creation)
    let resolved_slots = model
        .modules
        .iter()
        .find(|m| m.name == "flux_module")
        .and_then(|m| m.port_manifest.as_ref())
        .and_then(|p| p.resolved_state_slots.as_ref())
        .ok_or_else(|| "flux_module port_manifest missing resolved_state_slots".to_string())?;

    // Use shared helper to extract EOS params (handles both compressible and Constant EOS models)
    let eos_params = crate::solver::model::kernel::extract_eos_params(model);

    match flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel { kernel, .. } => {
            Ok(wgsl_flux::generate_flux_module_wgsl(
                resolved_slots,
                &flux_layout,
                flux_stride,
                &prims,
                kernel,
                &eos_params,
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
                resolved_slots,
                &flux_layout,
                flux_stride,
                &prims,
                &variants,
                &eos_params,
            ))
        }
    }
}
