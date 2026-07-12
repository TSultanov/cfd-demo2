use crate::solver::ir::{FieldKind, FluxLayout, StateLayout};
use crate::solver::model::flux_module::FluxModuleSpec;
use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::KernelBundleModule;
use crate::solver::model::KernelId;
use cfd2_codegen::solver::codegen::dsl::XY;
use std::collections::HashSet;

mod wgsl_flux {
    include!("flux_module_wgsl.rs");
}

mod wgsl_gradients {
    include!("flux_module_gradients_wgsl.rs");
}

mod resolver_pass {
    include!("flux_module_resolver_pass.rs");
}

use wgsl_flux::generate_flux_module_kernel_program_runtime_scheme;
use wgsl_gradients::generate_flux_module_gradients_kernel_program;

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
        let base_meta = layout_meta
            .fields_by_name
            .get(&base_field)
            .ok_or_else(|| format!("flux_module_gradients: base field '{base_field}' not found"))?;

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
            .component(XY::X.to_usize() as u32)
            .map(|c| c.full_offset())
            .ok_or_else(|| format!("flux_module_gradients: missing '{grad}[x]'"))?;
        let grad_y_offset = grad_port
            .component(XY::Y.to_usize() as u32)
            .map(|c| c.full_offset())
            .ok_or_else(|| format!("flux_module_gradients: missing '{grad}[y]'"))?;

        let bc_unknown_offset = flux_layout.offset_for(component);

        // SlipWall offsets for velocity-like vec2 fields
        let (slip_vec2_x_offset, slip_vec2_y_offset) = match base_field.as_str() {
            "u" | "U" | "rho_u" | "rhoU" => {
                let slip_port = registry
                    .register_vector2_field::<AnyDimension>(base_field.as_str())
                    .map_err(|e| format!("flux_module_gradients: {e}"))?;
                (
                    slip_port
                        .component(XY::X.to_usize() as u32)
                        .map(|c| c.full_offset()),
                    slip_port
                        .component(XY::Y.to_usize() as u32)
                        .map(|c| c.full_offset()),
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

    if component_idx >= base_meta.component_count {
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
    explicit_primitives: Option<&crate::solver::model::primitives::PrimitiveDerivations>,
) -> Result<KernelBundleModule, String> {
    let has_gradients = match &flux {
        FluxModuleSpec::Kernel { gradients, .. } => gradients.is_some(),
        FluxModuleSpec::Scheme { gradients, .. } => gradients.is_some(),
    };

    // Build a PortRegistry from the state layout — this becomes the single source of
    // truth for field offset resolution throughout this module.
    let registry = crate::solver::model::ports::PortRegistry::new(state_layout.clone());

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
        resolve_state_slots_for_flux(&flux, system, &registry, &ordered_primitives)?;

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
        out.generators.push(ModelKernelGeneratorSpec::new_dsl(
            KernelId::FLUX_MODULE_GRADIENTS,
            generate_flux_module_gradients_kernel_program_for_model,
        ));
    }

    // A differential RK stage consumes storage coefficients (EOS density,
    // pressure/temperature mass-block entries, Rhie--Chow d_p, and possibly
    // U.grad(p)) in the face flux and residual. Refresh those closures after
    // the current-stage gradients and immediately before the face flux. The
    // placement inside this bundle is load-bearing for the CPU schedule,
    // which preserves module order while grouping gradient/flux/assembly
    // kernels together.
    if !explicit_primitives.unwrap_or(primitives).is_identity() {
        out.kernels.push(ModelKernelSpec {
            id: KernelId::EXPLICIT_PRIMITIVE_RECOVERY,
            phase: KernelPhaseId::FluxComputation,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::RequiresExplicitStepping,
        });
        out.generators
            .push(ModelKernelGeneratorSpec::new_explicit_rk4_dsl(
                KernelId::EXPLICIT_PRIMITIVE_RECOVERY,
                crate::solver::model::kernel::generate_explicit_primitive_recovery_kernel_program,
            ));
    }

    out.kernels.push(ModelKernelSpec {
        id: KernelId::FLUX_MODULE,
        phase: KernelPhaseId::FluxComputation,
        dispatch: DispatchKindId::Faces,
        condition: KernelConditionId::Always,
    });
    out.generators.push(ModelKernelGeneratorSpec::new_dsl(
        KernelId::FLUX_MODULE,
        generate_flux_module_kernel_program_for_model,
    ));

    Ok(out)
}

fn generate_flux_module_gradients_kernel_program_for_model(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<cfd2_ir::kernel::KernelProgram, String> {
    let flux = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| {
            "flux_module_gradients requested but model has no flux module".to_string()
        })?;

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
    generate_flux_module_gradients_kernel_program(
        KernelId::FLUX_MODULE_GRADIENTS.as_str(),
        model.state_layout.stride(),
        &flux_layout,
        &targets,
        model.system.topology() == cfd2_ir::equation::TopologyMode::Structured2D,
    )
}

/// Resolve state slots for flux module based on the spec type.
///
/// Uses the `PortRegistry` as the single source of truth for field offset resolution,
/// replacing direct `StateLayout` scanning.
fn resolve_state_slots_for_flux(
    flux: &FluxModuleSpec,
    system: &crate::solver::model::backend::ast::EquationSystem,
    registry: &crate::solver::model::ports::PortRegistry,
    primitives: &[(String, cfd2_ir::ast::Expr)],
) -> Result<crate::solver::ir::ports::ResolvedStateSlotsSpec, String> {
    match flux {
        FluxModuleSpec::Kernel { kernel, .. } => {
            resolver_pass::resolve_flux_module_state_slots_via_registry(
                kernel, primitives, registry,
            )
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

            resolver_pass::resolve_flux_module_state_slots_runtime_scheme_via_registry(
                &variants, primitives, registry,
            )
        }
    }
}

fn generate_flux_module_kernel_program_for_model(
    model: &crate::solver::model::ModelSpec,
    schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<cfd2_ir::kernel::KernelProgram, String> {
    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let discrete = cfd2_codegen::solver::codegen::lower_system_unchecked(&model.system, schemes);
    let face_channels = cfd2_codegen::solver::codegen::explicit_liveness::ExplicitFaceChannelLiveness::from_discrete_system(&discrete);
    let flux_stride = face_channels.storage_stride();
    let prims = model
        .primitives
        .ordered()
        .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;

    let flux = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "flux_module requested but model has no flux module".to_string())?;

    let resolved_slots = model
        .modules
        .iter()
        .find(|m| m.name == "flux_module")
        .and_then(|m| m.port_manifest.as_ref())
        .and_then(|p| p.resolved_state_slots.as_ref())
        .ok_or_else(|| "flux_module port_manifest missing resolved_state_slots".to_string())?;

    let eos_params = crate::solver::model::kernel::extract_eos_params(model);
    let structured = model.system.topology() == cfd2_ir::equation::TopologyMode::Structured2D;

    match flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel { kernel, .. } => {
            wgsl_flux::generate_flux_module_kernel_program(
                KernelId::FLUX_MODULE.as_str(),
                resolved_slots,
                &flux_layout,
                &face_channels,
                flux_stride,
                &prims,
                kernel,
                &eos_params,
                structured,
            )
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

            // Env-gated per-phase timing (`CFD2_KGEN_PROFILE=1`): kernel-program
            // generation runs at solver construction on the CPU backend, so a
            // regression here is a GUI freeze — keep the phases observable.
            let profile = std::env::var("CFD2_KGEN_PROFILE").is_ok();
            let mut variants = Vec::new();
            for reconstruction in schemes {
                let t = std::time::Instant::now();
                let kernel = crate::solver::model::flux_schemes::lower_flux_scheme(
                    scheme,
                    &model.system,
                    reconstruction,
                )
                .map_err(|e| format!("flux scheme lowering failed: {e}"))?;
                if profile {
                    eprintln!(
                        "[kgen] lower_flux_scheme {:?}: {:.0} ms",
                        reconstruction,
                        t.elapsed().as_secs_f64() * 1e3
                    );
                }
                variants.push((reconstruction, kernel));
            }

            let t = std::time::Instant::now();
            let out = generate_flux_module_kernel_program_runtime_scheme(
                KernelId::FLUX_MODULE.as_str(),
                resolved_slots,
                &flux_layout,
                &face_channels,
                flux_stride,
                &prims,
                &variants,
                &eos_params,
                structured,
            );
            if profile {
                eprintln!(
                    "[kgen] runtime_scheme program gen: {:.0} ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            out
        }
    }
}
