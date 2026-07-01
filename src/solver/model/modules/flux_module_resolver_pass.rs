/// Resolver pass for flux module state field references.
///
/// This module provides a lowering pass that pre-resolves all state fields referenced by a
/// `FluxModuleKernelSpec` (and any referenced `Expr`s) into an IR-safe mapping that can
/// be stored on `PortManifest`.
///
/// The goal is to avoid probing `StateLayout` during WGSL generation: we resolve the offsets once
/// during module construction and pass the mapping into codegen.
use std::collections::{HashMap, HashSet};

use crate::solver::ir::ports::ResolvedStateSlotsSpec;
use crate::solver::ir::{FaceScalarExpr, FaceVec2Expr, FieldKind, FluxModuleKernelSpec, StateLayout};
use crate::solver::model::ports::PortRegistry;
use cfd2_ir::ast::{Expr, ExprNode};
use crate::solver::units::UnitDim;

/// Field metadata for precomputed layout lookups.
///
/// The `unit` and `offset` fields are only used in the legacy
/// `resolve_fields_against_layout` path (test-only).
#[allow(dead_code)]
struct FieldMetadata {
    kind: FieldKind,
    unit: UnitDim,
    offset: u32,
    component_count: u32,
}

/// Build a map of field metadata from the state layout.
fn build_field_metadata_map(layout: &StateLayout) -> HashMap<String, FieldMetadata> {
    layout
        .fields()
        .iter()
        .map(|f| {
            let meta = FieldMetadata {
                kind: f.kind(),
                unit: f.unit(),
                offset: f.offset(),
                component_count: f.component_count(),
            };
            (f.name().to_string(), meta)
        })
        .collect()
}

/// Legacy layout-based resolver — retained for test equivalence checks only.
#[cfg(test)]
pub fn resolve_flux_module_state_slots(
    spec: &FluxModuleKernelSpec,
    primitives: &[(String, Expr)],
    layout: &StateLayout,
) -> Result<ResolvedStateSlotsSpec, String> {
    let field_map = build_field_metadata_map(layout);
    let primitive_map: HashMap<&str, &Expr> =
        primitives.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut fields = HashSet::<String>::new();
    collect_fields_from_flux_spec(spec, &primitive_map, &field_map, &mut fields)?;
    resolve_fields_against_layout(fields, &field_map, layout.stride())
}

/// Resolve flux module state slots using a [`PortRegistry`] as the single source of truth.
///
/// This replaces `resolve_flux_module_state_slots` by delegating layout resolution to
/// `PortRegistry::to_resolved_state_slots_for()` instead of manually walking `StateLayout`.
pub fn resolve_flux_module_state_slots_via_registry(
    spec: &FluxModuleKernelSpec,
    primitives: &[(String, Expr)],
    registry: &PortRegistry,
) -> Result<ResolvedStateSlotsSpec, String> {
    let layout = registry.state_layout();
    let field_map = build_field_metadata_map(layout);
    let primitive_map: HashMap<&str, &Expr> =
        primitives.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut fields = HashSet::<String>::new();
    collect_fields_from_flux_spec(spec, &primitive_map, &field_map, &mut fields)?;
    registry
        .to_resolved_state_slots_for(&fields)
        .map_err(|e| format!("flux_module: {e}"))
}

/// Resolve flux module state slots for runtime scheme selection using a [`PortRegistry`].
///
/// This replaces `resolve_flux_module_state_slots_runtime_scheme` by delegating layout
/// resolution to `PortRegistry::to_resolved_state_slots_for()`.
pub fn resolve_flux_module_state_slots_runtime_scheme_via_registry(
    variants: &[(crate::solver::scheme::Scheme, FluxModuleKernelSpec)],
    primitives: &[(String, Expr)],
    registry: &PortRegistry,
) -> Result<ResolvedStateSlotsSpec, String> {
    let layout = registry.state_layout();
    let field_map = build_field_metadata_map(layout);
    let primitive_map: HashMap<&str, &Expr> =
        primitives.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut fields = HashSet::<String>::new();
    for (_, spec) in variants {
        collect_fields_from_flux_spec(spec, &primitive_map, &field_map, &mut fields)?;
    }
    registry
        .to_resolved_state_slots_for(&fields)
        .map_err(|e| format!("flux_module: {e}"))
}

fn collect_fields_from_flux_spec(
    spec: &FluxModuleKernelSpec,
    primitives: &HashMap<&str, &Expr>,
    field_map: &HashMap<String, FieldMetadata>,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    match spec {
        FluxModuleKernelSpec::ScalarReplicated { phi } => {
            collect_from_scalar_expr(phi, primitives, field_map, out)?;
        }
        FluxModuleKernelSpec::ScalarPerComponent { flux, .. } => {
            for expr in flux {
                collect_from_scalar_expr(expr, primitives, field_map, out)?;
            }
        }
        FluxModuleKernelSpec::CentralUpwind {
            u_left,
            u_right,
            flux_left,
            flux_right,
            a_plus,
            a_minus,
            ..
        } => {
            for expr in u_left {
                collect_from_scalar_expr(expr, primitives, field_map, out)?;
            }
            for expr in u_right {
                collect_from_scalar_expr(expr, primitives, field_map, out)?;
            }
            for expr in flux_left {
                collect_from_scalar_expr(expr, primitives, field_map, out)?;
            }
            for expr in flux_right {
                collect_from_scalar_expr(expr, primitives, field_map, out)?;
            }
            collect_from_scalar_expr(a_plus, primitives, field_map, out)?;
            collect_from_scalar_expr(a_minus, primitives, field_map, out)?;
        }
    }

    Ok(())
}

fn collect_from_scalar_expr(
    expr: &FaceScalarExpr,
    primitives: &HashMap<&str, &Expr>,
    field_map: &HashMap<String, FieldMetadata>,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    match expr {
        FaceScalarExpr::Literal(_)
        | FaceScalarExpr::Builtin(_)
        | FaceScalarExpr::Constant { .. }
        | FaceScalarExpr::LowMachParam(_) => {}

        FaceScalarExpr::State { name, .. } => {
            out.insert(name.clone());
        }

        FaceScalarExpr::Primitive { name, .. } => {
            let prim = primitives
                .get(name.as_str())
                .ok_or_else(|| format!("flux_module: primitive '{name}' not found"))?;
            collect_from_primitive_expr(prim, field_map, out)?;
        }

        FaceScalarExpr::Dot(a, b) => {
            collect_from_vec2_expr(a, primitives, field_map, out)?;
            collect_from_vec2_expr(b, primitives, field_map, out)?;
        }

        FaceScalarExpr::Add(a, b)
        | FaceScalarExpr::Sub(a, b)
        | FaceScalarExpr::Mul(a, b)
        | FaceScalarExpr::Div(a, b)
        | FaceScalarExpr::Max(a, b)
        | FaceScalarExpr::Min(a, b)
        | FaceScalarExpr::Lerp(a, b) => {
            collect_from_scalar_expr(a, primitives, field_map, out)?;
            collect_from_scalar_expr(b, primitives, field_map, out)?;
        }

        FaceScalarExpr::Neg(a) | FaceScalarExpr::Abs(a) | FaceScalarExpr::Sqrt(a) => {
            collect_from_scalar_expr(a, primitives, field_map, out)?;
        }
    }

    Ok(())
}

fn collect_from_vec2_expr(
    expr: &FaceVec2Expr,
    primitives: &HashMap<&str, &Expr>,
    field_map: &HashMap<String, FieldMetadata>,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    match expr {
        FaceVec2Expr::Builtin(_) => {}

        FaceVec2Expr::StateVec2 { field, .. } | FaceVec2Expr::CellStateVec2 { field, .. } => {
            out.insert(field.clone());
        }

        FaceVec2Expr::Vec2(x, y) => {
            collect_from_scalar_expr(x, primitives, field_map, out)?;
            collect_from_scalar_expr(y, primitives, field_map, out)?;
        }

        FaceVec2Expr::Add(a, b) | FaceVec2Expr::Sub(a, b) | FaceVec2Expr::Lerp(a, b) => {
            collect_from_vec2_expr(a, primitives, field_map, out)?;
            collect_from_vec2_expr(b, primitives, field_map, out)?;
        }

        FaceVec2Expr::Neg(a) => collect_from_vec2_expr(a, primitives, field_map, out)?,

        FaceVec2Expr::MulScalar(v, s) => {
            collect_from_vec2_expr(v, primitives, field_map, out)?;
            collect_from_scalar_expr(s, primitives, field_map, out)?;
        }
    }

    Ok(())
}

fn collect_from_primitive_expr(
    expr: &Expr,
    field_map: &HashMap<String, FieldMetadata>,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    match expr.node() {
        ExprNode::Literal(_) => {}

        ExprNode::Ident(name) => {
            let base = resolve_primitive_field_base(field_map, name)?;
            out.insert(base);
        }

        ExprNode::Binary { left, right, .. } => {
            collect_from_primitive_expr(left, field_map, out)?;
            collect_from_primitive_expr(right, field_map, out)?;
        }

        ExprNode::Unary { expr: inner, .. } => {
            collect_from_primitive_expr(inner, field_map, out)?;
        }

        ExprNode::Call { args, .. } => {
            for arg in args {
                collect_from_primitive_expr(arg, field_map, out)?;
            }
        }

        ExprNode::Field { base, .. } => {
            collect_from_primitive_expr(base, field_map, out)?;
        }

        ExprNode::Index { base, index } => {
            collect_from_primitive_expr(base, field_map, out)?;
            collect_from_primitive_expr(index, field_map, out)?;
        }
    }

    Ok(())
}

fn resolve_primitive_field_base(
    field_map: &HashMap<String, FieldMetadata>,
    name: &str,
) -> Result<String, String> {
    if let Some(meta) = field_map.get(name) {
        if meta.kind == FieldKind::Scalar {
            return Ok(name.to_string());
        }
    }

    let (base, suffix) = name.rsplit_once('_').ok_or_else(|| {
        format!("flux_module: primitive field '{name}' not found in state layout")
    })?;
    let component = match suffix {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => {
            return Err(format!(
                "flux_module: primitive field '{name}' not found in state layout"
            ))
        }
    };

    let base_meta = field_map.get(base).ok_or_else(|| {
        format!("flux_module: primitive field '{name}' not found in state layout")
    })?;
    if component >= base_meta.component_count {
        return Err(format!(
            "flux_module: primitive field '{name}' not found in state layout"
        ));
    }

    Ok(base.to_string())
}

/// Legacy layout-based resolution helper — retained for test equivalence checks only.
#[cfg(test)]
fn resolve_fields_against_layout(
    fields: HashSet<String>,
    field_map: &HashMap<String, FieldMetadata>,
    stride: u32,
) -> Result<ResolvedStateSlotsSpec, String> {
    use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec};
    let mut slots = Vec::new();
    for name in fields {
        let meta = field_map.get(&name).ok_or_else(|| {
            format!("flux_module: state field '{name}' not found in state layout")
        })?;

        let kind = match meta.kind {
            FieldKind::Scalar => PortFieldKind::Scalar,
            FieldKind::Vector2 => PortFieldKind::Vector2,
            FieldKind::Vector3 => PortFieldKind::Vector3,
        };

        slots.push(ResolvedStateSlotSpec {
            name,
            kind,
            unit: meta.unit,
            base_offset: meta.offset,
        });
    }

    slots.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(ResolvedStateSlotsSpec { stride, slots })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::dimensions::{Density, MomentumDensity};
    use crate::solver::ir::{vol_scalar_dim, vol_vector_dim, FaceScalarExpr, FaceSide, FluxModuleKernelSpec};

    #[test]
    fn primitive_field_component_selectors_resolve_to_base() {
        let rho = vol_scalar_dim::<Density>("rho");
        let rho_u = vol_vector_dim::<MomentumDensity>("rho_u");
        let layout = StateLayout::new(vec![rho, rho_u]);
        let field_map = build_field_metadata_map(&layout);

        assert_eq!(
            resolve_primitive_field_base(&field_map, "rho").unwrap(),
            "rho".to_string()
        );
        assert_eq!(
            resolve_primitive_field_base(&field_map, "rho_u_x").unwrap(),
            "rho_u".to_string()
        );
        assert_eq!(
            resolve_primitive_field_base(&field_map, "rho_u_y").unwrap(),
            "rho_u".to_string()
        );
    }

    #[test]
    fn primitive_field_missing_component_fails() {
        let rho_u = vol_vector_dim::<MomentumDensity>("rho_u");
        let layout = StateLayout::new(vec![rho_u]);
        let field_map = build_field_metadata_map(&layout);

        let err = resolve_primitive_field_base(&field_map, "rho_u_z").unwrap_err();
        assert!(
            err.contains("rho_u_z"),
            "expected error to mention missing field: {err}"
        );
    }

    #[test]
    fn resolves_state_slots_from_spec_and_primitives() {
        let rho = vol_scalar_dim::<Density>("rho");
        let rho_u = vol_vector_dim::<MomentumDensity>("rho_u");
        let layout = StateLayout::new(vec![rho, rho_u]);

        let primitives = vec![(
            "mom_x".to_string(),
            Expr::ident("rho_u_x"),
        )];

        let spec = FluxModuleKernelSpec::ScalarPerComponent {
            components: vec!["rho".to_string(), "rho_u_x".to_string()],
            flux: vec![
                FaceScalarExpr::state(FaceSide::Owner, "rho"),
                FaceScalarExpr::Primitive {
                    side: FaceSide::Owner,
                    name: "mom_x".to_string(),
                },
            ],
        };

        // Verify old (layout-based) and new (registry-based) resolvers produce the same result.
        let resolved_old = resolve_flux_module_state_slots(&spec, &primitives, &layout).unwrap();
        let registry = PortRegistry::new(layout);
        let resolved_new =
            resolve_flux_module_state_slots_via_registry(&spec, &primitives, &registry).unwrap();

        assert_eq!(resolved_old, resolved_new, "registry-based resolver must match layout-based");
        assert_eq!(resolved_new.stride, 3);
        assert_eq!(resolved_new.slots.len(), 2);
        assert!(resolved_new.slots.iter().any(|s| s.name == "rho"));
        assert!(resolved_new.slots.iter().any(|s| s.name == "rho_u"));
    }
}

