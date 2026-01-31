/// Resolver pass for flux module state field references.
///
/// This module provides a lowering pass that pre-resolves all state fields referenced by a
/// `FluxModuleKernelSpec` (and any referenced `PrimitiveExpr`s) into an IR-safe mapping that can
/// be stored on `PortManifest`.
///
/// The goal is to avoid probing `StateLayout` during WGSL generation: we resolve the offsets once
/// during module construction and pass the mapping into codegen.
use std::collections::{HashMap, HashSet};

use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
use crate::solver::ir::{FaceScalarExpr, FaceVec2Expr, FluxModuleKernelSpec, StateLayout};
use crate::solver::shared::PrimitiveExpr;

pub fn resolve_flux_module_state_slots(
    spec: &FluxModuleKernelSpec,
    primitives: &[(String, PrimitiveExpr)],
    layout: &StateLayout,
) -> Result<ResolvedStateSlotsSpec, String> {
    let primitive_map: HashMap<&str, &PrimitiveExpr> =
        primitives.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut fields = HashSet::<String>::new();
    collect_fields_from_flux_spec(spec, &primitive_map, layout, &mut fields)?;
    resolve_fields_against_layout(fields, layout)
}

pub fn resolve_flux_module_state_slots_runtime_scheme(
    variants: &[(crate::solver::scheme::Scheme, FluxModuleKernelSpec)],
    primitives: &[(String, PrimitiveExpr)],
    layout: &StateLayout,
) -> Result<ResolvedStateSlotsSpec, String> {
    let primitive_map: HashMap<&str, &PrimitiveExpr> =
        primitives.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut fields = HashSet::<String>::new();
    for (_, spec) in variants {
        collect_fields_from_flux_spec(spec, &primitive_map, layout, &mut fields)?;
    }
    resolve_fields_against_layout(fields, layout)
}

fn collect_fields_from_flux_spec(
    spec: &FluxModuleKernelSpec,
    primitives: &HashMap<&str, &PrimitiveExpr>,
    layout: &StateLayout,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    match spec {
        FluxModuleKernelSpec::ScalarReplicated { phi } => {
            collect_from_scalar_expr(phi, primitives, layout, out)?;
        }
        FluxModuleKernelSpec::ScalarPerComponent { flux, .. } => {
            for expr in flux {
                collect_from_scalar_expr(expr, primitives, layout, out)?;
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
                collect_from_scalar_expr(expr, primitives, layout, out)?;
            }
            for expr in u_right {
                collect_from_scalar_expr(expr, primitives, layout, out)?;
            }
            for expr in flux_left {
                collect_from_scalar_expr(expr, primitives, layout, out)?;
            }
            for expr in flux_right {
                collect_from_scalar_expr(expr, primitives, layout, out)?;
            }
            collect_from_scalar_expr(a_plus, primitives, layout, out)?;
            collect_from_scalar_expr(a_minus, primitives, layout, out)?;
        }
    }

    Ok(())
}

fn collect_from_scalar_expr(
    expr: &FaceScalarExpr,
    primitives: &HashMap<&str, &PrimitiveExpr>,
    layout: &StateLayout,
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
            collect_from_primitive_expr(prim, layout, out)?;
        }

        FaceScalarExpr::Dot(a, b) => {
            collect_from_vec2_expr(a, primitives, layout, out)?;
            collect_from_vec2_expr(b, primitives, layout, out)?;
        }

        FaceScalarExpr::Add(a, b)
        | FaceScalarExpr::Sub(a, b)
        | FaceScalarExpr::Mul(a, b)
        | FaceScalarExpr::Div(a, b)
        | FaceScalarExpr::Max(a, b)
        | FaceScalarExpr::Min(a, b)
        | FaceScalarExpr::Lerp(a, b) => {
            collect_from_scalar_expr(a, primitives, layout, out)?;
            collect_from_scalar_expr(b, primitives, layout, out)?;
        }

        FaceScalarExpr::Neg(a) | FaceScalarExpr::Abs(a) | FaceScalarExpr::Sqrt(a) => {
            collect_from_scalar_expr(a, primitives, layout, out)?;
        }
    }

    Ok(())
}

fn collect_from_vec2_expr(
    expr: &FaceVec2Expr,
    primitives: &HashMap<&str, &PrimitiveExpr>,
    layout: &StateLayout,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    match expr {
        FaceVec2Expr::Builtin(_) => {}

        FaceVec2Expr::StateVec2 { field, .. } | FaceVec2Expr::CellStateVec2 { field, .. } => {
            out.insert(field.clone());
        }

        FaceVec2Expr::Vec2(x, y) => {
            collect_from_scalar_expr(x, primitives, layout, out)?;
            collect_from_scalar_expr(y, primitives, layout, out)?;
        }

        FaceVec2Expr::Add(a, b) | FaceVec2Expr::Sub(a, b) | FaceVec2Expr::Lerp(a, b) => {
            collect_from_vec2_expr(a, primitives, layout, out)?;
            collect_from_vec2_expr(b, primitives, layout, out)?;
        }

        FaceVec2Expr::Neg(a) => collect_from_vec2_expr(a, primitives, layout, out)?,

        FaceVec2Expr::MulScalar(v, s) => {
            collect_from_vec2_expr(v, primitives, layout, out)?;
            collect_from_scalar_expr(s, primitives, layout, out)?;
        }
    }

    Ok(())
}

fn collect_from_primitive_expr(
    expr: &PrimitiveExpr,
    layout: &StateLayout,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    match expr {
        PrimitiveExpr::Literal(_) => {}

        PrimitiveExpr::Field(name) => {
            let base = resolve_primitive_field_base(layout, name)?;
            out.insert(base);
        }

        PrimitiveExpr::Add(a, b)
        | PrimitiveExpr::Sub(a, b)
        | PrimitiveExpr::Mul(a, b)
        | PrimitiveExpr::Div(a, b) => {
            collect_from_primitive_expr(a, layout, out)?;
            collect_from_primitive_expr(b, layout, out)?;
        }

        PrimitiveExpr::Sqrt(inner) | PrimitiveExpr::Neg(inner) => {
            collect_from_primitive_expr(inner, layout, out)?;
        }
    }

    Ok(())
}

fn resolve_primitive_field_base(layout: &StateLayout, name: &str) -> Result<String, String> {
    if let Some(field) = layout.field(name) {
        if field.kind() == crate::solver::ir::FieldKind::Scalar {
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

    let base_field = layout.field(base).ok_or_else(|| {
        format!("flux_module: primitive field '{name}' not found in state layout")
    })?;
    if component >= base_field.component_count() {
        return Err(format!(
            "flux_module: primitive field '{name}' not found in state layout"
        ));
    }

    Ok(base.to_string())
}

fn resolve_fields_against_layout(
    fields: HashSet<String>,
    layout: &StateLayout,
) -> Result<ResolvedStateSlotsSpec, String> {
    let mut slots = Vec::new();
    for name in fields {
        let field = layout.field(&name).ok_or_else(|| {
            format!("flux_module: state field '{name}' not found in state layout")
        })?;

        let kind = match field.kind() {
            crate::solver::ir::FieldKind::Scalar => PortFieldKind::Scalar,
            crate::solver::ir::FieldKind::Vector2 => PortFieldKind::Vector2,
            crate::solver::ir::FieldKind::Vector3 => PortFieldKind::Vector3,
        };

        slots.push(ResolvedStateSlotSpec {
            name,
            kind,
            unit: field.unit(),
            base_offset: field.offset(),
        });
    }

    slots.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(ResolvedStateSlotsSpec {
        stride: layout.stride(),
        slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::{vol_scalar, vol_vector, FaceScalarExpr, FaceSide, FluxModuleKernelSpec};
    use crate::solver::units::si;

    #[test]
    fn primitive_field_component_selectors_resolve_to_base() {
        let rho = vol_scalar("rho", si::DENSITY);
        let rho_u = vol_vector("rho_u", si::MOMENTUM_DENSITY);
        let layout = StateLayout::new(vec![rho, rho_u]);

        assert_eq!(
            resolve_primitive_field_base(&layout, "rho").unwrap(),
            "rho".to_string()
        );
        assert_eq!(
            resolve_primitive_field_base(&layout, "rho_u_x").unwrap(),
            "rho_u".to_string()
        );
        assert_eq!(
            resolve_primitive_field_base(&layout, "rho_u_y").unwrap(),
            "rho_u".to_string()
        );
    }

    #[test]
    fn primitive_field_missing_component_fails() {
        let rho_u = vol_vector("rho_u", si::MOMENTUM_DENSITY);
        let layout = StateLayout::new(vec![rho_u]);

        let err = resolve_primitive_field_base(&layout, "rho_u_z").unwrap_err();
        assert!(
            err.contains("rho_u_z"),
            "expected error to mention missing field: {err}"
        );
    }

    #[test]
    fn resolves_state_slots_from_spec_and_primitives() {
        let rho = vol_scalar("rho", si::DENSITY);
        let rho_u = vol_vector("rho_u", si::MOMENTUM_DENSITY);
        let layout = StateLayout::new(vec![rho, rho_u]);

        let primitives = vec![(
            "mom_x".to_string(),
            PrimitiveExpr::Field("rho_u_x".to_string()),
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

        let resolved = resolve_flux_module_state_slots(&spec, &primitives, &layout).unwrap();
        assert_eq!(resolved.stride, 3);
        assert_eq!(resolved.slots.len(), 2);
        assert!(resolved.slots.iter().any(|s| s.name == "rho"));
        assert!(resolved.slots.iter().any(|s| s.name == "rho_u"));
    }
}

