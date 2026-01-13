use crate::solver::codegen::ir::{DiscreteOpKind, DiscreteSystem};
use crate::solver::ir::{FieldKind, StateLayout};
use crate::solver::units::si;

#[derive(Debug, Clone)]
pub struct CoupledIncompressibleFields {
    pub u: String,
    pub p: String,
    pub d_p: String,
    pub grad_p: String,
}

pub fn derive_coupled_incompressible_fields(
    system: &DiscreteSystem,
    layout: &StateLayout,
) -> Result<CoupledIncompressibleFields, String> {
    let u = derive_momentum_field(system)?;
    let p = derive_pressure_field(system, &u)?;

    validate_layout_field(layout, &u, FieldKind::Vector2, Some(si::VELOCITY))?;
    validate_layout_field(layout, &p, FieldKind::Scalar, Some(si::PRESSURE))?;

    let d_p = derive_aux_field(layout, &format!("d_{p}"), FieldKind::Scalar, si::D_P, "d_p")?;
    let grad_p = derive_aux_field(
        layout,
        &format!("grad_{p}"),
        FieldKind::Vector2,
        si::PRESSURE_GRADIENT,
        "grad_p",
    )?;

    Ok(CoupledIncompressibleFields { u, p, d_p, grad_p })
}

fn derive_momentum_field(system: &DiscreteSystem) -> Result<String, String> {
    let mut vector_eq_targets = Vec::new();
    for eq in &system.equations {
        match eq.target.kind() {
            FieldKind::Vector2 | FieldKind::Vector3 => vector_eq_targets.push(eq.target),
            FieldKind::Scalar => {}
        }
    }

    match vector_eq_targets.as_slice() {
        [] => Err("missing momentum equation target (no vector equation targets found)".to_string()),
        [only] => Ok(only.name().to_string()),
        many => {
            let velocity: Vec<_> = many
                .iter()
                .copied()
                .filter(|f| f.unit() == si::VELOCITY)
                .collect();
            match velocity.as_slice() {
                [only] => Ok(only.name().to_string()),
                _ => Err(format!(
                    "ambiguous momentum target (expected 1 vector target): [{}]",
                    many.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
                )),
            }
        }
    }
}

fn derive_pressure_field(system: &DiscreteSystem, momentum_field: &str) -> Result<String, String> {
    let mut pressure_targets = Vec::new();
    for eq in &system.equations {
        if eq.target.kind() == FieldKind::Scalar && eq.target.unit() == si::PRESSURE {
            pressure_targets.push(eq.target);
        }
    }
    if pressure_targets.len() == 1 {
        return Ok(pressure_targets[0].name().to_string());
    }

    let momentum_equation = system
        .equations
        .iter()
        .find(|eq| eq.target.name() == momentum_field)
        .ok_or_else(|| {
            format!("missing momentum equation for derived momentum field '{momentum_field}'")
        })?;

    let mut gradient_fields = Vec::new();
    for op in &momentum_equation.ops {
        if op.kind != DiscreteOpKind::Gradient {
            continue;
        }
        if op.field.kind() != FieldKind::Scalar {
            continue;
        }
        gradient_fields.push(op.field);
    }

    let gradient_pressure: Vec<_> = gradient_fields
        .iter()
        .copied()
        .filter(|f| f.unit() == si::PRESSURE)
        .collect();
    match gradient_pressure.as_slice() {
        [only] => return Ok(only.name().to_string()),
        _ => {}
    }

    if pressure_targets.len() > 1 {
        let gradient_names: std::collections::HashSet<_> =
            gradient_fields.iter().map(|f| f.name()).collect();
        let referenced: Vec<_> = pressure_targets
            .iter()
            .copied()
            .filter(|f| gradient_names.contains(f.name()))
            .collect();
        if referenced.len() == 1 {
            return Ok(referenced[0].name().to_string());
        }
    }

    if gradient_fields.len() == 1 {
        return Ok(gradient_fields[0].name().to_string());
    }

    if pressure_targets.is_empty() {
        Err("missing pressure field (no scalar pressure equation target found and could not infer from momentum gradients)".to_string())
    } else {
        Err(format!(
            "ambiguous pressure field (found multiple scalar pressure equation targets): [{}]",
            pressure_targets
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

fn derive_aux_field(
    layout: &StateLayout,
    preferred_name: &str,
    expected_kind: FieldKind,
    expected_unit: crate::solver::units::UnitDim,
    label: &str,
) -> Result<String, String> {
    if let Some(field) = layout.field(preferred_name) {
        if field.kind() != expected_kind {
            return Err(format!(
                "derived {label} field '{preferred_name}' has wrong kind (expected {}, got {})",
                expected_kind.as_str(),
                field.kind().as_str()
            ));
        }
        if field.unit() != expected_unit {
            return Err(format!(
                "derived {label} field '{preferred_name}' has wrong unit (expected {expected_unit}, got {})",
                field.unit()
            ));
        }
        return Ok(preferred_name.to_string());
    }

    let candidates: Vec<_> = layout
        .fields()
        .iter()
        .filter(|f| f.kind() == expected_kind && f.unit() == expected_unit)
        .map(|f| f.name())
        .collect();

    match candidates.as_slice() {
        [] => Err(format!(
            "missing {label} field in state layout (expected kind={}, unit={expected_unit})",
            expected_kind.as_str()
        )),
        [only] => Ok((*only).to_string()),
        many => Err(format!(
            "ambiguous {label} field in state layout (expected 1 candidate, found [{}])",
            many.join(", ")
        )),
    }
}

fn validate_layout_field(
    layout: &StateLayout,
    name: &str,
    expected_kind: FieldKind,
    expected_unit: Option<crate::solver::units::UnitDim>,
) -> Result<(), String> {
    let field = layout
        .field(name)
        .ok_or_else(|| format!("state layout missing required field '{name}'"))?;

    if field.kind() != expected_kind {
        return Err(format!(
            "state layout field '{name}' has wrong kind (expected {}, got {})",
            expected_kind.as_str(),
            field.kind().as_str()
        ));
    }

    if let Some(unit) = expected_unit {
        if field.unit() != unit {
            return Err(format!(
                "state layout field '{name}' has wrong unit (expected {unit}, got {})",
                field.unit()
            ));
        }
    }

    Ok(())
}

