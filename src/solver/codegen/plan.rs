use crate::solver::codegen::ir::{DiscreteOp, DiscreteOpKind, DiscreteSystem};

#[derive(Debug, Clone)]
pub struct MomentumPlan {
    pub ddt: Option<DiscreteOp>,
    pub convection: Option<DiscreteOp>,
    pub diffusion: Option<DiscreteOp>,
    pub gradient: Option<DiscreteOp>,
    pub pressure_diffusion: Option<DiscreteOp>,
}

pub fn momentum_plan(
    system: &DiscreteSystem,
    momentum_field: &str,
    pressure_field: &str,
) -> MomentumPlan {
    let mut plan = MomentumPlan {
        ddt: None,
        convection: None,
        diffusion: None,
        gradient: None,
        pressure_diffusion: None,
    };

    let mut found = false;
    for equation in &system.equations {
        if equation.target.name() != momentum_field {
            continue;
        }
        found = true;
        for op in &equation.ops {
            match op.kind {
                DiscreteOpKind::TimeDerivative => plan.ddt = Some(op.clone()),
                DiscreteOpKind::Convection => {
                    if plan.convection.is_none() {
                        plan.convection = Some(op.clone());
                    }
                }
                DiscreteOpKind::Diffusion => plan.diffusion = Some(op.clone()),
                DiscreteOpKind::Gradient => plan.gradient = Some(op.clone()),
                DiscreteOpKind::Source => {}
            }
        }
    }

    if !found {
        panic!(
            "missing momentum equation for field '{}'",
            momentum_field
        );
    }

    for equation in &system.equations {
        if equation.target.name() != pressure_field {
            continue;
        }
        for op in &equation.ops {
            if op.kind == DiscreteOpKind::Diffusion {
                plan.pressure_diffusion = Some(op.clone());
                break;
            }
        }
    }

    plan
}
