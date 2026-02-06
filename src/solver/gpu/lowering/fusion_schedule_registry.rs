use crate::solver::gpu::modules::graph::DispatchKind;
use crate::solver::gpu::recipe::{KernelPhase, KernelSpec, SteppingMode};
use crate::solver::model::kernel::KernelFusionPolicy;
use crate::solver::model::KernelId;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/fusion_schedule_registry.rs"));
}

fn stepping_tag(stepping: SteppingMode) -> u8 {
    match stepping {
        SteppingMode::Explicit => 0,
        SteppingMode::Implicit { .. } => 1,
        SteppingMode::Coupled => 2,
    }
}

fn policy_tag(policy: KernelFusionPolicy) -> u8 {
    match policy {
        KernelFusionPolicy::Off => 0,
        KernelFusionPolicy::Safe => 1,
        KernelFusionPolicy::Aggressive => 2,
    }
}

fn decode_phase(tag: u8) -> Result<KernelPhase, String> {
    match tag {
        0 => Ok(KernelPhase::Preparation),
        1 => Ok(KernelPhase::Gradients),
        2 => Ok(KernelPhase::FluxComputation),
        3 => Ok(KernelPhase::Assembly),
        4 => Ok(KernelPhase::Apply),
        5 => Ok(KernelPhase::Update),
        _ => Err(format!(
            "unknown kernel phase tag in generated schedule: {tag}"
        )),
    }
}

fn decode_dispatch(tag: u8) -> Result<DispatchKind, String> {
    match tag {
        0 => Ok(DispatchKind::Cells),
        1 => Ok(DispatchKind::Faces),
        _ => Err(format!(
            "unknown dispatch kind tag in generated schedule: {tag}"
        )),
    }
}

pub(crate) fn schedule_for_model(
    model_id: &str,
    stepping: SteppingMode,
    has_grad_state: bool,
    policy: KernelFusionPolicy,
) -> Result<(Vec<KernelSpec>, Vec<&'static str>), String> {
    let Some((kernel_entries, applied_fusions)) = generated::schedule_entry(
        model_id,
        stepping_tag(stepping),
        has_grad_state,
        policy_tag(policy),
    ) else {
        return Err(format!(
            "missing compile-time kernel schedule for model='{}', stepping={:?}, has_grad_state={}, policy={:?}",
            model_id, stepping, has_grad_state, policy
        ));
    };

    let mut kernels = Vec::with_capacity(kernel_entries.len());
    for entry in kernel_entries {
        kernels.push(KernelSpec {
            id: KernelId(entry.id),
            phase: decode_phase(entry.phase)?,
            dispatch: decode_dispatch(entry.dispatch)?,
        });
    }

    Ok((kernels, applied_fusions.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gpu::recipe::KernelPhase;

    #[test]
    fn compile_time_schedule_exists_for_all_registered_models_and_contexts() {
        let models = crate::solver::model::all_models();
        let steppings = [
            SteppingMode::Explicit,
            SteppingMode::Implicit { outer_iters: 1 },
            SteppingMode::Coupled,
        ];
        let policies = [
            KernelFusionPolicy::Off,
            KernelFusionPolicy::Safe,
            KernelFusionPolicy::Aggressive,
        ];

        for model in models {
            for stepping in steppings {
                for has_grad_state in [false, true] {
                    for policy in policies {
                        let schedule =
                            schedule_for_model(model.id, stepping, has_grad_state, policy);
                        assert!(
                            schedule.is_ok(),
                            "missing schedule for model='{}', stepping={stepping:?}, has_grad_state={has_grad_state}, policy={policy:?}",
                            model.id,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn missing_schedule_returns_error_instead_of_runtime_fallback() {
        let err = schedule_for_model(
            "nonexistent_model",
            SteppingMode::Coupled,
            false,
            KernelFusionPolicy::Safe,
        )
        .unwrap_err();
        assert!(
            err.contains("missing compile-time kernel schedule"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn incompressible_coupled_aggressive_uses_dp_init_full_rhie_fused_kernel() {
        for has_grad_state in [false, true] {
            let (kernels, applied) = schedule_for_model(
                "incompressible_momentum",
                SteppingMode::Coupled,
                has_grad_state,
                KernelFusionPolicy::Aggressive,
            )
            .expect("missing aggressive coupled schedule for incompressible_momentum");

            assert!(
                applied
                    .iter()
                    .any(|name| *name
                        == "rhie_chow:dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1"),
                "expected aggressive schedule to apply dp_init+full Rhie-Chow fusion rule"
            );

            let ids: Vec<&str> = kernels.iter().map(|k| k.id.as_str()).collect();
            assert!(
                ids.contains(
                    &"rhie_chow/dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused"
                ),
                "missing dp_init+full Rhie-Chow fused kernel in aggressive schedule"
            );
            assert!(
                !ids.contains(&"dp_init"),
                "dp_init should be absorbed by aggressive fused replacement"
            );
            assert!(
                !ids.contains(
                    &"rhie_chow/dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused"
                ),
                "legacy aggressive 4-kernel Rhie-Chow fused kernel should be superseded"
            );
        }
    }

    #[test]
    fn incompressible_coupled_aggressive_reduces_update_dispatches_vs_safe() {
        for has_grad_state in [false, true] {
            let (safe_kernels, _) = schedule_for_model(
                "incompressible_momentum",
                SteppingMode::Coupled,
                has_grad_state,
                KernelFusionPolicy::Safe,
            )
            .expect("missing safe coupled schedule for incompressible_momentum");
            let (aggressive_kernels, _) = schedule_for_model(
                "incompressible_momentum",
                SteppingMode::Coupled,
                has_grad_state,
                KernelFusionPolicy::Aggressive,
            )
            .expect("missing aggressive coupled schedule for incompressible_momentum");

            let safe_update_dispatches = safe_kernels
                .iter()
                .filter(|k| k.phase == KernelPhase::Update)
                .count();
            let aggressive_update_dispatches = aggressive_kernels
                .iter()
                .filter(|k| k.phase == KernelPhase::Update)
                .count();

            assert!(
                aggressive_update_dispatches + 3 == safe_update_dispatches,
                "expected aggressive update dispatches to drop by 3 (safe={safe_update_dispatches}, aggressive={aggressive_update_dispatches}, has_grad_state={has_grad_state})"
            );
        }
    }
}
