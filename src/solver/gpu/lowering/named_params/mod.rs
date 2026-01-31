use crate::solver::gpu::program::plan::ProgramParamHandler;
use crate::solver::model::ModelSpec;
use std::collections::HashMap;

include!(concat!(env!("OUT_DIR"), "/named_params_registry.rs"));

pub(crate) fn named_params_for_model(
    model: &ModelSpec,
) -> Result<HashMap<&'static str, ProgramParamHandler>, String> {
    let mut out: HashMap<&'static str, ProgramParamHandler> = HashMap::new();

    for module in &model.modules {
        // Collect keys from both named_params (legacy) and port_manifest (new)
        let mut keys_to_process: Vec<&'static str> = Vec::new();

        // Add keys from named_params
        for key in &module.manifest.named_params {
            keys_to_process.push(*key);
        }

        // Add keys from port_manifest
        if let Some(ref port_manifest) = module.manifest.port_manifest {
            for param in &port_manifest.params {
                keys_to_process.push(param.key);
            }
        }

        // Deduplicate keys within this module
        keys_to_process.sort_unstable();
        keys_to_process.dedup();

        for key in keys_to_process {
            if !named_param_registry_exists(module.name) {
                return Err(format!(
                    "module '{}' declares named parameter '{key}', but no handler registry exists for that module",
                    module.name
                ));
            }

            let handler = named_param_handler_for_key(module.name, key).ok_or_else(|| {
                format!(
                    "module '{}' declares named parameter '{key}', but no handler exists for it",
                    module.name
                )
            })?;

            out.insert(key, handler);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_coupled_named_params_include_linear_solver_tuning_keys() {
        let model = crate::solver::model::generic_diffusion_demo_model();
        let params = named_params_for_model(&model).expect("named params");

        for key in [
            "linear_solver.max_restart",
            "linear_solver.max_iters",
            "linear_solver.tolerance",
            "linear_solver.tolerance_abs",
        ] {
            assert!(
                params.contains_key(key),
                "missing named param handler for '{key}'"
            );
        }
    }

    #[test]
    fn named_params_include_eos_keys_via_port_manifest() {
        // Regression test: EOS uniform params are declared via port_manifest,
        // not named_params. Ensure they are still discoverable.
        let model = crate::solver::model::compressible_model();
        let params = named_params_for_model(&model).expect("named params");

        // These keys come from eos.port_manifest, not eos.named_params
        assert!(
            params.contains_key("eos.gamma"),
            "eos.gamma should be discoverable via port_manifest"
        );
        assert!(
            params.contains_key("eos.r"),
            "eos.r should be discoverable via port_manifest"
        );
        assert!(
            params.contains_key("eos.theta_ref"),
            "eos.theta_ref should be discoverable via port_manifest"
        );

        // low_mach params should still work (they remain in named_params)
        assert!(
            params.contains_key("low_mach.model"),
            "low_mach.model should still be in named_params"
        );
    }

    #[test]
    fn named_params_include_generic_coupled_keys_via_port_manifest() {
        // Regression test: generic_coupled uniform params are declared via port_manifest,
        // not named_params. Ensure they are still discoverable.
        let model = crate::solver::model::generic_diffusion_demo_model();
        let params = named_params_for_model(&model).expect("named params");

        // These keys come from generic_coupled.port_manifest
        assert!(
            params.contains_key("dt"),
            "dt should be discoverable via port_manifest"
        );
        assert!(
            params.contains_key("dtau"),
            "dtau should be discoverable via port_manifest"
        );
        assert!(
            params.contains_key("viscosity"),
            "viscosity should be discoverable via port_manifest"
        );
        assert!(
            params.contains_key("density"),
            "density should be discoverable via port_manifest"
        );
        assert!(
            params.contains_key("advection_scheme"),
            "advection_scheme should be discoverable via port_manifest"
        );
        assert!(
            params.contains_key("time_scheme"),
            "time_scheme should be discoverable via port_manifest"
        );

        // Host-only keys should still work (they remain in named_params)
        assert!(
            params.contains_key("outer_iters"),
            "outer_iters should still be in named_params"
        );
    }

    #[test]
    fn named_params_include_generic_coupled_relaxation_keys_when_enabled() {
        // Test that alpha_u/alpha_p are present when relaxation is enabled
        // (compressible model has apply_relaxation_in_update = true)
        let model = crate::solver::model::compressible_model();
        let params = named_params_for_model(&model).expect("named params");

        // Relaxation params should be present for compressible model
        assert!(
            params.contains_key("alpha_u"),
            "alpha_u should be present when relaxation is enabled"
        );
        assert!(
            params.contains_key("alpha_p"),
            "alpha_p should be present when relaxation is enabled"
        );

        // nonconverged_relax should also be present (remains in named_params)
        assert!(
            params.contains_key("nonconverged_relax"),
            "nonconverged_relax should still be in named_params"
        );
    }
}
