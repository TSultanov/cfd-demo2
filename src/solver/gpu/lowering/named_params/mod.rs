use crate::solver::gpu::plans::program::ProgramParamHandler;
use crate::solver::model::ModelSpec;
use std::collections::HashMap;

include!(concat!(env!("OUT_DIR"), "/named_params_registry.rs"));

pub(crate) fn named_params_for_model(
    model: &ModelSpec,
) -> Result<HashMap<&'static str, ProgramParamHandler>, String> {
    let mut out: HashMap<&'static str, ProgramParamHandler> = HashMap::new();

    for module in &model.modules {
        for key in &module.manifest.named_params {
            let key = key.as_str();

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
}
