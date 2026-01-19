use crate::solver::gpu::plans::program::ProgramParamHandler;
use crate::solver::model::ModelSpec;
use std::collections::HashMap;

mod eos;
mod generic_coupled;

pub(crate) fn named_params_for_model(
    model: &ModelSpec,
) -> Result<HashMap<&'static str, ProgramParamHandler>, String> {
    let mut out: HashMap<&'static str, ProgramParamHandler> = HashMap::new();

    for module in &model.modules {
        for key in &module.manifest.named_params {
            let key = key.as_str();

            let handler = match module.name {
                "generic_coupled" => generic_coupled::handler_for_key(key),
                "eos" => eos::handler_for_key(key),
                other => {
                    return Err(format!(
                        "module '{other}' declares named parameter '{key}', but no handler registry exists for that module"
                    ));
                }
            }
            .ok_or_else(|| {
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
