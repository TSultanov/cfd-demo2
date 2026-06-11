use crate::solver::gpu::program::plan::ProgramParamHandler;

pub(crate) fn handler_for_key(key: &'static str) -> Option<ProgramParamHandler> {
    use crate::solver::gpu::lowering::programs::generic_coupled;

    match key {
        "buoyant.beta_g" => Some(generic_coupled::param_buoyant_beta_g),
        "buoyant.t0" => Some(generic_coupled::param_buoyant_t0),
        "buoyant.k_over_cp" => Some(generic_coupled::param_buoyant_k_over_cp),
        _ => None,
    }
}
