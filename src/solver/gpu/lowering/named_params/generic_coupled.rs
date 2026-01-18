use crate::solver::gpu::plans::program::ProgramParamHandler;

pub(crate) fn handler_for_key(key: &'static str) -> Option<ProgramParamHandler> {
    use crate::solver::gpu::lowering::models::generic_coupled;

    match key {
        "dt" => Some(generic_coupled::param_dt),
        "dtau" => Some(generic_coupled::param_dtau),
        "advection_scheme" => Some(generic_coupled::param_advection_scheme),
        "time_scheme" => Some(generic_coupled::param_time_scheme),
        "preconditioner" => Some(generic_coupled::param_preconditioner),
        "viscosity" => Some(generic_coupled::param_viscosity),
        "density" => Some(generic_coupled::param_density),
        "alpha_u" => Some(generic_coupled::param_alpha_u),
        "alpha_p" => Some(generic_coupled::param_alpha_p),
        "nonconverged_relax" => Some(generic_coupled::param_nonconverged_relax),
        "outer_iters" => Some(generic_coupled::param_outer_iters),
        "detailed_profiling_enabled" => Some(generic_coupled::param_detailed_profiling),
        _ => None,
    }
}
