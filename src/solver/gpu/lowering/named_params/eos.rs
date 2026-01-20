use crate::solver::gpu::program::plan::ProgramParamHandler;

pub(crate) fn handler_for_key(key: &'static str) -> Option<ProgramParamHandler> {
    use crate::solver::gpu::lowering::programs::generic_coupled;

    match key {
        "eos.gamma" => Some(generic_coupled::param_eos_gamma),
        "eos.gm1" => Some(generic_coupled::param_eos_gm1),
        "eos.r" => Some(generic_coupled::param_eos_r),
        "eos.dp_drho" => Some(generic_coupled::param_eos_dp_drho),
        "eos.p_offset" => Some(generic_coupled::param_eos_p_offset),
        "eos.theta_ref" => Some(generic_coupled::param_eos_theta_ref),
        "low_mach.model" => Some(generic_coupled::param_low_mach_model),
        "low_mach.theta_floor" => Some(generic_coupled::param_low_mach_theta_floor),
        _ => None,
    }
}
