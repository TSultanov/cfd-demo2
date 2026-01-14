use crate::solver::gpu::enums::GpuLowMachPrecondModel;
use crate::solver::gpu::plans::plan_instance::PlanParamValue;
use crate::solver::gpu::GpuUnifiedSolver;
use crate::solver::model::eos::EosSpec;

pub trait SolverPlanParamsExt {
    fn set_dtau(&mut self, dtau: f32);
    fn set_viscosity(&mut self, mu: f32);
    fn set_density(&mut self, rho: f32);
    fn set_alpha_u(&mut self, alpha_u: f32);
    fn set_alpha_p(&mut self, alpha_p: f32);
    fn set_inlet_velocity(&mut self, velocity: f32);
    fn set_ramp_time(&mut self, time: f32);
    fn set_outer_iters(&mut self, iters: usize);

    // Compressible/low-Mach helpers (model/method-specific; intentionally not inherent methods on the solver).
    fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String>;
    fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String>;
    fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String>;

    // EOS helpers (runtime constants).
    fn set_eos(&mut self, eos: &EosSpec);
    fn set_eos_gamma(&mut self, gamma: f32);
    fn set_eos_gm1(&mut self, gm1: f32);
    fn set_eos_r(&mut self, r_gas: f32);
    fn set_eos_dp_drho(&mut self, dp_drho: f32);
    fn set_eos_p_offset(&mut self, p_offset: f32);
    fn set_eos_theta_ref(&mut self, theta: f32);
}

impl SolverPlanParamsExt for GpuUnifiedSolver {
    fn set_dtau(&mut self, dtau: f32) {
        let _ = self.set_plan_named_param("dtau", PlanParamValue::F32(dtau));
    }

    fn set_viscosity(&mut self, mu: f32) {
        let _ = self.set_plan_named_param("viscosity", PlanParamValue::F32(mu));
    }

    fn set_density(&mut self, rho: f32) {
        let _ = self.set_plan_named_param("density", PlanParamValue::F32(rho));
    }

    fn set_alpha_u(&mut self, alpha_u: f32) {
        let _ = self.set_plan_named_param("alpha_u", PlanParamValue::F32(alpha_u));
    }

    fn set_alpha_p(&mut self, alpha_p: f32) {
        let _ = self.set_plan_named_param("alpha_p", PlanParamValue::F32(alpha_p));
    }

    fn set_inlet_velocity(&mut self, velocity: f32) {
        let _ = self.set_plan_named_param("inlet_velocity", PlanParamValue::F32(velocity));
    }

    fn set_ramp_time(&mut self, time: f32) {
        let _ = self.set_plan_named_param("ramp_time", PlanParamValue::F32(time));
    }

    fn set_outer_iters(&mut self, iters: usize) {
        let _ = self.set_plan_named_param("outer_iters", PlanParamValue::Usize(iters));
    }

    fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String> {
        self.set_plan_named_param("low_mach.model", PlanParamValue::LowMachModel(model))
    }

    fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String> {
        self.set_plan_named_param("low_mach.theta_floor", PlanParamValue::F32(theta))
    }

    fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String> {
        self.set_plan_named_param("nonconverged_relax", PlanParamValue::F32(alpha))
    }

    fn set_eos(&mut self, eos: &EosSpec) {
        let params = eos.runtime_params();
        let _ = self.set_plan_named_param("eos.gamma", PlanParamValue::F32(params.gamma));
        let _ = self.set_plan_named_param("eos.gm1", PlanParamValue::F32(params.gm1));
        let _ = self.set_plan_named_param("eos.r", PlanParamValue::F32(params.r));
        let _ = self.set_plan_named_param("eos.dp_drho", PlanParamValue::F32(params.dp_drho));
        let _ = self.set_plan_named_param("eos.p_offset", PlanParamValue::F32(params.p_offset));
        let _ = self.set_plan_named_param("eos.theta_ref", PlanParamValue::F32(params.theta_ref));
    }

    fn set_eos_gamma(&mut self, gamma: f32) {
        let _ = self.set_plan_named_param("eos.gamma", PlanParamValue::F32(gamma));
    }

    fn set_eos_gm1(&mut self, gm1: f32) {
        let _ = self.set_plan_named_param("eos.gm1", PlanParamValue::F32(gm1));
    }

    fn set_eos_r(&mut self, r_gas: f32) {
        let _ = self.set_plan_named_param("eos.r", PlanParamValue::F32(r_gas));
    }

    fn set_eos_dp_drho(&mut self, dp_drho: f32) {
        let _ = self.set_plan_named_param("eos.dp_drho", PlanParamValue::F32(dp_drho));
    }

    fn set_eos_p_offset(&mut self, p_offset: f32) {
        let _ = self.set_plan_named_param("eos.p_offset", PlanParamValue::F32(p_offset));
    }

    fn set_eos_theta_ref(&mut self, theta: f32) {
        let _ = self.set_plan_named_param("eos.theta_ref", PlanParamValue::F32(theta));
    }
}
