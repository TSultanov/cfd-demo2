use crate::solver::gpu::plans::plan_instance::{PlanParam, PlanParamValue};
use crate::solver::gpu::GpuUnifiedSolver;

pub trait SolverPlanParamsExt {
    fn set_dtau(&mut self, dtau: f32);
    fn set_viscosity(&mut self, mu: f32);
    fn set_density(&mut self, rho: f32);
    fn set_alpha_u(&mut self, alpha_u: f32);
    fn set_alpha_p(&mut self, alpha_p: f32);
    fn set_inlet_velocity(&mut self, velocity: f32);
    fn set_ramp_time(&mut self, time: f32);
    fn set_outer_iters(&mut self, iters: usize);
}

impl SolverPlanParamsExt for GpuUnifiedSolver {
    fn set_dtau(&mut self, dtau: f32) {
        let _ = self.set_plan_param(PlanParam::Dtau, PlanParamValue::F32(dtau));
    }

    fn set_viscosity(&mut self, mu: f32) {
        let _ = self.set_plan_param(PlanParam::Viscosity, PlanParamValue::F32(mu));
    }

    fn set_density(&mut self, rho: f32) {
        let _ = self.set_plan_param(PlanParam::Density, PlanParamValue::F32(rho));
    }

    fn set_alpha_u(&mut self, alpha_u: f32) {
        let _ = self.set_plan_param(PlanParam::AlphaU, PlanParamValue::F32(alpha_u));
    }

    fn set_alpha_p(&mut self, alpha_p: f32) {
        let _ = self.set_plan_param(PlanParam::AlphaP, PlanParamValue::F32(alpha_p));
    }

    fn set_inlet_velocity(&mut self, velocity: f32) {
        let _ = self.set_plan_param(
            PlanParam::InletVelocity,
            PlanParamValue::F32(velocity),
        );
    }

    fn set_ramp_time(&mut self, time: f32) {
        let _ = self.set_plan_param(PlanParam::RampTime, PlanParamValue::F32(time));
    }

    fn set_outer_iters(&mut self, iters: usize) {
        let _ = self.set_plan_param(PlanParam::OuterIters, PlanParamValue::Usize(iters));
    }
}

