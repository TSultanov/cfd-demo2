use crate::solver::gpu::GpuUnifiedSolver;
use crate::solver::gpu::plans::plan_instance::PlanParamValue;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::model::EosSpec;
use std::future::Future;
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

pub trait SolverFieldAliasesExt {
    fn set_u(&mut self, u: &[(f64, f64)]);
    fn set_p(&mut self, p: &[f64]);
    fn get_u(&self) -> BoxFuture<'_, Vec<(f64, f64)>>;
    fn get_p(&self) -> BoxFuture<'_, Vec<f64>>;
    fn get_rho(&self) -> BoxFuture<'_, Vec<f64>>;
}

impl SolverFieldAliasesExt for GpuUnifiedSolver {
    fn set_u(&mut self, u: &[(f64, f64)]) {
        let _ = self
            .set_field_vec2("U", u)
            .or_else(|_| self.set_field_vec2("u", u));
    }

    fn set_p(&mut self, p: &[f64]) {
        let _ = self.set_field_scalar("p", p);
    }

    fn get_u(&self) -> BoxFuture<'_, Vec<(f64, f64)>> {
        Box::pin(async move {
            if let Ok(v) = self.get_field_vec2("U").await {
                return v;
            }
            if let Ok(v) = self.get_field_vec2("u").await {
                return v;
            }
            vec![(0.0, 0.0); self.num_cells() as usize]
        })
    }

    fn get_p(&self) -> BoxFuture<'_, Vec<f64>> {
        Box::pin(async move {
            self.get_field_scalar("p")
                .await
                .unwrap_or_else(|_| vec![0.0; self.num_cells() as usize])
        })
    }

    fn get_rho(&self) -> BoxFuture<'_, Vec<f64>> {
        Box::pin(async move {
            self.get_field_scalar("rho")
                .await
                .unwrap_or_else(|_| vec![0.0; self.num_cells() as usize])
        })
    }
}

pub trait SolverCompressibleIdealGasExt {
    fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32);
    fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]);
}

impl SolverCompressibleIdealGasExt for GpuUnifiedSolver {
    fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        let gamma = match self.model().eos {
            EosSpec::IdealGas { gamma } => gamma as f32,
            _ => return,
        };

        let stride = self.model().state_layout.stride() as usize;
        let (Some(off_rho), Some(off_rho_u), Some(off_rho_e)) = (
            self.model().state_layout.offset_for("rho"),
            self.model().state_layout.offset_for("rho_u"),
            self.model().state_layout.offset_for("rho_e"),
        ) else {
            return;
        };

        let off_rho = off_rho as usize;
        let off_rho_u = off_rho_u as usize;
        let off_rho_e = off_rho_e as usize;
        let off_p = self.model().state_layout.offset_for("p").map(|v| v as usize);
        let off_t = self.model().state_layout.offset_for("T").map(|v| v as usize);
        let off_u = self
            .model()
            .state_layout
            .offset_for("u")
            .or_else(|| self.model().state_layout.offset_for("U"))
            .map(|v| v as usize);

        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let rho_e = p / (gamma - 1.0) + ke;

        let mut state = vec![0.0f32; self.num_cells() as usize * stride];
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            state[base + off_rho] = rho;
            state[base + off_rho_u] = rho * u[0];
            state[base + off_rho_u + 1] = rho * u[1];
            state[base + off_rho_e] = rho_e;
            if let Some(off_p) = off_p {
                state[base + off_p] = p;
            }
            if let Some(off_t) = off_t {
                state[base + off_t] = p / rho.max(1e-12);
            }
            if let Some(off_u) = off_u {
                state[base + off_u] = u[0];
                state[base + off_u + 1] = u[1];
            }
        }
        let _ = self.write_state_f32(&state);
    }

    fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        if rho.len() != self.num_cells() as usize
            || u.len() != self.num_cells() as usize
            || p.len() != self.num_cells() as usize
        {
            return;
        }

        let gamma = match self.model().eos {
            EosSpec::IdealGas { gamma } => gamma as f32,
            _ => return,
        };

        let stride = self.model().state_layout.stride() as usize;
        let (Some(off_rho), Some(off_rho_u), Some(off_rho_e)) = (
            self.model().state_layout.offset_for("rho"),
            self.model().state_layout.offset_for("rho_u"),
            self.model().state_layout.offset_for("rho_e"),
        ) else {
            return;
        };

        let off_rho = off_rho as usize;
        let off_rho_u = off_rho_u as usize;
        let off_rho_e = off_rho_e as usize;
        let off_p = self.model().state_layout.offset_for("p").map(|v| v as usize);
        let off_t = self.model().state_layout.offset_for("T").map(|v| v as usize);
        let off_u = self
            .model()
            .state_layout
            .offset_for("u")
            .or_else(|| self.model().state_layout.offset_for("U"))
            .map(|v| v as usize);

        let mut state = vec![0.0f32; self.num_cells() as usize * stride];
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            let rho_val = rho[cell];
            let u_val = u[cell];
            let p_val = p[cell];
            let ke = 0.5 * rho_val * (u_val[0] * u_val[0] + u_val[1] * u_val[1]);
            let rho_e = p_val / (gamma - 1.0) + ke;

            state[base + off_rho] = rho_val;
            state[base + off_rho_u] = rho_val * u_val[0];
            state[base + off_rho_u + 1] = rho_val * u_val[1];
            state[base + off_rho_e] = rho_e;
            if let Some(off_p) = off_p {
                state[base + off_p] = p_val;
            }
            if let Some(off_t) = off_t {
                state[base + off_t] = p_val / rho_val.max(1e-12);
            }
            if let Some(off_u) = off_u {
                state[base + off_u] = u_val[0];
                state[base + off_u + 1] = u_val[1];
            }
        }
        let _ = self.write_state_f32(&state);
    }
}

pub trait SolverIncompressibleStatsExt {
    fn incompressible_should_stop(&self) -> bool;
    fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)>;
    fn incompressible_linear_stats(
        &self,
    ) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)>;
    fn incompressible_degenerate_count(&self) -> Option<u32>;
}

impl SolverIncompressibleStatsExt for GpuUnifiedSolver {
    fn incompressible_should_stop(&self) -> bool {
        self.step_stats().should_stop.unwrap_or(false)
    }

    fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        let stats = self.step_stats();
        Some((stats.outer_iterations?, stats.outer_residual_u?, stats.outer_residual_p?))
    }

    fn incompressible_linear_stats(
        &self,
    ) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        self.step_stats().linear_stats
    }

    fn incompressible_degenerate_count(&self) -> Option<u32> {
        self.step_stats().degenerate_count
    }
}

pub trait SolverIncompressibleControlsExt {
    fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String>;
    fn incompressible_set_should_stop(&mut self, value: bool);
}

impl SolverIncompressibleControlsExt for GpuUnifiedSolver {
    fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String> {
        // Coupled/implicit paths use the unified `outer_iters` knob.
        self.set_plan_named_param("outer_iters", PlanParamValue::Usize(iters as usize))
    }

    fn incompressible_set_should_stop(&mut self, value: bool) {
        let _ = value;
    }
}
