use crate::solver::model::kernel::KernelFusionPolicy;

pub const SCHUR_MAX_U: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchurBlockLayout {
    /// Number of velocity-like unknown components in the Schur block.
    ///
    /// This is typically 2 (2D) or 3 (3D), but is model-defined.
    pub u_len: u32,
    /// Indices of the velocity-like unknowns within the per-cell unknown vector.
    ///
    /// Only the first `u_len` entries are used.
    pub u: [u32; SCHUR_MAX_U],
    /// Index of the pressure-like unknown within the per-cell unknown vector.
    pub p: u32,
}

impl SchurBlockLayout {
    pub fn from_u_p(u: &[u32], p: u32) -> Result<Self, String> {
        if u.is_empty() {
            return Err("SchurBlockLayout requires at least one u index".to_string());
        }
        if u.len() > SCHUR_MAX_U {
            return Err(format!(
                "SchurBlockLayout supports at most {} u indices (got {})",
                SCHUR_MAX_U,
                u.len()
            ));
        }
        let mut out = [0u32; SCHUR_MAX_U];
        out[..u.len()].copy_from_slice(u);
        Ok(Self {
            u_len: u.len() as u32,
            u: out,
            p,
        })
    }

    pub fn u_indices(&self) -> &[u32] {
        let n = (self.u_len as usize).min(SCHUR_MAX_U);
        &self.u[..n]
    }

    pub fn validate(self, unknowns_per_cell: u32) -> Result<(), String> {
        if unknowns_per_cell == 0 {
            return Err("SchurBlockLayout requires unknowns_per_cell > 0".to_string());
        }
        if self.u_len == 0 {
            return Err("SchurBlockLayout requires u_len > 0".to_string());
        }
        if self.u_len as usize > SCHUR_MAX_U {
            return Err(format!(
                "SchurBlockLayout u_len={} exceeds SCHUR_MAX_U={}",
                self.u_len, SCHUR_MAX_U
            ));
        }

        let mut seen = std::collections::BTreeSet::new();
        for (i, &idx) in self.u_indices().iter().enumerate() {
            if idx >= unknowns_per_cell {
                return Err(format!(
                    "SchurBlockLayout index u[{}]={} is out of range for unknowns_per_cell={}.",
                    i, idx, unknowns_per_cell
                ));
            }
            if !seen.insert(idx) {
                return Err(format!(
                    "SchurBlockLayout indices must be distinct (duplicate u index {}).",
                    idx
                ));
            }
        }

        if self.p >= unknowns_per_cell {
            return Err(format!(
                "SchurBlockLayout index p={} is out of range for unknowns_per_cell={}.",
                self.p, unknowns_per_cell
            ));
        }
        if !seen.insert(self.p) {
            return Err(format!(
                "SchurBlockLayout indices must be distinct (p={} overlaps u indices).",
                self.p
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ModelPreconditionerSpec {
    /// Leave preconditioner selection to the runtime config.
    ///
    /// This is the default for most models while solver plumbing is being
    /// migrated to be fully model-owned.
    #[default]
    Default,

    /// SIMPLE-like Schur complement preconditioner for saddle-point systems.
    Schur {
        /// Relaxation factor used in the pressure smoother.
        ///
        /// `1.0` means "auto": the GPU backend substitutes the heavy-ball
        /// weight (see `gpu::modules::coupled_schur::heavy_ball_omega`), which
        /// converges far faster than plain Jacobi. Any other value is used
        /// verbatim. The CPU backend runs its own inner solve and ignores this.
        omega: f32,
        /// Cap on the inner pressure-relaxation sweep count
        /// (`min(20 + sqrt(num_cells)/8, sweeps_cap)` per preconditioner
        /// apply; see `gpu::modules::coupled_schur::default_pressure_sweeps`).
        /// Poisson-like symmetric pressure blocks want 64; blocks with a
        /// strong diagonal mass term (all-Mach `psi/dtau`) saturate by ~24-32
        /// and extra sweeps are pure cost. Only binds above ~9k cells.
        sweeps_cap: u32,
        /// Model-declared block layout for the Schur complement preconditioner.
        ///
        /// This makes the preconditioner independent of any solver-side assumptions about
        /// unknown ordering.
        layout: SchurBlockLayout,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLinearSolverType {
    /// FGMRES with flexible preconditioning.
    Fgmres { max_restart: usize },

    /// Conjugate Gradient (for SPD systems).
    Cg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FgmresSolutionUpdateStrategy {
    #[default]
    FusedContiguous,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelLinearSolverSettings {
    pub solver_type: ModelLinearSolverType,
    pub max_iters: u32,
    pub tolerance: f32,
    pub tolerance_abs: f32,
    pub update_strategy: FgmresSolutionUpdateStrategy,
    /// Build-time fusion policy used when deriving the recipe kernel schedule.
    ///
    /// This is intentionally init-time only. Switching policy requires rebuilding
    /// the solver graph/plan so kernel availability and dispatch order stay coherent.
    pub kernel_fusion_policy: KernelFusionPolicy,
}

impl Default for ModelLinearSolverSettings {
    fn default() -> Self {
        Self {
            solver_type: ModelLinearSolverType::Fgmres { max_restart: 60 },
            max_iters: 200,
            // Inexact-Picard relative tolerance (June 2026). The outer loop
            // re-linearizes every iteration, so solving each linearization
            // past ~1e-4 of its initial residual is pure waste — measured on
            // the OpenFOAM reference suite, walls roughly halve with metrics
            // unchanged to the 5th decimal, because solves CONVERGE early
            // and the adaptive encoding budget stops encoding dead work.
            // 1e-3 was swept and REJECTED: it breaks the compressible MMS
            // pressure order (1.36 vs >= 1.65; dual-time fine-grid
            // corrections need the extra decade). 1e-4 reproduces all five
            // MMS suites at baseline orders. The scale is
            // min(||b||, ||r0||) on every path (host + encoded, aligned by
            // gmres_logic/clamp_rel_scale). tolerance_abs stays 1e-12:
            // the criterion is deliberately purely relative.
            tolerance: 1e-4,
            tolerance_abs: 1e-12,
            update_strategy: FgmresSolutionUpdateStrategy::FusedContiguous,
            kernel_fusion_policy: KernelFusionPolicy::Safe,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelLinearSolverSpec {
    pub preconditioner: ModelPreconditionerSpec,
    pub solver: ModelLinearSolverSettings,
}

impl Default for ModelLinearSolverSpec {
    fn default() -> Self {
        Self {
            preconditioner: ModelPreconditionerSpec::Default,
            solver: ModelLinearSolverSettings::default(),
        }
    }
}
