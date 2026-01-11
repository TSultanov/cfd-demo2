#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchurBlockLayout {
    /// Indices of the velocity-like unknowns within the per-cell unknown vector.
    ///
    /// For the current 2D incompressible bridge, this is `[u_x, u_y]`.
    pub u: [u32; 2],
    /// Index of the pressure-like unknown within the per-cell unknown vector.
    pub p: u32,
}

impl SchurBlockLayout {
    pub fn validate(self, unknowns_per_cell: u32) -> Result<(), String> {
        let [u0, u1] = self.u;
        let p = self.p;

        if unknowns_per_cell == 0 {
            return Err("SchurBlockLayout requires unknowns_per_cell > 0".to_string());
        }
        for (name, idx) in [("u[0]", u0), ("u[1]", u1), ("p", p)] {
            if idx >= unknowns_per_cell {
                return Err(format!(
                    "SchurBlockLayout index {}={} is out of range for unknowns_per_cell={}.",
                    name, idx, unknowns_per_cell
                ));
            }
        }
        if u0 == u1 || u0 == p || u1 == p {
            return Err(format!(
                "SchurBlockLayout indices must be distinct (got u=[{},{}], p={}).",
                u0, u1, p
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelPreconditionerSpec {
    /// Leave preconditioner selection to the runtime config.
    ///
    /// This is the default for most models while solver plumbing is being
    /// migrated to be fully model-owned.
    Default,

    /// SIMPLE-like Schur complement preconditioner for 2D incompressible systems.
    ///
    /// Assumes unknown ordering `[U_x, U_y, p]` (3 unknowns per cell).
    Schur {
        /// Relaxation factor used in the pressure smoother.
        omega: f32,
        /// Model-declared block layout for the Schur complement preconditioner.
        ///
        /// This makes the preconditioner independent of any solver-side assumptions about
        /// unknown ordering.
        layout: SchurBlockLayout,
    },
}

impl Default for ModelPreconditionerSpec {
    fn default() -> Self {
        Self::Default
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelLinearSolverSpec {
    pub preconditioner: ModelPreconditionerSpec,
}

impl Default for ModelLinearSolverSpec {
    fn default() -> Self {
        Self {
            preconditioner: ModelPreconditionerSpec::Default,
        }
    }
}
