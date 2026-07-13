use bytemuck::{Pod, Zeroable};

#[derive(Default, Clone, Copy, Debug)]
pub struct LinearSolverStats {
    pub iterations: u32,
    pub residual: f32,
    pub converged: bool,
    pub diverged: bool,
    pub time: std::time::Duration,
}

impl LinearSolverStats {
    /// Create stats for a diverged solve (non-finite residual, etc.).
    pub fn diverged(iterations: u32, residual: f32, time: std::time::Duration) -> Self {
        Self {
            iterations,
            residual,
            converged: false,
            diverged: true,
            time,
        }
    }

    /// Create stats for a converged solve.
    pub fn converged(iterations: u32, residual: f32, time: std::time::Duration) -> Self {
        Self {
            iterations,
            residual,
            converged: true,
            diverged: false,
            time,
        }
    }

    /// Create stats for a solve that hit max iterations without converging.
    pub fn max_iterations(iterations: u32, residual: f32, time: std::time::Duration) -> Self {
        Self {
            iterations,
            residual,
            converged: false,
            diverged: false,
            time,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PreconditionerType {
    Jacobi = 0,
    Amg = 1,
    BlockJacobi = 2,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuLowMachParams {
    pub model: u32,
    pub theta_floor: f32,
    pub pressure_coupling_alpha: f32,
    /// Biharmonic dissipation coefficient (epsilon_4).
    pub eps4: f32,
}

impl Default for GpuLowMachParams {
    fn default() -> Self {
        Self {
            // Default to no low-Mach preconditioning so transient acoustics behave like rhoCentralFoam.
            model: 2,
            theta_floor: 1e-6,
            // Low-Mach pressure coupling term (Rhie-Chow style) used by the compressible
            // central-upwind flux dissipation to prevent pressure checkerboarding on collocated
            // grids. This term is gated by `model != Off`, so it has no effect on fully
            // compressible runs unless preconditioning is enabled.
            //
            // Alpha=1.0 corresponds to adding `ρ' = p'/c^2` into the dissipation state's
            // density reconstruction.
            pressure_coupling_alpha: 1.0,
            // Biharmonic dissipation OFF by default (the term, when present, is x0).
            eps4: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuConstants {
    pub dt: f32,
    pub dt_old: f32,
    pub dtau: f32,
    pub time: f32,
    pub viscosity: f32,
    pub density: f32,
    pub component: u32, // 0: x, 1: y
    pub alpha_p: f32,   // Pressure relaxation
    pub scheme: u32,    // 0: Upwind, 1: SOU, 2: QUICK
    pub alpha_u: f32,   // Velocity under-relaxation
    pub stride_x: u32,
    pub time_scheme: u32, // 0: Euler, 1: BDF2
    /// Immutable target for a stage-time velocity-inlet soft start.
    pub inlet_velocity: f32,
    /// Ramp duration; zero disables the soft start.
    pub inlet_ramp_time: f32,

    // --- Equation of state (runtime) ---
    pub eos_gamma: f32,
    pub eos_gm1: f32,
    pub eos_r: f32,
    pub eos_dp_drho: f32,
    pub eos_p_ref: f32,
    pub eos_theta_ref: f32,
    pub eos_rho_ref: f32,
    // GAUGE STORAGE references for the density-based compressible family
    // (state stores deviations from a constant reference; all zero = absolute
    // storage). See `EosRuntimeParams::gauge_rho_ref`.
    pub eos_gauge_rho_ref: f32,
    pub eos_gauge_p_ref: f32,
    pub eos_gauge_e_ref: f32,
    pub eos_gauge_p_bias: f32,
    /// Compressible inlet driving mode (see `EosRuntimeParams::bc_pressure_inlet`):
    /// 0 = velocity inlet (default), 1 = pressure inlet + floating outlet.
    pub bc_pressure_inlet: f32,
    /// Stored-form recovered-pressure floor (`f32::MIN` = inert).
    pub eos_p_floor: f32,
    /// Recovered-temperature floor (`f32::MIN` = inert).
    pub eos_t_floor: f32,
    /// Stored-form conserved-density floor (`f32::MIN` = inert).
    pub eos_rho_floor: f32,

    // --- Buoyant Boussinesq model (runtime) ---
    // LAYOUT CONTRACT: tail fields must mirror the buoyant port manifest
    // order (`buoyant_uniform_port_manifest`); the WGSL Constants struct
    // appends those specs after the EOS block and this buffer is written
    // wholesale. Defaults are the canonical BUOYANT_* values that the MMS
    // manufactured solutions are derived from.
    pub buoyant_beta_g: f32,
    pub buoyant_t0: f32,
    pub buoyant_k_over_cp: f32,
}

impl Default for GpuConstants {
    fn default() -> Self {
        Self {
            dt: 0.0001,
            dt_old: 0.0001,
            dtau: 0.0,
            time: 0.0,
            viscosity: 0.01,
            density: 1.0,
            component: 0,
            alpha_p: 1.0,
            scheme: 0,
            alpha_u: 0.7,
            stride_x: 65535 * 64,
            time_scheme: 0,
            inlet_velocity: 0.0,
            inlet_ramp_time: 0.0,
            eos_gamma: 1.4,
            eos_gm1: 0.4,
            eos_r: 1.0,
            eos_dp_drho: 0.0,
            eos_p_ref: 0.0,
            eos_theta_ref: 1.0,
            eos_rho_ref: 0.0,
            eos_gauge_rho_ref: 0.0,
            eos_gauge_p_ref: 0.0,
            eos_gauge_e_ref: 0.0,
            eos_gauge_p_bias: 0.0,
            bc_pressure_inlet: 0.0,
            eos_p_floor: f32::MIN,
            eos_t_floor: f32::MIN,
            eos_rho_floor: f32::MIN,
            buoyant_beta_g: crate::solver::model::BUOYANT_BETA_G as f32,
            buoyant_t0: crate::solver::model::BUOYANT_T0 as f32,
            buoyant_k_over_cp: crate::solver::model::BUOYANT_K_OVER_CP as f32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SolverParams {
    pub n: u32,
    pub num_groups: u32,
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuSchurPrecondGenericParams {
    pub n: u32,
    pub num_cells: u32,
    pub omega: f32,
    pub unknowns_per_cell: u32,
    pub p: u32,
    pub u_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub u0123: [u32; 4],
    pub u4567: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuGenericCoupledSchurSetupParams {
    pub num_cells: u32,
    pub unknowns_per_cell: u32,
    pub p: u32,
    pub u_len: u32,
    pub u0123: [u32; 4],
    pub u4567: [u32; 4],
}
