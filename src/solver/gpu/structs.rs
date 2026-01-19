use bytemuck::{Pod, Zeroable};

#[derive(Default, Clone, Copy, Debug)]
pub struct LinearSolverStats {
    pub iterations: u32,
    pub residual: f32,
    pub converged: bool,
    pub diverged: bool,
    pub time: std::time::Duration,
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
    pub _pad0: f32,
}

impl Default for GpuLowMachParams {
    fn default() -> Self {
        Self {
            // Default to no low-Mach preconditioning so transient acoustics behave like rhoCentralFoam.
            model: 2,
            theta_floor: 1e-6,
            pressure_coupling_alpha: 1.0,
            _pad0: 0.0,
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

    // --- Equation of state (runtime) ---
    pub eos_gamma: f32,
    pub eos_gm1: f32,
    pub eos_r: f32,
    pub eos_dp_drho: f32,
    pub eos_p_offset: f32,
    pub eos_theta_ref: f32,
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
            eos_gamma: 1.4,
            eos_gm1: 0.4,
            eos_r: 1.0,
            eos_dp_drho: 0.0,
            eos_p_offset: 0.0,
            eos_theta_ref: 1.0,
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
