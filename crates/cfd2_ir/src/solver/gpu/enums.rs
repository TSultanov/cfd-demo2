#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBoundaryType {
    None = 0,
    Inlet = 1,
    Outlet = 2,
    Wall = 3,
    SlipWall = 4,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeScheme {
    Euler = 0,
    BDF2 = 1,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBcKind {
    ZeroGradient = 0,
    Dirichlet = 1,
    /// Value represents `dphi/dn` (outward normal gradient).
    Neumann = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuLowMachPrecondModel {
    Legacy = 0,
    WeissSmith = 1,
    Off = 2,
}
