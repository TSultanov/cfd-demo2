#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBoundaryType {
    None = 0,
    Inlet = 1,
    Outlet = 2,
    Wall = 3,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeScheme {
    Euler = 0,
    BDF2 = 1,
}

