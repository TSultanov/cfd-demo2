#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    Upwind,
    Central,
    QUICK,
}

impl Scheme {
    pub fn gpu_id(self) -> u32 {
        match self {
            Scheme::Upwind => 0,
            Scheme::Central => 1,
            Scheme::QUICK => 2,
        }
    }
}

impl Default for Scheme {
    fn default() -> Self {
        Scheme::Upwind
    }
}
