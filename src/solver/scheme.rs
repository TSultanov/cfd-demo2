#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum Scheme {
    #[default]
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

