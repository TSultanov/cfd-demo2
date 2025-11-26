#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scheme {
    #[default]
    Upwind,
    SecondOrderUpwind,
    QUICK,
}

impl Scheme {
    pub fn gpu_id(self) -> u32 {
        match self {
            Scheme::Upwind => 0,
            Scheme::SecondOrderUpwind => 1,
            Scheme::QUICK => 2,
        }
    }
}
