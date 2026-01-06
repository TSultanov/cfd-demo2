use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDim {
    pub m: i8,
    pub l: i8,
    pub t: i8,
}

impl UnitDim {
    pub const fn new(m: i8, l: i8, t: i8) -> Self {
        Self { m, l, t }
    }

    pub const fn dimensionless() -> Self {
        Self::new(0, 0, 0)
    }
}

impl Default for UnitDim {
    fn default() -> Self {
        Self::dimensionless()
    }
}

impl std::ops::Mul for UnitDim {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.m + rhs.m, self.l + rhs.l, self.t + rhs.t)
    }
}

impl std::ops::Div for UnitDim {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.m - rhs.m, self.l - rhs.l, self.t - rhs.t)
    }
}

impl fmt::Display for UnitDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == UnitDim::dimensionless() {
            return write!(f, "1");
        }
        write!(f, "M^{} L^{} T^{}", self.m, self.l, self.t)
    }
}

