use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct UnitExp {
    num: i32,
    den: i32,
}

impl UnitExp {
    pub const fn zero() -> Self {
        Self { num: 0, den: 1 }
    }

    pub const fn from_i32(value: i32) -> Self {
        Self { num: value, den: 1 }
    }

    pub fn new(num: i32, den: i32) -> Self {
        assert!(den != 0, "unit exponent denominator must be non-zero");
        if num == 0 {
            return Self::zero();
        }

        let (mut num, mut den) = (num, den);
        if den < 0 {
            num = -num;
            den = -den;
        }

        let gcd = gcd_i32(num.abs(), den);
        Self {
            num: num / gcd,
            den: den / gcd,
        }
    }

    pub fn is_zero(self) -> bool {
        self.num == 0
    }
}

impl fmt::Display for UnitExp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl std::ops::Add for UnitExp {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        let num =
            i64::from(self.num) * i64::from(rhs.den) + i64::from(rhs.num) * i64::from(self.den);
        let den = i64::from(self.den) * i64::from(rhs.den);
        Self::new(
            i32::try_from(num).expect("unit exponent overflow"),
            i32::try_from(den).expect("unit exponent overflow"),
        )
    }
}

impl std::ops::Sub for UnitExp {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return self;
        }
        let num =
            i64::from(self.num) * i64::from(rhs.den) - i64::from(rhs.num) * i64::from(self.den);
        let den = i64::from(self.den) * i64::from(rhs.den);
        Self::new(
            i32::try_from(num).expect("unit exponent overflow"),
            i32::try_from(den).expect("unit exponent overflow"),
        )
    }
}

impl std::ops::Mul for UnitExp {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let num = i64::from(self.num) * i64::from(rhs.num);
        let den = i64::from(self.den) * i64::from(rhs.den);
        Self::new(
            i32::try_from(num).expect("unit exponent overflow"),
            i32::try_from(den).expect("unit exponent overflow"),
        )
    }
}

impl std::ops::Div for UnitExp {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(rhs.num != 0, "unit exponent division by zero");
        self * Self::new(rhs.den, rhs.num)
    }
}

fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.abs().max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDim {
    m: UnitExp,
    l: UnitExp,
    t: UnitExp,
}

impl UnitDim {
    pub const fn new(m: i8, l: i8, t: i8) -> Self {
        Self {
            m: UnitExp::from_i32(m as i32),
            l: UnitExp::from_i32(l as i32),
            t: UnitExp::from_i32(t as i32),
        }
    }

    pub const fn dimensionless() -> Self {
        Self {
            m: UnitExp::zero(),
            l: UnitExp::zero(),
            t: UnitExp::zero(),
        }
    }

    pub fn pow_ratio(self, num: i32, den: i32) -> Self {
        let exp = UnitExp::new(num, den);
        Self {
            m: self.m * exp,
            l: self.l * exp,
            t: self.t * exp,
        }
    }

    pub fn powi(self, exp: i32) -> Self {
        self.pow_ratio(exp, 1)
    }

    pub fn sqrt(self) -> Self {
        self.pow_ratio(1, 2)
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
        Self {
            m: self.m + rhs.m,
            l: self.l + rhs.l,
            t: self.t + rhs.t,
        }
    }
}

impl std::ops::Div for UnitDim {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            m: self.m - rhs.m,
            l: self.l - rhs.l,
            t: self.t - rhs.t,
        }
    }
}

impl fmt::Display for UnitDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == UnitDim::dimensionless() {
            return write!(f, "1");
        }

        let mut parts = Vec::new();
        push_dim(&mut parts, "M", self.m);
        push_dim(&mut parts, "L", self.l);
        push_dim(&mut parts, "T", self.t);
        write!(f, "{}", parts.join(" "))
    }
}

fn push_dim(parts: &mut Vec<String>, name: &str, exp: UnitExp) {
    if exp.is_zero() {
        return;
    }
    if exp.den == 1 {
        parts.push(format!("{name}^{}", exp.num));
    } else {
        parts.push(format!("{name}^({exp})"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_sqrt_halves_exponents() {
        let length_sq = UnitDim::new(0, 2, 0);
        assert_eq!(length_sq.sqrt(), UnitDim::new(0, 1, 0));

        let length = UnitDim::new(0, 1, 0);
        let sqrt_length = length.sqrt();
        assert_eq!(sqrt_length * sqrt_length, length);
    }
}
