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

    pub const fn new(num: i32, den: i32) -> Self {
        assert!(den != 0, "unit exponent denominator must be non-zero");
        if num == 0 {
            return Self::zero();
        }

        let (mut num, mut den) = (num, den);
        if den < 0 {
            num = -num;
            den = -den;
        }

        let gcd = gcd_i32(abs_i32(num), den);
        Self {
            num: num / gcd,
            den: den / gcd,
        }
    }

    pub const fn is_zero(self) -> bool {
        self.num == 0
    }

    pub const fn add_exp(self, rhs: Self) -> Self {
        if self.num == 0 {
            return rhs;
        }
        if rhs.num == 0 {
            return self;
        }

        let num = (self.num as i64) * (rhs.den as i64) + (rhs.num as i64) * (self.den as i64);
        let den = (self.den as i64) * (rhs.den as i64);
        Self::new(i64_to_i32_checked(num), i64_to_i32_checked(den))
    }

    pub const fn sub_exp(self, rhs: Self) -> Self {
        if rhs.num == 0 {
            return self;
        }
        let num = (self.num as i64) * (rhs.den as i64) - (rhs.num as i64) * (self.den as i64);
        let den = (self.den as i64) * (rhs.den as i64);
        Self::new(i64_to_i32_checked(num), i64_to_i32_checked(den))
    }

    pub const fn mul_exp(self, rhs: Self) -> Self {
        if self.num == 0 || rhs.num == 0 {
            return Self::zero();
        }
        let num = (self.num as i64) * (rhs.num as i64);
        let den = (self.den as i64) * (rhs.den as i64);
        Self::new(i64_to_i32_checked(num), i64_to_i32_checked(den))
    }

    pub const fn div_exp(self, rhs: Self) -> Self {
        assert!(rhs.num != 0, "unit exponent division by zero");
        self.mul_exp(Self::new(rhs.den, rhs.num))
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
        self.add_exp(rhs)
    }
}

impl std::ops::Sub for UnitExp {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_exp(rhs)
    }
}

impl std::ops::Mul for UnitExp {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_exp(rhs)
    }
}

impl std::ops::Div for UnitExp {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_exp(rhs)
    }
}

const fn i64_to_i32_checked(value: i64) -> i32 {
    if value < i32::MIN as i64 || value > i32::MAX as i64 {
        panic!("unit exponent overflow");
    }
    value as i32
}

const fn abs_i32(value: i32) -> i32 {
    if value < 0 {
        -value
    } else {
        value
    }
}

const fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    let a = abs_i32(a);
    if a == 0 {
        1
    } else {
        a
    }
}

/// Physical dimension exponents in **SI base units** with rational powers.
///
/// - `kg` = mass (kilogram)
/// - `m` = length (meter)
/// - `s` = time (second)
/// - `K` = thermodynamic temperature (kelvin)
///
/// This encodes **dimensions only** (no scale factors); values are erased when lowering to WGSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDim {
    m: UnitExp,
    l: UnitExp,
    t: UnitExp,
    temp: UnitExp,
}

impl UnitDim {
    pub const fn new(m: i8, l: i8, t: i8) -> Self {
        Self {
            m: UnitExp::from_i32(m as i32),
            l: UnitExp::from_i32(l as i32),
            t: UnitExp::from_i32(t as i32),
            temp: UnitExp::zero(),
        }
    }

    pub const fn new_with_temp(m: i8, l: i8, t: i8, temp: i8) -> Self {
        Self {
            m: UnitExp::from_i32(m as i32),
            l: UnitExp::from_i32(l as i32),
            t: UnitExp::from_i32(t as i32),
            temp: UnitExp::from_i32(temp as i32),
        }
    }

    pub const fn dimensionless() -> Self {
        Self {
            m: UnitExp::zero(),
            l: UnitExp::zero(),
            t: UnitExp::zero(),
            temp: UnitExp::zero(),
        }
    }

    pub const fn mul_dim(self, rhs: Self) -> Self {
        Self {
            m: self.m.add_exp(rhs.m),
            l: self.l.add_exp(rhs.l),
            t: self.t.add_exp(rhs.t),
            temp: self.temp.add_exp(rhs.temp),
        }
    }

    pub const fn div_dim(self, rhs: Self) -> Self {
        Self {
            m: self.m.sub_exp(rhs.m),
            l: self.l.sub_exp(rhs.l),
            t: self.t.sub_exp(rhs.t),
            temp: self.temp.sub_exp(rhs.temp),
        }
    }

    pub const fn pow_ratio(self, num: i32, den: i32) -> Self {
        let exp = UnitExp::new(num, den);
        Self {
            m: self.m.mul_exp(exp),
            l: self.l.mul_exp(exp),
            t: self.t.mul_exp(exp),
            temp: self.temp.mul_exp(exp),
        }
    }

    pub const fn powi(self, exp: i32) -> Self {
        self.pow_ratio(exp, 1)
    }

    pub const fn sqrt(self) -> Self {
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
        self.mul_dim(rhs)
    }
}

impl std::ops::Div for UnitDim {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_dim(rhs)
    }
}

impl fmt::Display for UnitDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == UnitDim::dimensionless() {
            return write!(f, "1");
        }

        let mut parts = Vec::new();
        push_dim(&mut parts, "kg", self.m);
        push_dim(&mut parts, "m", self.l);
        push_dim(&mut parts, "s", self.t);
        push_dim(&mut parts, "K", self.temp);
        write!(f, "{}", parts.join(" "))
    }
}

fn push_dim(parts: &mut Vec<String>, name: &str, exp: UnitExp) {
    if exp.is_zero() {
        return;
    }
    if exp.den == 1 && exp.num == 1 {
        parts.push(name.to_string());
    } else if exp.den == 1 {
        parts.push(format!("{name}^{}", exp.num));
    } else {
        parts.push(format!("{name}^({exp})"));
    }
}

pub mod si {
    use super::UnitDim;

    pub const DIMENSIONLESS: UnitDim = UnitDim::dimensionless();

    pub const MASS: UnitDim = UnitDim::new(1, 0, 0);
    pub const LENGTH: UnitDim = UnitDim::new(0, 1, 0);
    pub const TIME: UnitDim = UnitDim::new(0, 0, 1);
    pub const TEMPERATURE: UnitDim = UnitDim::new_with_temp(0, 0, 0, 1);

    pub const AREA: UnitDim = LENGTH.powi(2);
    pub const VOLUME: UnitDim = AREA.mul_dim(LENGTH);

    pub const INV_TIME: UnitDim = TIME.powi(-1);

    pub const DENSITY: UnitDim = MASS.div_dim(VOLUME);
    pub const VELOCITY: UnitDim = LENGTH.div_dim(TIME);

    pub const FORCE: UnitDim = MASS.mul_dim(LENGTH).div_dim(TIME.powi(2)); // N = kg·m/s^2
    pub const PRESSURE: UnitDim = FORCE.div_dim(AREA);
    pub const DYNAMIC_VISCOSITY: UnitDim = PRESSURE.mul_dim(TIME);
    pub const POWER: UnitDim = FORCE.mul_dim(VELOCITY); // W = kg·m^2/s^3

    pub const MASS_FLUX: UnitDim = MASS.mul_dim(INV_TIME); // kg/s (integrated over face)
    pub const MOMENTUM_DENSITY: UnitDim = DENSITY.mul_dim(VELOCITY); // rho * U
    pub const ENERGY_DENSITY: UnitDim = PRESSURE; // rhoE ~ Pa
    pub const PRESSURE_GRADIENT: UnitDim = PRESSURE.div_dim(LENGTH); // grad(p)

    pub const D_P: UnitDim = VOLUME.mul_dim(TIME).div_dim(MASS); // pressure-correction mobility-like coefficient
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

    #[test]
    fn si_derived_units_match_expected_dimensions() {
        assert_eq!(si::AREA, UnitDim::new(0, 2, 0));
        assert_eq!(si::VOLUME, UnitDim::new(0, 3, 0));

        assert_eq!(si::INV_TIME, UnitDim::new(0, 0, -1));

        assert_eq!(si::TEMPERATURE, UnitDim::new_with_temp(0, 0, 0, 1));

        assert_eq!(si::DENSITY, UnitDim::new(1, -3, 0));
        assert_eq!(si::VELOCITY, UnitDim::new(0, 1, -1));

        assert_eq!(si::FORCE, UnitDim::new(1, 1, -2));
        assert_eq!(si::PRESSURE, UnitDim::new(1, -1, -2));
        assert_eq!(si::DYNAMIC_VISCOSITY, UnitDim::new(1, -1, -1));
        assert_eq!(si::POWER, UnitDim::new(1, 2, -3));

        assert_eq!(si::MASS_FLUX, UnitDim::new(1, 0, -1));
        assert_eq!(si::MOMENTUM_DENSITY, UnitDim::new(1, -2, -1));
        assert_eq!(si::ENERGY_DENSITY, UnitDim::new(1, -1, -2));
        assert_eq!(si::PRESSURE_GRADIENT, UnitDim::new(1, -2, -2));

        assert_eq!(si::D_P, UnitDim::new(-1, 3, 1));
    }

    #[test]
    fn unit_display_uses_si_base_names() {
        assert_eq!(UnitDim::dimensionless().to_string(), "1");
        assert_eq!(UnitDim::new(0, 1, 0).to_string(), "m");
        assert_eq!(UnitDim::new(1, 0, 0).to_string(), "kg");
        assert_eq!(UnitDim::new(0, 0, 1).to_string(), "s");
        assert_eq!(UnitDim::new_with_temp(0, 0, 0, 1).to_string(), "K");

        assert_eq!(UnitDim::new(1, -3, 0).to_string(), "kg m^-3");
        assert_eq!(UnitDim::new(1, 1, -2).to_string(), "kg m s^-2");

        assert_eq!(UnitDim::new(0, 1, 0).sqrt().to_string(), "m^(1/2)");
    }
}
