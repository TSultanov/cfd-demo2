//! Type-level physical dimensions with rational exponents.
//!
//! This module provides a canonical type-level representation of physical dimensions
//! that can be used across IR, codegen, and ports without violating the IR boundary.
//!
//! # Features
//!
//! - Rational exponents for all base dimensions (M, L, T, TEMP)
//! - Type constructors: [`MulDim`], [`DivDim`], [`PowDim`], [`SqrtDim`]
//! - Typed aliases for common SI dimensions
//! - Conversion to runtime [`UnitDim`](crate::solver::units::UnitDim)
//!
//! # Example
//!
//! ```rust,ignore
//! use cfd2_ir::solver::dimensions::*;
//!
//! // Define a custom dimension
//! type MyDim = DivDim<MulDim<Pressure, Area>, Force>;
//!
//! // Convert to runtime for serialization
//! let runtime: UnitDim = MyDim::to_runtime();
//! ```

use crate::solver::units::UnitDim;

/// Trait for type-level physical dimensions with rational exponents.
///
/// Implement this trait for types that represent physical dimensions.
/// The associated constants provide the rational exponents for each
/// SI base unit as `(numerator, denominator)` pairs.
///
/// # Example
///
/// ```rust,ignore
/// use cfd2_ir::solver::dimensions::UnitDimension;
///
/// struct MyCustomDim;
/// impl UnitDimension for MyCustomDim {
///     const M: (i32, i32) = (1, 2);      // kg^(1/2)
///     const L: (i32, i32) = (0, 1);      // m^0
///     const T: (i32, i32) = (-1, 1);     // s^(-1)
///     const TEMP: (i32, i32) = (0, 1);   // K^0
/// }
/// ```
pub trait UnitDimension: 'static + Copy + Send + Sync + Eq + PartialEq + std::fmt::Debug {
    /// Mass dimension exponent as (numerator, denominator) for kg^(M_num/M_den)
    const M: (i32, i32);
    /// Length dimension exponent as (numerator, denominator) for m^(L_num/L_den)
    const L: (i32, i32);
    /// Time dimension exponent as (numerator, denominator) for s^(T_num/T_den)
    const T: (i32, i32);
    /// Temperature dimension exponent as (numerator, denominator) for K^(TEMP_num/TEMP_den)
    const TEMP: (i32, i32);

    /// The runtime unit constant derived from this type-level dimension.
    const UNIT: UnitDim = UnitDim::from_rational(
        Self::M.0,
        Self::M.1,
        Self::L.0,
        Self::L.1,
        Self::T.0,
        Self::T.1,
        Self::TEMP.0,
        Self::TEMP.1,
    );

    /// Convert to runtime UnitDim for error messages and serialization.
    fn to_runtime() -> UnitDim {
        Self::UNIT
    }

    /// Check if this dimension is compatible with another at compile time.
    fn IS_COMPATIBLE_WITH<Other: UnitDimension>() -> bool {
        Self::M == Other::M
            && Self::L == Other::L
            && Self::T == Other::T
            && Self::TEMP == Other::TEMP
    }
}

/// Canonical dimension type with const generics for rational exponents.
///
/// This type ensures that dimensions with the same exponents are the same Rust type,
/// enabling typed builder operations (like term addition) to work correctly.
///
/// The type parameters are the rational exponents for each base dimension:
/// - `M_NUM/M_DEN`: Mass exponent
/// - `L_NUM/L_DEN`: Length exponent  
/// - `T_NUM/T_DEN`: Time exponent
/// - `TEMP_NUM/TEMP_DEN`: Temperature exponent
///
/// # Example
///
/// ```rust,ignore
/// use cfd2_ir::solver::dimensions::Dim;
///
/// // Velocity = Length / Time = L^1 * T^-1
/// type Velocity = Dim<0, 1, 1, 1, -1, 1, 0, 1>;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim<
    const M_NUM: i32,
    const M_DEN: i32,
    const L_NUM: i32,
    const L_DEN: i32,
    const T_NUM: i32,
    const T_DEN: i32,
    const TEMP_NUM: i32,
    const TEMP_DEN: i32,
>;

impl<
        const M_NUM: i32,
        const M_DEN: i32,
        const L_NUM: i32,
        const L_DEN: i32,
        const T_NUM: i32,
        const T_DEN: i32,
        const TEMP_NUM: i32,
        const TEMP_DEN: i32,
    > UnitDimension for Dim<M_NUM, M_DEN, L_NUM, L_DEN, T_NUM, T_DEN, TEMP_NUM, TEMP_DEN>
{
    const M: (i32, i32) = (M_NUM, M_DEN);
    const L: (i32, i32) = (L_NUM, L_DEN);
    const T: (i32, i32) = (T_NUM, T_DEN);
    const TEMP: (i32, i32) = (TEMP_NUM, TEMP_DEN);
}

/// Marker trait for dimensions that are compatible (same exponents).
///
/// This is automatically implemented when two dimensions have identical exponents.
pub trait DimCompatible<D: UnitDimension>: UnitDimension {}

impl<A: UnitDimension> DimCompatible<A> for A {}

// ============================================================================
// Base Dimensions
// ============================================================================

/// Dimensionless quantity (all exponents are zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dimensionless;

impl UnitDimension for Dimensionless {
    const M: (i32, i32) = (0, 1);
    const L: (i32, i32) = (0, 1);
    const T: (i32, i32) = (0, 1);
    const TEMP: (i32, i32) = (0, 1);
}

/// Base dimension: Mass (kg)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mass;

impl UnitDimension for Mass {
    const M: (i32, i32) = (1, 1);
    const L: (i32, i32) = (0, 1);
    const T: (i32, i32) = (0, 1);
    const TEMP: (i32, i32) = (0, 1);
}

/// Base dimension: Length (m)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Length;

impl UnitDimension for Length {
    const M: (i32, i32) = (0, 1);
    const L: (i32, i32) = (1, 1);
    const T: (i32, i32) = (0, 1);
    const TEMP: (i32, i32) = (0, 1);
}

/// Base dimension: Time (s)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Time;

impl UnitDimension for Time {
    const M: (i32, i32) = (0, 1);
    const L: (i32, i32) = (0, 1);
    const T: (i32, i32) = (1, 1);
    const TEMP: (i32, i32) = (0, 1);
}

/// Base dimension: Temperature (K)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Temperature;

impl UnitDimension for Temperature {
    const M: (i32, i32) = (0, 1);
    const L: (i32, i32) = (0, 1);
    const T: (i32, i32) = (0, 1);
    const TEMP: (i32, i32) = (1, 1);
}

// ============================================================================
// Type Constructors
// ============================================================================

/// Multiplication of two dimensions: A * B
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MulDim<A: UnitDimension, B: UnitDimension> {
    _a: std::marker::PhantomData<A>,
    _b: std::marker::PhantomData<B>,
}

impl<A: UnitDimension, B: UnitDimension> UnitDimension for MulDim<A, B> {
    const M: (i32, i32) = add_rational(A::M, B::M);
    const L: (i32, i32) = add_rational(A::L, B::L);
    const T: (i32, i32) = add_rational(A::T, B::T);
    const TEMP: (i32, i32) = add_rational(A::TEMP, B::TEMP);
}

/// Division of two dimensions: A / B
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DivDim<A: UnitDimension, B: UnitDimension> {
    _a: std::marker::PhantomData<A>,
    _b: std::marker::PhantomData<B>,
}

impl<A: UnitDimension, B: UnitDimension> UnitDimension for DivDim<A, B> {
    const M: (i32, i32) = sub_rational(A::M, B::M);
    const L: (i32, i32) = sub_rational(A::L, B::L);
    const T: (i32, i32) = sub_rational(A::T, B::T);
    const TEMP: (i32, i32) = sub_rational(A::TEMP, B::TEMP);
}

/// Rational power of a dimension: A^(NUM/DEN)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowDim<A: UnitDimension, const NUM: i32, const DEN: i32> {
    _a: std::marker::PhantomData<A>,
}

impl<A: UnitDimension, const NUM: i32, const DEN: i32> UnitDimension for PowDim<A, NUM, DEN> {
    const M: (i32, i32) = mul_rational(A::M, (NUM, DEN));
    const L: (i32, i32) = mul_rational(A::L, (NUM, DEN));
    const T: (i32, i32) = mul_rational(A::T, (NUM, DEN));
    const TEMP: (i32, i32) = mul_rational(A::TEMP, (NUM, DEN));
}

/// Square root of a dimension: sqrt(A) = A^(1/2)
pub type SqrtDim<A> = PowDim<A, 1, 2>;

/// Inverse square root: 1/sqrt(A) = A^(-1/2)
pub type InvSqrtDim<A> = PowDim<A, -1, 2>;

// ============================================================================
// Rational Arithmetic (const fn)
// ============================================================================

/// Add two rational numbers: a/b + c/d = (ad + bc) / bd
const fn add_rational(a: (i32, i32), b: (i32, i32)) -> (i32, i32) {
    let (an, ad) = a;
    let (bn, bd) = b;
    normalize_rational(an * bd + bn * ad, ad * bd)
}

/// Subtract two rational numbers: a/b - c/d = (ad - bc) / bd
const fn sub_rational(a: (i32, i32), b: (i32, i32)) -> (i32, i32) {
    let (an, ad) = a;
    let (bn, bd) = b;
    normalize_rational(an * bd - bn * ad, ad * bd)
}

/// Multiply two rational numbers: (a/b) * (c/d) = ac / bd
const fn mul_rational(a: (i32, i32), b: (i32, i32)) -> (i32, i32) {
    let (an, ad) = a;
    let (bn, bd) = b;
    normalize_rational(an * bn, ad * bd)
}

/// Normalize a rational number (reduce by GCD, fix sign)
const fn normalize_rational(num: i32, den: i32) -> (i32, i32) {
    if num == 0 {
        return (0, 1);
    }

    // Handle sign - denominator should always be positive
    let (mut n, mut d) = (num, den);
    if d < 0 {
        n = -n;
        d = -d;
    }

    // Compute GCD and reduce
    let g = gcd_i32(n.abs(), d.abs());
    (n / g, d / g)
}

/// Greatest common divisor (Euclidean algorithm)
const fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    // Ensure non-negative
    a = if a < 0 { -a } else { a };
    b = if b < 0 { -b } else { b };

    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }

    if a == 0 {
        1
    } else {
        a
    }
}

// ============================================================================
// Derived Dimensions (Typed Aliases)
// ============================================================================

/// Area = Length^2
pub type Area = PowDim<Length, 2, 1>;

/// Volume = Length^3
pub type Volume = PowDim<Length, 3, 1>;

/// Inverse time = 1/Time
pub type InvTime = PowDim<Time, -1, 1>;

/// Velocity = Length / Time
pub type Velocity = DivDim<Length, Time>;

/// Acceleration = Velocity / Time = Length / Time^2
pub type Acceleration = DivDim<Velocity, Time>;

/// Force = Mass * Acceleration = Mass * Length / Time^2
pub type Force = MulDim<Mass, Acceleration>;

/// Pressure = Force / Area = Mass / (Length * Time^2)
pub type Pressure = DivDim<Force, Area>;

/// Energy = Force * Length
pub type Energy = MulDim<Force, Length>;

/// Power = Energy / Time
pub type Power = DivDim<Energy, Time>;

/// Density = Mass / Volume
pub type Density = DivDim<Mass, Volume>;

/// Dynamic Viscosity = Pressure * Time
pub type DynamicViscosity = MulDim<Pressure, Time>;

/// Kinematic Viscosity = Dynamic Viscosity / Density = Area / Time
pub type KinematicViscosity = DivDim<DynamicViscosity, Density>;

/// Mass flux = Mass / Time (integrated over face)
pub type MassFlux = DivDim<Mass, Time>;

/// Momentum density = Density * Velocity
pub type MomentumDensity = MulDim<Density, Velocity>;

/// Energy density (same as Pressure for compressible flow)
pub type EnergyDensity = Pressure;

/// Pressure gradient = Pressure / Length
pub type PressureGradient = DivDim<Pressure, Length>;

/// D_P = Volume * Time / Mass (pressure-correction mobility-like coefficient)
pub type D_P = DivDim<MulDim<Volume, Time>, Mass>;

/// Diffusivity = Area / Time (e.g., thermal diffusivity, kinematic viscosity)
pub type Diffusivity = DivDim<Area, Time>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::units::UnitDim;

    #[test]
    fn base_dimensions_have_correct_exponents() {
        assert_eq!(Mass::M, (1, 1));
        assert_eq!(Mass::L, (0, 1));
        assert_eq!(Mass::T, (0, 1));
        assert_eq!(Mass::TEMP, (0, 1));

        assert_eq!(Length::M, (0, 1));
        assert_eq!(Length::L, (1, 1));
        assert_eq!(Length::T, (0, 1));

        assert_eq!(Time::M, (0, 1));
        assert_eq!(Time::L, (0, 1));
        assert_eq!(Time::T, (1, 1));

        assert_eq!(Temperature::M, (0, 1));
        assert_eq!(Temperature::TEMP, (1, 1));
    }

    #[test]
    fn dimensionless_has_zero_exponents() {
        assert_eq!(Dimensionless::M, (0, 1));
        assert_eq!(Dimensionless::L, (0, 1));
        assert_eq!(Dimensionless::T, (0, 1));
        assert_eq!(Dimensionless::TEMP, (0, 1));
    }

    #[test]
    fn dimension_multiplication_adds_exponents() {
        type TestMul = MulDim<Mass, Length>;
        assert_eq!(TestMul::M, (1, 1));
        assert_eq!(TestMul::L, (1, 1));
        assert_eq!(TestMul::T, (0, 1));
    }

    #[test]
    fn dimension_division_subtracts_exponents() {
        type TestDiv = DivDim<Length, Time>;
        assert_eq!(TestDiv::M, (0, 1));
        assert_eq!(TestDiv::L, (1, 1));
        assert_eq!(TestDiv::T, (-1, 1));
    }

    #[test]
    fn velocity_dimension_is_length_over_time() {
        assert_eq!(Velocity::M, (0, 1));
        assert_eq!(Velocity::L, (1, 1));
        assert_eq!(Velocity::T, (-1, 1));
    }

    #[test]
    fn pressure_dimension_is_correct() {
        // Pressure = Force / Area = (Mass * Length / Time^2) / Length^2
        //          = Mass / (Length * Time^2)
        assert_eq!(Pressure::M, (1, 1));
        assert_eq!(Pressure::L, (-1, 1));
        assert_eq!(Pressure::T, (-2, 1));
    }

    #[test]
    fn dimension_powers_work() {
        type LengthCubed = PowDim<Length, 3, 1>;
        assert_eq!(LengthCubed::L, (3, 1));

        type InvTime = PowDim<Time, -1, 1>;
        assert_eq!(InvTime::T, (-1, 1));
    }

    #[test]
    fn sqrt_dimension_halves_exponents() {
        type SqrtLength = SqrtDim<Length>;
        assert_eq!(SqrtLength::L, (1, 2));
        assert_eq!(SqrtLength::M, (0, 1));

        // Verify that sqrt(Length)^2 = Length
        type SqrtLengthSquared = MulDim<SqrtLength, SqrtLength>;
        assert_eq!(SqrtLengthSquared::L, (1, 1));
    }

    #[test]
    fn rational_arithmetic_reduces_fractions() {
        // Test that 1/2 + 1/2 = 1/1 (not 2/2)
        type HalfLength = PowDim<Length, 1, 2>;
        type Sum = MulDim<HalfLength, HalfLength>;
        assert_eq!(Sum::L, (1, 1));
    }

    #[test]
    fn runtime_conversion_matches_compile_time() {
        let runtime = Velocity::to_runtime();
        assert_eq!(runtime, UnitDim::new(0, 1, -1));
    }

    #[test]
    fn compatibility_check_works() {
        assert!(Velocity::IS_COMPATIBLE_WITH::<Velocity>());
        assert!(Dimensionless::IS_COMPATIBLE_WITH::<Dimensionless>());
        assert!(!Velocity::IS_COMPATIBLE_WITH::<Pressure>());
    }

    #[test]
    fn derived_dimensions_match_si_constants() {
        use crate::solver::units::si;

        // Verify that type-level dimensions match runtime SI constants
        assert_eq!(Dimensionless::UNIT, si::DIMENSIONLESS);
        assert_eq!(Mass::UNIT, si::MASS);
        assert_eq!(Length::UNIT, si::LENGTH);
        assert_eq!(Time::UNIT, si::TIME);
        assert_eq!(Temperature::UNIT, si::TEMPERATURE);

        assert_eq!(Area::UNIT, si::AREA);
        assert_eq!(Volume::UNIT, si::VOLUME);
        assert_eq!(Velocity::UNIT, si::VELOCITY);
        assert_eq!(Density::UNIT, si::DENSITY);
        assert_eq!(Pressure::UNIT, si::PRESSURE);
        assert_eq!(DynamicViscosity::UNIT, si::DYNAMIC_VISCOSITY);
        assert_eq!(Power::UNIT, si::POWER);
        assert_eq!(MassFlux::UNIT, si::MASS_FLUX);
        assert_eq!(MomentumDensity::UNIT, si::MOMENTUM_DENSITY);
        assert_eq!(EnergyDensity::UNIT, si::ENERGY_DENSITY);
        assert_eq!(PressureGradient::UNIT, si::PRESSURE_GRADIENT);
        assert_eq!(D_P::UNIT, si::D_P);
    }

    #[test]
    fn d_p_dimension_is_correct() {
        // D_P = Volume * Time / Mass = m^3 * s / kg
        // M: -1, L: 3, T: 1
        assert_eq!(D_P::M, (-1, 1));
        assert_eq!(D_P::L, (3, 1));
        assert_eq!(D_P::T, (1, 1));
    }

    #[test]
    fn pressure_gradient_dimension_is_correct() {
        // PressureGradient = Pressure / Length = (kg/(m*s^2)) / m = kg/(m^2*s^2)
        // M: 1, L: -2, T: -2
        assert_eq!(PressureGradient::M, (1, 1));
        assert_eq!(PressureGradient::L, (-2, 1));
        assert_eq!(PressureGradient::T, (-2, 1));
    }
}
