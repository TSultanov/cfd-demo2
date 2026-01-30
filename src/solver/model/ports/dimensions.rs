//! Compile-time physical dimension tracking for ports.
//!
//! This module provides type-level representation of physical dimensions,
//! enabling compile-time verification of dimensional consistency.
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::solver::model::ports::dimensions::*;
//!
//!<D: UnitDimension> // Velocity = Length / Time
//! type Velocity = DivDim<Length, Time>;
//!
//!<D: UnitDimension> // Pressure = Force / Area
//! type Pressure = DivDim<Force, Area>;
//! ```

use crate::solver::units::UnitDim;

/// Trait for compile-time physical dimensions.
///
/// Implement this trait for types that represent physical dimensions.
/// The associated constants provide the dimension exponents for each
/// SI base unit.
pub trait UnitDimension: 'static + Copy + Send + Sync + Eq + PartialEq + std::fmt::Debug {
    /// Mass dimension exponent (kg^M)
    const M: i8;
    /// Length dimension exponent (m^L)
    const L: i8;
    /// Time dimension exponent (s^T)
    const T: i8;
    /// Temperature dimension exponent (K^TEMP)
    const TEMP: i8;

    /// Check if this dimension is compatible with another at compile time.
    ///
    /// This is used by the `assert_dim_compatible!` macro.
    fn IS_COMPATIBLE_WITH<Other: UnitDimension>() -> bool {
        Self::M == Other::M
            && Self::L == Other::L
            && Self::T == Other::T
            && Self::TEMP == Other::TEMP
    }

    /// Convert to runtime UnitDim for error messages and serialization.
    fn to_runtime() -> UnitDim {
        UnitDim::new_with_temp(Self::M, Self::L, Self::T, Self::TEMP)
    }
}

/// Marker trait for dimensions that are compatible (same exponents).
///
/// This is automatically implemented when two dimensions have identical exponents.
pub trait DimCompatible<D: UnitDimension>: UnitDimension {}

impl<A: UnitDimension> DimCompatible<A> for A {}

/// Dimensionless quantity (all exponents are zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dimensionless;

impl UnitDimension for Dimensionless {
    const M: i8 = 0;
    const L: i8 = 0;
    const T: i8 = 0;
    const TEMP: i8 = 0;
}

/// Base dimension: Mass (kg)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mass;

impl UnitDimension for Mass {
    const M: i8 = 1;
    const L: i8 = 0;
    const T: i8 = 0;
    const TEMP: i8 = 0;
}

/// Base dimension: Length (m)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Length;

impl UnitDimension for Length {
    const M: i8 = 0;
    const L: i8 = 1;
    const T: i8 = 0;
    const TEMP: i8 = 0;
}

/// Base dimension: Time (s)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Time;

impl UnitDimension for Time {
    const M: i8 = 0;
    const L: i8 = 0;
    const T: i8 = 1;
    const TEMP: i8 = 0;
}

/// Base dimension: Temperature (K)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Temperature;

impl UnitDimension for Temperature {
    const M: i8 = 0;
    const L: i8 = 0;
    const T: i8 = 0;
    const TEMP: i8 = 1;
}

/// Multiplication of two dimensions: A * B
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MulDim<A: UnitDimension, B: UnitDimension> {
    _a: std::marker::PhantomData<A>,
    _b: std::marker::PhantomData<B>,
}

impl<A: UnitDimension, B: UnitDimension> UnitDimension for MulDim<A, B> {
    const M: i8 = A::M + B::M;
    const L: i8 = A::L + B::L;
    const T: i8 = A::T + B::T;
    const TEMP: i8 = A::TEMP + B::TEMP;
}

/// Division of two dimensions: A / B
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DivDim<A: UnitDimension, B: UnitDimension> {
    _a: std::marker::PhantomData<A>,
    _b: std::marker::PhantomData<B>,
}

impl<A: UnitDimension, B: UnitDimension> UnitDimension for DivDim<A, B> {
    const M: i8 = A::M - B::M;
    const L: i8 = A::L - B::L;
    const T: i8 = A::T - B::T;
    const TEMP: i8 = A::TEMP - B::TEMP;
}

/// Integer power of a dimension: A^N
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowDim<A: UnitDimension, const N: i8> {
    _a: std::marker::PhantomData<A>,
}

impl<A: UnitDimension, const N: i8> UnitDimension for PowDim<A, N> {
    const M: i8 = A::M * N;
    const L: i8 = A::L * N;
    const T: i8 = A::T * N;
    const TEMP: i8 = A::TEMP * N;
}

/// Square root of a dimension: sqrt(A) = A^(1/2)
pub type SqrtDim<A> = PowDim<A, 1>;

// Pre-defined common dimensions
/// Area = Length^2
pub type Area = PowDim<Length, 2>;

/// Volume = Length^3
pub type Volume = PowDim<Length, 3>;

/// Velocity = Length / Time
pub type Velocity = DivDim<Length, Time>;

/// Acceleration = Velocity / Time = Length / Time^2
pub type Acceleration = DivDim<Velocity, Time>;

/// Force = Mass * Acceleration
pub type Force = MulDim<Mass, Acceleration>;

/// Pressure = Force / Area
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_dimensions_have_correct_exponents() {
        assert_eq!(Mass::M, 1);
        assert_eq!(Mass::L, 0);
        assert_eq!(Mass::T, 0);

        assert_eq!(Length::M, 0);
        assert_eq!(Length::L, 1);
        assert_eq!(Length::T, 0);

        assert_eq!(Time::M, 0);
        assert_eq!(Time::L, 0);
        assert_eq!(Time::T, 1);
    }

    #[test]
    fn dimension_multiplication_adds_exponents() {
        type TestMul = MulDim<Mass, Length>;
        assert_eq!(TestMul::M, 1);
        assert_eq!(TestMul::L, 1);
        assert_eq!(TestMul::T, 0);
    }

    #[test]
    fn dimension_division_subtracts_exponents() {
        type TestDiv = DivDim<Length, Time>;
        assert_eq!(TestDiv::M, 0);
        assert_eq!(TestDiv::L, 1);
        assert_eq!(TestDiv::T, -1);
    }

    #[test]
    fn velocity_dimension_is_length_over_time() {
        assert_eq!(Velocity::M, 0);
        assert_eq!(Velocity::L, 1);
        assert_eq!(Velocity::T, -1);
    }

    #[test]
    fn pressure_dimension_is_correct() {
        // Pressure = Force / Area = (Mass * Length / Time^2) / Length^2
        //          = Mass / (Length * Time^2)
        assert_eq!(Pressure::M, 1);
        assert_eq!(Pressure::L, -1);
        assert_eq!(Pressure::T, -2);
    }

    #[test]
    fn dimension_powers_work() {
        type LengthCubed = PowDim<Length, 3>;
        assert_eq!(LengthCubed::L, 3);

        type InvTime = PowDim<Time, -1>;
        assert_eq!(InvTime::T, -1);
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
}
