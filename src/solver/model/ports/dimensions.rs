// Compile-time physical dimension tracking for ports.
//
// This module re-exports the canonical type-level dimension system from
// `cfd2_ir::solver::dimensions`. The canonical system provides:
//
// - Rational exponents for all base dimensions (M, L, T, TEMP)
// - Type constructors: [`MulDim`], [`DivDim`], [`PowDim`], [`SqrtDim`]
// - Typed aliases for common SI dimensions (including ports-relevant ones
//   like [`PressureGradient`], [`D_P`], [`MassFlux`], [`MomentumDensity`],
//   [`EnergyDensity`], [`InvTime`], etc.)
// - Conversion to runtime [`UnitDim`](crate::solver::units::UnitDim)
//
// # Migration Note
//
// This module previously contained a duplicate dimension system using `i8` exponents,
// and later had ports-specific aliases for CFD-related dimensions. All dimension
// aliases have now been moved to the canonical dimension system in `cfd2_ir`.
// This module now only provides ports-specific additions like [`AnyDimension`].
//
// # Example
//
// ```rust,ignore
// use crate::solver::model::ports::dimensions::*;
//
// // Velocity = Length / Time
// type MyVelocity = DivDim<Length, Time>;
//
// // Pressure = Force / Area
// type MyPressure = DivDim<Force, Area>;
//
// // Square root dimension (requires rational exponents)
// type SqrtPressure = SqrtDim<Pressure>;
//
// // Pressure gradient (available from canonical re-export)
// type PG = PressureGradient;
// ```

// Re-export everything from the canonical dimension system
pub use crate::solver::dimensions::*;

/// Any dimension - escape hatch for dynamic/unknown dimensions.
///
/// This type represents "skip unit enforcement" in port registration.
/// When used as the dimension type for field/param registration, unit
/// checks are bypassed, allowing fields with any runtime unit to be
/// registered without validation.
///
/// This is distinct from `Dimensionless` (which has a specific unit of 1).
/// Use `AnyDimension` when the actual dimension is not known at compile time
/// or when you explicitly want to opt out of dimension checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnyDimension;

// Use a large-but-safe sentinel value. This avoids accidental compatibility
// with real dimensions, while still allowing const arithmetic (e.g. MulDim)
// without overflowing `i32`.
const ANY_DIMENSION_EXPONENT: (i32, i32) = (1000, 1);

impl UnitDimension for AnyDimension {
    // Use a unique sentinel value that won't match any real dimension
    const M: (i32, i32) = ANY_DIMENSION_EXPONENT;
    const L: (i32, i32) = ANY_DIMENSION_EXPONENT;
    const T: (i32, i32) = ANY_DIMENSION_EXPONENT;
    const TEMP: (i32, i32) = ANY_DIMENSION_EXPONENT;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::units::UnitDim;

    #[test]
    fn canonical_dimensions_re_exported() {
        // Verify all base dimensions are available
        let _: UnitDim = Dimensionless::to_runtime();
        let _: UnitDim = Mass::to_runtime();
        let _: UnitDim = Length::to_runtime();
        let _: UnitDim = Time::to_runtime();
        let _: UnitDim = Temperature::to_runtime();
    }

    #[test]
    fn derived_dimensions_work() {
        // Test that type constructors work correctly
        type TestVel = DivDim<Length, Time>;
        assert_eq!(TestVel::to_runtime(), UnitDim::new(0, 1, -1));

        type TestPressure = DivDim<Force, Area>;
        assert_eq!(TestPressure::to_runtime(), UnitDim::new(1, -1, -2));
    }

    #[test]
    fn sqrt_dimension_requires_rational_exponents() {
        // SqrtDim should produce rational exponents
        assert_eq!(SqrtDim::<Length>::L, (1, 2));

        // Verify that sqrt(Length)^2 = Length
        type SqrtLenSq = MulDim<SqrtDim<Length>, SqrtDim<Length>>;
        assert_eq!(SqrtLenSq::L, (1, 1));
    }

    #[test]
    fn port_specific_dimensions_match_si() {
        use crate::solver::units::si;

        // These are re-exported from the canonical dimension system
        assert_eq!(PressureGradient::UNIT, si::PRESSURE_GRADIENT);
        assert_eq!(D_P::UNIT, si::D_P);
        assert_eq!(MassFlux::UNIT, si::MASS_FLUX);
        assert_eq!(MomentumDensity::UNIT, si::MOMENTUM_DENSITY);
        assert_eq!(EnergyDensity::UNIT, si::ENERGY_DENSITY);
        assert_eq!(InvTime::UNIT, si::INV_TIME);
    }

    #[test]
    fn compatibility_trait_works() {
        fn assert_compatible<A: UnitDimension, B: UnitDimension>()
        where
            A: DimCompatible<B>,
        {
        }

        assert_compatible::<Velocity, Velocity>();
        assert_compatible::<Pressure, Pressure>();
    }
}
