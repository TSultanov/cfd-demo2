//! Integration tests for port derive macros.

use crate::solver::model::ports::*;
use crate::solver::units::si;

// Tests from inline module
#[test]
fn port_id_uniqueness() {
    let id1 = PortId::new(1);
    let id2 = PortId::new(2);
    let id1_copy = PortId::new(1);

    assert_ne!(id1, id2);
    assert_eq!(id1, id1_copy);
    assert_eq!(id1.as_u32(), 1);
}

#[test]
fn dimension_compatibility() {
    // These should work - same dimensions
    assert!(Velocity::IS_COMPATIBLE_WITH::<Velocity>());

    // These should also work - both dimensionless
    assert!(Dimensionless::IS_COMPATIBLE_WITH::<Dimensionless>());

    // Different dimensions should not be compatible
    assert!(!Velocity::IS_COMPATIBLE_WITH::<Pressure>());
}

// Note: Derive macros use ::cfd2:: paths which don't work in tests.
// The macro infrastructure is in place - testing will be done through
// actual module migrations in Phase 2.

#[test]
fn macro_infrastructure_placeholder() {
    // Placeholder test to verify tests module compiles
    // Real testing happens through module migrations
}
