//! Integration tests for port derive macros.

use crate::solver::model::ports::*;

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
    assert!(Velocity::IS_COMPATIBLE_WITH::<Velocity>());
    assert!(Dimensionless::IS_COMPATIBLE_WITH::<Dimensionless>());
    assert!(!Velocity::IS_COMPATIBLE_WITH::<Pressure>());
}
