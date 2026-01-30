//! Integration tests for cfd2_macros derive macros.

use trybuild::TestCases;

#[test]
fn test_port_set_pass() {
    let t = TestCases::new();
    t.pass("tests/ui/port_set_pass.rs");
}

#[test]
fn test_port_set_fail() {
    let t = TestCases::new();
    t.compile_fail("tests/ui/port_set_fail/*.rs");
}
