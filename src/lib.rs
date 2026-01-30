#[cfg(feature = "meshgen")]
pub mod meshgen;
pub mod solver;
pub mod trace;
#[cfg(feature = "ui")]
pub mod ui;

// Allow macros to use `::cfd2::...` paths within this crate
extern crate self as cfd2;
