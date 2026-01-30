//! Type-level physical dimensions with rational exponents.
//!
//! This module re-exports the canonical dimension system from `cfd2_ir::solver::dimensions`,
//! providing a unified type-level dimension system for use in ports and throughout the solver.
//!
//! # Example
//!
//! ```rust,ignore
//! use cfd2::solver::dimensions::*;
//!
//! // Use typed dimensions in port definitions
//! let port: FieldPort<Pressure, Scalar> = ...;
//! ```

pub use cfd2_ir::solver::dimensions::*;
