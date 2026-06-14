//! Build-time transpiled CPU kernels (compiled Rust).
//!
//! `build.rs` emits one Rust function per supported model kernel via
//! `cfd2_codegen::solver::codegen::rust_emit`, plus a [`lookup`] dispatch, into
//! `$OUT_DIR/cpu_transpiled_kernels.rs`. The generated functions target the
//! [`super::transpile_rt`] prelude and operate on the shared atomic [`Buffers`].
//!
//! Kernels without a generated variant return `None` from [`lookup`]; the
//! `CpuSolver` then interprets them.

#[allow(unused_imports)]
use crate::solver::cpu::interpreter::Buffers;
#[allow(unused_imports)]
use crate::solver::cpu::transpile_rt::*;
#[allow(unused_imports)]
use crate::solver::gpu::structs::GpuConstants;

include!(concat!(env!("OUT_DIR"), "/cpu_transpiled_kernels.rs"));
