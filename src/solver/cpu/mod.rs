//! Experimental CPU backend for the solver.
//!
//! The GPU backend compiles the model's kernel IR (`cfd2_ir`) to WGSL and runs it
//! on wgpu. The CPU backend reuses *everything up to and including* the typed
//! `KernelProgram` IR and replaces only the runtime: instead of emitting WGSL and
//! dispatching compute passes, it executes the same `Stmt`/`Expr` AST directly on
//! CPU-side `Vec<f32>`/`Vec<u32>` buffers.
//!
//! Staging (see plan):
//! - **Phase 1 (this module today):** a tree-walking [`interpreter`] over the
//!   kernel AST — a GPU-free *reference* executor validated against the existing
//!   MMS order tests.
//! - **Phase 2:** runtime-switchable multithreading behind a single
//!   parallel-for abstraction (not committed to rayon).
//! - **Phase 3:** an IR→Rust transpiler emitting scalar + SIMD kernel variants,
//!   selected at runtime.

pub mod interpreter;
pub mod linalg;
pub mod lowering;
pub mod parallel;
pub mod solver;
pub mod transpile;

pub use solver::CpuSolver;

/// Runtime-selectable execution mode for the CPU backend.
///
/// Both multithreading and SIMD are intended to be switchable at runtime (the
/// user's requirement), so these live in a value passed at construction rather
/// than in cargo features. Phase 1 only honours single-threaded scalar
/// execution; the other fields are wired in later phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackendConfig {
    /// Worker threads for per-cell/face dispatch. `1` = serial. (Phase 2.)
    pub threads: usize,
    /// Use SIMD kernel variants when available. (Phase 3.)
    pub simd: bool,
}

impl Default for CpuBackendConfig {
    fn default() -> Self {
        // Phase-1 default: the deterministic, always-correct reference path.
        Self {
            threads: 1,
            simd: false,
        }
    }
}
