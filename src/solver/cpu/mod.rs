//! Experimental CPU backend for the solver.
//!
//! The GPU backend compiles the model's kernel IR (`cfd2_ir`) to WGSL and runs it
//! on wgpu. The CPU backend reuses *everything up to and including* the typed
//! `KernelProgram` IR and replaces only the runtime.
//!
//! Two execution engines share the same buffers, schedule, BCs and CPU linear
//! solver (see [`solver::CpuSolver`]):
//! - **Interpreter** ([`interpreter`]): tree-walks the kernel AST at runtime.
//!   GPU-free reference path; no codegen/compile step; tracks the IR automatically.
//! - **Transpiled** ([`generated`]): runs compiled Rust kernels emitted at build
//!   time by `cfd2_codegen::solver::codegen::rust_emit` (targeting the
//!   [`transpile_rt`] prelude). Native speed; falls back to the interpreter for
//!   any kernel without a generated variant.
//!
//! Multithreading ([`parallel`]) and the linear-solve SIMD path ([`linalg`]) are
//! selected at runtime via [`CpuBackendConfig`].

pub mod amg;
pub mod generated;
pub mod interpreter;
pub mod linalg;
pub mod lowering;
pub mod parallel;
pub(crate) mod pool;
pub mod solver;
pub mod transpile_rt;

pub use solver::CpuSolver;

/// Kernel execution engine for the CPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuEngine {
    /// Tree-walking interpreter over the kernel IR (reference path).
    #[default]
    Interpreter,
    /// Compiled Rust kernels (build-time transpiled), interpreter fallback.
    Transpiled,
}

/// Runtime-selectable execution configuration for the CPU backend. All knobs are
/// runtime values (not cargo features), per the user's requirement that engine,
/// threading and SIMD all be switchable at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackendConfig {
    /// Kernel engine: interpreter (default) or transpiled compiled-Rust.
    pub engine: CpuEngine,
    /// Worker threads for per-cell/face dispatch. `1` = serial.
    pub threads: usize,
    /// Use the SIMD path for the linear-solve reductions.
    pub simd: bool,
}

impl Default for CpuBackendConfig {
    fn default() -> Self {
        // Default: the deterministic, always-correct reference path.
        Self {
            engine: CpuEngine::Interpreter,
            threads: 1,
            simd: false,
        }
    }
}
