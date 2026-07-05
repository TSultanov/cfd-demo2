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

/// Scalar precision of the coupled linear solve (see `linalg::Real`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuPrecision {
    /// f64 internals — the bit-identical reference (default).
    #[default]
    F64,
    /// f32 internals — mirrors the GPU's arithmetic and halves vector
    /// bandwidth on the memory-bound solve phases. Reductions still
    /// accumulate in f64 (deterministic across thread counts); results
    /// differ from f64 at rounding level, validated by the tolerance suites.
    F32,
}

/// Runtime-selectable execution configuration for the CPU backend. Engine,
/// threading and SIMD are runtime knobs (not cargo features).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackendConfig {
    /// Kernel engine: interpreter (default) or transpiled compiled-Rust.
    pub engine: CpuEngine,
    /// Worker threads for per-cell/face dispatch. `1` = serial.
    pub threads: usize,
    /// Use the SIMD paths in the linear solve (vectorized block matvec,
    /// mixed-precision inner storage; see the linalg docs).
    pub simd: bool,
    /// Coupled linear-solve precision (`CFD2_CPU_PRECISION=f32|f64`).
    pub precision: CpuPrecision,
}

impl Default for CpuBackendConfig {
    fn default() -> Self {
        // Default: the deterministic, always-correct reference path.
        Self {
            engine: CpuEngine::Interpreter,
            threads: 1,
            simd: false,
            precision: CpuPrecision::F64,
        }
    }
}
