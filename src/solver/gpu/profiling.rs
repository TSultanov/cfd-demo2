#[cfg(not(feature = "profiling"))]
#[path = "profiling_disabled.rs"]
mod profiling_disabled;
#[cfg(feature = "profiling")]
#[path = "profiling_enabled.rs"]
mod profiling_enabled;

#[cfg(not(feature = "profiling"))]
pub use profiling_disabled::*;
#[cfg(feature = "profiling")]
pub use profiling_enabled::*;
