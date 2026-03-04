use std::time::Duration;

use super::{CategoryStats, MemoryDomain, MemoryStats, ProfileCategory};

#[derive(Debug, Default)]
pub struct ProfilingStats;

impl ProfilingStats {
    pub fn new() -> Self {
        Self
    }

    pub fn enable(&self) {}

    pub fn disable(&self) {}

    pub fn is_enabled(&self) -> bool {
        false
    }

    pub fn start_session(&self) {}

    pub fn end_session(&self) {}

    pub fn reset(&self) {}

    pub fn increment_iteration(&self) {}

    pub fn record(&self, _category: ProfileCategory, _duration: Duration, _bytes: u64) {}

    pub fn record_location(
        &self,
        _location: &str,
        _category: ProfileCategory,
        _duration: Duration,
        _bytes: u64,
    ) {
    }

    pub fn record_cpu_alloc(&self, _location: &str, _bytes: u64) {}

    pub fn record_cpu_free(&self, _location: &str, _bytes: u64) {}

    pub fn record_gpu_alloc(&self, _location: &str, _bytes: u64) {}

    pub fn record_gpu_free(&self, _location: &str, _bytes: u64) {}

    pub fn get_stats(&self, _category: ProfileCategory) -> CategoryStats {
        CategoryStats::default()
    }

    pub fn get_all_stats(&self) -> Vec<(ProfileCategory, CategoryStats)> {
        vec![
            (ProfileCategory::GpuRead, CategoryStats::default()),
            (ProfileCategory::GpuWrite, CategoryStats::default()),
            (ProfileCategory::GpuSync, CategoryStats::default()),
            (ProfileCategory::GpuDispatch, CategoryStats::default()),
            (ProfileCategory::CpuCompute, CategoryStats::default()),
            (
                ProfileCategory::GpuResourceCreation,
                CategoryStats::default(),
            ),
            (ProfileCategory::Other, CategoryStats::default()),
        ]
    }

    pub fn get_location_stats(&self) -> Vec<(String, CategoryStats)> {
        Vec::new()
    }

    pub fn get_memory_stats(&self) -> (MemoryStats, MemoryStats) {
        (MemoryStats::default(), MemoryStats::default())
    }

    pub fn get_memory_location_stats(&self, _domain: MemoryDomain) -> Vec<(String, MemoryStats)> {
        Vec::new()
    }

    pub fn get_session_total(&self) -> Duration {
        Duration::ZERO
    }

    pub fn get_iteration_count(&self) -> u64 {
        0
    }

    pub fn print_report(&self) {}
}

pub struct ProfileTimer<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> ProfileTimer<'a> {
    pub fn new(_stats: &'a ProfilingStats, _category: ProfileCategory) -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_location(self, _location: &'a str) -> Self {
        self
    }

    pub fn with_bytes(self, _bytes: u64) -> Self {
        self
    }
}

#[macro_export]
macro_rules! profile_scope {
    ($stats:expr, $category:expr, $location:expr) => {};
    ($stats:expr, $category:expr, $location:expr, $bytes:expr) => {};
}
