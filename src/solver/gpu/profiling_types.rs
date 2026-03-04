//! Shared types for the profiling subsystem.
//!
//! These types are used by both `profiling_enabled` and `profiling_disabled` implementations,
//! avoiding duplication across the two conditional-compilation paths.

use std::time::Duration;

/// Categories of operations that can be profiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfileCategory {
    /// GPU buffer read operations (GPU -> CPU transfer)
    GpuRead,
    /// GPU buffer write operations (CPU -> GPU transfer)
    GpuWrite,
    /// CPU waiting for GPU to complete (device.poll)
    GpuSync,
    /// GPU compute dispatch (command encoder submission)
    GpuDispatch,
    /// CPU-side computation (e.g., norm reduction, Gram-Schmidt)
    CpuCompute,
    /// Creating bind groups and other GPU resources
    GpuResourceCreation,
    /// Other overhead
    Other,
}

impl ProfileCategory {
    pub fn name(&self) -> &'static str {
        match self {
            ProfileCategory::GpuRead => "GPU -> CPU Transfer",
            ProfileCategory::GpuWrite => "CPU -> GPU Transfer",
            ProfileCategory::GpuSync => "GPU Sync Wait",
            ProfileCategory::GpuDispatch => "GPU Dispatch",
            ProfileCategory::CpuCompute => "CPU Compute",
            ProfileCategory::GpuResourceCreation => "GPU Resource Creation",
            ProfileCategory::Other => "Other",
        }
    }
}

/// Memory domains for allocation profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    Cpu,
    Gpu,
}

impl MemoryDomain {
    pub fn name(&self) -> &'static str {
        match self {
            MemoryDomain::Cpu => "CPU",
            MemoryDomain::Gpu => "GPU",
        }
    }
}

/// Allocation / deallocation statistics.
#[derive(Debug, Default, Clone)]
pub struct MemoryStats {
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub free_bytes: u64,
    pub free_count: u64,
    pub max_alloc_request: u64,
}

impl MemoryStats {
    pub fn record_alloc(&mut self, bytes: u64) {
        self.alloc_bytes += bytes;
        self.alloc_count += 1;
        self.max_alloc_request = self.max_alloc_request.max(bytes);
    }

    pub fn record_free(&mut self, bytes: u64) {
        self.free_bytes += bytes;
        self.free_count += 1;
    }

    pub fn net_bytes(&self) -> i64 {
        self.alloc_bytes as i64 - self.free_bytes as i64
    }
}

/// Detailed profiling statistics for a single category.
#[derive(Debug, Default, Clone)]
pub struct CategoryStats {
    pub total_time: Duration,
    pub call_count: u64,
    pub min_time: Duration,
    pub max_time: Duration,
    /// Total bytes transferred (for transfer categories)
    pub total_bytes: u64,
}

impl CategoryStats {
    pub fn new() -> Self {
        Self {
            total_time: Duration::ZERO,
            call_count: 0,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            total_bytes: 0,
        }
    }

    pub fn record(&mut self, duration: Duration, bytes: u64) {
        self.total_time += duration;
        self.call_count += 1;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        self.total_bytes += bytes;
    }

    pub fn avg_time(&self) -> Duration {
        if self.call_count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.call_count as u32
        }
    }

    pub fn throughput_mb_per_sec(&self) -> f64 {
        if self.total_time.as_secs_f64() == 0.0 {
            0.0
        } else {
            (self.total_bytes as f64 / 1_000_000.0) / self.total_time.as_secs_f64()
        }
    }
}
