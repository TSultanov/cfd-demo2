// GPU-CPU Communication Profiling Module
//
// This module provides detailed profiling of GPU-CPU data transfers and synchronization
// to identify performance bottlenecks and opportunities for GPU offloading.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Categories of operations that can be profiled
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

/// Detailed profiling statistics for a single category
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

/// Detailed profiling statistics for the solver
#[derive(Debug)]
pub struct ProfilingStats {
    enabled: AtomicBool,
    stats: Mutex<[CategoryStats; 7]>,
    /// Start time for the current profiling session
    session_start: Mutex<Option<Instant>>,
    /// Total wall-clock time for the session
    session_total: Mutex<Duration>,
    /// Number of solver iterations profiled
    iteration_count: AtomicU64,
    /// Per-location profiling for identifying hotspots
    location_stats: Mutex<Vec<(String, CategoryStats)>>,
}

impl Default for ProfilingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilingStats {
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            stats: Mutex::new([
                CategoryStats::new(),
                CategoryStats::new(),
                CategoryStats::new(),
                CategoryStats::new(),
                CategoryStats::new(),
                CategoryStats::new(),
                CategoryStats::new(),
            ]),
            session_start: Mutex::new(None),
            session_total: Mutex::new(Duration::ZERO),
            iteration_count: AtomicU64::new(0),
            location_stats: Mutex::new(Vec::new()),
        }
    }

    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn start_session(&self) {
        *self.session_start.lock().unwrap() = Some(Instant::now());
        self.reset();
    }

    pub fn end_session(&self) {
        if let Some(start) = self.session_start.lock().unwrap().take() {
            *self.session_total.lock().unwrap() = start.elapsed();
        }
    }

    pub fn reset(&self) {
        let mut stats = self.stats.lock().unwrap();
        for stat in stats.iter_mut() {
            *stat = CategoryStats::new();
        }
        self.iteration_count.store(0, Ordering::Relaxed);
        self.location_stats.lock().unwrap().clear();
    }

    pub fn increment_iteration(&self) {
        self.iteration_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record(&self, category: ProfileCategory, duration: Duration, bytes: u64) {
        if !self.is_enabled() {
            return;
        }
        let idx = category as usize;
        let mut stats = self.stats.lock().unwrap();
        stats[idx].record(duration, bytes);
    }

    pub fn record_location(
        &self,
        location: &str,
        category: ProfileCategory,
        duration: Duration,
        bytes: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        // Record in category stats
        self.record(category, duration, bytes);

        // Record in location stats
        let mut locations = self.location_stats.lock().unwrap();
        let key = format!("{}:{}", category.name(), location);
        if let Some(entry) = locations.iter_mut().find(|(k, _)| k == &key) {
            entry.1.record(duration, bytes);
        } else {
            let mut stat = CategoryStats::new();
            stat.record(duration, bytes);
            locations.push((key, stat));
        }
    }

    pub fn get_stats(&self, category: ProfileCategory) -> CategoryStats {
        let stats = self.stats.lock().unwrap();
        stats[category as usize].clone()
    }

    pub fn get_all_stats(&self) -> Vec<(ProfileCategory, CategoryStats)> {
        let stats = self.stats.lock().unwrap();
        vec![
            (ProfileCategory::GpuRead, stats[0].clone()),
            (ProfileCategory::GpuWrite, stats[1].clone()),
            (ProfileCategory::GpuSync, stats[2].clone()),
            (ProfileCategory::GpuDispatch, stats[3].clone()),
            (ProfileCategory::CpuCompute, stats[4].clone()),
            (ProfileCategory::GpuResourceCreation, stats[5].clone()),
            (ProfileCategory::Other, stats[6].clone()),
        ]
    }

    pub fn get_location_stats(&self) -> Vec<(String, CategoryStats)> {
        self.location_stats.lock().unwrap().clone()
    }

    pub fn get_session_total(&self) -> Duration {
        *self.session_total.lock().unwrap()
    }

    pub fn get_iteration_count(&self) -> u64 {
        self.iteration_count.load(Ordering::Relaxed)
    }

    /// Print a detailed report of the profiling data
    pub fn print_report(&self) {
        println!("\n============ GPU-CPU Communication Profile Report ============\n");

        let session_total = self.get_session_total();
        let iterations = self.get_iteration_count();

        println!("Session Statistics:");
        println!("  Total wall-clock time: {:?}", session_total);
        println!("  Total iterations: {}", iterations);
        if iterations > 0 {
            println!(
                "  Average time per iteration: {:?}",
                session_total / iterations as u32
            );
        }
        println!();

        // Category breakdown
        println!("Category Breakdown:");
        println!(
            "{:<25} {:>12} {:>10} {:>12} {:>12} {:>12}",
            "Category", "Total Time", "Calls", "Avg Time", "% of Total", "MB/s"
        );
        println!("{}", "-".repeat(85));

        let mut total_profiled = Duration::ZERO;
        for (category, stats) in self.get_all_stats() {
            if stats.call_count > 0 {
                total_profiled += stats.total_time;
                let pct = if session_total.as_nanos() > 0 {
                    (stats.total_time.as_nanos() as f64 / session_total.as_nanos() as f64) * 100.0
                } else {
                    0.0
                };
                let throughput = if stats.total_bytes > 0 {
                    format!("{:.1}", stats.throughput_mb_per_sec())
                } else {
                    "-".to_string()
                };
                println!(
                    "{:<25} {:>12?} {:>10} {:>12?} {:>11.1}% {:>12}",
                    category.name(),
                    stats.total_time,
                    stats.call_count,
                    stats.avg_time(),
                    pct,
                    throughput
                );
            }
        }

        // Unaccounted time
        if session_total > total_profiled {
            let unaccounted = session_total - total_profiled;
            let pct = (unaccounted.as_nanos() as f64 / session_total.as_nanos() as f64) * 100.0;
            println!(
                "{:<25} {:>12?} {:>10} {:>12} {:>11.1}% {:>12}",
                "Unaccounted", unaccounted, "-", "-", pct, "-"
            );
        }

        println!();

        // Top hotspots by location
        let mut location_stats = self.get_location_stats();
        location_stats.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        println!("Top 15 Hotspots by Location:");
        println!(
            "{:<50} {:>12} {:>10} {:>12}",
            "Location", "Total Time", "Calls", "Avg Time"
        );
        println!("{}", "-".repeat(86));

        for (location, stats) in location_stats.iter().take(15) {
            println!(
                "{:<50} {:>12?} {:>10} {:>12?}",
                location,
                stats.total_time,
                stats.call_count,
                stats.avg_time()
            );
        }

        println!();

        // Optimization suggestions
        self.print_optimization_suggestions();
    }

    fn print_optimization_suggestions(&self) {
        println!("Optimization Suggestions:");
        println!("{}", "-".repeat(50));

        let gpu_read = self.get_stats(ProfileCategory::GpuRead);
        let gpu_sync = self.get_stats(ProfileCategory::GpuSync);
        let cpu_compute = self.get_stats(ProfileCategory::CpuCompute);

        let mut suggestions = Vec::new();

        // Check for excessive GPU reads
        if gpu_read.call_count > 100 {
            suggestions.push(format!(
                "• HIGH GPU READ COUNT ({} calls): Consider batching reads or \
                 computing results on GPU to avoid frequent transfers.",
                gpu_read.call_count
            ));
        }

        // Check GPU sync time
        let total = self.get_session_total();
        if total.as_nanos() > 0 {
            let sync_pct =
                (gpu_sync.total_time.as_nanos() as f64 / total.as_nanos() as f64) * 100.0;
            if sync_pct > 20.0 {
                suggestions.push(format!(
                    "• HIGH GPU SYNC TIME ({:.1}%): Pipeline GPU operations to reduce \
                     synchronization. Use async operations where possible.",
                    sync_pct
                ));
            }
        }

        // Check CPU compute time
        if total.as_nanos() > 0 {
            let cpu_pct =
                (cpu_compute.total_time.as_nanos() as f64 / total.as_nanos() as f64) * 100.0;
            if cpu_pct > 10.0 {
                suggestions.push(format!(
                    "• SIGNIFICANT CPU COMPUTE ({:.1}%): Consider offloading these \
                     computations to GPU shaders.",
                    cpu_pct
                ));
            }
        }

        // Check transfer size efficiency
        if gpu_read.call_count > 0 {
            let avg_bytes = gpu_read.total_bytes / gpu_read.call_count;
            if avg_bytes < 1024 {
                suggestions.push(format!(
                    "• SMALL TRANSFER SIZE (avg {} bytes): Small transfers have high \
                     overhead. Batch multiple values or compute on GPU.",
                    avg_bytes
                ));
            }
        }

        if suggestions.is_empty() {
            println!("  No major optimization opportunities detected.");
        } else {
            for suggestion in suggestions {
                println!("{}", suggestion);
            }
        }
        println!();
    }
}

/// RAII timer for profiling a scope
pub struct ProfileTimer<'a> {
    stats: &'a ProfilingStats,
    category: ProfileCategory,
    location: Option<&'a str>,
    start: Instant,
    bytes: u64,
}

impl<'a> ProfileTimer<'a> {
    pub fn new(stats: &'a ProfilingStats, category: ProfileCategory) -> Self {
        Self {
            stats,
            category,
            location: None,
            start: Instant::now(),
            bytes: 0,
        }
    }

    pub fn with_location(mut self, location: &'a str) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self
    }
}

impl<'a> Drop for ProfileTimer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        if let Some(location) = self.location {
            self.stats
                .record_location(location, self.category, duration, self.bytes);
        } else {
            self.stats.record(self.category, duration, self.bytes);
        }
    }
}

/// Macro for easily timing a block of code
#[macro_export]
macro_rules! profile_scope {
    ($stats:expr, $category:expr, $location:expr) => {
        let _timer = $crate::solver::gpu::profiling::ProfileTimer::new($stats, $category)
            .with_location($location);
    };
    ($stats:expr, $category:expr, $location:expr, $bytes:expr) => {
        let _timer = $crate::solver::gpu::profiling::ProfileTimer::new($stats, $category)
            .with_location($location)
            .with_bytes($bytes);
    };
}
