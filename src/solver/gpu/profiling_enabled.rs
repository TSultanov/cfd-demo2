// GPU-CPU Communication Profiling Module (enabled via `profiling` feature)
//
// This module provides detailed profiling of GPU-CPU data transfers and synchronization
// to identify performance bottlenecks and opportunities for GPU offloading.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{CategoryStats, MemoryDomain, MemoryStats, ProfileCategory};

/// Detailed profiling statistics for the solver
#[derive(Debug)]
pub struct ProfilingStats {
    enabled: AtomicBool,
    stats: Mutex<[CategoryStats; 7]>,
    session_start: Mutex<Option<Instant>>,
    session_total: Mutex<Duration>,
    iteration_count: AtomicU64,
    location_stats: Mutex<Vec<(String, CategoryStats)>>,
    memory_cpu: Mutex<MemoryStats>,
    memory_gpu: Mutex<MemoryStats>,
    memory_locations_cpu: Mutex<HashMap<String, MemoryStats>>,
    memory_locations_gpu: Mutex<HashMap<String, MemoryStats>>,
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
            memory_cpu: Mutex::new(MemoryStats::default()),
            memory_gpu: Mutex::new(MemoryStats::default()),
            memory_locations_cpu: Mutex::new(HashMap::new()),
            memory_locations_gpu: Mutex::new(HashMap::new()),
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
        *self.memory_cpu.lock().unwrap() = MemoryStats::default();
        *self.memory_gpu.lock().unwrap() = MemoryStats::default();
        self.memory_locations_cpu.lock().unwrap().clear();
        self.memory_locations_gpu.lock().unwrap().clear();
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
        self.record(category, duration, bytes);

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

    pub fn record_memory_event(
        &self,
        domain: MemoryDomain,
        location: &str,
        bytes: u64,
        is_alloc: bool,
    ) {
        if !self.is_enabled() {
            return;
        }

        let (totals, locations) = match domain {
            MemoryDomain::Cpu => (&self.memory_cpu, &self.memory_locations_cpu),
            MemoryDomain::Gpu => (&self.memory_gpu, &self.memory_locations_gpu),
        };

        {
            let mut guard = totals.lock().unwrap();
            if is_alloc {
                guard.record_alloc(bytes);
            } else {
                guard.record_free(bytes);
            }
        }

        let mut map = locations.lock().unwrap();
        let entry = map
            .entry(location.to_string())
            .or_insert_with(MemoryStats::default);
        if is_alloc {
            entry.record_alloc(bytes);
        } else {
            entry.record_free(bytes);
        }
    }

    pub fn record_cpu_alloc(&self, location: &str, bytes: u64) {
        self.record_memory_event(MemoryDomain::Cpu, location, bytes, true);
    }

    pub fn record_cpu_free(&self, location: &str, bytes: u64) {
        self.record_memory_event(MemoryDomain::Cpu, location, bytes, false);
    }

    pub fn record_gpu_alloc(&self, location: &str, bytes: u64) {
        self.record_memory_event(MemoryDomain::Gpu, location, bytes, true);
    }

    pub fn record_gpu_free(&self, location: &str, bytes: u64) {
        self.record_memory_event(MemoryDomain::Gpu, location, bytes, false);
    }

    pub fn get_memory_stats(&self) -> (MemoryStats, MemoryStats) {
        let cpu = self.memory_cpu.lock().unwrap().clone();
        let gpu = self.memory_gpu.lock().unwrap().clone();
        (cpu, gpu)
    }

    pub fn get_memory_location_stats(&self, domain: MemoryDomain) -> Vec<(String, MemoryStats)> {
        match domain {
            MemoryDomain::Cpu => self
                .memory_locations_cpu
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            MemoryDomain::Gpu => self
                .memory_locations_gpu
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
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

        if session_total > total_profiled {
            let unaccounted = session_total - total_profiled;
            let pct = (unaccounted.as_nanos() as f64 / session_total.as_nanos() as f64) * 100.0;
            println!(
                "{:<25} {:>12?} {:>10} {:>12} {:>11.1}% {:>12}",
                "Unaccounted", unaccounted, "-", "-", pct, "-"
            );
        }

        println!();

        // Per-node wall + GPU time, per solver step: `wall` (dominates when the step is
        // bound by readback polls) vs real GPU-timeline time. `wall − gpu` is the node's
        // CPU/sync overhead — says whether the hot node is GPU-compute- or host-bound.
        self.print_per_node_time(iterations);

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

        self.print_memory_report();
        self.print_optimization_suggestions();
    }

    /// Per-node wall + GPU time per solver step. Merges the `CpuCompute:label` (wall)
    /// and `GpuDispatch:label` (GPU timeline) location stats into one row per node,
    /// sorted by wall. `wall − gpu` is the node's CPU/sync overhead.
    fn print_per_node_time(&self, iterations: u64) {
        let wall_prefix = format!("{}:", ProfileCategory::CpuCompute.name());
        let gpu_prefix = format!("{}:", ProfileCategory::GpuDispatch.name());

        // label -> (wall_total, wall_calls, gpu_total)
        let mut rows: std::collections::HashMap<String, (Duration, u64, Duration)> =
            std::collections::HashMap::new();
        for (key, stats) in self.get_location_stats() {
            if let Some(label) = key.strip_prefix(&wall_prefix) {
                let e = rows.entry(label.to_string()).or_default();
                e.0 += stats.total_time;
                e.1 += stats.call_count;
            } else if let Some(label) = key.strip_prefix(&gpu_prefix) {
                let e = rows.entry(label.to_string()).or_default();
                e.2 += stats.total_time;
            }
        }
        if rows.is_empty() {
            return;
        }

        let mut rows: Vec<(String, (Duration, u64, Duration))> = rows.into_iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

        let steps = iterations.max(1) as f64;
        let total_wall: Duration = rows.iter().map(|(_, v)| v.0).sum();
        let total_gpu: Duration = rows.iter().map(|(_, v)| v.2).sum();

        println!("Per-Node Time (per step, {} steps):", iterations);
        println!(
            "{:<44} {:>9} {:>11} {:>11} {:>8}",
            "Node", "calls/st", "wall ms/st", "gpu ms/st", "% wall"
        );
        println!("{}", "-".repeat(87));
        for (label, (wall, calls, gpu)) in &rows {
            let wall_ms = wall.as_secs_f64() * 1e3 / steps;
            let gpu_ms = gpu.as_secs_f64() * 1e3 / steps;
            let calls_per_step = *calls as f64 / steps;
            let pct = if total_wall.as_nanos() > 0 {
                (wall.as_nanos() as f64 / total_wall.as_nanos() as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "{:<44} {:>9.1} {:>11.4} {:>11.4} {:>7.1}%",
                truncate_label(label, 44),
                calls_per_step,
                wall_ms,
                gpu_ms,
                pct
            );
        }
        println!("{}", "-".repeat(87));
        println!(
            "{:<44} {:>9} {:>11.4} {:>11.4} {:>8}",
            "TOTAL",
            "",
            total_wall.as_secs_f64() * 1e3 / steps,
            total_gpu.as_secs_f64() * 1e3 / steps,
            "100.0%"
        );
        println!();
    }

    fn print_memory_report(&self) {
        println!("Memory Allocation / Deallocation:");
        println!("{}", "-".repeat(50));

        let (cpu, gpu) = self.get_memory_stats();
        println!(
            "{:<6} {:>14} {:>14} {:>14} {:>14} {:>14}",
            "Domain", "Alloc Bytes", "Alloc Count", "Free Bytes", "Free Count", "Net Bytes",
        );
        println!(
            "{:<6} {:>14} {:>14} {:>14} {:>14} {:>14}",
            "CPU",
            cpu.alloc_bytes,
            cpu.alloc_count,
            cpu.free_bytes,
            cpu.free_count,
            cpu.net_bytes(),
        );
        println!(
            "{:<6} {:>14} {:>14} {:>14} {:>14} {:>14}",
            "GPU",
            gpu.alloc_bytes,
            gpu.alloc_count,
            gpu.free_bytes,
            gpu.free_count,
            gpu.net_bytes(),
        );

        for domain in [MemoryDomain::Cpu, MemoryDomain::Gpu] {
            let mut locations = self.get_memory_location_stats(domain);
            locations.sort_by(|a, b| b.1.alloc_bytes.cmp(&a.1.alloc_bytes));
            if locations.is_empty() {
                continue;
            }

            println!("\nTop 10 {} allocations:", domain.name());
            println!(
                "{:<50} {:>14} {:>12} {:>14}",
                "Location", "Alloc Bytes", "Alloc Cnt", "Max Request"
            );
            println!("{}", "-".repeat(96));

            for (loc, stats) in locations.into_iter().take(10) {
                println!(
                    "{:<50} {:>14} {:>12} {:>14}",
                    loc, stats.alloc_bytes, stats.alloc_count, stats.max_alloc_request,
                );
            }
        }

        println!();
    }

    fn print_optimization_suggestions(&self) {
        println!("Optimization Suggestions:");
        println!("{}", "-".repeat(50));

        let gpu_read = self.get_stats(ProfileCategory::GpuRead);
        let gpu_sync = self.get_stats(ProfileCategory::GpuSync);
        let cpu_compute = self.get_stats(ProfileCategory::CpuCompute);

        let mut suggestions = Vec::new();

        if gpu_read.call_count > 100 {
            suggestions.push(format!(
                "• HIGH GPU READ COUNT ({} calls): Consider batching reads or \
                 computing results on GPU to avoid frequent transfers.",
                gpu_read.call_count
            ));
        }

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

/// Truncate a label to `max` chars, replacing the dropped tail with `..` so the
/// per-graph table stays aligned for long generated-kernel names.
fn truncate_label(label: &str, max: usize) -> String {
    if label.len() <= max {
        label.to_string()
    } else {
        format!("{}..", &label[..max.saturating_sub(2)])
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
