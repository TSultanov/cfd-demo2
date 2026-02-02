//! GPU Timestamp Query Profiler
//!
//! This module provides GPU-side timing using wgpu timestamp queries.
//! Unlike CPU-side timing, this captures actual GPU execution time,
//! excluding queue submission overhead and CPU-GPU synchronization.

use std::collections::HashMap;
use std::time::Duration;

use crate::solver::gpu::context::GpuContext;

/// Configuration for timestamp queries
#[derive(Debug, Clone, Copy)]
pub struct TimestampConfig {
    /// Number of query pairs (start/end) to allocate
    pub max_queries: u32,
    /// Whether profiling is enabled
    pub enabled: bool,
}

impl Default for TimestampConfig {
    fn default() -> Self {
        Self {
            max_queries: 1024,
            enabled: false,
        }
    }
}

/// A single GPU timestamp query pair (start and end)
#[derive(Debug, Clone, Copy)]
pub struct TimestampQuery {
    pub start_idx: u32,
    pub end_idx: u32,
    pub label: &'static str,
    pub category: KernelCategory,
}

/// Categories of GPU kernels for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelCategory {
    LinearSystemSetup,
    Preconditioner,
    Spmv,
    VectorOps,
    Reduction,
    FgmresIteration,
    AmgCycle,
    FluxComputation,
    GradientComputation,
    BoundaryConditions,
    Other,
}

impl KernelCategory {
    pub fn name(&self) -> &'static str {
        match self {
            KernelCategory::LinearSystemSetup => "Linear System Setup",
            KernelCategory::Preconditioner => "Preconditioner",
            KernelCategory::Spmv => "SpMV",
            KernelCategory::VectorOps => "Vector Operations",
            KernelCategory::Reduction => "Reduction",
            KernelCategory::FgmresIteration => "FGMRES Iteration",
            KernelCategory::AmgCycle => "AMG Cycle",
            KernelCategory::FluxComputation => "Flux Computation",
            KernelCategory::GradientComputation => "Gradient Computation",
            KernelCategory::BoundaryConditions => "Boundary Conditions",
            KernelCategory::Other => "Other",
        }
    }
}

/// GPU timestamp profiler using wgpu query sets
pub struct GpuTimestampProfiler {
    config: TimestampConfig,
    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: Option<wgpu::Buffer>,
    /// Query labels and their indices
    queries: Vec<TimestampQuery>,
    /// Current query index
    current_query: u32,
    /// Timestamp period (nanoseconds per tick)
    timestamp_period: f32,
    /// Whether a frame is currently being recorded
    recording: bool,
}

/// Results from a profiling session
#[derive(Debug, Clone)]
pub struct ProfilingResults {
    pub total_gpu_time: Duration,
    pub kernel_times: Vec<KernelTiming>,
    pub by_category: HashMap<KernelCategory, Duration>,
}

#[derive(Debug, Clone)]
pub struct KernelTiming {
    pub label: String,
    pub category: KernelCategory,
    pub duration: Duration,
    pub percentage: f64,
}

impl GpuTimestampProfiler {
    /// Create a new timestamp profiler (disabled by default)
    pub fn new(context: &GpuContext) -> Self {
        let config = TimestampConfig::default();
        let timestamp_period = context.queue.get_timestamp_period();

        Self {
            config,
            query_set: None,
            resolve_buffer: None,
            queries: Vec::new(),
            current_query: 0,
            timestamp_period,
            recording: false,
        }
    }

    /// Enable profiling and allocate GPU resources
    pub fn enable(&mut self, context: &GpuContext) {
        if self.config.enabled {
            return;
        }
        self.config.enabled = true;

        // Create query set for timestamps
        self.query_set = Some(context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Timestamp Queries"),
            count: self.config.max_queries,
            ty: wgpu::QueryType::Timestamp,
        }));

        // Create buffer for resolving query results
        // Each timestamp is 8 bytes (u64)
        self.resolve_buffer = Some(context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Resolve Buffer"),
            size: (self.config.max_queries * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        self.queries.clear();
        self.current_query = 0;
    }

    /// Disable profiling and free resources
    pub fn disable(&mut self) {
        self.config.enabled = false;
        self.query_set = None;
        self.resolve_buffer = None;
        self.queries.clear();
        self.current_query = 0;
        self.recording = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Begin a new profiling frame
    pub fn begin_frame(&mut self) {
        if !self.config.enabled {
            return;
        }
        self.recording = true;
        self.current_query = 0;
        self.queries.clear();
    }

    /// Record a timestamp at the current query index
    /// Returns the query index for later reference
    pub fn write_timestamp(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        _category: KernelCategory,
    ) -> Option<u32> {
        if !self.config.enabled || !self.recording {
            return None;
        }

        let query_idx = self.current_query;
        if query_idx >= self.config.max_queries {
            log::warn!("GPU timestamp query limit exceeded, dropping query for {}", label);
            return None;
        }

        if let Some(ref query_set) = self.query_set {
            encoder.write_timestamp(query_set, query_idx);
        }

        self.current_query += 1;
        Some(query_idx)
    }

    /// Record a kernel execution with automatic start/end tracking
    /// This should be called BEFORE the compute pass
    pub fn begin_kernel(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        category: KernelCategory,
    ) -> Option<u32> {
        let start_idx = self.write_timestamp(encoder, label, category)?;
        
        self.queries.push(TimestampQuery {
            start_idx,
            end_idx: 0, // Will be set by end_kernel
            label,
            category,
        });

        Some(start_idx)
    }

    /// End a kernel execution
    /// This should be called AFTER the compute pass
    pub fn end_kernel(&mut self, encoder: &mut wgpu::CommandEncoder, start_idx: u32) {
        if !self.config.enabled || !self.recording {
            return;
        }

        let end_idx = self.current_query;
        if end_idx >= self.config.max_queries {
            return;
        }

        if let Some(ref query_set) = self.query_set {
            encoder.write_timestamp(query_set, end_idx);
        }

        // Update the query with end index
        if let Some(query) = self.queries.iter_mut().find(|q| q.start_idx == start_idx) {
            query.end_idx = end_idx;
        }

        self.current_query += 1;
    }

    /// End the profiling frame and resolve results
    pub fn end_frame(&mut self, context: &GpuContext) -> Option<ProfilingResults> {
        if !self.config.enabled || !self.recording {
            return None;
        }
        self.recording = false;

        // Resolve queries to buffer
        let query_set = self.query_set.as_ref()?;
        let resolve_buffer = self.resolve_buffer.as_ref()?;

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Timestamp Resolve"),
            });

        encoder.resolve_query_set(
            query_set,
            0..self.current_query,
            resolve_buffer,
            0,
        );

        let submission_index = context.queue.submit(Some(encoder.finish()));

        // Map and read results
        let slice = resolve_buffer.slice(..(self.current_query * 8) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });
        rx.recv().ok().and_then(|v| v.ok())?;

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        // Calculate durations
        let mut kernel_times = Vec::new();
        let mut by_category: HashMap<KernelCategory, Duration> = HashMap::new();
        let mut total_gpu_time = Duration::ZERO;

        for query in &self.queries {
            if query.end_idx == 0 || query.end_idx as usize >= timestamps.len() {
                continue;
            }

            let start_ns = timestamps[query.start_idx as usize];
            let end_ns = timestamps[query.end_idx as usize];
            let delta_ns = end_ns.saturating_sub(start_ns);
            let duration = Duration::from_nanos((delta_ns as f32 * self.timestamp_period) as u64);

            kernel_times.push((query.label, query.category, duration));
            *by_category.entry(query.category).or_insert(Duration::ZERO) += duration;
            total_gpu_time += duration;
        }

        drop(data);
        resolve_buffer.unmap();

        // Calculate percentages and build results
        let kernel_times: Vec<KernelTiming> = kernel_times
            .into_iter()
            .map(|(label, category, duration)| KernelTiming {
                label: label.to_string(),
                category,
                duration,
                percentage: if total_gpu_time.as_nanos() > 0 {
                    (duration.as_nanos() as f64 / total_gpu_time.as_nanos() as f64) * 100.0
                } else {
                    0.0
                },
            })
            .collect();

        Some(ProfilingResults {
            total_gpu_time,
            kernel_times,
            by_category,
        })
    }

    /// Print profiling results
    pub fn print_results(results: &ProfilingResults) {
        println!("\n============ GPU Kernel Timing Report ============\n");
        println!("Total GPU time: {:?}", results.total_gpu_time);
        println!();

        // By category
        println!("Time by Category:");
        println!("{:<25} {:>12} {:>10}", "Category", "Time", "% of Total");
        println!("{}", "-".repeat(50));
        
        let mut categories: Vec<_> = results.by_category.iter().collect();
        categories.sort_by(|a, b| b.1.cmp(a.1));
        
        for (category, duration) in categories {
            let pct = if results.total_gpu_time.as_nanos() > 0 {
                (duration.as_nanos() as f64 / results.total_gpu_time.as_nanos() as f64) * 100.0
            } else {
                0.0
            };
            println!("{:<25} {:>12?} {:>9.1}%", category.name(), duration, pct);
        }

        // Individual kernels
        println!("\nTop 20 Individual Kernels:");
        println!("{:<40} {:>12} {:>10}", "Kernel", "Time", "% of Total");
        println!("{}", "-".repeat(65));
        
        let mut kernels = results.kernel_times.clone();
        kernels.sort_by(|a, b| b.duration.cmp(&a.duration));
        
        for kernel in kernels.iter().take(20) {
            let label = if kernel.label.len() > 38 {
                format!("{}..", &kernel.label[..38])
            } else {
                kernel.label.clone()
            };
            println!("{:<40} {:>12?} {:>9.1}%", label, kernel.duration, kernel.percentage);
        }

        println!();
    }
}

/// Macro for timing a kernel execution block
#[macro_export]
macro_rules! gpu_profile_kernel {
    ($profiler:expr, $encoder:expr, $label:expr, $category:expr, $block:block) => {{
        let _start_idx = $profiler.begin_kernel($encoder, $label, $category);
        let result = $block;
        if let Some(start_idx) = _start_idx {
            $profiler.end_kernel($encoder, start_idx);
        }
        result
    }};
}

/// Helper to create timestamp writes for compute pass (if enabled)
pub fn maybe_timestamp_writes<'a>(
    profiler: &'a GpuTimestampProfiler,
    query_set: &'a wgpu::QuerySet,
    start_idx: u32,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    if profiler.is_enabled() {
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(start_idx),
            end_of_pass_write_index: Some(start_idx + 1),
        })
    } else {
        None
    }
}
