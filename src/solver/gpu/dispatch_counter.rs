//! Dispatch Counter
//!
//! Tracks the number of GPU compute dispatches per solver step.
//! This helps identify overhead from too many small dispatches.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Debug, Default)]
pub struct DispatchCounter {
    total_dispatches: AtomicU64,
    by_category: Mutex<HashMap<&'static str, u64>>,
    by_kernel: Mutex<HashMap<String, u64>>,
    enabled: AtomicU64, // u64 for a lock-free atomic bool
}

#[derive(Debug, Clone)]
pub struct DispatchStats {
    pub total_dispatches: u64,
    pub by_category: HashMap<&'static str, u64>,
    pub by_kernel: HashMap<String, u64>,
}

impl DispatchCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enable(&self) {
        self.enabled.store(1, Ordering::Relaxed);
    }

    pub fn disable(&self) {
        self.enabled.store(0, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) != 0
    }

    pub fn record(&self, category: &'static str, kernel_label: &str) {
        if !self.is_enabled() {
            return;
        }

        self.total_dispatches.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut by_cat) = self.by_category.lock() {
            *by_cat.entry(category).or_insert(0) += 1;
        }

        if let Ok(mut by_ker) = self.by_kernel.lock() {
            *by_ker.entry(kernel_label.to_string()).or_insert(0) += 1;
        }
    }

    pub fn reset(&self) {
        self.total_dispatches.store(0, Ordering::Relaxed);
        if let Ok(mut by_cat) = self.by_category.lock() {
            by_cat.clear();
        }
        if let Ok(mut by_ker) = self.by_kernel.lock() {
            by_ker.clear();
        }
    }

    pub fn get_stats(&self) -> DispatchStats {
        DispatchStats {
            total_dispatches: self.total_dispatches.load(Ordering::Relaxed),
            by_category: self.by_category.lock().unwrap().clone(),
            by_kernel: self.by_kernel.lock().unwrap().clone(),
        }
    }

    pub fn print_stats(&self) {
        let stats = self.get_stats();
        Self::print_stats_static(&stats);
    }

    pub fn print_stats_static(stats: &DispatchStats) {
        println!("\n============ GPU Dispatch Statistics ============\n");
        println!("Total dispatches: {}", stats.total_dispatches);
        println!();

        println!("Dispatches by Category:");
        println!("{:<30} {:>10} {:>10}", "Category", "Count", "%");
        println!("{}", "-".repeat(52));

        let mut categories: Vec<_> = stats.by_category.iter().collect();
        categories.sort_by(|a, b| b.1.cmp(a.1));

        for (category, count) in categories {
            let pct = if stats.total_dispatches > 0 {
                (*count as f64 / stats.total_dispatches as f64) * 100.0
            } else {
                0.0
            };
            println!("{:<30} {:>10} {:>9.1}%", category, count, pct);
        }

        println!("\nTop 15 Kernels by Dispatch Count:");
        println!("{:<50} {:>10}", "Kernel", "Count");
        println!("{}", "-".repeat(62));

        let mut kernels: Vec<_> = stats.by_kernel.iter().collect();
        kernels.sort_by(|a, b| b.1.cmp(a.1));

        for (kernel, count) in kernels.iter().take(15) {
            let label = if kernel.len() > 48 {
                format!("{}..", &kernel[..48])
            } else {
                kernel.to_string()
            };
            println!("{:<50} {:>10}", label, count);
        }

        println!("\nAnalysis:");
        if stats.total_dispatches > 100 {
            println!("  WARNING: High dispatch count (>100) may indicate overhead.");
            println!("  Consider kernel fusion to reduce dispatch overhead.");
        } else if stats.total_dispatches > 50 {
            println!("  MODERATE: Dispatch count is reasonable but could be optimized.");
        } else {
            println!("  GOOD: Low dispatch count indicates efficient batching.");
        }

        let repeated: Vec<_> = stats.by_kernel.iter().filter(|(_, c)| **c > 10).collect();
        if !repeated.is_empty() {
            println!("\n  Note: Some kernels dispatched >10 times:");
            for (kernel, count) in repeated.iter().take(5) {
                println!("    - {}: {} times", kernel, count);
            }
        }

        println!();
    }
}

use std::sync::OnceLock;

pub fn global_dispatch_counter() -> &'static DispatchCounter {
    static COUNTER: OnceLock<DispatchCounter> = OnceLock::new();
    COUNTER.get_or_init(DispatchCounter::new)
}

pub mod categories {
    pub const LINEAR_SOLVER: &str = "Linear Solver";
    pub const PRECONDITIONER: &str = "Preconditioner";
    pub const FLUX_COMPUTATION: &str = "Flux Computation";
    pub const GRADIENT_COMPUTATION: &str = "Gradient Computation";
    pub const BOUNDARY_CONDITIONS: &str = "Boundary Conditions";
    pub const VECTOR_OPS: &str = "Vector Operations";
    pub const REDUCTION: &str = "Reduction";
    pub const STATE_UPDATE: &str = "State Update";
    pub const CONVERGENCE_CHECK: &str = "Convergence Check";
    pub const OTHER: &str = "Other";
}

#[macro_export]
macro_rules! count_dispatch {
    ($category:expr, $label:expr) => {
        $crate::solver::gpu::dispatch_counter::global_dispatch_counter().record($category, $label);
    };
}

/// RAII guard for counting dispatches within a scope
pub struct DispatchScope;

impl DispatchScope {
    pub fn new() -> Self {
        global_dispatch_counter().reset();
        global_dispatch_counter().enable();
        Self
    }
}

impl Default for DispatchScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DispatchScope {
    fn drop(&mut self) {
        global_dispatch_counter().disable();
    }
}

pub fn get_dispatch_stats() -> DispatchStats {
    global_dispatch_counter().get_stats()
}
