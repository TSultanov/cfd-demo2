//! Submission Counter
//!
//! Tracks the number of GPU queue submissions (`queue.submit(...)`) in a scope.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::sync::OnceLock;

#[derive(Debug, Default)]
pub struct SubmissionCounter {
    total_submissions: AtomicU64,
    by_category: Mutex<HashMap<&'static str, u64>>,
    by_label: Mutex<HashMap<String, u64>>,
    enabled: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct SubmissionStats {
    pub total_submissions: u64,
    pub by_category: HashMap<&'static str, u64>,
    pub by_label: HashMap<String, u64>,
}

impl SubmissionCounter {
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

    pub fn record(&self, category: &'static str, label: &str) {
        if !self.is_enabled() {
            return;
        }

        self.total_submissions.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut by_cat) = self.by_category.lock() {
            *by_cat.entry(category).or_insert(0) += 1;
        }

        if let Ok(mut by_lbl) = self.by_label.lock() {
            *by_lbl.entry(label.to_string()).or_insert(0) += 1;
        }
    }

    pub fn reset(&self) {
        self.total_submissions.store(0, Ordering::Relaxed);
        if let Ok(mut by_cat) = self.by_category.lock() {
            by_cat.clear();
        }
        if let Ok(mut by_lbl) = self.by_label.lock() {
            by_lbl.clear();
        }
    }

    pub fn get_stats(&self) -> SubmissionStats {
        SubmissionStats {
            total_submissions: self.total_submissions.load(Ordering::Relaxed),
            by_category: self.by_category.lock().unwrap().clone(),
            by_label: self.by_label.lock().unwrap().clone(),
        }
    }
}

pub fn global_submission_counter() -> &'static SubmissionCounter {
    static COUNTER: OnceLock<SubmissionCounter> = OnceLock::new();
    COUNTER.get_or_init(SubmissionCounter::new)
}

#[macro_export]
macro_rules! count_submission {
    ($category:expr, $label:expr) => {
        $crate::solver::gpu::submission_counter::global_submission_counter()
            .record($category, $label);
    };
}

pub struct SubmissionScope;

impl SubmissionScope {
    pub fn new() -> Self {
        global_submission_counter().reset();
        global_submission_counter().enable();
        Self
    }
}

impl Default for SubmissionScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SubmissionScope {
    fn drop(&mut self) {
        global_submission_counter().disable();
    }
}

pub fn get_submission_stats() -> SubmissionStats {
    global_submission_counter().get_stats()
}
