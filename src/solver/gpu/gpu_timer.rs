//! GPU-timeline profiling via `wgpu` timestamp queries.
//!
//! The [`crate::solver::gpu::profiling`] scope timer measures CPU wall-clock,
//! which — because GPU dispatches are asynchronous — captures nothing of the GPU
//! execution it brackets (a submit returns before the GPU runs). A poll-based
//! barrier fixes attribution but has a ~1 ms round-trip floor on integrated GPUs
//! that swamps the sub-ms graphs we want to compare.
//!
//! This module measures the *GPU timeline* directly. Each graph is bracketed by
//! two `write_timestamp` marks emitted on their own tiny encoders (so the graph's
//! own encoder is untouched); because queue submissions execute in order, the
//! delta between the two marks is the graph's true GPU time. Marks are cheap and
//! non-blocking — nothing serializes — and the whole batch is resolved with a
//! single poll at step end.
//!
//! Requires `TIMESTAMP_QUERY` + `TIMESTAMP_QUERY_INSIDE_ENCODERS` (see
//! [`GpuContext`]); when unavailable the caller falls back to the poll-based path.

use std::cell::{Cell, RefCell};
use std::time::Duration;

use super::context::GpuContext;
use super::profiling::{ProfileCategory, ProfilingStats};

/// Timestamp-query slots. Two are consumed per graph scope, so this allows
/// `CAP / 2` graph scopes between flushes. Flushing happens every solver step, so
/// this only has to cover one step's graphs (a handful of outer iterations × a
/// dozen graphs + linear solves ≈ low hundreds) with wide headroom.
const CAP: u32 = 4096;

/// Owns a timestamp `QuerySet` and the buffers to resolve it, and accumulates
/// per-graph GPU durations into a [`ProfilingStats`] once per step.
///
/// All mutable state uses interior mutability so the hot-path methods take `&self`
/// and compose with the plan's other `&self` field borrows in `execute_block`.
pub struct GpuTimestampProfiler {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    period_ns: f32,
    /// Next free timestamp index for this step's batch.
    next: Cell<u32>,
    /// `(label, start_idx, end_idx)` for each completed scope, drained on flush.
    pending: RefCell<Vec<(&'static str, u32, u32)>>,
    /// Count of scopes dropped this step because the query set filled (surfaced so
    /// silent truncation never masquerades as "measured everything").
    dropped: Cell<u32>,
}

impl GpuTimestampProfiler {
    /// Build a profiler if the context enabled the required timestamp features,
    /// else `None` (the caller then uses the poll-based fallback).
    pub fn new(context: &GpuContext) -> Option<Self> {
        if !(context.timestamp_query && context.timestamps_inside_encoders) {
            return None;
        }
        let device = &context.device;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cfd2 profiler timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: CAP,
        });
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfd2 profiler ts resolve"),
            size: (CAP as u64) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfd2 profiler ts read"),
            size: (CAP as u64) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buf,
            read_buf,
            period_ns: context.timestamp_period_ns,
            next: Cell::new(0),
            pending: RefCell::new(Vec::new()),
            dropped: Cell::new(0),
        })
    }

    /// Emit a timestamp mark on its own encoder, returning its index (or `None`
    /// when the batch is full — the scope is then dropped and counted).
    pub fn mark(&self, context: &GpuContext) -> Option<u32> {
        let idx = self.next.get();
        if idx >= CAP {
            return None;
        }
        let mut enc = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cfd2 ts mark"),
            });
        enc.write_timestamp(&self.query_set, idx);
        context.queue.submit(Some(enc.finish()));
        self.next.set(idx + 1);
        Some(idx)
    }

    /// Bracket `run` with GPU timestamps and remember its duration under `label`.
    /// Returns `run`'s value unchanged. If either mark cannot be allocated the
    /// scope still runs — only its measurement is dropped.
    pub fn scope<R>(
        &self,
        context: &GpuContext,
        label: &'static str,
        run: impl FnOnce() -> R,
    ) -> R {
        let start = self.mark(context);
        let value = run();
        let end = self.mark(context);
        self.record_scope(label, start, end);
        value
    }

    /// Remember a `(start, end)` mark pair (from [`Self::mark`]) under `label`.
    /// Use when the bracketed work needs `&mut` on the plan and so cannot be
    /// expressed as the closure [`Self::scope`] takes (e.g. a Host node).
    pub fn record_scope(&self, label: &'static str, start: Option<u32>, end: Option<u32>) {
        match (start, end) {
            (Some(s), Some(e)) => self.pending.borrow_mut().push((label, s, e)),
            _ => self.dropped.set(self.dropped.get() + 1),
        }
    }

    /// Resolve this step's timestamp batch and fold the per-graph GPU durations
    /// into `stats`, then reset for the next step. One poll per call.
    pub fn flush(&self, context: &GpuContext, stats: &ProfilingStats) {
        let count = self.next.get();
        if count == 0 {
            return;
        }
        let bytes = (count as u64) * 8;
        let mut enc = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cfd2 ts resolve"),
            });
        enc.resolve_query_set(&self.query_set, 0..count, &self.resolve_buf, 0);
        enc.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.read_buf, 0, bytes);
        let idx = context.queue.submit(Some(enc.finish()));

        let slice = self.read_buf.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        });

        if rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
            let data = slice.get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            for &(label, s, e) in self.pending.borrow().iter() {
                let (s, e) = (s as usize, e as usize);
                if s < ticks.len() && e < ticks.len() {
                    // Saturating: a timestamp can occasionally read as non-monotonic
                    // across the tiny bracketing encoders; clamp to zero rather than
                    // underflow into a huge bogus duration.
                    let dt = ticks[e].saturating_sub(ticks[s]);
                    let ns = (dt as f64) * (self.period_ns as f64);
                    stats.record_location(
                        label,
                        ProfileCategory::GpuDispatch,
                        Duration::from_nanos(ns as u64),
                        0,
                    );
                }
            }
            drop(data);
            self.read_buf.unmap();
        }

        self.next.set(0);
        self.pending.borrow_mut().clear();
        self.dropped.set(0);
    }
}
