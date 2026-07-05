//! Per-device compute-pipeline cache (M5 stage 1).
//!
//! The hand-written linear-algebra stack (scalar CG, FGMRES, Schur, AMG,
//! block/diag preconditioners, the outer-convergence monitor + gate) compiles
//! its WGSL into [`wgpu::ComputePipeline`] objects at construction time — and
//! historically *re-*compiled them on every Tier B topology refresh, because a
//! `refresh_mesh(Topology)` RECONSTRUCTS those modules to resize their bind
//! groups over the reallocated buffers. That recompile was the dominant cost of
//! a GPU topology refresh (measured ~10.6 ms of size-independent shader
//! compilation at both 20k and 300k cells — see
//! `tests/mesh_refresh_topology_test.rs::bench_topology_refresh_cost`).
//!
//! The pipelines are a pure function of the shader source, which is keyed by
//! `(model_id, KernelId)` and is INVARIANT across a topology refresh (same
//! kernels, only buffer sizes / bind groups differ). This cache memoizes the
//! compiled pipeline per `(model_id, KernelId)` so the refresh path (and the
//! lazy per-solve `ensure_*_pipelines` builders) reuse the already-compiled
//! pipeline instead of recompiling it. A `wgpu::ComputePipeline` is an
//! `Arc`-backed handle, so a cache hit is a cheap refcount clone.
//!
//! ## Device scoping / safety
//!
//! The cache lives inside [`crate::solver::gpu::context::GpuContext`], so there
//! is exactly ONE cache per device: every lookup uses the same device that owns
//! the context, and a pipeline compiled for device A can never be handed to
//! device B. A returned pipeline is byte-for-byte the same object the
//! reconstruction path would have compiled from the same source, so caching is
//! transparent to results — the no-op-refresh byte gate and the
//! refresh-vs-fresh physics gate both stay green.
//!
//! ## A/B toggle (`CFD2_GPU_PIPELINE_CACHE=0`)
//!
//! Setting `CFD2_GPU_PIPELINE_CACHE=0` disables the memoization: every
//! [`PipelineCache::pipeline`] call compiles a fresh pipeline, reproducing the
//! pre-M5 recompile-per-refresh cost. This is what makes the roadmap's
//! "topology refresh −68%" figure self-verifying — `bench_topology_refresh_cost`
//! measures the refresh with the cache both ON (default) and OFF and prints the
//! delta. It is a pure benchmarking/diagnostic knob: results are identical either
//! way (the compiled pipeline is a function of the static source), only the wall
//! time differs. Read per call (`pipeline()` is a setup/refresh-time call, never
//! in the per-iteration solve loop), so a test can flip it between legs.

use std::collections::HashMap;
use std::sync::Mutex;

/// Whether pipeline memoization is enabled (default true; `CFD2_GPU_PIPELINE_CACHE=0`
/// forces a fresh compile on every lookup for the refresh-cost A/B benchmark).
fn cache_enabled() -> bool {
    !matches!(std::env::var("CFD2_GPU_PIPELINE_CACHE").as_deref(), Ok("0"))
}

use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::model::KernelId;

/// Cache key: the model id (empty for the shared hand-written LA kernels) plus
/// the `&'static str` kernel id. Two kernels with the same id under different
/// models can generate different WGSL, so the model id is part of the key.
type CacheKey = (String, &'static str);

/// Memoizes compiled compute pipelines by `(model_id, KernelId)` for one device.
#[derive(Default)]
pub(crate) struct PipelineCache {
    map: Mutex<HashMap<CacheKey, wgpu::ComputePipeline>>,
}

impl PipelineCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Return the compiled pipeline for `(model_id, kernel)`, compiling and
    /// caching it on the first request. Subsequent requests (including after a
    /// topology refresh that reconstructed the owning module) return the cached
    /// `Arc` clone with no recompilation.
    pub(crate) fn pipeline(
        &self,
        device: &wgpu::Device,
        model_id: &str,
        kernel: KernelId,
    ) -> Result<wgpu::ComputePipeline, String> {
        let enabled = cache_enabled();
        let key: CacheKey = (model_id.to_string(), kernel.0);
        if enabled {
            if let Some(p) = self.map.lock().unwrap().get(&key) {
                return Ok(p.clone());
            }
        }
        let src = kernel_registry::kernel_source_by_id(model_id, kernel)?;
        let pipeline = (src.create_pipeline)(device);
        if enabled {
            self.map.lock().unwrap().insert(key, pipeline.clone());
        }
        Ok(pipeline)
    }

    /// Number of distinct pipelines currently memoized (test/introspection).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }
}
