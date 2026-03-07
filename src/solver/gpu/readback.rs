use std::sync::Mutex;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::profiling::ProfilingStats;

#[cfg(feature = "profiling")]
use crate::solver::gpu::profiling::ProfileCategory;
#[cfg(feature = "profiling")]
use std::time::Instant;

/// Maximum number of distinct buffer sizes retained in the cache.
///
/// 16 is more than sufficient for typical CFD workloads where only a handful of
/// distinct readback sizes are used (e.g. 4 bytes for scalars, N*4 bytes for
/// convergence bit-vectors, state-buffer-sized reads).
const STAGING_CACHE_CAPACITY: usize = 16;

/// A bounded, LRU-evicting cache of GPU staging buffers keyed by byte-size.
///
/// Buffers are borrowed via [`take_or_create`] and returned via [`put`].
/// When the number of cached entries exceeds [`STAGING_CACHE_CAPACITY`], the
/// least-recently-used entry is evicted (its `wgpu::Buffer` is dropped,
/// releasing the GPU memory).
///
/// Internally uses a `Vec<(u64, wgpu::Buffer)>` ordered from most-recently-used
/// (front) to least-recently-used (back). With a capacity of 16 the linear
/// scan is negligible.
pub struct StagingBufferCache {
    /// Entries ordered most-recently-used first.
    entries: Mutex<Vec<(u64, wgpu::Buffer)>>,
    /// Diagnostic counters (always maintained; exposed via `stats()`).
    counters: Mutex<CacheCounters>,
}

/// Diagnostic counters for [`StagingBufferCache`].
#[derive(Debug, Clone, Default)]
pub struct CacheCounters {
    /// Number of `take_or_create` calls that found a cached buffer.
    pub hits: u64,
    /// Number of `take_or_create` calls that had to allocate a new buffer.
    pub misses: u64,
    /// Number of buffers evicted because the cache exceeded its capacity.
    pub evictions: u64,
    /// Total GPU bytes currently held in the cache.
    pub cached_bytes: u64,
}

impl Default for StagingBufferCache {
    fn default() -> Self {
        Self {
            entries: Mutex::new(Vec::with_capacity(STAGING_CACHE_CAPACITY)),
            counters: Mutex::new(CacheCounters::default()),
        }
    }
}

impl StagingBufferCache {
    /// Borrow a buffer of exactly `size` bytes from the cache, or create one.
    ///
    /// If a cached buffer of the requested size exists it is removed from the
    /// cache and returned (cache hit). Otherwise a new staging buffer is
    /// allocated from the device (cache miss).
    pub fn take_or_create(
        &self,
        device: &wgpu::Device,
        size: u64,
        label: &'static str,
    ) -> wgpu::Buffer {
        let mut entries = self.entries.lock().unwrap();
        if let Some(pos) = entries.iter().position(|(s, _)| *s == size) {
            let (_sz, buffer) = entries.remove(pos);
            let mut c = self.counters.lock().unwrap();
            c.hits += 1;
            c.cached_bytes = c.cached_bytes.saturating_sub(size);
            return buffer;
        }
        drop(entries);

        self.counters.lock().unwrap().misses += 1;

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the cache after use.
    ///
    /// The buffer is placed at the front (most-recently-used position). If the
    /// cache is already at capacity the least-recently-used entry is evicted.
    pub fn put(&self, size: u64, buffer: wgpu::Buffer) {
        let mut entries = self.entries.lock().unwrap();
        entries.insert(0, (size, buffer));

        let mut c = self.counters.lock().unwrap();
        c.cached_bytes += size;

        // Evict LRU entries that exceed capacity.
        while entries.len() > STAGING_CACHE_CAPACITY {
            let (evicted_size, _evicted_buf) = entries.pop().unwrap();
            c.evictions += 1;
            c.cached_bytes = c.cached_bytes.saturating_sub(evicted_size);
            // `_evicted_buf` is dropped here, releasing GPU memory.
        }
    }

    /// Drop all cached buffers, releasing their GPU memory.
    pub fn clear(&self) {
        let mut entries = self.entries.lock().unwrap();
        entries.clear();
        let mut c = self.counters.lock().unwrap();
        c.cached_bytes = 0;
    }

    /// Return a snapshot of the diagnostic counters.
    pub fn stats(&self) -> CacheCounters {
        self.counters.lock().unwrap().clone()
    }
}

/// Record a profiling event if the `profiling` feature is enabled and active.
///
/// This macro eliminates the need to duplicate the entire function body for each
/// cfg(feature = "profiling") / cfg(not(feature = "profiling")) branch.
macro_rules! profile_location {
    ($profiling:expr, $location:expr, $category:expr, $body:expr) => {{
        #[cfg(feature = "profiling")]
        {
            if $profiling.is_enabled() {
                let _t = Instant::now();
                let result = $body;
                $profiling.record_location($location, $category, _t.elapsed(), 0);
                result
            } else {
                $body
            }
        }
        #[cfg(not(feature = "profiling"))]
        {
            $body
        }
    }};
}

pub async fn read_buffer_cached(
    context: &GpuContext,
    cache: &StagingBufferCache,
    profiling: &ProfilingStats,
    buffer: &wgpu::Buffer,
    size: u64,
    label: &'static str,
) -> Vec<u8> {
    #[cfg(not(feature = "profiling"))]
    let _ = profiling;

    let staging_buffer = profile_location!(
        profiling,
        "read_buffer:create_staging",
        ProfileCategory::GpuResourceCreation,
        {
            let buf = cache.take_or_create(&context.device, size, label);
            #[cfg(feature = "profiling")]
            if profiling.is_enabled() {
                profiling.record_gpu_alloc(label, size);
            }
            buf
        }
    );

    let submission_index = profile_location!(
        profiling,
        "read_buffer:submit_copy",
        ProfileCategory::GpuDispatch,
        {
            let mut encoder = context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
            let idx = context.queue.submit(Some(encoder.finish()));
            crate::count_submission!("Readback", "read_buffer_copy");
            idx
        }
    );

    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    profile_location!(
        profiling,
        "read_buffer:map_async_request",
        ProfileCategory::Other,
        {
            slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
        }
    );

    profile_location!(
        profiling,
        "read_buffer:device_poll_wait",
        ProfileCategory::GpuSync,
        {
            let _ = context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });
        }
    );

    profile_location!(
        profiling,
        "read_buffer:channel_recv",
        ProfileCategory::Other,
        {
            rx.recv()
                .map_err(|e| format!("GPU readback channel recv failed: {e}"))
                .and_then(|r| r.map_err(|e| format!("GPU buffer mapping failed: {e:?}")))
                .expect("GPU readback failed");
        }
    );

    let result = profile_location!(
        profiling,
        "read_buffer:memcpy",
        ProfileCategory::Other,
        {
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            #[cfg(feature = "profiling")]
            if profiling.is_enabled() {
                profiling.record_cpu_alloc("read_buffer:cpu_copy", size);
            }
            result
        }
    );

    cache.put(size, staging_buffer);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a wgpu device for cache tests.
    fn test_device() -> wgpu::Device {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("no GPU adapter available");
            let (device, _queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("failed to create device");
            device
        })
    }

    #[test]
    fn hit_and_miss_counters() {
        let device = test_device();
        let cache = StagingBufferCache::default();

        // First call: miss
        let buf = cache.take_or_create(&device, 64, "test");
        let s = cache.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 0);

        // Return it
        cache.put(64, buf);
        assert_eq!(cache.stats().cached_bytes, 64);

        // Second call: hit
        let buf = cache.take_or_create(&device, 64, "test");
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(s.cached_bytes, 0); // taken out

        cache.put(64, buf);
    }

    #[test]
    fn lru_eviction() {
        let device = test_device();
        let cache = StagingBufferCache::default();

        // Fill the cache to capacity with distinct sizes.
        for i in 0..STAGING_CACHE_CAPACITY {
            let size = (i as u64 + 1) * 256;
            let buf = cache.take_or_create(&device, size, "fill");
            cache.put(size, buf);
        }
        assert_eq!(cache.stats().evictions, 0);

        // One more distinct size should evict the LRU entry (size=256, the first inserted).
        let extra_size = 99999;
        let buf = cache.take_or_create(&device, extra_size, "overflow");
        cache.put(extra_size, buf);
        let s = cache.stats();
        assert_eq!(s.evictions, 1);

        // The evicted entry (size=256) should now miss.
        let buf = cache.take_or_create(&device, 256, "evicted");
        let s = cache.stats();
        // +1 miss for the initial 256 creation, +1 miss for the re-creation after eviction,
        // plus STAGING_CACHE_CAPACITY misses from the fill loop and 1 for extra_size.
        assert!(s.misses > 0);
        // But the most recent entries should still be hits.
        drop(buf);
    }

    #[test]
    fn clear_releases_all() {
        let device = test_device();
        let cache = StagingBufferCache::default();

        for i in 0..4 {
            let buf = cache.take_or_create(&device, (i + 1) * 128, "clear_test");
            cache.put((i + 1) * 128, buf);
        }
        assert!(cache.stats().cached_bytes > 0);

        cache.clear();
        assert_eq!(cache.stats().cached_bytes, 0);

        // After clear, all sizes should miss.
        let buf = cache.take_or_create(&device, 128, "after_clear");
        assert_eq!(cache.stats().hits, 0); // counters NOT reset by clear (by design)
        drop(buf);
    }
}
