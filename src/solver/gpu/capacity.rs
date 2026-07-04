//! Capacity-reserved allocation for topology-sized GPU buffers (M2 Tier B).
//!
//! Roadmap decision (docs/meshless-moving-mesh-roadmap.md §3 "Sparsity
//! format"): CSR semantics stay exact (logical sizes are the true face/nnz
//! counts) but device buffers MAY be allocated with headroom so a
//! topology-level mesh refresh reuses allocations and only rebuilds bind
//! groups. Kernels keep their `arrayLength` guards correct because these
//! buffers are bound as **sized ranges** (`wgpu::BufferBinding { offset: 0,
//! size: Some(logical) }`) — `arrayLength` of a sized binding is the bound
//! size, not the allocation size.
//!
//! Defaults are behavior-neutral: [`CapacityPlan::default`] reserves EXACTLY
//! the logical size (headroom 1.0), and a sized binding whose size equals the
//! full buffer size is definitionally equivalent to
//! `as_entire_buffer_binding` (which binds `size: None` = "rest of the
//! buffer"). Headroom > 1.0 is only requested by a refresh/build context that
//! expects topology churn (later stages).

/// Allocation policy for face-/nnz-sized device buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapacityPlan {
    /// Multiplier applied to the logical element count when sizing the
    /// allocation (clamped to >= logical). `1.0` = exact (default).
    pub headroom: f64,
}

impl Default for CapacityPlan {
    fn default() -> Self {
        Self { headroom: 1.0 }
    }
}

impl CapacityPlan {
    /// Exact-size allocation (no headroom) — the behavior-neutral default.
    pub const EXACT: CapacityPlan = CapacityPlan { headroom: 1.0 };

    /// Capacity in elements for a buffer whose logical length is `logical`.
    /// Always >= `logical`; rounds up.
    pub fn capacity_elems(&self, logical: usize) -> usize {
        (((logical as f64) * self.headroom).ceil() as usize).max(logical)
    }
}

/// Create a STORAGE-class buffer holding `contents`, allocated at
/// `capacity_bytes >= contents.len()`. The tail (if any) is zero-filled.
/// With `capacity_bytes == contents.len()` this is exactly
/// `device.create_buffer_init` (same usage, same bytes).
///
/// `capacity_bytes` must be a multiple of 4 (`wgpu::COPY_BUFFER_ALIGNMENT`);
/// every caller sizes in whole elements of 4-byte-multiple stride, so this
/// holds by construction (debug-asserted).
pub fn create_buffer_with_capacity(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    capacity_bytes: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    debug_assert!(
        capacity_bytes >= contents.len() as u64,
        "capacity {capacity_bytes} < logical {} for {label}",
        contents.len()
    );
    debug_assert_eq!(capacity_bytes % 4, 0, "capacity must be 4-aligned ({label})");
    if capacity_bytes == contents.len() as u64 {
        use wgpu::util::DeviceExt;
        return device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        });
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: capacity_bytes,
        usage,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()[..contents.len()]
        .copy_from_slice(contents);
    buffer.unmap();
    buffer
}

/// The sized-range binding for the first `logical_bytes` of `buffer` —
/// the binding shape every capacity-reserved buffer must use so kernel
/// `arrayLength` guards see the logical length, not the allocation.
///
/// `logical_bytes == 0` falls back to binding the entire buffer
/// (`size: None`): wgpu forbids zero-sized bindings, and a zero logical
/// length only occurs for buffers no kernel meaningfully indexes.
pub fn sized_binding(buffer: &wgpu::Buffer, logical_bytes: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(logical_bytes),
    })
}
