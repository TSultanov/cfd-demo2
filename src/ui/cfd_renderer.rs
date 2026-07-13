//! GPU-accelerated CFD mesh renderer using wgpu
//!
//! This module provides a shader-based renderer for CFD mesh data that significantly
//! improves rendering performance for very fine meshes (80k+ cells) compared to the
//! polygon-per-cell approach.
//!
//! The renderer:
//! 1. Uploads mesh geometry (triangulated cells) to GPU buffers
//! 2. Uploads field values as vertex attributes
//! 3. Renders all triangles in a single draw call with color interpolation

use wgpu::util::DeviceExt;

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

/// Order of the four ranges produced by the visualization reduction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CfdRangeField {
    Pressure = 0,
    VelocityX = 1,
    VelocityY = 2,
    VelocityMagnitude = 3,
}

impl CfdRangeField {
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Packed-state layout consumed by the range reducer. Offsets are arbitrary;
/// only the velocity pair is required to be adjacent, matching the renderer's
/// existing magnitude convention.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CfdFieldLayout {
    pub stride: u32,
    pub u_offset: u32,
    pub p_offset: u32,
    pub has_u: bool,
    pub has_p: bool,
}

impl CfdFieldLayout {
    fn validated(self) -> Self {
        let stride = self.stride.max(1);
        Self {
            stride,
            u_offset: self.u_offset,
            p_offset: self.p_offset,
            has_u: self.has_u
                && self
                    .u_offset
                    .checked_add(1)
                    .is_some_and(|last| last < stride),
            has_p: self.has_p && self.p_offset < stride,
        }
    }
}

/// GPU-produced metadata for one visualization snapshot. The sequence is
/// written by the compute pass (not inferred from a slot index), then verified
/// by the asynchronous readback path before it can update the legend.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CfdFieldRanges {
    pub minimum: [f32; 4],
    pub maximum: [f32; 4],
    sequence_lo: u32,
    sequence_hi: u32,
    _padding: [u32; 2],
}

impl CfdFieldRanges {
    pub const fn fallback(sequence: u64) -> Self {
        Self {
            minimum: [0.0; 4],
            maximum: [1.0; 4],
            sequence_lo: sequence as u32,
            sequence_hi: (sequence >> 32) as u32,
            _padding: [0; 2],
        }
    }

    pub const fn sequence(self) -> u64 {
        (self.sequence_lo as u64) | ((self.sequence_hi as u64) << 32)
    }

    pub fn range(self, field: CfdRangeField) -> [f32; 2] {
        let index = field.index();
        sanitize_range(self.minimum[index], self.maximum[index])
    }
}

impl Default for CfdFieldRanges {
    fn default() -> Self {
        Self::fallback(0)
    }
}

/// Minimum displayed span, in f32 quanta (ULPs) of the field's own magnitude.
///
/// The solver state is f32, so a field whose global span is only a handful of
/// quanta of its absolute value (e.g. sub-Pa acoustics riding on a 105 kPa
/// absolute pressure, one quantum ~= 0.0078 Pa) carries representation noise
/// of about one quantum per cell. Normalizing the colormap to such a span
/// renders that noise as full-scale speckle. Flooring the displayed span at
/// this many quanta keeps one quantum a small (~1.6%) fraction of the color
/// range. Fields stored near zero (gauge pressure, velocities) have tiny
/// magnitudes, so the floor never binds for them.
///
/// Keep in sync with the final guard in `cfd_range_reduce.wgsl` (the Direct
/// route's mesh shader reads the GPU-reduced buffer without host involvement).
pub const PRECISION_FLOOR_ULPS: f32 = 64.0;

fn sanitize_range(minimum: f32, maximum: f32) -> [f32; 2] {
    if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
        return [0.0, 1.0];
    }
    if (maximum - minimum).abs() < 1.0e-12 {
        let expanded = minimum + 1.0;
        if expanded.is_finite() && expanded > minimum {
            return [minimum, expanded];
        }
        return [0.0, 1.0];
    }
    let floor_span = PRECISION_FLOOR_ULPS * f32::EPSILON * minimum.abs().max(maximum.abs());
    if maximum - minimum < floor_span {
        let mid = 0.5 * (minimum + maximum);
        return [mid - 0.5 * floor_span, mid + 0.5 * floor_span];
    }
    [minimum, maximum]
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CfdRangeConfig {
    cell_count: u32,
    stride: u32,
    u_offset: u32,
    p_offset: u32,
    has_u: u32,
    has_p: u32,
    sequence_lo: u32,
    sequence_hi: u32,
}

impl CfdRangeConfig {
    fn new(layout: CfdFieldLayout, cell_count: u32, sequence: u64) -> Self {
        let layout = layout.validated();
        Self {
            cell_count,
            stride: layout.stride,
            u_offset: layout.u_offset,
            p_offset: layout.p_offset,
            has_u: u32::from(layout.has_u),
            has_p: u32::from(layout.has_p),
            sequence_lo: sequence as u32,
            sequence_hi: (sequence >> 32) as u32,
        }
    }
}

#[derive(Clone)]
struct CfdRangeReadbackSlot {
    buffer: wgpu::Buffer,
    busy: Arc<AtomicBool>,
}

struct PendingRangeReadback {
    buffer: wgpu::Buffer,
    busy: Arc<AtomicBool>,
    results: Arc<Mutex<CfdRangeReadbackQueue>>,
    sequence: u64,
}

impl PendingRangeReadback {
    fn map_on_submit(self, command: &wgpu::CommandBuffer) {
        let callback_buffer = self.buffer.clone();
        let busy = Arc::clone(&self.busy);
        let results = Arc::clone(&self.results);
        let sequence = self.sequence;
        let size = std::mem::size_of::<CfdFieldRanges>() as u64;
        command.map_buffer_on_submit(
            &self.buffer,
            wgpu::MapMode::Read,
            0..size,
            move |result| {
                if result.is_ok() {
                    let view = callback_buffer.slice(0..size).get_mapped_range();
                    let ranges = *bytemuck::from_bytes::<CfdFieldRanges>(&view);
                    drop(view);
                    callback_buffer.unmap();
                    // Verify the tag written by the GPU rather than trusting
                    // the host closure's requested sequence.
                    if ranges.sequence() == sequence {
                        results.lock().unwrap().publish(ranges);
                    }
                }
                busy.store(false, Ordering::Release);
            },
        );
    }
}

#[derive(Default)]
struct CfdRangeReadbackQueue {
    sequence_floor: Option<u64>,
    pending: Vec<CfdFieldRanges>,
}

impl CfdRangeReadbackQueue {
    fn publish(&mut self, ranges: CfdFieldRanges) {
        let sequence = ranges.sequence();
        if self.sequence_floor.is_some_and(|floor| sequence < floor) {
            return;
        }
        if let Some(existing) = self
            .pending
            .iter_mut()
            .find(|existing| existing.sequence() == sequence)
        {
            *existing = ranges;
            return;
        }
        self.pending.push(ranges);
        self.pending
            .sort_unstable_by_key(|ranges| ranges.sequence());
        // The GPU readback pool bounds normal occupancy. This cap is defensive
        // if callbacks accumulate while the UI is temporarily not consuming.
        if self.pending.len() > 8 {
            self.pending.drain(..self.pending.len() - 8);
        }
    }

    /// Consume only metadata for the snapshot that is actually bound. Older
    /// callbacks are discarded and future callbacks remain queued.
    fn take_exact(&mut self, sequence: u64) -> Option<CfdFieldRanges> {
        self.sequence_floor = Some(
            self.sequence_floor
                .map_or(sequence, |floor| floor.max(sequence)),
        );
        self.pending.retain(|ranges| ranges.sequence() >= sequence);
        let index = self
            .pending
            .iter()
            .position(|ranges| ranges.sequence() == sequence)?;
        Some(self.pending.remove(index))
    }
}

#[derive(Clone)]
struct CfdRangeSlot {
    config_buffer: wgpu::Buffer,
    range_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Frame-cadence GPU reduction plus a best-effort tiny asynchronous readback
/// for legend labels. The renderer consumes `range_buffer` directly, so a busy
/// readback slot can delay text but can never make the colormap stale.
#[derive(Clone)]
pub struct CfdRangeReducer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: CfdFieldLayout,
    slots: [CfdRangeSlot; 3],
    readback_slots: [CfdRangeReadbackSlot; 3],
    readback_queue: Arc<Mutex<CfdRangeReadbackQueue>>,
    #[cfg(test)]
    snapshot_submissions: Arc<AtomicU64>,
}

impl CfdRangeReducer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field_buffers: &[wgpu::Buffer; 3],
        layout: CfdFieldLayout,
    ) -> Self {
        let layout = layout.validated();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CFD Field Range Reduction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cfd_range_reduce.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CFD Field Range Reduction Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CFD Field Range Reduction Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CFD Field Range Reduction Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce_ranges"),
            compilation_options: Default::default(),
            cache: None,
        });

        let slots = std::array::from_fn(|index| {
            let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CFD Field Range Config"),
                contents: bytemuck::bytes_of(&CfdRangeConfig::new(layout, 0, 0)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let range_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CFD Field Range Metadata"),
                contents: bytemuck::bytes_of(&CfdFieldRanges::fallback(0)),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CFD Field Range Reduction Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: field_buffers[index].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: range_buffer.as_entire_binding(),
                    },
                ],
            });
            CfdRangeSlot {
                config_buffer,
                range_buffer,
                bind_group,
            }
        });

        let readback_slots = std::array::from_fn(|_| CfdRangeReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CFD Field Range Async Readback"),
                size: std::mem::size_of::<CfdFieldRanges>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            busy: Arc::new(AtomicBool::new(false)),
        });

        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            layout,
            slots,
            readback_slots,
            readback_queue: Arc::new(Mutex::new(CfdRangeReadbackQueue::default())),
            #[cfg(test)]
            snapshot_submissions: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn range_buffer(&self, slot: usize) -> &wgpu::Buffer {
        &self.slots[slot].range_buffer
    }

    pub const fn stride(&self) -> u32 {
        self.layout.stride
    }

    /// Range-only compatibility path used for the initialized slot zero.
    pub fn submit(&self, slot: usize, sequence: u64, cell_count: usize) {
        self.submit_snapshot(slot, sequence, cell_count, |_| Ok(()))
            .expect("range-only encoding cannot fail");
    }

    /// Encode the field snapshot and its range reduction into one command
    /// buffer and submit exactly once. The prefix runs before a readback slot is
    /// reserved, so a validation/capability error cannot strand that slot busy.
    pub(crate) fn submit_snapshot<F>(
        &self,
        slot: usize,
        sequence: u64,
        cell_count: usize,
        encode_snapshot: F,
    ) -> Result<wgpu::SubmissionIndex, String>
    where
        F: FnOnce(&mut wgpu::CommandEncoder) -> Result<(), String>,
    {
        if slot >= self.slots.len() {
            return Err(format!("CFD range slot {slot} is outside 0..{}", self.slots.len()));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CFD Direct Snapshot + Range"),
            });
        encode_snapshot(&mut encoder)?;
        let readback = self.encode_range(&mut encoder, slot, sequence, cell_count);
        let command = encoder.finish();
        if let Some(readback) = readback {
            readback.map_on_submit(&command);
        }
        let submission = self.queue.submit(Some(command));
        crate::count_submission!("UI Direct", "snapshot_and_range");
        #[cfg(test)]
        self.snapshot_submissions.fetch_add(1, Ordering::Relaxed);
        Ok(submission)
    }

    #[cfg(test)]
    pub(crate) fn snapshot_submission_count(&self) -> u64 {
        self.snapshot_submissions.load(Ordering::Relaxed)
    }

    fn encode_range(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot_index: usize,
        sequence: u64,
        cell_count: usize,
    ) -> Option<PendingRangeReadback> {
        let slot = &self.slots[slot_index];
        let cell_count = u32::try_from(cell_count).unwrap_or(u32::MAX);
        let config = CfdRangeConfig::new(self.layout, cell_count, sequence);
        self.queue
            .write_buffer(&slot.config_buffer, 0, bytemuck::bytes_of(&config));

        let readback = self.readback_slots.iter().find(|candidate| {
            candidate
                .busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CFD Field Range Reduction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &slot.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if let Some(readback) = readback {
            encoder.copy_buffer_to_buffer(
                &slot.range_buffer,
                0,
                &readback.buffer,
                0,
                std::mem::size_of::<CfdFieldRanges>() as u64,
            );
        }
        readback.map(|readback| PendingRangeReadback {
            buffer: readback.buffer.clone(),
            busy: Arc::clone(&readback.busy),
            results: Arc::clone(&self.readback_queue),
            sequence,
        })
    }

    /// Drive map callbacks without ever waiting for GPU completion.
    pub fn poll_nonblocking(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map(|_| ())
            .map_err(|error| format!("nonblocking CFD range poll failed: {error}"))
    }

    pub fn take_readback_for_sequence(&self, sequence: u64) -> Option<CfdFieldRanges> {
        self.readback_queue.lock().unwrap().take_exact(sequence)
    }
}

/// Vertex structure for the CFD mesh shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CfdVertex {
    /// Position in world coordinates
    pub position: [f32; 2],
    /// Cell index for looking up field values
    pub cell_index: u32,
    pub _padding: u32,
}

/// Uniform buffer for transform and rendering settings
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CfdUniforms {
    /// Transform matrix (2D affine: scale_x, scale_y, translate_x, translate_y)
    pub transform: [f32; 4],
    /// Viewport size in pixels
    pub viewport_size: [f32; 2],
    /// Value range for colormap [min, max]
    pub range: [f32; 2],
    /// Stride between elements (1 for scalar, 2 for vector)
    pub stride: u32,
    /// Offset to start reading (0 for scalar/x, 1 for y)
    pub offset: u32,
    /// Visualization mode (0: value, 1: magnitude)
    pub mode: u32,
    /// Index into [`CfdFieldRanges`] (pressure, Ux, Uy, |U|).
    pub range_index: u32,
}

/// Headroom over the initial tessellated count for a moving (ALE) mesh, which
/// re-tessellates every step with drifting per-cell vertex counts, so early
/// refreshes fit without a reallocation.
pub const VERTEX_HEADROOM: f32 = 1.5;

/// Headroom factor for a mesh that never grows (the static path): allocate
/// exactly the initial count.
pub const NO_HEADROOM: f32 = 1.0;

/// Over-allocation factor when a refresh outgrows the current allocation, so a
/// steadily growing topology does not reallocate every step.
const VERTEX_GROW_FACTOR: f32 = 1.5;

/// Round a required vertex count up to an allocation size using `factor`,
/// never returning less than `required` (or less than 1).
fn capacity_for(required: usize, factor: f32) -> usize {
    let scaled = (required as f32 * factor).ceil() as usize;
    scaled.max(required).max(1)
}

/// Allocate a vertex buffer holding `capacity` `CfdVertex` slots.
fn alloc_vertex_buffer(device: &wgpu::Device, label: &str, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (capacity.max(1) * std::mem::size_of::<CfdVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// GPU resources for CFD rendering
#[derive(Clone)]
pub struct CfdRenderResources {
    pub pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub line_vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub field_buffer: wgpu::Buffer,
    pub range_buffer: wgpu::Buffer,
    /// Generation of the visualization snapshot currently bound at binding 1.
    /// Slot indices are intentionally insufficient: triple-buffer slots are
    /// recycled, so the sequence prevents a delayed UI observation from
    /// rebinding an older generation after a newer one.
    pub field_sequence: u64,
    /// CPU-visible legend metadata is accepted only for `field_sequence`.
    /// `None` means the tiny asynchronous readback has not completed yet.
    legend_ranges: Option<CfdFieldRanges>,
    pub num_vertices: u32,
    pub num_line_vertices: u32,
    /// Allocated `CfdVertex` slots in `vertex_buffer` (the triangle-fill buffer).
    pub capacity_vertices: usize,
    /// Allocated `CfdVertex` slots in `line_vertex_buffer` (the wireframe buffer).
    pub capacity_line_vertices: usize,
}

impl CfdRenderResources {
    /// Create render resources for CFD visualization
    /// `headroom` scales the initial fill/line buffer allocation over
    /// `max_vertices`: pass [`VERTEX_HEADROOM`] for a moving (ALE) mesh that
    /// re-tessellates every step, or [`NO_HEADROOM`] for the static path so it
    /// allocates exactly what it needs (overflow is still impossible either way
    /// — `ensure_capacity` grows on demand before every write).
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        max_vertices: usize,
        headroom: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CFD Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cfd_mesh_shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CFD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CFD Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CFD Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CfdVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Uint32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CFD Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CfdVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Uint32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Draw range is always `num_vertices` (<= the written count), so a
        // larger allocation is visually identical for the static path.
        let capacity_vertices = capacity_for(max_vertices, headroom);
        let capacity_line_vertices = capacity_vertices;
        let vertex_buffer = alloc_vertex_buffer(device, "CFD Vertex Buffer", capacity_vertices);
        let line_vertex_buffer =
            alloc_vertex_buffer(device, "CFD Line Vertex Buffer", capacity_line_vertices);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CFD Uniform Buffer"),
            contents: bytemuck::bytes_of(&CfdUniforms {
                transform: [1.0, 1.0, 0.0, 0.0],
                viewport_size: [1.0, 1.0],
                range: [0.0, 1.0],
                stride: 1,
                offset: 0,
                mode: 0,
                range_index: CfdRangeField::Pressure as u32,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create dummy storage buffer for initial bind group
        let field_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Field Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let range_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy CFD Field Range Metadata"),
            contents: bytemuck::bytes_of(&CfdFieldRanges::fallback(0)),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CFD Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: range_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            line_pipeline,
            vertex_buffer,
            line_vertex_buffer,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            field_buffer,
            range_buffer,
            field_sequence: 0,
            legend_ranges: None,
            num_vertices: 0,
            num_line_vertices: 0,
            capacity_vertices,
            capacity_line_vertices,
        }
    }

    /// True if the current allocation can hold `vertex_count` fill vertices and
    /// `line_count` wireframe vertices without a reallocation.
    pub fn can_fit(&self, vertex_count: usize, line_count: usize) -> bool {
        vertex_count <= self.capacity_vertices && line_count <= self.capacity_line_vertices
    }

    /// Ensure the vertex/line buffers can hold at least `vertex_count` /
    /// `line_count` vertices, growing (reallocating) any buffer that is too
    /// small. Growing replaces the buffer handle only; the vertex buffers are
    /// bound per-draw via `set_vertex_buffer` (not through the bind group), so
    /// no rebind is required. Returns true if any buffer was reallocated.
    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        vertex_count: usize,
        line_count: usize,
    ) -> bool {
        let mut grew = false;
        if vertex_count > self.capacity_vertices {
            self.capacity_vertices = capacity_for(vertex_count, VERTEX_GROW_FACTOR);
            self.vertex_buffer =
                alloc_vertex_buffer(device, "CFD Vertex Buffer", self.capacity_vertices);
            grew = true;
        }
        if line_count > self.capacity_line_vertices {
            self.capacity_line_vertices = capacity_for(line_count, VERTEX_GROW_FACTOR);
            self.line_vertex_buffer = alloc_vertex_buffer(
                device,
                "CFD Line Vertex Buffer",
                self.capacity_line_vertices,
            );
            grew = true;
        }
        grew
    }

    /// Update the vertex buffer with new mesh data.
    ///
    /// Overflow-proof by construction: `ensure_capacity` grows the target
    /// buffer(s) to fit *before* any `write_buffer`, so the write can never
    /// exceed the allocation regardless of how the mesh topology (and hence the
    /// per-cell vertex count) changed.
    pub fn update_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[CfdVertex],
        line_vertices: &[CfdVertex],
    ) {
        self.ensure_capacity(device, vertices.len(), line_vertices.len());

        self.num_vertices = vertices.len() as u32;
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));

        self.num_line_vertices = line_vertices.len() as u32;
        queue.write_buffer(
            &self.line_vertex_buffer,
            0,
            bytemuck::cast_slice(line_vertices),
        );
    }

    /// Update the uniform buffer with new transform
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &CfdUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Bind field values and their GPU-produced ranges as one snapshot pair.
    fn update_bind_group(
        &mut self,
        device: &wgpu::Device,
        field_buffer: &wgpu::Buffer,
        range_buffer: &wgpu::Buffer,
    ) {
        self.field_buffer = field_buffer.clone();
        self.range_buffer = range_buffer.clone();
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CFD Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.range_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Bind the initialized slot-zero pair. Sequence zero is a valid initial
    /// snapshot, so it deliberately bypasses the strictly-newer live guard.
    pub fn bind_initial_snapshot(
        &mut self,
        device: &wgpu::Device,
        field_buffer: &wgpu::Buffer,
        range_buffer: &wgpu::Buffer,
    ) {
        self.update_bind_group(device, field_buffer, range_buffer);
        self.field_sequence = 0;
        self.legend_ranges = None;
    }

    /// Bind a completed visualization snapshot if it is newer than the field
    /// already displayed. The mailbox in `ui::app` supplies the sequence and
    /// owns the slot state transition; keeping the generation here adds a final
    /// monotonicity guard at the actual renderer-resource boundary.
    pub fn update_field_snapshot(
        &mut self,
        device: &wgpu::Device,
        field_buffer: &wgpu::Buffer,
        range_buffer: &wgpu::Buffer,
        sequence: u64,
    ) -> bool {
        if sequence <= self.field_sequence {
            return false;
        }
        self.update_bind_group(device, field_buffer, range_buffer);
        self.field_sequence = sequence;
        // Deliberately KEEP the previous frame's legend metadata: its exact
        // readback for this sequence arrives asynchronously a frame or so
        // later, and while frames stream continuously that gap covers most
        // paints. A <=1-frame-stale number is indistinguishable to the reader;
        // clearing here made the legend flash its 0..1 placeholder on nearly
        // every other paint. (Drawn COLORS never consult this CPU value — the
        // shader normalizes from the GPU-resident `field_ranges` buffer that
        // is always sequence-consistent with the bound field.)
        true
    }

    /// Adopt asynchronous legend metadata only when it belongs to the exact
    /// field/range pair currently bound for drawing.
    pub fn update_legend_ranges(&mut self, ranges: CfdFieldRanges) -> bool {
        if ranges.sequence() != self.field_sequence {
            return false;
        }
        self.legend_ranges = Some(ranges);
        true
    }

    /// Freshest measured range for the legend text. May lag the displayed
    /// field by a frame while its readback is in flight; the 0..1 placeholder
    /// appears only before the first readback of a run (or after a reset).
    pub fn legend_range(&self, field: CfdRangeField) -> [f32; 2] {
        self.legend_ranges
            .map_or([0.0, 1.0], |ranges| ranges.range(field))
    }

    /// Render the CFD mesh
    /// Note: The render pass lifetime is generic to work with egui_wgpu's CallbackTrait
    pub fn paint<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, draw_lines: bool) {
        if self.num_vertices > 0 {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        if draw_lines && self.num_line_vertices > 0 {
            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.num_line_vertices, 0..1);
        }
    }
}

/// Build mesh vertices from cell data for GPU rendering
pub fn build_mesh_vertices(cells: &[Vec<[f64; 2]>]) -> Vec<CfdVertex> {
    let mut vertices = Vec::new();

    for (cell_idx, polygon) in cells.iter().enumerate() {
        if polygon.len() < 3 {
            continue;
        }

        // Fan triangulation
        for i in 1..polygon.len() - 1 {
            vertices.push(CfdVertex {
                position: [polygon[0][0] as f32, polygon[0][1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
            vertices.push(CfdVertex {
                position: [polygon[i][0] as f32, polygon[i][1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
            vertices.push(CfdVertex {
                position: [polygon[i + 1][0] as f32, polygon[i + 1][1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
        }
    }

    vertices
}

/// Build line vertices from cell data for GPU rendering
pub fn build_line_vertices(cells: &[Vec<[f64; 2]>]) -> Vec<CfdVertex> {
    let mut vertices = Vec::new();

    for (cell_idx, polygon) in cells.iter().enumerate() {
        if polygon.len() < 2 {
            continue;
        }

        for i in 0..polygon.len() {
            let p1 = polygon[i];
            let p2 = polygon[(i + 1) % polygon.len()];

            vertices.push(CfdVertex {
                position: [p1[0] as f32, p1[1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
            vertices.push(CfdVertex {
                position: [p2[0] as f32, p2[1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
        }
    }

    vertices
}

/// Compute the bounding box of all cells
pub fn compute_bounds(cells: &[Vec<[f64; 2]>]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for polygon in cells {
        for &[x, y] in polygon {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    (min_x, max_x, min_y, max_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tagged_ranges(sequence: u64, minimum: f32, maximum: f32) -> CfdFieldRanges {
        let mut ranges = CfdFieldRanges::fallback(sequence);
        ranges.minimum[0] = minimum;
        ranges.maximum[0] = maximum;
        ranges
    }

    #[test]
    fn range_readbacks_are_consumed_only_for_the_exact_display_sequence() {
        let mut queue = CfdRangeReadbackQueue::default();
        queue.publish(tagged_ranges(4, 4.0, 5.0));
        queue.publish(tagged_ranges(2, 2.0, 3.0));

        // A future callback cannot replace the range for an older bound frame.
        assert!(queue.take_exact(3).is_none());
        queue.publish(tagged_ranges(3, 3.0, 4.0));
        assert_eq!(
            queue.take_exact(3).unwrap().range(CfdRangeField::Pressure),
            [3.0, 4.0]
        );

        // Sequence four was retained, while a late sequence two publication is
        // below the consumed floor and can never become visible again.
        queue.publish(tagged_ranges(2, -20.0, -10.0));
        assert_eq!(
            queue.take_exact(4).unwrap().range(CfdRangeField::Pressure),
            [4.0, 5.0]
        );
        assert!(queue.take_exact(2).is_none());
    }

    #[test]
    fn range_sanitization_is_deterministic_for_constant_and_invalid_fields() {
        assert_eq!(sanitize_range(7.0, 7.0), [7.0, 8.0]);
        assert_eq!(sanitize_range(f32::NAN, 2.0), [0.0, 1.0]);
        assert_eq!(sanitize_range(-1.0, f32::INFINITY), [0.0, 1.0]);
        assert_eq!(sanitize_range(3.0, 2.0), [0.0, 1.0]);
    }

    #[test]
    fn range_sanitization_floors_sub_ulp_spans_at_the_representable_precision() {
        // A 0.3 Pa acoustic span on a 105 kPa absolute f32 pressure is ~38
        // quanta; stretching it over the whole colormap displays f32
        // representation noise as full-scale speckle. The floor widens the
        // span to PRECISION_FLOOR_ULPS quanta around the same midpoint.
        let lo = 105472.35_f32;
        let hi = 105472.65_f32;
        let [flo, fhi] = sanitize_range(lo, hi);
        let floor_span = PRECISION_FLOOR_ULPS * f32::EPSILON * hi.abs();
        assert!(fhi - flo >= floor_span * 0.99);
        let mid = 0.5 * (lo + hi);
        assert!((0.5 * (flo + fhi) - mid).abs() <= floor_span * 0.01);

        // A healthy span (signal well above the representation quantum) is
        // returned untouched, as are gauge/zero-centered fields.
        assert_eq!(sanitize_range(105000.0, 106000.0), [105000.0, 106000.0]);
        assert_eq!(sanitize_range(-0.5, 0.5), [-0.5, 0.5]);
        assert_eq!(sanitize_range(-1.0e-6, 1.0e-6), [-1.0e-6, 1.0e-6]);
    }

    #[test]
    fn gpu_reduction_handles_arbitrary_packed_offsets_and_all_four_fields() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("skipping GPU range reduction test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        else {
            eprintln!("skipping GPU range reduction test: no device");
            return;
        };

        // stride=6, pressure at +1 and velocity at +3/+4. The unused values
        // make an accidental tightly-packed assumption immediately visible.
        let packed: [f32; 18] = [
            91.0,
            -4.0,
            92.0,
            3.0,
            4.0,
            93.0,
            81.0,
            8.0,
            82.0,
            -5.0,
            12.0,
            83.0,
            71.0,
            f32::NAN,
            72.0,
            f32::INFINITY,
            1.0,
            73.0,
        ];
        let field_buffers = std::array::from_fn(|_| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CFD range test packed field"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });
        let snapshot_source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CFD range test snapshot source"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let reducer = CfdRangeReducer::new(
            &device,
            &queue,
            &field_buffers,
            CfdFieldLayout {
                stride: 6,
                u_offset: 3,
                p_offset: 1,
                has_u: true,
                has_p: true,
            },
        );

        let submissions_before = reducer.snapshot_submission_count();
        let failed = reducer.submit_snapshot(1, 41, 3, |_| {
            Err("injected snapshot-prefix failure".to_string())
        });
        assert!(failed.is_err());
        assert_eq!(
            reducer.snapshot_submission_count(),
            submissions_before,
            "a failed prefix must not submit or reserve range readback work"
        );

        reducer
            .submit_snapshot(0, 0x1_0000_002a, 3, |encoder| {
                encoder.copy_buffer_to_buffer(
                    &snapshot_source,
                    0,
                    &field_buffers[0],
                    0,
                    std::mem::size_of_val(&packed) as u64,
                );
                Ok(())
            })
            .expect("single-submit Direct snapshot + range");
        assert_eq!(
            reducer.snapshot_submission_count(),
            submissions_before + 1,
            "field copy and range reduction must share one submission"
        );
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .expect("range reduction completion poll");
        let ranges = reducer
            .take_readback_for_sequence(0x1_0000_002a)
            .expect("sequence-tagged range readback");
        assert_eq!(ranges.sequence(), 0x1_0000_002a);
        assert_eq!(ranges.range(CfdRangeField::Pressure), [-4.0, 8.0]);
        assert_eq!(ranges.range(CfdRangeField::VelocityX), [-5.0, 3.0]);
        assert_eq!(ranges.range(CfdRangeField::VelocityY), [1.0, 12.0]);
        assert_eq!(ranges.range(CfdRangeField::VelocityMagnitude), [5.0, 13.0]);
    }

    #[test]
    fn legend_holds_previous_range_until_the_new_readback_lands() {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("skipping legend hold test: no adapter");
            return;
        };
        let Ok((device, _queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        else {
            eprintln!("skipping legend hold test: no device");
            return;
        };
        let buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let field_buffer = buffer("legend hold field");
        let range_buffer = buffer("legend hold range");

        let mut renderer =
            CfdRenderResources::new(&device, wgpu::TextureFormat::Rgba8Unorm, 3, NO_HEADROOM);

        // Before any readback the placeholder is all the legend can show.
        assert_eq!(renderer.legend_range(CfdRangeField::Pressure), [0.0, 1.0]);

        assert!(renderer.update_field_snapshot(&device, &field_buffer, &range_buffer, 1));
        assert!(renderer.update_legend_ranges(tagged_ranges(1, 2.0, 3.0)));
        assert_eq!(renderer.legend_range(CfdRangeField::Pressure), [2.0, 3.0]);

        // Binding the next streamed frame must NOT flash the 0..1 placeholder
        // while frame 2's readback is still in flight: the previous measured
        // range holds.
        assert!(renderer.update_field_snapshot(&device, &field_buffer, &range_buffer, 2));
        assert_eq!(renderer.legend_range(CfdRangeField::Pressure), [2.0, 3.0]);

        // A readback for a frame other than the displayed one is rejected...
        assert!(!renderer.update_legend_ranges(tagged_ranges(1, 7.0, 8.0)));
        assert_eq!(renderer.legend_range(CfdRangeField::Pressure), [2.0, 3.0]);

        // ...and the exact readback replaces the held text.
        assert!(renderer.update_legend_ranges(tagged_ranges(2, 4.0, 5.0)));
        assert_eq!(renderer.legend_range(CfdRangeField::Pressure), [4.0, 5.0]);
    }

    #[test]
    fn direct_renderer_rasterizes_velocity_magnitude_as_blue_green_red() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("skipping Direct raster test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        else {
            eprintln!("skipping Direct raster test: no device");
            return;
        };

        // Three disjoint cells carry |U|=0,5,10. This is the GUI's default
        // plotted field and the exact cold-start regression reported by the
        // interactive app. Sampling well inside each cell checks the final
        // field/range binding, strided vector lookup, magnitude branch, shader
        // normalization, and fragment output together.
        let cells = vec![
            vec![[0.0, 0.0], [1.0 / 3.0, 0.0], [1.0 / 3.0, 1.0], [0.0, 1.0]],
            vec![[1.0 / 3.0, 0.0], [2.0 / 3.0, 0.0], [2.0 / 3.0, 1.0], [1.0 / 3.0, 1.0]],
            vec![[2.0 / 3.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0 / 3.0, 1.0]],
        ];
        let vertices = build_mesh_vertices(&cells);
        let field_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Direct raster test field"),
            // Packed [p, Ux, Uy] per cell. Pressure is deliberately constant,
            // so a wrong offset/range-index path renders uniformly instead of
            // accidentally passing through a correlated scalar field.
            contents: bytemuck::cast_slice(&[
                7.0_f32, 0.0, 0.0, 7.0, 3.0, 4.0, 7.0, 6.0, 8.0,
            ]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut ranges = CfdFieldRanges::fallback(0);
        ranges.minimum[CfdRangeField::VelocityMagnitude as usize] = 0.0;
        ranges.maximum[CfdRangeField::VelocityMagnitude as usize] = 10.0;
        let range_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Direct raster test range"),
            contents: bytemuck::bytes_of(&ranges),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = CfdRenderResources::new(&device, format, vertices.len(), NO_HEADROOM);
        renderer.update_mesh(&device, &queue, &vertices, &[]);
        renderer.bind_initial_snapshot(&device, &field_buffer, &range_buffer);
        renderer.update_uniforms(
            &queue,
            &CfdUniforms {
                transform: [1.0, 1.0, 0.0, 0.0],
                viewport_size: [12.0, 4.0],
                range: [0.0, 10.0],
                stride: 3,
                offset: 1,
                mode: 1,
                range_index: CfdRangeField::VelocityMagnitude as u32,
            },
        );

        let extent = wgpu::Extent3d {
            width: 12,
            height: 4,
            depth_or_array_layers: 1,
        };
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Direct raster test target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let bytes_per_row = 256_u32;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Direct raster test readback"),
            size: u64::from(bytes_per_row * extent.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Direct raster test encoder"),
        });
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Direct raster test pass"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            renderer.paint(&mut pass, false);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(extent.height),
                },
            },
            extent,
        );
        queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .expect("Direct raster completion poll");
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("Direct raster map callback")
            .expect("Direct raster map");
        let mapped = readback.slice(..).get_mapped_range();
        let pixel = |x: usize, y: usize| {
            let offset = y * bytes_per_row as usize + x * 4;
            [mapped[offset], mapped[offset + 1], mapped[offset + 2], mapped[offset + 3]]
        };
        assert_eq!(pixel(2, 2), [0, 0, 255, 255], "minimum must render blue");
        assert_eq!(pixel(6, 2), [0, 255, 0, 255], "midpoint must render green");
        assert_eq!(pixel(10, 2), [255, 0, 0, 255], "maximum must render red");
        drop(mapped);
        readback.unmap();
    }
}
