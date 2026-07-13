//! GPU-resident `derive_faces` — turns the engine's cell-major padded face
//! slots into face-major addressing without a CPU round-trip, via count → scan:
//!
//!  1. `count_owned_faces` — one thread per cell counts the faces this cell
//!     *owns* under the canonical `owner = min(i, j)` rule (interior face `i↔j`
//!     owned by the lower id; every boundary/box face owned by its cell).
//!  2. [`crate::solver::gpu::voronoi::GpuScan`] — exclusive-scan those counts →
//!     per-cell base offsets into the face-major arrays; the grand total is the
//!     new `num_faces`.
//!
//! Capped at [`MAX_SCAN_ELEMS`] = 2²⁰ cells by the two-level scan.
//!
//! Owner-count assumes a reciprocal diagram (every `i↔j` seen from both cells,
//! both SUCCESS): `count_owned_faces` emits an interior face `i<j` from `i`'s
//! side without re-checking `j`. A non-reciprocal f32 clip would disagree with
//! `assemble_mesh`; the `emit`/geometry passes are not landed here.

use crate::solver::gpu::buffers::create_buffer;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

use super::engine::GpuVoronoiEngine;
use super::scan::GpuScan;
use super::K_FACE_MAX;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeriveParams {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-derived face addressing (count → scan half).
#[derive(Clone, Debug)]
pub struct DerivedFaceOffsets {
    /// Per-cell owned-face count (canonical `owner = min(i,j)` rule).
    pub owned_counts: Vec<u32>,
    /// Per-cell exclusive-scan base offset into the face-major arrays.
    pub offsets: Vec<u32>,
    /// Grand total = new `num_faces`.
    pub total: u32,
}

/// Count → scan result with the per-cell base offsets kept ON DEVICE (for the
/// emit pass to consume directly) plus the scalar `num_faces` total. Only the
/// n-u32 offsets/counts (the addressing, not the O(num_faces) face geometry) are
/// read back to compute the total.
pub struct ScanOffsets {
    /// Per-cell exclusive-scan base offsets, on device (size `n * 4` bytes).
    pub b_offsets: wgpu::Buffer,
    /// Grand total = new `num_faces`.
    pub total: u32,
}

/// The `count_owned_faces` pipeline (the scan is owned separately).
pub struct DeriveFaces {
    count_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    scan: GpuScan,
}

impl DeriveFaces {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("derive:bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),  // nbr_ids
                storage(2, true),  // cell_nfaces
                storage(3, true),  // status
                storage(4, false), // owned_counts (out)
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("derive:pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("derive:shader"),
            source: wgpu::ShaderSource::Wgsl(count_shader().into()),
        });
        let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("derive:count"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("count_owned_faces"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { count_pipeline, bgl, scan: GpuScan::new(device) }
    }

    /// Count → scan over the engine's cell-major outputs, producing the per-cell
    /// owned-face count buffer + exclusive-scan offset buffer ON DEVICE (one
    /// encoder, no intermediate readback). Both buffers have `COPY_SRC` so the
    /// caller can read them back (`derive_offsets`) or feed them straight to the
    /// emit pass (`encode_offsets`).
    fn run_count_scan(
        &self,
        ctx: &GpuContext,
        engine: &GpuVoronoiEngine,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        use wgpu::BufferUsages as U;
        let n = engine.n_seeds();
        assert!(n > 0, "run_count_scan called before a regen (n_seeds == 0)");
        assert!(
            n <= super::MAX_SCAN_ELEMS,
            "run_count_scan: {n} cells exceeds the two-level scan ceiling \
             ({} = 2^20); needs the v2 multi-level scan (design-gpu §4.1)",
            super::MAX_SCAN_ELEMS,
        );
        let params = DeriveParams { n, k: K_FACE_MAX as u32, _pad0: 0, _pad1: 0 };
        let b_params = create_buffer(
            &ctx.device,
            "derive:params",
            std::mem::size_of::<DeriveParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        ctx.queue.write_buffer(&b_params, 0, bytemuck::bytes_of(&params));

        let counts_bytes = (n as u64) * 4;
        let b_owned = create_buffer(
            &ctx.device,
            "derive:owned_counts",
            counts_bytes,
            U::STORAGE | U::COPY_SRC,
        );
        let b_offsets =
            create_buffer(&ctx.device, "derive:offsets", counts_bytes, U::STORAGE | U::COPY_SRC);
        let b_block_sums = create_buffer(
            &ctx.device,
            "derive:block_sums",
            (GpuScan::num_blocks(n) as u64) * 4,
            U::STORAGE | U::COPY_SRC,
        );
        let b_scan_params = create_buffer(
            &ctx.device,
            "derive:scan_params",
            16, // ScanParams: 4 x u32
            U::UNIFORM | U::COPY_DST,
        );

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("derive:bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_params.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: engine.outputs.b_nbr_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: engine.outputs.b_cell_nfaces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: engine.outputs.b_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 4, resource: b_owned.as_entire_binding() },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("derive:enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("derive:count_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y) = dispatch_2d(n.div_ceil(WORKGROUP_SIZE).max(1));
            pass.dispatch_workgroups(x, y, 1);
        }
        // count → scan in the same encoder — no intermediate readback.
        self.scan.encode(
            &ctx.device,
            &ctx.queue,
            &mut encoder,
            &b_owned,
            &b_offsets,
            &b_block_sums,
            &b_scan_params,
            n,
        );
        ctx.queue.submit(Some(encoder.finish()));
        (b_owned, b_offsets)
    }

    /// Run count → scan and read back the per-cell owned-face counts + offsets +
    /// total (the inspection/validation path — `gpu_derive_faces_test`).
    pub fn derive_offsets(
        &self,
        ctx: &GpuContext,
        cache: &StagingBufferCache,
        engine: &GpuVoronoiEngine,
    ) -> DerivedFaceOffsets {
        let n = engine.n_seeds();
        let counts_bytes = (n as u64) * 4;
        let (b_owned, b_offsets) = self.run_count_scan(ctx, engine);
        let prof = ProfilingStats::new();
        let owned_bytes = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_owned, counts_bytes, "derive:read_owned",
        ));
        let off_bytes = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_offsets, counts_bytes, "derive:read_offsets",
        ));
        let owned_counts: Vec<u32> = bytemuck::cast_slice(&owned_bytes).to_vec();
        let offsets: Vec<u32> = bytemuck::cast_slice(&off_bytes).to_vec();
        let last = n as usize - 1;
        let total = offsets[last] + owned_counts[last];
        DerivedFaceOffsets { owned_counts, offsets, total }
    }

    /// Run count → scan, keeping the per-cell offsets buffer ON DEVICE for the
    /// emit pass and returning the scalar `num_faces`. Only the n-u32
    /// owned/offsets addressing is read back (to compute the total); the
    /// O(num_faces) face geometry never round-trips.
    pub fn encode_offsets(
        &self,
        ctx: &GpuContext,
        cache: &StagingBufferCache,
        engine: &GpuVoronoiEngine,
    ) -> ScanOffsets {
        let n = engine.n_seeds();
        let counts_bytes = (n as u64) * 4;
        let (b_owned, b_offsets) = self.run_count_scan(ctx, engine);
        let prof = ProfilingStats::new();
        let owned_bytes = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_owned, counts_bytes, "derive:read_owned_total",
        ));
        let off_bytes = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_offsets, counts_bytes, "derive:read_offsets_total",
        ));
        let owned: Vec<u32> = bytemuck::cast_slice(&owned_bytes).to_vec();
        let offsets: Vec<u32> = bytemuck::cast_slice(&off_bytes).to_vec();
        let last = n as usize - 1;
        let total = offsets[last] + owned[last];
        ScanOffsets { b_offsets, total }
    }
}

fn count_shader() -> String {
    format!(
        r#"
struct Params {{ n: u32, k: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read>       nbr_ids: array<u32>;
@group(0) @binding(2) var<storage, read>       cell_nfaces: array<u32>;
@group(0) @binding(3) var<storage, read>       cell_status: array<u32>;
@group(0) @binding(4) var<storage, read_write> owned: array<u32>;

const NBR_NONE: u32 = 0xffffffffu;   // == u32::MAX (super::NBR_NONE)
const SUCCESS: u32  = 0u;            // == status::SUCCESS

@compute @workgroup_size({wg})
fn count_owned_faces(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(num_workgroups) nwg: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * nwg.x + wid.x) * {wg}u + lid.x;
    if (i >= P.n) {{ return; }}
    if (cell_status[i] != SUCCESS) {{ owned[i] = 0u; return; }}  // no geometry
    let nf = cell_nfaces[i];
    var c: u32 = 0u;
    for (var e: u32 = 0u; e < nf; e = e + 1u) {{
        let nbr = nbr_ids[i * P.k + e];
        if (nbr == NBR_NONE) {{
            c = c + 1u;           // boundary / bbox face — owned by this cell
        }} else if (i < nbr) {{
            c = c + 1u;           // interior face i<->j, owner = min(i,j)
        }}
    }}
    owned[i] = c;
}}
"#,
        wg = WORKGROUP_SIZE,
    )
}
