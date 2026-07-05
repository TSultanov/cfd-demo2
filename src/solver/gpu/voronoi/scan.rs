//! GPU exclusive prefix-sum (scan) primitive: turns per-cell face **counts**
//! into per-cell face **offsets** (and the total `num_faces`) for GPU-resident
//! regen (`derive_faces`) without a CPU round-trip.
//!
//! ## Algorithm — standard two-level exclusive scan
//!
//! `ELEMS_PER_BLOCK = 1024` (a `256`-thread workgroup, `4` elements/thread):
//!
//!  1. `scan_blocks` — each block does a per-block **exclusive** scan of its
//!     1024 elements into `output` and writes its block total to `block_sums[b]`.
//!     Each thread serial-scans its 4 contiguous elements, the per-thread totals
//!     are inclusive-scanned across the 256 threads with Hillis–Steele in a
//!     `array<u32, 256>` workgroup buffer, and each thread then offsets its 4
//!     local exclusive prefixes by its (exclusive) thread offset.
//!  2. `scan_block_sums` — a **single** workgroup exclusive-scans `block_sums`
//!     in place (≤ 1024 blocks ⇒ ≤ 1024 elements, one level suffices; that caps
//!     a single `scan()` at 1024·1024 = 2²⁰ elements, asserted).
//!  3. `add_offsets` — one thread per element adds `block_sums[i / 1024]`.
//!
//! Fully deterministic (disjoint per-element writes; `u32` adds are exact) and
//! byte-stable run-to-run on a fixed device. Reused by `derive_faces`.

use crate::solver::gpu::buffers::create_buffer;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

/// Elements handled by one workgroup (256 threads × 4 elements).
pub const ELEMS_PER_BLOCK: u32 = 1024;
const WORKGROUP_SIZE: u32 = 256;
/// Two levels of 1024 ⇒ the largest element count a single `scan` supports.
pub const MAX_SCAN_ELEMS: u32 = ELEMS_PER_BLOCK * ELEMS_PER_BLOCK;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScanParams {
    n: u32,
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

/// The scan pipelines + a shared bind-group layout. Buffers are caller-owned so
/// the scan can be encoded into an existing regen encoder (no readback).
pub struct GpuScan {
    scan_blocks: wgpu::ComputePipeline,
    scan_block_sums: wgpu::ComputePipeline,
    add_offsets: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuScan {
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
            label: Some("scan:bgl"),
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
                storage(1, true),  // input (read-only)
                storage(2, false), // output (read-write)
                storage(3, false), // block_sums (read-write)
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan:pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan:shader"),
            source: wgpu::ShaderSource::Wgsl(scan_shader().into()),
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            scan_blocks: make("scan_blocks", "scan:blocks"),
            scan_block_sums: make("scan_block_sums", "scan:block_sums"),
            add_offsets: make("add_offsets", "scan:add_offsets"),
            bgl,
        }
    }

    /// Number of block-sum slots a scan of `n` elements needs.
    pub fn num_blocks(n: u32) -> u32 {
        n.div_ceil(ELEMS_PER_BLOCK).max(1)
    }

    /// Encode an exclusive scan of `input[0..n]` into `output`, using
    /// `block_sums` (≥ [`Self::num_blocks`] `u32`s) and `params` (a `ScanParams`
    /// uniform, written here) as scratch. All buffers are caller-owned; no
    /// readback. The exclusive total lands as `output` sums; the grand total is
    /// `output[n-1] + input[n-1]` (or read `block_sums` — its last entry plus the
    /// last block's remaining sum). Callers that need the total explicitly should
    /// use [`Self::scan_to_vec`] or size the output one longer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        block_sums: &wgpu::Buffer,
        params: &wgpu::Buffer,
        n: u32,
    ) {
        assert!(
            n <= MAX_SCAN_ELEMS,
            "scan supports up to {MAX_SCAN_ELEMS} elements (two levels of 1024); got {n}"
        );
        let num_blocks = Self::num_blocks(n);
        queue.write_buffer(
            params,
            0,
            bytemuck::bytes_of(&ScanParams { n, num_blocks, _pad0: 0, _pad1: 0 }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan:bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: block_sums.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan:pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            // 1. per-block exclusive scan + block totals.
            pass.set_pipeline(&self.scan_blocks);
            let (bx, by) = dispatch_2d(num_blocks);
            pass.dispatch_workgroups(bx, by, 1);
            // 2. exclusive scan of the block totals (single workgroup).
            pass.set_pipeline(&self.scan_block_sums);
            pass.dispatch_workgroups(1, 1, 1);
            // 3. add each block's offset to its elements (one thread/element).
            pass.set_pipeline(&self.add_offsets);
            let (ax, ay) = dispatch_2d(n.div_ceil(WORKGROUP_SIZE).max(1));
            pass.dispatch_workgroups(ax, ay, 1);
        }
    }

    /// Convenience full round-trip for tests / one-off host calls: upload
    /// `data`, exclusive-scan on the GPU, return `(exclusive_scan, total)`.
    pub fn scan_to_vec(
        ctx: &GpuContext,
        cache: &StagingBufferCache,
        data: &[u32],
    ) -> (Vec<u32>, u32) {
        use wgpu::BufferUsages as U;
        if data.is_empty() {
            return (Vec::new(), 0);
        }
        let n = data.len() as u32;
        let scan = Self::new(&ctx.device);
        let cap = (n.max(1)) as u64 * 4;
        let input = crate::solver::gpu::buffers::create_buffer_init(
            &ctx.device,
            "scan:test_input",
            data,
            U::STORAGE | U::COPY_DST,
        );
        let output =
            create_buffer(&ctx.device, "scan:test_output", cap, U::STORAGE | U::COPY_SRC);
        let block_sums = create_buffer(
            &ctx.device,
            "scan:test_block_sums",
            (Self::num_blocks(n) as u64) * 4,
            U::STORAGE | U::COPY_SRC,
        );
        let params = create_buffer(
            &ctx.device,
            "scan:test_params",
            std::mem::size_of::<ScanParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scan:test") });
        scan.encode(
            &ctx.device,
            &ctx.queue,
            &mut encoder,
            &input,
            &output,
            &block_sums,
            &params,
            n,
        );
        ctx.queue.submit(Some(encoder.finish()));
        let prof = ProfilingStats::new();
        let bytes = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &output, cap, "scan:test_readback",
        ));
        let scan_out: Vec<u32> = bytemuck::cast_slice(&bytes)[..data.len()].to_vec();
        let total = if data.is_empty() {
            0
        } else {
            scan_out[data.len() - 1] + data[data.len() - 1]
        };
        (scan_out, total)
    }
}

fn scan_shader() -> String {
    format!(
        r#"
struct Params {{ n: u32, num_blocks: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read>       in_data: array<u32>;
@group(0) @binding(2) var<storage, read_write>  out_data: array<u32>;
@group(0) @binding(3) var<storage, read_write>  block_sums: array<u32>;

const WG: u32 = {wg}u;
const EPB: u32 = {epb}u;   // elements per block (WG * 4)

var<workgroup> temp: array<u32, {wg}u>;

// Inclusive Hillis-Steele scan of temp[0..WG); returns temp[t] afterwards.
fn scan_temp(t: u32) {{
    var off: u32 = 1u;
    loop {{
        if (off >= WG) {{ break; }}
        var add: u32 = 0u;
        if (t >= off) {{ add = temp[t - off]; }}
        workgroupBarrier();
        temp[t] = temp[t] + add;
        workgroupBarrier();
        off = off * 2u;
    }}
}}

@compute @workgroup_size({wg})
fn scan_blocks(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {{
    let t = lid.x;
    let b = wid.y * nwg.x + wid.x;       // linear block id (2D dispatch safe)
    let base = b * EPB + t * 4u;
    var vals: array<u32, 4>;
    var sum: u32 = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {{
        let idx = base + j;
        var v: u32 = 0u;
        if (idx < P.n) {{ v = in_data[idx]; }}
        vals[j] = v;
        sum = sum + v;
    }}
    temp[t] = sum;
    workgroupBarrier();
    scan_temp(t);
    let inclusive = temp[t];
    let thread_offset = inclusive - sum;  // exclusive prefix of thread totals
    var run: u32 = thread_offset;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {{
        let idx = base + j;
        if (idx < P.n) {{ out_data[idx] = run; }}
        run = run + vals[j];
    }}
    if (t == WG - 1u) {{ block_sums[b] = inclusive; }}  // block total
}}

@compute @workgroup_size({wg})
fn scan_block_sums(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let t = lid.x;
    let base = t * 4u;
    var vals: array<u32, 4>;
    var sum: u32 = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {{
        let idx = base + j;
        var v: u32 = 0u;
        if (idx < P.num_blocks) {{ v = block_sums[idx]; }}
        vals[j] = v;
        sum = sum + v;
    }}
    temp[t] = sum;
    workgroupBarrier();
    scan_temp(t);
    let thread_offset = temp[t] - sum;
    var run: u32 = thread_offset;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {{
        let idx = base + j;
        if (idx < P.num_blocks) {{ block_sums[idx] = run; }}  // exclusive, in place
        run = run + vals[j];
    }}
}}

@compute @workgroup_size({wg})
fn add_offsets(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {{
    let group = wid.y * nwg.x + wid.x;
    let i = group * WG + lid.x;
    if (i >= P.n) {{ return; }}
    out_data[i] = out_data[i] + block_sums[i / EPB];
}}
"#,
        wg = WORKGROUP_SIZE,
        epb = ELEMS_PER_BLOCK,
    )
}
