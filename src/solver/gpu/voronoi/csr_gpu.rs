//! GPU-resident CSR bridge (Phase C stage 2) — builds the cell→face adjacency
//! (`cell_faces` / `cell_face_offsets`) and the GPU sorted-adjacency scalar CSR
//! (`col_indices` / `row_offsets` / `diagonal_indices` /
//! `cell_face_matrix_indices`) entirely on device, reproducing
//! [`crate::solver::mesh::csr::build_sorted_scalar_csr`] without a CPU
//! `assemble_mesh` round-trip.
//!
//! Two passes over the M1 engine's cell-major clip slots + the emit's per-cell
//! owned-face offsets:
//!
//!  1. `count` — per cell, the total face count (`= nfaces`) and the sorted-CSR
//!     row size (`interior_faces + 1` for the diagonal). Two exclusive scans turn
//!     these into `cell_face_offsets` and `row_offsets`.
//!  2. `build` — per cell, deterministically resolves each slot's GLOBAL face
//!     index (owned face → `owned_offset[i] + rank`; a non-owned face to `j<i` →
//!     `owned_offset[j] + j`'s owned-rank of the `(j,i)` face, an O(K²) scan of
//!     `j`'s slots), collects + insertion-sorts the neighbour ids (+ self) into
//!     the sorted row, and binary-searches each face's column to fill
//!     `cell_face_matrix_indices`. Fully deterministic (fixed slot order, no
//!     atomics), so the byte-reproducibility the moving-mesh design relies on
//!     holds.
//!
//! Assumes a reciprocal, all-SUCCESS diagram (post `resolve_flagged`).

use crate::solver::gpu::buffers::create_buffer;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

use super::derive::DeriveFaces;
use super::engine::GpuVoronoiEngine;
use super::scan::GpuScan;
use super::K_FACE_MAX;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CsrParams {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-built scalar CSR + cell→face adjacency, read back for parity/inspection.
/// In the live loop these live only as device buffers.
#[derive(Clone, Debug)]
pub struct GpuCsrArrays {
    pub num_faces: u32,
    /// Row start offsets into `cell_faces`, length `num_cells + 1`.
    pub cell_face_offsets: Vec<u32>,
    /// Global face index per (cell, slot), length `cell_face_offsets[n]`.
    pub cell_faces: Vec<u32>,
    /// Sorted-CSR row offsets, length `num_cells + 1`.
    pub row_offsets: Vec<u32>,
    /// Sorted column indices (diagonal at its sorted position), length `nnz`.
    pub col_indices: Vec<u32>,
    /// CSR rank of each cell's diagonal, length `num_cells`.
    pub diagonal_indices: Vec<u32>,
    /// CSR rank for every (cell, slot) pair, indexed like `cell_faces`.
    pub cell_face_matrix_indices: Vec<u32>,
}

pub struct GpuCsr {
    count_pipeline: wgpu::ComputePipeline,
    count_bgl: wgpu::BindGroupLayout,
    build_pipeline: wgpu::ComputePipeline,
    build_bgl: wgpu::BindGroupLayout,
    scan: GpuScan,
    derive: DeriveFaces,
}

impl GpuCsr {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("csr:count_bgl"),
            entries: &[
                uniform(0),
                storage(1, true),  // nbr_ids
                storage(2, true),  // cell_nfaces
                storage(3, true),  // status
                storage(4, false), // out: cell_face_counts
                storage(5, false), // out: adj_counts
            ],
        });
        let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("csr:build_bgl"),
            entries: &[
                uniform(0),
                storage(1, true),   // nbr_ids
                storage(2, true),   // cell_nfaces
                storage(3, true),   // status
                storage(4, true),   // owned_offsets (emit base)
                storage(5, true),   // cell_face_offsets
                storage(6, true),   // row_offsets
                storage(7, false),  // out: cell_faces
                storage(8, false),  // out: col_indices
                storage(9, false),  // out: diagonal_indices
                storage(10, false), // out: cell_face_matrix_indices
            ],
        });
        let mk = |bgl: &wgpu::BindGroupLayout, src: String, entry: &str, label: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let count_pipeline = mk(&count_bgl, count_shader(), "count_csr", "csr:count");
        let build_pipeline = mk(&build_bgl, build_shader(), "build_csr", "csr:build");
        Self {
            count_pipeline,
            count_bgl,
            build_pipeline,
            build_bgl,
            scan: GpuScan::new(device),
            derive: DeriveFaces::new(device),
        }
    }

    pub fn build_csr(
        &self,
        ctx: &GpuContext,
        cache: &StagingBufferCache,
        engine: &GpuVoronoiEngine,
    ) -> GpuCsrArrays {
        use wgpu::BufferUsages as U;
        let n = engine.n_seeds();
        assert!(n > 0, "build_csr called before a regen");
        let scan_off = self.derive.encode_offsets(ctx, cache, engine);
        let num_faces = scan_off.total;

        let params = CsrParams { n, k: K_FACE_MAX as u32, _pad0: 0, _pad1: 0 };
        let b_params = create_buffer(
            &ctx.device,
            "csr:params",
            std::mem::size_of::<CsrParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        ctx.queue.write_buffer(&b_params, 0, bytemuck::bytes_of(&params));

        let nb = (n as u64) * 4;
        let mk = |label: &str, bytes: u64| {
            create_buffer(&ctx.device, label, bytes, U::STORAGE | U::COPY_SRC)
        };
        let b_face_counts = mk("csr:face_counts", nb);
        let b_adj_counts = mk("csr:adj_counts", nb);

        // --- pass 1: count ---
        let count_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("csr:count_bg"),
            layout: &self.count_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: engine.outputs.b_nbr_ids.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: engine.outputs.b_cell_nfaces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: engine.outputs.b_status.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_face_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_adj_counts.as_entire_binding() },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("csr:count_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("csr:count_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &count_bg, &[]);
            let (x, y) = dispatch_2d(n.div_ceil(WORKGROUP_SIZE).max(1));
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        // --- scans: cell_face_offsets, row_offsets ---
        let b_face_offsets = mk("csr:face_offsets", nb);
        let b_row_offsets = mk("csr:row_offsets", nb);
        self.scan_into(ctx, &b_face_counts, &b_face_offsets, n);
        self.scan_into(ctx, &b_adj_counts, &b_row_offsets, n);

        // totals from the last (offset + count) of each scan.
        let prof = ProfilingStats::new();
        let face_counts: Vec<u32> = bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_face_counts, nb, "csr:rb_face_counts",
        )))
        .to_vec();
        let face_offsets_v: Vec<u32> = bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_face_offsets, nb, "csr:rb_face_offsets",
        )))
        .to_vec();
        let adj_counts: Vec<u32> = bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_adj_counts, nb, "csr:rb_adj_counts",
        )))
        .to_vec();
        let row_offsets_v: Vec<u32> = bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_row_offsets, nb, "csr:rb_row_offsets",
        )))
        .to_vec();
        let last = n as usize - 1;
        let total_all = face_offsets_v[last] + face_counts[last];
        let nnz = row_offsets_v[last] + adj_counts[last];

        // --- pass 2: build ---
        let b_cell_faces = mk("csr:cell_faces", (total_all.max(1) as u64) * 4);
        let b_col = mk("csr:col_indices", (nnz.max(1) as u64) * 4);
        let b_diag = mk("csr:diagonal_indices", nb);
        let b_cfmi = mk("csr:cfmi", (total_all.max(1) as u64) * 4);
        let build_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("csr:build_bg"),
            layout: &self.build_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: engine.outputs.b_nbr_ids.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: engine.outputs.b_cell_nfaces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: engine.outputs.b_status.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scan_off.b_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_face_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b_row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: b_cell_faces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: b_col.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: b_diag.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: b_cfmi.as_entire_binding() },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("csr:build_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("csr:build_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &build_bg, &[]);
            let (x, y) = dispatch_2d(n.div_ceil(WORKGROUP_SIZE).max(1));
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        // readback (parity path). Build the n+1 offset arrays on the host.
        let mut cell_face_offsets = face_offsets_v.clone();
        cell_face_offsets.push(total_all);
        let mut row_offsets = row_offsets_v.clone();
        row_offsets.push(nnz);
        let cell_faces: Vec<u32> = bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_cell_faces, (total_all.max(1) as u64) * 4, "csr:rb_cell_faces",
        )))
        .to_vec();
        let col_indices: Vec<u32> = bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_col, (nnz.max(1) as u64) * 4, "csr:rb_col",
        )))
        .to_vec();
        let diagonal_indices: Vec<u32> = bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_diag, nb, "csr:rb_diag",
        )))
        .to_vec();
        let cell_face_matrix_indices: Vec<u32> =
            bytemuck::cast_slice(&pollster::block_on(read_buffer_cached(
                ctx, cache, &prof, &b_cfmi, (total_all.max(1) as u64) * 4, "csr:rb_cfmi",
            )))
            .to_vec();

        GpuCsrArrays {
            num_faces,
            cell_face_offsets,
            cell_faces: cell_faces[..total_all as usize].to_vec(),
            row_offsets,
            col_indices: col_indices[..nnz as usize].to_vec(),
            diagonal_indices,
            cell_face_matrix_indices: cell_face_matrix_indices[..total_all as usize].to_vec(),
        }
    }

    fn scan_into(&self, ctx: &GpuContext, input: &wgpu::Buffer, output: &wgpu::Buffer, n: u32) {
        use wgpu::BufferUsages as U;
        let b_block_sums = create_buffer(
            &ctx.device,
            "csr:block_sums",
            (GpuScan::num_blocks(n) as u64) * 4,
            U::STORAGE | U::COPY_SRC,
        );
        let b_scan_params =
            create_buffer(&ctx.device, "csr:scan_params", 16, U::UNIFORM | U::COPY_DST);
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("csr:scan_enc") });
        self.scan
            .encode(&ctx.device, &ctx.queue, &mut enc, input, output, &b_block_sums, &b_scan_params, n);
        ctx.queue.submit(Some(enc.finish()));
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
@group(0) @binding(4) var<storage, read_write> face_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> adj_counts: array<u32>;

const NBR_NONE: u32 = 0xffffffffu;
const SUCCESS: u32  = 0u;

@compute @workgroup_size({wg})
fn count_csr(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * nwg.x + wid.x) * {wg}u + lid.x;
    if (i >= P.n) {{ return; }}
    var nf: u32 = 0u;
    if (cell_status[i] == SUCCESS) {{ nf = cell_nfaces[i]; }}
    var interior: u32 = 0u;
    for (var e: u32 = 0u; e < nf; e = e + 1u) {{
        if (nbr_ids[i * P.k + e] != NBR_NONE) {{ interior = interior + 1u; }}
    }}
    face_counts[i] = nf;
    adj_counts[i] = interior + 1u;   // + diagonal
}}
"#,
        wg = WORKGROUP_SIZE,
    )
}

fn build_shader() -> String {
    // Local neighbour list per cell: interior neighbours + the diagonal.
    let cap = K_FACE_MAX + 1;
    format!(
        r#"
struct Params {{ n: u32, k: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read>       nbr_ids: array<u32>;
@group(0) @binding(2) var<storage, read>       cell_nfaces: array<u32>;
@group(0) @binding(3) var<storage, read>       cell_status: array<u32>;
@group(0) @binding(4) var<storage, read>       owned_offsets: array<u32>;
@group(0) @binding(5) var<storage, read>       cell_face_offsets: array<u32>;
@group(0) @binding(6) var<storage, read>       row_offsets: array<u32>;
@group(0) @binding(7)  var<storage, read_write> cell_faces: array<u32>;
@group(0) @binding(8)  var<storage, read_write> col_indices: array<u32>;
@group(0) @binding(9)  var<storage, read_write> diagonal_indices: array<u32>;
@group(0) @binding(10) var<storage, read_write> cell_face_matrix_indices: array<u32>;

const NBR_NONE: u32 = 0xffffffffu;
const SUCCESS: u32  = 0u;
const CAP: u32 = {cap}u;

@compute @workgroup_size({wg})
fn build_csr(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * nwg.x + wid.x) * {wg}u + lid.x;
    if (i >= P.n) {{ return; }}
    let row_base = row_offsets[i];
    var nf: u32 = 0u;
    if (cell_status[i] == SUCCESS) {{ nf = cell_nfaces[i]; }}

    // Collect interior neighbours + self, then insertion-sort ascending.
    var nbrs: array<u32, CAP>;
    var cnt: u32 = 0u;
    for (var e: u32 = 0u; e < nf; e = e + 1u) {{
        let nb = nbr_ids[i * P.k + e];
        if (nb != NBR_NONE) {{ nbrs[cnt] = nb; cnt = cnt + 1u; }}
    }}
    nbrs[cnt] = i; cnt = cnt + 1u;   // diagonal
    for (var a: u32 = 1u; a < cnt; a = a + 1u) {{
        let key = nbrs[a];
        var b: i32 = i32(a) - 1;
        loop {{
            if (b < 0) {{ break; }}
            if (nbrs[u32(b)] <= key) {{ break; }}
            nbrs[u32(b) + 1u] = nbrs[u32(b)];
            b = b - 1;
        }}
        nbrs[u32(b + 1)] = key;
    }}
    // Write the sorted row + record the diagonal rank.
    for (var r: u32 = 0u; r < cnt; r = r + 1u) {{
        col_indices[row_base + r] = nbrs[r];
        if (nbrs[r] == i) {{ diagonal_indices[i] = row_base + r; }}
    }}

    // Gather cell_faces (global face indices) + matrix indices.
    let face_base = cell_face_offsets[i];
    var owned_rank_i: u32 = 0u;
    for (var e: u32 = 0u; e < nf; e = e + 1u) {{
        let nb = nbr_ids[i * P.k + e];
        var global: u32;
        if (nb == NBR_NONE || i < nb) {{
            // owned by i
            global = owned_offsets[i] + owned_rank_i;
            owned_rank_i = owned_rank_i + 1u;
        }} else {{
            // owned by j = nb < i: find j's owned-rank of the (j,i) face.
            let j = nb;
            let nfj = cell_nfaces[j];
            var rank_j: u32 = 0u;
            for (var e2: u32 = 0u; e2 < nfj; e2 = e2 + 1u) {{
                let nb2 = nbr_ids[j * P.k + e2];
                if (nb2 == i) {{ break; }}
                if (nb2 == NBR_NONE || j < nb2) {{ rank_j = rank_j + 1u; }}
            }}
            global = owned_offsets[j] + rank_j;
        }}
        cell_faces[face_base + e] = global;

        // Column of this face (neighbour, or self for a boundary face) →
        // binary-search its sorted rank.
        var col: u32 = i;
        if (nb != NBR_NONE) {{ col = nb; }}
        var lo: u32 = 0u;
        var hi: u32 = cnt;
        loop {{
            if (lo >= hi) {{ break; }}
            let mid = (lo + hi) / 2u;
            if (nbrs[mid] < col) {{ lo = mid + 1u; }} else {{ hi = mid; }}
        }}
        cell_face_matrix_indices[face_base + e] = row_base + lo;
    }}
}}
"#,
        wg = WORKGROUP_SIZE,
        cap = cap,
    )
}
