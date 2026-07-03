//! GPU Lloyd/CVT relaxation (design §7): `lloyd_update` kernel + a two-pass
//! max-displacement reduce, chained with full regens in ONE encoder with no
//! readback (`encode_lloyd_iterations`).
//!
//! ## Semantics (mirrors M0 `lloyd_relax` exactly)
//!
//! Each iteration moves every non-fixed seed toward the DENSITY-WEIGHTED
//! centroid of its current cell: the ring is fanned from the seed and each
//! triangle `(seed, v_e, v_{e+1})` contributes `w = area · ρ(centroid)` with
//! the graded-CVT weight `ρ(x) = h(x)^-exponent` evaluated at the triangle
//! centroid (centroid-point quadrature) — the exact M0 `weighted_centroid`
//! quadrature. The sizing field `h(x)` is an arbitrary CPU closure in M0; on
//! the GPU it is a CPU-sampled node grid evaluated by bilinear interpolation
//! (`set_lloyd_density`), whose approximation error is far below the O(h)
//! quadrature error Lloyd already tolerates (M0 lloyd.rs module docs).
//! Fixed seeds: `SeedKind::Boundary` seeds never move (the M0 v1 contract,
//! keyed off the kind table so it CANNOT be forgotten), and the
//! `SEED_FLAG_FIXED` bit in `b_seed_flags` pins additional seeds. Cells with
//! no usable geometry this iteration (coalesced `EMPTY_CELL`, overflow —
//! `nfaces < 3`) leave their seed unmoved, matching M0's `cell_ring_xy =
//! None` rule. `NEEDS_EXACT` cells move on their best-known f32 ring (the
//! chained loop never patches mid-flight; the CPU oracle uses its exact ring
//! — covered by the relaxation-drift tolerance).
//!
//! ## The stale-grid problem and the displacement-slack derate
//!
//! Chained regens reuse the CPU-built `SeedGrid` (and coalescing table) of
//! the UPLOADED seed positions — after `lloyd_update` moves seeds, bin
//! membership is stale. Candidate enumeration stays complete (every seed id
//! is still listed exactly once; planes are computed from live `b_seeds`),
//! but the ring-sweep security stop's lower bound `lb = dist(p, ring-r
//! boxes)` is only valid for the OLD positions: a seed stored in a ring-r
//! bin may now be up to `slack = Σ_iters max_i |Δx_i|` closer. The reduce's
//! second pass therefore accumulates each iteration's max absolute
//! displacement into `b_slack[0]`, and `voronoi_cell` derates its stop to
//! `(lb − slack)² · SECURITY_SCALE > 4R²` — no readback, provably
//! conservative (per-seed total displacement ≤ sum of per-iteration maxima).
//! `upload_case` resets the slack to zero (fresh grid), so the non-Lloyd
//! path evaluates `lb − 0.0` — bitwise identical to stage 3. The coalescing
//! table is also held fixed across chained iterations: a coalesced duplicate
//! outputs `EMPTY_CELL` (centroid 0 ⇒ parked seed) and its planes stay
//! skipped; `refresh_after_lloyd` re-derives grid + coalescing from the
//! relaxed positions before any diagram is consumed.
//!
//! ## Convergence (standalone path)
//!
//! `lloyd_update` writes per-seed `(|Δx|, |Δx|/h(x))` displacements; the
//! two-pass reduce (`max_disp_pass1/2`, the dot_product.wgsl workgroup-
//! scratch pattern — the existing reductions are sum-typed and bindgen-bound,
//! not reusable) leaves the componentwise max in `b_dmax`, read back as one
//! tiny transfer by `read_max_disp`. f32 `max` is exact (no rounding), so the
//! whole relaxation stays byte-stable run-to-run for a fixed device.

use nalgebra::Point2;

use crate::solver::gpu::buffers::{create_buffer, create_buffer_init};
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

use super::engine::GpuVoronoiEngine;
use super::{K_FACE_MAX, SEED_FLAG_FIXED, SEG_NONE};

const WORKGROUP_SIZE: u32 = 64;

/// Uniform parameter block — must match the WGSL `LParams` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LloydParams {
    pub n_seeds: u32,
    /// Reduce pass-1 workgroup count = ceil(n_seeds / 64).
    pub num_groups: u32,
    /// Sizing-grid CELL counts (node counts are +1).
    pub sz_nx: u32,
    pub sz_ny: u32,
    /// Reciprocal sizing-grid pitches (node i sits at `i / inv_dx`).
    pub inv_dx: f32,
    pub inv_dy: f32,
    /// ρ(x) = h(x)^-exponent (M0 `density_exponent`; 0 ⇒ uniform density).
    pub exponent: f32,
    /// Under-/over-relaxation on the centroid move (M0 `omega`).
    pub omega: f32,
}

/// Pipelines + buffers of the Lloyd stage, owned by the engine.
pub(super) struct LloydResources {
    update_pipeline: wgpu::ComputePipeline,
    reduce1_pipeline: wgpu::ComputePipeline,
    reduce2_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// Rebuilt when `b_sizing` is replaced (`set_lloyd_density`).
    bind_group: wgpu::BindGroup,
    pub(super) b_params: wgpu::Buffer,
    b_sizing: wgpu::Buffer,
    b_disp: wgpu::Buffer,
    b_partial: wgpu::Buffer,
    /// `[0]` = h-relative max displacement, `[1]` = absolute max (last
    /// `lloyd_update` in flight).
    b_dmax: wgpu::Buffer,
    pub(super) params: LloydParams,
}

impl LloydResources {
    /// `b_slack` is engine-owned (the regen kernel binds it read-only);
    /// the other engine buffers are the Lloyd kernel's inputs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &wgpu::Device,
        capacity: u32,
        b_seeds: &wgpu::Buffer,
        b_seed_kind: &wgpu::Buffer,
        b_seed_flags: &wgpu::Buffer,
        b_ring_vert: &wgpu::Buffer,
        b_cell_nfaces: &wgpu::Buffer,
        b_slack: &wgpu::Buffer,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let cap = capacity as u64;
        let b_params = create_buffer(
            device,
            "voronoi:lloyd_params",
            std::mem::size_of::<LloydParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        // Default sizing: uniform h = 1 on a single-cell grid; with the
        // default exponent 0 the density is uniform regardless.
        let b_sizing = create_buffer_init(
            device,
            "voronoi:lloyd_sizing",
            &[1.0f32; 4],
            U::STORAGE,
        );
        let b_disp = create_buffer(device, "voronoi:lloyd_disp", cap * 8, U::STORAGE);
        let n_partial = cap.div_ceil(WORKGROUP_SIZE as u64).max(1);
        let b_partial = create_buffer(
            device,
            "voronoi:lloyd_disp_partial",
            n_partial * 8,
            U::STORAGE,
        );
        let b_dmax = create_buffer(
            device,
            "voronoi:lloyd_disp_max",
            8,
            U::STORAGE | U::COPY_SRC,
        );

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
        // One layout + one bind group shared by all three entry points
        // (unused bindings are legal with an explicit layout).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voronoi:lloyd_bgl"),
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
                storage(1, false), // seeds (updated in place)
                storage(2, true),  // seed_kind
                storage(3, true),  // seed_flags
                storage(4, true),  // ring_vert
                storage(5, true),  // cell_nfaces
                storage(6, true),  // sizing
                storage(7, false), // disp
                storage(8, false), // partial
                storage(9, false), // dmax
                storage(10, false), // slack (accumulated by pass 2)
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voronoi:lloyd_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voronoi:lloyd_shader"),
            source: wgpu::ShaderSource::Wgsl(lloyd_shader().into()),
        });
        let mk = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let update_pipeline = mk("lloyd_update", "voronoi:lloyd_update_pipeline");
        let reduce1_pipeline = mk("max_disp_pass1", "voronoi:lloyd_reduce1_pipeline");
        let reduce2_pipeline = mk("max_disp_pass2", "voronoi:lloyd_reduce2_pipeline");

        let bind_group = Self::build_bind_group(
            device,
            &bgl,
            &b_params,
            b_seeds,
            b_seed_kind,
            b_seed_flags,
            b_ring_vert,
            b_cell_nfaces,
            &b_sizing,
            &b_disp,
            &b_partial,
            &b_dmax,
            b_slack,
        );

        Self {
            update_pipeline,
            reduce1_pipeline,
            reduce2_pipeline,
            bgl,
            bind_group,
            b_params,
            b_sizing,
            b_disp,
            b_partial,
            b_dmax,
            params: LloydParams {
                n_seeds: 0,
                num_groups: 1,
                sz_nx: 1,
                sz_ny: 1,
                inv_dx: 1.0,
                inv_dy: 1.0,
                exponent: 0.0,
                omega: 1.0,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        b_params: &wgpu::Buffer,
        b_seeds: &wgpu::Buffer,
        b_seed_kind: &wgpu::Buffer,
        b_seed_flags: &wgpu::Buffer,
        b_ring_vert: &wgpu::Buffer,
        b_cell_nfaces: &wgpu::Buffer,
        b_sizing: &wgpu::Buffer,
        b_disp: &wgpu::Buffer,
        b_partial: &wgpu::Buffer,
        b_dmax: &wgpu::Buffer,
        b_slack: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            }
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voronoi:lloyd_bg"),
            layout: bgl,
            entries: &[
                entry(0, b_params),
                entry(1, b_seeds),
                entry(2, b_seed_kind),
                entry(3, b_seed_flags),
                entry(4, b_ring_vert),
                entry(5, b_cell_nfaces),
                entry(6, b_sizing),
                entry(7, b_disp),
                entry(8, b_partial),
                entry(9, b_dmax),
                entry(10, b_slack),
            ],
        })
    }
}

impl GpuVoronoiEngine {
    /// Upload a density/sizing configuration for the Lloyd stage: `h(x)` is
    /// sampled at the `(nx+1)×(ny+1)` nodes of a uniform grid over the
    /// domain bbox (bilinearly interpolated by the kernel), `exponent` is
    /// the M0 `density_exponent` (ρ = h^-exponent; 4 = 2D energy-CVT
    /// grading, 0 = uniform density) and `omega` the M0 relaxation factor.
    /// The default (never called) is uniform density, ω = 1.
    pub fn set_lloyd_density(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        nx: u32,
        ny: u32,
        exponent: f32,
        omega: f32,
        sizing: &dyn Fn(Point2<f64>) -> f64,
    ) {
        assert!(nx > 0 && ny > 0, "sizing grid needs at least one cell");
        let mut nodes = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
        for j in 0..=ny {
            for i in 0..=nx {
                let p = Point2::new(
                    self.domain.x * i as f64 / nx as f64,
                    self.domain.y * j as f64 / ny as f64,
                );
                let h = sizing(p);
                assert!(h > 0.0 && h.is_finite(), "sizing must be positive");
                nodes.push(h as f32);
            }
        }
        self.lloyd.b_sizing = create_buffer_init(
            device,
            "voronoi:lloyd_sizing",
            &nodes,
            wgpu::BufferUsages::STORAGE,
        );
        self.lloyd.params.sz_nx = nx;
        self.lloyd.params.sz_ny = ny;
        self.lloyd.params.inv_dx = (nx as f64 / self.domain.x) as f32;
        self.lloyd.params.inv_dy = (ny as f64 / self.domain.y) as f32;
        self.lloyd.params.exponent = exponent;
        self.lloyd.params.omega = omega;
        queue.write_buffer(&self.lloyd.b_params, 0, bytemuck::bytes_of(&self.lloyd.params));
        self.lloyd.bind_group = LloydResources::build_bind_group(
            device,
            &self.lloyd.bgl,
            &self.lloyd.b_params,
            &self.b_seeds,
            &self.b_seed_kind,
            &self.b_seed_flags,
            &self.outputs.b_ring_vert,
            &self.outputs.b_cell_nfaces,
            &self.lloyd.b_sizing,
            &self.lloyd.b_disp,
            &self.lloyd.b_partial,
            &self.lloyd.b_dmax,
            &self.b_slack,
        );
    }

    /// Encode ONE Lloyd move: `lloyd_update` (seeds ← density-weighted cell
    /// centroids, fixed seeds masked) + the two-pass max-displacement reduce
    /// (leaves `(rel, abs)` maxima in `b_dmax`, accumulates the abs max into
    /// the grid-staleness slack). Requires cell outputs from a prior regen.
    pub fn encode_lloyd_update(&self, enc: &mut wgpu::CommandEncoder) {
        assert!(self.n_seeds > 0, "upload_case must run before Lloyd");
        let n_groups = self.n_seeds.div_ceil(WORKGROUP_SIZE).max(1);
        debug_assert_eq!(n_groups, self.lloyd.params.num_groups);
        let (ux, uy) = dispatch_2d(n_groups);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voronoi:lloyd"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.lloyd.bind_group, &[]);
        pass.set_pipeline(&self.lloyd.update_pipeline);
        pass.dispatch_workgroups(ux, uy, 1);
        pass.set_pipeline(&self.lloyd.reduce1_pipeline);
        pass.dispatch_workgroups(ux, uy, 1);
        pass.set_pipeline(&self.lloyd.reduce2_pipeline);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Chain `iters` full Lloyd iterations (`lloyd_update` + regen each)
    /// into one encoder — no readback (design §7). The first update consumes
    /// the cell outputs already resident (run a regen after `upload_case`
    /// before the first call).
    pub fn encode_lloyd_iterations(
        &self,
        enc: &mut wgpu::CommandEncoder,
        n_seeds: u32,
        iters: u32,
    ) {
        assert_eq!(
            n_seeds, self.n_seeds,
            "encode_lloyd_iterations n_seeds must match the uploaded seed count"
        );
        for _ in 0..iters {
            self.encode_lloyd_update(enc);
            self.encode_regen(enc, n_seeds);
        }
    }

    /// Own encoder + submit convenience over `encode_lloyd_iterations`.
    pub fn run_lloyd_iterations(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        iters: u32,
    ) -> wgpu::SubmissionIndex {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voronoi:lloyd_encoder"),
        });
        self.encode_lloyd_iterations(&mut enc, self.n_seeds, iters);
        queue.submit(Some(enc.finish()))
    }

    /// One tiny readback of the LAST `lloyd_update`'s max displacement:
    /// `(h_relative, absolute)` — the standalone convergence check (the M0
    /// measure is the h-relative one, `LloydConfig::tol_disp`).
    pub fn read_max_disp(&self, ctx: &GpuContext, cache: &StagingBufferCache) -> (f32, f32) {
        let prof = ProfilingStats::new();
        let bytes = pollster::block_on(read_buffer_cached(
            ctx,
            cache,
            &prof,
            &self.lloyd.b_dmax,
            8,
            "voronoi:rb_dmax",
        ));
        let v: &[f32] = bytemuck::cast_slice(&bytes);
        (v[0], v[1])
    }

    /// Read the live (possibly Lloyd-moved) seed positions back as
    /// interleaved f32 pairs.
    pub fn read_seeds(&self, ctx: &GpuContext, cache: &StagingBufferCache) -> Vec<f32> {
        let prof = ProfilingStats::new();
        let bytes = pollster::block_on(read_buffer_cached(
            ctx,
            cache,
            &prof,
            &self.b_seeds,
            self.n_seeds as u64 * 8,
            "voronoi:rb_seeds",
        ));
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// End a chained-Lloyd episode: read the relaxed seeds back and re-run
    /// `upload_case` on them (same flags/kinds/boundary), rebuilding the
    /// seed grid, the coalescing table, the CPU seed mirror (`resolve_flagged`
    /// / `read_diagram` inputs) and resetting the grid-staleness slack to
    /// zero. Callers then run a fresh regen + `resolve_flagged` as usual.
    /// Returns the relaxed interleaved-f32 seeds.
    pub fn refresh_after_lloyd(
        &mut self,
        ctx: &GpuContext,
        cache: &StagingBufferCache,
    ) -> Vec<f32> {
        let seeds = self.read_seeds(ctx, cache);
        let flags = self.flags.clone();
        let kinds = self.kinds.clone();
        let boundary = self.boundary.clone();
        self.upload_case(&ctx.device, &ctx.queue, &seeds, &flags, &kinds, &boundary);
        seeds
    }
}

/// Build the Lloyd + reduce shader source (constants injected from Rust).
fn lloyd_shader() -> String {
    format!(
        "const K_FACE_MAX: u32 = {k}u;\n\
         const SEG_NONE: u32 = {seg_none}u;\n\
         const FLAG_FIXED: u32 = {fixed}u;\n{BODY}",
        k = K_FACE_MAX,
        seg_none = SEG_NONE,
        fixed = SEED_FLAG_FIXED,
    )
}

const BODY: &str = r#"
struct LParams {
    n_seeds: u32,
    num_groups: u32,
    sz_nx: u32,
    sz_ny: u32,
    inv_dx: f32,
    inv_dy: f32,
    exponent: f32,
    omega: f32,
};

@group(0) @binding(0) var<uniform> lp: LParams;
@group(0) @binding(1) var<storage, read_write> seeds: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> seed_kind: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> seed_flags: array<u32>;
@group(0) @binding(4) var<storage, read> ring_vert: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> cell_nfaces: array<u32>;
// Sizing-field node samples, (sz_nx+1) x (sz_ny+1) row-major.
@group(0) @binding(6) var<storage, read> sizing: array<f32>;
// Per-seed (absolute, h-relative) displacement of the last update.
@group(0) @binding(7) var<storage, read_write> disp: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> partial_max: array<vec2<f32>>;
// [0] = h-relative max, [1] = absolute max.
@group(0) @binding(9) var<storage, read_write> dmax: array<f32>;
// [0] = accumulated grid-staleness slack (absolute distance units), read by
// voronoi_cell's security stop.
@group(0) @binding(10) var<storage, read_write> slack: array<f32>;

// Bilinear interpolation of the sizing field at an absolute position.
fn sizing_at(x: vec2<f32>) -> f32 {
    let fx = clamp(x.x * lp.inv_dx, 0.0, f32(lp.sz_nx));
    let fy = clamp(x.y * lp.inv_dy, 0.0, f32(lp.sz_ny));
    let ix = min(u32(fx), lp.sz_nx - 1u);
    let iy = min(u32(fy), lp.sz_ny - 1u);
    let tx = fx - f32(ix);
    let ty = fy - f32(iy);
    let w = lp.sz_nx + 1u;
    let h00 = sizing[iy * w + ix];
    let h10 = sizing[iy * w + ix + 1u];
    let h01 = sizing[(iy + 1u) * w + ix];
    let h11 = sizing[(iy + 1u) * w + ix + 1u];
    return mix(mix(h00, h10, tx), mix(h01, h11, tx), ty);
}

// seeds[i] <- p + omega * (c_w - p), c_w = the density-weighted centroid of
// the cell ring (fan quadrature, rho = h^-exponent at triangle centroids —
// the exact M0 weighted_centroid rule, in seed-relative coordinates: the
// fan triangle (0, a, b) has centroid (a+b)/3 relative to the seed, and
// sum(w*(p+t_rel))/sum(w) - p == sum(w*t_rel)/sum(w)). Fixed seeds
// (SeedKind::Boundary via seg_prev != SEG_NONE, or the FLAG_FIXED bit) and
// cells without usable geometry (nfaces < 3: coalesced EMPTY, overflow)
// stay put with zero displacement, matching M0.
@compute @workgroup_size(64)
fn lloyd_update(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= lp.n_seeds) {
        return;
    }
    let p = seeds[i];
    let fixed = ((seed_flags[i] & FLAG_FIXED) != 0u) || (seed_kind[i].x != SEG_NONE);
    let nf = min(cell_nfaces[i], K_FACE_MAX);
    var d = vec2<f32>(0.0, 0.0);
    if (!fixed && nf >= 3u) {
        var w_sum = 0.0;
        var cx = 0.0;
        var cy = 0.0;
        for (var e = 0u; e < nf; e = e + 1u) {
            var w2 = e + 1u;
            if (w2 == nf) {
                w2 = 0u;
            }
            let a = ring_vert[i * K_FACE_MAX + e];
            let b = ring_vert[i * K_FACE_MAX + w2];
            let area = 0.5 * (a.x * b.y - a.y * b.x);
            let t = (a + b) / 3.0;
            let h = sizing_at(p + t);
            let wgt = area * pow(h, -lp.exponent);
            w_sum = w_sum + wgt;
            cx = cx + wgt * t.x;
            cy = cy + wgt * t.y;
        }
        if (w_sum > 0.0) {
            d = lp.omega * vec2<f32>(cx, cy) / w_sum;
        }
    }
    seeds[i] = p + d;
    let ad = length(d);
    disp[i] = vec2<f32>(ad, ad / sizing_at(p));
}

var<workgroup> scratch: array<vec2<f32>, 64>;

// Two-pass componentwise max over disp (dot_product.wgsl scratch pattern;
// f32 max is exact, so the reduction is order-insensitive and byte-stable).
@compute @workgroup_size(64)
fn max_disp_pass1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    var v = vec2<f32>(0.0, 0.0);
    if (idx < lp.n_seeds) {
        v = disp[idx];
    }
    scratch[lid.x] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        let g = wid.y * nwg.x + wid.x;
        if (g < lp.num_groups) {
            partial_max[g] = scratch[0u];
        }
    }
}

// Single-workgroup grid-stride finish; also accumulates the absolute max
// into the grid-staleness slack (module docs).
@compute @workgroup_size(64)
fn max_disp_pass2(@builtin(local_invocation_id) lid: vec3<u32>) {
    var v = vec2<f32>(0.0, 0.0);
    for (var k = lid.x; k < lp.num_groups; k = k + 64u) {
        v = max(v, partial_max[k]);
    }
    scratch[lid.x] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        dmax[0] = scratch[0u].y;
        dmax[1] = scratch[0u].x;
        slack[0] = slack[0] + scratch[0u].x;
    }
}
"#;
