//! GPU-resident ALE swept-flux GEOMETRY (Phase C stage D1+D2) — the per-face
//! swept-quad areas the moving-mesh GCL closure consumes, computed entirely on
//! device from the OLD and NEW seed sets.
//!
//! ## Why this can skip the CPU union-find merge
//!
//! The CPU path (`solver::mesh::ale`) computes swept quads from MERGED vertex
//! positions (`face_v1/face_v2` into `vx/vy`), with the old positions supplied
//! by `align_old_vertices_by_seed_set` (a vertex ≡ its incident-seed SET). The
//! key identity that unlocks the on-device port: a Voronoi vertex incident to
//! cells `{i,j,k}` **is** the circumcenter of the seed triple `{i,j,k}` — a pure
//! function of the seed positions — so
//!
//!  - its NEW position = circumcenter at the new seeds,
//!  - its OLD position = circumcenter at the old seeds,
//!
//! with NO vertex-id correspondence to track (the triple IS the id). This is the
//! same canonical re-evaluation `meshless::assemble::canonical_vertex` performs;
//! reproducing it here makes shared endpoints consistent between the two cells
//! incident to a face without any global merge. For a boundary vertex the third
//! plane is a domain-box side (`bisector ∩ box`), a domain corner (`box ∩ box`),
//! or a polyline BOUNDARY-segment line (`bisector ∩ segment`, and at polyline
//! corners `segment ∩ segment`) — the segment tables at BOTH seed sets are
//! bound (the engine's t^{n+1} table + a caller-supplied t^n table, so a moving
//! boundary sweeps its wall faces correctly). Adjacent ring edges are never
//! EXACTLY collinear (a redundant clip plane creates no edge), but f32
//! rounding of nearly-collinear segment lines can leave an ill-conditioned
//! bracket — `vertex_at` flags those to the CPU fallback via a RELATIVE
//! near-parallel guard instead of fabricating a far vertex.
//!
//! ## Precision
//!
//! All four endpoints of a face's swept quad are evaluated RELATIVE to a common
//! per-face origin (the owner's new seed). The swept-area cross products then
//! subtract O(h) relative coordinates rather than O(1) absolutes, so the f32
//! cancellation that would otherwise dominate `p_new − p_old` (a displacement
//! ~10⁻² formed from positions ~1) never happens.
//!
//! ## Orientation & ownership
//!
//! One thread per OWNER cell (`owner = min(i,j)`; boundary faces owned by their
//! cell) writes each owned face's swept area into the face-major slot (via the
//! emit's owned-offset addressing), oriented onto the stored owner-outward
//! normal exactly as `swept_area_along_normal` does. The non-owner cell reads
//! the same face with the opposite sign in the per-cell GCL sum, so the flux is
//! single-valued by construction.

use crate::meshgen::meshless::BoundarySpec;
use crate::solver::gpu::buffers::create_buffer;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

use super::derive::DeriveFaces;
use super::engine::{boundary_spec_f32, GpuVoronoiEngine};
use super::K_FACE_MAX;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SweptParams {
    n: u32,
    k: u32,
    det_eps: f32,
    _pad0: u32,
    domain_x: f32,
    domain_y: f32,
    _pad1: u32,
    _pad2: u32,
}

/// Per-face swept-quad areas (owner-signed, face-major) + the per-cell CANONICAL
/// polygon areas at both seed sets, read back for parity/inspection. In the live
/// loop these live only as device buffers.
///
/// The swept areas telescope to the canonical area CHANGE by construction (both
/// come from the same canonical vertices), so the moving path uses
/// `canon_area_new` for `cell_vols` (self-consistent ⇒ the GCL closes to f32),
/// NOT the engine's clipped `b_cell_area` (whose streaming-clip drift is what the
/// canonical vertices avoid).
#[derive(Clone, Debug)]
pub struct GpuSweptAreas {
    pub num_faces: u32,
    /// Signed swept area per face (owner-outward normal convention), same
    /// quantity as the CPU `swept[f]` before the `/dt` cast + closure.
    pub swept: Vec<f32>,
    /// Per-face flag: `1` if the swept quad could not be built on device (an
    /// edge tag that is neither a bisector, a box side, nor a polyline
    /// segment — defensive; no such tag is emitted today), so the caller must
    /// fall back to the CPU path for this regen. Written per OWNED face; a cell
    /// that hits an unsupported edge flags ALL its owned faces.
    pub needs_cpu: Vec<u32>,
    /// Per-cell canonical polygon area at the NEW seeds (shoelace of the same
    /// canonical vertices the swept quads use). The moving path's `cell_vols`.
    pub canon_area_new: Vec<f32>,
    /// Per-cell canonical polygon area at the OLD seeds (the closure's `V^n`).
    pub canon_area_old: Vec<f32>,
}

/// The swept-flux geometry pass: composes [`DeriveFaces`] (for the owned-face
/// offsets that address the face-major output) with the swept-quad kernel.
pub struct SweptFluxGeometry {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    derive: DeriveFaces,
}

impl SweptFluxGeometry {
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
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("swept:bgl"),
            entries: &[
                uniform(0),        // params
                storage(1, true),  // old_seeds
                storage(2, true),  // new_seeds
                storage(3, true),  // nbr_ids
                storage(4, true),  // face_bc
                storage(5, true),  // face_geom  [nx,ny,len,0]
                storage(6, true),  // cell_nfaces
                storage(7, true),  // status
                storage(8, true),  // offsets (owned-face base)
                storage(9, false), // out: swept
                storage(10, false), // out: needs_cpu
                storage(11, false), // out: canon_area_new (per cell)
                storage(12, false), // out: canon_area_old (per cell)
                storage(13, true), // segments at the NEW boundary (engine's table)
                storage(14, true), // segments at the OLD boundary (caller-packed)
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("swept:pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("swept:shader"),
            source: wgpu::ShaderSource::Wgsl(swept_shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("swept:areas"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("swept_areas"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bgl, derive: DeriveFaces::new(device) }
    }

    /// Compute per-face swept areas from the engine's current (NEW-seed) diagram
    /// and the supplied OLD seed positions (interleaved f32 x/y, one pair per
    /// cell — the seeds BEFORE this step's motion). `old_spec` is the boundary
    /// spec at t^n — its segment table gives polyline BOUNDARY-segment faces
    /// their old line (identical to the engine's table for a static boundary;
    /// segment COUNT must match, the diagram's seg ids index both tables). The
    /// engine must already hold the new diagram (post `run_regen` +
    /// `resolve_flagged`).
    pub fn compute(
        &self,
        ctx: &GpuContext,
        cache: &StagingBufferCache,
        engine: &GpuVoronoiEngine,
        old_seeds_xy: &[f32],
        old_spec: &BoundarySpec,
    ) -> GpuSweptAreas {
        use wgpu::BufferUsages as U;
        let n = engine.n_seeds();
        assert!(n > 0, "swept::compute called before a regen");
        assert_eq!(
            old_seeds_xy.len(),
            2 * n as usize,
            "swept::compute: old_seeds has {} floats, expected {}",
            old_seeds_xy.len(),
            2 * n
        );
        // Pack the OLD segment table exactly like `upload_case` packs the new
        // one: f32-rounded spec, `[ax, ay, bx, by]` per segment, one degenerate
        // pad entry when empty (storage buffers must be non-empty; never
        // referenced then — no face carries a segment tag).
        let old_spec32 = boundary_spec_f32(old_spec);
        let nseg_old = old_spec32.num_segments();
        assert_eq!(
            nseg_old,
            engine.num_segments(),
            "swept::compute: old_spec has {} segments, engine holds {} — the \
             diagram's segment ids must index both tables",
            nseg_old,
            engine.num_segments()
        );
        let mut segs_old: Vec<[f32; 4]> = Vec::with_capacity(nseg_old.max(1));
        for s in 0..nseg_old {
            let (a, b) = old_spec32.segment_points(s as u32);
            segs_old.push([a.x as f32, a.y as f32, b.x as f32, b.y as f32]);
        }
        if segs_old.is_empty() {
            segs_old.push([0.0; 4]);
        }

        let scan = self.derive.encode_offsets(ctx, cache, engine);
        let num_faces = scan.total;
        assert!(num_faces > 0, "swept::compute: zero faces");

        let domain = engine.domain();
        let params = SweptParams {
            n,
            k: K_FACE_MAX as u32,
            det_eps: engine.determinant_eps() as f32,
            _pad0: 0,
            domain_x: domain.x as f32,
            domain_y: domain.y as f32,
            _pad1: 0,
            _pad2: 0,
        };
        let b_params = create_buffer(
            &ctx.device,
            "swept:params",
            std::mem::size_of::<SweptParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        ctx.queue.write_buffer(&b_params, 0, bytemuck::bytes_of(&params));

        let b_old = create_buffer(
            &ctx.device,
            "swept:old_seeds",
            (old_seeds_xy.len() * 4) as u64,
            U::STORAGE | U::COPY_DST,
        );
        ctx.queue.write_buffer(&b_old, 0, bytemuck::cast_slice(old_seeds_xy));

        let nf = num_faces as u64;
        let b_swept = create_buffer(&ctx.device, "swept:areas", nf * 4, U::STORAGE | U::COPY_SRC);
        let b_needs = create_buffer(&ctx.device, "swept:needs_cpu", nf * 4, U::STORAGE | U::COPY_SRC);
        let nn = n as u64;
        let b_area_new =
            create_buffer(&ctx.device, "swept:canon_area_new", nn * 4, U::STORAGE | U::COPY_SRC);
        let b_area_old =
            create_buffer(&ctx.device, "swept:canon_area_old", nn * 4, U::STORAGE | U::COPY_SRC);
        let b_segs_old = create_buffer(
            &ctx.device,
            "swept:segments_old",
            (segs_old.len() * 16) as u64,
            U::STORAGE | U::COPY_DST,
        );
        ctx.queue.write_buffer(&b_segs_old, 0, bytemuck::cast_slice(&segs_old));

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("swept:bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_old.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: engine.b_seeds.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: engine.outputs.b_nbr_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: engine.outputs.b_face_bc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: engine.outputs.b_face_geom.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: engine.outputs.b_cell_nfaces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: engine.outputs.b_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 8, resource: scan.b_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: b_swept.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: b_needs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: b_area_new.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: b_area_old.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: engine.segments_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 14, resource: b_segs_old.as_entire_binding() },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("swept:enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("swept:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y) = dispatch_2d(n.div_ceil(WORKGROUP_SIZE).max(1));
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let prof = ProfilingStats::new();
        let swept_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_swept, nf * 4, "swept:rb_swept",
        ));
        let needs_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_needs, nf * 4, "swept:rb_needs",
        ));
        let area_new_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_area_new, nn * 4, "swept:rb_area_new",
        ));
        let area_old_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_area_old, nn * 4, "swept:rb_area_old",
        ));
        GpuSweptAreas {
            num_faces,
            swept: bytemuck::cast_slice(&swept_b).to_vec(),
            needs_cpu: bytemuck::cast_slice(&needs_b).to_vec(),
            canon_area_new: bytemuck::cast_slice(&area_new_b).to_vec(),
            canon_area_old: bytemuck::cast_slice(&area_old_b).to_vec(),
        }
    }
}

fn swept_shader() -> String {
    format!(
        r#"
struct Params {{ n: u32, k: u32, det_eps: f32, pad0: u32,
                 domain_x: f32, domain_y: f32, pad1: u32, pad2: u32 }};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read>       old_seeds: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       new_seeds: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read>       nbr_ids: array<u32>;
@group(0) @binding(4) var<storage, read>       face_bc: array<u32>;
@group(0) @binding(5) var<storage, read>       face_geom: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       cell_nfaces: array<u32>;
@group(0) @binding(7) var<storage, read>       cell_status: array<u32>;
@group(0) @binding(8) var<storage, read>       offsets: array<u32>;
@group(0) @binding(9)  var<storage, read_write> out_swept: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_needs: array<u32>;
@group(0) @binding(11) var<storage, read_write> out_area_new: array<f32>;
@group(0) @binding(12) var<storage, read_write> out_area_old: array<f32>;
@group(0) @binding(13) var<storage, read>       segments_new: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read>       segments_old: array<vec4<f32>>;

const NBR_NONE: u32 = 0xffffffffu;
const BC_NONE: u32  = 0xffffffffu;
const BC_SEG_FLAG: u32 = 0x80000000u;
const SUCCESS: u32  = 0u;

// A clip plane of a ring edge, resolved to a line n·x = c in the frame with the
// owner's NEW seed subtracted (a common per-face origin). `ok=false` marks an
// unsupported (polyline BOUNDARY-segment) edge.
struct Line {{ n: vec2<f32>, c: f32, ok: bool }};

// Bisector of seeds i and j in the `origin`-subtracted frame:
//   n·(x-m) = 0, m = midpoint(p_i,p_j), n = p_j - p_i  ->  n·x_rel = n·(m-origin).
fn bisector_line(pi: vec2<f32>, pj: vec2<f32>, origin: vec2<f32>) -> Line {{
    let nrm = pj - pi;
    let m = 0.5 * (pi + pj);
    return Line(nrm, dot(nrm, m - origin), true);
}}

// Domain-box side line in the origin-subtracted frame (side: 0=left x=0,
// 1=right x=domain_x, 2=bottom y=0, 3=top y=domain_y).
fn box_line(side: u32, origin: vec2<f32>) -> Line {{
    if (side == 0u) {{ return Line(vec2<f32>(1.0, 0.0), -origin.x, true); }}
    if (side == 1u) {{ return Line(vec2<f32>(1.0, 0.0), P.domain_x - origin.x, true); }}
    if (side == 2u) {{ return Line(vec2<f32>(0.0, 1.0), -origin.y, true); }}
    if (side == 3u) {{ return Line(vec2<f32>(0.0, 1.0), P.domain_y - origin.y, true); }}
    return Line(vec2<f32>(0.0), 0.0, false);
}}

fn line_intersect(a: Line, b: Line) -> vec2<f32> {{
    // Solve [a.n; b.n] x = [a.c; b.c].
    let det = a.n.x * b.n.y - a.n.y * b.n.x;
    if (abs(det) <= P.det_eps) {{ return vec2<f32>(0.0); }}
    let x = (a.c * b.n.y - b.c * a.n.y) / det;
    let y = (a.n.x * b.c - b.n.x * a.c) / det;
    return vec2<f32>(x, y);
}}

// Polyline BOUNDARY-segment line in the origin-subtracted frame, from the
// segment table at the given seed set (the new table is the engine's own
// clip table; the old table is caller-packed with the identical rule). Any
// normal orientation defines the same line — only the intersection point is
// consumed here, never a side test.
fn segment_line(seg: u32, origin: vec2<f32>, is_new: bool) -> Line {{
    var s4: vec4<f32>;
    if (is_new) {{ s4 = segments_new[seg]; }} else {{ s4 = segments_old[seg]; }}
    let d = s4.zw - s4.xy;
    let nrm = vec2<f32>(d.y, -d.x);
    return Line(nrm, dot(nrm, s4.xy - origin), true);
}}

// The line of ring edge `e` of cell `i`, in the origin-subtracted frame at the
// given seed set (old or new). `pi` = seed[i] at that set.
fn edge_line(i: u32, nbr: u32, bc: u32, pi: vec2<f32>, origin: vec2<f32>, is_new: bool) -> Line {{
    if (nbr != NBR_NONE) {{
        var pj: vec2<f32>;
        if (is_new) {{ pj = new_seeds[nbr]; }} else {{ pj = old_seeds[nbr]; }}
        return bisector_line(pi, pj, origin);
    }}
    // Boundary edge: box side (< 4) or polyline segment (BC_SEG_FLAG | seg).
    if (bc < 4u) {{ return box_line(bc, origin); }}
    if (bc != BC_NONE && (bc & BC_SEG_FLAG) != 0u) {{
        return segment_line(bc & 0x7fffffffu, origin, is_new);
    }}
    return Line(vec2<f32>(0.0), 0.0, false);   // defensive: unknown tag
}}

// The canonical vertex where ring edges `ea` and `eb` of cell `i` meet, in the
// origin-subtracted frame at the given seed set. Interior-interior collapses to
// the circumcenter of the seed triple {{i, ja, jb}} (== line_intersect of the two
// bisectors, which is the perpendicular-bisector construction of the
// circumcenter); the mixed / box cases fall out of the same line intersect.
fn vertex_at(i: u32, na: u32, ba: u32, nb: u32, bb: u32,
             origin: vec2<f32>, is_new: bool, needs: ptr<function, bool>) -> vec2<f32> {{
    var pi: vec2<f32>;
    if (is_new) {{ pi = new_seeds[i]; }} else {{ pi = old_seeds[i]; }}
    let la = edge_line(i, na, ba, pi, origin, is_new);
    let lb = edge_line(i, nb, bb, pi, origin, is_new);
    if (!la.ok || !lb.ok) {{ *needs = true; return vec2<f32>(0.0); }}
    // RELATIVE near-parallel guard: a bounded cell's bracket cannot be this
    // flat (its vertex would sit orders of magnitude outside the domain —
    // valid brackets have angles >> h/L_domain), but f32 rounding of
    // nearly-collinear adjacent segment lines can be. An ill-conditioned
    // intersect must fall back to the CPU path, never fabricate a far (or
    // origin-substituted) vertex silently.
    let det = la.n.x * lb.n.y - la.n.y * lb.n.x;
    if (abs(det) <= max(P.det_eps, 1e-8 * length(la.n) * length(lb.n))) {{
        *needs = true;
        return vec2<f32>(0.0);
    }}
    return line_intersect(la, lb);
}}

// Ring vertex `v` of cell `i` (the vertex STARTING edge `v`, i.e. bracketed by
// ring edges `v-1` and `v`), in the origin-subtracted frame at one seed set.
fn ring_vertex(i: u32, v: u32, nf: u32, origin: vec2<f32>, is_new: bool,
               needs: ptr<function, bool>) -> vec2<f32> {{
    let prev = (v + nf - 1u) % nf;
    let n_prev = nbr_ids[i * P.k + prev]; let b_prev = face_bc[i * P.k + prev];
    let n_cur  = nbr_ids[i * P.k + v];    let b_cur  = face_bc[i * P.k + v];
    return vertex_at(i, n_prev, b_prev, n_cur, b_cur, origin, is_new, needs);
}}

@compute @workgroup_size({wg})
fn swept_areas(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * nwg.x + wid.x) * {wg}u + lid.x;
    if (i >= P.n) {{ return; }}
    if (cell_status[i] != SUCCESS) {{ return; }}
    let nf = cell_nfaces[i];
    if (nf < 3u) {{ return; }}
    let origin = new_seeds[i];       // common per-face frame

    // Canonical ring vertices (old + new) once into local arrays, so the swept
    // quads and the polygon-area shoelace share the SAME vertices — the
    // telescoping identity Σσ·swept = ΔV then holds by construction.
    var vold: array<vec2<f32>, {kmax}>;
    var vnew: array<vec2<f32>, {kmax}>;
    var need = false;
    for (var v: u32 = 0u; v < nf; v = v + 1u) {{
        vold[v] = ring_vertex(i, v, nf, origin, false, &need);
        vnew[v] = ring_vertex(i, v, nf, origin, true,  &need);
    }}
    if (need) {{
        // Unsupported edge tag (defensive — bisector/box/segment all resolve):
        // flag every owned face; the caller falls back to the CPU path this
        // regen. Areas left 0 (unused on fallback).
        var rank0: u32 = offsets[i];
        for (var e: u32 = 0u; e < nf; e = e + 1u) {{
            let nbr = nbr_ids[i * P.k + e];
            if ((nbr == NBR_NONE) || (i < nbr)) {{ out_needs[rank0] = 1u; out_swept[rank0] = 0.0; rank0 = rank0 + 1u; }}
        }}
        out_area_new[i] = 0.0; out_area_old[i] = 0.0;
        return;
    }}

    // Shoelace polygon areas from the canonical vertices.
    var a_new = 0.0;
    var a_old = 0.0;
    for (var v: u32 = 0u; v < nf; v = v + 1u) {{
        let w = (v + 1u) % nf;
        a_new = a_new + (vnew[v].x * vnew[w].y - vnew[v].y * vnew[w].x);
        a_old = a_old + (vold[v].x * vold[w].y - vold[v].y * vold[w].x);
    }}
    out_area_new[i] = 0.5 * a_new;
    out_area_old[i] = 0.5 * a_old;

    // Per-owned-face swept quad, oriented onto the stored owner-outward normal
    // via the OLD tangent (matches swept_area_along_normal). Edge e runs from
    // ring vertex e (start) to e+1 (end).
    var rank: u32 = offsets[i];
    for (var e: u32 = 0u; e < nf; e = e + 1u) {{
        let nbr = nbr_ids[i * P.k + e];
        if (!((nbr == NBR_NONE) || (i < nbr))) {{ continue; }}
        let g = rank;
        rank = rank + 1u;
        out_needs[g] = 0u;
        let w = (e + 1u) % nf;
        let pA_old = vold[e]; let pB_old = vold[w];
        let pA_new = vnew[e]; let pB_new = vnew[w];
        // Swept-quad area ½·cross(p2n - p1o, p1n - p2o) with p1=A (ring start),
        // p2=B (ring end). Because we walk cell i's CCW ring and i OWNS this edge,
        // the ring-oriented quad already carries the owner-outward (σ=+1) sign the
        // per-cell telescoping Σσ·swept = ΔV needs — no normal-flip (that is only
        // the CPU's fix for its arbitrary face_v1/v2 order; here the winding is
        // canonical). A global sign aligns the quad to the +Δarea convention.
        let d1 = pB_new - pA_old;
        let d2 = pA_new - pB_old;
        let quad = 0.5 * (d1.x * d2.y - d1.y * d2.x);
        out_swept[g] = -quad;
    }}
}}
"#,
        wg = WORKGROUP_SIZE,
        kmax = K_FACE_MAX,
    )
}
