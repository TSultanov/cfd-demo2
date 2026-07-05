//! WGSL source for the `voronoi_cell` kernel: one thread per seed, all
//! arithmetic seed-relative f32. The cell starts as the seed-relative domain
//! bbox; `SeedKind::Boundary` seeds clip their own segment line(s) first (they
//! pass through the seed, distance 0, so precede every bisector), then bisector
//! planes stream in Chebyshev grid-ring order with the security-radius stop
//! `d² > 4·R²`.
//!
//! ## Edge-tag encoding (kernel-internal AND `b_face_bc` output)
//!
//! A clip plane is one u32 `tag` mirroring the CPU `PlaneTag`:
//! - `tag < TAG_SEG_FLAG (0x8000_0000)` — `Bisector(tag)`, the neighbor seed id;
//! - `TAG_SEG_FLAG ≤ tag < TAG_BOX_BASE` — `Boundary(tag & 0x7fff_ffff)`, a
//!   boundary-segment id into the uploaded segment table;
//! - `tag ≥ TAG_BOX_BASE (0xffff_fff0)` — `Box(tag − TAG_BOX_BASE)`, a bbox side
//!   (0=left, 1=right, 2=bottom, 3=top).
//!
//! `b_nbr_ids` holds the canonicalized seed id for bisector faces and `NBR_NONE`
//! otherwise; `b_face_bc` holds the box side id (< 4) or the `TAG_SEG_FLAG|seg`
//! boundary tag, `BC_NONE` for interior/unused slots.
//!
//! ## Canonical vertices + certified error bounds
//!
//! Every intersection vertex is the 2×2 solve of its two defining edge planes
//! (plane coefficients are pure functions of the f32 inputs — no chain error),
//! NOT the Sutherland-Hodgman lerp. A per-vertex COMPONENTWISE error bound
//! `ve = (ve_x, ve_y)` is carried: componentwise matters because axis-aligned
//! boundary structures produce coordinates that are EXACT in both f32 and f64,
//! and a scalar norm bound would flag every such cell.
//!
//! ## Conservative epsilon filter (one-sided by design)
//!
//! `NEEDS_EXACT` is flagged whenever f32 cannot certify agreement with the f64
//! oracle on the same f32-rounded inputs. The classification band is a
//! RUNNING-ERROR bound so exactly-zero / exactly-cancelling structured
//! arithmetic yields a zero band instead of a domain-scale one. Near-parallel
//! plane pairs do not flag on a det threshold alone: the certified vertex bound
//! (honest under cancellation) propagates into the bands and the final
//! `VE_MAX_REL` check; only an exactly-degenerate det (≤ DET_ZERO·scale) falls
//! back to the lerp vertex and flags. Load-bearing for boundary support: a
//! straight-wall vertex seed's two own lines meet with a det of order 1e-6·scale
//! but both offsets are EXACTLY zero, so the solve yields exactly (0,0) with a
//! zero certified bound.

use super::{status, BC_NONE, K_FACE_MAX, MAX_VERTS, NBR_NONE};

/// Dot-product/coefficient rounding slack: 8 × f32 unit roundoff. Scales the
/// running-error classification band (see module docs); the worst analyzed
/// chain (product roundings + coefficient representation error + the f64
/// off computation) is ≤ 5u on the product terms and ≤ 8u on the offset
/// term, which the `(products + 2|off|)` band shape covers.
pub(super) const C_DOT_EPS: f32 = 4.8e-7;

/// Canonical-vertex error-bound slack for single-rounding terms (bbox
/// corner subtractions): 8 × f32 unit roundoff.
pub(super) const C_VERT_EPS: f32 = 4.8e-7;

/// Cramer numerator/determinant product-rounding slack: 4 × f32 unit
/// roundoff (each product term carries one multiply rounding plus ≤ 2
/// roundings of input representation — pa.z of a bisector is
/// `fl(0.5·fl(dot))`). Deliberately tighter than `C_DOT_EPS`: the
/// componentwise numerator-magnitude bound is honest under cancellation
/// but scales like `(|A|+|B|)/sin θ`, so slack here directly multiplies
/// the flag rate on moderate-angle plane pairs.
///
/// HEADROOM NOTE: the `/det` in the canonical solve is
/// only 2.5 ULP under the WGSL accuracy spec (division is NOT correctly
/// rounded). The bound still covers it because `dmag/|det| ≥ 1` makes the
/// `dmag·|vi|` term contribute ≥ 4u and `|num|/det ≈ |vi|` adds another
/// ≥ 4u — ~8u available vs ~7u consumed (product roundings + the 2.5-ULP
/// division). The headroom is thin: do NOT tighten below 4u without
/// redoing that budget.
pub(super) const C_NUM_EPS: f32 = 2.4e-7;

/// Exact-degeneracy threshold for the 2×2 canonical solve (relative to
/// `|A||B|`); below it the lerp fallback vertex is kept and the cell is
/// flagged. Genuinely near-parallel-but-nonzero dets take the canonical
/// solve and carry their (large) certified error bound instead.
pub(super) const DET_ZERO: f32 = 1e-12;

/// Faces shorter than `EPS_SHORT · h_min` (error-adjusted) are flagged:
/// 4× the downstream `eps_face = 1e-6·h` topology filter, covering the
/// worst face-length error so threshold-straddling faces always resolve
/// through the f64 fallback.
pub(super) const EPS_SHORT: f32 = 4e-6;

/// Max tolerated certified vertex error BOUND in units of `h_min`, matching
/// the `1e-5·h` geometry parity tolerance; the bound carries ≥ 8× slack over
/// the real rounding error.
pub(super) const VE_MAX_REL: f32 = 1e-5;

/// Relative security-radius margin: the ring-sweep stop requires
/// `lb²·(1−SECURITY_MARGIN) > 4·R²`. Budget: on an UNFLAGGED cell every vertex
/// may carry a
/// certified componentwise error up to `VE_MAX_REL·h_min` (norm ≤ √2×),
/// and R can be as small as `h_min/2`, so `4·r2` can undersell the true
/// `4·R²` by up to `~4·√2·VE_MAX_REL ≈ 5.7e-5` relative; 1e-4 carries
/// ~1.75× headroom. The lb leg's f32 error is ABSOLUTE (ulps of DOMAIN
/// scale, not of lb — see `ring_lower_bound`) and cannot be covered by any
/// relative margin; it is handled by the `grid_slack` baseline instead
/// (`lb_abs_slack` in engine.rs).
pub(super) const SECURITY_MARGIN: f32 = 1e-4;

/// Build the shader source. Numeric constants are injected from the Rust
/// consts so the two can never drift.
pub fn voronoi_cell_shader() -> String {
    let header = format!(
        "const MAX_VERTS: u32 = {max_verts}u;\n\
         const K_FACE_MAX: u32 = {k_face_max}u;\n\
         const NBR_NONE: u32 = {nbr_none}u;\n\
         const BC_NONE: u32 = {bc_none}u;\n\
         const STATUS_SUCCESS: u32 = {s_ok}u;\n\
         const STATUS_VERT_OVERFLOW: u32 = {s_vert}u;\n\
         const STATUS_FACE_OVERFLOW: u32 = {s_face}u;\n\
         const STATUS_NEEDS_EXACT: u32 = {s_exact}u;\n\
         const STATUS_EMPTY_CELL: u32 = {s_empty}u;\n\
         const C_DOT_EPS: f32 = {c_dot:e};\n\
         const C_VERT_EPS: f32 = {c_vert:e};\n\
         const C_NUM_EPS: f32 = {c_num:e};\n\
         const DET_ZERO: f32 = {det_zero:e};\n\
         const EPS_SHORT: f32 = {eps_short:e};\n\
         const VE_MAX_REL: f32 = {ve_max:e};\n\
         const SECURITY_SCALE: f32 = {sec_scale:.9};\n",
        max_verts = MAX_VERTS,
        k_face_max = K_FACE_MAX,
        nbr_none = NBR_NONE,
        bc_none = BC_NONE,
        s_ok = status::SUCCESS,
        s_vert = status::VERT_OVERFLOW,
        s_face = status::FACE_OVERFLOW,
        s_exact = status::NEEDS_EXACT,
        s_empty = status::EMPTY_CELL,
        c_dot = C_DOT_EPS,
        c_vert = C_VERT_EPS,
        c_num = C_NUM_EPS,
        det_zero = DET_ZERO,
        eps_short = EPS_SHORT,
        ve_max = VE_MAX_REL,
        sec_scale = 1.0 - SECURITY_MARGIN,
    );
    format!("{header}{BODY}")
}

const BODY: &str = r#"
// Edge-tag encoding (see wgsl.rs module docs): bisector = neighbor seed id;
// TAG_SEG_FLAG | seg = boundary-segment line; TAG_BOX_BASE + side = domain
// bbox side (0=left, 1=right, 2=bottom, 3=top, matching PlaneTag::Box).
const TAG_SEG_FLAG: u32 = 0x80000000u;
const TAG_BOX_BASE: u32 = 0xfffffff0u;
// seed_kind sentinel: Interior seed (no own segments).
const SEG_NONE: u32 = 0xffffffffu;
// clip_plane() sentinel: result exceeded MAX_VERTS.
const CLIP_OVERFLOW: u32 = 0xffffffffu;

struct Params {
    n_seeds: u32,
    gw: u32,
    gh: u32,
    flag_cap: u32,
    cell_size: f32,
    edge_len_eps: f32,
    domain_x: f32,
    domain_y: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> seeds: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> canon: array<u32>;
@group(0) @binding(3) var<storage, read> grid_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> grid_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> nbr_ids: array<u32>;
@group(0) @binding(6) var<storage, read_write> face_bc: array<u32>;
@group(0) @binding(7) var<storage, read_write> face_geom: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> face_mid: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> cell_centroid: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read_write> cell_area: array<f32>;
@group(0) @binding(11) var<storage, read_write> cell_nfaces: array<u32>;
@group(0) @binding(12) var<storage, read_write> status_out: array<u32>;
// [0] = atomic append cursor, [1..] = flagged cell ids (order is the only
// nondeterministic output; consumers sort on the CPU).
@group(0) @binding(13) var<storage, read_write> flagged: array<atomic<u32>>;
// Per-seed (seg_prev, seg_next); (SEG_NONE, SEG_NONE) = Interior.
@group(0) @binding(14) var<storage, read> seed_kind: array<vec2<u32>>;
// Flattened boundary-segment table: (ax, ay, bx, by), fluid on the LEFT of
// a -> b (the CPU BoundarySpec convention).
@group(0) @binding(15) var<storage, read> segments: array<vec4<f32>>;
// Ring vertex per face slot (seed-relative): vertex e STARTS edge e, whose
// tag/geometry live in the same slot — the CPU MeshlessDiagram layout.
@group(0) @binding(16) var<storage, read_write> ring_vert: array<vec2<f32>>;
// Diagnostics (deterministic): low 24 bits = grid bins processed by this
// cell's traversal (the F3 graded-set instrument), high 8 bits = the
// epsilon-filter condition mask of a NEEDS_EXACT cell.
@group(0) @binding(17) var<storage, read_write> visited_bins: array<u32>;
// [0] = distance-lower-bound derate, subtracted from the ring sweep's
// lower bound. upload_case initializes it to the ABSOLUTE f32 slack
// `lb_abs_slack` (covers ring_lower_bound's domain-scale ulp overestimate,
// see its comment); chained Lloyd iterations accumulate the per-iteration
// max seed displacement on top (grid staleness — seeds move without a CPU
// SeedGrid rebuild, see lloyd.rs).
@group(0) @binding(18) var<storage, read> grid_slack: array<f32>;

// Zero/PAD the padded face slots and scalar outputs of a cell that has no
// usable geometry (empty, overflow — hard-failure statuses).
fn write_empty_outputs(i: u32) {
    for (var s = 0u; s < K_FACE_MAX; s = s + 1u) {
        let base = i * K_FACE_MAX + s;
        nbr_ids[base] = NBR_NONE;
        face_bc[base] = BC_NONE;
        face_geom[base] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        face_mid[base] = vec2<f32>(0.0, 0.0);
        ring_vert[base] = vec2<f32>(0.0, 0.0);
    }
    cell_nfaces[i] = 0u;
    cell_area[i] = 0.0;
    cell_centroid[i] = vec2<f32>(0.0, 0.0);
}

// Seed-relative boundary-segment line as (q.x, q.y, off, oerr):
// { x : dot(x, q) <= off } keeps the fluid (fluid on the left of d = b - a,
// outward solid-side normal q = (d.y, -d.x)) — the CPU
// HalfPlane::segment_line formula. `oerr` is a certified bound on the f32
// off-computation error vs the f64 oracle on the same rounded endpoints.
// Exactness detection: when the seed IS one of the segment endpoints
// (vertex seeds), a_rel == 0 or a_rel == -d bitwise and the two products
// cancel EXACTLY in both f32 and f64 (fl(-x) == -fl(x)), so off = 0 with
// zero error — load-bearing for keeping straight-wall seeds unflagged.
fn segment_plane(seg: u32, p: vec2<f32>) -> vec4<f32> {
    let s4 = segments[seg];
    let a = s4.xy - p;
    let d = s4.zw - s4.xy;
    let q = vec2<f32>(d.y, -d.x);
    let t1 = a.x * q.x;
    let t2 = a.y * q.y;
    var off = t1 + t2;
    var oerr = C_DOT_EPS * (abs(t1) + abs(t2));
    if ((a.x == 0.0 && a.y == 0.0) || (a.x == -d.x && a.y == -d.y)) {
        off = 0.0;
        oerr = 0.0;
    }
    return vec4<f32>(q.x, q.y, off, oerr);
}

// Seed-relative plane of an edge tag as (q.x, q.y, off, oerr):
// { x : dot(x, q) <= off + eps }. Plane coefficients are pure functions of
// the seeds/segments/domain — no chain error; oerr bounds the offset's own
// f32 computation error (zero for bisectors/box sides, whose offsets are
// single-rounding and covered by the |off| running term).
fn plane_of(tag: u32, p: vec2<f32>) -> vec4<f32> {
    if (tag >= TAG_BOX_BASE) {
        switch (tag - TAG_BOX_BASE) {
            case 0u: { return vec4<f32>(-1.0, 0.0, p.x, 0.0); }
            case 1u: { return vec4<f32>(1.0, 0.0, params.domain_x - p.x, 0.0); }
            case 2u: { return vec4<f32>(0.0, -1.0, p.y, 0.0); }
            default: { return vec4<f32>(0.0, 1.0, params.domain_y - p.y, 0.0); }
        }
    }
    if ((tag & TAG_SEG_FLAG) != 0u) {
        return segment_plane(tag & 0x7fffffffu, p);
    }
    let q = seeds[tag] - p;
    return vec4<f32>(q.x, q.y, 0.5 * dot(q, q), 0.0);
}

struct ClipResult {
    // New vertex count: 0 = cell clipped away, CLIP_OVERFLOW = > MAX_VERTS,
    // n unchanged for a redundant plane.
    n: u32,
    // Updated max squared vertex distance (security radius squared).
    r2: f32,
};

// Sutherland-Hodgman clip of the convex ring by the half-plane
// { x : dot(x, q) - off <= eps } (seed-relative). Mirrors the CPU sh_emit
// CLASSIFICATION rule exactly (vertices with |s| <= eps count as inside,
// degenerate verts kept), but intersection vertices are constructed
// canonically (module docs): the 2x2 solve of the crossed edge's plane with
// the new plane, with a certified componentwise error bound written to
// `pe`; the lerp (t = s_u / (s_u - s_w)) survives only as the
// exactly-degenerate-det fallback. A redundant plane (no vertex outside)
// leaves the ring bitwise untouched. Epsilon-filter hits are OR-ed into
// *pu. `pl` = (q.x, q.y, off, oerr).
fn clip_plane(
    pv: ptr<function, array<vec2<f32>, MAX_VERTS>>,
    pt: ptr<function, array<u32, MAX_VERTS>>,
    pe: ptr<function, array<vec2<f32>, MAX_VERTS>>,
    n: u32,
    r2_in: f32,
    pl: vec4<f32>,
    eps: f32,
    ql: f32,
    tag: u32,
    p: vec2<f32>,
    pu: ptr<function, u32>,
) -> ClipResult {
    let q = pl.xy;
    let off = pl.z;
    let oerr = pl.w;
    var s: array<f32, MAX_VERTS>;
    var any_out = false;
    var any_in = false;
    for (var e = 0u; e < n; e = e + 1u) {
        let v = (*pv)[e];
        let t1 = v.x * q.x;
        let t2 = v.y * q.y;
        let sv = t1 + t2 - off;
        s[e] = sv;
        // Filter condition 1 (orientation band, running-error form): if the
        // certified f32-vs-f64 error band around s reaches the
        // classification threshold, the f64 oracle may classify this vertex
        // differently. Exact structured arithmetic (t1 = t2 = off = 0)
        // yields a zero band — no false flag on axis-aligned walls.
        let band = C_DOT_EPS * (abs(t1) + abs(t2) + 2.0 * abs(off))
            + abs(q.x) * (*pe)[e].x + abs(q.y) * (*pe)[e].y + oerr;
        if (abs(sv - eps) <= band) {
            *pu = *pu | 1u;
        }
        if (sv > eps) {
            any_out = true;
        } else {
            any_in = true;
        }
    }
    if (!any_out) {
        return ClipResult(n, r2_in);
    }
    if (!any_in) {
        return ClipResult(0u, 0.0);
    }
    var nx: array<vec2<f32>, MAX_VERTS>;
    var nt: array<u32, MAX_VERTS>;
    var ne: array<vec2<f32>, MAX_VERTS>;
    var m = 0u;
    var r2 = 0.0;
    for (var e = 0u; e < n; e = e + 1u) {
        var w = e + 1u;
        if (w == n) {
            w = 0u;
        }
        let u_in = s[e] <= eps;
        let w_in = s[w] <= eps;
        if (u_in) {
            if (m == MAX_VERTS) {
                return ClipResult(CLIP_OVERFLOW, 0.0);
            }
            let v = (*pv)[e];
            nx[m] = v;
            nt[m] = (*pt)[e];
            ne[m] = (*pe)[e];
            r2 = max(r2, dot(v, v));
            m = m + 1u;
        }
        if (u_in != w_in) {
            if (m == MAX_VERTS) {
                return ClipResult(CLIP_OVERFLOW, 0.0);
            }
            // Canonical vertex: intersection of the crossed edge's plane
            // with the new plane (both exact functions of the inputs),
            // Cramer 2x2, with a certified error bound that stays honest
            // under numerator cancellation and offset error.
            let pa = plane_of((*pt)[e], p);
            let det = pa.x * q.y - pa.y * q.x;
            let la = length(pa.xy);
            let scale = la * ql;
            var vi: vec2<f32>;
            var ei: vec2<f32>;
            if (abs(det) > DET_ZERO * scale) {
                let numx = pa.z * q.y - pa.y * off;
                let numy = pa.x * off - pa.z * q.x;
                vi = vec2<f32>(numx, numy) / det;
                // COMPONENTWISE certified bound: per-component numerator
                // product-rounding magnitudes + the det-rounding term
                // (proportional to the solution component) + the plane
                // offset errors. Componentwise is load-bearing: axis-
                // aligned boundary structures produce vertices with one
                // EXACT coordinate (all products in that numerator exactly
                // zero) whose later — equally exact — classifications
                // against collinear wall lines must see a zero band, or
                // every straight-wall seed flags (measured on backstep).
                let dmag = abs(pa.x * q.y) + abs(pa.y * q.x);
                ei = vec2<f32>(
                    C_NUM_EPS * (abs(pa.z * q.y) + abs(pa.y * off) + dmag * abs(vi.x))
                        + pa.w * abs(q.y) + oerr * abs(pa.y),
                    C_NUM_EPS * (abs(pa.x * off) + abs(pa.z * q.x) + dmag * abs(vi.y))
                        + pa.w * abs(q.x) + oerr * abs(pa.x),
                ) / abs(det);
            } else {
                // Filter condition 2 (exactly-degenerate denominator): keep
                // the lerp vertex, give it a garbage-level error bound, flag.
                let t = clamp(s[e] / (s[e] - s[w]), 0.0, 1.0);
                vi = (*pv)[e] + t * ((*pv)[w] - (*pv)[e]);
                ei = vec2<f32>(length(vi), length(vi));
                *pu = *pu | 2u;
            }
            nx[m] = vi;
            // Leaving the keep-set starts the new plane's edge; re-entering
            // resumes the original edge.
            nt[m] = select((*pt)[e], tag, u_in);
            ne[m] = ei;
            r2 = max(r2, dot(vi, vi));
            m = m + 1u;
        }
    }
    if (m < 3u) {
        return ClipResult(0u, 0.0);
    }
    for (var e = 0u; e < m; e = e + 1u) {
        (*pv)[e] = nx[e];
        (*pt)[e] = nt[e];
        (*pe)[e] = ne[e];
    }
    return ClipResult(m, r2);
}

// Clip every candidate stored in grid bin `bin` (in-bin ids are ascending —
// the CPU counting sort guarantees it). Skips candidates sharing the cell's
// quantize bin (canon[j] == ci covers both j == i and coalesced duplicates
// of i, exactly the CPU retain rule). Returns 0 to continue, else a
// terminal status.
fn process_bin(
    bin: u32,
    ci: u32,
    p: vec2<f32>,
    pv: ptr<function, array<vec2<f32>, MAX_VERTS>>,
    pt: ptr<function, array<u32, MAX_VERTS>>,
    pe: ptr<function, array<vec2<f32>, MAX_VERTS>>,
    pn: ptr<function, u32>,
    pr2: ptr<function, f32>,
    pu: ptr<function, u32>,
) -> u32 {
    let lo = grid_offsets[bin];
    let hi = grid_offsets[bin + 1u];
    for (var e = lo; e < hi; e = e + 1u) {
        let j = grid_ids[e];
        if (canon[j] == ci) {
            continue;
        }
        // Bisector of the seed (origin) and neighbor j: s(x) = x.q - 0.5|q|^2,
        // eps = |q| * edge_len_eps (the CPU HalfPlane::bisector formula).
        // Bitwise symmetry with thread j needs no (min,max) ordering: IEEE
        // subtraction gives fl(p_j - p_i) == -fl(p_i - p_j) exactly, and off
        // and eps depend only on componentwise magnitudes, so both threads
        // evaluate the exactly-negated coefficients of the same line.
        let q = seeds[j] - p;
        let q2 = dot(q, q);
        let ql = sqrt(q2);
        let res = clip_plane(
            pv, pt, pe, *pn, *pr2,
            vec4<f32>(q.x, q.y, 0.5 * q2, 0.0),
            ql * params.edge_len_eps, ql, j, p, pu,
        );
        if (res.n == CLIP_OVERFLOW) {
            return STATUS_VERT_OVERFLOW;
        }
        if (res.n == 0u) {
            return STATUS_EMPTY_CELL;
        }
        *pn = res.n;
        *pr2 = res.r2;
    }
    return 0u;
}

// Lower bound (in distance units) on the distance from p to any seed in a
// grid bin of Chebyshev ring r or beyond — the CPU SeedGrid::ring_lower_bound
// formula. Ring r=0 has bound 0. f32 CAVEAT (stage-5 review): `cell_size` is
// the f32 rounding of the f64 grid pitch and the products below round once
// more, so the result can OVERESTIMATE the true bound by up to
// ~2·2⁻²⁴·domain ABSOLUTE — ulps of DOMAIN scale, not of the result. The
// caller subtracts `grid_slack[0]`, whose upload_case baseline
// (`lb_abs_slack` ≥ 4·2⁻²⁴·max_domain, engine.rs) covers this with 2×
// headroom; Lloyd staleness accumulates on top of that baseline.
fn ring_lower_bound(p: vec2<f32>, bx: u32, by: u32, r: u32) -> f32 {
    let cs = params.cell_size;
    let rf = f32(r);
    let x_lo = (f32(bx) - (rf - 1.0)) * cs;
    let x_hi = (f32(bx) + rf) * cs;
    let y_lo = (f32(by) - (rf - 1.0)) * cs;
    let y_hi = (f32(by) + rf) * cs;
    let dx = min(p.x - x_lo, x_hi - p.x);
    let dy = min(p.y - y_lo, y_hi - p.y);
    return max(min(dx, dy), 0.0);
}

fn box_normal(side: u32) -> vec2<f32> {
    switch side {
        case 0u: { return vec2<f32>(-1.0, 0.0); }  // left
        case 1u: { return vec2<f32>(1.0, 0.0); }   // right
        case 2u: { return vec2<f32>(0.0, -1.0); }  // bottom
        default: { return vec2<f32>(0.0, 1.0); }   // top
    }
}

@compute @workgroup_size(64)
fn voronoi_cell(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride_x = nwg.x * 64u;
    let i = gid.y * stride_x + gid.x;
    if (i >= params.n_seeds) {
        return;
    }

    let ci = canon[i];
    if (ci != i) {
        // Coalesced duplicate: a lower-index seed shares this seed's
        // quantize bin — that seed keeps the whole cell (CPU EmptyCell
        // rule). A legitimate outcome, NOT flagged for fallback.
        write_empty_outputs(i);
        status_out[i] = STATUS_EMPTY_CELL;
        visited_bins[i] = 0u;
        return;
    }
    let p = seeds[i];

    // Seed-relative CCW domain bbox: verts (x0,y0)(x1,y0)(x1,y1)(x0,y1),
    // edge tags bottom/right/top/left = Box(2)/Box(1)/Box(3)/Box(0) — the
    // CPU bbox_ring layout. Componentwise corner error bounds: negations
    // (-p.x, -p.y) are EXACT; domain-side subtractions carry one rounding.
    let x0 = -p.x;
    let y0 = -p.y;
    let x1 = params.domain_x - p.x;
    let y1 = params.domain_y - p.y;
    var vx: array<vec2<f32>, MAX_VERTS>;
    var vt: array<u32, MAX_VERTS>;
    var ve: array<vec2<f32>, MAX_VERTS>;
    vx[0] = vec2<f32>(x0, y0);
    vx[1] = vec2<f32>(x1, y0);
    vx[2] = vec2<f32>(x1, y1);
    vx[3] = vec2<f32>(x0, y1);
    vt[0] = TAG_BOX_BASE + 2u;
    vt[1] = TAG_BOX_BASE + 1u;
    vt[2] = TAG_BOX_BASE + 3u;
    vt[3] = TAG_BOX_BASE + 0u;
    let ex = C_VERT_EPS * abs(x1);
    let ey = C_VERT_EPS * abs(y1);
    ve[0] = vec2<f32>(0.0, 0.0);
    ve[1] = vec2<f32>(ex, 0.0);
    ve[2] = vec2<f32>(ex, ey);
    ve[3] = vec2<f32>(0.0, ey);
    var n = 4u;
    var r2 = 0.0;
    for (var e = 0u; e < 4u; e = e + 1u) {
        r2 = max(r2, dot(vx[e], vx[e]));
    }
    // Epsilon-filter accumulator (any nonzero => NEEDS_EXACT).
    var unc = 0u;
    var st = 0u;
    var bins = 0u;

    // Boundary-kind seeds clip their OWN segment line(s) FIRST — the exact
    // M0 own_planes/clip_own order (seg_prev, then seg_next if different).
    // These lines pass through the seed (distance 0), so they precede every
    // bisector in distance order; the r2 they leave feeds the security stop.
    let sk = seed_kind[i];
    if (sk.x != SEG_NONE) {
        for (var o = 0u; o < 2u; o = o + 1u) {
            var seg = sk.x;
            if (o == 1u) {
                seg = sk.y;
                if (seg == sk.x) {
                    break;
                }
            }
            let pl = segment_plane(seg, p);
            let ql = length(pl.xy);
            let res = clip_plane(
                &vx, &vt, &ve, n, r2, pl,
                ql * params.edge_len_eps, ql, TAG_SEG_FLAG | seg, p, &unc,
            );
            if (res.n == CLIP_OVERFLOW) {
                st = STATUS_VERT_OVERFLOW;
                break;
            }
            if (res.n == 0u) {
                // Seed on the wrong side of its own wall: the CPU calls
                // this EmptyCell (broken input); flag it so the f64
                // fallback renders the authoritative verdict.
                st = STATUS_EMPTY_CELL;
                break;
            }
            n = res.n;
            r2 = res.r2;
        }
    }

    // Home bin (clamped; kernel f32 binning may disagree with the CPU's f64
    // assignment by one bin near an edge — the ring lower bound stays valid
    // because p then lies strictly inside the ring-(r-1) box, and coverage
    // is unaffected: every bin is visited by ring r_max).
    let gw = params.gw;
    let gh = params.gh;
    let bx = min(u32(max(p.x / params.cell_size, 0.0)), gw - 1u);
    let by = min(u32(max(p.y / params.cell_size, 0.0)), gh - 1u);
    let r_max = max(max(bx, gw - 1u - bx), max(by, gh - 1u - by));

    let bxi = i32(bx);
    let byi = i32(by);
    let gwi = i32(gw);
    let ghi = i32(gh);

    // Chebyshev ring sweep with the security-radius stop. Completing the
    // sweep without the stop means every seed in the grid was clipped —
    // exhaustive, therefore exact (SUCCESS, matching CPU semantics). The
    // stop is derated by SECURITY_SCALE so an R^2 underestimated by vertex
    // error (bounded by VE_MAX_REL, else the cell is flagged) can never
    // hide a plane that would cut a post-eps_face-visible face.
    if (st == 0u) {
        for (var r = 0u; r <= r_max; r = r + 1u) {
            if (r > 0u) {
                // Derate the lower bound by grid_slack: the absolute f32
                // lb slack baseline + any Lloyd grid-staleness displacement
                // (binding 18 docs).
                let lb = ring_lower_bound(p, bx, by, r) - grid_slack[0];
                if (lb > 0.0 && lb * lb * SECURITY_SCALE > 4.0 * r2) {
                    break;
                }
            }
            if (r == 0u) {
                bins = bins + 1u;
                st = process_bin(by * gw + bx, ci, p, &vx, &vt, &ve, &n, &r2, &unc);
                if (st != 0u) {
                    break;
                }
                continue;
            }
            let ri = i32(r);
            // Top/bottom rows of the ring (CPU for_each_ring_bin order).
            for (var gx = bxi - ri; gx <= bxi + ri; gx = gx + 1) {
                if (gx < 0 || gx >= gwi) {
                    continue;
                }
                for (var k = 0u; k < 2u; k = k + 1u) {
                    var gy = byi - ri;
                    if (k == 1u) {
                        gy = byi + ri;
                    }
                    if (gy >= 0 && gy < ghi) {
                        bins = bins + 1u;
                        st = process_bin(u32(gy) * gw + u32(gx), ci, p, &vx, &vt, &ve, &n, &r2, &unc);
                        if (st != 0u) {
                            break;
                        }
                    }
                }
                if (st != 0u) {
                    break;
                }
            }
            if (st != 0u) {
                break;
            }
            // Left/right columns (excluding the corners already visited).
            for (var gy = byi - ri + 1; gy <= byi + ri - 1; gy = gy + 1) {
                if (gy < 0 || gy >= ghi) {
                    continue;
                }
                for (var k = 0u; k < 2u; k = k + 1u) {
                    var gx = bxi - ri;
                    if (k == 1u) {
                        gx = bxi + ri;
                    }
                    if (gx >= 0 && gx < gwi) {
                        bins = bins + 1u;
                        st = process_bin(u32(gy) * gw + u32(gx), ci, p, &vx, &vt, &ve, &n, &r2, &unc);
                        if (st != 0u) {
                            break;
                        }
                    }
                }
                if (st != 0u) {
                    break;
                }
            }
            if (st != 0u) {
                break;
            }
        }
    }

    if (st == 0u && n > K_FACE_MAX) {
        st = STATUS_FACE_OVERFLOW;
    }
    if (st != 0u) {
        write_empty_outputs(i);
        status_out[i] = st;
        visited_bins[i] = bins;
        // The ONLY atomic in the kernel: append to the flag list. Order is
        // nondeterministic; ids are sorted on the CPU before use.
        let slot = atomicAdd(&flagged[0], 1u);
        if (slot < params.flag_cap) {
            atomicStore(&flagged[1u + slot], i);
        }
        return;
    }

    // Local scale h_min = distance to the nearest final bisector neighbor
    // (drives the short-face / vertex-error filter conditions).
    var h2_min = 0.0;
    var have_h = false;
    for (var e = 0u; e < n; e = e + 1u) {
        let tag = vt[e];
        if (tag < TAG_SEG_FLAG) {
            let dq = seeds[tag] - p;
            let d2 = dot(dq, dq);
            if (!have_h || d2 < h2_min) {
                h2_min = d2;
                have_h = true;
            }
        }
    }
    // Cells with NO bisector edge (near-single-seed regions, wall-enclosed
    // pockets) get a domain-scale h so conditions 3/4 still run (stage-5
    // review): their real box/segment faces are domain scale, so genuine
    // geometry never flags, while a threshold-straddling sliver face does
    // instead of silently bypassing the filter.
    var h_min = min(params.domain_x, params.domain_y);
    if (have_h) {
        h_min = sqrt(h2_min);
    }

    // Emit faces + shoelace area/centroid (seed-relative; interior seeds
    // are strictly inside their cell so every cross term has the same sign;
    // boundary vertex seeds sit ON their ring apex — the shoelace sum is
    // still exact for the convex wedge). Vertices are already canonical
    // (creation-time 2x2 solves), so no re-evaluation pass is needed.
    var signed2 = 0.0;
    var cx = 0.0;
    var cy = 0.0;
    for (var e = 0u; e < n; e = e + 1u) {
        var w = e + 1u;
        if (w == n) {
            w = 0u;
        }
        let v0 = vx[e];
        let v1 = vx[w];
        let cross_t = v0.x * v1.y - v1.x * v0.y;
        signed2 = signed2 + cross_t;
        cx = cx + (v0.x + v1.x) * cross_t;
        cy = cy + (v0.y + v1.y) * cross_t;

        let flen = distance(v0, v1);
        // Filter conditions 3/4: short/untrustworthy faces and
        // geometry-degrading vertex error bounds (h_min falls back to the
        // domain scale for bisector-free rings — see above).
        let vee = length(ve[e]);
        let vew = length(ve[w]);
        if (flen - 4.0 * (vee + vew) <= EPS_SHORT * h_min) {
            unc = unc | 4u;
        }
        if (max(vee, vew) > VE_MAX_REL * h_min) {
            unc = unc | 8u;
        }

        let base = i * K_FACE_MAX + e;
        let tag = vt[e];
        var nbr = NBR_NONE;
        var bc = BC_NONE;
        var nrm = vec2<f32>(0.0, 0.0);
        if (tag >= TAG_BOX_BASE) {
            bc = tag - TAG_BOX_BASE;
            nrm = box_normal(bc);
        } else if ((tag & TAG_SEG_FLAG) != 0u) {
            // Boundary-segment face: bc keeps the full TAG_SEG_FLAG|seg
            // encoding (documented in mod.rs); outward normal = the
            // solid-side segment normal (d.y, -d.x)/|d|.
            bc = tag;
            let s4 = segments[tag & 0x7fffffffu];
            let d = s4.zw - s4.xy;
            nrm = normalize(vec2<f32>(d.y, -d.x));
        } else {
            // Neighbor id is CANONICALIZED (coalesced duplicates report
            // their bin representative) so topology consumers see one id
            // per physical cell regardless of which duplicate's plane
            // happened to survive eps-redundant clipping. The geometry
            // still uses the actual plane seed (sub-pitch identical).
            nbr = canon[tag];
            // Face normal = owner->neighbor bisector direction (the M5
            // face-normal convention).
            nrm = normalize(seeds[tag] - p);
        }
        nbr_ids[base] = nbr;
        face_bc[base] = bc;
        face_geom[base] = vec4<f32>(nrm.x, nrm.y, flen, 0.0);
        face_mid[base] = 0.5 * (v0 + v1);
        ring_vert[base] = v0;
    }
    for (var e = n; e < K_FACE_MAX; e = e + 1u) {
        let base = i * K_FACE_MAX + e;
        nbr_ids[base] = NBR_NONE;
        face_bc[base] = BC_NONE;
        face_geom[base] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        face_mid[base] = vec2<f32>(0.0, 0.0);
        ring_vert[base] = vec2<f32>(0.0, 0.0);
    }

    let signed_area = 0.5 * signed2;
    let area = abs(signed_area);
    var cen = vec2<f32>(0.0, 0.0);
    if (area > 1e-12) {
        cen = vec2<f32>(cx, cy) / (6.0 * signed_area);
    } else {
        // Degenerate-area fallback: vertex average (CPU ring_geometry).
        for (var e = 0u; e < n; e = e + 1u) {
            cen = cen + vx[e];
        }
        cen = cen / f32(max(n, 1u));
    }
    cell_area[i] = area;
    cell_centroid[i] = cen;
    cell_nfaces[i] = n;
    visited_bins[i] = (bins & 0x00ffffffu) | (unc << 24u);

    if (unc != 0u) {
        // Best-known f32 geometry stays in the output slots (the CPU f64
        // fallback overwrites them); the status + flag route the cell to
        // resolve_flagged.
        status_out[i] = STATUS_NEEDS_EXACT;
        let slot = atomicAdd(&flagged[0], 1u);
        if (slot < params.flag_cap) {
            atomicStore(&flagged[1u + slot], i);
        }
    } else {
        status_out[i] = STATUS_SUCCESS;
    }
}
"#;
