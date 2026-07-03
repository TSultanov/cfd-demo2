//! WGSL source for the `voronoi_cell` kernel (stage 1: domain bbox only).
//!
//! One thread per seed; the cell starts as the seed-relative domain bbox and
//! is cut by one half-plane per candidate neighbor, streamed in Chebyshev
//! grid-ring order (see `mod.rs` for the traversal decision) with the
//! security-radius stop `d² > 4·R²`. All arithmetic is seed-relative f32;
//! function-scope arrays follow the shipped `block_precond.wgsl` precedent
//! (~2 KB/thread with dynamic indexing).

use super::{status, BC_NONE, K_FACE_MAX, MAX_VERTS, NBR_NONE};

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
         const STATUS_EMPTY_CELL: u32 = {s_empty}u;\n",
        max_verts = MAX_VERTS,
        k_face_max = K_FACE_MAX,
        nbr_none = NBR_NONE,
        bc_none = BC_NONE,
        s_ok = status::SUCCESS,
        s_vert = status::VERT_OVERFLOW,
        s_face = status::FACE_OVERFLOW,
        s_empty = status::EMPTY_CELL,
    );
    format!("{header}{BODY}")
}

const BODY: &str = r#"
// Edge-tag encoding while clipping: a bisector edge carries the neighbor
// seed id; a domain-bbox edge carries TAG_BOX_BASE + side (0=left, 1=right,
// 2=bottom, 3=top, matching the CPU PlaneTag::Box ids). Real seed ids stay
// far below TAG_BOX_BASE.
const TAG_BOX_BASE: u32 = 0xfffffff0u;
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

// Zero/PAD the padded face slots and scalar outputs of a cell that has no
// usable geometry (empty, overflow — statuses other than SUCCESS).
fn write_empty_outputs(i: u32) {
    for (var s = 0u; s < K_FACE_MAX; s = s + 1u) {
        let base = i * K_FACE_MAX + s;
        nbr_ids[base] = NBR_NONE;
        face_bc[base] = BC_NONE;
        face_geom[base] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        face_mid[base] = vec2<f32>(0.0, 0.0);
    }
    cell_nfaces[i] = 0u;
    cell_area[i] = 0.0;
    cell_centroid[i] = vec2<f32>(0.0, 0.0);
}

struct ClipResult {
    // New vertex count: 0 = cell clipped away, CLIP_OVERFLOW = > MAX_VERTS,
    // n unchanged for a redundant plane.
    n: u32,
    // Updated max squared vertex distance (security radius squared).
    r2: f32,
};

// Sutherland-Hodgman clip of the convex ring by the half-plane
// { x : dot(x, q) - off <= eps } (seed-relative). Mirrors the CPU sh_emit:
// vertices with |s| <= eps count as inside (degenerate verts kept), the
// intersection parameter t = s_u / (s_u - s_w) is clamped to [0, 1], kept
// vertices keep their outgoing-edge tag, an exit intersection starts the
// new plane's edge, a re-entry intersection resumes the original edge.
// A redundant plane (no vertex outside) leaves the ring bitwise untouched.
fn clip_plane(
    pv: ptr<function, array<vec2<f32>, MAX_VERTS>>,
    pt: ptr<function, array<u32, MAX_VERTS>>,
    n: u32,
    r2_in: f32,
    q: vec2<f32>,
    off: f32,
    eps: f32,
    tag: u32,
) -> ClipResult {
    var s: array<f32, MAX_VERTS>;
    var any_out = false;
    var any_in = false;
    for (var e = 0u; e < n; e = e + 1u) {
        let sv = dot((*pv)[e], q) - off;
        s[e] = sv;
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
            r2 = max(r2, dot(v, v));
            m = m + 1u;
        }
        if (u_in != w_in) {
            if (m == MAX_VERTS) {
                return ClipResult(CLIP_OVERFLOW, 0.0);
            }
            let t = clamp(s[e] / (s[e] - s[w]), 0.0, 1.0);
            let vi = (*pv)[e] + t * ((*pv)[w] - (*pv)[e]);
            nx[m] = vi;
            // Leaving the keep-set starts the new plane's edge; re-entering
            // resumes the original edge.
            nt[m] = select((*pt)[e], tag, u_in);
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
    pn: ptr<function, u32>,
    pr2: ptr<function, f32>,
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
        let res = clip_plane(pv, pt, *pn, *pr2, q, 0.5 * q2, sqrt(q2) * params.edge_len_eps, j);
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

// Lower bound (in distance units, conservative up to ~1 ulp) on the distance
// from p to any seed in a grid bin of Chebyshev ring r or beyond — the CPU
// SeedGrid::ring_lower_bound formula. Ring r=0 has bound 0.
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

// Seed-relative plane of an edge tag as (q.x, q.y, off): { x : dot(x,q) = off }.
fn plane_of(tag: u32, p: vec2<f32>) -> vec3<f32> {
    if (tag >= TAG_BOX_BASE) {
        switch (tag - TAG_BOX_BASE) {
            case 0u: { return vec3<f32>(-1.0, 0.0, p.x); }
            case 1u: { return vec3<f32>(1.0, 0.0, params.domain_x - p.x); }
            case 2u: { return vec3<f32>(0.0, -1.0, p.y); }
            default: { return vec3<f32>(0.0, 1.0, params.domain_y - p.y); }
        }
    }
    let q = seeds[tag] - p;
    return vec3<f32>(q.x, q.y, 0.5 * dot(q, q));
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
        return;
    }
    let p = seeds[i];

    // Seed-relative CCW domain bbox: verts (x0,y0)(x1,y0)(x1,y1)(x0,y1),
    // edge tags bottom/right/top/left = Box(2)/Box(1)/Box(3)/Box(0) — the
    // CPU bbox_ring layout.
    let x0 = -p.x;
    let y0 = -p.y;
    let x1 = params.domain_x - p.x;
    let y1 = params.domain_y - p.y;
    var vx: array<vec2<f32>, MAX_VERTS>;
    var vt: array<u32, MAX_VERTS>;
    vx[0] = vec2<f32>(x0, y0);
    vx[1] = vec2<f32>(x1, y0);
    vx[2] = vec2<f32>(x1, y1);
    vx[3] = vec2<f32>(x0, y1);
    vt[0] = TAG_BOX_BASE + 2u;
    vt[1] = TAG_BOX_BASE + 1u;
    vt[2] = TAG_BOX_BASE + 3u;
    vt[3] = TAG_BOX_BASE + 0u;
    var n = 4u;
    var r2 = 0.0;
    for (var e = 0u; e < 4u; e = e + 1u) {
        r2 = max(r2, dot(vx[e], vx[e]));
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

    var st = 0u;
    // Chebyshev ring sweep with the security-radius stop. Completing the
    // sweep without the stop means every seed in the grid was clipped —
    // exhaustive, therefore exact (SUCCESS, matching CPU semantics).
    for (var r = 0u; r <= r_max; r = r + 1u) {
        if (r > 0u) {
            let lb = ring_lower_bound(p, bx, by, r);
            if (lb * lb > 4.0 * r2) {
                break;
            }
        }
        if (r == 0u) {
            st = process_bin(by * gw + bx, ci, p, &vx, &vt, &n, &r2);
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
                    st = process_bin(u32(gy) * gw + u32(gx), ci, p, &vx, &vt, &n, &r2);
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
                    st = process_bin(u32(gy) * gw + u32(gx), ci, p, &vx, &vt, &n, &r2);
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

    if (st == 0u && n > K_FACE_MAX) {
        st = STATUS_FACE_OVERFLOW;
    }
    if (st != 0u) {
        write_empty_outputs(i);
        status_out[i] = st;
        // The ONLY atomic in the kernel: append to the flag list. Order is
        // nondeterministic; ids are sorted on the CPU before use.
        let slot = atomicAdd(&flagged[0], 1u);
        if (slot < params.flag_cap) {
            atomicStore(&flagged[1u + slot], i);
        }
        return;
    }

    // Canonical vertex re-evaluation (the M0 assembly trick): the clipped
    // vertex chain carries ~domain·eps_f32 absolute error from the early
    // bbox-scale clips, which is ~1e-5·h relative at 30k seeds — too coarse
    // for the geometry parity gates. Each final vertex is the intersection
    // of its two adjacent edge planes, whose seed-relative coefficients are
    // h-scale, so re-solving the 2x2 system recovers ~eps_f32-relative
    // accuracy. Near-parallel plane pairs (sliver vertices) keep the
    // clipped position.
    var rx: array<vec2<f32>, K_FACE_MAX>;
    for (var e = 0u; e < n; e = e + 1u) {
        var prev = n - 1u;
        if (e > 0u) {
            prev = e - 1u;
        }
        let pa = plane_of(vt[prev], p);
        let pb = plane_of(vt[e], p);
        let det = pa.x * pb.y - pa.y * pb.x;
        let scale = length(pa.xy) * length(pb.xy);
        if (abs(det) > 1e-6 * scale) {
            rx[e] = vec2<f32>(
                (pa.z * pb.y - pa.y * pb.z) / det,
                (pa.x * pb.z - pa.z * pb.x) / det,
            );
        } else {
            rx[e] = vx[e];
        }
    }

    // Emit faces + shoelace area/centroid (seed-relative; the seed is
    // strictly inside its cell, so every cross term has the same sign — no
    // cancellation).
    var signed2 = 0.0;
    var cx = 0.0;
    var cy = 0.0;
    for (var e = 0u; e < n; e = e + 1u) {
        var w = e + 1u;
        if (w == n) {
            w = 0u;
        }
        let v0 = rx[e];
        let v1 = rx[w];
        let cross_t = v0.x * v1.y - v1.x * v0.y;
        signed2 = signed2 + cross_t;
        cx = cx + (v0.x + v1.x) * cross_t;
        cy = cy + (v0.y + v1.y) * cross_t;

        let base = i * K_FACE_MAX + e;
        let tag = vt[e];
        var nbr = NBR_NONE;
        var bc = BC_NONE;
        var nrm = vec2<f32>(0.0, 0.0);
        if (tag >= TAG_BOX_BASE) {
            bc = tag - TAG_BOX_BASE;
            nrm = box_normal(bc);
        } else {
            nbr = tag;
            // Face normal = owner->neighbor bisector direction (the M5
            // face-normal convention).
            nrm = normalize(seeds[tag] - p);
        }
        nbr_ids[base] = nbr;
        face_bc[base] = bc;
        face_geom[base] = vec4<f32>(nrm.x, nrm.y, distance(v0, v1), 0.0);
        face_mid[base] = 0.5 * (v0 + v1);
    }
    for (var e = n; e < K_FACE_MAX; e = e + 1u) {
        let base = i * K_FACE_MAX + e;
        nbr_ids[base] = NBR_NONE;
        face_bc[base] = BC_NONE;
        face_geom[base] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        face_mid[base] = vec2<f32>(0.0, 0.0);
    }

    let signed_area = 0.5 * signed2;
    let area = abs(signed_area);
    var cen = vec2<f32>(0.0, 0.0);
    if (area > 1e-12) {
        cen = vec2<f32>(cx, cy) / (6.0 * signed_area);
    } else {
        // Degenerate-area fallback: vertex average (CPU ring_geometry).
        for (var e = 0u; e < n; e = e + 1u) {
            cen = cen + rx[e];
        }
        cen = cen / f32(max(n, 1u));
    }
    cell_area[i] = area;
    cell_centroid[i] = cen;
    cell_nfaces[i] = n;
    status_out[i] = STATUS_SUCCESS;
}
"#;
