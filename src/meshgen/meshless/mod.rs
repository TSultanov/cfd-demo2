//! Meshless Voronoi engine (roadmap M0): the mesh *is* the seed set; each
//! Voronoi cell is computed independently — kNN candidate search
//! (`seed_grid`) + half-plane clipping with a security-radius stop (`clip`)
//! — per Ray/Sokolov/Lefebvre/Lévy, "Meshless Voronoi on the GPU" (ACM TOG
//! 2018), adapted to 2D. Purely additive alongside the incumbent generators;
//! nothing in the static pipeline changes.
//!
//! Invariants (stronger than the incumbent Voronoi path):
//! - **Seed i == cell i.** Diagram slot `i` is a pure function of the input;
//!   no compaction, no cell splitting, statuses instead of failures.
//! - **Exactness.** The kNN + security-radius engine is an accelerator, not
//!   an approximation: every cell is bit-identical to clipping against *all*
//!   other seeds (`compute_cell_exhaustive` is the in-tree oracle).
//! - **Determinism.** Per-cell writes go to disjoint padded slots, neighbor
//!   ties break on id, and there is no shared mutable state — output is
//!   byte-identical for any rayon thread count by construction.
//!
//! Stage coverage: this file + `seed_grid` + `clip` implement M0.1/M0.2;
//! `boundary` adds M0.3 — boundary loops, the review-F1 seeding protocol
//! (vertex seeds at convex fluid corners, equidistant guard seeds around
//! reflex ones) and own-segment-line clipping for `SeedKind::Boundary`
//! seeds; `assemble` adds M0.4 — tag-canonical `Mesh` assembly and the
//! `generate_meshless_voronoi_mesh` entry point; `lloyd` adds M0.5 —
//! Lloyd/CVT relaxation and the `generate_cvt_mesh` entry point.

mod assemble;
mod boundary;
mod clip;
mod lloyd;
mod seed_grid;

pub use assemble::{assemble_mesh, generate_meshless_voronoi_mesh};
pub use lloyd::{generate_cvt_mesh, lloyd_relax, LloydConfig, LloydStats};
pub use boundary::{
    boundary_seeds, circle_loop, distance_to_loops, meshless_seed_points, point_in_fluid,
    polyline_loop, shielding_violations, tag_boundary_type, BoundaryLoop, BoundarySpec, SeedKind,
    SegId,
};
pub use clip::MAX_CLIP_VERTS;
pub use seed_grid::SeedGrid;

use clip::{drive, Attempt, CellRing, ClipOutcome, ClipPoly, ClipPolyVec, HalfPlane};
use seed_grid::dist2;

use super::tolerances::MeshgenTolerances;
use nalgebra::{Point2, Vector2};

/// Engine tuning knobs. `k` is the initial candidate count; uncertified
/// cells retry with `k` doubled up to `k_max`, then stream every remaining
/// seed in distance order (the CPU engine never fails).
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub k: usize,
    pub k_max: usize,
    /// Ring stride of the padded diagram; reserved knob — the v1 fast path
    /// is compiled at `MAX_CLIP_VERTS` and this must equal it.
    pub max_ring: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            k: 16,
            k_max: 64,
            max_ring: MAX_CLIP_VERTS,
        }
    }
}

/// Per-cell outcome. Statuses, not failures: flagged cells still carry their
/// best-known geometry, and `RingOverflow` rings live in the diagram's
/// `overflow` spill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellStatus {
    Ok,
    /// Certified after `n` k-doublings (the exhaustive fallback counts as
    /// one more doubling). Telemetry that feeds the M1 GPU k choice.
    OkEscalated(u8),
    /// Even the exhaustive pass failed to certify — impossible on the CPU
    /// path (an exhaustive clip is exact by construction); reserved for
    /// genuinely broken inputs.
    SecurityRadiusFailed,
    /// Final ring exceeds `MAX_CLIP_VERTS`; the ring is in
    /// `MeshlessDiagram::overflow` and the padded slot is empty.
    RingOverflow,
    /// The cell was clipped away entirely (input error, e.g. a seed outside
    /// the domain).
    EmptyCell,
}

/// Identity of the plane that created a ring edge. The per-cell tag sequence
/// doubles as the (padded) neighbor list: `Bisector` entries in ring order
/// are exactly the cell's Voronoi neighbors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlaneTag {
    /// Bisector against seed `j`.
    Bisector(u32),
    /// Boundary segment (global `SegId` into the input's `BoundarySpec`).
    Boundary(u32),
    /// Domain bbox side: 0=left(Inlet) 1=right(Outlet) 2=bottom(Wall) 3=top(Wall).
    Box(u8),
}

impl PlaneTag {
    /// Filler for unused padded slots (`Bisector(u32::MAX)` — a real seed id
    /// would need 4 billion seeds).
    pub const PAD: PlaneTag = PlaneTag::Bisector(u32::MAX);
}

/// Engine input. `seeds` must lie inside `[0, domain.x] × [0, domain.y]` and
/// be pairwise distinct; `kinds` may be empty, meaning every seed is
/// `Interior`.
pub struct MeshlessInput<'a> {
    pub seeds: &'a [Point2<f64>],
    pub kinds: &'a [SeedKind],
    pub boundary: &'a BoundarySpec,
    /// Clip bbox `[0, x] × [0, y]`.
    pub domain: Vector2<f64>,
    pub tol: &'a MeshgenTolerances,
    pub cfg: EngineConfig,
}

impl<'a> MeshlessInput<'a> {
    /// Stage-1 constructor: no boundary loops, every seed `Interior`.
    pub fn interior_only(
        seeds: &'a [Point2<f64>],
        boundary: &'a BoundarySpec,
        domain: Vector2<f64>,
        tol: &'a MeshgenTolerances,
        cfg: EngineConfig,
    ) -> Self {
        Self {
            seeds,
            kinds: &[],
            boundary,
            domain,
            tol,
            cfg,
        }
    }

    /// Kind of seed `i` (empty `kinds` ⇒ all `Interior`).
    #[inline]
    pub fn kind(&self, i: usize) -> SeedKind {
        if self.kinds.is_empty() {
            SeedKind::Interior
        } else {
            self.kinds[i]
        }
    }
}

/// The whole diagram as a padded SoA (stride `MAX_CLIP_VERTS`): fixed-size
/// disjoint slots per cell make parallel writes deterministic and are the
/// exact shape the M1 GPU port produces. Ring vertices are CCW in absolute
/// (not seed-relative) f64 coordinates; edge `e` of cell `i` runs
/// `ring_xy[i*M + e] -> ring_xy[i*M + (e+1) % ring_len[i]]` and was created
/// by `ring_plane[i*M + e]`.
#[derive(Clone, Debug)]
pub struct MeshlessDiagram {
    pub n: usize,
    pub status: Vec<CellStatus>,
    /// `n * MAX_CLIP_VERTS`, padded with `[0.0, 0.0]`.
    pub ring_xy: Vec<[f64; 2]>,
    /// `n * MAX_CLIP_VERTS`, padded with `PlaneTag::PAD`.
    pub ring_plane: Vec<PlaneTag>,
    /// Ring vertex count per cell (0 for `EmptyCell`/`RingOverflow` slots).
    pub ring_len: Vec<u8>,
    pub centroid: Vec<[f64; 2]>,
    pub area: Vec<f64>,
    /// Slow-path spill for `RingOverflow` cells (rare): `(cell, ring)` in
    /// ascending cell order, absolute CCW coordinates like `ring_xy`.
    pub overflow: Vec<(u32, Vec<([f64; 2], PlaneTag)>)>,
}

impl MeshlessDiagram {
    /// Status census `(ok, escalated, overflow, empty, failed)` — the
    /// escalation-rate telemetry the M0 gates report.
    pub fn status_counts(&self) -> (usize, usize, usize, usize, usize) {
        let mut counts = (0, 0, 0, 0, 0);
        for s in &self.status {
            match s {
                CellStatus::Ok => counts.0 += 1,
                CellStatus::OkEscalated(_) => counts.1 += 1,
                CellStatus::RingOverflow => counts.2 += 1,
                CellStatus::EmptyCell => counts.3 += 1,
                CellStatus::SecurityRadiusFailed => counts.4 += 1,
            }
        }
        counts
    }
}

/// One computed cell — the pure per-cell result `build_diagram` scatters
/// into the diagram, and the entry point the M1 GPU fallback recomputes
/// flagged cells through. Ring coordinates are absolute; `len == 0` with
/// `spill = Some(..)` for rings beyond `MAX_CLIP_VERTS`.
pub struct CellOut {
    pub status: CellStatus,
    pub len: usize,
    pub xy: [[f64; 2]; MAX_CLIP_VERTS],
    pub plane: [PlaneTag; MAX_CLIP_VERTS],
    pub centroid: [f64; 2],
    pub area: f64,
    pub spill: Option<Vec<([f64; 2], PlaneTag)>>,
}

impl CellOut {
    fn empty(seed: Point2<f64>) -> Self {
        Self {
            status: CellStatus::EmptyCell,
            len: 0,
            xy: [[0.0; 2]; MAX_CLIP_VERTS],
            plane: [PlaneTag::PAD; MAX_CLIP_VERTS],
            centroid: [seed.x, seed.y],
            area: 0.0,
            spill: None,
        }
    }

    /// Translate the finished (seed-relative) ring to absolute coordinates
    /// and finalize area/centroid. Oversized rings spill and the status
    /// upgrades to `RingOverflow` regardless of how the clip went.
    fn from_ring<R: CellRing>(ring: &R, seed: Point2<f64>, status: CellStatus) -> Self {
        let n = ring.len();
        let (area, centroid) = clip::ring_geometry(ring, seed);
        let mut out = Self {
            status,
            len: 0,
            xy: [[0.0; 2]; MAX_CLIP_VERTS],
            plane: [PlaneTag::PAD; MAX_CLIP_VERTS],
            centroid,
            area,
            spill: None,
        };
        if n <= MAX_CLIP_VERTS {
            for e in 0..n {
                let (x, y) = ring.vert(e);
                out.xy[e] = [x + seed.x, y + seed.y];
                out.plane[e] = ring.tag(e);
            }
            out.len = n;
        } else {
            let mut spill = Vec::with_capacity(n);
            for e in 0..n {
                let (x, y) = ring.vert(e);
                spill.push(([x + seed.x, y + seed.y], ring.tag(e)));
            }
            out.status = CellStatus::RingOverflow;
            out.spill = Some(spill);
        }
        out
    }
}

#[inline]
fn escalated_status(doublings: u8) -> CellStatus {
    if doublings == 0 {
        CellStatus::Ok
    } else {
        CellStatus::OkEscalated(doublings)
    }
}

/// The seed's own boundary-segment half-planes: the lines through the
/// segments the seed sits on, clipped *first* (they pass through the seed —
/// distance 0, so they precede every bisector in distance order). One plane
/// for a mid-segment guard seed (`seg_prev == seg_next`), two for a vertex
/// seed. Pure function of the input ⇒ determinism is unaffected.
fn own_planes(input: &MeshlessInput, i: usize) -> [Option<HalfPlane>; 2] {
    match input.kind(i) {
        SeedKind::Interior => [None, None],
        SeedKind::Boundary { seg_prev, seg_next } => {
            let p = input.seeds[i];
            let eps = input.tol.edge_len_eps;
            let (a, b) = input.boundary.segment_points(seg_prev);
            let first = HalfPlane::segment_line(a, b, p, seg_prev, eps);
            let second = if seg_next != seg_prev {
                let (a, b) = input.boundary.segment_points(seg_next);
                Some(HalfPlane::segment_line(a, b, p, seg_next, eps))
            } else {
                None
            };
            [Some(first), second]
        }
    }
}

/// Apply the own-segment planes to a fresh bbox ring. Returns `false` if the
/// cell was clipped away entirely (a seed on the wrong side of its own wall
/// — broken input).
fn clip_own<R: CellRing>(ring: &mut R, own: &[Option<HalfPlane>; 2]) -> bool {
    for hp in own.iter().flatten() {
        match ring.clip(hp) {
            ClipOutcome::Empty => return false,
            // 4 bbox verts + at most one net vertex per clip stays far
            // below MAX_CLIP_VERTS.
            ClipOutcome::Overflow => unreachable!("own-segment clips cannot overflow a bbox ring"),
            ClipOutcome::Redundant | ClipOutcome::Cut => {}
        }
    }
    true
}

/// Compute cell `i` — pure function of `(input, i)`; `grid` is just the
/// accelerator index over `input.seeds`. Escalation (k-doubling up to
/// `k_max`, then exhaustive) happens per cell and functionally, so thread
/// count cannot affect the result. `SeedKind::Boundary` seeds clip their
/// own segment lines before the bisector loop (both re-applied on every
/// escalation attempt, since the ring restarts from the bbox).
pub fn compute_cell(input: &MeshlessInput, grid: &SeedGrid, i: usize) -> CellOut {
    let seeds = input.seeds;
    let p = seeds[i];
    let eps = input.tol.edge_len_eps;
    let own = own_planes(input, i);
    let max_nb = seeds.len() - 1;
    let mut k = input.cfg.k.max(1);
    let mut doublings: u8 = 0;
    let mut use_slow = false;
    let mut nbrs: Vec<(f64, u32)> = Vec::new();
    loop {
        let exhaustive = k >= max_nb;
        grid.knn(seeds, i as u32, k, &mut nbrs);

        let attempt = if use_slow {
            let mut ring = ClipPolyVec::from_bbox(p, input.domain);
            if !clip_own(&mut ring, &own) {
                return CellOut::empty(p);
            }
            let a = drive(&mut ring, p, seeds, &nbrs, eps);
            match a {
                Attempt::Certified => {
                    return CellOut::from_ring(&ring, p, escalated_status(doublings))
                }
                Attempt::Uncertified if exhaustive => {
                    // The list covered every other seed: the clip is exact
                    // even though the security radius never certified it.
                    return CellOut::from_ring(&ring, p, escalated_status(doublings));
                }
                Attempt::Overflow => unreachable!("Vec-backed ring cannot overflow"),
                _ => a,
            }
        } else {
            let mut ring = ClipPoly::from_bbox(p, input.domain);
            if !clip_own(&mut ring, &own) {
                return CellOut::empty(p);
            }
            let a = drive(&mut ring, p, seeds, &nbrs, eps);
            match a {
                Attempt::Certified => {
                    return CellOut::from_ring(&ring, p, escalated_status(doublings))
                }
                Attempt::Uncertified if exhaustive => {
                    return CellOut::from_ring(&ring, p, escalated_status(doublings));
                }
                _ => a,
            }
        };

        match attempt {
            Attempt::Empty => return CellOut::empty(p),
            Attempt::Overflow => {
                // Retry the same neighbor list with the heap-backed ring
                // (same arithmetic ⇒ same bits); stay there for any further
                // escalations of this cell.
                use_slow = true;
            }
            Attempt::Uncertified => {
                doublings = doublings.saturating_add(1);
                k = if k * 2 <= input.cfg.k_max { k * 2 } else { max_nb };
            }
            Attempt::Certified => unreachable!("handled above"),
        }
    }
}

/// Brute-force oracle: clip cell `i` against its own segment lines (if any)
/// and then **every** other seed's bisector in ascending `(d², id)` order —
/// no kNN, no security radius, same clip arithmetic. The engine must
/// reproduce this bit-for-bit (the accelerator is exactness-preserving, not
/// approximate); kept public for the test gates and the M1 GPU-parity
/// harness.
pub fn compute_cell_exhaustive(input: &MeshlessInput, i: usize) -> CellOut {
    let seeds = input.seeds;
    let p = seeds[i];
    let mut nbrs: Vec<(f64, u32)> = (0..seeds.len())
        .filter(|&j| j != i)
        .map(|j| (dist2(p, seeds[j]), j as u32))
        .collect();
    nbrs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut ring = ClipPolyVec::from_bbox(p, input.domain);
    if !clip_own(&mut ring, &own_planes(input, i)) {
        return CellOut::empty(p);
    }
    for &(_, j) in &nbrs {
        let q = seeds[j as usize] - p;
        if ring.clip(&HalfPlane::bisector(q, j, input.tol.edge_len_eps)) == ClipOutcome::Empty {
            return CellOut::empty(p);
        }
    }
    CellOut::from_ring(&ring, p, CellStatus::Ok)
}

/// Build the full diagram: rayon over cells, each writing its own disjoint
/// padded slot — byte-identical output for any thread count by construction.
/// The rare `RingOverflow` spills are collected in a serial second pass (in
/// cell order) to keep the parallel loop free of shared state.
pub fn build_diagram(input: &MeshlessInput) -> MeshlessDiagram {
    let n = input.seeds.len();
    assert!(
        input.kinds.is_empty() || input.kinds.len() == n,
        "kinds must be empty or one per seed"
    );
    let m = MAX_CLIP_VERTS;
    let mut d = MeshlessDiagram {
        n,
        status: vec![CellStatus::Ok; n],
        ring_xy: vec![[0.0; 2]; n * m],
        ring_plane: vec![PlaneTag::PAD; n * m],
        ring_len: vec![0u8; n],
        centroid: vec![[0.0; 2]; n],
        area: vec![0.0; n],
        overflow: Vec::new(),
    };
    if n == 0 {
        return d;
    }
    let grid = SeedGrid::build(input.seeds, input.domain);

    {
        use rayon::prelude::*;
        let MeshlessDiagram {
            status,
            ring_xy,
            ring_plane,
            ring_len,
            centroid,
            area,
            ..
        } = &mut d;
        status
            .par_iter_mut()
            .zip(ring_len.par_iter_mut())
            .zip(centroid.par_iter_mut())
            .zip(area.par_iter_mut())
            .zip(ring_xy.par_chunks_mut(m))
            .zip(ring_plane.par_chunks_mut(m))
            .enumerate()
            .with_min_len(1024)
            .for_each(|(i, (((((st, len), ce), ar), rxy), rpl))| {
                let out = compute_cell(input, &grid, i);
                *st = out.status;
                *len = out.len as u8;
                *ce = out.centroid;
                *ar = out.area;
                rxy[..out.len].copy_from_slice(&out.xy[..out.len]);
                rpl[..out.len].copy_from_slice(&out.plane[..out.len]);
            });
    }

    for i in 0..n {
        if d.status[i] == CellStatus::RingOverflow {
            let out = compute_cell(input, &grid, i);
            let spill = out
                .spill
                .expect("RingOverflow cell must produce a spill ring");
            d.overflow.push((i as u32, spill));
        }
    }

    d
}
