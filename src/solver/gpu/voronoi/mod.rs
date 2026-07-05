//! GPU meshless Voronoi engine (roadmap M1): a WGSL port of the M0 CPU
//! engine (`crate::meshgen::meshless`) — one thread per seed computes its
//! Voronoi cell by half-plane clipping with a security-radius stop, per
//! Ray/Sokolov/Lefebvre/Lévy, "Meshless Voronoi on the GPU" (ACM TOG 2018),
//! adapted to 2D. Cloned structurally from the `srd.rs` hand-written-kernel
//! seam: WGSL string literal, manual bind group layouts, own encoder +
//! submit, outside the solver's step graph.
//!
//! Stage 3 scope (this module version): full boundary support on top of the
//! stage-2 filter/fallback machinery — flattened boundary-segment table +
//! per-seed `SeedKind` upload, `Boundary`-kind seeds clipping their own
//! segment line(s) FIRST (the M0 `own_planes`/`clip_own` order),
//! `Boundary`/`Box` tag equivalents in the output slots (see the encoding
//! below), CPU-built `SeedGrid` uploaded as CSR `u32` buffers, conservative
//! epsilon filter (`NEEDS_EXACT`, see `wgsl.rs` docs), CPU f64 fallback
//! (`resolve_flagged`: recompute flagged cells via M0 `compute_cell` **on
//! the f32-rounded seeds and segments** — review F4 — and patch via
//! `write_buffer`), unconditional (release-mode) reciprocity enforcement
//! over the merged diagram, and `read_diagram` — the ring readback bridge
//! into the M0 `assemble_mesh` for an end-to-end GPU-diagram `Mesh`.
//!
//! ## Output tag encoding (`b_nbr_ids` / `b_face_bc`)
//!
//! Face slot tags mirror the CPU `PlaneTag` in u32 space:
//!
//! - interior face: `b_nbr_ids` = the (coalescing-canonicalized) neighbor
//!   seed id, `b_face_bc` = `BC_NONE`;
//! - domain-bbox face: `b_nbr_ids` = `NBR_NONE`, `b_face_bc` = side id
//!   `< 4` (0=left, 1=right, 2=bottom, 3=top — `PlaneTag::Box`);
//! - boundary-segment face: `b_nbr_ids` = `NBR_NONE`, `b_face_bc` =
//!   `BC_SEG_FLAG | seg` (`PlaneTag::Boundary(seg)`, high bit set; real
//!   segment counts stay far below 2³¹ and `BC_NONE` is reserved);
//! - unused slot: `NBR_NONE` / `BC_NONE`.
//!
//! ## Traversal decision: streaming ring clip (not kNN-then-clip)
//!
//! The kernel clips candidates directly while walking Chebyshev grid rings
//! (bins in the CPU `for_each_ring_bin` order, ids ascending within each bin
//! — the CPU-built counting sort guarantees that for free), stopping when
//! the next ring's distance lower bound exceeds `2R`. We deliberately do NOT
//! reproduce M0's kNN ascending-(d², id) clip order: the final clipped
//! polygon is order-independent up to f32 rounding, and the M1 parity gates
//! compare *eps_face-filtered neighbor sets* and geometry tolerances — never
//! bits — so matching the CPU order buys nothing for parity triage while
//! costing a k-array, an in-register distance sort, and a "k too small"
//! escalation/failure mode. Streaming clip leaves exactly two overflow
//! classes (`MAX_VERTS`, `K_FACE_MAX`), and its traversal order is fully
//! deterministic on a fixed device, which is the load-bearing property
//! (byte-stable run-to-run). This choice is kept for all M1 stages.
//!
//! ## Determinism
//!
//! Per-cell writes go to disjoint padded slots; the clip order is pinned by
//! (ring, bin, in-bin id) traversal; the only atomic is the flag-list append
//! (`b_flagged[0]`), which perturbs only the flag-list order — consumers
//! sort the ids on the CPU before acting on them.
//!
//! ## Precision layout
//!
//! All clip arithmetic is seed-relative f32 (the M0 precision trick).
//! Bisector planes are bitwise symmetric between threads i and j WITHOUT an
//! explicit (min, max) ordering because IEEE-754 subtraction is
//! sign-symmetric: `fl(p_j − p_i) == −fl(p_i − p_j)` bit-for-bit, and the
//! offset `0.5·|q|²` and tolerance `|q|·eps` depend only on componentwise
//! magnitudes, so the two threads evaluate the exactly-negated coefficients
//! of the identical geometric line — the design §5.2 rule, obtained
//! structurally. Cell centroids and face midpoints are stored SEED-RELATIVE
//! (add the seed position to get absolute coordinates): at 30k+ seeds the
//! absolute-f32 quantum (~1e-7 at x≈2) is larger than the 1e-5·h parity
//! tolerance, while relative coordinates are O(h) with ~1e-10 ulps.

//!
//! ## Stage 4: GPU Lloyd/CVT relaxation (lloyd.rs)
//!
//! `lloyd_update` moves every non-fixed seed to the ρ = h⁻ᵉˣᵖ density-
//! weighted centroid of its cell (the exact M0 `weighted_centroid` fan
//! quadrature; sizing field = CPU-sampled bilerp grid), chained with full
//! regens in one encoder and no readback; a two-pass max-displacement
//! reduce feeds both the standalone convergence check (one tiny readback)
//! and the grid-staleness slack that keeps the stale CPU `SeedGrid`'s
//! security stop conservative across chained iterations (lloyd.rs docs).

mod derive;
mod engine;
mod lloyd;
mod scan;
mod wgsl;

pub use derive::{DeriveFaces, DerivedFaceOffsets};
pub use engine::{
    boundary_spec_f32, GpuVoronoiCells, GpuVoronoiEngine, VoronoiResolveReport,
};
pub use scan::{GpuScan, ELEMS_PER_BLOCK, MAX_SCAN_ELEMS};

/// Max clip-polygon vertices per cell (intermediate ring). 2D Voronoi cells
/// of Poisson-disk sets average 6 vertices with tails under 12; the bbox
/// start ring has 4. 2× margin; overflow is a status, never UB.
pub const MAX_VERTS: usize = 24;

/// Padded output faces per cell (final ring size). Overflow is a status.
pub const K_FACE_MAX: usize = 16;

/// Per-cell status codes written to `b_status` (mirrors design §5.3).
pub mod status {
    /// Cell certified by the security radius, or clipped against every seed
    /// in the grid (exhaustive ⇒ exact).
    pub const SUCCESS: u32 = 0;
    /// Reserved: with a CPU-built full-coverage grid the ring sweep always
    /// either certifies or goes exhaustive, so this cannot fire in v1.
    pub const SECURITY_RADIUS_NOT_REACHED: u32 = 1;
    /// Clip ring exceeded `MAX_VERTS`.
    pub const VERT_OVERFLOW: u32 = 2;
    /// Final ring exceeded `K_FACE_MAX` faces.
    pub const FACE_OVERFLOW: u32 = 3;
    /// Reserved: CPU-built CSR bins have no capacity limit (GPU counting
    /// sort would use it).
    pub const GRID_OVERFLOW: u32 = 4;
    /// Conservative epsilon filter fired (f32 cannot certify agreement
    /// with the f64 oracle — see `wgsl.rs` docs for the conditions). The
    /// cell's slots still hold its best-known f32 geometry;
    /// `resolve_flagged` recomputes it in f64 and patches.
    pub const NEEDS_EXACT: u32 = 5;
    /// Reserved for boundary-segment failures that are not expressible as
    /// EMPTY/overflow (none exist in the current kernel: a seed on the
    /// wrong side of its own wall clips to empty and flags).
    pub const BOUNDARY_ERROR: u32 = 6;
    /// Coalesced duplicate (shares a `quantize_point` bin with a
    /// lower-index seed — the CPU `CellStatus::EmptyCell` rule) or the cell
    /// was clipped away entirely (broken input).
    pub const EMPTY_CELL: u32 = 7;
}

/// `b_nbr_ids` sentinel: boundary face (see `b_face_bc`) or unused slot.
pub const NBR_NONE: u32 = u32::MAX;

/// `b_face_bc` sentinel for unused slots / interior faces (see the tag
/// encoding in the module docs for the boundary-face values).
pub const BC_NONE: u32 = u32::MAX;

/// High bit of a `b_face_bc` value marking a boundary-SEGMENT face: the
/// low 31 bits are the global `SegId` into the uploaded `BoundarySpec`
/// (`PlaneTag::Boundary`). Values `< 4` are bbox sides; `BC_NONE` (all
/// ones) stays reserved for unused slots.
pub const BC_SEG_FLAG: u32 = 0x8000_0000;

/// Per-seed kind-table sentinel (`b_seed_kind`): no own segment, i.e.
/// `SeedKind::Interior`.
pub const SEG_NONE: u32 = u32::MAX;

/// `b_seed_flags` bit: the seed is pinned — `lloyd_update` never moves it.
/// `SeedKind::Boundary` seeds are ALWAYS fixed (keyed off the kind table,
/// matching M0 `lloyd_relax`); this flag additionally pins interior seeds.
pub const SEED_FLAG_FIXED: u32 = 1;
