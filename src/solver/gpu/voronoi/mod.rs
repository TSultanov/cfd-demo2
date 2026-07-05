//! GPU meshless Voronoi engine: a WGSL port of the CPU engine
//! (`crate::meshgen::meshless`) — one thread per seed computes its Voronoi
//! cell by half-plane clipping with a security-radius stop, adapted to 2D.
//! Runs outside the solver's step graph: WGSL string literal, manual bind
//! group layouts, own encoder + submit.
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
//! ## Traversal: streaming ring clip (not kNN-then-clip)
//!
//! The kernel clips candidates directly while walking Chebyshev grid rings
//! (bins in `for_each_ring_bin` order, ids ascending within each bin — the
//! counting sort guarantees that for free), stopping when the next ring's
//! distance lower bound exceeds `2R`. The final clipped polygon is
//! order-independent up to f32 rounding, so we do not reproduce the CPU's
//! kNN ascending-(d², id) order; streaming clip leaves exactly two overflow
//! classes (`MAX_VERTS`, `K_FACE_MAX`) and its traversal order is fully
//! deterministic on a fixed device (byte-stable run-to-run).
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
//! All clip arithmetic is seed-relative f32. Bisector planes are bitwise
//! symmetric between threads i and j WITHOUT an explicit (min, max) ordering
//! because IEEE-754 subtraction is sign-symmetric:
//! `fl(p_j − p_i) == −fl(p_i − p_j)` bit-for-bit, and the offset `0.5·|q|²`
//! and tolerance `|q|·eps` depend only on componentwise magnitudes, so the
//! two threads evaluate the exactly-negated coefficients of the identical
//! geometric line. Cell centroids and face midpoints are stored SEED-RELATIVE
//! (add the seed position to get absolute coordinates): at 30k+ seeds the
//! absolute-f32 quantum (~1e-7 at x≈2) is larger than the 1e-5·h parity
//! tolerance, while relative coordinates are O(h) with ~1e-10 ulps.

mod cell_geom;
mod csr_gpu;
mod derive;
mod emit;
mod engine;
mod lloyd;
mod regen;
mod scan;
mod solver_mesh;
mod swept_gpu;
mod wgsl;

pub use cell_geom::{CellGeometry, GpuCellGeometry};
pub use regen::{GpuMeshRegen, GpuRegenResult};
pub use solver_mesh::assemble_solver_mesh;
pub use csr_gpu::{GpuCsr, GpuCsrArrays};
pub use derive::{DeriveFaces, DerivedFaceOffsets, ScanOffsets};
pub use emit::{EmitFaces, GpuFaceGeometry};
pub use engine::{
    boundary_spec_f32, GpuVoronoiCells, GpuVoronoiEngine, VoronoiResolveReport,
};
pub use scan::{GpuScan, ELEMS_PER_BLOCK, MAX_SCAN_ELEMS};
pub use swept_gpu::{GpuSweptAreas, SweptFluxGeometry};

/// Max clip-polygon vertices per cell (intermediate ring). 2D Voronoi cells
/// of Poisson-disk sets average 6 vertices with tails under 12; the bbox
/// start ring has 4. 2× margin; overflow is a status, never UB.
pub const MAX_VERTS: usize = 24;

/// Padded output faces per cell (final ring size). Overflow is a status.
pub const K_FACE_MAX: usize = 16;

/// Per-cell status codes written to `b_status`.
pub mod status {
    /// Cell certified by the security radius, or clipped against every seed
    /// in the grid (exhaustive ⇒ exact).
    pub const SUCCESS: u32 = 0;
    /// Reserved: with a CPU-built full-coverage grid the ring sweep always
    /// either certifies or goes exhaustive, so this cannot fire.
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
/// `SeedKind::Boundary` seeds are ALWAYS fixed (keyed off the kind table);
/// this flag additionally pins interior seeds.
pub const SEED_FLAG_FIXED: u32 = 1;
