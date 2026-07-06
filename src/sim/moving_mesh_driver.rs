//! The moving-mesh loop orchestrator.
//!
//! [`MovingMeshDriver`] wraps a [`SolverDriver`] and drives the per-step
//! moving-mesh cycle around it WITHOUT touching `SolverDriver::step`, so
//! static-mesh users are unaffected. One `step` is:
//!
//! 1. **dt handshake**: pin a fixed dt — the configured `requested_dt`, capped
//!    by the mesh-motion CFL `dt ≤ cfl_mesh · min_h / max|w|` — BEFORE the swept
//!    fluxes are closed, so the fluxes and the step march with exactly one dt
//!    (an adaptive re-scale after the closure would silently break the GCL;
//!    `adaptive_dt` is rejected).
//! 2. **advect seeds** per [`MeshMotionSpec`] (f64). `Frozen` is a no-op.
//! 3. **regen** the mesh from the advected seeds
//!    ([`assemble_meshless_from_seeds`]) — deterministic; an UNCHANGED seed set
//!    reproduces the current mesh byte-for-byte.
//! 4. **swept-quad mesh fluxes** old→new with the pinned dt: f64 telescoping
//!    geometry + the f32 SCL closure. A byte-identical regen ⇒ all-zero fluxes.
//! 5. **refresh + rotate**: [`SolverDriver::begin_ale_step_topology`] rotates
//!    the volume history, rebuilds the topology-derived solver stack, uploads
//!    the closed fluxes.
//! 6. **step**: `SolverDriver::step` (fixed dt, divergence detection).
//!
//! **Fixed seed count within a step**: seed `i` is cell `i` across every
//! per-step regen; no insertion/deletion happens inside the moving cycle, so
//! every cell-indexed solver buffer (state, BDF2 history, warm-start `x`)
//! survives the refresh untouched. The cell COUNT may change BETWEEN steps
//! through the [`resize_cells`](MovingMeshDriver::resize_cells) seam (a
//! rebuild-and-gather event: the wrapped solver is rebuilt at the new count
//! and every surviving cell's full state row is transferred), which the
//! flow-adaptive sizing ([`set_adaptive_sizing`](MovingMeshDriver::set_adaptive_sizing))
//! drives from the flow's velocity/pressure/strain gradients.
//!
//! **Model scope**: `incompressible_momentum_ale` only.
//!
//! **Moving boundaries**: [`BoundaryMotionSpec`] is orthogonal to
//! [`MeshMotionSpec`] (which governs interior seeds). A declared moving boundary
//! ([`BoundaryMotionSpec::RigidLoop`]) moves its boundary-bound seeds RIGIDLY
//! each step (keeping their parametric position on the moving wall) while the
//! regen clips against the moved loops. `moved_spec`/`apply_boundary_motion`
//! evaluate the rigid transform from the t=0 labels (no drift); `w_wall` records
//! the per-seed material velocity for the `MovingWall` Dirichlet BC. `Static`
//! (the default) is byte-identical to the static-boundary path.

use std::time::Instant;

use nalgebra::{Point2, Vector2};

use super::{RuntimeParams, SolverDriver, StepOutcome};
use crate::meshgen::meshless::{
    assemble_meshless_from_seeds, lloyd_relax, BoundarySpec, CvtMeshSeeds, EngineConfig,
    LloydConfig, SeedKind,
};
use crate::meshgen::MeshgenTolerances;
use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::mesh::{
    align_old_vertices_by_seed_set, detect_flips, swept_mesh_fluxes_closed,
    swept_mesh_fluxes_closed_flip, BoundaryType, FlipReport, Mesh,
};
use crate::solver::model::incompressible_momentum_ale_model;
use crate::solver::GpuLowMachPrecondModel;

/// Default per-step seed-displacement cap, as a fraction of the local cell
/// radius `R_i = √(V_i/π)`. Under `FlowCoupled` a seed cannot move more than
/// `FLOW_DISP_CAP · R_i` in one step regardless of flow speed — a hard
/// anti-tangling clamp that keeps the swept-quad linear-motion assumption and
/// the born/dead-face flip remap in their valid regime.
pub const DEFAULT_FLOW_DISP_CAP: f64 = 0.25;

/// Width (in `min_cell_size` units) of the open-boundary advection ramp for
/// `FlowCoupled` motion: within this band of the domain box the seed
/// ADVECTION fades to zero (the mesh is ~Eulerian there), so through-flow
/// cannot pile interior seeds against the FIXED boundary guard seeds.
/// Steering and the displacement clamp stay active in the band.
pub const FLOW_ADVECT_BOX_RAMP_CELLS: f64 = 3.0;

/// Dead-zone width (in `min_cell_size` units) of the advection ramp:
/// advection is FULLY zero within this distance of the box. A pure linear
/// ramp only SLOWS the approach (`d' ∝ d` — an exponential decay toward the
/// boundary that never stops), which measurably compressed the boundary
/// half-cells without bound (vol_min halved every ~1000 steps on the GUI
/// obstacle case); the dead zone stalls the creep at its edge, so the last
/// interior seed row parks ≥ this far from the fixed guards.
pub const FLOW_ADVECT_BOX_DEAD_CELLS: f64 = 1.5;

/// FlowCoupled cell-SIZING band, relative to the INITIAL mesh's realized
/// volume extremes (which encode the user's selected min/max cell size): the
/// quality escalation triggers when any cell volume falls below
/// `LO · initial_vol_min` or rises above `HI · initial_vol_max`, pulling the
/// seeds back toward the CVT sizing before cells mutilate into slivers
/// (near-degenerate faces flip-flopping through zero length every step).
/// Skew alone cannot catch this: a uniformly squeezed cell stays centroidal.
pub const QUALITY_VOL_BAND_LO: f64 = 0.5;
pub const QUALITY_VOL_BAND_HI: f64 = 2.0;

/// Minimum distance (in `min_cell_size` units) from a recycled seed's spawn
/// slot to the nearest existing seed. Deliberately BELOW the fresh-CVT
/// deepest-hole depth (~0.58·h, the hexagon circumradius): a spawn must
/// always be possible, or crossing seeds pile into a parked column at the
/// recycle line while the inlet stretches unfilled — the intake rate must
/// match the exit rate BY CONSTRUCTION, and the Lloyd sizing hold relaxes
/// the temporarily tight spawn neighborhood within a few steps. The floor
/// only guards the coalescing pitch (sub-pitch duplicates).
pub const RECYCLE_MIN_SEP_CELLS: f64 = 0.35;

/// Width (in `min_cell_size` units) of the OUTLET strip in which a squeezed
/// cell is eligible for recycling. The trigger is DENSITY, not position: a
/// position line either parks arrivals into a compression column (if spawns
/// lag) or drains the strip into oversized cells (if exports outrun the
/// upstream advection refill) — both observed. A squeeze trigger
/// (vol < `QUALITY_VOL_BAND_LO`·initial_min) is self-regulating: a drained
/// strip has no squeezed cells ⇒ exports stop ⇒ advection refills ⇒ density
/// recovers ⇒ exports resume — cells exit when they are compressed out,
/// exactly like the fluid they carry.
pub const RECYCLE_TRIGGER_CELLS: f64 = 3.0;

/// Max seeds recycled per step: staggers bulk squeeze events (a whole column
/// compressing at once would otherwise teleport in ONE giant flip step).
pub const RECYCLE_MAX_PER_STEP: usize = 8;

/// Squeeze fraction for the recycle trigger, applied to the OUTLET STRIP's
/// initial interior minimum volume (captured at build). The strip's natural
/// sizing differs from the bulk (its cells neighbor the boundary half-cells),
/// so neither the global minimum (a boundary half-cell — interior cells never
/// compress 10× against the steering) nor the bulk average (already below it
/// on a fresh strip) can calibrate the trigger; the strip's own initial
/// minimum is the reference that fires on genuine compression and never on
/// the fresh mesh.
pub const RECYCLE_SQUEEZE_FRACTION: f64 = 0.75;

/// Max cells BORN (split) per flow-adaptive sizing event. Rate-limited like
/// recycling: each event is a full solver rebuild-and-gather, so bulk changes
/// are spread over successive events instead of one giant remesh.
pub const ADAPT_MAX_BIRTHS_PER_EVENT: usize = 8;

/// Max cells KILLED (coarsened away) per flow-adaptive sizing event.
pub const ADAPT_MAX_KILLS_PER_EVENT: usize = 8;

/// Refine trigger: a cell is split once its volume exceeds
/// `ADAPT_REFINE_RATIO ×` its flow-derived target volume. The split halves the
/// volume, landing the children near the target — comfortably above the kill
/// trigger below (hysteresis: no birth→kill flip-flop).
pub const ADAPT_REFINE_RATIO: f64 = 2.0;

/// Coarsen trigger: a cell is removed once its volume falls below
/// `ADAPT_COARSEN_RATIO ×` its target. Its volume is absorbed by the
/// neighbors on the next regen (each grows by a fraction of one target —
/// well below the refine trigger; hysteresis again). Deliberately below 0.5
/// (the post-split child ratio) so a freshly split pair is never re-merged.
pub const ADAPT_COARSEN_RATIO: f64 = 0.45;

/// Robust normalization quantile for the flow-adaptation indicators: each raw
/// indicator (|∇U|, |∇p|, strain rate) is normalized by its q-quantile over
/// the eligible cells rather than its max, so one spike cell cannot flatten
/// the whole indicator field to ~0.
pub const ADAPT_INDICATOR_QUANTILE: f64 = 0.90;

/// Hard cell-count budget of the flow-adaptive sizing, as factors of the
/// INITIAL cell count: births stop at `MAX_FACTOR·n0`, kills at
/// `MIN_FACTOR·n0`. The GUI sizes its per-cell viz buffers against
/// `MAX_FACTOR` at build, so the cap is a contract, not a tuning knob.
pub const ADAPT_BUDGET_MAX_FACTOR: f64 = 2.0;
pub const ADAPT_BUDGET_MIN_FACTOR: f64 = 0.5;

/// Max TARGET-SPACING ratio between face-adjacent cells (the mesh grading
/// constraint): the raw indicator can step from 1 to 0 across one cell (a
/// front), demanding the full band jump between neighbors — Voronoi cells
/// with a several-× size jump mutilate (high skew, degenerate wall clips).
/// The target field is smoothed by min-propagation sweeps until every
/// face's spacing ratio is ≤ this factor, so refinement fans out in graded
/// layers exactly like the meshgen's `growth_rate`.
pub const ADAPT_GRADING_FACTOR: f64 = 1.3;

/// Max grading min-propagation sweeps (each sweep relaxes one adjacency
/// layer; a full band traverse at 1.3/layer needs ~5 — 16 is safely past
/// convergence and the loop breaks early when nothing changes).
pub const ADAPT_GRADING_SWEEPS: usize = 16;

/// Split a WALL polyline segment once its length exceeds this multiple of
/// its wall cells' target spacing — the boundary-discretization adaptation.
/// The wall's own seeds keep whatever spacing the segments give them, so
/// near-wall flow refinement is only reachable if the wall subdivides
/// along; a midpoint split halves the length to `0.75×` the trigger, so a
/// fresh split never immediately re-fires (hysteresis).
pub const ADAPT_WALL_SPLIT_RATIO: f64 = 1.5;

/// Max wall-segment splits per adaptation event (each split births 1–2 wall
/// seeds; rate-limited like the interior births).
pub const ADAPT_MAX_WALL_SPLITS_PER_EVENT: usize = 8;

/// Realized-fluid guard on wall splits: a segment may subdivide only while
/// its length exceeds this fraction of the ADJACENT interior cells' realized
/// spacing — the wall may lead the fluid by at most ONE subdivision level.
/// Without it the wall chases an aggressive target band unboundedly while
/// the interior is budget-capped, combing the boundary into sliver guard
/// cells (observed: wall seeds ~10× finer than the adjacent fluid). The
/// wall/fluid co-refinement staircase: wall splits one level ahead → the
/// wall-adjacent fluid unlocks (its own gate compares the segment to ITS
/// realized spacing) and splits → the wall unlocks again.
pub const ADAPT_WALL_FLUID_RATIO: f64 = 0.9;

/// Sizing-hysteresis PERSISTENCE window, in STEPS: a cell acts on its
/// split/kill threshold only after violating it for
/// `ceil(ADAPT_PERSIST_STEPS / adapt_every_n)` consecutive adapt events —
/// exactly 1 event at cadences ≥ 5 (the validated regime keeps single-event
/// behavior), 5 events at cadence 1. Breaks the kill→neighbor-inflates→
/// split-back ping-pong that saturated the per-event caps indefinitely under
/// every-step adaptation, and rejects noise-driven one-event violations.
pub const ADAPT_PERSIST_STEPS: usize = 5;

/// Default AREPO distortion trigger `η`: the centroid steering ramps in once a
/// cell's seed-to-centroid offset exceeds `η · R_i` and saturates at `1.1 η`.
/// Below `0.9 η` the steering is OFF, so well-shaped cells advect PURELY with
/// the flow.
pub const DEFAULT_AREPO_ETA: f64 = 0.25;

/// Default max-skewness above which a `FlowCoupled` step runs a Lloyd
/// regularization escalation on the advected seeds before committing. Below the
/// solver's comfortable skew band so quality is caught rising.
pub const DEFAULT_QUALITY_SKEW_TARGET: f64 = 0.5;

/// Default mesh-motion CFL cap factor (`dt ≤ cfl_mesh · min_h / max|w|`).
/// Conservative (0.2) so a fast seed cannot sweep more than ~a fifth of a cell
/// per step — the regime where the swept-quad linear-motion assumption holds.
/// Inert under `Frozen` (max|w| = 0).
pub const DEFAULT_MESH_CFL: f64 = 0.2;

/// Anti-swallow cap on a [`BoundaryMotionSpec::Oscillation`] amplitude, as a
/// fraction of the mesh's near-wall cell spacing (`min_cell_size`, the meshgen
/// length scale). The `v1` moving-mesh driver holds a FIXED seed count — cells
/// change shape, never existence — and the frozen/flow interior seeds adjacent
/// to a rigidly-moving wall do NOT step aside for it. If the wall's PEAK
/// displacement (its amplitude) exceeds the near-wall cell spacing, the wall
/// sweeps THROUGH those seeds and swallows them: the clipped near-wall cells
/// collapse to slivers / invert, and the swept-quad telescoping identity fails
/// (a mesh tangle) or the moving-wall velocity on a near-degenerate cell blows
/// the local CFL. The whole test envelope keeps `amplitude < cell spacing` by
/// construction (see `tests/moving_boundary_test.rs`); this clamps
/// `set_boundary_motion` to enforce that contract for every caller (the GUI
/// slider ranges up to 6× the cell size). `0.6` leaves the validated gates
/// (ratios ≤ 0.5) untouched while capping the too-large GUI default.
pub const OSC_AMPLITUDE_CELL_FRACTION: f64 = 0.6;

/// How the seeds move each step.
#[derive(Clone, Copy)]
pub enum MeshMotionSpec {
    /// Seeds never move (regen reproduces the same mesh byte-for-byte).
    Frozen,
    /// Prescribed analytic motion `new_pos = f(seed0, t)` from the t=0 seed
    /// label and absolute time (interior seeds only; boundary seeds fixed).
    /// Voronoi topology flips are handled by the born/dead-face conservative
    /// remap.
    Prescribed(fn([f64; 2], f64) -> [f64; 2]),
    /// Flow-coupled motion (cell velocity + AREPO centroid steering); the
    /// `regularization` is the steering strength χ.
    FlowCoupled { regularization: f64 },
}

/// How the *boundary* moves. Orthogonal to [`MeshMotionSpec`], which governs
/// the INTERIOR seeds: when a boundary is declared moving here, its
/// boundary-bound seeds move RIGIDLY with it each step (staying exactly on the
/// moving boundary) while the mesh regenerates against the moved loops.
///
/// `Static` holds all boundary seeds; the moving-mesh path is byte-identical to
/// a static-boundary run under it. `RigidLoop` prescribes an analytic rigid
/// transform for ONE boundary loop (the obstacle): an oscillating cylinder is
/// `transform(t, p) = [p.x + A·sin(ω t), p.y]`.
#[derive(Clone, Copy)]
pub enum BoundaryMotionSpec {
    /// All boundaries fixed (default). Byte-identical to the static-boundary
    /// path.
    Static,
    /// A rigidly-moving boundary loop. `loop_index` selects the loop in the
    /// [`BoundarySpec`] that moves (e.g. the obstacle circle is loop 1 of a
    /// [`crate::meshgen::ChannelWithObstacle`]); `transform(t, p)` maps a t=0
    /// point to its position at absolute time `t`. Applied to BOTH the loop's
    /// polyline points (so the regen clips against the moved wall) and the
    /// loop's boundary seeds (so seed `i` stays on the moving wall). **Contract:**
    /// `transform(0, p) == p` — the driver is built on the t=0 mesh, so the
    /// motion law must be the identity at t=0 (an oscillation `A·sin(ω t)`
    /// satisfies this).
    ///
    /// **Scope: pure TRANSLATION.** A rigid translation preserves chord lengths
    /// ⇒ the loop's seed count / segment structure are invariant, and —
    /// critically for the `MovingWall` BC — the per-seed material velocity
    /// `w_wall` from `(new−old)/dt` is UNIFORM across the seed's wall face,
    /// exactly matching the per-vertex swept `mesh_flux`. A ROTATION would break
    /// that match: the face's vertices sit at different radii/angles than the
    /// seed, so the uniform seed-velocity Dirichlet no longer cancels the
    /// per-face swept flux (a spurious wall mass flux O(ω·Δr)). Do not pass a
    /// rotating `transform`.
    RigidLoop {
        loop_index: usize,
        transform: fn(f64, [f64; 2]) -> [f64; 2],
    },
    /// A sinusoidally-oscillating boundary loop. Carries its own
    /// `amplitude`/`omega`/`axis` so the analytic rigid map is `p ↦ p +
    /// amplitude·sin(ω t)·ê_axis` — identity at t=0 (`sin 0 = 0`), so the
    /// build-mesh contract holds — WITHOUT a captured closure (the `RigidLoop`
    /// `fn` pointer cannot carry runtime-tuned parameters). Pure rigid
    /// translation, so the same chord-length / fixed-seed invariants carry over
    /// as for `RigidLoop`. `axis` selects in-line vs cross-stream forcing.
    Oscillation {
        loop_index: usize,
        amplitude: f64,
        omega: f64,
        axis: OscAxis,
    },
}

/// The axis a [`BoundaryMotionSpec::Oscillation`] translates the loop along:
/// `InLine` = streamwise (x), `CrossStream` = transverse (y). Cross-stream is
/// the clean forcing signal for a channel demo (a symmetric static obstacle
/// yields ~zero transverse velocity, so any measured `v` is the forced
/// response).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OscAxis {
    /// Streamwise (x) oscillation.
    InLine,
    /// Cross-stream (y) oscillation.
    CrossStream,
}

impl BoundaryMotionSpec {
    /// The moving loop's index, or `None` under [`Static`](Self::Static).
    fn loop_index(&self) -> Option<usize> {
        match *self {
            BoundaryMotionSpec::Static => None,
            BoundaryMotionSpec::RigidLoop { loop_index, .. }
            | BoundaryMotionSpec::Oscillation { loop_index, .. } => Some(loop_index),
        }
    }

    /// The rigid map applied to the moving loop's points + boundary seeds at
    /// absolute time `t`. Identity under `Static` (and required to be the
    /// identity at `t = 0` for every variant — the build-mesh contract).
    fn eval(&self, t: f64, p: [f64; 2]) -> [f64; 2] {
        match *self {
            BoundaryMotionSpec::Static => p,
            BoundaryMotionSpec::RigidLoop { transform, .. } => transform(t, p),
            BoundaryMotionSpec::Oscillation {
                amplitude,
                omega,
                axis,
                ..
            } => {
                let d = amplitude * (omega * t).sin();
                match axis {
                    OscAxis::InLine => [p[0] + d, p[1]],
                    OscAxis::CrossStream => [p[0], p[1] + d],
                }
            }
        }
    }
}

/// Where a moving step's mesh reconstruction ran (per-step telemetry; the
/// GUI surfaces it verbatim so the regen mode is obvious to the user).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegenBackend {
    /// CPU `assemble_meshless_from_seeds` + CPU swept-quad path, at compute
    /// time every step.
    Cpu,
    /// Fully on-device ([`crate::solver::gpu::voronoi::GpuMeshRegen`]):
    /// Voronoi + topology + geometry + ALE swept fluxes on the GPU.
    GpuOnDevice,
    /// The device regen ran but could not certify this step (reason
    /// attached: a Voronoi flip, a sub-tolerance sliver, or an unresolved
    /// cell); the step re-ran on the CPU path. The next step retries the
    /// device — fallback is per-step, not sticky.
    GpuFallback(&'static str),
    /// Skip-regen passthrough (`set_regen_each_step(false)`) — no
    /// reconstruction happened.
    Skipped,
}

impl RegenBackend {
    /// Short human-readable label for status lines.
    pub fn label(&self) -> &'static str {
        match self {
            RegenBackend::Cpu => "CPU (per step)",
            RegenBackend::GpuOnDevice => "GPU on-device",
            RegenBackend::GpuFallback(_) => "GPU→CPU fallback",
            RegenBackend::Skipped => "off",
        }
    }
}

/// Per-step moving-mesh telemetry (always-on diagnostics the gates and the UI
/// observe).
#[derive(Clone, Copy, Debug)]
pub struct MovingMeshStats {
    /// Wall time of the pre-regen seed-motion planning (ms): the dt handshake,
    /// seed advection, the FlowCoupled velocity readback, AND the
    /// quality-escalation PROBE regen (a full `assemble_meshless_from_seeds`
    /// before the authoritative regen). Counted separately so the overhead split
    /// stays honest. 0 on the skip-regen passthrough.
    pub plan_ms: f32,
    /// Wall time to regenerate the mesh from the advected seeds (ms).
    pub regen_ms: f32,
    /// Wall time to compute + SCL-close the swept mesh fluxes (ms).
    pub swept_ms: f32,
    /// Wall time of the topology refresh + volume rotation + flux upload (ms).
    pub refresh_ms: f32,
    /// Post-closure per-cell SCL defect (relative) — f32-roundoff scale by
    /// construction; the GCL diagnostic asserted every step.
    pub scl_defect: f64,
    /// Pre-closure f64 telescoping-identity residual (relative) — a check that
    /// the swept-quad geometry is self-consistent (f64-roundoff scale).
    pub identity_err: f64,
    /// Max mesh skewness `max_f (1 − |d̂·n̂|)` of the regenerated mesh.
    pub max_skew: f64,
    pub n_cells: usize,
    pub n_faces: usize,
    /// Vertex count of the committed mesh (0 for the VERTEX-LESS
    /// device-built mesh — the on-device regen never materializes vertices).
    pub n_vertices: usize,
    /// Cell-volume extremes of the committed mesh — the realized sizing the
    /// GUI mesh-stats panel tracks live (with adaptive sizing the initial
    /// snapshot goes stale immediately).
    pub vol_min: f64,
    pub vol_max: f64,
    /// The regenerated mesh's face ARRAYS (order and/or adjacency) differ from
    /// the previous step's, so the step drove the TOPOLOGY seam (a full CSR
    /// rebuild) rather than the surgical geometry seam. Any real seed motion
    /// reorders the deterministic face emission, so this is `true` every motion
    /// step — it is NOT a flip flag (see `flipped`). Always `false` under
    /// `Frozen` (byte-identical regen ⇒ geometry seam).
    pub topo_changed: bool,
    /// A genuine Voronoi topology FLIP happened this step (born/dead faces —
    /// an ADJACENCY change, not merely a face-array reorder). A flip always
    /// takes the topology seam and the flip-aware swept-flux path.
    pub flipped: bool,
    /// Number of NEWBORN faces (new `(i,j)` adjacency, no t^n swept quad) whose
    /// swept contribution was forced to zero. 0 on non-flip steps.
    pub born_faces: usize,
    /// Number of DIED faces (a t^n adjacency gone at t^{n+1}). 0 on non-flip
    /// steps.
    pub died_faces: usize,
    /// Number of cells a flip touched (incident to a born face). 0 on non-flip
    /// steps. The flip RATE diagnostic (÷ n_cells).
    pub flipped_cells: usize,
    /// Pre-closure per-cell **flip defect** (relative) — the residual the
    /// born/dead faces leave that the forest closure repairs onto the slack
    /// faces. O(motion·h/V) on a flip step, 0 otherwise. NOT a GCL error —
    /// `scl_defect` (the post-closure per-cell sum) stays at roundoff.
    pub flip_defect: f64,
    /// The pinned dt the swept fluxes were closed against AND the solver
    /// stepped with (they are equal by the dt handshake). f64-widened f32 —
    /// exactly `params.requested_dt as f64`.
    pub dt: f64,
    /// Where THIS step's mesh reconstruction ran (GPU on-device, CPU, a
    /// per-step GPU→CPU fallback, or skipped).
    pub regen_backend: RegenBackend,
    /// Seeds RECYCLED this step (outflow → inflow relabel-in-place): their
    /// cells were re-seeded as fresh inlet parcels. 0 unless FlowCoupled
    /// with recycling enabled.
    pub recycled: usize,
    /// Cells BORN this step by the flow-adaptive sizing (a split before the
    /// step's mesh motion; the step ran at the new count). 0 unless
    /// [`MovingMeshDriver::set_adaptive_sizing`] fired this step.
    pub cells_born: usize,
    /// Cells KILLED this step by the flow-adaptive sizing (coarsened away
    /// before the step's mesh motion).
    pub cells_killed: usize,
    /// The adaptivity GROWTH budget is exhausted (count at
    /// `budget factor × initial` — births suppressed until kills free
    /// room). Surfaced so a cell count that stops growing is explained.
    pub at_adapt_budget: bool,
    /// Fixed-point mesh-motion iterations this step actually ran
    /// ([`MovingMeshDriver::set_implicit_mesh_motion`]): `1` = the explicit
    /// single pass; `k > 1` = the step was re-solved `k` times, advecting
    /// with the previous attempt's end-of-step velocity, until the planned
    /// seed set converged (or the cap).
    pub motion_iters: usize,
    /// Mass-row transfer projection at THIS step's resize event: the
    /// inf-norm of the solver's OWN continuity-row residual at the
    /// transferred state, BEFORE the projected velocity correction.
    /// `0.0` when no resize fired (or the projection is disabled).
    pub transfer_defect_pre: f64,
    /// ... and AFTER. `post == pre` means the correction was rejected by
    /// the re-assembly verification (kept only when it measurably helps).
    pub transfer_defect_post: f64,
}

/// One step's shared pre-regen plan ([`MovingMeshDriver::plan_step`]): the
/// pinned dt, the advected seed set, the t^{n+1} boundary spec, the quality-
/// escalation flag, and the plan wall time — identical inputs for the CPU and
/// device regen paths, computed exactly once per step.
struct StepPlan {
    dt: f64,
    new_time: f64,
    new_seeds: Vec<Point2<f64>>,
    step_spec: BoundarySpec,
    escalated: bool,
    plan_ms: f32,
    /// Seeds RECYCLED this step (outflow → inflow relabel-in-place, see
    /// [`MovingMeshDriver::set_seed_recycling`]): their slot keeps its index,
    /// but the cell is a REMOVED + INSERTED pair — the step is forced onto
    /// the flip path, the closure targets ΔV = 0 for these cells, and their
    /// solver rows (state, time history, volume history, warm-start x) are
    /// re-seeded from the nearest surviving cell after the ALE seam.
    recycled: Vec<usize>,
}

/// Outcome of a device-regen attempt: the completed step, or a per-step
/// fallback request the caller routes to the CPU path (reason surfaced in
/// [`MovingMeshStats::regen_backend`]).
enum DeviceStep {
    Done((StepOutcome, MovingMeshStats)),
    Fallback(&'static str),
}

/// The rewind point of an implicit-mesh-motion attempt: the solver's
/// full-history snapshot plus every driver field a committed step mutates
/// (see [`MovingMeshDriver::motion_checkpoint`]).
struct MotionCheckpoint {
    snap: crate::solver::SolverStateSnapshot,
    seeds: Vec<Point2<f64>>,
    mesh: Mesh,
    prev_vx: Vec<f64>,
    prev_vy: Vec<f64>,
    w_wall: Vec<[f64; 2]>,
    time: f64,
    step_index: usize,
    last_escalated: bool,
}

/// The moving-mesh loop driver. Owns the authoritative seed set, the current
/// realized mesh, and the wrapped [`SolverDriver`].
pub struct MovingMeshDriver {
    driver: SolverDriver,
    /// The ALE model spec the wrapped solver was built with — retained so a
    /// cell-count RESIZE ([`Self::resize_cells`]) can rebuild the solver at
    /// the new count with the identical physics.
    model: crate::solver::model::ModelSpec,
    /// The INITIAL cell count — the flow-adaptive sizing budget anchor
    /// (births/kills are bounded to
    /// `[ADAPT_BUDGET_MIN_FACTOR, ADAPT_BUDGET_MAX_FACTOR]·initial_cell_count`).
    initial_cell_count: usize,
    /// Flow-adaptive sizing cadence: every `adapt_every_n` committed steps,
    /// derive a per-cell target volume from the flow's velocity/pressure/
    /// strain gradients and birth/kill cells toward it
    /// ([`Self::set_adaptive_sizing`]). `0` = off (default).
    adapt_every_n: usize,
    /// Explicit flow-adaptation TARGET band (cell VOLUMES, `lo ≤ hi`), set
    /// via [`Self::set_adaptive_sizing_band`] in cell-size units. `None`
    /// (default) = the initial mesh's realized volume band. When set, the
    /// FlowCoupled sizing hold widens to the UNION of this band and the
    /// initial band ([`Self::hold_vol_band`]) so the quality escalation
    /// never fights cells the adaptation deliberately refined/coarsened
    /// past the initial sizing.
    adapt_band: Option<(f64, f64)>,
    /// The adaptivity growth budget factor: births stop once the count
    /// reaches `adapt_budget_factor × initial_cell_count`
    /// ([`Self::set_adaptive_budget_factor`]; default
    /// [`ADAPT_BUDGET_MAX_FACTOR`]). The GUI sizes its per-cell viz buffers
    /// from the SAME requested factor, so the two stay a contract.
    adapt_budget_factor: f64,
    /// Per-component indicator THRESHOLD multipliers `(|∇U|, |∇p|, strain)`
    /// ([`Self::set_adaptive_indicator_thresholds`]): each component's
    /// auto-calibrated quantile scale is multiplied by its factor before
    /// normalization — `> 1` demands stronger features to trigger
    /// refinement, `< 1` refines at weaker ones, `0` disables the
    /// component. Default `(1, 1, 1)` (pure auto-calibration).
    adapt_thresholds: (f64, f64, f64),
    /// The last adaptation event's per-cell TARGET volumes (refreshed each
    /// event at the current indexing; permuted by reorder; invalidated by a
    /// direct resize). The per-cell SQUEEZE reference: a cell is
    /// "compressed" when its volume falls below `LO ×` its OWN target — a
    /// wide explicit band made the global hold-band floor useless as a
    /// squeeze detector (an adapted-fine cell and a squeezed cell are
    /// indistinguishable by volume alone), which let the stagnation-side
    /// pileup against the obstacle run unchecked.
    adapt_targets: Option<Vec<f64>>,
    /// Per-cell PERSISTENCE counters for the sizing hysteresis: `+k` = the
    /// cell exceeded its SPLIT threshold on `k` consecutive adapt events,
    /// `-k` = below its KILL threshold, `0` = in band. At fast cadences a
    /// single-event trigger PING-PONGS — a kill inflates its absorbing
    /// neighbors past the split threshold, which splits them back; at
    /// cadence 1 the cycle saturated the per-event birth/kill caps
    /// indefinitely (the salt-and-pepper size field + permanent recycle
    /// storm of the visual probe). Acting only on violations persisting
    /// ~[`ADAPT_PERSIST_STEPS`] steps breaks the cycle: fresh births and
    /// gathered survivors re-earn their next event from 0, and noise-driven
    /// violations flip sign before they accumulate. Cadences ≥ 5 keep
    /// today's single-event behavior exactly. Lifecycle: gathered through
    /// resizes (births at 0), permuted by reorder, reset on recycle.
    adapt_persist: Vec<i16>,
    /// IMPLICIT mesh motion ([`Self::set_implicit_mesh_motion`]): max
    /// fixed-point iterations of {advect with the previous attempt's
    /// END-of-step velocity → regen → ALE solve} per step. `1` = the
    /// explicit (lagged-velocity) motion (default).
    motion_outer_iters: usize,
    /// Fixed-point convergence tolerance: accept once the planned seed set
    /// changes by less than `tol × min_cell_size` between attempts.
    motion_outer_tol: f64,
    /// The previous attempt's end-of-step cell velocities — the advection
    /// source of the NEXT attempt's plan (implicit motion). `None` = read
    /// the solver's current (t^n) state, the explicit behaviour.
    motion_u_override: Option<Vec<(f64, f64)>>,
    /// Mass-row transfer projection at resize events (default ON; env kill
    /// switch `CFD2_ADAPT_PROJECTION=0`, setter
    /// [`Self::set_transfer_projection`]): project the interpolated state
    /// onto the solver's OWN continuity row before the first post-resize
    /// step, so the split/merge mass defect is carried away by a least-norm
    /// velocity correction instead of a phantom pressure dipole.
    transfer_projection: bool,
    /// The last resize event's projection measurement `(pre, post)` — the
    /// inf-norm mass-row residual before/after. Surfaced per adapt step in
    /// [`MovingMeshStats`].
    last_transfer_projection: (f64, f64),
    /// FLOW-adaptive dt for the moving path ([`Self::set_adaptive_dt`]):
    /// `Some(target_cfl)` re-computes the [`Self::pin_dt`] BASE each step as
    /// `cfl · min_h / (max(|U|, |U_in|) + c_eos)` — the same acoustic-aware
    /// controller as the static `SolverDriver::step` adaptive branch — but
    /// INSIDE the ALE dt handshake: the dt is chosen BEFORE the swept-flux
    /// closure, so the GCL contract holds for any per-step dt sequence (the
    /// BDF2 lowering carries variable-step coefficients `r = dt/dt_old`).
    /// `params.adaptive_dt` stays hard-rejected on this path — the
    /// solver-side controller re-scales dt AFTER the fluxes are closed,
    /// which would silently violate the GCL.
    adaptive_dt_cfl: Option<f64>,
    /// The last COMMITTED step's pinned dt — the adaptive growth-limit
    /// reference (dt may grow at most 1.2× per committed step; shrink is
    /// unlimited). Attempts inside the implicit-motion loop re-pin from the
    /// same reference, so retries stay deterministic.
    last_pinned_dt: Option<f64>,
    /// Authoritative seed positions (f64), seed `i` == cell `i` — the CURRENT
    /// (t^n) realized set. Advanced each step by [`MeshMotionSpec`].
    seeds: Vec<Point2<f64>>,
    /// The t=0 seed positions (the Lagrangian labels). Prescribed motion is
    /// evaluated as `f(seed0_i, t)` from these — sampling the analytic
    /// trajectory absolutely, so there is no incremental round-off drift.
    seeds0: Vec<Point2<f64>>,
    /// Accumulated simulated time (Σ of the actually-pinned dt's) — the `t`
    /// argument of a [`MeshMotionSpec::Prescribed`] law.
    time: f64,
    /// Per-seed kind (interior vs boundary), fixed for the run.
    kinds: Vec<SeedKind>,
    /// Boundary loops the regen clips against.
    spec: BoundarySpec,
    /// Clip domain `[0, x] × [0, y]`.
    domain: Vector2<f64>,
    /// Meshgen length scale — reconstructs the tolerances + engine config so a
    /// regen reproduces the mesh byte-for-byte.
    min_cell_size: f64,
    /// The dt the driver was CONFIGURED with at build (`params.requested_dt`,
    /// f64-widened) — the immutable base the mesh-motion CFL cap is applied to
    /// each step. Cached (rather than reading back the already-capped
    /// `params.requested_dt`) so the pinned dt cannot ratchet monotonically
    /// downward: a transient fast step must not permanently lower the timestep
    /// after the flow slows.
    configured_dt: f64,
    /// The current realized mesh (owned; the driver holds it across steps).
    mesh: Mesh,
    /// Vertex positions of `mesh` at t^n (the swept-quad `old` positions).
    prev_vx: Vec<f64>,
    prev_vy: Vec<f64>,
    /// Seed motion law (INTERIOR seeds).
    motion: MeshMotionSpec,
    /// Boundary motion law. `Static` ⇒ byte-identical to a static-boundary run.
    boundary_motion: BoundaryMotionSpec,
    /// Per-seed material velocity `w_wall = (new_pos − old_pos)/dt` of the last
    /// committed step, seed `i` == cell `i`. Zero for interior + static-boundary
    /// seeds; the rigid boundary velocity for moving-wall seeds. Feeds the
    /// `MovingWall` Dirichlet BC (the Dirichlet U at the wall must be the wall's
    /// material velocity). All-zero until the first moving step.
    w_wall: Vec<[f64; 2]>,
    /// Feed the moving wall's material velocity into the fluid. When `true`
    /// (opt-in via [`set_moving_wall_bc`]) AND a `RigidLoop` boundary motion is
    /// declared, each regenerated mesh's moving-loop open faces are re-tagged
    /// [`BoundaryType::MovingWall`] (from the engine's default `Wall`) and, after
    /// the ALE refresh, their per-face Dirichlet velocity `bc_value` is set to
    /// `w_wall[owner]` — so the no-slip ghost at the wall carries the wall's
    /// material velocity (no-penetration + no-slip). Static + fixed walls stay
    /// `Wall`/`Slip`. Default `false` (the obstacle stays a static-velocity
    /// `Wall`).
    moving_wall_bc: bool,
    /// Mesh-motion CFL cap factor.
    mesh_cfl: f64,
    /// Whether `step` regenerates + refreshes each step. `false` = the
    /// skip-regen variant (a pure `SolverDriver::step` passthrough,
    /// byte-identical to a static run).
    regen_each_step: bool,
    /// Optional per-regen boundary-tag rewrite, applied to each freshly
    /// regenerated mesh before the ALE seam. The engine tags the domain box
    /// sides by a FIXED rule (left Inlet / right Outlet / bottom+top Wall), so a
    /// run that wants a different arrangement must re-stamp `face_boundary` every
    /// step — the topology seam rebuilds the bc tables from the regenerated
    /// mesh's tags, so a one-time retag of the initial mesh would be lost.
    /// Touches only `face_boundary` (never geometry/adjacency), so it does not
    /// affect the swept fluxes or the topology-diff. `None` = keep the engine's
    /// tags.
    boundary_retag: Option<fn(&mut Mesh)>,
    /// Route every step through the topology seam even when the topology is
    /// unchanged (default `false` — use the geometry seam when it suffices).
    /// The topology seam clears the AMG hierarchy + re-scatters bc tables, so
    /// this deliberately perturbs the solve away from a static run; it exists to
    /// MEASURE that perturbation.
    force_topology_seam: bool,
    /// Per-step seed-displacement cap (fraction of local cell radius), FlowCoupled.
    flow_disp_cap: f64,
    /// AREPO centroid-steering distortion trigger `η`, FlowCoupled.
    arepo_eta: f64,
    /// Regenerated-mesh max-skew above which a FlowCoupled step runs a Lloyd
    /// regularization escalation on the advected seeds.
    quality_skew_target: f64,
    /// The INITIAL mesh's (vol_min, vol_max) — the realized cell sizing of the
    /// user's selected mesh settings. FlowCoupled escalation also triggers
    /// when the advected mesh's volumes leave
    /// `[QUALITY_VOL_BAND_LO·min, QUALITY_VOL_BAND_HI·max]` (sizing hold).
    initial_vol_band: (f64, f64),
    /// FlowCoupled seed RECYCLING: an interior seed whose cell is squeezed
    /// below [`Self::recycle_vol_floor`] inside the outlet strip is relabeled
    /// in place to the deepest fluid hole in the inlet band, its cell
    /// re-seeded as a fresh parcel — the mesh "flows through" the domain
    /// instead of straining against the fixed boundary guards. Default on
    /// (inert for non-FlowCoupled motion).
    seed_recycling: bool,
    /// Recycle squeeze threshold: `RECYCLE_SQUEEZE_FRACTION ×` the OUTLET
    /// strip's initial interior minimum cell volume (see
    /// [`RECYCLE_SQUEEZE_FRACTION`]).
    recycle_vol_floor: f64,
    /// Periodic mesh smoothing cadence: every `smoothing_every_n` committed
    /// steps, run `smoothing_iters` blended Lloyd sweeps at weight
    /// `smoothing_omega` over the interior seeds — UNCONDITIONAL, unlike the
    /// quality escalation (which fires only on skew/sizing violations).
    /// `0` = off (default). FlowCoupled + Frozen (a stationary mesh advances
    /// from its current seeds, so the smooth persists); ignored under
    /// Prescribed (positions recomputed from the t=0 labels every step
    /// would overwrite it).
    smoothing_every_n: usize,
    smoothing_iters: usize,
    smoothing_omega: f64,
    /// Periodic MEMORY reordering cadence: every `reorder_every_n` committed
    /// steps, relabel the cells in Morton order of the CURRENT seed
    /// positions ([`Self::set_reorder_every_n`]). Long FlowCoupled runs with
    /// recycling migrate slots arbitrarily far from their neighbors (a slot
    /// dies at the outlet and respawns at the inlet keeping its index), so
    /// the initial near-optimal generator order decays toward random —
    /// measured on this codebase at +30% CPU / +55% GPU step cost. `0` = off
    /// (default).
    reorder_every_n: usize,
    /// Number of Lloyd regularization iterations a quality escalation runs
    /// (0 = escalation disabled). Blended (`omega`) so it nudges toward CVT
    /// without erasing the flow displacement.
    lloyd_escalation_iters: usize,
    /// Under-relaxation of the Lloyd escalation move (0..1); small so the
    /// escalation regularizes gently.
    lloyd_escalation_omega: f64,
    /// Committed step count.
    step_index: usize,
    /// Last committed FlowCoupled quality-escalation flag (telemetry).
    last_escalated: bool,
    /// GPU device/queue clones, retained so the opt-in on-device regen path can
    /// build its own [`GpuMeshRegen`]. `None` when the caller supplied no GPU
    /// handles. NOTE: presence does NOT imply the GPU solver backend — the GUI
    /// passes its RENDER device even for CPU-solver runs (`CFD2_BACKEND=cpu`),
    /// so device-path eligibility additionally checks
    /// `!driver.solver().is_cpu()` (see [`Self::gpu_regen_active`]).
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    /// Opt-in: reconstruct the mesh ENTIRELY on the GPU each step
    /// ([`Self::set_gpu_regen`]) instead of the CPU `assemble_meshless_from_seeds`
    /// + CPU swept path. Default `false` (the shipped CPU path is byte-identical).
    /// Requires the GPU solver backend. A step the device cannot certify (a
    /// Voronoi flip — the device swept assumes a persistent adjacency —, a
    /// sub-tolerance sliver, or an unresolved cell) re-runs on the CPU path for
    /// that step and retries the device on the next
    /// ([`RegenBackend::GpuFallback`] in the step stats).
    gpu_regen: bool,
    /// Lazily-built on-device regen bundle (engine + passes), created on the
    /// first `gpu_regen` step.
    gpu_regen_state: Option<crate::solver::gpu::voronoi::GpuMeshRegen>,
    /// Lazily-built GPU context (reusing `device`/`queue`) for the regen passes.
    gpu_ctx: Option<crate::solver::gpu::context::GpuContext>,
}

impl MovingMeshDriver {
    /// Build a moving-mesh driver from an initial CVT mesh + its authoritative
    /// seeds ([`crate::meshgen::meshless::generate_cvt_mesh_with_seeds`]), using
    /// the default `incompressible_momentum_ale` model. For the all-Mach
    /// compressible ALE variants use [`MovingMeshDriver::build_with_model`].
    #[allow(clippy::too_many_arguments)]
    pub async fn build(
        cvt: CvtMeshSeeds,
        params: &RuntimeParams,
        motion: MeshMotionSpec,
        initial_u: &[(f64, f64)],
        initial_p: &[f64],
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        Self::build_with_model(
            cvt,
            incompressible_momentum_ale_model()?,
            params,
            motion,
            initial_u,
            initial_p,
            device,
            queue,
        )
        .await
    }

    /// Build a moving-mesh driver with an explicit ALE `model` (e.g.
    /// `incompressible_momentum_ale` or `allmach_pressure_ale`). The wrapped
    /// [`SolverDriver`] is constructed on `cvt.mesh`, seeded with
    /// `initial_u`/`initial_p` (plus any model-specific state — psi/psi_precond/
    /// rho/dt_local for the all-Mach models — via `SolverDriver::build`). `params`
    /// must have `adaptive_dt == false` (the ALE seam SCL-closes fluxes against a
    /// fixed dt; an adaptive re-scale would break the GCL — rejected here, up
    /// front, rather than at the first `begin_ale_step_topology`). The model MUST
    /// be an ALE model (its convection is `.with_mesh_relative()`); a static model
    /// is rejected by the `begin_ale_step*` seam. All-Mach models keep
    /// `dt_local ≡ 0` under motion (time-accurate global dt).
    #[allow(clippy::too_many_arguments)]
    pub async fn build_with_model(
        cvt: CvtMeshSeeds,
        model: crate::solver::model::ModelSpec,
        params: &RuntimeParams,
        motion: MeshMotionSpec,
        initial_u: &[(f64, f64)],
        initial_p: &[f64],
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        if params.adaptive_dt {
            return Err(
                "MovingMeshDriver requires params.adaptive_dt == false: the swept mesh fluxes \
                 are SCL-closed against a fixed dt, and an adaptive re-scale after the closure \
                 silently violates the GCL (the mesh-motion CFL cap is applied by the driver \
                 itself)."
                    .into(),
            );
        }
        let CvtMeshSeeds {
            mesh,
            seeds,
            kinds,
            spec,
            domain,
            min_cell_size,
        } = cvt;

        // Clone the GPU handles for the opt-in on-device regen path before the
        // build consumes them (wgpu Device/Queue are cheap Arc clones).
        let device_kept = device.clone();
        let queue_kept = queue.clone();
        // Keep the model spec for cell-count RESIZE rebuilds (resize_cells).
        let model_kept = model.clone();
        let build =
            SolverDriver::build(&mesh, model, params, initial_u, initial_p, device, queue).await?;

        let prev_vx = mesh.vx.clone();
        let prev_vy = mesh.vy.clone();
        let n_seeds = seeds.len();
        // The realized cell sizing of the user's mesh settings — the
        // FlowCoupled sizing-hold band is anchored to it.
        let initial_vol_band = mesh
            .cell_vol
            .iter()
            .fold((f64::MAX, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        // Recycle trigger reference: the OUTLET strip's initial interior
        // minimum (see `RECYCLE_SQUEEZE_FRACTION`).
        let recycle_vol_floor = {
            let strip_x = domain.x - RECYCLE_TRIGGER_CELLS * min_cell_size;
            let strip_min = (0..n_seeds)
                .filter(|&i| kinds[i] == SeedKind::Interior && seeds[i].x > strip_x)
                .map(|i| mesh.cell_vol[i])
                .fold(f64::MAX, f64::min);
            if strip_min.is_finite() {
                RECYCLE_SQUEEZE_FRACTION * strip_min
            } else {
                QUALITY_VOL_BAND_LO * initial_vol_band.0
            }
        };

        Ok(Self {
            driver: build.driver,
            model: model_kept,
            initial_cell_count: n_seeds,
            adapt_every_n: 0,
            adapt_band: None,
            adapt_budget_factor: ADAPT_BUDGET_MAX_FACTOR,
            adapt_thresholds: (1.0, 1.0, 1.0),
            adapt_targets: None,
            adapt_persist: vec![0; n_seeds],
            motion_outer_iters: 1,
            motion_outer_tol: 0.02,
            motion_u_override: None,
            transfer_projection: std::env::var("CFD2_ADAPT_PROJECTION")
                .map(|v| v != "0")
                .unwrap_or(true),
            last_transfer_projection: (0.0, 0.0),
            adaptive_dt_cfl: None,
            last_pinned_dt: None,
            seeds0: seeds.clone(),
            time: 0.0,
            seeds,
            kinds,
            spec,
            domain,
            min_cell_size,
            configured_dt: params.requested_dt as f64,
            mesh,
            prev_vx,
            prev_vy,
            motion,
            boundary_motion: BoundaryMotionSpec::Static,
            w_wall: vec![[0.0, 0.0]; n_seeds],
            moving_wall_bc: false,
            mesh_cfl: DEFAULT_MESH_CFL,
            boundary_retag: None,
            regen_each_step: true,
            force_topology_seam: false,
            flow_disp_cap: DEFAULT_FLOW_DISP_CAP,
            arepo_eta: DEFAULT_AREPO_ETA,
            quality_skew_target: DEFAULT_QUALITY_SKEW_TARGET,
            initial_vol_band,
            seed_recycling: true,
            recycle_vol_floor,
            smoothing_every_n: 0,
            smoothing_iters: 1,
            smoothing_omega: 0.5,
            reorder_every_n: 0,
            lloyd_escalation_iters: 1,
            lloyd_escalation_omega: 0.4,
            step_index: 0,
            last_escalated: false,
            device: device_kept,
            queue: queue_kept,
            gpu_regen: false,
            gpu_regen_state: None,
            gpu_ctx: None,
        })
    }

    /// Opt in to reconstructing the mesh ENTIRELY on the GPU each step (Voronoi
    /// diagram + topology + geometry + ALE swept fluxes) instead of the CPU
    /// `assemble_meshless_from_seeds` + CPU swept path. Requires the GPU solver
    /// backend AND a device handle — otherwise the flag is inert and every step
    /// runs the CPU path (see [`Self::gpu_regen_active`]). A step the device
    /// cannot certify (Voronoi flip / sub-tolerance sliver / unresolved cell)
    /// re-runs on the CPU path for that step and retries the device on the
    /// next; the per-step [`MovingMeshStats::regen_backend`] reports which path
    /// ran. Default `false`.
    pub fn set_gpu_regen(&mut self, on: bool) {
        self.gpu_regen = on;
    }

    /// Whether the on-device regen path is enabled AND available: the flag is
    /// set, a device handle exists, and the wrapped solver actually runs on the
    /// GPU (the GUI passes its RENDER device even for `CFD2_BACKEND=cpu` runs,
    /// so the device handle alone is not a backend discriminator).
    pub fn gpu_regen_active(&self) -> bool {
        self.gpu_regen && self.device.is_some() && !self.driver.solver().is_cpu()
    }

    /// Set the mesh-motion CFL cap factor (default [`DEFAULT_MESH_CFL`]).
    pub fn set_mesh_cfl(&mut self, cfl: f64) {
        self.mesh_cfl = cfl;
    }

    /// Configure periodic mesh smoothing: every `every_n` committed steps,
    /// run `iters` blended Lloyd sweeps at weight `omega` (0..1) over the
    /// interior seeds — a scheduled, unconditional regularization on top of
    /// the on-demand quality escalation. `every_n == 0` disables (default).
    /// Active under FlowCoupled AND Frozen (a stationary mesh advances from
    /// its current seeds, so the smooth persists — with flow-adaptive
    /// sizing enabled, the Lloyd target preserves the LOCAL adapted spacing
    /// instead of pulling toward uniform, see `lloyd_sizing`); ignored
    /// under Prescribed (positions are recomputed from the t=0 labels every
    /// step, which would overwrite it). Small `omega` keeps the smooth from
    /// erasing the flow-coupled displacement; the smooth itself is mesh
    /// motion like any other — the swept-flux closure keeps the GCL.
    pub fn set_smoothing(&mut self, every_n: usize, iters: usize, omega: f64) {
        self.smoothing_every_n = every_n;
        self.smoothing_iters = iters.max(1);
        self.smoothing_omega = omega.clamp(0.0, 1.0);
    }

    /// Configure periodic memory REORDERING: every `every_n` committed steps,
    /// relabel the cells in Morton order of the current seed positions —
    /// state, time history, volume history and warm-start rows are permuted
    /// on the solver ([`crate::solver::gpu::unified_solver::UnifiedSolver::permute_cells`]),
    /// the committed mesh is relabeled in place, and the next step's topology
    /// seam rebuilds the face-indexed stacks. Restores cache locality that
    /// long FlowCoupled+recycling runs erode (slots migrate arbitrarily far
    /// from their spatial neighbors). `0` = off (default). A pure relabel:
    /// zero mesh motion, zero physics.
    pub fn set_reorder_every_n(&mut self, every_n: usize) {
        self.reorder_every_n = every_n;
    }

    /// Configure FLOW-ADAPTIVE sizing: every `every_n` committed steps, derive
    /// a per-cell target volume from the flow (|∇U|, |∇p| and the strain-rate
    /// magnitude, Green–Gauss over the current mesh; high gradients → the
    /// initial mesh's finest realized volume, smooth regions → its coarsest)
    /// and BIRTH/KILL cells toward it through [`Self::resize_cells`]:
    /// over-resolved cells (volume < [`ADAPT_COARSEN_RATIO`]·target) are
    /// removed, under-resolved ones (volume > [`ADAPT_REFINE_RATIO`]·target)
    /// are split. Rate-limited ([`ADAPT_MAX_BIRTHS_PER_EVENT`] /
    /// [`ADAPT_MAX_KILLS_PER_EVENT`]) and budget-bounded
    /// ([`ADAPT_BUDGET_MIN_FACTOR`]‥[`ADAPT_BUDGET_MAX_FACTOR`] × the initial
    /// count). `0` = off (default). Available under FlowCoupled AND Frozen —
    /// a STATIONARY mesh adapts to the developing flow too (pair with
    /// [`Self::set_smoothing`] for post-resize relaxation; the quality
    /// escalation activates alongside). The BOUNDARY discretization adapts
    /// along: static `Wall`/`SlipWall` polyline segments whose wall cells'
    /// target falls below [`ADAPT_WALL_SPLIT_RATIO`]⁻¹ × their length are
    /// subdivided at their midpoints (geometry-exact — shape and fluid area
    /// unchanged) so near-wall refinement is actually reachable. Skipped for
    /// `Prescribed` motion (its analytic law is indexed by the t=0 labels,
    /// which a resize re-anchors).
    pub fn set_adaptive_sizing(&mut self, every_n: usize) {
        self.adapt_every_n = every_n;
    }

    /// Set the flow-adaptation TARGET band explicitly, in CELL-SIZE units
    /// (the meshgen length scale): the indicator maps high gradients →
    /// `min_size` and smooth regions → `max_size`, independent of the mesh
    /// the run was built with — e.g. build coarse and let the adaptation
    /// refine below the built sizing. Sizes are converted to cell volumes
    /// via the regular-hexagon area `(√3/2)·h²` (the realized CVT volume at
    /// pitch `h`) and ordered. `None` (default) restores the initial mesh's
    /// realized band. The FlowCoupled sizing hold widens to the union of
    /// this band and the initial band, so the escalation never undoes
    /// deliberate refinement; the cell-count budget
    /// ([`ADAPT_BUDGET_MIN_FACTOR`]‥[`ADAPT_BUDGET_MAX_FACTOR`] × the
    /// initial count) still caps growth regardless of the band.
    pub fn set_adaptive_sizing_band(&mut self, band: Option<(f64, f64)>) {
        self.adapt_band = band.map(|(a, b)| {
            let hex_vol = |h: f64| 3.0f64.sqrt() / 2.0 * h * h;
            let (va, vb) = (hex_vol(a.max(0.0)), hex_vol(b.max(0.0)));
            (va.min(vb), va.max(vb))
        });
    }

    /// Set the adaptivity GROWTH budget factor: births stop once the cell
    /// count reaches `factor × initial_cell_count` (default
    /// [`ADAPT_BUDGET_MAX_FACTOR`]). The GUI allocates its per-cell viz
    /// buffers from the same requested factor — keep the two in sync when
    /// calling this directly. Clamped to `[1, 16]`.
    pub fn set_adaptive_budget_factor(&mut self, factor: f64) {
        self.adapt_budget_factor = factor.clamp(1.0, 16.0);
    }

    /// The adaptivity growth cap in CELLS (`factor × initial count`).
    fn adapt_cell_cap(&self) -> usize {
        (self.initial_cell_count as f64 * self.adapt_budget_factor) as usize
    }

    /// Set the per-component indicator THRESHOLD multipliers
    /// `(|∇U|, |∇p|, strain-rate)`. Each component's auto-calibrated scale
    /// (the [`ADAPT_INDICATOR_QUANTILE`] quantile over the eligible cells)
    /// is multiplied by its factor before normalization: `> 1` = only
    /// stronger features refine, `< 1` = refine at weaker ones, `0` =
    /// ignore this component entirely. Default `(1, 1, 1)`. Clamped to
    /// `[0, 100]`.
    pub fn set_adaptive_indicator_thresholds(&mut self, u: f64, p: f64, strain: f64) {
        let c = |v: f64| v.clamp(0.0, 100.0);
        self.adapt_thresholds = (c(u), c(p), c(strain));
    }

    /// IMPLICIT (fixed-point) mesh motion: iterate `{advect the seeds with
    /// the previous attempt's END-of-step cell velocity → regenerate → ALE
    /// solve}` from a byte-exactly rewound t^n state, until the planned
    /// seed set changes by less than `tol_cells × min_cell_size` between
    /// attempts (or `max_iters` attempts ran). The accepted attempt's mesh
    /// motion is consistent with its own end-of-step solution — the
    /// explicit (lagged-velocity) coupling noise a per-step remesh injects
    /// is solved away instead of committed. `max_iters = 1` (default) is
    /// the explicit single pass, byte-identical to the shipped behaviour.
    /// Each extra attempt costs a full regen + ALE solve.
    ///
    /// FlowCoupled + CPU backend only: only FlowCoupled motion depends on
    /// the solution (Frozen/Prescribed converge trivially), and the GPU
    /// snapshot restores with initial-condition semantics (it would reset
    /// the BDF2 history every attempt) — elsewhere the knob is inert.
    pub fn set_implicit_mesh_motion(&mut self, max_iters: usize, tol_cells: f64) {
        self.motion_outer_iters = max_iters.max(1);
        self.motion_outer_tol = tol_cells.max(0.0);
    }

    /// Enable/disable the mass-row transfer projection at resize events
    /// (default ON; also killable via `CFD2_ADAPT_PROJECTION=0`). OFF means
    /// the raw first-order interpolant is stepped as-is — the pre-projection
    /// behaviour, kept reachable for A/B measurement.
    pub fn set_transfer_projection(&mut self, on: bool) {
        self.transfer_projection = on;
    }

    /// FLOW-adaptive timestep for the moving path: `Some(target_cfl)` makes
    /// every step pin `dt = target_cfl · min_h / (max(|U|, |U_in|) + c_eos)`
    /// (growth-limited to 1.2× per committed step, still capped by the
    /// mesh-motion CFL), `None` restores the fixed configured dt. This is
    /// the GCL-safe analogue of `params.adaptive_dt` (which stays rejected
    /// on the moving path): the dt is chosen at the HEAD of the handshake,
    /// before the swept fluxes are closed against it.
    pub fn set_adaptive_dt(&mut self, target_cfl: Option<f64>) {
        self.adaptive_dt_cfl = target_cfl.filter(|c| *c > 0.0);
    }

    /// Re-base the fixed timestep mid-run (the GUI dt slider): the
    /// configured dt was previously captured at build only, which made a
    /// live dt change on the moving path silently inert — `pin_dt` re-bases
    /// from the configured value every step (the anti-ratchet), so THIS is
    /// the knob a runtime dt change must turn. Under adaptive dt it is the
    /// fallback base for steps where the flow readback is unavailable.
    pub fn set_configured_dt(&mut self, dt: f64) {
        if dt.is_finite() && dt > 0.0 {
            self.configured_dt = dt;
        }
    }

    /// Whether the adaptivity growth budget is exhausted (births suppressed;
    /// kills still free budget). Surfaced per step in
    /// [`MovingMeshStats::at_adapt_budget`] so a count that stops growing is
    /// explained, not mysterious.
    pub fn adapt_budget_reached(&self) -> bool {
        self.adapt_every_n > 0 && self.seeds.len() >= self.adapt_cell_cap()
    }

    /// Consecutive violating adapt EVENTS required before the planner acts:
    /// `ceil(ADAPT_PERSIST_STEPS / adapt_every_n)`, i.e. ~5 STEPS of
    /// persistence at any cadence (exactly 1 event at cadences ≥ 5).
    fn adapt_persist_needed(&self) -> i16 {
        let every = self.adapt_every_n.max(1);
        ADAPT_PERSIST_STEPS
            .div_ceil(every)
            .clamp(1, ADAPT_PERSIST_STEPS) as i16
    }

    /// The flow-adaptation TARGET volume band: the explicit band when set,
    /// else the initial mesh's realized band.
    fn adapt_vol_band(&self) -> (f64, f64) {
        self.adapt_band.unwrap_or(self.initial_vol_band)
    }

    /// The FlowCoupled sizing-HOLD volume band (the chi_size ramp + the
    /// quality-escalation sizing trigger): the initial realized band widened
    /// to include any explicit adaptation band — never narrower than the
    /// initial band (a fresh mesh must not self-trigger), never excluding
    /// volumes the adaptation deliberately targets.
    fn hold_vol_band(&self) -> (f64, f64) {
        match self.adapt_band {
            Some((lo, hi)) => (
                self.initial_vol_band.0.min(lo),
                self.initial_vol_band.1.max(hi),
            ),
            None => self.initial_vol_band,
        }
    }

    /// Whether the flow-adaptive sizing will PLAN a resize on the upcoming
    /// `step()` call (cadence hit; the event may still be empty if no cell
    /// violates its target band). The GUI worker uses this to force a
    /// readback+mesh publish on resize steps so the per-cell viz arrays and
    /// the polygon set change together.
    pub fn adapt_fires_this_step(&self) -> bool {
        self.adapt_every_n > 0
            && self.regen_each_step
            && self.step_index > 0
            && self.step_index % self.adapt_every_n == 0
            && !matches!(self.motion, MeshMotionSpec::Prescribed(_))
    }

    /// Enable/disable FlowCoupled seed RECYCLING (default on): seeds advected
    /// into the outlet dead zone relabel in place to the largest fluid gap in
    /// the inlet band, and their cells are re-seeded as fresh parcels (state
    /// + time history + volume history + warm-start x from the nearest
    /// surviving cell) — the mesh flows through the domain instead of
    /// straining/compressing against the fixed boundary guards. Inert for
    /// Frozen/Prescribed motion.
    pub fn set_seed_recycling(&mut self, on: bool) {
        self.seed_recycling = on;
    }

    /// Install a per-regen boundary-tag rewrite (see the `boundary_retag`
    /// field). Apply the same rewrite to the initial mesh before `build` so the
    /// first step and every regen agree. `None` restores the engine's tags.
    pub fn set_boundary_retag(&mut self, retag: Option<fn(&mut Mesh)>) {
        self.boundary_retag = retag;
    }

    /// Force every step through the topology-refresh seam (default off — the
    /// driver uses the geometry seam when the topology is unchanged). Used to
    /// measure the topology-seam perturbation (AMG reset + bc re-scatter).
    pub fn set_force_topology_seam(&mut self, force: bool) {
        self.force_topology_seam = force;
    }

    /// FlowCoupled tuning: the per-step seed-displacement cap (fraction of the
    /// local cell radius; default [`DEFAULT_FLOW_DISP_CAP`]) and the AREPO
    /// distortion trigger `η` (default [`DEFAULT_AREPO_ETA`]).
    pub fn set_flow_coupled_tuning(&mut self, disp_cap: f64, arepo_eta: f64) {
        self.flow_disp_cap = disp_cap;
        self.arepo_eta = arepo_eta;
    }

    /// FlowCoupled quality escalation: the regenerated-mesh max-skew above which
    /// a step runs `iters` blended Lloyd regularization sweeps (`iters = 0`
    /// disables it) at under-relaxation `omega`.
    pub fn set_quality_escalation(&mut self, skew_target: f64, iters: usize, omega: f64) {
        self.quality_skew_target = skew_target;
        self.lloyd_escalation_iters = iters;
        self.lloyd_escalation_omega = omega;
    }

    /// Whether the last committed step ran a Lloyd quality escalation.
    pub fn last_escalated(&self) -> bool {
        self.last_escalated
    }

    /// Switch the seed-motion law mid-run (e.g. develop a flow on a cheap static
    /// `Frozen`+skip-regen warm-up, then hand off to `FlowCoupled` to show the
    /// moving mesh sustains it). The current realized mesh + all warm-started
    /// solver state carry over untouched; the next `step` advects from here.
    pub fn set_motion(&mut self, motion: MeshMotionSpec) {
        self.motion = motion;
    }

    /// Declare a moving boundary. `Static` (the default) holds all boundaries;
    /// [`BoundaryMotionSpec::RigidLoop`] moves one loop's boundary seeds rigidly
    /// each step. Orthogonal to [`set_motion`]: the interior seeds still follow
    /// the [`MeshMotionSpec`]. Panics if a `RigidLoop` `loop_index` is out of
    /// range for the current boundary spec.
    ///
    /// An [`BoundaryMotionSpec::Oscillation`] amplitude is CLAMPED to
    /// [`OSC_AMPLITUDE_CELL_FRACTION`] × the near-wall cell spacing (the fixed
    /// `min_cell_size`): a wall whose peak displacement exceeds the cell spacing
    /// swallows the frozen interior seeds it sweeps into and tangles the mesh
    /// (see [`OSC_AMPLITUDE_CELL_FRACTION`]). The clamp is a no-op for a `RigidLoop`
    /// (its motion is baked into an opaque `fn`, kept within envelope by its
    /// caller) and for any in-envelope amplitude.
    pub fn set_boundary_motion(&mut self, boundary_motion: BoundaryMotionSpec) {
        if let Some(loop_index) = boundary_motion.loop_index() {
            assert!(
                loop_index < self.spec.loops.len(),
                "BoundaryMotionSpec moving loop_index {loop_index} out of range \
                 ({} loops)",
                self.spec.loops.len()
            );
        }
        self.boundary_motion = self.clamp_oscillation_amplitude(boundary_motion);
    }

    /// Clamp an [`BoundaryMotionSpec::Oscillation`] amplitude to the anti-swallow
    /// cap (peak displacement < near-wall cell spacing). Other variants pass
    /// through unchanged. See [`OSC_AMPLITUDE_CELL_FRACTION`].
    fn clamp_oscillation_amplitude(&self, spec: BoundaryMotionSpec) -> BoundaryMotionSpec {
        match spec {
            BoundaryMotionSpec::Oscillation {
                loop_index,
                amplitude,
                omega,
                axis,
            } if self.min_cell_size.is_finite() && self.min_cell_size > 0.0 => {
                let cap = OSC_AMPLITUDE_CELL_FRACTION * self.min_cell_size;
                BoundaryMotionSpec::Oscillation {
                    loop_index,
                    amplitude: amplitude.min(cap),
                    omega,
                    axis,
                }
            }
            other => other,
        }
    }

    /// Per-seed material velocity `w_wall` (seed `i` == cell `i`) of the last
    /// committed step — zero for interior + static-boundary seeds, the rigid
    /// wall velocity for moving-wall seeds. The `MovingWall` BC reads this to set
    /// the Dirichlet wall velocity per boundary face.
    pub fn w_wall(&self) -> &[[f64; 2]] {
        &self.w_wall
    }

    /// Enable feeding the moving wall's material velocity into the fluid
    /// (opt-in; default off). With this on AND a
    /// [`BoundaryMotionSpec::RigidLoop`] declared, each regenerated mesh's
    /// moving-loop open faces are re-tagged [`BoundaryType::MovingWall`] and
    /// their per-face Dirichlet velocity `bc_value` is set to `w_wall[owner]`
    /// after the ALE refresh, so the fluid satisfies no-slip AND no-penetration
    /// at the wall's material velocity (`U_wall = w_wall`; the convective part
    /// already sees the relative velocity `phi − ρ·mesh_flux` from the ALE path
    /// — this is the other half, the Dirichlet wall value). Inert under `Static`
    /// (no moving seeds ⇒ no faces tagged) and byte-neutral for a run that never
    /// enables it.
    pub fn set_moving_wall_bc(&mut self, enable: bool) {
        self.moving_wall_bc = enable;
    }

    /// Disable per-step regeneration: `step` becomes a pure `SolverDriver::step`
    /// passthrough (no seed advection, no regen, no ALE refresh) — byte-identical
    /// to a static ALE run. Only meaningful with `Frozen` motion.
    pub fn set_regen_each_step(&mut self, regen: bool) {
        self.regen_each_step = regen;
    }

    /// Advance one moving-mesh step; returns the solver outcome + the per-step
    /// moving-mesh telemetry.
    pub fn step(&mut self, readback: bool) -> Result<(StepOutcome, MovingMeshStats), String> {
        let out = self.step_inner(readback)?;
        // The COMMITTED pinned dt — the flow-adaptive controller's
        // growth-limit reference (discarded implicit-motion attempts re-pin
        // from the same committed value, so retries stay deterministic).
        self.last_pinned_dt = Some(out.1.dt);
        Ok(out)
    }

    fn step_inner(&mut self, readback: bool) -> Result<(StepOutcome, MovingMeshStats), String> {
        // Re-assert the dt handshake invariant on EVERY path: the skip-regen
        // passthrough never reaches an ALE seam, so a caller that enabled
        // `adaptive_dt` via `driver_mut().apply_params(..)` after build could
        // otherwise silently step on an adaptive dt (ignoring the pinned value)
        // with no seam to catch it. Guard here, up front, for both paths.
        if self.driver.params().adaptive_dt {
            return Err(
                "MovingMeshDriver::step: adaptive_dt was enabled after build — the swept mesh \
                 fluxes are SCL-closed against the driver-pinned fixed dt, and an adaptive \
                 re-scale silently violates the GCL. Keep adaptive_dt == false."
                    .into(),
            );
        }

        // Skip-regen variant: pin the fixed dt and step. No seed motion, no
        // regen, no ALE refresh — a pure passthrough, byte-identical to a static
        // ALE run driven by `SolverDriver::step`.
        if !self.regen_each_step {
            let dt = self.pin_dt(0.0);
            let outcome = self.driver.step(readback);
            self.step_index += 1;
            let (vol_min, vol_max) = vol_extremes(&self.mesh.cell_vol);
            let stats = MovingMeshStats {
                plan_ms: 0.0,
                regen_ms: 0.0,
                swept_ms: 0.0,
                refresh_ms: 0.0,
                scl_defect: 0.0,
                identity_err: 0.0,
                max_skew: self.mesh.calculate_max_skewness(),
                n_cells: self.mesh.num_cells(),
                n_faces: self.mesh.num_faces(),
                n_vertices: self.mesh.num_vertices(),
                vol_min,
                vol_max,
                topo_changed: false,
                flipped: false,
                born_faces: 0,
                died_faces: 0,
                flipped_cells: 0,
                flip_defect: 0.0,
                dt,
                regen_backend: RegenBackend::Skipped,
                recycled: 0,
                cells_born: 0,
                cells_killed: 0,
                at_adapt_budget: false,
                motion_iters: 1,
                transfer_defect_pre: 0.0,
                transfer_defect_post: 0.0,
            };
            return Ok((outcome, stats));
        }

        // Periodic memory reordering (a pure relabel between steps — zero
        // mesh motion, zero physics; the regen below reproduces the same
        // diagram in the new slot order and the topology seam rebuilds the
        // face-indexed stacks).
        if self.reorder_every_n > 0
            && self.step_index > 0
            && self.step_index % self.reorder_every_n == 0
        {
            self.reorder_cells()?;
        }

        // Flow-adaptive sizing event (BETWEEN steps, before any seed motion):
        // derive per-cell target volumes from the flow gradients, plan the
        // interior birth/kill lists AND the wall-segment subdivisions (the
        // boundary discretization adapts along), and execute them as ONE
        // resize event. The step then proceeds at the new count with a
        // freshly rebuilt, state-transferred solver.
        let (mut cells_born, mut cells_killed) = (0usize, 0usize);
        let mut transfer_defect = (0.0f64, 0.0f64);
        if self.adapt_fires_this_step() {
            let targets = self.adapt_target_vols()?;
            let (kills, births) = self.plan_adaptation(&targets)?;
            let wall_segs = self.plan_wall_refinement(&targets);
            if !kills.is_empty() || !births.is_empty() || !wall_segs.is_empty() {
                let wall_born = self.resize_cells_impl(&kills, &births, &wall_segs)?;
                cells_born = births.len() + wall_born;
                cells_killed = kills.len();
                transfer_defect = self.last_transfer_projection;
                // Re-derive the per-cell target cache at the NEW indexing
                // (the resize invalidated it) — the per-cell squeeze
                // reference for chi_size and the escalation.
                self.adapt_targets = Some(self.adapt_target_vols()?);
            } else {
                self.adapt_targets = Some(targets);
            }
        }

        // IMPLICIT (fixed-point) mesh motion — FlowCoupled + CPU backend +
        // opt-in (`set_implicit_mesh_motion`): iterate {advect with the
        // previous attempt's end-of-step velocity → regen → ALE solve} from
        // a byte-exactly rewound t^n state, until the planned seed set
        // converges or the attempt cap. Every attempt is a fully
        // GCL-consistent ALE step; discarded attempts leave no trace (the
        // full-history CPU snapshot rewinds state, time levels, warm start,
        // volume history AND current volumes). The adaptation event above
        // deliberately stays OUTSIDE the loop: a resize rebuilds the
        // solver, which would invalidate the checkpoint.
        let outer = if self.motion_outer_iters > 1
            && matches!(self.motion, MeshMotionSpec::FlowCoupled { .. })
            && self.driver.solver().is_cpu()
        {
            self.motion_outer_iters
        } else {
            1
        };
        if outer > 1 {
            let checkpoint = self.motion_checkpoint();
            let tol = self.motion_outer_tol * self.min_cell_size;
            let mut prev_planned: Option<Vec<Point2<f64>>> = None;
            let mut iters_done = 0usize;
            loop {
                iters_done += 1;
                let plan = self.plan_step()?;
                // Fixed-point residual: how far THIS attempt's planned seed
                // set moved from the previous attempt's. Converged ⇒ this
                // attempt's mesh is (within tol) the one its own end-state
                // velocity would produce — solve it and accept.
                let converged = prev_planned.as_ref().is_some_and(|prev| {
                    plan.new_seeds
                        .iter()
                        .zip(prev.iter())
                        .all(|(a, b)| (a - b).norm() <= tol)
                });
                let planned = plan.new_seeds.clone();
                let mut out = self.step_cpu_planned(plan, readback, None)?;
                out.1.cells_born = cells_born;
                out.1.cells_killed = cells_killed;
                out.1.at_adapt_budget = self.adapt_budget_reached();
                out.1.motion_iters = iters_done;
                out.1.transfer_defect_pre = out.1.transfer_defect_pre.max(transfer_defect.0);
                out.1.transfer_defect_post = out.1.transfer_defect_post.max(transfer_defect.1);
                if converged || iters_done >= outer || out.0.diverged.is_some() {
                    self.motion_u_override = None;
                    return Ok(out);
                }
                // Rewind to t^n and retry, advecting with the velocities the
                // discarded attempt ENDED with (the implicit coupling).
                self.motion_u_override = Some(self.read_cell_velocities()?);
                self.motion_restore(&checkpoint)?;
                prev_planned = Some(planned);
            }
        }

        // Shared step plan: pinned dt, advected seeds, moved boundary spec,
        // quality escalation, recorded wall velocity — identical for the CPU
        // and device paths.
        let plan = self.plan_step()?;

        // Opt-in on-device regen: build the mesh + swept fluxes ENTIRELY on
        // the GPU. A step the device cannot certify (Voronoi flip /
        // sub-tolerance sliver / unresolved cell) falls back to the CPU path
        // below FOR THIS STEP — the device is retried next step — with the
        // reason surfaced in `MovingMeshStats::regen_backend`.
        let mut fallback: Option<&'static str> = None;
        if self.gpu_regen_active() {
            match self.try_step_device(&plan, readback)? {
                DeviceStep::Done(mut out) => {
                    out.1.cells_born = cells_born;
                    out.1.cells_killed = cells_killed;
                    out.1.at_adapt_budget = self.adapt_budget_reached();
                    out.1.transfer_defect_pre = out.1.transfer_defect_pre.max(transfer_defect.0);
                    out.1.transfer_defect_post = out.1.transfer_defect_post.max(transfer_defect.1);
                    return Ok(out);
                }
                DeviceStep::Fallback(reason) => fallback = Some(reason),
            }
        }

        let mut out = self.step_cpu_planned(plan, readback, fallback)?;
        out.1.cells_born = cells_born;
        out.1.cells_killed = cells_killed;
        out.1.at_adapt_budget = self.adapt_budget_reached();
        out.1.transfer_defect_pre = out.1.transfer_defect_pre.max(transfer_defect.0);
        out.1.transfer_defect_post = out.1.transfer_defect_post.max(transfer_defect.1);
        Ok(out)
    }

    /// Everything an implicit-mesh-motion retry must rewind: the full
    /// solver snapshot (state × time levels, warm start, volume history +
    /// current volumes, mesh fluxes, dt bookkeeping) plus the driver's own
    /// step-committed fields.
    fn motion_checkpoint(&self) -> MotionCheckpoint {
        MotionCheckpoint {
            snap: self.driver.snapshot(),
            seeds: self.seeds.clone(),
            mesh: self.mesh.clone(),
            prev_vx: self.prev_vx.clone(),
            prev_vy: self.prev_vy.clone(),
            w_wall: self.w_wall.clone(),
            time: self.time,
            step_index: self.step_index,
            last_escalated: self.last_escalated,
        }
    }

    /// Rewind a discarded implicit-motion attempt (see
    /// [`Self::motion_checkpoint`]). The solver's face-indexed stacks keep
    /// the discarded attempt's topology — the NEXT attempt's
    /// `begin_ale_step_topology` rebuilds them from the restored t^n mesh,
    /// and the restored CURRENT volumes make its history rotation exact.
    fn motion_restore(&mut self, c: &MotionCheckpoint) -> Result<(), String> {
        self.driver.restore(&c.snap)?;
        self.seeds = c.seeds.clone();
        self.mesh = c.mesh.clone();
        self.prev_vx = c.prev_vx.clone();
        self.prev_vy = c.prev_vy.clone();
        self.w_wall = c.w_wall.clone();
        self.time = c.time;
        self.step_index = c.step_index;
        self.last_escalated = c.last_escalated;
        Ok(())
    }

    /// The shared pre-regen work of one moving step (both paths): pin the
    /// fixed dt (mesh-motion CFL capped) BEFORE the swept fluxes are closed —
    /// so the flux dt == the step dt exactly — advect the interior seeds, move
    /// the boundary-bound seeds RIGIDLY with the moving boundary at t^{n+1},
    /// build the t^{n+1} boundary spec, run the FlowCoupled quality
    /// escalation, and record the per-seed material velocity `w_wall` while
    /// `self.seeds` still holds the t^n set (the `MovingWall` Dirichlet BC
    /// applied later feeds THIS step's solve a wall velocity consistent with
    /// the mesh motion swept during [t^n, t^{n+1}]).
    fn plan_step(&mut self) -> Result<StepPlan, String> {
        let plan_start = Instant::now();
        let (dt, mut new_seeds, recycled) = self.plan_seed_motion()?;
        let new_time = self.time + dt;
        self.apply_boundary_motion(new_time, &mut new_seeds);
        // The boundary spec at t^{n+1}: the moved loops the regen clips against
        // (an identity clone of `self.spec` under Static ⇒ byte-identical regen).
        let step_spec = self.moved_spec(new_time);
        let new_seeds = self.maybe_periodic_smooth(&step_spec, new_seeds);
        let (mut new_seeds, escalated) = self.maybe_quality_escalate(&step_spec, new_seeds);
        // HOLE containment barrier — after EVERY seed-motion stage
        // (advection/steering, recycling, smoothing, escalation): an interior
        // seed may not end the step inside an embedded boundary loop.
        self.contain_seeds(&step_spec, &mut new_seeds);
        let plan_ms = ms_since(plan_start);
        self.record_wall_velocity(&new_seeds, dt);
        Ok(StepPlan {
            dt,
            new_time,
            new_seeds,
            step_spec,
            escalated,
            plan_ms,
            recycled,
        })
    }

    /// The CPU regen + swept step body, from an already-computed [`StepPlan`].
    /// `gpu_fallback` carries the device-rejection reason when this step is a
    /// per-step GPU→CPU fallback (stats-only — the physics is identical).
    fn step_cpu_planned(
        &mut self,
        plan: StepPlan,
        readback: bool,
        gpu_fallback: Option<&'static str>,
    ) -> Result<(StepOutcome, MovingMeshStats), String> {
        let StepPlan {
            dt,
            new_time,
            new_seeds,
            step_spec,
            escalated,
            plan_ms,
            recycled,
        } = plan;

        // A device-committed t^n mesh is vertex-less, but the CPU swept path
        // aligns old vertices through the t^n mesh's `cell_vertices` rings.
        // Re-assemble the t^n mesh from the authoritative seeds (deterministic).
        //
        // Volumes: the rebuilt mesh keeps its own CPU f64 volumes — the
        // degeneracy-only flip closure hard-asserts the telescoping identity
        // against `self.mesh.cell_vol`, and the CPU swept quads telescope to
        // the CPU ring volumes, so patching device volumes in here would blow
        // that assert by the f64-vs-f32 volume gap (~1e-7 rel). The DEVICE
        // volumes (what the solver's volume history actually holds) are
        // stashed separately and used as the closure target only on a GENUINE
        // flip, where the identity check is relaxed and the forest closure is
        // the sole guarantee — there the GCL then closes EXACTLY against the
        // solver-held V^n. On non-flip / degeneracy-only fallback steps the
        // solver-held V^n differs from the CPU closure target by the same
        // O(f32) gap for ONE step — the device path's own accepted floor.
        //
        // The SOLVER also still holds the DEVICE face ordering, which the CPU
        // assembler's deterministic emission does not reproduce — so this step
        // must take the TOPOLOGY seam even when the CPU-side face arrays are
        // unchanged between the rebuilt t^n mesh and the new one
        // (`rebuilt_from_device` below); the geometry seam would reject the
        // order mismatch.
        let mut rebuilt_from_device = false;
        let mut device_vols: Option<Vec<f64>> = None;
        if self.mesh.vx.is_empty() && self.mesh.num_cells() > 0 {
            rebuilt_from_device = true;
            device_vols = Some(self.mesh.cell_vol.clone());
            let spec_n = self.moved_spec(self.time);
            let mut rebuilt = assemble_meshless_from_seeds(
                &self.seeds,
                &self.kinds,
                &spec_n,
                self.domain,
                self.min_cell_size,
            );
            if let Some(retag) = self.boundary_retag {
                retag(&mut rebuilt);
            }
            self.retag_moving_wall_faces(&mut rebuilt);
            self.mesh = rebuilt;
        }

        // 3. Regenerate the mesh from the advected seeds (deterministic; a
        //    frozen seed set reproduces `self.mesh` byte-for-byte).
        let regen_start = Instant::now();
        let mut new_mesh = assemble_meshless_from_seeds(
            &new_seeds,
            &self.kinds,
            &step_spec,
            self.domain,
            self.min_cell_size,
        );
        // Re-stamp boundary tags (the topology seam rebuilds bc tables from
        // these) before anything downstream reads them. face_boundary only —
        // geometry/adjacency untouched.
        if let Some(retag) = self.boundary_retag {
            retag(&mut new_mesh);
        }
        // Re-tag the moving loop's open faces MovingWall (from the engine's
        // default Wall) so the ALE seam builds a MovingWall face list to receive
        // the wall velocity below. face_boundary only. No-op unless
        // moving_wall_bc is on.
        self.retag_moving_wall_faces(&mut new_mesh);
        let regen_ms = ms_since(regen_start);
        if new_mesh.num_cells() != self.mesh.num_cells() {
            return Err(format!(
                "MovingMeshDriver: regen changed the cell count ({} -> {}); v1 is fixed-seed \
                 (cells change shape, not existence)",
                self.mesh.num_cells(),
                new_mesh.num_cells()
            ));
        }
        // Whether the regenerated face ARRAYS (order and/or adjacency) differ
        // from the previous mesh. Seed motion can REORDER the deterministic face
        // emission without changing the adjacency, so this is NOT a flip flag —
        // it just decides geometry-vs-topology seam below. The genuine-flip
        // discriminator is the VERTEX correspondence: a born vertex — an
        // incident-seed set with no t^n counterpart — is precisely what the
        // swept-quad path cannot close.
        let face_arrays_differ = topology_differs(&self.mesh, &new_mesh);

        // 4. Swept-quad mesh fluxes old→new with the pinned dt. Across the
        //    Voronoi regen the new mesh's vertex ids are unrelated to the old
        //    mesh's, so we first map each NEW vertex to its t^n position via the
        //    seed-set correspondence (vertex ≡ seed-triple).
        //
        //    Then we detect ADJACENCY flips (born/dead faces). Two regimes:
        //    * NO flip (persistent adjacency, at worst a face-array reorder): the
        //      aligned old ring reproduces each cell's t^n polygon, so the
        //      f64-telescoping + f32-forest-closure path applies directly and its
        //      telescoping identity is HARD-asserted.
        //    * FLIP (born/dead faces): the born faces have no swept quad, so we
        //      take the flip-aware path — born faces carry zero swept
        //      contribution and the per-cell defect they (and the born-vertex
        //      partial sweeps) leave is distributed onto the slack faces by the
        //      SAME spanning-forest closure, keeping `Σ_f σ·flux = ΔV_i/dt`
        //      EXACT per cell (⇒ GCL survives the flip). `unmatched > 0` (a born
        //      vertex) always coincides with a flip and forces this path even in
        //      the rare case where the adjacency scan alone would miss it.
        let swept_start = Instant::now();
        let (old_vx_aligned, old_vy_aligned, unmatched) =
            align_old_vertices_by_seed_set(&self.mesh, &new_mesh)?;
        let mut flip = detect_flips(&self.mesh, &new_mesh)?;
        // A GENUINE flip — an actual adjacency change (born/died face) or a born
        // vertex — captured BEFORE the degeneracy forcing below. On a genuine flip
        // the born faces legitimately break the telescoping identity and the
        // closure is the sole guarantee. On a step that is NOT a genuine flip but
        // has a sliver face, the identity is still load-bearing everywhere else, so
        // we keep the hard check alive there (see the `hard_assert_exclude` arg).
        // A recycle step is BY CONSTRUCTION a flip: the recycled cell's whole
        // adjacency teleported (removed at the outlet + inserted at the inlet
        // sharing a slot).
        let genuine_flip = flip.is_flip() || unmatched != 0 || !recycled.is_empty();
        // Robustness: a face that is geometrically DEGENERATE (near-zero length at
        // t^n or t^{n+1}) is a collapsing/near-flip face the adjacency scan did
        // not flag — it has no reliable swept quad and the persistent path would
        // hard-error on it. Mark such faces born so the conservative flip closure
        // absorbs their (negligible) swept volume onto the slack face per cell.
        let degen = force_degenerate_faces_born(
            &new_mesh,
            &old_vx_aligned,
            &old_vy_aligned,
            &mut flip,
            self.min_cell_size,
        );
        let mut is_flip = genuine_flip || degen > 0;
        // Closure old-volume target. Base: the actual t^n cell volumes
        // (`self.mesh` is still the old mesh here) — the born vertices'
        // aligned old positions cannot reconstruct the old polygon across a
        // flip, so the ring-reconstructed old volume would be wrong. On a
        // GENUINE flip after a device-committed step, target the DEVICE
        // volumes the solver actually holds (the identity check is relaxed
        // there, so the f64-vs-f32 gap cannot trip it) — the closure then
        // repairs the GCL exactly against the solver's V^n. RECYCLED cells'
        // target is ΔV = 0 (their volume history is re-zeroed to the new
        // volume after the seam): no mesh flux may connect the outlet cell
        // they were to the inlet cell they become.
        let base_old_vols: &[f64] = match (&device_vols, genuine_flip) {
            (Some(dv), true) => dv,
            _ => &self.mesh.cell_vol,
        };
        let old_vols_owned: Option<Vec<f64>> = if recycled.is_empty() {
            None
        } else {
            let mut v = base_old_vols.to_vec();
            for &i in &recycled {
                v[i] = new_mesh.cell_vol[i];
            }
            Some(v)
        };
        let old_vols: &[f64] = old_vols_owned.as_deref().unwrap_or(base_old_vols);
        let strict = if is_flip {
            // Degeneracy-ONLY step (no genuine adjacency change): keep the >1e-9
            // telescoping-identity assert LIVE on every cell not incident to a
            // forced-degenerate face — a lone sliver must not disable the
            // whole-step guard. On a genuine flip, pass None (relaxed).
            let hard_assert_exclude = if genuine_flip {
                None
            } else {
                Some(flip.cell_flipped.as_slice())
            };
            swept_mesh_fluxes_closed_flip(
                &new_mesh,
                &old_vx_aligned,
                &old_vy_aligned,
                old_vols,
                &flip.born_face_mask,
                hard_assert_exclude,
                dt,
            )
        } else {
            swept_mesh_fluxes_closed(&new_mesh, &old_vx_aligned, &old_vy_aligned, dt)
        };
        let swept = match strict {
            Ok(s) => s,
            // CONTINUOUS FLIP recovery: a face can pass through zero length
            // WITHIN the step (endpoints swap; adjacency and endpoint lengths
            // both look healthy), which no discriminator can see — the
            // adjacency scan compares sets, the degeneracy forcing only sees
            // the t^n/t^{n+1} endpoints. The swept quads of the affected
            // cells are then genuinely inconsistent and the telescoping
            // identity FAILS — but that is precisely the event class the
            // flip-aware closure exists for: re-run the step as a flip
            // (identity relaxed, born faces as detected, GCL guaranteed
            // per-cell by the forest closure against the actual t^n volumes;
            // the residual is reported as `flip_defect`). Any other error
            // (bad inputs, periodic mesh, invalid dt) propagates unchanged.
            Err(e) if e.contains("telescoping identity") => {
                is_flip = true;
                swept_mesh_fluxes_closed_flip(
                    &new_mesh,
                    &old_vx_aligned,
                    &old_vy_aligned,
                    old_vols,
                    &flip.born_face_mask,
                    None,
                    dt,
                )?
            }
            Err(e) => return Err(e),
        };
        let swept_ms = ms_since(swept_start);

        // 5. Refresh (rotate volume history → rebuild/upload geometry → upload
        //    closed fluxes). SEAM CHOICE:
        //    * Face arrays BYTE-IDENTICAL (a frozen regen): the surgical
        //      GEOMETRY seam re-uploads geometry while keeping the AMG hierarchy
        //      + per-face BC overrides — a zero-motion step is byte-identical to
        //      a static run.
        //    * Face arrays differ (ANY real motion reorders the face emission):
        //      the geometry seam would upload against a stale face order, so we
        //      take the CPU-surgical TOPOLOGY seam (rebuilds the face-indexed
        //      CSR stack for the new order; cell-indexed state incl. the
        //      warm-start `x` and BDF2 history is preserved). The rebuild
        //      re-scatters bc tables from the model per-type defaults, dropping
        //      per-face overrides, so we re-apply them (`bc_overrides_reset`).
        //    A genuine flip changes adjacency ⇒ `face_arrays_differ` is already
        //    true; the explicit `is_flip` guard makes the coupling defensive.
        let refresh_start = Instant::now();
        if face_arrays_differ || is_flip || self.force_topology_seam || rebuilt_from_device {
            let report = self
                .driver
                .begin_ale_step_topology(&new_mesh, &swept.fluxes)?;
            if report.bc_overrides_reset {
                self.driver.reapply_boundary_conditions();
            }
        } else {
            self.driver.begin_ale_step(&new_mesh, &swept.fluxes)?;
        }
        // Feed the moving wall's material velocity into the fluid. AFTER the
        // refresh (the seam just rebuilt the MovingWall face list + BC tables)
        // and BEFORE the solve, set each MovingWall face's Dirichlet velocity to
        // w_wall[owner] — no-slip + no-penetration at the wall's material
        // velocity. No-op unless moving_wall_bc is on.
        self.apply_moving_wall_velocity(&new_mesh)?;
        // Re-seed the RECYCLED cells as fresh parcels — state + every time
        // level + volume history + warm-start x — from the nearest surviving
        // cell's state row (model-agnostic: copies whatever the state layout
        // carries — U/p, psi/rho/T for the all-Mach families). AFTER the seam
        // (so the re-zeroed volume history survives the rotation), BEFORE the
        // solve.
        let mut recycle_defect = (0.0f64, 0.0f64);
        if !recycled.is_empty() {
            // A recycled slot is a FRESH parcel: its sizing-persistence
            // history belongs to the dead outlet cell, not the respawn.
            for &i in &recycled {
                if let Some(c) = self.adapt_persist.get_mut(i) {
                    *c = 0;
                }
            }
            let state = pollster::block_on(self.driver.solver().read_state_f32());
            let stride = self.driver.solver().model().state_layout.stride() as usize;
            let recycled_set: std::collections::HashSet<usize> = recycled.iter().cloned().collect();
            let mut cells: Vec<u32> = Vec::with_capacity(recycled.len());
            let mut rows: Vec<f32> = Vec::with_capacity(recycled.len() * stride);
            let mut vols: Vec<f64> = Vec::with_capacity(recycled.len());
            for &i in &recycled {
                let p = new_seeds[i];
                let (mut best, mut best_d2) = (usize::MAX, f64::INFINITY);
                for (j, q) in new_seeds.iter().enumerate() {
                    if recycled_set.contains(&j) {
                        continue;
                    }
                    let d2 = (q - p).norm_squared();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = j;
                    }
                }
                cells.push(i as u32);
                rows.extend_from_slice(&state[best * stride..(best + 1) * stride]);
                vols.push(new_mesh.cell_vol[i]);
            }
            self.driver.solver().reinit_cells(&cells, &rows, &vols)?;
            // MASS-ROW PROJECTION of the recycle transfer: the zeroth-order
            // neighbor copy leaves a per-parcel defect in the continuity row
            // at the inlet re-seed site; the next step answers it with a
            // pressure speckle that the |grad p| adaptation indicator then
            // chases (births at the noise). Same disease and same cure as
            // the resize transfer — the target is masked to the parcels'
            // neighborhood and applied history-preserving. CPU backend only
            // (a recycle step already runs the CPU regen path; the
            // GPU-solver variant would need a mid-run assembly mirror no
            // seam provides).
            #[cfg(feature = "cpu")]
            if self.transfer_projection {
                if let Some(layout) = self.projection_layout() {
                    if let Some(cpu) = self.driver.solver_mut().cpu_solver_mut() {
                        match super::mass_projection::project_recycled_state(
                            cpu, &new_mesh, layout, &recycled, &vols,
                        ) {
                            Ok(o) if o.pre > 0.0 => recycle_defect = (o.pre, o.post),
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("moving-mesh: recycle projection skipped: {e}")
                            }
                        }
                    }
                }
            }
        }
        let refresh_ms = ms_since(refresh_start);
        let topo_changed = face_arrays_differ || is_flip || rebuilt_from_device;

        // 6. Step.
        let outcome = self.driver.step(readback);

        // Commit the new mesh as the current realized mesh; its vertices become
        // next step's `old` positions.
        let max_skew = new_mesh.calculate_max_skewness();
        let n_cells = new_mesh.num_cells();
        let n_faces = new_mesh.num_faces();
        let n_vertices = new_mesh.num_vertices();
        let (vol_min, vol_max) = vol_extremes(&new_mesh.cell_vol);
        self.prev_vx = new_mesh.vx.clone();
        self.prev_vy = new_mesh.vy.clone();
        self.mesh = new_mesh;
        // (w_wall was recorded before the solve — see above — so the MovingWall
        // BC fed this step's solve the correct wall velocity.)
        self.seeds = new_seeds;
        self.time = new_time;
        self.step_index += 1;
        self.last_escalated = escalated;

        // On a flip step the pre-closure per-cell residual is the intended flip
        // defect, not a telescoping-identity violation, so it is reported via
        // `flip_defect` and `identity_err` is left at 0 (the persistent identity
        // is not enforced across a flip). On a non-flip step it is the roundoff
        // telescoping residual, reported as `identity_err` and asserted by the
        // GCL gates.
        let stats = MovingMeshStats {
            plan_ms,
            regen_ms,
            swept_ms,
            refresh_ms,
            scl_defect: swept.max_defect_rel,
            identity_err: if is_flip {
                0.0
            } else {
                swept.max_identity_err_rel
            },
            max_skew,
            n_cells,
            n_faces,
            n_vertices,
            vol_min,
            vol_max,
            topo_changed,
            flipped: is_flip,
            born_faces: flip.born_faces,
            died_faces: flip.died_faces,
            flipped_cells: flip.flipped_cells,
            flip_defect: if is_flip {
                swept.max_identity_err_rel
            } else {
                0.0
            },
            dt,
            regen_backend: gpu_fallback
                .map(RegenBackend::GpuFallback)
                .unwrap_or(RegenBackend::Cpu),
            recycled: recycled.len(),
            cells_born: 0,
            cells_killed: 0,
            at_adapt_budget: false,
            motion_iters: 1,
            transfer_defect_pre: recycle_defect.0,
            transfer_defect_post: recycle_defect.1,
        };
        Ok((outcome, stats))
    }

    /// The opt-in ON-DEVICE regen attempt: the mesh is rebuilt ENTIRELY on the
    /// GPU (Voronoi + topology + geometry + ALE swept fluxes) via
    /// [`GpuMeshRegen`] — no CPU `assemble_meshless_from_seeds`, no CPU swept
    /// path. The motion planning is shared ([`StepPlan`]) and the
    /// solver-driving seam (`begin_ale_step_topology` + BC reapply + step) is
    /// the SAME as the CPU path; only the mesh source differs.
    ///
    /// Non-flip only: the device swept assumes a persistent adjacency, so a
    /// Voronoi flip (interior adjacency changed vs the committed t^n mesh —
    /// derived FRESH from `self.mesh` every attempt, so the check holds on the
    /// first device step and across CPU-path interludes), a sub-tolerance
    /// sliver, or a resolve-unresolved cell (`needs_cpu`) returns
    /// [`DeviceStep::Fallback`] and the caller re-runs the step on the CPU
    /// path (which handles flips natively).
    fn try_step_device(&mut self, plan: &StepPlan, readback: bool) -> Result<DeviceStep, String> {
        use std::collections::HashSet;

        let dt = plan.dt;
        let new_seeds = &plan.new_seeds;

        // A recycle step is a guaranteed topology flip (the recycled cell's
        // whole adjacency teleports) AND needs the per-cell solver reinit the
        // CPU path performs — skip the device build outright.
        if !plan.recycled.is_empty() {
            return Ok(DeviceStep::Fallback("seed recycle"));
        }

        // Lazy-init the GPU context (reusing the solver device/queue) + the regen
        // bundle (engine + passes) at the fixed seed count.
        let n = self.seeds.len();
        if self.gpu_ctx.is_none() {
            let ctx = pollster::block_on(crate::solver::gpu::context::GpuContext::new(
                self.device.clone(),
                self.queue.clone(),
            ))
            .map_err(|e| format!("gpu_regen: GPU context init failed: {e}"))?;
            self.gpu_ctx = Some(ctx);
        }
        if self.gpu_regen_state.is_none() {
            let tol = MeshgenTolerances::from_geometry(self.min_cell_size, self.domain);
            let device = &self.gpu_ctx.as_ref().unwrap().device;
            self.gpu_regen_state = Some(crate::solver::gpu::voronoi::GpuMeshRegen::new(
                device,
                n,
                self.domain,
                &tol,
            ));
        }

        // 3-4. Regenerate the solver mesh + ALE fluxes ON DEVICE, clipping
        //      against the t^{n+1} boundary spec (an identity clone of
        //      `self.spec` under Static boundary motion).
        let regen_start = Instant::now();
        let old_f32: Vec<f32> = self
            .seeds
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32])
            .collect();
        let new_f32: Vec<f32> = new_seeds
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32])
            .collect();
        let dt_f = self.driver.params().requested_dt;
        let old_spec = self.moved_spec(self.time);
        let result = {
            let ctx = self.gpu_ctx.as_ref().unwrap();
            let regen = self.gpu_regen_state.as_mut().unwrap();
            regen.regen(
                ctx,
                &new_f32,
                &old_f32,
                &self.kinds,
                &plan.step_spec,
                &old_spec,
                dt_f,
            )
        };
        let mut new_mesh = result.mesh;
        let regen_ms = ms_since(regen_start);

        if new_mesh.num_cells() != self.mesh.num_cells() {
            return Err(format!(
                "gpu_regen: regen changed the cell count ({} -> {})",
                self.mesh.num_cells(),
                new_mesh.num_cells()
            ));
        }
        // Sliver / unresolved cell ⇒ the device build is not certified for
        // this step; the CPU path takes it.
        if result.needs_cpu > 0 {
            return Ok(DeviceStep::Fallback("sliver/unresolved cell"));
        }
        // Topology discriminator vs the COMMITTED t^n mesh (derived fresh —
        // valid on the first device step and across CPU-path interludes; both
        // mesh sources carry owner/neighbor). Two components:
        // * interior adjacency set — a changed set is a Voronoi flip;
        // * per-cell OPEN-face count — a boundary (wall/box) face birth/death
        //   changes NO interior pair, yet the device swept would evaluate the
        //   t^n ring with a bracket that did not exist at t^n (wrong old
        //   vertex, silent GCL defect). Segment IDS need no comparison:
        //   boundary seeds move rigidly with their loop, so a face's segment
        //   assignment persists whenever the loop moves, and a static loop
        //   makes old and new segment tables identical.
        // Either change ⇒ the device swept quads are invalid ⇒ the CPU path
        // (which handles flips natively via the flip-aware closure) takes the
        // step.
        let topo_signature =
            |mesh: &crate::solver::mesh::Mesh| -> (HashSet<(usize, usize)>, Vec<u32>) {
                let mut open = vec![0u32; mesh.num_cells()];
                let mut adj = HashSet::new();
                for f in 0..mesh.num_faces() {
                    match mesh.face_neighbor[f] {
                        Some(nb) => {
                            let o = mesh.face_owner[f];
                            adj.insert((o.min(nb), o.max(nb)));
                        }
                        None => open[mesh.face_owner[f]] += 1,
                    }
                }
                (adj, open)
            };
        if topo_signature(&self.mesh) != topo_signature(&new_mesh) {
            return Ok(DeviceStep::Fallback("Voronoi flip"));
        }

        // Re-stamp boundary tags (the topology seam rebuilds bc tables from them).
        if let Some(retag) = self.boundary_retag {
            retag(&mut new_mesh);
        }
        self.retag_moving_wall_faces(&mut new_mesh);

        // The device swept telescopes to the device cell-volume change; the solver
        // GCL defect is |Σσ·flux·dt − (V^{n+1} − V^n)| with V^n = the CURRENT mesh
        // volumes the solver holds (`self.mesh.cell_vol`, cell i == cell i).
        let mut scl_defect = 0.0f64;
        for i in 0..n {
            let (fb, fe) = (
                new_mesh.cell_face_offsets[i],
                new_mesh.cell_face_offsets[i + 1],
            );
            let mut s = 0.0f64;
            for &f in &new_mesh.cell_faces[fb..fe] {
                let sgn = if new_mesh.face_owner[f] == i {
                    1.0
                } else {
                    -1.0
                };
                s += sgn * result.mesh_fluxes[f] as f64 * dt;
            }
            let dv = new_mesh.cell_vol[i] - self.mesh.cell_vol[i];
            scl_defect =
                scl_defect.max((s - dv).abs() / new_mesh.cell_vol[i].max(f64::MIN_POSITIVE));
        }

        // 5. Refresh (always the topology seam — the device face order changes
        //    every regen) + reapply runtime BC overrides the refresh drops.
        let refresh_start = Instant::now();
        let report = self
            .driver
            .begin_ale_step_topology(&new_mesh, &result.mesh_fluxes)?;
        if report.bc_overrides_reset {
            self.driver.reapply_boundary_conditions();
        }
        self.apply_moving_wall_velocity(&new_mesh)?;
        let refresh_ms = ms_since(refresh_start);

        // 6. Step.
        let outcome = self.driver.step(readback);

        let n_cells = new_mesh.num_cells();
        let n_faces = new_mesh.num_faces();
        let n_vertices = new_mesh.num_vertices();
        let (vol_min, vol_max) = vol_extremes(&new_mesh.cell_vol);
        // Skewness is vertex-free (cell centres + face normals only), so the
        // vertex-less device mesh reports it like the CPU path does.
        let max_skew = new_mesh.calculate_max_skewness();
        // Commit. The device mesh is vertex-less (`prev_vx/vy` stay empty — the
        // device swept needs no old vertices; a CPU-fallback step re-assembles
        // the t^n mesh from the authoritative seeds).
        self.mesh = new_mesh;
        self.prev_vx.clear();
        self.prev_vy.clear();
        self.seeds = plan.new_seeds.clone();
        self.time = plan.new_time;
        self.step_index += 1;
        self.last_escalated = plan.escalated;

        let stats = MovingMeshStats {
            plan_ms: plan.plan_ms,
            regen_ms,
            swept_ms: 0.0,
            refresh_ms,
            scl_defect,
            identity_err: scl_defect,
            max_skew,
            n_cells,
            n_faces,
            n_vertices,
            vol_min,
            vol_max,
            // The device path always drives the topology seam (its face order
            // changes every regen).
            topo_changed: true,
            flipped: false,
            born_faces: 0,
            died_faces: 0,
            flipped_cells: 0,
            flip_defect: 0.0,
            dt,
            regen_backend: RegenBackend::GpuOnDevice,
            recycled: 0,
            cells_born: 0,
            cells_killed: 0,
            at_adapt_budget: false,
            motion_iters: 1,
            transfer_defect_pre: 0.0,
            transfer_defect_post: 0.0,
        };
        Ok(DeviceStep::Done((outcome, stats)))
    }

    /// The seed positions at absolute time `t`, evaluated from the t=0 labels
    /// `f(seed0_i, t)` for interior seeds; boundary seeds are held fixed.
    /// Frozen advances from the CURRENT seeds (t is irrelevant): a stationary
    /// mesh has no analytic law to re-sample, and holding the current set —
    /// rather than the t=0 labels — lets a scheduled smooth or a quality
    /// escalation PERSIST (label-based advection would revert it next step).
    /// Identical to the label set whenever nothing has moved the seeds.
    fn advect_to(&self, t: f64) -> Vec<Point2<f64>> {
        match self.motion {
            MeshMotionSpec::Frozen | MeshMotionSpec::FlowCoupled { .. } => self.seeds.clone(),
            MeshMotionSpec::Prescribed(f) => self
                .seeds0
                .iter()
                .zip(&self.kinds)
                .map(|(s0, kind)| {
                    if *kind == SeedKind::Interior {
                        let p = f([s0.x, s0.y], t);
                        Point2::new(p[0], p[1])
                    } else {
                        // Boundary seeds fixed here (moving boundaries handled by
                        // BoundaryMotionSpec) ⇒ boundary faces static, zero flux.
                        *s0
                    }
                })
                .collect(),
        }
    }

    /// The max interior-seed speed `max_i |w_i|` over the upcoming base step,
    /// a finite-difference estimate `|f(seed0,t+dt) − f(seed0,t)|/dt` used only
    /// to size the mesh-motion CFL cap (conservative — the cap has its own 0.2
    /// safety factor). Zero for Frozen/FlowCoupled.
    fn max_seed_speed(&self, dt_base: f64) -> f64 {
        let f = match self.motion {
            MeshMotionSpec::Prescribed(f) => f,
            _ => return 0.0,
        };
        if dt_base <= 0.0 {
            return 0.0;
        }
        let mut w_max = 0.0f64;
        for (s0, kind) in self.seeds0.iter().zip(&self.kinds) {
            if *kind != SeedKind::Interior {
                continue;
            }
            let a = f([s0.x, s0.y], self.time);
            let b = f([s0.x, s0.y], self.time + dt_base);
            let w = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() / dt_base;
            w_max = w_max.max(w);
        }
        w_max
    }

    // Moving boundary: rigidly-moving boundary seeds + moved loop spec.

    /// The `[lo, hi)` global-segment range of the moving loop, or `None` under
    /// `Static`. A boundary seed whose adjacent segment falls in this range is a
    /// moving-wall seed.
    fn moving_loop_range(&self) -> Option<(usize, usize)> {
        self.boundary_motion.loop_index().map(|loop_index| {
            (
                self.spec.seg_offsets[loop_index],
                self.spec.seg_offsets[loop_index + 1],
            )
        })
    }

    /// Whether seed `i` is a boundary seed bound to the moving loop.
    fn is_moving_boundary_seed(&self, i: usize) -> bool {
        match (self.moving_loop_range(), self.kinds[i]) {
            (Some((lo, hi)), SeedKind::Boundary { seg_next, .. }) => {
                let s = seg_next as usize;
                s >= lo && s < hi
            }
            _ => false,
        }
    }

    /// The boundary spec at absolute time `t`: the t=0 label spec with the
    /// moving loop's polyline points rigidly transformed. Under `Static` this is
    /// a byte-identical clone of `self.spec`, so the regen reproduces the mesh
    /// byte-for-byte. A rigid transform preserves chord
    /// lengths, so segment tags/structure are unchanged — only the points move.
    fn moved_spec(&self, t: f64) -> BoundarySpec {
        self.moved_spec_from(&self.spec, t)
    }

    /// [`Self::moved_spec`] against an EXPLICIT base spec: a cell-count
    /// resize applies wall subdivisions to a WORKING copy of the label spec
    /// and needs the moved variant of THAT copy for the assembly (seg ids in
    /// the updated kinds index the working spec, not `self.spec`).
    fn moved_spec_from(&self, base: &BoundarySpec, t: f64) -> BoundarySpec {
        let mut spec = base.clone();
        if let Some(loop_index) = self.boundary_motion.loop_index() {
            for p in spec.loops[loop_index].pts.iter_mut() {
                let q = self.boundary_motion.eval(t, [p.x, p.y]);
                *p = Point2::new(q[0], q[1]);
            }
        }
        spec
    }

    /// Overwrite the moving loop's boundary seeds with their rigidly-transformed
    /// t=0 positions at absolute time `t` (interior + static-boundary seeds
    /// untouched). Evaluating from the t=0 label `seed0_i` keeps the seed exactly
    /// on the moving wall with no incremental drift. A no-op under `Static`.
    fn apply_boundary_motion(&self, t: f64, seeds: &mut [Point2<f64>]) {
        if matches!(self.boundary_motion, BoundaryMotionSpec::Static) {
            return;
        }
        for i in 0..seeds.len() {
            if self.is_moving_boundary_seed(i) {
                let s0 = self.seeds0[i];
                let q = self.boundary_motion.eval(t, [s0.x, s0.y]);
                seeds[i] = Point2::new(q[0], q[1]);
            }
        }
    }

    /// Max moving-boundary-seed speed over the upcoming base step, a
    /// finite-difference estimate `|w(t+dt) − w(t)|/dt` used only to size the
    /// mesh-motion CFL cap. Zero under `Static`.
    fn max_boundary_speed(&self, dt_base: f64) -> f64 {
        if matches!(self.boundary_motion, BoundaryMotionSpec::Static) {
            return 0.0;
        }
        if dt_base <= 0.0 {
            return 0.0;
        }
        let mut w_max = 0.0f64;
        for i in 0..self.seeds0.len() {
            if !self.is_moving_boundary_seed(i) {
                continue;
            }
            let s0 = [self.seeds0[i].x, self.seeds0[i].y];
            let a = self.boundary_motion.eval(self.time, s0);
            let b = self.boundary_motion.eval(self.time + dt_base, s0);
            let w = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() / dt_base;
            w_max = w_max.max(w);
        }
        w_max
    }

    /// Record the per-seed material velocity `w_wall = (new − old)/dt` for the
    /// committed step (seed `i` == cell `i`) — the rigid wall velocity on moving
    /// boundary seeds, zero elsewhere. Called with the OLD `self.seeds` still in
    /// place. `dt > 0` by the CFL cap; guarded anyway.
    fn record_wall_velocity(&mut self, new_seeds: &[Point2<f64>], dt: f64) {
        let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        for i in 0..self.w_wall.len() {
            if self.is_moving_boundary_seed(i) {
                self.w_wall[i] = [
                    (new_seeds[i].x - self.seeds[i].x) * inv_dt,
                    (new_seeds[i].y - self.seeds[i].y) * inv_dt,
                ];
            } else {
                self.w_wall[i] = [0.0, 0.0];
            }
        }
    }

    /// Re-tag the moving loop's open (boundary) faces as
    /// [`BoundaryType::MovingWall`] on a freshly regenerated mesh. An open face
    /// whose OWNER cell is a moving-boundary seed lies on the moving obstacle
    /// contour (the boundary seeds sit ON the wall; their cells' only open face
    /// is the clipped obstacle edge), so this catches exactly the moving wall
    /// while leaving the fixed channel walls / inlet / outlet untouched. Only
    /// `face_boundary` is rewritten (geometry/adjacency untouched ⇒ swept fluxes
    /// and the topology-diff unaffected). A no-op unless `moving_wall_bc` is on
    /// and a `RigidLoop` is declared. Must run BEFORE the ALE seam so the solver
    /// builds a `MovingWall` boundary-face list to receive the velocity.
    fn retag_moving_wall_faces(&self, mesh: &mut Mesh) {
        self.retag_moving_wall_faces_with(mesh, &self.kinds);
    }

    /// [`Self::retag_moving_wall_faces`] against an EXPLICIT per-seed kind
    /// array: during a cell-count resize the freshly assembled mesh is indexed
    /// by the NEW seed order while `self.kinds` still holds the old one, so
    /// the moving-wall test must read the caller's array.
    fn retag_moving_wall_faces_with(&self, mesh: &mut Mesh, kinds: &[SeedKind]) {
        if !self.moving_wall_bc {
            return;
        }
        let Some((lo, hi)) = self.moving_loop_range() else {
            return;
        };
        for f in 0..mesh.num_faces() {
            if mesh.face_neighbor[f].is_none() {
                if let SeedKind::Boundary { seg_next, .. } = kinds[mesh.face_owner[f]] {
                    let s = seg_next as usize;
                    if s >= lo && s < hi {
                        mesh.face_boundary[f] = Some(BoundaryType::MovingWall);
                    }
                }
            }
        }
    }

    /// Push the per-face Dirichlet wall velocity into the solver for the current
    /// step. Each `MovingWall` open face gets `bc_value = w_wall` of its owner
    /// cell (the rigid wall material velocity — uniform across the face under the
    /// pure-translation scope; a rotating wall would need a per-face `w_wall`).
    /// Re-applied EVERY step: the topology seam re-scatters the bc tables from
    /// the model's per-type defaults (MovingWall Dirichlet 0) dropping the
    /// per-face override, and `w_wall` itself changes each step. A no-op unless
    /// `moving_wall_bc` is on. Called AFTER the refresh (so the `MovingWall` face
    /// list + tables are rebuilt) and BEFORE the solve (so this step feels this
    /// step's wall velocity).
    fn apply_moving_wall_velocity(&mut self, mesh: &Mesh) -> Result<(), String> {
        if !self.moving_wall_bc || self.moving_loop_range().is_none() {
            return Ok(());
        }
        // Index against `mesh` — the just-refreshed NEW mesh whose face ordering
        // the solver's MovingWall face list matches (`self.mesh` is still the t^n
        // mesh at this point). Precompute per-face x/y wall velocity into owned
        // locals so the closures do not borrow `self` while `solver_mut()` does.
        let nf = mesh.num_faces();
        let (mut vx, mut vy) = (vec![0.0f32; nf], vec![0.0f32; nf]);
        for f in 0..nf {
            if mesh.face_neighbor[f].is_none()
                && matches!(mesh.face_boundary[f], Some(BoundaryType::MovingWall))
            {
                let owner = mesh.face_owner[f];
                vx[f] = self.w_wall[owner][0] as f32;
                vy[f] = self.w_wall[owner][1] as f32;
            }
        }
        let solver = self.driver.solver_mut();
        // Field name is "U" (upper) in the ALE model; fall back to "u" to match
        // `set_inlet_velocity`'s upper/lower convention.
        set_moving_wall_component(solver, 0, &vx)?;
        set_moving_wall_component(solver, 1, &vy)?;
        Ok(())
    }

    /// Pin the fixed dt and produce the advected seed set for the current
    /// [`MeshMotionSpec`]. This is the whole dt-handshake + advection front of a
    /// step, factored so the regen→swept→refresh→step tail is motion-agnostic.
    fn plan_seed_motion(&mut self) -> Result<(f64, Vec<Point2<f64>>, Vec<usize>), String> {
        match self.motion {
            // Frozen: no INTERIOR motion. `advect_to` returns the labels; the dt
            // is capped only by the moving BOUNDARY speed, zero otherwise — so a
            // fully static step pins exactly `requested_dt`.
            MeshMotionSpec::Frozen => {
                let w_max = self.max_boundary_speed(self.configured_dt);
                let dt = self.pin_dt(w_max);
                Ok((dt, self.advect_to(self.time + dt), Vec::new()))
            }
            // Prescribed: FD-estimate max seed speed over the base step, cap dt,
            // then sample the analytic law at the pinned t^{n+1}. The moving
            // boundary speed also enters the cap.
            MeshMotionSpec::Prescribed(_) => {
                let dt_base = self.configured_dt;
                let w_max = self
                    .max_seed_speed(dt_base)
                    .max(self.max_boundary_speed(dt_base));
                let dt = self.pin_dt(w_max);
                Ok((dt, self.advect_to(self.time + dt), Vec::new()))
            }
            MeshMotionSpec::FlowCoupled { regularization } => {
                self.plan_flow_coupled(regularization)
            }
        }
    }

    /// FlowCoupled seed motion: read the current cell velocities, pin dt off the
    /// flow speed (mesh-motion CFL), then displace each interior seed by `U_i·dt`
    /// plus an AREPO-style distortion-ramped steering toward its cell centroid,
    /// clamped to a fraction of the local cell radius.
    ///
    /// * The steering `χ_i·(c_i − s_i)` is a partial (under-relaxed) Lloyd move.
    ///   `χ_i` ramps from 0 (well-shaped cell — pure flow advection, so the mesh
    ///   follows the flow and cuts advective dissipation) up to `regularization`
    ///   once the seed-to-centroid offset exceeds `η·R_i`. This holds quality
    ///   WITHOUT overwhelming advection.
    /// * The per-step displacement clamp `|Δ| ≤ flow_disp_cap·R_i` is the hard
    ///   anti-tangling guard (keeps the swept-quad + flip-remap regime valid).
    /// * The mesh-motion CFL cap on dt is sized from the flow speed; the steering
    ///   displacement is bounded independently by the clamp, so it never violates
    ///   the GCL (fluxes are geometric, closed against exactly this pinned dt).
    /// * **Open-boundary anti-pileup**: the ADVECTION part ramps to zero within
    ///   [`FLOW_ADVECT_BOX_RAMP_CELLS`]`·min_cell` of the domain box. Boundary
    ///   guard seeds are fixed, so flow THROUGH an inlet/outlet would otherwise
    ///   pile the advected interior seeds against them without bound (measured:
    ///   outlet-band cell volumes shrank 7× over 300 steps, collapsing the
    ///   mesh-motion dt cap). Near the box the mesh is ~Eulerian; the wake /
    ///   obstacle region (interior loops are NOT the box) keeps full advection,
    ///   and the steering stays active everywhere (quality hold). No-slip walls
    ///   are unaffected in practice (u ≈ 0 there anyway).
    fn plan_flow_coupled(
        &mut self,
        chi_max: f64,
    ) -> Result<(f64, Vec<Point2<f64>>, Vec<usize>), String> {
        use std::f64::consts::PI;
        // Advection velocity: the implicit-motion override (the previous
        // attempt's END-of-step velocities) when live, else the solver's
        // current (t^n) state — the explicit coupling.
        let u = match &self.motion_u_override {
            Some(u) if u.len() == self.mesh.num_cells() => u.clone(),
            _ => self.read_cell_velocities()?,
        };
        let dead_len = FLOW_ADVECT_BOX_DEAD_CELLS * self.min_cell_size;
        let ramp_len = ((FLOW_ADVECT_BOX_RAMP_CELLS - FLOW_ADVECT_BOX_DEAD_CELLS)
            * self.min_cell_size)
            .max(1e-30);
        let domain = self.domain;
        // With seed RECYCLING active the x (through-flow) sides must NOT park
        // seeds: they advect at full speed to the recycle line and relabel to
        // the inlet — the dead zone would otherwise stall them just short of
        // the trigger and recycling would never fire (measured: 0 recycles in
        // 2000 steps), while freshly spawned inlet seeds would sit at ramp=0
        // and never leave. The wall (y) ramp stays either way (inert for
        // no-slip walls where u ≈ 0, protective for slip walls).
        let through_flow = self.seed_recycling;
        let box_ramp = move |s: &Point2<f64>| -> f64 {
            let mut d = s.y.min(domain.y - s.y);
            if !through_flow {
                d = d.min(s.x).min(domain.x - s.x);
            }
            ((d - dead_len) / ramp_len).clamp(0.0, 1.0)
        };
        // dt handshake: cap off the max EFFECTIVE advection speed (the ramped
        // velocity each seed actually moves with).
        let mut w_flow_max = 0.0f64;
        for (i, &(ux, uy)) in u.iter().enumerate() {
            if self.kinds[i] != SeedKind::Interior {
                continue;
            }
            let w = (ux * ux + uy * uy).sqrt() * box_ramp(&self.seeds[i]);
            w_flow_max = w_flow_max.max(w);
        }
        // The moving boundary speed also enters the dt cap.
        let w_max = w_flow_max.max(self.max_boundary_speed(self.configured_dt));
        let dt = self.pin_dt(w_max);

        let eta = self.arepo_eta;
        let cap_frac = self.flow_disp_cap;
        // Per-cell squeeze reference: the cell's OWN target when the
        // adaptation cache is live (a wide explicit band makes the global
        // hold floor blind to compression — an adapted-fine cell and a
        // squeezed cell have the same volume); the hold-band floor
        // otherwise.
        let hold_lo = QUALITY_VOL_BAND_LO * self.hold_vol_band().0;
        let squeeze_lo = |i: usize| -> f64 {
            match &self.adapt_targets {
                Some(t) if t.len() == self.seeds.len() => QUALITY_VOL_BAND_LO * t[i],
                _ => hold_lo,
            }
        };
        let mut new_seeds = self.seeds.clone();
        for i in 0..self.seeds.len() {
            if self.kinds[i] != SeedKind::Interior {
                continue; // boundary seeds fixed
            }
            let s = self.seeds[i];
            let ramp = box_ramp(&s);
            let (ux, uy) = (u[i].0 * ramp, u[i].1 * ramp);
            // Effective cell radius (uniform-density Lloyd target is the geometric
            // centroid, already stored on the mesh as cell_cx/cell_cy).
            let r = (self.mesh.cell_vol[i].max(0.0) / PI).sqrt().max(1e-30);
            let dcx = self.mesh.cell_cx[i] - s.x;
            let dcy = self.mesh.cell_cy[i] - s.y;
            let dist = (dcx * dcx + dcy * dcy).sqrt();
            // AREPO ramp on the distortion ratio d/R.
            let ratio = dist / r;
            let chi_shape = if ratio < 0.9 * eta {
                0.0
            } else if ratio < 1.1 * eta {
                chi_max * (ratio - 0.9 * eta) / (0.2 * eta)
            } else {
                chi_max
            };
            // SIZING ramp: a cell squeezed below its reference volume
            // (persistent advective compression — e.g. converging streamlines
            // against fixed boundary guards) escalates its centroid pull
            // toward FULL Lloyd weight (χ=1), regardless of `chi_max`. The
            // shape ramp cannot see this (a uniformly squeezed cell stays
            // centroidal), and the global blended escalation is too gentle
            // to balance a steady seed inflow. χ ramps 0→1 as the volume
            // falls from `LO·reference` to half that; the hard displacement
            // clamp below keeps the move flip-safe. Reference = the cell's
            // own adaptation target when live, else the hold-band floor.
            let lo = squeeze_lo(i);
            let vol = self.mesh.cell_vol[i];
            let chi_size = if vol >= lo {
                0.0
            } else {
                ((lo - vol) / (0.5 * lo)).clamp(0.0, 1.0)
            };
            let chi = chi_shape.max(chi_size);
            let mut dx = ux * dt + chi * dcx;
            let mut dy = uy * dt + chi * dcy;
            // Hard anti-tangling clamp.
            let disp = (dx * dx + dy * dy).sqrt();
            let cap = cap_frac * r;
            if disp > cap && disp > 1e-300 {
                let scale = cap / disp;
                dx *= scale;
                dy *= scale;
            }
            new_seeds[i] = Point2::new(s.x + dx, s.y + dy);
        }
        // Outflow -> inflow seed recycling (relabel-in-place; inert when
        // disabled). Runs AFTER advection/steering/clamp so the trigger sees
        // the step's final positions; the teleport is not advection, so it
        // does not enter the mesh-CFL dt cap above.
        let recycled = if self.seed_recycling {
            self.recycle_seeds(&mut new_seeds)
        } else {
            Vec::new()
        };
        Ok((dt, new_seeds, recycled))
    }

    /// Relabel every interior seed that crossed into the OUTLET dead zone
    /// (x > domain.x − dead) to the largest FLUID gap along the inlet band —
    /// the mesh analogue of fluid leaving the outflow and entering the
    /// inflow. The slot keeps its index (fixed-seed contract); the caller
    /// re-seeds the cell's solver rows after the ALE seam. A respawn keeps
    /// the Poisson spacing discipline ([`RECYCLE_MIN_HALFGAP_CELLS`]); when
    /// no admissible gap exists the seed simply stays parked in the dead
    /// zone until one opens.
    fn recycle_seeds(&self, seeds: &mut [Point2<f64>]) -> Vec<usize> {
        use crate::meshgen::meshless::point_in_fluid;
        let h = self.min_cell_size;
        let strip_x = self.domain.x - RECYCLE_TRIGGER_CELLS * h;
        let park_x = self.domain.x - FLOW_ADVECT_BOX_DEAD_CELLS * h;
        let vol_lo = self.recycle_vol_floor;

        // DENSITY trigger: interior seeds in the outlet strip whose CURRENT
        // cell (seed i == cell i, t^n mesh) is squeezed below the strip's
        // own initial sizing — being compressed out, most-squeezed first.
        // ADAPTATION-AWARE: the build-time strip reference goes stale the
        // moment the flow-adaptive sizing targets volumes below it — a
        // deliberately-fine adapted cell advecting into the strip is NOT
        // being squeezed (vol ~ its own target), but the stale reference
        // read it as one, and the resulting permanent recycle storm
        // (measured 429/500 steps under every-step default-band adaptation)
        // fed the outlet-band churn loop that slowly diverged the run. The
        // per-cell floor is the FINER of the strip reference and
        // `RECYCLE_SQUEEZE_FRACTION x` the cell's own adapted target —
        // byte-identical without adaptation (no targets cached), and the
        // same per-cell-target pattern that cured the chi_size squeeze
        // misread in the leak arc.
        let targets = self.adapt_targets.as_deref();
        let squeeze_floor = |i: usize| -> f64 {
            match targets.and_then(|t| t.get(i)) {
                Some(&t_vol) if t_vol.is_finite() && t_vol > 0.0 => {
                    vol_lo.min(RECYCLE_SQUEEZE_FRACTION * t_vol)
                }
                _ => vol_lo,
            }
        };
        let mut out: Vec<usize> = (0..seeds.len())
            .filter(|&i| {
                self.kinds[i] == SeedKind::Interior
                    && seeds[i].x > strip_x
                    && self.mesh.cell_vol[i] < squeeze_floor(i)
            })
            .collect();
        out.sort_by(|&a, &b| self.mesh.cell_vol[a].total_cmp(&self.mesh.cell_vol[b]));
        out.truncate(RECYCLE_MAX_PER_STEP);

        // Anti-guard-squeeze backstop for everyone else: advection is
        // full-speed on the through-flow sides when recycling is on, so a
        // seed that presses past the park line (1.5 cells from the outlet
        // guards) is clamped there — squeezing then shows up as shrinking
        // volumes IN the strip, which is exactly the recycle trigger.
        for i in 0..seeds.len() {
            if self.kinds[i] == SeedKind::Interior && !out.contains(&i) && seeds[i].x > park_x {
                seeds[i].x = park_x;
            }
        }
        if out.is_empty() {
            return Vec::new();
        }

        // Spawn placement: the deepest 2D hole over a candidate grid spanning
        // the inlet band (x ∈ [1.5h, 3h], y at h/4), scored by distance to
        // the nearest occupant. 2D distance, not a y-projection gap — the
        // x=0 inlet guards' own ~h spacing caps any projected gap regardless
        // of real holes. The admissibility floor is BELOW the fresh-CVT
        // deepest hole (~0.58h), so a spawn is always possible and the
        // intake rate matches the export rate by construction; the sizing
        // hold relaxes the temporarily tight neighborhood within steps.
        let out_set: std::collections::HashSet<usize> = out.iter().cloned().collect();
        let band_lim = (RECYCLE_TRIGGER_CELLS + 1.0) * h;
        let mut occupants: Vec<Point2<f64>> = (0..seeds.len())
            .filter(|&i| seeds[i].x < band_lim && !out_set.contains(&i))
            .map(|i| seeds[i])
            .collect();

        let min_sep = RECYCLE_MIN_SEP_CELLS * h;
        let cand_step = 0.25 * h;
        let x_levels = [1.5 * h, 2.0 * h, 2.5 * h, 3.0 * h];
        let n_cand = (self.domain.y / cand_step).floor() as usize;
        let mut recycled = Vec::new();
        for &i in &out {
            let mut best: Option<(f64, Point2<f64>)> = None;
            for &cx in &x_levels {
                for k in 1..n_cand {
                    let p = Point2::new(cx, k as f64 * cand_step);
                    if !point_in_fluid(p, &self.spec) {
                        continue;
                    }
                    let d2 = occupants
                        .iter()
                        .map(|q| (q - p).norm_squared())
                        .fold(f64::INFINITY, f64::min);
                    if best.map_or(true, |(b, _)| d2 > b) {
                        best = Some((d2, p));
                    }
                }
            }
            match best {
                Some((d2, p)) if d2.sqrt() >= min_sep => {
                    seeds[i] = p;
                    occupants.push(p);
                    recycled.push(i);
                }
                _ => {
                    // No admissible hole (pathological): park instead.
                    seeds[i].x = seeds[i].x.min(park_x);
                }
            }
        }
        recycled
    }

    /// Read the current per-cell velocity `(u_x, u_y)` (f64-widened f32) from the
    /// solver state — the FlowCoupled seed-velocity source ("cell velocity from
    /// the last readback"). Cheap on CPU (clones the f32 store).
    fn read_cell_velocities(&self) -> Result<Vec<(f64, f64)>, String> {
        let layout = &self.driver.solver().model().state_layout;
        let stride = layout.stride() as usize;
        let u_off = layout
            .offset_for("U")
            .ok_or("MovingMeshDriver::FlowCoupled: model has no U field")?
            as usize;
        let state = pollster::block_on(self.driver.solver().read_state_f32());
        let n = self.mesh.num_cells();
        if state.len() != n * stride {
            return Err(format!(
                "MovingMeshDriver: state length {} != n_cells*stride {}",
                state.len(),
                n * stride
            ));
        }
        Ok((0..n)
            .map(|c| {
                (
                    state[c * stride + u_off] as f64,
                    state[c * stride + u_off + 1] as f64,
                )
            })
            .collect())
    }

    /// Relabel the cells in Morton (z-curve) order of the CURRENT seed
    /// positions: permute the driver's per-seed arrays and the committed
    /// mesh's cell-indexed arrays in place, and gather the solver's
    /// cell-indexed stores through the [`UnifiedSolver::permute_cells`] seam.
    /// A no-op when the current order is already Morton.
    fn reorder_cells(&mut self) -> Result<(), String> {
        let n = self.seeds.len();
        // Morton key: 16-bit quantized x/y, bits interleaved.
        let key = |p: &Point2<f64>| -> u64 {
            let qx = ((p.x / self.domain.x).clamp(0.0, 1.0) * 65535.0) as u64;
            let qy = ((p.y / self.domain.y).clamp(0.0, 1.0) * 65535.0) as u64;
            let mut z = 0u64;
            for b in 0..16 {
                z |= ((qx >> b) & 1) << (2 * b) | ((qy >> b) & 1) << (2 * b + 1);
            }
            z
        };
        // gather[new] = old, stable by (key, old id) for determinism.
        let mut gather: Vec<usize> = (0..n).collect();
        gather.sort_by_key(|&i| (key(&self.seeds[i]), i));
        if gather.iter().enumerate().all(|(i, &s)| i == s) {
            return Ok(());
        }
        let mut inv = vec![0usize; n];
        for (new, &old) in gather.iter().enumerate() {
            inv[old] = new;
        }

        // Driver per-seed arrays.
        let take =
            |v: &Vec<Point2<f64>>| -> Vec<Point2<f64>> { gather.iter().map(|&s| v[s]).collect() };
        self.seeds = take(&self.seeds);
        self.seeds0 = take(&self.seeds0);
        self.kinds = gather.iter().map(|&s| self.kinds[s]).collect();
        self.w_wall = gather.iter().map(|&s| self.w_wall[s]).collect();
        if self.adapt_persist.len() == n {
            self.adapt_persist = gather.iter().map(|&s| self.adapt_persist[s]).collect();
        }
        if let Some(t) = &self.adapt_targets {
            self.adapt_targets = Some(gather.iter().map(|&s| t[s]).collect());
        }

        // Committed mesh: cell-indexed arrays gather; face arrays keep their
        // ids but owner/neighbor cell ids remap through the inverse.
        let m = &mut self.mesh;
        let gather_f64 = |v: &[f64]| -> Vec<f64> { gather.iter().map(|&s| v[s]).collect() };
        m.cell_cx = gather_f64(&m.cell_cx);
        m.cell_cy = gather_f64(&m.cell_cy);
        m.cell_vol = gather_f64(&m.cell_vol);
        let mut cell_faces = Vec::with_capacity(m.cell_faces.len());
        let mut cell_face_offsets = Vec::with_capacity(n + 1);
        cell_face_offsets.push(0usize);
        for &src in &gather {
            cell_faces.extend_from_slice(
                &m.cell_faces[m.cell_face_offsets[src]..m.cell_face_offsets[src + 1]],
            );
            cell_face_offsets.push(cell_faces.len());
        }
        m.cell_faces = cell_faces;
        m.cell_face_offsets = cell_face_offsets;
        if !m.cell_vertex_offsets.is_empty() {
            let mut cv = Vec::with_capacity(m.cell_vertices.len());
            let mut cvo = Vec::with_capacity(n + 1);
            cvo.push(0usize);
            for &src in &gather {
                cv.extend_from_slice(
                    &m.cell_vertices[m.cell_vertex_offsets[src]..m.cell_vertex_offsets[src + 1]],
                );
                cvo.push(cv.len());
            }
            m.cell_vertices = cv;
            m.cell_vertex_offsets = cvo;
        }
        for f in 0..m.face_owner.len() {
            m.face_owner[f] = inv[m.face_owner[f]];
            m.face_neighbor[f] = m.face_neighbor[f].map(|nb| inv[nb]);
        }

        // Solver cell-indexed stores.
        let gather_u32: Vec<u32> = gather.iter().map(|&s| s as u32).collect();
        self.driver.solver().permute_cells(&gather_u32)
    }

    /// RESIZE the cell count between steps (the variable-cell-count seam):
    /// remove the `kills` cells and insert one new cell per `births` entry
    /// `(position, donor)`. A **rebuild-and-gather** event, not an
    /// incremental edit:
    ///
    /// 1. the survivor seeds (slot order preserved) plus the birth positions
    ///    form the new seed set; a fresh mesh is assembled from it at the
    ///    current time's boundary spec;
    /// 2. the wrapped [`SolverDriver`] is REBUILT on that mesh at the new
    ///    count (same model, same params — `apply_params` restores the
    ///    phase-2 knobs and the params-derived BCs);
    /// 3. every new cell's FULL state row (all fields, all time levels,
    ///    volume history, warm-start x) is written through the
    ///    [`UnifiedSolver::reinit_cells`](crate::solver::UnifiedSolver::reinit_cells)
    ///    seam from its source row — a survivor keeps its own row, a birth
    ///    copies its donor's — so the run RESUMES, it does not restart.
    ///
    /// The resize happens at fixed simulated time (zero mesh motion): the
    /// volume history restarts at the new mesh's volumes on every level, so
    /// the next ALE step sees zero mesh-volume rate, and the equal rows at
    /// all time levels degrade the first post-resize ddt to backward-Euler
    /// for exactly one step (the recycling seam's proven behaviour).
    ///
    /// Kills must be interior cells; birth positions must lie in the fluid;
    /// birth donors must survive the event. `Prescribed` motion is rejected
    /// (its analytic law is indexed by the t=0 labels, which a resize
    /// re-anchors for the inserted cells). On `Err` before the commit point
    /// the driver is untouched; a failed solver rebuild propagates with the
    /// driver still on the OLD mesh/solver (the event simply did not happen).
    pub fn resize_cells(
        &mut self,
        kills: &[usize],
        births: &[(Point2<f64>, usize)],
    ) -> Result<(), String> {
        self.resize_cells_impl(kills, births, &[]).map(|_| ())
    }

    /// [`Self::resize_cells`] plus WALL refinement: subdivide the given
    /// boundary segments at their midpoints (geometry-EXACT — the polyline
    /// shape and the fluid area are unchanged; only the discretization
    /// refines), re-kind the affected wall seeds against the new segment
    /// ids, and birth the new wall seeds in the same rebuild-and-gather
    /// event. Returns the number of wall seeds born.
    fn resize_cells_impl(
        &mut self,
        kills: &[usize],
        births: &[(Point2<f64>, usize)],
        wall_segs: &[usize],
    ) -> Result<usize, String> {
        use crate::meshgen::meshless::point_in_fluid;
        use std::collections::HashSet;

        if matches!(self.motion, MeshMotionSpec::Prescribed(_)) {
            return Err(
                "MovingMeshDriver::resize_cells: Prescribed motion samples the analytic law \
                 from the t=0 seed labels, which a resize re-anchors — resize supports \
                 Frozen/FlowCoupled motion only."
                    .into(),
            );
        }
        let n = self.seeds.len();
        let kill_set: HashSet<usize> = kills.iter().cloned().collect();
        if kill_set.len() != kills.len() {
            return Err("resize_cells: duplicate kill indices".into());
        }
        for &i in kills {
            if i >= n {
                return Err(format!(
                    "resize_cells: kill index {i} out of range ({n} cells)"
                ));
            }
            if self.kinds[i] != SeedKind::Interior {
                return Err(format!("resize_cells: kill index {i} is a boundary seed"));
            }
        }
        let spec_t = self.moved_spec(self.time);
        for &(p, donor) in births {
            if donor >= n || kill_set.contains(&donor) {
                return Err(format!(
                    "resize_cells: birth donor {donor} is out of range or killed"
                ));
            }
            if !point_in_fluid(p, &spec_t) {
                return Err(format!(
                    "resize_cells: birth position ({}, {}) is outside the fluid",
                    p.x, p.y
                ));
            }
        }

        // WALL refinement: apply the segment subdivisions to a WORKING copy
        // of the label spec + kinds. Processed in DESCENDING segment order
        // so the pending (original-numbering) ids stay valid — each split
        // renumbers only the segments ABOVE it, and the generic remap inside
        // `split_wall_segment` keeps the already-updated kinds and the
        // already-emitted wall births consistent.
        let mut spec_new = self.spec.clone();
        let mut kinds_upd = self.kinds.clone();
        let mut wall_births: Vec<(Point2<f64>, usize, SeedKind)> = Vec::new();
        {
            let mut segs: Vec<usize> = wall_segs.to_vec();
            segs.sort_unstable_by(|x, y| y.cmp(x));
            segs.dedup();
            for &g in &segs {
                split_wall_segment(
                    &mut spec_new,
                    &self.seeds,
                    &mut kinds_upd,
                    &mut wall_births,
                    g,
                );
            }
        }

        // New seed set: survivors in slot order, then interior births, then
        // wall births. `src[new]` is the OLD cell whose state row seeds the
        // new cell (survivor: itself; birth: its donor). `seeds0` labels are
        // GATHERED, not reset: boundary seeds keep their t=0 labels (the
        // rigid boundary transform is evaluated absolutely from them); a
        // birth's label is its spawn position (wall refinement is restricted
        // to STATIC segments, so a wall birth's label is exact).
        let n_new = n - kills.len() + births.len() + wall_births.len();
        let mut new_seeds: Vec<Point2<f64>> = Vec::with_capacity(n_new);
        let mut new_seeds0: Vec<Point2<f64>> = Vec::with_capacity(n_new);
        let mut new_kinds: Vec<SeedKind> = Vec::with_capacity(n_new);
        let mut new_w_wall: Vec<[f64; 2]> = Vec::with_capacity(n_new);
        let mut src: Vec<usize> = Vec::with_capacity(n_new);
        // Persistence counters: survivors carry theirs (gathered like every
        // per-seed array); births start at 0 — a fresh cell must re-earn any
        // further adaptation event (the ping-pong breaker).
        let mut new_persist: Vec<i16> = Vec::with_capacity(n_new);
        let persist_of = |i: usize| self.adapt_persist.get(i).copied().unwrap_or(0);
        for i in 0..n {
            if kill_set.contains(&i) {
                continue;
            }
            new_seeds.push(self.seeds[i]);
            new_seeds0.push(self.seeds0[i]);
            new_kinds.push(kinds_upd[i]);
            new_w_wall.push(self.w_wall[i]);
            new_persist.push(persist_of(i));
            src.push(i);
        }
        for &(p, donor) in births {
            new_seeds.push(p);
            new_seeds0.push(p);
            new_kinds.push(SeedKind::Interior);
            new_w_wall.push([0.0, 0.0]);
            new_persist.push(0);
            src.push(donor);
        }
        for &(p, donor, kind) in &wall_births {
            new_seeds.push(p);
            new_seeds0.push(p);
            new_kinds.push(kind);
            new_w_wall.push([0.0, 0.0]);
            new_persist.push(0);
            src.push(donor);
        }

        // PRE-RELAX the new seed set (gentle local Lloyd toward the adapted
        // sizing) BEFORE assembling: a fresh child at its split position sits
        // far from CVT, and without this the FIRST post-resize smoothing/
        // escalation pass moves the patch violently — that one-step local
        // mesh deformation published as the residual 1-3-cell pressure mark
        // (the re-solve below absorbs the state/topology transition, but not
        // motion that happens on the NEXT step). Relaxing here folds the
        // settling INTO the resize event, where the re-solve absorbs it too;
        // the first-order transfer handles the resulting centroid shifts by
        // construction.
        let assemble_spec = self.moved_spec_from(&spec_new, self.time);
        if !births.is_empty() || !wall_births.is_empty() {
            let sizing = self.lloyd_sizing();
            let tol = MeshgenTolerances::from_geometry(self.min_cell_size, self.domain);
            let cfg = EngineConfig::default();
            let lcfg = LloydConfig {
                max_iters: 2,
                tol_disp: 0.0,
                omega: 0.7,
                density_exponent: 4.0,
            };
            lloyd_relax(
                &mut new_seeds,
                &new_kinds,
                &assemble_spec,
                &sizing,
                self.domain,
                &tol,
                &cfg,
                &lcfg,
            );
        }
        let mut new_mesh = assemble_meshless_from_seeds(
            &new_seeds,
            &new_kinds,
            &assemble_spec,
            self.domain,
            self.min_cell_size,
        );
        if let Some(retag) = self.boundary_retag {
            retag(&mut new_mesh);
        }
        self.retag_moving_wall_faces_with(&mut new_mesh, &new_kinds);
        if new_mesh.num_cells() != n_new {
            return Err(format!(
                "resize_cells: regen produced {} cells for {} seeds",
                new_mesh.num_cells(),
                n_new
            ));
        }

        // Gather the full state rows (all fields, packed by the state layout
        // — model-agnostic) through the old→new source map.
        let state = pollster::block_on(self.driver.solver().read_state_f32());
        let layout = &self.driver.solver().model().state_layout;
        let stride = layout.stride() as usize;
        if state.len() != n * stride {
            return Err(format!(
                "resize_cells: state length {} != n_cells*stride {}",
                state.len(),
                n * stride
            ));
        }
        let u_off = layout
            .offset_for("U")
            .ok_or("resize_cells: model has no U field")? as usize;
        let p_off = layout
            .offset_for("p")
            .ok_or("resize_cells: model has no p field")? as usize;
        // FIRST-ORDER transfer: each new cell's row is its source row plus
        // the source's Green–Gauss gradient times the centroid offset,
        // clamped to the source neighborhood's min/max (monotone — no new
        // extremum can be manufactured). A zeroth-order copy places the
        // donor's point value at a DIFFERENT location, and the elliptic
        // pressure reacts to that inconsistency with an unphysical spike;
        // linear reconstruction places the field's local expansion instead.
        // A survivor whose cell is geometrically unchanged has a zero
        // centroid offset — an exact copy, so a no-op resize stays a no-op.
        let (gsx, gsy, glo, ghi) = state_gradients_and_bounds(&self.mesh, &state, stride);
        let mut rows: Vec<f32> = Vec::with_capacity(n_new * stride);
        let mut init_u: Vec<(f64, f64)> = Vec::with_capacity(n_new);
        let mut init_p: Vec<f64> = Vec::with_capacity(n_new);
        for (c, &s) in src.iter().enumerate() {
            let dx = new_mesh.cell_cx[c] - self.mesh.cell_cx[s];
            let dy = new_mesh.cell_cy[c] - self.mesh.cell_cy[s];
            for k in 0..stride {
                let idx = s * stride + k;
                let v = state[idx] as f64 + gsx[idx] * dx + gsy[idx] * dy;
                rows.push((v as f32).clamp(glo[idx], ghi[idx]));
            }
            let row = &rows[c * stride..(c + 1) * stride];
            init_u.push((row[u_off] as f64, row[u_off + 1] as f64));
            init_p.push(row[p_off] as f64);
        }

        // Rebuild the wrapped solver at the new count and transfer the rows.
        let params = *self.driver.params();
        let build = pollster::block_on(SolverDriver::build(
            &new_mesh,
            self.model.clone(),
            &params,
            &init_u,
            &init_p,
            self.device.clone(),
            self.queue.clone(),
        ))?;
        let mut driver = build.driver;
        driver.apply_params(&params);
        let cells: Vec<u32> = (0..n_new as u32).collect();
        driver
            .solver()
            .reinit_cells(&cells, &rows, &new_mesh.cell_vol)?;

        // MASS-ROW TRANSFER PROJECTION: the interpolated rows carry a
        // residual defect in the solver's OWN continuity row, and the first
        // post-resize step balances whatever is left of it with a
        // `defect/dt` pressure response — the phantom dipoles at split/merge
        // sites. Project the transferred velocity onto the mass-row
        // constraint (least-norm through the momentum diagonal) BEFORE the
        // step; the correction is verified by re-assembly and reverted if it
        // does not measurably shrink the residual. A projection failure
        // never aborts the resize — the unprojected transfer is the
        // fallback behaviour.
        self.last_transfer_projection = (0.0, 0.0);
        #[cfg(feature = "cpu")]
        if self.transfer_projection {
            match self.project_transferred_rows(
                &mut driver,
                &new_mesh,
                &mut rows,
                &init_u,
                &init_p,
                &params,
            ) {
                Ok(Some(out)) => self.last_transfer_projection = (out.pre, out.post),
                Ok(None) => {}
                Err(e) => eprintln!("moving-mesh: transfer projection skipped: {e}"),
            }
        }

        // BDF CONTINUITY across the rebuild (CPU backend): the fresh solver
        // starts at step_count = 0 with flat history, so the ENTIRE field
        // takes a backward-Euler step on every resize event — under
        // every-step adaptation the integrator effectively never runs BDF2,
        // and the BDF1-vs-BDF2 solution difference concentrates at the
        // steepest transients (measured as recurring above-floor pressure
        // jumps on the obstacle arc FAR from any event — the "ambient
        // stragglers" of the dipole watch). Transfer the full time history
        // through the same first-order map and restore integrator
        // continuity (dt_old / step_count / time) via the snapshot seam.
        // The VOLUME history stays FLAT (zero mesh-rate on the rebuild
        // step, exactly like the reinit seam): a fresh cell has no
        // meaningful V^{n-1}, and flat volumes keep the GCL cancellation
        // intact — the STATE history is what BDF2's temporal order needs.
        // GPU keeps the flat-history reinit (its snapshot has no history).
        #[cfg(feature = "cpu")]
        if driver.solver().is_cpu() {
            let old_snap = self.driver.snapshot();
            if old_snap.has_history
                && old_snap.state_old.len() == n * stride
                && old_snap.state_old_old.len() == n * stride
            {
                let transfer_level = |src_level: &[f32]| -> Vec<f32> {
                    let (gx, gy, lo, hi) =
                        state_gradients_and_bounds(&self.mesh, src_level, stride);
                    let mut out = Vec::with_capacity(n_new * stride);
                    for (c, &s) in src.iter().enumerate() {
                        let dx = new_mesh.cell_cx[c] - self.mesh.cell_cx[s];
                        let dy = new_mesh.cell_cy[c] - self.mesh.cell_cy[s];
                        for k in 0..stride {
                            let idx = s * stride + k;
                            let v = src_level[idx] as f64 + gx[idx] * dx + gy[idx] * dy;
                            out.push((v as f32).clamp(lo[idx], hi[idx]));
                        }
                    }
                    out
                };
                let state_old = transfer_level(&old_snap.state_old);
                let state_old_old = transfer_level(&old_snap.state_old_old);
                let offsets: Vec<usize> = driver
                    .solver_mut()
                    .cpu_solver_mut()
                    .map(|c| {
                        c.unknown_state_offsets()
                            .iter()
                            .map(|&o| o as usize)
                            .collect()
                    })
                    .unwrap_or_default();
                let s_unk = offsets.len();
                let mut x = vec![0.0f32; n_new * s_unk];
                for c in 0..n_new {
                    for (r, &off) in offsets.iter().enumerate() {
                        x[c * s_unk + r] = rows[c * stride + off];
                    }
                }
                let vols_f32: Vec<f32> = new_mesh.cell_vol.iter().map(|&v| v as f32).collect();
                let snap = crate::solver::SolverStateSnapshot {
                    num_cells: n_new,
                    num_faces: new_mesh.num_faces(),
                    state_stride: stride as u32,
                    unknowns_per_cell: s_unk,
                    state: rows.clone(),
                    state_old: state_old.clone(),
                    state_old_old: state_old_old.clone(),
                    x,
                    cell_vols: vols_f32.clone(),
                    cell_vols_old: vols_f32.clone(),
                    cell_vols_old_old: vols_f32.clone(),
                    mesh_fluxes: vec![0.0; new_mesh.num_faces()],
                    time: old_snap.time,
                    dt: old_snap.dt,
                    dt_old: old_snap.dt_old,
                    dtau: old_snap.dtau,
                    step_count: old_snap.step_count,
                    last_rel_delta: old_snap.last_rel_delta,
                    schur_amg_active: old_snap.schur_amg_active,
                    has_history: true,
                };
                if let Err(e) = driver.restore(&snap) {
                    eprintln!("moving-mesh: resize BDF continuity skipped: {e}");
                } else {
                    // RESIZE RE-SOLVE: the published post-resize frame used
                    // to be the INTERPOLATED state — the first real step
                    // then carried the transition between the old and the
                    // new discretization as a visible 1-3-cell pressure
                    // mark at every re-meshed site (transfer-order
                    // independent; it is the difference between the two
                    // meshes' own solutions, not a state error). Absorb the
                    // transition BEFORE anything is published: rewind one
                    // level (state <- t^{n-1}, old <- t^{n-2}, time -= dt)
                    // and RE-SOLVE the same physical step on the NEW mesh —
                    // no extra physical time, the original dt (no 1/dt
                    // amplification: this is NOT the refuted dt/M
                    // settle-substeps), one extra solve per resize event.
                    // The state the next real step starts from is then the
                    // new mesh's OWN converged level-n solution. On any
                    // re-solve failure, fall back to the transferred state.
                    let mut x_prev = vec![0.0f32; n_new * s_unk];
                    for c in 0..n_new {
                        for (r, &off) in offsets.iter().enumerate() {
                            x_prev[c * s_unk + r] = state_old[c * stride + off];
                        }
                    }
                    let redo = crate::solver::SolverStateSnapshot {
                        num_cells: n_new,
                        num_faces: new_mesh.num_faces(),
                        state_stride: stride as u32,
                        unknowns_per_cell: s_unk,
                        state: state_old.clone(),
                        state_old: state_old_old.clone(),
                        state_old_old: state_old_old.clone(),
                        x: x_prev,
                        cell_vols: vols_f32.clone(),
                        cell_vols_old: vols_f32.clone(),
                        cell_vols_old_old: vols_f32,
                        mesh_fluxes: vec![0.0; new_mesh.num_faces()],
                        time: old_snap.time - old_snap.dt,
                        dt: old_snap.dt,
                        dt_old: old_snap.dt_old,
                        dtau: old_snap.dtau,
                        step_count: old_snap.step_count.saturating_sub(1),
                        last_rel_delta: old_snap.last_rel_delta,
                        schur_amg_active: old_snap.schur_amg_active,
                        has_history: true,
                    };
                    if driver.restore(&redo).is_ok() {
                        let outcome = driver.step(false);
                        if outcome.diverged.is_some() {
                            // Fall back to the transferred t^n state.
                            if let Err(e) = driver.restore(&snap) {
                                eprintln!("moving-mesh: resize re-solve fallback failed: {e}");
                            }
                        }
                    } else if let Err(e) = driver.restore(&snap) {
                        eprintln!("moving-mesh: resize re-solve fallback failed: {e}");
                    }
                }
            }
        }

        // Commit.
        self.driver = driver;
        self.prev_vx = new_mesh.vx.clone();
        self.prev_vy = new_mesh.vy.clone();
        self.mesh = new_mesh;
        self.seeds = new_seeds;
        self.seeds0 = new_seeds0;
        self.kinds = new_kinds;
        self.w_wall = new_w_wall;
        self.adapt_persist = new_persist;
        // The refined label spec (identical shape, subdivided segments).
        self.spec = spec_new;
        // The per-cell target cache is indexed by the OLD cells — invalid
        // now (the adapt event re-derives it right after the resize).
        self.adapt_targets = None;
        // The on-device regen bundle is sized at a fixed seed count —
        // rebuild it lazily at the new count on the next device attempt.
        self.gpu_regen_state = None;
        Ok(wall_births.len())
    }

    /// The [`super::mass_projection::ProjectionLayout`] of the wrapped
    /// solver's model; `None` when the model has no U/p system.
    #[cfg(feature = "cpu")]
    fn projection_layout(&self) -> Option<super::mass_projection::ProjectionLayout> {
        let l = &self.driver.solver().model().state_layout;
        let u_off = l.offset_for("U")?;
        l.offset_for("p")?;
        Some(super::mass_projection::ProjectionLayout {
            stride: l.stride() as usize,
            u_off: u_off as usize,
            rho_off: l.offset_for("rho").map(|o| o as usize),
            upwind_rho: l.offset_for("t_ref").is_some(),
        })
    }

    /// Run the mass-row transfer projection against a CPU assembly of the
    /// freshly built solver: in place on a CPU run; on a GPU run through a
    /// params-faithful throwaway CPU companion (built via the same
    /// `SolverDriver::build` path, so the assembled inlet/outlet closures
    /// match the production solver), with the corrected rows re-transferred
    /// into the GPU solver. `Ok(None)` = model without a U/p system.
    #[cfg(feature = "cpu")]
    fn project_transferred_rows(
        &self,
        driver: &mut SolverDriver,
        new_mesh: &Mesh,
        rows: &mut [f32],
        init_u: &[(f64, f64)],
        init_p: &[f64],
        params: &RuntimeParams,
    ) -> Result<Option<super::mass_projection::MassProjectionOutcome>, String> {
        use super::mass_projection::{project_transferred_state, ProjectionLayout};
        let layout = {
            let l = &driver.solver().model().state_layout;
            let Some(u_off) = l.offset_for("U") else {
                return Ok(None);
            };
            if l.offset_for("p").is_none() {
                return Ok(None);
            }
            ProjectionLayout {
                stride: l.stride() as usize,
                u_off: u_off as usize,
                rho_off: l.offset_for("rho").map(|o| o as usize),
                upwind_rho: l.offset_for("t_ref").is_some(),
            }
        };
        if driver.solver().is_cpu() {
            let cpu = driver
                .solver_mut()
                .cpu_solver_mut()
                .ok_or("transfer projection: CPU backend expected")?;
            return project_transferred_state(cpu, new_mesh, layout, rows, &new_mesh.cell_vol)
                .map(Some);
        }
        // GPU backend: the assembly seam is CPU-only, so project through a
        // throwaway CPU companion on the same mesh/model/params.
        let build = pollster::block_on(SolverDriver::build_forced_cpu(
            new_mesh,
            self.model.clone(),
            params,
            init_u,
            init_p,
        ))?;
        let mut tmp = build.driver;
        tmp.apply_params(params);
        let cells: Vec<u32> = (0..new_mesh.num_cells() as u32).collect();
        tmp.solver()
            .reinit_cells(&cells, rows, &new_mesh.cell_vol)?;
        let cpu = tmp
            .solver_mut()
            .cpu_solver_mut()
            .ok_or("transfer projection: build_forced_cpu returned a non-CPU backend")?;
        let out = project_transferred_state(cpu, new_mesh, layout, rows, &new_mesh.cell_vol)?;
        if out.applied {
            driver
                .solver()
                .reinit_cells(&cells, rows, &new_mesh.cell_vol)?;
        }
        Ok(Some(out))
    }

    /// The current boundary discretization size (total polyline segments
    /// across all loops) — grows when the wall refinement subdivides.
    pub fn boundary_segment_count(&self) -> usize {
        self.spec.num_segments()
    }

    /// Expanded bounding boxes of the spec's HOLE loops (negative signed
    /// area — embedded obstacles), the cheap prefilter for the containment
    /// tests (`point_in_fluid` walks every polyline point; a subdivided
    /// obstacle has hundreds).
    fn hole_bboxes(spec: &BoundarySpec, pad: f64) -> Vec<(f64, f64, f64, f64)> {
        spec.loops
            .iter()
            .filter(|lp| lp.signed_area() < 0.0)
            .map(|lp| {
                let (mut x0, mut y0, mut x1, mut y1) = (
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                );
                for p in &lp.pts {
                    x0 = x0.min(p.x);
                    y0 = y0.min(p.y);
                    x1 = x1.max(p.x);
                    y1 = y1.max(p.y);
                }
                (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            })
            .collect()
    }

    /// HOLE containment barrier: an INTERIOR seed whose planned position
    /// falls inside an embedded (hole) loop reverts to its current position
    /// — the wall polyline is a CLIP for the Voronoi cells, not a physical
    /// barrier for the seeds, and the stagnation-side advection presses
    /// seeds against the fixed obstacle guards until one slips through the
    /// chord line; once inside, nothing ever brings it back (observed as a
    /// growing cell colony filling the obstacle). A seed ALREADY outside
    /// the fluid (a pre-barrier leak) is left for the adaptation cleanup
    /// ([`Self::plan_adaptation`] kills it).
    fn contain_seeds(&self, spec_t: &BoundarySpec, new_seeds: &mut [Point2<f64>]) {
        use crate::meshgen::meshless::point_in_fluid;
        let boxes = Self::hole_bboxes(spec_t, self.min_cell_size);
        if boxes.is_empty() {
            return;
        }
        for i in 0..new_seeds.len() {
            if self.kinds[i] != SeedKind::Interior {
                continue;
            }
            let p = new_seeds[i];
            if p == self.seeds[i] {
                continue;
            }
            let near = boxes
                .iter()
                .any(|&(x0, y0, x1, y1)| p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1);
            if near && !point_in_fluid(p, spec_t) && point_in_fluid(self.seeds[i], spec_t) {
                new_seeds[i] = self.seeds[i];
            }
        }
    }

    /// Interior seeds currently OUTSIDE the fluid (inside a hole loop) —
    /// pre-barrier leaks or pathological crossings. Killed unconditionally
    /// by the adaptation cleanup: their cells cover non-fluid area, their
    /// state is junk, and they attract further Lloyd/adaptation effort.
    fn leaked_seeds(&self, spec_t: &BoundarySpec) -> Vec<usize> {
        use crate::meshgen::meshless::point_in_fluid;
        let boxes = Self::hole_bboxes(spec_t, 0.0);
        if boxes.is_empty() {
            return Vec::new();
        }
        (0..self.seeds.len())
            .filter(|&i| {
                if self.kinds[i] != SeedKind::Interior {
                    return false;
                }
                let p = self.seeds[i];
                let near = boxes
                    .iter()
                    .any(|&(x0, y0, x1, y1)| p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1);
                near && !point_in_fluid(p, spec_t)
            })
            .collect()
    }

    /// Whether cell `i` may be birthed into / killed by the flow-adaptive
    /// sizing: an interior seed outside the open-boundary bands that does
    /// not TOUCH a boundary cell. The through-flow x-bands belong to the
    /// recycling machinery (its density trigger + hole-spawn own that
    /// traffic); the y-wall dead zone keeps splits/kills off the fixed box
    /// guards. The wall-adjacency buffer (no face shared with a
    /// boundary-seed cell, no open face) is load-bearing at EMBEDDED
    /// boundaries: the wall's own seeds keep their build-time spacing —
    /// adaptation cannot re-discretize the polyline — and fluid cells
    /// refined well below that spacing degenerate the wall cells until one
    /// bulges THROUGH the wall into the obstacle interior (observed as the
    /// GUI "hernia": covered area exceeding the fluid area, cell centroids
    /// inside the obstacle). One untouched interior layer plus the target
    /// grading keeps the wall-adjacent size ratio in the regime the clip
    /// machinery is validated for.
    fn adapt_eligible(&self, i: usize) -> bool {
        self.adapt_in_bands(i) && self.touching_wall_seg_len(i).is_none()
    }

    /// The box-band half of eligibility: an interior seed clear of the
    /// through-flow and wall dead zones.
    fn adapt_in_bands(&self, i: usize) -> bool {
        let h = self.min_cell_size;
        let s = self.seeds[i];
        self.kinds[i] == SeedKind::Interior
            && s.x > FLOW_ADVECT_BOX_RAMP_CELLS * h
            && s.x < self.domain.x - FLOW_ADVECT_BOX_RAMP_CELLS * h
            && s.y > FLOW_ADVECT_BOX_DEAD_CELLS * h
            && s.y < self.domain.y - FLOW_ADVECT_BOX_DEAD_CELLS * h
    }

    /// If cell `i` TOUCHES the boundary (shares a face with a boundary-seed
    /// cell, or has an open face), the coarsest adjacent wall discretization
    /// scale: the max polyline segment length among the touched wall cells'
    /// segments (`+∞` for a bare open face). `None` = a bulk interior cell.
    /// Birth eligibility compares this against the cell's target spacing —
    /// a wall-adjacent cell may refine once the WALL is at least as fine
    /// (the wall subdivision keeps up event-by-event), so the boundary
    /// layer is resolvable without re-opening the out-refine-the-wall
    /// hernia.
    fn touching_wall_seg_len(&self, i: usize) -> Option<f64> {
        let m = &self.mesh;
        let mut worst: Option<f64> = None;
        let mut bump = |len: f64, worst: &mut Option<f64>| {
            *worst = Some(worst.map_or(len, |w: f64| w.max(len)));
        };
        let (fb, fe) = (m.cell_face_offsets[i], m.cell_face_offsets[i + 1]);
        for &f in &m.cell_faces[fb..fe] {
            let other = if m.face_owner[f] == i {
                m.face_neighbor[f]
            } else {
                Some(m.face_owner[f])
            };
            match other {
                None => bump(f64::INFINITY, &mut worst),
                Some(nb) => {
                    if let SeedKind::Boundary { seg_prev, seg_next } = self.kinds[nb] {
                        let seg_len = |g: u32| {
                            let (a, b) = self.spec.segment_points(g);
                            (b - a).norm()
                        };
                        bump(seg_len(seg_prev).max(seg_len(seg_next)), &mut worst);
                    }
                }
            }
        }
        worst
    }

    /// Per-cell TARGET volumes for the flow-adaptive sizing: Green–Gauss
    /// |∇U| (Frobenius), strain-rate magnitude (the symmetric part) and
    /// |∇p| over the current mesh, each normalized by its
    /// [`ADAPT_INDICATOR_QUANTILE`] quantile over the adapt-eligible cells
    /// (robust to a single spike; boundary-band jumps excluded from the
    /// scale), combined by max, then mapped linearly onto the adaptation
    /// band ([`Self::adapt_vol_band`] — the explicit
    /// [`Self::set_adaptive_sizing_band`] when set, else the initial mesh's
    /// realized volume band): indicator 0 → the coarsest band volume, 1 →
    /// the finest.
    fn adapt_target_vols(&self) -> Result<Vec<f64>, String> {
        let layout = &self.driver.solver().model().state_layout;
        let stride = layout.stride() as usize;
        let u_off = layout
            .offset_for("U")
            .ok_or("adaptive sizing: model has no U field")? as usize;
        let p_off = layout
            .offset_for("p")
            .ok_or("adaptive sizing: model has no p field")? as usize;
        let state = pollster::block_on(self.driver.solver().read_state_f32());
        let m = &self.mesh;
        let n = m.num_cells();
        if state.len() != n * stride {
            return Err(format!(
                "adaptive sizing: state length {} != n_cells*stride {}",
                state.len(),
                n * stride
            ));
        }
        let val = |c: usize| -> (f64, f64, f64) {
            (
                state[c * stride + u_off] as f64,
                state[c * stride + u_off + 1] as f64,
                state[c * stride + p_off] as f64,
            )
        };
        // Green–Gauss accumulation (face normals point owner→neighbor).
        // Boundary faces use the owner value (a zero-gradient closure —
        // adequate for an indicator; the boundary bands are adapt-ineligible
        // anyway). acc = V·[du/dx, du/dy, dv/dx, dv/dy, dp/dx, dp/dy].
        let mut acc = vec![[0.0f64; 6]; n];
        for f in 0..m.num_faces() {
            let o = m.face_owner[f];
            let (uo, vo, po) = val(o);
            let (uf, vf, pf) = match m.face_neighbor[f] {
                Some(nb) => {
                    let (un, vn, pn) = val(nb);
                    (0.5 * (uo + un), 0.5 * (vo + vn), 0.5 * (po + pn))
                }
                None => (uo, vo, po),
            };
            let (anx, any) = (m.face_area[f] * m.face_nx[f], m.face_area[f] * m.face_ny[f]);
            let add = [uf * anx, uf * any, vf * anx, vf * any, pf * anx, pf * any];
            for k in 0..6 {
                acc[o][k] += add[k];
            }
            if let Some(nb) = m.face_neighbor[f] {
                for k in 0..6 {
                    acc[nb][k] -= add[k];
                }
            }
        }
        let mut grad_u = vec![0.0f64; n];
        let mut strain = vec![0.0f64; n];
        let mut grad_p = vec![0.0f64; n];
        for i in 0..n {
            let inv_v = 1.0 / m.cell_vol[i].max(f64::MIN_POSITIVE);
            let g = [
                acc[i][0] * inv_v,
                acc[i][1] * inv_v,
                acc[i][2] * inv_v,
                acc[i][3] * inv_v,
            ];
            grad_u[i] = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2] + g[3] * g[3]).sqrt();
            let s01 = 0.5 * (g[1] + g[2]);
            strain[i] = (g[0] * g[0] + 2.0 * s01 * s01 + g[3] * g[3]).sqrt();
            grad_p[i] = (acc[i][4] * inv_v).hypot(acc[i][5] * inv_v);
        }
        let eligible: Vec<usize> = (0..n).filter(|&i| self.adapt_eligible(i)).collect();
        let scale = |q: &[f64]| -> f64 {
            let mut v: Vec<f64> = eligible.iter().map(|&i| q[i]).collect();
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(|a, b| a.total_cmp(b));
            v[((v.len() - 1) as f64 * ADAPT_INDICATOR_QUANTILE) as usize]
        };
        let (su, ss, sp) = (scale(&grad_u), scale(&strain), scale(&grad_p));
        // Per-component user threshold multipliers (0 disables a component;
        // the auto-calibrated quantile scale is the reference at 1.0).
        let (ku, kp, kstrain) = self.adapt_thresholds;
        let norm = |v: f64, s: f64, k: f64| {
            if k > 0.0 && s > 0.0 {
                (v / (s * k)).min(1.0)
            } else {
                0.0
            }
        };
        let (vmin_t, vmax_t) = self.adapt_vol_band();
        let mut targets: Vec<f64> = (0..n)
            .map(|i| {
                let ind = norm(grad_u[i], su, ku)
                    .max(norm(strain[i], ss, kstrain))
                    .max(norm(grad_p[i], sp, kp));
                vmax_t + (vmin_t - vmax_t) * ind
            })
            .collect();
        // GRADING: the raw indicator can step from 1 to 0 across one cell (a
        // front), demanding the full band jump between face neighbors —
        // several-× Voronoi size jumps mutilate cells. Min-propagate until
        // every face's target-SPACING ratio is ≤ ADAPT_GRADING_FACTOR
        // (volume ratio ≤ its square): refinement fans out in graded layers.
        let cap = ADAPT_GRADING_FACTOR * ADAPT_GRADING_FACTOR;
        for _ in 0..ADAPT_GRADING_SWEEPS {
            let mut changed = false;
            for f in 0..m.num_faces() {
                let Some(nb) = m.face_neighbor[f] else {
                    continue;
                };
                let o = m.face_owner[f];
                if targets[nb] > targets[o] * cap {
                    targets[nb] = targets[o] * cap;
                    changed = true;
                }
                if targets[o] > targets[nb] * cap {
                    targets[o] = targets[nb] * cap;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(targets)
    }

    /// Plan one flow-adaptive sizing event against the per-cell target
    /// volumes (`targets`, from [`Self::adapt_target_vols`]): the kill list
    /// (over-resolved cells, most-squeezed first, thinned to an
    /// adjacency-independent set so no survivor absorbs two removals at
    /// once) and the birth list (under-resolved cells split at the midpoint
    /// toward their farthest face centre — the cell's long axis — subject
    /// to the fluid test and a separation floor scaled to the cell's TARGET
    /// spacing, capped by the meshgen scale). Pure planning;
    /// [`Self::resize_cells`] executes it.
    fn plan_adaptation(
        &mut self,
        targets: &[f64],
    ) -> Result<(Vec<usize>, Vec<(Point2<f64>, usize)>), String> {
        use crate::meshgen::meshless::point_in_fluid;
        use std::collections::HashSet;
        let n = self.seeds.len();

        // PERSISTENCE counters (see [`ADAPT_PERSIST_STEPS`]): classify every
        // cell against THIS event's targets; a violation must hold its sign
        // for `persist_needed` consecutive events before the planner may act
        // on it. Noise flips the sign and resets the counter.
        if self.adapt_persist.len() != n {
            self.adapt_persist = vec![0; n];
        }
        for i in 0..n {
            let v = self.mesh.cell_vol[i];
            let c = self.adapt_persist[i];
            self.adapt_persist[i] = if v > ADAPT_REFINE_RATIO * targets[i] {
                c.max(0).saturating_add(1).min(16)
            } else if v < ADAPT_COARSEN_RATIO * targets[i] {
                c.min(0).saturating_sub(1).max(-16)
            } else {
                0
            };
        }
        let persist_needed = self.adapt_persist_needed();

        let m = &self.mesh;
        let n_max = self.adapt_cell_cap();
        let n_min = (self.initial_cell_count as f64 * ADAPT_BUDGET_MIN_FACTOR) as usize;

        // CONTAINMENT cleanup first: interior seeds sitting inside a hole
        // loop (leaks that predate the barrier) are killed UNCONDITIONALLY —
        // exempt from the rate limit, the eligibility bands, the
        // adjacency-independence rule and the budget floor (their cells
        // cover non-fluid area; removing them is repair, not adaptation).
        let spec_now = self.moved_spec(self.time);
        let mut kills: Vec<usize> = self.leaked_seeds(&spec_now);
        let mut kill_set: HashSet<usize> = kills.iter().cloned().collect();

        let mut kill_cand: Vec<usize> = (0..n)
            .filter(|&i| {
                !kill_set.contains(&i)
                    && self.adapt_persist[i] <= -persist_needed
                    && self.adapt_eligible(i)
                    && m.cell_vol[i] < ADAPT_COARSEN_RATIO * targets[i]
            })
            .collect();
        kill_cand.sort_by(|&a, &b| {
            (m.cell_vol[a] / targets[a]).total_cmp(&(m.cell_vol[b] / targets[b]))
        });
        let kill_budget = kills.len()
            + ADAPT_MAX_KILLS_PER_EVENT.min(n.saturating_sub(n_min).saturating_sub(kills.len()));
        'cand: for &i in &kill_cand {
            if kills.len() >= kill_budget {
                break;
            }
            let (fb, fe) = (m.cell_face_offsets[i], m.cell_face_offsets[i + 1]);
            for &f in &m.cell_faces[fb..fe] {
                let other = if m.face_owner[f] == i {
                    m.face_neighbor[f]
                } else {
                    Some(m.face_owner[f])
                };
                if let Some(nb) = other {
                    if kill_set.contains(&nb) {
                        continue 'cand;
                    }
                }
            }
            kills.push(i);
            kill_set.insert(i);
        }

        let spec_t = spec_now;
        let hex = 3.0f64.sqrt() / 2.0;
        let mut birth_cand: Vec<usize> = (0..n)
            .filter(|&i| {
                if self.adapt_persist[i] < persist_needed
                    || !self.adapt_in_bands(i)
                    || kill_set.contains(&i)
                    || m.cell_vol[i] <= ADAPT_REFINE_RATIO * targets[i]
                {
                    return false;
                }
                // Wall-adjacent cells (the boundary-layer band) may SPLIT
                // once the adjacent wall discretization is at least as fine
                // as their own REALIZED spacing — the wall then leads by at
                // most one level and the pair staircases down together, so
                // the interior can never out-refine the wall (the hernia
                // regime) and an extreme target band cannot deadlock the
                // unlock (a target-based gate would demand a wall the
                // subdivision only reaches level by level). Kills keep the
                // hard buffer (adapt_eligible).
                match self.touching_wall_seg_len(i) {
                    None => true,
                    Some(len) => len <= (m.cell_vol[i].max(0.0) / hex).sqrt(),
                }
            })
            .collect();
        birth_cand.sort_by(|&a, &b| {
            (m.cell_vol[b] / targets[b]).total_cmp(&(m.cell_vol[a] / targets[a]))
        });
        let birth_budget =
            ADAPT_MAX_BIRTHS_PER_EVENT.min(n_max.saturating_sub(n.saturating_sub(kills.len())));
        let mut births: Vec<(Point2<f64>, usize)> = Vec::new();
        for &i in &birth_cand {
            if births.len() >= birth_budget {
                break;
            }
            let s = self.seeds[i];
            let (fb, fe) = (m.cell_face_offsets[i], m.cell_face_offsets[i + 1]);
            let mut far: Option<(f64, Point2<f64>)> = None;
            for &f in &m.cell_faces[fb..fe] {
                let c = Point2::new(m.face_cx[f], m.face_cy[f]);
                let d2 = (c - s).norm_squared();
                if far.map_or(true, |(b, _)| d2 > b) {
                    far = Some((d2, c));
                }
            }
            let Some((_, fc)) = far else { continue };
            let p = Point2::new(0.5 * (s.x + fc.x), 0.5 * (s.y + fc.y));
            if !point_in_fluid(p, &spec_t) {
                continue;
            }
            // Clearance floor scaled to the CELL'S target spacing, capped by
            // the global meshgen scale: an explicit adaptation band FINER
            // than the built mesh must be reachable — a split toward target
            // pitch h_t legitimately places the child ~h_t/2 from its
            // parent, which the global 0.35·min_cell floor would reject.
            // For targets at/above the mesh scale this reduces to the old
            // global floor (the coalescing-pitch guard).
            let target_h = (targets[i] / (3.0f64.sqrt() / 2.0)).sqrt();
            let min_sep2 = (RECYCLE_MIN_SEP_CELLS * target_h.min(self.min_cell_size)).powi(2);
            let clear = self
                .seeds
                .iter()
                .enumerate()
                .filter(|&(j, _)| !kill_set.contains(&j))
                .all(|(_, q)| (q - p).norm_squared() >= min_sep2)
                && births
                    .iter()
                    .all(|(q, _)| (q - p).norm_squared() >= min_sep2);
            if clear {
                births.push((p, i));
            }
        }
        Ok((kills, births))
    }

    /// Plan the WALL-refinement half of an adaptation event: the global ids
    /// of boundary segments whose length exceeds
    /// [`ADAPT_WALL_SPLIT_RATIO`] × their wall cells' target spacing —
    /// worst first, rate-limited, budget-bounded. Restricted to segments
    /// the midpoint subdivision is valid for:
    /// * STATIC `Wall`/`SlipWall` segments only — the moving loop's seeds
    ///   follow a rigid transform of their t=0 labels (a mid-run birth has
    ///   no valid label), and the Inlet/Outlet guard structure belongs to
    ///   the recycling machinery;
    /// * segments whose wall seeds follow one of the two canonical
    ///   patterns (a collapsed mid-chord guard, or endpoint vertex seeds) —
    ///   an OFF-midpoint guard (a reflex-corner guard) would leave one half
    ///   uncovered after the split, so such segments are skipped.
    fn plan_wall_refinement(&self, targets: &[f64]) -> Vec<usize> {
        let total = self.spec.num_segments();
        if total == 0 {
            return Vec::new();
        }
        // Diagnostic kill switch (probes/A-B only): no wall subdivision —
        // walls keep their build discretization and the wall-adjacent
        // interior stays locked at its unlock gate.
        if std::env::var("CFD2_ADAPT_NO_WALL_SPLITS")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            return Vec::new();
        }
        let hex = 3.0f64.sqrt() / 2.0;
        let tol = MeshgenTolerances::from_geometry(self.min_cell_size, self.domain);
        // Per-segment wall-seed registry.
        let mut seg_seeds: Vec<Vec<usize>> = vec![Vec::new(); total];
        for (i, k) in self.kinds.iter().enumerate() {
            if let SeedKind::Boundary { seg_prev, seg_next } = *k {
                seg_seeds[seg_prev as usize].push(i);
                if seg_next != seg_prev {
                    seg_seeds[seg_next as usize].push(i);
                }
            }
        }
        let moving = self.moving_loop_range();
        let mut cand: Vec<(f64, usize)> = Vec::new();
        for g in 0..total {
            if let Some((lo, hi)) = moving {
                if g >= lo && g < hi {
                    continue;
                }
            }
            if !matches!(
                self.spec.segment_tag(g as u32),
                BoundaryType::Wall | BoundaryType::SlipWall
            ) {
                continue;
            }
            if seg_seeds[g].is_empty() {
                continue;
            }
            let (a, b) = self.spec.segment_points(g as u32);
            let len = (b - a).norm();
            // A half must stay well clear of the degenerate-segment floor.
            if 0.5 * len <= 8.0 * tol.edge_len_eps {
                continue;
            }
            let mid = Point2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y));
            let splittable = seg_seeds[g].iter().all(|&i| match self.kinds[i] {
                SeedKind::Boundary { seg_prev, seg_next } if seg_prev == seg_next => {
                    (self.seeds[i] - mid).norm() <= 0.05 * len
                }
                _ => true,
            });
            if !splittable {
                continue;
            }
            let t_h = seg_seeds[g]
                .iter()
                .map(|&i| (targets[i].max(0.0) / hex).sqrt())
                .fold(f64::INFINITY, f64::min);
            // Realized-fluid guard (see ADAPT_WALL_FLUID_RATIO): the finest
            // INTERIOR cell adjacent to this segment's wall cells bounds how
            // far the wall may run ahead of the fluid.
            let mut h_adj = f64::INFINITY;
            for &i in &seg_seeds[g] {
                let (fb, fe) = (
                    self.mesh.cell_face_offsets[i],
                    self.mesh.cell_face_offsets[i + 1],
                );
                for &f in &self.mesh.cell_faces[fb..fe] {
                    let other = if self.mesh.face_owner[f] == i {
                        self.mesh.face_neighbor[f]
                    } else {
                        Some(self.mesh.face_owner[f])
                    };
                    if let Some(nb) = other {
                        if self.kinds[nb] == SeedKind::Interior {
                            h_adj = h_adj.min((self.mesh.cell_vol[nb].max(0.0) / hex).sqrt());
                        }
                    }
                }
            }
            let fluid_ok = !h_adj.is_finite() || len > ADAPT_WALL_FLUID_RATIO * h_adj;
            if t_h.is_finite() && len > ADAPT_WALL_SPLIT_RATIO * t_h && fluid_ok {
                cand.push((len / t_h, g));
            }
        }
        cand.sort_by(|x, y| y.0.total_cmp(&x.0));
        // Budget: each split births at most 2 wall seeds.
        let budget = self.adapt_cell_cap().saturating_sub(self.seeds.len()) / 2;
        cand.truncate(ADAPT_MAX_WALL_SPLITS_PER_EVENT.min(budget));
        cand.into_iter().map(|(_, g)| g).collect()
    }

    /// The Lloyd SIZING function for scheduled smoothing / quality
    /// escalation. Without adaptation: the uniform meshgen scale (the
    /// shipped behaviour — a uniform-density CVT pull). With adaptation
    /// enabled (`adapt_every_n > 0`): the LOCAL CURRENT spacing — each
    /// query returns the nearest seed's realized cell pitch
    /// `√(vol/(√3/2))` via a bucket grid — so relaxation regularizes
    /// SHAPES while PRESERVING the adapted density distribution (a uniform
    /// sizing would erode the refinement the adaptation just built,
    /// pointfully so on a stationary mesh with no steering to counter it).
    fn lloyd_sizing(&self) -> Box<dyn Fn(Point2<f64>) -> f64 + Sync + '_> {
        let min_cell = self.min_cell_size;
        if self.adapt_every_n == 0 {
            return Box::new(move |_| min_cell);
        }
        // Bucket the seeds on a uniform grid (pitch = 2·min_cell) for O(1)
        // nearest-seed queries; per-seed pitch from the committed volumes.
        let hex = 3.0f64.sqrt() / 2.0;
        let pitch = 2.0 * min_cell;
        let nx = (self.domain.x / pitch).ceil().max(1.0) as usize;
        let ny = (self.domain.y / pitch).ceil().max(1.0) as usize;
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); nx * ny];
        let cell_of = move |p: &Point2<f64>| -> (usize, usize) {
            (
                ((p.x / pitch) as isize).clamp(0, nx as isize - 1) as usize,
                ((p.y / pitch) as isize).clamp(0, ny as isize - 1) as usize,
            )
        };
        for (i, s) in self.seeds.iter().enumerate() {
            let (bx, by) = cell_of(s);
            buckets[by * nx + bx].push(i as u32);
        }
        let mut h: Vec<f64> = self
            .mesh
            .cell_vol
            .iter()
            .map(|&v| (v.max(0.0) / hex).sqrt())
            .collect();
        // GRADE the realized spacing field (the same per-layer constraint as
        // the adaptation targets): a fresh split is locally 2:1 against its
        // neighbors, and an ungraded sizing would make the Lloyd relaxation
        // PRESERVE that jump. Min-propagating the spacing lets the smooth
        // actively fan rapid realized transitions out into graded layers.
        {
            let m = &self.mesh;
            for _ in 0..ADAPT_GRADING_SWEEPS {
                let mut changed = false;
                for f in 0..m.num_faces() {
                    let Some(nb) = m.face_neighbor[f] else {
                        continue;
                    };
                    let o = m.face_owner[f];
                    if h[nb] > h[o] * ADAPT_GRADING_FACTOR {
                        h[nb] = h[o] * ADAPT_GRADING_FACTOR;
                        changed = true;
                    }
                    if h[o] > h[nb] * ADAPT_GRADING_FACTOR {
                        h[o] = h[nb] * ADAPT_GRADING_FACTOR;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        let seeds = self.seeds.clone();
        Box::new(move |p: Point2<f64>| -> f64 {
            let (bx, by) = cell_of(&p);
            // Expanding ring search (radius 2 covers 5·min_cell — beyond any
            // seed gap); fall back to the meshgen scale if somehow empty.
            for r in 0..3usize {
                let (mut best, mut best_d2) = (usize::MAX, f64::INFINITY);
                for gy in by.saturating_sub(r)..=(by + r).min(ny - 1) {
                    for gx in bx.saturating_sub(r)..=(bx + r).min(nx - 1) {
                        for &i in &buckets[gy * nx + gx] {
                            let d2 = (seeds[i as usize] - p).norm_squared();
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best = i as usize;
                            }
                        }
                    }
                }
                if best != usize::MAX {
                    return h[best];
                }
            }
            min_cell
        })
    }

    /// Scheduled periodic Lloyd smoothing (see [`Self::set_smoothing`]): on
    /// every `smoothing_every_n`-th step, blend the interior seeds toward
    /// the CVT of [`Self::lloyd_sizing`]. A no-op when disabled or under
    /// `Prescribed` motion (positions are recomputed from the t=0 labels
    /// each step, so a smooth would be overwritten). Under `Frozen` the
    /// mesh advances from its current seeds, so the smooth persists — the
    /// stationary-adaptive relaxation path.
    fn maybe_periodic_smooth(
        &self,
        spec: &BoundarySpec,
        seeds: Vec<Point2<f64>>,
    ) -> Vec<Point2<f64>> {
        if self.smoothing_every_n == 0
            || matches!(self.motion, MeshMotionSpec::Prescribed(_))
            || (self.step_index + 1) % self.smoothing_every_n != 0
        {
            return seeds;
        }
        let before = seeds.clone();
        let mut relaxed = seeds;
        let sizing = self.lloyd_sizing();
        let tol = MeshgenTolerances::from_geometry(self.min_cell_size, self.domain);
        let cfg = EngineConfig::default();
        let lcfg = LloydConfig {
            max_iters: self.smoothing_iters,
            tol_disp: 0.0,
            omega: self.smoothing_omega,
            density_exponent: 4.0,
        };
        lloyd_relax(
            &mut relaxed,
            &self.kinds,
            spec,
            &sizing,
            self.domain,
            &tol,
            &cfg,
            &lcfg,
        );
        let _ = before;
        relaxed
    }

    /// Quality escalation: if the advected seed set would regenerate a mesh
    /// whose max skew exceeds `quality_skew_target` (or whose volumes leave
    /// the sizing-hold band), run `lloyd_escalation_iters` blended Lloyd
    /// regularization sweeps (reusing the meshgen [`lloyd_relax`], sizing
    /// per [`Self::lloyd_sizing`]) to pull the interior seeds back toward
    /// CVT, then return the regularized seeds. Returns `(seeds, escalated?)`.
    /// Active under FlowCoupled always, and under Frozen only when the
    /// flow-adaptive sizing is enabled (a stationary ADAPTIVE mesh needs the
    /// same post-resize quality hold; a plain Frozen run must stay
    /// byte-identical AND probe-free — the escalation assembles a probe mesh
    /// every step). Never under Prescribed (labels would overwrite it).
    fn maybe_quality_escalate(
        &self,
        spec: &BoundarySpec,
        seeds: Vec<Point2<f64>>,
    ) -> (Vec<Point2<f64>>, bool) {
        let active = match self.motion {
            MeshMotionSpec::FlowCoupled { .. } => true,
            MeshMotionSpec::Frozen => self.adapt_every_n > 0,
            MeshMotionSpec::Prescribed(_) => false,
        };
        if !active || self.lloyd_escalation_iters == 0 {
            return (seeds, false);
        }
        let probe = assemble_meshless_from_seeds(
            &seeds,
            &self.kinds,
            spec,
            self.domain,
            self.min_cell_size,
        );
        // Sizing hold: volumes leaving the HOLD band (the initial mesh's
        // realized sizing — the user's min/max cell settings — widened by
        // any explicit adaptation band) trigger the same Lloyd pull-back as
        // excess skew. A shape-only trigger cannot see this — a uniformly
        // squeezed cell stays centroidal — and mutilated undersized cells
        // are what breed the continuous-flip slivers (faces flip-flopping
        // through zero length every step).
        let hold = self.hold_vol_band();
        let (lo, hi) = (QUALITY_VOL_BAND_LO * hold.0, QUALITY_VOL_BAND_HI * hold.1);
        let (vmin, vmax) = probe
            .cell_vol
            .iter()
            .fold((f64::MAX, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
        // Per-cell squeeze test against the adaptation targets when live
        // (probe cell i == candidate seed i): a wide explicit band makes
        // the global floor blind to compression.
        let size_violated = match &self.adapt_targets {
            Some(t) if t.len() == probe.num_cells() => {
                (0..probe.num_cells()).any(|i| probe.cell_vol[i] < QUALITY_VOL_BAND_LO * t[i])
                    || vmax > hi
            }
            _ => vmin < lo || vmax > hi,
        };
        if !size_violated && probe.calculate_max_skewness() <= self.quality_skew_target {
            return (seeds, false);
        }
        // Gentle, blended Lloyd toward the CVT of `lloyd_sizing` (uniform
        // without adaptation; the local adapted spacing with it). `tol_disp
        // = 0` forces the full iteration budget (no early convergence
        // break); the small `omega` keeps it from erasing the flow
        // displacement.
        let mut relaxed = seeds;
        let sizing = self.lloyd_sizing();
        let tol = MeshgenTolerances::from_geometry(self.min_cell_size, self.domain);
        let cfg = EngineConfig::default();
        let lcfg = LloydConfig {
            max_iters: self.lloyd_escalation_iters,
            tol_disp: 0.0,
            omega: self.lloyd_escalation_omega,
            density_exponent: 4.0,
        };
        lloyd_relax(
            &mut relaxed,
            &self.kinds,
            spec,
            &sizing,
            self.domain,
            &tol,
            &cfg,
            &lcfg,
        );
        (relaxed, true)
    }

    /// Pin the fixed dt: the configured `requested_dt`, capped by the
    /// mesh-motion CFL `dt ≤ cfl_mesh · min_h / w_max`, pushed to the solver via
    /// [`SolverDriver::set_requested_dt`]. Returns the actually-pinned dt as the
    /// f64-widened f32 (`params.requested_dt as f64`) — the exact value the
    /// swept-flux closure must use so the flux dt == the step dt.
    fn pin_dt(&mut self, w_max: f64) -> f64 {
        // Base off the IMMUTABLE configured dt, not the previously-pinned
        // `params.requested_dt` — else the cap would ratchet the timestep down
        // permanently (a momentary fast step could never recover). Under
        // flow-adaptive dt the base is the CFL controller's candidate
        // instead (self-recovering by construction: it re-derives from the
        // flow every step).
        let mut dt = self.configured_dt;
        if let Some(cfl) = self.adaptive_dt_cfl {
            if let Some(flow_dt) = self.flow_adaptive_dt(cfl) {
                dt = flow_dt;
            }
        }
        if w_max > 1e-30 {
            let cap = self.mesh_cfl * self.driver.min_cell() / w_max;
            if cap.is_finite() && cap < dt {
                dt = cap;
            }
        }
        self.driver.set_requested_dt(dt as f32);
        // Read the pinned value BACK through params so the flux dt is the exact
        // f32 the solver steps with (widened) — not the f64 pre-cast value.
        self.driver.params().requested_dt as f64
    }

    /// The flow-CFL dt candidate: `dt = cfl / max_i((|U_i| + c_eff) / h_i)`
    /// — the TRUE per-cell advective CFL over the t^n state, not the static
    /// controller's conservative `min_h / max|U|` law (which assumes the
    /// fastest velocity lives in the smallest cell; under flow-adaptive
    /// sizing the finest cells sit in the SLOW near-wall bands, so the
    /// conservative law under-runs the realized CFL by 3-5x and leaves that
    /// much 1/CFL amplification of re-meshing pressure flicker on the
    /// table — measured on the extreme-band dipole watch). `c_eff` is the
    /// low-Mach-reduced EOS sound speed, GATED (exactly like the static
    /// branch) on the MODEL declaring a real EOS — NOT on `params.eos`: the
    /// GUI forwards the fluid's physical EOS (Air c≈347) even for the
    /// constant-EOS incompressible/all-Mach ALE families, and dividing
    /// their advective dt by a sound speed the model never resolves would
    /// collapse dt by ~4 orders of magnitude (the all-Mach acoustic
    /// stiffness is handled implicitly by `psi_precond`, not by the
    /// timestep). The inflow appears through the inlet-adjacent cells'
    /// own (|U|, h) — the state is read fresh every pin, so no synthetic
    /// global `|U_in|/min_h` floor is needed (it would re-impose the
    /// conservative law whenever the inlet speed rivals the interior).
    /// Growth is limited to 1.2× the last COMMITTED pinned dt. `None` = no
    /// usable estimate this step (fall back to the configured dt).
    fn flow_adaptive_dt(&self, target_cfl: f64) -> Option<f64> {
        let params = *self.driver.params();
        let u = self.read_cell_velocities().ok()?;
        if u.len() != self.mesh.num_cells() {
            return None;
        }
        let mut adv = (params.inlet_velocity as f64).abs();
        for &(x, y) in &u {
            adv = adv.max(x.hypot(y));
        }
        let sound = if self.driver.supports_sound_speed() {
            params.eos.sound_speed(params.density as f64)
        } else {
            0.0
        };
        let eff_sound = match params.low_mach_model {
            GpuLowMachPrecondModel::Off => sound,
            GpuLowMachPrecondModel::Legacy => sound.min(adv),
            GpuLowMachPrecondModel::WeissSmith => {
                let theta = (params.low_mach_theta_floor as f64).max(0.0);
                sound.min(adv.max(sound * theta.sqrt()))
            }
        };
        // max_i (wave_i / h_i), h_i = sqrt(cell area).
        let mut rate = 0.0f64;
        for (i, &(x, y)) in u.iter().enumerate() {
            let h = self.mesh.cell_vol[i].max(0.0).sqrt();
            if h > 1e-12 {
                let w = x.hypot(y) + eff_sound;
                if w.is_finite() {
                    rate = rate.max(w / h);
                }
            }
        }
        if !(rate.is_finite() && rate > 1e-12) {
            return None;
        }
        let mut dt = target_cfl / rate;
        if let Some(prev) = self.last_pinned_dt {
            dt = dt.min(prev * 1.2);
        }
        Some(dt.clamp(1e-9, 100.0))
    }

    /// The wrapped [`SolverDriver`] (field reads, stats, BC setup).
    pub fn driver(&self) -> &SolverDriver {
        &self.driver
    }

    /// Mutable access to the wrapped [`SolverDriver`] (e.g. inlet-vector BC
    /// overrides the ALE gates apply after build).
    pub fn driver_mut(&mut self) -> &mut SolverDriver {
        &mut self.driver
    }

    /// The current realized mesh.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// The EFFECTIVE boundary-motion spec (an [`BoundaryMotionSpec::Oscillation`]
    /// amplitude reflects the anti-swallow clamp applied by [`set_boundary_motion`],
    /// not the requested value). Lets a caller/gate observe the realized motion.
    pub fn boundary_motion(&self) -> BoundaryMotionSpec {
        self.boundary_motion
    }

    /// The authoritative seed positions (seed `i` == cell `i`).
    pub fn seeds(&self) -> &[Point2<f64>] {
        &self.seeds
    }

    /// The per-seed kinds (seed `i` == cell `i`) — diagnostics/probes.
    pub fn seed_kinds(&self) -> &[SeedKind] {
        &self.kinds
    }

    /// Committed step count.
    pub fn step_index(&self) -> usize {
        self.step_index
    }
}

/// Milliseconds elapsed since `start`, as f32.
#[inline]
fn ms_since(start: Instant) -> f32 {
    start.elapsed().as_secs_f32() * 1000.0
}

/// `(min, max)` cell volume — the per-step realized-sizing telemetry.
fn vol_extremes(vols: &[f64]) -> (f64, f64) {
    vols.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        })
}

/// Green–Gauss gradient of EVERY packed state component plus per-component
/// monotonicity bounds (min/max over the cell and its face neighbors) — the
/// first-order state-transfer stencil of a cell-count resize. Boundary faces
/// use the owner value (zero-gradient closure). Returns
/// `(gx, gy, lo, hi)`, each `n_cells × stride`.
fn state_gradients_and_bounds(
    m: &Mesh,
    state: &[f32],
    stride: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f32>, Vec<f32>) {
    let n = m.num_cells();
    let mut gx = vec![0.0f64; n * stride];
    let mut gy = vec![0.0f64; n * stride];
    let mut lo = state.to_vec();
    let mut hi = state.to_vec();
    for f in 0..m.num_faces() {
        let o = m.face_owner[f];
        let (anx, any) = (m.face_area[f] * m.face_nx[f], m.face_area[f] * m.face_ny[f]);
        match m.face_neighbor[f] {
            Some(nb) => {
                for k in 0..stride {
                    let vo = state[o * stride + k];
                    let vn = state[nb * stride + k];
                    let vf = 0.5 * (vo as f64 + vn as f64);
                    gx[o * stride + k] += vf * anx;
                    gy[o * stride + k] += vf * any;
                    gx[nb * stride + k] -= vf * anx;
                    gy[nb * stride + k] -= vf * any;
                    lo[o * stride + k] = lo[o * stride + k].min(vn);
                    hi[o * stride + k] = hi[o * stride + k].max(vn);
                    lo[nb * stride + k] = lo[nb * stride + k].min(vo);
                    hi[nb * stride + k] = hi[nb * stride + k].max(vo);
                }
            }
            None => {
                for k in 0..stride {
                    let vf = state[o * stride + k] as f64;
                    gx[o * stride + k] += vf * anx;
                    gy[o * stride + k] += vf * any;
                }
            }
        }
    }
    for c in 0..n {
        let inv_v = 1.0 / m.cell_vol[c].max(f64::MIN_POSITIVE);
        for k in 0..stride {
            gx[c * stride + k] *= inv_v;
            gy[c * stride + k] *= inv_v;
        }
    }
    (gx, gy, lo, hi)
}

/// Set one component of the per-face `MovingWall` Dirichlet velocity, resolving
/// the velocity field name as `"U"` (upper) with a `"u"` (lower) fallback —
/// mirroring [`set_inlet_velocity`](crate::solver::model::helpers) so the caller
/// need not know the model's field-name casing.
fn set_moving_wall_component(
    solver: &mut crate::solver::UnifiedSolver,
    component: u32,
    values: &[f32],
) -> Result<(), String> {
    solver
        .set_boundary_values_per_face(GpuBoundaryType::MovingWall, "U", component, &|f| {
            values[f as usize]
        })
        .or_else(|_| {
            solver.set_boundary_values_per_face(GpuBoundaryType::MovingWall, "u", component, &|f| {
                values[f as usize]
            })
        })
}

/// Mark geometrically degenerate NEW faces (near-zero length at t^n via the
/// aligned old positions, or at t^{n+1}) as born in `flip`, so the conservative
/// flip closure treats them as slack (zero swept contribution absorbed per cell)
/// rather than letting the persistent swept-flux path hard-error on them. The
/// threshold is a small fraction of the cell size, so only truly collapsing
/// faces (whose real swept volume is negligible) are caught. Returns how many
/// faces were newly forced born.
///
/// Keeps the flip telemetry honest: a forced-degenerate face marks its incident
/// cells in `flip.cell_flipped` and re-derives `flip.flipped_cells`, so a
/// degeneracy-only step reports `flipped_cells > 0` (not a `flipped=true,
/// flipped_cells=0` phantom). That per-cell flag is ALSO the hard-assert exclude
/// set the caller passes to the flip closure.
fn force_degenerate_faces_born(
    mesh: &Mesh,
    old_vx: &[f64],
    old_vy: &[f64],
    flip: &mut FlipReport,
    min_cell: f64,
) -> usize {
    let tol = 1e-3 * min_cell.max(f64::MIN_POSITIVE);
    let tol2 = tol * tol;
    let mut added = 0usize;
    for f in 0..mesh.num_faces() {
        if flip.born_face_mask[f] {
            continue;
        }
        let (v1, v2) = (mesh.face_v1[f], mesh.face_v2[f]);
        let nlen2 = (mesh.vx[v1] - mesh.vx[v2]).powi(2) + (mesh.vy[v1] - mesh.vy[v2]).powi(2);
        let olen2 = (old_vx[v1] - old_vx[v2]).powi(2) + (old_vy[v1] - old_vy[v2]).powi(2);
        if nlen2 < tol2 || olen2 < tol2 {
            flip.born_face_mask[f] = true;
            flip.born_faces += 1;
            flip.cell_flipped[mesh.face_owner[f]] = true;
            if let Some(nb) = mesh.face_neighbor[f] {
                flip.cell_flipped[nb] = true;
            }
            added += 1;
        }
    }
    if added > 0 {
        flip.flipped_cells = flip.cell_flipped.iter().filter(|&&b| b).count();
    }
    added
}

/// Subdivide global boundary segment `g` of `spec` at its MIDPOINT — the
/// boundary-discretization refinement primitive. Geometry-EXACT: the new
/// polyline vertex lies on the old chord, so the wall shape and the fluid
/// area are bit-unchanged; only the segment (and thus wall-seed) resolution
/// doubles. Mutates the spec (vertex + duplicated tag inserted, later
/// `seg_offsets` shifted), remaps every seg id `> g` by `+1` in `kinds` AND
/// in the already-emitted `births`, and re-kinds/births the wall seeds by
/// pattern:
/// * collapsed mid-chord GUARD (`seg_prev == seg_next == g`, at the chord
///   midpoint — the hole/circle pattern): the guard becomes the new
///   (straight) vertex seed spanning both halves, and TWO guard births land
///   at the half-chord midpoints;
/// * endpoint VERTEX seeds (the straight-wall pattern): ONE vertex-seed
///   birth lands at the midpoint, and the far endpoint's `seg_prev` moves
///   to the new half.
///
/// Returns `false` (no mutation) for an OFF-midpoint guard (a reflex-corner
/// guard: splitting would leave one half uncovered) — the planner filters
/// these, so this is a defensive re-check. Birth donors are the segment's
/// existing wall seeds (their cells carry the wall-adjacent state).
fn split_wall_segment(
    spec: &mut BoundarySpec,
    seeds: &[Point2<f64>],
    kinds: &mut [SeedKind],
    births: &mut Vec<(Point2<f64>, usize, SeedKind)>,
    g: usize,
) -> bool {
    let (l, s) = spec.locate(g as u32);
    let n_pts = spec.loops[l].pts.len();
    let a = spec.loops[l].pts[s];
    let b = spec.loops[l].pts[(s + 1) % n_pts];
    let m = Point2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y));
    let len = (b - a).norm();

    // Locate the wall seeds referencing `g` and classify the pattern.
    let mut guard: Option<usize> = None;
    let mut a_end: Option<usize> = None;
    let mut b_end: Option<usize> = None;
    for (i, k) in kinds.iter().enumerate() {
        if let SeedKind::Boundary { seg_prev, seg_next } = *k {
            let (p, q) = (seg_prev as usize, seg_next as usize);
            if p == g && q == g {
                if (seeds[i] - m).norm() > 0.05 * len {
                    return false; // off-midpoint (reflex-corner) guard
                }
                guard = Some(i);
            } else if q == g {
                a_end = Some(i);
            } else if p == g {
                b_end = Some(i);
            }
        }
    }
    if guard.is_none() && a_end.is_none() && b_end.is_none() {
        return false; // uncovered segment — nothing to re-kind, do not split
    }

    // Insert the midpoint vertex (duplicating the segment tag) and shift
    // the later loops' segment offsets.
    let tag = spec.loops[l].tags[s];
    spec.loops[l].pts.insert(s + 1, m);
    spec.loops[l].tags.insert(s + 1, tag);
    for off in spec.seg_offsets[l + 1..].iter_mut() {
        *off += 1;
    }

    // Remap every seg id above `g` (the inserted segment renumbers them).
    let remap = |k: &mut SeedKind| {
        if let SeedKind::Boundary { seg_prev, seg_next } = k {
            if (*seg_prev as usize) > g {
                *seg_prev += 1;
            }
            if (*seg_next as usize) > g {
                *seg_next += 1;
            }
        }
    };
    for k in kinds.iter_mut() {
        remap(k);
    }
    for (_, _, k) in births.iter_mut() {
        remap(k);
    }

    let (ga, gb) = (g as u32, (g + 1) as u32);
    match guard {
        Some(i) => {
            kinds[i] = SeedKind::Boundary {
                seg_prev: ga,
                seg_next: gb,
            };
            births.push((
                Point2::new(0.5 * (a.x + m.x), 0.5 * (a.y + m.y)),
                i,
                SeedKind::Boundary {
                    seg_prev: ga,
                    seg_next: ga,
                },
            ));
            births.push((
                Point2::new(0.5 * (m.x + b.x), 0.5 * (m.y + b.y)),
                i,
                SeedKind::Boundary {
                    seg_prev: gb,
                    seg_next: gb,
                },
            ));
        }
        None => {
            // The b-endpoint's vertex seed now abuts the SECOND half.
            if let Some(ib) = b_end {
                if let SeedKind::Boundary { seg_prev, .. } = &mut kinds[ib] {
                    *seg_prev = gb;
                }
            }
            let donor = a_end.or(b_end).expect("checked non-empty above");
            births.push((
                m,
                donor,
                SeedKind::Boundary {
                    seg_prev: ga,
                    seg_next: gb,
                },
            ));
        }
    }
    true
}

/// Whether two meshes with the same cell count differ in face set / adjacency
/// (a Voronoi flip) OR in per-face boundary TAGS. Compares the face count, the
/// per-face owner/neighbor and per-cell face lists — the connectivity the
/// topology refresh rebuilds — AND `face_boundary` (by `bc_table_index`).
///
/// The `face_boundary` comparison is load-bearing for the `Wall → MovingWall`
/// retag: the geometry seam ([`Mesh::refresh_mesh_geometry`]) asserts identical
/// `face_boundary` and hard-errors on a tag flip, but a step whose motion is too
/// small to change connectivity would otherwise take that seam and die.
/// Including the tags here routes any tag change through
/// `begin_ale_step_topology`, which rebuilds the BC tables + `face_boundary`
/// snapshot; `apply_moving_wall_velocity` then repopulates the values.
/// Byte-neutral under `Static`/no-retag (tags identical ⇒ `false`). Cheap
/// (O(faces)).
fn topology_differs(a: &Mesh, b: &Mesh) -> bool {
    a.num_faces() != b.num_faces()
        || a.face_owner != b.face_owner
        || a.face_neighbor != b.face_neighbor
        || a.cell_face_offsets != b.cell_face_offsets
        || a.cell_faces != b.cell_faces
        || a.face_boundary.iter().zip(&b.face_boundary).any(|(x, y)| {
            x.map(|t| t.bc_table_index()).unwrap_or(0) != y.map(|t| t.bc_table_index()).unwrap_or(0)
        })
}
