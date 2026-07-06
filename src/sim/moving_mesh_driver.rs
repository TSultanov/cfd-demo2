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
    /// `0` = off (default). FlowCoupled only: Frozen/Prescribed recompute
    /// positions from the t=0 labels every step, so a smooth would be
    /// overwritten.
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
        let build = SolverDriver::build(
            &mesh, model, params, initial_u, initial_p, device, queue,
        )
        .await?;

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

    /// Configure periodic mesh smoothing (FlowCoupled): every `every_n`
    /// committed steps, run `iters` blended Lloyd sweeps at weight `omega`
    /// (0..1) over the interior seeds — a scheduled, unconditional
    /// regularization on top of the on-demand quality escalation. `every_n
    /// == 0` disables (default). Small `omega` keeps the smooth from erasing
    /// the flow-coupled displacement; the hard per-step displacement clamp
    /// still applies to the following advection, and the smooth itself is
    /// mesh motion like any other — the swept-flux closure keeps the GCL.
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
    /// count). `0` = off (default). Skipped for `Prescribed` motion (its
    /// analytic law is indexed by the t=0 labels, which a resize re-anchors).
    pub fn set_adaptive_sizing(&mut self, every_n: usize) {
        self.adapt_every_n = every_n;
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
        // derive per-cell target volumes from the flow gradients and
        // birth/kill cells toward them through the resize seam. The step then
        // proceeds at the new count with a freshly rebuilt, state-transferred
        // solver.
        let (mut cells_born, mut cells_killed) = (0usize, 0usize);
        if self.adapt_fires_this_step() {
            let (kills, births) = self.plan_adaptation()?;
            if !kills.is_empty() || !births.is_empty() {
                self.resize_cells(&kills, &births)?;
                cells_born = births.len();
                cells_killed = kills.len();
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
                    return Ok(out);
                }
                DeviceStep::Fallback(reason) => fallback = Some(reason),
            }
        }

        let mut out = self.step_cpu_planned(plan, readback, fallback)?;
        out.1.cells_born = cells_born;
        out.1.cells_killed = cells_killed;
        Ok(out)
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
        let (new_seeds, escalated) = self.maybe_quality_escalate(&step_spec, new_seeds);
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
        if !recycled.is_empty() {
            let state = pollster::block_on(self.driver.solver().read_state_f32());
            let stride = self.driver.solver().model().state_layout.stride() as usize;
            let recycled_set: std::collections::HashSet<usize> =
                recycled.iter().cloned().collect();
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
            identity_err: if is_flip { 0.0 } else { swept.max_identity_err_rel },
            max_skew,
            n_cells,
            n_faces,
            topo_changed,
            flipped: is_flip,
            born_faces: flip.born_faces,
            died_faces: flip.died_faces,
            flipped_cells: flip.flipped_cells,
            flip_defect: if is_flip { swept.max_identity_err_rel } else { 0.0 },
            dt,
            regen_backend: gpu_fallback
                .map(RegenBackend::GpuFallback)
                .unwrap_or(RegenBackend::Cpu),
            recycled: recycled.len(),
            cells_born: 0,
            cells_killed: 0,
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
    fn try_step_device(
        &mut self,
        plan: &StepPlan,
        readback: bool,
    ) -> Result<DeviceStep, String> {
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
                device, n, self.domain, &tol,
            ));
        }

        // 3-4. Regenerate the solver mesh + ALE fluxes ON DEVICE, clipping
        //      against the t^{n+1} boundary spec (an identity clone of
        //      `self.spec` under Static boundary motion).
        let regen_start = Instant::now();
        let old_f32: Vec<f32> = self.seeds.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
        let new_f32: Vec<f32> = new_seeds.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
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
        let topo_signature = |mesh: &crate::solver::mesh::Mesh| -> (HashSet<(usize, usize)>, Vec<u32>) {
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
            let (fb, fe) = (new_mesh.cell_face_offsets[i], new_mesh.cell_face_offsets[i + 1]);
            let mut s = 0.0f64;
            for &f in &new_mesh.cell_faces[fb..fe] {
                let sgn = if new_mesh.face_owner[f] == i { 1.0 } else { -1.0 };
                s += sgn * result.mesh_fluxes[f] as f64 * dt;
            }
            let dv = new_mesh.cell_vol[i] - self.mesh.cell_vol[i];
            scl_defect = scl_defect
                .max((s - dv).abs() / new_mesh.cell_vol[i].max(f64::MIN_POSITIVE));
        }

        // 5. Refresh (always the topology seam — the device face order changes
        //    every regen) + reapply runtime BC overrides the refresh drops.
        let refresh_start = Instant::now();
        let report = self.driver.begin_ale_step_topology(&new_mesh, &result.mesh_fluxes)?;
        if report.bc_overrides_reset {
            self.driver.reapply_boundary_conditions();
        }
        self.apply_moving_wall_velocity(&new_mesh)?;
        let refresh_ms = ms_since(refresh_start);

        // 6. Step.
        let outcome = self.driver.step(readback);

        let n_cells = new_mesh.num_cells();
        let n_faces = new_mesh.num_faces();
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
        };
        Ok(DeviceStep::Done((outcome, stats)))
    }

    /// The seed positions at absolute time `t`, evaluated from the t=0 labels
    /// `f(seed0_i, t)` for interior seeds; boundary seeds are held fixed.
    /// Frozen returns the labels unchanged (t is irrelevant).
    fn advect_to(&self, t: f64) -> Vec<Point2<f64>> {
        match self.motion {
            MeshMotionSpec::Frozen | MeshMotionSpec::FlowCoupled { .. } => self.seeds0.clone(),
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
        let mut spec = self.spec.clone();
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
        let u = self.read_cell_velocities()?;
        let dead_len = FLOW_ADVECT_BOX_DEAD_CELLS * self.min_cell_size;
        let ramp_len =
            ((FLOW_ADVECT_BOX_RAMP_CELLS - FLOW_ADVECT_BOX_DEAD_CELLS) * self.min_cell_size)
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
            // SIZING ramp: a cell squeezed below the initial sizing band
            // (persistent advective compression — e.g. converging streamlines
            // against fixed boundary guards) escalates its centroid pull
            // toward FULL Lloyd weight (χ=1), regardless of `chi_max`. The
            // shape ramp cannot see this (a uniformly squeezed cell stays
            // centroidal), and the global blended escalation is too gentle
            // to balance a steady seed inflow. χ ramps 0→1 as the volume
            // falls from `LO·initial_min` to half that; the hard
            // displacement clamp below keeps the move flip-safe.
            let lo = QUALITY_VOL_BAND_LO * self.initial_vol_band.0;
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
        let mut out: Vec<usize> = (0..seeds.len())
            .filter(|&i| {
                self.kinds[i] == SeedKind::Interior
                    && seeds[i].x > strip_x
                    && self.mesh.cell_vol[i] < vol_lo
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
        let take = |v: &Vec<Point2<f64>>| -> Vec<Point2<f64>> {
            gather.iter().map(|&s| v[s]).collect()
        };
        self.seeds = take(&self.seeds);
        self.seeds0 = take(&self.seeds0);
        self.kinds = gather.iter().map(|&s| self.kinds[s]).collect();
        self.w_wall = gather.iter().map(|&s| self.w_wall[s]).collect();

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
            cell_faces
                .extend_from_slice(&m.cell_faces[m.cell_face_offsets[src]..m.cell_face_offsets[src + 1]]);
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
                return Err(format!("resize_cells: kill index {i} out of range ({n} cells)"));
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

        // New seed set: survivors in slot order, births appended. `src[new]`
        // is the OLD cell whose state row seeds the new cell (survivor:
        // itself; birth: its donor). `seeds0` labels are GATHERED, not reset:
        // boundary seeds keep their t=0 labels (the rigid boundary transform
        // is evaluated absolutely from them); a birth's label is its spawn
        // position (interior labels are unused under Frozen-equivalence and
        // FlowCoupled).
        let n_new = n - kills.len() + births.len();
        let mut new_seeds: Vec<Point2<f64>> = Vec::with_capacity(n_new);
        let mut new_seeds0: Vec<Point2<f64>> = Vec::with_capacity(n_new);
        let mut new_kinds: Vec<SeedKind> = Vec::with_capacity(n_new);
        let mut new_w_wall: Vec<[f64; 2]> = Vec::with_capacity(n_new);
        let mut src: Vec<usize> = Vec::with_capacity(n_new);
        for i in 0..n {
            if kill_set.contains(&i) {
                continue;
            }
            new_seeds.push(self.seeds[i]);
            new_seeds0.push(self.seeds0[i]);
            new_kinds.push(self.kinds[i]);
            new_w_wall.push(self.w_wall[i]);
            src.push(i);
        }
        for &(p, donor) in births {
            new_seeds.push(p);
            new_seeds0.push(p);
            new_kinds.push(SeedKind::Interior);
            new_w_wall.push([0.0, 0.0]);
            src.push(donor);
        }

        // Assemble the new mesh (CPU — a resize is a rare, full topology
        // event) and re-stamp the tags against the NEW indexing.
        let mut new_mesh = assemble_meshless_from_seeds(
            &new_seeds,
            &new_kinds,
            &spec_t,
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
        let mut rows: Vec<f32> = Vec::with_capacity(n_new * stride);
        let mut init_u: Vec<(f64, f64)> = Vec::with_capacity(n_new);
        let mut init_p: Vec<f64> = Vec::with_capacity(n_new);
        for &s in &src {
            let row = &state[s * stride..(s + 1) * stride];
            rows.extend_from_slice(row);
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
        driver.solver().reinit_cells(&cells, &rows, &new_mesh.cell_vol)?;

        // Commit.
        self.driver = driver;
        self.prev_vx = new_mesh.vx.clone();
        self.prev_vy = new_mesh.vy.clone();
        self.mesh = new_mesh;
        self.seeds = new_seeds;
        self.seeds0 = new_seeds0;
        self.kinds = new_kinds;
        self.w_wall = new_w_wall;
        // The on-device regen bundle is sized at a fixed seed count —
        // rebuild it lazily at the new count on the next device attempt.
        self.gpu_regen_state = None;
        Ok(())
    }

    /// Whether cell `i` may be birthed into / killed by the flow-adaptive
    /// sizing: an interior seed outside the open-boundary bands. The
    /// through-flow x-bands belong to the recycling machinery (its density
    /// trigger + hole-spawn own that traffic); the y-wall dead zone keeps
    /// splits/kills off the fixed guard seeds.
    fn adapt_eligible(&self, i: usize) -> bool {
        let h = self.min_cell_size;
        let s = self.seeds[i];
        self.kinds[i] == SeedKind::Interior
            && s.x > FLOW_ADVECT_BOX_RAMP_CELLS * h
            && s.x < self.domain.x - FLOW_ADVECT_BOX_RAMP_CELLS * h
            && s.y > FLOW_ADVECT_BOX_DEAD_CELLS * h
            && s.y < self.domain.y - FLOW_ADVECT_BOX_DEAD_CELLS * h
    }

    /// Per-cell TARGET volumes for the flow-adaptive sizing: Green–Gauss
    /// |∇U| (Frobenius), strain-rate magnitude (the symmetric part) and
    /// |∇p| over the current mesh, each normalized by its
    /// [`ADAPT_INDICATOR_QUANTILE`] quantile over the adapt-eligible cells
    /// (robust to a single spike; boundary-band jumps excluded from the
    /// scale), combined by max, then mapped linearly onto the INITIAL mesh's
    /// realized volume band — the user's selected sizing: indicator 0 →
    /// the coarsest initial volume, 1 → the finest.
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
        let norm = |v: f64, s: f64| if s > 0.0 { (v / s).min(1.0) } else { 0.0 };
        let (vmin_t, vmax_t) = self.initial_vol_band;
        Ok((0..n)
            .map(|i| {
                let ind = norm(grad_u[i], su)
                    .max(norm(strain[i], ss))
                    .max(norm(grad_p[i], sp));
                vmax_t + (vmin_t - vmax_t) * ind
            })
            .collect())
    }

    /// Plan one flow-adaptive sizing event against the current per-cell
    /// target volumes: the kill list (over-resolved cells, most-squeezed
    /// first, thinned to an adjacency-independent set so no survivor absorbs
    /// two removals at once) and the birth list (under-resolved cells split
    /// at the midpoint toward their farthest face centre — the cell's long
    /// axis — subject to the fluid test and the coalescing-pitch separation
    /// floor). Pure planning; [`Self::resize_cells`] executes it.
    fn plan_adaptation(&self) -> Result<(Vec<usize>, Vec<(Point2<f64>, usize)>), String> {
        use crate::meshgen::meshless::point_in_fluid;
        use std::collections::HashSet;
        let targets = self.adapt_target_vols()?;
        let m = &self.mesh;
        let n = self.seeds.len();
        let n_max = (self.initial_cell_count as f64 * ADAPT_BUDGET_MAX_FACTOR) as usize;
        let n_min = (self.initial_cell_count as f64 * ADAPT_BUDGET_MIN_FACTOR) as usize;

        let mut kill_cand: Vec<usize> = (0..n)
            .filter(|&i| {
                self.adapt_eligible(i) && m.cell_vol[i] < ADAPT_COARSEN_RATIO * targets[i]
            })
            .collect();
        kill_cand.sort_by(|&a, &b| {
            (m.cell_vol[a] / targets[a]).total_cmp(&(m.cell_vol[b] / targets[b]))
        });
        let kill_budget = ADAPT_MAX_KILLS_PER_EVENT.min(n.saturating_sub(n_min));
        let mut kills: Vec<usize> = Vec::new();
        let mut kill_set: HashSet<usize> = HashSet::new();
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

        let spec_t = self.moved_spec(self.time);
        let min_sep2 = (RECYCLE_MIN_SEP_CELLS * self.min_cell_size).powi(2);
        let mut birth_cand: Vec<usize> = (0..n)
            .filter(|&i| {
                self.adapt_eligible(i)
                    && !kill_set.contains(&i)
                    && m.cell_vol[i] > ADAPT_REFINE_RATIO * targets[i]
            })
            .collect();
        birth_cand.sort_by(|&a, &b| {
            (m.cell_vol[b] / targets[b]).total_cmp(&(m.cell_vol[a] / targets[a]))
        });
        let birth_budget = ADAPT_MAX_BIRTHS_PER_EVENT
            .min(n_max.saturating_sub(n.saturating_sub(kills.len())));
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

    /// Scheduled periodic Lloyd smoothing (see [`Self::set_smoothing`]): on
    /// every `smoothing_every_n`-th step, blend the interior seeds toward the
    /// uniform-density CVT. A no-op when disabled or for non-FlowCoupled
    /// motion (Frozen/Prescribed recompute positions from labels each step).
    fn maybe_periodic_smooth(
        &self,
        spec: &BoundarySpec,
        seeds: Vec<Point2<f64>>,
    ) -> Vec<Point2<f64>> {
        if self.smoothing_every_n == 0
            || !matches!(self.motion, MeshMotionSpec::FlowCoupled { .. })
            || (self.step_index + 1) % self.smoothing_every_n != 0
        {
            return seeds;
        }
        let mut relaxed = seeds;
        let min_cell = self.min_cell_size;
        let sizing = move |_: Point2<f64>| min_cell;
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
        relaxed
    }

    /// FlowCoupled quality escalation: if the advected seed set would regenerate
    /// a mesh whose max skew exceeds `quality_skew_target`, run
    /// `lloyd_escalation_iters` blended Lloyd regularization sweeps (reusing the
    /// meshgen [`lloyd_relax`]) to pull the interior seeds back toward CVT, then
    /// return the regularized seeds. Returns `(seeds, escalated?)`. A no-op for
    /// non-FlowCoupled motion, when escalation is disabled, or when the mesh is
    /// already well-shaped — so it never perturbs the frozen/prescribed gates.
    fn maybe_quality_escalate(
        &self,
        spec: &BoundarySpec,
        seeds: Vec<Point2<f64>>,
    ) -> (Vec<Point2<f64>>, bool) {
        if !matches!(self.motion, MeshMotionSpec::FlowCoupled { .. })
            || self.lloyd_escalation_iters == 0
        {
            return (seeds, false);
        }
        let probe = assemble_meshless_from_seeds(
            &seeds,
            &self.kinds,
            spec,
            self.domain,
            self.min_cell_size,
        );
        // Sizing hold: volumes leaving the band anchored to the INITIAL
        // mesh's realized sizing (the user's min/max cell settings) trigger
        // the same Lloyd pull-back as excess skew. A shape-only trigger
        // cannot see this — a uniformly squeezed cell stays centroidal — and
        // mutilated undersized cells are what breed the continuous-flip
        // slivers (faces flip-flopping through zero length every step).
        let (lo, hi) = (
            QUALITY_VOL_BAND_LO * self.initial_vol_band.0,
            QUALITY_VOL_BAND_HI * self.initial_vol_band.1,
        );
        let (vmin, vmax) = probe
            .cell_vol
            .iter()
            .fold((f64::MAX, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
        let size_violated = vmin < lo || vmax > hi;
        if !size_violated && probe.calculate_max_skewness() <= self.quality_skew_target {
            return (seeds, false);
        }
        // Gentle, blended Lloyd toward the (uniform-density) CVT. `tol_disp = 0`
        // forces the full iteration budget (no early convergence break); the
        // small `omega` keeps it from erasing the flow displacement.
        let mut relaxed = seeds;
        let min_cell = self.min_cell_size;
        let sizing = move |_: Point2<f64>| min_cell;
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
        // permanently (a momentary fast step could never recover).
        let mut dt = self.configured_dt;
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
            x.map(|t| t.bc_table_index()).unwrap_or(0)
                != y.map(|t| t.bc_table_index()).unwrap_or(0)
        })
}
