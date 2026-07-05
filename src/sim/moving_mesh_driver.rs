//! The moving-mesh loop orchestrator (roadmap M4, CPU-first).
//!
//! [`MovingMeshDriver`] wraps a [`SolverDriver`] and drives the per-step
//! moving-mesh cycle around it — WITHOUT touching `SolverDriver::step`, so
//! static-mesh users are provably unaffected (the wrapper is purely additive).
//! One `step` is:
//!
//! 1. **dt handshake** (review-solver-ale F2): pin a fixed dt — the driver's
//!    configured `requested_dt`, additionally capped by the mesh-motion CFL
//!    `dt ≤ cfl_mesh · min_h / max|w|` — via [`SolverDriver::set_requested_dt`]
//!    BEFORE the swept fluxes are closed, so the fluxes and the step march with
//!    exactly one dt (an adaptive re-scale after the closure would silently
//!    break the GCL; `begin_ale_step_topology` also hard-rejects `adaptive_dt`).
//! 2. **advect seeds** per [`MeshMotionSpec`] (f64). `Frozen` is a no-op.
//! 3. **regen** the mesh from the advected seeds via the M0 engine
//!    ([`assemble_meshless_from_seeds`]) — deterministic; from an UNCHANGED
//!    seed set it reproduces the current mesh byte-for-byte.
//! 4. **swept-quad mesh fluxes** old→new with the pinned dt
//!    ([`swept_mesh_fluxes_closed`]): f64 telescoping geometry + the f32 SCL
//!    closure. A frozen (byte-identical) regen ⇒ all-zero fluxes, zero defect.
//! 5. **refresh + rotate**: [`SolverDriver::begin_ale_step_topology`] rotates
//!    the volume history, rebuilds the topology-derived solver stack for the
//!    new mesh, and uploads the closed fluxes.
//! 6. **step**: `SolverDriver::step` (fixed dt, divergence detection).
//!
//! **Fixed seed count (v1)**: seed `i` is cell `i` for the whole run; no
//! insertion/deletion (that needs a conservative remap, deferred to M6). This
//! is what lets every cell-indexed solver buffer (state, BDF2 history, the
//! warm-start `x`) survive the refresh untouched.
//!
//! **Model scope (v1)**: `incompressible_momentum_ale` only. The constructor
//! builds that model; no other model is accepted.
//!
//! **Moving boundaries (roadmap M6, stage 1)**: [`BoundaryMotionSpec`] is
//! orthogonal to [`MeshMotionSpec`] — the interior seeds follow the mesh-motion
//! law, and additionally a declared moving boundary
//! ([`BoundaryMotionSpec::RigidLoop`]) moves its boundary-bound seeds RIGIDLY
//! each step (they keep their parametric position on the moving wall) while the
//! regen clips against the moved loops. `moved_spec`/`apply_boundary_motion`
//! evaluate the rigid transform from the t=0 labels (no drift); the SAME
//! `align_old_vertices_by_seed_set` + `swept_mesh_fluxes_closed` path closes the
//! now-real boundary-face swept areas; and `w_wall` records the per-seed
//! material velocity for stage 2's `MovingWall` Dirichlet BC. `Static` (the
//! default) is byte-identical to the M4 static-boundary path.
//!
//! Scope so far: [`MeshMotionSpec::Frozen`] (stage 1 — the static-limit
//! plumbing) and [`MeshMotionSpec::Prescribed`] (stage 2 — analytic seed motion
//! through the full Voronoi-regen + swept-flux path; stage 3 — Voronoi
//! topology FLIPS via the born/dead-face conservative remap: born faces carry
//! zero swept contribution and the per-cell defect is closed onto the slack
//! faces by the spanning forest, so `Σ_f σ·flux = ΔV_i/dt` stays exact per cell
//! and free-stream flow survives the flip). Stage 4 adds `FlowCoupled`: the
//! seeds move with the readback cell velocity plus an AREPO-style
//! distortion-ramped centroid steering (`χ·(centroid − seed)`), clamped per
//! step to a fraction of the local cell radius, with a Lloyd regularization
//! escalation when the regenerated mesh's skew rises — a genuinely
//! flow-following moving mesh.

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
/// radius `R_i = √(V_i/π)`. Under `FlowCoupled`, a seed cannot move more than
/// `FLOW_DISP_CAP · R_i` in one step regardless of the flow speed — a hard
/// anti-tangling clamp that keeps the swept-quad linear-motion assumption and
/// the born/dead-face flip remap in their valid regime.
pub const DEFAULT_FLOW_DISP_CAP: f64 = 0.25;

/// Default AREPO distortion trigger `η` (Springel 2010, Eq. 63 style): the
/// centroid steering ramps in once a cell's seed-to-centroid offset exceeds
/// `η · R_i` and saturates at `1.1 η`. Below `0.9 η` the steering is OFF, so
/// well-shaped cells advect PURELY with the flow (the milestone premise — a
/// moving mesh must reduce advective dissipation, not regularize it away).
pub const DEFAULT_AREPO_ETA: f64 = 0.25;

/// Default max-skewness above which a `FlowCoupled` step runs a Lloyd
/// regularization escalation on the advected seeds before committing. Chosen
/// below the solver's comfortable skew band so quality is caught rising, not
/// after it has hurt the solve.
pub const DEFAULT_QUALITY_SKEW_TARGET: f64 = 0.5;

/// Default mesh-motion CFL cap factor (`dt ≤ cfl_mesh · min_h / max|w|`).
/// Conservative (0.2) so a fast seed cannot sweep more than ~a fifth of a cell
/// per step — the regime where the swept-quad linear-motion assumption and the
/// warm-start-validity argument hold. Inert under `Frozen` (max|w| = 0).
pub const DEFAULT_MESH_CFL: f64 = 0.2;

/// How the seeds move each step.
///
/// `Frozen` and `Prescribed` are implemented; `FlowCoupled` is declared for the
/// milestone shape and rejected by [`MovingMeshDriver::step`] until its stage.
#[derive(Clone, Copy)]
pub enum MeshMotionSpec {
    /// Seeds never move (regen reproduces the same mesh byte-for-byte). The
    /// static-limit / do-no-harm case.
    Frozen,
    /// Prescribed analytic motion `new_pos = f(seed0, t)` from the t=0 seed
    /// label and absolute time (interior seeds only; boundary seeds fixed in
    /// v1). Voronoi topology flips are handled by the born/dead-face
    /// conservative remap (stage 3).
    Prescribed(fn([f64; 2], f64) -> [f64; 2]),
    /// Flow-coupled motion (cell velocity + AREPO centroid steering); the
    /// `regularization` is the steering strength χ (stage 4).
    FlowCoupled { regularization: f64 },
}

/// How the *boundary* moves (roadmap M6). Orthogonal to [`MeshMotionSpec`],
/// which governs the INTERIOR seeds: the interior seeds always follow the
/// `MeshMotionSpec` law, and additionally, when a boundary is declared moving
/// here, its boundary-bound seeds move RIGIDLY with it each step (staying
/// exactly on the moving boundary) while the mesh regenerates against the moved
/// loops.
///
/// `Static` is the M0–M4 behaviour (all boundary seeds held); the moving-mesh
/// path is byte-identical to a static-boundary run under it. `RigidLoop`
/// prescribes an analytic rigid transform for ONE boundary loop (the obstacle):
/// an oscillating cylinder is `transform(t, p) = [p.x + A·sin(ω t), p.y]`.
#[derive(Clone, Copy)]
pub enum BoundaryMotionSpec {
    /// All boundaries fixed (v1 default). The moving-mesh loop under this is
    /// byte-identical to the M4 static-boundary path (the do-no-harm anchor).
    Static,
    /// A rigidly-moving boundary loop. `loop_index` selects the loop in the
    /// [`BoundarySpec`] that moves (e.g. the obstacle circle is loop 1 of a
    /// [`crate::meshgen::ChannelWithObstacle`]); `transform(t, p)` maps a t=0
    /// point to its position at absolute time `t`. It is applied to BOTH the
    /// loop's polyline points (so the regen clips against the moved wall) and the
    /// loop's boundary seeds (so seed `i` stays on the moving wall). **Contract:**
    /// `transform(0, p) == p` — the driver is built on the t=0 mesh, so the
    /// motion law must be the identity at t=0 (an oscillation `A·sin(ω t)`
    /// satisfies this).
    ///
    /// **v1 scope: pure TRANSLATION.** A rigid translation preserves chord lengths
    /// ⇒ the loop's seed count / segment structure are invariant (fixed-seed +
    /// watertightness carry over), and — critically for the `MovingWall` BC — the
    /// per-seed material velocity `w_wall` recorded from the seed's `(new−old)/dt`
    /// is UNIFORM across the seed's wall face, exactly matching the per-vertex
    /// swept `mesh_flux` (all chords sweep the same displacement). A ROTATION would
    /// break that match: the face's vertices sit at different radii/angles than the
    /// seed, so the uniform seed-velocity Dirichlet no longer cancels the per-face
    /// swept flux (a spurious wall mass flux O(ω·Δr)). Rotation is left to a future
    /// stage that evaluates `w_wall` per wall-face rather than per seed; do not
    /// pass a rotating `transform` in v1.
    RigidLoop {
        loop_index: usize,
        transform: fn(f64, [f64; 2]) -> [f64; 2],
    },
    /// A sinusoidally-oscillating boundary loop (the headline M6 demo — an
    /// oscillating cylinder). A first-class variant that carries its own
    /// `amplitude`/`omega`/`axis` so the analytic rigid map is `p ↦ p +
    /// amplitude·sin(ω t)·ê_axis` — identity at t=0 (`sin 0 = 0`), so the
    /// build-mesh contract holds — WITHOUT a captured closure (the `RigidLoop`
    /// `fn` pointer cannot carry runtime-tuned parameters, e.g. from the GUI
    /// sliders). It is a pure rigid translation, so the chord-length /
    /// fixed-seed / watertightness invariants carry over exactly as for
    /// `RigidLoop`. `axis` selects in-line (`InLine`, cross-stream-free) vs
    /// cross-stream (`CrossStream`) forcing.
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

/// Per-step moving-mesh telemetry (the always-on diagnostics the M4 gates and
/// the UI observe).
#[derive(Clone, Copy, Debug)]
pub struct MovingMeshStats {
    /// Wall time of the pre-regen seed-motion planning (ms): the dt handshake,
    /// seed advection, the FlowCoupled velocity readback, AND the quality-escalation
    /// PROBE regen (which runs a full `assemble_meshless_from_seeds` before the
    /// authoritative regen). Counted separately so the moving-mesh overhead split
    /// is HONEST — this work would otherwise fall into the `solve` residual (review
    /// July 2026). 0 on the skip-regen passthrough.
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
    /// rebuild) rather than the surgical geometry seam. Note: any real seed
    /// motion reorders the deterministic face emission, so this is `true` every
    /// motion step — it is NOT a flip flag (see `flipped` for the genuine
    /// adjacency-change flag). Always `false` under `Frozen` (byte-identical
    /// regen ⇒ geometry seam).
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
    /// faces. O(motion·h/V) on a flip step, 0 otherwise. The always-on flip
    /// diagnostic (roadmap: "defect magnitude is the diagnostic"). NOT a GCL
    /// error — `scl_defect` (the post-closure per-cell sum) stays at roundoff.
    pub flip_defect: f64,
    /// The pinned dt the swept fluxes were closed against AND the solver
    /// stepped with (they are equal by the F2 handshake). f64-widened f32 —
    /// exactly `params.requested_dt as f64`.
    pub dt: f64,
}

/// The moving-mesh loop driver (roadmap M4). Owns the authoritative seed set,
/// the current realized mesh, and the wrapped [`SolverDriver`].
pub struct MovingMeshDriver {
    driver: SolverDriver,
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
    /// each step. Caching it (rather than reading back the already-capped
    /// `params.requested_dt`) keeps the pinned dt from ratcheting monotonically
    /// downward: a transient fast step must not permanently lower the timestep
    /// after the flow slows (review July 2026).
    configured_dt: f64,
    /// The current realized mesh (owned; the driver holds it across steps).
    mesh: Mesh,
    /// Vertex positions of `mesh` at t^n (the swept-quad `old` positions).
    prev_vx: Vec<f64>,
    prev_vy: Vec<f64>,
    /// Seed motion law (INTERIOR seeds).
    motion: MeshMotionSpec,
    /// Boundary motion law (roadmap M6). `Static` ⇒ the M4 path (byte-identical).
    boundary_motion: BoundaryMotionSpec,
    /// Per-seed material velocity `w_wall = (new_pos − old_pos)/dt` of the last
    /// committed step, seed `i` == cell `i`. Zero for interior + static-boundary
    /// seeds; the rigid boundary velocity for moving-wall seeds. Stored here for
    /// stage 2's `MovingWall` Dirichlet BC (the Dirichlet U at the wall must be
    /// the wall's material velocity). All-zero until the first moving step.
    w_wall: Vec<[f64; 2]>,
    /// M6 stage 2: feed the moving wall's material velocity into the fluid.
    /// When `true` (opt-in via [`set_moving_wall_bc`]) AND a `RigidLoop`
    /// boundary motion is declared, each regenerated mesh's moving-loop open
    /// faces are re-tagged [`BoundaryType::MovingWall`] (from the engine's
    /// default `Wall`) and, after the ALE refresh, their per-face Dirichlet
    /// velocity `bc_value` is set to `w_wall[owner]` — so the no-slip ghost at
    /// the wall carries the wall's material velocity (no-penetration +
    /// no-slip). Static + fixed walls stay `Wall`/`Slip`. Default `false`
    /// (stage-1 behaviour: the obstacle stays a static-velocity `Wall`).
    moving_wall_bc: bool,
    /// Mesh-motion CFL cap factor.
    mesh_cfl: f64,
    /// Whether `step` regenerates + refreshes each step. `false` = the
    /// do-no-harm skip-regen variant (a pure `SolverDriver::step` passthrough,
    /// byte-identical to a static run).
    regen_each_step: bool,
    /// Optional per-regen boundary-tag rewrite, applied to each freshly
    /// regenerated mesh before the ALE seam. The M0 engine tags the domain box
    /// sides by a FIXED rule (left Inlet / right Outlet / bottom+top Wall), so a
    /// run that wants a different arrangement (e.g. slip channel walls, or
    /// inlet/outlet on the bottom) must re-stamp `face_boundary` every step —
    /// the topology seam rebuilds the bc tables from the regenerated mesh's
    /// tags, so a one-time retag of the initial mesh would be lost. Touches only
    /// `face_boundary` (never geometry/adjacency), so it does not affect the
    /// swept fluxes or the topology-diff. `None` = keep the engine's tags.
    boundary_retag: Option<fn(&mut Mesh)>,
    /// Route every step through the topology seam even when the topology is
    /// unchanged (default `false` — use the geometry seam when it suffices).
    /// The topology seam clears the AMG hierarchy + re-scatters bc tables, so
    /// this deliberately perturbs the solve away from a static run; it exists to
    /// MEASURE that perturbation (the do-no-harm finding).
    force_topology_seam: bool,
    /// Per-step seed-displacement cap (fraction of local cell radius), FlowCoupled.
    flow_disp_cap: f64,
    /// AREPO centroid-steering distortion trigger `η`, FlowCoupled.
    arepo_eta: f64,
    /// Regenerated-mesh max-skew above which a FlowCoupled step runs a Lloyd
    /// regularization escalation on the advected seeds.
    quality_skew_target: f64,
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
}

impl MovingMeshDriver {
    /// Build a moving-mesh driver from an initial CVT mesh + its authoritative
    /// seeds ([`crate::meshgen::meshless::generate_cvt_mesh_with_seeds`]).
    ///
    /// Constructs the wrapped [`SolverDriver`] with the `incompressible_momentum_ale`
    /// model on `cvt.mesh`, seeded with `initial_u`/`initial_p`. `params` must
    /// have `adaptive_dt == false` (the ALE seam SCL-closes fluxes against a
    /// fixed dt; an adaptive re-scale would break the GCL — rejected here, up
    /// front, rather than at the first `begin_ale_step_topology`).
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
        if params.adaptive_dt {
            return Err(
                "MovingMeshDriver requires params.adaptive_dt == false: the swept mesh fluxes \
                 are SCL-closed against a fixed dt, and an adaptive re-scale after the closure \
                 silently violates the GCL (the mesh-motion CFL cap is applied by the driver \
                 itself)."
                    .into(),
            );
        }
        let model = incompressible_momentum_ale_model()?;
        let CvtMeshSeeds {
            mesh,
            seeds,
            kinds,
            spec,
            domain,
            min_cell_size,
        } = cvt;

        let build = SolverDriver::build(
            &mesh, model, params, initial_u, initial_p, device, queue,
        )
        .await?;

        let prev_vx = mesh.vx.clone();
        let prev_vy = mesh.vy.clone();
        let n_seeds = seeds.len();

        Ok(Self {
            driver: build.driver,
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
            lloyd_escalation_iters: 1,
            lloyd_escalation_omega: 0.4,
            step_index: 0,
            last_escalated: false,
        })
    }

    /// Set the mesh-motion CFL cap factor (default [`DEFAULT_MESH_CFL`]).
    pub fn set_mesh_cfl(&mut self, cfl: f64) {
        self.mesh_cfl = cfl;
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

    /// Declare a moving boundary (roadmap M6). `Static` (the default) is the
    /// M4 path; [`BoundaryMotionSpec::RigidLoop`] moves one loop's boundary
    /// seeds rigidly each step. Orthogonal to [`set_motion`]: the interior
    /// seeds still follow the [`MeshMotionSpec`]. Panics if a `RigidLoop`
    /// `loop_index` is out of range for the current boundary spec.
    pub fn set_boundary_motion(&mut self, boundary_motion: BoundaryMotionSpec) {
        if let Some(loop_index) = boundary_motion.loop_index() {
            assert!(
                loop_index < self.spec.loops.len(),
                "BoundaryMotionSpec moving loop_index {loop_index} out of range \
                 ({} loops)",
                self.spec.loops.len()
            );
        }
        self.boundary_motion = boundary_motion;
    }

    /// Per-seed material velocity `w_wall` (seed `i` == cell `i`) of the last
    /// committed step — zero for interior + static-boundary seeds, the rigid
    /// wall velocity for moving-wall seeds. The stage-2 `MovingWall` BC reads
    /// this to set the Dirichlet wall velocity per boundary face.
    pub fn w_wall(&self) -> &[[f64; 2]] {
        &self.w_wall
    }

    /// M6 stage 2: enable feeding the moving wall's material velocity into the
    /// fluid (opt-in; default off). With this on AND a
    /// [`BoundaryMotionSpec::RigidLoop`] declared, each regenerated mesh's
    /// moving-loop open faces are re-tagged [`BoundaryType::MovingWall`] and
    /// their per-face Dirichlet velocity `bc_value` is set to `w_wall[owner]`
    /// after the ALE refresh, so the fluid satisfies no-slip AND no-penetration
    /// at the wall's material velocity (`U_wall = w_wall`; the convective part
    /// already sees the relative velocity `phi − ρ·mesh_flux` from the M3 ALE
    /// path — this is the other half, the Dirichlet wall value). Inert under
    /// `Static` (no moving seeds ⇒ no faces tagged) and byte-neutral for a run
    /// that never enables it (stage-1 keeps the obstacle a static `Wall`).
    pub fn set_moving_wall_bc(&mut self, enable: bool) {
        self.moving_wall_bc = enable;
    }

    /// Disable per-step regeneration: `step` becomes a pure `SolverDriver::step`
    /// passthrough (no seed advection, no regen, no ALE refresh). The
    /// do-no-harm anchor — byte-identical to a static ALE run — used by the
    /// skip-regen gate. Only meaningful with `Frozen` motion.
    pub fn set_regen_each_step(&mut self, regen: bool) {
        self.regen_each_step = regen;
    }

    /// Advance one moving-mesh step; returns the solver outcome + the per-step
    /// moving-mesh telemetry.
    pub fn step(&mut self, readback: bool) -> Result<(StepOutcome, MovingMeshStats), String> {
        // Re-assert the dt handshake invariant on EVERY path (review July 2026):
        // `adaptive_dt` is rejected at build and at both ALE seams, but the
        // skip-regen passthrough never reaches an ALE seam, so a caller that
        // flipped it on via `driver_mut().apply_params(..)` after build could
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

        // Skip-regen (do-no-harm) variant: pin the fixed dt and step. No seed
        // motion, no regen, no ALE refresh — a pure passthrough, byte-identical
        // to a static ALE run driven by `SolverDriver::step`.
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
            };
            return Ok((outcome, stats));
        }

        // 1-2. Motion: pin the fixed dt (mesh-motion CFL capped) BEFORE the
        //      swept fluxes are closed — so the flux dt == the step dt exactly —
        //      and advect the interior seeds (boundary seeds fixed in v1).
        //      * Frozen/Prescribed: the analytic label path (`f(seed0, t)`).
        //      * FlowCoupled: read back the current cell velocities, cap dt off
        //        the flow speed, then displace each seed by `U_i·dt` plus an
        //        AREPO distortion-ramped centroid steering, clamped to a
        //        fraction of the local cell radius.
        let plan_start = Instant::now();
        let (dt, mut new_seeds) = self.plan_seed_motion()?;
        let new_time = self.time + dt;
        // M6: move the boundary-bound seeds RIGIDLY with the moving boundary at
        // t^{n+1} — overwrites the moving loop's boundary seeds (they keep their
        // parametric position on the moving wall); interior + static-boundary
        // seeds are untouched. A no-op under `BoundaryMotionSpec::Static`.
        self.apply_boundary_motion(new_time, &mut new_seeds);
        // The boundary spec at t^{n+1}: the moved loops the regen clips against
        // (an identity clone of `self.spec` under Static ⇒ byte-identical regen).
        let step_spec = self.moved_spec(new_time);

        // 2b. FlowCoupled quality escalation: if the advected seed set would
        //     regenerate a too-skewed mesh, run a gentle Lloyd regularization
        //     pass (reusing the meshgen Lloyd machinery) to pull it back toward
        //     CVT before the authoritative regen — "more steering if quality
        //     rises". A no-op for Frozen/Prescribed and for well-shaped sets.
        let (new_seeds, escalated) = self.maybe_quality_escalate(&step_spec, new_seeds);
        // The pre-regen planning cost (readback + escalation probe regen): timed
        // into its own bucket so the overhead split stays honest.
        let plan_ms = ms_since(plan_start);

        // M6: record the per-seed material velocity w_wall = (new − old)/dt for
        // THIS step's wall motion, while `self.seeds` still holds the t^n set. It
        // is computed here (before the solve, not after) so the stage-2
        // `MovingWall` Dirichlet BC applied below feeds this step's solve the
        // wall velocity consistent with the mesh motion swept during [t^n,t^{n+1}].
        // Zero for interior + static-boundary seeds (a no-op under `Static`).
        self.record_wall_velocity(&new_seeds, dt);

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
        // M6 stage 2: re-tag the moving loop's open faces MovingWall (from the
        // engine's default Wall) so the ALE seam builds a MovingWall face list to
        // receive the wall velocity below. face_boundary only — no effect on the
        // swept fluxes or the topology-diff. No-op unless moving_wall_bc is on.
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
        // from the previous mesh. Stage-2 finding: seed motion can REORDER the
        // deterministic face emission without changing the adjacency, so this is
        // NOT a flip flag — it just decides geometry-vs-topology seam below. (In
        // practice a swirl toggles it on ~1/5 of steps; a rigid translation
        // trips it almost immediately.) The genuine-flip discriminator is the
        // VERTEX correspondence: a born vertex — an incident-seed set with no
        // t^n counterpart — is precisely what the swept-quad path cannot close.
        let face_arrays_differ = topology_differs(&self.mesh, &new_mesh);

        // 4. Swept-quad mesh fluxes old→new with the pinned dt. Across the
        //    Voronoi regen the new mesh's vertex ids are unrelated to the old
        //    mesh's, so we first map each NEW vertex to its t^n position via the
        //    seed-set correspondence (roadmap R2: vertex ≡ seed-triple).
        //
        //    Then we detect ADJACENCY flips (born/dead faces). Two regimes:
        //    * NO flip (`is_flip()` false — persistent adjacency, at worst a
        //      face-array reorder): the aligned old ring reproduces each cell's
        //      t^n polygon, so the M3 f64-telescoping + f32-forest-closure path
        //      applies directly and its telescoping identity is HARD-asserted.
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
        let genuine_flip = flip.is_flip() || unmatched != 0;
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
        let is_flip = genuine_flip || degen > 0;
        let swept = if is_flip {
            // Degeneracy-ONLY step (no genuine adjacency change): keep the >1e-9
            // telescoping-identity assert LIVE on every cell not incident to a
            // forced-degenerate face — a lone sliver must not disable the whole-step
            // guard (review July 2026 F1). On a genuine flip, pass None (relaxed).
            let hard_assert_exclude = if genuine_flip {
                None
            } else {
                Some(flip.cell_flipped.as_slice())
            };
            // The actual t^n cell volumes (`self.mesh` is still the old mesh
            // here) are the closure target's old-volume — the born vertices'
            // aligned old positions cannot reconstruct the old polygon across a
            // flip, so the ring-reconstructed old volume would be wrong.
            swept_mesh_fluxes_closed_flip(
                &new_mesh,
                &old_vx_aligned,
                &old_vy_aligned,
                &self.mesh.cell_vol,
                &flip.born_face_mask,
                hard_assert_exclude,
                dt,
            )?
        } else {
            swept_mesh_fluxes_closed(&new_mesh, &old_vx_aligned, &old_vy_aligned, dt)?
        };
        let swept_ms = ms_since(swept_start);

        // 5. Refresh (rotate volume history → rebuild/upload geometry → upload
        //    closed fluxes). SEAM CHOICE:
        //    * Face arrays BYTE-IDENTICAL (a frozen regen): the surgical
        //      GEOMETRY seam re-uploads geometry while keeping the AMG hierarchy
        //      + per-face BC overrides — a zero-motion step is byte-identical to
        //      a static run (the stage-1 do-no-harm anchor).
        //    * Face arrays differ (ANY real motion reorders the face emission):
        //      the geometry seam would upload against a stale face order, so we
        //      take the CPU-surgical TOPOLOGY seam (rebuilds the face-indexed
        //      CSR stack for the new order; cell-indexed state incl. the
        //      warm-start `x` and BDF2 history is preserved — M3 proved this
        //      holds the GCL at ~1e-6). The rebuild re-scatters bc tables from
        //      the model per-type defaults, dropping per-face overrides, so we
        //      re-apply them (`bc_overrides_reset`).
        //    A genuine flip changes adjacency ⇒ `face_arrays_differ` is already
        //    true; the explicit `is_flip` guard makes the coupling defensive.
        let refresh_start = Instant::now();
        if face_arrays_differ || is_flip || self.force_topology_seam {
            let report = self
                .driver
                .begin_ale_step_topology(&new_mesh, &swept.fluxes)?;
            if report.bc_overrides_reset {
                self.driver.reapply_boundary_conditions();
            }
        } else {
            self.driver.begin_ale_step(&new_mesh, &swept.fluxes)?;
        }
        // M6 stage 2: feed the moving wall's material velocity into the fluid.
        // AFTER the refresh (the seam just rebuilt the MovingWall face list + BC
        // tables) and BEFORE the solve, set each MovingWall face's Dirichlet
        // velocity to w_wall[owner] — no-slip + no-penetration at the wall's
        // material velocity. No-op unless moving_wall_bc is on.
        self.apply_moving_wall_velocity(&new_mesh)?;
        let refresh_ms = ms_since(refresh_start);
        let topo_changed = face_arrays_differ || is_flip;

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
        };
        Ok((outcome, stats))
    }

    /// The seed positions at absolute time `t`, evaluated from the t=0 labels
    /// `f(seed0_i, t)` for interior seeds; boundary seeds are held fixed (v1).
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
                        // Boundary seeds are fixed in v1 (moving boundaries are
                        // M6); this keeps boundary faces static ⇒ zero mesh flux.
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

    // -----------------------------------------------------------------------
    // M6 — moving boundary (rigidly-moving boundary seeds + moved loop spec).
    // -----------------------------------------------------------------------

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
    /// byte-for-byte (the do-no-harm anchor). A rigid transform preserves chord
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

    /// M6 stage 2: re-tag the moving loop's open (boundary) faces as
    /// [`BoundaryType::MovingWall`] on a freshly regenerated mesh. An open face
    /// whose OWNER cell is a moving-boundary seed lies on the moving obstacle
    /// contour (the boundary seeds sit ON the wall; their cells' only open face
    /// is the clipped obstacle edge), so this catches exactly the moving wall
    /// while leaving the fixed channel walls / inlet / outlet untouched. Only
    /// `face_boundary` is rewritten (geometry/adjacency untouched ⇒ swept fluxes
    /// and the topology-diff are unaffected). A no-op unless `moving_wall_bc` is
    /// on and a `RigidLoop` is declared. Must run BEFORE the ALE seam so the
    /// solver builds a `MovingWall` boundary-face list to receive the velocity.
    fn retag_moving_wall_faces(&self, mesh: &mut Mesh) {
        if !self.moving_wall_bc || self.moving_loop_range().is_none() {
            return;
        }
        for f in 0..mesh.num_faces() {
            if mesh.face_neighbor[f].is_none() && self.is_moving_boundary_seed(mesh.face_owner[f]) {
                mesh.face_boundary[f] = Some(BoundaryType::MovingWall);
            }
        }
    }

    /// M6 stage 2: push the per-face Dirichlet wall velocity into the solver for
    /// the current step. Each `MovingWall` open face gets `bc_value = w_wall`
    /// of its owner cell (the rigid wall material velocity — uniform across the
    /// face under the v1 pure-translation scope; a rotating wall would need a
    /// per-face `w_wall`, out of v1). Re-applied EVERY step: the topology
    /// seam re-scatters the bc tables from the model's per-type defaults
    /// (MovingWall Dirichlet 0) dropping the per-face override, and `w_wall`
    /// itself changes each step. A no-op unless `moving_wall_bc` is on. Called
    /// AFTER the refresh (so the `MovingWall` face list + tables are rebuilt)
    /// and BEFORE the solve (so this step feels this step's wall velocity).
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
    fn plan_seed_motion(&mut self) -> Result<(f64, Vec<Point2<f64>>), String> {
        match self.motion {
            // Frozen: no INTERIOR motion. `advect_to` returns the labels; the dt
            // is capped only by the moving BOUNDARY speed (M6), zero otherwise —
            // so a fully static step pins exactly `requested_dt`.
            MeshMotionSpec::Frozen => {
                let w_max = self.max_boundary_speed(self.configured_dt);
                let dt = self.pin_dt(w_max);
                Ok((dt, self.advect_to(self.time + dt)))
            }
            // Prescribed: FD-estimate max seed speed over the base step, cap dt,
            // then sample the analytic law at the pinned t^{n+1}. The moving
            // boundary speed (M6) also enters the cap.
            MeshMotionSpec::Prescribed(_) => {
                let dt_base = self.configured_dt;
                let w_max = self
                    .max_seed_speed(dt_base)
                    .max(self.max_boundary_speed(dt_base));
                let dt = self.pin_dt(w_max);
                Ok((dt, self.advect_to(self.time + dt)))
            }
            MeshMotionSpec::FlowCoupled { regularization } => {
                self.plan_flow_coupled(regularization)
            }
        }
    }

    /// FlowCoupled seed motion (deliverable 1): read the current cell velocities,
    /// pin dt off the flow speed (mesh-motion CFL), then displace each interior
    /// seed by `U_i·dt` plus an AREPO-style distortion-ramped steering toward its
    /// cell centroid, clamped to a fraction of the local cell radius.
    ///
    /// * The steering `χ_i·(c_i − s_i)` is a partial (under-relaxed) Lloyd move.
    ///   `χ_i` ramps from 0 (well-shaped cell — pure flow advection, so the mesh
    ///   follows the flow and cuts advective dissipation) up to `regularization`
    ///   once the seed-to-centroid offset exceeds `η·R_i` (Springel 2010 Eq. 63
    ///   style). This is what holds quality WITHOUT overwhelming advection.
    /// * The per-step displacement clamp `|Δ| ≤ flow_disp_cap·R_i` is the hard
    ///   anti-tangling guard (keeps the swept-quad + flip-remap regime valid).
    /// * The mesh-motion CFL cap on dt is sized from the flow speed; the steering
    ///   displacement is bounded independently by the clamp, so it never violates
    ///   the GCL (fluxes are geometric, closed against exactly this pinned dt).
    fn plan_flow_coupled(&mut self, chi_max: f64) -> Result<(f64, Vec<Point2<f64>>), String> {
        use std::f64::consts::PI;
        let u = self.read_cell_velocities()?;
        // dt handshake: cap off the max interior flow speed.
        let mut w_flow_max = 0.0f64;
        for (i, &(ux, uy)) in u.iter().enumerate() {
            if self.kinds[i] != SeedKind::Interior {
                continue;
            }
            w_flow_max = w_flow_max.max((ux * ux + uy * uy).sqrt());
        }
        // The moving boundary speed (M6) also enters the dt cap.
        let w_max = w_flow_max.max(self.max_boundary_speed(self.configured_dt));
        let dt = self.pin_dt(w_max);

        let eta = self.arepo_eta;
        let cap_frac = self.flow_disp_cap;
        let mut new_seeds = self.seeds.clone();
        for i in 0..self.seeds.len() {
            if self.kinds[i] != SeedKind::Interior {
                continue; // boundary seeds fixed (v1)
            }
            let s = self.seeds[i];
            let (ux, uy) = u[i];
            // Effective cell radius (uniform-density Lloyd target is the geometric
            // centroid, already stored on the mesh as cell_cx/cell_cy).
            let r = (self.mesh.cell_vol[i].max(0.0) / PI).sqrt().max(1e-30);
            let dcx = self.mesh.cell_cx[i] - s.x;
            let dcy = self.mesh.cell_cy[i] - s.y;
            let dist = (dcx * dcx + dcy * dcy).sqrt();
            // AREPO ramp on the distortion ratio d/R.
            let ratio = dist / r;
            let chi = if ratio < 0.9 * eta {
                0.0
            } else if ratio < 1.1 * eta {
                chi_max * (ratio - 0.9 * eta) / (0.2 * eta)
            } else {
                chi_max
            };
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
        Ok((dt, new_seeds))
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

    /// FlowCoupled quality escalation (deliverable 2): if the advected seed set
    /// would regenerate a mesh whose max skew exceeds `quality_skew_target`, run
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
        if probe.calculate_max_skewness() <= self.quality_skew_target {
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
/// Keeps the flip telemetry honest (review July 2026): a forced-degenerate face
/// marks its incident cells in `flip.cell_flipped` and re-derives
/// `flip.flipped_cells`, so a degeneracy-only step reports `flipped_cells > 0`
/// (not a `flipped=true, flipped_cells=0` phantom). That per-cell flag is ALSO
/// the hard-assert exclude set the caller passes to the flip closure.
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
/// The `face_boundary` comparison is load-bearing for the M6 `Wall → MovingWall`
/// retag (review July 2026, HIGH): the geometry seam ([`Mesh::refresh_mesh_geometry`])
/// asserts identical `face_boundary` and hard-errors on a tag flip, but a step
/// whose motion is too small to change connectivity (e.g. the GUI slider minima)
/// would otherwise take that seam and die on step 0. Including the tags here
/// routes any tag change through `begin_ale_step_topology`, which rebuilds the BC
/// tables + `face_boundary` snapshot; `apply_moving_wall_velocity` then repopulates
/// the values. Byte-neutral under `Static`/no-retag (tags identical ⇒ `false`),
/// and it hardens every `boundary_retag` path too. Cheap (O(faces)).
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
