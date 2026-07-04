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
//! Scope so far: [`MeshMotionSpec::Frozen`] (stage 1 — the static-limit
//! plumbing) and [`MeshMotionSpec::Prescribed`] (stage 2 — analytic seed motion
//! through the full Voronoi-regen + swept-flux path; stage 3 — Voronoi
//! topology FLIPS via the born/dead-face conservative remap: born faces carry
//! zero swept contribution and the per-cell defect is closed onto the slack
//! faces by the spanning forest, so `Σ_f σ·flux = ΔV_i/dt` stays exact per cell
//! and free-stream flow survives the flip). `FlowCoupled` is declared but
//! rejected.

use std::time::Instant;

use nalgebra::{Point2, Vector2};

use super::{RuntimeParams, SolverDriver, StepOutcome};
use crate::meshgen::meshless::{assemble_meshless_from_seeds, BoundarySpec, CvtMeshSeeds, SeedKind};
use crate::solver::mesh::{
    align_old_vertices_by_seed_set, detect_flips, swept_mesh_fluxes_closed,
    swept_mesh_fluxes_closed_flip, Mesh,
};
use crate::solver::model::incompressible_momentum_ale_model;

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

/// Per-step moving-mesh telemetry (the always-on diagnostics the M4 gates and
/// the UI observe).
#[derive(Clone, Copy, Debug)]
pub struct MovingMeshStats {
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
    /// The current realized mesh (owned; the driver holds it across steps).
    mesh: Mesh,
    /// Vertex positions of `mesh` at t^n (the swept-quad `old` positions).
    prev_vx: Vec<f64>,
    prev_vy: Vec<f64>,
    /// Seed motion law.
    motion: MeshMotionSpec,
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
    /// Committed step count.
    step_index: usize,
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

        Ok(Self {
            driver: build.driver,
            seeds0: seeds.clone(),
            time: 0.0,
            seeds,
            kinds,
            spec,
            domain,
            min_cell_size,
            mesh,
            prev_vx,
            prev_vy,
            motion,
            mesh_cfl: DEFAULT_MESH_CFL,
            boundary_retag: None,
            regen_each_step: true,
            force_topology_seam: false,
            step_index: 0,
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
        // Skip-regen (do-no-harm) variant: pin the fixed dt and step. No seed
        // motion, no regen, no ALE refresh — a pure passthrough, byte-identical
        // to a static ALE run driven by `SolverDriver::step`.
        if !self.regen_each_step {
            let dt = self.pin_dt(0.0);
            let outcome = self.driver.step(readback);
            self.step_index += 1;
            let stats = MovingMeshStats {
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

        // Flow-coupled motion is a later stage (needs the on-device cell
        // velocity + AREPO steering); reject it up front with a crisp message.
        if let MeshMotionSpec::FlowCoupled { .. } = self.motion {
            return Err(
                "MovingMeshDriver: FlowCoupled motion is not implemented in this stage \
                 (Frozen + Prescribed only)"
                    .into(),
            );
        }

        // 1. dt handshake: estimate max seed speed over the upcoming base step,
        //    then pin the fixed dt (mesh-motion CFL capped) BEFORE the swept
        //    fluxes are closed, so the flux dt == the step dt exactly.
        let dt_base = self.driver.params().requested_dt as f64;
        let w_max = self.max_seed_speed(dt_base);
        let dt = self.pin_dt(w_max);
        let new_time = self.time + dt;

        // 2. Advect the seeds to t^{n+1} along the prescribed law (interior
        //    seeds only; boundary seeds are fixed in v1). Frozen ⇒ unchanged.
        let new_seeds = self.advect_to(new_time);

        // 3. Regenerate the mesh from the advected seeds (deterministic; a
        //    frozen seed set reproduces `self.mesh` byte-for-byte).
        let regen_start = Instant::now();
        let mut new_mesh = assemble_meshless_from_seeds(
            &new_seeds,
            &self.kinds,
            &self.spec,
            self.domain,
            self.min_cell_size,
        );
        // Re-stamp boundary tags (the topology seam rebuilds bc tables from
        // these) before anything downstream reads them. face_boundary only —
        // geometry/adjacency untouched.
        if let Some(retag) = self.boundary_retag {
            retag(&mut new_mesh);
        }
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
        let flip = detect_flips(&self.mesh, &new_mesh)?;
        let is_flip = flip.is_flip() || unmatched != 0;
        let swept = if is_flip {
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
        self.seeds = new_seeds;
        self.time = new_time;
        self.step_index += 1;

        // On a flip step the pre-closure per-cell residual is the intended flip
        // defect, not a telescoping-identity violation, so it is reported via
        // `flip_defect` and `identity_err` is left at 0 (the persistent identity
        // is not enforced across a flip). On a non-flip step it is the roundoff
        // telescoping residual, reported as `identity_err` and asserted by the
        // GCL gates.
        let stats = MovingMeshStats {
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

    /// Pin the fixed dt: the configured `requested_dt`, capped by the
    /// mesh-motion CFL `dt ≤ cfl_mesh · min_h / w_max`, pushed to the solver via
    /// [`SolverDriver::set_requested_dt`]. Returns the actually-pinned dt as the
    /// f64-widened f32 (`params.requested_dt as f64`) — the exact value the
    /// swept-flux closure must use so the flux dt == the step dt.
    fn pin_dt(&mut self, w_max: f64) -> f64 {
        let mut dt = self.driver.params().requested_dt as f64;
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

/// Whether two meshes with the same cell count differ in face set / adjacency
/// (a Voronoi flip). Compares the face count and the per-face owner/neighbor
/// and per-cell face lists — the connectivity the topology refresh rebuilds.
/// Cheap (O(faces)); byte-identical regen ⇒ `false`.
fn topology_differs(a: &Mesh, b: &Mesh) -> bool {
    a.num_faces() != b.num_faces()
        || a.face_owner != b.face_owner
        || a.face_neighbor != b.face_neighbor
        || a.cell_face_offsets != b.cell_face_offsets
        || a.cell_faces != b.cell_faces
}
