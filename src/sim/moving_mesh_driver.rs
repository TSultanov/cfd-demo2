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
//! Stage-1 scope: the [`MeshMotionSpec::Frozen`] leg (seeds never move — the
//! plumbing proven before any motion). `Prescribed`/`FlowCoupled` are declared
//! but not yet wired (they return an error from `step`).

use std::time::Instant;

use nalgebra::{Point2, Vector2};

use super::{RuntimeParams, SolverDriver, StepOutcome};
use crate::meshgen::meshless::{assemble_meshless_from_seeds, BoundarySpec, CvtMeshSeeds, SeedKind};
use crate::solver::mesh::{swept_mesh_fluxes_closed, Mesh};
use crate::solver::model::incompressible_momentum_ale_model;

/// Default mesh-motion CFL cap factor (`dt ≤ cfl_mesh · min_h / max|w|`).
/// Conservative (0.2) so a fast seed cannot sweep more than ~a fifth of a cell
/// per step — the regime where the swept-quad linear-motion assumption and the
/// warm-start-validity argument hold. Inert under `Frozen` (max|w| = 0).
pub const DEFAULT_MESH_CFL: f64 = 0.2;

/// How the seeds move each step.
///
/// Stage 1 implements [`MeshMotionSpec::Frozen`] only; the other variants are
/// declared for the milestone shape and rejected by [`MovingMeshDriver::step`]
/// until their stages land.
#[derive(Clone, Copy)]
pub enum MeshMotionSpec {
    /// Seeds never move (regen reproduces the same mesh byte-for-byte). The
    /// static-limit / do-no-harm case.
    Frozen,
    /// Prescribed analytic motion `new_pos = f(seed, t)` (stage 2).
    Prescribed(fn([f64; 2], f64) -> [f64; 2]),
    /// Flow-coupled motion (cell velocity + AREPO centroid steering); the
    /// `regularization` is the steering strength χ (stage 4).
    FlowCoupled { regularization: f64 },
}

impl MeshMotionSpec {
    fn label(&self) -> &'static str {
        match self {
            MeshMotionSpec::Frozen => "Frozen",
            MeshMotionSpec::Prescribed(_) => "Prescribed",
            MeshMotionSpec::FlowCoupled { .. } => "FlowCoupled",
        }
    }
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
    /// The regenerated mesh's face set / adjacency differs from the previous
    /// step's (a Voronoi flip). Always `false` under `Frozen` (byte-identical
    /// regen); a real signal once seeds move.
    pub topo_changed: bool,
    /// The pinned dt the swept fluxes were closed against AND the solver
    /// stepped with (they are equal by the F2 handshake). f64-widened f32 —
    /// exactly `params.requested_dt as f64`.
    pub dt: f64,
}

/// The moving-mesh loop driver (roadmap M4). Owns the authoritative seed set,
/// the current realized mesh, and the wrapped [`SolverDriver`].
pub struct MovingMeshDriver {
    driver: SolverDriver,
    /// Authoritative seed positions (f64), seed `i` == cell `i`.
    seeds: Vec<Point2<f64>>,
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
            regen_each_step: true,
            force_topology_seam: false,
            step_index: 0,
        })
    }

    /// Set the mesh-motion CFL cap factor (default [`DEFAULT_MESH_CFL`]).
    pub fn set_mesh_cfl(&mut self, cfl: f64) {
        self.mesh_cfl = cfl;
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
                dt,
            };
            return Ok((outcome, stats));
        }

        // 2. Advect the seeds per the motion law (also yields max|w| for the
        //    dt cap). Frozen ⇒ unchanged seeds, max|w| = 0.
        let (new_seeds, w_max) = self.advect_seeds()?;

        // 1. dt handshake: pin the fixed dt (mesh-motion CFL capped) BEFORE the
        //    swept fluxes are closed, so the flux dt == the step dt exactly.
        let dt = self.pin_dt(w_max);

        // 3. Regenerate the mesh from the advected seeds (deterministic; a
        //    frozen seed set reproduces `self.mesh` byte-for-byte).
        let regen_start = Instant::now();
        let new_mesh = assemble_meshless_from_seeds(
            &new_seeds,
            &self.kinds,
            &self.spec,
            self.domain,
            self.min_cell_size,
        );
        let regen_ms = ms_since(regen_start);
        if new_mesh.num_cells() != self.mesh.num_cells() {
            return Err(format!(
                "MovingMeshDriver: regen changed the cell count ({} -> {}); v1 is fixed-seed \
                 (cells change shape, not existence)",
                self.mesh.num_cells(),
                new_mesh.num_cells()
            ));
        }
        let topo_changed = topology_differs(&self.mesh, &new_mesh);

        // 4. Swept-quad mesh fluxes old→new with the pinned dt. `prev_vx/vy`
        //    are `self.mesh`'s t^n vertex positions; a frozen regen makes them
        //    equal to `new_mesh`'s vertices ⇒ all-zero fluxes.
        let swept_start = Instant::now();
        let swept = swept_mesh_fluxes_closed(&new_mesh, &self.prev_vx, &self.prev_vy, dt)?;
        let swept_ms = ms_since(swept_start);

        // 5. Refresh (rotate volume history → upload new geometry → upload
        //    closed fluxes). SEAM CHOICE (do-no-harm finding, stage 1): use the
        //    GEOMETRY seam whenever the face set / adjacency is unchanged — the
        //    CPU geometry refresh is fully surgical (re-uploads geometry, keeps
        //    the AMG hierarchy, keeps per-face BC overrides), so a zero-motion
        //    (frozen) step is byte-identical to a static run. The TOPOLOGY seam
        //    is required only on a real flip: it rebuilds the CSR stack AND
        //    clears the Schur-AMG hierarchy + re-scatters bc tables from the
        //    model per-type defaults — both of which perturb a subsequent solve
        //    away from a static run (the AMG reset alone drives an O(1) drift on
        //    a developing channel). `force_topology_seam` routes every step
        //    through the topology seam to MEASURE that perturbation.
        let refresh_start = Instant::now();
        if topo_changed || self.force_topology_seam {
            let report = self
                .driver
                .begin_ale_step_topology(&new_mesh, &swept.fluxes)?;
            // The topology rebuild re-scatters bc tables from the model per-type
            // defaults, dropping the per-face inlet override — re-apply it
            // (`bc_overrides_reset`), exactly as a hand-written ALE loop does.
            if report.bc_overrides_reset {
                self.driver.reapply_boundary_conditions();
            }
        } else {
            self.driver.begin_ale_step(&new_mesh, &swept.fluxes)?;
        }
        let refresh_ms = ms_since(refresh_start);

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
        self.step_index += 1;

        let stats = MovingMeshStats {
            regen_ms,
            swept_ms,
            refresh_ms,
            scl_defect: swept.max_defect_rel,
            identity_err: swept.max_identity_err_rel,
            max_skew,
            n_cells,
            n_faces,
            topo_changed,
            dt,
        };
        Ok((outcome, stats))
    }

    /// Advance the seeds by one step per [`MeshMotionSpec`], returning the new
    /// seed positions and the max seed speed `max_i |w_i|` (for the dt cap).
    fn advect_seeds(&self) -> Result<(Vec<Point2<f64>>, f64), String> {
        match self.motion {
            MeshMotionSpec::Frozen => Ok((self.seeds.clone(), 0.0)),
            other => Err(format!(
                "MovingMeshDriver: {} motion is not implemented in this stage (only Frozen)",
                other.label()
            )),
        }
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
