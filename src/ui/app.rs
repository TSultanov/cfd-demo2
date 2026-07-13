use crate::solver::cpu::structured::StructuredModelSolver;
use crate::solver::gpu::structured::{
    BcComp as StructBc, Edge as StructEdge, StructuredAutonomousStatus, StructuredGpuSolver,
    StructuredGrid,
};
use crate::solver::mesh::{
    generate_cut_cell_mesh, generate_cvt_mesh, generate_delaunay_mesh,
    generate_structured_rect_mesh, generate_structured_symmetric_nozzle_mesh,
    generate_voronoi_mesh, BackwardsStep, BoundarySides, BoundaryType, ChannelWithObstacle,
    Geometry, LloydConfig, Mesh, Nozzle,
};
use crate::solver::model::{
    allmach_pressure_ale_model, allmach_pressure_model, allmach_thermal_ale_model,
    allmach_thermal_model, compressible_model_with_eos, incompressible_momentum_ale_model,
    incompressible_momentum_model, ModelPreconditionerSpec, ModelSpec,
};
use crate::solver::model::eos::{EosRuntimeParams, EosSpec};
use crate::solver::scheme::Scheme;
use crate::solver::{
    GpuLowMachPrecondModel, LinearSolverStats, OuterStepStatus, PreconditionerType,
    TimeScheme as GpuTimeScheme, UiPortSet, UnifiedSolver,
};
use crate::trace as tracefmt;
use crate::ui::{cfd_renderer, fluid::Fluid};
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Polygon};
use nalgebra::{Point2, Vector2};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use crate::meshgen::meshless::{generate_cvt_mesh_with_seeds, CvtMeshSeeds};
use crate::sim::{
    BoundaryMotionSpec, DivergeReason, DriverBuild, FieldStats, MeshMotionSpec, MovingMeshDriver,
    MovingMeshStats, OscAxis, Readback, RegenBackend, RuntimeParams, SolverDriver, StepOutcome,
    OSC_AMPLITUDE_CELL_FRACTION,
};

/// Rendering mode for the mesh visualization
#[derive(PartialEq, Clone, Copy)]
enum RenderMode {
    /// Use egui_plot polygons (slow for large meshes)
    EguiPlot,
    /// Render directly on GPU (zero-copy)
    GpuDirect,
}

/// Discretisation mode. `Unstructured` is the incumbent face-connectivity mesh
/// (cut-cell / Voronoi / Delaunay) driven by the `SolverDriver`/`UnifiedSolver`.
/// `Structured2D` is the dense-Cartesian `TopologyMode::Structured2D` path driven
/// by a `StructuredGpuSolver` — no connectivity indirection, immersed obstacles.
/// Selecting it hides the unstructured meshers, ALE/moving-mesh, and the
/// curvilinear nozzle geometry, and narrows the model list to structured models.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum MeshMode {
    Unstructured,
    Structured2D,
}

/// Keep a pending moving-mesh request on a time integrator the ALE path can
/// construct. RK4 is intentionally static-only because its four stages do not
/// yet advance stage-consistent mesh geometry; BDF2 is the deterministic
/// implicit fallback already used by every GUI model family.
///
/// Returning whether a coercion occurred makes the policy directly testable
/// without constructing an `eframe` application. Static requests are an exact
/// no-op, so selecting RK4 for the normal fixed-mesh solver is unaffected.
fn enforce_moving_mesh_time_scheme(
    moving_mesh: bool,
    time_scheme: &mut GpuTimeScheme,
) -> bool {
    if moving_mesh && *time_scheme == GpuTimeScheme::RK4 {
        *time_scheme = GpuTimeScheme::BDF2;
        true
    } else {
        false
    }
}

impl Default for MeshMode {
    fn default() -> Self {
        Self::Unstructured
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum GeometryType {
    BackwardsStep,
    ChannelObstacle,
    /// Converging–diverging nozzle (structured, no cut cells). Paired with an
    /// all-Mach model + a sub-critical outlet back-pressure it drives the
    /// supersonic demo (subsonic inflow → choked throat → supersonic exit).
    Nozzle,
}

impl Default for GeometryType {
    fn default() -> Self {
        Self::BackwardsStep
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MeshType {
    CutCell,
    Delaunay,
    Voronoi,
    /// Meshless CVT (Lloyd-relaxed) Voronoi mesh (`generate_cvt_mesh`). Seed
    /// positions are optimized instead of running `Mesh::smooth`: vertex
    /// smoothing would move Voronoi vertices off the bisectors, so this mesh
    /// type must never be smoothed after generation.
    VoronoiCvt,
    /// Body-fitted curvilinear structured grid. Only meaningful — and only
    /// offered in the UI — for the converging–diverging nozzle geometry, whose
    /// walls it conforms to exactly (see `generate_structured_nozzle_mesh`).
    Fitted,
}

impl MeshType {
    /// The fitted (structured) mesh is a uniform curvilinear grid: it has no
    /// local refinement or grading, so only a single target cell size applies.
    /// The unstructured meshers honour the full min/max/growth sizing controls.
    fn supports_size_grading(self) -> bool {
        !matches!(self, MeshType::Fitted)
    }
}

/// Build a slider whose range expands to include the current value and which
/// never snaps a hand-typed value back to an end (`SliderClamping::Never`).
///
/// The user can type any value into the drag-value field — finer or coarser than
/// the nominal `base` range — and it stays put; the visible track adapts around it
/// so the handle stays reachable. Dragging the handle is still bounded to the
/// (adapted) track. Consumers add `.text(…)`, `.logarithmic(…)`, etc. as usual.
fn adaptive_slider<Num: egui::emath::Numeric>(
    value: &mut Num,
    base: std::ops::RangeInclusive<Num>,
) -> egui::Slider<'_> {
    let current = value.to_f64();
    let lo = (*base.start()).to_f64().min(current);
    let hi = (*base.end()).to_f64().max(current);
    egui::Slider::new(value, Num::from_f64(lo)..=Num::from_f64(hi))
        .clamping(egui::SliderClamping::Never)
}

impl Default for MeshType {
    fn default() -> Self {
        Self::CutCell
    }
}

/// Seed-motion law for the moving-mesh (ALE) GUI mode. Maps to
/// [`MeshMotionSpec`]; the analytic-swirl variant supplies the concrete
/// [`prescribed_swirl`] function pointer the driver requires.
#[derive(PartialEq, Clone, Copy)]
enum MovingMotionChoice {
    /// Seeds never move — the do-no-harm anchor (a regen from an unchanged seed
    /// set reproduces the mesh byte-for-byte, so the swept fluxes are zero).
    Frozen,
    /// A fixed analytic swirl, independent of the flow (prescribed motion).
    PrescribedSwirl,
    /// Seeds follow the flow (cell velocity + AREPO centroid steering).
    FlowCoupled,
}

impl MovingMotionChoice {
    const ALL: [MovingMotionChoice; 3] = [
        MovingMotionChoice::Frozen,
        MovingMotionChoice::PrescribedSwirl,
        MovingMotionChoice::FlowCoupled,
    ];

    fn label(self) -> &'static str {
        match self {
            MovingMotionChoice::Frozen => "Frozen (do-no-harm)",
            MovingMotionChoice::PrescribedSwirl => "Prescribed swirl",
            MovingMotionChoice::FlowCoupled => "Flow-coupled",
        }
    }

    /// Convert to the driver's motion spec. `regularization` is the FlowCoupled
    /// centroid-steering strength χ (ignored by the other variants).
    fn to_spec(self, regularization: f64) -> MeshMotionSpec {
        match self {
            MovingMotionChoice::Frozen => MeshMotionSpec::Frozen,
            MovingMotionChoice::PrescribedSwirl => MeshMotionSpec::Prescribed(prescribed_swirl),
            MovingMotionChoice::FlowCoupled => MeshMotionSpec::FlowCoupled { regularization },
        }
    }
}

impl Default for MovingMotionChoice {
    fn default() -> Self {
        Self::FlowCoupled
    }
}

/// A small, bounded analytic swirl about the domain mid-line, used by the GUI's
/// "Prescribed swirl" moving-mesh motion. Rotates each interior seed by a
/// time-oscillating angle that decays away from the centre, so the mesh visibly
/// deforms and relaxes without depending on the flow solution — a demonstrator
/// for the prescribed-motion path. The amplitude is kept tiny so the per-step
/// seed displacement stays inside the swept-quad / flip-remap regime.
fn prescribed_swirl(seed0: [f64; 2], t: f64) -> [f64; 2] {
    let (cx, cy) = (1.0, 0.5);
    let dx = seed0[0] - cx;
    let dy = seed0[1] - cy;
    let r2 = dx * dx + dy * dy;
    let ang = 0.15 * t.sin() * (-2.0 * r2).exp();
    let (s, c) = ang.sin_cos();
    [cx + c * dx - s * dy, cy + s * dx + c * dy]
}

#[derive(PartialEq, Clone, Copy)]
enum PlotField {
    Pressure,
    VelocityX,
    VelocityY,
    VelocityMag,
}

struct PlotCache {
    snapshot_seq: u64,
    field: PlotField,
    min: f64,
    max: f64,
    values: Option<Vec<f64>>,
}

#[derive(Default, Clone)]
struct ModelUiCaps {
    plot_stride: u32,
    plot_u_offset: u32,
    plot_p_offset: u32,
    plot_has_u: bool,
    plot_has_p: bool,

    supports_preconditioner: bool,
    model_owns_preconditioner: bool,
    unknowns_per_cell: u32,

    supports_outer_iters: bool,
    supports_dtau: bool,
    supports_low_mach: bool,
    supports_alpha_u: bool,
    supports_alpha_p: bool,
    supports_eos_tuning: bool,
    supports_explicit_rk4: bool,
}

struct SolverInitRequest {
    generation: u64,
    model_id: &'static str,
    selected_geometry: GeometryType,
    mesh_type: MeshType,
    mesh_mode: MeshMode,
    /// STRUCTURED-only: coupled banded-solve preconditioner (block-Jacobi / Schur
    /// / Schur+AMG). Applied by `build_structured_init` via
    /// `StructuredGpuSolver::set_preconditioner`.
    structured_precond: crate::solver::banded_schur::CoupledPrecondKind,
    /// Compute backend: structured honours GPU and CPU Interpreter (the CPU
    /// transpiled paths have no structured kernels — the interpreter runs the
    /// same codegen IR the GPU lowers).
    backend: BackendChoice,
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    // Solver settings travel as a single `RuntimeParams` snapshot (built via
    // `current_runtime_params`) that the shared `SolverDriver` consumes.
    current_fluid: Fluid,
    params: RuntimeParams,
    // Moving-mesh (ALE) request: when `enable_moving_mesh`, build a
    // `MovingMeshDriver` (forcing the CVT mesh + incompressible ALE model)
    // instead of a plain `SolverDriver`.
    enable_moving_mesh: bool,
    moving_motion: MovingMotionChoice,
    moving_regularization: f64,
    // Oscillating-obstacle boundary motion (ChannelObstacle only). When set,
    // `build_moving_init` declares the obstacle loop a cross-stream
    // `BoundaryMotionSpec::Oscillation` and enables the MovingWall BC.
    moving_oscillate_obstacle: bool,
    moving_osc_amplitude: f64,
    moving_osc_frequency: f64,
    // Reconstruct the moving mesh ON DEVICE each step (GPU solver backend
    // only — pre-gated by `make_init_request`, so `build_moving_init` can
    // apply it directly).
    moving_gpu_regen: bool,
    // Periodic Lloyd smoothing cadence for FlowCoupled (0 = off).
    moving_smooth_every_n: usize,
    // Periodic Morton memory-reordering cadence (0 = off).
    moving_reorder_every_n: usize,
    // Flow-adaptive sizing cadence (0 = off): every N steps birth/kill cells
    // toward a target volume derived from |∇U|/|∇p|/strain gradients.
    moving_adapt_every_n: usize,
    // Explicit adaptation target band in cell-size units (fine/coarse limit
    // the indicator maps onto). 0 = auto (the mesh's realized sizing band).
    moving_adapt_min_size: f64,
    moving_adapt_max_size: f64,
    // Adaptivity growth budget: births stop at this multiple of the initial
    // cell count. The viz buffers are allocated from the same factor.
    moving_adapt_budget: f64,
    // Per-indicator threshold multipliers (|∇U|, |∇p|, strain); 1 = auto
    // calibration, 0 = component disabled.
    moving_adapt_thresh_u: f64,
    moving_adapt_thresh_p: f64,
    moving_adapt_thresh_strain: f64,
    // Implicit mesh-motion fixed-point iterations (1 = explicit).
    moving_motion_iters: usize,
    wgpu_device: Option<wgpu::Device>,
    wgpu_queue: Option<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
}

struct SolverInitOutcome {
    mode: SolverMode,
    mesh: Mesh,
    cached_cells: Vec<Vec<[f64; 2]>>,
    actual_min_cell_size: f64,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    model_caps: ModelUiCaps,
    renderer: Option<cfd_renderer::CfdRenderResources>,
    viz_field: Option<VizFieldBuffers>,
    trace_init_events: Vec<tracefmt::TraceInitEvent>,
}

#[derive(Clone)]
struct VizFieldBuffers {
    buffers: [wgpu::Buffer; 3],
    size_bytes: u64,
    range_reducer: cfd_renderer::CfdRangeReducer,
    mailbox: Arc<VizFrameMailbox>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VizSnapshotSource {
    Current,
    AutonomousAccepted,
}

const VIZ_SLOT_STATE_BITS: u32 = 2;
const VIZ_SLOT_STATE_MASK: u64 = (1 << VIZ_SLOT_STATE_BITS) - 1;
const VIZ_MAX_SEQUENCE: u64 = u64::MAX >> VIZ_SLOT_STATE_BITS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
enum VizSlotState {
    Free = 0,
    Writing = 1,
    Ready = 2,
    Display = 3,
}

impl VizSlotState {
    fn from_word(word: u64) -> Self {
        match word & VIZ_SLOT_STATE_MASK {
            0 => Self::Free,
            1 => Self::Writing,
            2 => Self::Ready,
            3 => Self::Display,
            _ => unreachable!("slot state is encoded in two bits"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VizSlotToken {
    index: usize,
    sequence: u64,
}

impl VizSlotToken {
    fn word(self, state: VizSlotState) -> u64 {
        (self.sequence << VIZ_SLOT_STATE_BITS) | state as u64
    }
}

/// Lock-free, frame-demanded mailbox between the solver worker and renderer.
///
/// A slot's atomic word contains both its state and the frame-request sequence,
/// so an old observation can never claim or release a recycled slot (the ABA
/// problem in an index-only front/ready protocol). There is one producer (the
/// solver worker) and one consumer (the UI/render thread):
///
/// `FREE -> WRITING -> READY -> DISPLAY -> FREE`.
///
/// Requests are monotonically numbered. The worker samples the newest request
/// only at an accepted-step boundary, so multiple screen frames during a long
/// step coalesce, while a fast solver performs at most one visualization copy
/// per rendered frame. An unconsumed older READY slot may be discarded after a
/// newer slot is claimed; it was never visible and is therefore safe to free.
struct VizFrameMailbox {
    gpu_consumer_enabled: AtomicBool,
    requested_sequence: AtomicU64,
    serviced_sequence: AtomicU64,
    write_cursor: AtomicUsize,
    slots: [AtomicU64; 3],
}

impl VizFrameMailbox {
    fn new() -> Self {
        let initial_display = VizSlotToken {
            index: 0,
            sequence: 0,
        };
        Self {
            gpu_consumer_enabled: AtomicBool::new(false),
            requested_sequence: AtomicU64::new(0),
            serviced_sequence: AtomicU64::new(0),
            write_cursor: AtomicUsize::new(1),
            slots: [
                AtomicU64::new(initial_display.word(VizSlotState::Display)),
                AtomicU64::new(
                    VizSlotToken {
                        index: 1,
                        sequence: 0,
                    }
                    .word(VizSlotState::Free),
                ),
                AtomicU64::new(
                    VizSlotToken {
                        index: 2,
                        sequence: 0,
                    }
                    .word(VizSlotState::Free),
                ),
            ],
        }
    }

    fn initial_display(&self) -> VizSlotToken {
        VizSlotToken {
            index: 0,
            sequence: 0,
        }
    }

    /// Post a screen-frame request. The producer deliberately observes only
    /// the latest sequence, so this never creates an unbounded work queue.
    fn request_frame(&self) -> u64 {
        self.requested_sequence
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |sequence| {
                (sequence < VIZ_MAX_SEQUENCE).then_some(sequence + 1)
            })
            .expect("visualization frame sequence exhausted")
            + 1
    }

    fn set_gpu_consumer_enabled(&self, enabled: bool) {
        self.gpu_consumer_enabled.store(enabled, Ordering::Release);
    }

    fn gpu_consumer_enabled(&self) -> bool {
        self.gpu_consumer_enabled.load(Ordering::Acquire)
    }

    /// True while a requested snapshot has not reached the renderer yet.
    ///
    /// The producer publishes READY before advancing `serviced_sequence`, so
    /// the acquire loads below cannot miss a completed slot after observing
    /// that sequence. The UI uses this only to schedule another screen tick;
    /// it never participates in slot ownership.
    fn presentation_refresh_pending(&self) -> bool {
        if self.requested_sequence.load(Ordering::Acquire)
            != self.serviced_sequence.load(Ordering::Acquire)
        {
            return true;
        }
        self.slots.iter().any(|slot| {
            matches!(
                VizSlotState::from_word(slot.load(Ordering::Acquire)),
                VizSlotState::Writing | VizSlotState::Ready
            )
        })
    }

    /// Claim a free target for the newest outstanding frame request.
    /// Called by the single solver-worker producer at accepted-step boundaries.
    fn try_begin_write(&self) -> Option<VizSlotToken> {
        let requested = self.requested_sequence.load(Ordering::Acquire);
        if requested == self.serviced_sequence.load(Ordering::Acquire) {
            return None;
        }

        let n = self.slots.len();
        let start = self.write_cursor.load(Ordering::Relaxed) % n;
        for offset in 0..n {
            let index = (start + offset) % n;
            let observed = self.slots[index].load(Ordering::Acquire);
            if VizSlotState::from_word(observed) != VizSlotState::Free {
                continue;
            }
            let token = VizSlotToken {
                index,
                sequence: requested,
            };
            if self.slots[index]
                .compare_exchange(
                    observed,
                    token.word(VizSlotState::Writing),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                self.write_cursor.store((index + 1) % n, Ordering::Relaxed);
                return Some(token);
            }
        }
        None
    }

    /// Publish a copy only after its queue copy/write has been enqueued. A
    /// renderer that acquires READY therefore submits its draw after the copy
    /// on the same wgpu queue (including `Queue::write_buffer` CPU bridges).
    fn finish_write(&self, token: VizSlotToken) -> bool {
        if self.slots[token.index]
            .compare_exchange(
                token.word(VizSlotState::Writing),
                token.word(VizSlotState::Ready),
                Ordering::Release,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        self.serviced_sequence
            .store(token.sequence, Ordering::Release);
        true
    }

    /// Return an unpublished producer slot to the pool after a snapshot copy
    /// failed before READY publication. The outstanding request deliberately
    /// remains unserviced so a later accepted boundary can retry it.
    fn cancel_write(&self, token: VizSlotToken) -> bool {
        self.slots[token.index]
            .compare_exchange(
                token.word(VizSlotState::Writing),
                token.word(VizSlotState::Free),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Claim the newest completed snapshot. The exact generation participates
    /// in the CAS, so a stale READY observation cannot claim a recycled slot.
    fn try_claim_latest_ready(&self) -> Option<VizSlotToken> {
        loop {
            let mut newest: Option<(VizSlotToken, u64)> = None;
            for (index, slot) in self.slots.iter().enumerate() {
                let word = slot.load(Ordering::Acquire);
                if VizSlotState::from_word(word) != VizSlotState::Ready {
                    continue;
                }
                let token = VizSlotToken {
                    index,
                    sequence: word >> VIZ_SLOT_STATE_BITS,
                };
                if newest
                    .as_ref()
                    .map_or(true, |(current, _)| token.sequence > current.sequence)
                {
                    newest = Some((token, word));
                }
            }

            let (token, observed) = newest?;
            if self.slots[token.index]
                .compare_exchange(
                    observed,
                    token.word(VizSlotState::Display),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }

            // Coalesce completed snapshots that the screen never displayed.
            // A concurrently published *newer* snapshot is retained for the
            // next frame; only generations older than this display are freed.
            for (index, slot) in self.slots.iter().enumerate() {
                if index == token.index {
                    continue;
                }
                let word = slot.load(Ordering::Acquire);
                let sequence = word >> VIZ_SLOT_STATE_BITS;
                if VizSlotState::from_word(word) == VizSlotState::Ready && sequence < token.sequence
                {
                    let _ = slot.compare_exchange(
                        word,
                        VizSlotToken { index, sequence }.word(VizSlotState::Free),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                }
            }
            return Some(token);
        }
    }

    /// Release the previous display only after the renderer has rebound to the
    /// replacement. Exact-token comparison makes stale releases harmless.
    fn release_display(&self, token: VizSlotToken) -> bool {
        self.slots[token.index]
            .compare_exchange(
                token.word(VizSlotState::Display),
                token.word(VizSlotState::Free),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    #[cfg(test)]
    fn state(&self, index: usize) -> (u64, VizSlotState) {
        let word = self.slots[index].load(Ordering::Acquire);
        (word >> VIZ_SLOT_STATE_BITS, VizSlotState::from_word(word))
    }
}

/// Activate Direct presentation and post its cold-start snapshot exactly once.
///
/// The false -> true edge is atomic. This matters because a render callback can
/// still be completing while the next UI update changes presentation mode: a
/// late callback may post one harmless coalesced request, but it must never
/// resurrect the autonomous Direct consumer after Plot disabled it.
fn activate_direct_viz(mailbox: &VizFrameMailbox) -> bool {
    if mailbox
        .gpu_consumer_enabled
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return false;
    }
    mailbox.request_frame();
    true
}

fn apply_worker_running_event(
    requested_generation: u64,
    is_running: &mut bool,
    event_generation: u64,
    worker_running: bool,
) -> bool {
    if event_generation != requested_generation {
        return false;
    }
    *is_running = worker_running;
    true
}

struct SolverInitResponse {
    generation: u64,
    result: Result<SolverInitOutcome, String>,
}

// Cached GPU solver stats for UI display (avoids lock contention)
#[derive(Default, Clone)]
struct CachedGpuStats {
    /// Completion-fenced simulation time. GPU submission may be ahead of this.
    sim_time: f64,
    dt: f32,
    linear_solves: u32,
    linear_last: LinearSolverStats,
    outer_residual_u: f32,
    outer_residual_p: f32,
    outer_iterations: u32,
    outer_step_status: Option<OuterStepStatus>,
    positivity_min_rho: Option<f32>,
    positivity_min_p: Option<f32>,
    positivity_rho_undershoots: u32,
    positivity_pressure_undershoots: u32,
    /// Completion-fenced worker throughput, not GPU queue submission latency.
    step_time_ms: f32,
    steps_per_second: f32,
    sim_seconds_per_wall_second: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct CompletedStepSample {
    step_time_ms: f32,
    steps_per_second: f32,
    sim_seconds_per_wall_second: f32,
}

/// Zero-overhead completion timing for the GUI worker.
///
/// GPU submission is asynchronous, so timing the Rust `step()` call compares
/// enqueue latency against synchronous CPU execution. The worker already asks
/// for a blocking state snapshot initially and about every 100 ms. At those
/// existing completion fences we divide the elapsed wall interval by every step
/// completed in it; unsynchronised iterations retain the last honest sample.
/// This neither adds a submission nor serialises the solver once per step.
struct CompletedStepTiming {
    window_start: std::time::Instant,
    window_steps: u64,
    window_sim_time: f64,
    has_completion_anchor: bool,
    last: CompletedStepSample,
}

impl CompletedStepTiming {
    fn new() -> Self {
        Self::new_at(std::time::Instant::now())
    }

    fn new_at(now: std::time::Instant) -> Self {
        Self {
            window_start: now,
            window_steps: 0,
            window_sim_time: 0.0,
            has_completion_anchor: false,
            last: CompletedStepSample::default(),
        }
    }

    fn reset(&mut self) {
        self.reset_at(std::time::Instant::now());
    }

    fn reset_at(&mut self, now: std::time::Instant) {
        self.window_start = now;
        self.window_steps = 0;
        self.window_sim_time = 0.0;
        self.has_completion_anchor = false;
        self.last = CompletedStepSample::default();
    }

    fn record_step(&mut self, dt: f32, completion_fenced: bool) -> CompletedStepSample {
        self.record_step_at(dt, completion_fenced, std::time::Instant::now())
    }

    fn record_step_at(
        &mut self,
        dt: f32,
        completion_fenced: bool,
        now: std::time::Instant,
    ) -> CompletedStepSample {
        self.record_batch_at(1, f64::from(dt), completion_fenced, now)
    }

    fn record_batch_at(
        &mut self,
        steps: u64,
        simulated_seconds: f64,
        completion_fenced: bool,
        now: std::time::Instant,
    ) -> CompletedStepSample {
        self.window_steps = self.window_steps.saturating_add(steps);
        if simulated_seconds.is_finite() && simulated_seconds > 0.0 {
            self.window_sim_time += simulated_seconds;
        }
        if completion_fenced {
            if self.has_completion_anchor {
                let elapsed = now.saturating_duration_since(self.window_start);
                let seconds = elapsed.as_secs_f64();
                if self.window_steps > 0 && seconds > 0.0 {
                    let steps_per_second = self.window_steps as f64 / seconds;
                    self.last.step_time_ms = (1.0e3 / steps_per_second) as f32;
                    self.last.steps_per_second = steps_per_second as f32;
                    self.last.sim_seconds_per_wall_second = (self.window_sim_time / seconds) as f32;
                }
            } else {
                // The first fence establishes a completion-to-completion
                // anchor. Measuring from worker reset/start would omit the
                // previous loop's pacing and post-step work, making the first
                // displayed sample systematically optimistic.
                self.has_completion_anchor = true;
            }
            self.window_start = now;
            self.window_steps = 0;
            self.window_sim_time = 0.0;
        }
        self.last
    }
}

const AUTONOMOUS_MAX_IN_FLIGHT: usize = 2;
const AUTONOMOUS_TARGET_BATCH_TIME: std::time::Duration = std::time::Duration::from_millis(3);
const AUTONOMOUS_MAX_BATCH_STEPS: u32 = 4096;

/// Host-side state of the bounded autonomous GPU producer.
///
/// `Pausing` is deliberately distinct from `Paused`: no more work is submitted,
/// but the worker continues nonblocking device polls until the (at most two)
/// already-submitted batches reach their completion callbacks. Only then may a
/// `Running(false)` acknowledgement be published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutonomousRunPhase {
    Paused,
    Running,
    Pausing,
}

/// Whether a backend can append a new batch as soon as one FIFO credit retires.
/// A three-phase backend remains continuously refillable when every ticket is
/// a whole history-ring cycle: its submitted and reconciled buffer phase are
/// then identical at every healthy batch boundary. A rejected prefix halts the
/// already-queued tail before the scheduler can refill it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutonomousRefillPolicy {
    Continuous,
    ContinuousPhaseCycle3,
    DrainWindow,
}

/// Structured autonomous marching is a presentation optimization, never a
/// distinct numerical policy. If the runtime EOS is outside the certified
/// device audit, both Direct (`presentation_ready=true`) and Plot (`false`)
/// return `None` and therefore use the same ordinary `structured_step` CFL
/// route.
fn structured_autonomous_refill_policy(
    presentation_ready: bool,
    adaptive_dt: bool,
    supports_fixed: bool,
    supports_adaptive: bool,
) -> Option<AutonomousRefillPolicy> {
    (presentation_ready
        && if adaptive_dt {
            supports_adaptive
        } else {
            supports_fixed
        })
    .then_some(AutonomousRefillPolicy::Continuous)
}

impl AutonomousRefillPolicy {
    fn is_continuous(self) -> bool {
        matches!(self, Self::Continuous | Self::ContinuousPhaseCycle3)
    }

    fn normalize_steps(self, steps: u32) -> u32 {
        if self != Self::ContinuousPhaseCycle3 {
            return steps.clamp(1, AUTONOMOUS_MAX_BATCH_STEPS);
        }
        // 4095 is the largest whole three-buffer cycle inside the global cap.
        steps
            .max(1)
            .div_ceil(3)
            .saturating_mul(3)
            .min(AUTONOMOUS_MAX_BATCH_STEPS - AUTONOMOUS_MAX_BATCH_STEPS % 3)
            .max(3)
    }
}

/// A slot reserved before backend submission. The exact generation and serial
/// travel through the completion callback, so replacing a solver or changing
/// parameters cannot turn an old callback into progress for a new run.
#[derive(Debug, Clone, Copy)]
struct AutonomousBatchTicket {
    backend_epoch: u64,
    generation: u64,
    serial: u64,
    requested_steps: u32,
    submitted_at: std::time::Instant,
}

/// Backend-independent, completion-fenced result of one autonomous batch.
///
/// Backends populate this from their tiny GPU control/status block, never by
/// copying the cell state. `accepted_steps` may be smaller than the ticket's
/// requested count when a health kernel freezes/rolls back an invalid step.
#[derive(Debug, Clone, Copy)]
struct AutonomousBatchCompletion {
    ticket: AutonomousBatchTicket,
    accepted_steps: u32,
    completed_time: f64,
    last_dt: f32,
    next_dt: f32,
    halted: bool,
    invalid_count: u32,
    completed_at: std::time::Instant,
    backend_status: Option<AutonomousBackendStatus>,
}

struct AutonomousCallbackMessage {
    ticket: AutonomousBatchTicket,
    result: Result<AutonomousBackendStatus, String>,
    completed_at: std::time::Instant,
}

#[derive(Debug, Clone, Copy)]
enum AutonomousBackendStatus {
    Structured(StructuredAutonomousStatus),
    Unstructured(crate::solver::gpu::unified_solver::AdaptiveExplicitGpuBatchCompletion),
}

impl AutonomousBatchCompletion {
    fn is_healthy(self) -> bool {
        !self.halted
            && self.invalid_count == 0
            && self.accepted_steps == self.ticket.requested_steps
    }
}

/// Duration feedback for batch sizing. The target is the middle of the 2--4 ms
/// latency envelope. An EWMA filters callback/poll jitter and a 2x slew limit
/// prevents one delayed UI tick from making the next command buffer enormous.
#[derive(Debug, Clone)]
struct AutonomousBatchController {
    steps: u32,
    seconds_per_step_ewma: Option<f64>,
}

impl Default for AutonomousBatchController {
    fn default() -> Self {
        Self {
            steps: 1,
            seconds_per_step_ewma: None,
        }
    }
}

impl AutonomousBatchController {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn steps(&self) -> u32 {
        self.steps
    }

    fn observe(&mut self, completed_steps: u32, elapsed: std::time::Duration) {
        if completed_steps == 0 || elapsed.is_zero() {
            return;
        }
        let sample = elapsed.as_secs_f64() / f64::from(completed_steps);
        if !sample.is_finite() || sample <= 0.0 {
            return;
        }
        let filtered = self
            .seconds_per_step_ewma
            .map_or(sample, |old| old.mul_add(0.75, sample * 0.25));
        self.seconds_per_step_ewma = Some(filtered);

        let desired = (AUTONOMOUS_TARGET_BATCH_TIME.as_secs_f64() / filtered)
            .round()
            .clamp(1.0, f64::from(AUTONOMOUS_MAX_BATCH_STEPS)) as u32;
        let lower = (self.steps / 2).max(1);
        let upper = self.steps.saturating_mul(2).min(AUTONOMOUS_MAX_BATCH_STEPS);
        self.steps = desired.clamp(lower, upper);
    }
}

#[derive(Debug)]
struct FixedCompletionBatch {
    slots: [Option<AutonomousBatchCompletion>; AUTONOMOUS_MAX_IN_FLIGHT],
    len: usize,
}

impl Default for FixedCompletionBatch {
    fn default() -> Self {
        Self {
            slots: [None; AUTONOMOUS_MAX_IN_FLIGHT],
            len: 0,
        }
    }
}

impl FixedCompletionBatch {
    fn push(&mut self, completion: AutonomousBatchCompletion) {
        assert!(
            self.len < AUTONOMOUS_MAX_IN_FLIGHT,
            "autonomous completion drain exceeded the two-credit bound"
        );
        self.slots[self.len] = Some(completion);
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn get(&self, index: usize) -> Option<&AutonomousBatchCompletion> {
        (index < self.len).then(|| self.slots[index].as_ref()).flatten()
    }

    fn iter(&self) -> impl Iterator<Item = &AutonomousBatchCompletion> {
        self.slots[..self.len].iter().flatten()
    }
}

#[derive(Debug, Default)]
struct AutonomousDrain {
    /// Every physically retired FIFO completion, including stale generations;
    /// same-backend stale entries still need host phase/time reconciliation.
    retired: FixedCompletionBatch,
    completed: FixedCompletionBatch,
    became_paused: bool,
}

/// Two-credit FIFO scheduler shared by structured and unstructured GPU paths.
/// It contains no solver-specific assumptions: a backend may use it only after
/// declaring that every encoded accepted state has GPU health checking and
/// rollback/freeze semantics.
struct AutonomousBatchScheduler {
    backend_epoch: u64,
    generation: u64,
    next_serial: u64,
    phase: AutonomousRunPhase,
    refill_policy: AutonomousRefillPolicy,
    window_slots_remaining: usize,
    controller: AutonomousBatchController,
    in_flight: VecDeque<AutonomousBatchTicket>,
    pending_completions: Vec<AutonomousBatchCompletion>,
    last_completion_at: Option<std::time::Instant>,
}

impl Default for AutonomousBatchScheduler {
    fn default() -> Self {
        Self {
            backend_epoch: 0,
            generation: 0,
            next_serial: 0,
            phase: AutonomousRunPhase::Paused,
            refill_policy: AutonomousRefillPolicy::Continuous,
            window_slots_remaining: AUTONOMOUS_MAX_IN_FLIGHT,
            controller: AutonomousBatchController::default(),
            in_flight: VecDeque::with_capacity(AUTONOMOUS_MAX_IN_FLIGHT),
            pending_completions: Vec::with_capacity(AUTONOMOUS_MAX_IN_FLIGHT),
            last_completion_at: None,
        }
    }
}

impl AutonomousBatchScheduler {
    fn begin_run(&mut self, refill_policy: AutonomousRefillPolicy) {
        self.generation = self.generation.wrapping_add(1);
        self.phase = AutonomousRunPhase::Running;
        self.refill_policy = refill_policy;
        self.window_slots_remaining = if self.in_flight.is_empty() {
            AUTONOMOUS_MAX_IN_FLIGHT
        } else {
            0
        };
        self.controller.reset();
        self.last_completion_at = None;
    }

    fn replace_backend(&mut self) {
        self.backend_epoch = self.backend_epoch.wrapping_add(1);
        self.invalidate(false);
    }

    /// Start a new callback generation without forgetting physical queue
    /// occupancy. Stale tickets retain their credits until their callbacks are
    /// observed, which keeps the global in-flight bound true across updates.
    fn invalidate(&mut self, continue_running: bool) {
        self.generation = self.generation.wrapping_add(1);
        self.phase = if continue_running {
            AutonomousRunPhase::Running
        } else if self.in_flight.is_empty() {
            AutonomousRunPhase::Paused
        } else {
            AutonomousRunPhase::Pausing
        };
        self.window_slots_remaining = if self.in_flight.is_empty() {
            AUTONOMOUS_MAX_IN_FLIGHT
        } else {
            0
        };
        self.controller.reset();
        self.last_completion_at = None;
    }

    /// Returns `true` when already fully drained.
    fn request_pause(&mut self) -> bool {
        self.phase = if self.in_flight.is_empty() {
            AutonomousRunPhase::Paused
        } else {
            AutonomousRunPhase::Pausing
        };
        self.phase == AutonomousRunPhase::Paused
    }

    fn abort(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.phase = AutonomousRunPhase::Paused;
        self.in_flight.clear();
        self.pending_completions.clear();
        self.window_slots_remaining = AUTONOMOUS_MAX_IN_FLIGHT;
        self.last_completion_at = None;
    }

    fn is_active(&self) -> bool {
        self.phase != AutonomousRunPhase::Paused || !self.in_flight.is_empty()
    }

    fn can_submit(&self) -> bool {
        self.phase == AutonomousRunPhase::Running
            && self.in_flight.len() < AUTONOMOUS_MAX_IN_FLIGHT
            && (self.refill_policy.is_continuous() || self.window_slots_remaining > 0)
    }

    fn reserve(&mut self, now: std::time::Instant) -> Option<AutonomousBatchTicket> {
        if !self.can_submit() {
            return None;
        }
        let ticket = AutonomousBatchTicket {
            backend_epoch: self.backend_epoch,
            generation: self.generation,
            serial: self.next_serial,
            requested_steps: self.refill_policy.normalize_steps(self.controller.steps()),
            submitted_at: now,
        };
        self.next_serial = self.next_serial.wrapping_add(1);
        self.in_flight.push_back(ticket);
        if self.refill_policy == AutonomousRefillPolicy::DrainWindow {
            self.window_slots_remaining = self.window_slots_remaining.saturating_sub(1);
        }
        Some(ticket)
    }

    fn cancel_reservation(&mut self, ticket: AutonomousBatchTicket) {
        if let Some(index) = self.in_flight.iter().position(|queued| {
            queued.generation == ticket.generation && queued.serial == ticket.serial
        }) {
            self.in_flight.remove(index);
            if self.refill_policy == AutonomousRefillPolicy::DrainWindow
                && ticket.generation == self.generation
            {
                self.window_slots_remaining =
                    (self.window_slots_remaining + 1).min(AUTONOMOUS_MAX_IN_FLIGHT);
            }
        }
        if self.phase == AutonomousRunPhase::Pausing && self.in_flight.is_empty() {
            self.phase = AutonomousRunPhase::Paused;
        }
    }

    /// Release a batch whose completion status could not be decoded.  The run
    /// generation is invalidated by the caller first, so any already-arrived
    /// tail statuses are tombstones: retire them in FIFO order without ever
    /// exposing their progress. This prevents an out-of-order tail callback
    /// from being stranded forever behind the failed front ticket.
    fn discard_failed_completion(&mut self, ticket: AutonomousBatchTicket) {
        self.cancel_reservation(ticket);
        loop {
            let Some(front) = self.in_flight.front().copied() else {
                break;
            };
            let Some(index) = self.pending_completions.iter().position(|pending| {
                pending.ticket.backend_epoch == front.backend_epoch
                    && pending.ticket.generation == front.generation
                    && pending.ticket.serial == front.serial
            }) else {
                break;
            };
            self.pending_completions.swap_remove(index);
            self.in_flight.pop_front();
        }
        if self.phase == AutonomousRunPhase::Pausing && self.in_flight.is_empty() {
            self.phase = AutonomousRunPhase::Paused;
        }
    }

    /// A retired device may be lost independently of the replacement backend.
    /// Its callbacks can no longer arrive, so release only stale-epoch credits;
    /// current-backend work and poison state are untouched.
    fn discard_retired_epochs(&mut self) {
        let epoch = self.backend_epoch;
        self.in_flight
            .retain(|ticket| ticket.backend_epoch == epoch);
        self.pending_completions
            .retain(|pending| pending.ticket.backend_epoch == epoch);
        if self.phase == AutonomousRunPhase::Pausing && self.in_flight.is_empty() {
            self.phase = AutonomousRunPhase::Paused;
        }
    }

    /// Accept callback delivery in any host order, then retire it strictly in
    /// GPU FIFO order. Queue callbacks are normally ordered; retaining this
    /// small reorder buffer makes duplicate/adversarial callbacks harmless and
    /// keeps pause acknowledgement deterministic.
    fn complete(&mut self, completion: AutonomousBatchCompletion) -> AutonomousDrain {
        let known = self.in_flight.iter().any(|queued| {
            queued.generation == completion.ticket.generation
                && queued.serial == completion.ticket.serial
        });
        let duplicate = self.pending_completions.iter().any(|pending| {
            pending.ticket.generation == completion.ticket.generation
                && pending.ticket.serial == completion.ticket.serial
        });
        if !known || duplicate {
            return AutonomousDrain::default();
        }
        self.pending_completions.push(completion);

        let mut drain = AutonomousDrain::default();
        loop {
            let Some(front) = self.in_flight.front().copied() else {
                break;
            };
            let Some(index) = self.pending_completions.iter().position(|pending| {
                pending.ticket.generation == front.generation
                    && pending.ticket.serial == front.serial
            }) else {
                break;
            };
            let completed = self.pending_completions.swap_remove(index);
            self.in_flight.pop_front();
            drain.retired.push(completed);

            // A callback from an invalidated generation releases its physical
            // credit but cannot tune or publish progress for the current run.
            if completed.ticket.generation != self.generation {
                continue;
            }

            let interval_start = self
                .last_completion_at
                .unwrap_or(completed.ticket.submitted_at);
            let elapsed = completed
                .completed_at
                .saturating_duration_since(interval_start);
            self.last_completion_at = Some(completed.completed_at);
            self.controller.observe(completed.accepted_steps, elapsed);
            if !completed.is_healthy() {
                self.phase = AutonomousRunPhase::Pausing;
            }
            drain.completed.push(completed);
        }

        if self.in_flight.is_empty()
            && self.phase == AutonomousRunPhase::Running
            && self.refill_policy == AutonomousRefillPolicy::DrainWindow
        {
            self.window_slots_remaining = AUTONOMOUS_MAX_IN_FLIGHT;
        }

        if self.phase == AutonomousRunPhase::Pausing && self.in_flight.is_empty() {
            self.phase = AutonomousRunPhase::Paused;
            drain.became_paused = true;
        }
        drain
    }
}

/// What the solver worker is driving. `Static` wraps a [`SolverDriver`];
/// `MovingMesh` is the ALE path — a [`MovingMeshDriver`] that re-generates the
/// CVT-Voronoi mesh each step. Both delegate field/stats/params access to the
/// wrapped [`SolverDriver`] via [`SolverMode::driver`]/[`SolverMode::driver_mut`],
/// so the worker's telemetry, trace, and `apply_params` plumbing is shared and
/// the step dispatch is the only fork.
enum SolverMode {
    Static(SolverDriver),
    MovingMesh(MovingMeshDriver),
    /// Dense-Cartesian `TopologyMode::Structured2D` path — a standalone
    /// [`StructuredGpuSolver`] (no `SolverDriver`/`Mesh`). The worker drives it
    /// through the mode-level abstraction methods below rather than `driver()`.
    Structured(StructuredGpuSolver),
    /// Structured on the CPU interpreter ([`StructuredModelSolver`]) — the compute
    /// runs on the host, and each step uploads the packed state to a GPU buffer so
    /// the same on-device renderer viz is reused. Same abstraction surface as
    /// [`Self::Structured`].
    StructuredCpu(StructuredCpuBridge),
}

/// Bridges the CPU structured solver ([`StructuredModelSolver`]) to the GPU-direct
/// renderer: it holds a GPU `state` buffer seeded from the CPU state so the viz
/// init copy works, and `copy_state_to_buffer` writes the current CPU packed
/// state straight into the target viz buffer (`queue.write_buffer`). The compute
/// is 100% host-side; only the viz feed touches the GPU (as it must — the GUI is
/// a wgpu app).
struct StructuredCpuBridge {
    solver: StructuredModelSolver,
    /// GPU copy of the packed state (COPY_SRC), for the one-time viz-buffer init.
    state_buf: wgpu::Buffer,
    size_bytes: u64,
    queue: wgpu::Queue,
}

impl StructuredCpuBridge {
    fn new(solver: StructuredModelSolver, device: &wgpu::Device, queue: wgpu::Queue) -> Self {
        use wgpu::util::DeviceExt;
        let packed = solver.packed_state_f32();
        let size_bytes = (packed.len() * 4) as u64;
        let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("structured_cpu_state"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            solver,
            state_buf,
            size_bytes,
            queue,
        }
    }

    fn step(&mut self) {
        self.solver.step();
    }

    fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buf
    }

    fn copy_state_to_buffer(&self, dst: &wgpu::Buffer) {
        // Write the CURRENT host state straight into the viz buffer.
        let packed = self.solver.packed_state_f32();
        self.queue
            .write_buffer(dst, 0, bytemuck::cast_slice(&packed));
    }

    fn upload_state_to_buffer(&self, dst: &wgpu::Buffer) -> Result<(), String> {
        if dst.size() < self.size_bytes {
            return Err(format!(
                "structured CPU visualization destination is {} bytes, needs {}",
                dst.size(),
                self.size_bytes
            ));
        }
        if !dst.usage().contains(wgpu::BufferUsages::COPY_DST) {
            return Err(
                "structured CPU visualization destination lacks COPY_DST usage".to_string(),
            );
        }
        self.copy_state_to_buffer(dst);
        Ok(())
    }
}

impl SolverMode {
    /// The wrapped solver driver (Static/Moving only). Panics for `Structured`,
    /// which has no `SolverDriver` — structured-reachable worker code must use the
    /// abstraction methods below, never `driver()`.
    fn driver(&self) -> &SolverDriver {
        match self {
            SolverMode::Static(d) => d,
            SolverMode::MovingMesh(m) => m.driver(),
            SolverMode::Structured(_) | SolverMode::StructuredCpu(_) => {
                panic!("driver() called on a Structured SolverMode")
            }
        }
    }

    fn install_viz_frame_targets(
        &mut self,
        destinations: &[wgpu::Buffer; 3],
    ) -> Result<(), String> {
        match self {
            SolverMode::Static(driver) => {
                driver.install_autonomous_frame_copy_targets(destinations)
            }
            _ => Ok(()),
        }
    }

    fn encode_viz_snapshot_to_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_slot: usize,
        dst: &wgpu::Buffer,
        source: VizSnapshotSource,
    ) -> Result<(), String> {
        if source == VizSnapshotSource::AutonomousAccepted {
            if let SolverMode::Static(driver) = self {
                return driver.encode_autonomous_accepted_state_copy(encoder, target_slot);
            }
        }
        match self {
            SolverMode::Structured(solver) => {
                solver.encode_state_copy_to_buffer(encoder, dst)
            }
            SolverMode::StructuredCpu(bridge) => bridge.upload_state_to_buffer(dst),
            _ => self
                .driver()
                .solver()
                .encode_or_upload_state_to_buffer(encoder, dst),
        }
    }

    /// Mutable access to the wrapped solver driver (Static/Moving only).
    fn driver_mut(&mut self) -> &mut SolverDriver {
        match self {
            SolverMode::Static(d) => d,
            SolverMode::MovingMesh(m) => m.driver_mut(),
            SolverMode::Structured(_) | SolverMode::StructuredCpu(_) => {
                panic!("driver_mut() called on a Structured SolverMode")
            }
        }
    }

    /// The wrapped `UnifiedSolver`, if this mode has one (Static/Moving). `None`
    /// for `Structured`, whose telemetry/residual surface does not exist.
    fn unified_solver(&self) -> Option<&UnifiedSolver> {
        match self {
            SolverMode::Static(d) => Some(d.solver()),
            SolverMode::MovingMesh(m) => Some(m.driver().solver()),
            SolverMode::Structured(_) | SolverMode::StructuredCpu(_) => None,
        }
    }

    fn unified_solver_mut(&mut self) -> Option<&mut UnifiedSolver> {
        match self {
            SolverMode::Static(d) => Some(d.solver_mut()),
            SolverMode::MovingMesh(m) => Some(m.driver_mut().solver_mut()),
            SolverMode::Structured(_) | SolverMode::StructuredCpu(_) => None,
        }
    }

    /// The packed cell-major `state` GPU buffer feeding the renderer viz color.
    fn state_buffer(&self) -> &wgpu::Buffer {
        match self {
            SolverMode::Structured(s) => s.state_buffer(),
            SolverMode::StructuredCpu(s) => s.state_buffer(),
            _ => self.driver().solver().state_buffer(),
        }
    }

    fn state_size_bytes(&self) -> u64 {
        match self {
            SolverMode::Structured(s) => s.state_size_bytes(),
            SolverMode::StructuredCpu(s) => s.size_bytes,
            _ => self.driver().solver().state_size_bytes(),
        }
    }

    /// Same-device copy of packed `state` into a renderer viz buffer.
    fn copy_state_to_buffer(&self, dst: &wgpu::Buffer) {
        match self {
            SolverMode::Structured(s) => s.copy_state_to_buffer(dst),
            SolverMode::StructuredCpu(s) => s.copy_state_to_buffer(dst),
            _ => self.driver().solver().copy_state_to_buffer(dst),
        }
    }

    /// The state-layout-derived UI ports (velocity/pressure offsets, stride).
    fn ui_ports(&self) -> UiPortSet {
        match self {
            SolverMode::Structured(s) => UiPortSet::from_layout(s.state_layout()),
            SolverMode::StructuredCpu(s) => UiPortSet::from_layout(s.solver.state_layout()),
            _ => self.driver().solver().ui_ports(),
        }
    }

    fn model_id_str(&self) -> &'static str {
        match self {
            SolverMode::Structured(s) => s.model_id(),
            SolverMode::StructuredCpu(s) => s.solver.model_id(),
            _ => self.driver().solver().model().id,
        }
    }

    fn sim_time(&self) -> f32 {
        self.sim_time_f64() as f32
    }

    fn sim_time_f64(&self) -> f64 {
        match self {
            SolverMode::Structured(s) => s.time(),
            SolverMode::StructuredCpu(s) => s.solver.time(),
            _ => f64::from(self.driver().solver().time()),
        }
    }

    fn sim_dt(&self) -> f32 {
        match self {
            SolverMode::Structured(s) => s.dt() as f32,
            SolverMode::StructuredCpu(s) => s.solver.dt() as f32,
            _ => self.driver().solver().dt(),
        }
    }

    /// Apply live runtime params. Structured supports fluid + outer Picard knobs
    /// and fixed-dt when adaptive is off; the unstructured arms route to the full
    /// `SolverDriver::apply_params`.
    fn apply_params_any(&mut self, params: &RuntimeParams) {
        match self {
            SolverMode::Structured(s) => {
                // Match SolverDriver: only pin the slider seed when adaptive is off;
                // otherwise the per-step CFL controller owns dt.
                if !params.adaptive_dt {
                    s.set_dt(params.requested_dt as f64);
                }
                s.set_fluid(params.density as f64, params.viscosity as f64);
                s.set_outer_iters(params.outer_iters.max(1) as usize);
                s.set_outer_auto_converge(params.outer_auto_converge);
                s.set_alpha_u(params.alpha_u);
                s.set_alpha_p(params.alpha_p);
                s.set_time_scheme(params.time_scheme);
                if s.model_id() == "allmach_thermal_structured" {
                    let grid = s.grid();
                    refresh_structured_allmach_runtime_fields(s, params, grid.dx, grid.dy);
                    setup_structured_bcs(
                        s,
                        "allmach_thermal_structured",
                        params.inlet_velocity as f64,
                        4,
                        params,
                    );
                }
                if s.model_id() == "compressible_structured" {
                    apply_structured_compressible_runtime(s, params);
                    let stride = s.state_layout().stride() as usize;
                    setup_structured_bcs(
                        s,
                        "compressible_structured",
                        params.inlet_velocity as f64,
                        stride,
                        params,
                    );
                }
            }
            SolverMode::StructuredCpu(s) => {
                if !params.adaptive_dt {
                    s.solver.set_dt(params.requested_dt as f64);
                }
                s.solver
                    .set_fluid(params.density as f64, params.viscosity as f64);
                s.solver.set_outer_iters(params.outer_iters.max(1) as usize);
                s.solver.set_outer_auto_converge(params.outer_auto_converge);
                s.solver.set_alpha_u(params.alpha_u);
                s.solver.set_alpha_p(params.alpha_p);
                s.solver.set_time_scheme(params.time_scheme);
                if s.solver.model_id() == "allmach_thermal_structured" {
                    let grid = s.solver.grid();
                    refresh_structured_allmach_runtime_fields(
                        &mut s.solver,
                        params,
                        grid.dx,
                        grid.dy,
                    );
                    setup_structured_bcs(
                        &mut s.solver,
                        "allmach_thermal_structured",
                        params.inlet_velocity as f64,
                        4,
                        params,
                    );
                }
                if s.solver.model_id() == "compressible_structured" {
                    apply_structured_compressible_runtime(&mut s.solver, params);
                    let stride = s.solver.state_layout().stride() as usize;
                    setup_structured_bcs(
                        &mut s.solver,
                        "compressible_structured",
                        params.inlet_velocity as f64,
                        stride,
                        params,
                    );
                }
            }
            _ => self.driver_mut().apply_params(params),
        }
    }
}

impl VizFieldBuffers {
    /// Capture one sequence-tagged visualization snapshot. Queue ordering is
    /// field copy -> range reduction/readback copy -> READY publication.
    fn capture_snapshot(
        &self,
        mode: &SolverMode,
        write: VizSlotToken,
        source: VizSnapshotSource,
    ) -> Result<(), String> {
        let field_buffer = &self.buffers[write.index];
        let bytes_per_cell = u64::from(self.range_reducer.stride()) * 4;
        let copied_bytes = mode.state_size_bytes().min(field_buffer.size());
        let cell_count = if bytes_per_cell == 0 {
            0
        } else {
            (copied_bytes / bytes_per_cell) as usize
        };
        let submitted = self.range_reducer.submit_snapshot(
            write.index,
            write.sequence,
            cell_count,
            |encoder| {
                mode.encode_viz_snapshot_to_buffer(
                    encoder,
                    write.index,
                    field_buffer,
                    source,
                )
            },
        );
        if let Err(error) = submitted {
            let cancelled = self.mailbox.cancel_write(write);
            debug_assert!(
                cancelled,
                "failed visualization copy lost exclusive WRITING ownership"
            );
            return Err(error);
        }
        if self.mailbox.finish_write(write) {
            Ok(())
        } else {
            let _ = self.mailbox.cancel_write(write);
            Err("visualization producer lost exclusive WRITING ownership".to_string())
        }
    }
}

fn service_pending_viz_snapshot(
    mode: &SolverMode,
    viz: &VizFieldBuffers,
    source: VizSnapshotSource,
) -> Result<bool, String> {
    let Some(write) = viz.mailbox.try_begin_write() else {
        return Ok(false);
    };
    viz.capture_snapshot(mode, write, source)?;
    Ok(true)
}

enum SolverWorkerCommand {
    SetSolver {
        mode: SolverMode,
        viz_field: Option<VizFieldBuffers>,
    },
    ClearSolver,
    SetRunning {
        running: bool,
        generation: u64,
    },
    UpdateParams(RuntimeParams),
    StartTrace {
        path: String,
        header: tracefmt::TraceHeader,
    },
    AppendTraceInit {
        events: Vec<tracefmt::TraceInitEvent>,
    },
    StopTrace,
    Shutdown,
}

enum SolverWorkerEvent {
    Stats {
        stats: CachedGpuStats,
    },
    Snapshot {
        u: Vec<(f64, f64)>,
        p: Vec<f64>,
        stats: CachedGpuStats,
    },
    /// The moving-mesh (ALE) path re-generated the mesh this step: carries the
    /// re-tessellation input (per-cell polygons, seed `i` == cell `i`) and the
    /// per-step moving-mesh telemetry. Emitted every moving step; the UI thread
    /// coalesces (only the latest matters) and re-tessellates at the frame
    /// boundary. Empty on the static path (never sent).
    MeshRefreshed {
        cached_cells: Vec<Vec<[f64; 2]>>,
        stats: MovingMeshStats,
    },
    Message(String),
    Error(String),
    Running {
        running: bool,
        generation: u64,
    },
}

struct SolverTraceSession {
    writer: tracefmt::TraceWriter,
    profiling_enabled: bool,
}

struct SolverWorkerHandle {
    tx: mpsc::Sender<SolverWorkerCommand>,
    rx: mpsc::Receiver<SolverWorkerEvent>,
    thread: Option<thread::JoinHandle<()>>,
}

impl SolverWorkerHandle {
    fn spawn() -> Self {
        let (tx, cmd_rx) = mpsc::channel::<SolverWorkerCommand>();
        let (evt_tx, rx) = mpsc::channel::<SolverWorkerEvent>();
        let thread = thread::spawn(move || solver_worker_main(cmd_rx, evt_tx));
        Self {
            tx,
            rx,
            thread: Some(thread),
        }
    }

    fn send(&self, cmd: SolverWorkerCommand) {
        let _ = self.tx.send(cmd);
    }
}

impl Drop for SolverWorkerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(SolverWorkerCommand::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Observations from a headless moving-mesh worker smoke run (see
/// [`moving_mesh_worker_smoke`]).
#[doc(hidden)]
#[derive(Default, Debug)]
pub struct MovingWorkerSmoke {
    /// Number of `MeshRefreshed` events the worker emitted.
    pub mesh_refresh_events: usize,
    /// Min / max cell count across all emitted refreshes (should be equal —
    /// fixed seed). `None` if no refresh was seen.
    pub min_cells: Option<usize>,
    pub max_cells: Option<usize>,
    /// A refresh carried an empty / degenerate (< 3 vertex) polygon set.
    pub saw_empty_cells: bool,
    /// A refresh carried a non-finite `MovingMeshStats` field.
    pub saw_nonfinite_stats: bool,
    /// Max post-closure per-cell SCL defect over all refreshes — the GCL health
    /// of the moving-mesh run (should stay at f32-roundoff scale; a physical
    /// moving mesh conserves volume). 0 if no refresh was seen.
    pub max_scl_defect: f64,
    /// Max mesh skew over all refreshes — the quality watchdog.
    pub max_skew: f64,
    /// The emitted per-refresh cell polygons (exactly the `cached_cells` the UI
    /// thread re-tessellates + uploads). Collected so a headless test can replay
    /// the live render loop's `build_mesh_vertices` → `update_mesh` capacity path
    /// on the REAL moving meshes. Empty unless `collect_meshes` was requested.
    pub meshes: Vec<Vec<Vec<[f64; 2]>>>,
    /// Refreshes whose step ran fully on device ([`RegenBackend::GpuOnDevice`]).
    pub gpu_ondevice_refreshes: usize,
    /// Refreshes whose step fell back to the CPU path
    /// ([`RegenBackend::GpuFallback`]).
    pub gpu_fallback_refreshes: usize,
    /// The worker reported an error (step failure / divergence).
    pub error: Option<String>,
}

/// Are the numeric [`MovingMeshStats`] fields all finite? (The discrete counts
/// are always finite; this guards the float telemetry the gates watch.)
fn moving_stats_finite(s: &MovingMeshStats) -> bool {
    s.plan_ms.is_finite()
        && s.regen_ms.is_finite()
        && s.swept_ms.is_finite()
        && s.refresh_ms.is_finite()
        && s.scl_defect.is_finite()
        && s.identity_err.is_finite()
        && s.max_skew.is_finite()
        && s.flip_defect.is_finite()
        && s.dt.is_finite()
}

/// Headless test hook (no window): drive the *real* private solver worker through
/// the moving-mesh message path — `SetSolver { MovingMesh }`,
/// `SetRunning { running: true, .. }`,
/// collect `MeshRefreshed` events, then shut it down. Returns per-run
/// observations so `tests/moving_mesh_gui_test.rs` can assert the actual channel
/// plumbing without a display. `pub` only because the worker + command / event
/// enums are otherwise private to this module.
///
/// Stops at whichever comes first: `target_refreshes` refreshes collected (0 =
/// no count target, run the whole budget) or `run_ms` elapsed. `collect_meshes`
/// retains each emitted `cached_cells` polygon set on `smoke.meshes` so a test
/// can replay the renderer's re-tessellation / capacity-growth path on the REAL
/// moving meshes (the closest headless proxy for the live render loop).
#[doc(hidden)]
pub fn moving_mesh_worker_smoke(
    moving: MovingMeshDriver,
    run_ms: u64,
    target_refreshes: usize,
    collect_meshes: bool,
) -> MovingWorkerSmoke {
    use std::time::{Duration, Instant};

    let handle = SolverWorkerHandle::spawn();
    // Mirror the GUI's init sequence (`init_solver` → `sync_worker_params`):
    // the worker's SetSolver arm re-applies ITS params snapshot over the
    // driver (phase-2 application), so the driver's own runtime params must
    // be sent first or the worker silently overwrites the test's carefully
    // configured solver (a default-params overwrite destabilized the
    // thermal-ALE smoke while coincidentally matching the incompressible
    // ones).
    let driver_params = *moving.driver().params();
    handle.send(SolverWorkerCommand::UpdateParams(driver_params));
    handle.send(SolverWorkerCommand::SetSolver {
        mode: SolverMode::MovingMesh(moving),
        viz_field: None,
    });
    handle.send(SolverWorkerCommand::SetRunning {
        running: true,
        generation: 1,
    });

    let mut smoke = MovingWorkerSmoke::default();
    let deadline = Instant::now() + Duration::from_millis(run_ms);
    'outer: while Instant::now() < deadline {
        while let Ok(evt) = handle.rx.try_recv() {
            match evt {
                SolverWorkerEvent::MeshRefreshed {
                    cached_cells,
                    stats,
                } => {
                    smoke.mesh_refresh_events += 1;
                    if cached_cells.is_empty() || cached_cells.iter().any(|c| c.len() < 3) {
                        smoke.saw_empty_cells = true;
                    }
                    let n = cached_cells.len();
                    smoke.min_cells = Some(smoke.min_cells.map_or(n, |m| m.min(n)));
                    smoke.max_cells = Some(smoke.max_cells.map_or(n, |m| m.max(n)));
                    if !moving_stats_finite(&stats) {
                        smoke.saw_nonfinite_stats = true;
                    }
                    smoke.max_scl_defect = smoke.max_scl_defect.max(stats.scl_defect);
                    smoke.max_skew = smoke.max_skew.max(stats.max_skew);
                    match stats.regen_backend {
                        RegenBackend::GpuOnDevice => smoke.gpu_ondevice_refreshes += 1,
                        RegenBackend::GpuFallback(_) => smoke.gpu_fallback_refreshes += 1,
                        _ => {}
                    }
                    if collect_meshes {
                        smoke.meshes.push(cached_cells);
                    }
                    if target_refreshes != 0 && smoke.mesh_refresh_events >= target_refreshes {
                        break 'outer;
                    }
                }
                SolverWorkerEvent::Error(e) => smoke.error = Some(e),
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    // `handle` drops here: sends Shutdown + joins the worker thread.
    smoke
}

/// Compute-backend dropdown choice (env-driven; applied on Initialize / Reset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Gpu,
    CpuInterpreter,
    CpuTranspiled,
    /// Transpiled kernels + the SIMD path for the linear-solve reductions.
    CpuTranspiledSimd,
}

impl BackendChoice {
    const ALL: [BackendChoice; 4] = [
        BackendChoice::Gpu,
        BackendChoice::CpuInterpreter,
        BackendChoice::CpuTranspiled,
        BackendChoice::CpuTranspiledSimd,
    ];

    fn label(self) -> &'static str {
        match self {
            BackendChoice::Gpu => "GPU",
            BackendChoice::CpuInterpreter => "CPU Interpreter",
            BackendChoice::CpuTranspiled => "CPU Transpiled",
            BackendChoice::CpuTranspiledSimd => "CPU Transpiled (SIMD linear)",
        }
    }

    fn is_cpu(self) -> bool {
        !matches!(self, BackendChoice::Gpu)
    }
}

pub struct CFDApp {
    solver_worker: SolverWorkerHandle,
    pending_init_request: Option<SolverInitRequest>,
    init_rx: Option<mpsc::Receiver<SolverInitResponse>>,
    init_in_flight: Option<u64>,
    next_init_generation: u64,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    snapshot_seq: u64,
    plot_cache: Option<PlotCache>,
    cached_gpu_stats: CachedGpuStats,
    cached_error: Option<String>,
    cached_message: Option<String>,
    last_init_trace_events: Vec<tracefmt::TraceInitEvent>,
    mesh: Option<Mesh>,
    cached_cells: Vec<Vec<[f64; 2]>>,
    actual_min_cell_size: f64,
    // --- Moving-mesh (ALE) opt-in mode ---
    /// Enable the moving-mesh (ALE) path on the next Initialize / Reset.
    /// Selectable on both compute backends. Enabling it locks Mesh Type to
    /// Voronoi (CVT) + a fixed dt and maps the user's Model to its ALE variant at
    /// build time (never silently swaps the model). Disabled for models without
    /// an ALE variant (see [`CFDApp::ale_model_for`]).
    enable_moving_mesh: bool,
    /// The Mesh Type the user had selected before enabling moving mesh; restored
    /// when moving mesh is disabled (moving locks Mesh Type to Voronoi (CVT)).
    pre_moving_mesh_type: Option<MeshType>,
    /// How the CVT seeds move each step (Frozen / prescribed swirl / flow-coupled).
    moving_motion: MovingMotionChoice,
    /// FlowCoupled centroid-steering strength χ (0 = pure flow advection).
    moving_regularization: f64,
    /// Cross-stream-oscillate the obstacle (ChannelObstacle geometry only).
    /// The obstacle loop's boundary seeds move rigidly with it and its contour
    /// faces carry the moving-wall material velocity (MovingWall BC). Orthogonal
    /// to `moving_motion` (which governs the interior seeds). Applied on
    /// Initialize / Reset.
    moving_oscillate_obstacle: bool,
    /// Oscillating-obstacle cross-stream amplitude (domain units).
    moving_osc_amplitude: f64,
    /// Oscillating-obstacle forcing frequency (Hz); `ω = 2π·f`.
    moving_osc_frequency: f64,
    /// Reconstruct the moving mesh ON DEVICE each step (the meshless-Voronoi
    /// GPU pipeline). Only meaningful on the GPU compute backend — the
    /// request pre-gates it (`backend == Gpu`), and steps the device cannot
    /// certify re-run on the CPU transparently. Default on: a GPU-backend
    /// moving run reconstructs on-device out of the box.
    moving_gpu_regen: bool,
    /// Periodic Lloyd smoothing cadence for FlowCoupled moving runs: every N
    /// steps, one gentle blended Lloyd sweep regularizes the interior seeds
    /// (0 = off; the on-demand quality escalation stays active either way).
    moving_smooth_every_n: usize,
    /// Periodic Morton memory-reordering cadence: every N steps, relabel the
    /// cells in Morton order of the current seed positions (0 = off). Long
    /// FlowCoupled+recycling runs erode the initial memory locality (measured
    /// 63→180 mean slot distance over 2000 steps; reordering holds ~70).
    moving_reorder_every_n: usize,
    /// Flow-adaptive sizing cadence: every N steps the moving driver derives
    /// a per-cell target volume from the flow's velocity/pressure/strain
    /// gradients and births/kills cells toward it — the cell COUNT changes at
    /// runtime, bounded to [0.5, 2]× the initial count (0 = off).
    moving_adapt_every_n: usize,
    /// Explicit adaptation target band in cell-size units: the FINEST cell
    /// size high-gradient regions refine toward, and the COARSEST smooth
    /// regions may grow to — independent of the built mesh's sizing (e.g.
    /// build coarse, adapt finer). `0` = auto: the mesh's realized band.
    /// Applied only when BOTH are > 0.
    moving_adapt_min_size: f64,
    moving_adapt_max_size: f64,
    /// Adaptivity growth budget factor: births stop once the cell count
    /// reaches this multiple of the initial count (default 2×). The per-cell
    /// viz buffers are allocated at the SAME factor, so raising it costs GPU
    /// memory up front.
    moving_adapt_budget: f64,
    /// Per-indicator threshold multipliers (|∇U|, |∇p|, strain rate): each
    /// scales the auto-calibrated (90th-percentile) reference before
    /// normalization — > 1 refines only stronger features, < 1 refines at
    /// weaker ones, 0 disables the component. Default 1.
    moving_adapt_thresh_u: f64,
    moving_adapt_thresh_p: f64,
    moving_adapt_thresh_strain: f64,
    /// IMPLICIT mesh motion: max fixed-point iterations per step — the mesh
    /// is re-advected with the attempt's own end-of-step velocity and the
    /// step re-solved from a rewound t^n state until the motion converges.
    /// 1 = explicit (default). FlowCoupled + CPU backend only.
    moving_motion_iters: usize,
    /// Latest per-step moving-mesh telemetry (from `MeshRefreshed`), for display.
    cached_moving_stats: Option<MovingMeshStats>,
    /// Whether the driver the worker is *actually running* is a moving-mesh
    /// (ALE) driver — captured at Initialize/Reset from the built `SolverMode`,
    /// independent of the `enable_moving_mesh` checkbox (which the user can
    /// un-tick mid-run without reinitializing). Used to keep controls the
    /// moving driver forbids (e.g. adaptive dt) disabled while it is live.
    solver_is_moving: bool,
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    timestep: f64,
    selected_geometry: GeometryType,
    mesh_type: MeshType,
    /// Discretisation mode: dense-Cartesian structured vs unstructured mesh.
    mesh_mode: MeshMode,
    /// STRUCTURED-only: coupled banded-solve preconditioner (block-Jacobi / Schur
    /// / Schur+AMG).
    structured_precond: crate::solver::banded_schur::CoupledPrecondKind,
    /// STRUCTURED-only: per-cell "is this cell inside the immersed obstacle?"
    /// (one entry per `cached_cells` polygon, cell-order). The dense grid never
    /// cuts the obstacle out, so the solid region is drawn as a grey overlay so
    /// the geometry is visible. Empty for the unstructured (cut-cell) path.
    structured_solid_mask: Vec<bool>,
    plot_field: PlotField,
    is_running: bool,
    /// Monotonic ownership token for Run/Pause requests. Worker acknowledgments
    /// from an older initialization or pause must not overwrite a newer Run
    /// click and suppress Direct-frame requests.
    run_request_generation: u64,
    selected_scheme: Scheme,
    current_fluid: Fluid,
    show_mesh_lines: bool,
    // Compute-backend selection (env-driven; applied on Initialize / Reset).
    backend: BackendChoice,
    cpu_threads: usize,
    /// Coupled linear-solve precision for the CPU backends (f64 = reference,
    /// f32 = GPU-like arithmetic, lower bandwidth).
    cpu_precision_f32: bool,
    adaptive_dt: bool,
    target_cfl: f64,
    dual_time: bool,
    dtau: f64,
    log_convergence: bool,
    log_every_steps: u32,
    trace_enabled: bool,
    trace_path: String,
    trace_pending_start: bool,
    render_mode: RenderMode,
    alpha_u: f64,
    alpha_p: f64,
    time_scheme: GpuTimeScheme,
    outer_iters: u32,
    /// Enable the per-step outer-convergence monitor (GUI residual readout +
    /// opportunistic early exit). Decoupled from the console logging flag.
    outer_auto_converge: bool,
    low_mach_model: GpuLowMachPrecondModel,
    low_mach_theta_floor: f32,
    low_mach_pressure_coupling_alpha: f32,
    inlet_velocity: f32,
    /// All-Mach compressibility `psi = 1/c^2` (the wire value). DERIVED, not stored:
    /// the REAL `current_fluid.compressibility()`, recomputed whenever the fluid or
    /// model changes. No exaggeration — always physical.
    compressibility_psi: f32,
    /// Gauge back-pressure pinned at the outlet (all-Mach models). `0.0` is the
    /// standard outlet; the supersonic-nozzle demo sets a negative value to pull the
    /// diverging section past Mach 1. Applied per-case via the GUI defaults.
    outlet_back_pressure: f32,
    /// All-Mach preconditioner reference-velocity floor (see
    /// [`crate::sim::RuntimeParams::allmach_precond_uref_min`]). Higher (~1.0) cleans
    /// the residual standing pressure mode toward the incompressible field; 0.2 is the
    /// bare-stability floor. Live slider (no rebuild).
    allmach_precond_uref_min: f32,
    /// Drive the CD nozzle with a pressure inlet + supersonic outlet (see
    /// `RuntimeParams::pressure_inlet`). Set per-case via the GUI defaults.
    pressure_inlet: bool,
    /// Inlet gauge pressure pinned when `pressure_inlet` is set.
    inlet_pressure: f32,
    selected_preconditioner: PreconditionerType,
    model_id: &'static str,
    model_caps: ModelUiCaps,
    wgpu_device: Option<wgpu::Device>,
    wgpu_queue: Option<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
    cfd_renderer: Option<Arc<Mutex<cfd_renderer::CfdRenderResources>>>,
    viz_field: Option<VizFieldBuffers>,
    viz_display: VizSlotToken,
    /// A moving-mesh refresh arrived and `cached_cells` changed since the GPU
    /// renderer's vertex buffers were last re-tessellated. Multiple solver
    /// steps may land between frames; only the latest matters, so this is a
    /// coalescing flag consumed once per frame at the render-frame boundary.
    pending_mesh_upload: bool,
    /// CFD2_AUTOSTART: start the run as soon as the pending init completes.
    autostart_run_pending: bool,
}

struct CfdRenderCallback {
    renderer: Arc<Mutex<cfd_renderer::CfdRenderResources>>,
    uniforms: cfd_renderer::CfdUniforms,
    draw_lines: bool,
    /// Present only while marching. Posting from `prepare` ties solver-state
    /// copies to actual rendered frames rather than UI-loop guesses.
    viz_frame_request: Option<Arc<VizFrameMailbox>>,
}

impl eframe::egui_wgpu::CallbackTrait for CfdRenderCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(mailbox) = &self.viz_frame_request {
            // `request_frame` intentionally does not change presentation
            // ownership. If this callback races a Direct -> Plot switch, its
            // stale request may be serviced later, but it cannot turn the
            // autonomous Direct producer back on behind Plot's back.
            if mailbox.gpu_consumer_enabled() {
                mailbox.request_frame();
            }
        }
        let renderer = self.renderer.lock().unwrap();
        renderer.update_uniforms(queue, &self.uniforms);
        resources.insert(renderer.clone());
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<cfd_renderer::CfdRenderResources>() {
            let clip_rect = info.clip_rect;
            render_pass.set_viewport(
                clip_rect.min.x as f32,
                clip_rect.min.y as f32,
                clip_rect.width() as f32,
                clip_rect.height() as f32,
                0.0,
                1.0,
            );

            // SAFETY: The renderer lives in CallbackResources which lives as long as the egui renderer.
            // The render pass is executed synchronously and the resources are valid for the duration.
            // We need to extend the lifetime to 'static to match the RenderPass<'static> signature required by egui_wgpu.
            let renderer: &'static cfd_renderer::CfdRenderResources =
                unsafe { std::mem::transmute(renderer) };

            let pixels_per_point = info.pixels_per_point;
            render_pass.set_viewport(
                clip_rect.min.x * pixels_per_point,
                clip_rect.min.y * pixels_per_point,
                clip_rect.width() * pixels_per_point,
                clip_rect.height() * pixels_per_point,
                0.0,
                1.0,
            );

            renderer.paint(render_pass, self.draw_lines);
        }
    }
}

impl CFDApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (wgpu_device, wgpu_queue, target_format) =
            if let Some(render_state) = &cc.wgpu_render_state {
                (
                    Some(render_state.device.clone()),
                    Some(render_state.queue.clone()),
                    render_state.target_format,
                )
            } else {
                (None, None, wgpu::TextureFormat::Bgra8Unorm)
            };

        let default_fluid = Fluid::presets()[1].clone();

        let mut app = Self {
            solver_worker: SolverWorkerHandle::spawn(),
            pending_init_request: None,
            init_rx: None,
            init_in_flight: None,
            next_init_generation: 1,
            cached_u: Vec::new(),
            cached_p: Vec::new(),
            snapshot_seq: 0,
            plot_cache: None,
            cached_gpu_stats: CachedGpuStats::default(),
            cached_error: None,
            cached_message: None,
            last_init_trace_events: Vec::new(),
            mesh: None,
            cached_cells: Vec::new(),
            actual_min_cell_size: 0.01,
            enable_moving_mesh: false,
            pre_moving_mesh_type: None,
            moving_motion: MovingMotionChoice::default(),
            moving_regularization: 0.5,
            moving_oscillate_obstacle: false,
            // Below the default near-wall cell spacing (min_cell_size 0.025): a
            // larger peak displacement sweeps the rigidly-moving obstacle through
            // the frozen interior seeds and tangles the mesh. The driver clamps
            // this to `OSC_AMPLITUDE_CELL_FRACTION × cell spacing` regardless, but
            // an honest default keeps the slider value == the realized motion.
            moving_osc_amplitude: 0.01,
            moving_osc_frequency: 0.5,
            moving_gpu_regen: true,
            moving_smooth_every_n: 0,
            moving_reorder_every_n: 500,
            moving_adapt_every_n: 0,
            moving_adapt_min_size: 0.0,
            moving_adapt_max_size: 0.0,
            moving_adapt_budget: 2.0,
            moving_adapt_thresh_u: 1.0,
            moving_adapt_thresh_p: 1.0,
            moving_adapt_thresh_strain: 1.0,
            moving_motion_iters: 1,
            cached_moving_stats: None,
            solver_is_moving: false,
            min_cell_size: 0.025,
            max_cell_size: 0.025,
            growth_rate: 1.2,
            timestep: 0.02,
            selected_geometry: GeometryType::default(),
            mesh_type: MeshType::default(),
            mesh_mode: MeshMode::default(),
            structured_precond: crate::solver::banded_schur::CoupledPrecondKind::BlockJacobi,
            structured_solid_mask: Vec::new(),
            plot_field: PlotField::VelocityMag,
            is_running: false,
            run_request_generation: 0,
            selected_scheme: Scheme::Upwind,
            current_fluid: default_fluid,
            show_mesh_lines: true,
            backend: BackendChoice::Gpu,
            cpu_threads: 1,
            cpu_precision_f32: false,
            adaptive_dt: true,
            target_cfl: 0.9,
            dual_time: false,
            dtau: 1e-5,
            log_convergence: false,
            log_every_steps: 25,
            trace_enabled: false,
            trace_path: "cfd2_trace.jsonl".to_string(),
            trace_pending_start: false,
            render_mode: RenderMode::GpuDirect,
            alpha_u: 0.7,
            alpha_p: 0.3,
            time_scheme: GpuTimeScheme::BDF2,
            outer_iters: 8,
            outer_auto_converge: true,
            low_mach_model: GpuLowMachPrecondModel::Off,
            low_mach_theta_floor: 1e-6,
            low_mach_pressure_coupling_alpha: 1.0,
            inlet_velocity: 1.0,
            compressibility_psi: 0.0,
            outlet_back_pressure: 0.0,
            allmach_precond_uref_min: 0.2,
            pressure_inlet: false,
            inlet_pressure: 0.0,
            selected_preconditioner: PreconditionerType::Jacobi,
            model_id: "incompressible_momentum",
            model_caps: ModelUiCaps::default(),
            wgpu_device,
            wgpu_queue,
            target_format,
            cfd_renderer: None,
            viz_field: None,
            viz_display: VizSlotToken {
                index: 0,
                sequence: 0,
            },
            pending_mesh_upload: false,
            autostart_run_pending: false,
        };
        app.apply_model_defaults();
        app.refresh_model_caps();
        app.sync_worker_params();
        // Headed perf harness: CFD2_AUTOSTART=structured-rk4|unstructured-rk4
        // configures the GUI obstacle case (matched ~4.8k cells), initializes,
        // and starts running as soon as the solver is ready — so a scripted
        // headed run can A/B the two explicit compressible GPU paths under
        // the real renderer. Pair with CFD2_PERF_LOG=1 for stderr stats.
        if let Ok(autostart) = std::env::var("CFD2_AUTOSTART") {
            app.selected_geometry = GeometryType::ChannelObstacle;
            app.min_cell_size = 0.025;
            app.max_cell_size = 0.025;
            // Optional overrides appended as ":cell=<f64>,u=<f32>", e.g.
            // "unstructured-rk4:cell=0.005,u=1000" for stress configs.
            let (autostart, overrides) = match autostart.split_once(':') {
                Some((mode, rest)) => (mode.to_string(), Some(rest.to_string())),
                None => (autostart, None),
            };
            if let Some(overrides) = overrides {
                for part in overrides.split(',') {
                    if let Some(value) = part.strip_prefix("cell=") {
                        let cell: f64 = value.parse().expect("CFD2_AUTOSTART cell override");
                        app.min_cell_size = cell;
                        app.max_cell_size = cell;
                    } else if let Some(value) = part.strip_prefix("u=") {
                        app.inlet_velocity =
                            value.parse().expect("CFD2_AUTOSTART inlet override");
                    }
                }
            }
            match autostart.as_str() {
                "structured-rk4" => {
                    app.mesh_mode = MeshMode::Structured2D;
                    app.model_id = "compressible_structured";
                }
                "unstructured-rk4" => {
                    app.mesh_mode = MeshMode::Unstructured;
                    app.mesh_type = MeshType::CutCell;
                    app.model_id = "compressible";
                }
                other => panic!("unknown CFD2_AUTOSTART '{other}'"),
            }
            app.apply_model_defaults();
            app.time_scheme = GpuTimeScheme::RK4;
            app.adaptive_dt = true;
            app.dual_time = false;
            if app.mesh_mode == MeshMode::Structured2D {
                app.backend = app.structured_backend_default();
            } else {
                app.backend = BackendChoice::Gpu;
            }
            app.refresh_model_caps();
            app.sync_worker_params();
            app.init_solver();
            app.autostart_run_pending = true;
        }
        app
    }

    fn current_runtime_params(&self) -> RuntimeParams {
        RuntimeParams {
            adaptive_dt: self.adaptive_dt,
            target_cfl: self.target_cfl,
            requested_dt: self.timestep as f32,
            dtau: if self.dual_time {
                self.dtau.max(0.0) as f32
            } else {
                0.0
            },
            log_convergence: self.log_convergence,
            log_every_steps: self.log_every_steps.max(1),
            advection_scheme: self.selected_scheme,
            time_scheme: self.time_scheme,
            preconditioner: self.selected_preconditioner,
            outer_iters: self.outer_iters.max(1),
            outer_auto_converge: self.outer_auto_converge,
            low_mach_model: self.low_mach_model,
            low_mach_theta_floor: self.low_mach_theta_floor,
            low_mach_pressure_coupling_alpha: self.low_mach_pressure_coupling_alpha,
            alpha_u: self.alpha_u as f32,
            alpha_p: self.alpha_p as f32,
            inlet_velocity: self.inlet_velocity,
            density: self.current_fluid.density as f32,
            viscosity: self.current_fluid.viscosity as f32,
            eos: self.current_fluid.eos,
            compressibility_psi: self.compressibility_psi,
            outlet_back_pressure: self.outlet_back_pressure,
            allmach_precond_uref_min: self.allmach_precond_uref_min,
            pressure_inlet: self.pressure_inlet,
            inlet_pressure: self.inlet_pressure,
        }
    }

    fn sync_worker_params(&self) {
        self.solver_worker.send(SolverWorkerCommand::UpdateParams(
            self.current_runtime_params(),
        ));
    }

    fn set_worker_running(&mut self, running: bool) {
        self.run_request_generation = self.run_request_generation.wrapping_add(1);
        self.solver_worker.send(SolverWorkerCommand::SetRunning {
            running,
            generation: self.run_request_generation,
        });
    }

    /// Apply the per-model GUI default solver parameters (single source of truth
    /// in `model_defaults`). Called at startup and whenever the active model
    /// changes, so the incompressible and compressible solvers each get sane,
    /// non-diverging knobs instead of one shared set tuned for neither.
    fn apply_model_defaults(&mut self) {
        // The supersonic-nozzle demo is the composition (all-Mach model + nozzle
        // geometry): override the per-model defaults with the tuned nozzle case
        // (choking inlet speed + sub-critical outlet back-pressure) so selecting the
        // nozzle "just works" as a supersonic case. Every other case keeps a 0
        // back-pressure (standard outlet).
        let is_allmach = matches!(
            self.model_id,
            "allmach_pressure" | "allmach_thermal" | "allmach_thermal_structured"
        );
        let is_compressible = matches!(
            self.model_id,
            "compressible" | "compressible_structured"
        );
        let d = if self.selected_geometry == GeometryType::Nozzle && is_allmach {
            crate::ui::model_defaults::ALLMACH_THERMAL_NOZZLE
        } else if self.selected_geometry == GeometryType::Nozzle && is_compressible {
            crate::ui::model_defaults::COMPRESSIBLE_NOZZLE
        } else {
            crate::ui::model_defaults::gui_defaults_for(self.model_id)
        };
        self.selected_scheme = d.advection_scheme;
        self.time_scheme = d.time_scheme;
        self.selected_preconditioner = d.preconditioner;
        // STRUCTURED coupled preconditioner defaults (fast path first):
        // saddle models (incompressible / all-Mach thermal) → Schur+AMG — the
        // h-independent pressure solve. BlockJacobi "converges" on fine grids but
        // leaves an unphysical checkerboard pressure field (no Kármán street).
        // Adaptive AMG latch: heavy-ball until A_pp is well-conditioned / until a
        // slow outer is seen, then V-cycle (must flip MID-step so remaining
        // Picard outers of the same step benefit). Compressible → block-Jacobi
        // (no Schur layout). User can override via the radio.
        self.structured_precond = match self.model_id {
            "incompressible_momentum_structured" | "allmach_thermal_structured" => {
                crate::solver::banded_schur::CoupledPrecondKind::SchurAmg
            }
            _ => crate::solver::banded_schur::CoupledPrecondKind::BlockJacobi,
        };
        self.alpha_u = d.alpha_u;
        self.alpha_p = d.alpha_p;
        self.outer_iters = d.outer_iters;
        self.outer_auto_converge = d.outer_auto_converge;
        self.target_cfl = d.target_cfl;
        self.timestep = d.timestep;
        // Adaptive dt is supported on BOTH paths now: static runs use the
        // solver-side controller; moving (ALE) runs route the same checkbox
        // to the driver-side flow-CFL controller (pinned inside the GCL dt
        // handshake), and the worker strips `params.adaptive_dt` before it
        // can reach the inner solver — so the old force-off guard here would
        // only silently revert the user's explicit choice on a model or
        // geometry change.
        self.adaptive_dt = d.adaptive_dt;
        // Pseudo-transient continuation: the low-Mach stabilizer for the
        // compressible default (see model_defaults). Off for incompressible.
        self.dual_time = d.dual_time;
        self.dtau = d.dtau;
        self.low_mach_model = d.low_mach_model;
        self.low_mach_theta_floor = d.low_mach_theta_floor;
        self.low_mach_pressure_coupling_alpha = d.low_mach_pressure_coupling_alpha;
        self.inlet_velocity = d.inlet_velocity;
        // All-Mach compressibility: the REAL `psi = 1/c^2` of the current fluid, no
        // exaggeration. Recomputed here (model switch) and on fluid change.
        self.compressibility_psi = self.current_fluid.compressibility() as f32;
        // Outlet back-pressure: negative only for the supersonic-nozzle case (above).
        self.outlet_back_pressure = d.outlet_back_pressure;
        self.allmach_precond_uref_min = d.allmach_precond_uref_min;
        self.pressure_inlet = d.pressure_inlet;
        self.inlet_pressure = d.inlet_pressure;
    }

    fn current_trace_runtime_params(&self) -> tracefmt::TraceRuntimeParams {
        tracefmt::TraceRuntimeParams {
            adaptive_dt: self.adaptive_dt,
            target_cfl: self.target_cfl,
            requested_dt: self.timestep as f32,
            dtau: if self.dual_time {
                self.dtau.max(0.0) as f32
            } else {
                0.0
            },
            advection_scheme: tracefmt::TraceScheme::from(self.selected_scheme),
            time_scheme: tracefmt::TraceTimeScheme::from(self.time_scheme),
            preconditioner: tracefmt::TracePreconditioner::from(self.selected_preconditioner),
            outer_iters: self.outer_iters.max(1),
            low_mach_model: tracefmt::TraceLowMachModel::from(self.low_mach_model),
            low_mach_theta_floor: self.low_mach_theta_floor,
            low_mach_pressure_coupling_alpha: self.low_mach_pressure_coupling_alpha,
            alpha_u: self.alpha_u as f32,
            alpha_p: self.alpha_p as f32,
            inlet_velocity: self.inlet_velocity,
            density: self.current_fluid.density as f32,
            viscosity: self.current_fluid.viscosity as f32,
            eos: tracefmt::TraceEosSpec::from(self.current_fluid.eos),
        }
    }

    fn build_trace_header(&self, mesh: &Mesh) -> Result<tracefmt::TraceHeader, String> {
        let mesh = tracefmt::TraceMesh::from_mesh(mesh)?;

        let geometry = match self.selected_geometry {
            GeometryType::BackwardsStep => tracefmt::TraceGeometry::BackwardsStep,
            GeometryType::ChannelObstacle => tracefmt::TraceGeometry::ChannelObstacle,
            GeometryType::Nozzle => tracefmt::TraceGeometry::Nozzle,
        };
        let mesh_type = match self.mesh_type {
            MeshType::CutCell => tracefmt::TraceMeshType::CutCell,
            MeshType::Delaunay => tracefmt::TraceMeshType::Delaunay,
            MeshType::Voronoi => tracefmt::TraceMeshType::Voronoi,
            MeshType::VoronoiCvt => tracefmt::TraceMeshType::VoronoiCvt,
            MeshType::Fitted => tracefmt::TraceMeshType::Fitted,
        };

        let stepping_mode = if self.model_caps.supports_eos_tuning {
            tracefmt::TraceSteppingMode::Implicit
        } else {
            tracefmt::TraceSteppingMode::Coupled
        };

        let fluid = tracefmt::TraceFluid {
            name: Some(self.current_fluid.name.clone()),
            density: self.current_fluid.density,
            viscosity: self.current_fluid.viscosity,
            eos: tracefmt::TraceEosSpec::from(self.current_fluid.eos),
        };

        let case = tracefmt::TraceCase {
            model_id: self.model_id.to_string(),
            geometry,
            mesh_type,
            min_cell_size: self.min_cell_size,
            max_cell_size: self.max_cell_size,
            growth_rate: self.growth_rate,
            fluid,
            stepping_mode,
            mesh,
        };

        Ok(tracefmt::make_header(
            case,
            self.current_trace_runtime_params(),
            cfg!(feature = "profiling"),
        ))
    }

    fn start_trace(&mut self) {
        let Some(mesh) = &self.mesh else {
            self.trace_pending_start = true;
            return;
        };

        let header = match self.build_trace_header(mesh) {
            Ok(v) => v,
            Err(err) => {
                self.trace_enabled = false;
                self.trace_pending_start = false;
                self.cached_error = Some(format!("failed to start trace: {err}"));
                return;
            }
        };

        self.trace_pending_start = false;
        self.solver_worker.send(SolverWorkerCommand::StartTrace {
            path: self.trace_path.clone(),
            header,
        });
        if !self.last_init_trace_events.is_empty() {
            self.solver_worker
                .send(SolverWorkerCommand::AppendTraceInit {
                    events: self.last_init_trace_events.clone(),
                });
        }
    }

    fn stop_trace(&mut self) {
        self.trace_pending_start = false;
        self.solver_worker.send(SolverWorkerCommand::StopTrace);
    }

    fn model_label(model_id: &'static str) -> &'static str {
        match model_id {
            "incompressible_momentum" => "Incompressible momentum",
            "compressible" => "Compressible",
            "allmach_pressure" => "All-Mach (pressure-based)",
            "allmach_thermal" => "All-Mach (pressure-based, thermal)",
            other => other,
        }
    }

    /// The moving-mesh (ALE) model id for a selected base model, or `None` if the
    /// model has no ALE variant. This is the single source of truth for "does
    /// this model support moving mesh": the toggle is disabled (with a tooltip)
    /// for models that return `None`, and `build_moving_init` maps the user's
    /// chosen model to this variant at build time. The user's model selection is
    /// therefore NEVER silently overwritten — enabling moving mesh keeps the same
    /// physics family and only substitutes the mesh-relative (ALE) discretization.
    fn ale_model_for(model_id: &str) -> Option<&'static str> {
        match model_id {
            "incompressible_momentum" | "incompressible_momentum_ale" => {
                Some("incompressible_momentum_ale")
            }
            "allmach_pressure" | "allmach_pressure_ale" => Some("allmach_pressure_ale"),
            "allmach_thermal" | "allmach_thermal_ale" => Some("allmach_thermal_ale"),
            // Density-based `compressible` has no ALE variant (roadmap off-ramp:
            // it stays the static true-supersonic solver). Any unknown id: no ALE.
            _ => None,
        }
    }

    /// The physical models the GUI Model dropdown offers. Static lists only —
    /// never call `all_models()` here: that rebuilds every registered model
    /// (including ~120 ms compressible flux-module kernel programs ×4) and was
    /// measured at ~485 ms per call, paid on every UI frame that painted this
    /// panel. Verification variants (`*_mms*`, `*biharmonic*`, `*demo*`) are
    /// intentionally excluded.
    fn supported_ui_models(mesh_mode: MeshMode) -> Vec<(&'static str, &'static str)> {
        // Structured mode offers exactly the models with a `TopologyMode::Structured2D`
        // variant (dense-Cartesian, immersed obstacles) — a fixed short list.
        if mesh_mode == MeshMode::Structured2D {
            return vec![
                ("incompressible_momentum_structured", "Incompressible"),
                ("allmach_thermal_structured", "All-Mach Thermal"),
                ("compressible_structured", "Compressible"),
            ];
        }
        // Only physical-flow models belong in the GUI (these are exactly the ones
        // `model_label` names). Sorted by id so the ComboBox order is stable.
        vec![
            ("allmach_pressure", CFDApp::model_label("allmach_pressure")),
            ("allmach_thermal", CFDApp::model_label("allmach_thermal")),
            ("compressible", CFDApp::model_label("compressible")),
            (
                "incompressible_momentum",
                CFDApp::model_label("incompressible_momentum"),
            ),
        ]
    }

    fn build_selected_model(&self) -> Result<ModelSpec, String> {
        if self.mesh_mode == MeshMode::Structured2D {
            return structured_model_by_id(self.model_id);
        }
        unstructured_model_by_id(self.model_id, self.current_fluid.eos)
    }

    fn refresh_model_caps(&mut self) {
        let model = match self.build_selected_model() {
            Ok(m) => m,
            Err(_) => {
                self.model_id = "incompressible_momentum";
                self.build_selected_model()
                    .expect("default UI model 'incompressible_momentum' must exist")
            }
        };

        let named_params = model.named_param_keys();
        self.model_caps.supports_preconditioner =
            named_params.iter().any(|&k| k == "preconditioner");
        self.model_caps.model_owns_preconditioner = model
            .linear_solver
            .map(|s| matches!(s.preconditioner, ModelPreconditionerSpec::Schur { .. }))
            .unwrap_or(false);
        self.model_caps.unknowns_per_cell = model.system.unknowns_per_cell();

        self.model_caps.supports_alpha_u = named_params.iter().any(|&k| k == "alpha_u");
        self.model_caps.supports_alpha_p = named_params.iter().any(|&k| k == "alpha_p");
        self.model_caps.supports_eos_tuning = named_params.iter().any(|&k| k == "eos.gamma");
        self.model_caps.supports_outer_iters = named_params.iter().any(|&k| k == "outer_iters");
        self.model_caps.supports_dtau = named_params.iter().any(|&k| k == "dtau");
        self.model_caps.supports_low_mach = named_params.iter().any(|&k| k == "low_mach.model");
        self.model_caps.supports_explicit_rk4 = model.validate_explicit_rk4().is_ok();

        // Use UiPortSet for field offset resolution (validates field types)
        let ui_ports = UiPortSet::from_layout(&model.state_layout);
        self.model_caps.plot_stride = ui_ports.stride;
        self.model_caps.plot_has_u = ui_ports.u_offset.is_some();
        self.model_caps.plot_u_offset = ui_ports.u_offset.unwrap_or(0);
        self.model_caps.plot_has_p = ui_ports.p_offset.is_some();
        self.model_caps.plot_p_offset = ui_ports.p_offset.unwrap_or(0);

        const UI_MAX_BLOCK_JACOBI: u32 = 16;
        if matches!(
            self.selected_preconditioner,
            PreconditionerType::BlockJacobi
        ) && self.model_caps.unknowns_per_cell > UI_MAX_BLOCK_JACOBI
        {
            self.selected_preconditioner = PreconditionerType::Jacobi;
        }
    }

    fn make_init_request(&mut self) -> SolverInitRequest {
        // Defense in depth for programmatic/request-order paths: the checkbox
        // handler below normally performs this coercion immediately, but an
        // initialization request must never carry the static-only RK4 scheme
        // into an ALE model even if UI events were delivered in another order.
        enforce_moving_mesh_time_scheme(self.enable_moving_mesh, &mut self.time_scheme);
        let generation = self.next_init_generation;
        self.next_init_generation = self.next_init_generation.wrapping_add(1);
        SolverInitRequest {
            generation,
            model_id: self.model_id,
            selected_geometry: self.selected_geometry,
            mesh_type: self.mesh_type,
            mesh_mode: self.mesh_mode,
            structured_precond: self.structured_precond,
            backend: self.backend,
            min_cell_size: self.min_cell_size,
            max_cell_size: self.max_cell_size,
            growth_rate: self.growth_rate,
            current_fluid: self.current_fluid.clone(),
            params: self.current_runtime_params(),
            enable_moving_mesh: self.enable_moving_mesh,
            moving_motion: self.moving_motion,
            moving_regularization: self.moving_regularization,
            moving_oscillate_obstacle: self.moving_oscillate_obstacle,
            moving_osc_amplitude: self.moving_osc_amplitude,
            moving_osc_frequency: self.moving_osc_frequency,
            // Pre-gate on the COMPUTE backend: the render device exists even
            // for CPU-solver runs, so the checkbox alone must not enable the
            // device path there.
            moving_gpu_regen: self.moving_gpu_regen && self.backend == BackendChoice::Gpu,
            moving_smooth_every_n: self.moving_smooth_every_n,
            moving_reorder_every_n: self.moving_reorder_every_n,
            moving_adapt_every_n: self.moving_adapt_every_n,
            moving_adapt_min_size: self.moving_adapt_min_size,
            moving_adapt_max_size: self.moving_adapt_max_size,
            moving_adapt_budget: self.moving_adapt_budget.clamp(1.0, 16.0),
            moving_adapt_thresh_u: self.moving_adapt_thresh_u,
            moving_adapt_thresh_p: self.moving_adapt_thresh_p,
            moving_adapt_thresh_strain: self.moving_adapt_thresh_strain,
            moving_motion_iters: self.moving_motion_iters.max(1),
            wgpu_device: self.wgpu_device.clone(),
            wgpu_queue: self.wgpu_queue.clone(),
            target_format: self.target_format,
        }
    }

    fn update_renderer_field(&self) {
        // No-op: the renderer bind group is created at solver init time and points at the
        // visualization buffer. Field selection is driven by uniforms (stride/offset/mode).
    }

    /// Apply the selected compute backend via environment (read by
    /// `UnifiedSolver::new`). CPU options are runtime-switchable; changes take
    /// effect on the next solver (re)build.
    fn apply_backend_env(&self) {
        if self.backend.is_cpu() {
            std::env::set_var("CFD2_BACKEND", "cpu");
            std::env::set_var(
                "CFD2_CPU_ENGINE",
                if self.backend == BackendChoice::CpuInterpreter {
                    "interpreter"
                } else {
                    "transpiled"
                },
            );
            std::env::set_var("CFD2_CPU_THREADS", self.cpu_threads.max(1).to_string());
            std::env::set_var(
                "CFD2_CPU_SIMD",
                if self.backend == BackendChoice::CpuTranspiledSimd {
                    "1"
                } else {
                    "0"
                },
            );
            std::env::set_var(
                "CFD2_CPU_PRECISION",
                if self.cpu_precision_f32 { "f32" } else { "f64" },
            );
        } else {
            std::env::remove_var("CFD2_BACKEND");
        }
    }

    fn init_solver(&mut self) {
        self.apply_backend_env();
        self.is_running = false;
        self.set_worker_running(false);
        self.solver_worker.send(SolverWorkerCommand::StopTrace);
        self.trace_pending_start = self.trace_enabled;
        self.solver_worker.send(SolverWorkerCommand::ClearSolver);
        self.refresh_model_caps();
        self.pending_init_request = Some(self.make_init_request());
        self.mesh = None;
        self.cached_cells.clear();
        self.cfd_renderer = None;
        self.cached_u.clear();
        self.cached_p.clear();
        self.snapshot_seq = 0;
        self.invalidate_plot_cache();
        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_moving_stats = None;
        self.cached_error = None;
        self.cached_message = None;
        self.last_init_trace_events.clear();
    }

    fn build_mesh_with(
        selected_geometry: GeometryType,
        mesh_type: MeshType,
        min_cell_size: f64,
        max_cell_size: f64,
        growth_rate: f64,
        trace_init_events: &mut Vec<tracefmt::TraceInitEvent>,
    ) -> Mesh {
        // The sizing sliders accept any hand-typed value (`SliderClamping::Never`),
        // so sanitise before the values reach the mesh generators: sizes must be
        // finite and positive, `min <= max`, and the growth rate at least 1. This
        // keeps the UI free while preventing a NaN/∞/divide-by-zero, and — crucially
        // for the UNSTRUCTURED meshers — bounding the base grid. `generate_cut_cell_mesh`
        // et al. build an uncapped `(domain/max_cell_size)^2` base grid, so a tiny
        // hand-typed size would OOM; the `MIN_CELL_SIZE` floor keeps the base grid
        // ≲ few·1e6 cells on these domains.
        // The fitted structured grid is unaffected: its resolution is bounded by the
        // nx/ny clamp (≈5.9e-3 cell), coarser than this floor, so the floor never binds.
        const MIN_CELL_SIZE: f64 = 1e-3;
        let max_cell_size = if max_cell_size.is_finite() {
            max_cell_size.max(MIN_CELL_SIZE)
        } else {
            0.05
        };
        let min_cell_size = if min_cell_size.is_finite() {
            min_cell_size.clamp(MIN_CELL_SIZE, max_cell_size)
        } else {
            max_cell_size
        };
        let growth_rate = if growth_rate.is_finite() {
            growth_rate.max(1.0)
        } else {
            1.2
        };

        fn mesh_type_id(mesh_type: MeshType) -> &'static str {
            match mesh_type {
                MeshType::CutCell => "cutcell",
                MeshType::Delaunay => "delaunay",
                MeshType::Voronoi => "voronoi",
                MeshType::VoronoiCvt => "voronoi_cvt",
                MeshType::Fitted => "fitted",
            }
        }

        fn geometry_id(geometry: GeometryType) -> &'static str {
            match geometry {
                GeometryType::BackwardsStep => "backwards_step",
                GeometryType::ChannelObstacle => "channel_obstacle",
                GeometryType::Nozzle => "nozzle",
            }
        }

        let geometry = geometry_id(selected_geometry);
        let mesh_kind = mesh_type_id(mesh_type);
        let total_start = std::time::Instant::now();

        let mesh = match selected_geometry {
            GeometryType::BackwardsStep => {
                let geo = gui_backstep_geometry();
                let domain_size = Vector2::new(geo.length, geo.height_outlet);

                let gen_start = std::time::Instant::now();
                // `Fitted` is a nozzle-only option; it never reaches this geometry
                // from the UI, so fall back to the cut-cell mesher if it does.
                let mut mesh = match mesh_type {
                    MeshType::CutCell | MeshType::Fitted => generate_cut_cell_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Delaunay => generate_delaunay_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Voronoi => generate_voronoi_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::VoronoiCvt => generate_cvt_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                        &LloydConfig::default(),
                    ),
                };

                CFDApp::push_trace_init_event(
                    trace_init_events,
                    format!("mesh.generate.{geometry}.{mesh_kind}"),
                    gen_start.elapsed(),
                    Some(format!(
                        "min={min_cell_size:.4e} max={max_cell_size:.4e} growth={growth_rate:.3} cells={} faces={} vertices={}",
                        mesh.num_cells(),
                        mesh.num_faces(),
                        mesh.num_vertices()
                    )),
                );

                // The CVT mesh optimizes seed positions instead: vertex
                // smoothing would move Voronoi vertices off the bisectors
                // and destroy the mesh's defining property.
                if mesh_type != MeshType::VoronoiCvt {
                    let smooth_start = std::time::Instant::now();
                    mesh.smooth(&geo, 0.3, 50);
                    CFDApp::push_trace_init_event(
                        trace_init_events,
                        format!("mesh.smooth.{geometry}.{mesh_kind}"),
                        smooth_start.elapsed(),
                        Some("factor=0.3 iters=50".to_string()),
                    );
                }

                mesh
            }
            GeometryType::ChannelObstacle => {
                let geo = gui_channel_obstacle_geometry();
                let domain_size = Vector2::new(geo.length, geo.height);

                let gen_start = std::time::Instant::now();
                // `Fitted` is a nozzle-only option; it never reaches this geometry
                // from the UI, so fall back to the cut-cell mesher if it does.
                let mut mesh = match mesh_type {
                    MeshType::CutCell | MeshType::Fitted => generate_cut_cell_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Delaunay => generate_delaunay_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Voronoi => generate_voronoi_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::VoronoiCvt => generate_cvt_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                        &LloydConfig::default(),
                    ),
                };

                CFDApp::push_trace_init_event(
                    trace_init_events,
                    format!("mesh.generate.{geometry}.{mesh_kind}"),
                    gen_start.elapsed(),
                    Some(format!(
                        "min={min_cell_size:.4e} max={max_cell_size:.4e} growth={growth_rate:.3} cells={} faces={} vertices={}",
                        mesh.num_cells(),
                        mesh.num_faces(),
                        mesh.num_vertices()
                    )),
                );

                // The CVT mesh optimizes seed positions instead: vertex
                // smoothing would move Voronoi vertices off the bisectors
                // and destroy the mesh's defining property.
                if mesh_type != MeshType::VoronoiCvt {
                    let smooth_iters = match mesh_type {
                        MeshType::CutCell | MeshType::Fitted => 100,
                        MeshType::Delaunay | MeshType::Voronoi => 50,
                        MeshType::VoronoiCvt => unreachable!(),
                    };

                    let smooth_start = std::time::Instant::now();
                    mesh.smooth(&geo, 0.3, smooth_iters);
                    CFDApp::push_trace_init_event(
                        trace_init_events,
                        format!("mesh.smooth.{geometry}.{mesh_kind}"),
                        smooth_start.elapsed(),
                        Some(format!("factor=0.3 iters={smooth_iters}")),
                    );
                }

                mesh
            }
            GeometryType::Nozzle => {
                // Converging–diverging nozzle, area ratio exit/throat = 2, matching
                // the validated `allmach_thermal_supersonic_test`. `Fitted` builds
                // the body-fitted structured grid (the validated default); the
                // unstructured mesh types conform to the same wall profile via the
                // `Nozzle` SDF geometry and honour the full sizing controls.
                let shared_geo = gui_nozzle_geometry();
                let length = shared_geo.length;
                let height = shared_geo.height;
                let throat_h = shared_geo.throat_height;
                let throat_frac = shared_geo.throat_frac;
                let exit_h = shared_geo.exit_height;

                match mesh_type {
                    MeshType::Fitted => {
                        // Uniform curvilinear structured grid. Resolution is derived
                        // from the (target) cell-size slider. The lower clamp keeps the
                        // throat resolved enough to choke; the upper clamp lets the user
                        // drive the mesh much finer than before (down to ~0.006) while
                        // still bounding a hand-typed size so it can't blow up memory.
                        let nx = ((length / max_cell_size).round() as usize).clamp(64, 512);
                        let ny = ((height / max_cell_size).round() as usize).clamp(24, 192);

                        let gen_start = std::time::Instant::now();
                        // Symmetric (both walls curved) CD nozzle — the iconic bell
                        // shape with a clean symmetric core jet (no asymmetric
                        // flat-bottom boundary layer). Same area ratio / throat as the
                        // flat-bottom profile.
                        let mesh = generate_structured_symmetric_nozzle_mesh(
                            nx,
                            ny,
                            length,
                            height,
                            throat_h,
                            throat_frac,
                            exit_h,
                            BoundarySides {
                                left: BoundaryType::Inlet,
                                right: BoundaryType::Outlet,
                                bottom: BoundaryType::Wall,
                                top: BoundaryType::Wall,
                            },
                        );
                        CFDApp::push_trace_init_event(
                            trace_init_events,
                            format!("mesh.generate.{geometry}.{mesh_kind}"),
                            gen_start.elapsed(),
                            Some(format!(
                                "nx={nx} ny={ny} area_ratio={:.2} cells={} faces={} vertices={}",
                                exit_h / throat_h,
                                mesh.num_cells(),
                                mesh.num_faces(),
                                mesh.num_vertices()
                            )),
                        );

                        mesh
                    }
                    MeshType::CutCell
                    | MeshType::Delaunay
                    | MeshType::Voronoi
                    | MeshType::VoronoiCvt => {
                        // Unstructured mesh conforming to the nozzle SDF. The bounding
                        // box is the inlet-height rectangle; the mesher tags the left
                        // edge Inlet, the right edge Outlet, and the flat bottom Wall.
                        let domain_size = Vector2::new(length, height);
                        let geo = shared_geo;

                        let gen_start = std::time::Instant::now();
                        let mut mesh = match mesh_type {
                            MeshType::Delaunay => generate_delaunay_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                            ),
                            MeshType::Voronoi => generate_voronoi_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                            ),
                            MeshType::VoronoiCvt => generate_cvt_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                                &LloydConfig::default(),
                            ),
                            // CutCell (and the unreachable Fitted, already handled).
                            _ => generate_cut_cell_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                            ),
                        };
                        CFDApp::push_trace_init_event(
                            trace_init_events,
                            format!("mesh.generate.{geometry}.{mesh_kind}"),
                            gen_start.elapsed(),
                            Some(format!(
                                "min={min_cell_size:.4e} max={max_cell_size:.4e} growth={growth_rate:.3} cells={} faces={} vertices={}",
                                mesh.num_cells(),
                                mesh.num_faces(),
                                mesh.num_vertices()
                            )),
                        );

                        // The CVT mesh optimizes seed positions instead:
                        // vertex smoothing would move Voronoi vertices off
                        // the bisectors and destroy its defining property.
                        if mesh_type != MeshType::VoronoiCvt {
                            let smooth_iters = match mesh_type {
                                MeshType::Delaunay | MeshType::Voronoi => 50,
                                _ => 100,
                            };
                            let smooth_start = std::time::Instant::now();
                            mesh.smooth(&geo, 0.3, smooth_iters);
                            CFDApp::push_trace_init_event(
                                trace_init_events,
                                format!("mesh.smooth.{geometry}.{mesh_kind}"),
                                smooth_start.elapsed(),
                                Some(format!("factor=0.3 iters={smooth_iters}")),
                            );
                        }

                        // The curved top wall (untagged by `classify_boundary`) is
                        // closed as a no-slip wall inside every unstructured mesh
                        // generator (`close_untagged_boundary_faces`).
                        mesh
                    }
                }
            }
        };

        CFDApp::push_trace_init_event(
            trace_init_events,
            format!("mesh.total.{geometry}.{mesh_kind}"),
            total_start.elapsed(),
            Some(format!(
                "cells={} faces={} vertices={}",
                mesh.num_cells(),
                mesh.num_faces(),
                mesh.num_vertices()
            )),
        );

        mesh
    }

    fn push_trace_init_event(
        events: &mut Vec<tracefmt::TraceInitEvent>,
        stage: impl Into<String>,
        elapsed: std::time::Duration,
        detail: Option<String>,
    ) {
        events.push(tracefmt::TraceInitEvent {
            stage: stage.into(),
            wall_time_ms: elapsed.as_secs_f32() * 1000.0,
            detail,
        });
    }

    fn cache_cells(mesh: &Mesh) -> Vec<Vec<[f64; 2]>> {
        // An on-device (GPU-regen) mesh is vertex-less: reconstruct each
        // cell's polygon from its faces' stored geometry instead. A face is a
        // segment of length `area` through `(face_cx, face_cy)` perpendicular
        // to `(face_nx, face_ny)`; its two endpoints are the cell's Voronoi
        // vertices. Order corners by angle around the cell centre (cells are
        // convex) and merge the shared endpoint of adjacent faces.
        if mesh.cell_vertex_offsets.is_empty() && mesh.num_cells() > 0 {
            let mut cells = Vec::with_capacity(mesh.num_cells());
            for i in 0..mesh.num_cells() {
                let (cx, cy) = (mesh.cell_cx[i], mesh.cell_cy[i]);
                let mut corners: Vec<[f64; 2]> = Vec::new();
                let mut merge_tol2 = f64::INFINITY;
                for &f in &mesh.cell_faces[mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1]]
                {
                    let (tx, ty) = (-mesh.face_ny[f], mesh.face_nx[f]);
                    let h = 0.5 * mesh.face_area[f];
                    corners.push([mesh.face_cx[f] + tx * h, mesh.face_cy[f] + ty * h]);
                    corners.push([mesh.face_cx[f] - tx * h, mesh.face_cy[f] - ty * h]);
                    merge_tol2 = merge_tol2.min(mesh.face_area[f] * mesh.face_area[f]);
                }
                // Duplicate-merge tolerance: well below the shortest face, so
                // only true shared corners (f32-noise apart) collapse.
                let merge_tol2 = merge_tol2 * 1e-6;
                corners.sort_by(|a, b| {
                    let ta = (a[1] - cy).atan2(a[0] - cx);
                    let tb = (b[1] - cy).atan2(b[0] - cx);
                    ta.total_cmp(&tb)
                });
                let mut poly: Vec<[f64; 2]> = Vec::with_capacity(corners.len() / 2 + 1);
                for c in corners {
                    if let Some(last) = poly.last() {
                        let d2 = (c[0] - last[0]).powi(2) + (c[1] - last[1]).powi(2);
                        if d2 <= merge_tol2 {
                            continue;
                        }
                    }
                    poly.push(c);
                }
                if poly.len() > 1 {
                    let first = poly[0];
                    let last = *poly.last().unwrap();
                    let d2 = (first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2);
                    if d2 <= merge_tol2 {
                        poly.pop();
                    }
                }
                cells.push(poly);
            }
            return cells;
        }
        let mut cells = Vec::with_capacity(mesh.num_cells());
        for i in 0..mesh.num_cells() {
            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            let polygon_points: Vec<[f64; 2]> = (start..end)
                .map(|k| {
                    let v_idx = mesh.cell_vertices[k];
                    [mesh.vx[v_idx], mesh.vy[v_idx]]
                })
                .collect();
            cells.push(polygon_points);
        }
        cells
    }

    fn build_initial_velocity_with(
        mesh: &Mesh,
        selected_geometry: GeometryType,
        max_cell_size: f64,
        inlet_velocity: f32,
    ) -> Vec<(f64, f64)> {
        // Nozzle: develop FROM REST (zero velocity). No seeded freestream — the flow
        // accelerates purely from the rocket-scale inlet pressure drop. The large
        // pressure ratio (1 MPa gauge, see `ALLMACH_THERMAL_NOZZLE`) forces the throat
        // to choke, so the supersonic branch forms from scratch.
        if selected_geometry == GeometryType::Nozzle {
            let _ = inlet_velocity; // unused for the nozzle IC
            return vec![(0.0, 0.0); mesh.num_cells()];
        }
        let mut u = vec![(0.0, 0.0); mesh.num_cells()];
        for (i, _vel) in u.iter_mut().enumerate() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];

            if cx < max_cell_size {
                match selected_geometry {
                    GeometryType::BackwardsStep => {
                        if cy > 0.5 {
                            // Inlet ramp handled by shader
                        }
                    }
                    GeometryType::ChannelObstacle => {
                        // Inlet handled by shader
                    }
                    GeometryType::Nozzle => {
                        // Handled by the uniform-freestream early return above.
                    }
                }
            }
        }
        u
    }

    fn spawn_pending_init(&mut self) {
        if self.pending_init_request.is_none() {
            return;
        }

        // Cancel any in-flight init by dropping the receiver; the background thread will finish
        // but its result will be ignored.
        self.init_rx = None;
        self.init_in_flight = None;

        let request = match self.pending_init_request.take() {
            Some(r) => r,
            None => return, // should be unreachable due to guard above
        };
        self.init_in_flight = Some(request.generation);
        let (tx, rx) = mpsc::channel();
        self.init_rx = Some(rx);

        thread::spawn(move || {
            let generation = request.generation;
            let run = std::panic::AssertUnwindSafe(|| CFDApp::build_init_outcome(request));
            let result = std::panic::catch_unwind(run)
                .map_err(|_| "Solver init panicked".to_string())
                .and_then(|r| r);
            let _ = tx.send(SolverInitResponse { generation, result });
        });
    }

    fn poll_init(&mut self) {
        let Some(rx) = &self.init_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(resp) => {
                self.init_rx = None;
                let in_flight = self.init_in_flight;
                self.init_in_flight = None;
                if in_flight.is_some() && in_flight != Some(resp.generation) {
                    return;
                }
                match resp.result {
                    Ok(outcome) => {
                        self.apply_init_outcome(outcome);
                    }
                    Err(err) => {
                        self.mesh = None;
                        self.cached_cells.clear();
                        self.cfd_renderer = None;
                        self.viz_field = None;
                        self.viz_display = VizSlotToken {
                            index: 0,
                            sequence: 0,
                        };
                        self.cached_u.clear();
                        self.cached_p.clear();
                        self.snapshot_seq = 0;
                        self.invalidate_plot_cache();
                        self.cached_gpu_stats = CachedGpuStats::default();
                        self.cached_error = Some(err);
                    }
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.init_rx = None;
                self.init_in_flight = None;
                self.cached_error = Some("Solver init thread disconnected".to_string());
            }
        }
    }

    fn poll_solver_worker(&mut self) {
        while let Ok(evt) = self.solver_worker.rx.try_recv() {
            match evt {
                SolverWorkerEvent::Error(ref error_message) if std::env::var_os("CFD2_PERF_LOG").is_some() => {
                    eprintln!("[worker-error] {error_message}");
                    self.cached_error = Some(error_message.clone());
                }
                SolverWorkerEvent::Stats { stats } => {
                    if std::env::var_os("CFD2_PERF_LOG").is_some() {
                        eprintln!(
                            "[perf] t={:.4e} dt={:.3e} step={:.3} ms ({:.0} steps/s, {:.3e} sim s/s)",
                            stats.sim_time,
                            stats.dt,
                            stats.step_time_ms,
                            stats.steps_per_second,
                            stats.sim_seconds_per_wall_second
                        );
                    }
                    self.cached_gpu_stats = stats;
                    self.cached_error = None;
                }
                SolverWorkerEvent::Snapshot { u, p, stats } => {
                    self.cached_u = u;
                    self.cached_p = p;
                    self.snapshot_seq = self.snapshot_seq.wrapping_add(1);
                    self.invalidate_plot_cache();
                    self.cached_gpu_stats = stats;
                    self.cached_error = None;
                }
                SolverWorkerEvent::MeshRefreshed {
                    cached_cells,
                    stats,
                } => {
                    // The moving mesh re-generated: adopt the new polygons. The
                    // egui-plot fallback renders `cached_cells` directly; the
                    // GPU-direct renderer re-tessellates from them once per frame
                    // (below, gated on `pending_mesh_upload`), growing its vertex
                    // buffers if the new topology needs more room. Coalesced —
                    // only the latest cells per frame matter, so intermediate
                    // steps between frames are dropped.
                    self.cached_cells = cached_cells;
                    self.cached_moving_stats = Some(stats);
                    self.pending_mesh_upload = true;
                    self.snapshot_seq = self.snapshot_seq.wrapping_add(1);
                    self.invalidate_plot_cache();
                    self.cached_error = None;
                }
                SolverWorkerEvent::Message(message) => {
                    self.cached_message = Some(message);
                }
                SolverWorkerEvent::Error(err) => {
                    self.cached_error = Some(err);
                    self.cached_message = None;
                    self.is_running = false;
                }
                SolverWorkerEvent::Running {
                    running,
                    generation,
                } => {
                    if !apply_worker_running_event(
                        self.run_request_generation,
                        &mut self.is_running,
                        generation,
                        running,
                    ) {
                        continue;
                    }
                    if running {
                        // A resumed worker needs a fresh completion-to-completion
                        // sample. Do not leave the pre-pause rate visible while
                        // its first completion fence is still pending.
                        self.cached_gpu_stats.step_time_ms = 0.0;
                        self.cached_gpu_stats.steps_per_second = 0.0;
                        self.cached_gpu_stats.sim_seconds_per_wall_second = 0.0;
                    }
                }
            }
        }

        if let (Some(viz), Some(renderer), Some(device)) = (
            self.viz_field.as_ref(),
            self.cfd_renderer.as_ref(),
            self.wgpu_device.as_ref(),
        ) {
            // Mapping callbacks are driven opportunistically and never wait for
            // the GPU. The colormap itself reads the range buffer directly;
            // this poll serves only the tiny CPU legend labels.
            let _ = viz.range_reducer.poll_nonblocking();
            if let Some(next_display) = viz.mailbox.try_claim_latest_ready() {
                let mut renderer = renderer.lock().unwrap();
                if renderer.update_field_snapshot(
                    device,
                    &viz.buffers[next_display.index],
                    viz.range_reducer.range_buffer(next_display.index),
                    next_display.sequence,
                ) {
                    let previous_display = std::mem::replace(&mut self.viz_display, next_display);
                    let released = viz.mailbox.release_display(previous_display);
                    debug_assert!(
                        released,
                        "renderer released a stale visualization display token"
                    );
                } else {
                    // Defensive only: mailbox generations are monotonic. Keep
                    // the current display owned if a stale token ever arrives.
                    let released = viz.mailbox.release_display(next_display);
                    debug_assert!(released);
                }
            }

            if let Some(ranges) = viz
                .range_reducer
                .take_readback_for_sequence(self.viz_display.sequence)
            {
                let mut renderer = renderer.lock().unwrap();
                let updated = renderer.update_legend_ranges(ranges);
                debug_assert!(
                    updated,
                    "range readback sequence did not match the displayed snapshot"
                );
            }
        }

        // Re-tessellate the GPU-direct renderer once per frame if a moving-mesh
        // refresh arrived. Done here at the render-frame boundary (not per event)
        // so multiple solver steps between frames upload only the latest mesh.
        // `update_mesh` grows the vertex/line buffers if the new topology needs
        // more room, so this can never overflow. The cell count is invariant
        // (only geometry/topology moves), so the field bind path — which indexes
        // by fixed `cell_index` — is untouched.
        if self.pending_mesh_upload {
            self.pending_mesh_upload = false;
            if let (Some(renderer), Some(device), Some(queue)) = (
                self.cfd_renderer.as_ref(),
                self.wgpu_device.as_ref(),
                self.wgpu_queue.as_ref(),
            ) {
                if !self.cached_cells.is_empty() {
                    let vertices = cfd_renderer::build_mesh_vertices(&self.cached_cells);
                    let line_vertices = cfd_renderer::build_line_vertices(&self.cached_cells);
                    let mut renderer = renderer.lock().unwrap();
                    renderer.update_mesh(device, queue, &vertices, &line_vertices);
                }
            }
        }
    }

    fn apply_init_outcome(&mut self, outcome: SolverInitOutcome) {
        let SolverInitOutcome {
            mode,
            mesh,
            cached_cells,
            actual_min_cell_size,
            cached_u,
            cached_p,
            model_caps,
            renderer,
            viz_field,
            trace_init_events,
        } = outcome;

        self.is_running = false;
        self.set_worker_running(false);

        self.model_caps = model_caps;
        self.cached_cells = cached_cells;
        // In structured mode the immersed obstacle is a Brinkman mask, not a cut
        // cell, so cache which cells fall inside the selected geometry's solid so
        // the renderer can draw the obstacle as an overlay (the mesh is a full
        // rectangle). The mask uses the same ground-truth geometry objects as
        // the unstructured meshers.
        self.structured_solid_mask = if self.mesh_mode == MeshMode::Structured2D {
            let geom = self.selected_geometry;
            self.cached_cells
                .iter()
                .map(|poly| {
                    let n = poly.len().max(1) as f64;
                    let (cx, cy) = poly
                        .iter()
                        .fold((0.0, 0.0), |(ax, ay), p| (ax + p[0], ay + p[1]));
                    structured_geometry_is_solid(geom, cx / n, cy / n)
                })
                .collect()
        } else {
            Vec::new()
        };
        self.actual_min_cell_size = actual_min_cell_size;
        self.cached_u = cached_u;
        self.cached_p = cached_p;
        self.snapshot_seq = self.snapshot_seq.wrapping_add(1);
        self.invalidate_plot_cache();
        self.mesh = Some(mesh);

        self.cfd_renderer = renderer.map(|r| Arc::new(Mutex::new(r)));
        self.viz_field = viz_field.clone();
        self.viz_display = self.viz_field.as_ref().map_or(
            VizSlotToken {
                index: 0,
                sequence: 0,
            },
            |viz| viz.mailbox.initial_display(),
        );
        self.last_init_trace_events = trace_init_events;

        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_moving_stats = None;
        self.cached_error = None;
        self.cached_message = None;

        // Track what the worker is about to run so controls the moving driver
        // forbids stay disabled for the driver's whole life, not just while the
        // enable checkbox happens to be ticked.
        self.solver_is_moving = matches!(mode, SolverMode::MovingMesh(_));

        // SetSolver phase-2 applies the worker's current parameter snapshot.
        // Queue the new model's parameters first; otherwise a slow structured
        // all-Mach install briefly rewrites the freshly built solver with the
        // previous model's inlet/EOS settings before UpdateParams repairs it.
        self.sync_worker_params();
        self.solver_worker
            .send(SolverWorkerCommand::SetSolver { mode, viz_field });

        if self.trace_enabled {
            self.start_trace();
        } else {
            self.trace_pending_start = false;
        }
    }

    fn build_init_outcome(request: SolverInitRequest) -> Result<SolverInitOutcome, String> {
        let init_start = std::time::Instant::now();
        let mut trace_init_events: Vec<tracefmt::TraceInitEvent> = Vec::new();

        // Build the mesh + driver. The static path builds the selected mesh +
        // model and a plain `SolverDriver`; the moving-mesh path builds a CVT mesh
        // WITH its seeds and wraps a `MovingMeshDriver` around the incompressible
        // ALE model. Both yield the same downstream tuple, so the renderer / viz /
        // return tail below is shared.
        let (mut mode, mesh, cached_u, cached_p, mut model_caps) =
            if request.mesh_mode == MeshMode::Structured2D {
                CFDApp::build_structured_init(&request, &mut trace_init_events)?
            } else if request.enable_moving_mesh {
                CFDApp::build_moving_init(&request, &mut trace_init_events)?
            } else {
                CFDApp::build_static_init(&request, &mut trace_init_events)?
            };
        let n_cells = mesh.num_cells();

        // Update model_caps from the solver's ui_ports() (prefers PortRegistry over StateLayout)
        let ui_ports = mode.ui_ports();
        model_caps.plot_stride = ui_ports.stride;
        model_caps.plot_u_offset = ui_ports.u_offset.unwrap_or(0);
        model_caps.plot_p_offset = ui_ports.p_offset.unwrap_or(0);
        model_caps.plot_has_u = ui_ports.u_offset.is_some();
        model_caps.plot_has_p = ui_ports.p_offset.is_some();

        let cache_start = std::time::Instant::now();
        let cached_cells = CFDApp::cache_cells(&mesh);
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "mesh.cache_cells",
            cache_start.elapsed(),
            Some(format!("cells={}", mesh.num_cells())),
        );

        let renderer_start = std::time::Instant::now();
        let (renderer, viz_field) = if let (Some(device), Some(queue)) =
            (&request.wgpu_device, &request.wgpu_queue)
        {
            let state_size_bytes = mesh.num_cells() as u64 * model_caps.plot_stride as u64 * 4;
            // Moving (ALE) runs may GROW the cell count at runtime (the
            // flow-adaptive sizing births cells, hard-capped by the driver at
            // the requested budget factor × the initial count) — allocate the
            // per-cell viz buffers at that cap so a resized solver's state
            // still fits. Static runs allocate exactly their fixed size.
            let viz_capacity_bytes = if request.enable_moving_mesh {
                let factor = request
                    .moving_adapt_budget
                    .max(crate::sim::ADAPT_BUDGET_MAX_FACTOR);
                (state_size_bytes as f64 * factor).ceil() as u64
            } else {
                state_size_bytes
            };
            let viz_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CFD Viz Field Buffer 0"),
                size: viz_capacity_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let viz_buffer_1 = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CFD Viz Field Buffer 1"),
                size: viz_capacity_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let viz_buffer_2 = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CFD Viz Field Buffer 2"),
                size: viz_capacity_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let viz_buffers = [viz_buffer, viz_buffer_1, viz_buffer_2];
            mode.install_viz_frame_targets(&viz_buffers)?;
            if state_size_bytes > 0 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cfd_viz:init_copy_state"),
                });
                encoder.copy_buffer_to_buffer(
                    mode.state_buffer(),
                    0,
                    &viz_buffers[0],
                    0,
                    state_size_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    mode.state_buffer(),
                    0,
                    &viz_buffers[1],
                    0,
                    state_size_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    mode.state_buffer(),
                    0,
                    &viz_buffers[2],
                    0,
                    state_size_bytes,
                );
                queue.submit(Some(encoder.finish()));
            }

            let range_reducer = cfd_renderer::CfdRangeReducer::new(
                device,
                queue,
                &viz_buffers,
                cfd_renderer::CfdFieldLayout {
                    stride: model_caps.plot_stride,
                    u_offset: model_caps.plot_u_offset,
                    p_offset: model_caps.plot_p_offset,
                    has_u: model_caps.plot_has_u,
                    has_p: model_caps.plot_has_p,
                },
            );
            // Initialize the range metadata for the already-bound slot zero.
            // The state copy above, this reduction, and the first render are
            // ordered on the same queue; no host wait is needed.
            range_reducer.submit(0, 0, n_cells);

            // Size the render buffers from the actual triangulated data: a
            // per-cell heuristic undersizes polygonal (Voronoi) meshes, whose
            // fan triangulation needs 3*(n-2) and wireframe 2*n vertices per
            // n-gon (~12+ for the typical hexagon).
            let vertices = cfd_renderer::build_mesh_vertices(&cached_cells);
            let line_vertices = cfd_renderer::build_line_vertices(&cached_cells);
            let max_vertices = vertices.len().max(line_vertices.len()).max(1);
            // Only the moving (ALE) mesh re-tessellates per step, so only it
            // needs pre-grown buffers; the static path allocates exactly its
            // one-time count (no 50% over-allocation for never-moving runs).
            let headroom = if request.enable_moving_mesh {
                cfd_renderer::VERTEX_HEADROOM
            } else {
                cfd_renderer::NO_HEADROOM
            };
            let mut renderer = cfd_renderer::CfdRenderResources::new(
                device,
                request.target_format,
                max_vertices,
                headroom,
            );
            renderer.update_mesh(device, queue, &vertices, &line_vertices);
            renderer.bind_initial_snapshot(device, &viz_buffers[0], range_reducer.range_buffer(0));
            (
                Some(renderer),
                Some(VizFieldBuffers {
                    buffers: viz_buffers,
                    size_bytes: state_size_bytes,
                    range_reducer,
                    mailbox: Arc::new(VizFrameMailbox::new()),
                }),
            )
        } else {
            (None, None)
        };
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "renderer.init",
            renderer_start.elapsed(),
            Some(if renderer.is_some() {
                format!("mesh_vertices_cap={}", mesh.num_cells() * 10)
            } else {
                "skipped".to_string()
            }),
        );

        let min_cell_start = std::time::Instant::now();
        let actual_min_cell_size = mesh
            .cell_vol
            .iter()
            .map(|&v| v.sqrt())
            .fold(f64::INFINITY, f64::min);
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "mesh.actual_min_cell_size",
            min_cell_start.elapsed(),
            Some(format!("{actual_min_cell_size:.4e}")),
        );

        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "init.total",
            init_start.elapsed(),
            Some(format!("cells={n_cells}")),
        );

        Ok(SolverInitOutcome {
            mode,
            mesh,
            cached_cells,
            actual_min_cell_size,
            cached_u,
            cached_p,
            model_caps,
            renderer,
            viz_field,
            trace_init_events,
        })
    }

    /// Initial (layout-derived) UI capabilities for a model. Refined from the
    /// built solver's `ui_ports()` afterwards. Shared by the static and
    /// moving-mesh init paths so both report caps identically.
    fn model_ui_caps(model: &ModelSpec) -> ModelUiCaps {
        let named_params = model.named_param_keys();
        let ui_ports_fallback = UiPortSet::from_layout(&model.state_layout);
        ModelUiCaps {
            plot_stride: ui_ports_fallback.stride,
            plot_u_offset: ui_ports_fallback.u_offset.unwrap_or(0),
            plot_p_offset: ui_ports_fallback.p_offset.unwrap_or(0),
            plot_has_u: ui_ports_fallback.u_offset.is_some(),
            plot_has_p: ui_ports_fallback.p_offset.is_some(),
            supports_preconditioner: named_params.iter().any(|&k| k == "preconditioner"),
            model_owns_preconditioner: model
                .linear_solver
                .map(|s| matches!(s.preconditioner, ModelPreconditionerSpec::Schur { .. }))
                .unwrap_or(false),
            unknowns_per_cell: model.system.unknowns_per_cell(),
            supports_outer_iters: named_params.iter().any(|&k| k == "outer_iters"),
            supports_dtau: named_params.iter().any(|&k| k == "dtau"),
            supports_low_mach: named_params.iter().any(|&k| k == "low_mach.model"),
            supports_alpha_u: named_params.iter().any(|&k| k == "alpha_u"),
            supports_alpha_p: named_params.iter().any(|&k| k == "alpha_p"),
            supports_eos_tuning: named_params.iter().any(|&k| k == "eos.gamma"),
            supports_explicit_rk4: model.validate_explicit_rk4().is_ok(),
        }
    }

    /// Static (non-moving) init path: build the selected mesh + model and a plain
    /// `SolverDriver`.
    fn build_static_init(
        request: &SolverInitRequest,
        trace_init_events: &mut Vec<tracefmt::TraceInitEvent>,
    ) -> Result<(SolverMode, Mesh, Vec<(f64, f64)>, Vec<f64>, ModelUiCaps), String> {
        let mesh = CFDApp::build_mesh_with(
            request.selected_geometry,
            request.mesh_type,
            request.min_cell_size,
            request.max_cell_size,
            request.growth_rate,
            trace_init_events,
        );

        let fields_start = std::time::Instant::now();
        let n_cells = mesh.num_cells();
        let initial_u = CFDApp::build_initial_velocity_with(
            &mesh,
            request.selected_geometry,
            request.max_cell_size,
            request.params.inlet_velocity,
        );
        let initial_p = vec![0.0; n_cells];
        CFDApp::push_trace_init_event(
            trace_init_events,
            "init.fields",
            fields_start.elapsed(),
            Some(format!("cells={n_cells}")),
        );

        let model = unstructured_model_by_id(request.model_id, request.current_fluid.eos)?;
        let model_caps = CFDApp::model_ui_caps(&model);

        // The shared driver derives the `SolverConfig` (stepping mode + effective
        // preconditioner), constructs the solver, and applies the phase-1 setters +
        // initial / boundary conditions. Phase-2 knobs arrive via `sync_worker_params`
        // (→ `apply_params`) after `SetSolver`.
        let solver_start = std::time::Instant::now();
        let init_guard = tracefmt::install_init_collector(trace_init_events);
        let DriverBuild {
            driver,
            cached_u,
            cached_p,
        } = pollster::block_on(SolverDriver::build(
            &mesh,
            model,
            &request.params,
            &initial_u,
            &initial_p,
            request.wgpu_device.clone(),
            request.wgpu_queue.clone(),
        ))?;
        drop(init_guard);
        CFDApp::push_trace_init_event(
            trace_init_events,
            "solver.new",
            solver_start.elapsed(),
            Some(format!("model_id={} cells={}", request.model_id, n_cells)),
        );

        Ok((
            SolverMode::Static(driver),
            mesh,
            cached_u,
            cached_p,
            model_caps,
        ))
    }

    /// Structured (dense-Cartesian `TopologyMode::Structured2D`) init path: build a
    /// uniform grid + a companion render `Mesh` (row-major `p=j*nx+i`, so cell ids
    /// line up 1:1 with the packed state) and a standalone [`StructuredGpuSolver`]
    /// on the GUI's device. The scenario is a lid-driven cavity (momentum /
    /// all-Mach thermal) or a uniform gas box (compressible) — the proven
    /// structured demos. Obstacles are immersed (Brinkman), never cut.
    fn build_structured_init(
        request: &SolverInitRequest,
        trace_init_events: &mut Vec<tracefmt::TraceInitEvent>,
    ) -> Result<(SolverMode, Mesh, Vec<(f64, f64)>, Vec<f64>, ModelUiCaps), String> {
        let device = request
            .wgpu_device
            .clone()
            .ok_or("structured mode requires a GPU device")?;
        let queue = request
            .wgpu_queue
            .clone()
            .ok_or("structured mode requires a GPU queue")?;

        // Every structured flow model runs a CHANNEL (inlet left, outlet right,
        // no-slip walls) whose immersed obstacle — a Brinkman `ibm_penalty_U`
        // mask, never a cut cell — rasterizes the SAME ground-truth geometry the
        // unstructured meshers cut (identical domain and shape). This includes
        // the density-based compressible model, which drives the channel with a
        // uniform-freestream momentum IC + conserved-Dirichlet inlet (its
        // bc_expr closure keeps the dependent entries thermodynamically
        // consistent).
        let (lx, ly) = gui_geometry_domain(request.selected_geometry);
        // Resolution tracks the user's max-cell-size slider with no artificial
        // DOF cap (the old clamp(…, 192)×clamp(…, 96) floor at 18 432 cells was a
        // GUI convenience limit, not a solver bound). Only reject non-positive /
        // non-finite sizes and keep at least one cell per axis.
        let cell = if request.max_cell_size.is_finite() && request.max_cell_size > 0.0 {
            request.max_cell_size
        } else {
            0.025
        };
        let nx = ((lx / cell).round() as usize).max(1);
        let ny = ((ly / cell).round() as usize).max(1);

        let mesh_start = std::time::Instant::now();
        let sides = BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        };
        let mesh = generate_structured_rect_mesh(nx, ny, lx, ly, sides);
        CFDApp::push_trace_init_event(
            trace_init_events,
            "mesh.generate.structured",
            mesh_start.elapsed(),
            Some(format!("{nx}x{ny}")),
        );

        let grid = StructuredGrid::new(nx, ny, lx, ly);
        let model = structured_model_by_id(request.model_id)?;
        let model_caps = CFDApp::model_ui_caps(&model);
        let s = model.system.unknowns_per_cell() as usize;
        let dt = request.params.requested_dt.max(1.0e-6) as f64;
        let outer = request.params.outer_iters.max(1) as usize;

        let u_in = request.params.inlet_velocity as f64;
        let (density, viscosity) = (
            request.params.density as f64,
            request.params.viscosity as f64,
        );
        let (scheme, time_scheme) = (request.params.advection_scheme, request.params.time_scheme);
        let precond = request.structured_precond;
        let solver_start = std::time::Instant::now();

        // The CPU backends run the SAME Structured2D codegen IR the GPU lowers,
        // bridged to the GPU-direct renderer by uploading the packed state to a
        // viz buffer. The interpreter is the correctness oracle; the transpiled
        // engine runs the compiled-Rust structured kernels (generated::
        // lookup_structured, per-kernel interpreter fallback). The seeding + BCs
        // are backend-generic (StructuredSeed).
        let (mode, cached_u, cached_p) = if request.backend.is_cpu() {
            let mut cpu =
                StructuredModelSolver::with_config(grid, &model, dt, outer, scheme, time_scheme)?;
            let engine = if request.backend == BackendChoice::CpuInterpreter {
                crate::solver::cpu::CpuEngine::Interpreter
            } else {
                // CpuTranspiled / CpuTranspiledSimd — the SIMD variant only affects
                // the unstructured linear solve; the structured banded solve is the
                // shared host routine, so both map to the transpiled kernel engine.
                crate::solver::cpu::CpuEngine::Transpiled
            };
            let threads = std::env::var("CFD2_CPU_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1)
                .max(1);
            cpu.set_engine(engine, threads);
            cpu.set_fluid(density, viscosity);
            cpu.set_preconditioner(precond);
            cpu.set_outer_iters(outer);
            cpu.set_outer_auto_converge(request.params.outer_auto_converge);
            cpu.set_alpha_u(request.params.alpha_u);
            cpu.set_alpha_p(request.params.alpha_p);
            if request.model_id == "compressible_structured" {
                apply_structured_compressible_runtime(&mut cpu, &request.params);
            }
            seed_structured_state(
                &mut cpu,
                request.model_id,
                &request.params,
                lx / nx as f64,
                ly / ny as f64,
            );
            seed_structured_freestream(&mut cpu, request.model_id, u_in, &request.params);
            seed_structured_ibm(&mut cpu, request.selected_geometry);
            setup_structured_bcs(&mut cpu, request.model_id, u_in, s, &request.params);
            let ports = UiPortSet::from_layout(cpu.state_layout());
            let cached_u = ports
                .u_offset
                .map(|o| cpu.get_u(o as usize))
                .unwrap_or_default();
            let cached_p = ports
                .p_offset
                .map(|o| cpu.get_scalar(o as usize))
                .unwrap_or_default();
            let bridge = StructuredCpuBridge::new(cpu, &device, queue);
            (SolverMode::StructuredCpu(bridge), cached_u, cached_p)
        } else {
            let ctx = pollster::block_on(crate::solver::gpu::context::GpuContext::new(
                Some(device),
                Some(queue),
            ))?;
            // Honour the selected advection + time-integration schemes (structured
            // kernels read them at runtime — full parity with the unstructured path).
            let mut solver = StructuredGpuSolver::with_config(
                ctx,
                grid,
                &model,
                dt,
                outer,
                scheme,
                time_scheme,
            )?;
            solver.set_fluid(density, viscosity);
            // Coupled-solve preconditioner: the model-owned SIMPLE Schur (if the
            // model declares a layout) or block-Jacobi. No-op where unsupported.
            solver.set_preconditioner(precond);
            solver.set_outer_iters(outer);
            solver.set_outer_auto_converge(request.params.outer_auto_converge);
            solver.set_alpha_u(request.params.alpha_u);
            solver.set_alpha_p(request.params.alpha_p);
            if request.model_id == "compressible_structured" {
                apply_structured_compressible_runtime(&mut solver, &request.params);
            }
            seed_structured_state(
                &mut solver,
                request.model_id,
                &request.params,
                lx / nx as f64,
                ly / ny as f64,
            );
            // Uniform-freestream momentum IC (compressible: rho_u = rho*u_in) so the
            // channel starts from a moving state — the stable configuration; a rest
            // IC on the fine grid can stall/diverge the density-based model.
            seed_structured_freestream(&mut solver, request.model_id, u_in, &request.params);
            // Immersed obstacle from the selected geometry (Brinkman mask; all three
            // structured models now declare an `ibm_penalty_U` field).
            seed_structured_ibm(&mut solver, request.selected_geometry);
            setup_structured_bcs(&mut solver, request.model_id, u_in, s, &request.params);
            let ports = UiPortSet::from_layout(solver.state_layout());
            let cached_u = ports
                .u_offset
                .map(|o| solver.get_u(o as usize))
                .unwrap_or_default();
            let cached_p = ports
                .p_offset
                .map(|o| solver.get_scalar(o as usize))
                .unwrap_or_default();
            (SolverMode::Structured(solver), cached_u, cached_p)
        };
        CFDApp::push_trace_init_event(
            trace_init_events,
            "solver.new",
            solver_start.elapsed(),
            Some(format!("model_id={} cells={}", request.model_id, nx * ny)),
        );

        Ok((mode, mesh, cached_u, cached_p, model_caps))
    }

    /// Moving-mesh (ALE) init path: build a CVT mesh WITH its authoritative seeds
    /// and wrap a [`MovingMeshDriver`] around the incompressible ALE model
    /// The mesh type is forced to CVT-Voronoi and the model to
    /// `incompressible_momentum_ale`; `adaptive_dt` is forced off (the swept mesh
    /// fluxes are SCL-closed against a fixed dt — the driver rejects an adaptive
    /// re-scale). `cached_u`/`cached_p` are the seeded IC (the driver's initial
    /// state), used for the initial plot before the first readback.
    fn build_moving_init(
        request: &SolverInitRequest,
        trace_init_events: &mut Vec<tracefmt::TraceInitEvent>,
    ) -> Result<(SolverMode, Mesh, Vec<(f64, f64)>, Vec<f64>, ModelUiCaps), String> {
        let cvt = CFDApp::build_cvt_seeds_with(
            request.selected_geometry,
            request.min_cell_size,
            request.max_cell_size,
            request.growth_rate,
            trace_init_events,
        );
        // Keep a copy of the realized mesh for the outcome; `cvt` is consumed by
        // `MovingMeshDriver::build`.
        let mesh = cvt.mesh.clone();

        let fields_start = std::time::Instant::now();
        let n_cells = mesh.num_cells();
        let initial_u = CFDApp::build_initial_velocity_with(
            &mesh,
            request.selected_geometry,
            request.max_cell_size,
            request.params.inlet_velocity,
        );
        let initial_p = vec![0.0; n_cells];
        CFDApp::push_trace_init_event(
            trace_init_events,
            "init.fields",
            fields_start.elapsed(),
            Some(format!("cells={n_cells} (moving)")),
        );

        // Map the user's selected model to its ALE (moving-mesh) variant — same
        // physics family, mesh-relative convection. The base model is never
        // silently swapped; models without an ALE variant are gated out of the
        // toggle in the UI, so reaching here with an unsupported model is a
        // programming error (surfaced as an init Err, not a wrong-physics run).
        let ale_id = CFDApp::ale_model_for(request.model_id).ok_or_else(|| {
            format!(
                "model '{}' has no moving-mesh (ALE) variant",
                request.model_id
            )
        })?;
        let model = ale_model_by_id(ale_id)?;
        let model_caps = CFDApp::model_ui_caps(&model);

        // The SOLVER-side adaptive dt stays off for the ALE seam (its
        // re-scale happens after the swept fluxes are closed — a GCL
        // violation); the user's Adaptive Timestep choice routes to the
        // DRIVER-side controller below, which pins the CFL-derived dt at the
        // head of the handshake instead.
        let adaptive_dt = request.params.adaptive_dt;
        let mut params = request.params;
        params.adaptive_dt = false;
        let motion = request.moving_motion.to_spec(request.moving_regularization);

        let solver_start = std::time::Instant::now();
        let init_guard = tracefmt::install_init_collector(trace_init_events);
        // `build_with_model` seeds any model-specific state (psi/psi_precond/rho/
        // dt_local≡0 for the all-Mach models) via `SolverDriver::build`, and
        // applies the nozzle pressure-inlet BCs when `params.pressure_inlet`.
        let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
            cvt,
            model,
            &params,
            motion,
            &initial_u,
            &initial_p,
            request.wgpu_device.clone(),
            request.wgpu_queue.clone(),
        ))?;
        drop(init_guard);

        // An oscillating obstacle (ChannelObstacle only — the obstacle is loop 1
        // of its boundary spec). Cross-stream sinusoidal rigid motion; the
        // MovingWall BC feeds the wall's material velocity into the fluid. A
        // no-op for any other geometry (no obstacle loop) or when off — so the
        // FlowCoupled / Frozen / swirl interior-motion demos are unchanged.
        if request.moving_oscillate_obstacle
            && request.selected_geometry == GeometryType::ChannelObstacle
        {
            const OBSTACLE_LOOP: usize = 1;
            moving.set_boundary_motion(BoundaryMotionSpec::Oscillation {
                loop_index: OBSTACLE_LOOP,
                amplitude: request.moving_osc_amplitude,
                omega: std::f64::consts::TAU * request.moving_osc_frequency,
                axis: OscAxis::CrossStream,
            });
            moving.set_moving_wall_bc(true);
        }
        // On-device reconstruction (GPU backend only — the request pre-gates
        // it, and the driver additionally checks the solver backend). Steps
        // the device cannot certify (flip/sliver) re-run on the CPU for that
        // step; the per-step `regen_backend` stat surfaces the live path.
        moving.set_gpu_regen(request.moving_gpu_regen);
        // Scheduled Lloyd smoothing (0 = off; one gentle sweep at ω = 0.5).
        moving.set_smoothing(request.moving_smooth_every_n, 1, 0.5);
        // Periodic Morton memory reordering (0 = off) — a pure relabel that
        // restores cache locality eroded by recycling slot migration.
        moving.set_reorder_every_n(request.moving_reorder_every_n);
        // Flow-adaptive sizing (0 = off): birth/kill cells toward a target
        // volume derived from the flow's velocity/pressure/strain gradients.
        // The viz buffers are allocated with the driver's 2× budget headroom.
        moving.set_adaptive_sizing(request.moving_adapt_every_n);
        // Explicit adaptation band (both sliders > 0); otherwise the driver
        // defaults to the built mesh's realized sizing band.
        if request.moving_adapt_min_size > 0.0 && request.moving_adapt_max_size > 0.0 {
            moving.set_adaptive_sizing_band(Some((
                request.moving_adapt_min_size,
                request.moving_adapt_max_size,
            )));
        }
        // Growth budget (the viz buffers were allocated from the same factor).
        moving.set_adaptive_budget_factor(request.moving_adapt_budget);
        // Per-indicator threshold multipliers (1 = auto, 0 = disabled).
        moving.set_adaptive_indicator_thresholds(
            request.moving_adapt_thresh_u,
            request.moving_adapt_thresh_p,
            request.moving_adapt_thresh_strain,
        );
        // Implicit mesh motion (1 = explicit; FlowCoupled + CPU only —
        // inert elsewhere). Tolerance: 2% of the near-wall cell spacing.
        moving.set_implicit_mesh_motion(request.moving_motion_iters, 0.02);
        // Flow-adaptive dt (GCL-safe: pinned per step INSIDE the handshake).
        moving.set_adaptive_dt(adaptive_dt.then_some(params.target_cfl));
        CFDApp::push_trace_init_event(
            trace_init_events,
            "solver.new",
            solver_start.elapsed(),
            Some(format!(
                "model_id={ale_id} motion={} cells={}",
                request.moving_motion.label(),
                n_cells
            )),
        );

        Ok((
            SolverMode::MovingMesh(moving),
            mesh,
            initial_u,
            initial_p,
            model_caps,
        ))
    }

    /// Build a CVT-Voronoi mesh together with its authoritative seed set
    /// ([`CvtMeshSeeds`]) for the selected geometry — the moving-mesh driver owns
    /// the seeds so it can advect + regenerate deterministically. Mirrors the
    /// geometry construction + size sanitisation of [`build_mesh_with`] (same
    /// `MIN_CELL_SIZE` floor so a hand-typed size cannot OOM the base grid), but
    /// returns the seeds instead of discarding them.
    fn build_cvt_seeds_with(
        selected_geometry: GeometryType,
        min_cell_size: f64,
        max_cell_size: f64,
        growth_rate: f64,
        trace_init_events: &mut Vec<tracefmt::TraceInitEvent>,
    ) -> CvtMeshSeeds {
        const MIN_CELL_SIZE: f64 = 1e-3;
        let max_cell_size = if max_cell_size.is_finite() {
            max_cell_size.max(MIN_CELL_SIZE)
        } else {
            0.05
        };
        let min_cell_size = if min_cell_size.is_finite() {
            min_cell_size.clamp(MIN_CELL_SIZE, max_cell_size)
        } else {
            max_cell_size
        };
        let growth_rate = if growth_rate.is_finite() {
            growth_rate.max(1.0)
        } else {
            1.2
        };
        let lloyd = LloydConfig::default();
        let gen_start = std::time::Instant::now();
        let cvt = match selected_geometry {
            GeometryType::BackwardsStep => {
                let domain = Vector2::new(3.5, 1.0);
                let geo = BackwardsStep {
                    length: 3.5,
                    height_inlet: 0.5,
                    height_outlet: 1.0,
                    step_x: 0.5,
                };
                generate_cvt_mesh_with_seeds(
                    &geo,
                    min_cell_size,
                    max_cell_size,
                    growth_rate,
                    domain,
                    &lloyd,
                )
            }
            GeometryType::ChannelObstacle => {
                let domain = Vector2::new(3.0, 1.0);
                let geo = ChannelWithObstacle {
                    length: 3.0,
                    height: 1.0,
                    obstacle_center: Point2::new(1.0, 0.51),
                    obstacle_radius: 0.1,
                };
                generate_cvt_mesh_with_seeds(
                    &geo,
                    min_cell_size,
                    max_cell_size,
                    growth_rate,
                    domain,
                    &lloyd,
                )
            }
            GeometryType::Nozzle => {
                let domain = Vector2::new(3.0, 1.0);
                let geo = Nozzle {
                    length: 3.0,
                    height: 1.0,
                    throat_height: 0.40,
                    throat_frac: 0.40,
                    exit_height: 0.80,
                };
                generate_cvt_mesh_with_seeds(
                    &geo,
                    min_cell_size,
                    max_cell_size,
                    growth_rate,
                    domain,
                    &lloyd,
                )
            }
        };
        CFDApp::push_trace_init_event(
            trace_init_events,
            "mesh.generate.voronoi_cvt.moving",
            gen_start.elapsed(),
            Some(format!(
                "min={min_cell_size:.4e} max={max_cell_size:.4e} growth={growth_rate:.3} cells={} faces={} seeds={}",
                cvt.mesh.num_cells(),
                cvt.mesh.num_faces(),
                cvt.seeds.len()
            )),
        );
        cvt
    }

    fn invalidate_plot_cache(&mut self) {
        self.plot_cache = None;
    }

    fn ensure_plot_cache(&mut self, want_values: bool) {
        let snapshot_seq = self.snapshot_seq;
        let field = self.plot_field;

        let cache_ok = self.plot_cache.as_ref().map_or(false, |cache| {
            cache.snapshot_seq == snapshot_seq
                && cache.field == field
                && (!want_values || cache.values.is_some())
        });
        if cache_ok {
            return;
        }

        let (len, has_data) = match field {
            PlotField::Pressure => (self.cached_p.len(), !self.cached_p.is_empty()),
            PlotField::VelocityX | PlotField::VelocityY | PlotField::VelocityMag => {
                (self.cached_u.len(), !self.cached_u.is_empty())
            }
        };

        if !has_data || len == 0 {
            self.plot_cache = Some(PlotCache {
                snapshot_seq,
                field,
                min: 0.0,
                max: 1.0,
                values: None,
            });
            return;
        }

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut values = want_values.then(|| Vec::with_capacity(len));

        for i in 0..len {
            let val = match field {
                PlotField::Pressure => self.cached_p[i],
                PlotField::VelocityX => self.cached_u[i].0,
                PlotField::VelocityY => self.cached_u[i].1,
                PlotField::VelocityMag => {
                    let (vx, vy) = self.cached_u[i];
                    (vx * vx + vy * vy).sqrt()
                }
            };
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            if let Some(values) = values.as_mut() {
                values.push(val);
            }
        }

        if !(min_val.is_finite() && max_val.is_finite()) || min_val > max_val {
            min_val = 0.0;
            max_val = 1.0;
            values = None;
        } else if (max_val - min_val).abs() < 1e-12 {
            max_val = min_val + 1.0;
        } else {
            // Representable-precision floor, matching the Direct route (see
            // `sanitize_range` in cfd_renderer.rs): the plotted values come
            // from an f32 state, so never normalize the colormap to a span
            // below ~64 f32 quanta of the field magnitude — that renders
            // representation noise as full-scale speckle.
            let floor_span = f64::from(cfd_renderer::PRECISION_FLOOR_ULPS)
                * f64::from(f32::EPSILON)
                * min_val.abs().max(max_val.abs());
            if max_val - min_val < floor_span {
                let mid = 0.5 * (min_val + max_val);
                min_val = mid - 0.5 * floor_span;
                max_val = mid + 0.5 * floor_span;
            }
        }

        self.plot_cache = Some(PlotCache {
            snapshot_seq,
            field,
            min: min_val,
            max: max_val,
            values,
        });
    }

    fn render_layout_for_field(&self) -> (u32, u32, u32) {
        let stride = self.model_caps.plot_stride;
        let u_offset = self.model_caps.plot_u_offset;
        let p_offset = self.model_caps.plot_p_offset;

        match self.plot_field {
            PlotField::Pressure => (stride, p_offset, 0),
            PlotField::VelocityX => (stride, u_offset, 0),
            PlotField::VelocityY => (stride, u_offset + 1, 0),
            PlotField::VelocityMag => (stride, u_offset, 1),
        }
    }

    fn render_range_field(&self) -> cfd_renderer::CfdRangeField {
        match self.plot_field {
            PlotField::Pressure => cfd_renderer::CfdRangeField::Pressure,
            PlotField::VelocityX => cfd_renderer::CfdRangeField::VelocityX,
            PlotField::VelocityY => cfd_renderer::CfdRangeField::VelocityY,
            PlotField::VelocityMag => cfd_renderer::CfdRangeField::VelocityMagnitude,
        }
    }

    fn update_gpu_fluid(&mut self) {
        // The fluid's EOS sets the sound speed, so the REAL compressibility
        // `psi = 1/c^2` must be recomputed when the fluid (preset, density, or
        // viscosity) changes. No exaggeration.
        self.compressibility_psi = self.current_fluid.compressibility() as f32;
        self.sync_worker_params();
    }

    fn update_gpu_dt(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_dtau(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_scheme(&mut self) {
        // The structured solver bakes the scheme into its recipe at construction —
        // a change rebuilds; the unstructured solver applies it live.
        if self.mesh_mode == MeshMode::Structured2D {
            self.init_solver();
        } else {
            self.sync_worker_params();
        }
    }

    fn update_gpu_alpha_u(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_alpha_p(&self) {
        self.sync_worker_params();
    }

    /// Recommended structured compute backend for the CURRENT time scheme.
    ///
    /// The implicit coupled saddle solve is host-side either way (the GPU
    /// path reads the banded matrix back every outer iteration), so the
    /// CPU-transpiled backend wins there. The explicit RK4 program is fully
    /// GPU-resident with no per-step readbacks: the structured GPU runs it at
    /// ~0.4 ms/step at 120x40 versus ~6 ms/step for the single-threaded CPU
    /// bridge — leaving the coupled-era CPU default in place silently made
    /// the structured RK4 case slower than the unstructured GPU one. The
    /// backend radio remains manually selectable after this default applies.
    fn structured_backend_default(&self) -> BackendChoice {
        if self.time_scheme == GpuTimeScheme::RK4 {
            BackendChoice::Gpu
        } else {
            BackendChoice::CpuTranspiled
        }
    }

    fn update_gpu_time_scheme(&mut self) {
        // RK4 selects a different, matrix-free solver program; rebuilding on
        // every time-scheme change keeps transitions to/from the implicit
        // Euler/BDF2 paths unambiguous for both mesh families.
        //
        // The structured backend default is scheme-aware (see
        // structured_backend_default): re-apply it so picking RK4 lands on
        // the GPU program and picking an implicit scheme returns to the
        // host-side coupled solver.
        if self.mesh_mode == MeshMode::Structured2D {
            self.backend = self.structured_backend_default();
        }
        self.init_solver();
    }

    fn update_gpu_outer_iters(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_low_mach(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_inlet_velocity(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_preconditioner(&self) {
        self.sync_worker_params();
    }

    /// Render the right panel showing mesh stats and color legend.
    fn render_right_panel(
        &self,
        root_ui: &mut egui::Ui,
        has_solver: bool,
        min_val: f32,
        max_val: f32,
    ) {
        egui::Panel::right("legend").show(root_ui, |ui| {
            if let Some(mesh) = &self.mesh {
                ui.heading("Mesh Stats");
                // A moving (ALE) run regenerates the mesh every step — and
                // with adaptive sizing even the CELL COUNT changes — so the
                // init-time `self.mesh` snapshot goes stale immediately.
                // Prefer the live per-step telemetry when the running solver
                // is a moving one.
                let live = if self.solver_is_moving {
                    self.cached_moving_stats.as_ref()
                } else {
                    None
                };
                if let Some(m) = live {
                    ui.label(format!("Cells: {}", m.n_cells));
                    ui.label(format!("Faces: {}", m.n_faces));
                    // The on-device regen builds a vertex-less mesh (0).
                    if m.n_vertices > 0 {
                        ui.label(format!("Vertices: {}", m.n_vertices));
                    }
                    if m.vol_min.is_finite() {
                        ui.label(format!("Cell vol: {:.2e} - {:.2e}", m.vol_min, m.vol_max));
                    }
                } else {
                    ui.label(format!("Cells: {}", mesh.num_cells()));
                    ui.label(format!("Faces: {}", mesh.num_faces()));
                    ui.label(format!("Vertices: {}", mesh.num_vertices()));
                    if !mesh.cell_vol.is_empty() {
                        let min_vol = mesh.cell_vol.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_vol = mesh
                            .cell_vol
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        ui.label(format!("Cell vol: {:.2e} - {:.2e}", min_vol, max_vol));
                    }
                }
                ui.separator();
            }

            if has_solver {
                ui.heading("Legend");
                ui.label(format!("Max: {:.4}", max_val));

                let (rect, _response) =
                    ui.allocate_at_least(egui::vec2(30.0, 200.0), egui::Sense::hover());
                if ui.is_rect_visible(rect) {
                    let mut mesh = egui::Mesh::default();
                    let n_steps = 20;
                    for i in 0..n_steps {
                        let t0 = i as f32 / n_steps as f32;
                        let y0 = rect.max.y - t0 * rect.height();
                        let y1 = rect.max.y - (i as f32 + 1.0) / n_steps as f32 * rect.height();
                        let c0 = get_color(t0 as f64);

                        mesh.add_colored_rect(
                            egui::Rect::from_min_max(
                                egui::pos2(rect.min.x, y1),
                                egui::pos2(rect.max.x, y0),
                            ),
                            c0,
                        );
                    }
                    ui.painter().add(mesh);
                }

                ui.label(format!("Min: {:.4}", min_val));
            }
        });
    }

    /// Render the bottom panel with plot field selection and render mode.
    fn render_bottom_panel(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::bottom("plot_controls").show(root_ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Plot Field:");
                let old_field = self.plot_field;
                ui.radio_value(&mut self.plot_field, PlotField::Pressure, "Pressure");
                ui.radio_value(&mut self.plot_field, PlotField::VelocityX, "Velocity X");
                ui.radio_value(&mut self.plot_field, PlotField::VelocityY, "Velocity Y");
                ui.radio_value(&mut self.plot_field, PlotField::VelocityMag, "Velocity Mag");
                if old_field != self.plot_field {
                    self.update_renderer_field();
                    self.invalidate_plot_cache();
                }

                ui.separator();
                ui.checkbox(&mut self.show_mesh_lines, "Show Mesh Lines");

                ui.separator();
                ui.label("Render Mode:");
                let old_render_mode = self.render_mode;
                ui.radio_value(&mut self.render_mode, RenderMode::GpuDirect, "Direct");
                ui.radio_value(&mut self.render_mode, RenderMode::EguiPlot, "Plot (Slow)");
                if old_render_mode != self.render_mode {
                    self.invalidate_plot_cache();
                }
            });
        });
    }

    /// Render the central panel with the CFD visualization.
    fn render_central_panel(
        &self,
        root_ui: &mut egui::Ui,
        is_initializing: bool,
        has_solver: bool,
        min_val: f32,
        max_val: f32,
    ) {
        let ctx = &root_ui.ctx().clone();
        egui::CentralPanel::default().show(root_ui, |ui| {
            if self.is_running {
                // GPU solver runs in background thread
            }

            let cells = if has_solver && !self.cached_cells.is_empty() {
                Some(self.cached_cells.as_slice())
            } else {
                None
            };

            if let Some(cells) = cells {
                match self.render_mode {
                    RenderMode::GpuDirect => {
                        if let Some(viz) = &self.viz_field {
                            if activate_direct_viz(&viz.mailbox) {
                                // This request is posted after `update`'s normal
                                // repaint decision. Keep ticking until the idle
                                // worker publishes and the UI adopts it.
                                ctx.request_repaint_after(std::time::Duration::from_millis(16));
                            }
                        }
                        let rect = ui.available_rect_before_wrap();
                        let (rect, _response) =
                            ui.allocate_exact_size(rect.size(), egui::Sense::drag());

                        if let Some(renderer) = &self.cfd_renderer {
                            let renderer = renderer.clone();
                            let viewport_size = [rect.width(), rect.height()];

                            let (min_x, max_x, min_y, max_y) = cfd_renderer::compute_bounds(cells);
                            let mesh_width = max_x - min_x;
                            let mesh_height = max_y - min_y;

                            // Fit to screen preserving aspect ratio
                            let s = (rect.width() / mesh_width as f32)
                                .min(rect.height() / mesh_height as f32);

                            let scale_x = s / rect.width();
                            let scale_y = s / rect.height();

                            let mesh_center_x = (min_x + max_x) / 2.0;
                            let mesh_center_y = (min_y + max_y) / 2.0;

                            let tx = 0.5 - mesh_center_x as f32 * scale_x;
                            let ty = 0.5 - mesh_center_y as f32 * scale_y;

                            let (stride, offset, mode) = self.render_layout_for_field();
                            let range_field = self.render_range_field();

                            let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                                rect,
                                CfdRenderCallback {
                                    renderer: renderer.clone(),
                                    uniforms: cfd_renderer::CfdUniforms {
                                        transform: [scale_x, scale_y, tx, ty],
                                        viewport_size,
                                        range: [min_val, max_val],
                                        stride,
                                        offset,
                                        mode,
                                        range_index: range_field as u32,
                                    },
                                    draw_lines: self.show_mesh_lines,
                                    viz_frame_request: if self.is_running {
                                        self.viz_field.as_ref().map(|viz| Arc::clone(&viz.mailbox))
                                    } else {
                                        None
                                    },
                                },
                            );

                            ui.painter().add(cb);

                            // Immersed-obstacle overlay: the dense grid is not cut, so
                            // draw the solid cells as a grey overlay so the geometry is
                            // visible. Map domain → screen with the SAME fit the shader
                            // uses (centre the mesh in `rect` at `s` px per domain unit;
                            // screen-y is flipped vs domain-y).
                            if !self.structured_solid_mask.is_empty() {
                                let center = rect.center();
                                let painter = ui.painter_at(rect);
                                let grey = egui::Color32::from_rgba_unmultiplied(90, 90, 90, 220);
                                for (i, poly) in cells.iter().enumerate() {
                                    if !self.structured_solid_mask.get(i).copied().unwrap_or(false)
                                    {
                                        continue;
                                    }
                                    let pts: Vec<egui::Pos2> = poly
                                        .iter()
                                        .map(|p| {
                                            egui::pos2(
                                                center.x + (p[0] as f32 - mesh_center_x as f32) * s,
                                                center.y - (p[1] as f32 - mesh_center_y as f32) * s,
                                            )
                                        })
                                        .collect();
                                    if pts.len() >= 3 {
                                        painter.add(egui::Shape::convex_polygon(
                                            pts,
                                            grey,
                                            egui::Stroke::NONE,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    RenderMode::EguiPlot => {
                        if let Some(viz) = &self.viz_field {
                            // This path consumes host `cached_u/p`, so ordinary
                            // readback stepping must remain authoritative.
                            viz.mailbox.set_gpu_consumer_enabled(false);
                        }
                        let Some(vals) = self
                            .plot_cache
                            .as_ref()
                            .and_then(|cache| cache.values.as_deref())
                        else {
                            ui.centered_and_justified(|ui| {
                                ui.label("Waiting for field snapshot...");
                            });
                            return;
                        };

                        let solid = &self.structured_solid_mask;
                        Plot::new("cfd_plot").data_aspect(1.0).show(ui, |plot_ui| {
                            for (i, polygon_points) in cells.iter().enumerate() {
                                // Immersed-obstacle cells render as opaque grey so the
                                // geometry is visible on the (uncut) dense grid.
                                let color = if solid.get(i).copied().unwrap_or(false) {
                                    egui::Color32::from_gray(90)
                                } else {
                                    let val = vals.get(i).copied().unwrap_or_default();
                                    let t = (val - min_val as f64) / (max_val - min_val) as f64;
                                    get_color(t)
                                };

                                plot_ui.polygon(
                                    Polygon::new("", PlotPoints::new(polygon_points.clone()))
                                        .fill_color(color)
                                        .stroke(if self.show_mesh_lines {
                                            egui::Stroke::new(1.0, egui::Color32::BLACK)
                                        } else {
                                            egui::Stroke::NONE
                                        }),
                                );
                            }
                        });
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    if is_initializing {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Initializing solver...");
                        });
                    } else {
                        ui.label("Press Initialize to start");
                    }
                });
            }
        });
    }
}

impl eframe::App for CFDApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &root_ui.ctx().clone();
        self.poll_init();
        self.spawn_pending_init();
        self.poll_solver_worker();
        let init_in_progress = self.init_rx.is_some();
        let init_pending = self.pending_init_request.is_some();
        let is_initializing = init_in_progress || init_pending;
        // Headed perf harness (CFD2_AUTOSTART): press Run once init completes.
        if self.autostart_run_pending && !is_initializing {
            self.autostart_run_pending = false;
            self.is_running = true;
            self.set_worker_running(true);
        }

        egui::Panel::left("controls").show(root_ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.heading("CFD Controls");
                    if init_in_progress {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Initializing solver...");
                        });
                    } else if init_pending {
                        ui.label("Initializing solver...");
                    }

                    ui.add_enabled_ui(!is_initializing, |ui| {
                    ui.group(|ui| {
                        ui.label("Discretization");
                        let mut mode_changed = false;
                        mode_changed |= ui
                            .radio_value(
                                &mut self.mesh_mode,
                                MeshMode::Unstructured,
                                "Unstructured mesh",
                            )
                            .on_hover_text(
                                "Face-connectivity mesh (cut-cell / Voronoi / Delaunay), \
                                 the full solver + moving-mesh (ALE) feature set.",
                            )
                            .changed();
                        mode_changed |= ui
                            .radio_value(
                                &mut self.mesh_mode,
                                MeshMode::Structured2D,
                                "Structured (dense grid)",
                            )
                            .on_hover_text(
                                "Dense Cartesian grid with NO connectivity indirection; \
                                 immersed (Brinkman) obstacles instead of cut cells. The \
                                 backend default is scheme-aware: CPU-transpiled for the \
                                 implicit coupled solve (host-side banded solver), GPU for \
                                 explicit RK4 (fully GPU-resident); still selectable. \
                                 A lid-driven cavity / uniform-gas box demo. Hides the \
                                 unstructured meshers, mesh grading, moving mesh (ALE), and \
                                 the nozzle geometry.",
                            )
                            .changed();
                        if mode_changed {
                            if self.mesh_mode == MeshMode::Structured2D {
                                // Structured has no ALE and a fixed model family:
                                // force-disable moving mesh and switch to a structured
                                // model id. DEFAULT to the CPU-transpiled backend: the
                                // structured coupled saddle solve is host-side either
                                // way (the GPU path reads the full banded matrix back
                                // every outer iteration to solve in f64), so the GPU
                                // backend adds per-outer readback + sync stalls for no
                                // compute win — the CPU-transpiled solver is measurably
                                // faster (~24 ms/step at 120×40) and drives the same
                                // GPU-direct renderer via the state-upload bridge. The
                                // GPU radio remains available for manual selection.
                                self.enable_moving_mesh = false;
                                if !self.model_id.ends_with("_structured") {
                                    self.model_id = "incompressible_momentum_structured";
                                }
                            } else if let Some(base) =
                                self.model_id.strip_suffix("_structured")
                            {
                                // Back to unstructured: map the structured id → base.
                                self.model_id = match base {
                                    "incompressible_momentum" => "incompressible_momentum",
                                    "allmach_thermal" => "allmach_thermal",
                                    "compressible" => "compressible",
                                    _ => "incompressible_momentum",
                                };
                            }
                            self.apply_model_defaults();
                            if self.mesh_mode == MeshMode::Structured2D {
                                // Scheme-aware backend default (after
                                // apply_model_defaults has set the scheme).
                                self.backend = self.structured_backend_default();
                            }
                            self.init_solver();
                        }
                    });

                    // The geometry selector applies to BOTH paths: the unstructured
                    // solver cuts the shape from the mesh; the structured solver
                    // immerses it as a Brinkman `ibm_penalty_U` mask in a dense box.
                    let structured = self.mesh_mode == MeshMode::Structured2D;
                    ui.group(|ui| {
                        ui.label("Geometry");
                        let mut geom_changed = false;
                        geom_changed |= ui
                            .radio_value(
                                &mut self.selected_geometry,
                                GeometryType::BackwardsStep,
                                "Backwards Step",
                            )
                            .changed();
                        geom_changed |= ui
                            .radio_value(
                                &mut self.selected_geometry,
                                GeometryType::ChannelObstacle,
                                "Channel w/ Obstacle",
                            )
                            .changed();
                        geom_changed |= ui
                            .radio_value(
                                &mut self.selected_geometry,
                                GeometryType::Nozzle,
                                "Supersonic Nozzle",
                            )
                            .on_hover_text(
                                "Converging–diverging nozzle. Selecting it switches to the \
                                 All-Mach thermal model with a PRESSURE INLET + supersonic \
                                 outlet (toggle in Solver Parameters), driving the flow \
                                 through a choked throat to a SUPERSONIC exit (with \
                                 expansion cooling).",
                            )
                            .changed();
                        if geom_changed {
                            // The nozzle is meaningless without compressibility: when it is
                            // picked, switch to the supersonic-capable thermal model (unless
                            // already on an all-Mach model). Then re-seed the per-case
                            // defaults (the nozzle override sets the choking inlet speed +
                            // sub-critical back-pressure) and rebuild — mirroring the Model
                            // dropdown's apply-defaults-then-reinit behaviour. In structured
                            // mode the mesh type is fixed (dense Cartesian) and the models
                            // are the *_structured set, so those unstructured-only side
                            // effects are skipped — the obstacle is a Brinkman mask instead.
                            if structured && self.selected_geometry == GeometryType::Nozzle {
                                self.model_id = "allmach_thermal_structured";
                            } else if !structured {
                                if self.selected_geometry == GeometryType::Nozzle {
                                    if self.model_id != "allmach_pressure"
                                        && self.model_id != "allmach_thermal"
                                    {
                                        self.model_id = "allmach_thermal";
                                    }
                                    // Default the nozzle to the body-fitted structured
                                    // grid (the validated configuration). The user can
                                    // still switch to an unstructured mesh below.
                                    self.mesh_type = MeshType::Fitted;
                                } else if self.mesh_type == MeshType::Fitted {
                                    // `Fitted` is nozzle-only; fall back for other shapes.
                                    self.mesh_type = MeshType::CutCell;
                                }
                            }
                            self.apply_model_defaults();
                            self.init_solver();
                        }
                    });

                    ui.group(|ui| {
                        ui.label("Mesh Parameters");
                        // The fitted (structured) mesh AND the dense structured grid are
                        // uniform: only a single target cell size applies, so hide the
                        // min-size / growth-rate controls they do not honour.
                        let show_grading =
                            self.mesh_type.supports_size_grading() && !structured;
                        if show_grading {
                            ui.add(
                                adaptive_slider(
                                    &mut self.min_cell_size,
                                    0.001..=self.max_cell_size,
                                )
                                .text("Min Cell Size"),
                            );
                        }
                        // For the fitted grid the sole control is the target cell
                        // size; give it a low floor (and see the nx/ny clamp in
                        // `build_mesh_with`) so the user can drive the mesh finer by
                        // typing.
                        let cell_size_base = if show_grading {
                            self.min_cell_size..=0.5
                        } else {
                            0.002..=0.5
                        };
                        ui.add(
                            adaptive_slider(&mut self.max_cell_size, cell_size_base).text(
                                if show_grading {
                                    "Max Cell Size"
                                } else {
                                    "Cell Size"
                                },
                            ),
                        )
                        .on_hover_text(if show_grading {
                            "Upper bound on cell size (coarse regions). Type any value \
                             to go beyond the slider ends."
                        } else {
                            "Target cell size for the structured grid (sets resolution). \
                             Type any value to go finer than the slider end."
                        });
                        if show_grading {
                            ui.add(
                                adaptive_slider(&mut self.growth_rate, 1.0..=2.0)
                                    .text("Growth Rate"),
                            );
                        }
                        // The unstructured mesher choice is meaningless for a dense
                        // Cartesian structured grid — hide it entirely in that mode.
                        if !structured {
                        ui.separator();
                        ui.label("Mesh Type");
                        // Moving mesh locks Mesh Type to Voronoi (CVT) — grey the
                        // radios (rather than silently overriding them) so the lock
                        // is visible and the selection can't diverge from what runs.
                        let mesh_locked = self.enable_moving_mesh || self.solver_is_moving;
                        let mesh_resp = ui.add_enabled_ui(!mesh_locked, |ui| {
                            // The body-fitted structured grid is only meaningful for the
                            // nozzle (it conforms to the CD-nozzle walls), so offer it only
                            // there. The unstructured meshers work for every geometry.
                            if self.selected_geometry == GeometryType::Nozzle {
                                ui.radio_value(&mut self.mesh_type, MeshType::Fitted, "Fitted")
                                    .on_hover_text(
                                        "Body-fitted curvilinear structured grid conforming to \
                                         the nozzle walls. The validated configuration — \
                                         recommended for this case.",
                                    );
                            }
                            ui.radio_value(&mut self.mesh_type, MeshType::CutCell, "CutCell");
                            ui.radio_value(&mut self.mesh_type, MeshType::Delaunay, "Delaunay");
                            ui.radio_value(&mut self.mesh_type, MeshType::Voronoi, "Voronoi");
                            ui.radio_value(&mut self.mesh_type, MeshType::VoronoiCvt, "Voronoi (CVT)")
                                .on_hover_text(
                                    "Meshless Voronoi mesh with Lloyd/CVT seed relaxation: \
                                     near-hexagonal cells with close-to-zero interior-face \
                                     skewness (no post-generation vertex smoothing).",
                                );
                        });
                        if mesh_locked {
                            mesh_resp.response.on_hover_text(
                                "Locked to Voronoi (CVT) while Moving Mesh (ALE) is enabled.",
                            );
                        }
                        } // end if !structured (Mesh Type)
                    });

                        // Moving mesh (ALE) has no structured analogue.
                        if !structured {
                        ui.group(|ui| {
                        ui.label("Moving Mesh (ALE)");
                        // Moving mesh keeps the user's SELECTED model — it maps it to
                        // the same-physics ALE variant at build (`build_moving_init`),
                        // never silently swaps it. Only models that have an ALE variant
                        // may enable it; the rest disable the toggle with an
                        // explanatory tooltip so the choice is never quietly discarded.
                        let ale_variant = CFDApp::ale_model_for(self.model_id);
                        let mut enable = self.enable_moving_mesh;
                        let checkbox = ui.add_enabled(
                            ale_variant.is_some() || self.enable_moving_mesh,
                            egui::Checkbox::new(&mut enable, "Enable Moving Mesh (ALE)"),
                        );
                        if ale_variant.is_some() {
                            checkbox.on_hover_text(format!(
                                "Advect the CVT-Voronoi mesh with the flow (ALE) using the \
                                 selected solver's moving-mesh variant ({}). The mesh is \
                                 RECONSTRUCTED every step: fully on-device on the GPU backend \
                                 (when 'On-device mesh regen' is on), on the CPU at compute \
                                 time otherwise — the live 'Mesh regen' stats line shows which \
                                 path ran. While enabled, Mesh Type is locked to Voronoi (CVT) \
                                 and a fixed timestep is pinned; the Model selection is kept \
                                 (mapped to its ALE variant), not changed. Applied on \
                                 Initialize / Reset.",
                                CFDApp::model_label(ale_variant.unwrap()),
                            ));
                        } else {
                            checkbox.on_hover_text(format!(
                                "Moving Mesh (ALE) is not available for the '{}' solver (it has \
                                 no moving-mesh variant). Select an incompressible or all-Mach \
                                 model to enable it.",
                                CFDApp::model_label(self.model_id),
                            ));
                        }
                        if enable != self.enable_moving_mesh {
                            self.enable_moving_mesh = enable;
                            if enable {
                                // RK4 has no stage-consistent ALE geometry path.
                                // Reflect the supported scheme in the controls
                                // immediately; `make_init_request` repeats this
                                // guard so request construction is authoritative.
                                enforce_moving_mesh_time_scheme(
                                    true,
                                    &mut self.time_scheme,
                                );
                                // Save the pre-moving mesh selection so disabling
                                // restores it; the model selection is untouched.
                                self.pre_moving_mesh_type = Some(self.mesh_type);
                                // Moving mesh requires the meshless CVT-Voronoi mesh.
                                // Adaptive dt is KEPT: on this path it routes to the
                                // driver-side flow-CFL controller, pinned inside the
                                // GCL dt handshake. The MODEL is left as the user
                                // chose it — mapped to its ALE variant at build time.
                                self.mesh_type = MeshType::VoronoiCvt;
                                self.refresh_model_caps();
                            } else if let Some(prev) = self.pre_moving_mesh_type.take() {
                                // Restore the mesh selection the user had before.
                                self.mesh_type = prev;
                            }
                        }
                        if self.enable_moving_mesh {
                            // Where the per-step mesh reconstruction runs. GPU
                            // backend: fully on-device (opt-out); any step the
                            // device cannot certify (Voronoi flip / sliver)
                            // transparently re-runs on the CPU for that step.
                            // CPU backends: always CPU at compute time.
                            let gpu_backend = self.backend == BackendChoice::Gpu;
                            if gpu_backend {
                                ui.checkbox(
                                    &mut self.moving_gpu_regen,
                                    "On-device mesh regen (GPU)",
                                )
                                .on_hover_text(
                                    "Rebuild the mesh ENTIRELY on the GPU every step — \
                                     Voronoi diagram, topology, geometry, and the ALE swept \
                                     fluxes (the meshless-Voronoi-on-GPU pipeline). Steps the \
                                     device cannot certify (a Voronoi topology flip or a \
                                     sub-tolerance sliver) automatically re-run on the CPU \
                                     for that step and retry the GPU on the next; the 'Mesh \
                                     regen' stats line shows the live path. Off = rebuild on \
                                     the CPU every step. Applied on Initialize / Reset.",
                                );
                            } else {
                                ui.label("Mesh regen: CPU at compute time (per step)")
                                    .on_hover_text(
                                        "On a CPU compute backend the mesh is re-assembled \
                                         from the advected seeds by the CPU meshless engine \
                                         every step. Select the GPU backend to enable \
                                         on-device reconstruction.",
                                    );
                            }
                            egui::ComboBox::from_label("Seed motion")
                                .selected_text(self.moving_motion.label())
                                .show_ui(ui, |ui| {
                                    for choice in MovingMotionChoice::ALL {
                                        ui.selectable_value(
                                            &mut self.moving_motion,
                                            choice,
                                            choice.label(),
                                        );
                                    }
                                });
                            if self.moving_motion == MovingMotionChoice::FlowCoupled {
                                ui.add(
                                    adaptive_slider(&mut self.moving_regularization, 0.0..=2.0)
                                        .text("Regularization χ"),
                                )
                                .on_hover_text(
                                    "Centroid-steering strength for flow-coupled motion: \
                                     0 = pure flow advection; higher pulls seeds toward \
                                     cell centroids to hold mesh quality.",
                                );
                                ui.add(
                                    egui::Slider::new(&mut self.moving_motion_iters, 1..=5)
                                        .text("Implicit motion iterations"),
                                )
                                .on_hover_text(
                                    "Fixed-point coupling of the mesh motion to the \
                                     solution: each step is re-solved from a rewound \
                                     state with the seeds advected by its own \
                                     end-of-step velocity, until the motion converges \
                                     (2% of a cell) or this many attempts ran. Cures \
                                     the phantom pressure noise the explicit \
                                     (lagged-velocity) remeshing injects. 1 = explicit \
                                     (default). Each extra iteration costs a full \
                                     regen + solve. CPU backend only. Applied on \
                                     Initialize / Reset.",
                                );
                            }
                            // Smoothing, memory reordering and flow-adaptive
                            // sizing apply to FlowCoupled AND Frozen (a
                            // STATIONARY mesh adapts to the flow and relaxes
                            // in place; only Prescribed motion — recomputed
                            // from t=0 labels each step — excludes them).
                            if matches!(
                                self.moving_motion,
                                MovingMotionChoice::Frozen | MovingMotionChoice::FlowCoupled
                            ) {
                                ui.add(
                                    egui::Slider::new(&mut self.moving_smooth_every_n, 0..=100)
                                        .text("Lloyd smooth every N steps"),
                                )
                                .on_hover_text(
                                    "Scheduled mesh smoothing: every N steps, one gentle \
                                     blended Lloyd sweep regularizes the interior seeds \
                                     (0 = off). With adaptive sizing enabled the Lloyd \
                                     target preserves the local adapted spacing (it will \
                                     not pull the mesh back toward uniform). Applied on \
                                     Initialize / Reset.",
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.moving_reorder_every_n,
                                        0..=2000,
                                    )
                                    .text("Memory reorder every N steps"),
                                )
                                .on_hover_text(
                                    "Periodically relabel the cells in Morton order of the \
                                     current seed positions (0 = off) — a pure relabel with \
                                     zero effect on the physics. Seed recycling migrates \
                                     slots far from their spatial neighbors over time, \
                                     degrading memory locality (measured: mean neighbor \
                                     slot distance 63 → 180 over 2000 steps; reordering \
                                     holds ~70). Applied on Initialize / Reset.",
                                );
                                ui.add(
                                    egui::Slider::new(&mut self.moving_adapt_every_n, 0..=500)
                                        .text("Adaptive sizing every N steps"),
                                )
                                .on_hover_text(
                                    "Flow-adaptive cell birth/kill (0 = off): every N steps \
                                     a per-cell target volume is derived from the flow's \
                                     velocity, pressure and strain gradients (high gradients \
                                     → the adapt band's finest sizing, smooth regions → its \
                                     coarsest); over-resolved cells are removed and \
                                     under-resolved cells split, so the CELL COUNT adapts at \
                                     runtime (bounded to 0.5–2× the initial count; each \
                                     event rebuilds the solver and transfers the state). \
                                     Static wall/obstacle discretization refines along \
                                     (segments subdivide where the flow demands finer wall \
                                     cells). Applied on Initialize / Reset.",
                                );
                                if self.moving_adapt_every_n > 0 {
                                    ui.add(
                                        adaptive_slider(
                                            &mut self.moving_adapt_min_size,
                                            0.0..=0.1,
                                        )
                                        .text("Adapt min cell size"),
                                    )
                                    .on_hover_text(
                                        "The FINEST cell size the adaptation refines \
                                         high-gradient regions toward — independent of the \
                                         built mesh (build coarse, adapt finer). 0 = auto \
                                         (the mesh's realized sizing band; both sliders \
                                         must be > 0 to take effect). Growth stays capped \
                                         at 2× the initial cell count. Applied on \
                                         Initialize / Reset.",
                                    );
                                    ui.add(
                                        adaptive_slider(
                                            &mut self.moving_adapt_max_size,
                                            0.0..=0.2,
                                        )
                                        .text("Adapt max cell size"),
                                    )
                                    .on_hover_text(
                                        "The COARSEST cell size smooth regions may grow \
                                         to (their cells are merged away until local \
                                         sizing reaches it). 0 = auto (the mesh's \
                                         realized sizing band; both sliders must be > 0 \
                                         to take effect). Shrink stays capped at 0.5× \
                                         the initial cell count. Applied on Initialize / \
                                         Reset.",
                                    );
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.moving_adapt_budget,
                                            1.0..=8.0,
                                        )
                                        .text("Adapt cell budget ×"),
                                    )
                                    .on_hover_text(
                                        "Growth budget: adaptive births stop once the \
                                         cell count reaches this multiple of the initial \
                                         count (the stats line shows \"adapt budget \
                                         reached\" while capped). The per-cell viz \
                                         buffers are allocated at this factor, so a \
                                         larger budget costs GPU memory up front. \
                                         Applied on Initialize / Reset.",
                                    );
                                    for (value, label, what) in [
                                        (
                                            &mut self.moving_adapt_thresh_u,
                                            "U-gradient threshold ×",
                                            "the velocity-gradient magnitude |∇U|",
                                        ),
                                        (
                                            &mut self.moving_adapt_thresh_p,
                                            "p-gradient threshold ×",
                                            "the pressure-gradient magnitude |∇p|",
                                        ),
                                        (
                                            &mut self.moving_adapt_thresh_strain,
                                            "Strain threshold ×",
                                            "the strain-rate magnitude",
                                        ),
                                    ] {
                                        ui.add(
                                            adaptive_slider(value, 0.0..=5.0).text(label),
                                        )
                                        .on_hover_text(format!(
                                            "Refinement threshold for {what}, relative to \
                                             its auto-calibrated scale (the 90th \
                                             percentile over the adaptable cells): 1 = \
                                             auto, higher = only stronger features \
                                             refine, lower = refine at weaker ones, 0 = \
                                             ignore this indicator. The three indicators \
                                             combine by maximum. Applied on Initialize / \
                                             Reset.",
                                        ));
                                    }
                                }
                            }
                            // Oscillating obstacle (ChannelObstacle only — its
                            // boundary spec has the obstacle as loop 1).
                            if self.selected_geometry == GeometryType::ChannelObstacle {
                                ui.separator();
                                ui.checkbox(
                                    &mut self.moving_oscillate_obstacle,
                                    "Oscillating obstacle",
                                )
                                .on_hover_text(
                                    "Cross-stream sinusoidally oscillate the cylinder. Its \
                                     boundary seeds move rigidly with it and the contour \
                                     carries the wall's material velocity (MovingWall BC), so \
                                     the fluid feels the moving wall. Applied on \
                                     Initialize / Reset.",
                                );
                                if self.moving_oscillate_obstacle {
                                    // Cap the slider MAX at the driver's anti-swallow clamp
                                    // (`OSC_AMPLITUDE_CELL_FRACTION × near-wall spacing`) so the
                                    // displayed amplitude can never exceed the REALIZED motion —
                                    // the driver clamps larger requests, which would otherwise
                                    // show a value the obstacle never reaches. The cap tracks the
                                    // mesh-size slider (finer mesh ⇒ smaller safe amplitude).
                                    let osc_cap =
                                        (OSC_AMPLITUDE_CELL_FRACTION * self.min_cell_size).max(0.005);
                                    ui.add(
                                        adaptive_slider(
                                            &mut self.moving_osc_amplitude,
                                            0.005..=osc_cap,
                                        )
                                        .text("Amplitude"),
                                    )
                                    .on_hover_text(
                                        "Cross-stream oscillation amplitude (domain units). \
                                         Capped at the near-wall cell spacing (0.6× the mesh size): \
                                         a larger peak displacement sweeps the moving obstacle \
                                         through the frozen interior seeds and tangles the mesh, so \
                                         the driver clamps it — the slider max is that clamp.",
                                    );
                                    ui.add(
                                        adaptive_slider(
                                            &mut self.moving_osc_frequency,
                                            0.1..=3.0,
                                        )
                                        .text("Frequency (Hz)"),
                                    )
                                    .on_hover_text(
                                        "Forcing frequency; ω = 2π·f. A faster wall shrinks \
                                         the mesh-motion-CFL-capped timestep.",
                                    );
                                }
                            }
                            ui.label("Applied on Initialize / Reset.");
                        }
                        });
                        } // end if !structured (Moving Mesh ALE group)

                        ui.group(|ui| {
                        ui.label("Fluid Properties");
                        egui::ComboBox::from_label("Preset")
                            .selected_text(&self.current_fluid.name)
                            .show_ui(ui, |ui| {
                                for fluid in Fluid::presets() {
                                    if ui
                                        .selectable_value(
                                            &mut self.current_fluid,
                                            fluid.clone(),
                                            &fluid.name,
                                        )
                                        .clicked()
                                    {
                                        self.update_gpu_fluid();
                                    }
                                }
                            });

                        let mut density = self.current_fluid.density;
                        if ui
                            .add(
                                adaptive_slider(&mut density, 0.1..=20000.0)
                                    .text("Density (kg/m³)"),
                            )
                            .changed()
                        {
                            self.current_fluid.density = density;
                            self.current_fluid.name = "Custom".to_string();
                            if let crate::solver::model::eos::EosSpec::LinearCompressibility {
                                rho_ref,
                                ..
                            } = &mut self.current_fluid.eos
                            {
                                *rho_ref = density;
                            }
                            self.update_gpu_fluid();
                        }

                        let mut viscosity = self.current_fluid.viscosity;
                        if ui
                            .add(
                                adaptive_slider(&mut viscosity, 1e-6..=0.1)
                                    .logarithmic(true)
                                    .text("Viscosity (Pa·s)"),
                            )
                            .changed()
                        {
                            self.current_fluid.viscosity = viscosity;
                            self.current_fluid.name = "Custom".to_string();
                            self.update_gpu_fluid();
                        }
                        });

                        ui.group(|ui| {
                        ui.label("Inlet Conditions");

                        if ui
                            .add(
                                adaptive_slider(&mut self.inlet_velocity, 0.0..=10.0)
                                    .text("Inlet Velocity (m/s)"),
                            )
                            .changed()
                        {
                            self.update_gpu_inlet_velocity();
                        }

                        // All-Mach compressibility: ψ = 1/c² is the REAL value from the
                        // fluid's EOS (sound speed) — no exaggeration. Read-only physical
                        // readout of the regime (ψ, sound speed, inlet Mach).
                        if matches!(
                            self.model_id,
                            "allmach_pressure" | "allmach_thermal" | "allmach_thermal_structured"
                        ) {
                            let c_phys = self.current_fluid.sound_speed();
                            let mach_real = if c_phys > 0.0 {
                                self.inlet_velocity.abs() as f64 / c_phys
                            } else {
                                0.0
                            };
                            ui.label(format!(
                                "Compressibility ψ = 1/c² = {:.2e} (real EOS) · c = {c_phys:.0} m/s \
                                 · inlet Mach ≈ {mach_real:.2e}",
                                self.current_fluid.compressibility()
                            ));
                            // Preconditioner floor (pressure smoothness). Live via
                            // `apply_params` (no rebuild). Raising it toward ~1.0
                            // collapses the residual standing pseudo-acoustic pressure
                            // mode onto the incompressible field; 0.2 is the bare
                            // divergence-stability floor. Step-0 safe via the moving
                            // driver's startup dt growth-cap.
                            let mut floor = self.allmach_precond_uref_min;
                            if ui
                                .add(
                                    adaptive_slider(&mut floor, 0.2..=2.0)
                                        .text("Preconditioner floor (pressure smoothness)"),
                                )
                                .on_hover_text(
                                    "Low-Mach preconditioner reference-velocity floor. \
                                     Higher (~1.0) makes the pressure nearly elliptic under \
                                     the adaptive dt, cleaning the residual standing pressure \
                                     mode toward the incompressible field; 0.2 is the minimum \
                                     that stops the compressible outlet divergence. Live, no \
                                     rebuild; step-0 safe via the startup dt cap.",
                                )
                                .changed()
                            {
                                self.allmach_precond_uref_min = floor.max(0.0);
                                self.sync_worker_params();
                            }
                        }

                        // Supersonic-nozzle driver: a sub-critical (negative gauge)
                        // outlet back-pressure pulls the diverging section past Mach 1.
                        // Live (no rebuild) via `apply_params`, so the user can drag
                        // from 0 (subsonic) toward the stable floor and watch the exit
                        // go supersonic. Shown only for the all-Mach nozzle case; the
                        // floor (≈ −0.05) stays inside the gauge-EOS envelope (below it
                        // the outlet density crosses zero and the solve diverges).
                        if self.selected_geometry == GeometryType::Nozzle
                            && matches!(
                                self.model_id,
                                "allmach_pressure"
                                    | "allmach_thermal"
                                    | "allmach_thermal_structured"
                                    | "compressible"
                                    | "compressible_structured"
                            )
                        {
                            // Driving-mode toggle. Flipping it changes the Inlet/Outlet
                            // boundary KINDS (baked at build via `apply_pressure_inlet_nozzle_bcs`),
                            // so it must REBUILD the solver — unlike the pressure sliders below,
                            // which are live via `apply_params`. `init_solver` rebuilds from the
                            // current params (which carry the toggled `pressure_inlet`); we do NOT
                            // call `apply_model_defaults` here, which would reset the toggle.
                            let mut pin = self.pressure_inlet;
                            if ui
                                .checkbox(&mut pin, "Pressure inlet + supersonic outlet")
                                .on_hover_text(
                                    "ON: pin the inlet gauge pressure (gauge anchored upstream) \
                                     and let the outlet float — the physically-correct \
                                     supersonic-outlet driving. OFF: velocity inlet + a \
                                     sub-critical outlet back-pressure. Changing this REBUILDS \
                                     the solver (it flips the inlet/outlet boundary kinds).",
                                )
                                .changed()
                            {
                                self.pressure_inlet = pin;
                                self.init_solver();
                            }
                            if self.pressure_inlet {
                                // Pressure-inlet nozzle: tune the pinned inlet gauge
                                // pressure (the gauge anchor). The outlet floats —
                                // supersonic outlet, no back-pressure. The slider
                                // range follows the model family's pressure scale
                                // (all-Mach gauge units vs compressible Pa).
                                // Compressible ceiling covers the full regime
                                // ladder of the area-ratio-2 nozzle (p_e/p0 ~
                                // 0.094 on the supersonic branch): started
                                // (shock swallowed) above ~1e5 gauge,
                                // over-expanded interior-supersonic beyond,
                                // UNDERexpanded exit above ~1.0e6 gauge.
                                let pressure_ceiling = if matches!(
                                    self.model_id,
                                    "compressible" | "compressible_structured"
                                ) {
                                    1.5e6
                                } else {
                                    0.12
                                };
                                let mut p_in = self.inlet_pressure;
                                if ui
                                    .add(
                                        adaptive_slider(&mut p_in, 0.0..=pressure_ceiling)
                                            .text("Inlet pressure (gauge)"),
                                    )
                                    .on_hover_text(
                                        "Pressure-inlet CD nozzle: pins the inlet gauge \
                                         pressure (the gauge anchor); the outlet floats \
                                         (supersonic, no back-pressure). For the \
                                         compressible model this is the TOTAL (reservoir) \
                                         gauge pressure — the ghost static state follows \
                                         the extrapolated inflow via the stagnation \
                                         relations. Higher ⇒ stronger drop ⇒ supersonic \
                                         exit.",
                                    )
                                    .changed()
                                {
                                    self.inlet_pressure = p_in;
                                    self.sync_worker_params();
                                }
                            } else {
                                let mut p_back = self.outlet_back_pressure;
                                if ui
                                    .add(
                                        adaptive_slider(&mut p_back, -0.05..=0.0)
                                            .text("Outlet back-pressure (gauge)"),
                                    )
                                    .on_hover_text(
                                        "Drives the converging–diverging nozzle. 0 ⇒ subsonic \
                                         exit; lowering it past the critical value pulls the \
                                         diverging section SUPERSONIC (M_exit up to ≈ 1.07 at \
                                         the stable floor).",
                                    )
                                    .changed()
                                {
                                    self.outlet_back_pressure = p_back;
                                    self.sync_worker_params();
                                }
                            }
                        }

                        let char_length = 1.0; // Characteristic length (channel height)
                        let re = self.current_fluid.density
                            * self.inlet_velocity.abs() as f64
                            * char_length
                            / self.current_fluid.viscosity;
                        ui.label(format!("Est. Reynolds Number: {:.0}", re));
                        });

                        ui.group(|ui| {
                        ui.label("Solver Parameters");

                        // SI units:
                        // - Mesh coordinates/volumes are in meters.
                        // - `timestep` is seconds.
                        // - CFL is dimensionless: (wave_speed * dt / dx).
                        let dt_for_cfl = if self.mesh.is_some()
                            && self.cached_gpu_stats.dt.is_finite()
                            && self.cached_gpu_stats.dt > 0.0
                        {
                            self.cached_gpu_stats.dt as f64
                        } else {
                            self.timestep
                        };
                        let max_vel = self
                            .cached_u
                            .iter()
                            .map(|(vx, vy)| (vx * vx + vy * vy).sqrt())
                            .fold(0.0_f64, f64::max);
                        let adv_speed = max_vel.max(self.inlet_velocity.abs() as f64);
                        let allmach_rk4 = self.time_scheme == GpuTimeScheme::RK4
                            && matches!(
                                self.model_id,
                                "allmach_pressure"
                                    | "allmach_thermal"
                                    | "allmach_thermal_structured"
                            );
                        let sound_speed = if allmach_rk4 {
                            let psi = self.current_fluid.compressibility().max(1.0e-30);
                            let u_ref = 2.0
                                * (self.inlet_velocity.abs() as f64)
                                    .max(self.allmach_precond_uref_min as f64);
                            let chi = psi.max(1.0 / u_ref.max(1.0e-15).powi(2));
                            1.0 / chi.sqrt()
                        } else if self.model_caps.supports_eos_tuning {
                            self.current_fluid.sound_speed()
                        } else {
                            0.0
                        };
                        let effective_sound_speed = if self.model_caps.supports_low_mach {
                            match self.low_mach_model {
                                GpuLowMachPrecondModel::Off => sound_speed,
                                GpuLowMachPrecondModel::Legacy => sound_speed.min(adv_speed),
                                GpuLowMachPrecondModel::WeissSmith => {
                                    let theta = (self.low_mach_theta_floor as f64).max(0.0);
                                    let c_floor = sound_speed * theta.sqrt();
                                    sound_speed.min(adv_speed.max(c_floor))
                                }
                            }
                        } else {
                            sound_speed
                        };
                        let wave_speed = adv_speed + effective_sound_speed;
                        let (recommended_dt, cfl) = if self.min_cell_size > 0.0 && wave_speed > 1e-12
                        {
                            (
                                0.5 * self.min_cell_size / wave_speed,
                                wave_speed * dt_for_cfl / self.min_cell_size,
                            )
                        } else {
                            (0.0, 0.0)
                        };

                        if ui
                            .add(
                                adaptive_slider(&mut self.timestep, 0.0001..=0.1)
                                    .text("Timestep (s)"),
                            )
                            .changed()
                        {
                            self.update_gpu_dt();
                        }

                        // The moving-mesh (ALE) driver pins a fixed dt and hard-
                        // On the moving path the checkbox routes to the
                        // DRIVER-side flow-adaptive controller
                        // (`MovingMeshDriver::set_adaptive_dt`): the dt is
                        // chosen at the head of the ALE handshake, before the
                        // swept-flux closure, so the GCL holds. The
                        // solver-side `params.adaptive_dt` stays hard-rejected
                        // there (it re-scales dt AFTER closure) — the worker
                        // strips it from the params it forwards.
                        if ui
                            .checkbox(&mut self.adaptive_dt, "Adaptive Timestep")
                            .on_hover_text(
                                "Acoustically-adaptive timestep (CFL-targeted, 1.2× growth \
                                 cap). Static unstructured and structured both recompute \
                                 dt each step from max(|U|, |U_in|) + EOS sound speed. On \
                                 the moving mesh (ALE) it is applied inside the GCL dt \
                                 handshake: pinned per step before the swept-flux \
                                 closure.",
                            )
                            .changed()
                        {
                            self.sync_worker_params();
                        }
                        if self.adaptive_dt {
                            if ui
                                .add(
                                    adaptive_slider(&mut self.target_cfl, 0.1..=1.0)
                                        .text("Target CFL"),
                                )
                                .changed()
                            {
                                self.sync_worker_params();
                            }
                        }

                        if self.model_caps.supports_outer_iters
                            || self.model_caps.supports_low_mach
                            || self.model_caps.supports_dtau
                        {
                            ui.separator();
                            ui.label("Coupled/Implicit Controls");
                            if self.model_caps.supports_outer_iters
                                && ui
                                    .add(
                                        adaptive_slider(&mut self.outer_iters, 1..=100)
                                            .text("Outer Iterations"),
                                    )
                                    .changed()
                            {
                                self.update_gpu_outer_iters();
                            }
                            if self.model_caps.supports_dtau
                                && self.time_scheme != GpuTimeScheme::RK4
                            {
                                ui.separator();
                                ui.label("Dual Time Stepping");
                                if ui
                                    .checkbox(&mut self.dual_time, "Enable pseudo-time (dtau)")
                                    .changed()
                                {
                                    self.update_gpu_dtau();
                                    if self.dual_time && self.model_id == "compressible" {
                                        // Pressure under-relaxation is destabilizing for the
                                        // compressible solver in dual-time mode (unphysical striping
                                        // + runaway velocities). Keep pressure unrelaxed.
                                        if (self.alpha_p - 1.0).abs() > 1e-12 {
                                            self.alpha_p = 1.0;
                                            self.update_gpu_alpha_p();
                                        }
                                        // Provide conservative defaults if the UI is still at the
                                        // typical incompressible starting point.
                                        if (self.alpha_u - 0.7).abs() < 1e-12 {
                                            self.alpha_u = 0.7;
                                            self.update_gpu_alpha_u();
                                        }
                                    }
                                }
                                ui.add_enabled_ui(self.dual_time, |ui| {
                                    if ui
                                        .add(
                                            adaptive_slider(&mut self.dtau, 1e-8..=0.1)
                                                .logarithmic(true)
                                                .text("dtau (s)"),
                                        )
                                        .changed()
                                    {
                                        self.update_gpu_dtau();
                                    }

                                    let pseudo_wave_speed = adv_speed + effective_sound_speed;
                                    if self.min_cell_size > 0.0 && pseudo_wave_speed > 1e-12 {
                                        let pseudo_cfl = pseudo_wave_speed * self.dtau
                                            / self.min_cell_size.max(1e-12);
                                        ui.label(format!("Pseudo CFL≈{:.2}", pseudo_cfl));
                                    }
                                });
                            }
                            if self.model_caps.supports_low_mach {
                                ui.separator();
                                ui.label("Low-Mach Preconditioning");
                                let prev_model = self.low_mach_model;
                                egui::ComboBox::from_label("Model")
                                    .selected_text(match self.low_mach_model {
                                        GpuLowMachPrecondModel::Off => "Off",
                                        GpuLowMachPrecondModel::Legacy => "Legacy",
                                        GpuLowMachPrecondModel::WeissSmith => "Weiss-Smith",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.low_mach_model,
                                            GpuLowMachPrecondModel::Off,
                                            "Off",
                                        );
                                        ui.selectable_value(
                                            &mut self.low_mach_model,
                                            GpuLowMachPrecondModel::Legacy,
                                            "Legacy",
                                        );
                                        ui.selectable_value(
                                            &mut self.low_mach_model,
                                            GpuLowMachPrecondModel::WeissSmith,
                                            "Weiss-Smith",
                                        );
                                    });
                                if self.low_mach_model != prev_model {
                                    self.update_gpu_low_mach();
                                }

                                let theta_enabled = matches!(
                                    self.low_mach_model,
                                    GpuLowMachPrecondModel::WeissSmith
                                );
                                ui.add_enabled_ui(theta_enabled, |ui| {
                                    if ui
                                        .add(
                                            adaptive_slider(
                                                &mut self.low_mach_theta_floor,
                                                1e-8f32..=1e-2f32,
                                            )
                                            .logarithmic(true)
                                            .text("θ floor"),
                                        )
                                        .changed()
                                    {
                                        self.update_gpu_low_mach();
                                    }
                                });
                                if !theta_enabled {
                                    ui.weak("θ floor is used by Weiss-Smith.");
                                }

                                let coupling_enabled = !matches!(
                                    self.low_mach_model,
                                    GpuLowMachPrecondModel::Off
                                );
                                ui.add_enabled_ui(coupling_enabled, |ui| {
                                    if ui
                                        .add(
                                            adaptive_slider(
                                                &mut self.low_mach_pressure_coupling_alpha,
                                                0.0f32..=1.0f32,
                                            )
                                            .text("Pressure coupling α"),
                                        )
                                        .changed()
                                    {
                                        self.update_gpu_low_mach();
                                    }
                                });
                                if !coupling_enabled {
                                    ui.weak("Pressure coupling is active when preconditioning is on.");
                                }

                                if sound_speed > 1e-12 {
                                    let mach = adv_speed / sound_speed;
                                    ui.label(format!("Estimated Mach: {:.3e}", mach));
                                    if mach < 0.3
                                        && matches!(self.low_mach_model, GpuLowMachPrecondModel::Off)
                                    {
                                        ui.weak("Tip: enable preconditioning for low Mach.");
                                    }
                                }
                            }
                        }

                        ui.separator();
                        ui.label("Debug");
                        if ui
                            .checkbox(&mut self.log_convergence, "Log convergence to console")
                            .changed()
                        {
                            self.sync_worker_params();
                        }
                        ui.add_enabled_ui(self.log_convergence, |ui| {
                            if ui
                                .add(
                                    adaptive_slider(&mut self.log_every_steps, 1..=200)
                                        .text("Log every N steps"),
                                )
                                .changed()
                            {
                                self.sync_worker_params();
                            }
                        });

                        ui.separator();
                        ui.label("Tracing / Profiling");
                        let prev_trace = self.trace_enabled;
                        if ui
                            .checkbox(
                                &mut self.trace_enabled,
                                "Save trace to file (for replay + performance)",
                            )
                            .changed()
                        {
                            if self.trace_enabled {
                                self.start_trace();
                            } else {
                                self.stop_trace();
                            }
                        }

                        ui.horizontal(|ui| {
                            ui.label("Trace file");
                            ui.text_edit_singleline(&mut self.trace_path);
                        });
                        if prev_trace && self.trace_path.is_empty() {
                            ui.weak("Tip: set a path before restarting the trace.");
                        } else if self.trace_pending_start {
                            ui.weak("Trace will start after initialization.");
                        } else if self.trace_enabled && !cfg!(feature = "profiling") {
                            ui.weak("Built without `profiling` feature: trace has timings, but no GPU-CPU profiling counters.");
                        }

                        if cfl > 1.0 {
                            if self.model_caps.supports_dtau && self.dual_time {
                                ui.colored_label(
                                    egui::Color32::YELLOW,
                                    format!("Acoustic CFL≈{:.1} (>1) (dual time enabled)", cfl),
                                );
                            } else {
                                ui.colored_label(
                                    egui::Color32::RED,
                                    format!("⚠ CFL≈{:.1} (>1, may be unstable!)", cfl),
                                );
                                ui.colored_label(
                                    egui::Color32::YELLOW,
                                    format!("Recommended dt ≤ {:.4}", recommended_dt),
                                );
                            }
                        } else if cfl > 0.5 {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!("CFL≈{:.2} (moderate)", cfl),
                            );
                        }

                        ui.separator();
                        ui.label("Advection Scheme");
                        if ui
                            .radio(matches!(self.selected_scheme, Scheme::Upwind), "First order (Upwind)")
                            .clicked()
                        {
                            self.selected_scheme = Scheme::Upwind;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::SecondOrderUpwind),
                                "SOU (Second order upwind)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::SecondOrderUpwind;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::SecondOrderUpwindMinMod),
                                "SOU (MinMod limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::SecondOrderUpwindMinMod;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::SecondOrderUpwindVanLeer),
                                "SOU (VanLeer limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::SecondOrderUpwindVanLeer;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(matches!(self.selected_scheme, Scheme::QUICK), "QUICK")
                            .clicked()
                        {
                            self.selected_scheme = Scheme::QUICK;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::QUICKMinMod),
                                "QUICK (MinMod limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::QUICKMinMod;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::QUICKVanLeer),
                                "QUICK (VanLeer limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::QUICKVanLeer;
                            self.update_gpu_scheme();
                        }

                        if self.model_caps.supports_eos_tuning {
                            ui.label("Compressible solver uses this for KT flux reconstruction + deferred-correction advection.");
                        }

                        ui.separator();
                        ui.label("Preconditioner");
                        const UI_MAX_BLOCK_JACOBI: u32 = 16;
                        let structured_precond = self.mesh_mode == MeshMode::Structured2D;
                        let supports_preconditioner =
                            self.model_caps.supports_preconditioner && !structured_precond;
                        let model_owns_preconditioner = self.model_caps.model_owns_preconditioner;
                        let block_jacobi_supported =
                            self.model_caps.unknowns_per_cell <= UI_MAX_BLOCK_JACOBI;

                        if structured_precond {
                            // The dense-banded matrix-free structured solve carries its
                            // own preconditioner. The coupled U–p GMRES offers the SAME
                            // menu as the unstructured path where the model declares a
                            // Schur block layout (incompressible + thermal): block-Jacobi,
                            // the model-owned SIMPLE Schur (velocity predict + heavy-ball
                            // pressure solve, via FGMRES), or Schur with an AMG pressure
                            // solve (the same cpu::amg V-cycle on A_pp). The density-based
                            // compressible declares no layout → block-Jacobi only. Point-
                            // Jacobi is singular on the saddle-point pressure diagonal.
                            use crate::solver::banded_schur::CoupledPrecondKind as PK;
                            if model_owns_preconditioner {
                                ui.weak("Coupled banded solve preconditioner:");
                                let mut chosen: Option<PK> = None;
                                for (kind, label, hover) in [
                                    (PK::BlockJacobi, "Block-Jacobi",
                                     "Per-cell s×s block-inverse — the robust default."),
                                    (PK::Schur, "Schur (SIMPLE)",
                                     "Model-owned SIMPLE Schur complement with a heavy-ball \
                                      pressure solve — the same saddle-point preconditioner \
                                      the unstructured solver uses."),
                                    (PK::SchurAmg, "Schur + AMG",
                                     "Schur with an algebraic-multigrid pressure solve — the \
                                      same cpu::amg V-cycle the unstructured Schur uses on \
                                      the pressure Poisson."),
                                ] {
                                    if ui
                                        .radio(self.structured_precond == kind, label)
                                        .on_hover_text(hover)
                                        .clicked()
                                        && self.structured_precond != kind
                                    {
                                        chosen = Some(kind);
                                    }
                                }
                                if let Some(kind) = chosen {
                                    self.structured_precond = kind;
                                    self.init_solver();
                                }
                            } else {
                                // Compressible etc.: no Schur layout — block-Jacobi only.
                                ui.weak(
                                    "Structured banded solve: block-Jacobi (fixed — this \
                                     model declares no Schur layout).",
                                );
                            }
                        } else if model_owns_preconditioner {
                            ui.weak("Model-owned Schur: selector chooses pressure solve (Chebyshev vs AMG).");
                        } else {
                            ui.weak("Krylov preconditioner for the coupled linear solve.");
                        }
                        if supports_preconditioner {
                            if ui
                                .radio(
                                    matches!(self.selected_preconditioner, PreconditionerType::Jacobi),
                                    "Jacobi (diag)",
                                )
                                .clicked()
                            {
                                self.selected_preconditioner = PreconditionerType::Jacobi;
                                self.update_gpu_preconditioner();
                            }
                            ui.add_enabled_ui(block_jacobi_supported, |ui| {
                                if ui
                                    .radio(
                                        matches!(
                                            self.selected_preconditioner,
                                            PreconditionerType::BlockJacobi
                                        ),
                                        "BlockJacobi (cell block)",
                                    )
                                    .clicked()
                                {
                                    self.selected_preconditioner = PreconditionerType::BlockJacobi;
                                    self.update_gpu_preconditioner();
                                }
                            });
                            if !block_jacobi_supported {
                                ui.weak(format!(
                                    "BlockJacobi requires ≤{UI_MAX_BLOCK_JACOBI} unknowns/cell."
                                ));
                            }
                            if ui
                                .radio(
                                    matches!(self.selected_preconditioner, PreconditionerType::Amg),
                                    "AMG (Multigrid)",
                                )
                                .clicked()
                            {
                                self.selected_preconditioner = PreconditionerType::Amg;
                                self.update_gpu_preconditioner();
                            }
                        } else if !structured_precond {
                            ui.weak("(not declared by model)");
                        }

                        if self.model_caps.supports_alpha_u || self.model_caps.supports_alpha_p {
                            ui.separator();
                            ui.label("Under-Relaxation Factors");
                            let compressible_dual_time =
                                self.model_id == "compressible" && self.dual_time;
                            ui.weak(if compressible_dual_time {
                                "Tip: compressible dual-time now defaults to α_U=1.0; lower it or set nonconverged fallback damping only if pseudo-time steps stall."
                            } else {
                                "Tip: start with α_U≈0.7 and α_P≈0.3; α=1 can diverge at high Re."
                            });
                            if self.model_caps.supports_alpha_u
                                && ui
                                    .add(
                                        adaptive_slider(&mut self.alpha_u, 0.1..=1.0)
                                            .text("α_U (Velocity)"),
                                    )
                                    .changed()
                            {
                                self.update_gpu_alpha_u();
                            }
                            if self.model_caps.supports_alpha_p
                            {
                                if compressible_dual_time {
                                    if (self.alpha_p - 1.0).abs() > 1e-12 {
                                        self.alpha_p = 1.0;
                                        self.update_gpu_alpha_p();
                                    }
                                    ui.label("α_P (Pressure): 1.0 (locked)");
                                } else if ui
                                    .add(
                                        adaptive_slider(&mut self.alpha_p, 0.1..=1.0)
                                            .text("α_P (Pressure)"),
                                    )
                                    .changed()
                                {
                                    self.update_gpu_alpha_p();
                                }
                            }
                        }
                        });

                        ui.separator();
                        ui.label("Time Stepping Scheme");
                        egui::ComboBox::from_label("Time Scheme")
                            .selected_text(format!("{:?}", self.time_scheme))
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(
                                        &mut self.time_scheme,
                                        GpuTimeScheme::Euler,
                                        "Euler",
                                    )
                                    .clicked()
                                {
                                    self.update_gpu_time_scheme();
                                }
                                if ui
                                    .selectable_value(
                                        &mut self.time_scheme,
                                        GpuTimeScheme::BDF2,
                                        "BDF2",
                                    )
                                    .clicked()
                                {
                                    self.update_gpu_time_scheme();
                                }
                                let rk4_enabled = self.model_caps.supports_explicit_rk4
                                    && !self.enable_moving_mesh
                                    && !self.solver_is_moving;
                                let rk4 = ui.add_enabled(
                                    rk4_enabled,
                                    egui::Button::selectable(
                                        self.time_scheme == GpuTimeScheme::RK4,
                                        "RK4 (Explicit, adaptive CFL)",
                                    ),
                                );
                                if rk4.clicked() {
                                    self.time_scheme = GpuTimeScheme::RK4;
                                    self.adaptive_dt = true;
                                    self.dual_time = false;
                                    self.update_gpu_time_scheme();
                                }
                                if !rk4_enabled {
                                    rk4.on_disabled_hover_text(
                                        "Explicit RK4 requires a method-of-lines model and a static mesh.",
                                    );
                                }
                            });

                        ui.separator();
                        ui.label("Model");
                        let prev_model_id = self.model_id;
                        // Moving mesh keeps the user's model and runs it as its ALE
                        // variant. The dropdown stays LIVE while moving is enabled
                        // (the selection applies on Initialize / Reset like any
                        // model change); only entries WITHOUT an ALE variant
                        // (density-based `compressible`) are disabled, with the
                        // reason on hover — so switching e.g. incompressible →
                        // All-Mach thermal under ALE is one click, never a
                        // disable-retick-reenable dance.
                        let moving_active = self.enable_moving_mesh || self.solver_is_moving;
                        let ui_models = CFDApp::supported_ui_models(self.mesh_mode);
                        let cur_label = ui_models
                            .iter()
                            .find(|(id, _)| *id == self.model_id)
                            .map(|(_, l)| *l)
                            .unwrap_or_else(|| CFDApp::model_label(self.model_id));
                        egui::ComboBox::from_label("Model")
                            .selected_text(cur_label)
                            .show_ui(ui, |ui| {
                                for (id, label) in ui_models {
                                    // ALE has no structured variant; structured
                                    // mode is never `moving_active`.
                                    let selectable =
                                        !moving_active || CFDApp::ale_model_for(id).is_some();
                                    let entry = ui.add_enabled(
                                        selectable,
                                        egui::Button::selectable(self.model_id == id, label),
                                    );
                                    if entry.clicked() && selectable {
                                        self.model_id = id;
                                    }
                                    if !selectable {
                                        entry.on_disabled_hover_text(format!(
                                            "'{label}' has no moving-mesh (ALE) variant — \
                                             disable Moving Mesh (ALE) to select it.",
                                        ));
                                    }
                                }
                            });
                        if prev_model_id != self.model_id {
                            // Re-seed every solver knob from the per-model defaults
                            // (scheme, relaxation, outer cap, adaptive-dt target,
                            // low-Mach preconditioning, viscosity floor) so each
                            // model starts from settings that converge and do not
                            // diverge. `init_solver` refreshes the model caps.
                            self.apply_model_defaults();
                            self.init_solver();
                        }

                        ui.separator();
                        egui::ComboBox::from_label("Compute Backend")
                            .selected_text(self.backend.label())
                            .show_ui(ui, |ui| {
                                // Structured runs on the GPU or ANY CPU backend: the
                                // interpreter and the transpiled (compiled-Rust)
                                // kernels both execute the same Structured2D codegen
                                // IR the GPU lowers. (The SIMD-linear variant only
                                // affects the unstructured solve; on structured it is
                                // the same transpiled kernels + shared banded solve.)
                                for choice in BackendChoice::ALL {
                                    let entry = ui.add(egui::Button::selectable(
                                            self.backend == choice,
                                            choice.label(),
                                    ));
                                    if entry.clicked() {
                                        self.backend = choice;
                                    }
                                }
                            })
                            .response
                            .on_hover_text(
                                "CPU backends run the selected model without a GPU \
                                 adapter, at parity with the GPU; GPU-only telemetry \
                                 (profiling, per-graph timings) is unavailable. \
                                 Interpreter = reference tree-walker; Transpiled = \
                                 compiled kernels (fast); SIMD adds vectorized \
                                 block-matvec kernels plus mixed-precision (f32) \
                                 storage for the pressure inner solve and AMG \
                                 levels — a rounding-level result change, fastest \
                                 on solve-heavy runs.",
                            );
                        // Moving mesh runs on both backends: switching the compute
                        // backend keeps the moving-mesh selection intact — the next
                        // Initialize / Reset rebuilds on the chosen backend.
                        if self.backend.is_cpu() {
                            ui.add(
                                adaptive_slider(&mut self.cpu_threads, 1..=16).text("Cores"),
                            );
                            egui::ComboBox::from_label("Solver Precision")
                                .selected_text(if self.cpu_precision_f32 {
                                    "f32 (GPU-like)"
                                } else {
                                    "f64 (reference)"
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.cpu_precision_f32,
                                        false,
                                        "f64 (reference)",
                                    );
                                    ui.selectable_value(
                                        &mut self.cpu_precision_f32,
                                        true,
                                        "f32 (GPU-like)",
                                    );
                                })
                                .response
                                .on_hover_text(
                                    "Scalar precision of the coupled linear solve. f64 is \
                                     the reference; f32 mirrors the GPU's arithmetic and \
                                     halves solve bandwidth (results differ at rounding \
                                     level).",
                                );
                            ui.label("Applied on Initialize / Reset.");
                        }

                        ui.separator();
                        ui.label("Pressure-Velocity Coupling");
                        ui.label("Coupled solver (block system)");

                        if ui.button("Initialize / Reset").clicked() {
                            self.init_solver();
                        }
                    });

                    let has_solver = self.mesh.is_some();

                    if ui
                        .add_enabled(
                            has_solver && !is_initializing,
                            egui::Button::new(if self.is_running { "Pause" } else { "Run" }),
                        )
                        .clicked()
                    {
                        self.is_running = !self.is_running;

                        if self.is_running {
                            self.cached_error = None;
                            self.cached_message = None;
                            if matches!(self.render_mode, RenderMode::GpuDirect) {
                                if let Some(viz) = &self.viz_field {
                                    // Keep the UI repaint loop alive while the
                                    // worker drains initialization commands. A
                                    // second request is posted worker-side once
                                    // Run is actually accepted, so an idle
                                    // snapshot cannot consume the only demand.
                                    viz.mailbox.request_frame();
                                }
                            }
                            self.sync_worker_params();
                            self.set_worker_running(true);
                        } else {
                            self.set_worker_running(false);
                        }
                    }

                    ui.separator();

                    if let Some(err) = &self.cached_error {
                        ui.colored_label(egui::Color32::RED, err);
                    }
                    if let Some(message) = &self.cached_message {
                        ui.colored_label(egui::Color32::YELLOW, message);
                    }

                    if has_solver {
                        let stats = &self.cached_gpu_stats;
                        ui.label(format!("Completed simulation time: {:.6e} s", stats.sim_time));
                        if stats.dt.is_finite() && stats.dt > 0.0 {
                            ui.label(format!("Last accepted dt: {:.2e} s", stats.dt));
                        } else {
                            ui.label("Last accepted dt: measuring…");
                        }
                        if stats.linear_solves > 0 {
                            let status = if stats.linear_last.diverged
                                || !stats.linear_last.residual.is_finite()
                            {
                                "diverged"
                            } else if stats.linear_last.converged {
                                "converged"
                            } else {
                                "running"
                            };
                            ui.label(format!(
                                "Linear: {} solve(s), iters={} res={:.2e} ({})",
                                stats.linear_solves,
                                stats.linear_last.iterations,
                                stats.linear_last.residual,
                                status
                            ));
                        }
                        if stats.outer_iterations > 0 {
                            let status_suffix = stats
                                .outer_step_status
                                .map(|status| format!(" ({})", status.as_str()))
                                .unwrap_or_default();
                            ui.label(format!(
                                "Coupled: {} iters, U:{:.2e} P:{:.2e}{}",
                                stats.outer_iterations,
                                stats.outer_residual_u,
                                stats.outer_residual_p,
                                status_suffix,
                            ));
                        }
                        if stats.positivity_min_rho.is_some() || stats.positivity_min_p.is_some() {
                            ui.label(format!(
                                "Positivity: rho_min={:.2e} (n={}) p_min={:.2e} (n={})",
                                stats.positivity_min_rho.unwrap_or(f32::NAN),
                                stats.positivity_rho_undershoots,
                                stats.positivity_min_p.unwrap_or(f32::NAN),
                                stats.positivity_pressure_undershoots,
                            ));
                        }
                        let timing_response = if stats.steps_per_second > 0.0 {
                            ui.label(format!(
                                "Effective step wall time: {:.1} ms ({:.1} completed steps/s)",
                                stats.step_time_ms, stats.steps_per_second
                            ))
                        } else {
                            ui.label("Effective step wall time: measuring…")
                        };
                        timing_response.on_hover_text(
                            "Average wall time over the latest completion-to-completion worker \
                             window. GPU queue submission time is intentionally not shown; the \
                             sample includes work that must finish before the solution can be \
                             observed and adds no per-step synchronization.",
                        );
                        if stats.steps_per_second > 0.0 {
                            ui.label(format!(
                                "Simulation rate: {:.3e} simulated s / wall s",
                                stats.sim_seconds_per_wall_second
                            ));
                        }

                        // Moving-mesh (ALE) per-step telemetry, when active.
                        if let Some(m) = &self.cached_moving_stats {
                            ui.separator();
                            // Where THIS step's mesh was rebuilt — the
                            // requirement is that the reconstruction path is
                            // obvious, so it leads the ALE block.
                            let regen_line = match m.regen_backend {
                                RegenBackend::GpuFallback(reason) => {
                                    format!("Mesh regen: GPU→CPU this step ({reason})")
                                }
                                other => format!("Mesh regen: {}", other.label()),
                            };
                            ui.label(regen_line).on_hover_text(
                                "GPU on-device: Voronoi diagram, topology, geometry and ALE \
                                 swept fluxes are rebuilt entirely on the GPU every step. \
                                 CPU (per step): the CPU meshless engine re-assembles the \
                                 mesh from the advected seeds at compute time. GPU→CPU: the \
                                 device could not certify this step (Voronoi flip / sliver) \
                                 and the CPU path took it; the GPU is retried next step.",
                            );
                            ui.label(format!(
                                "ALE mesh: {} cells, {} faces{}{}{}{}",
                                m.n_cells,
                                m.n_faces,
                                if m.flipped {
                                    format!(
                                        " (flip: {} born / {} died, {} cells)",
                                        m.born_faces, m.died_faces, m.flipped_cells
                                    )
                                } else {
                                    String::new()
                                },
                                if m.recycled > 0 {
                                    format!(" (recycled {} seeds)", m.recycled)
                                } else {
                                    String::new()
                                },
                                if m.cells_born > 0 || m.cells_killed > 0 {
                                    format!(
                                        " (adapted: +{} / −{} cells, Δmass {:.1e}→{:.1e})",
                                        m.cells_born,
                                        m.cells_killed,
                                        m.transfer_defect_pre,
                                        m.transfer_defect_post
                                    )
                                } else if m.transfer_defect_pre > 0.0 {
                                    // Recycle-only transfer projection.
                                    format!(
                                        " (Δmass {:.1e}→{:.1e})",
                                        m.transfer_defect_pre, m.transfer_defect_post
                                    )
                                } else {
                                    String::new()
                                },
                                if m.at_adapt_budget {
                                    " (adapt budget reached)"
                                } else {
                                    ""
                                }
                            ));
                            ui.label(format!(
                                "ALE: dt={:.2e} skew={:.3} SCL={:.1e} flip_defect={:.1e}{}",
                                m.dt,
                                m.max_skew,
                                m.scl_defect,
                                m.flip_defect,
                                if m.motion_iters > 1 {
                                    format!(" motion_iters={}", m.motion_iters)
                                } else {
                                    String::new()
                                },
                            ));
                            ui.label(format!(
                                "ALE time (ms): plan={:.1} regen={:.1} swept={:.1} refresh={:.1}",
                                m.plan_ms, m.regen_ms, m.swept_ms, m.refresh_ms
                            ));
                        }
                    }
                });
        });

        let has_solver = self.mesh.is_some();

        if has_solver && matches!(self.render_mode, RenderMode::EguiPlot) {
            // The slow fallback intentionally retains the host cached_u/p
            // implementation. GPU-direct ranges come from the sequence-matched
            // reduction metadata below and never consult this cache.
            self.ensure_plot_cache(true);
        }

        let (min_val, max_val) = match self.render_mode {
            RenderMode::GpuDirect => self
                .cfd_renderer
                .as_ref()
                .map(|renderer| {
                    let range = renderer
                        .lock()
                        .unwrap()
                        .legend_range(self.render_range_field());
                    (range[0], range[1])
                })
                .unwrap_or((0.0, 1.0)),
            RenderMode::EguiPlot => self
                .plot_cache
                .as_ref()
                .map(|cache| (cache.min as f32, cache.max as f32))
                .unwrap_or((0.0, 1.0)),
        };

        self.render_right_panel(root_ui, has_solver, min_val, max_val);
        self.render_bottom_panel(root_ui);
        self.render_central_panel(root_ui, is_initializing, has_solver, min_val, max_val);

        // Schedule after controls and Direct activation have run. In
        // particular, a paused Plot -> Direct transition and the first Run
        // click both happen later than the old start-of-update decision.
        let direct_refresh_pending = matches!(self.render_mode, RenderMode::GpuDirect)
            && self
                .viz_field
                .as_ref()
                .is_some_and(|viz| viz.mailbox.presentation_refresh_pending());
        if self.is_running || is_initializing || direct_refresh_pending {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}

fn trace_runtime_params_from_worker(params: RuntimeParams) -> tracefmt::TraceRuntimeParams {
    tracefmt::TraceRuntimeParams {
        adaptive_dt: params.adaptive_dt,
        target_cfl: params.target_cfl,
        requested_dt: params.requested_dt,
        dtau: params.dtau,
        advection_scheme: tracefmt::TraceScheme::from(params.advection_scheme),
        time_scheme: tracefmt::TraceTimeScheme::from(params.time_scheme),
        preconditioner: tracefmt::TracePreconditioner::from(params.preconditioner),
        outer_iters: params.outer_iters,
        low_mach_model: tracefmt::TraceLowMachModel::from(params.low_mach_model),
        low_mach_theta_floor: params.low_mach_theta_floor,
        low_mach_pressure_coupling_alpha: params.low_mach_pressure_coupling_alpha,
        alpha_u: params.alpha_u,
        alpha_p: params.alpha_p,
        inlet_velocity: params.inlet_velocity,
        density: params.density,
        viscosity: params.viscosity,
        eos: tracefmt::TraceEosSpec::from(params.eos),
    }
}

fn solver_worker_stop_trace(trace: &mut Option<SolverTraceSession>, mode: &mut Option<SolverMode>) {
    let Some(mut session) = trace.take() else {
        return;
    };

    if let Some(s) = mode.as_mut().and_then(|m| m.unified_solver_mut()) {
        s.set_collect_trace(false);
        let _ = s.enable_detailed_profiling(false);
        if session.profiling_enabled {
            let _ = s.end_profiling_session();
            if let Ok(stats) = s.get_profiling_stats() {
                let categories = stats
                    .get_all_stats()
                    .into_iter()
                    .map(|(category, s)| {
                        (
                            category.name().to_string(),
                            tracefmt::TraceCategoryStats {
                                total_seconds: s.total_time.as_secs_f64(),
                                call_count: s.call_count,
                                min_seconds: s.min_time.as_secs_f64(),
                                max_seconds: s.max_time.as_secs_f64(),
                                total_bytes: s.total_bytes,
                            },
                        )
                    })
                    .collect::<Vec<_>>();

                let mut locations = stats.get_location_stats();
                locations.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

                let locations = locations
                    .into_iter()
                    .take(200)
                    .map(|(name, s)| {
                        (
                            name,
                            tracefmt::TraceCategoryStats {
                                total_seconds: s.total_time.as_secs_f64(),
                                call_count: s.call_count,
                                min_seconds: s.min_time.as_secs_f64(),
                                max_seconds: s.max_time.as_secs_f64(),
                                total_bytes: s.total_bytes,
                            },
                        )
                    })
                    .collect();

                let (mem_cpu, mem_gpu) = stats.get_memory_stats();
                let memory_cpu = tracefmt::TraceMemoryStats {
                    alloc_bytes: mem_cpu.alloc_bytes,
                    alloc_count: mem_cpu.alloc_count,
                    free_bytes: mem_cpu.free_bytes,
                    free_count: mem_cpu.free_count,
                    max_alloc_request: mem_cpu.max_alloc_request,
                };
                let memory_gpu = tracefmt::TraceMemoryStats {
                    alloc_bytes: mem_gpu.alloc_bytes,
                    alloc_count: mem_gpu.alloc_count,
                    free_bytes: mem_gpu.free_bytes,
                    free_count: mem_gpu.free_count,
                    max_alloc_request: mem_gpu.max_alloc_request,
                };

                let memory_locations_cpu = stats
                    .get_memory_location_stats(crate::solver::gpu::profiling::MemoryDomain::Cpu)
                    .into_iter()
                    .map(|(name, s)| {
                        (
                            name,
                            tracefmt::TraceMemoryStats {
                                alloc_bytes: s.alloc_bytes,
                                alloc_count: s.alloc_count,
                                free_bytes: s.free_bytes,
                                free_count: s.free_count,
                                max_alloc_request: s.max_alloc_request,
                            },
                        )
                    })
                    .collect();
                let memory_locations_gpu = stats
                    .get_memory_location_stats(crate::solver::gpu::profiling::MemoryDomain::Gpu)
                    .into_iter()
                    .map(|(name, s)| {
                        (
                            name,
                            tracefmt::TraceMemoryStats {
                                alloc_bytes: s.alloc_bytes,
                                alloc_count: s.alloc_count,
                                free_bytes: s.free_bytes,
                                free_count: s.free_count,
                                max_alloc_request: s.max_alloc_request,
                            },
                        )
                    })
                    .collect();

                let profiling = tracefmt::TraceProfilingEvent {
                    session_total_seconds: stats.get_session_total().as_secs_f64(),
                    iteration_count: stats.get_iteration_count(),
                    categories,
                    locations,
                    memory_cpu,
                    memory_gpu,
                    memory_locations_cpu,
                    memory_locations_gpu,
                };

                let _ = session
                    .writer
                    .write_event(&tracefmt::TraceEvent::Profiling(profiling));
            }
        }
    }

    let _ = session
        .writer
        .write_event(&tracefmt::TraceEvent::Footer(tracefmt::TraceFooter {
            closed_unix_ms: tracefmt::now_unix_ms(),
        }));

    let path = session.writer.path().display().to_string();
    let _ = session.writer.close();
    eprintln!("[cfd2][trace] closed {}", path);
}

// One implicit step of the structured GPU solver synthesises the same
// `StepOutcome`/`Readback`/`FieldStats` shape the worker consumes from the
// unstructured `SolverDriver::step`. The step + readback surface is shared by
// the GPU and CPU structured solvers; `st_` avoids inherent-method collisions.
#[cfg(test)]
thread_local! {
    static STRUCTURED_PACKED_STATE_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
fn note_structured_packed_state_read() {
    #[cfg(test)]
    STRUCTURED_PACKED_STATE_READS.with(|count| count.set(count.get() + 1));
}

trait StructuredSteppable {
    fn st_step(&mut self);
    fn st_dt(&self) -> f64;
    fn st_set_dt(&mut self, dt: f64);
    fn st_set_fluid(&mut self, density: f64, viscosity: f64);
    fn st_set_alpha_u(&mut self, alpha_u: f32);
    fn st_set_alpha_p(&mut self, alpha_p: f32);
    fn st_set_inlet_ramp(&mut self, velocity: f32, duration: f32);
    /// Min Cartesian spacing `min(dx, dy)` — the structured analogue of the
    /// driver's mesh-derived `min_cell_size` used by adaptive CFL.
    fn st_min_cell_size(&self) -> f64;
    /// Square-root cell area, matching the generic driver's `sqrt(cell_vol)`
    /// inlet-ramp metric on an equivalent Cartesian mesh.
    fn st_inlet_ramp_cell_size(&self) -> f64;
    fn st_grid(&self) -> StructuredGrid;
    fn st_geometry_rates(&self) -> (f64, f64);
    fn st_layout(&self) -> &crate::solver::model::backend::state_layout::StateLayout;
    fn st_packed_state(&self) -> Vec<f32>;
    /// Convergence telemetry from the most recent `st_step` — plumbed into the
    /// GUI's "Linear"/"Coupled" readout via `structured_step`.
    fn st_stats(&self) -> crate::solver::banded_schur::StructuredStepStats;
}
impl StructuredSteppable for StructuredGpuSolver {
    fn st_step(&mut self) {
        self.step()
    }
    fn st_dt(&self) -> f64 {
        self.dt()
    }
    fn st_set_dt(&mut self, dt: f64) {
        self.set_dt(dt)
    }
    fn st_set_fluid(&mut self, density: f64, viscosity: f64) {
        self.set_fluid(density, viscosity)
    }
    fn st_set_alpha_u(&mut self, alpha_u: f32) {
        self.set_alpha_u(alpha_u)
    }
    fn st_set_alpha_p(&mut self, alpha_p: f32) {
        self.set_alpha_p(alpha_p)
    }
    fn st_set_inlet_ramp(&mut self, velocity: f32, duration: f32) {
        self.set_inlet_ramp(velocity, duration)
    }
    fn st_min_cell_size(&self) -> f64 {
        self.grid().dx.min(self.grid().dy)
    }
    fn st_inlet_ramp_cell_size(&self) -> f64 {
        (self.grid().dx * self.grid().dy).sqrt()
    }
    fn st_grid(&self) -> StructuredGrid {
        self.grid()
    }
    fn st_geometry_rates(&self) -> (f64, f64) {
        let grid = self.grid();
        (
            1.0 / grid.dx + 1.0 / grid.dy,
            2.0 / (grid.dx * grid.dx) + 2.0 / (grid.dy * grid.dy),
        )
    }
    fn st_layout(&self) -> &crate::solver::model::backend::state_layout::StateLayout {
        self.state_layout()
    }
    fn st_packed_state(&self) -> Vec<f32> {
        note_structured_packed_state_read();
        self.packed_state_f32()
    }
    fn st_stats(&self) -> crate::solver::banded_schur::StructuredStepStats {
        self.last_stats()
    }
}
impl StructuredSteppable for StructuredModelSolver {
    fn st_step(&mut self) {
        self.step()
    }
    fn st_dt(&self) -> f64 {
        self.dt()
    }
    fn st_set_dt(&mut self, dt: f64) {
        self.set_dt(dt)
    }
    fn st_set_fluid(&mut self, density: f64, viscosity: f64) {
        self.set_fluid(density, viscosity)
    }
    fn st_set_alpha_u(&mut self, alpha_u: f32) {
        self.set_alpha_u(alpha_u)
    }
    fn st_set_alpha_p(&mut self, alpha_p: f32) {
        self.set_alpha_p(alpha_p)
    }
    fn st_set_inlet_ramp(&mut self, velocity: f32, duration: f32) {
        self.set_inlet_ramp(velocity, duration)
    }
    fn st_min_cell_size(&self) -> f64 {
        self.grid().dx.min(self.grid().dy)
    }
    fn st_inlet_ramp_cell_size(&self) -> f64 {
        (self.grid().dx * self.grid().dy).sqrt()
    }
    fn st_grid(&self) -> StructuredGrid {
        self.grid()
    }
    fn st_geometry_rates(&self) -> (f64, f64) {
        let grid = self.grid();
        (
            1.0 / grid.dx + 1.0 / grid.dy,
            2.0 / (grid.dx * grid.dx) + 2.0 / (grid.dy * grid.dy),
        )
    }
    fn st_layout(&self) -> &crate::solver::model::backend::state_layout::StateLayout {
        self.state_layout()
    }
    fn st_packed_state(&self) -> Vec<f32> {
        note_structured_packed_state_read();
        self.packed_state_f32()
    }
    fn st_stats(&self) -> crate::solver::banded_schur::StructuredStepStats {
        self.last_stats()
    }
}

/// Magnitude of the structured GUI's Brinkman momentum sink. All-Mach RK4
/// enforces the solid velocity algebraically after each stage; the density-based
/// explicit path still uses this value for its reaction-rate bound.
const STRUCTURED_IBM_PENALTY_RATE: f64 = 1.0e5;

/// Conservative extent of classical RK4's stability interval on the negative
/// real axis (the exact endpoint is about 2.785).
const RK4_NEGATIVE_REAL_SAFETY: f64 = 2.5;

const STRUCTURED_EXPLICIT_RK_SAFETY: f64 = 0.8;

#[derive(Clone, Copy, Default)]
struct StructuredExplicitSample {
    max_rate: f64,
    max_vel: f64,
    invalid: usize,
    rho_min: f64,
    rho_max: f64,
}

fn structured_rhie_chow_turnover_rate(
    grid: StructuredGrid,
    pressure: &[f64],
    density: &[f64],
    d_p: &[f64],
    penalty: &[f64],
    pressure_inlet: bool,
    inlet_pressure: f64,
    outlet_pressure: f64,
) -> f64 {
    let cells = grid.num_cells();
    if pressure.len() != cells || density.len() != cells || d_p.len() != cells {
        return f64::INFINITY;
    }
    let p_at = |i: usize, j: usize| pressure[j * grid.nx + i];
    let mut gradient = vec![(0.0, 0.0); cells];
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let cell = j * grid.nx + i;
            let p_w = if i > 0 {
                0.5 * (pressure[cell] + p_at(i - 1, j))
            } else if pressure_inlet {
                inlet_pressure
            } else {
                pressure[cell]
            };
            let p_e = if i + 1 < grid.nx {
                0.5 * (pressure[cell] + p_at(i + 1, j))
            } else if pressure_inlet {
                pressure[cell]
            } else {
                outlet_pressure
            };
            let p_s = if j > 0 {
                0.5 * (pressure[cell] + p_at(i, j - 1))
            } else {
                pressure[cell]
            };
            let p_n = if j + 1 < grid.ny {
                0.5 * (pressure[cell] + p_at(i, j + 1))
            } else {
                pressure[cell]
            };
            gradient[cell] = ((p_e - p_w) / grid.dx, (p_n - p_s) / grid.dy);
        }
    }

    let ibm = !penalty.is_empty();
    let mut row_sum = vec![0.0; cells];
    let mut add_face = |a: usize, b: Option<usize>, flux: f64| {
        let magnitude = flux.abs();
        row_sum[a] += magnitude;
        if let Some(b) = b {
            row_sum[b] += magnitude;
        }
    };
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let cell = j * grid.nx + i;
            if i + 1 < grid.nx {
                let other = cell + 1;
                let seal = if ibm {
                    1.0 - (penalty[cell] + penalty[other]).min(1.0)
                } else {
                    1.0
                };
                let (kappa, qx) = if ibm {
                    let kappa = 0.5 * (density[cell] + density[other]) * d_p[cell].min(d_p[other]);
                    (kappa, kappa * 0.5 * (gradient[cell].0 + gradient[other].0))
                } else {
                    let ka = density[cell] * d_p[cell];
                    let kb = density[other] * d_p[other];
                    (
                        0.5 * (ka + kb),
                        0.5 * (ka * gradient[cell].0 + kb * gradient[other].0),
                    )
                };
                let compact = kappa * (pressure[other] - pressure[cell]) / grid.dx;
                add_face(cell, Some(other), seal * grid.dy * (qx - compact));
            }
            if j + 1 < grid.ny {
                let other = cell + grid.nx;
                let seal = if ibm {
                    1.0 - (penalty[cell] + penalty[other]).min(1.0)
                } else {
                    1.0
                };
                let (kappa, qy) = if ibm {
                    let kappa = 0.5 * (density[cell] + density[other]) * d_p[cell].min(d_p[other]);
                    (kappa, kappa * 0.5 * (gradient[cell].1 + gradient[other].1))
                } else {
                    let ka = density[cell] * d_p[cell];
                    let kb = density[other] * d_p[other];
                    (
                        0.5 * (ka + kb),
                        0.5 * (ka * gradient[cell].1 + kb * gradient[other].1),
                    )
                };
                let compact = kappa * (pressure[other] - pressure[cell]) / grid.dy;
                add_face(cell, Some(other), seal * grid.dx * (qy - compact));
            }

            // The only pressure-Dirichlet outer boundary is left for the
            // pressure-inlet nozzle and right for the ordinary velocity inlet.
            // Zero-Neumann boundaries have identically zero corrected RC flux.
            let boundary = if pressure_inlet && i == 0 {
                Some((-gradient[cell].0, inlet_pressure, grid.dy, 0.5 * grid.dx))
            } else if !pressure_inlet && i + 1 == grid.nx {
                Some((gradient[cell].0, outlet_pressure, grid.dy, 0.5 * grid.dx))
            } else {
                None
            };
            if let Some((normal_gradient, boundary_p, area, distance)) = boundary {
                let kappa = density[cell] * d_p[cell];
                let compact = kappa * (boundary_p - pressure[cell]) / distance;
                let seal = if ibm {
                    1.0 - (2.0 * penalty[cell]).min(1.0)
                } else {
                    1.0
                };
                add_face(
                    cell,
                    None,
                    seal * area * (kappa * normal_gradient - compact),
                );
            }
        }
    }

    let volume = grid.dx * grid.dy;
    row_sum
        .iter()
        .enumerate()
        .map(|(cell, &sum)| sum / (density[cell] * volume))
        .fold(0.0, f64::max)
}

fn structured_allmach_explicit_sample_from_state(
    s: &impl StructuredSteppable,
    params: &RuntimeParams,
    state: &[f32],
) -> StructuredExplicitSample {
    let layout = s.st_layout();
    let stride = layout.stride() as usize;
    let off = |name: &str| layout.offset_for(name).map(|value| value as usize);
    let (Some(u), Some(p), Some(t), Some(t_ref), Some(dt_local)) =
        (off("U"), off("p"), off("T"), off("t_ref"), off("dt_local"))
    else {
        return StructuredExplicitSample {
            invalid: 1,
            rho_min: f64::INFINITY,
            rho_max: f64::NEG_INFINITY,
            ..Default::default()
        };
    };
    let ibm_penalty = off("ibm_penalty_U");
    let (g_h, g_d) = s.st_geometry_rates();
    let psi0 = (params.compressibility_psi as f64).max(0.0);
    let u_ref = 2.0
        * (params.inlet_velocity.abs() as f64).max(params.allmach_precond_uref_min.max(0.0) as f64);
    let mut sample = StructuredExplicitSample {
        rho_min: f64::INFINITY,
        rho_max: f64::NEG_INFINITY,
        ..Default::default()
    };
    let grid = s.st_grid();
    let mut pressure = vec![0.0; grid.num_cells()];
    let mut density_cells = vec![0.0; grid.num_cells()];
    let mut dp_cells = vec![0.0; grid.num_cells()];
    let mut penalty_cells = ibm_penalty.map(|_| vec![0.0; grid.num_cells()]);
    for (cell, row) in state.chunks_exact(stride).enumerate() {
        if row.iter().any(|value| !value.is_finite()) {
            sample.invalid += 1;
            continue;
        }
        let speed = (row[u] as f64).hypot(row[u + 1] as f64);
        let temperature = row[t] as f64;
        if !(temperature > 0.0) {
            sample.invalid += 1;
            continue;
        }
        let reference_t = row[t_ref] as f64;
        let rho_numer = params.density as f64 * reference_t
            + crate::solver::model::ALLMACH_GAMMA * psi0 * reference_t * row[p] as f64;
        let rho_raw = rho_numer / temperature;
        let density_floor = psi0 * 1.0e-5;
        let density = rho_raw.max(density_floor);
        pressure[cell] = row[p] as f64;
        density_cells[cell] = density;
        let rho_dt = -rho_numer / (temperature * temperature);
        let psi_local = (psi0 * reference_t / temperature).max(psi0);
        let beta2 = (speed * speed).max(u_ref * u_ref).max(1.0e-12);
        let mass_pp = if psi0 > 0.0 {
            (crate::solver::model::ALLMACH_GAMMA - 1.0) * psi0 * reference_t / temperature
                + psi_local.max(1.0 / beta2)
        } else {
            0.0
        };
        let inv_cp = (crate::solver::model::ALLMACH_GAMMA - 1.0) * psi0 * reference_t;
        let chi = mass_pp + rho_dt * inv_cp / density.max(1.0e-30);
        let chi_floor = mass_pp.abs().max(1.0e-30) * 1.0e-6;
        if !(density > 0.0
            && mass_pp > 0.0
            && chi.is_finite()
            && chi > chi_floor
            && rho_raw > density_floor * (1.0 + 1.0e-6))
        {
            sample.invalid += 1;
            continue;
        }
        sample.max_vel = sample.max_vel.max(speed);
        sample.rho_min = sample.rho_min.min(density);
        sample.rho_max = sample.rho_max.max(density);
        let sound = 1.0 / chi.sqrt();
        let nu_long = 4.0 * (params.viscosity as f64).abs() / (3.0 * density);
        let alpha_t = crate::solver::model::ALLMACH_K_OVER_CP * mass_pp / (density * chi);
        let dp0 = (row[dt_local] as f64).max(0.0) / density;
        let penalty = ibm_penalty.map_or(0.0, |field| (row[field] as f64).abs());
        let d_p = dp0 / (1.0 + penalty * dp0);
        dp_cells[cell] = d_p;
        if let Some(values) = &mut penalty_cells {
            values[cell] = penalty;
        }
        let alpha_p = density * d_p.abs() / chi;
        let rate = g_h * (speed + sound) + 2.0 * g_d * nu_long.max(alpha_t).max(alpha_p);
        if rate.is_finite() {
            sample.max_rate = sample.max_rate.max(rate);
        } else {
            sample.invalid += 1;
        }
    }
    if sample.invalid == 0 {
        sample.max_rate += structured_rhie_chow_turnover_rate(
            grid,
            &pressure,
            &density_cells,
            &dp_cells,
            penalty_cells.as_deref().unwrap_or(&[]),
            params.pressure_inlet,
            params.inlet_pressure as f64,
            params.outlet_back_pressure as f64,
        );
    }
    sample
}

fn structured_allmach_explicit_sample(
    s: &impl StructuredSteppable,
    params: &RuntimeParams,
) -> StructuredExplicitSample {
    let state = s.st_packed_state();
    structured_allmach_explicit_sample_from_state(s, params, &state)
}

/// Pin `dt` for one structured step — the same CFL policy as
/// [`crate::sim::SolverDriver::step`].
fn structured_pin_dt(
    s: &mut impl StructuredSteppable,
    params: &RuntimeParams,
    prev_max_vel: &mut f64,
    prev_explicit_rate: &mut Option<f64>,
    supports_sound_speed: bool,
    model_id: &str,
) -> Result<(), String> {
    if params.adaptive_dt {
        if params.time_scheme == GpuTimeScheme::RK4 && model_id == "allmach_thermal_structured" {
            let rate = match prev_explicit_rate.filter(|rate| rate.is_finite() && *rate > 0.0) {
                Some(rate) => rate,
                None => {
                    // Bootstrap once after construction/a live parameter edit. Every
                    // accepted RK4 step caches its post-step sample below, so the normal
                    // path does not perform a second pre-step GPU readback.
                    let sample = structured_allmach_explicit_sample(s, params);
                    if sample.invalid > 0
                        || !sample.max_rate.is_finite()
                        || !(sample.max_rate > 0.0)
                    {
                        return Err(format!(
                            "invalid structured all-Mach RK4 pre-step state ({} invalid cells)",
                            sample.invalid
                        ));
                    }
                    *prev_max_vel = sample.max_vel;
                    *prev_explicit_rate = Some(sample.max_rate);
                    sample.max_rate
                }
            };
            let mut next_dt =
                STRUCTURED_EXPLICIT_RK_SAFETY * params.target_cfl.clamp(1.0e-6, 1.0) / rate;
            let current_dt = s.st_dt();
            if next_dt > current_dt * 1.2 {
                next_dt = current_dt * 1.2;
            }
            if next_dt.is_finite() && next_dt > 0.0 {
                s.st_set_dt(next_dt.min(100.0));
            }
            return Ok(());
        }
        let sound_speed = if supports_sound_speed {
            params.eos.sound_speed(params.density as f64)
        } else {
            0.0
        };
        let adv_speed = (*prev_max_vel).max(params.inlet_velocity.abs() as f64);
        let effective_sound_speed = match params.low_mach_model {
            GpuLowMachPrecondModel::Off => sound_speed,
            GpuLowMachPrecondModel::Legacy => sound_speed.min(adv_speed),
            GpuLowMachPrecondModel::WeissSmith => {
                let theta = (params.low_mach_theta_floor as f64).max(0.0);
                let c_floor = sound_speed * theta.sqrt();
                sound_speed.min(adv_speed.max(c_floor))
            }
        };
        // The all-Mach Turkel pseudo-sound-speed β = k·max(U_in, uref_min) does NOT
        // enter the CFL — see the derivation on `SolverDriver::step`. The pressure
        // row is solved implicitly, so pinning the pseudo-acoustic Courant number to
        // target_cfl only makes the acoustic mass term `psi_precond·V/dt` comparable
        // to the pressure Laplacian (R = 1/(4·α_u·CFL_β²)), which de-ellipticises the
        // pressure and stalls/destabilises the flow.
        // Density-based RK4 integrates physical acoustics directly, so its
        // acoustic CFL uses the true sound speed. Pressure-based all-Mach RK4
        // returned above after using its local p/T mass-block spectral rate.
        let acoustic_speed = if params.time_scheme == GpuTimeScheme::RK4 {
            sound_speed
        } else {
            effective_sound_speed
        };
        let wave_speed = adv_speed + acoustic_speed;
        let min_h = s.st_min_cell_size();
        let mut stable_dt = if min_h > 1e-12 && wave_speed.is_finite() && wave_speed > 1e-12 {
            Some(params.target_cfl * min_h / wave_speed)
        } else {
            None
        };
        if params.time_scheme == GpuTimeScheme::RK4 && min_h > 1e-12 {
            let rho = (params.density as f64).abs().max(1.0e-12);
            let mut alpha = (params.viscosity as f64).abs() / rho;
            if model_id.contains("thermal") {
                alpha = alpha.max(crate::solver::model::ALLMACH_K_OVER_CP / rho);
            }
            if alpha.is_finite() && alpha > 1.0e-14 {
                let diffusion_dt = 0.25 * params.target_cfl * min_h * min_h / alpha;
                stable_dt = Some(stable_dt.map_or(diffusion_dt, |dt| dt.min(diffusion_dt)));
            }

            // The structured density-based model carries a -1e5 1/s Brinkman
            // reaction in immersed-solid cells, but the RK4 stage kernels
            // enforce the solid as an algebraic momentum projection
            // (rho_u = 0 at every abscissa), so the penalty never enters the
            // explicit stability spectrum and no reaction dt bound applies.
        }
        if let Some(mut next_dt) = stable_dt {
            let current_dt = s.st_dt();
            if next_dt > current_dt * 1.2 {
                next_dt = current_dt * 1.2;
            }
            next_dt = next_dt.clamp(1e-9, 100.0);
            s.st_set_dt(next_dt);
        }
    } else {
        s.st_set_dt(params.requested_dt.max(1.0e-9) as f64);
    }
    Ok(())
}

/// Whether this structured model carries a thermodynamic EOS that contributes
/// physical sound speed to the adaptive CFL (density-based compressible only).
fn structured_supports_sound_speed(model_id: &str) -> bool {
    model_id == "compressible_structured" || model_id == "compressible"
}

fn structured_step(
    s: &mut (impl StructuredSteppable + StructuredSeed),
    params: &RuntimeParams,
    prev_max_vel: &mut f64,
    prev_explicit_rate: &mut Option<f64>,
    model_id: &str,
    readback: bool,
) -> StepOutcome {
    if let Err(error) = structured_pin_dt(
        s,
        params,
        prev_max_vel,
        prev_explicit_rate,
        structured_supports_sound_speed(model_id),
        model_id,
    ) {
        return StepOutcome {
            dt: s.st_dt() as f32,
            step_time_ms: 0.0,
            linear_stats: Vec::new(),
            outer_iters: None,
            outer_residual_u: None,
            outer_residual_p: None,
            diverged: Some(DivergeReason::StepError(error)),
            should_stop: false,
            readback: None,
        };
    }

    if model_id == "allmach_thermal_structured" {
        apply_structured_allmach_inlet_ramp(s, params);
    }

    let t0 = std::time::Instant::now();
    s.st_step();
    let solver_step_time_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let dt = s.st_dt() as f32;
    let cstats = s.st_stats();
    let ports = UiPortSet::from_layout(s.st_layout());
    let stride = ports.stride as usize;
    let explicit_rk4 = params.time_scheme == GpuTimeScheme::RK4;
    // When adaptive CFL is on, refresh `prev_max_vel` EVERY step (not only the
    // throttled GUI snapshot cadence). A stale velocity scale lets dt lag a
    // developing jet/wake and overshoot the true Courant number for several
    // steps. Explicit RK4 also has no linear solve whose residual can carry a
    // divergence bit, so its accepted state is inspected every step.
    //
    // Read the packed state ONCE and derive every U/p/sample view from it. The
    // GPU read is already a completion fence for all four RK stages; the old
    // path independently read the same full buffer for the all-Mach sample, U,
    // and p (and once more before adaptive steps), making the displayed enqueue
    // time look fast while three/four blocking transfers controlled cadence.
    let need_u = readback || params.adaptive_dt || explicit_rk4;
    let need_p = readback || explicit_rk4;
    let packed_state = (need_u || need_p).then(|| s.st_packed_state());

    let explicit_allmach_sample = if explicit_rk4 && model_id == "allmach_thermal_structured" {
        packed_state
            .as_deref()
            .map(|state| structured_allmach_explicit_sample_from_state(s, params, state))
    } else {
        None
    };
    if let Some(sample) = explicit_allmach_sample {
        if sample.invalid == 0 && sample.max_rate.is_finite() && sample.max_rate > 0.0 {
            *prev_max_vel = sample.max_vel;
            *prev_explicit_rate = Some(sample.max_rate);
        } else {
            *prev_explicit_rate = None;
        }
    }

    let rows = packed_state
        .as_deref()
        .filter(|_| stride > 0)
        .map(|state| state.chunks_exact(stride));
    let cell_count = rows.as_ref().map_or(0, |rows| rows.len());
    let mut u = if readback {
        Vec::with_capacity(cell_count)
    } else {
        Vec::new()
    };
    let mut p = if readback {
        Vec::with_capacity(cell_count)
    } else {
        Vec::new()
    };
    let mut max_vel = 0.0f64;
    let mut step_nonfinite_u = 0usize;
    let mut step_nonfinite_p = 0usize;
    let mut p_min = f64::INFINITY;
    let mut p_max = f64::NEG_INFINITY;
    if let Some(rows) = rows {
        for row in rows {
            if need_u {
                if let Some(off) = ports.u_offset.map(|off| off as usize) {
                    let ux = row[off] as f64;
                    let uy = row[off + 1] as f64;
                    if readback {
                        u.push((ux, uy));
                    }
                    if ux.is_finite() && uy.is_finite() {
                        max_vel = max_vel.max(ux.hypot(uy));
                    } else {
                        step_nonfinite_u += 1;
                    }
                }
            }
            if need_p {
                if let Some(off) = ports.p_offset.map(|off| off as usize) {
                    let value = row[off] as f64;
                    if readback {
                        p.push(value);
                    }
                    if value.is_finite() {
                        p_min = p_min.min(value);
                        p_max = p_max.max(value);
                    } else {
                        step_nonfinite_p += 1;
                    }
                }
            }
        }
    }
    if need_u {
        if params.adaptive_dt || readback {
            *prev_max_vel = max_vel;
        }
    }

    // At this point the accepted GPU step and all mandatory safety/CFL work are
    // complete. This is the meaningful per-step latency; optional UI event
    // construction below performs no further device transfer.
    let step_time_ms = if explicit_rk4 || params.adaptive_dt {
        // The packed-state audit/CFL scan is mandatory accepted-step work and
        // provides the explicit GPU completion fence.
        t0.elapsed().as_secs_f32() * 1000.0
    } else {
        // A fixed implicit snapshot is optional GUI observation. Keep it out of
        // the step-local trace even though it reuses the same packed transfer.
        solver_step_time_ms
    };

    let readback = if readback {
        let p_finite = step_nonfinite_p == 0 && !p.is_empty();
        let stats = FieldStats {
            max_vel,
            nonfinite_u: step_nonfinite_u,
            p_min: if p_finite { p_min } else { 0.0 },
            p_max: if p_finite { p_max } else { 0.0 },
            p_finite,
            nonfinite_p: step_nonfinite_p,
            rho: explicit_allmach_sample.map(|sample| (sample.rho_min, sample.rho_max)),
        };
        Some(Readback { u, p, stats })
    } else {
        None
    };

    let (nonfinite_u, nonfinite_p) = readback
        .as_ref()
        .map_or((step_nonfinite_u, step_nonfinite_p), |rb| {
            (rb.stats.nonfinite_u, rb.stats.nonfinite_p)
        });
    let mut diverged = (nonfinite_u > 0 || nonfinite_p > 0).then_some(DivergeReason::NonFinite {
        u: nonfinite_u,
        p: nonfinite_p,
    });
    if let Some(sample) = explicit_allmach_sample {
        if sample.invalid > 0 || !(sample.max_rate > 0.0) {
            diverged = Some(DivergeReason::StepError(format!(
                "invalid structured all-Mach RK4 thermodynamic state ({} invalid cells)",
                sample.invalid
            )));
        }
    }

    // Convergence telemetry for the GUI readout. The banded solve is a direct
    // f64 GMRES per outer, so we report the LAST outer's linear residual as a
    // single [`LinearSolverStats`] (drives the "Linear:" line) plus the coupled
    // (Picard) increment residuals as the "Coupled: U/P" line — matching the
    // unstructured `UnifiedSolver` reporting the worker reads elsewhere.
    let linear_stats = if cstats.outer_iters > 0 {
        vec![LinearSolverStats {
            iterations: cstats.linear_iters,
            residual: cstats.linear_res,
            converged: cstats.linear_res <= crate::solver::banded_schur::default_step_tol() as f32,
            diverged: !cstats.linear_res.is_finite(),
            time: std::time::Duration::ZERO,
        }]
    } else {
        Vec::new()
    };

    StepOutcome {
        dt,
        step_time_ms,
        linear_stats,
        outer_iters: Some(cstats.outer_iters),
        outer_residual_u: Some(cstats.outer_du),
        outer_residual_p: Some(cstats.outer_dp),
        diverged,
        should_stop: false,
        readback,
    }
}

/// Resolve the `TopologyMode::Structured2D` model for a structured model id.
fn structured_model_by_id(id: &str) -> Result<ModelSpec, String> {
    Ok(match id {
        "incompressible_momentum_structured" => {
            crate::solver::model::incompressible_momentum_structured_model()?
        }
        "allmach_thermal_structured" => crate::solver::model::allmach_thermal_structured_model()?,
        "compressible_structured" => crate::solver::model::compressible_structured_model()?,
        other => return Err(format!("no structured model for id '{other}'")),
    })
}

/// Resolve a static (non-ALE) unstructured physical model by id. Dispatches to
/// the individual builders — never through `all_models()`, which pays for every
/// registered model (~0.5 s, dominated by compressible flux-module generation).
fn unstructured_model_by_id(
    id: &str,
    eos: crate::solver::model::eos::EosSpec,
) -> Result<ModelSpec, String> {
    match id {
        "incompressible_momentum" => incompressible_momentum_model(),
        "compressible" => compressible_model_with_eos(eos),
        "allmach_pressure" => allmach_pressure_model(),
        "allmach_thermal" => allmach_thermal_model(),
        other => Err(format!("unknown model id '{other}'")),
    }
}

/// Resolve a moving-mesh (ALE) model by id. Same direct-dispatch rule as
/// [`unstructured_model_by_id`] — only the three ALE variants the GUI can select.
fn ale_model_by_id(id: &str) -> Result<ModelSpec, String> {
    match id {
        "incompressible_momentum_ale" => incompressible_momentum_ale_model(),
        "allmach_pressure_ale" => allmach_pressure_ale_model(),
        "allmach_thermal_ale" => allmach_thermal_ale_model(),
        other => Err(format!("unknown ALE model id '{other}'")),
    }
}

/// Seed the non-solved state fields a structured model needs (all-Mach `psi`/`rho`
/// reference fields; compressible conserved rest state). Mirrors the driver's
/// seeding + the structured tests.
/// The state-seeding + BC surface shared by the GPU (`StructuredGpuSolver`) and
/// CPU (`StructuredModelSolver`) structured solvers, so the GUI's seed/BC helpers
/// drive either backend. The `sc_`-prefixed names avoid colliding with the
/// identically-named inherent methods the impls forward to.
trait StructuredSeed {
    fn sc_field_offset(&self, name: &str) -> Option<usize>;
    fn sc_set_eos(&mut self, params: EosRuntimeParams);
    fn sc_set_inlet_velocity(&mut self, velocity: f32);
    fn sc_set_named<F: Fn(f64, f64) -> f64>(&mut self, name: &str, f: F);
    fn sc_set_component<F: Fn(f64, f64) -> f64>(&mut self, offset: usize, f: F);
    fn sc_set_boundaries<F: Fn(StructEdge, f64, f64) -> (u32, Vec<StructBc>)>(&mut self, f: F);
}
impl StructuredSeed for StructuredGpuSolver {
    fn sc_field_offset(&self, name: &str) -> Option<usize> {
        self.field_offset(name)
    }
    fn sc_set_eos(&mut self, params: EosRuntimeParams) {
        self.set_eos(params)
    }
    fn sc_set_inlet_velocity(&mut self, velocity: f32) {
        self.set_inlet_ramp(velocity, 0.0)
    }
    fn sc_set_named<F: Fn(f64, f64) -> f64>(&mut self, name: &str, f: F) {
        self.set_named_field(name, f)
    }
    fn sc_set_component<F: Fn(f64, f64) -> f64>(&mut self, offset: usize, f: F) {
        self.set_state_component(offset, f)
    }
    fn sc_set_boundaries<F: Fn(StructEdge, f64, f64) -> (u32, Vec<StructBc>)>(&mut self, f: F) {
        self.set_boundaries(f)
    }
}
impl StructuredSeed for StructuredModelSolver {
    fn sc_field_offset(&self, name: &str) -> Option<usize> {
        self.field_offset(name)
    }
    fn sc_set_eos(&mut self, params: EosRuntimeParams) {
        self.set_eos(params)
    }
    fn sc_set_inlet_velocity(&mut self, velocity: f32) {
        self.set_inlet_ramp(velocity, 0.0)
    }
    fn sc_set_named<F: Fn(f64, f64) -> f64>(&mut self, name: &str, f: F) {
        self.set_named_field(name, f)
    }
    fn sc_set_component<F: Fn(f64, f64) -> f64>(&mut self, offset: usize, f: F) {
        self.set_state(offset, f)
    }
    fn sc_set_boundaries<F: Fn(StructEdge, f64, f64) -> (u32, Vec<StructBc>)>(&mut self, f: F) {
        self.set_boundaries(f)
    }
}

/// Apply the two runtime inputs that expression-valued compressible kernels do
/// not obtain from the state buffers. This is deliberately called before IC/BC
/// construction so the host reference state, generated kernels, and the GPU
/// autonomous oracle all observe one EOS and one inlet target.
fn apply_structured_compressible_runtime(
    s: &mut impl StructuredSeed,
    params: &RuntimeParams,
) {
    // GAUGE STORAGE (matches the unstructured driver): the density-based
    // compressible state stores deviations from the quiescent reference at
    // the fluid density. See docs/compressible-explicit-acoustics.md.
    let mut eos = params.eos.runtime_params_gauged(params.density as f64);
    // Inlet driving mode: the committed bc_expr closures branch on this
    // runtime constant (velocity inlet when 0; prescribed gauge pressure +
    // reservoir temperature with a floating outlet when 1 — the CD nozzle).
    eos.bc_pressure_inlet = if params.pressure_inlet { 1.0 } else { 0.0 };
    s.sc_set_eos(eos);
    s.sc_set_inlet_velocity(params.inlet_velocity);
}

#[derive(Clone, Copy, Debug)]
struct StructuredCompressibleReferenceState {
    rho: f64,
    momentum_x: f64,
    total_energy_density: f64,
    pressure: f64,
    temperature: f64,
    /// STORED (gauge) values of rho / rho_e / p for this state under the
    /// gauge references at the fluid density (what the state buffer and BC
    /// tables actually hold). `momentum_x`/`temperature` are stored absolute.
    stored_rho: f64,
    stored_total_energy_density: f64,
    stored_pressure: f64,
}

/// Uniform conserved/primitive state implied by the selected physical EOS.
/// Pressure comes from `EosSpec` itself, avoiding a second affine-pressure
/// implementation in the GUI. The energy gauge is the model helper's declared
/// zero gauge for a barotropic liquid and the conventional ideal-gas energy for
/// calorically-perfect gases.
fn structured_compressible_reference_state(
    eos: EosSpec,
    rho: f64,
    velocity_x: f64,
) -> StructuredCompressibleReferenceState {
    structured_compressible_reference_state_gauged(eos, rho, velocity_x, rho)
}

/// [`structured_compressible_reference_state`] with the gauge base density made
/// explicit (the stored values subtract the references of the quiescent state
/// at `gauge_rho0`; the default takes the state's own density as the base).
fn structured_compressible_reference_state_gauged(
    eos: EosSpec,
    rho: f64,
    velocity_x: f64,
    gauge_rho0: f64,
) -> StructuredCompressibleReferenceState {
    let pressure = eos.pressure_for_density(rho);
    let runtime = eos.runtime_params();
    let kinetic = 0.5 * rho * velocity_x * velocity_x;
    let internal = eos
        .barotropic_internal_energy_density(rho, 0.0)
        .unwrap_or_else(|| {
            if runtime.gm1 > 0.0 {
                pressure / f64::from(runtime.gm1)
            } else {
                0.0
            }
        });
    let temperature = if rho > 0.0 && runtime.r > 0.0 {
        pressure / (rho * f64::from(runtime.r))
    } else {
        0.0
    };
    let gauged = eos.runtime_params_gauged(gauge_rho0);
    let total_energy_density = internal + kinetic;
    StructuredCompressibleReferenceState {
        rho,
        momentum_x: rho * velocity_x,
        total_energy_density,
        pressure,
        temperature,
        stored_rho: rho - f64::from(gauged.gauge_rho_ref),
        stored_total_energy_density: total_energy_density - f64::from(gauged.gauge_e_ref),
        stored_pressure: pressure - f64::from(gauged.gauge_p_ref),
    }
}

/// Install the stage-time inlet target used by both ordinary and autonomous
/// structured RK4. The cell-area metric matches the generic solver driver.
fn apply_structured_allmach_inlet_ramp(
    s: &mut impl StructuredSteppable,
    params: &RuntimeParams,
) {
    let ramp_time = crate::sim::explicit_allmach_inlet_ramp_time(
        params,
        s.st_inlet_ramp_cell_size(),
        true,
    );
    s.st_set_inlet_ramp(params.inlet_velocity, ramp_time);
}

/// Refresh configuration fields that are algebraic inputs to the structured
/// all-Mach closure without resetting solved p/T/U. This is safe for live GUI
/// edits and makes the next pre-step spectral sample use the new fluid,
/// compressibility, inlet, and preconditioner settings.
fn refresh_structured_allmach_runtime_fields(
    s: &mut (impl StructuredSeed + StructuredSteppable),
    params: &RuntimeParams,
    dx: f64,
    dy: f64,
) {
    let psi = (params.compressibility_psi as f64).max(0.0);
    let rho_ref = params.density as f64;
    let t_ref = 1.0_f64;
    let u_ref = 2.0
        * (params.inlet_velocity.abs() as f64).max(params.allmach_precond_uref_min.max(0.0) as f64);
    let psi_precond = if psi > 0.0 && u_ref > 0.0 {
        (crate::solver::model::ALLMACH_GAMMA - 1.0) * psi + psi.max(1.0 / (u_ref * u_ref))
    } else {
        psi
    };
    let chi = (psi_precond - (crate::solver::model::ALLMACH_GAMMA - 1.0) * psi)
        .max(psi_precond.abs().max(1.0e-30) * 1.0e-6);
    let g_h = 1.0 / dx.max(1.0e-30) + 1.0 / dy.max(1.0e-30);
    let stabilization_time = 1.0 / (g_h * (1.0 / chi.sqrt()));
    let rc_scale = if params.time_scheme == GpuTimeScheme::RK4 {
        1.0
    } else {
        params.alpha_u as f64
    };
    let d_p = rc_scale * stabilization_time / rho_ref.abs().max(1.0e-12);

    s.sc_set_named("rho_t_ref", move |_, _| rho_ref * t_ref);
    s.sc_set_named("t_ref", move |_, _| t_ref);
    s.sc_set_named("rho_floor", move |_, _| psi * 1.0e-5);
    s.sc_set_named("psi_ref", move |_, _| psi);
    s.sc_set_named(
        "precond_mask",
        move |_, _| if psi > 0.0 { 1.0 } else { 0.0 },
    );
    s.sc_set_named("u_ref", move |_, _| u_ref);
    s.sc_set_named("psi", move |_, _| psi);
    s.sc_set_named("psi_precond", move |_, _| psi_precond);
    s.sc_set_named("dt_local", move |_, _| stabilization_time);
    s.sc_set_named("d_p", move |_, _| d_p);
    // Autonomous Direct RK4 bypasses `structured_step`, so these stage-time
    // boundary constants must be live at construction and after every runtime
    // parameter update rather than being lazily primed by the Plot route.
    apply_structured_allmach_inlet_ramp(s, params);
}

fn seed_structured_state(
    s: &mut (impl StructuredSeed + StructuredSteppable),
    model_id: &str,
    params: &RuntimeParams,
    dx: f64,
    dy: f64,
) {
    match model_id {
        "allmach_thermal_structured" => {
            // Match SolverDriver all-Mach seeds: real fluid compressibility
            // `psi = 1/c²` (not a hardcoded 0.5), density from the fluid, and
            // psi_precond floored by the uref preconditioner target — NOT
            // `psi.max(1.0)`, which forced an O(1) acoustic mass and stalled
            // low-Mach development under adaptive dt.
            let psi = (params.compressibility_psi as f64).max(0.0);
            let rho = params.density as f64;
            let t_ref = 1.0_f64;
            s.sc_set_named("psi", move |_, _| psi);
            s.sc_set_named("rho", move |_, _| rho);
            s.sc_set_named("rho_t_ref", move |_, _| rho * t_ref);
            s.sc_set_named("T", move |_, _| t_ref);
            if s.sc_field_offset("t_ref").is_some() {
                s.sc_set_named("t_ref", move |_, _| t_ref);
            }
            if s.sc_field_offset("rho_floor").is_some() {
                // ABS_PRESSURE_FLOOR ≈ 1e-5 gauge units (driver parity).
                s.sc_set_named("rho_floor", move |_, _| psi * 1.0e-5);
            }
            if s.sc_field_offset("psi_ref").is_some() {
                s.sc_set_named("psi_ref", move |_, _| psi);
            }
            // psi_precond host seed; on-device recovery refreshes it from u_ref.
            let u_ref = 2.0
                * (params.inlet_velocity.abs() as f64)
                    .max(params.allmach_precond_uref_min.max(0.0) as f64);
            let psi_precond = if u_ref > 0.0 {
                (crate::solver::model::ALLMACH_GAMMA - 1.0) * psi + psi.max(1.0 / (u_ref * u_ref))
            } else {
                psi
            };
            s.sc_set_named("psi_precond", move |_, _| psi_precond);
            s.sc_set_named("rho_dT", move |_, _| -rho / t_ref);
            s.sc_set_named("u_dot_grad_p", |_, _| 0.0);
            let chi = (psi_precond - (crate::solver::model::ALLMACH_GAMMA - 1.0) * psi)
                .max(psi_precond.abs().max(1.0e-30) * 1.0e-6);
            let g_h = 1.0 / dx.max(1.0e-30) + 1.0 / dy.max(1.0e-30);
            let stabilization_time = 1.0 / (g_h * (1.0 / chi.sqrt()));
            s.sc_set_named("dt_local", move |_, _| stabilization_time);
            let rc_scale = if params.time_scheme == GpuTimeScheme::RK4 {
                1.0
            } else {
                params.alpha_u as f64
            };
            let d_p = rc_scale * stabilization_time / rho.abs().max(1.0e-12);
            s.sc_set_named("d_p", move |_, _| d_p);
            apply_structured_allmach_inlet_ramp(s, params);
        }
        "compressible_structured" => {
            // GAUGE STORAGE: rho/rho_e/p store deviations from this very
            // reference state, so the quiescent seed is exactly zero.
            let reference = structured_compressible_reference_state(
                params.eos,
                params.density as f64,
                0.0,
            );
            s.sc_set_named("rho", move |_, _| reference.stored_rho);
            s.sc_set_named("rho_e", move |_, _| reference.stored_total_energy_density);
            s.sc_set_named("p", move |_, _| reference.stored_pressure);
            s.sc_set_named("T", move |_, _| reference.temperature);
            if s.sc_field_offset("mu").is_some() {
                let mu = params.viscosity as f64;
                s.sc_set_named("mu", move |_, _| mu);
            }
        }
        _ => {}
    }
}

/// Is `(x, y)` inside the SOLID of the structured geometry (for the Brinkman mask)?
/// `ChannelObstacle` = a cylinder; `BackwardsStep` = a solid block at the inlet
/// floor; `Nozzle` = the region outside a converging–diverging profile.
/// The unstructured GUI geometries are the GROUND TRUTH for every mesh mode.
/// These shared constructors keep `build_mesh_with` (SDF meshing), the fitted
/// nozzle grid, and the structured Brinkman mask in exact agreement.
fn gui_channel_obstacle_geometry() -> ChannelWithObstacle {
    ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51), // Offset to trigger vortex shedding
        obstacle_radius: 0.1,
    }
}

fn gui_backstep_geometry() -> BackwardsStep {
    BackwardsStep {
        length: 3.5,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    }
}

/// Converging–diverging nozzle, area ratio exit/throat = 2, matching the
/// validated `allmach_thermal_supersonic_test`.
fn gui_nozzle_geometry() -> Nozzle {
    Nozzle {
        length: 3.0,
        height: 1.0,
        throat_height: 0.40,
        throat_frac: 0.40,
        exit_height: 0.80,
    }
}

/// Bounding-box domain of the selected geometry — identical for the
/// unstructured meshers and the structured (dense-Cartesian) grid.
fn gui_geometry_domain(geom: GeometryType) -> (f64, f64) {
    match geom {
        GeometryType::BackwardsStep => {
            let geo = gui_backstep_geometry();
            (geo.length, geo.height_outlet)
        }
        GeometryType::ChannelObstacle => {
            let geo = gui_channel_obstacle_geometry();
            (geo.length, geo.height)
        }
        GeometryType::Nozzle => {
            let geo = gui_nozzle_geometry();
            (geo.length, geo.height)
        }
    }
}

fn structured_geometry_is_solid(geom: GeometryType, x: f64, y: f64) -> bool {
    let point = Point2::new(x, y);
    match geom {
        GeometryType::ChannelObstacle => !gui_channel_obstacle_geometry().is_inside(&point),
        GeometryType::BackwardsStep => !gui_backstep_geometry().is_inside(&point),
        GeometryType::Nozzle => {
            // The GUI-default (validated) nozzle mesh is the FITTED symmetric
            // bell, so the mask rasterizes that profile: fluid occupies
            // `|y - height/2| <= nozzle_height(x)/2` (the SDF mesh variants'
            // flat-bottom profile is a non-default alternative).
            let geo = gui_nozzle_geometry();
            let xi = (x / geo.length).clamp(0.0, 1.0);
            let h = crate::solver::mesh::structured::nozzle_height(
                xi,
                geo.height,
                geo.throat_height,
                geo.throat_frac,
                geo.exit_height,
            );
            (y - 0.5 * geo.height).abs() > 0.5 * h
        }
    }
}

/// Rasterize the selected geometry into the momentum Brinkman penalty field
/// `ibm_penalty_U` (large negative inside the solid; a `source_coeff(Sp,U)` sink
/// pins U→0 there, with an exact per-stage projection under all-Mach RK4).
/// No-op for models without the field.
fn seed_structured_ibm(s: &mut impl StructuredSeed, geom: GeometryType) {
    let Some(off) = s.sc_field_offset("ibm_penalty_U") else {
        return;
    };
    s.sc_set_component(off, move |x, y| {
        if structured_geometry_is_solid(geom, x, y) {
            -STRUCTURED_IBM_PENALTY_RATE
        } else {
            0.0
        }
    });
}

/// Seed the uniform-freestream momentum IC the density-based compressible
/// channel needs: `rho_u = rho * u_in` (rightward). No-op for the pressure-based
/// models, whose primitive-velocity channel drives fine from a rest IC.
fn seed_structured_freestream(
    s: &mut impl StructuredSeed,
    model_id: &str,
    u_in: f64,
    params: &RuntimeParams,
) {
    if model_id == "compressible_structured" {
        // Explicit (RK4) stepping develops naturally FROM REST like the
        // pressure-based models (mirrors the unstructured driver's compressible
        // IC): the inlet BC keeps driving with `u_in`, only the initial state
        // is quiescent. The uniform-freestream IC remains the implicit
        // pseudo-transient path's low-Mach stabilizer.
        let ic_velocity = if params.time_scheme == GpuTimeScheme::RK4 {
            0.0
        } else {
            u_in
        };
        let reference = structured_compressible_reference_state(
            params.eos,
            params.density as f64,
            ic_velocity,
        );
        if s.sc_field_offset("rho_u").is_some() {
            s.sc_set_named("rho_u", move |_, _| reference.momentum_x);
        }
        if s.sc_field_offset("rho_e").is_some() {
            s.sc_set_named("rho_e", move |_, _| reference.stored_total_energy_density);
        }
        if s.sc_field_offset("p").is_some() {
            s.sc_set_named("p", move |_, _| reference.stored_pressure);
        }
        if s.sc_field_offset("T").is_some() {
            s.sc_set_named("T", move |_, _| reference.temperature);
        }
        if s.sc_field_offset("u").is_some() {
            s.sc_set_named("u", move |_, _| ic_velocity);
        }
    }
    if model_id == "allmach_thermal_structured" {
        // Low-Mach preconditioner CONFIG (mirrors the unstructured driver's all-Mach
        // seeds). The on-device psi_precond recovery reads beta^2 = max(|U|^2,
        // u_ref^2); WITHOUT u_ref the from-REST field over-damps. u_ref tracks the
        // inlet slider floored by the GUI preconditioner floor — model config, NOT
        // a velocity head-start (flow still starts from rest).
        let psi = (params.compressibility_psi as f64).max(0.0);
        let u_ref = 2.0
            * u_in
                .abs()
                .max(params.allmach_precond_uref_min.max(0.0) as f64);
        if s.sc_field_offset("u_ref").is_some() {
            s.sc_set_named("u_ref", move |_, _| u_ref);
        }
        if s.sc_field_offset("psi_ref").is_some() {
            s.sc_set_named("psi_ref", move |_, _| psi);
        }
        if s.sc_field_offset("precond_mask").is_some() {
            let mask = if psi > 0.0 { 1.0 } else { 0.0 };
            s.sc_set_named("precond_mask", move |_, _| mask);
        }
    }
}

/// Apply the structured CHANNEL boundary conditions (inlet left, outlet right,
/// no-slip top/bottom walls). The two solved-unknown layouts need different
/// component conventions:
/// - pressure-based (incompressible / all-Mach thermal): primitive `[Ux, Uy, p]`
///   (+`T`) — inlet Dirichlet velocity, outlet pressure-Dirichlet + zero-gradient
///   velocity.
/// - density-based (compressible): conserved `[rho, rho_u_x, rho_u_y, rho_e]` —
///   conserved-Dirichlet inlet (`rho`, `rho*u_in`, `rho_e`), zero-gradient
///   outlet; the model's `bc_expr` closure keeps the dependent entries
///   thermodynamically consistent.
fn setup_structured_bcs(
    s: &mut impl StructuredSeed,
    model_id: &str,
    u_in: f64,
    stride_s: usize,
    params: &RuntimeParams,
) {
    if model_id == "compressible_structured" {
        // GAUGE STORAGE: BC tables hold STORED values; the gauge base is the
        // fluid density (matching apply_structured_compressible_runtime), so
        // inlet-state deviations from it survive in the stored form.
        let reference = structured_compressible_reference_state_gauged(
            params.eos,
            f64::from(params.density.max(1.0e-6)),
            u_in,
            params.density as f64,
        );
        let rho0 = reference.stored_rho as f32;
        let momentum_x = reference.momentum_x as f32;
        let e0 = reference.stored_total_energy_density as f32;
        let t0 = reference.temperature as f32;
        let u_in_f32 = u_in as f32;
        let outlet_p = params.outlet_back_pressure;
        let inlet_p = params.inlet_pressure;
        s.sc_set_boundaries(move |edge, _x, _y| {
            let d = |v: f32| StructBc { kind: 1, value: v };
            let n = || StructBc {
                kind: 2,
                value: 0.0,
            };
            // TRUE ZeroGradient (kind 0): the ghost extrapolates the owner and
            // the table VALUE is ignored by every kernel — safe for channels
            // the bc_expr closure overwrites (kind 2 would misread those
            // writes as prescribed GRADIENTS). This matches the unstructured
            // model's outlet declarations (GpuBcKind::ZeroGradient), giving
            // the same quasi-non-reflective, extrapolating outlet.
            let z = || StructBc {
                kind: 0,
                value: 0.0,
            };
            // Coupled channel order: rho, rho_u_x, rho_u_y, rho_e, u_x, u_y,
            // p, T. The u CHANNELS ARE LOAD-BEARING at the inlet: the model's
            // bc_expr kernel recomputes the dependent inlet entries every
            // stage as `rho_u = rho_abs * bc(u)` (and the rho_e/T closures
            // from the same prescribed velocity), so leaving u unset (zero)
            // silently zeroes the momentum drive each stage — the inlet then
            // acts as a wall and the flow never develops. Walls pin u to 0
            // (no-slip, matching the rho_u Dirichlet and the unstructured
            // declarations); outlet extrapolates everything.
            // KIND CONTRACT: the ghost builder consumes `bc_value` only for
            // kind-1 channels; kind 2 reinterprets it as a GRADIENT
            // (`ghost = owner + value*d`), and the viscous residual adds
            // `coeff*area*value` outright. The bc_expr closure REWRITES the
            // dependent channels' VALUES every stage, so every closure-written
            // channel a kernel consumes must be kind 1 — leaving one at kind 2
            // feeds the closure's absolute value in as a gradient (the T
            // channel then injects ~kappa*area*300 of heat per boundary face,
            // a positive feedback that pressurizes the whole channel).
            let (btype, mut v): (u32, Vec<StructBc>) = match edge {
                // Inlet channel 6 holds the prescribed gauge pressure: unused
                // in velocity mode (the closure writes the interior-following
                // p over it), THE drive in pressure-inlet mode (the closure
                // reads it to rebuild rho/rho_u/rho_e at the reservoir
                // temperature in channel 7).
                StructEdge::Left => (
                    1,
                    vec![
                        d(rho0),
                        d(momentum_x),
                        d(0.0),
                        d(e0),
                        d(u_in_f32),
                        d(0.0),
                        d(inlet_p),
                        d(t0),
                    ],
                ),
                // Outlet: TRUE ZeroGradient extrapolation on every state
                // channel, matching the unstructured model (whose outlet
                // declarations are GpuBcKind::ZeroGradient — its ghost builder
                // ignores the closure values entirely). Consuming the
                // closure's p-anchored ghost energy as a kind-1 value instead
                // makes the outlet a pressure-release surface: an outgoing
                // compression reflects as an expansion with DOUBLED velocity
                // (a strong visible bounce of the startup wave). Channel 6
                // still carries the gauge back-pressure as the closure input
                // (nothing consumes its ghost). Kind 2 with value 0 would also
                // extrapolate, but the closure REWRITES these channel values
                // every stage and kind 2 misreads them as prescribed
                // gradients — the historical boundary heat-source runaway.
                StructEdge::Right => (
                    2,
                    vec![z(), z(), z(), z(), z(), z(), d(outlet_p), z()],
                ),
                // No-slip wall: momentum and velocity pinned to 0,
                // rho/rho_e/p/T zero-gradient.
                _ => (
                    3,
                    vec![n(), d(0.0), d(0.0), n(), d(0.0), d(0.0), n(), n()],
                ),
            };
            v.truncate(stride_s);
            (btype, v)
        });
    } else {
        // Pressure-based channel. bc_kind 1=Dirichlet, 2=Neumann(zero-grad).
        // Components 0=Ux, 1=Uy, 2=p (+3=T).
        let pressure_inlet = model_id == "allmach_thermal_structured" && params.pressure_inlet;
        let inlet_pressure = params.inlet_pressure;
        let outlet_back_pressure = params.outlet_back_pressure;
        s.sc_set_boundaries(move |edge, _x, _y| {
            let (btype, mut v): (u32, Vec<StructBc>) = match edge {
                StructEdge::Left => (
                    1,
                    vec![
                        StructBc {
                            kind: if pressure_inlet { 2 } else { 1 },
                            value: if pressure_inlet { 0.0 } else { u_in as f32 },
                        },
                        StructBc {
                            kind: 1,
                            value: 0.0,
                        },
                        StructBc {
                            kind: if pressure_inlet { 1 } else { 2 },
                            value: if pressure_inlet { inlet_pressure } else { 0.0 },
                        },
                    ],
                ),
                StructEdge::Right => (
                    2,
                    vec![
                        StructBc {
                            kind: 2,
                            value: 0.0,
                        },
                        StructBc {
                            kind: 2,
                            value: 0.0,
                        },
                        StructBc {
                            kind: if pressure_inlet { 2 } else { 1 },
                            value: if pressure_inlet {
                                0.0
                            } else {
                                outlet_back_pressure
                            },
                        },
                    ],
                ),
                _ => (
                    3,
                    vec![
                        StructBc {
                            kind: 1,
                            value: 0.0,
                        },
                        StructBc {
                            kind: 1,
                            value: 0.0,
                        },
                        StructBc {
                            kind: 2,
                            value: 0.0,
                        },
                    ],
                ),
            };
            if stride_s >= 4 {
                // T: Dirichlet at inlet, zero-gradient elsewhere.
                v.push(if matches!(edge, StructEdge::Left) {
                    StructBc {
                        kind: 1,
                        value: 1.0,
                    }
                } else {
                    StructBc {
                        kind: 2,
                        value: 0.0,
                    }
                });
            }
            (btype, v)
        });
    }
}

#[cfg(test)]
mod structured_boundary_tests {
    use super::*;
    use crate::solver::model::eos::EosSpec;
    use crate::ui::model_defaults::gui_defaults_for;

    #[test]
    fn moving_mesh_coerces_only_static_rk4_to_bdf2() {
        let mut rk4 = GpuTimeScheme::RK4;
        assert!(enforce_moving_mesh_time_scheme(true, &mut rk4));
        assert_eq!(rk4, GpuTimeScheme::BDF2);

        let mut static_rk4 = GpuTimeScheme::RK4;
        assert!(!enforce_moving_mesh_time_scheme(
            false,
            &mut static_rk4,
        ));
        assert_eq!(static_rk4, GpuTimeScheme::RK4);

        for mut implicit in [GpuTimeScheme::Euler, GpuTimeScheme::BDF2] {
            let expected = implicit;
            assert!(!enforce_moving_mesh_time_scheme(true, &mut implicit));
            assert_eq!(implicit, expected);
        }
    }

    #[test]
    fn actual_gui_model_options_are_exact_for_each_topology() {
        let unstructured: Vec<_> = CFDApp::supported_ui_models(MeshMode::Unstructured)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            unstructured,
            [
                "allmach_pressure",
                "allmach_thermal",
                "compressible",
                "incompressible_momentum",
            ]
        );

        let structured: Vec<_> = CFDApp::supported_ui_models(MeshMode::Structured2D)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            structured,
            [
                "incompressible_momentum_structured",
                "allmach_thermal_structured",
                "compressible_structured",
            ]
        );
    }

    #[test]
    fn structured_allmach_runtime_primes_inlet_before_first_cpu_step() {
        let params = autonomous_allmach_params();
        let model = crate::solver::model::allmach_thermal_structured_model()
            .expect("structured all-Mach model");
        let grid = StructuredGrid::new(8, 4, 1.0, 0.5);
        let mut solver = StructuredModelSolver::with_config(
            grid,
            &model,
            f64::from(params.requested_dt),
            1,
            Scheme::Upwind,
            GpuTimeScheme::RK4,
        )
        .expect("structured CPU explicit solver");
        solver.set_fluid(f64::from(params.density), f64::from(params.viscosity));
        seed_structured_state(
            &mut solver,
            "allmach_thermal_structured",
            &params,
            grid.dx,
            grid.dy,
        );
        refresh_structured_allmach_runtime_fields(&mut solver, &params, grid.dx, grid.dy);

        let expected_ramp = crate::sim::explicit_allmach_inlet_ramp_time(
            &params,
            (grid.dx * grid.dy).sqrt(),
            true,
        );
        assert_eq!(
            solver.inlet_velocity_for_test().to_bits(),
            params.inlet_velocity.to_bits()
        );
        assert_eq!(
            solver.inlet_ramp_time_for_test().to_bits(),
            expected_ramp.to_bits()
        );
        assert_eq!(
            solver.time().to_bits(),
            0.0_f64.to_bits(),
            "test must not prime via Plot"
        );
    }

    #[test]
    fn structured_compressible_air_gui_runtime_reaches_cpu_kernels_seed_and_bc() {
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let mut params = gui_defaults_for("compressible_structured")
            .to_runtime_params(1.225, 1.81e-5, air);
        params.inlet_velocity = 12.5;
        params.time_scheme = GpuTimeScheme::RK4;

        let model = crate::solver::model::compressible_structured_model().unwrap();
        let grid = StructuredGrid::new(3, 2, 0.3, 0.2);
        let mut solver = StructuredModelSolver::with_config(
            grid,
            &model,
            f64::from(params.requested_dt),
            1,
            Scheme::Upwind,
            GpuTimeScheme::RK4,
        )
        .unwrap();
        solver.set_fluid(f64::from(params.density), f64::from(params.viscosity));
        apply_structured_compressible_runtime(&mut solver, &params);
        seed_structured_state(&mut solver, model.id, &params, grid.dx, grid.dy);
        seed_structured_freestream(
            &mut solver,
            model.id,
            f64::from(params.inlet_velocity),
            &params,
        );
        let unknowns = solver.unknowns();
        setup_structured_bcs(
            &mut solver,
            model.id,
            f64::from(params.inlet_velocity),
            unknowns,
            &params,
        );

        // GAUGE STORAGE: the structured compressible runtime activates the
        // gauge around the fluid density.
        assert_eq!(
            solver.runtime_eos_for_test(),
            air.runtime_params_gauged(f64::from(params.density))
        );
        assert_eq!(
            solver.inlet_velocity_for_test().to_bits(),
            params.inlet_velocity.to_bits(),
            "live inlet target must reach generated-kernel constants"
        );

        // Explicit (RK4) runs seed the STATE from rest while the inlet BC
        // table carries the prescribed freestream — the flow develops from
        // the boundary drive (docs/compressible-explicit-acoustics.md).
        let expected_seed = structured_compressible_reference_state(
            air,
            f64::from(params.density),
            0.0,
        );
        let expected_inlet = structured_compressible_reference_state_gauged(
            air,
            f64::from(params.density),
            f64::from(params.inlet_velocity),
            f64::from(params.density),
        );
        let stored = |value: f64| f64::from(value as f32);
        let scalar = |name: &str| {
            solver.get_scalar(solver.field_offset(name).expect("structured field"))[0]
        };
        // GAUGE STORAGE: rho/rho_e/p seed and BC tables hold STORED values.
        assert_eq!(scalar("rho"), stored(expected_seed.stored_rho));
        assert_eq!(
            scalar("rho_e"),
            stored(expected_seed.stored_total_energy_density)
        );
        assert_eq!(scalar("p"), stored(expected_seed.stored_pressure));
        assert_eq!(scalar("T"), stored(expected_seed.temperature));
        let velocity = solver.get_u(solver.field_offset("u").unwrap())[0];
        assert_eq!(velocity, (0.0, 0.0), "RK4 seeds the velocity from rest");

        // Cell 0's west face is the inlet; coupled unknown order is
        // rho, rho_u.x, rho_u.y, rho_e, u.x, u.y, p, T.
        assert_eq!(solver.bc_kind_at(0, 1, 0), 1);
        assert_eq!(solver.bc_kind_at(0, 1, 1), 1);
        assert_eq!(solver.bc_kind_at(0, 1, 3), 1);
        assert_eq!(solver.bc_value_at(0, 1, 0), stored(expected_inlet.stored_rho));
        assert_eq!(
            solver.bc_value_at(0, 1, 1),
            stored(expected_inlet.momentum_x)
        );
        assert_eq!(
            solver.bc_value_at(0, 1, 3),
            stored(expected_inlet.stored_total_energy_density)
        );
        // The inlet u channels are LOAD-BEARING: the bc_expr kernel rebuilds
        // the dependent inlet entries from the prescribed velocity every
        // stage, so an unset (zero) u channel silently kills the momentum
        // drive (the "small pulse then nothing" failure).
        assert_eq!(solver.bc_kind_at(0, 1, 4), 1);
        assert_eq!(solver.bc_kind_at(0, 1, 5), 1);
        assert_eq!(
            solver.bc_value_at(0, 1, 4),
            stored(f64::from(params.inlet_velocity))
        );
        assert_eq!(solver.bc_value_at(0, 1, 5), 0.0);
    }

    #[test]
    fn unsupported_structured_eos_uses_same_ordinary_policy_in_plot_and_direct() {
        for adaptive in [false, true] {
            let plot = structured_autonomous_refill_policy(false, adaptive, false, false);
            let direct = structured_autonomous_refill_policy(true, adaptive, false, false);
            assert_eq!(plot, None);
            assert_eq!(direct, None);
        }
    }

    #[test]
    fn real_gpu_structured_air_gui_runtime_reaches_oracle_and_water_falls_back() {
        let ctx = match pollster::block_on(crate::solver::gpu::context::GpuContext::new(None, None))
        {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("skipping structured EOS GPU gate: {error}");
                return;
            }
        };
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let mut params = gui_defaults_for("compressible_structured")
            .to_runtime_params(1.225, 1.81e-5, air);
        params.inlet_velocity = 6.25;
        params.target_cfl = 0.5;
        params.requested_dt = 1.0e-5;
        params.adaptive_dt = true;
        params.time_scheme = GpuTimeScheme::RK4;

        let model = crate::solver::model::compressible_structured_model().unwrap();
        let grid = StructuredGrid::new(4, 2, 1.0e-4, 5.0e-5);
        let mut solver = StructuredGpuSolver::with_config(
            ctx,
            grid,
            &model,
            f64::from(params.requested_dt),
            1,
            Scheme::Upwind,
            GpuTimeScheme::RK4,
        )
        .unwrap();
        solver.set_fluid(f64::from(params.density), f64::from(params.viscosity));
        apply_structured_compressible_runtime(&mut solver, &params);
        seed_structured_state(&mut solver, model.id, &params, grid.dx, grid.dy);
        seed_structured_freestream(
            &mut solver,
            model.id,
            f64::from(params.inlet_velocity),
            &params,
        );
        let unknowns = solver.unknowns();
        setup_structured_bcs(
            &mut solver,
            model.id,
            f64::from(params.inlet_velocity),
            unknowns,
            &params,
        );

        // Exercise the worker's live-update seam, not only construction-time
        // setup: the new target must reach the same constants/oracle block.
        let mut live = params;
        live.inlet_velocity = 12.5;
        let mut mode = SolverMode::Structured(solver);
        mode.apply_params_any(&live);
        let SolverMode::Structured(mut solver) = mode else {
            unreachable!("structured GPU mode changed variant")
        };
        params = live;

        assert_eq!(
            solver.runtime_eos_for_test(),
            air.runtime_params_gauged(f64::from(params.density))
        );
        assert_eq!(
            solver.inlet_velocity_for_test().to_bits(),
            params.inlet_velocity.to_bits()
        );
        assert!(solver.supports_autonomous_fixed());
        assert!(solver.supports_autonomous_adaptive());
        let target_cfl = params.target_cfl as f32;
        solver
            .step_autonomous_batch(1, Some(target_cfl))
            .expect("Air adaptive oracle submission");
        let status = solver.autonomous_status().expect("Air oracle status");
        assert_eq!(status.accepted_steps, 1, "Air oracle rejected GUI state");
        assert!(!status.halted, "Air oracle halted: {status:?}");

        let h = grid.dx.min(grid.dy) as f32;
        let sound = (air.runtime_params().gamma * air.runtime_params().theta_ref).sqrt();
        let wave_dt = target_cfl * h / (params.inlet_velocity.abs() + sound);
        let diffusion_dt = 0.25 * target_cfl * h * h
            / (params.viscosity.abs() / params.density.abs().max(1.0e-12));
        // No Brinkman reaction bound: the RK4 stages project rho_u to zero
        // in solid cells, so the -1e5 penalty never limits the explicit dt.
        let expected = wave_dt
            .min(diffusion_dt)
            .min(params.requested_dt * 1.2)
            .clamp(1.0e-9, 100.0);
        assert!(
            (status.dt - expected).abs() <= expected * 2.0e-4,
            "Air EOS/inlet did not reach adaptive oracle: dt={} expected={expected}",
            status.dt
        );

        let water = EosSpec::LinearCompressibility {
            bulk_modulus: 2.2e9,
            rho_ref: 1000.0,
            p_ref: 1.0e5,
        };
        for unsupported in [water, EosSpec::Constant] {
            solver.set_eos(unsupported.runtime_params());
            assert_eq!(solver.runtime_eos_for_test(), unsupported.runtime_params());
            assert!(!solver.supports_autonomous_fixed());
            assert!(!solver.supports_autonomous_adaptive());
            assert!(solver.step_autonomous_batch(1, None).is_err());
            for presentation_ready in [false, true] {
                assert_eq!(
                    structured_autonomous_refill_policy(
                        presentation_ready,
                        params.adaptive_dt,
                        solver.supports_autonomous_fixed(),
                        solver.supports_autonomous_adaptive(),
                    ),
                    None,
                    "unsupported EOS must use the ordinary timestep route in Plot and Direct"
                );
            }
        }
    }

    fn batch_completion(
        ticket: AutonomousBatchTicket,
        accepted_steps: u32,
        completed_at: std::time::Instant,
    ) -> AutonomousBatchCompletion {
        AutonomousBatchCompletion {
            ticket,
            accepted_steps,
            completed_time: (ticket.serial + 1) as f64,
            last_dt: 1.0e-3,
            next_dt: 1.0e-3,
            halted: false,
            invalid_count: 0,
            completed_at,
            backend_status: None,
        }
    }

    #[test]
    fn autonomous_batch_controller_converges_to_three_millisecond_batches() {
        let mut controller = AutonomousBatchController::default();
        for _ in 0..12 {
            let steps = controller.steps();
            // Deterministic 0.5 ms/step device throughput.
            controller.observe(
                steps,
                std::time::Duration::from_secs_f64(f64::from(steps) * 0.0005),
            );
        }
        assert_eq!(controller.steps(), 6);
        let predicted_ms = f64::from(controller.steps()) * 0.5;
        assert!((2.0..=4.0).contains(&predicted_ms));
    }

    #[test]
    fn autonomous_scheduler_bounds_fifo_and_acks_pause_only_after_drain() {
        let start = std::time::Instant::now();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(AutonomousRefillPolicy::Continuous);
        let first = scheduler.reserve(start).unwrap();
        let second = scheduler
            .reserve(start + std::time::Duration::from_micros(20))
            .unwrap();
        assert!(
            scheduler.reserve(start).is_none(),
            "only two GPU batches may be queued"
        );
        assert!(!scheduler.request_pause());
        assert_eq!(scheduler.phase, AutonomousRunPhase::Pausing);

        // Even adversarial callback delivery cannot retire the FIFO tail first.
        let tail = scheduler.complete(batch_completion(
            second,
            second.requested_steps,
            start + std::time::Duration::from_millis(6),
        ));
        assert!(tail.completed.is_empty());
        assert!(!tail.became_paused);

        let drained = scheduler.complete(batch_completion(
            first,
            first.requested_steps,
            start + std::time::Duration::from_millis(3),
        ));
        assert_eq!(drained.completed.len(), 2);
        assert_eq!(drained.completed.get(0).unwrap().ticket.serial, first.serial);
        assert_eq!(drained.completed.get(1).unwrap().ticket.serial, second.serial);
        assert_eq!(drained.retired.get(0).unwrap().ticket.serial, first.serial);
        assert_eq!(drained.retired.get(1).unwrap().ticket.serial, second.serial);
        assert!(drained.became_paused);
        assert_eq!(scheduler.phase, AutonomousRunPhase::Paused);
        assert!(!scheduler.is_active());
    }

    #[test]
    fn autonomous_scheduler_invalidates_stale_callback_generations() {
        let start = std::time::Instant::now();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(AutonomousRefillPolicy::Continuous);
        let stale = scheduler.reserve(start).unwrap();

        // A live parameter/solver change gets a new generation without
        // forgetting that the old queue submission still occupies one credit.
        scheduler.invalidate(true);
        let current = scheduler
            .reserve(start + std::time::Duration::from_millis(1))
            .unwrap();
        assert!(scheduler.reserve(start).is_none());

        let mut stale_completion = batch_completion(
            stale,
            stale.requested_steps,
            start + std::time::Duration::from_millis(2),
        );
        stale_completion.halted = true;
        stale_completion.invalid_count = 7;
        let ignored = scheduler.complete(stale_completion);
        assert!(ignored.completed.is_empty());
        assert_eq!(ignored.retired.len(), 1);
        assert_eq!(
            ignored.retired.get(0).unwrap().ticket.backend_epoch,
            scheduler.backend_epoch,
            "same-solver stale work must still be offered for host reconciliation"
        );
        assert_eq!(scheduler.phase, AutonomousRunPhase::Running);

        let accepted = scheduler.complete(batch_completion(
            current,
            current.requested_steps,
            start + std::time::Duration::from_millis(5),
        ));
        assert_eq!(accepted.completed.len(), 1);
        assert_eq!(
            accepted.completed.get(0).unwrap().ticket.generation,
            scheduler.generation
        );
        assert_eq!(scheduler.phase, AutonomousRunPhase::Running);
    }

    #[test]
    fn autonomous_scheduler_separates_backend_epoch_from_run_generation() {
        let start = std::time::Instant::now();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(AutonomousRefillPolicy::Continuous);
        let replaced = scheduler.reserve(start).unwrap();
        scheduler.replace_backend();

        let drain = scheduler.complete(batch_completion(
            replaced,
            replaced.requested_steps,
            start + std::time::Duration::from_millis(2),
        ));
        assert!(drain.completed.is_empty());
        assert_eq!(drain.retired.len(), 1);
        assert_ne!(
            drain.retired.get(0).unwrap().ticket.backend_epoch,
            scheduler.backend_epoch,
            "replacement-solver callbacks must never reconcile into the new backend"
        );
    }

    #[test]
    fn autonomous_failed_front_tombstones_an_already_arrived_tail() {
        let start = std::time::Instant::now();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(AutonomousRefillPolicy::Continuous);
        let front = scheduler.reserve(start).unwrap();
        let tail = scheduler.reserve(start).unwrap();

        let parked = scheduler.complete(batch_completion(
            tail,
            tail.requested_steps,
            start + std::time::Duration::from_millis(2),
        ));
        assert!(parked.completed.is_empty());
        assert_eq!(scheduler.pending_completions.len(), 1);

        // The front map/decode fails. Invalidating first makes the parked tail
        // a tombstone; discarding the failure must release both credits.
        scheduler.invalidate(false);
        scheduler.discard_failed_completion(front);
        assert!(scheduler.in_flight.is_empty());
        assert!(scheduler.pending_completions.is_empty());
        assert_eq!(scheduler.phase, AutonomousRunPhase::Paused);
    }

    #[test]
    fn autonomous_health_failure_freezes_submission_and_drains_tail() {
        let start = std::time::Instant::now();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(AutonomousRefillPolicy::Continuous);
        let first = scheduler.reserve(start).unwrap();
        let second = scheduler.reserve(start).unwrap();

        let mut failed = batch_completion(
            first,
            first.requested_steps.saturating_sub(1),
            start + std::time::Duration::from_millis(3),
        );
        failed.halted = true;
        failed.invalid_count = 1;
        let failure = scheduler.complete(failed);
        assert_eq!(failure.completed.len(), 1);
        assert_eq!(scheduler.phase, AutonomousRunPhase::Pausing);
        assert!(scheduler.reserve(start).is_none());

        let tail = scheduler.complete(batch_completion(
            second,
            0,
            start + std::time::Duration::from_millis(4),
        ));
        assert!(tail.became_paused);
        assert_eq!(scheduler.phase, AutonomousRunPhase::Paused);
    }

    #[test]
    fn autonomous_drain_window_does_not_refill_after_only_one_completion() {
        let start = std::time::Instant::now();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(AutonomousRefillPolicy::DrainWindow);
        let first = scheduler.reserve(start).unwrap();
        let second = scheduler.reserve(start).unwrap();
        assert!(scheduler.reserve(start).is_none());

        let first_done = scheduler.complete(batch_completion(
            first,
            first.requested_steps,
            start + std::time::Duration::from_millis(2),
        ));
        assert_eq!(first_done.completed.len(), 1);
        assert!(
            scheduler.reserve(start).is_none(),
            "phase-sensitive backend must drain both ping-pong batches before refill"
        );

        let second_done = scheduler.complete(batch_completion(
            second,
            second.requested_steps,
            start + std::time::Duration::from_millis(4),
        ));
        assert_eq!(second_done.completed.len(), 1);
        assert!(scheduler.reserve(start).is_some());
    }

    #[test]
    fn autonomous_three_phase_policy_refills_only_whole_history_cycles() {
        let start = std::time::Instant::now();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(AutonomousRefillPolicy::ContinuousPhaseCycle3);
        scheduler.controller.steps = 1;
        let first = scheduler.reserve(start).unwrap();
        assert_eq!(first.requested_steps, 3);
        scheduler.controller.steps = 4;
        let second = scheduler.reserve(start).unwrap();
        assert_eq!(second.requested_steps, 6);

        let first_done = scheduler.complete(batch_completion(
            first,
            first.requested_steps,
            start + std::time::Duration::from_millis(2),
        ));
        assert_eq!(first_done.completed.len(), 1);
        let refill = scheduler.reserve(start).expect("continuous refill credit");
        assert_eq!(refill.requested_steps % 3, 0);
        assert!((3..=4095).contains(&refill.requested_steps));
    }

    fn autonomous_allmach_params() -> RuntimeParams {
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let mut params = gui_defaults_for("allmach_thermal").to_runtime_params(1.225, 1.81e-5, air);
        params.time_scheme = GpuTimeScheme::RK4;
        params.adaptive_dt = true;
        params.target_cfl = 0.5;
        params.requested_dt = 2.0e-4;
        params.log_convergence = false;
        params.inlet_velocity = 0.2;
        params.allmach_precond_uref_min = 0.2;
        params.outer_iters = 1;
        params
    }

    /// Drive the same two-credit scheduler/callback/reconciliation seam as the
    /// worker, but with direct assertions on its private FIFO state. This is a
    /// real-device smoke: both batches execute, map their exact tail status via
    /// Poll-only callbacks, and publish one frame-demanded device copy.
    #[derive(Debug)]
    struct DirectVizProbe {
        pressure_values: Vec<f32>,
        velocity_magnitude_values: Vec<f32>,
        pressure_range: [f32; 2],
        velocity_magnitude_range: [f32; 2],
    }

    fn read_gpu_f32(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        size_bytes: u64,
    ) -> Vec<f32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("direct-viz-probe-readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("direct-viz-probe-copy"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size_bytes);
        queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::sync_channel(1);
        staging.slice(..size_bytes).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .expect("direct viz probe GPU wait");
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("direct viz probe map callback")
            .expect("direct viz probe map");
        let view = staging.slice(..size_bytes).get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        staging.unmap();
        values
    }

    fn expected_direct_range(minimum: f32, maximum: f32) -> [f32; 2] {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return [0.0, 1.0];
        }
        if (maximum - minimum).abs() < 1.0e-12 {
            let expanded = minimum + 1.0;
            if expanded.is_finite() && expanded > minimum {
                return [minimum, expanded];
            }
            return [0.0, 1.0];
        }
        [minimum, maximum]
    }

    fn probe_paused_plot_to_direct(
        mode: &SolverMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> DirectVizProbe {
        let ports = mode.ui_ports();
        let pressure_offset = ports.p_offset.expect("paused Direct pressure offset");
        let velocity_offset = ports.u_offset.expect("paused Direct velocity offset");
        let size_bytes = mode.state_size_bytes();
        let expected = read_gpu_f32(device, queue, mode.state_buffer(), size_bytes);
        let buffers = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("paused-direct-viz-field"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let range_reducer = cfd_renderer::CfdRangeReducer::new(
            device,
            queue,
            &buffers,
            cfd_renderer::CfdFieldLayout {
                stride: ports.stride,
                u_offset: ports.u_offset.unwrap_or(0),
                p_offset: pressure_offset,
                has_u: ports.u_offset.is_some(),
                has_p: true,
            },
        );
        let mailbox = Arc::new(VizFrameMailbox::new());
        let viz = VizFieldBuffers {
            buffers,
            size_bytes,
            range_reducer,
            mailbox: Arc::clone(&mailbox),
        };

        // A cold Direct renderer starts with no GPU consumer and slot zero's
        // bootstrap field. Production activation must post exactly one request,
        // and idle service must replace it without a Plot transition or another
        // solver step.
        assert!(mailbox.try_begin_write().is_none());
        assert!(activate_direct_viz(&mailbox));
        let submissions_before = viz.range_reducer.snapshot_submission_count();
        assert!(
            service_pending_viz_snapshot(mode, &viz, VizSnapshotSource::Current)
                .expect("cold Direct snapshot service")
        );
        assert_eq!(
            viz.range_reducer.snapshot_submission_count(),
            submissions_before + 1,
            "Direct field copy and range reduction must share one submission"
        );
        let display = mailbox
            .try_claim_latest_ready()
            .expect("paused Direct snapshot becomes READY");
        assert_eq!(display.sequence, 1);

        let packed = read_gpu_f32(device, queue, &viz.buffers[display.index], size_bytes);
        assert_eq!(
            packed.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "paused Direct copy must match the latest ordinary-step state"
        );
        let pressure_values: Vec<f32> = packed
            .chunks_exact(ports.stride as usize)
            .map(|cell| cell[pressure_offset as usize])
            .collect();
        let expected_min = pressure_values
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let expected_max = pressure_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        viz.range_reducer
            .poll_nonblocking()
            .expect("paused Direct range poll");
        let ranges = viz
            .range_reducer
            .take_readback_for_sequence(display.sequence)
            .expect("paused Direct sequence-matched range");
        let pressure_range = ranges.range(cfd_renderer::CfdRangeField::Pressure);
        let expected_pressure_range = expected_direct_range(expected_min, expected_max);
        assert_eq!(pressure_range, expected_pressure_range);

        let velocity_magnitude_values: Vec<f32> = packed
            .chunks_exact(ports.stride as usize)
            .map(|cell| {
                let ux = cell[velocity_offset as usize];
                let uy = cell[velocity_offset as usize + 1];
                ux.hypot(uy)
            })
            .collect();
        let (expected_velocity_min, expected_velocity_max) = velocity_magnitude_values
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), value| {
                (lo.min(value), hi.max(value))
            });
        let expected_velocity_range =
            expected_direct_range(expected_velocity_min, expected_velocity_max);
        let velocity_magnitude_range =
            ranges.range(cfd_renderer::CfdRangeField::VelocityMagnitude);
        for (actual, expected) in velocity_magnitude_range
            .into_iter()
            .zip(expected_velocity_range)
        {
            assert!(
                (actual - expected).abs() <= 1.0e-6_f32.max(expected.abs() * 2.0e-5),
                "paused Direct |U| range mismatch: actual={actual} expected={expected}"
            );
        }
        DirectVizProbe {
            pressure_values,
            velocity_magnitude_values,
            pressure_range,
            velocity_magnitude_range,
        }
    }

    fn drive_real_gpu_autonomous_window(
        mut mode: SolverMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: RuntimeParams,
        policy: AutonomousRefillPolicy,
    ) -> DirectVizProbe {
        let size_bytes = mode.state_size_bytes();
        let ports = mode.ui_ports();
        let pressure_offset = ports
            .p_offset
            .expect("direct visualization probe requires pressure");
        let velocity_offset = ports
            .u_offset
            .expect("direct visualization probe requires velocity");
        let buffers: [wgpu::Buffer; 3] = std::array::from_fn(|index| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match index {
                    0 => "autonomous-smoke-viz-0",
                    1 => "autonomous-smoke-viz-1",
                    _ => "autonomous-smoke-viz-2",
                }),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        mode.install_viz_frame_targets(&buffers)
            .expect("install cached Direct frame targets");
        if let SolverMode::Static(driver) = &mode {
            assert_eq!(
                driver.autonomous_frame_copy_bind_group_count(),
                3,
                "one cached accepted-copy bind group per mailbox slot"
            );
        }
        let range_reducer = cfd_renderer::CfdRangeReducer::new(
            device,
            queue,
            &buffers,
            cfd_renderer::CfdFieldLayout {
                stride: ports.stride,
                u_offset: ports.u_offset.unwrap_or(0),
                p_offset: pressure_offset,
                has_u: ports.u_offset.is_some(),
                has_p: true,
            },
        );
        let mailbox = Arc::new(VizFrameMailbox::new());
        let viz = VizFieldBuffers {
            buffers,
            size_bytes,
            range_reducer,
            mailbox: Arc::clone(&mailbox),
        };
        mailbox.set_gpu_consumer_enabled(true);
        mailbox.request_frame();

        let (tx, rx) = mpsc::channel::<AutonomousCallbackMessage>();
        let mut scheduler = AutonomousBatchScheduler::default();
        scheduler.begin_run(policy);
        let initial_time = mode.sim_time_f64();
        let mut max_in_flight = 0usize;
        let mut requested_steps = 0_u64;

        while let Some(ticket) = scheduler.reserve(std::time::Instant::now()) {
            max_in_flight = max_in_flight.max(scheduler.in_flight.len());
            requested_steps = requested_steps.saturating_add(u64::from(ticket.requested_steps));
            assert!(scheduler.in_flight.len() <= AUTONOMOUS_MAX_IN_FLIGHT);
            let callback_tx = tx.clone();
            let submission = match &mut mode {
                SolverMode::Structured(s) => s.submit_autonomous_batch(
                    ticket.requested_steps as usize,
                    Some(params.target_cfl as f32),
                    move |result| {
                        let _ = callback_tx.send(AutonomousCallbackMessage {
                            ticket,
                            result: result.map(AutonomousBackendStatus::Structured),
                            completed_at: std::time::Instant::now(),
                        });
                    },
                ),
                SolverMode::Static(d) => d
                    .submit_autonomous_explicit_gpu_batch(ticket.requested_steps, move |result| {
                        let _ = callback_tx.send(AutonomousCallbackMessage {
                            ticket,
                            result: result.map(AutonomousBackendStatus::Unstructured),
                            completed_at: std::time::Instant::now(),
                        });
                    })
                    .map(|submission| Some(submission.submission_index)),
                _ => panic!("real autonomous smoke requires a GPU solver"),
            }
            .expect("health-ready autonomous submission");
            assert!(submission.is_some());
        }
        assert_eq!(max_in_flight, AUTONOMOUS_MAX_IN_FLIGHT);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        let mut accepted_steps = 0u64;
        let mut completed_time = initial_time;
        let mut frame_published = false;
        let mut refill_submitted = false;
        while scheduler.is_active() {
            assert!(
                std::time::Instant::now() < deadline,
                "autonomous completion callbacks did not drain before timeout"
            );
            match &mode {
                SolverMode::Structured(s) => s.poll_gpu_completions().unwrap(),
                SolverMode::Static(d) => d.poll_gpu_batch_completions().unwrap(),
                _ => unreachable!(),
            }

            while let Ok(message) = rx.try_recv() {
                let completion = match message.result.expect("mapped autonomous status") {
                    AutonomousBackendStatus::Structured(status) => AutonomousBatchCompletion {
                        ticket: message.ticket,
                        accepted_steps: status.accepted_steps,
                        completed_time: f64::from(status.time),
                        last_dt: status.dt_old,
                        next_dt: status.dt,
                        halted: status.halted,
                        invalid_count: status.invalid_cells,
                        completed_at: message.completed_at,
                        backend_status: Some(AutonomousBackendStatus::Structured(status)),
                    },
                    AutonomousBackendStatus::Unstructured(status) => AutonomousBatchCompletion {
                        ticket: message.ticket,
                        accepted_steps: status.accepted_batch,
                        completed_time: f64::from(status.time),
                        last_dt: status.last_dt,
                        next_dt: status.next_dt,
                        halted: status.halted,
                        invalid_count: status.invalid_count,
                        completed_at: message.completed_at,
                        backend_status: Some(AutonomousBackendStatus::Unstructured(status)),
                    },
                };
                let drain = scheduler.complete(completion);
                for retired in drain.retired.iter() {
                    assert_eq!(retired.ticket.backend_epoch, scheduler.backend_epoch);
                    match (retired.backend_status, &mut mode) {
                        (
                            Some(AutonomousBackendStatus::Structured(status)),
                            SolverMode::Structured(s),
                        ) => s.reconcile_autonomous_status(status),
                        (
                            Some(AutonomousBackendStatus::Unstructured(status)),
                            SolverMode::Static(d),
                        ) => d
                            .reconcile_autonomous_explicit_gpu_batch(status)
                            .expect("unstructured phase reconciliation"),
                        _ => panic!("status/backend mismatch"),
                    }
                }
                for completed in drain.completed.iter().copied() {
                    assert!(completed.is_healthy(), "GPU health status: {completed:?}");
                    assert!(completed.completed_time.is_finite());
                    assert!(completed.last_dt.is_finite() && completed.last_dt > 0.0);
                    assert!(completed.next_dt.is_finite() && completed.next_dt > 0.0);
                    accepted_steps =
                        accepted_steps.saturating_add(u64::from(completed.accepted_steps));
                    completed_time = completed.completed_time;
                    let frame_boundary = policy.is_continuous() || scheduler.in_flight.is_empty();
                    if frame_boundary {
                        if let Some(write) = mailbox.try_begin_write() {
                            let source = if matches!(mode, SolverMode::Static(_)) {
                                VizSnapshotSource::AutonomousAccepted
                            } else {
                                VizSnapshotSource::Current
                            };
                            viz.capture_snapshot(&mode, write, source)
                                .expect("accepted Direct snapshot");
                            frame_published = true;
                        }
                    }
                }

                // Retire/reconcile the front batch while the second remains
                // physically queued, then immediately append batch three. This
                // is the production continuous-refill seam: the unstructured
                // route must preserve its whole-cycle phase before encoding the
                // refill, and both routes must retain exactly two credits.
                if !refill_submitted
                    && scheduler.phase == AutonomousRunPhase::Running
                    && scheduler.in_flight.len() == 1
                {
                    let ticket = scheduler
                        .reserve(std::time::Instant::now())
                        .expect("front retirement must release one refill credit");
                    assert_eq!(scheduler.in_flight.len(), AUTONOMOUS_MAX_IN_FLIGHT);
                    requested_steps = requested_steps
                        .saturating_add(u64::from(ticket.requested_steps));
                    let callback_tx = tx.clone();
                    let submission = match &mut mode {
                        SolverMode::Structured(s) => s.submit_autonomous_batch(
                            ticket.requested_steps as usize,
                            Some(params.target_cfl as f32),
                            move |result| {
                                let _ = callback_tx.send(AutonomousCallbackMessage {
                                    ticket,
                                    result: result.map(AutonomousBackendStatus::Structured),
                                    completed_at: std::time::Instant::now(),
                                });
                            },
                        ),
                        SolverMode::Static(d) => d
                            .submit_autonomous_explicit_gpu_batch(
                                ticket.requested_steps,
                                move |result| {
                                    let _ = callback_tx.send(AutonomousCallbackMessage {
                                        ticket,
                                        result: result.map(AutonomousBackendStatus::Unstructured),
                                        completed_at: std::time::Instant::now(),
                                    });
                                },
                            )
                            .map(|submission| Some(submission.submission_index)),
                        _ => panic!("real autonomous smoke requires a GPU solver"),
                    }
                    .expect("continuous third-batch refill submission");
                    assert!(submission.is_some());
                    refill_submitted = true;
                    assert!(
                        !scheduler.request_pause(),
                        "pause must drain the queued second and third batches"
                    );
                }
            }
            thread::yield_now();
        }

        assert_eq!(scheduler.phase, AutonomousRunPhase::Paused);
        assert!(scheduler.in_flight.is_empty());
        assert!(refill_submitted, "real GPU path never submitted batch three");
        assert_eq!(accepted_steps, requested_steps);
        assert!(completed_time > initial_time);
        assert!(frame_published);
        if let SolverMode::Static(driver) = &mode {
            assert_eq!(
                driver.autonomous_frame_copy_bind_group_count(),
                3,
                "frame capture must not allocate additional bind groups"
            );
        }
        assert!(
            (0..3).any(|index| mailbox.state(index).1 == VizSlotState::Ready),
            "accepted boundary must publish a READY GPU-direct frame"
        );

        let display = mailbox
            .try_claim_latest_ready()
            .expect("accepted Direct frame must become displayable");
        let packed = read_gpu_f32(device, queue, &viz.buffers[display.index], size_bytes);
        let pressure_values: Vec<f32> = packed
            .chunks_exact(ports.stride as usize)
            .map(|cell| cell[pressure_offset as usize])
            .collect();
        let expected_min = pressure_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::INFINITY, f32::min);
        let expected_max = pressure_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        viz.range_reducer
            .poll_nonblocking()
            .expect("Direct range callback poll");
        let ranges = viz
            .range_reducer
            .take_readback_for_sequence(display.sequence)
            .expect("Direct range must match displayed sequence");
        let pressure_range = ranges.range(cfd_renderer::CfdRangeField::Pressure);
        let expected_range = expected_direct_range(expected_min, expected_max);
        assert_eq!(
            pressure_range, expected_range,
            "Direct range reducer must consume the same packed snapshot the renderer binds"
        );
        let velocity_magnitude_values: Vec<f32> = packed
            .chunks_exact(ports.stride as usize)
            .map(|cell| {
                let ux = cell[velocity_offset as usize];
                let uy = cell[velocity_offset as usize + 1];
                ux.hypot(uy)
            })
            .collect();
        let (velocity_min, velocity_max) = velocity_magnitude_values
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), value| {
                (lo.min(value), hi.max(value))
            });
        let expected_velocity_range = expected_direct_range(velocity_min, velocity_max);
        let velocity_magnitude_range =
            ranges.range(cfd_renderer::CfdRangeField::VelocityMagnitude);
        for (actual, expected) in velocity_magnitude_range
            .into_iter()
            .zip(expected_velocity_range)
        {
            assert!(
                (actual - expected).abs() <= 1.0e-6_f32.max(expected.abs() * 2.0e-5),
                "default Direct |U| range mismatch: actual={actual} expected={expected}"
            );
        }
        DirectVizProbe {
            pressure_values,
            velocity_magnitude_values,
            pressure_range,
            velocity_magnitude_range,
        }
    }

    #[test]
    fn real_gpu_structured_adaptive_worker_window_advances_and_drains() {
        let ctx = match pollster::block_on(crate::solver::gpu::context::GpuContext::new(None, None))
        {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("skipping structured autonomous GPU smoke: {error}");
                return;
            }
        };
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let params = autonomous_allmach_params();
        let model = crate::solver::model::allmach_thermal_structured_model()
            .expect("structured all-Mach model");
        let grid = StructuredGrid::new(8, 4, 1.0, 0.5);
        let mut solver = StructuredGpuSolver::with_config(
            ctx,
            grid,
            &model,
            f64::from(params.requested_dt),
            1,
            Scheme::Upwind,
            GpuTimeScheme::RK4,
        )
        .expect("structured explicit solver");
        solver.set_fluid(f64::from(params.density), f64::from(params.viscosity));
        seed_structured_state(
            &mut solver,
            "allmach_thermal_structured",
            &params,
            grid.dx,
            grid.dy,
        );
        seed_structured_freestream(
            &mut solver,
            "allmach_thermal_structured",
            f64::from(params.inlet_velocity),
            &params,
        );
        refresh_structured_allmach_runtime_fields(&mut solver, &params, grid.dx, grid.dy);
        setup_structured_bcs(
            &mut solver,
            "allmach_thermal_structured",
            f64::from(params.inlet_velocity),
            4,
            &params,
        );

        // Production starts this pressure-based model from rest. Direct must
        // install the stage-time inlet target itself: no ordinary/Plot step is
        // allowed to prime these constants before autonomous marching.
        let expected_ramp = crate::sim::explicit_allmach_inlet_ramp_time(
            &params,
            (grid.dx * grid.dy).sqrt(),
            true,
        );
        assert_eq!(
            solver.inlet_velocity_for_test().to_bits(),
            params.inlet_velocity.to_bits()
        );
        assert_eq!(
            solver.inlet_ramp_time_for_test().to_bits(),
            expected_ramp.to_bits()
        );

        let mode = SolverMode::Structured(solver);
        let cold = probe_paused_plot_to_direct(&mode, &device, &queue);
        assert!(
            cold.velocity_magnitude_values
                .iter()
                .all(|value| value.to_bits() == 0.0_f32.to_bits()),
            "cold structured all-Mach regression must begin from rest"
        );
        let live = drive_real_gpu_autonomous_window(
            mode,
            &device,
            &queue,
            params,
            AutonomousRefillPolicy::Continuous,
        );
        assert!(
            live.velocity_magnitude_values
                .iter()
                .any(|value| value.is_finite() && *value > 0.0),
            "cold autonomous Direct run never applied the configured inlet target"
        );
        assert!(
            live.velocity_magnitude_values
                .windows(2)
                .any(|pair| pair[0].to_bits() != pair[1].to_bits()),
            "cold autonomous Direct run remained uniformly blue"
        );
    }

    #[test]
    fn real_gpu_unstructured_adaptive_worker_window_advances_and_drains() {
        let ctx = match pollster::block_on(crate::solver::gpu::context::GpuContext::new(None, None))
        {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("skipping unstructured autonomous GPU smoke: {error}");
                return;
            }
        };
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let params = autonomous_allmach_params();
        let mesh = generate_structured_rect_mesh(
            8,
            4,
            1.0,
            0.5,
            BoundarySides {
                left: BoundaryType::Inlet,
                right: BoundaryType::Outlet,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        );
        let n = mesh.num_cells();
        let model = allmach_thermal_model().expect("unstructured all-Mach model");
        let mut build = pollster::block_on(SolverDriver::build(
            &mesh,
            model,
            &params,
            &vec![(0.0, 0.0); n],
            &vec![0.0; n],
            Some(device.clone()),
            Some(queue),
        ))
        .expect("unstructured GPU driver");
        build.driver.apply_params(&params);

        // Deterministic nonuniform valid state. This makes a stale slot-zero
        // capture (the release-blocking all-blue failure) impossible to hide
        // behind an accidentally uniform flow field.
        let ports = build.driver.solver().ui_ports();
        let stride = ports.stride as usize;
        let p_offset = ports.p_offset.expect("all-Mach pressure offset") as usize;
        let mut packed = pollster::block_on(build.driver.solver().read_state_f32());
        for (cell, values) in packed.chunks_exact_mut(stride).enumerate() {
            values[p_offset] = cell as f32 * 0.01;
        }
        build
            .driver
            .solver_mut()
            .write_state_f32(&packed)
            .expect("seed deterministic Direct field");

        // Evolve through the ordinary (Plot-compatible) route, then pause and
        // switch to Direct. This is the second all-blue release regression:
        // the initial slot must be replaced even though no next step exists.
        let outcome = build.driver.step(true);
        assert!(outcome.diverged.is_none(), "ordinary Plot step diverged");
        let mode = SolverMode::Static(build.driver);
        let paused_probe = probe_paused_plot_to_direct(&mode, &device, &ctx.queue);
        assert!(paused_probe.pressure_range[1] > paused_probe.pressure_range[0]);
        assert!(
            paused_probe.velocity_magnitude_range[1]
                > paused_probe.velocity_magnitude_range[0]
        );

        let probe = drive_real_gpu_autonomous_window(
            mode,
            &device,
            &ctx.queue,
            params,
            AutonomousRefillPolicy::ContinuousPhaseCycle3,
        );
        assert!(
            probe.pressure_range[1] > probe.pressure_range[0],
            "Direct pressure range must be nondegenerate"
        );
        assert!(
            probe.velocity_magnitude_range[1] > probe.velocity_magnitude_range[0],
            "default Direct velocity-magnitude range must be nondegenerate"
        );
        assert!(
            probe
                .pressure_values
                .windows(2)
                .any(|pair| pair[0].to_bits() != pair[1].to_bits()),
            "Direct accepted-state buffer must contain a nonconstant pressure field"
        );
    }

    #[test]
    fn viz_mailbox_coalesces_frames_and_never_exposes_writing_slots() {
        let mailbox = VizFrameMailbox::new();
        let initial = mailbox.initial_display();
        assert_eq!(mailbox.state(0), (0, VizSlotState::Display));
        assert_eq!(mailbox.state(1), (0, VizSlotState::Free));
        assert_eq!(mailbox.state(2), (0, VizSlotState::Free));
        assert!(mailbox.try_begin_write().is_none());

        // Three screen callbacks before an accepted solver boundary collapse
        // into one copy tagged with the newest request generation.
        assert_eq!(mailbox.request_frame(), 1);
        assert_eq!(mailbox.request_frame(), 2);
        assert_eq!(mailbox.request_frame(), 3);
        let write = mailbox.try_begin_write().expect("coalesced frame copy");
        assert_eq!(write.sequence, 3);
        assert_eq!(mailbox.state(write.index), (3, VizSlotState::Writing));
        assert!(
            mailbox.try_claim_latest_ready().is_none(),
            "the renderer must never observe a copy before queue submission"
        );

        assert!(mailbox.finish_write(write));
        assert!(mailbox.try_begin_write().is_none(), "one copy per request");
        let next = mailbox
            .try_claim_latest_ready()
            .expect("submitted snapshot becomes displayable");
        assert_eq!(next, write);

        // Claiming the replacement does not free the old field out from under
        // the renderer; rebinding is followed by an explicit release.
        assert_eq!(mailbox.state(initial.index), (0, VizSlotState::Display));
        assert_eq!(mailbox.state(next.index), (3, VizSlotState::Display));
        assert!(mailbox.release_display(initial));
        assert_eq!(mailbox.state(initial.index), (0, VizSlotState::Free));
    }

    #[test]
    fn initial_direct_activation_posts_one_cold_snapshot_request() {
        let mailbox = VizFrameMailbox::new();
        assert!(!mailbox.gpu_consumer_enabled());
        assert!(!mailbox.presentation_refresh_pending());

        assert!(activate_direct_viz(&mailbox));
        assert!(mailbox.gpu_consumer_enabled());
        assert!(mailbox.presentation_refresh_pending());
        let write = mailbox
            .try_begin_write()
            .expect("cold Direct activation must be serviceable");
        assert_eq!(write.sequence, 1);

        // Re-rendering Direct does not create a second request. The outstanding
        // generation remains the only work until the worker publishes it.
        assert!(!activate_direct_viz(&mailbox));
        assert!(mailbox.finish_write(write));
        assert!(mailbox.presentation_refresh_pending());
        let display = mailbox
            .try_claim_latest_ready()
            .expect("cold Direct snapshot must become displayable");
        assert_eq!(display.sequence, 1);
        assert!(!mailbox.presentation_refresh_pending());
    }

    #[test]
    fn stale_initialization_pause_cannot_cancel_a_newer_run_request() {
        let mut running = true;
        assert!(!apply_worker_running_event(7, &mut running, 6, false));
        assert!(running, "stale pause acknowledgment cancelled Run");

        assert!(apply_worker_running_event(7, &mut running, 7, false));
        assert!(!running, "current worker stop was not adopted");
    }

    #[test]
    fn direct_activation_edge_is_atomic_under_competing_ui_observers() {
        let mailbox = Arc::new(VizFrameMailbox::new());
        let start = Arc::new(std::sync::Barrier::new(9));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let mailbox = Arc::clone(&mailbox);
            let start = Arc::clone(&start);
            workers.push(thread::spawn(move || {
                start.wait();
                activate_direct_viz(&mailbox)
            }));
        }
        start.wait();
        let activated = workers
            .into_iter()
            .map(|worker| worker.join().expect("activation observer"))
            .filter(|activated| *activated)
            .count();

        assert_eq!(activated, 1, "only one false -> true edge may win");
        assert_eq!(mailbox.requested_sequence.load(Ordering::Acquire), 1);
        assert!(mailbox.gpu_consumer_enabled());
    }

    #[test]
    fn late_direct_frame_request_cannot_resurrect_plot_consumer() {
        let mailbox = VizFrameMailbox::new();
        assert!(activate_direct_viz(&mailbox));
        mailbox.set_gpu_consumer_enabled(false);

        // Emulate a Direct paint callback which observed the old mode before
        // the next UI update selected Plot. It may leave one coalesced request,
        // but request publication no longer owns the presentation-mode bit.
        assert_eq!(mailbox.request_frame(), 2);
        assert!(!mailbox.gpu_consumer_enabled());

        assert!(activate_direct_viz(&mailbox));
        assert_eq!(mailbox.requested_sequence.load(Ordering::Acquire), 3);
        assert!(mailbox.gpu_consumer_enabled());
    }

    #[test]
    fn viz_mailbox_cancelled_copy_retries_the_same_outstanding_request() {
        let mailbox = VizFrameMailbox::new();
        mailbox.request_frame();
        let failed = mailbox.try_begin_write().expect("first copy reservation");
        assert!(mailbox.cancel_write(failed));
        assert_eq!(mailbox.state(failed.index).1, VizSlotState::Free);

        let retry = mailbox
            .try_begin_write()
            .expect("cancelled request must remain outstanding");
        assert_eq!(retry.sequence, failed.sequence);
        assert!(mailbox.finish_write(retry));
    }

    #[test]
    fn viz_mailbox_generation_prevents_aba_and_discards_stale_ready_frames() {
        let mailbox = VizFrameMailbox::new();
        let initial = mailbox.initial_display();

        mailbox.request_frame();
        let first = mailbox.try_begin_write().unwrap();
        assert!(mailbox.finish_write(first));
        mailbox.request_frame();
        let second = mailbox.try_begin_write().unwrap();
        assert!(mailbox.finish_write(second));

        let displayed = mailbox.try_claim_latest_ready().unwrap();
        assert_eq!(displayed.sequence, 2);
        assert_eq!(
            mailbox.state(first.index).1,
            VizSlotState::Free,
            "an older completed frame is coalesced without becoming visible"
        );
        assert!(mailbox.release_display(initial));

        // Cycle until `first.index` is displayed again with a newer sequence.
        let mut current = displayed;
        let recycled = loop {
            mailbox.request_frame();
            let write = mailbox.try_begin_write().unwrap();
            assert!(mailbox.finish_write(write));
            let next = mailbox.try_claim_latest_ready().unwrap();
            assert!(mailbox.release_display(current));
            current = next;
            if current.index == first.index {
                break current;
            }
        };
        assert!(recycled.sequence > first.sequence);

        // An index-only protocol would free the newly displayed buffer here.
        // The stale generation is part of the expected CAS word, so it cannot.
        assert!(!mailbox.release_display(first));
        assert_eq!(
            mailbox.state(recycled.index),
            (recycled.sequence, VizSlotState::Display)
        );
    }

    #[test]
    fn completed_step_timing_ignores_unfenced_submission_latency() {
        let start = std::time::Instant::now();
        let mut timing = CompletedStepTiming::new_at(start);

        let first = timing.record_step_at(0.02, false, start + std::time::Duration::from_millis(1));
        assert_eq!(first.step_time_ms, 0.0);

        // The first completion only anchors the next honest window. Time from
        // worker creation/reset is not representative steady-state throughput.
        let anchored =
            timing.record_step_at(0.02, true, start + std::time::Duration::from_millis(40));
        assert_eq!(anchored.step_time_ms, 0.0);

        // An asynchronous enqueue cannot manufacture a throughput sample from
        // its tiny host return time.
        let retained =
            timing.record_step_at(0.01, false, start + std::time::Duration::from_millis(41));
        assert_eq!(retained.step_time_ms, 0.0);

        let completed =
            timing.record_step_at(0.01, true, start + std::time::Duration::from_millis(80));
        assert!((completed.step_time_ms - 20.0).abs() < 1.0e-5);
        assert!((completed.steps_per_second - 50.0).abs() < 1.0e-5);
        assert!((completed.sim_seconds_per_wall_second - 0.5).abs() < 1.0e-5);

        let retained =
            timing.record_step_at(0.01, false, start + std::time::Duration::from_millis(81));
        assert_eq!(retained.step_time_ms, completed.step_time_ms);
        assert_eq!(retained.steps_per_second, completed.steps_per_second);
    }

    #[test]
    fn structured_adaptive_rk4_uses_one_packed_read_per_accepted_step_after_bootstrap() {
        STRUCTURED_PACKED_STATE_READS.with(|count| count.set(0));
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let params =
            gui_defaults_for("allmach_thermal_structured").to_runtime_params(1.225, 1.81e-5, air);
        let smoke = structured_allmach_rk4_smoke("backstep", false, 12, 4, 3, params)
            .expect("structured adaptive RK4 smoke");
        assert_eq!(smoke.steps, 3);
        let reads = STRUCTURED_PACKED_STATE_READS.with(std::cell::Cell::get);
        assert_eq!(
            reads, 4,
            "one initial stability bootstrap plus one accepted-state read per step"
        );
    }

    #[test]
    fn allmach_structured_outlet_uses_live_back_pressure() {
        let model = crate::solver::model::allmach_thermal_structured_model().unwrap();
        let grid = StructuredGrid::new(4, 3, 1.0, 1.0);
        let mut solver = StructuredModelSolver::with_config(
            grid,
            &model,
            1.0e-3,
            1,
            Scheme::Upwind,
            GpuTimeScheme::RK4,
        )
        .unwrap();
        let mut params = gui_defaults_for("allmach_thermal_structured").to_runtime_params(
            1.0,
            1.0e-3,
            EosSpec::Constant,
        );
        params.pressure_inlet = false;
        params.outlet_back_pressure = -0.03125;
        setup_structured_bcs(
            &mut solver,
            "allmach_thermal_structured",
            params.inlet_velocity as f64,
            4,
            &params,
        );

        let right_cell = 4 + 3;
        assert_eq!(solver.bc_kind_at(right_cell, 2, 2), 1);
        assert_eq!(
            solver.bc_value_at(right_cell, 2, 2),
            params.outlet_back_pressure as f64
        );

        params.pressure_inlet = true;
        params.inlet_pressure = 0.125;
        setup_structured_bcs(
            &mut solver,
            "allmach_thermal_structured",
            params.inlet_velocity as f64,
            4,
            &params,
        );
        assert_eq!(solver.bc_kind_at(right_cell, 2, 2), 2);
        let left_cell = 4;
        assert_eq!(solver.bc_kind_at(left_cell, 1, 2), 1);
        assert_eq!(
            solver.bc_value_at(left_cell, 1, 2),
            params.inlet_pressure as f64
        );
    }
}

/// One row of the explicit-RK4 capability inventory derived from the actual
/// GUI model dropdown.  A rejected model remains in the inventory with its
/// validator diagnostic; callers must not treat it as a skipped test case.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuiExplicitRk4Capability {
    pub topology: &'static str,
    pub model_id: &'static str,
    pub supported: bool,
    pub rejection: Option<String>,
}

/// Return the RK4 capability of every model the two real GUI dropdowns expose.
#[doc(hidden)]
pub fn gui_explicit_rk4_capability_matrix(
) -> Result<Vec<GuiExplicitRk4Capability>, String> {
    let air = Fluid::presets()
        .into_iter()
        .find(|fluid| fluid.name == "Air")
        .ok_or("Air GUI fluid preset is missing")?;
    let mut rows = Vec::new();
    for (mesh_mode, topology) in [
        (MeshMode::Unstructured, "unstructured"),
        (MeshMode::Structured2D, "structured"),
    ] {
        for (model_id, _) in CFDApp::supported_ui_models(mesh_mode) {
            let model = if mesh_mode == MeshMode::Structured2D {
                structured_model_by_id(model_id)?
            } else {
                unstructured_model_by_id(model_id, air.eos)?
            };
            let rejection = model.validate_explicit_rk4().err();
            rows.push(GuiExplicitRk4Capability {
                topology,
                model_id,
                supported: rejection.is_none(),
                rejection,
            });
        }
    }
    Ok(rows)
}

/// A headless case selected only through controls that exist in the GUI.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct GuiExplicitRk4Case<'a> {
    pub model_id: &'a str,
    /// Exact entry from the GUI fluid preset dropdown.
    pub fluid: &'a str,
    pub geometry: &'a str,
    /// `structured` for dense Cartesian mode; otherwise one of the real
    /// unstructured mesh radio values (`cutcell`, `delaunay`, `voronoi`, `cvt`,
    /// or nozzle-only `fitted`).
    pub mesh_kind: &'a str,
    /// `gpu`, `cpu-interpreter`, `cpu-transpiled`, or
    /// `cpu-transpiled-simd`, matching the compute dropdown.
    pub backend: &'a str,
    pub adaptive: bool,
    /// `direct` or `plot`.  Direct GPU cases take the autonomous controller
    /// whenever that exact solver reports the required health capability.
    pub presentation: &'a str,
    pub moving_mesh: bool,
    pub cell_size: f64,
    pub steps: usize,
    /// Optional exact value of the GUI timestep slider. `None` uses the lean
    /// matrix seed (the smaller of the model default and 5e-6 s).
    pub requested_dt: Option<f32>,
    /// Optional exact advection radio selection. `None` uses the model's real
    /// GUI default.
    pub advection_scheme: Option<Scheme>,
    /// Optional inlet-velocity slider override. `None` uses the model's real
    /// GUI default.
    pub inlet_velocity: Option<f32>,
    /// Optional gauge inlet-pressure slider override (pressure-inlet cases).
    /// `None` uses the model's real GUI default.
    pub inlet_pressure: Option<f32>,
}

/// Stability observations from the real GUI mesh/model/seed/BC/runtime path.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GuiExplicitRk4Smoke {
    pub topology: &'static str,
    pub model_id: String,
    pub backend: String,
    pub presentation: String,
    /// `ordinary` is the readback-capable Plot/CPU router; `autonomous` is the
    /// health-audited Direct GPU router.
    pub route: &'static str,
    pub cells: usize,
    pub steps: usize,
    pub final_time: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub min_rho: Option<f64>,
    pub min_temperature: Option<f64>,
    pub min_pressure: Option<f64>,
    pub min_total_energy_density: Option<f64>,
    pub min_internal_energy_density: Option<f64>,
    pub packed_state: Vec<f32>,
    /// Cell-center positions, index-aligned with the packed state rows
    /// (structured: row-major `p = j*nx + i`; unstructured: mesh centroids).
    pub cell_centers: Vec<(f64, f64)>,
    /// Structured Brinkman-solid mask (all-`false` for unstructured meshes,
    /// which have no cells inside the solid).
    pub cell_solid: Vec<bool>,
    /// Per-cell velocity (`u` primitive / `U` state vector), absolute.
    pub velocity: Vec<(f32, f32)>,
    /// Per-cell pressure AS STORED (gauge for the compressible family; both
    /// topologies share the same gauge references, so values compare 1:1).
    pub pressure: Vec<f32>,
    /// Per-cell density as stored, when the model has a `rho` field.
    pub density: Option<Vec<f32>>,
    /// Per-cell temperature, when the model has a `T` field.
    pub temperature: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy)]
enum GuiMatrixBackend {
    Gpu,
    CpuInterpreter,
    CpuTranspiled,
    CpuTranspiledSimd,
}

impl GuiMatrixBackend {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "gpu" => Ok(Self::Gpu),
            "cpu-interpreter" => Ok(Self::CpuInterpreter),
            "cpu-transpiled" => Ok(Self::CpuTranspiled),
            "cpu-transpiled-simd" => Ok(Self::CpuTranspiledSimd),
            other => Err(format!(
                "unknown GUI compute backend '{other}'; expected gpu, cpu-interpreter, cpu-transpiled, or cpu-transpiled-simd"
            )),
        }
    }

    fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

fn gui_matrix_geometry(value: &str) -> Result<GeometryType, String> {
    match value {
        "backstep" => Ok(GeometryType::BackwardsStep),
        "obstacle" => Ok(GeometryType::ChannelObstacle),
        "nozzle" => Ok(GeometryType::Nozzle),
        other => Err(format!(
            "unknown GUI geometry '{other}'; expected backstep, obstacle, or nozzle"
        )),
    }
}

fn gui_matrix_mesh(value: &str, geometry: GeometryType) -> Result<MeshType, String> {
    match value {
        "cutcell" => Ok(MeshType::CutCell),
        "delaunay" => Ok(MeshType::Delaunay),
        "voronoi" => Ok(MeshType::Voronoi),
        "cvt" => Ok(MeshType::VoronoiCvt),
        "fitted" if geometry == GeometryType::Nozzle => Ok(MeshType::Fitted),
        "fitted" => Err("the fitted GUI mesh is nozzle-only".to_string()),
        "structured" => Err(
            "the structured mesh token is valid only for a *_structured GUI model".to_string(),
        ),
        other => Err(format!(
            "unknown GUI mesh '{other}'; expected cutcell, delaunay, voronoi, cvt, or fitted"
        )),
    }
}

fn gui_matrix_params(
    model_id: &str,
    geometry: GeometryType,
    adaptive: bool,
    fluid_name: &str,
) -> Result<(Fluid, RuntimeParams), String> {
    let fluid = Fluid::presets()
        .into_iter()
        .find(|fluid| fluid.name == fluid_name)
        .ok_or_else(|| format!("unknown GUI fluid preset '{fluid_name}'"))?;
    let is_allmach = matches!(
        model_id,
        "allmach_pressure" | "allmach_thermal" | "allmach_thermal_structured"
    );
    let is_compressible = matches!(model_id, "compressible" | "compressible_structured");
    let defaults = if geometry == GeometryType::Nozzle && is_allmach {
        crate::ui::model_defaults::ALLMACH_THERMAL_NOZZLE
    } else if geometry == GeometryType::Nozzle && is_compressible {
        crate::ui::model_defaults::COMPRESSIBLE_NOZZLE
    } else {
        crate::ui::model_defaults::gui_defaults_for(model_id)
    };
    let mut params = defaults.to_runtime_params(
        fluid.density as f32,
        fluid.viscosity as f32,
        fluid.eos,
    );
    params.time_scheme = GpuTimeScheme::RK4;
    params.adaptive_dt = adaptive;
    // This is an ordinary, user-selectable timestep value.  It is below the
    // acoustic limit of the deliberately lean matrix meshes and below the
    // structured Brinkman reaction limit, so the fixed-dt row tests RK4 rather
    // than intentionally asking the stability region to reject an oversized
    // implicit-default slider value.
    params.requested_dt = params.requested_dt.min(5.0e-6);
    Ok((fluid, params))
}

fn gui_matrix_gpu_context(
) -> Result<crate::solver::gpu::context::GpuContext, String> {
    use crate::solver::gpu::context::GpuContext;
    static CONTEXT: std::sync::OnceLock<Result<GpuContext, String>> =
        std::sync::OnceLock::new();
    let shared = CONTEXT.get_or_init(|| pollster::block_on(GpuContext::new(None, None)));
    match shared {
        Ok(context) => Ok(GpuContext {
            device: context.device.clone(),
            queue: context.queue.clone(),
            timestamp_query: context.timestamp_query,
            timestamps_inside_encoders: context.timestamps_inside_encoders,
            timestamp_period_ns: context.timestamp_period_ns,
            pipeline_cache: Arc::clone(&context.pipeline_cache),
        }),
        Err(error) => Err(error.clone()),
    }
}

/// Cheap adapter probe used by the matrix gate to distinguish a legitimate
/// headless skip from a solver construction failure.
#[doc(hidden)]
pub fn gui_explicit_rk4_gpu_available() -> Result<(), String> {
    gui_matrix_gpu_context().map(|_| ())
}

static GUI_MATRIX_CPU_ENV_LOCK: Mutex<()> = Mutex::new(());

struct GuiMatrixEnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl GuiMatrixEnvRestore {
    fn install(backend: GuiMatrixBackend) -> Self {
        const KEYS: [&str; 5] = [
            "CFD2_BACKEND",
            "CFD2_CPU_ENGINE",
            "CFD2_CPU_THREADS",
            "CFD2_CPU_SIMD",
            "CFD2_CPU_PRECISION",
        ];
        let old = KEYS
            .iter()
            .map(|&key| (key, std::env::var_os(key)))
            .collect();
        if backend.is_gpu() {
            std::env::remove_var("CFD2_BACKEND");
        } else {
            std::env::set_var("CFD2_BACKEND", "cpu");
            std::env::set_var(
                "CFD2_CPU_ENGINE",
                if matches!(backend, GuiMatrixBackend::CpuInterpreter) {
                    "interpreter"
                } else {
                    "transpiled"
                },
            );
            std::env::set_var("CFD2_CPU_THREADS", "1");
            std::env::set_var(
                "CFD2_CPU_SIMD",
                if matches!(backend, GuiMatrixBackend::CpuTranspiledSimd) {
                    "1"
                } else {
                    "0"
                },
            );
            std::env::set_var("CFD2_CPU_PRECISION", "f64");
        }
        Self(old)
    }
}

impl Drop for GuiMatrixEnvRestore {
    fn drop(&mut self) {
        for (key, value) in self.0.drain(..) {
            if let Some(value) = value {
                std::env::set_var(key, value);
            } else {
                std::env::remove_var(key);
            }
        }
    }
}

fn gui_matrix_state_minima(
    model_id: &str,
    layout: &crate::solver::model::backend::state_layout::StateLayout,
    state: &[f32],
    eos: EosSpec,
    // Gauge-storage references `[rho_ref, p_ref, e_ref]` active on the run
    // (zero = absolute storage): positivity is checked on ABSOLUTE values.
    gauge: [f32; 3],
) -> Result<
    (
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    ),
    String,
> {
    let stride = layout.stride() as usize;
    if stride == 0 || state.len() % stride != 0 || state.is_empty() {
        return Err(format!(
            "GUI RK4 {model_id} returned invalid packed-state length {} for stride {stride}",
            state.len()
        ));
    }
    if let Some((index, value)) = state
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "GUI RK4 {model_id} returned non-finite state[{index}]={value}"
        ));
    }

    let offset = |name: &str| layout.offset_for(name).map(|value| value as usize);
    let rows = || state.chunks_exact(stride);
    let min_rho = offset("rho").map(|rho| {
        rows()
            .map(|row| f64::from(row[rho] + gauge[0]))
            .fold(f64::INFINITY, f64::min)
    });
    // FLOORED-RECOVERY CONTRACT: the production compressible models clamp
    // the recovered p/T at 1 Pa / 1 K instead of halting, so conserved
    // rho/rho_e positivity is no longer a run-fatal invariant here — only
    // non-finite values are (checked above). Minima keep being REPORTED so
    // gate tests can still assert positivity on healthy cases.
    let _ = &min_rho;
    let min_temperature = offset("T").map(|temperature| {
        rows()
            .map(|row| row[temperature] as f64)
            .fold(f64::INFINITY, f64::min)
    });
    let _ = &min_temperature;

    let conserved_offsets = (
        offset("rho"),
        offset("rho_u"),
        offset("rho_e"),
    );
    let min_total_energy_density = conserved_offsets.2.map(|total_energy| {
        rows()
            .map(|row| f64::from(row[total_energy] + gauge[2]))
            .fold(f64::INFINITY, f64::min)
    });
    let _ = &min_total_energy_density;
    let min_pressure = if conserved_offsets.2.is_some() {
        offset("p").map(|pressure| {
            rows()
                .map(|row| f64::from(row[pressure] + gauge[1]))
                .fold(f64::INFINITY, f64::min)
        })
    } else {
        None
    };
    let _ = &min_pressure;
    let min_internal_energy_density = match conserved_offsets {
        (Some(rho), Some(momentum), Some(total_energy)) => Some(
            rows()
                .map(|row| {
                    let density = f64::from(row[rho] + gauge[0]);
                    let mx = row[momentum] as f64;
                    let my = row[momentum + 1] as f64;
                    f64::from(row[total_energy] + gauge[2])
                        - (mx * mx + my * my) / (2.0 * density)
                })
                .fold(f64::INFINITY, f64::min),
        ),
        _ => None,
    };
    let internal_invalid = match eos {
        EosSpec::IdealGas { .. } => {
            min_internal_energy_density.is_some_and(|value| !(value > 0.0))
        }
        // A barotropic internal-energy primitive has an arbitrary additive
        // C*rho gauge. GUI liquid presets choose zero at rho_ref, and the
        // thermodynamically compatible energy is negative for a sufficiently
        // small rarefaction about that reference. Its sign is therefore not a
        // domain invariant; finite rho, pressure, and sound-speed closure are.
        EosSpec::LinearCompressibility { .. } | EosSpec::Constant => false,
    };
    let _ = internal_invalid;
    Ok((
        min_rho,
        min_temperature,
        min_pressure,
        min_total_energy_density,
        min_internal_energy_density,
    ))
}

fn gui_matrix_state_diagnostic(
    layout: &crate::solver::model::backend::state_layout::StateLayout,
    state: &[f32],
    mesh: Option<&Mesh>,
) -> String {
    let stride = layout.stride() as usize;
    if stride == 0 || state.len() % stride != 0 {
        return format!("state_len={} stride={stride}", state.len());
    }
    let nonfinite_cells: Vec<usize> = state
        .chunks_exact(stride)
        .enumerate()
        .filter_map(|(cell, row)| row.iter().any(|value| !value.is_finite()).then_some(cell))
        .collect();
    let nonfinite_boundary_cells = mesh.map(|mesh| {
        nonfinite_cells
            .iter()
            .filter(|&&cell| {
                mesh.cell_faces[mesh.cell_face_offsets[cell]..mesh.cell_face_offsets[cell + 1]]
                    .iter()
                    .any(|&face| mesh.face_boundary[face].is_some())
            })
            .count()
    });
    let mut fields = Vec::new();
    for name in ["rho", "p", "T", "psi_precond", "d_p", "rho_e"] {
        let Some(offset) = layout.offset_for(name).map(|value| value as usize) else {
            continue;
        };
        let (min, max) = state
            .chunks_exact(stride)
            .map(|row| row[offset])
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            });
        fields.push(format!("{name}=[{min:.6e},{max:.6e}]"));
    }
    format!(
        "nonfinite_rows={} boundary_nonfinite={nonfinite_boundary_cells:?} bad_cells={nonfinite_cells:?} {}",
        nonfinite_cells.len(),
        fields.join(" ")
    )
}

/// Cell centers + Brinkman solid mask for a structured grid, index-aligned
/// with the packed state rows (row-major `p = j*nx + i`).
fn structured_cell_geometry(
    grid: &StructuredGrid,
    geometry: GeometryType,
) -> (Vec<(f64, f64)>, Vec<bool>) {
    let cells = grid.num_cells();
    let mut centers = Vec::with_capacity(cells);
    let mut solid = Vec::with_capacity(cells);
    for p in 0..cells {
        let (x, y) = grid.cell_center(p);
        centers.push((x, y));
        solid.push(structured_geometry_is_solid(geometry, x, y));
    }
    (centers, solid)
}

/// Split the packed state into per-cell physical samples for cross-topology
/// comparison: velocity (`u` primitive, falling back to the `U` state vector),
/// pressure, and — where present — density and temperature, all AS STORED.
#[allow(clippy::type_complexity)]
fn gui_matrix_cell_fields(
    layout: &crate::solver::model::backend::state_layout::StateLayout,
    state: &[f32],
) -> Result<
    (
        Vec<(f32, f32)>,
        Vec<f32>,
        Option<Vec<f32>>,
        Option<Vec<f32>>,
    ),
    String,
> {
    let stride = layout.stride() as usize;
    let offset = |name: &str| layout.offset_for(name).map(|value| value as usize);
    let velocity_offset = offset("u")
        .or_else(|| offset("U"))
        .ok_or("explicit GUI model exposes neither a 'u' nor a 'U' velocity field")?;
    let pressure_offset =
        offset("p").ok_or("explicit GUI model exposes no 'p' pressure field")?;
    let rows = || state.chunks_exact(stride);
    let velocity = rows()
        .map(|row| (row[velocity_offset], row[velocity_offset + 1]))
        .collect();
    let pressure = rows().map(|row| row[pressure_offset]).collect();
    let density = offset("rho").map(|rho| rows().map(|row| row[rho]).collect());
    let temperature = offset("T").map(|t| rows().map(|row| row[t]).collect());
    Ok((velocity, pressure, density, temperature))
}

/// Execute one bounded matrix row through the same GUI mesh/model/IC/BC/runtime
/// helpers and the same ordinary/autonomous routing policy as the worker.
#[doc(hidden)]
pub fn gui_explicit_rk4_smoke(
    case: GuiExplicitRk4Case<'_>,
) -> Result<GuiExplicitRk4Smoke, String> {
    if case.moving_mesh {
        return Err(
            "explicit RK4 is static-only; the GUI coerces a moving-mesh request to BDF2"
                .to_string(),
        );
    }
    if case.steps == 0 {
        return Err("GUI RK4 smoke requires at least one step".to_string());
    }
    if !case.cell_size.is_finite() || !(case.cell_size > 0.0) {
        return Err(format!(
            "GUI RK4 smoke cell size must be finite and positive, got {}",
            case.cell_size
        ));
    }
    let backend = GuiMatrixBackend::parse(case.backend)?;
    let direct = match case.presentation {
        "direct" => true,
        "plot" => false,
        other => {
            return Err(format!(
                "unknown GUI presentation '{other}'; expected direct or plot"
            ))
        }
    };
    let geometry = gui_matrix_geometry(case.geometry)?;
    let structured = case.model_id.ends_with("_structured");
    let mesh_mode = if structured {
        MeshMode::Structured2D
    } else {
        MeshMode::Unstructured
    };
    if !CFDApp::supported_ui_models(mesh_mode)
        .iter()
        .any(|(model_id, _)| *model_id == case.model_id)
    {
        return Err(format!(
            "model '{}' is not an actual {mesh_mode:?} GUI dropdown option",
            case.model_id
        ));
    }
    if structured && case.mesh_kind != "structured" {
        return Err(format!(
            "structured GUI model '{}' requires mesh_kind='structured'; its mesh dropdown is hidden",
            case.model_id
        ));
    }
    let mesh_kind = if structured {
        None
    } else {
        Some(gui_matrix_mesh(case.mesh_kind, geometry)?)
    };
    let (fluid, params) =
        gui_matrix_params(case.model_id, geometry, case.adaptive, case.fluid)?;
    let mut params = params;
    if let Some(requested_dt) = case.requested_dt {
        if !(requested_dt.is_finite() && requested_dt > 0.0) {
            return Err(format!(
                "GUI RK4 requested timestep must be finite and positive, got {requested_dt}"
            ));
        }
        params.requested_dt = requested_dt;
    }
    if let Some(advection_scheme) = case.advection_scheme {
        params.advection_scheme = advection_scheme;
    }
    if let Some(inlet_velocity) = case.inlet_velocity {
        if !inlet_velocity.is_finite() {
            return Err(format!(
                "GUI RK4 inlet velocity override must be finite, got {inlet_velocity}"
            ));
        }
        params.inlet_velocity = inlet_velocity;
    }
    if let Some(inlet_pressure) = case.inlet_pressure {
        if !inlet_pressure.is_finite() {
            return Err(format!(
                "GUI RK4 inlet pressure override must be finite, got {inlet_pressure}"
            ));
        }
        params.inlet_pressure = inlet_pressure;
    }

    let model = if structured {
        structured_model_by_id(case.model_id)?
    } else {
        unstructured_model_by_id(case.model_id, fluid.eos)?
    };
    model.validate_explicit_rk4().map_err(|error| {
        format!(
            "GUI model '{}' explicitly rejects RK4: {error}",
            case.model_id
        )
    })?;

    let mut min_dt = f64::INFINITY;
    let mut max_dt = 0.0_f64;
    let (route, cells, final_time, layout, packed_state, cell_centers, cell_solid) = if structured
    {
        let (lx, ly) = gui_geometry_domain(geometry);
        let nx = ((lx / case.cell_size).round() as usize).max(1);
        let ny = ((ly / case.cell_size).round() as usize).max(1);
        let grid = StructuredGrid::new(nx, ny, lx, ly);
        let unknowns = model.system.unknowns_per_cell() as usize;

        if backend.is_gpu() {
            let mut solver = StructuredGpuSolver::with_config(
                gui_matrix_gpu_context()?,
                grid,
                &model,
                params.requested_dt as f64,
                params.outer_iters.max(1) as usize,
                params.advection_scheme,
                GpuTimeScheme::RK4,
            )?;
            solver.set_fluid(params.density as f64, params.viscosity as f64);
            solver.set_outer_iters(params.outer_iters.max(1) as usize);
            solver.set_outer_auto_converge(params.outer_auto_converge);
            solver.set_alpha_u(params.alpha_u);
            solver.set_alpha_p(params.alpha_p);
            if case.model_id == "compressible_structured" {
                apply_structured_compressible_runtime(&mut solver, &params);
            }
            seed_structured_state(
                &mut solver,
                case.model_id,
                &params,
                grid.dx,
                grid.dy,
            );
            seed_structured_freestream(
                &mut solver,
                case.model_id,
                params.inlet_velocity as f64,
                &params,
            );
            seed_structured_ibm(&mut solver, geometry);
            setup_structured_bcs(
                &mut solver,
                case.model_id,
                params.inlet_velocity as f64,
                unknowns,
                &params,
            );

            let autonomous = direct
                && if params.adaptive_dt {
                    solver.supports_autonomous_adaptive()
                } else {
                    solver.supports_autonomous_fixed()
                };
            if autonomous {
                solver.step_autonomous_batch(
                    case.steps,
                    params.adaptive_dt.then_some(params.target_cfl as f32),
                )?;
                let status = solver
                    .autonomous_status()
                    .ok_or("structured Direct controller returned no status")?;
                if status.halted || status.invalid_cells != 0 {
                    return Err(format!(
                        "structured Direct controller rejected GUI case: {status:?}"
                    ));
                }
                if status.accepted_steps as usize != case.steps {
                    return Err(format!(
                        "structured Direct controller accepted {} of {} requested steps",
                        status.accepted_steps, case.steps
                    ));
                }
                min_dt = f64::from(status.dt_old.min(status.dt));
                max_dt = f64::from(status.dt_old.max(status.dt));
                solver.reconcile_autonomous_status(status);
            } else {
                let mut prev_max_vel = 0.0;
                let mut prev_explicit_rate = None;
                for step in 0..case.steps {
                    let outcome = structured_step(
                        &mut solver,
                        &params,
                        &mut prev_max_vel,
                        &mut prev_explicit_rate,
                        case.model_id,
                        true,
                    );
                    if let Some(reason) = outcome.diverged {
                        return Err(format!(
                            "structured GUI RK4 ordinary step {step}: {reason:?}"
                        ));
                    }
                    min_dt = min_dt.min(outcome.dt as f64);
                    max_dt = max_dt.max(outcome.dt as f64);
                }
            }
            let final_time = solver.time();
            let layout = solver.state_layout().clone();
            let state = solver.packed_state_f32();
            let (cell_centers, cell_solid) = structured_cell_geometry(&grid, geometry);
            (
                if autonomous { "autonomous" } else { "ordinary" },
                grid.num_cells(),
                final_time,
                layout,
                state,
                cell_centers,
                cell_solid,
            )
        } else {
            let mut solver = StructuredModelSolver::with_config(
                grid,
                &model,
                params.requested_dt as f64,
                params.outer_iters.max(1) as usize,
                params.advection_scheme,
                GpuTimeScheme::RK4,
            )?;
            let engine = if matches!(backend, GuiMatrixBackend::CpuInterpreter) {
                crate::solver::cpu::CpuEngine::Interpreter
            } else {
                // The GUI documents SIMD as a linear-solve optimization. RK4 has
                // no linear solve, so its transpiled and transpiled+SIMD rows
                // deliberately execute the same structured kernel engine.
                crate::solver::cpu::CpuEngine::Transpiled
            };
            solver.set_engine(engine, 1);
            solver.set_fluid(params.density as f64, params.viscosity as f64);
            solver.set_outer_iters(params.outer_iters.max(1) as usize);
            solver.set_outer_auto_converge(params.outer_auto_converge);
            solver.set_alpha_u(params.alpha_u);
            solver.set_alpha_p(params.alpha_p);
            if case.model_id == "compressible_structured" {
                apply_structured_compressible_runtime(&mut solver, &params);
            }
            seed_structured_state(
                &mut solver,
                case.model_id,
                &params,
                grid.dx,
                grid.dy,
            );
            seed_structured_freestream(
                &mut solver,
                case.model_id,
                params.inlet_velocity as f64,
                &params,
            );
            seed_structured_ibm(&mut solver, geometry);
            setup_structured_bcs(
                &mut solver,
                case.model_id,
                params.inlet_velocity as f64,
                unknowns,
                &params,
            );
            let mut prev_max_vel = 0.0;
            let mut prev_explicit_rate = None;
            for step in 0..case.steps {
                let outcome = structured_step(
                    &mut solver,
                    &params,
                    &mut prev_max_vel,
                    &mut prev_explicit_rate,
                    case.model_id,
                    true,
                );
                if let Some(reason) = outcome.diverged {
                    return Err(format!(
                        "structured CPU GUI RK4 step {step}: {reason:?}"
                    ));
                }
                min_dt = min_dt.min(outcome.dt as f64);
                max_dt = max_dt.max(outcome.dt as f64);
            }
            let (cell_centers, cell_solid) = structured_cell_geometry(&grid, geometry);
            (
                "ordinary",
                grid.num_cells(),
                solver.time(),
                solver.state_layout().clone(),
                solver.packed_state_f32(),
                cell_centers,
                cell_solid,
            )
        }
    } else {
        let mut trace = Vec::new();
        let mesh = CFDApp::build_mesh_with(
            geometry,
            mesh_kind.expect("unstructured mesh parsed"),
            case.cell_size,
            case.cell_size,
            1.2,
            &mut trace,
        );
        if mesh.cell_vol.iter().any(|volume| !volume.is_finite() || !(*volume > 0.0)) {
            return Err(format!(
                "GUI {geometry:?}/{:?} mesh contains a non-positive or non-finite cell volume",
                mesh_kind.expect("unstructured mesh parsed")
            ));
        }
        let initial_u = CFDApp::build_initial_velocity_with(
            &mesh,
            geometry,
            case.cell_size,
            params.inlet_velocity,
        );
        let initial_p = vec![0.0; mesh.num_cells()];
        let _env_lock = GUI_MATRIX_CPU_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = GuiMatrixEnvRestore::install(backend);
        let mut build = if backend.is_gpu() {
            let context = gui_matrix_gpu_context()?;
            pollster::block_on(SolverDriver::build(
                &mesh,
                model,
                &params,
                &initial_u,
                &initial_p,
                Some(context.device),
                Some(context.queue),
            ))?
        } else {
            pollster::block_on(SolverDriver::build_forced_cpu(
                &mesh,
                model,
                &params,
                &initial_u,
                &initial_p,
            ))?
        };
        build.driver.apply_params(&params);
        let autonomous = direct
            && backend.is_gpu()
            && build
                .driver
                .autonomous_explicit_capabilities()
                .is_some_and(|capabilities| {
                    capabilities.accepted_state_copy
                        && if params.adaptive_dt {
                            capabilities.adaptive_health
                        } else {
                            capabilities.fixed_health
                        }
                });
        if autonomous {
            if case.steps % 3 != 0 {
                return Err(format!(
                    "unstructured Direct autonomous GUI smoke requires a whole three-history cycle; got {} steps",
                    case.steps
                ));
            }
            let (tx, rx) = mpsc::channel();
            build.driver.submit_autonomous_explicit_gpu_batch(
                case.steps as u32,
                move |result| {
                    let _ = tx.send(result);
                },
            )?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            let completion = loop {
                build.driver.poll_gpu_batch_completions()?;
                if let Ok(result) = rx.try_recv() {
                    break result?;
                }
                if std::time::Instant::now() >= deadline {
                    return Err("unstructured Direct controller completion timed out".to_string());
                }
                thread::yield_now();
            };
            if completion.halted || completion.invalid_count != 0 {
                return Err(format!(
                    "unstructured Direct controller rejected GUI case: {completion:?}"
                ));
            }
            if completion.accepted_batch as usize != case.steps {
                return Err(format!(
                    "unstructured Direct controller accepted {} of {} requested steps",
                    completion.accepted_batch, case.steps
                ));
            }
            min_dt = f64::from(completion.last_dt.min(completion.next_dt));
            max_dt = f64::from(completion.last_dt.max(completion.next_dt));
            build
                .driver
                .reconcile_autonomous_explicit_gpu_batch(completion)?;
        } else {
            for step in 0..case.steps {
                let outcome = build.driver.step(true);
                if let Some(reason) = outcome.diverged {
                    let state = pollster::block_on(build.driver.solver().read_state_f32());
                    let diagnostic = gui_matrix_state_diagnostic(
                        &build.driver.solver().model().state_layout,
                        &state,
                        Some(&mesh),
                    );
                    return Err(format!(
                        "unstructured GUI RK4 ordinary step {step}: {reason:?}; {diagnostic}"
                    ));
                }
                if let Some(readback) = &outcome.readback {
                    if readback.stats.nonfinite_u != 0 || readback.stats.nonfinite_p != 0 {
                        return Err(format!(
                            "unstructured GUI RK4 ordinary step {step} returned non-finite fields"
                        ));
                    }
                }
                min_dt = min_dt.min(outcome.dt as f64);
                max_dt = max_dt.max(outcome.dt as f64);
            }
        }
        let cells = mesh.num_cells();
        let final_time = build.driver.solver().time() as f64;
        let layout = build.driver.solver().model().state_layout.clone();
        let state = pollster::block_on(build.driver.solver().read_state_f32());
        let cell_centers = mesh
            .cell_cx
            .iter()
            .zip(&mesh.cell_cy)
            .map(|(&x, &y)| (x, y))
            .collect();
        (
            if autonomous { "autonomous" } else { "ordinary" },
            cells,
            final_time,
            layout,
            state,
            cell_centers,
            vec![false; cells],
        )
    };

    if !(final_time.is_finite() && final_time > 0.0) {
        return Err(format!(
            "GUI RK4 '{}' did not advance to a finite positive time: {final_time}",
            case.model_id
        ));
    }
    if !(min_dt.is_finite() && min_dt > 0.0 && max_dt.is_finite() && max_dt > 0.0) {
        return Err(format!(
            "GUI RK4 '{}' returned invalid timestep range [{min_dt}, {max_dt}]",
            case.model_id
        ));
    }
    let (
        min_rho,
        min_temperature,
        min_pressure,
        min_total_energy_density,
        min_internal_energy_density,
    ) = gui_matrix_state_minima(
        case.model_id,
        &layout,
        &packed_state,
        params.eos,
        if matches!(case.model_id, "compressible" | "compressible_structured") {
            let gauged = params.eos.runtime_params_gauged(params.density as f64);
            [gauged.gauge_rho_ref, gauged.gauge_p_ref, gauged.gauge_e_ref]
        } else {
            [0.0; 3]
        },
    )?;
    let (velocity, pressure, density, temperature) =
        gui_matrix_cell_fields(&layout, &packed_state)?;
    Ok(GuiExplicitRk4Smoke {
        topology: if structured { "structured" } else { "unstructured" },
        model_id: case.model_id.to_string(),
        backend: case.backend.to_string(),
        presentation: case.presentation.to_string(),
        route,
        cells,
        steps: case.steps,
        final_time,
        min_dt,
        max_dt,
        min_rho,
        min_temperature,
        min_pressure,
        min_total_energy_density,
        min_internal_energy_density,
        packed_state,
        cell_centers,
        cell_solid,
        velocity,
        pressure,
        density,
        temperature,
    })
}

/// Headless observations from the exact unstructured GUI mesh, initial-field,
/// driver, adaptive-controller, and readback path.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredAllmachRk4Smoke {
    pub cells: usize,
    pub steps: usize,
    pub final_time: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub max_vel: f64,
    pub min_rho: f64,
    pub max_rho: f64,
    /// Volume-normalized energy in the actual conductance-weighted Rhie--Chow
    /// pressure bracket, relative to the de-meaned pressure energy and RMS
    /// cell conductance. This uses the solver's normalized WLS reconstruction,
    /// complete `rho*d_p` interpolation, IBM seal, and distance fallback.
    /// A smooth linear pressure field gives zero; a collocated checkerboard gives
    /// an O(1/h) bracket and therefore a large value.
    pub rhie_chow_bracket_fraction: f64,
    /// RMS visibility of the reconstructed conductance-weighted pressure term
    /// relative to the compact term. A true collocated checkerboard has an
    /// O(1) compact jump but an almost invisible reconstructed contribution.
    pub pressure_gradient_visibility: f64,
    /// RMS disagreement between the actual reconstructed and compact flux
    /// terms, normalized by compact-term energy.
    pub pressure_jump_defect: f64,
    /// Grid-Nyquist fractions of pressure on Cartesian-like meshes.  These are
    /// diagnostic only (the bracket fraction above is topology-independent).
    pub pressure_nyquist_x: f64,
    pub pressure_nyquist_y: f64,
    pub pressure_nyquist_xy: f64,
    /// RMS of the column-mean streamwise velocity after removal of a
    /// 17-column moving average, normalized by its RMS. This isolates the
    /// multi-cell vertical wave train visible during low-Mach startup; unlike
    /// the Nyquist metrics it is intentionally sensitive to wavelengths of
    /// roughly 4--32 cells.
    pub velocity_stripe_fraction: f64,
}

fn unstructured_rhie_chow_diagnostics(
    mesh: &Mesh,
    p: &[f64],
    density: &[f64],
    d_p: &[f64],
    penalty: Option<&[f64]>,
    params: &RuntimeParams,
    nominal_h: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    if p.len() != mesh.num_cells()
        || density.len() != mesh.num_cells()
        || d_p.len() != mesh.num_cells()
    {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let penalty = penalty.filter(|values| values.len() == mesh.num_cells());

    // Recompute grad(p) from the FINAL RK state. The stored grad_p belongs to
    // the k4 input abscissa until the next residual preparation, so reading it
    // immediately after a step would overstate the bracket defect.
    let grad_p = crate::sim::explicit_pressure_gradients(mesh, p, params);

    let total_volume = mesh.cell_vol.iter().sum::<f64>().max(1.0e-300);
    let mean = p
        .iter()
        .zip(&mesh.cell_vol)
        .map(|(&value, &volume)| value * volume)
        .sum::<f64>()
        / total_volume;
    let pressure_energy = p
        .iter()
        .zip(&mesh.cell_vol)
        .map(|(&value, &volume)| volume * (value - mean).powi(2))
        .sum::<f64>();
    let conductance_rms_sq = density
        .iter()
        .zip(d_p)
        .zip(&mesh.cell_vol)
        .map(|((&rho, &dp), &volume)| volume * (rho * dp).powi(2))
        .sum::<f64>()
        / total_volume;

    let mut bracket_energy = 0.0;
    let mut compact_jump_energy = 0.0;
    let mut reconstructed_gradient_energy = 0.0;
    let mut jump_defect_energy = 0.0;
    for face in 0..mesh.num_faces() {
        let Some(neighbor) = mesh.face_neighbor[face] else {
            continue;
        };
        let owner = mesh.face_owner[face];
        let wrap = mesh
            .face_wrap_shift
            .get(face)
            .copied()
            .unwrap_or([0.0, 0.0]);
        let dx = mesh.cell_cx[neighbor] + wrap[0] - mesh.cell_cx[owner];
        let dy = mesh.cell_cy[neighbor] + wrap[1] - mesh.cell_cy[owner];
        let projected = (dx * mesh.face_nx[face] + dy * mesh.face_ny[face]).abs();
        let normal_distance = if projected > 1.0e-6 {
            projected
        } else {
            dx.hypot(dy).max(1.0e-6)
        };
        let d_owner = ((mesh.face_cx[face] - mesh.cell_cx[owner]) * mesh.face_nx[face]
            + (mesh.face_cy[face] - mesh.cell_cy[owner]) * mesh.face_ny[face])
            .abs();
        let d_neighbor = ((mesh.cell_cx[neighbor] + wrap[0] - mesh.face_cx[face])
            * mesh.face_nx[face]
            + (mesh.cell_cy[neighbor] + wrap[1] - mesh.face_cy[face]) * mesh.face_ny[face])
            .abs();
        let lambda = if d_owner + d_neighbor > 1.0e-6 {
            d_neighbor / (d_owner + d_neighbor)
        } else {
            0.5
        };
        let other = 1.0 - lambda;
        let (reconstructed, compact) = if let Some(penalty) = penalty {
            let seal = 1.0 - (penalty[owner].abs() + penalty[neighbor].abs()).min(1.0);
            let kappa = (lambda * density[owner] + other * density[neighbor])
                * d_p[owner].min(d_p[neighbor]);
            let gx = lambda * grad_p[owner].0 + other * grad_p[neighbor].0;
            let gy = lambda * grad_p[owner].1 + other * grad_p[neighbor].1;
            (
                seal * kappa * (gx * dx + gy * dy) / normal_distance,
                seal * kappa * (p[neighbor] - p[owner]) / normal_distance,
            )
        } else {
            let k_owner = density[owner] * d_p[owner];
            let k_neighbor = density[neighbor] * d_p[neighbor];
            let kappa = lambda * k_owner + other * k_neighbor;
            let qx = lambda * k_owner * grad_p[owner].0 + other * k_neighbor * grad_p[neighbor].0;
            let qy = lambda * k_owner * grad_p[owner].1 + other * k_neighbor * grad_p[neighbor].1;
            (
                (qx * dx + qy * dy) / normal_distance,
                kappa * (p[neighbor] - p[owner]) / normal_distance,
            )
        };
        let bracket = reconstructed - compact;
        // In 2-D, A_f*d_f has volume units, so this is directly comparable
        // with the cell-volume-weighted pressure energy above.
        let weight = mesh.face_area[face] * normal_distance;
        bracket_energy += weight * bracket * bracket;
        compact_jump_energy += weight * compact * compact;
        reconstructed_gradient_energy += weight * reconstructed * reconstructed;
        jump_defect_energy += weight * bracket * bracket;
    }
    let h = nominal_h.abs().max(1.0e-14);
    let bracket_fraction = h
        * (bracket_energy / (conductance_rms_sq.max(1.0e-300) * pressure_energy.max(1.0e-300)))
            .sqrt();
    let (gradient_visibility, jump_defect) = if compact_jump_energy > 1.0e-300 {
        (
            (reconstructed_gradient_energy / compact_jump_energy).sqrt(),
            (jump_defect_energy / compact_jump_energy).sqrt(),
        )
    } else {
        (1.0, 0.0)
    };
    let rms = (pressure_energy / total_volume).sqrt();
    let nyquist = |mode: u8| {
        let projection = (0..mesh.num_cells())
            .map(|cell| {
                let i = (mesh.cell_cx[cell] / h - 0.5).round() as i64;
                let j = (mesh.cell_cy[cell] / h - 0.5).round() as i64;
                let parity = match mode {
                    0 => i,
                    1 => j,
                    _ => i + j,
                };
                let sign = if parity & 1 == 0 { 1.0 } else { -1.0 };
                mesh.cell_vol[cell] * (p[cell] - mean) * sign
            })
            .sum::<f64>()
            / total_volume;
        projection.abs() / rms.max(1.0e-300)
    };

    (
        bracket_fraction,
        gradient_visibility,
        jump_defect,
        nyquist(0),
        nyquist(1),
        nyquist(2),
    )
}

fn unstructured_velocity_stripe_fraction(
    mesh: &Mesh,
    velocity: &[(f64, f64)],
    nominal_h: f64,
) -> f64 {
    if velocity.len() != mesh.num_cells() {
        return f64::NAN;
    }
    let h = nominal_h.abs().max(1.0e-14);
    let min_i = mesh
        .cell_cx
        .iter()
        .map(|&x| (x / h - 0.5).round() as i64)
        .min()
        .unwrap_or(0);
    let max_i = mesh
        .cell_cx
        .iter()
        .map(|&x| (x / h - 0.5).round() as i64)
        .max()
        .unwrap_or(min_i);
    let n = (max_i - min_i + 1).max(0) as usize;
    let mut sum = vec![0.0; n];
    let mut weight = vec![0.0; n];
    for cell in 0..mesh.num_cells() {
        let i = ((mesh.cell_cx[cell] / h - 0.5).round() as i64 - min_i) as usize;
        sum[i] += mesh.cell_vol[cell] * velocity[cell].0;
        weight[i] += mesh.cell_vol[cell];
    }
    let columns: Vec<f64> = sum
        .iter()
        .zip(&weight)
        .map(|(&value, &volume)| value / volume.max(1.0e-300))
        .collect();
    if columns.len() < 19 {
        return 0.0;
    }
    let radius = 8usize;
    let mut residual_energy = 0.0;
    let mut samples = 0usize;
    for i in radius..columns.len() - radius {
        let smooth = columns[i - radius..=i + radius].iter().sum::<f64>() / (2 * radius + 1) as f64;
        residual_energy += (columns[i] - smooth).powi(2);
        samples += 1;
    }
    let amplitude = columns.iter().map(|value| value.abs()).fold(0.0, f64::max);
    (residual_energy / samples.max(1) as f64).sqrt() / amplitude.max(1.0e-300)
}

/// Headless static-GUI hook. `geometry` is `backstep`, `obstacle`, or
/// `nozzle`; `mesh_kind` is `cutcell`, `delaunay`, `voronoi`, `cvt`, or
/// `fitted` (the last is nozzle-only).
#[doc(hidden)]
pub fn unstructured_allmach_rk4_smoke(
    geometry: &str,
    mesh_kind: &str,
    gpu: bool,
    cell_size: f64,
    steps: usize,
    params: RuntimeParams,
) -> Result<UnstructuredAllmachRk4Smoke, String> {
    unstructured_allmach_rk4_smoke_impl(geometry, mesh_kind, gpu, cell_size, steps, None, params)
}

/// Exact [`unstructured_allmach_rk4_smoke`] path, stopped at the first
/// adaptive step ending at or beyond `target_time`. `max_steps` is a guard
/// against a controller regression that otherwise makes a test march forever.
#[doc(hidden)]
pub fn unstructured_allmach_rk4_smoke_to_time(
    geometry: &str,
    mesh_kind: &str,
    gpu: bool,
    cell_size: f64,
    target_time: f64,
    max_steps: usize,
    params: RuntimeParams,
) -> Result<UnstructuredAllmachRk4Smoke, String> {
    if !target_time.is_finite() || target_time <= 0.0 {
        return Err(format!(
            "unstructured all-Mach RK4 smoke target time must be finite and positive, got {target_time}"
        ));
    }
    unstructured_allmach_rk4_smoke_impl(
        geometry,
        mesh_kind,
        gpu,
        cell_size,
        max_steps,
        Some(target_time),
        params,
    )
}

fn unstructured_allmach_rk4_smoke_impl(
    geometry: &str,
    mesh_kind: &str,
    gpu: bool,
    cell_size: f64,
    max_steps: usize,
    target_time: Option<f64>,
    mut params: RuntimeParams,
) -> Result<UnstructuredAllmachRk4Smoke, String> {
    let geometry = match geometry {
        "backstep" => GeometryType::BackwardsStep,
        "obstacle" => GeometryType::ChannelObstacle,
        "nozzle" => GeometryType::Nozzle,
        other => return Err(format!("unknown GUI smoke geometry '{other}'")),
    };
    let mesh_kind = match mesh_kind {
        "cutcell" => MeshType::CutCell,
        "delaunay" => MeshType::Delaunay,
        "voronoi" => MeshType::Voronoi,
        "cvt" => MeshType::VoronoiCvt,
        "fitted" if geometry == GeometryType::Nozzle => MeshType::Fitted,
        "fitted" => return Err("the fitted mesh is nozzle-only".into()),
        other => return Err(format!("unknown GUI smoke mesh '{other}'")),
    };
    params.time_scheme = GpuTimeScheme::RK4;
    params.adaptive_dt = true;
    if !(params.compressibility_psi > 0.0) {
        return Err("unstructured all-Mach RK4 smoke requires positive compressibility".into());
    }

    let mut trace = Vec::new();
    let mesh = CFDApp::build_mesh_with(geometry, mesh_kind, cell_size, cell_size, 1.2, &mut trace);
    let initial_u =
        CFDApp::build_initial_velocity_with(&mesh, geometry, cell_size, params.inlet_velocity);
    let initial_p = vec![0.0; mesh.num_cells()];
    let model = allmach_thermal_model()?;
    let mut build = if gpu {
        let context = pollster::block_on(crate::solver::gpu::context::GpuContext::new(None, None))?;
        pollster::block_on(SolverDriver::build(
            &mesh,
            model,
            &params,
            &initial_u,
            &initial_p,
            Some(context.device),
            Some(context.queue),
        ))?
    } else {
        pollster::block_on(SolverDriver::build_forced_cpu_transpiled(
            &mesh, model, &params, &initial_u, &initial_p,
        ))?
    };
    build.driver.apply_params(&params);

    let mut smoke = UnstructuredAllmachRk4Smoke {
        cells: mesh.num_cells(),
        steps: 0,
        final_time: 0.0,
        min_dt: f64::INFINITY,
        max_dt: 0.0,
        max_vel: 0.0,
        min_rho: f64::INFINITY,
        max_rho: f64::NEG_INFINITY,
        rhie_chow_bracket_fraction: f64::NAN,
        pressure_gradient_visibility: f64::NAN,
        pressure_jump_defect: f64::NAN,
        pressure_nyquist_x: f64::NAN,
        pressure_nyquist_y: f64::NAN,
        pressure_nyquist_xy: f64::NAN,
        velocity_stripe_fraction: f64::NAN,
    };
    for step in 0..max_steps {
        if target_time.is_some_and(|target| build.driver.solver().time() as f64 >= target) {
            break;
        }
        let outcome = build.driver.step(true);
        if let Some(reason) = outcome.diverged {
            return Err(format!(
                "unstructured {geometry:?}/{mesh_kind:?} RK4 step {step}: {reason:?}"
            ));
        }
        smoke.steps += 1;
        smoke.min_dt = smoke.min_dt.min(outcome.dt as f64);
        smoke.max_dt = smoke.max_dt.max(outcome.dt as f64);
        if let Some(readback) = outcome.readback {
            smoke.max_vel = smoke.max_vel.max(readback.stats.max_vel);
            if let Some((lo, hi)) = readback.stats.rho {
                smoke.min_rho = smoke.min_rho.min(lo);
                smoke.max_rho = smoke.max_rho.max(hi);
            }
        }
    }
    smoke.final_time = build.driver.solver().time() as f64;
    if let Some(target) = target_time {
        if smoke.final_time < target {
            return Err(format!(
                "unstructured {geometry:?}/{mesh_kind:?} RK4 reached t={:.6e}, below target t={target:.6e}, after {max_steps} steps",
                smoke.final_time
            ));
        }
    }
    let p = pollster::block_on(build.driver.solver().get_field_scalar("p"))?;
    let temperature = pollster::block_on(build.driver.solver().get_field_scalar("T"))?;
    let dt_local = pollster::block_on(build.driver.solver().get_field_scalar("dt_local"))?;
    let penalty = pollster::block_on(build.driver.solver().get_field_scalar("ibm_penalty_U")).ok();
    let psi0 = (params.compressibility_psi as f64).max(0.0);
    let t_ref = crate::solver::model::ALLMACH_T_REF;
    let mut density = Vec::with_capacity(mesh.num_cells());
    let mut d_p = Vec::with_capacity(mesh.num_cells());
    for cell in 0..mesh.num_cells() {
        let numerator = params.density as f64 * t_ref
            + crate::solver::model::ALLMACH_GAMMA * psi0 * t_ref * p[cell];
        let rho = (numerator / temperature[cell]).max(psi0 * 1.0e-5);
        let dp0 = dt_local[cell].max(0.0) / rho.max(1.0e-30);
        let cell_penalty = penalty.as_ref().map_or(0.0, |values| values[cell].abs());
        density.push(rho);
        d_p.push(dp0 / (1.0 + cell_penalty * dp0));
    }
    let velocity = pollster::block_on(build.driver.solver().get_field_vec2("U"))?;
    let (bracket, gradient_visibility, jump_defect, nyquist_x, nyquist_y, nyquist_xy) =
        unstructured_rhie_chow_diagnostics(
            &mesh,
            &p,
            &density,
            &d_p,
            penalty.as_deref(),
            &params,
            cell_size,
        );
    smoke.rhie_chow_bracket_fraction = bracket;
    smoke.pressure_gradient_visibility = gradient_visibility;
    smoke.pressure_jump_defect = jump_defect;
    smoke.pressure_nyquist_x = nyquist_x;
    smoke.pressure_nyquist_y = nyquist_y;
    smoke.pressure_nyquist_xy = nyquist_xy;
    smoke.velocity_stripe_fraction =
        unstructured_velocity_stripe_fraction(&mesh, &velocity, cell_size);
    Ok(smoke)
}

/// Headless observations from the exact structured GUI seeding, boundary,
/// adaptive-controller, and step path.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct StructuredAllmachRk4Smoke {
    pub steps: usize,
    pub min_dt: f64,
    pub max_dt: f64,
    pub max_vel: f64,
    pub min_rho: f64,
    pub max_rho: f64,
}

/// Headless test hook for the structured all-Mach GUI path. `geometry` is one
/// of `backstep`, `obstacle`, or `nozzle`; `gpu=false` selects the transpiled
/// CPU backend and `gpu=true` selects the structured GPU backend.
#[doc(hidden)]
pub fn structured_allmach_rk4_smoke(
    geometry: &str,
    gpu: bool,
    nx: usize,
    ny: usize,
    steps: usize,
    params: RuntimeParams,
) -> Result<StructuredAllmachRk4Smoke, String> {
    structured_allmach_rk4_smoke_impl(geometry, gpu, nx, ny, steps, params, None)
}

/// Same exact structured GUI path, with a live parameter update applied halfway
/// through the run. Used to gate the worker's live fluid/CFL/BC refresh seam.
#[doc(hidden)]
pub fn structured_allmach_rk4_live_update_smoke(
    geometry: &str,
    gpu: bool,
    nx: usize,
    ny: usize,
    steps: usize,
    params: RuntimeParams,
    updated: RuntimeParams,
) -> Result<StructuredAllmachRk4Smoke, String> {
    structured_allmach_rk4_smoke_impl(geometry, gpu, nx, ny, steps, params, Some(updated))
}

fn structured_allmach_rk4_smoke_impl(
    geometry: &str,
    gpu: bool,
    nx: usize,
    ny: usize,
    steps: usize,
    mut params: RuntimeParams,
    mut live_update: Option<RuntimeParams>,
) -> Result<StructuredAllmachRk4Smoke, String> {
    let geometry = match geometry {
        "backstep" => GeometryType::BackwardsStep,
        "obstacle" => GeometryType::ChannelObstacle,
        "nozzle" => GeometryType::Nozzle,
        other => return Err(format!("unknown structured smoke geometry '{other}'")),
    };
    params.time_scheme = GpuTimeScheme::RK4;
    params.adaptive_dt = true;
    if let Some(updated) = &mut live_update {
        updated.time_scheme = GpuTimeScheme::RK4;
        updated.adaptive_dt = true;
    }
    if !(params.compressibility_psi > 0.0) {
        return Err("structured all-Mach RK4 smoke requires positive compressibility".into());
    }
    let (lx, ly) = (3.0, 1.0);
    let model = crate::solver::model::allmach_thermal_structured_model()?;
    let unknowns = model.system.unknowns_per_cell() as usize;

    fn drive(
        solver: &mut (impl StructuredSteppable + StructuredSeed),
        geometry: GeometryType,
        nx: usize,
        ny: usize,
        lx: f64,
        ly: f64,
        unknowns: usize,
        steps: usize,
        params: &RuntimeParams,
        live_update: Option<&RuntimeParams>,
    ) -> Result<StructuredAllmachRk4Smoke, String> {
        seed_structured_state(
            solver,
            "allmach_thermal_structured",
            params,
            lx / nx.max(1) as f64,
            ly / ny.max(1) as f64,
        );
        seed_structured_freestream(
            solver,
            "allmach_thermal_structured",
            params.inlet_velocity as f64,
            params,
        );
        seed_structured_ibm(solver, geometry);
        setup_structured_bcs(
            solver,
            "allmach_thermal_structured",
            params.inlet_velocity as f64,
            unknowns,
            params,
        );

        let mut active_params = *params;
        let mut prev_max_vel = 0.0;
        let mut prev_explicit_rate = None;
        let mut smoke = StructuredAllmachRk4Smoke {
            steps: 0,
            min_dt: f64::INFINITY,
            max_dt: 0.0,
            max_vel: 0.0,
            min_rho: f64::INFINITY,
            max_rho: f64::NEG_INFINITY,
        };
        for step in 0..steps {
            if step == steps / 2 {
                if let Some(updated) = live_update {
                    active_params = *updated;
                    prev_explicit_rate = None;
                    solver
                        .st_set_fluid(active_params.density as f64, active_params.viscosity as f64);
                    solver.st_set_alpha_u(active_params.alpha_u);
                    solver.st_set_alpha_p(active_params.alpha_p);
                    refresh_structured_allmach_runtime_fields(
                        solver,
                        &active_params,
                        lx / nx.max(1) as f64,
                        ly / ny.max(1) as f64,
                    );
                    setup_structured_bcs(
                        solver,
                        "allmach_thermal_structured",
                        active_params.inlet_velocity as f64,
                        unknowns,
                        &active_params,
                    );
                }
            }
            let outcome = structured_step(
                solver,
                &active_params,
                &mut prev_max_vel,
                &mut prev_explicit_rate,
                "allmach_thermal_structured",
                true,
            );
            if let Some(reason) = outcome.diverged {
                return Err(format!(
                    "structured {geometry:?} RK4 step {step}: {reason:?}"
                ));
            }
            smoke.steps += 1;
            smoke.min_dt = smoke.min_dt.min(outcome.dt as f64);
            smoke.max_dt = smoke.max_dt.max(outcome.dt as f64);
            if let Some(readback) = outcome.readback {
                smoke.max_vel = smoke.max_vel.max(readback.stats.max_vel);
                if let Some((lo, hi)) = readback.stats.rho {
                    smoke.min_rho = smoke.min_rho.min(lo);
                    smoke.max_rho = smoke.max_rho.max(hi);
                }
            }
        }
        Ok(smoke)
    }

    if gpu {
        let context = pollster::block_on(crate::solver::gpu::context::GpuContext::new(None, None))?;
        let mut solver = StructuredGpuSolver::with_config(
            context,
            StructuredGrid::new(nx, ny, lx, ly),
            &model,
            params.requested_dt.max(1.0e-12) as f64,
            1,
            params.advection_scheme,
            GpuTimeScheme::RK4,
        )?;
        solver.set_fluid(params.density as f64, params.viscosity as f64);
        solver.set_alpha_u(params.alpha_u);
        solver.set_alpha_p(params.alpha_p);
        drive(
            &mut solver,
            geometry,
            nx,
            ny,
            lx,
            ly,
            unknowns,
            steps,
            &params,
            live_update.as_ref(),
        )
    } else {
        let mut solver = StructuredModelSolver::with_config(
            StructuredGrid::new(nx, ny, lx, ly),
            &model,
            params.requested_dt.max(1.0e-12) as f64,
            1,
            params.advection_scheme,
            GpuTimeScheme::RK4,
        )?;
        solver.set_engine(crate::solver::cpu::CpuEngine::Transpiled, 1);
        solver.set_fluid(params.density as f64, params.viscosity as f64);
        solver.set_alpha_u(params.alpha_u);
        solver.set_alpha_p(params.alpha_p);
        drive(
            &mut solver,
            geometry,
            nx,
            ny,
            lx,
            ly,
            unknowns,
            steps,
            &params,
            live_update.as_ref(),
        )
    }
}

fn solver_worker_main(
    cmd_rx: mpsc::Receiver<SolverWorkerCommand>,
    evt_tx: mpsc::Sender<SolverWorkerEvent>,
) {
    let mut mode: Option<SolverMode> = None;
    let mut trace: Option<SolverTraceSession> = None;
    let mut model_id: &'static str = "<uninitialized>";
    let mut viz_field: Option<VizFieldBuffers> = None;
    let mut params = RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.001,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: GpuTimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 1,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 1.0,
        alpha_p: 1.0,
        inlet_velocity: 0.0,
        density: 1.0,
        viscosity: 0.0,
        eos: crate::solver::model::eos::EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    };

    let mut running = false;
    let mut run_generation = 0_u64;
    let mut step_idx: u64 = 0;
    // Adaptive-dt velocity scale for the structured path (mirrors
    // `SolverDriver::prev_max_vel`). Updated from structured field readbacks.
    let mut structured_prev_max_vel: f64 = 0.0;
    // Post-step all-Mach spectral rate used to size the NEXT adaptive RK4 step.
    // Caching it avoids a second full-state GPU read before every accepted step.
    let mut structured_prev_explicit_rate: Option<f64> = None;
    let mut last_stats_publish = std::time::Instant::now();
    let mut last_snapshot_publish = std::time::Instant::now();
    let mut completed_step_timing = CompletedStepTiming::new();
    let stats_publish_interval = std::time::Duration::from_millis(33);
    let snapshot_publish_interval = std::time::Duration::from_millis(100);
    let mut autonomous_scheduler = AutonomousBatchScheduler::default();
    let (autonomous_completion_tx, autonomous_completion_rx) =
        mpsc::channel::<AutonomousCallbackMessage>();
    let mut autonomous_fallback = false;
    // A failed tail-status decode means the GPU executed work whose accepted
    // time/history phase the host can no longer prove.  Such a backend must be
    // rebuilt; silently falling back to the ordinary router would risk
    // advancing from the wrong ping-pong buffer.
    let mut autonomous_backend_poisoned = false;
    // Live parameter changes are policy requests.  Applying them while an
    // autonomous window is queued can both block on whole-state readbacks and
    // let an older completion overwrite the new dt.  Drain first, then apply
    // the latest request exactly once.
    let mut pending_autonomous_param_apply = false;
    let mut autonomous_error: Option<String> = None;
    let mut autonomous_completed_time: Option<f64> = None;
    // Keep a replaced solver (and therefore its wgpu Device) alive until all
    // callbacks from its invalidated generation have released their credits.
    let mut retired_autonomous_modes: Vec<SolverMode> = Vec::new();
    // CFD2_PERF_LOG=1: stderr telemetry for the headed perf harness — per-batch
    // submit->completion latency, batch-level stats, and >2ms submit/poll
    // stalls (the vsync lock-convoy signature).
    let perf_log = std::env::var_os("CFD2_PERF_LOG").is_some();

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !solver_worker_handle_cmd(
                cmd,
                &mut mode,
                &mut trace,
                &mut model_id,
                &mut viz_field,
                &mut params,
                &mut running,
                &mut run_generation,
                &mut step_idx,
                &mut structured_prev_max_vel,
                &mut structured_prev_explicit_rate,
                &mut completed_step_timing,
                &mut last_stats_publish,
                &mut last_snapshot_publish,
                &mut autonomous_scheduler,
                &mut autonomous_fallback,
                &mut autonomous_backend_poisoned,
                &mut pending_autonomous_param_apply,
                &mut retired_autonomous_modes,
                &evt_tx,
            ) {
                return;
            }
        }

        // Autonomous marching is intentionally a presentation-mode fast path:
        // the GPU-direct renderer consumes frame-demanded device copies, while
        // EguiPlot needs host u/p snapshots and therefore stays on the ordinary
        // readback path. Fixed-dt structured RK4 has an accepted-state GPU
        // health/rollback controller. Other modes fall through until they
        // expose the same safety contract.
        let autonomous_presentation_ready = !autonomous_fallback
            && trace.is_none()
            && params.time_scheme == GpuTimeScheme::RK4
            && !params.log_convergence
            && viz_field
                .as_ref()
                .is_some_and(|viz| viz.size_bytes > 0 && viz.mailbox.gpu_consumer_enabled());
        let autonomous_policy = match mode.as_ref() {
            // Structured fixed-dt RK4 has finite-state rollback only when
            // its current EOS family is covered by that audit. The exact
            // all-Mach controller additionally supports adaptive CFL.
            Some(SolverMode::Structured(s)) => structured_autonomous_refill_policy(
                autonomous_presentation_ready,
                params.adaptive_dt,
                s.supports_autonomous_fixed(),
                s.supports_autonomous_adaptive(),
            ),
            // Unstructured eligibility is declared by the controller, not
            // inferred from a model id. Whole three-step history cycles
            // keep the host submission phase stable while the GPU-selected
            // frame copy follows any accepted prefix exactly.
            Some(SolverMode::Static(d))
                if autonomous_presentation_ready
                    && d.autonomous_explicit_capabilities().is_some_and(|capabilities| {
                        capabilities.accepted_state_copy
                            && if params.adaptive_dt {
                                capabilities.adaptive_health
                            } else {
                                capabilities.fixed_health
                            }
                    }) =>
            {
                Some(AutonomousRefillPolicy::ContinuousPhaseCycle3)
            }
            _ => None,
        };
        let autonomous_eligible = autonomous_policy.is_some();

        if autonomous_scheduler.is_active() {
            let perf_poll_start = std::time::Instant::now();
            let mut current_pollable = false;
            let current_poll = match mode.as_ref() {
                Some(SolverMode::Structured(s)) => {
                    current_pollable = true;
                    s.poll_gpu_completions()
                }
                // A generation-changing solver replacement normally uses the
                // same UI device. Polling its current plan also drives stale
                // callbacks on that shared device until their credits retire.
                Some(SolverMode::Static(d)) => {
                    current_pollable = true;
                    d.poll_gpu_batch_completions()
                }
                _ => Ok(()),
            };
            let perf_poll_ms = perf_poll_start.elapsed().as_secs_f64() * 1.0e3;
            if perf_log && perf_poll_ms > 2.0 {
                eprintln!("[stall] device.poll took {perf_poll_ms:.3} ms");
            }
            let current_poll_error = current_poll.err();
            let mut retired_poll_succeeded = false;
            let mut retired_poll_error = None;
            for retired in &retired_autonomous_modes {
                let result = match retired {
                    SolverMode::Structured(s) => s.poll_gpu_completions(),
                    SolverMode::Static(d) => d.poll_gpu_batch_completions(),
                    _ => Ok(()),
                };
                match result {
                    Ok(()) => retired_poll_succeeded = true,
                    Err(error) => retired_poll_error = Some(error),
                }
            }

            if let Some(error) = current_poll_error {
                autonomous_error.get_or_insert(error);
                autonomous_backend_poisoned = true;
                autonomous_fallback = true;
                running = false;
                // The current device may have executed work whose tail status
                // can no longer be observed.  Its state is therefore
                // non-resumable until Initialize/Reset rebuilds the backend.
                autonomous_scheduler.abort();
            } else if retired_poll_error.is_some() {
                // A retired device failure cannot poison its replacement.
                // Its stale callbacks are unrecoverable, so release only those
                // old-epoch credits and keep current work intact.
                autonomous_scheduler.discard_retired_epochs();
                retired_autonomous_modes.clear();
            } else if !current_pollable
                && !retired_poll_succeeded
                && autonomous_scheduler.is_active()
            {
                let has_current_work = autonomous_scheduler
                    .in_flight
                    .iter()
                    .any(|ticket| ticket.backend_epoch == autonomous_scheduler.backend_epoch);
                if has_current_work {
                    autonomous_error.get_or_insert_with(|| {
                        "autonomous GPU completion lost its polling device".to_string()
                    });
                    autonomous_backend_poisoned = true;
                    autonomous_fallback = true;
                    running = false;
                    autonomous_scheduler.abort();
                } else {
                    autonomous_scheduler.discard_retired_epochs();
                }
            }
        }

        let mut accepted_batch_boundary = false;
        while let Ok(message) = autonomous_completion_rx.try_recv() {
            let completion = match message.result {
                Ok(AutonomousBackendStatus::Structured(status)) => AutonomousBatchCompletion {
                    ticket: message.ticket,
                    accepted_steps: status.accepted_steps,
                    completed_time: f64::from(status.time),
                    last_dt: status.dt_old,
                    next_dt: status.dt,
                    halted: status.halted,
                    invalid_count: status.invalid_cells,
                    completed_at: message.completed_at,
                    backend_status: Some(AutonomousBackendStatus::Structured(status)),
                },
                Ok(AutonomousBackendStatus::Unstructured(status)) => AutonomousBatchCompletion {
                    ticket: message.ticket,
                    accepted_steps: status.accepted_batch,
                    completed_time: f64::from(status.time),
                    last_dt: status.last_dt,
                    next_dt: status.next_dt,
                    halted: status.halted,
                    invalid_count: status.invalid_count,
                    completed_at: message.completed_at,
                    backend_status: Some(AutonomousBackendStatus::Unstructured(status)),
                },
                Err(error) => {
                    // The GPU batch is fenced even when decoding its tiny
                    // status failed, so release its physical credit but never
                    // invent accepted progress.
                    if message.ticket.backend_epoch == autonomous_scheduler.backend_epoch {
                        // Make every queued tail callback stale before
                        // tombstoning the failed front.  A late error from a
                        // retired backend must not re-poison its replacement.
                        autonomous_scheduler.invalidate(false);
                        autonomous_error.get_or_insert(error);
                        autonomous_backend_poisoned = true;
                        autonomous_fallback = true;
                        running = false;
                    }
                    autonomous_scheduler.discard_failed_completion(message.ticket);
                    continue;
                }
            };

            let drain = autonomous_scheduler.complete(completion);
            for retired in drain.retired.iter() {
                if retired.ticket.backend_epoch != autonomous_scheduler.backend_epoch {
                    continue;
                }
                let reconciliation = match (retired.backend_status, mode.as_mut()) {
                    (
                        Some(AutonomousBackendStatus::Structured(status)),
                        Some(SolverMode::Structured(s)),
                    ) => {
                        s.reconcile_autonomous_status(status);
                        Ok(())
                    }
                    (
                        Some(AutonomousBackendStatus::Unstructured(status)),
                        Some(SolverMode::Static(d)),
                    ) => d.reconcile_autonomous_explicit_gpu_batch(status),
                    (None, _) => Ok(()),
                    _ => {
                        Err("autonomous completion reached a different solver backend".to_string())
                    }
                };
                if let Err(error) = reconciliation {
                    autonomous_error.get_or_insert(error);
                    running = false;
                    autonomous_scheduler.request_pause();
                } else if retired.ticket.generation != autonomous_scheduler.generation {
                    // Generation invalidation suppresses publication/tuning,
                    // not physical work already accepted by this same
                    // backend. Keep trace step numbering and the next timing
                    // delta aligned with the reconciled solver clock.
                    step_idx = step_idx.wrapping_add(u64::from(retired.accepted_steps));
                    autonomous_completed_time = Some(retired.completed_time);
                }
            }
            for completed in drain.completed.iter().copied() {
                // Unstructured ping-pong ownership is reconciled only after
                // the complete two-batch window retires. A copy submitted on
                // the first callback would execute behind batch 2 while bound
                // to batch 1's now-history phase. Structured storage has no
                // such phase alias and may present every retired completion.
                accepted_batch_boundary |= autonomous_scheduler.refill_policy.is_continuous()
                    || autonomous_scheduler.in_flight.is_empty();
                let simulated_seconds = autonomous_completed_time
                    .map(|previous| (completed.completed_time - previous).max(0.0))
                    .unwrap_or_else(|| {
                        f64::from(completed.accepted_steps) * f64::from(completed.last_dt)
                    });
                autonomous_completed_time = Some(completed.completed_time);
                let timing = completed_step_timing.record_batch_at(
                    u64::from(completed.accepted_steps),
                    simulated_seconds,
                    true,
                    completed.completed_at,
                );
                step_idx = step_idx.wrapping_add(u64::from(completed.accepted_steps));
                if perf_log {
                    eprintln!(
                        "[batch] steps={} submit->done={:.3} ms",
                        completed.accepted_steps,
                        completed
                            .completed_at
                            .duration_since(completed.ticket.submitted_at)
                            .as_secs_f64()
                            * 1.0e3,
                    );
                }

                if !completed.is_healthy() {
                    autonomous_error.get_or_insert_with(|| {
                        format!(
                            "GPU explicit health check halted after {} accepted batch steps ({} invalid cells)",
                            completed.accepted_steps, completed.invalid_count
                        )
                    });
                    running = false;
                }

                if last_stats_publish.elapsed() >= stats_publish_interval
                    || !completed.is_healthy()
                    || autonomous_scheduler.phase == AutonomousRunPhase::Paused
                {
                    last_stats_publish = std::time::Instant::now();
                    let _ = evt_tx.send(SolverWorkerEvent::Stats {
                        stats: CachedGpuStats {
                            sim_time: completed.completed_time,
                            // `next_dt` is the controller value which will be
                            // used by the next accepted step; fixed mode keeps
                            // it equal to `last_dt`.
                            dt: completed.next_dt,
                            step_time_ms: timing.step_time_ms,
                            steps_per_second: timing.steps_per_second,
                            sim_seconds_per_wall_second: timing.sim_seconds_per_wall_second,
                            ..Default::default()
                        },
                    });
                }
            }

            if drain.became_paused && !running {
                if let Some(error) = autonomous_error.take() {
                    let _ = evt_tx.send(SolverWorkerEvent::Error(error));
                }
                let _ = evt_tx.send(SolverWorkerEvent::Running {
                    running: false,
                    generation: run_generation,
                });
            }
        }

        if !autonomous_scheduler.is_active() && !running {
            if let Some(error) = autonomous_error.take() {
                let _ = evt_tx.send(SolverWorkerEvent::Error(error));
                let _ = evt_tx.send(SolverWorkerEvent::Running {
                    running: false,
                    generation: run_generation,
                });
            }
        }
        if !autonomous_scheduler.is_active() {
            retired_autonomous_modes.clear();
        }

        // Parameter mutation is intentionally downstream of physical status
        // reconciliation.  Therefore no stale completion can restore an old
        // dt after this point, and any packed-state readback performed by
        // apply_params_any sees a fully drained queue.
        if !autonomous_scheduler.is_active()
            && pending_autonomous_param_apply
            && !autonomous_backend_poisoned
        {
            pending_autonomous_param_apply = false;
            solver_worker_apply_live_params(
                &mut mode,
                &mut params,
                &mut structured_prev_explicit_rate,
            );
            solver_worker_trace_params(&mut trace, mode.as_ref(), step_idx, params);
        }

        // A visualization copy is requested by the render callback and is
        // serviced only after a completion-fenced accepted batch boundary.
        // `finish_write` publishes queue order, not CPU completion: rendering
        // is submitted after this copy on the same wgpu queue.
        if accepted_batch_boundary {
            if let (Some(mode), Some(viz)) = (
                mode.as_ref(),
                viz_field.as_ref().filter(|viz| viz.size_bytes > 0),
            ) {
                if let Some(write) = viz.mailbox.try_begin_write() {
                    if viz
                        .capture_snapshot(mode, write, VizSnapshotSource::AutonomousAccepted)
                        .is_err()
                    {
                        // A missing/invalid accepted-state copy is a clean
                        // autonomous capability failure. The cancelled mailbox
                        // request remains outstanding for the ordinary path.
                        autonomous_fallback = true;
                        autonomous_scheduler.request_pause();
                    }
                }
            }
        }

        if !autonomous_eligible && autonomous_scheduler.phase == AutonomousRunPhase::Running {
            autonomous_scheduler.request_pause();
        }

        if running && autonomous_eligible && !autonomous_scheduler.is_active() {
            autonomous_completed_time = mode.as_ref().map(SolverMode::sim_time_f64);
            autonomous_scheduler.begin_run(autonomous_policy.expect("checked above"));
        }

        if running && autonomous_eligible {
            while let Some(ticket) = autonomous_scheduler.reserve(std::time::Instant::now()) {
                let tx = autonomous_completion_tx.clone();
                let perf_submit_start = std::time::Instant::now();
                let submitted = match mode.as_mut() {
                    Some(SolverMode::Structured(s)) => s.submit_autonomous_batch(
                        ticket.requested_steps as usize,
                        params.adaptive_dt.then_some(params.target_cfl as f32),
                        move |result| {
                            let _ = tx.send(AutonomousCallbackMessage {
                                ticket,
                                result: result.map(AutonomousBackendStatus::Structured),
                                completed_at: std::time::Instant::now(),
                            });
                        },
                    ),
                    Some(SolverMode::Static(d)) => d
                        .submit_autonomous_explicit_gpu_batch(
                            ticket.requested_steps,
                            move |result| {
                                let _ = tx.send(AutonomousCallbackMessage {
                                    ticket,
                                    result: result.map(AutonomousBackendStatus::Unstructured),
                                    completed_at: std::time::Instant::now(),
                                });
                            },
                        )
                        .map(|submission| Some(submission.submission_index)),
                    _ => Err("autonomous structured solver disappeared".to_string()),
                };
                let perf_submit_ms = perf_submit_start.elapsed().as_secs_f64() * 1.0e3;
                if perf_log && perf_submit_ms > 2.0 {
                    eprintln!("[stall] submit took {perf_submit_ms:.3} ms");
                }
                match submitted {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        autonomous_scheduler.cancel_reservation(ticket);
                        autonomous_fallback = true;
                        break;
                    }
                    Err(_) => {
                        // Eligibility can change underneath a queued GUI
                        // command. A pre-submit rejection is a clean fallback,
                        // not a solver failure.
                        autonomous_scheduler.cancel_reservation(ticket);
                        autonomous_fallback = true;
                        autonomous_scheduler.request_pause();
                        break;
                    }
                }
            }
        }

        if autonomous_scheduler.is_active() {
            // Never block here: Poll callbacks return completion credits. A
            // short yield keeps the command/UI threads schedulable without
            // placing a latency floor on sub-millisecond GPU batches.
            thread::yield_now();
            continue;
        }

        if !running {
            // Plot -> Direct while paused must not keep displaying slot zero's
            // initial rest state forever. Service at most the one coalesced
            // outstanding request; no request means no paused-frame copy.
            if !autonomous_backend_poisoned {
                if let (Some(mode), Some(viz)) = (
                    mode.as_ref(),
                    viz_field.as_ref().filter(|viz| {
                        viz.size_bytes > 0 && viz.mailbox.gpu_consumer_enabled()
                    }),
                ) {
                    let result =
                        service_pending_viz_snapshot(mode, viz, VizSnapshotSource::Current);
                    debug_assert!(result.is_ok(), "paused visualization snapshot failed");
                }
            }
            match cmd_rx.recv_timeout(std::time::Duration::from_millis(16)) {
                Ok(cmd) => {
                    if !solver_worker_handle_cmd(
                        cmd,
                        &mut mode,
                        &mut trace,
                        &mut model_id,
                        &mut viz_field,
                        &mut params,
                        &mut running,
                        &mut run_generation,
                        &mut step_idx,
                        &mut structured_prev_max_vel,
                        &mut structured_prev_explicit_rate,
                        &mut completed_step_timing,
                        &mut last_stats_publish,
                        &mut last_snapshot_publish,
                        &mut autonomous_scheduler,
                        &mut autonomous_fallback,
                        &mut autonomous_backend_poisoned,
                        &mut pending_autonomous_param_apply,
                        &mut retired_autonomous_modes,
                        &evt_tx,
                    ) {
                        return;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }

        let Some(mode) = mode.as_mut() else {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(
                "solver worker entered running state without an initialized solver".to_string(),
            ));
            let _ = evt_tx.send(SolverWorkerEvent::Running {
                running: false,
                generation: run_generation,
            });
            continue;
        };

        // Logging / readback cadence (depends on the current step index + timers).
        let log_every_steps = params.log_every_steps.max(1) as u64;
        let should_log = params.log_convergence && (step_idx % log_every_steps == 0);
        // A flow-adaptive sizing event changes the CELL COUNT this step: force
        // a full readback so the published u/p snapshot and the published
        // polygon set change length together (a stale-length snapshot against
        // the resized mesh would mis-index the per-cell colors).
        let adapt_step = matches!(
            &*mode,
            SolverMode::MovingMesh(m) if m.adapt_fires_this_step()
        );
        let should_readback = step_idx == 0
            || last_snapshot_publish.elapsed() >= snapshot_publish_interval
            || should_log
            || adapt_step;

        // Adaptive timestep + step + divergence / steady-state detection live in
        // the shared driver; GUI-only concerns (viz upload, publishing, trace)
        // stay here.
        //
        // Static vs moving is the ONLY step-dispatch fork: the static arm runs
        // `SolverDriver::step`; the moving arm runs the ALE cycle (advect → regen
        // → swept fluxes → refresh → step), yielding the re-generated polygons +
        // per-step telemetry to publish.
        let (outcome, moving_refresh) = match mode {
            SolverMode::Static(d) => (d.step(should_readback), None),
            SolverMode::Structured(s) => (
                structured_step(
                    s,
                    &params,
                    &mut structured_prev_max_vel,
                    &mut structured_prev_explicit_rate,
                    model_id,
                    should_readback,
                ),
                None,
            ),
            SolverMode::StructuredCpu(s) => (
                structured_step(
                    &mut s.solver,
                    &params,
                    &mut structured_prev_max_vel,
                    &mut structured_prev_explicit_rate,
                    model_id,
                    should_readback,
                ),
                None,
            ),
            SolverMode::MovingMesh(m) => match m.step(should_readback) {
                Ok((o, mstats)) => {
                    // Clone + publish the regenerated mesh only at readback
                    // cadence. The UI coalesces `MeshRefreshed` to one upload
                    // per frame anyway, so cloning the full f64 polygon set on
                    // every solver step (many per frame on a fast CPU solve)
                    // is wasted worker CPU + channel churn — and an unbounded
                    // queue balloon if the UI stalls while the worker steps.
                    // Geometry + ALE diagnostics tolerate frame-cadence lag;
                    // colors track per-step via the separate viz-field path
                    // regardless.
                    //
                    // EXCEPTION — recycle steps publish immediately: a
                    // recycled cell's slot keeps its index but its polygon
                    // teleports outlet → inlet, and the per-step viz colors
                    // would otherwise paint the STALE outlet polygon with the
                    // slot's fresh inlet-donor state until the next snapshot
                    // (a visible wrong-velocity/pressure flash at the
                    // outlet). Recycle steps are rare (≤ RECYCLE_MAX_PER_STEP
                    // seeds, ~0.2/step measured), so the extra publish is
                    // negligible.
                    // Resize (adapt) steps likewise publish immediately: the
                    // polygon COUNT changed, so every consumer of the cached
                    // cells must see the new set alongside the (forced, see
                    // `adapt_step`) fresh state snapshot.
                    let refresh = if should_readback
                        || mstats.recycled > 0
                        || mstats.cells_born > 0
                        || mstats.cells_killed > 0
                    {
                        Some((CFDApp::cache_cells(m.mesh()), mstats))
                    } else {
                        None
                    };
                    (o, refresh)
                }
                Err(err) => {
                    running = false;
                    let _ = evt_tx.send(SolverWorkerEvent::Error(format!(
                        "moving-mesh step failed: {err}"
                    )));
                    let _ = evt_tx.send(SolverWorkerEvent::Running {
                        running: false,
                        generation: run_generation,
                    });
                    continue;
                }
            },
        };
        if let Some(DivergeReason::StepError(err)) = &outcome.diverged {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(format!(
                "solver step failed: {err}"
            )));
            let _ = evt_tx.send(SolverWorkerEvent::Running {
                running: false,
                generation: run_generation,
            });
            continue;
        }
        if let Some(DivergeReason::NonFinite { u, p }) = &outcome.diverged {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(format!(
                "divergence detected (nonfinite u={u}, p={p})"
            )));
            let _ = evt_tx.send(SolverWorkerEvent::Running {
                running: false,
                generation: run_generation,
            });
            continue;
        }
        // Publish the re-generated mesh (moving only). Emitted every moving step;
        // the UI thread coalesces and re-tessellates at the frame boundary.
        if let Some((cached_cells, mstats)) = moving_refresh {
            let _ = evt_tx.send(SolverWorkerEvent::MeshRefreshed {
                cached_cells,
                stats: mstats,
            });
        }
        // `outcome.readback` is an existing GPU completion fence. Use it to
        // close a multi-step wall-time window; on intervening async iterations
        // retain the last completed sample instead of publishing enqueue time.
        let trace_step_time_ms = outcome.step_time_ms;
        let completed_sample =
            completed_step_timing.record_step(outcome.dt, outcome.readback.is_some());
        let step_time_ms = completed_sample.step_time_ms;

        let linear_solves = outcome.linear_stats.len() as u32;
        let linear_last = outcome.linear_stats.last().copied().unwrap_or_default();
        if matches!(outcome.diverged, Some(DivergeReason::LinearSolver)) {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(
                "divergence detected (linear solver)".to_string(),
            ));
            let _ = evt_tx.send(SolverWorkerEvent::Running {
                running: false,
                generation: run_generation,
            });
            continue;
        }

        // Service at most one coalesced screen-frame request at this accepted
        // step boundary. Fast solvers no longer copy their complete state on
        // every step; a slow step still publishes its accepted state as soon
        // as it reaches the boundary. `finish_write` runs after the wgpu copy
        // enqueue, establishing copy-before-render queue order.
        if let Some(viz_field) = viz_field.as_ref().filter(|viz| viz.size_bytes > 0) {
            let result =
                service_pending_viz_snapshot(mode, viz_field, VizSnapshotSource::Current);
            debug_assert!(
                result.is_ok(),
                "visualization producer lost exclusive WRITING ownership"
            );
        }

        let mut stats = CachedGpuStats {
            sim_time: mode.sim_time_f64(),
            dt: mode.sim_dt(),
            step_time_ms,
            steps_per_second: completed_sample.steps_per_second,
            sim_seconds_per_wall_second: completed_sample.sim_seconds_per_wall_second,
            linear_solves,
            linear_last,
            ..Default::default()
        };

        // Outer-iteration / positivity telemetry on the UnifiedSolver path comes
        // off `step_stats()`; the structured banded solver has no UnifiedSolver, so
        // it rides the `StepOutcome` fields `structured_step` populated instead.
        if let Some(solver) = mode.unified_solver() {
            let step_stats = solver.step_stats();
            if let Some(iters) = step_stats.outer_iterations {
                stats.outer_iterations = iters;
            }
            if let Some(res_u) = step_stats.outer_residual_u {
                stats.outer_residual_u = res_u;
            }
            if let Some(res_p) = step_stats.outer_residual_p {
                stats.outer_residual_p = res_p;
            }
            stats.outer_step_status = step_stats.outer_step_status;
            stats.positivity_min_rho = step_stats.positivity_min_rho;
            stats.positivity_min_p = step_stats.positivity_min_p;
            stats.positivity_rho_undershoots =
                step_stats.positivity_rho_undershoot_count.unwrap_or(0);
            stats.positivity_pressure_undershoots =
                step_stats.positivity_pressure_undershoot_count.unwrap_or(0);
        } else {
            if let Some(iters) = outcome.outer_iters {
                stats.outer_iterations = iters;
            }
            if let Some(res_u) = outcome.outer_residual_u {
                stats.outer_residual_u = res_u;
            }
            if let Some(res_p) = outcome.outer_residual_p {
                stats.outer_residual_p = res_p;
            }
        }

        let mut trace_max_u: Option<f64> = None;
        let mut trace_p_min: Option<f64> = None;
        let mut trace_p_max: Option<f64> = None;

        if let Some(rb) = outcome.readback {
            let now = std::time::Instant::now();
            last_snapshot_publish = now;
            last_stats_publish = now;
            let fs = rb.stats;
            trace_max_u = Some(fs.max_vel);
            trace_p_min = Some(fs.p_min);
            trace_p_max = Some(fs.p_max);

            if should_log || fs.nonfinite_u > 0 || fs.nonfinite_p > 0 {
                let outer_str = mode
                    .unified_solver()
                    .map(|solver| {
                        let step_stats = solver.step_stats();
                        step_stats
                            .outer_iterations
                            .map(|iters| {
                                let status_suffix = step_stats
                                    .outer_step_status
                                    .map(|status| format!(", status={}", status.as_str()))
                                    .unwrap_or_default();
                                let abs_residuals = solver.outer_field_residuals();
                                let scaled_residuals = solver.outer_field_residuals_scaled();

                                if let (Some(abs), Some(scaled)) = (abs_residuals, scaled_residuals) {
                                    let fields = abs
                                        .iter()
                                        .zip(scaled.iter())
                                        .map(|((name, abs_res), (_, scaled_res))| {
                                            format!("{name}={scaled_res:.3e} (abs={abs_res:.3e})")
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    format!(" outer(iters={iters}, res=[{fields}]{status_suffix})")
                                } else if let Some(fields) = abs_residuals {
                                    let fields = fields
                                        .iter()
                                        .map(|(name, res)| format!("{name}={res:.3e}"))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    format!(" outer(iters={iters}, res=[{fields}]{status_suffix})")
                                } else if let (Some(res_u), Some(res_p)) =
                                    (step_stats.outer_residual_u, step_stats.outer_residual_p)
                                {
                                    format!(" outer(iters={iters}, u={res_u:.3e}, p={res_p:.3e}{status_suffix})")
                                } else {
                                    format!(" outer(iters={iters}{status_suffix})")
                                }
                            })
                            .unwrap_or_default()
                    })
                    .unwrap_or_default();

                eprintln!(
                    "[cfd2][{}] step={} t={:.4e} dt={:.2e} max|u|={:.3e} p=[{:.3e},{:.3e}] solves={} last(iters={}, res={:.3e}, conv={}, div={}){} nonfinite(u={}, p={})",
                    model_id,
                    step_idx,
                    mode.sim_time(),
                    mode.sim_dt(),
                    fs.max_vel,
                    fs.p_min,
                    fs.p_max,
                    linear_solves,
                    linear_last.iterations,
                    linear_last.residual,
                    linear_last.converged,
                    linear_last.diverged,
                    outer_str,
                    fs.nonfinite_u,
                    fs.nonfinite_p,
                );
            }

            if fs.nonfinite_u > 0 || fs.nonfinite_p > 0 {
                running = false;
                let _ = evt_tx.send(SolverWorkerEvent::Error(format!(
                    "divergence detected (nonfinite u={}, p={})",
                    fs.nonfinite_u, fs.nonfinite_p
                )));
                let _ = evt_tx.send(SolverWorkerEvent::Running {
                    running: false,
                    generation: run_generation,
                });
                continue;
            }

            let _ = evt_tx.send(SolverWorkerEvent::Snapshot {
                u: rb.u,
                p: rb.p,
                stats,
            });
        } else if outcome.should_stop
            || step_idx == 0
            || last_stats_publish.elapsed() >= stats_publish_interval
        {
            last_stats_publish = std::time::Instant::now();
            let _ = evt_tx.send(SolverWorkerEvent::Stats { stats });
        }

        // Per-step trace is a UnifiedSolver capability (graph timings); the
        // structured banded path has none, so it is skipped entirely.
        if let (Some(trace), Some(solver)) = (trace.as_mut(), mode.unified_solver()) {
            let linear_solves = outcome
                .linear_stats
                .iter()
                .copied()
                .map(tracefmt::TraceLinearSolverStats::from)
                .collect::<Vec<_>>();

            let graph = solver
                .step_graph_timings()
                .iter()
                .map(|t| {
                    let nodes = t.detail.as_ref().and_then(|detail| match detail {
                        crate::solver::gpu::execution_plan::GraphDetail::Module(detail) => Some(
                            detail
                                .nodes
                                .iter()
                                .map(|n| tracefmt::TraceGraphNodeTiming {
                                    label: n.label.to_string(),
                                    seconds: n.seconds,
                                })
                                .collect::<Vec<_>>(),
                        ),
                    });
                    tracefmt::TraceGraphTiming {
                        label: t.label.to_string(),
                        seconds: t.seconds,
                        nodes,
                    }
                })
                .collect::<Vec<_>>();

            let event = tracefmt::TraceEvent::Step(tracefmt::TraceStepEvent {
                step: step_idx,
                sim_time: solver.time(),
                dt: solver.dt(),
                // Traces retain their step-local backend timing. The UI-only
                // completion window must not be repeated across per-step trace
                // events, whose consumers sum and correlate this field.
                wall_time_ms: trace_step_time_ms,
                linear_solves,
                graph,
                max_u: trace_max_u,
                p_min: trace_p_min,
                p_max: trace_p_max,
            });
            let _ = trace.writer.write_event(&event);
        }

        step_idx = step_idx.wrapping_add(1);
        if outcome.should_stop {
            // The converged step is accepted: publish its final t/dt/stats and
            // trace before telling the UI that automatic marching stopped.
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Running {
                running: false,
                generation: run_generation,
            });
            continue;
        }
        // Keep command/UI threads schedulable without imposing a fixed 1 ms
        // latency floor on every accepted step. A sleep materially throttles
        // small explicit GPU cases whose completed solve is already sub-ms.
        thread::yield_now();
    }
}

/// Apply the worker's latest runtime policy only when no autonomous command
/// buffer remains in flight.  Some unstructured setters need one packed-state
/// readback, so this boundary is also what keeps command handling free of an
/// accidental GPU `Wait` behind queued batches.
fn solver_worker_apply_live_params(
    mode: &mut Option<SolverMode>,
    params: &mut RuntimeParams,
    structured_prev_explicit_rate: &mut Option<f64>,
) {
    *structured_prev_explicit_rate = None;
    let Some(mode) = mode.as_mut() else {
        return;
    };
    // The moving path never runs solver-side adaptive dt: its driver owns the
    // GCL-compatible controller and the configured timestep.
    if let SolverMode::MovingMesh(moving) = mode {
        let adaptive_dt = params.adaptive_dt;
        params.adaptive_dt = false;
        moving.set_adaptive_dt(adaptive_dt.then_some(params.target_cfl));
        moving.set_configured_dt(params.requested_dt as f64);
    }
    mode.apply_params_any(params);
}

fn solver_worker_trace_params(
    trace: &mut Option<SolverTraceSession>,
    mode: Option<&SolverMode>,
    step_idx: u64,
    params: RuntimeParams,
) {
    if let (Some(trace), Some(mode)) = (trace.as_mut(), mode) {
        let event = tracefmt::TraceEvent::Params(tracefmt::TraceParamsEvent {
            step: step_idx,
            sim_time: mode.sim_time(),
            params: trace_runtime_params_from_worker(params),
        });
        let _ = trace.writer.write_event(&event);
    }
}

fn solver_worker_handle_cmd(
    cmd: SolverWorkerCommand,
    mode: &mut Option<SolverMode>,
    trace: &mut Option<SolverTraceSession>,
    model_id: &mut &'static str,
    viz_field: &mut Option<VizFieldBuffers>,
    params: &mut RuntimeParams,
    running: &mut bool,
    run_generation: &mut u64,
    step_idx: &mut u64,
    structured_prev_max_vel: &mut f64,
    structured_prev_explicit_rate: &mut Option<f64>,
    completed_step_timing: &mut CompletedStepTiming,
    last_stats_publish: &mut std::time::Instant,
    last_snapshot_publish: &mut std::time::Instant,
    autonomous_scheduler: &mut AutonomousBatchScheduler,
    autonomous_fallback: &mut bool,
    autonomous_backend_poisoned: &mut bool,
    pending_autonomous_param_apply: &mut bool,
    retired_autonomous_modes: &mut Vec<SolverMode>,
    evt_tx: &mpsc::Sender<SolverWorkerEvent>,
) -> bool {
    match cmd {
        SolverWorkerCommand::SetSolver {
            mode: next,
            viz_field: next_viz_field,
        } => {
            autonomous_scheduler.replace_backend();
            *autonomous_fallback = false;
            *autonomous_backend_poisoned = false;
            *pending_autonomous_param_apply = false;
            if trace.is_some() {
                solver_worker_stop_trace(trace, mode);
            }
            if autonomous_scheduler.is_active() {
                if let Some(previous) = mode.take() {
                    retired_autonomous_modes.push(previous);
                }
            }
            *model_id = next.model_id_str();
            *mode = Some(next);
            *viz_field = next_viz_field;
            *running = false;
            *step_idx = 0;
            *structured_prev_max_vel = 0.0;
            *structured_prev_explicit_rate = None;
            completed_step_timing.reset();
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;

            // Phase-2 parameter application: build sets phase 1, this sets phase 2.
            if let Some(m) = mode.as_mut() {
                // Strip the solver-side adaptive_dt for a moving driver (its
                // step() rejects it — GCL handshake). Unlike the UpdateParams
                // arm, do NOT touch the driver-side controller or the
                // configured dt here: the builder (build_moving_init / a
                // headless harness) configured them from ITS OWN request, and
                // the worker's params snapshot may be stale at SetSolver time
                // (the smoke path would silently disable a pre-configured
                // controller). A user-driven UpdateParams re-routes both.
                if let SolverMode::MovingMesh(_) = m {
                    params.adaptive_dt = false;
                }
                m.apply_params_any(params);
            }
            // This is a readiness acknowledgment, so publish it only after the
            // solver has accepted its phase-2 runtime configuration.
            let _ = evt_tx.send(SolverWorkerEvent::Running {
                running: false,
                generation: *run_generation,
            });
        }
        SolverWorkerCommand::ClearSolver => {
            autonomous_scheduler.replace_backend();
            *autonomous_fallback = false;
            *autonomous_backend_poisoned = false;
            *pending_autonomous_param_apply = false;
            solver_worker_stop_trace(trace, mode);
            if autonomous_scheduler.is_active() {
                if let Some(previous) = mode.take() {
                    retired_autonomous_modes.push(previous);
                }
            }
            *mode = None;
            *model_id = "<uninitialized>";
            *running = false;
            *step_idx = 0;
            *structured_prev_max_vel = 0.0;
            *structured_prev_explicit_rate = None;
            completed_step_timing.reset();
            *viz_field = None;
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;
            let _ = evt_tx.send(SolverWorkerEvent::Running {
                running: false,
                generation: *run_generation,
            });
        }
        SolverWorkerCommand::SetRunning {
            running: next_running,
            generation,
        } => {
            *run_generation = generation;
            if next_running && *autonomous_backend_poisoned {
                let _ = evt_tx.send(SolverWorkerEvent::Error(
                    "the previous autonomous GPU status could not be decoded; initialize/reset the solver before resuming"
                        .to_string(),
                ));
                let _ = evt_tx.send(SolverWorkerEvent::Running {
                    running: false,
                    generation,
                });
                *running = false;
                return true;
            }
            if next_running && mode.is_none() {
                let _ = evt_tx.send(SolverWorkerEvent::Error(
                    "cannot start solver: no solver is initialized".to_string(),
                ));
                let _ = evt_tx.send(SolverWorkerEvent::Running {
                    running: false,
                    generation,
                });
                *running = false;
                return true;
            }
            if next_running {
                autonomous_scheduler.invalidate(false);
                *autonomous_fallback = false;
                *step_idx = 0;
                completed_step_timing.reset();
                let now = std::time::Instant::now();
                *last_stats_publish = now;
                *last_snapshot_publish = now;
            }
            *running = next_running;
            if next_running {
                if let Some(viz) = viz_field.as_ref().filter(|viz| {
                    viz.size_bytes > 0 && viz.mailbox.gpu_consumer_enabled()
                }) {
                    // Ordered after Run acceptance: unlike the cold activation
                    // request, this demand cannot be consumed by the paused
                    // service path before the first accepted solver boundary.
                    viz.mailbox.request_frame();
                }
            }
            if next_running || autonomous_scheduler.request_pause() {
                let _ = evt_tx.send(SolverWorkerEvent::Running {
                    running: *running,
                    generation,
                });
            }
        }
        SolverWorkerCommand::UpdateParams(next_params) => {
            if *autonomous_backend_poisoned {
                *params = next_params;
                let _ = evt_tx.send(SolverWorkerEvent::Message(
                    "parameter update retained, but the solver must be initialized/reset after an ambiguous GPU completion"
                        .to_string(),
                ));
                return true;
            }
            let defer_apply = autonomous_scheduler.is_active();
            autonomous_scheduler.invalidate(false);
            *autonomous_fallback = false;
            *params = next_params;
            // The all-Mach stability rate is parameter-dependent.  Defer the
            // mutation itself until every older completion has reconciled, so
            // its dt cannot overwrite the requested policy.
            *structured_prev_explicit_rate = None;
            if defer_apply {
                *pending_autonomous_param_apply = true;
            } else {
                *pending_autonomous_param_apply = false;
                solver_worker_apply_live_params(mode, params, structured_prev_explicit_rate);
                solver_worker_trace_params(trace, mode.as_ref(), *step_idx, *params);
            }
        }
        SolverWorkerCommand::StartTrace { path, header } => {
            autonomous_scheduler.invalidate(false);
            *autonomous_fallback = true;
            solver_worker_stop_trace(trace, mode);

            match tracefmt::TraceWriter::create(&path) {
                Ok(mut writer) => {
                    let profiling_enabled = header.ui.profiling_enabled;
                    let event = tracefmt::TraceEvent::Header(Box::new(header));
                    let _ = writer.write_event(&event);

                    if let Some(s) = mode.as_mut().and_then(|m| m.unified_solver_mut()) {
                        s.set_collect_trace(true);
                        let _ = s.enable_detailed_profiling(profiling_enabled);
                        if profiling_enabled {
                            let _ = s.start_profiling_session();
                        }
                    }

                    *trace = Some(SolverTraceSession {
                        writer,
                        profiling_enabled,
                    });
                    eprintln!("[cfd2][trace] recording to {}", path);
                }
                Err(err) => {
                    let _ = evt_tx.send(SolverWorkerEvent::Message(format!(
                        "failed to start trace: {err}"
                    )));
                }
            }
        }
        SolverWorkerCommand::AppendTraceInit { events } => {
            if let Some(trace) = trace.as_mut() {
                for init in events {
                    let _ = trace.writer.write_event(&tracefmt::TraceEvent::Init(init));
                }
            }
        }
        SolverWorkerCommand::StopTrace => {
            solver_worker_stop_trace(trace, mode);
            *autonomous_fallback = false;
        }
        SolverWorkerCommand::Shutdown => return false,
    }
    true
}

fn get_color(t: f64) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        (0.0, t * 2.0, 1.0 - t * 2.0)
    } else {
        ((t - 0.5) * 2.0, 1.0 - (t - 0.5) * 2.0, 0.0)
    };

    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
