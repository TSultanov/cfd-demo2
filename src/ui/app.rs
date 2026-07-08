use crate::solver::mesh::{
    generate_cut_cell_mesh, generate_cvt_mesh, generate_delaunay_mesh,
    generate_structured_nozzle_mesh, generate_structured_symmetric_nozzle_mesh,
    generate_voronoi_mesh, BackwardsStep, BoundarySides,
    BoundaryType, ChannelWithObstacle, LloydConfig, Mesh, Nozzle,
};
use crate::solver::model::{
    all_models, compressible_model_with_eos, ModelPreconditionerSpec, ModelSpec,
};
use crate::solver::scheme::Scheme;
use crate::solver::{
    GpuLowMachPrecondModel, LinearSolverStats, OuterStepStatus, PreconditionerType,
    TimeScheme as GpuTimeScheme, UiPortSet,
};
use crate::trace as tracefmt;
use crate::ui::{cfd_renderer, fluid::Fluid};
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Polygon};
use nalgebra::{Point2, Vector2};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use crate::meshgen::meshless::{generate_cvt_mesh_with_seeds, CvtMeshSeeds};
use crate::sim::{
    BoundaryMotionSpec, DivergeReason, DriverBuild, MeshMotionSpec, MovingMeshDriver,
    MovingMeshStats, OscAxis, RegenBackend, RuntimeParams, SolverDriver,
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

#[derive(PartialEq, Clone, Copy)]
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
}

struct SolverInitRequest {
    generation: u64,
    model_id: &'static str,
    selected_geometry: GeometryType,
    mesh_type: MeshType,
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
    front_idx: Arc<AtomicUsize>,
    ready_idx: Arc<AtomicUsize>,
}

struct SolverInitResponse {
    generation: u64,
    result: Result<SolverInitOutcome, String>,
}

// Cached GPU solver stats for UI display (avoids lock contention)
#[derive(Default, Clone)]
struct CachedGpuStats {
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
    step_time_ms: f32,
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
}

impl SolverMode {
    /// The wrapped solver driver (shared telemetry/params/trace access).
    fn driver(&self) -> &SolverDriver {
        match self {
            SolverMode::Static(d) => d,
            SolverMode::MovingMesh(m) => m.driver(),
        }
    }

    /// Mutable access to the wrapped solver driver.
    fn driver_mut(&mut self) -> &mut SolverDriver {
        match self {
            SolverMode::Static(d) => d,
            SolverMode::MovingMesh(m) => m.driver_mut(),
        }
    }
}

enum SolverWorkerCommand {
    SetSolver {
        mode: SolverMode,
        viz_field: Option<VizFieldBuffers>,
    },
    ClearSolver,
    SetRunning(bool),
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
    Running(bool),
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
/// the moving-mesh message path — `SetSolver { MovingMesh }`, `SetRunning(true)`,
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
    handle.send(SolverWorkerCommand::SetRunning(true));

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
    plot_field: PlotField,
    is_running: bool,
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
    viz_field_front: usize,
    /// A moving-mesh refresh arrived and `cached_cells` changed since the GPU
    /// renderer's vertex buffers were last re-tessellated. Multiple solver
    /// steps may land between frames; only the latest matters, so this is a
    /// coalescing flag consumed once per frame at the render-frame boundary.
    pending_mesh_upload: bool,
}

struct CfdRenderCallback {
    renderer: Arc<Mutex<cfd_renderer::CfdRenderResources>>,
    uniforms: cfd_renderer::CfdUniforms,
    draw_lines: bool,
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
            plot_field: PlotField::VelocityMag,
            is_running: false,
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
            viz_field_front: 0,
            pending_mesh_upload: false,
        };
        app.apply_model_defaults();
        app.refresh_model_caps();
        app.sync_worker_params();
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
        let is_allmach = self.model_id == "allmach_pressure" || self.model_id == "allmach_thermal";
        let d = if self.selected_geometry == GeometryType::Nozzle && is_allmach {
            crate::ui::model_defaults::ALLMACH_THERMAL_NOZZLE
        } else {
            crate::ui::model_defaults::gui_defaults_for(self.model_id)
        };
        self.selected_scheme = d.advection_scheme;
        self.time_scheme = d.time_scheme;
        self.selected_preconditioner = d.preconditioner;
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

    /// The physical models the GUI Model dropdown offers. `all_models()` also
    /// contains MMS / biharmonic / demo *verification* variants which carry the
    /// velocity+pressure ports (so `UiPortSet::is_complete()` alone would expose
    /// them) but only ever reproduce a manufactured solution — never a physical
    /// flow a user would want to run. The dropdown is therefore restricted to the
    /// genuine flow models; the completeness check is kept as a secondary guard.
    fn supported_ui_models() -> Vec<(&'static str, &'static str)> {
        // Only physical-flow models belong in the GUI (these are exactly the ones
        // `model_label` names). Verification variants (`*_mms*`, `*biharmonic*`,
        // `*demo*`) are excluded.
        const GUI_PHYSICAL_MODELS: &[&str] = &[
            "incompressible_momentum",
            "compressible",
            "allmach_pressure",
            "allmach_thermal",
        ];
        let mut out = Vec::new();
        for model in all_models().expect("failed to build model definitions") {
            if !GUI_PHYSICAL_MODELS.contains(&model.id) {
                continue;
            }
            // Use UiPortSet to check for required fields (validates types too)
            let ui_ports = UiPortSet::from_layout(&model.state_layout);
            if !ui_ports.is_complete() {
                continue;
            }
            out.push((model.id, CFDApp::model_label(model.id)));
        }
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn build_selected_model(&self) -> Result<ModelSpec, String> {
        if self.model_id == "compressible" {
            return compressible_model_with_eos(self.current_fluid.eos);
        }
        all_models()?
            .into_iter()
            .find(|m| m.id == self.model_id)
            .ok_or_else(|| format!("unknown model id '{}'", self.model_id))
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
        let generation = self.next_init_generation;
        self.next_init_generation = self.next_init_generation.wrapping_add(1);
        SolverInitRequest {
            generation,
            model_id: self.model_id,
            selected_geometry: self.selected_geometry,
            mesh_type: self.mesh_type,
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
                if self.backend == BackendChoice::CpuTranspiledSimd { "1" } else { "0" },
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
        self.solver_worker
            .send(SolverWorkerCommand::SetRunning(false));
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
                let length = 3.5;
                let height_outlet = 1.0;
                let height_inlet = 0.5;
                let step_x = 0.5;

                let domain_size = Vector2::new(length, height_outlet);
                let geo = BackwardsStep {
                    length,
                    height_inlet,
                    height_outlet,
                    step_x,
                };

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
                let length = 3.0;
                let domain_size = Vector2::new(length, 1.0);
                let geo = ChannelWithObstacle {
                    length,
                    height: 1.0,
                    obstacle_center: Point2::new(1.0, 0.51), // Offset to trigger vortex shedding
                    obstacle_radius: 0.1,
                };

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
                let length = 3.0;
                let height = 1.0;
                let throat_h = 0.40;
                let throat_frac = 0.40;
                let exit_h = 0.80;

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
                        let geo = Nozzle {
                            length,
                            height,
                            throat_height: throat_h,
                            throat_frac,
                            exit_height: exit_h,
                        };

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
                for &f in &mesh.cell_faces
                    [mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1]]
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
                        self.viz_field_front = 0;
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
                SolverWorkerEvent::Stats { stats } => {
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
                SolverWorkerEvent::Running(running) => {
                    self.is_running = running;
                }
            }
        }

        if let (Some(viz), Some(renderer), Some(device)) = (
            self.viz_field.as_ref(),
            self.cfd_renderer.as_ref(),
            self.wgpu_device.as_ref(),
        ) {
            let ready = viz.ready_idx.load(Ordering::Acquire) % viz.buffers.len();
            if ready != self.viz_field_front {
                let mut renderer = renderer.lock().unwrap();
                renderer.update_bind_group(device, &viz.buffers[ready]);
                self.viz_field_front = ready;
                viz.front_idx.store(ready, Ordering::Release);
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
        self.solver_worker
            .send(SolverWorkerCommand::SetRunning(false));

        self.model_caps = model_caps;
        self.cached_cells = cached_cells;
        self.actual_min_cell_size = actual_min_cell_size;
        self.cached_u = cached_u;
        self.cached_p = cached_p;
        self.snapshot_seq = self.snapshot_seq.wrapping_add(1);
        self.invalidate_plot_cache();
        self.mesh = Some(mesh);

        self.cfd_renderer = renderer.map(|r| Arc::new(Mutex::new(r)));
        self.viz_field = viz_field.clone();
        self.viz_field_front = 0;
        if let Some(viz) = self.viz_field.as_ref() {
            viz.front_idx.store(0, Ordering::Release);
            viz.ready_idx.store(0, Ordering::Release);
        }
        self.last_init_trace_events = trace_init_events;

        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_moving_stats = None;
        self.cached_error = None;
        self.cached_message = None;

        // Track what the worker is about to run so controls the moving driver
        // forbids stay disabled for the driver's whole life, not just while the
        // enable checkbox happens to be ticked.
        self.solver_is_moving = matches!(mode, SolverMode::MovingMesh(_));

        self.solver_worker.send(SolverWorkerCommand::SetSolver {
            mode,
            viz_field,
        });
        self.sync_worker_params();

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
        let (mode, mesh, cached_u, cached_p, mut model_caps) = if request.enable_moving_mesh {
            CFDApp::build_moving_init(&request, &mut trace_init_events)?
        } else {
            CFDApp::build_static_init(&request, &mut trace_init_events)?
        };
        let n_cells = mesh.num_cells();

        // Update model_caps from the solver's ui_ports() (prefers PortRegistry over StateLayout)
        let ui_ports = mode.driver().solver().ui_ports();
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
            if state_size_bytes > 0 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cfd_viz:init_copy_state"),
                });
                encoder.copy_buffer_to_buffer(
                    mode.driver().solver().state_buffer(),
                    0,
                    &viz_buffer,
                    0,
                    state_size_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    mode.driver().solver().state_buffer(),
                    0,
                    &viz_buffer_1,
                    0,
                    state_size_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    mode.driver().solver().state_buffer(),
                    0,
                    &viz_buffer_2,
                    0,
                    state_size_bytes,
                );
                queue.submit(Some(encoder.finish()));
            }

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
            renderer.update_bind_group(device, &viz_buffer);
            (
                Some(renderer),
                Some(VizFieldBuffers {
                    buffers: [viz_buffer, viz_buffer_1, viz_buffer_2],
                    size_bytes: state_size_bytes,
                    front_idx: Arc::new(AtomicUsize::new(0)),
                    ready_idx: Arc::new(AtomicUsize::new(0)),
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

        let model = if request.model_id == "compressible" {
            compressible_model_with_eos(request.current_fluid.eos)?
        } else {
            all_models()?
                .into_iter()
                .find(|m| m.id == request.model_id)
                .ok_or_else(|| format!("unknown model id '{}'", request.model_id))?
        };
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
        let ale_id = CFDApp::ale_model_for(request.model_id)
            .ok_or_else(|| format!("model '{}' has no moving-mesh (ALE) variant", request.model_id))?;
        let model = all_models()?
            .into_iter()
            .find(|m| m.id == ale_id)
            .ok_or_else(|| format!("unknown ALE model id '{ale_id}'"))?;
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

    fn update_gpu_scheme(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_alpha_u(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_alpha_p(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_time_scheme(&self) {
        self.sync_worker_params();
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
        ctx: &egui::Context,
        has_solver: bool,
        min_val: f32,
        max_val: f32,
    ) {
        egui::SidePanel::right("legend").show(ctx, |ui| {
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
                        let min_vol =
                            mesh.cell_vol.iter().cloned().fold(f64::INFINITY, f64::min);
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
    fn render_bottom_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("plot_controls").show(ctx, |ui| {
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
                ui.radio_value(&mut self.render_mode, RenderMode::GpuDirect, "Direct");
                ui.radio_value(&mut self.render_mode, RenderMode::EguiPlot, "Plot (Slow)");
            });
        });
    }

    /// Render the central panel with the CFD visualization.
    fn render_central_panel(
        &self,
        ctx: &egui::Context,
        is_initializing: bool,
        has_solver: bool,
        min_val: f32,
        max_val: f32,
    ) {
        egui::CentralPanel::default().show(ctx, |ui| {
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
                                        _padding: 0,
                                    },
                                    draw_lines: self.show_mesh_lines,
                                },
                            );

                            ui.painter().add(cb);
                        }
                    }
                    RenderMode::EguiPlot => {
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

                        Plot::new("cfd_plot").data_aspect(1.0).show(ui, |plot_ui| {
                            for (i, polygon_points) in cells.iter().enumerate() {
                                let val = vals.get(i).copied().unwrap_or_default();
                                let t = (val - min_val as f64) / (max_val - min_val) as f64;
                                let color = get_color(t);

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
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_init();
        self.spawn_pending_init();
        self.poll_solver_worker();
        let init_in_progress = self.init_rx.is_some();
        let init_pending = self.pending_init_request.is_some();
        let is_initializing = init_in_progress || init_pending;

        if self.is_running || is_initializing {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        egui::SidePanel::left("controls").show(ctx, |ui| {
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
                            // dropdown's apply-defaults-then-reinit behaviour.
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
                            self.apply_model_defaults();
                            self.init_solver();
                        }
                    });

                    ui.group(|ui| {
                        ui.label("Mesh Parameters");
                        // The fitted (structured) mesh is a uniform grid: only a single
                        // target cell size applies, so hide the min-size / growth-rate
                        // controls it does not honour.
                        let show_grading = self.mesh_type.supports_size_grading();
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
                    });

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
                        if self.model_id == "allmach_pressure"
                            || self.model_id == "allmach_thermal"
                        {
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
                            && (self.model_id == "allmach_pressure"
                                || self.model_id == "allmach_thermal")
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
                                // supersonic outlet, no back-pressure.
                                let mut p_in = self.inlet_pressure;
                                if ui
                                    .add(
                                        adaptive_slider(&mut p_in, 0.0..=0.12)
                                            .text("Inlet pressure (gauge)"),
                                    )
                                    .on_hover_text(
                                        "Pressure-inlet CD nozzle: pins the inlet gauge \
                                         pressure (the gauge anchor); the outlet floats \
                                         (supersonic, no back-pressure). Higher ⇒ stronger \
                                         drop ⇒ supersonic exit (M_exit ≈ 1.07 at 0.07). \
                                         Over-expanded here (the throat over-chokes).",
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
                        let sound_speed = if self.model_caps.supports_eos_tuning {
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
                                "Acoustically-adaptive timestep (CFL-targeted). On the \
                                 moving mesh (ALE) it is applied inside the GCL dt \
                                 handshake: pinned per step before the swept-flux \
                                 closure, growth-limited 1.2×/step.",
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
                            if self.model_caps.supports_dtau {
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
                        let supports_preconditioner = self.model_caps.supports_preconditioner;
                        let model_owns_preconditioner = self.model_caps.model_owns_preconditioner;
                        let block_jacobi_supported =
                            self.model_caps.unknowns_per_cell <= UI_MAX_BLOCK_JACOBI;

                        if model_owns_preconditioner {
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
                        } else {
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
                        egui::ComboBox::from_label("Model")
                            .selected_text(CFDApp::model_label(self.model_id))
                            .show_ui(ui, |ui| {
                                for (id, label) in CFDApp::supported_ui_models() {
                                    let selectable =
                                        !moving_active || CFDApp::ale_model_for(id).is_some();
                                    let entry = ui.add_enabled(
                                        selectable,
                                        egui::SelectableLabel::new(self.model_id == id, label),
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
                                for choice in BackendChoice::ALL {
                                    ui.selectable_value(
                                        &mut self.backend,
                                        choice,
                                        choice.label(),
                                    );
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
                            self.sync_worker_params();
                            self.solver_worker.send(SolverWorkerCommand::SetRunning(true));
                        } else {
                            self.solver_worker.send(SolverWorkerCommand::SetRunning(false));
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
                        ui.label(format!("dt: {:.2e}", stats.dt));
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
                        ui.label(format!("Step time: {:.1} ms", stats.step_time_ms));

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

        if has_solver {
            self.ensure_plot_cache(matches!(self.render_mode, RenderMode::EguiPlot));
        }

        let (min_val, max_val) = self
            .plot_cache
            .as_ref()
            .map(|cache| (cache.min as f32, cache.max as f32))
            .unwrap_or((0.0, 1.0));

        self.render_right_panel(ctx, has_solver, min_val, max_val);
        self.render_bottom_panel(ctx);
        self.render_central_panel(ctx, is_initializing, has_solver, min_val, max_val);
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

fn solver_worker_stop_trace(
    trace: &mut Option<SolverTraceSession>,
    mode: &mut Option<SolverMode>,
) {
    let Some(mut session) = trace.take() else {
        return;
    };

    if let Some(s) = mode.as_mut().map(|m| m.driver_mut().solver_mut()) {
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
    let mut step_idx: u64 = 0;
    let mut last_stats_publish = std::time::Instant::now();
    let mut last_snapshot_publish = std::time::Instant::now();
    let stats_publish_interval = std::time::Duration::from_millis(33);
    let snapshot_publish_interval = std::time::Duration::from_millis(100);

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
                &mut step_idx,
                &mut last_stats_publish,
                &mut last_snapshot_publish,
                &evt_tx,
            ) {
                return;
            }
        }

        if !running {
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
                        &mut step_idx,
                        &mut last_stats_publish,
                        &mut last_snapshot_publish,
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
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
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
                    let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                    continue;
                }
            },
        };
        if let Some(DivergeReason::StepError(err)) = &outcome.diverged {
            running = false;
            let _ =
                evt_tx.send(SolverWorkerEvent::Error(format!("solver step failed: {err}")));
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
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
        let solver = mode.driver().solver();
        let step_time_ms = outcome.step_time_ms;

        if let Some(viz_field) = viz_field.as_ref() {
            if viz_field.size_bytes > 0 {
                let n = viz_field.buffers.len().max(1);
                let front = viz_field.front_idx.load(Ordering::Acquire) % n;
                let ready = viz_field.ready_idx.load(Ordering::Acquire) % n;

                let mut write_idx = (ready + 1) % n;
                for _ in 0..n {
                    if write_idx != front && write_idx != ready {
                        break;
                    }
                    write_idx = (write_idx + 1) % n;
                }
                if write_idx == front || write_idx == ready {
                    // Should be unreachable with n>=3, but keep a safe fallback.
                    write_idx = (front + 1) % n;
                }
                solver.copy_state_to_buffer(&viz_field.buffers[write_idx]);
                viz_field.ready_idx.store(write_idx, Ordering::Release);
            }
        }

        if outcome.should_stop {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        }

        let linear_solves = outcome.linear_stats.len() as u32;
        let linear_last = outcome.linear_stats.last().copied().unwrap_or_default();
        if matches!(outcome.diverged, Some(DivergeReason::LinearSolver)) {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(
                "divergence detected (linear solver)".to_string(),
            ));
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        }

        let mut stats = CachedGpuStats {
            dt: solver.dt(),
            step_time_ms,
            linear_solves,
            linear_last,
            ..Default::default()
        };

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
        stats.positivity_rho_undershoots = step_stats.positivity_rho_undershoot_count.unwrap_or(0);
        stats.positivity_pressure_undershoots = step_stats
            .positivity_pressure_undershoot_count
            .unwrap_or(0);

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
                let step_stats = solver.step_stats();
                let outer_str = step_stats
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
                            format!(
                                " outer(iters={iters}, u={res_u:.3e}, p={res_p:.3e}{status_suffix})"
                            )
                        } else {
                            format!(" outer(iters={iters}{status_suffix})")
                        }
                    })
                    .unwrap_or_default();

                eprintln!(
                    "[cfd2][{}] step={} t={:.4e} dt={:.2e} max|u|={:.3e} p=[{:.3e},{:.3e}] solves={} last(iters={}, res={:.3e}, conv={}, div={}){} nonfinite(u={}, p={})",
                    model_id,
                    step_idx,
                    solver.time(),
                    solver.dt(),
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
                let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                continue;
            }

            let _ = evt_tx.send(SolverWorkerEvent::Snapshot {
                u: rb.u,
                p: rb.p,
                stats,
            });
        } else if step_idx == 0 || last_stats_publish.elapsed() >= stats_publish_interval {
            last_stats_publish = std::time::Instant::now();
            let _ = evt_tx.send(SolverWorkerEvent::Stats { stats });
        }

        if let Some(trace) = trace.as_mut() {
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
                wall_time_ms: step_time_ms,
                linear_solves,
                graph,
                max_u: trace_max_u,
                p_min: trace_p_min,
                p_max: trace_p_max,
            });
            let _ = trace.writer.write_event(&event);
        }

        step_idx = step_idx.wrapping_add(1);
        thread::sleep(std::time::Duration::from_millis(1));
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
    step_idx: &mut u64,
    last_stats_publish: &mut std::time::Instant,
    last_snapshot_publish: &mut std::time::Instant,
    evt_tx: &mpsc::Sender<SolverWorkerEvent>,
) -> bool {
    match cmd {
        SolverWorkerCommand::SetSolver {
            mode: next,
            viz_field: next_viz_field,
        } => {
            if trace.is_some() {
                solver_worker_stop_trace(trace, mode);
            }
            *model_id = next.driver().solver().model().id;
            *mode = Some(next);
            *viz_field = next_viz_field;
            *running = false;
            *step_idx = 0;
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));

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
                m.driver_mut().apply_params(params);
            }
        }
        SolverWorkerCommand::ClearSolver => {
            solver_worker_stop_trace(trace, mode);
            *mode = None;
            *model_id = "<uninitialized>";
            *running = false;
            *step_idx = 0;
            *viz_field = None;
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
        }
        SolverWorkerCommand::SetRunning(next_running) => {
            if next_running && mode.is_none() {
                let _ = evt_tx.send(SolverWorkerEvent::Error(
                    "cannot start solver: no solver is initialized".to_string(),
                ));
                let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                *running = false;
                return true;
            }
            if next_running {
                *step_idx = 0;
                let now = std::time::Instant::now();
                *last_stats_publish = now;
                *last_snapshot_publish = now;
            }
            *running = next_running;
            let _ = evt_tx.send(SolverWorkerEvent::Running(*running));
        }
        SolverWorkerCommand::UpdateParams(next_params) => {
            *params = next_params;
            if let Some(m) = mode.as_mut() {
                // The moving path never runs the SOLVER-side adaptive dt (a
                // post-closure re-scale breaks the GCL) — the user's choice
                // routes to the driver-side controller instead, and the live
                // dt slider re-bases the pinned dt (configured_dt was
                // previously captured at build only, leaving the slider
                // silently inert mid-run on this path).
                if let SolverMode::MovingMesh(moving) = m {
                    let adaptive_dt = params.adaptive_dt;
                    params.adaptive_dt = false;
                    moving.set_adaptive_dt(adaptive_dt.then_some(params.target_cfl));
                    moving.set_configured_dt(params.requested_dt as f64);
                }
                m.driver_mut().apply_params(params);
            }
            if let (Some(trace), Some(m)) = (trace.as_mut(), mode.as_ref()) {
                let event = tracefmt::TraceEvent::Params(tracefmt::TraceParamsEvent {
                    step: *step_idx,
                    sim_time: m.driver().solver().time(),
                    params: trace_runtime_params_from_worker(*params),
                });
                let _ = trace.writer.write_event(&event);
            }
        }
        SolverWorkerCommand::StartTrace { path, header } => {
            solver_worker_stop_trace(trace, mode);

            match tracefmt::TraceWriter::create(&path) {
                Ok(mut writer) => {
                    let profiling_enabled = header.ui.profiling_enabled;
                    let event = tracefmt::TraceEvent::Header(Box::new(header));
                    let _ = writer.write_event(&event);

                    if let Some(s) = mode.as_mut().map(|m| m.driver_mut().solver_mut()) {
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
