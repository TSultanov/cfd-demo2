use crate::solver::mesh::{BoundaryType, Mesh};
use crate::solver::model::helpers::{
    SolverCompressibleInletExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use crate::solver::scheme::Scheme;
use crate::solver::{GpuLowMachPrecondModel, PreconditionerType, SteppingMode, TimeScheme};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::io::Write;
use std::path::{Path, PathBuf};

const TRACE_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TraceEvent {
    Header(TraceHeader),
    Init(TraceInitEvent),
    Params(TraceParamsEvent),
    Step(TraceStepEvent),
    Profiling(TraceProfilingEvent),
    Footer(TraceFooter),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHeader {
    pub format_version: u32,
    pub created_unix_ms: u64,
    pub case: TraceCase,
    pub initial_params: TraceRuntimeParams,
    pub ui: TraceUiInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceInitEvent {
    pub stage: String,
    pub wall_time_ms: f32,
    pub detail: Option<String>,
}

thread_local! {
    static INIT_COLLECTOR: RefCell<Option<*mut Vec<TraceInitEvent>>> = const { RefCell::new(None) };
}

pub struct InitCollectorGuard {
    prev: Option<*mut Vec<TraceInitEvent>>,
}

impl Drop for InitCollectorGuard {
    fn drop(&mut self) {
        INIT_COLLECTOR.with(|collector| {
            let mut collector = collector.borrow_mut();
            *collector = self.prev.take();
        });
    }
}

pub fn install_init_collector(events: &mut Vec<TraceInitEvent>) -> InitCollectorGuard {
    INIT_COLLECTOR.with(|collector| {
        let mut collector = collector.borrow_mut();
        let prev = collector.replace(events as *mut Vec<TraceInitEvent>);
        InitCollectorGuard { prev }
    })
}

pub fn record_init_event(
    stage: impl Into<String>,
    elapsed: std::time::Duration,
    detail: Option<String>,
) {
    let event = TraceInitEvent {
        stage: stage.into(),
        wall_time_ms: elapsed.as_secs_f32() * 1000.0,
        detail,
    };

    INIT_COLLECTOR.with(|collector| {
        let collector = collector.borrow();
        let Some(ptr) = *collector else {
            return;
        };

        // SAFETY: `install_init_collector` is only meant to be used by the UI init thread,
        // and the guard ensures the referenced Vec outlives this call. The pointer is
        // thread-local, so there is no cross-thread access.
        unsafe { (&mut *ptr).push(event) };
    });
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceUiInfo {
    pub profiling_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFooter {
    pub closed_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCase {
    pub model_id: String,
    pub geometry: TraceGeometry,
    pub mesh_type: TraceMeshType,
    pub min_cell_size: f64,
    pub max_cell_size: f64,
    pub growth_rate: f64,
    pub fluid: TraceFluid,
    pub stepping_mode: TraceSteppingMode,
    pub mesh: TraceMesh,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceGeometry {
    BackwardsStep,
    ChannelObstacle,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceMeshType {
    CutCell,
    Delaunay,
    Voronoi,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceSteppingMode {
    Explicit,
    Implicit,
    Coupled,
}

impl TraceSteppingMode {
    pub fn from_solver(mode: SteppingMode) -> Self {
        match mode {
            SteppingMode::Explicit => TraceSteppingMode::Explicit,
            SteppingMode::Implicit { .. } => TraceSteppingMode::Implicit,
            SteppingMode::Coupled => TraceSteppingMode::Coupled,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFluid {
    pub name: Option<String>,
    pub density: f64,
    pub viscosity: f64,
    pub eos: TraceEosSpec,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceEosSpec {
    IdealGas {
        gamma: f64,
        gas_constant: f64,
        temperature: f64,
    },
    LinearCompressibility {
        bulk_modulus: f64,
        rho_ref: f64,
        p_ref: f64,
    },
    Constant,
}

impl From<crate::solver::model::EosSpec> for TraceEosSpec {
    fn from(value: crate::solver::model::EosSpec) -> Self {
        match value {
            crate::solver::model::EosSpec::IdealGas {
                gamma,
                gas_constant,
                temperature,
            } => TraceEosSpec::IdealGas {
                gamma,
                gas_constant,
                temperature,
            },
            crate::solver::model::EosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                p_ref,
            } => TraceEosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                p_ref,
            },
            crate::solver::model::EosSpec::Constant => TraceEosSpec::Constant,
        }
    }
}

impl From<TraceEosSpec> for crate::solver::model::EosSpec {
    fn from(value: TraceEosSpec) -> Self {
        match value {
            TraceEosSpec::IdealGas {
                gamma,
                gas_constant,
                temperature,
            } => crate::solver::model::EosSpec::IdealGas {
                gamma,
                gas_constant,
                temperature,
            },
            TraceEosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                p_ref,
            } => crate::solver::model::EosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                p_ref,
            },
            TraceEosSpec::Constant => crate::solver::model::EosSpec::Constant,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRuntimeParams {
    pub adaptive_dt: bool,
    pub target_cfl: f64,
    pub requested_dt: f32,
    pub dtau: f32,
    pub advection_scheme: TraceScheme,
    pub time_scheme: TraceTimeScheme,
    pub preconditioner: TracePreconditioner,
    pub outer_iters: u32,
    pub low_mach_model: TraceLowMachModel,
    pub low_mach_theta_floor: f32,
    pub low_mach_pressure_coupling_alpha: f32,
    pub alpha_u: f32,
    pub alpha_p: f32,
    pub inlet_velocity: f32,
    pub density: f32,
    pub viscosity: f32,
    pub eos: TraceEosSpec,
}

impl TraceRuntimeParams {
    pub fn apply_to_solver(&self, solver: &mut crate::solver::UnifiedSolver) {
        solver.set_dt(self.requested_dt);

        let named_params = solver.model().named_param_keys();
        let has_param = |key: &str| named_params.iter().any(|&k| k == key);

        if has_param("dtau") {
            let _ = solver.set_dtau(self.dtau);
        }
        if has_param("density") {
            let _ = solver.set_density(self.density);
        }
        if has_param("viscosity") {
            let _ = solver.set_viscosity(self.viscosity);
        }
        if has_param("eos.gamma") {
            let _ = solver.set_eos(&crate::solver::model::EosSpec::from(self.eos));
        }
        if has_param("outer_iters") {
            let _ = solver.set_outer_iters(self.outer_iters as usize);
        }
        if has_param("low_mach.model") {
            let _ = solver.set_precond_model(GpuLowMachPrecondModel::from(self.low_mach_model));
        }
        if has_param("low_mach.theta_floor") {
            let _ = solver.set_precond_theta_floor(self.low_mach_theta_floor);
        }
        if has_param("low_mach.pressure_coupling_alpha") {
            let _ = solver.set_precond_pressure_coupling_alpha(self.low_mach_pressure_coupling_alpha);
        }

        if has_param("advection_scheme") {
            solver.set_advection_scheme(Scheme::from(self.advection_scheme));
        }
        if has_param("time_scheme") {
            solver.set_time_scheme(TimeScheme::from(self.time_scheme));
        }
        if has_param("preconditioner") {
            solver.set_preconditioner(PreconditionerType::from(self.preconditioner));
        }

        if has_param("alpha_u") {
            let _ = solver.set_alpha_u(self.alpha_u);
        }
        if has_param("alpha_p") {
            let _ = solver.set_alpha_p(self.alpha_p);
        }

        // Boundary conditions depend on whether this is the compressible model.
        let model = solver.model();
        let mut has_rho = false;
        let mut has_rho_u = false;
        let mut has_rho_e = false;
        let mut has_u = false;
        for eqn in model.system.equations() {
            match eqn.target().name() {
                "rho" => has_rho = true,
                "rho_u" => has_rho_u = true,
                "rho_e" => has_rho_e = true,
                "u" => has_u = true,
                _ => {}
            }
        }

        if has_rho && has_rho_u && has_rho_e && has_u {
            let eos = crate::solver::model::EosSpec::from(self.eos);
            let _ = solver.set_compressible_inlet_isothermal_x(self.density, self.inlet_velocity, &eos);
        } else {
            let _ = solver.set_inlet_velocity(self.inlet_velocity);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceScheme {
    Upwind,
    SecondOrderUpwind,
    QUICK,
    SecondOrderUpwindMinMod,
    SecondOrderUpwindVanLeer,
    QUICKMinMod,
    QUICKVanLeer,
}

impl From<Scheme> for TraceScheme {
    fn from(value: Scheme) -> Self {
        match value {
            Scheme::Upwind => TraceScheme::Upwind,
            Scheme::SecondOrderUpwind => TraceScheme::SecondOrderUpwind,
            Scheme::QUICK => TraceScheme::QUICK,
            Scheme::SecondOrderUpwindMinMod => TraceScheme::SecondOrderUpwindMinMod,
            Scheme::SecondOrderUpwindVanLeer => TraceScheme::SecondOrderUpwindVanLeer,
            Scheme::QUICKMinMod => TraceScheme::QUICKMinMod,
            Scheme::QUICKVanLeer => TraceScheme::QUICKVanLeer,
        }
    }
}

impl From<TraceScheme> for Scheme {
    fn from(value: TraceScheme) -> Self {
        match value {
            TraceScheme::Upwind => Scheme::Upwind,
            TraceScheme::SecondOrderUpwind => Scheme::SecondOrderUpwind,
            TraceScheme::QUICK => Scheme::QUICK,
            TraceScheme::SecondOrderUpwindMinMod => Scheme::SecondOrderUpwindMinMod,
            TraceScheme::SecondOrderUpwindVanLeer => Scheme::SecondOrderUpwindVanLeer,
            TraceScheme::QUICKMinMod => Scheme::QUICKMinMod,
            TraceScheme::QUICKVanLeer => Scheme::QUICKVanLeer,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceTimeScheme {
    Euler,
    BDF2,
}

impl From<TimeScheme> for TraceTimeScheme {
    fn from(value: TimeScheme) -> Self {
        match value {
            TimeScheme::Euler => TraceTimeScheme::Euler,
            TimeScheme::BDF2 => TraceTimeScheme::BDF2,
        }
    }
}

impl From<TraceTimeScheme> for TimeScheme {
    fn from(value: TraceTimeScheme) -> Self {
        match value {
            TraceTimeScheme::Euler => TimeScheme::Euler,
            TraceTimeScheme::BDF2 => TimeScheme::BDF2,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TracePreconditioner {
    Jacobi,
    Amg,
    BlockJacobi,
}

impl From<PreconditionerType> for TracePreconditioner {
    fn from(value: PreconditionerType) -> Self {
        match value {
            PreconditionerType::Jacobi => TracePreconditioner::Jacobi,
            PreconditionerType::Amg => TracePreconditioner::Amg,
            PreconditionerType::BlockJacobi => TracePreconditioner::BlockJacobi,
        }
    }
}

impl From<TracePreconditioner> for PreconditionerType {
    fn from(value: TracePreconditioner) -> Self {
        match value {
            TracePreconditioner::Jacobi => PreconditionerType::Jacobi,
            TracePreconditioner::Amg => PreconditionerType::Amg,
            TracePreconditioner::BlockJacobi => PreconditionerType::BlockJacobi,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceLowMachModel {
    Off,
    Legacy,
    WeissSmith,
}

impl From<GpuLowMachPrecondModel> for TraceLowMachModel {
    fn from(value: GpuLowMachPrecondModel) -> Self {
        match value {
            GpuLowMachPrecondModel::Off => TraceLowMachModel::Off,
            GpuLowMachPrecondModel::Legacy => TraceLowMachModel::Legacy,
            GpuLowMachPrecondModel::WeissSmith => TraceLowMachModel::WeissSmith,
        }
    }
}

impl From<TraceLowMachModel> for GpuLowMachPrecondModel {
    fn from(value: TraceLowMachModel) -> Self {
        match value {
            TraceLowMachModel::Off => GpuLowMachPrecondModel::Off,
            TraceLowMachModel::Legacy => GpuLowMachPrecondModel::Legacy,
            TraceLowMachModel::WeissSmith => GpuLowMachPrecondModel::WeissSmith,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceParamsEvent {
    pub step: u64,
    pub sim_time: f32,
    pub params: TraceRuntimeParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStepEvent {
    pub step: u64,
    pub sim_time: f32,
    pub dt: f32,
    pub wall_time_ms: f32,
    pub linear_solves: Vec<TraceLinearSolverStats>,
    pub graph: Vec<TraceGraphTiming>,
    pub max_u: Option<f64>,
    pub p_min: Option<f64>,
    pub p_max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceLinearSolverStats {
    pub iterations: u32,
    pub residual: f32,
    pub converged: bool,
    pub diverged: bool,
    pub time_ms: f32,
}

impl From<crate::solver::LinearSolverStats> for TraceLinearSolverStats {
    fn from(value: crate::solver::LinearSolverStats) -> Self {
        Self {
            iterations: value.iterations,
            residual: value.residual,
            converged: value.converged,
            diverged: value.diverged,
            time_ms: value.time.as_secs_f32() * 1000.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceGraphTiming {
    pub label: String,
    pub seconds: f64,
    pub nodes: Option<Vec<TraceGraphNodeTiming>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceGraphNodeTiming {
    pub label: String,
    pub seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceProfilingEvent {
    pub session_total_seconds: f64,
    pub iteration_count: u64,
    pub categories: Vec<(String, TraceCategoryStats)>,
    pub locations: Vec<(String, TraceCategoryStats)>,
    pub memory_cpu: TraceMemoryStats,
    pub memory_gpu: TraceMemoryStats,
    pub memory_locations_cpu: Vec<(String, TraceMemoryStats)>,
    pub memory_locations_gpu: Vec<(String, TraceMemoryStats)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCategoryStats {
    pub total_seconds: f64,
    pub call_count: u64,
    pub min_seconds: f64,
    pub max_seconds: f64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMemoryStats {
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub free_bytes: u64,
    pub free_count: u64,
    pub max_alloc_request: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMesh {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub v_fixed: Vec<bool>,

    pub face_v1: Vec<u32>,
    pub face_v2: Vec<u32>,
    pub face_owner: Vec<u32>,
    pub face_neighbor: Vec<i32>,
    pub face_boundary: Vec<i8>,
    pub face_nx: Vec<f64>,
    pub face_ny: Vec<f64>,
    pub face_area: Vec<f64>,
    pub face_cx: Vec<f64>,
    pub face_cy: Vec<f64>,

    pub cell_cx: Vec<f64>,
    pub cell_cy: Vec<f64>,
    pub cell_vol: Vec<f64>,

    pub cell_faces: Vec<u32>,
    pub cell_face_offsets: Vec<u32>,

    pub cell_vertices: Vec<u32>,
    pub cell_vertex_offsets: Vec<u32>,
}

impl TraceMesh {
    pub fn from_mesh(mesh: &Mesh) -> Result<Self, String> {
        fn to_u32_vec(src: &[usize], label: &str) -> Result<Vec<u32>, String> {
            src.iter()
                .map(|&v| {
                    u32::try_from(v).map_err(|_| format!("{label} index {v} overflows u32"))
                })
                .collect()
        }

        let face_neighbor = mesh
            .face_neighbor
            .iter()
            .map(|&v| v.map(|n| n as i32).unwrap_or(-1))
            .collect();

        let face_boundary = mesh
            .face_boundary
            .iter()
            .map(|b| match b {
                None => -1,
                Some(BoundaryType::Inlet) => 0,
                Some(BoundaryType::Outlet) => 1,
                Some(BoundaryType::Wall) => 2,
                Some(BoundaryType::SlipWall) => 3,
                Some(BoundaryType::MovingWall) => 4,
            })
            .collect();

        Ok(Self {
            vx: mesh.vx.clone(),
            vy: mesh.vy.clone(),
            v_fixed: mesh.v_fixed.clone(),
            face_v1: to_u32_vec(&mesh.face_v1, "face_v1")?,
            face_v2: to_u32_vec(&mesh.face_v2, "face_v2")?,
            face_owner: to_u32_vec(&mesh.face_owner, "face_owner")?,
            face_neighbor,
            face_boundary,
            face_nx: mesh.face_nx.clone(),
            face_ny: mesh.face_ny.clone(),
            face_area: mesh.face_area.clone(),
            face_cx: mesh.face_cx.clone(),
            face_cy: mesh.face_cy.clone(),
            cell_cx: mesh.cell_cx.clone(),
            cell_cy: mesh.cell_cy.clone(),
            cell_vol: mesh.cell_vol.clone(),
            cell_faces: to_u32_vec(&mesh.cell_faces, "cell_faces")?,
            cell_face_offsets: to_u32_vec(&mesh.cell_face_offsets, "cell_face_offsets")?,
            cell_vertices: to_u32_vec(&mesh.cell_vertices, "cell_vertices")?,
            cell_vertex_offsets: to_u32_vec(&mesh.cell_vertex_offsets, "cell_vertex_offsets")?,
        })
    }

    pub fn to_mesh(&self) -> Result<Mesh, String> {
        fn to_usize_vec(src: &[u32]) -> Vec<usize> {
            src.iter().map(|&v| v as usize).collect()
        }

        let face_neighbor = self
            .face_neighbor
            .iter()
            .map(|&v| if v < 0 { None } else { Some(v as usize) })
            .collect();

        let face_boundary = self
            .face_boundary
            .iter()
            .map(|&b| match b {
                -1 => Ok(None),
                0 => Ok(Some(BoundaryType::Inlet)),
                1 => Ok(Some(BoundaryType::Outlet)),
                2 => Ok(Some(BoundaryType::Wall)),
                3 => Ok(Some(BoundaryType::SlipWall)),
                4 => Ok(Some(BoundaryType::MovingWall)),
                v => Err(format!("invalid face_boundary value {v}")),
            })
            .collect::<Result<_, _>>()?;

        Ok(Mesh {
            vx: self.vx.clone(),
            vy: self.vy.clone(),
            v_fixed: self.v_fixed.clone(),
            face_v1: to_usize_vec(&self.face_v1),
            face_v2: to_usize_vec(&self.face_v2),
            face_owner: to_usize_vec(&self.face_owner),
            face_neighbor,
            face_boundary,
            face_nx: self.face_nx.clone(),
            face_ny: self.face_ny.clone(),
            face_area: self.face_area.clone(),
            face_cx: self.face_cx.clone(),
            face_cy: self.face_cy.clone(),
            cell_cx: self.cell_cx.clone(),
            cell_cy: self.cell_cy.clone(),
            cell_vol: self.cell_vol.clone(),
            cell_faces: to_usize_vec(&self.cell_faces),
            cell_face_offsets: to_usize_vec(&self.cell_face_offsets),
            cell_vertices: to_usize_vec(&self.cell_vertices),
            cell_vertex_offsets: to_usize_vec(&self.cell_vertex_offsets),
        })
    }
}

pub struct TraceWriter {
    path: PathBuf,
    writer: std::io::BufWriter<std::fs::File>,
    events_written: usize,
    flush_every: usize,
}

impl TraceWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_owned();
        let file = std::fs::File::create(&path)
            .map_err(|err| format!("failed to create trace file '{}': {err}", path.display()))?;
        Ok(Self {
            path,
            writer: std::io::BufWriter::new(file),
            events_written: 0,
            flush_every: 25,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write_event(&mut self, event: &TraceEvent) -> Result<(), String> {
        serde_json::to_writer(&mut self.writer, event)
            .map_err(|err| format!("failed to serialize trace event: {err}"))?;
        self.writer
            .write_all(b"\n")
            .map_err(|err| format!("failed to write trace event: {err}"))?;
        self.events_written += 1;
        if self.events_written % self.flush_every == 0 {
            let _ = self.writer.flush();
        }
        Ok(())
    }

    pub fn close(mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|err| format!("failed to flush trace file: {err}"))?;
        Ok(())
    }
}

pub fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn make_header(case: TraceCase, initial_params: TraceRuntimeParams, profiling_enabled: bool) -> TraceHeader {
    TraceHeader {
        format_version: TRACE_FORMAT_VERSION,
        created_unix_ms: now_unix_ms(),
        case,
        initial_params,
        ui: TraceUiInfo { profiling_enabled },
    }
}
