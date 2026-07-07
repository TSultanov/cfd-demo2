//! End-to-end on-device moving-mesh GCL (Phase C stage D4+D5): drive the FULL
//! all-Mach compressible ALE loop where the mesh is reconstructed **entirely on
//! the GPU** each step — Voronoi diagram (engine) → face topology + geometry
//! (emit) → cell↔face CSR (csr_gpu) → cell volumes (cell_geom) → ALE swept
//! fluxes (swept_gpu) — assembled into the solver mesh by
//! [`assemble_solver_mesh`] with NO CPU `assemble_meshless_from_seeds` and NO CPU
//! swept-flux path.
//!
//! The gate is free-stream preservation (the discrete GCL): a uniform
//! compressible free stream `U=(U0,0)`, `p=0` gauge, `rho=rho_ref` is an exact
//! fixed point of the moving-mesh scheme, so ANY drift in U/p/rho over a
//! prescribed (flip-free) swirl means the device-built cell volumes and swept
//! fluxes are not GCL-consistent. Because the swept quads and the cell volumes
//! come from the SAME canonical vertices (see `swept_gpu`), the identity closes
//! to the f32 floor with no spanning-forest closure.
//!
//! ```sh
//! cargo test --features meshgen --test gpu_moving_gcl_test -- --nocapture
//! ```
#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{meshless_seed_points, SeedKind};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{
    assemble_solver_mesh, boundary_spec_f32, CellGeometry, EmitFaces, GpuCsr, GpuVoronoiEngine,
    SweptFluxGeometry,
};
use cfd2::solver::mesh::{BoundaryType, Mesh, RectangularChannel};
use cfd2::solver::model::allmach_pressure_ale_model;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use cfd2::sim::{RuntimeParams, SolverDriver};
use nalgebra::{Point2, Vector2};

const LX: f64 = 1.0;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const DT: f32 = 0.005;
const PERIOD: f64 = 80.0 * DT as f64;
const STEPS: usize = 60;
const U0: (f32, f32) = (1.0, 0.0);
const PSI: f32 = 1.0e-4;
const RHO_REF: f32 = 1.0;

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

fn test_params() -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: DT,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 1000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 6,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: U0.0,
        density: RHO_REF,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: PSI,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

/// Flip-free interior swirl (identical to the CPU free-stream gate): rotate each
/// seed about the centre by a boundary-vanishing bump.
fn swirl(p: [f64; 2], t: f64) -> [f64; 2] {
    let (cx, cy) = (0.5 * LX, 0.5 * LY);
    let bump = (std::f64::consts::PI * p[0] / LX).sin().powi(2)
        * (std::f64::consts::PI * p[1] / LY).sin().powi(2);
    let theta = 0.03 * (2.0 * std::f64::consts::PI * t / PERIOD).sin() * bump;
    let (dx, dy) = (p[0] - cx, p[1] - cy);
    let (c, s) = (theta.cos(), theta.sin());
    [cx + c * dx - s * dy, cy + s * dx + c * dy]
}

/// Tag boundary faces of the device-built mesh by centre position — Inlet left,
/// Outlet right, SlipWall top/bottom (the free-stream BC convention).
fn tag_channel(mesh: &mut Mesh) {
    let eps = 1e-4;
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_some() {
            continue;
        }
        let (x, y) = (mesh.face_cx[f], mesh.face_cy[f]);
        mesh.face_boundary[f] = if x < eps {
            Some(BoundaryType::Inlet)
        } else if x > LX - eps {
            Some(BoundaryType::Outlet)
        } else if y < eps || y > LY - eps {
            Some(BoundaryType::SlipWall)
        } else {
            continue;
        };
    }
}

/// The on-device regen orchestrator: holds the GPU passes and rebuilds the
/// solver mesh + ALE mesh fluxes from a seed set entirely on device.
struct DeviceRegen {
    engine: GpuVoronoiEngine,
    emit: EmitFaces,
    csr: GpuCsr,
    cellgeom: CellGeometry,
    swept: SweptFluxGeometry,
    cache: StagingBufferCache,
    kinds: Vec<SeedKind>,
    spec: cfd2::meshgen::meshless::BoundarySpec,
    flags: Vec<u32>,
}

impl DeviceRegen {
    fn new(ctx: &GpuContext, n: usize, domain: Vector2<f64>, tol: &MeshgenTolerances,
           kinds: Vec<SeedKind>, spec: cfd2::meshgen::meshless::BoundarySpec) -> Self {
        Self {
            engine: GpuVoronoiEngine::new(&ctx.device, n as u32, domain, tol),
            emit: EmitFaces::new(&ctx.device),
            csr: GpuCsr::new(&ctx.device),
            cellgeom: CellGeometry::new(&ctx.device),
            swept: SweptFluxGeometry::new(&ctx.device),
            cache: StagingBufferCache::default(),
            flags: vec![0u32; n],
            kinds,
            spec,
        }
    }

    /// Regen the mesh at `new_seeds` (with `old_seeds` for the swept fluxes) and
    /// return the assembled solver mesh + the per-face ALE mesh fluxes (swept
    /// area / dt), all built on device.
    fn regen(&mut self, ctx: &GpuContext, new_seeds: &[f32], old_seeds: &[f32], dt: f32)
             -> (Mesh, Vec<f32>, usize) {
        self.engine.upload_case(&ctx.device, &ctx.queue, new_seeds, &self.flags, &self.kinds, &self.spec);
        let idx = self.engine.run_regen(&ctx.device, &ctx.queue);
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
        let _ = self.engine.resolve_flagged(ctx, &self.cache);

        let faces = self.emit.emit(ctx, &self.cache, &self.engine);
        let csr = self.csr.build_csr(ctx, &self.cache, &self.engine);
        let cells = self.cellgeom.build(ctx, &self.cache, &self.engine);
        // Static boundary: the t^n spec == the uploaded t^{n+1} spec.
        let sw = self.swept.compute(ctx, &self.cache, &self.engine, old_seeds, &self.spec);
        let needs_cpu: usize = sw.needs_cpu.iter().filter(|&&x| x != 0).count();

        let mut mesh = assemble_solver_mesh(&faces, &csr, &cells, &sw).expect("assemble device mesh");
        tag_channel(&mut mesh);
        let fluxes: Vec<f32> = sw.swept.iter().map(|&s| s / dt).collect();
        (mesh, fluxes, needs_cpu)
    }
}

/// interleaved f32 seeds at time `t` (interior swirl; boundary fixed at t=0).
fn seeds_at(seeds0: &[Point2<f64>], kinds: &[SeedKind], t: f64) -> Vec<f32> {
    seeds0
        .iter()
        .zip(kinds)
        .flat_map(|(s0, k)| {
            let p = if *k == SeedKind::Interior { swirl([s0.x, s0.y], t) } else { [s0.x, s0.y] };
            [p[0] as f32, p[1] as f32]
        })
        .collect()
}

/// ISOLATION reference: the SAME free stream driven by the CPU-mesh + certified
/// CPU-swept `MovingMeshDriver`, but on the GPU BACKEND. If this also drifts, the
/// fault is the GPU all-Mach ALE free stream itself (untested elsewhere — the CPU
/// gate forces the CPU backend), NOT the device regen.
#[test]
fn movingmesh_driver_gpu_freestream_reference() {
    use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
    use cfd2::meshgen::LloydConfig;
    use cfd2::sim::{MeshMotionSpec, MovingMeshDriver};
    let Some(ctx) = gpu_context() else { return };
    let geo = RectangularChannel { length: LX, height: LY };
    let domain = Vector2::new(LX, LY);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();
    let params = test_params();
    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        allmach_pressure_ale_model().expect("model"),
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("gpu moving driver");
    moving.set_boundary_retag(Some(tag_channel));
    moving.driver_mut().apply_params(&params);
    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U") as usize;
    let mut max_du = 0.0f32;
    for step in 0..STEPS {
        let (outcome, _stats) = moving.step(false).unwrap_or_else(|e| panic!("step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        let state = pollster::block_on(moving.driver().solver().read_state_f32());
        let mut du = 0.0f32;
        for c in 0..n {
            du = du.max((state[c * stride + u_off] - U0.0).abs())
                .max((state[c * stride + u_off + 1] - U0.1).abs());
        }
        if step < 3 || du > 1e-2 {
            eprintln!("[gpu-ref-movingmesh]   step {step}: du={du:.3e}");
        }
        max_du = max_du.max(du);
    }
    eprintln!("[gpu-ref-movingmesh] GPU-backend MovingMeshDriver free-stream: max|U-U0| = {max_du:.3e}");
}

/// Production path: the SAME free stream driven by `MovingMeshDriver` with the
/// opt-in `set_gpu_regen(true)` — the driver reconstructs the mesh entirely on
/// the GPU each step (no CPU assemble / swept) through its normal `step()`.
#[test]
fn movingmesh_driver_gpu_regen_freestream() {
    use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
    use cfd2::meshgen::LloydConfig;
    use cfd2::sim::{MeshMotionSpec, MovingMeshDriver};
    let Some(ctx) = gpu_context() else { return };
    let geo = RectangularChannel { length: LX, height: LY };
    let domain = Vector2::new(LX, LY);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();
    let params = test_params();
    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        allmach_pressure_ale_model().expect("model"),
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("gpu moving driver");
    moving.set_gpu_regen(true);
    assert!(moving.gpu_regen_active(), "gpu_regen should be active on the GPU backend");
    moving.set_boundary_retag(Some(tag_channel));
    moving.driver_mut().apply_params(&params);
    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U") as usize;
    let mut max_du = 0.0f32;
    let mut max_scl = 0.0f64;
    for step in 0..STEPS {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("gpu_regen step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        max_scl = max_scl.max(stats.scl_defect);
        // On-device per-step SCL defect: the f32 gap between two INDEPENDENT
        // regens' canonical areas at the shared seed positions (different ring
        // orders ⇒ different shoelace summation ~1e-7), plus a one-time ~1.4e-6
        // seam at step 0 (V^n is the CPU-built initial mesh). Tiny; the free-stream
        // drift below is the load-bearing GCL gate.
        assert!(stats.scl_defect < 5e-6, "step {step}: on-device SCL defect {:.3e}", stats.scl_defect);
        let state = pollster::block_on(moving.driver().solver().read_state_f32());
        let mut du = 0.0f32;
        for c in 0..n {
            du = du.max((state[c * stride + u_off] - U0.0).abs())
                .max((state[c * stride + u_off + 1] - U0.1).abs());
        }
        max_du = max_du.max(du);
    }
    println!(
        "[gpu-regen-driver] MovingMeshDriver::step with set_gpu_regen(true): {STEPS} steps, \
         max|U-U0| = {max_du:.3e}, max on-device SCL defect = {max_scl:.3e}"
    );
    assert!(max_du < 1e-4, "free-stream U drift {max_du:.3e} via the driver gpu_regen path");
}

#[test]
fn device_regen_freestream_gcl() {
    let Some(ctx) = gpu_context() else { return };
    let geo = RectangularChannel { length: LX, height: LY };
    let domain = Vector2::new(LX, LY);
    let (seeds0, kinds, spec) = meshless_seed_points(&geo, H, H, 1.2, domain);
    let n = seeds0.len();
    let tol = MeshgenTolerances::from_geometry(H, domain);
    let spec_r = boundary_spec_f32(&spec);

    let mut dev = DeviceRegen::new(&ctx, n, domain, &tol, kinds.clone(), spec_r);

    // Initial device mesh at t=0 (old == new ⇒ zero swept, canonical vols).
    let s0 = seeds_at(&seeds0, &kinds, 0.0);
    let (mesh0, _f0, needs0) = dev.regen(&ctx, &s0, &s0, DT);
    assert_eq!(needs0, 0, "initial regen flagged {needs0} needs_cpu faces");
    assert_eq!(mesh0.num_cells(), n);

    // --- DIAGNOSTIC: discrete divergence of a uniform field per cell. --------
    // Free-stream preservation REQUIRES Σ_f σ(i,f)·area_f·normal_f = 0 per cell
    // (a closed polygon in the solver's face geometry). If this is not ~0 the
    // device face areas/normals are inconsistent and div(U0) ≠ 0 seeds momentum.
    {
        let mut max_divx = 0.0f64;
        let mut max_divy = 0.0f64;
        for i in 0..n {
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            let (fb, fe) = (mesh0.cell_face_offsets[i], mesh0.cell_face_offsets[i + 1]);
            for &f in &mesh0.cell_faces[fb..fe] {
                let sgn = if mesh0.face_owner[f] == i { 1.0 } else { -1.0 };
                sx += sgn * mesh0.face_area[f] * mesh0.face_nx[f];
                sy += sgn * mesh0.face_area[f] * mesh0.face_ny[f];
            }
            max_divx = max_divx.max(sx.abs());
            max_divy = max_divy.max(sy.abs());
        }
        // Free-stream needs div(U0) = 0 discretely (closed polygons).
        assert!(
            max_divx < 1e-5 && max_divy < 1e-5,
            "device mesh div(uniform) not ~0: ({max_divx:.3e}, {max_divy:.3e}) — face geometry inconsistent"
        );
        // Boundary-tag census (a missing Outlet ⇒ ungauged pressure ⇒ blow-up).
        let (mut nin, mut nout, mut nslip, mut nuntag) = (0, 0, 0, 0);
        for f in 0..mesh0.num_faces() {
            if mesh0.face_neighbor[f].is_some() { continue; }
            match mesh0.face_boundary[f] {
                Some(BoundaryType::Inlet) => nin += 1,
                Some(BoundaryType::Outlet) => nout += 1,
                Some(BoundaryType::SlipWall) => nslip += 1,
                None => nuntag += 1,
                _ => {}
            }
        }
        eprintln!("[gpu-device-regen] boundary faces: {nin} Inlet + {nout} Outlet + {nslip} SlipWall + {nuntag} untagged; div(U0)~({max_divx:.1e},{max_divy:.1e})");
        assert!(nin > 0 && nout > 0 && nslip > 0 && nuntag == 0, "boundary tagging incomplete");
    }

    // The ACTUAL solver GCL between two consecutive device regens: the solver's
    // V^n = mesh0.cell_vol (@ s0); V^{n+1} = the next regen (@ s1). GCL needs
    // Σσ·flux·dt == V^{n+1}-V^n per cell — the device swept must telescope to the
    // volume change the solver actually sees across two INDEPENDENT regens.
    {
        let s1 = seeds_at(&seeds0, &kinds, DT as f64);
        let (mesh1, fluxes1, _n1) = dev.regen(&ctx, &s1, &s0, DT);
        let mut max_gcl = 0.0f64;
        for i in 0..n {
            let (fb, fe) = (mesh1.cell_face_offsets[i], mesh1.cell_face_offsets[i + 1]);
            let mut s = 0.0f64;
            for &f in &mesh1.cell_faces[fb..fe] {
                let sgn = if mesh1.face_owner[f] == i { 1.0 } else { -1.0 };
                s += sgn * fluxes1[f] as f64 * DT as f64;
            }
            max_gcl = max_gcl.max((s - (mesh1.cell_vol[i] - mesh0.cell_vol[i])).abs());
        }
        eprintln!("[gpu-device-regen] cross-regen solver GCL: max|Σσ·flux·dt - ΔV| = {max_gcl:.3e}");
        assert!(max_gcl < 1e-6, "cross-regen GCL defect {max_gcl:.3e} — swept ≠ solver ΔV");
    }

    let params = test_params();
    let build = pollster::block_on(SolverDriver::build(
        &mesh0,
        allmach_pressure_ale_model().expect("allmach ale model"),
        &params,
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("solver build on device mesh");
    let mut driver = build.driver;
    driver.apply_params(&params);

    let layout = driver.solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U") as usize;
    let p_off = layout.offset_for("p").expect("p") as usize;
    let rho_off = layout.offset_for("rho").expect("rho") as usize;

    let mut prev = s0;
    let (mut max_du, mut max_dp, mut max_drho, mut late_du) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for step in 0..STEPS {
        let t = (step + 1) as f64 * DT as f64;
        let new_seeds = seeds_at(&seeds0, &kinds, t);
        driver.set_requested_dt(DT);
        let dt_f = driver.params().requested_dt;

        let (mesh, fluxes, needs) = dev.regen(&ctx, &new_seeds, &prev, dt_f);
        assert_eq!(needs, 0, "step {step}: {needs} needs_cpu faces (sliver fallback not wired)");
        assert_eq!(mesh.num_cells(), n, "step {step}: cell count changed");

        let report = driver
            .begin_ale_step_topology(&mesh, &fluxes)
            .unwrap_or_else(|e| panic!("step {step} begin_ale_step_topology: {e}"));
        // The topology refresh re-scatters BC tables from the model defaults,
        // dropping runtime per-face overrides (e.g. the inlet velocity vector) —
        // reapply them, exactly as MovingMeshDriver::step does.
        if report.bc_overrides_reset {
            driver.reapply_boundary_conditions();
        }
        let outcome = driver.step(false);
        assert!(outcome.diverged.is_none(), "step {step} diverged: {:?}", outcome.diverged);

        let state = pollster::block_on(driver.solver().read_state_f32());
        let (mut du, mut dp, mut drho) = (0.0f32, 0.0f32, 0.0f32);
        for c in 0..n {
            du = du
                .max((state[c * stride + u_off] - U0.0).abs())
                .max((state[c * stride + u_off + 1] - U0.1).abs());
            dp = dp.max(state[c * stride + p_off].abs());
            drho = drho.max((state[c * stride + rho_off] - RHO_REF).abs());
        }
        assert!(du.is_finite() && dp.is_finite(), "step {step}: non-finite state");
        max_du = max_du.max(du);
        max_dp = max_dp.max(dp);
        max_drho = max_drho.max(drho);
        if step >= 3 * STEPS / 4 {
            late_du = late_du.max(du);
        }
        prev = new_seeds;
    }

    println!(
        "[gpu-device-regen] freestream GCL over {STEPS} steps (n={n}, ENTIRELY on-device regen): \
         max|U-U0| = {max_du:.3e} (late {late_du:.3e}), max|p| = {max_dp:.3e}, \
         max|rho-rho_ref| = {max_drho:.3e}"
    );
    // Free-stream preservation. The fully on-device regen holds |U-U0| ~ 1.6e-5,
    // matching the GPU-backend CPU-mesh reference (1.4e-5); cap at ~6× for f32
    // adapter variation. A gross regression (e.g. dropping the BC reapply after
    // the topology refresh) blows this to O(1).
    assert!(max_du < 1e-4, "free-stream U drift {max_du:.3e} — device regen not GCL-consistent");
    assert!(max_dp < 1e-3, "free-stream p drift {max_dp:.3e}");
    assert!(max_drho < 1e-4, "free-stream rho drift {max_drho:.3e}");
}

/// POLYLINE-boundary production path (the UI geometries): a channel with an
/// embedded obstacle loop — its boundary faces are polyline SEGMENTS, not box
/// sides — driven by `MovingMeshDriver` with `set_gpu_regen(true)`. Gates:
/// (1) every step succeeds (pre-port, the device swept flagged every segment
/// face `needs_cpu` and the driver errored out); (2) the device path actually
/// runs (`RegenBackend::GpuOnDevice` steps observed — a per-step fallback that
/// silently ate every step would pass (1) while running nothing on device);
/// (3) the on-device SCL defect stays at the f32 floor on every device step;
/// (4) any fallback step (Voronoi flip near the obstacle ring) also succeeds —
/// exercising the vertex-less t^n re-assembly inside the CPU fallback.
#[test]
fn movingmesh_driver_gpu_regen_obstacle_polyline() {
    use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
    use cfd2::meshgen::{ChannelWithObstacle, LloydConfig};
    use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RegenBackend};
    let Some(ctx) = gpu_context() else { return };
    let geo = ChannelWithObstacle {
        length: LX,
        height: LY,
        obstacle_center: Point2::new(0.5 * LX, 0.5 * LY),
        obstacle_radius: 0.18,
    };
    let domain = Vector2::new(LX, LY);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_channel(&mut cvt.mesh); // box sides; obstacle faces keep the engine's Wall tag
    let n = cvt.mesh.num_cells();
    let params = test_params();
    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        allmach_pressure_ale_model().expect("model"),
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("gpu moving driver (obstacle)");
    moving.set_gpu_regen(true);
    assert!(moving.gpu_regen_active(), "gpu_regen should be active on the GPU backend");
    moving.set_boundary_retag(Some(tag_channel));
    moving.driver_mut().apply_params(&params);

    let (mut gpu_steps, mut fallback_steps) = (0usize, 0usize);
    let mut max_scl_gpu = 0.0f64;
    for step in 0..STEPS {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("obstacle gpu_regen step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        match stats.regen_backend {
            RegenBackend::GpuOnDevice => {
                gpu_steps += 1;
                max_scl_gpu = max_scl_gpu.max(stats.scl_defect);
                assert!(
                    stats.scl_defect < 5e-6,
                    "step {step}: on-device SCL defect {:.3e} above the f32 floor",
                    stats.scl_defect
                );
            }
            RegenBackend::GpuFallback(reason) => {
                fallback_steps += 1;
                eprintln!("[gpu-regen-obstacle]   step {step}: CPU fallback ({reason})");
            }
            other => panic!("step {step}: unexpected regen backend {other:?}"),
        }
    }
    println!(
        "[gpu-regen-obstacle] polyline-boundary driver run: {STEPS} steps (n={n}), \
         {gpu_steps} on-device / {fallback_steps} CPU-fallback, \
         max on-device SCL defect = {max_scl_gpu:.3e}"
    );
    assert!(
        gpu_steps > 0,
        "no step ran on device — the polyline swept port is not being exercised"
    );
    assert!(
        gpu_steps >= STEPS / 2,
        "only {gpu_steps}/{STEPS} steps ran on device — the gentle swirl should not \
         flip the diagram most steps"
    );
}

/// Deterministic vertex-less-fallback gate: one on-device step commits a
/// VERTEX-LESS mesh; the next step is forced onto the CPU path (toggle
/// `set_gpu_regen(false)`), which must re-assemble the t^n mesh from the
/// authoritative seeds (keeping the solver-held volumes) before the swept
/// alignment — the exact path a mid-run flip fallback takes after a device
/// step. Pre-fix, the CPU path indexed the empty `cell_vertex_offsets` /
/// aligned zero vertices and blew the telescoping identity.
#[test]
fn movingmesh_driver_cpu_step_after_device_step() {
    use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
    use cfd2::meshgen::LloydConfig;
    use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RegenBackend};
    let Some(ctx) = gpu_context() else { return };
    let geo = RectangularChannel { length: LX, height: LY };
    let domain = Vector2::new(LX, LY);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();
    let params = test_params();
    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        allmach_pressure_ale_model().expect("model"),
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("gpu moving driver");
    moving.set_boundary_retag(Some(tag_channel));
    moving.driver_mut().apply_params(&params);

    // Step 1 on device: commits a vertex-less mesh.
    moving.set_gpu_regen(true);
    let (_, s1) = moving.step(false).expect("device step");
    assert_eq!(s1.regen_backend, RegenBackend::GpuOnDevice, "step 1 should run on device");
    assert!(moving.mesh().vx.is_empty(), "device mesh should be vertex-less");

    // Step 2 on CPU: must rebuild the t^n mesh internally (no panic, identity
    // intact, finite stats) and re-commit a vertexed mesh.
    moving.set_gpu_regen(false);
    let (outcome, s2) = moving.step(false).expect("cpu step after device step");
    assert!(outcome.diverged.is_none(), "cpu step diverged");
    assert_eq!(s2.regen_backend, RegenBackend::Cpu);
    assert!(!moving.mesh().vx.is_empty(), "cpu step should commit a vertexed mesh");
    assert!(
        s2.identity_err < 1e-8 || s2.flipped,
        "telescoping identity {:.3e} blown after the vertex-less rebuild",
        s2.identity_err
    );
    assert!(s2.scl_defect < 5e-6, "post-rebuild SCL defect {:.3e}", s2.scl_defect);

    // Step 3 back on device: the flip discriminator must compare against the
    // CPU-committed mesh (derived fresh), not a stale cache.
    moving.set_gpu_regen(true);
    let (_, s3) = moving.step(false).expect("device step after cpu step");
    assert!(
        matches!(s3.regen_backend, RegenBackend::GpuOnDevice | RegenBackend::GpuFallback(_)),
        "step 3 should attempt the device path (got {:?})",
        s3.regen_backend
    );
}
