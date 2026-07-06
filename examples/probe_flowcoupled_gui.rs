//! Headless reproduction of the GUI's moving-mesh ChannelObstacle case
//! (FlowCoupled, CPU backend, GUI incompressible defaults) — diagnoses the
//! "no meaningful solution" report: spurious outlet-band velocities cap the
//! FlowCoupled dt and freeze the run.
//!
//! ```sh
//! CFD2_BACKEND=cpu CFD2_CPU_ENGINE=transpiled \
//!   cargo run --release --features "meshgen cpu" --example probe_flowcoupled_gui
//! ```
#[cfg(all(feature = "meshgen", feature = "cpu"))]
fn main() {
    probe::run();
}

#[cfg(not(all(feature = "meshgen", feature = "cpu")))]
fn main() {
    eprintln!("probe_flowcoupled_gui requires --features \"meshgen cpu\"");
}

#[cfg(all(feature = "meshgen", feature = "cpu"))]
mod probe {
    use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
    use cfd2::meshgen::{ChannelWithObstacle, LloydConfig};
    use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
    use cfd2::solver::model::eos::EosSpec;
    use cfd2::solver::scheme::Scheme;
    use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
    use nalgebra::{Point2, Vector2};

    const LX: f64 = 3.0;
    const LY: f64 = 1.0;
    const H: f64 = 0.025;
    const INLET: f32 = 0.011;
    const STEPS: usize = 300;

    fn gui_params() -> RuntimeParams {
        RuntimeParams {
            adaptive_dt: false,
            target_cfl: 0.9,
            requested_dt: 0.02,
            dtau: 0.0,
            log_convergence: false,
            log_every_steps: 100_000,
            advection_scheme: Scheme::SecondOrderUpwindVanLeer,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            outer_iters: 8,
            outer_auto_converge: true,
            low_mach_model: GpuLowMachPrecondModel::Off,
            low_mach_theta_floor: 1e-6,
            low_mach_pressure_coupling_alpha: 1.0,
            alpha_u: 0.7,
            alpha_p: 0.3,
            inlet_velocity: INLET,
            density: 1.225,
            viscosity: 1.81e-5,
            eos: EosSpec::Constant,
            compressibility_psi: 0.0,
            outlet_back_pressure: 0.0,
            pressure_inlet: false,
            inlet_pressure: 0.0,
        }
    }

    fn run_case(label: &str, motion: MeshMotionSpec) {
        run_case_steps(label, motion, STEPS);
    }

    fn run_case_steps(label: &str, motion: MeshMotionSpec, steps: usize) {
        let domain = Vector2::new(LX, LY);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();

        // Initial-mesh anatomy: volume extremes + where they live.
        {
            let m = &cvt.mesh;
            let (mut vmin, mut vmax, mut imin, mut imax) = (f64::MAX, 0.0f64, 0, 0);
            for i in 0..n {
                if m.cell_vol[i] < vmin {
                    vmin = m.cell_vol[i];
                    imin = i;
                }
                if m.cell_vol[i] > vmax {
                    vmax = m.cell_vol[i];
                    imax = i;
                }
            }
            let band = (0..n).filter(|&i| m.cell_cx[i] > LX - 2.5 * H).count();
            let band_vmax = (0..n)
                .filter(|&i| m.cell_cx[i] > LX - 2.5 * H)
                .map(|i| m.cell_vol[i])
                .fold(0.0f64, f64::max);
            println!(
                "[{label}] initial mesh: {n} cells, vol [{vmin:.3e} @({:.3},{:.3}), \
                 {vmax:.3e} @({:.3},{:.3})], outlet band (x>{:.3}): {band} cells, \
                 band vmax {band_vmax:.3e}, skew {:.3}",
                m.cell_cx[imin], m.cell_cy[imin], m.cell_cx[imax], m.cell_cy[imax],
                LX - 2.5 * H, m.calculate_max_skewness()
            );
        }

        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt, &params, motion, &initial_u, &initial_p, None, None,
        ))
        .expect("driver build");
        if let Some(n_reorder) = std::env::var("PROBE_REORDER").ok().and_then(|s| s.parse().ok()) {
            moving.set_reorder_every_n(n_reorder);
        }
        moving.driver_mut().apply_params(&params);

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;

        let report_every = (steps / 15).max(20);
        let mut total_recycled = 0usize;
        for step in 0..steps {
            let (outcome, stats) = match moving.step(false) {
                Ok(r) => r,
                Err(e) => {
                    println!("[{label}] step {step}: ERROR {e}");
                    return;
                }
            };
            if let Some(reason) = &outcome.diverged {
                println!("[{label}] step {step}: DIVERGED {reason:?}");
                return;
            }
            total_recycled += stats.recycled;
            if step % report_every == 0 || step == steps - 1 {
                let state =
                    pollster::block_on(moving.driver().solver().read_state_f32());
                let mesh = moving.mesh();
                let (mut umax, mut iu) = (0.0f32, 0usize);
                let mut umax_interior = 0.0f32;
                for c in 0..n {
                    let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                    let m = (ux * ux + uy * uy).sqrt();
                    if m > umax {
                        umax = m;
                        iu = c;
                    }
                    if mesh.cell_cx[c] < LX - 2.5 * H {
                        umax_interior = umax_interior.max(m);
                    }
                }
                let (mut vmin, mut ivm) = (f64::MAX, 0usize);
                for c in 0..n {
                    if mesh.cell_vol[c] < vmin {
                        vmin = mesh.cell_vol[c];
                        ivm = c;
                    }
                }
                let vmax = mesh.cell_vol.iter().cloned().fold(0.0f64, f64::max);
                println!(
                    "[{label}] step {step:3}: dt={:.2e} |U|max={umax:.3e} \
                     @({:.3},{:.3}) vol@argmax={:.2e} |U|max_interior={umax_interior:.3e} \
                     vol_min={vmin:.2e} @({:.3},{:.3}) vol_max={vmax:.2e} skew={:.3} \
                     flip={} SCL={:.1e} recycled_total={total_recycled} locality={:.1}",
                    stats.dt, mesh.cell_cx[iu], mesh.cell_cy[iu], mesh.cell_vol[iu],
                    mesh.cell_cx[ivm], mesh.cell_cy[ivm],
                    stats.max_skew, stats.flipped, stats.scl_defect,
                    locality(mesh),
                );
            }
        }
    }

    /// Flow-adaptive sizing long-run probe (arc d): a GRADED obstacle channel
    /// under FlowCoupled with `set_adaptive_sizing(every_n)` — reports the
    /// cell count trajectory, cumulative births/kills, and the mean cell
    /// volume near the obstacle+wake vs the far field (adaptation should
    /// steer volume DOWN where the gradients live). Cell count changes at
    /// runtime, so every per-cell scan re-derives `n` from the current mesh.
    fn run_adapt(steps: usize, every_n: usize) {
        let domain = Vector2::new(LX, LY);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };
        // Graded 0.025→0.05: a wide realized volume band gives the indicator
        // room in both directions.
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, H, 2.0 * H, 1.2, domain, &LloydConfig::default());
        let n0 = cvt.mesh.num_cells();
        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n0];
        let initial_p = vec![0.0; n0];
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::FlowCoupled { regularization: 0.5 },
            &initial_u,
            &initial_p,
            None,
            None,
        ))
        .expect("adapt driver build");
        moving.driver_mut().apply_params(&params);
        moving.set_adaptive_sizing(every_n);

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        // "Near" = within 2.5 obstacle radii of the obstacle centre or its
        // 1-diameter wake box; everything else is "far".
        let near = |x: f64, y: f64| -> bool {
            let (dx, dy) = (x - 1.0, y - 0.51);
            (dx * dx + dy * dy).sqrt() < 0.25 || (x > 1.0 && x < 1.8 && (y - 0.51).abs() < 0.2)
        };
        let report_every = (steps / 20).max(10);
        let (mut born, mut killed, mut recycled) = (0usize, 0usize, 0usize);
        for step in 0..steps {
            let (outcome, stats) = match moving.step(false) {
                Ok(r) => r,
                Err(e) => {
                    println!("[adapt] step {step}: ERROR {e}");
                    return;
                }
            };
            if let Some(reason) = &outcome.diverged {
                println!("[adapt] step {step}: DIVERGED {reason:?}");
                return;
            }
            born += stats.cells_born;
            killed += stats.cells_killed;
            recycled += stats.recycled;
            if step % report_every == 0 || step == steps - 1 || stats.cells_born > 0 {
                let mesh = moving.mesh();
                let n = mesh.num_cells();
                let state = pollster::block_on(moving.driver().solver().read_state_f32());
                let mut umax = 0.0f32;
                let (mut v_near, mut c_near, mut v_far, mut c_far) = (0.0f64, 0usize, 0.0f64, 0usize);
                for c in 0..n {
                    let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                    umax = umax.max((ux * ux + uy * uy).sqrt());
                    if near(mesh.cell_cx[c], mesh.cell_cy[c]) {
                        v_near += mesh.cell_vol[c];
                        c_near += 1;
                    } else {
                        v_far += mesh.cell_vol[c];
                        c_far += 1;
                    }
                }
                println!(
                    "[adapt] step {step:4}: n={n} (+{born}/−{killed}, recycled {recycled}) \
                     dt={:.2e} |U|max={umax:.3e} mean_vol near/far = {:.3e}/{:.3e} ({:.2}x) \
                     skew={:.3}",
                    stats.dt,
                    v_near / c_near.max(1) as f64,
                    v_far / c_far.max(1) as f64,
                    (v_far / c_far.max(1) as f64) / (v_near / c_near.max(1) as f64),
                    stats.max_skew,
                );
            }
        }
        println!(
            "[adapt] done: {n0} → {} cells, born={born} killed={killed} recycled={recycled}",
            moving.mesh().num_cells()
        );
    }

    /// Static reference: a plain `SolverDriver` on the SAME CVT mesh (no
    /// MovingMeshDriver, no ALE seam) — separates "base solver on this mesh"
    /// from "ALE driver machinery". `obstacle=false` runs a plain rectangular
    /// CVT channel (minimal repro candidate).
    fn run_static(label: &str, scheme: Scheme, obstacle: bool, steps: usize) {
        use cfd2::sim::SolverDriver;
        use cfd2::solver::mesh::RectangularChannel;
        let domain = Vector2::new(LX, LY);
        let cvt = if obstacle {
            let geo = ChannelWithObstacle {
                length: LX,
                height: LY,
                obstacle_center: Point2::new(1.0, 0.51),
                obstacle_radius: 0.1,
            };
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default())
        } else {
            let geo = RectangularChannel { length: LX, height: LY };
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default())
        };
        let mesh = cvt.mesh;
        let n = mesh.num_cells();
        let mut params = gui_params();
        params.advection_scheme = scheme;
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let build = pollster::block_on(SolverDriver::build(
            &mesh,
            cfd2::solver::model::incompressible_momentum_model().expect("model"),
            &params,
            &initial_u,
            &initial_p,
            None,
            None,
        ))
        .expect("static driver build");
        let mut driver = build.driver;
        driver.apply_params(&params);
        let layout = driver.solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        for step in 0..steps {
            let outcome = driver.step(false);
            if let Some(reason) = &outcome.diverged {
                println!("[{label}] step {step}: DIVERGED {reason:?}");
                return;
            }
            if step % 20 == 0 || step == steps - 1 {
                let state = pollster::block_on(driver.solver().read_state_f32());
                let (mut umax, mut iu) = (0.0f32, 0usize);
                let mut umax_interior = 0.0f32;
                for c in 0..n {
                    let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                    let m = (ux * ux + uy * uy).sqrt();
                    if m > umax {
                        umax = m;
                        iu = c;
                    }
                    if mesh.cell_cx[c] < LX - 2.5 * H {
                        umax_interior = umax_interior.max(m);
                    }
                }
                println!(
                    "[{label}] step {step:3}: |U|max={umax:.3e} @({:.3},{:.3}) \
                     |U|max_interior={umax_interior:.3e}",
                    mesh.cell_cx[iu], mesh.cell_cy[iu],
                );
            }
        }
    }

    /// ALE passthrough: MovingMeshDriver with regen disabled — the pure
    /// `SolverDriver::step` loop through the ALE model, no seam churn.
    fn run_passthrough(label: &str) {
        let domain = Vector2::new(LX, LY);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();
        let mesh = cvt.mesh.clone();
        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt, &params, MeshMotionSpec::Frozen, &initial_u, &initial_p, None, None,
        ))
        .expect("driver build");
        moving.driver_mut().apply_params(&params);
        moving.set_regen_each_step(false);
        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        for step in 0..STEPS {
            let (outcome, _stats) = match moving.step(false) {
                Ok(r) => r,
                Err(e) => {
                    println!("[{label}] step {step}: ERROR {e}");
                    return;
                }
            };
            if let Some(reason) = &outcome.diverged {
                println!("[{label}] step {step}: DIVERGED {reason:?}");
                return;
            }
            if step % 20 == 0 || step == STEPS - 1 {
                let state = pollster::block_on(moving.driver().solver().read_state_f32());
                let (mut umax, mut iu) = (0.0f32, 0usize);
                let mut umax_interior = 0.0f32;
                for c in 0..n {
                    let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                    let m = (ux * ux + uy * uy).sqrt();
                    if m > umax {
                        umax = m;
                        iu = c;
                    }
                    if mesh.cell_cx[c] < LX - 2.5 * H {
                        umax_interior = umax_interior.max(m);
                    }
                }
                println!(
                    "[{label}] step {step:3}: |U|max={umax:.3e} @({:.3},{:.3}) \
                     |U|max_interior={umax_interior:.3e}",
                    mesh.cell_cx[iu], mesh.cell_cy[iu],
                );
            }
        }
    }

    /// Mode-shape dump: run the static obstacle case and print every
    /// outlet-band cell's (x, y, ux, uy, p) at a late step, sorted by y — a
    /// checkerboard (sign-alternating in y) vs smooth profile discriminates
    /// the instability mechanism.
    fn run_mode_dump(dump_step: usize) {
        use cfd2::sim::SolverDriver;
        let domain = Vector2::new(LX, LY);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default());
        let mesh = cvt.mesh;
        let n = mesh.num_cells();
        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let build = pollster::block_on(SolverDriver::build(
            &mesh,
            cfd2::solver::model::incompressible_momentum_model().expect("model"),
            &params,
            &initial_u,
            &initial_p,
            None,
            None,
        ))
        .expect("static driver build");
        let mut driver = build.driver;
        driver.apply_params(&params);
        let layout = driver.solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        let p_off = layout.offset_for("p").expect("p") as usize;
        for _ in 0..dump_step {
            let outcome = driver.step(false);
            if outcome.diverged.is_some() {
                break;
            }
        }
        let state = pollster::block_on(driver.solver().read_state_f32());
        // Outlet band = the boundary half-cells (owner of an Outlet face).
        let mut band: Vec<usize> = (0..mesh.num_faces())
            .filter(|&f| {
                mesh.face_neighbor[f].is_none()
                    && matches!(
                        mesh.face_boundary[f],
                        Some(cfd2::solver::mesh::BoundaryType::Outlet)
                    )
            })
            .map(|f| mesh.face_owner[f])
            .collect();
        band.sort_unstable();
        band.dedup();
        band.sort_by(|&a, &b| mesh.cell_cy[a].total_cmp(&mesh.cell_cy[b]));
        println!("[mode-dump] step {dump_step}: outlet-band cells (sorted by y):");
        for &c in &band {
            println!(
                "[mode-dump]   y={:7.4} x={:7.4} vol={:.2e}  ux={:+.4e} uy={:+.4e} p={:+.4e}",
                mesh.cell_cy[c],
                mesh.cell_cx[c],
                mesh.cell_vol[c],
                state[c * stride + u_off],
                state[c * stride + u_off + 1],
                state[c * stride + p_off],
            );
        }
        // Second column (interior neighbors of band cells) summary.
        let band_set: std::collections::HashSet<usize> = band.iter().cloned().collect();
        let mut col2: Vec<usize> = Vec::new();
        for f in 0..mesh.num_faces() {
            if let Some(nb) = mesh.face_neighbor[f] {
                let o = mesh.face_owner[f];
                if band_set.contains(&o) && !band_set.contains(&nb) {
                    col2.push(nb);
                }
                if band_set.contains(&nb) && !band_set.contains(&o) {
                    col2.push(o);
                }
            }
        }
        col2.sort_unstable();
        col2.dedup();
        let c2max = col2
            .iter()
            .map(|&c| {
                let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                (ux * ux + uy * uy).sqrt()
            })
            .fold(0.0f32, f32::max);
        println!("[mode-dump] second-column max|U| = {c2max:.3e} over {} cells", col2.len());
    }

    /// Quantify the RECYCLE transient at the outlet: on steps where seeds
    /// recycle (and the two steps after), print the outlet-strip |U| / p
    /// extremes vs the running ambient — separates a real solver spike (the
    /// old neighbors absorb the dying cell's area in one step) from the
    /// rendering-cadence artifact (stale polygon painted with the slot's new
    /// inlet state).
    fn run_recycle_spike(steps: usize) {
        let domain = Vector2::new(LX, LY);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();
        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::FlowCoupled { regularization: 0.5 },
            &initial_u,
            &initial_p,
            None,
            None,
        ))
        .expect("driver build");
        moving.driver_mut().apply_params(&params);
        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        let p_off = layout.offset_for("p").expect("p") as usize;

        let strip_x = LX - 4.0 * H;
        let mut watch = 0usize; // steps left to report after a recycle
        let (mut peak_u, mut peak_p) = (0.0f32, 0.0f32);
        for step in 0..steps {
            let (outcome, stats) = moving.step(false).expect("step");
            assert!(outcome.diverged.is_none(), "diverged at step {step}");
            if stats.recycled > 0 {
                watch = 3;
            }
            if watch > 0 || step % 100 == 0 {
                let state = pollster::block_on(moving.driver().solver().read_state_f32());
                let mesh = moving.mesh();
                let (mut umax, mut pmin, mut pmax) = (0.0f32, f32::MAX, f32::MIN);
                for c in 0..n {
                    if mesh.cell_cx[c] < strip_x {
                        continue;
                    }
                    let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                    umax = umax.max((ux * ux + uy * uy).sqrt());
                    let p = state[c * stride + p_off];
                    pmin = pmin.min(p);
                    pmax = pmax.max(p);
                }
                let tag = if stats.recycled > 0 { "RECYCLE" } else if watch > 0 { "after  " } else { "ambient" };
                println!(
                    "[recycle-spike] step {step:4} {tag} n_rec={} strip |U|max={umax:.3e} \
                     p=[{pmin:+.3e},{pmax:+.3e}]",
                    stats.recycled
                );
                if stats.recycled > 0 || watch == 3 {
                    peak_u = peak_u.max(umax);
                    peak_p = peak_p.max(pmax.abs().max(pmin.abs()));
                }
                watch = watch.saturating_sub(1);
            }
        }
        println!(
            "[recycle-spike] PEAK on recycle steps: |U|={peak_u:.3e} (inlet {INLET:.3e}), \
             |p|={peak_p:.3e}"
        );
    }

    /// STABILITY MATRIX: the `visual` case exposed a SLOW blow-up of the
    /// every-step-adaptation + every-step-smoothing + recycling regime at
    /// GUI params (healthy at step 100, |U| 5x inlet by 400, diverged by
    /// 1200, churn saturated at the 8+8 cap). Isolate the driver: run the
    /// ingredient combinations at a coarser mesh and print the |U|max
    /// trajectory + churn/recycle rates for each.
    fn run_stability_matrix(steps: usize) {
        struct Variant {
            label: &'static str,
            adapt_every: usize,
            band: Option<(f64, f64)>,
            smooth_every: usize,
            recycling: bool,
            projection: bool,
        }
        let variants = [
            Variant { label: "A adapt1+smooth1+recycle (fail cfg)", adapt_every: 1, band: None, smooth_every: 1, recycling: true, projection: true },
            Variant { label: "B adapt1+smooth1 no-recycle      ", adapt_every: 1, band: None, smooth_every: 1, recycling: false, projection: true },
            Variant { label: "C adapt5+smooth1+recycle         ", adapt_every: 5, band: None, smooth_every: 1, recycling: true, projection: true },
            Variant { label: "D adapt1 WIDE band+smooth1+recycle", adapt_every: 1, band: Some((0.03, 0.07)), smooth_every: 1, recycling: true, projection: true },
            Variant { label: "E smooth1+recycle no-adapt       ", adapt_every: 0, band: None, smooth_every: 1, recycling: true, projection: true },
            Variant { label: "F adapt1+smooth1+recycle proj-OFF", adapt_every: 1, band: None, smooth_every: 1, recycling: true, projection: false },
        ];
        let hh = 0.035;
        for v in &variants {
            let domain = Vector2::new(LX, LY);
            let geo = ChannelWithObstacle {
                length: LX,
                height: LY,
                obstacle_center: Point2::new(1.0, 0.51),
                obstacle_radius: 0.1,
            };
            let cvt =
                generate_cvt_mesh_with_seeds(&geo, hh, hh, 1.2, domain, &LloydConfig::default());
            let n0 = cvt.mesh.num_cells();
            let params = gui_params();
            let mut moving = pollster::block_on(MovingMeshDriver::build(
                cvt,
                &params,
                MeshMotionSpec::FlowCoupled {
                    regularization: 0.5,
                },
                &vec![(INLET as f64, 0.0); n0],
                &vec![0.0; n0],
                None,
                None,
            ))
            .expect("driver build");
            moving.driver_mut().apply_params(&params);
            moving.set_transfer_projection(v.projection);
            moving.set_seed_recycling(v.recycling);
            if v.adapt_every > 0 {
                moving.set_adaptive_sizing(v.adapt_every);
                if let Some(band) = v.band {
                    moving.set_adaptive_sizing_band(Some(band));
                }
            }
            if v.smooth_every > 0 {
                moving.set_smoothing(v.smooth_every, 1, 0.5);
            }

            let layout = moving.driver().solver().model().state_layout.clone();
            let stride = layout.stride() as usize;
            let u_off = layout.offset_for("U").expect("U") as usize;
            let (mut births, mut kills, mut recycles) = (0usize, 0usize, 0usize);
            let mut track: Vec<(usize, f64)> = Vec::new();
            let mut died = None;
            for step in 0..steps {
                let (outcome, stats) = match moving.step(false) {
                    Ok(x) => x,
                    Err(e) => {
                        println!("[matrix] {} step {step}: ERR {e}", v.label);
                        died = Some(step);
                        break;
                    }
                };
                if outcome.diverged.is_some() {
                    died = Some(step);
                    break;
                }
                births += stats.cells_born;
                kills += stats.cells_killed;
                recycles += usize::from(stats.recycled > 0);
                if step % 100 == 99 || step + 1 == steps {
                    let state = pollster::block_on(moving.driver().solver().read_state_f32());
                    let n = moving.mesh().num_cells();
                    let umax = (0..n)
                        .map(|c| {
                            (state[c * stride + u_off] as f64)
                                .hypot(state[c * stride + u_off + 1] as f64)
                        })
                        .fold(0.0f64, f64::max);
                    track.push((step + 1, umax));
                }
            }
            let traj: Vec<String> = track
                .iter()
                .map(|(s, u)| format!("{s}:{u:.2e}"))
                .collect();
            println!(
                "[matrix] {} | umax@[{}] births {births} kills {kills} recycle-steps {recycles}{}",
                v.label,
                traj.join(", "),
                died.map(|s| format!(" DIED@{s}")).unwrap_or_default()
            );
        }
    }

    /// VISUAL verification: run the GUI screenshot regime (FlowCoupled,
    /// adaptation + smoothing every step, recycling active) and render
    /// pressure / velocity-magnitude / cell-size maps to PNGs at snapshot
    /// steps — metrics don't show everything (per-parcel speckle, mesh
    /// churn, and inlet/outlet band artifacts are pattern problems). Also
    /// prints LATE-half strip metrics (past the impulsive transient) so the
    /// numbers aren't cold-start-contaminated.
    fn run_visual(steps: usize, snaps: &[usize]) {
        let family = std::env::var("PROBE_VISUAL_FAMILY").unwrap_or_else(|_| "incomp".into());
        let out_dir_owned = format!("target/probe_visual_{family}");
        let out_dir = std::path::Path::new(&out_dir_owned);
        std::fs::create_dir_all(out_dir).expect("mkdir probe_visual");
        let domain = Vector2::new(LX, LY);
        let (ocx, ocy, orad) = (1.0f64, 0.51f64, 0.1f64);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(ocx, ocy),
            obstacle_radius: orad,
        };
        let cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default());
        let n0 = cvt.mesh.num_cells();
        // PROBE_VISUAL_FAMILY=allmach runs the thermal all-Mach ALE family
        // (the GUI screenshot regime) with its proven recipe knobs; default
        // is the incompressible GUI config.
        let thermal = std::env::var("PROBE_VISUAL_FAMILY")
            .map(|v| v == "allmach")
            .unwrap_or(false);
        let mut params = gui_params();
        if thermal {
            params.time_scheme = TimeScheme::Euler;
            params.requested_dt = 0.005;
            params.outer_iters = 6;
            params.compressibility_psi = 1.0e-4;
            params.viscosity = 1e-2;
            params.inlet_velocity = 0.4;
        }
        let inlet = params.inlet_velocity;
        let mut moving = if thermal {
            use cfd2::solver::model::allmach_thermal_ale_model;
            pollster::block_on(MovingMeshDriver::build_with_model(
                cvt,
                allmach_thermal_ale_model().expect("thermal ale model"),
                &params,
                MeshMotionSpec::FlowCoupled {
                    regularization: 0.5,
                },
                &vec![(inlet as f64, 0.0); n0],
                &vec![0.0; n0],
                None,
                None,
            ))
            .expect("driver build")
        } else {
            pollster::block_on(MovingMeshDriver::build(
                cvt,
                &params,
                MeshMotionSpec::FlowCoupled {
                    regularization: 0.5,
                },
                &vec![(inlet as f64, 0.0); n0],
                &vec![0.0; n0],
                None,
                None,
            ))
            .expect("driver build")
        };
        moving.driver_mut().apply_params(&params);
        moving.set_adaptive_sizing(1);
        moving.set_smoothing(1, 1, 0.5);

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        let p_off = layout.offset_for("p").expect("p") as usize;

        // Late-half strip telemetry (inlet x<0.5, outlet x>LX-0.5).
        let half = steps / 2;
        let (mut in_jump, mut out_jump) = (0.0f64, 0.0f64);
        let (mut prev_in, mut prev_out) = (0.0f64, 0.0f64);
        let (mut late_births, mut late_kills, mut late_recycles) = (0usize, 0usize, 0usize);
        let mut late_umax = 0.0f64;
        for step in 0..steps {
            let (outcome, stats) = moving.step(false).expect("step");
            assert!(outcome.diverged.is_none(), "[visual] diverged at step {step}");
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            let mesh = moving.mesh();
            let n = mesh.num_cells();
            let (mut ip, mut op, mut um) = (0.0f64, 0.0f64, 0.0f64);
            for c in 0..n {
                let p = (state[c * stride + p_off] as f64).abs();
                let (ux, uy) = (
                    state[c * stride + u_off] as f64,
                    state[c * stride + u_off + 1] as f64,
                );
                um = um.max(ux.hypot(uy));
                if mesh.cell_cx[c] < 0.5 {
                    ip = ip.max(p);
                } else if mesh.cell_cx[c] > LX - 0.5 {
                    op = op.max(p);
                }
            }
            if step >= half {
                in_jump = in_jump.max(ip - prev_in);
                out_jump = out_jump.max(op - prev_out);
                late_births += stats.cells_born;
                late_kills += stats.cells_killed;
                late_recycles += usize::from(stats.recycled > 0);
                late_umax = late_umax.max(um);
            }
            prev_in = ip;
            prev_out = op;

            if snaps.contains(&step) {
                let fields: [(&str, Box<dyn Fn(usize) -> f64>); 3] = [
                    (
                        "p",
                        Box::new(|c| state[c * stride + p_off] as f64),
                    ),
                    (
                        "umag",
                        Box::new(|c| {
                            (state[c * stride + u_off] as f64)
                                .hypot(state[c * stride + u_off + 1] as f64)
                        }),
                    ),
                    ("cellvol", Box::new(|c| mesh.cell_vol[c].ln())),
                ];
                for (name, f) in &fields {
                    render_voronoi_field(
                        mesh,
                        f,
                        (ocx, ocy, orad),
                        &out_dir.join(format!("step{step:04}_{name}.png")),
                    );
                }
                println!(
                    "[visual] step {step:4}: {} cells, snapshot written (p range on file)",
                    n
                );
            }
        }
        println!(
            "[visual] LATE half (steps {half}..{steps}): inlet-strip worst |p| jump {in_jump:.3e}, \
             outlet-strip {out_jump:.3e}, births {late_births} kills {late_kills} over {} steps, \
             recycle steps {late_recycles}, max|U| {late_umax:.3e} (inlet {inlet:.3e})",
            steps - half
        );
    }

    /// Rasterize a per-cell scalar onto a PNG via nearest-centroid lookup
    /// (Voronoi cells are nearest-seed regions; post-Lloyd centroids track
    /// the seeds closely enough for visualization). Blue -> green -> red
    /// over the field's own range; obstacle interior black.
    fn render_voronoi_field(
        mesh: &cfd2::solver::mesh::Mesh,
        value: &dyn Fn(usize) -> f64,
        obstacle: (f64, f64, f64),
        path: &std::path::Path,
    ) {
        let n = mesh.num_cells();
        let (w, h) = (900u32, 300u32);
        let (sx, sy) = (LX / w as f64, LY / h as f64);
        // Bucket grid over centroids for nearest lookup.
        let (bw, bh) = (90usize, 30usize);
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); bw * bh];
        let bidx = |x: f64, y: f64| -> (usize, usize) {
            (
                ((x / LX * bw as f64) as usize).min(bw - 1),
                ((y / LY * bh as f64) as usize).min(bh - 1),
            )
        };
        for c in 0..n {
            let (bx, by) = bidx(mesh.cell_cx[c], mesh.cell_cy[c]);
            buckets[by * bw + bx].push(c);
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for c in 0..n {
            let v = value(c);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let span = (hi - lo).max(1e-300);
        let mut img = image::RgbImage::new(w, h);
        for py in 0..h {
            for px in 0..w {
                let (x, y) = ((px as f64 + 0.5) * sx, LY - (py as f64 + 0.5) * sy);
                if (x - obstacle.0).hypot(y - obstacle.1) < obstacle.2 {
                    img.put_pixel(px, py, image::Rgb([0, 0, 0]));
                    continue;
                }
                let (bx, by) = bidx(x, y);
                let (mut best, mut best_d2) = (usize::MAX, f64::INFINITY);
                for dby in by.saturating_sub(1)..=(by + 1).min(bh - 1) {
                    for dbx in bx.saturating_sub(2)..=(bx + 2).min(bw - 1) {
                        for &c in &buckets[dby * bw + dbx] {
                            let d2 = (mesh.cell_cx[c] - x).powi(2)
                                + (mesh.cell_cy[c] - y).powi(2);
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best = c;
                            }
                        }
                    }
                }
                let rgb = if best == usize::MAX {
                    [255u8, 255, 255]
                } else {
                    let t = ((value(best) - lo) / span).clamp(0.0, 1.0);
                    // blue (0) -> green (0.5) -> red (1)
                    if t < 0.5 {
                        let s = t * 2.0;
                        [0, (s * 255.0) as u8, ((1.0 - s) * 255.0) as u8]
                    } else {
                        let s = (t - 0.5) * 2.0;
                        [(s * 255.0) as u8, ((1.0 - s) * 255.0) as u8, 0]
                    }
                };
                img.put_pixel(px, py, image::Rgb(rgb));
            }
        }
        img.save(path).expect("write png");
        println!(
            "[visual]   {} range [{lo:.4e}, {hi:.4e}]",
            path.file_name().unwrap().to_string_lossy()
        );
    }

    /// A/B the RECYCLE mass-row projection under the GUI screenshot regime
    /// (FlowCoupled + adaptation & smoothing every step): the recycle
    /// re-seed's zeroth-order copy leaves a continuity defect at the INLET
    /// respawn site — visible as per-parcel pressure speckle that the
    /// |grad p| adaptation indicator then chases. Reports, for projection
    /// ON vs OFF: recycle events, the worst inlet-strip |p| jump on
    /// recycle steps, births landing in the inlet strip, and the defect
    /// closure telemetry.
    fn run_recycle_inlet(steps: usize) {
        for projection in [true, false] {
            let tag = if projection { "proj-ON " } else { "proj-OFF" };
            let domain = Vector2::new(LX, LY);
            let geo = ChannelWithObstacle {
                length: LX,
                height: LY,
                obstacle_center: Point2::new(1.0, 0.51),
                obstacle_radius: 0.1,
            };
            let cvt =
                generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default());
            let n0 = cvt.mesh.num_cells();
            let params = gui_params();
            let mut moving = pollster::block_on(MovingMeshDriver::build(
                cvt,
                &params,
                MeshMotionSpec::FlowCoupled {
                    regularization: 0.5,
                },
                &vec![(INLET as f64, 0.0); n0],
                &vec![0.0; n0],
                None,
                None,
            ))
            .expect("driver build");
            moving.driver_mut().apply_params(&params);
            moving.set_transfer_projection(projection);
            // The screenshot regime: adaptation + smoothing EVERY step.
            moving.set_adaptive_sizing(1);
            moving.set_smoothing(1, 1, 0.5);

            let layout = moving.driver().solver().model().state_layout.clone();
            let stride = layout.stride() as usize;
            let p_off = layout.offset_for("p").expect("p") as usize;
            let strip_x = 0.5; // the inlet respawn strip
            let (mut recycles, mut births_in_strip, mut total_births) = (0usize, 0usize, 0usize);
            let (mut worst_jump, mut prev_strip_pmax) = (0.0f64, 0.0f64);
            let (mut defect_pre, mut defect_post) = (0.0f64, 0.0f64);
            for step in 0..steps {
                let n_before = moving.mesh().num_cells();
                let (outcome, stats) = moving.step(false).expect("step");
                assert!(outcome.diverged.is_none(), "[{tag}] diverged at step {step}");
                let state = pollster::block_on(moving.driver().solver().read_state_f32());
                let mesh = moving.mesh();
                let n = mesh.num_cells();
                let strip_pmax = (0..n)
                    .filter(|&c| mesh.cell_cx[c] < strip_x)
                    .map(|c| (state[c * stride + p_off] as f64).abs())
                    .fold(0.0f64, f64::max);
                if stats.recycled > 0 {
                    recycles += 1;
                    if step > 50 {
                        worst_jump = worst_jump.max(strip_pmax - prev_strip_pmax);
                    }
                    defect_pre = defect_pre.max(stats.transfer_defect_pre);
                    defect_post = defect_post.max(stats.transfer_defect_post);
                }
                let _ = n_before;
                if stats.cells_born > 0 {
                    total_births += stats.cells_born;
                    // Births landing in the inlet strip = the indicator
                    // chasing the recycle noise (the resize appends births
                    // at the tail of the index range).
                    for c in n.saturating_sub(stats.cells_born)..n {
                        if mesh.cell_cx[c] < strip_x {
                            births_in_strip += 1;
                        }
                    }
                }
                prev_strip_pmax = strip_pmax;
            }
            println!(
                "[recycle-inlet] {tag}: {recycles} recycle steps, worst inlet-strip |p| jump \
                 {worst_jump:.3e}, births {total_births} (in strip {births_in_strip}), \
                 defect max {defect_pre:.3e} -> {defect_post:.3e}"
            );
        }
    }

    /// Thermal-ALE divergence ladder: the allmach_thermal model on the GUI
    /// obstacle CVT with the PROVEN gate params (Euler, 6 outers, dt 5e-3,
    /// psi 1e-4, inlet 0.4) — static solver vs Frozen-ALE vs FlowCoupled,
    /// each run to divergence or `steps`. Prints the death step.
    fn run_thermal_ladder(steps: usize) {
        use cfd2::sim::SolverDriver;
        use cfd2::solver::model::{allmach_thermal_ale_model, all_models};
        let domain = Vector2::new(LX, LY);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };
        let mut params = gui_params();
        params.time_scheme = TimeScheme::Euler;
        params.outer_iters = 6;
        params.requested_dt = 5e-3;
        params.compressibility_psi = 1e-4;
        params.inlet_velocity = 0.4;
        params.viscosity = 0.01;
        params.density = 1.0;

        // Static (non-ALE) thermal on the same CVT mesh.
        {
            let cvt =
                generate_cvt_mesh_with_seeds(&geo, 0.06, 0.06, 1.0, domain, &LloydConfig::default());
            let mesh = cvt.mesh;
            let n = mesh.num_cells();
            let model = all_models()
                .expect("models")
                .into_iter()
                .find(|m| m.id == "allmach_thermal")
                .expect("allmach_thermal");
            let build = pollster::block_on(SolverDriver::build(
                &mesh,
                model,
                &params,
                &vec![(params.inlet_velocity as f64, 0.0); n],
                &vec![0.0; n],
                None,
                None,
            ))
            .expect("static thermal build");
            let mut driver = build.driver;
            driver.apply_params(&params);
            let mut died = None;
            for step in 0..steps {
                let outcome = driver.step(false);
                if outcome.diverged.is_some() {
                    died = Some(step);
                    break;
                }
            }
            println!("[thermal-ladder] static:       died={died:?} (None = survived {steps})");
        }

        // Frozen-ALE and FlowCoupled thermal.
        for (label, motion) in [
            ("frozen-ale ", MeshMotionSpec::Frozen),
            ("flowcoupled", MeshMotionSpec::FlowCoupled { regularization: 0.5 }),
        ] {
            let cvt =
                generate_cvt_mesh_with_seeds(&geo, 0.06, 0.06, 1.0, domain, &LloydConfig::default());
            let n = cvt.mesh.num_cells();
            let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
                cvt,
                allmach_thermal_ale_model().expect("thermal ale"),
                &params,
                motion,
                &vec![(params.inlet_velocity as f64, 0.0); n],
                &vec![0.0; n],
                None,
                None,
            ))
            .expect("thermal moving build");
            moving.driver_mut().apply_params(&params);
            let mut died = None;
            for step in 0..steps {
                match moving.step(false) {
                    Ok((outcome, _)) => {
                        if outcome.diverged.is_some() {
                            died = Some(step);
                            break;
                        }
                    }
                    Err(e) => {
                        println!("[thermal-ladder] {label}: step {step} ERROR {e}");
                        died = Some(step);
                        break;
                    }
                }
            }
            println!("[thermal-ladder] {label}: died={died:?} (None = survived {steps})");
        }
    }

    /// One long thermal FlowCoupled run with the worker smoke's exact params
    /// (outer_auto_converge OFF, dtau 0) and the given advection scheme.
    fn run_thermal_long(scheme: Scheme, steps: usize) {
        use cfd2::solver::model::allmach_thermal_ale_model;
        let domain = Vector2::new(LX, LY);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };
        let mut params = gui_params();
        params.advection_scheme = scheme;
        params.time_scheme = TimeScheme::Euler;
        params.outer_iters = 6;
        params.outer_auto_converge = false;
        params.requested_dt = 5e-3;
        params.compressibility_psi = 1e-4;
        params.inlet_velocity = 0.4;
        params.viscosity = 0.01;
        params.density = 1.0;
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, 0.06, 0.06, 1.0, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();
        let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
            cvt,
            allmach_thermal_ale_model().expect("thermal ale"),
            &params,
            MeshMotionSpec::FlowCoupled { regularization: 0.5 },
            &vec![(params.inlet_velocity as f64, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("thermal moving build");
        moving.driver_mut().apply_params(&params);
        let mut died = None;
        let mut total_recycled = 0usize;
        for step in 0..steps {
            match moving.step(false) {
                Ok((outcome, stats)) => {
                    total_recycled += stats.recycled;
                    if let Some(r) = &outcome.diverged {
                        println!("[thermal-long] {scheme:?}: DIVERGED {r:?} at step {step}");
                        died = Some(step);
                        break;
                    }
                }
                Err(e) => {
                    println!("[thermal-long] {scheme:?}: step {step} ERROR {e}");
                    died = Some(step);
                    break;
                }
            }
        }
        println!(
            "[thermal-long] {scheme:?}: died={died:?} recycled_total={total_recycled} \
             (None = survived {steps})"
        );
    }

    /// Mean slot distance |owner − neighbor| over interior faces — the memory
    /// locality the solver's gather/scatter feels.
    fn locality(mesh: &cfd2::solver::mesh::Mesh) -> f64 {
        let (mut sum, mut cnt) = (0.0f64, 0usize);
        for f in 0..mesh.num_faces() {
            if let Some(nb) = mesh.face_neighbor[f] {
                sum += (mesh.face_owner[f] as f64 - nb as f64).abs();
                cnt += 1;
            }
        }
        sum / cnt.max(1) as f64
    }

    /// Reorder INVARIANCE gate: a rectangular free stream under FROZEN motion
    /// with periodic Morton reordering — a pure relabel must preserve the
    /// free stream exactly (any permutation bug corrupts it immediately).
    fn run_reorder_frozen(steps: usize, every_n: usize) {
        use cfd2::solver::mesh::{BoundaryType, RectangularChannel};
        let domain = Vector2::new(LX, LY);
        let geo = RectangularChannel { length: LX, height: LY };
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.2, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();
        fn retag(mesh: &mut cfd2::solver::mesh::Mesh) {
            let eps = 1e-4;
            for f in 0..mesh.num_faces() {
                if mesh.face_neighbor[f].is_some() {
                    continue;
                }
                let (x, y) = (mesh.face_cx[f], mesh.face_cy[f]);
                mesh.face_boundary[f] = if x < eps {
                    Some(BoundaryType::Inlet)
                } else if x > 3.0 - eps {
                    Some(BoundaryType::Outlet)
                } else if y < eps || y > 1.0 - eps {
                    Some(BoundaryType::SlipWall)
                } else {
                    continue;
                };
            }
        }
        let params = gui_params();
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(INLET as f64, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("driver build");
        moving.set_boundary_retag(Some(retag));
        moving.set_reorder_every_n(every_n);
        moving.driver_mut().apply_params(&params);
        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        let mut max_du = 0.0f32;
        for step in 0..steps {
            let (outcome, _stats) = moving
                .step(false)
                .unwrap_or_else(|e| panic!("reorder-frozen step {step}: {e}"));
            assert!(outcome.diverged.is_none(), "diverged at step {step}");
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            for c in 0..n {
                max_du = max_du
                    .max((state[c * stride + u_off] - INLET).abs())
                    .max(state[c * stride + u_off + 1].abs());
            }
        }
        println!(
            "[reorder-frozen] {steps} steps, reorder every {every_n}: max|U-U0| = {max_du:.3e}, \
             locality(final) = {:.1}",
            locality(moving.mesh())
        );
        assert!(
            max_du < 1e-4,
            "free stream corrupted by reordering: {max_du:.3e}"
        );
    }

    pub fn run() {
        let which = std::env::var("PROBE_CASES").unwrap_or_else(|_| "all".into());
        let has = |k: &str| which == "all" || which.split(',').any(|c| c == k);
        if has("static") {
            run_static("static-vanleer", Scheme::SecondOrderUpwindVanLeer, true, STEPS);
        }
        if has("static-upwind") {
            run_static("static-upwind", Scheme::Upwind, true, STEPS);
        }
        if has("static-rect") {
            run_static("static-rect-vanleer", Scheme::SecondOrderUpwindVanLeer, false, 600);
        }
        if has("recycle-spike") {
            run_recycle_spike(800);
        }
        if has("recycle-inlet") {
            run_recycle_inlet(800);
        }
        if has("visual") {
            run_visual(1200, &[100, 400, 800, 1199]);
        }
        if has("stability-matrix") {
            run_stability_matrix(500);
        }
        if has("reorder-frozen") {
            run_reorder_frozen(200, 20);
        }
        if has("thermal-ladder") {
            run_thermal_ladder(1500);
        }
        if has("thermal-long") {
            // Replicate the WORKER smoke's exact configuration (which
            // diverges within its 120s budget ≈ 5000–8000 steps) at a fixed
            // long horizon, with the advection scheme as the variable.
            for scheme in [Scheme::Upwind, Scheme::SecondOrderUpwindVanLeer] {
                run_thermal_long(scheme, 8000);
            }
        }
        if has("mode-dump") {
            let step = std::env::var("PROBE_DUMP_STEP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(260);
            run_mode_dump(step);
        }
        if has("passthrough") {
            run_passthrough("ale-passthrough");
        }
        if has("frozen") {
            run_case("frozen", MeshMotionSpec::Frozen);
        }
        if has("flowcoupled") {
            run_case("flowcoupled", MeshMotionSpec::FlowCoupled { regularization: 0.5 });
        }
        if has("flowcoupled-long") {
            run_case_steps(
                "flowcoupled-long",
                MeshMotionSpec::FlowCoupled { regularization: 0.5 },
                2000,
            );
        }
        if has("adapt") {
            run_adapt(2000, 50);
        }
    }
}
