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
                     flip={} SCL={:.1e} recycled_total={total_recycled}",
                    stats.dt, mesh.cell_cx[iu], mesh.cell_cy[iu], mesh.cell_vol[iu],
                    mesh.cell_cx[ivm], mesh.cell_cy[ivm],
                    stats.max_skew, stats.flipped, stats.scl_defect,
                );
            }
        }
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
    }
}
