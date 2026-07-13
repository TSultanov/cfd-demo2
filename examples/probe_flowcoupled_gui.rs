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
    use cfd2::meshgen::meshless::SeedKind;
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

    /// Initial mesh cell size, overridable via `PROBE_DIPOLE_H` (default [`H`]).
    /// Used to calibrate the mesh-aware preconditioner floor: coarser initial
    /// meshes lose step-0 acoustic damping margin at higher floors.
    fn probe_h() -> f64 {
        std::env::var("PROBE_DIPOLE_H")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(H)
    }

    fn gui_params() -> RuntimeParams {
        RuntimeParams {
        filter_sigma: 0.0,
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
            allmach_precond_uref_min: 0.2,
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
        let cvt = generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
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
                m.cell_cx[imin],
                m.cell_cy[imin],
                m.cell_cx[imax],
                m.cell_cy[imax],
                LX - 2.5 * H,
                m.calculate_max_skewness()
            );
        }

        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt, &params, motion, &initial_u, &initial_p, None, None,
        ))
        .expect("driver build");
        if let Some(n_reorder) = std::env::var("PROBE_REORDER")
            .ok()
            .and_then(|s| s.parse().ok())
        {
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
                let state = pollster::block_on(moving.driver().solver().read_state_f32());
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
                    stats.dt,
                    mesh.cell_cx[iu],
                    mesh.cell_cy[iu],
                    mesh.cell_vol[iu],
                    mesh.cell_cx[ivm],
                    mesh.cell_cy[ivm],
                    stats.max_skew,
                    stats.flipped,
                    stats.scl_defect,
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
            MeshMotionSpec::FlowCoupled {
                regularization: 0.5,
            },
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
                let (mut v_near, mut c_near, mut v_far, mut c_far) =
                    (0.0f64, 0usize, 0.0f64, 0usize);
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
            generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default())
        } else {
            let geo = RectangularChannel {
                length: LX,
                height: LY,
            };
            generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default())
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
        let cvt = generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();
        let mesh = cvt.mesh.clone();
        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &initial_u,
            &initial_p,
            None,
            None,
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
        let cvt = generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
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
        println!(
            "[mode-dump] second-column max|U| = {c2max:.3e} over {} cells",
            col2.len()
        );
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
        let cvt = generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();
        let params = gui_params();
        let initial_u = vec![(INLET as f64, 0.0); n];
        let initial_p = vec![0.0; n];
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::FlowCoupled {
                regularization: 0.5,
            },
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
                let tag = if stats.recycled > 0 {
                    "RECYCLE"
                } else if watch > 0 {
                    "after  "
                } else {
                    "ambient"
                };
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

    /// PHANTOM-DIPOLE watch at the user's EXACT GUI config: incompressible,
    /// FlowCoupled, obstacle, adaptation every step, budget 5x, smoothing
    /// every step, band (0.001, 0.03) — the extreme-refinement regime.
    /// Renders the PRESSURE field to PNG EVERY STEP and screens each frame
    /// with a dipole metric: the max adjacent-cell |p| jump normalized by
    /// the frame's robust p-range (5th..95th percentile). A phantom dipole
    /// is a +/- pair on neighboring cells with amplitude comparable to (or
    /// exceeding) the whole smooth field's range — exactly what saturates
    /// the GUI color scale. Prints the worst frames for eyeball follow-up.
    fn run_dipole_watch(steps: usize) {
        // PROBE_DIPOLE_OUT: frame/zoom output dir (default target/probe_dipole)
        // — set per-run when two backends run concurrently, or the PNGs
        // interleave and the visual record is useless.
        let out_dir_owned = std::env::var("PROBE_DIPOLE_OUT")
            .unwrap_or_else(|_| "target/probe_dipole".into());
        let out_dir = std::path::Path::new(&out_dir_owned);
        std::fs::create_dir_all(out_dir).expect("mkdir probe_dipole");
        let domain = Vector2::new(LX, LY);
        let (ocx, ocy, orad) = (1.0f64, 0.51f64, 0.1f64);
        let geo = ChannelWithObstacle {
            length: LX,
            height: LY,
            obstacle_center: Point2::new(ocx, ocy),
            obstacle_radius: orad,
        };
        let cvt = generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
        let n0 = cvt.mesh.num_cells();
        // PROBE_DIPOLE_FAMILY=allmach runs the COMPRESSIBLE thermal all-Mach
        // ALE family (same recipe knobs as the visual probe's GUI regime) —
        // the birth/death dipole discipline must hold there too.
        let thermal = std::env::var("PROBE_DIPOLE_FAMILY")
            .map(|v| v == "allmach")
            .unwrap_or(false);
        let mut params = gui_params();
        if thermal {
            // PROBE_DIPOLE_GUI=1 reproduces the ACTUAL GUI thermal-obstacle
            // config (gui_params() verbatim: BDF2, dt 0.02, target_cfl 0.9,
            // outer 8 auto-converge, inlet 0.011, real Air viscosity) — the
            // reported divergence config. Otherwise the STABLE gate override
            // (Euler, dt 5e-3, inlet 0.4) used by the shipped dipole gate.
            let gui_faithful = std::env::var("PROBE_DIPOLE_GUI")
                .map(|v| v == "1")
                .unwrap_or(false);
            if gui_faithful {
                // The thermal model needs a positive compressibility; the GUI
                // derives psi = 1/c^2 from the gas EOS (real Air ~8.3e-6).
                params.compressibility_psi = std::env::var("PROBE_DIPOLE_PSI")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8.3e-6);
                // PROBE_DIPOLE_INLET speeds up the fix-iteration loop: a higher
                // inlet develops the wake to the outlet in fewer steps (the
                // divergence is a through-flow-reaches-outlet event). Default
                // 0.011 (the reported config).
                if let Some(u) = std::env::var("PROBE_DIPOLE_INLET")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    params.inlet_velocity = u;
                }
            } else {
                params.time_scheme = TimeScheme::Euler;
                params.requested_dt = 0.005;
                params.outer_iters = 6;
                params.compressibility_psi = 1.0e-4;
                params.viscosity = 1e-2;
                params.inlet_velocity = 0.4;
            }
        }
        if let Some(outers) = std::env::var("PROBE_DIPOLE_OUTERS")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            // Fixed outer budget for the experiment: auto-converge's plateau
            // exit would cut it before the slow wall modes relax.
            params.outer_iters = outers;
            params.outer_auto_converge = false;
        }
        // PROBE_DIPOLE_DTAU enables pseudo-transient continuation (dual-time):
        // a dtau>0 pseudo-time ddt term that damps the re-excited low-Mach
        // pseudo-acoustics each step (the standard cure for the transient
        // preconditioner mismatch).
        if let Some(dtau) = std::env::var("PROBE_DIPOLE_DTAU")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            params.dtau = dtau;
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
        // The user's GUI settings, verbatim.
        moving.set_adaptive_sizing(1);
        moving.set_adaptive_sizing_band(Some((0.001, 0.03)));
        // PROBE_DIPOLE_BUDGET overrides the growth budget (default 5x); the
        // reported outlet-divergence config uses 8x.
        let budget = std::env::var("PROBE_DIPOLE_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5.0);
        moving.set_adaptive_budget_factor(budget);
        moving.set_smoothing(1, 1, 0.5);
        // PROBE_DIPOLE_REORDER=N enables periodic cell reordering every N steps
        // (default off); the reported config uses 20.
        if let Some(n) = std::env::var("PROBE_DIPOLE_REORDER")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            moving.set_reorder_every_n(n);
        }
        // PROBE_DIPOLE_ADAPTIVE_DT=cfl enables the driver-side flow-CFL dt
        // (the shipped GUI default on the moving path). The transfer-noise
        // pressure response scales ~ rho*du*h/dt = interp-error/CFL — at the
        // GUI's fixed dt=0.02 with inlet 0.011 and 0.001-band cells the CFL
        // is ~0.02, a ~45x amplifier of interpolation noise into pressure.
        if let Some(cfl) = std::env::var("PROBE_DIPOLE_ADAPTIVE_DT")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            moving.set_adaptive_dt(Some(cfl));
        }
        // PROBE_DIPOLE_TRIAL=1: trial-step adaptation — plan each event
        // from the UPCOMING solution (trial step, rewind, resize, re-solve
        // on the final mesh).
        if std::env::var("PROBE_DIPOLE_TRIAL")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            moving.set_trial_step_adaptation(true);
        }

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let p_off = layout.offset_for("p").expect("p") as usize;
        // DISCRIMINATOR: PROBE_DIPOLE_FREEZE=N disables adaptation+smoothing
        // at step N. If the wall checkerboard decays afterwards, it is
        // churn-maintained (re-injected by the every-step rebuilds); if it
        // persists on the frozen mesh, it is a steady discrete mode of the
        // refined wall discretization itself.
        let freeze_at: Option<usize> = std::env::var("PROBE_DIPOLE_FREEZE")
            .ok()
            .and_then(|s| s.parse().ok());
        // What the freeze disables: "both" (default) | "adapt" | "smooth".
        let freeze_mode =
            std::env::var("PROBE_DIPOLE_FREEZE_MODE").unwrap_or_else(|_| "both".into());
        let mut worst: Vec<(f64, usize)> = Vec::new(); // (dipole metric, step)
                                                       // AMBIENT vs EVENT-ZONE split: recent events' positions (births and
                                                       // wall births land at the cell-index tail of each resize) with their
                                                       // step. A face within EVENT_R_CELLS local spacings of an event
                                                       // younger than EVENT_AGE steps is "event zone" — the unavoidable
                                                       // transient of an actively-refining site (a binary split is a
                                                       // factor-2 discretization jump; its solution transition cannot be
                                                       // zero). "No phantom dipoles" then decomposes into two testable
                                                       // claims: the AMBIENT dip sits at the frozen-mesh floor on EVERY
                                                       // frame, and the event-zone excess DECAYS within EVENT_AGE steps
                                                       // instead of accumulating.
        const EVENT_AGE: usize = 6;
        const EVENT_R_CELLS: f64 = 4.0;
        let mut events: Vec<(f64, f64, usize)> = Vec::new();
        let mut ambient_worst = 0.0f64;
        let mut ambient_top: Vec<(f64, usize, f64, f64)> = Vec::new();
        let mut osc_series: Vec<(usize, f64)> = Vec::new();
        let mut osc_sites: Vec<(f64, usize, f64, f64)> = Vec::new();
        let mut prev_seeds: Option<Vec<Point2<f64>>> = None;
        let (mut outlet_iso_worst, mut outlet_vr_worst, mut outlet_skew_worst) =
            (0.0f64, f64::INFINITY, 0.0f64);
        let mut outlet_iso_top = (0.0f64, 0usize, 0.0f64, 0.0f64);
        // STANDING-MODE AMPLITUDE (phase-robust): pressure spread in the
        // UPSTREAM inlet strip (x<0.9, upstream of the obstacle at x≈1.1) has
        // NO physical wake — any large max-min there is the domain-scale
        // pseudo-acoustic standing mode. Track running mean/max over the
        // developed regime (step>=200) so the metric is a phase-independent
        // amplitude, not a single-snapshot phase.
        let (mut ff_sum, mut ff_max, mut ff_n) = (0.0f64, 0.0f64, 0usize);
        // Late window (final 200 steps): with PROBE_DIPOLE_FREEZE set before it,
        // this is the POST-freeze amplitude — if the standing mode is fed by the
        // adaptation transfer, it decays here relative to the full-run mean.
        let late_from = steps.saturating_sub(200);
        let (mut ff_late_sum, mut ff_late_n) = (0.0f64, 0usize);
        for step in 0..steps {
            if freeze_at == Some(step) {
                if freeze_mode != "smooth" {
                    moving.set_adaptive_sizing(0);
                }
                if freeze_mode != "adapt" {
                    moving.set_smoothing(0, 1, 0.5);
                }
                println!("[dipole-watch] step {step}: FROZE ({freeze_mode})");
            }
            let (outcome, stats) = moving.step(false).expect("step");
            // OUTLET-DIVERGENCE DIAGNOSTIC: track T / rho / |U| / p extremes
            // (and their locations) BEFORE the divergence assert so the
            // proximate cause (T->0 => rho spike, sub-vacuum p, or |U| blow-up)
            // is visible. Print each step once the flow is stressed, and a full
            // dump on the diverging step. `T`/`rho` offsets exist only for the
            // thermal model.
            {
                let st = pollster::block_on(moving.driver().solver().read_state_f32());
                let m = moving.mesh();
                let nn = m.num_cells();
                let t_off = layout.offset_for("T").map(|o| o as usize);
                let rho_off = layout.offset_for("rho").map(|o| o as usize);
                let u_off = layout.offset_for("U").expect("U") as usize;
                let mut tmin = (f64::INFINITY, 0usize);
                let mut rhomax = (0.0f64, 0usize);
                let mut umax = (0.0f64, 0usize);
                let mut pmin = (f64::INFINITY, 0usize);
                let (mut p_in_lo, mut p_in_hi) = (f64::INFINITY, f64::NEG_INFINITY);
                for c in 0..nn {
                    if let Some(to) = t_off {
                        let t = st[c * stride + to] as f64;
                        if t < tmin.0 { tmin = (t, c); }
                    }
                    if let Some(ro) = rho_off {
                        let r = st[c * stride + ro] as f64;
                        if r > rhomax.0 { rhomax = (r, c); }
                    }
                    let uu = (st[c * stride + u_off] as f64).hypot(st[c * stride + u_off + 1] as f64);
                    if uu > umax.0 { umax = (uu, c); }
                    let pp = st[c * stride + p_off] as f64;
                    if pp < pmin.0 { pmin = (pp, c); }
                    // Upstream-strip pressure spread (standing-mode amplitude).
                    if m.cell_cx[c] < 0.9 {
                        p_in_lo = p_in_lo.min(pp);
                        p_in_hi = p_in_hi.max(pp);
                    }
                }
                if step >= 200 && p_in_hi > p_in_lo {
                    let sp = p_in_hi - p_in_lo;
                    ff_sum += sp;
                    ff_max = ff_max.max(sp);
                    ff_n += 1;
                    if step >= late_from {
                        ff_late_sum += sp;
                        ff_late_n += 1;
                    }
                }
                let loc = |c: usize| (m.cell_cx[c], m.cell_cy[c], m.cell_vol[c]);
                let diverged = outcome.diverged.is_some()
                    || !umax.0.is_finite()
                    || umax.0 > 100.0 * inlet as f64;
                if diverged || step % 50 == 0 {
                    let (tx, ty, tv) = loc(tmin.1);
                    let (rx, ry, rv) = loc(rhomax.1);
                    let (ux, uy, uv) = loc(umax.1);
                    let (px, py, pv) = loc(pmin.1);
                    println!(
                        "[diag] step {step} {}| Tmin={:.4e}@({:.3},{:.3},vol={:.2e}) rhomax={:.4e}@({:.3},{:.3},vol={:.2e}) |U|max={:.4e}@({:.3},{:.3},vol={:.2e}) pmin={:.4e}@({:.3},{:.3},vol={:.2e}) cells={} born={} killed={} recyc={} dt={:.3e}",
                        if diverged { "DIVERGING " } else { "" },
                        tmin.0, tx, ty, tv, rhomax.0, rx, ry, rv, umax.0, ux, uy, uv,
                        pmin.0, px, py, pv, nn, stats.cells_born, stats.cells_killed, stats.recycled, stats.dt,
                    );
                }
            }
            assert!(
                outcome.diverged.is_none(),
                "[dipole-watch] diverged at step {step}"
            );
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            let mesh = moving.mesh();
            let n = mesh.num_cells();
            let born = stats.cells_born.min(n);
            for c in n - born..n {
                events.push((mesh.cell_cx[c], mesh.cell_cy[c], step));
            }
            // KILL sites are event zones too (the hole the absorbing
            // neighbors settle into) — without them the "ambient" split
            // still contains kill transients.
            for &(kx, ky) in moving.last_kill_sites() {
                events.push((kx, ky, step));
            }
            // RECYCLED slots keep their index but teleport outlet -> inlet;
            // both the drained site and the respawn site are event zones.
            let seeds_now = moving.seeds().to_vec();
            if let Some(prev) = &prev_seeds {
                for i in 0..prev.len().min(seeds_now.len()) {
                    if (seeds_now[i] - prev[i]).norm() > 0.5 {
                        events.push((prev[i].x, prev[i].y, step));
                        events.push((seeds_now[i].x, seeds_now[i].y, step));
                    }
                }
            }
            prev_seeds = Some(seeds_now);
            events.retain(|&(_, _, s)| step - s < EVENT_AGE);
            let p_of = |c: usize| state[c * stride + p_off] as f64;
            // Robust field range: 5th..95th percentile of p.
            let mut ps: Vec<f64> = (0..n).map(p_of).collect();
            ps.sort_by(f64::total_cmp);
            let range = (ps[(n * 95) / 100] - ps[(n * 5) / 100]).max(1e-30);
            // Dipole metric: worst adjacent-cell jump / robust range,
            // globally AND excluding the event zones (ambient).
            let (mut max_jump, mut max_face, mut ambient_jump, mut ambient_face) =
                (0.0f64, 0usize, 0.0f64, 0usize);
            for f in 0..mesh.num_faces() {
                if let Some(nb) = mesh.face_neighbor[f] {
                    let d = (p_of(mesh.face_owner[f]) - p_of(nb)).abs();
                    if d > max_jump {
                        max_jump = d;
                        max_face = f;
                    }
                    if d > ambient_jump {
                        let (fx, fy) = (mesh.face_cx[f], mesh.face_cy[f]);
                        let h_loc = mesh.cell_vol[mesh.face_owner[f]].max(1e-30).sqrt();
                        let r = EVENT_R_CELLS * h_loc.max(H);
                        let near_event = events
                            .iter()
                            .any(|&(ex, ey, _)| (ex - fx).hypot(ey - fy) < r);
                        if !near_event {
                            ambient_jump = d;
                            ambient_face = f;
                        }
                    }
                }
            }
            let dip = max_jump / range;
            let ambient_dip = ambient_jump / range;
            // PHANTOM-DIPOLE-SPECIFIC metric: the `dip` jump above flags any
            // steep adjacent |dp| — at deep refinement its worst faces are
            // MONOTONE resolved-gradient steps (suction slope, corner
            // singularities), NOT dipoles. A phantom dipole is a +/-
            // OSCILLATION pair: adjacent cells whose deviations from their
            // own neighborhood means carry OPPOSITE signs. For a linear
            // field on a centroidal mesh the deviation is ~0 (gradients are
            // nulled exactly); a smooth resolved extremum scores at
            // truncation size O(h^2 * lap p); a checkerboard scores at its
            // full amplitude. osc = max over opposite-signed adjacent pairs
            // of min(|r_o|, |r_n|) / range.
            let mut resid = vec![0.0f64; n];
            for c in 0..n {
                let (fb, fe) = (mesh.cell_face_offsets[c], mesh.cell_face_offsets[c + 1]);
                let (mut sum, mut cnt) = (0.0f64, 0usize);
                for &f in &mesh.cell_faces[fb..fe] {
                    let other = if mesh.face_owner[f] == c {
                        mesh.face_neighbor[f]
                    } else {
                        Some(mesh.face_owner[f])
                    };
                    if let Some(nb) = other {
                        sum += p_of(nb);
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    resid[c] = p_of(c) - sum / cnt as f64;
                }
            }
            let (mut osc_amp, mut osc_face) = (0.0f64, 0usize);
            // AMBIENT variant: excludes the event zones (fresh births +
            // recycle teleport sites, same window as the dip split above)
            // — the "does adaptation leave dipoles AWAY from its own
            // in-flight transients" discriminator.
            let (mut osc_amb_amp, mut osc_amb_face) = (0.0f64, 0usize);
            for f in 0..mesh.num_faces() {
                if let Some(nb) = mesh.face_neighbor[f] {
                    let (ro, rn) = (resid[mesh.face_owner[f]], resid[nb]);
                    let amp = ro.abs().min(rn.abs());
                    if ro * rn < 0.0 && amp > osc_amb_amp.min(osc_amp) {
                        if amp > osc_amp {
                            osc_amp = amp;
                            osc_face = f;
                        }
                        if amp > osc_amb_amp {
                            let (fx, fy) = (mesh.face_cx[f], mesh.face_cy[f]);
                            let h_loc = mesh.cell_vol[mesh.face_owner[f]].max(1e-30).sqrt();
                            let r = EVENT_R_CELLS * h_loc.max(H);
                            let near_event = events
                                .iter()
                                .any(|&(ex, ey, _)| (ex - fx).hypot(ey - fy) < r);
                            if !near_event {
                                osc_amb_amp = amp;
                                osc_amb_face = f;
                            }
                        }
                    }
                }
            }
            let osc = osc_amp / range;
            let osc_amb = osc_amb_amp / range;
            osc_series.push((step, osc));
            osc_sites.push((osc, step, mesh.face_cx[osc_face], mesh.face_cy[osc_face]));
            println!(
                "[osc-frame] step {step} osc {osc:.4} @({:.3},{:.3}) ambient {osc_amb:.4} @({:.3},{:.3})",
                mesh.face_cx[osc_face],
                mesh.face_cy[osc_face],
                mesh.face_cx[osc_amb_face],
                mesh.face_cy[osc_amb_face]
            );
            // STRIP-GUARD watch: the y-wall guard cells inside the outlet
            // strip are recycle-exempt and immovable — track how deeply the
            // advected crowd shaves them (the corner-hill suspect).
            {
                let kinds_now = moving.seed_kinds();
                let (mut gv_min, mut gx, mut gy) = (f64::INFINITY, 0.0, 0.0);
                for c in 0..n {
                    if mesh.cell_cx[c] > 2.8
                        && !matches!(kinds_now[c], SeedKind::Interior)
                        && mesh.cell_vol[c] < gv_min
                    {
                        gv_min = mesh.cell_vol[c];
                        gx = mesh.cell_cx[c];
                        gy = mesh.cell_cy[c];
                    }
                }
                if gv_min.is_finite() {
                    println!("[strip-guards] step {step} vmin {gv_min:.3e} @({gx:.3},{gy:.3})");
                }
            }
            // Flare anatomy: is the worst +/- pair the TWIN half-guards born
            // by a wall split (two near-identical tiny wall cells sharing a
            // face = weakly damped two-cell checkerboard mode)?
            if std::env::var("PROBE_OSC_ANATOMY").is_ok() && osc > 1.5 {
                let kinds = moving.seed_kinds();
                let (o, nb) = (mesh.face_owner[osc_face], mesh.face_neighbor[osc_face].unwrap());
                for c in [o, nb] {
                    let (fb, fe) = (mesh.cell_face_offsets[c], mesh.cell_face_offsets[c + 1]);
                    let nbp: Vec<String> = mesh.cell_faces[fb..fe]
                        .iter()
                        .filter_map(|&f| {
                            let other = if mesh.face_owner[f] == c {
                                mesh.face_neighbor[f]
                            } else {
                                Some(mesh.face_owner[f])
                            };
                            other.map(|x| {
                                let kc = match kinds[x] {
                                    SeedKind::Interior => "I",
                                    _ => "B",
                                };
                                format!(
                                    "{x}{kc}:{:+.4e}(v{:.2e})",
                                    p_of(x),
                                    mesh.cell_vol[x]
                                )
                            })
                        })
                        .collect();
                    println!(
                        "[osc-anatomy] step {step} osc {osc:.3} cell {c} kind {:?} @({:.4},{:.4}) vol {:.3e} p {:+.4e} resid {:+.4e} nbrs [{}]",
                        kinds[c], mesh.cell_cx[c], mesh.cell_cy[c], mesh.cell_vol[c], p_of(c), resid[c], nbp.join(", ")
                    );
                }
            }
            if step >= 30 {
                ambient_worst = ambient_worst.max(ambient_dip);
                ambient_top.push((
                    ambient_dip,
                    step,
                    mesh.face_cx[ambient_face],
                    mesh.face_cy[ambient_face],
                ));
            }
            worst.push((dip, step));
            // OUTLET-strip cell quality: the recycle squeeze compresses
            // cells against the outlet by design — watch that it never
            // degenerates them into slivers. Per interior cell in the strip:
            // isoperimetric ratio P^2/(4*pi*A) (1 = circle; slivers blow
            // up), volume vs the strip median, and worst per-face skew
            // (1 - |d_hat . n_hat|).
            {
                let strip_x = LX - 5.0 * H;
                let strip: Vec<usize> = (0..n).filter(|&c| mesh.cell_cx[c] > strip_x).collect();
                if !strip.is_empty() {
                    let mut vols: Vec<f64> = strip.iter().map(|&c| mesh.cell_vol[c]).collect();
                    vols.sort_by(f64::total_cmp);
                    let med = vols[vols.len() / 2];
                    let (mut worst_iso, mut worst_iso_c) = (0.0f64, 0usize);
                    let mut worst_vr = f64::INFINITY;
                    let mut worst_skew = 0.0f64;
                    for &c in &strip {
                        let (fb, fe) = (mesh.cell_face_offsets[c], mesh.cell_face_offsets[c + 1]);
                        let perim: f64 = mesh.cell_faces[fb..fe]
                            .iter()
                            .map(|&f| mesh.face_area[f])
                            .sum();
                        let a = mesh.cell_vol[c].max(1e-30);
                        let iso = perim * perim / (4.0 * std::f64::consts::PI * a);
                        if iso > worst_iso {
                            worst_iso = iso;
                            worst_iso_c = c;
                        }
                        worst_vr = worst_vr.min(mesh.cell_vol[c] / med);
                        for &f in &mesh.cell_faces[fb..fe] {
                            if let Some(nb) = mesh.face_neighbor[f] {
                                let o = mesh.face_owner[f];
                                let (dx, dy) = (
                                    mesh.cell_cx[nb] - mesh.cell_cx[o],
                                    mesh.cell_cy[nb] - mesh.cell_cy[o],
                                );
                                let d = dx.hypot(dy).max(1e-30);
                                let dot = (dx * mesh.face_nx[f] + dy * mesh.face_ny[f]).abs() / d;
                                worst_skew = worst_skew.max(1.0 - dot);
                            }
                        }
                    }
                    if step >= 30 {
                        outlet_iso_worst = outlet_iso_worst.max(worst_iso);
                        outlet_vr_worst = outlet_vr_worst.min(worst_vr);
                        outlet_skew_worst = outlet_skew_worst.max(worst_skew);
                        if worst_iso > outlet_iso_top.0 {
                            outlet_iso_top = (
                                worst_iso,
                                step,
                                mesh.cell_cx[worst_iso_c],
                                mesh.cell_cy[worst_iso_c],
                            );
                        }
                    }
                }
            }
            // PROBE_DIPOLE_NORENDER=1 skips the per-step PNG (long divergence
            // hunts): the [diag]/[osc-frame] telemetry is enough to localize.
            if std::env::var("PROBE_DIPOLE_NORENDER").map(|v| v != "1").unwrap_or(true) {
                render_voronoi_field(
                    mesh,
                    &p_of,
                    (ocx, ocy, orad),
                    &out_dir.join(format!("p{step:04}.png")),
                );
            }
            if step % 50 == 0 || dip > 1.5 {
                println!(
                    "[dipole-watch] step {step:4}: {} cells, dip {dip:.3} ambient {ambient_dip:.3} \
                     (jump {max_jump:.3e} / range {range:.3e}) at face ({:.3},{:.3}), \
                     +{}/-{} recycled {} defect {:.2e}->{:.2e}",
                    n,
                    mesh.face_cx[max_face],
                    mesh.face_cy[max_face],
                    stats.cells_born,
                    stats.cells_killed,
                    stats.recycled,
                    stats.transfer_defect_pre,
                    stats.transfer_defect_post,
                );
            }
        }
        // STREAMWISE PROFILE (final field): mean p and mean |U| in 0.5-wide x
        // bands, plus the outlet-corner |U|. Diagnoses the pressure-gradient
        // SIGN (favorable = p falls toward the outlet) and whether the flow
        // sustains its inlet flux downstream or decays.
        {
            let st = pollster::block_on(moving.driver().solver().read_state_f32());
            let m = moving.mesh();
            let nn = m.num_cells();
            let uo = layout.offset_for("U").expect("U") as usize;
            let nb = (LX / 0.5).ceil() as usize;
            let mut psum = vec![0.0f64; nb];
            let mut usum = vec![0.0f64; nb];
            let mut cnt = vec![0usize; nb];
            let (mut corner_tr, mut corner_br) = (0.0f64, 0.0f64);
            for c in 0..nn {
                let (x, y) = (m.cell_cx[c], m.cell_cy[c]);
                let b = ((x / 0.5) as usize).min(nb - 1);
                let p = st[c * stride + p_off] as f64;
                let um = (st[c * stride + uo] as f64).hypot(st[c * stride + uo + 1] as f64);
                psum[b] += p;
                usum[b] += um;
                cnt[b] += 1;
                if x > LX - 0.15 && y > LY - 0.15 {
                    corner_tr = corner_tr.max(um);
                }
                if x > LX - 0.15 && y < 0.15 {
                    corner_br = corner_br.max(um);
                }
            }
            let pmean: Vec<String> = (0..nb)
                .map(|b| format!("{:.2e}", psum[b] / cnt[b].max(1) as f64))
                .collect();
            let umean: Vec<String> = (0..nb)
                .map(|b| format!("{:.2e}", usum[b] / cnt[b].max(1) as f64))
                .collect();
            println!("[profile] p(x-bands 0..{LX}): [{}]", pmean.join(", "));
            println!("[profile] |U|(x-bands): [{}]", umean.join(", "));
            println!(
                "[profile] outlet-corner |U|max: top-right={corner_tr:.3e} bottom-right={corner_br:.3e} (inlet {inlet:.3e})"
            );
            // PROBE_DIPOLE_FINAL_PNG=<tag>: dump the final pressure + |U| field
            // (full domain and an outlet-corner zoom) so the rendered solution
            // can be eyeballed the way the GUI shows it. Autoscaled per image.
            if let Ok(tag) = std::env::var("PROBE_DIPOLE_FINAL_PNG") {
                let p_fld = |c: usize| st[c * stride + p_off] as f64;
                let u_fld = |c: usize| {
                    (st[c * stride + uo] as f64).hypot(st[c * stride + uo + 1] as f64)
                };
                render_voronoi_field(m, &p_fld, (ocx, ocy, orad), &out_dir.join(format!("final_p_{tag}.png")));
                render_voronoi_field(m, &u_fld, (ocx, ocy, orad), &out_dir.join(format!("final_u_{tag}.png")));
                // Outlet strip zoom (x in [LX-0.6, LX], full height) — where the
                // reported "weirdness" lives.
                let win = (LX - 0.6, LX, 0.0, LY);
                render_voronoi_window(m, &p_fld, &out_dir.join(format!("outlet_p_{tag}.png")), win, (360, 300));
                render_voronoi_window(m, &u_fld, &out_dir.join(format!("outlet_u_{tag}.png")), win, (360, 300));
                println!("[final-png] wrote final_{{p,u}}_{tag}.png + outlet_{{p,u}}_{tag}.png to {}", out_dir.display());
            }
        }
        println!(
            "[dipole-watch] AMBIENT worst dip (steps 30..{steps}, faces beyond \
             {EVENT_R_CELLS} local spacings of any event younger than {EVENT_AGE} steps): \
             {ambient_worst:.3}"
        );
        println!(
            "[standing-mode] upstream-strip (x<0.9) pressure spread over steps 200..{steps}: \
             mean={:.3e} max={:.3e} (n={ff_n}) | late[{late_from}..{steps}] mean={:.3e} (n={ff_late_n}) \
             — phase-robust standing pseudo-acoustic amplitude",
            if ff_n > 0 { ff_sum / ff_n as f64 } else { 0.0 },
            ff_max,
            if ff_late_n > 0 { ff_late_sum / ff_late_n as f64 } else { 0.0 },
        );
        // Full OSC series to the log (one line per step) so any window can
        // be analyzed offline — the phantom-dipole-specific A/B needs
        // running-vs-frozen windows.
        for (s, o) in &osc_series {
            println!("[osc-series] step {s} osc {o:.5}");
        }
        osc_sites.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = osc_sites
            .iter()
            .filter(|(_, s, _, _)| *s >= 30)
            .take(12)
            .map(|(o, s, x, y)| format!("step {s}: {o:.2} @({x:.3},{y:.3})"))
            .collect();
        println!("[dipole-watch] OSC worst sites: {}", top.join("; "));
        println!(
            "[dipole-watch] OUTLET-strip quality (steps 30..{steps}): worst isoperimetric \
             {outlet_iso_worst:.2} (1 = circle; hexagonal CVT ~1.1) at step {} @({:.3},{:.3}); \
             worst vol/median {outlet_vr_worst:.3}; worst face skew {outlet_skew_worst:.3}",
            outlet_iso_top.1, outlet_iso_top.2, outlet_iso_top.3
        );
        // ZOOM of the outlet strip (final frame): ln(cell volume) over the
        // last 0.3 of the channel — the visual sliver check.
        {
            let mesh = moving.mesh();
            render_voronoi_window(
                mesh,
                &|c| mesh.cell_vol[c].max(1e-30).ln(),
                &out_dir.join("outlet_zoom_cellvol.png"),
                (LX - 0.3, LX, 0.0, LY),
                (300, 1000),
            );
            println!("[dipole-watch] outlet zoom written: outlet_zoom_cellvol.png");
        }
        ambient_top.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top_amb: Vec<String> = ambient_top
            .iter()
            .take(8)
            .map(|(d, s, x, y)| format!("step {s}: {d:.2} @({x:.3},{y:.3})"))
            .collect();
        println!("[dipole-watch] AMBIENT worst faces: {}", top_amb.join("; "));
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = worst
            .iter()
            .take(10)
            .map(|(d, s)| format!("step {s}: dip {d:.3}"))
            .collect();
        println!("[dipole-watch] WORST frames: {}", top.join("; "));
        let late_max = worst
            .iter()
            .filter(|(_, s)| *s >= steps / 2)
            .map(|(d, _)| *d)
            .fold(0.0f64, f64::max);
        println!(
            "[dipole-watch] late-half worst dip {late_max:.3} over {} frames",
            steps - steps / 2
        );

        // ANATOMY of the final frame's worst faces: are the offending cells
        // slivers, wall guards, freshly-born, or normal fluid? A persistent
        // dipole at a FIXED face is a steady discretization artifact, not a
        // transfer transient — the geometry is the suspect.
        {
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            let mesh = moving.mesh();
            let n = mesh.num_cells();
            let p_of = |c: usize| state[c * stride + p_off] as f64;
            let mut faces: Vec<(f64, usize)> = (0..mesh.num_faces())
                .filter_map(|f| {
                    mesh.face_neighbor[f].map(|nb| ((p_of(mesh.face_owner[f]) - p_of(nb)).abs(), f))
                })
                .collect();
            faces.sort_by(|a, b| b.0.total_cmp(&a.0));
            let kinds = moving.seed_kinds();
            for &(jump, f) in faces.iter().take(10) {
                let (o, nb) = (mesh.face_owner[f], mesh.face_neighbor[f].unwrap());
                println!(
                    "[dipole-anatomy] face ({:.4},{:.4}) area {:.2e} jump {jump:.3e}: \
                     owner c{o} kind {:?} vol {:.2e} p {:+.3e} | neighbor c{nb} kind {:?} \
                     vol {:.2e} p {:+.3e}",
                    mesh.face_cx[f],
                    mesh.face_cy[f],
                    mesh.face_area[f],
                    kinds[o],
                    mesh.cell_vol[o],
                    p_of(o),
                    kinds[nb],
                    mesh.cell_vol[nb],
                    p_of(nb),
                );
            }
            let _ = n;
        }
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
            Variant {
                label: "A adapt1+smooth1+recycle (fail cfg)",
                adapt_every: 1,
                band: None,
                smooth_every: 1,
                recycling: true,
                projection: true,
            },
            Variant {
                label: "B adapt1+smooth1 no-recycle      ",
                adapt_every: 1,
                band: None,
                smooth_every: 1,
                recycling: false,
                projection: true,
            },
            Variant {
                label: "C adapt5+smooth1+recycle         ",
                adapt_every: 5,
                band: None,
                smooth_every: 1,
                recycling: true,
                projection: true,
            },
            Variant {
                label: "D adapt1 WIDE band+smooth1+recycle",
                adapt_every: 1,
                band: Some((0.03, 0.07)),
                smooth_every: 1,
                recycling: true,
                projection: true,
            },
            Variant {
                label: "E smooth1+recycle no-adapt       ",
                adapt_every: 0,
                band: None,
                smooth_every: 1,
                recycling: true,
                projection: true,
            },
            Variant {
                label: "F adapt1+smooth1+recycle proj-OFF",
                adapt_every: 1,
                band: None,
                smooth_every: 1,
                recycling: true,
                projection: false,
            },
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
            let wall_t = std::time::Instant::now();
            let mut done_steps = 0usize;
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
                done_steps = step + 1;
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
            let traj: Vec<String> = track.iter().map(|(s, u)| format!("{s}:{u:.2e}")).collect();
            println!(
                "[matrix] {} | umax@[{}] births {births} kills {kills} recycle-steps {recycles}{}",
                v.label,
                traj.join(", "),
                died.map(|s| format!(" DIED@{s}")).unwrap_or_default()
            );
            // Throughput line, PROFILE-gated and separate from the physics
            // line above so cross-thread determinism diffs stay byte-clean.
            if std::env::var("CFD2_ALE_PROFILE").map_or(false, |v| v.trim() != "" && v.trim() != "0")
            {
                let secs = wall_t.elapsed().as_secs_f64();
                println!(
                    "[matrix-time] {} | {done_steps} steps in {secs:.2}s = {:.2} steps/s",
                    v.label,
                    done_steps as f64 / secs.max(1e-9),
                );
            }
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
        let cvt = generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
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
            assert!(
                outcome.diverged.is_none(),
                "[visual] diverged at step {step}"
            );
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
                    ("p", Box::new(|c| state[c * stride + p_off] as f64)),
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

    /// Windowed variant of [`render_voronoi_field`]: rasterize a per-cell
    /// scalar over `(x0, x1, y0, y1)` at the given pixel size — zoom views
    /// (e.g. the outlet strip sliver check).
    fn render_voronoi_window(
        mesh: &cfd2::solver::mesh::Mesh,
        value: &dyn Fn(usize) -> f64,
        path: &std::path::Path,
        window: (f64, f64, f64, f64),
        size: (u32, u32),
    ) {
        let n = mesh.num_cells();
        let (x0, x1, y0, y1) = window;
        let (w, h) = size;
        let (sx, sy) = ((x1 - x0) / w as f64, (y1 - y0) / h as f64);
        let cells: Vec<usize> = (0..n)
            .filter(|&c| {
                mesh.cell_cx[c] > x0 - 0.05
                    && mesh.cell_cx[c] < x1 + 0.05
                    && mesh.cell_cy[c] > y0 - 0.05
                    && mesh.cell_cy[c] < y1 + 0.05
            })
            .collect();
        if cells.is_empty() {
            return;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &c in &cells {
            let v = value(c);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let span = (hi - lo).max(1e-300);
        let mut img = image::RgbImage::new(w, h);
        for py in 0..h {
            for px in 0..w {
                let (x, y) = (x0 + (px as f64 + 0.5) * sx, y1 - (py as f64 + 0.5) * sy);
                let (mut best, mut best_d2) = (usize::MAX, f64::INFINITY);
                for &c in &cells {
                    let d2 = (mesh.cell_cx[c] - x).powi(2) + (mesh.cell_cy[c] - y).powi(2);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = c;
                    }
                }
                let t = ((value(best) - lo) / span).clamp(0.0, 1.0);
                let rgb = if t < 0.5 {
                    let s = t * 2.0;
                    [0, (s * 255.0) as u8, ((1.0 - s) * 255.0) as u8]
                } else {
                    let s = (t - 0.5) * 2.0;
                    [(s * 255.0) as u8, ((1.0 - s) * 255.0) as u8, 0]
                };
                img.put_pixel(px, py, image::Rgb(rgb));
            }
        }
        img.save(path).expect("write png");
        println!(
            "[visual]   {} range [{lo:.4e}, {hi:.4e}] ({} cells in window)",
            path.file_name().unwrap().to_string_lossy(),
            cells.len()
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
                            let d2 = (mesh.cell_cx[c] - x).powi(2) + (mesh.cell_cy[c] - y).powi(2);
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
                generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
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
                assert!(
                    outcome.diverged.is_none(),
                    "[{tag}] diverged at step {step}"
                );
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
        use cfd2::solver::model::{all_models, allmach_thermal_ale_model};
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
            let cvt = generate_cvt_mesh_with_seeds(
                &geo,
                0.06,
                0.06,
                1.0,
                domain,
                &LloydConfig::default(),
            );
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
            (
                "flowcoupled",
                MeshMotionSpec::FlowCoupled {
                    regularization: 0.5,
                },
            ),
        ] {
            let cvt = generate_cvt_mesh_with_seeds(
                &geo,
                0.06,
                0.06,
                1.0,
                domain,
                &LloydConfig::default(),
            );
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
            MeshMotionSpec::FlowCoupled {
                regularization: 0.5,
            },
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
        let geo = RectangularChannel {
            length: LX,
            height: LY,
        };
        let cvt = generate_cvt_mesh_with_seeds(&geo, probe_h(), probe_h(), 1.2, domain, &LloydConfig::default());
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
            run_static(
                "static-vanleer",
                Scheme::SecondOrderUpwindVanLeer,
                true,
                STEPS,
            );
        }
        if has("static-upwind") {
            run_static("static-upwind", Scheme::Upwind, true, STEPS);
        }
        if has("static-rect") {
            run_static(
                "static-rect-vanleer",
                Scheme::SecondOrderUpwindVanLeer,
                false,
                600,
            );
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
        if has("dipole-watch") {
            let steps = std::env::var("PROBE_DIPOLE_STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(600);
            run_dipole_watch(steps);
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
            run_case(
                "flowcoupled",
                MeshMotionSpec::FlowCoupled {
                    regularization: 0.5,
                },
            );
        }
        if has("flowcoupled-long") {
            run_case_steps(
                "flowcoupled-long",
                MeshMotionSpec::FlowCoupled {
                    regularization: 0.5,
                },
                2000,
            );
        }
        if has("adapt") {
            run_adapt(2000, 50);
        }
    }
}
