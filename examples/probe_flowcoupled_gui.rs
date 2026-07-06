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
                println!(
                    "[{label}] step {step:3}: dt={:.2e} |U|max={umax:.3e} \
                     @({:.3},{:.3}) vol@argmax={:.2e} |U|max_interior={umax_interior:.3e} \
                     vol_min={vmin:.2e} @({:.3},{:.3}) skew={:.3} flip={} SCL={:.1e}",
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
