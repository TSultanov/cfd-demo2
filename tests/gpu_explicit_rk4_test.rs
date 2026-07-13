//! GPU integration gates for the matrix-free explicit RK4 path.
//!
//! The tests stay deliberately small: one unstructured pipeline compilation
//! covers both the exact constant-source seam and a temporal-order study, and
//! one structured compilation covers the corresponding direct GPU frontend.
//! A missing headless adapter is an allowed platform skip, matching the other
//! native GPU integration tests.
#![cfg(feature = "dev-tests")]

use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::structured::{BcComp, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::gpu::unified_solver::PlanParamValue;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    allmach_thermal_structured_model, compressible_structured_model,
    generic_diffusion_demo_mms_model, generic_diffusion_demo_structured_ibm_model,
    incompressible_momentum_model, MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[cfg(feature = "meshgen")]
use cfd2::sim::{RuntimeParams, SolverDriver};
#[cfg(feature = "meshgen")]
use cfd2::solver::model::compressible_model_with_eos;
#[cfg(feature = "meshgen")]
use cfd2::solver::model::helpers::SolverFieldAliasesExt;
#[cfg(feature = "meshgen")]
use cfd2::solver::GpuLowMachPrecondModel;

fn gpu_context(gate: &str) -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(error) => {
            eprintln!("[{gate}] no compatible GPU adapter ({error}); skipping GPU gate");
            None
        }
    }
}

fn observed_orders(dts: &[f64], errors: &[f64]) -> Vec<f64> {
    dts.windows(2)
        .zip(errors.windows(2))
        .map(|(h, e)| (e[0] / e[1]).ln() / (h[0] / h[1]).ln())
        .collect()
}

fn explicit_config() -> SolverConfig {
    SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::RK4,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Explicit,
    }
}

#[cfg(feature = "meshgen")]
fn explicit_driver_params() -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.5,
        requested_dt: 0.01,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 100,
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::RK4,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 1,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1.0e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 1.0,
        alpha_p: 1.0,
        inlet_velocity: 0.0,
        density: 1.0,
        viscosity: 0.0,
        eos: EosSpec::Constant,
        compressibility_psi: (1.0 / (347.0_f64 * 347.0)) as f32,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 1.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

/// `SolverDriver` reuses the packed state captured by the explicit GPU finite
/// audit. Scalar-only models have neither velocity nor pressure ports, so this
/// pins the legacy alias-getter contract: both fields are still cell-sized zero
/// vectors (and pressure extrema are zero), rather than empty vectors derived
/// from the absent offsets.
#[cfg(feature = "meshgen")]
#[test]
fn gpu_driver_packed_readback_preserves_missing_ui_port_fallbacks() {
    let Some(ctx) = gpu_context("gpu-packed-readback-missing-ports") else {
        return;
    };
    let mesh = generate_structured_rect_mesh(4, 3, 1.0, 0.75, BoundarySides::wall());
    let n = mesh.num_cells();
    let params = explicit_driver_params();
    let build = pollster::block_on(SolverDriver::build(
        &mesh,
        generic_diffusion_demo_mms_model().expect("diffusion MMS model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("GPU explicit scalar driver construction");
    let mut driver = build.driver;
    assert!(!driver.solver().is_cpu(), "test must exercise GPU reuse");
    assert!(matches!(
        driver.solver().config().stepping,
        SteppingMode::Explicit
    ));
    let ports = driver.solver().ui_ports();
    assert_eq!(ports.u_offset, None, "scalar model unexpectedly gained U/u");
    assert_eq!(ports.p_offset, None, "scalar model unexpectedly gained p");
    driver.apply_params(&params);
    driver
        .solver_mut()
        .set_field_scalar("phi", &vec![1.0; n])
        .expect("seed scalar field");
    driver.solver().initialize_history();

    let outcome = driver.step(true);
    assert!(
        outcome.diverged.is_none(),
        "explicit scalar step diverged: {:?}",
        outcome.diverged
    );
    let readback = outcome.readback.expect("requested packed readback");

    // Compare directly with the legacy public aliases on the unchanged
    // accepted state, in addition to pinning their exact fallback shape/stats.
    let getter_u = pollster::block_on(driver.solver().get_u());
    let getter_p = pollster::block_on(driver.solver().get_p());
    assert_eq!(readback.u, getter_u);
    assert_eq!(readback.p, getter_p);
    assert_eq!(readback.u, vec![(0.0, 0.0); n]);
    assert_eq!(readback.p, vec![0.0; n]);
    assert_eq!(readback.stats.max_vel.to_bits(), 0.0_f64.to_bits());
    assert_eq!(readback.stats.nonfinite_u, 0);
    assert_eq!(readback.stats.p_min.to_bits(), 0.0_f64.to_bits());
    assert_eq!(readback.stats.p_max.to_bits(), 0.0_f64.to_bits());
    assert!(readback.stats.p_finite);
    assert_eq!(readback.stats.nonfinite_p, 0);
    assert_eq!(readback.stats.rho, None);
}

/// The driver's explicit-GPU finite audit already owns a packed state readback.
/// When the GUI asks for fields on that same step, the reused packed snapshot
/// must be byte-for-byte equivalent to the legacy per-field getter surface.
#[cfg(feature = "meshgen")]
#[test]
fn gpu_driver_reused_packed_readback_matches_live_field_getters() {
    let Some(ctx) = gpu_context("gpu-rk4-driver-packed-readback") else {
        return;
    };
    let mesh = generate_structured_rect_mesh(2, 2, 1.0, 1.0, BoundarySides::wall());
    let n = mesh.num_cells();
    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };
    let params = RuntimeParams {
        requested_dt: 1.0e-7,
        log_every_steps: 1,
        density: 1.225,
        eos,
        compressibility_psi: 0.0,
        allmach_precond_uref_min: 0.2,
        ..explicit_driver_params()
    };
    let mut build = pollster::block_on(SolverDriver::build(
        &mesh,
        compressible_model_with_eos(eos).expect("compressible model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("GPU explicit driver construction");
    assert!(!build.driver.solver().is_cpu());
    build.driver.apply_params(&params);

    let outcome = build.driver.step(true);
    assert!(
        outcome.diverged.is_none(),
        "one uniform RK4 step diverged: {:?}",
        outcome.diverged
    );
    let readback = outcome.readback.expect("requested driver readback");
    let getter_u = pollster::block_on(build.driver.solver().get_u());
    let getter_p = pollster::block_on(build.driver.solver().get_p());
    let getter_rho = pollster::block_on(build.driver.solver().get_rho());

    assert_eq!(readback.u, getter_u, "packed U differs from getter U");
    assert_eq!(readback.p, getter_p, "packed p differs from getter p");
    let expected_max_u = getter_u
        .iter()
        .map(|&(ux, uy)| (ux * ux + uy * uy).sqrt())
        .fold(0.0_f64, f64::max);
    let expected_p = (
        getter_p.iter().copied().fold(f64::INFINITY, f64::min),
        getter_p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    assert_eq!(readback.stats.max_vel, expected_max_u);
    assert_eq!((readback.stats.p_min, readback.stats.p_max), expected_p);
    let expected_rho = (
        getter_rho.iter().copied().fold(f64::INFINITY, f64::min),
        getter_rho.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    assert_eq!(
        readback.stats.rho,
        Some(expected_rho),
        "packed rho telemetry differs from getter rho"
    );
    assert_eq!(readback.stats.nonfinite_u, 0);
    assert_eq!(readback.stats.nonfinite_p, 0);
    assert!(readback.stats.p_finite);
}

#[test]
fn gpu_unstructured_rk4_integrates_source_and_converges_at_order_four() {
    let Some(ctx) = gpu_context("gpu-rk4-unstructured") else {
        return;
    };
    let mesh = generate_structured_rect_mesh(4, 4, 1.0, 1.0, BoundarySides::wall());

    // The backend-independent capability gate must remain authoritative after
    // enabling the GPU runtime: a saddle-point pressure constraint is not an
    // explicit ODE and must still fail for its mathematical reason.
    let invalid = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model().expect("incompressible model"),
        explicit_config(),
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .err()
    .expect("incompressible saddle-point model must reject explicit RK4");
    assert!(
        invalid.contains("d_p")
            || invalid.contains("Rhie")
            || invalid.contains("no own-variable ddt")
            || invalid.contains("not a method-of-lines row"),
        "unexpected explicit-capability rejection: {invalid}"
    );
    assert!(
        !invalid.contains("CPU matrix-free backend") && !invalid.contains("select the CPU backend"),
        "GPU RK4 must reach the model capability gate, not the retired backend gate: {invalid}"
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        generic_diffusion_demo_mms_model().expect("diffusion MMS model"),
        explicit_config(),
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("GPU explicit RK4 solver construction");
    assert!(
        !solver.is_cpu(),
        "the RK4 integration gate must exercise the GPU backend"
    );
    assert_eq!(solver.config().time_scheme, TimeScheme::RK4);
    assert_eq!(solver.config().stepping, SteppingMode::Explicit);
    assert!(
        pollster::block_on(solver.get_linear_rhs()).is_err(),
        "explicit GPU runtime must not expose an implicit linear system"
    );
    let dtau_error = solver
        .set_dtau(1.0e-3)
        .expect_err("explicit RK4 must reject pseudo-time stepping");
    assert!(dtau_error.contains("dtau") || dtau_error.contains("pseudo-time"));
    let switch_error = solver
        .try_set_time_scheme(TimeScheme::Euler)
        .expect_err("RK4-to-implicit switching requires a rebuilt GPU program");
    assert!(switch_error.contains("rebuild"));
    assert_eq!(solver.config().time_scheme, TimeScheme::RK4);
    let named_switch_error = solver
        .set_named_param("time_scheme", PlanParamValue::TimeScheme(TimeScheme::BDF2))
        .expect_err("the public named-param path must not bypass the RK4 family guard");
    assert!(named_switch_error.contains("rebuild"));
    assert_eq!(solver.config().time_scheme, TimeScheme::RK4);

    // A uniform field removes the spatial operator exactly. This pins the
    // residual sign, local ddt inversion, RK workspaces, and the public
    // step-with-stats path. Explicit stepping performs no linear solve.
    solver.set_dt(0.125);
    solver
        .set_field_scalar("phi", &vec![1.0; mesh.num_cells()])
        .expect("set phi");
    solver
        .set_field_scalar(MMS_SOURCE_FIELD, &vec![-2.0; mesh.num_cells()])
        .expect("set source");
    solver.initialize_history();
    let linear_stats = solver.step_with_stats().expect("explicit GPU step");
    assert!(
        linear_stats.is_empty(),
        "matrix-free RK4 must not report linear-solver work"
    );
    let phi = pollster::block_on(solver.get_field_scalar("phi")).expect("read phi");
    let max_error = phi
        .iter()
        .map(|&value| (value - 0.75).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 2.0e-6,
        "uniform GPU RK4 source error = {max_error:e}"
    );

    // On this orthogonal 4x4 mesh, cos(pi*x)cos(pi*y) is an exact eigenvector
    // of the discrete zero-Neumann Laplacian. Comparing against that discrete
    // eigenvalue removes spatial truncation error. Crucially, this is a
    // state-dependent residual: evaluating it only once and then dispatching
    // all four stage updates degenerates to first order and fails this gate.
    let n = 4usize;
    let t_end = 0.2_f64;
    let lambda_h = 8.0 * (n * n) as f64 * (std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2);
    let exact_scale = (-lambda_h * t_end).exp();
    let initial: Vec<f64> = (0..mesh.num_cells())
        .map(|cell| {
            (std::f64::consts::PI * mesh.cell_cx[cell]).cos()
                * (std::f64::consts::PI * mesh.cell_cy[cell]).cos()
        })
        .collect();
    let dts = [0.04_f64, 0.02, 0.01];
    let mut errors = Vec::with_capacity(dts.len());

    for &dt in &dts {
        solver.set_dt(dt as f32);
        solver.set_field_scalar("phi", &initial).expect("reset phi");
        solver
            .set_field_scalar(MMS_SOURCE_FIELD, &vec![0.0; mesh.num_cells()])
            .expect("clear source");
        solver.initialize_history();
        for _ in 0..(t_end / dt).round() as usize {
            solver.step();
        }
        let got =
            pollster::block_on(solver.get_field_scalar("phi")).expect("read temporal-order field");
        let l2 = (got
            .iter()
            .zip(initial.iter())
            .map(|(&value, &value0)| (value - exact_scale * value0).powi(2))
            .sum::<f64>()
            / mesh.num_cells() as f64)
            .sqrt();
        errors.push(l2);
    }

    let orders = observed_orders(&dts, &errors);
    eprintln!("[gpu-rk4-unstructured] dt={dts:?} error={errors:?} order={orders:?}");
    assert!(
        orders.iter().all(|&order| order > 3.5),
        "GPU RK4 temporal order below four: {orders:?}, errors={errors:?}"
    );

    // Matrix-free stepping has no linear solve whose residual could carry a
    // failure bit. The stats API must still surface a non-finite RK state on
    // the same step so the driver stops immediately.
    solver.set_dt(0.01);
    solver.set_field_scalar("phi", &initial).expect("reset phi");
    solver
        .set_field_scalar(MMS_SOURCE_FIELD, &vec![f64::NAN; mesh.num_cells()])
        .expect("set non-finite source");
    solver.initialize_history();
    let stats = solver
        .step_with_stats()
        .expect("non-finite explicit step must still return stats");
    assert!(
        stats.iter().any(|stat| stat.diverged),
        "non-finite GPU RK4 state was not reported as diverged"
    );
}

#[test]
fn gpu_structured_rk4_converges_at_order_four() {
    let Some(ctx) = gpu_context("gpu-rk4-structured") else {
        return;
    };
    let model =
        generic_diffusion_demo_structured_ibm_model().expect("structured reaction-diffusion model");
    let dts = [0.125_f64, 0.0625, 0.03125];
    let t_end = 0.5_f64;
    let exact = (-4.0 * t_end).exp();
    let mut solver = StructuredGpuSolver::with_config(
        ctx,
        StructuredGrid::new(4, 4, 1.0, 1.0),
        &model,
        dts[0],
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("structured GPU explicit RK4 solver construction");
    let switch_error = solver
        .try_set_time_scheme(TimeScheme::BDF2)
        .expect_err("structured RK4-to-implicit switching requires reconstruction");
    assert!(switch_error.contains("rebuild"));
    solver.set_boundaries(|_edge, _x, _y| {
        (
            3,
            vec![BcComp {
                kind: 0,
                value: 0.0,
            }],
        )
    });

    // Uniform phi and zero-gradient faces remove diffusion, leaving the
    // declared autonomous ODE phi'=-4 phi. The reaction depends on the current
    // stage state, so this independently pins four residual evaluations in the
    // structured GPU frontend.
    let mut errors = Vec::with_capacity(dts.len());
    for &dt in &dts {
        solver.set_dt(dt);
        solver.set_named_field("phi", |_x, _y| 1.0);
        solver.set_named_field("ibm_penalty", |_x, _y| -4.0);
        for _ in 0..(t_end / dt).round() as usize {
            solver.step();
        }
        let phi = solver.state_field(solver.field_offset("phi").expect("phi offset"));
        let l2 = (phi
            .iter()
            .map(|&value| (value - exact).powi(2))
            .sum::<f64>()
            / phi.len() as f64)
            .sqrt();
        errors.push(l2);
    }

    let orders = observed_orders(&dts, &errors);
    eprintln!("[gpu-rk4-structured] dt={dts:?} error={errors:?} order={orders:?}");
    assert!(
        orders.iter().all(|&order| order > 3.5),
        "structured GPU RK4 temporal order below four: {orders:?}, errors={errors:?}"
    );
}

fn shared_gpu_context(ctx: &GpuContext) -> GpuContext {
    pollster::block_on(GpuContext::new(
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("shared GPU context")
}

/// Batching may change only the command routing, never the mathematical state
/// machine.  Exercise successive B=1,2,3 batches against B single-step batches
/// and pin both the accepted state and the two public history levels.
#[test]
fn gpu_structured_rk4_batches_match_single_steps_for_b1_b2_b3() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-batch-parity") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let model = generic_diffusion_demo_structured_ibm_model().expect("structured diffusion model");
    let grid = StructuredGrid::new(7, 5, 1.0, 1.0);
    let build = |context| {
        StructuredGpuSolver::with_config(
            context,
            grid,
            &model,
            0.01,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured RK4 solver")
    };
    let mut batched = build(ctx);
    let mut reference = build(reference_ctx);
    for solver in [&mut batched, &mut reference] {
        solver.set_boundaries(|_edge, _x, _y| {
            (
                3,
                vec![BcComp {
                    kind: 2,
                    value: 0.0,
                }],
            )
        });
        solver.set_named_field("phi", |x, y| {
            0.8 + 0.1
                * (2.0 * std::f64::consts::PI * x).cos()
                * (2.0 * std::f64::consts::PI * y).cos()
        });
        solver.set_named_field("ibm_penalty", |x, y| -1.0 - 0.25 * x + 0.1 * y);
    }

    let initial = batched.read_named("state", grid.num_cells() * 2);
    let initial_time = batched.time();
    assert!(batched.step_batch(0).is_none(), "B=0 must not submit");
    assert_eq!(
        batched.read_named("state", grid.num_cells() * 2),
        initial,
        "B=0 changed state"
    );
    assert_eq!(batched.time(), initial_time, "B=0 changed time");

    for batch in 1..=3usize {
        for _ in 0..batch {
            assert!(reference.step_batch(1).is_some());
        }
        assert!(batched.step_batch(batch).is_some());

        for name in ["state", "state_old", "state_old_old"] {
            let got = batched.read_named(name, grid.num_cells() * 2);
            let expected = reference.read_named(name, grid.num_cells() * 2);
            assert_eq!(
                got, expected,
                "B={batch} changed `{name}` relative to {batch} single-step batches"
            );
        }
        assert_eq!(
            batched.time().to_bits(),
            reference.time().to_bits(),
            "B={batch} accumulated a different physical time"
        );
    }
}

fn seed_structured_allmach_rk4(solver: &mut StructuredGpuSolver, cells: usize) {
    let stride = solver.state_size_bytes() as usize / (cells * std::mem::size_of::<f32>());
    let mut state = vec![0.0_f32; cells * stride];
    let mut set = |name: &str, value: f32| {
        if let Some(offset) = solver.field_offset(name) {
            for row in state.chunks_exact_mut(stride) {
                row[offset] = value;
            }
        }
    };
    let psi = 1.0e-4_f32;
    for (name, value) in [
        ("p", 0.0),
        ("T", 1.0),
        ("rho", 1.225),
        ("rho_t_ref", 1.225),
        ("rho_dT", -1.225),
        ("rho_floor", psi * 1.0e-5),
        ("t_ref", 1.0),
        ("psi_ref", psi),
        ("psi", psi),
        ("psi_precond", 0.25),
        ("u_ref", 2.0),
        ("precond_mask", 1.0),
        ("dt_local", 2.0e-4),
        ("d_p", 2.0e-4 / 1.225),
        ("ibm_penalty_U", 0.0),
    ] {
        set(name, value);
    }
    solver
        .set_packed_state_f32(&state)
        .expect("seed all-Mach packed state");
}

fn configure_structured_allmach_rk4(
    solver: &mut StructuredGpuSolver,
    grid: StructuredGrid,
    target: f32,
    ramp_duration: f32,
) {
    solver.set_fluid(1.225, 0.0);
    solver.set_inlet_ramp(target, ramp_duration);
    seed_structured_allmach_rk4(solver, grid.num_cells());
    solver.set_boundaries(|edge, _x, _y| match edge {
        cfd2::solver::gpu::structured::Edge::Left => (
            1,
            vec![
                BcComp {
                    kind: 1,
                    value: target,
                },
                BcComp {
                    kind: 1,
                    value: 0.0,
                },
                BcComp {
                    kind: 2,
                    value: 0.0,
                },
                BcComp {
                    kind: 1,
                    value: 1.0,
                },
            ],
        ),
        cfd2::solver::gpu::structured::Edge::Right => (
            2,
            vec![
                BcComp {
                    kind: 2,
                    value: 0.0,
                },
                BcComp {
                    kind: 2,
                    value: 0.0,
                },
                BcComp {
                    kind: 1,
                    value: 0.0,
                },
                BcComp {
                    kind: 2,
                    value: 0.0,
                },
            ],
        ),
        _ => (
            3,
            vec![
                BcComp {
                    kind: 1,
                    value: 0.0,
                },
                BcComp {
                    kind: 1,
                    value: 0.0,
                },
                BcComp {
                    kind: 2,
                    value: 0.0,
                },
                BcComp {
                    kind: 2,
                    value: 0.0,
                },
            ],
        ),
    });
}

fn host_structured_allmach_rates(
    solver: &StructuredGpuSolver,
    grid: StructuredGrid,
    state: &[f32],
) -> (f64, f64, f64) {
    let cells = grid.num_cells();
    let stride = state.len() / cells;
    let u = solver.field_offset("U").unwrap();
    let p = solver.field_offset("p").unwrap();
    let t = solver.field_offset("T").unwrap();
    let t_ref = solver.field_offset("t_ref").unwrap();
    let dt_local = solver.field_offset("dt_local").unwrap();
    let penalty = solver.field_offset("ibm_penalty_U").unwrap();
    let psi0 = 1.0e-4_f64;
    let density0 = 1.225_f64;
    let u_ref = 2.0_f64;
    let (gh, gd) = (
        1.0 / grid.dx + 1.0 / grid.dy,
        2.0 / grid.dx.powi(2) + 2.0 / grid.dy.powi(2),
    );
    let mut pressure = vec![0.0; cells];
    let mut density = vec![0.0; cells];
    let mut dp = vec![0.0; cells];
    let mut penalties = vec![0.0; cells];
    let mut max_base = 0.0_f64;
    let mut max_vel = 0.0_f64;
    for (cell, row) in state.chunks_exact(stride).enumerate() {
        let speed = (row[u] as f64).hypot(row[u + 1] as f64);
        let temperature = row[t] as f64;
        let reference_t = row[t_ref] as f64;
        let pressure_cell = row[p] as f64;
        let rho_numer = density0 * reference_t + 1.4 * psi0 * reference_t * pressure_cell;
        let rho = (rho_numer / temperature).max(psi0 * 1.0e-5);
        let rho_dt = -rho_numer / temperature.powi(2);
        let psi_local = (psi0 * reference_t / temperature).max(psi0);
        let beta2 = speed.powi(2).max(u_ref.powi(2)).max(1.0e-12);
        let mass_pp = 0.4 * psi0 * reference_t / temperature + psi_local.max(1.0 / beta2);
        let chi = mass_pp + rho_dt * (0.4 * psi0 * reference_t) / rho.max(1.0e-30);
        let sound = 1.0 / chi.sqrt();
        let nu_long = 0.0_f64;
        let alpha_t = 1.0e-2 * mass_pp / (rho * chi);
        let dp0 = (row[dt_local] as f64).max(0.0) / rho;
        let penalty_cell = (row[penalty] as f64).abs();
        let local_dp = dp0 / (1.0 + penalty_cell * dp0);
        let alpha_p = rho * local_dp.abs() / chi;
        let rate = gh * (speed + sound) + 2.0 * gd * nu_long.max(alpha_t).max(alpha_p);
        max_base = max_base.max(rate);
        max_vel = max_vel.max(speed);
        pressure[cell] = pressure_cell;
        density[cell] = rho;
        dp[cell] = local_dp;
        penalties[cell] = penalty_cell;
    }

    let at = |i: usize, j: usize| pressure[j * grid.nx + i];
    let mut gradient = vec![(0.0_f64, 0.0_f64); cells];
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let cell = j * grid.nx + i;
            let pw = if i > 0 {
                0.5 * (pressure[cell] + at(i - 1, j))
            } else {
                pressure[cell]
            };
            let pe = if i + 1 < grid.nx {
                0.5 * (pressure[cell] + at(i + 1, j))
            } else {
                0.0
            };
            let ps = if j > 0 {
                0.5 * (pressure[cell] + at(i, j - 1))
            } else {
                pressure[cell]
            };
            let pn = if j + 1 < grid.ny {
                0.5 * (pressure[cell] + at(i, j + 1))
            } else {
                pressure[cell]
            };
            gradient[cell] = ((pe - pw) / grid.dx, (pn - ps) / grid.dy);
        }
    }
    let mut row_sum = vec![0.0_f64; cells];
    let mut add = |a: usize, b: Option<usize>, flux: f64| {
        row_sum[a] += flux.abs();
        if let Some(b) = b {
            row_sum[b] += flux.abs();
        }
    };
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let cell = j * grid.nx + i;
            if i + 1 < grid.nx {
                let other = cell + 1;
                let seal = 1.0 - (penalties[cell] + penalties[other]).min(1.0);
                let kappa = 0.5 * (density[cell] + density[other]) * dp[cell].min(dp[other]);
                let q = kappa * 0.5 * (gradient[cell].0 + gradient[other].0);
                let compact = kappa * (pressure[other] - pressure[cell]) / grid.dx;
                add(cell, Some(other), seal * grid.dy * (q - compact));
            }
            if j + 1 < grid.ny {
                let other = cell + grid.nx;
                let seal = 1.0 - (penalties[cell] + penalties[other]).min(1.0);
                let kappa = 0.5 * (density[cell] + density[other]) * dp[cell].min(dp[other]);
                let q = kappa * 0.5 * (gradient[cell].1 + gradient[other].1);
                let compact = kappa * (pressure[other] - pressure[cell]) / grid.dy;
                add(cell, Some(other), seal * grid.dx * (q - compact));
            }
            if i + 1 == grid.nx {
                let kappa = density[cell] * dp[cell];
                let compact = kappa * (0.0 - pressure[cell]) / (0.5 * grid.dx);
                let seal = 1.0 - (2.0 * penalties[cell]).min(1.0);
                add(
                    cell,
                    None,
                    seal * grid.dy * (kappa * gradient[cell].0 - compact),
                );
            }
        }
    }
    let volume = grid.dx * grid.dy;
    let max_rc = row_sum
        .iter()
        .enumerate()
        .map(|(cell, sum)| sum / (density[cell] * volume))
        .fold(0.0_f64, f64::max);
    (max_base, max_rc, max_vel)
}

#[test]
fn gpu_structured_autonomous_q64_clock_is_exact_and_stage_times_round_once() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-q64-clock") else {
        return;
    };
    let model = generic_diffusion_demo_structured_ibm_model().expect("structured diffusion model");
    let solver = StructuredGpuSolver::with_config(
        ctx,
        StructuredGrid::new(2, 2, 1.0, 1.0),
        &model,
        1.0e-5,
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("structured Q64 probe solver");

    let fixed = solver
        .autonomous_clock_probe(false)
        .expect("fixed Q64 device probe");
    let fixed_dt = f32::from_bits(0x30a9ad7f);
    let fixed_base = f64::from(fixed_dt) * 100_003.0;
    let fixed_probe_dt = f32::from_bits(0x3157e37c);
    assert_eq!(fixed.time, fixed_base, "tiny-step Q64 accumulation drift");
    for (actual, a, label) in [
        (fixed.dt, 0.0_f64, "a=0"),
        (fixed.dt_old, 0.5, "a=1/2"),
        (fixed.max_base_rate, 1.0, "a=1"),
    ] {
        let expected = (fixed_base + a * f64::from(fixed_probe_dt)) as f32;
        assert_eq!(actual.to_bits(), expected.to_bits(), "fixed probe {label}");
    }
    let mut naive = 0.0_f32;
    for _ in 0..100_003 {
        naive += fixed_dt;
    }
    assert!((f64::from(naive) - fixed_base).abs() > 1.0e-8);

    let random = solver
        .autonomous_clock_probe(true)
        .expect("random Q64 device probe");
    let mut seed = 0x6d2b79f5_u32;
    let mut tiny_sum = 0.0_f64;
    for _ in 0..4096 {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let bits = 0x3089705f_u32 + (seed & 0x001f_ffff);
        tiny_sum += f64::from(f32::from_bits(bits));
    }
    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let random_probe_dt = f32::from_bits(0x3089705f_u32 + (seed & 0x001f_ffff));
    let random_base = 31.0 + f64::from(0xfffff000_u32) * (1.0 / 4_294_967_296.0)
        + tiny_sum;
    assert!(random_base > 32.0 && random_base < 32.000_01);
    assert_eq!(random.time, random_base, "random Q64 carry drift");
    for (actual, a, label) in [
        (random.dt, 0.0_f64, "a=0"),
        (random.dt_old, 0.5, "a=1/2"),
        (random.max_base_rate, 1.0, "a=1"),
    ] {
        let expected = (random_base + a * f64::from(random_probe_dt)) as f32;
        assert_eq!(actual.to_bits(), expected.to_bits(), "random probe {label}");
    }
}

fn configure_structured_compressible_rk4(
    solver: &mut StructuredGpuSolver,
    grid: StructuredGrid,
    inlet_velocity: f32,
    viscosity: f32,
) {
    solver.set_fluid(1.0, f64::from(viscosity));
    solver.set_eos(
        EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
            temperature: 1.0,
        }
        .runtime_params(),
    );
    solver.set_inlet_ramp(inlet_velocity, 0.0);

    let cells = grid.num_cells();
    let stride = solver.state_size_bytes() as usize / (cells * std::mem::size_of::<f32>());
    let mut state = vec![0.0_f32; cells * stride];
    let rho = solver.field_offset("rho").expect("rho offset");
    let rho_e = solver.field_offset("rho_e").expect("rho_e offset");
    let pressure = solver.field_offset("p").expect("pressure offset");
    let temperature = solver.field_offset("T").expect("temperature offset");
    for row in state.chunks_exact_mut(stride) {
        row[rho] = 1.0;
        row[rho_e] = 2.5;
        row[pressure] = 1.0;
        row[temperature] = 1.0;
    }
    solver
        .set_packed_state_f32(&state)
        .expect("seed compressible conserved state");

    let unknowns = solver.unknowns();
    solver.set_boundaries(move |_edge, _x, _y| {
        let mut values = vec![
            BcComp {
                kind: 0,
                value: 0.0,
            };
            unknowns
        ];
        values[0] = BcComp {
            kind: 1,
            value: 1.0,
        };
        if unknowns >= 4 {
            values[3] = BcComp {
                kind: 1,
                value: 2.5,
            };
        }
        (3, values)
    });
}

#[test]
fn gpu_structured_compressible_adaptive_batch_matches_one_step_submissions() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-compressible-adaptive") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let fixed_ctx = shared_gpu_context(&ctx);
    let model = compressible_structured_model().expect("structured compressible model");
    // A small physical grid makes the wave limit, rather than the always-present
    // IBM reaction ceiling, the active member of the host policy's min set.
    let grid = StructuredGrid::new(6, 3, 1.0e-4, 5.0e-5);
    let initial_dt = 1.0e-5_f64;
    let build = |context| {
        StructuredGpuSolver::with_config(
            context,
            grid,
            &model,
            initial_dt,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured compressible RK4 solver")
    };
    let mut batched = build(ctx);
    let mut reference = build(reference_ctx);
    let mut fixed = build(fixed_ctx);
    for solver in [&mut batched, &mut reference, &mut fixed] {
        configure_structured_compressible_rk4(solver, grid, 0.2, 0.0);
        assert!(solver.supports_autonomous_fixed());
        assert!(solver.supports_autonomous_adaptive());
    }

    fixed
        .step_autonomous_batch(1, None)
        .expect("fixed compressible autonomous submission");
    let fixed_status = fixed.autonomous_status().expect("fixed status");
    assert_eq!(fixed_status.accepted_steps, 1);
    assert!(!fixed_status.halted, "fixed audit halted: {fixed_status:?}");
    assert_eq!(fixed_status.max_base_rate.to_bits(), 0);
    assert_eq!(fixed_status.max_rhie_chow_rate.to_bits(), 0);
    assert_eq!(fixed_status.max_velocity.to_bits(), 0);

    batched
        .step_autonomous_batch(4, Some(0.5))
        .expect("compressible adaptive B=4 submission");
    let batched_status = batched.autonomous_status().expect("batched status");
    for _ in 0..4 {
        reference
            .step_autonomous_batch(1, Some(0.5))
            .expect("compressible adaptive B=1 submission");
        let status = reference.autonomous_status().expect("single-step status");
        assert!(!status.halted, "B=1 controller halted: {status:?}");
    }
    let reference_status = reference.autonomous_status().expect("reference tail");
    assert!(!batched_status.halted, "B=4 halted: {batched_status:?}");
    assert_eq!(batched_status.accepted_steps, 4);
    assert_eq!(reference_status.accepted_steps, 1);
    assert_eq!(batched_status.accepted_total, 4);
    assert_eq!(batched_status.accepted_total, reference_status.accepted_total);
    assert_eq!(batched_status.invalid_cells, reference_status.invalid_cells);
    assert_eq!(batched_status.time.to_bits(), reference_status.time.to_bits());
    assert_eq!(batched_status.dt.to_bits(), reference_status.dt.to_bits());
    assert_eq!(
        batched_status.dt_old.to_bits(),
        reference_status.dt_old.to_bits()
    );
    assert_eq!(
        batched_status.max_velocity.to_bits(),
        reference_status.max_velocity.to_bits()
    );

    let h = grid.dx.min(grid.dy) as f32;
    let wave_dt = 0.5 * h / (0.2 + 1.4_f32.sqrt());
    // The RK4 stages project rho_u to zero in solid cells, so the Brinkman
    // penalty no longer bounds the explicit dt (acoustic/diffusion only).
    let expected_dt = wave_dt.clamp(1.0e-9, 100.0);
    assert_eq!(
        batched_status.dt.to_bits(),
        expected_dt.to_bits(),
        "GPU next dt differs from host wave/diffusion/IBM min oracle"
    );
    assert_eq!(batched_status.max_velocity.to_bits(), 0.0_f32.to_bits());

    let state_len = batched.state_size_bytes() as usize / 4;
    for name in ["state", "state_old", "state_old_old"] {
        assert_eq!(
            batched.read_named(name, state_len),
            reference.read_named(name, state_len),
            "compressible adaptive B=4 changed {name} relative to B=1"
        );
    }
}

#[test]
fn gpu_structured_compressible_hot_state_cfl_uses_local_characteristic_speed() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-compressible-hot-local-cfl") else {
        return;
    };
    let model = compressible_structured_model().expect("structured compressible model");
    let grid = StructuredGrid::new(6, 4, 6.0e-4, 4.0e-4);
    let initial_dt = 1.0e-4_f64;
    let mut solver = StructuredGpuSolver::with_config(
        ctx,
        grid,
        &model,
        initial_dt,
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("hot structured compressible RK4 solver");
    configure_structured_compressible_rk4(&mut solver, grid, 0.0, 0.0);

    // Uniform hot ideal gas: p=100 at rho=1, so c_local=sqrt(gamma*p/rho)
    // is ten times the configured reference sqrt(gamma*theta_ref). Zero-gradient
    // boundaries keep this an exact freestream and isolate the timestep oracle.
    let cells = grid.num_cells();
    let stride = solver.state_size_bytes() as usize / (cells * std::mem::size_of::<f32>());
    let mut state = vec![0.0_f32; cells * stride];
    let rho = solver.field_offset("rho").expect("rho offset");
    let rho_e = solver.field_offset("rho_e").expect("rho_e offset");
    let pressure = solver.field_offset("p").expect("pressure offset");
    let temperature = solver.field_offset("T").expect("temperature offset");
    let hot_pressure = 100.0_f32;
    for row in state.chunks_exact_mut(stride) {
        row[rho] = 1.0;
        row[rho_e] = hot_pressure / 0.4;
        row[pressure] = hot_pressure;
        row[temperature] = hot_pressure;
    }
    solver
        .set_packed_state_f32(&state)
        .expect("hot conserved state seed");
    let unknowns = solver.unknowns();
    solver.set_boundaries(move |_edge, _x, _y| {
        (
            3,
            vec![
                BcComp {
                    kind: 2,
                    value: 0.0,
                };
                unknowns
            ],
        )
    });

    let target_cfl = 0.5_f32;
    solver
        .step_autonomous_batch(1, Some(target_cfl))
        .expect("hot-state adaptive submission");
    let status = solver.autonomous_status().expect("hot-state status");
    assert_eq!(status.accepted_steps, 1);
    assert!(!status.halted, "hot state rejected: {status:?}");

    let reference_sound = 1.4_f32.sqrt();
    let local_sound = (1.4_f32 * hot_pressure).sqrt();
    assert!(
        status.max_base_rate > 0.99 * local_sound,
        "accepted-state local characteristic was discarded: {status:?}"
    );
    let h = grid.dx.min(grid.dy) as f32;
    // No Brinkman dt bound: the stage projection keeps the penalty out of
    // the explicit stability spectrum (acoustic bounds only below).
    let reference_only_dt =
        (target_cfl * h / reference_sound).min(initial_dt as f32 * 1.2);
    assert!(
        status.dt_old < 0.5 * reference_only_dt,
        "hot accepted-step dt {} did not shrink below reference-only bound {reference_only_dt}",
        status.dt_old
    );
    assert!(
        status.dt < 0.5 * reference_only_dt,
        "hot next dt {} did not shrink below reference-only bound {reference_only_dt}",
        status.dt
    );
    let metric_dt = (target_cfl * h / status.max_base_rate)
        .min(status.dt_old * 1.2)
        .clamp(1.0e-9, 100.0);
    assert!(
        (status.dt - metric_dt).abs() <= metric_dt * 2.0e-5,
        "next dt {} does not follow accepted local characteristic {} (expected {metric_dt})",
        status.dt,
        status.max_base_rate
    );
}

#[test]
fn gpu_structured_compressible_rejects_finite_negative_internal_energy() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-compressible-negative-energy") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let model = compressible_structured_model().expect("structured compressible model");
    let grid = StructuredGrid::new(5, 3, 1.0, 0.6);
    let build = |context| {
        let mut solver = StructuredGpuSolver::with_config(
            context,
            grid,
            &model,
            1.0e-5,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured compressible RK4 solver");
        configure_structured_compressible_rk4(&mut solver, grid, 0.0, 0.0);
        solver
    };
    let mut batched = build(ctx);
    let mut reference = build(reference_ctx);
    reference
        .step_autonomous_batch(1, None)
        .expect("accepted-prefix reference step");
    let reference_status = reference.autonomous_status().expect("reference status");
    assert_eq!(reference_status.accepted_total, 1);

    batched.set_autonomous_test_negative_energy_after(Some(1));
    batched
        .step_autonomous_batch(5, None)
        .expect("finite-invalid candidate batch");
    let status = batched.autonomous_status().expect("finite-invalid status");
    assert!(
        status.halted,
        "negative internal energy was accepted: {status:?}"
    );
    assert!(status.invalid_cells > 0);
    assert_eq!(status.accepted_steps, 1);
    assert_eq!(status.accepted_total, 1);
    assert_eq!(status.time.to_bits(), reference_status.time.to_bits());
    assert_eq!(status.dt.to_bits(), reference_status.dt.to_bits());
    assert_eq!(status.dt_old.to_bits(), reference_status.dt_old.to_bits());

    let state_len = batched.state_size_bytes() as usize / 4;
    for name in ["state", "state_old", "state_old_old"] {
        assert_eq!(
            batched.read_named(name, state_len),
            reference.read_named(name, state_len),
            "finite-domain rejection did not restore accepted-prefix {name}"
        );
    }
}

#[test]
fn gpu_structured_autonomous_accepted_total_carries_across_u32_wrap() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-accepted-total-wrap") else {
        return;
    };
    let model = compressible_structured_model().expect("structured compressible model");
    let grid = StructuredGrid::new(3, 2, 1.0, 0.5);
    let mut solver = StructuredGpuSolver::with_config(
        ctx,
        grid,
        &model,
        1.0e-5,
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("structured compressible RK4 solver");
    configure_structured_compressible_rk4(&mut solver, grid, 0.0, 0.0);
    let before = u64::from(u32::MAX) - 1;
    solver.set_autonomous_test_accepted_total(before);
    solver
        .step_autonomous_batch(3, None)
        .expect("wrap-adjacent autonomous batch");
    let status = solver.autonomous_status().expect("wrap-adjacent status");
    assert!(!status.halted, "wrap probe halted: {status:?}");
    assert_eq!(status.accepted_steps, 3);
    assert_eq!(status.accepted_total, before + 3);
    solver.reconcile_autonomous_status(status);
    assert_eq!(solver.committed_steps(), before + 3);
}

#[test]
fn gpu_structured_autonomous_status_ring_preserves_three_inflight_tails() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-status-ring") else {
        return;
    };
    let model = compressible_structured_model().expect("structured compressible model");
    let grid = StructuredGrid::new(3, 2, 1.0, 0.5);
    let mut solver = StructuredGpuSolver::with_config(
        ctx,
        grid,
        &model,
        1.0e-5,
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("structured compressible RK4 solver");
    configure_structured_compressible_rk4(&mut solver, grid, 0.0, 0.0);

    let (tx, rx) = std::sync::mpsc::channel();
    for serial in 0..3_u32 {
        let tx = tx.clone();
        solver
            .submit_autonomous_batch(1, None, move |status| {
                tx.send((serial, status)).expect("send ring status");
            })
            .expect("three-slot submission");
    }
    let overflow = solver.submit_autonomous_batch(1, None, |_| {});
    assert!(
        overflow
            .expect_err("a fourth unmapped tail must not alias a busy staging slot")
            .contains("status readback ring is full")
    );

    let start = std::time::Instant::now();
    let mut tails = Vec::new();
    while tails.len() < 3 {
        solver
            .poll_gpu_completions()
            .expect("poll three-slot completions");
        while let Ok(tail) = rx.try_recv() {
            tails.push(tail);
        }
        assert!(start.elapsed().as_secs_f32() < 5.0, "status ring timed out");
        std::thread::yield_now();
    }
    tails.sort_by_key(|(serial, _)| *serial);
    for (serial, status) in tails {
        let status = status.expect("mapped ring status");
        assert!(!status.halted, "ring tail {serial} halted: {status:?}");
        assert_eq!(status.accepted_steps, 1);
        assert_eq!(status.accepted_total, u64::from(serial + 1));
    }

    // Mapping/unmapping released all three slots; the next submission can
    // immediately reuse one without allocating another buffer.
    solver
        .submit_autonomous_batch(1, None, |_| {})
        .expect("status slot reuse after callback");
}

#[test]
fn gpu_structured_adaptive_batch_matches_one_step_submissions() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-adaptive-batch") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let fixed_ctx = shared_gpu_context(&ctx);
    let model = allmach_thermal_structured_model().expect("structured all-Mach model");
    let grid = StructuredGrid::new(6, 3, 1.0, 0.5);
    let dt = 2.0e-4_f64;
    let build = |context| {
        StructuredGpuSolver::with_config(
            context,
            grid,
            &model,
            dt,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured all-Mach RK4 solver")
    };
    let mut batched = build(ctx);
    let mut reference = build(reference_ctx);
    let mut fixed = build(fixed_ctx);
    for solver in [&mut batched, &mut reference, &mut fixed] {
        configure_structured_allmach_rk4(solver, grid, 0.2, 0.001);
        solver.set_named_field("p", |x, y| {
            0.02 * (2.0 * std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y / 0.5).cos()
        });
    }

    fixed
        .step_autonomous_batch(1, None)
        .expect("fixed all-Mach autonomous submission");
    let fixed_status = fixed.autonomous_status().expect("fixed all-Mach status");
    assert_eq!(fixed_status.accepted_steps, 1);
    assert!(
        !fixed_status.halted,
        "fixed health audit halted: {fixed_status:?}"
    );
    assert_eq!(fixed_status.max_base_rate.to_bits(), 0);
    assert_eq!(fixed_status.max_rhie_chow_rate.to_bits(), 0);
    assert_eq!(fixed_status.max_velocity.to_bits(), 0);

    batched
        .step_autonomous_batch(3, Some(0.5))
        .expect("adaptive B=3 submission");
    let batched_status = batched.autonomous_status().expect("batched status");
    for _ in 0..3 {
        reference
            .step_autonomous_batch(1, Some(0.5))
            .expect("adaptive B=1 submission");
        let status = reference.autonomous_status().expect("single-step status");
        assert!(
            !status.halted,
            "single-step adaptive controller halted: {status:?}"
        );
    }
    let reference_status = reference.autonomous_status().expect("reference status");
    assert!(
        !batched_status.halted,
        "batched controller halted: {batched_status:?}"
    );
    assert_eq!(batched_status.accepted_steps, 3);
    assert_eq!(batched_status.accepted_total, 3);
    assert_eq!(reference_status.accepted_total, 3);
    assert_eq!(
        batched_status.time.to_bits(),
        reference_status.time.to_bits(),
        "adaptive time differs"
    );
    for (a, b, label) in [
        (batched_status.dt_old, reference_status.dt_old, "last dt"),
        (batched_status.dt, reference_status.dt, "next dt"),
        (
            batched_status.max_base_rate,
            reference_status.max_base_rate,
            "base rate",
        ),
        (
            batched_status.max_rhie_chow_rate,
            reference_status.max_rhie_chow_rate,
            "Rhie-Chow rate",
        ),
    ] {
        assert_eq!(a.to_bits(), b.to_bits(), "adaptive {label} differs");
    }
    let state_len = batched.state_size_bytes() as usize / 4;
    let batched_state = batched.read_named("state", state_len);
    let (host_base, host_rc, host_max_vel) =
        host_structured_allmach_rates(&batched, grid, &batched_state);
    let close = |gpu: f32, host: f64, label: &str| {
        let scale = host.abs().max(1.0);
        assert!(
            ((gpu as f64 - host).abs() / scale) < 2.0e-5,
            "GPU {label} {gpu:e} differs from host oracle {host:e}"
        );
    };
    close(batched_status.max_base_rate, host_base, "base rate");
    close(batched_status.max_rhie_chow_rate, host_rc, "Rhie-Chow rate");
    close(batched_status.max_velocity, host_max_vel, "max velocity");
    assert!(
        host_rc > 0.0,
        "nonuniform pressure did not exercise RC turnover"
    );
    let expected_next = (0.8 * 0.5 / (host_base + host_rc))
        .min(1.2 * batched_status.dt_old as f64)
        .min(100.0);
    close(batched_status.dt, expected_next, "next dt");
    for name in ["state", "state_old", "state_old_old"] {
        assert_eq!(
            batched.read_named(name, state_len),
            reference.read_named(name, state_len),
            "adaptive B=3 changed {name}"
        );
    }
}

#[test]
fn gpu_structured_autonomous_nan_halts_at_exact_accepted_prefix() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-autonomous-nan") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let model = generic_diffusion_demo_structured_ibm_model().expect("structured diffusion model");
    let grid = StructuredGrid::new(7, 5, 1.0, 1.0);
    let build = |context| {
        let mut solver = StructuredGpuSolver::with_config(
            context,
            grid,
            &model,
            0.01,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured RK4 solver");
        solver.set_boundaries(|_edge, _x, _y| {
            (
                3,
                vec![BcComp {
                    kind: 2,
                    value: 0.0,
                }],
            )
        });
        solver.set_named_field("phi", |x, y| 0.9 + 0.05 * x - 0.03 * y);
        solver.set_named_field("ibm_penalty", |_x, _y| -2.0);
        solver
    };
    let mut batched = build(ctx);
    let mut reference = build(reference_ctx);
    reference
        .step_autonomous_batch(1, None)
        .expect("accepted-prefix step");
    let reference_status = reference.autonomous_status().expect("reference status");
    assert_eq!(reference_status.accepted_total, 1);

    batched.set_autonomous_test_nan_after(Some(1));
    let (tx, rx) = std::sync::mpsc::channel();
    batched
        .submit_autonomous_batch(5, None, move |status| {
            tx.send(status).expect("send completion status");
        })
        .expect("injected-invalid batch submission");
    let start = std::time::Instant::now();
    let batched_status = loop {
        batched
            .poll_gpu_completions()
            .expect("nonblocking completion poll");
        match rx.try_recv() {
            Ok(status) => break status.expect("mapped completion status"),
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                assert!(
                    start.elapsed().as_secs_f32() < 5.0,
                    "status callback timed out"
                );
                std::thread::yield_now();
            }
            Err(error) => panic!("status channel failed: {error}"),
        }
    };
    assert!(
        batched_status.halted,
        "injected NaN did not halt: {batched_status:?}"
    );
    assert!(batched_status.invalid_cells > 0);
    assert_eq!(batched_status.accepted_steps, 1);
    assert_eq!(batched_status.accepted_total, 1);
    assert_eq!(
        batched_status.time.to_bits(),
        reference_status.time.to_bits()
    );
    assert_eq!(batched_status.dt.to_bits(), reference_status.dt.to_bits());
    assert_eq!(
        batched_status.dt_old.to_bits(),
        reference_status.dt_old.to_bits()
    );

    let state_len = batched.state_size_bytes() as usize / 4;
    for name in ["state", "state_old", "state_old_old"] {
        assert_eq!(
            batched.read_named(name, state_len),
            reference.read_named(name, state_len),
            "halted batch did not preserve accepted-prefix {name}"
        );
    }
    batched.reconcile_autonomous_status(batched_status);
    reference.reconcile_autonomous_status(reference_status);
    assert_eq!(batched.time().to_bits(), reference.time().to_bits());
    assert_eq!(batched.dt().to_bits(), reference.dt().to_bits());
    assert_eq!(batched.committed_steps(), reference.committed_steps());

    // Switching back to ordinary stepping makes the host clock authoritative.
    // The next autonomous entry must reseed from that clock rather than resume
    // the stale pre-ordinary device controller time.
    batched.set_autonomous_test_nan_after(None);
    batched.step();
    let after_ordinary = batched.time();
    let ordinary_dt = batched.dt();
    batched
        .step_autonomous_batch(1, None)
        .expect("autonomous restart after ordinary step");
    let restarted = batched.autonomous_status().expect("restarted status");
    let expected_restart = after_ordinary + ordinary_dt;
    assert!(
        (restarted.time - expected_restart).abs() < 1.0e-12,
        "autonomous Q64 clock resumed stale tail: got {:.12}, expected {:.12}",
        restarted.time,
        expected_restart
    );
}

/// The production all-Mach inlet is a generated expression of
/// `constants.time`.  It is the critical non-autonomous case: a batch must make
/// each RK abscissa visible on-device rather than freezing the first or final
/// host uniform for all dispatches.
#[test]
fn gpu_structured_rk4_batch_preserves_time_dependent_inlet_stages() {
    let Some(ctx) = gpu_context("gpu-rk4-structured-time-dependent-batch") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let model = allmach_thermal_structured_model().expect("structured all-Mach model");
    let grid = StructuredGrid::new(6, 3, 1.0, 0.5);
    let dt = 2.0e-4_f64;
    let batch = 3usize;
    let target = 0.2_f32;
    let ramp_duration = (batch as f64 * dt) as f32;
    let build = |context| {
        StructuredGpuSolver::with_config(
            context,
            grid,
            &model,
            dt,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured all-Mach RK4 solver")
    };
    let mut batched = build(ctx);
    let mut reference = build(reference_ctx);
    let unknowns = model.system.unknowns_per_cell() as usize;

    for solver in [&mut batched, &mut reference] {
        solver.set_fluid(1.225, 0.0);
        solver.set_inlet_ramp(target, ramp_duration);
        seed_structured_allmach_rk4(solver, grid.num_cells());
        solver.set_boundaries(|edge, _x, _y| match edge {
            cfd2::solver::gpu::structured::Edge::Left => (
                1,
                vec![
                    BcComp {
                        kind: 1,
                        value: target,
                    },
                    BcComp {
                        kind: 1,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 2,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 1,
                        value: 1.0,
                    },
                ],
            ),
            cfd2::solver::gpu::structured::Edge::Right => (
                2,
                vec![
                    BcComp {
                        kind: 2,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 2,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 1,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 2,
                        value: 0.0,
                    },
                ],
            ),
            _ => (
                3,
                vec![
                    BcComp {
                        kind: 1,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 1,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 2,
                        value: 0.0,
                    },
                    BcComp {
                        kind: 2,
                        value: 0.0,
                    },
                ],
            ),
        });
    }

    for _ in 0..batch {
        assert!(reference.step_batch(1).is_some());
    }
    assert!(batched.step_batch(batch).is_some());

    let len = grid.num_cells() * (batched.state_size_bytes() as usize / 4 / grid.num_cells());
    let got = batched.read_named("state", len);
    let expected = reference.read_named("state", len);
    assert_eq!(
        got, expected,
        "time-dependent B=3 state differs from singles"
    );
    assert!(got.iter().all(|value| value.is_finite()));
    assert_eq!(batched.time().to_bits(), reference.time().to_bits());

    // Direction W=1 on the first left-edge cell; component 0 is inlet Ux.
    // The last encoded stage is at t=batch*dt, where the smootherstep reaches
    // exactly its target. This rejects a frozen t=0 uniform even if both state
    // trajectories happened to stay close over this short stability probe.
    let bc = batched.read_named("bc_value", grid.num_cells() * 4 * unknowns);
    let left_ux = bc[unknowns];
    assert!(
        (left_ux - target).abs() < 2.0e-6,
        "final stage-time inlet value {left_ux:e} did not reach target {target:e}"
    );
}
