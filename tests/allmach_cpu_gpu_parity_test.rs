//! CPU vs GPU parity for the pressure-based all-Mach thermal solver.
//!
//! The critic flagged that no CPU/GPU parity gate existed for the `allmach` path
//! (the only compressible parity test was density-based and `#[ignore]`'d). This
//! drives the SAME `allmach_thermal` model through the CPU backend (f64) and the GPU
//! backend (f32) for a handful of steps and checks the fields agree to an f32-level
//! tolerance — catching a codegen divergence between the two lowerings of the new
//! compressible physics (real-EOS rho recovery, viscous dissipation, local sound
//! speed, the psi/psi_precond recoveries). Skips gracefully when no GPU adapter is
//! available (CPU-only CI), so it never blocks.
#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::allmach_thermal_model;
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;

fn air() -> Fluid {
    Fluid::presets()[1].clone()
}

/// A small closed channel with an inlet on the left; the flow develops and exercises
/// the full coupled compressible path (Rhie–Chow, EOS recovery, block solve) every
/// step, without needing a huge mesh.
fn mesh() -> Mesh {
    generate_structured_rect_mesh(
        16,
        12,
        1.0,
        0.75,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

/// Build an `allmach_thermal` solver on the requested backend (forced-CPU or the
/// default GPU). Returns `None` if backend construction fails (e.g. no GPU adapter).
fn build(m: &Mesh, forced_cpu: bool) -> Option<UnifiedSolver> {
    let air = air();
    let mut d = gui_defaults_for("allmach_pressure");
    d.inlet_velocity = 0.05;
    // Unfloor the preconditioner reference velocity: the GUI ALLMACH preset raises
    // `allmach_precond_uref_min` to 1.0 (a MOVING-MESH pressure-quality knob — it lowers
    // psi_precond so the developed pressure is nearly elliptic). This STATIC 20-step
    // backend-parity check has no moving mesh, and the lower psi_precond leaves the gauge
    // pressure (a Lagrange multiplier ~O(1e-4)) less damped, so f32-vs-f64 rounding is a
    // larger fraction of it — tripping the demeaned-pressure sanity bound even though the
    // physical fields still agree to 1e-4. Pin the floor OFF so this test measures the
    // backend lowering, not the preconditioner-floor regime.
    d.allmach_precond_uref_min = 0.0;
    let params = d.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);
    let n = m.num_cells();
    let u0 = vec![(0.05_f64, 0.0); n];
    let p0 = vec![0.0; n];
    let built: DriverBuild = if forced_cpu {
        pollster::block_on(SolverDriver::build_forced_cpu(
            m,
            allmach_thermal_model().ok()?,
            &params,
            &u0,
            &p0,
        ))
        .ok()?
    } else {
        pollster::block_on(SolverDriver::build(
            m,
            allmach_thermal_model().ok()?,
            &params,
            &u0,
            &p0,
            None,
            None,
        ))
        .ok()?
    };
    let mut driver = built.driver;
    driver.apply_params(&params);
    Some(driver.into_solver())
}

fn read(s: &UnifiedSolver, field: &str) -> Vec<f64> {
    pollster::block_on(s.get_field_scalar(field)).unwrap_or_else(|_| panic!("read {field}"))
}

#[test]
fn allmach_thermal_cpu_gpu_parity() {
    let m = mesh();

    let mut cpu = build(&m, true).expect("forced-CPU allmach_thermal build");
    let gpu = match build(&m, false) {
        Some(g) => g,
        None => {
            eprintln!("[allmach-parity] no GPU adapter — skipping CPU/GPU parity check");
            return;
        }
    };
    let mut gpu = gpu;

    const STEPS: usize = 20;
    for _ in 0..STEPS {
        cpu.step_with_stats().expect("cpu step");
        gpu.step_with_stats().expect("gpu step");
    }

    // Compare the solved fields. GPU is f32, CPU is f64, so use an f32-scale relative
    // tolerance rather than a byte match.
    let mag = |v: &[f64]| v.iter().fold(1e-6f64, |a, &b| a.max(b.abs()));
    let field_diff = |c: &[f64], g: &[f64], demean: bool| -> f64 {
        assert_eq!(c.len(), g.len(), "field length mismatch");
        // The pressure is a gauge / Lagrange-multiplier field — each backend's iterative
        // solve settles to its own additive gauge — so compare it demeaned.
        let (oc, og) = if demean {
            (c.iter().sum::<f64>() / c.len() as f64, g.iter().sum::<f64>() / g.len() as f64)
        } else {
            (0.0, 0.0)
        };
        let scale = if demean {
            c.iter().fold(1e-6f64, |a, &b| a.max((b - oc).abs()))
        } else {
            mag(c)
        };
        c.iter()
            .zip(g.iter())
            .map(|(a, b)| {
                assert!(a.is_finite() && b.is_finite(), "non-finite field value");
                ((a - oc) - (b - og)).abs() / scale
            })
            .fold(0.0f64, f64::max)
    };

    let u_cpu = pollster::block_on(cpu.get_field_vec2("U")).expect("U cpu");
    let u_gpu = pollster::block_on(gpu.get_field_vec2("U")).expect("U gpu");
    let u_scale = u_cpu.iter().fold(1e-6f64, |a, (x, y)| a.max(x.abs()).max(y.abs()));
    let u_diff = u_cpu
        .iter()
        .zip(u_gpu.iter())
        .map(|((cx, cy), (gx, gy))| (cx - gx).abs().max((cy - gy).abs()) / u_scale)
        .fold(0.0f64, f64::max);

    let t_diff = field_diff(&read(&cpu, "T"), &read(&gpu, "T"), false);
    let rho_diff = field_diff(&read(&cpu, "rho"), &read(&gpu, "rho"), false);
    let p_diff = field_diff(&read(&cpu, "p"), &read(&gpu, "p"), true); // demeaned gauge

    println!(
        "[allmach-parity] {STEPS} steps, relative CPU/GPU diff:  U={u_diff:.3e}  T={t_diff:.3e}  \
         rho={rho_diff:.3e}  p(demeaned)={p_diff:.3e}"
    );

    // Physical fields (gauge-independent) must agree to the f32-vs-f64 envelope: a few
    // e-3 relative over 20 coupled steps with iterative solves. A larger diff is a real
    // codegen divergence between the two backend lowerings of the compressible physics.
    assert!(
        u_diff < 5.0e-3 && t_diff < 5.0e-3 && rho_diff < 5.0e-3,
        "CPU/GPU allmach_thermal physical fields diverged: U={u_diff:.3e} T={t_diff:.3e} rho={rho_diff:.3e}"
    );
    // The gauge pressure is far more sensitive — it is a tiny perturbation field
    // (~O(rho*U^2)) AND a Lagrange multiplier, so f32-vs-f64 noise is a large fraction
    // of its small scale (here ~8e-2 relative = ~1e-4 absolute) even though the physical
    // fields agree to ~1e-5. A loose sanity bound catches a gross divergence (blow-up /
    // NaN / wrong structure) without failing on the expected f32 pressure noise.
    assert!(
        p_diff < 2.0e-1,
        "CPU/GPU allmach_thermal pressure (demeaned) grossly diverged: {p_diff:.3e}"
    );
}
