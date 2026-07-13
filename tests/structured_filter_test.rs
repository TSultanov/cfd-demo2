//! Gates for the selective conserved-field low-pass filter on the structured
//! explicit RK4 path (`set_filter_sigma`): a dimension-split binomial filter
//! over `rho`/`rho_u`/`rho_e` that damps grid-Nyquist modes by `sigma` per
//! pass, leaves resolved scales untouched, never modifies immersed-solid
//! cells, and reduces its stencil near solids and domain boundaries.
//!
//! Route coverage mirrors `gpu_explicit_rk4_test.rs`: the plot batched route
//! (`step_batch`), the GPU-resident autonomous route
//! (`step_autonomous_batch`), and the hand-written CPU implementation.
//! A missing headless adapter is an allowed platform skip.
#![cfg(feature = "dev-tests")]

use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::structured::{BcComp, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::model::compressible_structured_model;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;

const SIGMA: f32 = 0.2;

fn gpu_context(gate: &str) -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(error) => {
            eprintln!("[{gate}] no compatible GPU adapter ({error}); skipping GPU gate");
            None
        }
    }
}

fn shared_gpu_context(ctx: &GpuContext) -> GpuContext {
    pollster::block_on(GpuContext::new(
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("shared GPU context")
}

fn nondimensional_air() -> EosSpec {
    EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 1.0,
        temperature: 1.0,
    }
}

fn build_gpu(ctx: GpuContext, grid: StructuredGrid, dt: f64) -> StructuredGpuSolver {
    StructuredGpuSolver::with_config(ctx, grid, &compressible_structured_model()
        .expect("structured compressible model"), dt, 1, Scheme::Upwind, TimeScheme::RK4)
    .expect("structured compressible RK4 solver")
}

/// Uniform gas at rest (rho=1, p=1, T=1, rho_e=2.5) with walls that pin the
/// conserved density/energy — the same known-stable configuration the RK4
/// route-parity gates use.
fn configure_gpu(solver: &mut StructuredGpuSolver) {
    solver.set_fluid(1.0, 0.0);
    solver.set_eos(nondimensional_air().runtime_params());
    solver.set_inlet_ramp(0.0, 0.0);
    for (name, value) in [("rho", 1.0), ("rho_e", 2.5), ("p", 1.0), ("T", 1.0)] {
        solver.set_named_field(name, move |_, _| value);
    }
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

/// Seed the x-momentum from a closure of the integer cell coordinates,
/// identically reproducible on the CPU solver (same f64 arithmetic path).
fn seed_rho_u_x_gpu<F: Fn(usize, usize) -> f64 + 'static>(
    solver: &mut StructuredGpuSolver,
    grid: StructuredGrid,
    f: F,
) {
    let offset = solver.field_offset("rho_u").expect("rho_u offset");
    let (dx, dy) = (grid.dx, grid.dy);
    solver.set_state_component(offset, move |x, y| {
        f((x / dx).floor() as usize, (y / dy).floor() as usize)
    });
}

fn state_bits(solver: &StructuredGpuSolver, name: &str) -> Vec<u32> {
    let len = solver.state_size_bytes() as usize / 4;
    solver
        .read_named(name, len)
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// One step with sigma=0.2 must multiply the interior grid-Nyquist amplitude
/// of a seeded odd-even x-momentum checkerboard by ~(1 - sigma) relative to a
/// sigma=0 run of the same step (the transfer at k*h = pi is exactly
/// 1 - sigma for every stencil order; the y-pass is inert on an x-mode).
#[test]
fn structured_filter_damps_nyquist_by_sigma_per_step() {
    let Some(ctx) = gpu_context("structured-filter-nyquist") else {
        return;
    };
    let filtered_ctx = shared_gpu_context(&ctx);
    let grid = StructuredGrid::new(48, 12, 1.0, 0.25);
    let eps = 1.0e-3_f64;
    let pert = move |i: usize, _j: usize| if i % 2 == 0 { eps } else { -eps };

    let mut baseline = build_gpu(ctx, grid, 1.0e-4);
    let mut filtered = build_gpu(filtered_ctx, grid, 1.0e-4);
    for solver in [&mut baseline, &mut filtered] {
        configure_gpu(solver);
        seed_rho_u_x_gpu(solver, grid, pert);
    }
    filtered.set_filter_sigma(SIGMA);
    assert!(baseline.step_batch(1).is_some());
    assert!(filtered.step_batch(1).is_some());

    let offset = baseline.field_offset("rho_u").expect("rho_u offset");
    let amplitude = |solver: &StructuredGpuSolver| -> f64 {
        let field = solver.state_field(offset);
        let mut sum = 0.0;
        let mut count = 0usize;
        // Interior window: full 8-cell x-clearance and 4-cell y-clearance.
        for j in 4..8 {
            for i in 8..40 {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sum += sign * field[j * grid.nx + i];
                count += 1;
            }
        }
        sum / count as f64
    };
    let amp_baseline = amplitude(&baseline);
    let amp_filtered = amplitude(&filtered);
    assert!(
        amp_baseline > 0.5 * eps,
        "RK step destroyed the seeded Nyquist mode: baseline amplitude {amp_baseline:e}"
    );
    let ratio = amp_filtered / amp_baseline;
    eprintln!(
        "[structured-filter-nyquist] baseline={amp_baseline:e} filtered={amp_filtered:e} ratio={ratio}"
    );
    assert!(
        (ratio - f64::from(1.0 - SIGMA)).abs() < 0.1,
        "Nyquist damping ratio {ratio} is not ~{} (baseline {amp_baseline:e}, filtered {amp_filtered:e})",
        1.0 - SIGMA
    );
}

/// A resolved long-wavelength mode (48 cells/wavelength >= the 16-cell floor)
/// must pass through the filter essentially untouched: the filtered and
/// unfiltered runs of the same step differ by < 1e-3 of the perturbation
/// amplitude at EVERY cell (interior 8th-order transfer ~1e-9; even the
/// boundary 4th-order stencils are ~4e-6).
#[test]
fn structured_filter_leaves_resolved_scales_untouched() {
    let Some(ctx) = gpu_context("structured-filter-smooth") else {
        return;
    };
    let filtered_ctx = shared_gpu_context(&ctx);
    let grid = StructuredGrid::new(48, 12, 1.0, 0.25);
    let eps = 1.0e-3_f64;
    let nx = grid.nx as f64;
    let pert = move |i: usize, _j: usize| {
        eps * (2.0 * std::f64::consts::PI * (i as f64 + 0.5) / nx).sin()
    };

    let mut baseline = build_gpu(ctx, grid, 1.0e-4);
    let mut filtered = build_gpu(filtered_ctx, grid, 1.0e-4);
    for solver in [&mut baseline, &mut filtered] {
        configure_gpu(solver);
        seed_rho_u_x_gpu(solver, grid, pert);
    }
    filtered.set_filter_sigma(SIGMA);
    assert!(baseline.step_batch(1).is_some());
    assert!(filtered.step_batch(1).is_some());

    let offset = baseline.field_offset("rho_u").expect("rho_u offset");
    let base_field = baseline.state_field(offset);
    let filt_field = filtered.state_field(offset);
    let max_diff = base_field
        .iter()
        .zip(&filt_field)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("[structured-filter-smooth] max relative change {:e}", max_diff / eps);
    assert!(
        max_diff / eps < 1.0e-3,
        "filter touched a resolved 48-cell mode by {:e} of its amplitude",
        max_diff / eps
    );
}

/// Immersed-solid protection: solid cells (ibm_penalty_U < 0) are never
/// modified by the filter (their conserved state stays bit-identical to the
/// sigma=0 run of the same step), the filtered state stays finite, and fluid
/// cells near the solid block demonstrably use reduced stencils — on an
/// 8-cell-wavelength mode the 4th-order near-ring stencil damps ~46x harder
/// than the interior 8th-order one, and the 1-cell ring is identity.
#[test]
fn structured_filter_protects_solid_cells_and_reduces_stencils_nearby() {
    let Some(ctx) = gpu_context("structured-filter-solid") else {
        return;
    };
    let filtered_ctx = shared_gpu_context(&ctx);
    let grid = StructuredGrid::new(40, 16, 1.0, 0.4);
    let (nx, ny) = (grid.nx, grid.ny);
    // Solid block: i in 16..20, j in 6..10.
    let solid = |i: usize, j: usize| (16..20).contains(&i) && (6..10).contains(&j);
    let eps = 0.05_f64;
    // 8-cell wavelength: strongly separated N=4 vs N=2 responses, well below
    // Nyquist so the responses genuinely differ per stencil order.
    let wave = move |i: usize| eps * (std::f64::consts::PI * i as f64 / 4.0).cos();

    let mut baseline = build_gpu(ctx, grid, 1.0e-5);
    let mut filtered = build_gpu(filtered_ctx, grid, 1.0e-5);
    for solver in [&mut baseline, &mut filtered] {
        configure_gpu(solver);
        let (dx, dy) = (grid.dx, grid.dy);
        let cell_of = move |x: f64, y: f64| -> (usize, usize) {
            ((x / dx).floor() as usize, (y / dy).floor() as usize)
        };
        // Solid sentinels on every conserved component + the Brinkman mask.
        let rho_off = solver.field_offset("rho").expect("rho offset");
        let rho_u_off = solver.field_offset("rho_u").expect("rho_u offset");
        let rho_e_off = solver.field_offset("rho_e").expect("rho_e offset");
        solver.set_state_component(rho_off, move |x, y| {
            let (i, j) = cell_of(x, y);
            if solid(i, j) {
                3.0
            } else {
                1.0
            }
        });
        solver.set_state_component(rho_u_off, move |x, y| {
            let (i, j) = cell_of(x, y);
            if solid(i, j) {
                0.5
            } else {
                wave(i)
            }
        });
        solver.set_state_component(rho_u_off + 1, move |x, y| {
            let (i, j) = cell_of(x, y);
            if solid(i, j) {
                -0.25
            } else {
                0.0
            }
        });
        solver.set_state_component(rho_e_off, move |x, y| {
            let (i, j) = cell_of(x, y);
            if solid(i, j) {
                9.0
            } else {
                2.5
            }
        });
        let penalty_off = solver
            .field_offset("ibm_penalty_U")
            .expect("structured compressible model must carry ibm_penalty_U");
        solver.set_state_component(penalty_off, move |x, y| {
            let (i, j) = cell_of(x, y);
            if solid(i, j) {
                -1.0e5
            } else {
                0.0
            }
        });
    }
    filtered.set_filter_sigma(SIGMA);
    assert!(baseline.step_batch(1).is_some());
    assert!(filtered.step_batch(1).is_some());

    let stride = baseline.state_size_bytes() as usize / (nx * ny * 4);
    let base_state = baseline.read_named("state", nx * ny * stride);
    let filt_state = filtered.read_named("state", nx * ny * stride);
    assert!(
        filt_state.iter().all(|v| v.is_finite()),
        "filtered state went non-finite near the solid block"
    );

    let rho_off = baseline.field_offset("rho").expect("rho offset");
    let rho_u_off = baseline.field_offset("rho_u").expect("rho_u offset");
    let rho_e_off = baseline.field_offset("rho_e").expect("rho_e offset");
    for j in 0..ny {
        for i in 0..nx {
            if !solid(i, j) {
                continue;
            }
            let cell = j * nx + i;
            for comp in [rho_off, rho_u_off, rho_u_off + 1, rho_e_off] {
                let a = base_state[cell * stride + comp];
                let b = filt_state[cell * stride + comp];
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "filter modified solid cell ({i},{j}) component {comp}: {a} -> {b}"
                );
            }
        }
    }

    // Filter-only effect per cell (both runs share the identical pre-filter
    // state, so the difference isolates the filter's stencil order).
    let diff_at = |i: usize, j: usize| -> f64 {
        let cell = j * nx + i;
        f64::from(
            (filt_state[cell * stride + rho_u_off] - base_state[cell * stride + rho_u_off]).abs(),
        )
    };
    let diff_far = diff_at(8, 8); // clearance 4 (full 8th-order stencil)
    let diff_ring = diff_at(13, 8); // 3 cells from the block: clearance 2
    let diff_identity = diff_at(15, 8); // 1 cell from the block: identity in x
    eprintln!(
        "[structured-filter-solid] far={diff_far:e} ring={diff_ring:e} identity={diff_identity:e}"
    );
    assert!(
        diff_far > 1.0e-7,
        "far-field filter response missing: {diff_far:e}"
    );
    assert!(
        diff_ring > 2.0 * diff_far,
        "near-solid reduced stencil did not bite harder on the 8-cell mode: ring {diff_ring:e} vs far {diff_far:e}"
    );
    assert!(
        diff_identity < 0.2 * diff_ring,
        "1-cell ring should be identity in x: {diff_identity:e} vs ring {diff_ring:e}"
    );
}

/// Plot-route batching parity with the filter ON: `step_batch(4)` must be
/// bit-identical to four `step_batch(1)` submissions, including both public
/// history levels (mirror of `gpu_structured_rk4_batches_match_single_steps...`
/// with sigma = 0.2).
#[test]
fn gpu_structured_rk4_plot_batches_match_single_steps_with_filter_on() {
    let Some(ctx) = gpu_context("structured-filter-plot-batch-parity") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let grid = StructuredGrid::new(24, 10, 1.0, 0.4);
    let eps = 1.0e-3_f64;
    let pert = move |i: usize, _j: usize| if i % 2 == 0 { eps } else { -eps };

    let mut batched = build_gpu(ctx, grid, 1.0e-5);
    let mut reference = build_gpu(reference_ctx, grid, 1.0e-5);
    for solver in [&mut batched, &mut reference] {
        configure_gpu(solver);
        seed_rho_u_x_gpu(solver, grid, pert);
        solver.set_filter_sigma(SIGMA);
    }

    for _ in 0..4 {
        assert!(reference.step_batch(1).is_some());
    }
    assert!(batched.step_batch(4).is_some());

    for name in ["state", "state_old", "state_old_old"] {
        assert_eq!(
            state_bits(&batched, name),
            state_bits(&reference, name),
            "filtered B=4 changed `{name}` relative to 4 single-step batches"
        );
    }
    assert_eq!(
        batched.time().to_bits(),
        reference.time().to_bits(),
        "filtered B=4 accumulated a different physical time"
    );
}

/// Autonomous-route batching parity with the filter ON: a fixed-dt B=4
/// autonomous batch must be bit-identical to four B=1 submissions — the
/// filter is dispatched through the same cells-indirect-args slot as the
/// stage kernels and runs before the accepted-state audit in both routings.
#[test]
fn gpu_structured_autonomous_batches_match_single_steps_with_filter_on() {
    let Some(ctx) = gpu_context("structured-filter-autonomous-batch-parity") else {
        return;
    };
    let reference_ctx = shared_gpu_context(&ctx);
    let grid = StructuredGrid::new(24, 10, 1.0, 0.4);
    let eps = 1.0e-3_f64;
    let pert = move |i: usize, _j: usize| if i % 2 == 0 { eps } else { -eps };

    let mut batched = build_gpu(ctx, grid, 1.0e-5);
    let mut reference = build_gpu(reference_ctx, grid, 1.0e-5);
    for solver in [&mut batched, &mut reference] {
        configure_gpu(solver);
        seed_rho_u_x_gpu(solver, grid, pert);
        solver.set_filter_sigma(SIGMA);
        assert!(solver.supports_autonomous_fixed());
    }

    batched
        .step_autonomous_batch(4, None)
        .expect("filtered autonomous B=4 submission");
    let batched_status = batched.autonomous_status().expect("batched status");
    for _ in 0..4 {
        reference
            .step_autonomous_batch(1, None)
            .expect("filtered autonomous B=1 submission");
        let status = reference.autonomous_status().expect("single-step status");
        assert!(!status.halted, "filtered B=1 controller halted: {status:?}");
    }
    let reference_status = reference.autonomous_status().expect("reference tail");
    assert!(
        !batched_status.halted,
        "filtered B=4 halted: {batched_status:?}"
    );
    assert_eq!(batched_status.accepted_steps, 4);
    assert_eq!(batched_status.accepted_total, 4);
    assert_eq!(
        batched_status.accepted_total,
        reference_status.accepted_total
    );
    assert_eq!(
        batched_status.time.to_bits(),
        reference_status.time.to_bits()
    );
    assert_eq!(batched_status.dt.to_bits(), reference_status.dt.to_bits());
    assert_eq!(
        batched_status.dt_old.to_bits(),
        reference_status.dt_old.to_bits()
    );
    for name in ["state", "state_old", "state_old_old"] {
        assert_eq!(
            state_bits(&batched, name),
            state_bits(&reference, name),
            "filtered autonomous B=4 changed `{name}` relative to B=1"
        );
    }
}

/// CPU vs GPU: one filtered RK4 step from identical seeds must agree on every
/// conserved component within 1e-5 (relative to the fields' O(1) scale). The
/// two implementations are hand-written independently (WGSL vs Rust) but pin
/// the identical f32 evaluation order.
#[cfg(feature = "cpu")]
#[test]
fn structured_filter_cpu_matches_gpu_one_step() {
    use cfd2::solver::cpu::structured::StructuredModelSolver;

    let Some(ctx) = gpu_context("structured-filter-cpu-gpu") else {
        return;
    };
    let grid = StructuredGrid::new(24, 10, 1.0, 0.4);
    let model = compressible_structured_model().expect("structured compressible model");
    let dt = 1.0e-5;
    let eps = 1.0e-3_f64;
    let pert = move |i: usize, _j: usize| if i % 2 == 0 { eps } else { -eps };

    let mut gpu = build_gpu(ctx, grid, dt);
    configure_gpu(&mut gpu);
    seed_rho_u_x_gpu(&mut gpu, grid, pert);
    gpu.set_filter_sigma(SIGMA);

    let mut cpu = StructuredModelSolver::with_config(
        grid,
        &model,
        dt,
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("structured compressible CPU RK4 solver");
    cpu.set_fluid(1.0, 0.0);
    cpu.set_eos(nondimensional_air().runtime_params());
    cpu.set_inlet_ramp(0.0, 0.0);
    for (name, value) in [("rho", 1.0), ("rho_e", 2.5), ("p", 1.0), ("T", 1.0)] {
        cpu.set_named_field(name, move |_, _| value);
    }
    let unknowns = cpu.unknowns();
    cpu.set_boundaries(move |_edge, _x, _y| {
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
    let rho_u_off = cpu.field_offset("rho_u").expect("rho_u offset");
    let (dx, dy) = (grid.dx, grid.dy);
    cpu.set_state(rho_u_off, move |x, y| {
        pert((x / dx).floor() as usize, (y / dy).floor() as usize)
    });
    cpu.set_filter_sigma(SIGMA);

    assert!(gpu.step_batch(1).is_some());
    cpu.step();

    let rho_off = cpu.field_offset("rho").expect("rho offset");
    let rho_e_off = cpu.field_offset("rho_e").expect("rho_e offset");
    for (label, offset) in [
        ("rho", rho_off),
        ("rho_u_x", rho_u_off),
        ("rho_u_y", rho_u_off + 1),
        ("rho_e", rho_e_off),
    ] {
        let gpu_field = gpu.state_field(offset);
        let cpu_field = cpu.state_field(offset);
        for (cell, (&g, &c)) in gpu_field.iter().zip(&cpu_field).enumerate() {
            let scale = g.abs().max(c.abs()).max(1.0);
            assert!(
                (g - c).abs() <= 1.0e-5 * scale,
                "{label} diverged between CPU and GPU at cell {cell}: gpu {g:e}, cpu {c:e}"
            );
        }
    }
}

/// sigma = 0.0 must be a bitwise no-op on every route: a run that calls
/// `set_filter_sigma(0.2)` and then returns to 0.0 must match a run that never
/// called the setter at all — and the setter must be harmless on a structured
/// model without the conserved layout.
#[test]
fn structured_filter_sigma_zero_is_bitwise_noop() {
    let Some(ctx) = gpu_context("structured-filter-sigma-zero") else {
        return;
    };
    let toggled_ctx = shared_gpu_context(&ctx);
    let diffusion_ctx = shared_gpu_context(&ctx);
    let diffusion_reference_ctx = shared_gpu_context(&ctx);
    let grid = StructuredGrid::new(16, 8, 1.0, 0.5);
    let eps = 1.0e-3_f64;
    let pert = move |i: usize, _j: usize| if i % 2 == 0 { eps } else { -eps };

    let mut untouched = build_gpu(ctx, grid, 1.0e-5);
    let mut toggled = build_gpu(toggled_ctx, grid, 1.0e-5);
    for solver in [&mut untouched, &mut toggled] {
        configure_gpu(solver);
        seed_rho_u_x_gpu(solver, grid, pert);
    }
    // Arm and disarm: returning to 0.0 must restore exact bit-inertness.
    toggled.set_filter_sigma(SIGMA);
    toggled.set_filter_sigma(0.0);

    assert!(untouched.step_batch(1).is_some());
    assert!(toggled.step_batch(1).is_some());
    untouched
        .step_autonomous_batch(1, None)
        .expect("untouched autonomous step");
    toggled
        .step_autonomous_batch(1, None)
        .expect("sigma-zero autonomous step");
    for name in ["state", "state_old", "state_old_old"] {
        assert_eq!(
            state_bits(&untouched, name),
            state_bits(&toggled, name),
            "sigma=0.0 changed `{name}` relative to a run that never set it"
        );
    }

    // The setter must be a harmless no-op on models without rho/rho_u/rho_e.
    let model = cfd2::solver::model::generic_diffusion_demo_structured_ibm_model()
        .expect("structured diffusion model");
    let build_diffusion = |context| {
        let mut solver = StructuredGpuSolver::with_config(
            context,
            grid,
            &model,
            0.01,
            1,
            Scheme::Upwind,
            TimeScheme::RK4,
        )
        .expect("structured diffusion RK4 solver");
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
    let mut diffusion = build_diffusion(diffusion_ctx);
    let mut diffusion_reference = build_diffusion(diffusion_reference_ctx);
    diffusion.set_filter_sigma(SIGMA); // no conserved layout: stored harmlessly
    assert!(diffusion.step_batch(1).is_some());
    assert!(diffusion_reference.step_batch(1).is_some());
    assert_eq!(
        state_bits(&diffusion, "state"),
        state_bits(&diffusion_reference, "state"),
        "set_filter_sigma on a non-conserved model must not change stepping"
    );
}
