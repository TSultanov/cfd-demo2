//! Focused GPU benchmark for IR-derived explicit liveness compaction.
//!
//! Runs the density-based compressible RK4 model, whose coupled layout has
//! eight rows but only four differential components. A high-order scheme also
//! audits that precomputed `DivFlux` reconstruction does not schedule the
//! unrelated packed-state gradient pass.

use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::dispatch_counter::{get_dispatch_stats, DispatchScope};
use cfd2::solver::gpu::structured::{BcComp, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides};
use cfd2::solver::model::{compressible_model, compressible_structured_model};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use std::time::{Duration, Instant};

fn shared_context(ctx: &GpuContext) -> GpuContext {
    GpuContext {
        device: ctx.device.clone(),
        queue: ctx.queue.clone(),
        timestamp_query: ctx.timestamp_query,
        timestamps_inside_encoders: ctx.timestamps_inside_encoders,
        timestamp_period_ns: ctx.timestamp_period_ns,
        pipeline_cache: ctx.pipeline_cache.clone(),
    }
}

fn wait(ctx: &GpuContext, submission: wgpu::SubmissionIndex) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("GPU fence");
}

fn elapsed(ctx: &GpuContext, run: impl FnOnce()) -> Duration {
    let start = Instant::now();
    run();
    let encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("explicit compaction fence"),
        });
    wait(ctx, ctx.queue.submit(Some(encoder.finish())));
    start.elapsed()
}

fn config(scheme: Scheme) -> SolverConfig {
    SolverConfig {
        advection_scheme: scheme,
        time_scheme: TimeScheme::RK4,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Explicit,
    }
}

fn seed_state(
    layout: &cfd2::solver::model::backend::state_layout::StateLayout,
    cells: usize,
) -> Vec<f32> {
    let stride = layout.stride() as usize;
    let mut state = vec![0.0f32; cells * stride];
    for (name, value) in [("rho", 1.0), ("rho_e", 2.5), ("p", 1.0), ("T", 1.0)] {
        let offset = layout.offset_for(name).expect("compressible state field") as usize;
        for row in state.chunks_exact_mut(stride) {
            row[offset] = value;
        }
    }
    state
}

fn structured(ctx: &GpuContext, n: usize, steps: usize, warmup: usize, scheme: Scheme) -> Duration {
    let model = compressible_structured_model().expect("structured compressible model");
    let mut solver = StructuredGpuSolver::with_config(
        shared_context(ctx),
        StructuredGrid::new(n, n, 1.0, 1.0),
        &model,
        1.0e-7,
        1,
        scheme,
        TimeScheme::RK4,
    )
    .expect("structured explicit solver");
    solver.set_fluid(1.0, 1.0e-3);
    solver.set_boundaries(|_, _, _| {
        (
            3,
            (0..8)
                .map(|_| BcComp {
                    kind: 2,
                    value: 0.0,
                })
                .collect(),
        )
    });
    solver
        .set_packed_state_f32(&seed_state(&model.state_layout, n * n))
        .expect("seed structured state");
    for _ in 0..warmup {
        solver.step();
    }
    let _ = solver.read_named("state", 1);
    let time = elapsed(ctx, || {
        for _ in 0..steps {
            solver.step();
        }
    });
    let state = solver.read_named("state", n * n * model.state_layout.stride() as usize);
    assert!(state.iter().all(|x| x.is_finite()));
    time
}

fn unstructured(
    ctx: &GpuContext,
    n: usize,
    steps: usize,
    warmup: usize,
    scheme: Scheme,
) -> (Duration, f64) {
    let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
    let model = compressible_model().expect("unstructured compressible model");
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        config(scheme),
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("unstructured explicit solver");
    solver.set_dt(1.0e-7);
    let cells = mesh.num_cells();
    solver
        .set_field_scalar("rho", &vec![1.0; cells])
        .expect("rho");
    solver
        .set_field_vec2("rho_u", &vec![(0.0, 0.0); cells])
        .expect("rho_u");
    solver
        .set_field_scalar("rho_e", &vec![2.5; cells])
        .expect("rho_e");
    solver
        .set_field_vec2("u", &vec![(0.0, 0.0); cells])
        .expect("u");
    solver.set_field_scalar("p", &vec![1.0; cells]).expect("p");
    solver.set_field_scalar("T", &vec![1.0; cells]).expect("T");
    solver.initialize_history();
    for _ in 0..warmup {
        solver.step();
    }
    let _ = pollster::block_on(solver.get_field_scalar("rho")).expect("warmup fence");
    let scope = DispatchScope::new();
    let time = elapsed(ctx, || {
        for _ in 0..steps {
            solver.step();
        }
    });
    let state = pollster::block_on(solver.read_state_f32());
    assert!(state.iter().all(|x| x.is_finite()));
    let dispatches_per_step = get_dispatch_stats().total_dispatches as f64 / steps as f64;
    drop(scope);
    (time, dispatches_per_step)
}

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(512usize);
    let steps = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(30usize);
    let scheme = match std::env::args().nth(3).as_deref() {
        Some("upwind") => Scheme::Upwind,
        _ => Scheme::QUICKVanLeer,
    };
    let warmup = 5;
    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("GPU context");

    let structured = structured(&ctx, n, steps, warmup, scheme);
    let (unstructured, dispatches) = unstructured(&ctx, n, steps, warmup, scheme);
    println!(
        "compressible RK4 {n}x{n} scheme={scheme:?}: structured={:.3} ms/step unstructured={:.3} ms/step unstructured_dispatches={dispatches:.1}/step",
        structured.as_secs_f64() * 1.0e3 / steps as f64,
        unstructured.as_secs_f64() * 1.0e3 / steps as f64,
    );
}
