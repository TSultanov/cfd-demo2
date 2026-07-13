//! Reproducible roofline probe for the matrix-free explicit GPU paths.
//!
//! This is intentionally a report-only harness.  It measures an empirical
//! storage-buffer streaming ceiling on the same wgpu device, then runs the same
//! model-derived scalar diffusion operator through the structured and
//! unstructured RK4 frontends.  Run a release build so shader compilation and
//! debug assertions do not contaminate the steady-state measurements:
//!
//! ```text
//! cargo run --release --features meshgen --example profile_explicit_bandwidth -- \
//!     --nx 1024 --ny 1024 --steps 20
//! ```

use cfd2::sim::{RuntimeParams, SolverDriver};
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::dispatch_counter::{get_dispatch_stats, DispatchScope};
use cfd2::solver::gpu::structured::{BcComp, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides};
use cfd2::solver::model::{
    allmach_thermal_model, allmach_thermal_structured_model, eos::EosSpec,
    generic_diffusion_demo_model, generic_diffusion_demo_structured_model, ALLMACH_GAMMA,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    GpuLowMachPrecondModel, PreconditionerType, SolverConfig, SteppingMode, TimeScheme,
    UnifiedSolver,
};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Options {
    nx: usize,
    ny: usize,
    steps: usize,
    warmup: usize,
    stream_mib: usize,
    stream_passes: usize,
    allmach: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            nx: 1024,
            ny: 1024,
            steps: 20,
            warmup: 5,
            // The M3 Max sweep plateaus only at 512--1024 MiB per buffer;
            // 256 MiB remains cache-inflated by roughly six percent.
            stream_mib: 512,
            stream_passes: 8,
            allmach: false,
        }
    }
}

fn parse_options() -> Options {
    let mut out = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("missing value after {flag}"))
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid integer after {flag}"))
        };
        match flag.as_str() {
            "--nx" => out.nx = value(),
            "--ny" => out.ny = value(),
            "--steps" => out.steps = value(),
            "--warmup" => out.warmup = value(),
            "--stream-mib" => out.stream_mib = value(),
            "--stream-passes" => out.stream_passes = value(),
            "--allmach" => out.allmach = true,
            "--help" | "-h" => {
                println!(
                    "profile_explicit_bandwidth [--nx N] [--ny N] [--steps N] \
                     [--warmup N] [--stream-mib N] [--stream-passes N] [--allmach]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown option {other}"),
        }
    }
    assert!(out.nx > 0 && out.ny > 0 && out.steps > 0);
    assert!(out.stream_mib > 0 && out.stream_passes > 0);
    out
}

fn wait_for(ctx: &GpuContext, submission: wgpu::SubmissionIndex) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("wait for GPU submission");
}

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

fn validate_f32_state(label: &str, values: &[f32]) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} produced a non-finite state"
    );
    let checksum = values.iter().enumerate().fold(0.0_f64, |sum, (i, value)| {
        sum + *value as f64 * ((i % 251 + 1) as f64)
    });
    std::hint::black_box(checksum);
}

/// Queue-to-completion time for work spanning multiple internal submissions.
/// Metal exposes only a timestamp counter here, and cross-command-buffer marks
/// are not stable enough to be a correctness-grade measurement (they can read
/// zero or jump clock domains). A single final fence avoids charging readback
/// and map latency while retaining honest end-to-end solver time.
fn queue_elapsed(ctx: &GpuContext, run: impl FnOnce()) -> Duration {
    let start = Instant::now();
    run();
    let encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("explicit bandwidth completion fence"),
        });
    wait_for(ctx, ctx.queue.submit(Some(encoder.finish())));
    start.elapsed()
}

/// Empirical read+write storage bandwidth.  The working set is deliberately
/// larger than GPU caches and alternates two buffers, so every pass must stream
/// one full read and one full write.  Encoder construction is outside the timed
/// interval; the reported duration is the GPU timestamp delta.
fn streaming_ceiling(
    ctx: &GpuContext,
    bytes: u64,
    passes: usize,
) -> (Duration, Option<Duration>, f64) {
    let bytes = bytes / 16 * 16;
    let vectors = (bytes / 16) as u32;
    let a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roofline stream a"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE,
        // Materialize incompressible pages. Fresh Metal buffers read as zero;
        // repeatedly copying those can exercise zero-page allocation and
        // framebuffer-style compression instead of external memory bandwidth,
        // producing impossible multi-terabyte/s "ceilings".
        mapped_at_creation: true,
    });
    let b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roofline stream b"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    {
        let mut mapped = a.slice(..).get_mapped_range_mut();
        let mut hash = 0x9e37_79b9u32;
        let words = mapped.len() / 4;
        let mut fill = vec![0u8; words * 4];
        for (index, word) in fill.chunks_exact_mut(4).enumerate() {
            hash ^= (index as u32).wrapping_mul(0x85eb_ca6b);
            hash ^= hash >> 16;
            hash = hash.wrapping_mul(0x7feb_352d);
            hash ^= hash >> 15;
            word.copy_from_slice(&hash.to_le_bytes());
        }
        mapped.slice(..fill.len()).copy_from_slice(&fill);
    }
    a.unmap();
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("roofline stream shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.y * (65535u * 256u) + gid.x;
    if (linear < arrayLength(&src)) {
        dst[linear] = src[linear];
    }
}
"#
                .into(),
            ),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("roofline stream pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let layout = pipeline.get_bind_group_layout(0);
    let bind = |label, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
            ],
        })
    };
    let ab = bind("roofline a to b", &a, &b);
    let ba = bind("roofline b to a", &b, &a);

    let timing = ctx.timestamp_query.then(|| {
        let queries = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("roofline stream timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roofline stream timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roofline stream timestamp read"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (queries, resolve, read)
    });

    let encode = |timed: bool| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roofline stream encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roofline stream pass"),
                timestamp_writes: if timed {
                    timing
                        .as_ref()
                        .map(|(queries, _, _)| wgpu::ComputePassTimestampWrites {
                            query_set: queries,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        })
                } else {
                    None
                },
            });
            pass.set_pipeline(&pipeline);
            for i in 0..passes {
                pass.set_bind_group(0, if i & 1 == 0 { &ab } else { &ba }, &[]);
                let groups = vectors.div_ceil(256);
                pass.dispatch_workgroups(groups.min(65_535), groups.div_ceil(65_535), 1);
            }
        }
        if timed {
            if let Some((queries, resolve, read)) = &timing {
                encoder.resolve_query_set(queries, 0..2, resolve, 0);
                encoder.copy_buffer_to_buffer(resolve, 0, read, 0, 16);
            }
        }
        encoder.finish()
    };

    let warm = ctx.queue.submit(Some(encode(false)));
    wait_for(ctx, warm);
    let command = encode(true);
    let start = Instant::now();
    let submission = ctx.queue.submit(Some(command));
    let (wall_elapsed, timestamp_elapsed) = if let Some((_, _, read)) = &timing {
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        wait_for(ctx, submission);
        let wall = start.elapsed();
        rx.recv()
            .expect("stream timestamp map callback")
            .expect("stream timestamp map");
        let mapped = slice.get_mapped_range();
        let ticks: &[u64] = bytemuck::cast_slice(&mapped);
        let delta = ticks[1].saturating_sub(ticks[0]);
        let gpu = (delta != 0).then(|| {
            Duration::from_nanos((delta as f64 * ctx.timestamp_period_ns as f64).round() as u64)
        });
        drop(mapped);
        read.unmap();
        (wall, gpu)
    } else {
        wait_for(ctx, submission);
        (start.elapsed(), None)
    };
    let logical_bytes = 2.0 * bytes as f64 * passes as f64;
    // Queue-to-completion is the accepted ceiling denominator. On Metal the
    // same-pass timestamp can be dramatically shorter than physically possible
    // for large storage copies (likely a counter-domain/encoder issue), so keep
    // it as an explicit diagnostic rather than silently claiming TB/s DRAM.
    (
        wall_elapsed,
        timestamp_elapsed,
        logical_bytes / wall_elapsed.as_secs_f64(),
    )
}

fn explicit_config() -> SolverConfig {
    SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::RK4,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Explicit,
    }
}

fn measure_structured(ctx: &GpuContext, options: Options) -> (Duration, f64) {
    // Match the unstructured scalar model's one-component state. IBM is a
    // separate workload; including its second state slot here confounds the
    // topology comparison with an extra field read and wider history copies.
    let model = generic_diffusion_demo_structured_model().expect("structured model");
    let h = (1.0 / options.nx as f64).min(1.0 / options.ny as f64);
    let stable_dt = 0.1 * h * h;
    let mut solver = StructuredGpuSolver::with_config(
        shared_context(ctx),
        StructuredGrid::new(options.nx, options.ny, 1.0, 1.0),
        &model,
        stable_dt,
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("structured solver");
    solver.set_boundaries(|_, _, _| {
        (
            1,
            vec![BcComp {
                kind: 2,
                value: 0.0,
            }],
        )
    });
    solver.set_named_field("phi", |x, y| {
        (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).sin()
    });
    for _ in 0..options.warmup {
        solver.step();
    }
    let _ = solver.read_named("state", 1);

    let elapsed = queue_elapsed(ctx, || {
        for _ in 0..options.steps {
            solver.step();
        }
    });
    let state = solver.read_named(
        "state",
        options.nx * options.ny * model.state_layout.stride() as usize,
    );
    validate_f32_state("structured scalar", &state);
    let cell_stages = (options.nx * options.ny * options.steps * 4) as f64;
    (elapsed, cell_stages / elapsed.as_secs_f64())
}

fn measure_unstructured(ctx: &GpuContext, options: Options) -> (Duration, f64, u64) {
    let mesh =
        generate_structured_rect_mesh(options.nx, options.ny, 1.0, 1.0, BoundarySides::wall());
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        generic_diffusion_demo_model().expect("unstructured model"),
        explicit_config(),
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("unstructured solver");
    let h = (1.0 / options.nx as f64).min(1.0 / options.ny as f64);
    solver.set_dt((0.1 * h * h) as f32);
    solver
        .set_field_scalar(
            "phi",
            &mesh
                .cell_cx
                .iter()
                .zip(mesh.cell_cy.iter())
                .map(|(&x, &y)| {
                    (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).sin()
                })
                .collect::<Vec<_>>(),
        )
        .expect("seed phi");
    solver.initialize_history();
    for _ in 0..options.warmup {
        solver.step();
    }
    let _ = pollster::block_on(solver.get_field_scalar("phi")).expect("warmup sync");

    let dispatch_scope = DispatchScope::new();
    let elapsed = queue_elapsed(ctx, || {
        for _ in 0..options.steps {
            solver.step();
        }
    });
    let state = pollster::block_on(solver.read_state_f32());
    validate_f32_state("unstructured scalar", &state);
    let dispatches = get_dispatch_stats().total_dispatches;
    drop(dispatch_scope);
    let cell_stages = (options.nx * options.ny * options.steps * 4) as f64;
    (elapsed, cell_stages / elapsed.as_secs_f64(), dispatches)
}

fn allmach_params() -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 1.0e-7,
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
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 0.0,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: EosSpec::Constant,
        compressibility_psi: (1.0 / (347.0_f64 * 347.0)) as f32,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 1.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn measure_structured_allmach(ctx: &GpuContext, options: Options) -> (Duration, f64) {
    let model = allmach_thermal_structured_model().expect("structured all-Mach model");
    let params = allmach_params();
    let grid = StructuredGrid::new(options.nx, options.ny, 1.0, 1.0);
    let mut solver = StructuredGpuSolver::with_config(
        shared_context(ctx),
        grid,
        &model,
        params.requested_dt as f64,
        1,
        params.advection_scheme,
        TimeScheme::RK4,
    )
    .expect("structured all-Mach solver");
    solver.set_fluid(params.density as f64, params.viscosity as f64);
    solver.set_boundaries(|_, _, _| {
        (
            3,
            (0..4)
                .map(|_| BcComp {
                    kind: 2,
                    value: 0.0,
                })
                .collect(),
        )
    });

    let layout = &model.state_layout;
    let stride = layout.stride() as usize;
    let cells = options.nx * options.ny;
    let mut state = vec![0.0_f32; cells * stride];
    let psi = params.compressibility_psi as f64;
    let rho = params.density as f64;
    let t_ref = 1.0_f64;
    let u_ref = 2.0_f64;
    let psi_precond = (ALLMACH_GAMMA - 1.0) * psi + psi.max(1.0 / (u_ref * u_ref));
    let chi = (psi_precond - (ALLMACH_GAMMA - 1.0) * psi).max(1.0e-30);
    let tau = chi.sqrt() / (1.0 / grid.dx + 1.0 / grid.dy);
    let d_p = tau / rho;
    let fields = [
        ("psi", psi),
        ("rho", rho),
        ("rho_t_ref", rho * t_ref),
        ("T", t_ref),
        ("t_ref", t_ref),
        ("rho_floor", psi * 1.0e-5),
        ("psi_ref", psi),
        ("psi_precond", psi_precond),
        ("rho_dT", -rho / t_ref),
        ("u_dot_grad_p", 0.0),
        ("dt_local", tau),
        ("d_p", d_p),
        ("u_ref", u_ref),
        ("precond_mask", 1.0),
        ("ibm_penalty_U", 0.0),
    ];
    for (name, value) in fields {
        if let Some(offset) = layout.offset_for(name) {
            let offset = offset as usize;
            for row in state.chunks_exact_mut(stride) {
                row[offset] = value as f32;
            }
        }
    }
    solver
        .set_packed_state_f32(&state)
        .expect("seed structured all-Mach state");

    for _ in 0..options.warmup {
        solver.step();
    }
    let _ = solver.read_named("state", 1);
    let elapsed = queue_elapsed(ctx, || {
        for _ in 0..options.steps {
            solver.step();
        }
    });
    let state = solver.read_named("state", cells * model.state_layout.stride() as usize);
    validate_f32_state("structured all-Mach", &state);
    let cell_stages = (cells * options.steps * 4) as f64;
    (elapsed, cell_stages / elapsed.as_secs_f64())
}

fn measure_unstructured_allmach(ctx: &GpuContext, options: Options) -> (Duration, f64, u64) {
    let mesh =
        generate_structured_rect_mesh(options.nx, options.ny, 1.0, 1.0, BoundarySides::wall());
    let params = allmach_params();
    let mut build = pollster::block_on(SolverDriver::build(
        &mesh,
        allmach_thermal_model().expect("unstructured all-Mach model"),
        &params,
        &vec![(0.0, 0.0); mesh.num_cells()],
        &vec![0.0; mesh.num_cells()],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("unstructured all-Mach driver");
    build.driver.apply_params(&params);
    let mut solver = build.driver.into_solver();
    for _ in 0..options.warmup {
        solver.step();
    }
    let _ = pollster::block_on(solver.get_field_scalar("p")).expect("warmup sync");

    let dispatch_scope = DispatchScope::new();
    let elapsed = queue_elapsed(ctx, || {
        for _ in 0..options.steps {
            solver.step();
        }
    });
    let state = pollster::block_on(solver.read_state_f32());
    validate_f32_state("unstructured all-Mach", &state);
    let dispatches = get_dispatch_stats().total_dispatches;
    drop(dispatch_scope);
    let cell_stages = (mesh.num_cells() * options.steps * 4) as f64;
    (elapsed, cell_stages / elapsed.as_secs_f64(), dispatches)
}

fn main() {
    let options = parse_options();
    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("GPU context");
    let router = if std::env::var_os("CFD2_STRUCTGPU_LEGACY_EXPLICIT_ROUTER").is_some() {
        "legacy-uncached-separate"
    } else if std::env::var_os("CFD2_STRUCTGPU_NO_EXPLICIT_BATCH").is_some() {
        "cached-separate"
    } else {
        "cached-batched"
    };
    println!(
        "measurement: solver_timing=queue-to-completion stream_timing=queue-to-completion gpu_timestamp=diagnostic-only gpu_timestamps={} inside_encoders={} period_ns={} structured_router={router}",
        ctx.timestamp_query, ctx.timestamps_inside_encoders, ctx.timestamp_period_ns
    );
    let stream_bytes = options.stream_mib as u64 * 1024 * 1024;
    let (stream_time, stream_gpu_time, stream_bw) =
        streaming_ceiling(&ctx, stream_bytes, options.stream_passes);
    println!(
        "stream: working_set={} MiB passes={} wall={:.3} ms gpu_timestamp={} read+write={:.1} GB/s",
        options.stream_mib,
        options.stream_passes,
        stream_time.as_secs_f64() * 1e3,
        stream_gpu_time
            .map(|time| format!("{:.3}ms", time.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "n/a".to_string()),
        stream_bw / 1e9,
    );

    let (structured_time, structured_cell_stages) = measure_structured(&ctx, options);
    println!(
        "structured: {}x{} steps={} time={:.3} ms step={:.3} ms cell-stages={:.1} M/s",
        options.nx,
        options.ny,
        options.steps,
        structured_time.as_secs_f64() * 1e3,
        structured_time.as_secs_f64() * 1e3 / options.steps as f64,
        structured_cell_stages / 1e6,
    );

    let (unstructured_time, unstructured_cell_stages, dispatches) =
        measure_unstructured(&ctx, options);
    println!(
        "unstructured: {}x{} steps={} time={:.3} ms step={:.3} ms cell-stages={:.1} M/s dispatches/step={:.1}",
        options.nx,
        options.ny,
        options.steps,
        unstructured_time.as_secs_f64() * 1e3,
        unstructured_time.as_secs_f64() * 1e3 / options.steps as f64,
        unstructured_cell_stages / 1e6,
        dispatches as f64 / options.steps as f64,
    );

    if options.allmach {
        let (time, cell_stages) = measure_structured_allmach(&ctx, options);
        println!(
            "structured-allmach: {}x{} steps={} time={:.3} ms step={:.3} ms cell-stages={:.1} M/s",
            options.nx,
            options.ny,
            options.steps,
            time.as_secs_f64() * 1e3,
            time.as_secs_f64() * 1e3 / options.steps as f64,
            cell_stages / 1e6,
        );
        let (time, cell_stages, dispatches) = measure_unstructured_allmach(&ctx, options);
        println!(
            "unstructured-allmach: {}x{} steps={} time={:.3} ms step={:.3} ms cell-stages={:.1} M/s dispatches/step={:.1}",
            options.nx,
            options.ny,
            options.steps,
            time.as_secs_f64() * 1e3,
            time.as_secs_f64() * 1e3 / options.steps as f64,
            cell_stages / 1e6,
            dispatches as f64 / options.steps as f64,
        );
    }
}
