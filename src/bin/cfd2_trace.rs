use cfd2::solver::model::{all_models, compressible_model_with_eos};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use cfd2::trace as tracefmt;
use std::collections::BTreeMap;
use std::io::BufRead;

fn usage() -> &'static str {
    "Usage:
  cfd2_trace info <trace.jsonl>
  cfd2_trace replay <trace.jsonl> [--steps N] [--report-every N] [--no-collect-trace] [--no-profiling]

Notes:
  - `replay` uses the mesh snapshot embedded in the trace file.
  - `--steps` overrides the number of recorded steps (required if the trace has no Step events)."
}

struct LoadedTrace {
    header: tracefmt::TraceHeader,
    init_events: Vec<tracefmt::TraceInitEvent>,
    init_total_ms: Option<f64>,
    params_by_step: BTreeMap<u64, Vec<tracefmt::TraceRuntimeParams>>,
    step_meta: Vec<tracefmt::TraceStepEvent>,
    recorded_total_wall_ms: f64,
    baseline_step: u64,
}

fn load_trace(path: &str) -> Result<LoadedTrace, String> {
    let file = std::fs::File::open(path)
        .map_err(|err| format!("failed to open trace file '{}': {err}", path))?;
    let reader = std::io::BufReader::new(file);

    let mut header: Option<tracefmt::TraceHeader> = None;
    let mut init_events: Vec<tracefmt::TraceInitEvent> = Vec::new();
    let mut init_total_ms: Option<f64> = None;
    let mut params_by_step: BTreeMap<u64, Vec<tracefmt::TraceRuntimeParams>> = BTreeMap::new();
    let mut step_meta: Vec<tracefmt::TraceStepEvent> = Vec::new();
    let mut recorded_total_wall_ms: f64 = 0.0;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| format!("failed reading line {}: {err}", line_idx + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event: tracefmt::TraceEvent = serde_json::from_str(line)
            .map_err(|err| format!("failed to parse JSON on line {}: {err}", line_idx + 1))?;
        match event {
            tracefmt::TraceEvent::Header(h) => {
                if header.is_some() {
                    return Err(format!(
                        "trace file '{}' has multiple Header events (line {})",
                        path,
                        line_idx + 1
                    ));
                }
                header = Some(h);
            }
            tracefmt::TraceEvent::Init(init) => {
                if init.stage == "init.total" && init_total_ms.is_none() {
                    init_total_ms = Some(init.wall_time_ms as f64);
                }
                init_events.push(init);
            }
            tracefmt::TraceEvent::Params(p) => {
                params_by_step.entry(p.step).or_default().push(p.params);
            }
            tracefmt::TraceEvent::Step(s) => {
                recorded_total_wall_ms += s.wall_time_ms as f64;
                step_meta.push(s);
            }
            tracefmt::TraceEvent::Profiling(_) | tracefmt::TraceEvent::Footer(_) => {}
        }
    }

    let header = header.ok_or_else(|| format!("trace file '{}' is missing a Header event", path))?;

    let baseline_step = step_meta
        .first()
        .map(|s| s.step)
        .or_else(|| params_by_step.keys().next().copied())
        .unwrap_or(0);

    Ok(LoadedTrace {
        header,
        init_events,
        init_total_ms,
        params_by_step,
        step_meta,
        recorded_total_wall_ms,
        baseline_step,
    })
}

fn build_model(case: &tracefmt::TraceCase) -> Result<cfd2::solver::model::ModelSpec, String> {
    if case.model_id == "compressible" {
        return Ok(compressible_model_with_eos(case.fluid.eos.into()));
    }
    all_models()
        .into_iter()
        .find(|m| m.id == case.model_id)
        .ok_or_else(|| format!("unknown model id '{}'", case.model_id))
}

fn build_config(
    trace: &LoadedTrace,
    model: &cfd2::solver::model::ModelSpec,
) -> SolverConfig {
    let stepping = match trace.header.case.stepping_mode {
        tracefmt::TraceSteppingMode::Explicit => SteppingMode::Explicit,
        tracefmt::TraceSteppingMode::Implicit => SteppingMode::Implicit { outer_iters: 1 },
        tracefmt::TraceSteppingMode::Coupled => SteppingMode::Coupled,
    };

    let named_params = model.named_param_keys();
    let supports_preconditioner = named_params.iter().any(|&k| k == "preconditioner");
    let config_preconditioner = if supports_preconditioner {
        PreconditionerType::from(trace.header.initial_params.preconditioner)
    } else {
        PreconditionerType::Jacobi
    };

    SolverConfig {
        advection_scheme: Scheme::from(trace.header.initial_params.advection_scheme),
        time_scheme: TimeScheme::from(trace.header.initial_params.time_scheme),
        preconditioner: config_preconditioner,
        stepping,
    }
}

fn cmd_info(path: &str) -> Result<(), String> {
    let trace = load_trace(path)?;
    let mesh = trace.header.case.mesh.to_mesh()?;

    println!("Trace: {}", path);
    println!("  format_version: {}", trace.header.format_version);
    println!("  created_unix_ms: {}", trace.header.created_unix_ms);
    println!("  model_id: {}", trace.header.case.model_id);
    println!("  geometry: {:?}", trace.header.case.geometry);
    println!("  mesh: cells={} faces={} vertices={}", mesh.num_cells(), mesh.face_v1.len(), mesh.vx.len());
    println!(
        "  steps: {} (baseline step={})",
        trace.step_meta.len(),
        trace.baseline_step
    );
    if !trace.init_events.is_empty() {
        if let Some(total_ms) = trace.init_total_ms {
            println!("  init_total_ms: {:.3}", total_ms);
        }
        println!(
            "  init_events: {}",
            trace.init_events.len(),
        );

        let mut slowest: Vec<&tracefmt::TraceInitEvent> = trace
            .init_events
            .iter()
            .filter(|e| e.stage != "init.total")
            .collect();
        slowest.sort_by(|a, b| {
            b.wall_time_ms
                .partial_cmp(&a.wall_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!("  init_slowest:");
        for ev in slowest.into_iter().take(10) {
            let pct = trace
                .init_total_ms
                .and_then(|total| (total > 0.0).then_some(ev.wall_time_ms as f64 / total * 100.0));
            let pct_str = pct.map(|p| format!(" ({p:5.1}%)")).unwrap_or_default();
            let detail = ev.detail.as_deref().unwrap_or("");
            if detail.is_empty() {
                println!("    {:10.3} ms{}  {}", ev.wall_time_ms, pct_str, ev.stage);
            } else {
                println!(
                    "    {:10.3} ms{}  {}  {}",
                    ev.wall_time_ms, pct_str, ev.stage, detail
                );
            }
        }
    }
    if !trace.step_meta.is_empty() {
        println!(
            "  recorded_total_wall_ms: {:.3}",
            trace.recorded_total_wall_ms
        );
    }
    println!(
        "  profiling_enabled (original build): {}",
        trace.header.ui.profiling_enabled
    );
    Ok(())
}

struct ReplayOpts {
    steps_override: Option<u64>,
    report_every: u64,
    collect_trace: bool,
    profiling: bool,
}

fn parse_replay_opts(args: &[String]) -> Result<(String, ReplayOpts), String> {
    let mut path: Option<String> = None;
    let mut steps_override: Option<u64> = None;
    let mut report_every: u64 = 25;
    let mut collect_trace = true;
    let mut profiling = true;

    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--steps" => {
                let Some(v) = it.next() else {
                    return Err("missing value after --steps".into());
                };
                steps_override = Some(
                    v.parse::<u64>()
                        .map_err(|_| format!("invalid --steps value '{v}'"))?,
                );
            }
            "--report-every" => {
                let Some(v) = it.next() else {
                    return Err("missing value after --report-every".into());
                };
                report_every = v
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --report-every value '{v}'"))?
                    .max(1);
            }
            "--no-collect-trace" => collect_trace = false,
            "--no-profiling" => profiling = false,
            v if v.starts_with('-') => return Err(format!("unknown option '{v}'")),
            v => {
                if path.is_some() {
                    return Err(format!("unexpected positional argument '{v}'"));
                }
                path = Some(v.to_string());
            }
        }
    }

    let path = path.ok_or_else(|| "missing trace file path".to_string())?;
    Ok((
        path,
        ReplayOpts {
            steps_override,
            report_every,
            collect_trace,
            profiling,
        },
    ))
}

fn cmd_replay(path: &str, opts: ReplayOpts) -> Result<(), String> {
    let trace = load_trace(path)?;
    let mesh = trace.header.case.mesh.to_mesh()?;
    let model = build_model(&trace.header.case)?;
    let config = build_config(&trace, &model);

    let mut solver = pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))?;

    // Match UI behavior: start from a zeroed state buffer.
    let stride = solver.model().state_layout.stride() as usize;
    let n_cells = mesh.num_cells();
    let _ = solver.write_state_f32(&vec![0.0f32; n_cells * stride]);

    trace.header
        .initial_params
        .apply_to_solver(&mut solver);

    let wants_profiling = opts.profiling && trace.header.ui.profiling_enabled;
    if wants_profiling && !cfg!(feature = "profiling") {
        eprintln!(
            "[cfd2_trace] profiling requested, but this build is missing the `profiling` feature"
        );
    }

    let profiling_enabled = wants_profiling && cfg!(feature = "profiling");
    if profiling_enabled {
        solver
            .enable_detailed_profiling(true)
            .map_err(|err| format!("failed to enable profiling: {err}"))?;
        solver
            .start_profiling_session()
            .map_err(|err| format!("failed to start profiling: {err}"))?;
    }

    if opts.collect_trace {
        solver.set_collect_trace(true);
    }

    let steps = if let Some(override_steps) = opts.steps_override {
        override_steps
    } else {
        trace.step_meta.len() as u64
    };
    if steps == 0 {
        return Err(
            "no steps to replay (pass --steps N if the trace contains no Step events)".into(),
        );
    }

    let mut recorded_total_ms: f64 = 0.0;
    if !trace.step_meta.is_empty() {
        let available = trace.step_meta.len() as u64;
        let slice = steps.min(available) as usize;
        recorded_total_ms = trace.step_meta[..slice]
            .iter()
            .map(|s| s.wall_time_ms as f64)
            .sum();
    }

    println!(
        "Replaying {} step(s) from '{}' (baseline step={})",
        steps, path, trace.baseline_step
    );
    if recorded_total_ms > 0.0 {
        println!("Recorded wall time: {:.3} ms", recorded_total_ms);
    }

    let start_total = std::time::Instant::now();

    for local_step in 0..steps {
        let trace_step = trace.baseline_step.wrapping_add(local_step);

        if let Some(changes) = trace.params_by_step.get(&trace_step) {
            for p in changes {
                p.apply_to_solver(&mut solver);
            }
        }

        if let Some(meta) = trace
            .step_meta
            .get(local_step as usize)
            .filter(|m| m.step == trace_step)
        {
            solver.set_dt(meta.dt);
        }

        let start_step = std::time::Instant::now();
        let linear_stats = solver
            .step_with_stats()
            .map_err(|err| format!("solver step failed at step {local_step}: {err}"))?;
        let step_ms = start_step.elapsed().as_secs_f64() * 1000.0;

        let linear_diverged = linear_stats
            .iter()
            .any(|s| s.diverged || !s.residual.is_finite() || s.residual > 1e12);
        if linear_diverged {
            return Err(format!(
                "linear divergence detected at step {local_step} (t={:.4e}, dt={:.2e})",
                solver.time(),
                solver.dt()
            ));
        }

        if (local_step + 1) % opts.report_every == 0 || local_step + 1 == steps {
            println!(
                "step {:>6}/{:<6} t={:.4e} dt={:.2e} step_ms={:.3}",
                local_step + 1,
                steps,
                solver.time(),
                solver.dt(),
                step_ms
            );
        }
    }

    let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;
    println!("Replay wall time: {:.3} ms", total_ms);
    if recorded_total_ms > 0.0 {
        println!("Replay / recorded: {:.3}x", total_ms / recorded_total_ms);
    }

    if profiling_enabled {
        solver
            .end_profiling_session()
            .map_err(|err| format!("failed to end profiling: {err}"))?;
        let stats = solver
            .get_profiling_stats()
            .map_err(|err| format!("failed to fetch profiling stats: {err}"))?;
        println!(
            "Profiling session total: {:.3}s (iters={})",
            stats.get_session_total().as_secs_f64(),
            stats.get_iteration_count()
        );
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", usage());
        std::process::exit(2);
    }

    let result = match args[1].as_str() {
        "info" => {
            if args.len() != 3 {
                Err("expected: cfd2_trace info <trace.jsonl>".to_string())
            } else {
                cmd_info(&args[2])
            }
        }
        "replay" => {
            match parse_replay_opts(&args[2..]) {
                Ok((path, opts)) => cmd_replay(&path, opts),
                Err(err) => Err(err),
            }
        }
        _ => Err(format!("unknown command '{}'\n\n{}", args[1], usage())),
    };

    if let Err(err) = result {
        eprintln!("[cfd2_trace] {err}");
        std::process::exit(1);
    }
}
