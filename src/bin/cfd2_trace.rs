use cfd2::solver::model::{all_models, compressible_model_with_eos};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use cfd2::solver::model::helpers::SolverCompressibleIdealGasExt;
use cfd2::trace as tracefmt;
use std::collections::BTreeMap;
use std::io::BufRead;
use std::time::Instant;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};

fn usage() -> &'static str {
    "Usage:
  cfd2_trace info <trace.jsonl>
  cfd2_trace init [--model <id>] [--nx N] [--ny N]
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
                header = Some(*h);
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

#[derive(Default, Clone, Copy)]
struct StatsAgg {
    count: u64,
    total: f64,
    min: f64,
    max: f64,
}

impl StatsAgg {
    fn push(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.count += 1;
        self.total += value;
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.total / self.count as f64)
    }
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
        let mut wall_ms = StatsAgg::default();
        let mut outer_iters = StatsAgg::default();
        let mut linear_solve_ms = StatsAgg::default();
        let mut linear_iters = StatsAgg::default();
        let mut linear_residual = StatsAgg::default();
        let mut linear_converged = 0u64;
        let mut linear_diverged = 0u64;

        let mut graph_label_total_s: BTreeMap<String, f64> = BTreeMap::new();
        let mut graph_node_total_s: BTreeMap<String, f64> = BTreeMap::new();

        for step in &trace.step_meta {
            wall_ms.push(step.wall_time_ms as f64);
            let iters = step
                .graph
                .iter()
                .filter(|g| g.label.ends_with(":assembly"))
                .count() as f64;
            if iters > 0.0 {
                outer_iters.push(iters);
            }

            for solve in &step.linear_solves {
                linear_solve_ms.push(solve.time_ms as f64);
                linear_iters.push(solve.iterations as f64);
                linear_residual.push(solve.residual as f64);
                if solve.converged {
                    linear_converged += 1;
                }
                if solve.diverged {
                    linear_diverged += 1;
                }
            }

            for timing in &step.graph {
                *graph_label_total_s.entry(timing.label.clone()).or_insert(0.0) += timing.seconds;
                if let Some(nodes) = &timing.nodes {
                    for node in nodes {
                        *graph_node_total_s.entry(node.label.clone()).or_insert(0.0) += node.seconds;
                    }
                }
            }
        }

        let total_linear_ms = linear_solve_ms.total;
        let total_wall_ms = wall_ms.total;

        println!(
            "  recorded_total_wall_ms: {:.3}",
            trace.recorded_total_wall_ms
        );
        if let Some(mean) = wall_ms.mean() {
            println!(
                "  step_wall_ms: avg={:.3} min={:.3} max={:.3}",
                mean, wall_ms.min, wall_ms.max
            );
        }
        if outer_iters.count > 0 {
            println!(
                "  outer_iters: avg={:.2} min={:.0} max={:.0}",
                outer_iters.mean().unwrap_or(0.0),
                outer_iters.min,
                outer_iters.max
            );
        }
        if linear_solve_ms.count > 0 {
            let pct_wall = if total_wall_ms > 0.0 {
                total_linear_ms / total_wall_ms * 100.0
            } else {
                0.0
            };
            println!(
                "  linear_solves: count={} total_ms={:.3} ({:.1}% wall) avg_ms={:.3}",
                linear_solve_ms.count,
                total_linear_ms,
                pct_wall,
                linear_solve_ms.mean().unwrap_or(0.0)
            );
            println!(
                "  linear_iters: avg={:.2} min={:.0} max={:.0}  converged={} diverged={}",
                linear_iters.mean().unwrap_or(0.0),
                linear_iters.min,
                linear_iters.max,
                linear_converged,
                linear_diverged
            );
            println!(
                "  linear_residual: avg={:.3e} min={:.3e} max={:.3e}",
                linear_residual.mean().unwrap_or(0.0),
                linear_residual.min,
                linear_residual.max
            );
        }

        if !graph_label_total_s.is_empty() {
            let mut totals: Vec<(String, f64)> = graph_label_total_s.into_iter().collect();
            totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("  graph_top:");
            for (label, seconds) in totals.into_iter().take(10) {
                let pct = if total_wall_ms > 0.0 {
                    seconds * 1000.0 / total_wall_ms * 100.0
                } else {
                    0.0
                };
                println!("    {:10.3} ms ({:5.1}%)  {}", seconds * 1000.0, pct, label);
            }
        }

        if !graph_node_total_s.is_empty() {
            let mut totals: Vec<(String, f64)> = graph_node_total_s.into_iter().collect();
            totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("  graph_nodes_top:");
            for (label, seconds) in totals.into_iter().take(10) {
                let pct = if total_wall_ms > 0.0 {
                    seconds * 1000.0 / total_wall_ms * 100.0
                } else {
                    0.0
                };
                println!("    {:10.3} ms ({:5.1}%)  {}", seconds * 1000.0, pct, label);
            }
        }
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

    let mut init_events: Vec<tracefmt::TraceInitEvent> = Vec::new();
    let init_start = Instant::now();
    let init_guard = tracefmt::install_init_collector(&mut init_events);
    let mut solver = pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))?;
    drop(init_guard);
    let init_ms = init_start.elapsed().as_secs_f64() * 1000.0;

    // Match UI behavior: start from a zeroed state buffer.
    let stride = solver.model().state_layout.stride() as usize;
    let n_cells = mesh.num_cells();
    let _ = solver.write_state_f32(&vec![0.0f32; n_cells * stride]);

    trace.header
        .initial_params
        .apply_to_solver(&mut solver);

    if trace.header.case.model_id == "compressible" {
        let eos: cfd2::solver::model::EosSpec = trace.header.initial_params.eos.clone().into();
        let rho = trace.header.initial_params.density;
        let p_ref = eos.pressure_for_density(rho as f64) as f32;
        solver.set_uniform_state(rho, [0.0, 0.0], p_ref);
    }
    solver.initialize_history();

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
    if steps == 0 && trace.step_meta.is_empty() && opts.steps_override.is_none() {
        return Err("no steps to replay (pass --steps N)".into());
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
    print_init_report(init_ms, &mut init_events);
    if recorded_total_ms > 0.0 {
        println!("Recorded wall time: {:.3} ms", recorded_total_ms);
    }

    if steps == 0 {
        return Ok(());
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

fn print_init_report(init_ms: f64, init_events: &mut Vec<tracefmt::TraceInitEvent>) {
    println!("Init wall time: {:.3} ms", init_ms);
    if init_events.is_empty() {
        return;
    }

    init_events.sort_by(|a, b| {
        b.wall_time_ms
            .partial_cmp(&a.wall_time_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("Init slowest:");
    for ev in init_events.iter().take(10) {
        let pct = if init_ms > 0.0 {
            ev.wall_time_ms as f64 / init_ms * 100.0
        } else {
            0.0
        };
        let detail = ev.detail.as_deref().unwrap_or("");
        if detail.is_empty() {
            println!("  {:10.3} ms ({:5.1}%)  {}", ev.wall_time_ms, pct, ev.stage);
        } else {
            println!(
                "  {:10.3} ms ({:5.1}%)  {}  {}",
                ev.wall_time_ms, pct, ev.stage, detail
            );
        }
    }
}

struct InitOpts {
    model_id: String,
    nx: usize,
    ny: usize,
}

fn parse_init_opts(args: &[String]) -> Result<InitOpts, String> {
    let mut model_id = "compressible".to_string();
    let mut nx: usize = 64;
    let mut ny: usize = 32;

    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let Some(v) = it.next() else {
                    return Err("missing value after --model".into());
                };
                model_id = v.to_string();
            }
            "--nx" => {
                let Some(v) = it.next() else {
                    return Err("missing value after --nx".into());
                };
                nx = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --nx value '{v}'"))?
                    .max(1);
            }
            "--ny" => {
                let Some(v) = it.next() else {
                    return Err("missing value after --ny".into());
                };
                ny = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --ny value '{v}'"))?
                    .max(1);
            }
            v if v.starts_with('-') => return Err(format!("unknown option '{v}'")),
            v => return Err(format!("unexpected positional argument '{v}'")),
        }
    }

    Ok(InitOpts { model_id, nx, ny })
}

fn cmd_init(opts: InitOpts) -> Result<(), String> {
    let mesh = generate_structured_rect_mesh(
        opts.nx,
        opts.ny,
        10.0,
        1.0,
        BoundaryType::Inlet,
        BoundaryType::Outlet,
        BoundaryType::Wall,
        BoundaryType::Wall,
    );

    let model = if opts.model_id == "compressible" {
        compressible_model_with_eos(cfd2::solver::model::EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        })
    } else {
        all_models()
            .into_iter()
            .find(|m| m.id == opts.model_id)
            .ok_or_else(|| format!("unknown model id '{}'", opts.model_id))?
    };

    let named_params = model.named_param_keys();
    let stepping = if named_params.iter().any(|&k| k == "eos.gamma") {
        SteppingMode::Implicit { outer_iters: 1 }
    } else {
        SteppingMode::Coupled
    };

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
        stepping,
    };

    let mut init_events: Vec<tracefmt::TraceInitEvent> = Vec::new();
    let init_start = Instant::now();
    let init_guard = tracefmt::install_init_collector(&mut init_events);
    let _solver = pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))?;
    drop(init_guard);

    let init_ms = init_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Init case: model_id={} mesh={}x{} (cells={})",
        opts.model_id,
        opts.nx,
        opts.ny,
        mesh.num_cells()
    );
    print_init_report(init_ms, &mut init_events);
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
        "init" => match parse_init_opts(&args[2..]) {
            Ok(opts) => cmd_init(opts),
            Err(err) => Err(err),
        },
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
