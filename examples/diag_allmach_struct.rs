//! Headless repro of the GUI's structured all-Mach thermal channel-obstacle case.
//!
//! Mirrors `seed_structured_state` / `seed_structured_freestream` /
//! `setup_structured_bcs` / `structured_pin_dt` from `src/ui/app.rs`.
//!
//! Env knobs:
//!   MODEL=allmach|incomp     (default allmach)
//!   STEPS=200
//!   PSI=<value>              (default air 1/347.2^2; 0 => incompressible limit)
//!   UREF_MIN=1.0
//!   CFL=beta|u               (default beta = the GUI's all-Mach pseudo-sound CFL)
//!   TARGET_CFL=0.9
//!   NX=120 NY=40
//!   THREADS=6
//!   REPORT=25

use cfd2::solver::banded_schur::CoupledPrecondKind;
use cfd2::solver::cpu::structured::{StructuredGrid, StructuredModelSolver};
use cfd2::solver::gpu::structured::{BcComp, Edge};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_str(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let model_kind = env_str("MODEL", "allmach");
    let steps = env_usize("STEPS", 200);
    let psi_air = 1.0 / (1.4 * 287.0 * 300.0);
    let psi = env_f64("PSI", psi_air);
    let uref_min = env_f64("UREF_MIN", 1.0);
    let cfl_mode = env_str("CFL", "beta");
    let target_cfl = env_f64("TARGET_CFL", 0.9);
    let (nx, ny) = (env_usize("NX", 120), env_usize("NY", 40));
    let threads = env_usize("THREADS", 6);
    let report = env_usize("REPORT", 25);

    let (lx, ly) = (3.0, 1.0);
    let (rho, mu) = (1.225f64, 1.81e-5f64);
    let u_in = env_f64("U_IN", 0.011);
    let (alpha_u, alpha_p) = (0.7f32, 0.3f32);

    let model = match model_kind.as_str() {
        "incomp" => cfd2::solver::model::incompressible_momentum_structured_model().unwrap(),
        _ => cfd2::solver::model::allmach_thermal_structured_model().unwrap(),
    };
    let s = model.system.unknowns_per_cell() as usize;
    let grid = StructuredGrid::new(nx, ny, lx, ly);
    let n = grid.num_cells();
    let dx = lx / nx as f64;
    let dy = ly / ny as f64;
    let min_h = dx.min(dy);

    let mut solver = StructuredModelSolver::with_config(
        grid,
        &model,
        0.02,
        8,
        Scheme::SecondOrderUpwindVanLeer,
        TimeScheme::BDF2,
    )
    .unwrap();
    let engine = if env_str("ENGINE", "t") == "i" {
        cfd2::solver::cpu::CpuEngine::Interpreter
    } else {
        cfd2::solver::cpu::CpuEngine::Transpiled
    };
    solver.set_engine(engine, threads);
    solver.set_fluid(rho, mu);
    let alpha_u = env_f64("ALPHA_U", alpha_u as f64) as f32;
    let alpha_p = env_f64("ALPHA_P", alpha_p as f64) as f32;
    solver.set_alpha_u(alpha_u);
    solver.set_alpha_p(alpha_p);
    solver.set_preconditioner(match env_str("PRECOND", "amg").as_str() {
        "bj" => CoupledPrecondKind::BlockJacobi,
        "schur" => CoupledPrecondKind::Schur,
        _ => CoupledPrecondKind::SchurAmg,
    });
    solver.set_outer_auto_converge(env_str("AUTOCONV", "1") == "1");
    solver.set_outer_iters(env_usize("OUTERS", 8));

    // ---- seed_structured_state (allmach only)
    if model_kind != "incomp" {
        let t_ref = 1.0f64;
        solver.set_named_field("psi", move |_, _| psi);
        solver.set_named_field("rho", move |_, _| rho);
        solver.set_named_field("rho_t_ref", move |_, _| rho * t_ref);
        solver.set_named_field("T", move |_, _| t_ref);
        if solver.field_offset("t_ref").is_some() {
            solver.set_named_field("t_ref", move |_, _| t_ref);
        }
        if solver.field_offset("rho_floor").is_some() {
            solver.set_named_field("rho_floor", move |_, _| psi * 1.0e-5);
        }
        if solver.field_offset("psi_ref").is_some() {
            solver.set_named_field("psi_ref", move |_, _| psi);
        }
        let u_ref = 2.0 * u_in.abs().max(uref_min.max(0.2));
        let psi_precond = if u_ref > 0.0 { psi.max(1.0 / (u_ref * u_ref)) } else { psi };
        solver.set_named_field("psi_precond", move |_, _| psi_precond);
        if solver.field_offset("u_ref").is_some() {
            solver.set_named_field("u_ref", move |_, _| u_ref);
        }
        if solver.field_offset("precond_mask").is_some() {
            let m = if psi > 0.0 { 1.0 } else { 0.0 };
            solver.set_named_field("precond_mask", move |_, _| m);
        }
        eprintln!("psi={psi:.4e} psi_precond={psi_precond:.4e} u_ref={u_ref}");
    }

    // ---- IBM cylinder (structured_geometry_is_solid, ChannelObstacle)
    let (cx, cy, r) = (lx / 3.0, ly * 0.51, ly * 0.1);
    let no_obst = std::env::var("NO_OBST").is_ok();
    let is_solid = move |x: f64, y: f64| !no_obst && (x - cx).hypot(y - cy) < r;
    if let Some(pen) = solver.field_offset("ibm_penalty_U") {
        solver.set_state(pen, move |x, y| if is_solid(x, y) { -1.0e5 } else { 0.0 });
    }

    // ---- setup_structured_bcs (pressure-based branch)
    solver.set_boundaries(move |edge, _x, _y| {
        let d = |v: f32| BcComp { kind: 1, value: v };
        let g = || BcComp { kind: 2, value: 0.0 };
        let (bt, mut v): (u32, Vec<BcComp>) = match edge {
            Edge::Left => (1, vec![d(u_in as f32), d(0.0), g()]),
            Edge::Right => (2, vec![g(), g(), d(0.0)]),
            _ => (3, vec![d(0.0), d(0.0), g()]),
        };
        if s >= 4 {
            v.push(if matches!(edge, Edge::Left) { d(1.0) } else { g() });
        }
        (bt, v)
    });

    let p_off = solver.field_offset("p").unwrap();
    let u_off = solver.field_offset("U").unwrap();

    let solid: Vec<bool> = (0..n)
        .map(|c| {
            let (x, y) = solver.grid().cell_center(c);
            is_solid(x, y)
        })
        .collect();

    let tmax = env_f64("TMAX", f64::INFINITY);
    let mut prev_max_vel = 0.0f64;
    for st in 1..=steps {
        if solver.time() >= tmax {
            println!("reached TMAX={tmax} at step {st}");
            break;
        }
        // structured_pin_dt
        let adv = prev_max_vel.max(u_in.abs());
        let acoustic = match cfl_mode.as_str() {
            "u" => 0.0,
            _ => {
                if psi > 0.0 && model_kind != "incomp" {
                    2.0 * u_in.abs().max(uref_min)
                } else {
                    0.0
                }
            }
        };
        let wave = adv + acoustic;
        if let Ok(fixed) = std::env::var("DT") {
            solver.set_dt(fixed.parse().unwrap());
        } else if wave > 1e-12 {
            let cur = solver.dt();
            let mut next = target_cfl * min_h / wave;
            if next > cur * 1.2 {
                next = cur * 1.2;
            }
            solver.set_dt(next.clamp(1e-9, 100.0));
        }

        solver.step();

        let u = solver.get_u(u_off);
        prev_max_vel = u
            .iter()
            .map(|&(a, b)| a.hypot(b))
            .filter(|v| v.is_finite())
            .fold(0.0, f64::max);

        if st % report == 0 || st == steps {
            let p = solver.get_scalar(p_off);
            let stt = solver.last_stats();
            // Checkerboard metric on p: interior FLUID cells only, cell minus the
            // 4-neighbour average, normalised by the rms pressure fluctuation.
            let (mut c2, mut cn) = (0.0f64, 0usize);
            let pm: f64 = p.iter().sum::<f64>() / n as f64;
            let prms = (p.iter().map(|v| (v - pm) * (v - pm)).sum::<f64>() / n as f64).sqrt();
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let c = j * nx + i;
                    if solid[c] || solid[c - 1] || solid[c + 1] || solid[c - nx] || solid[c + nx] {
                        continue;
                    }
                    let lap = p[c] - 0.25 * (p[c - 1] + p[c + 1] + p[c - nx] + p[c + nx]);
                    c2 += lap * lap;
                    cn += 1;
                }
            }
            let cb = (c2 / cn.max(1) as f64).sqrt() / prms.max(1e-30);
            // Same for |U| (velocity odd-even).
            let (mut v2, mut vn) = (0.0f64, 0usize);
            let umag: Vec<f64> = u.iter().map(|&(a, b)| a.hypot(b)).collect();
            let um: f64 = umag.iter().sum::<f64>() / n as f64;
            let urms = (umag.iter().map(|v| (v - um) * (v - um)).sum::<f64>() / n as f64).sqrt();
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let c = j * nx + i;
                    if solid[c] || solid[c - 1] || solid[c + 1] || solid[c - nx] || solid[c + nx] {
                        continue;
                    }
                    let lap =
                        umag[c] - 0.25 * (umag[c - 1] + umag[c + 1] + umag[c - nx] + umag[c + nx]);
                    v2 += lap * lap;
                    vn += 1;
                }
            }
            let vcb = (v2 / vn.max(1) as f64).sqrt() / urms.max(1e-30);
            let u_solid = (0..n)
                .filter(|&c| solid[c])
                .map(|c| umag[c])
                .fold(0.0, f64::max);
            let pmin = p.iter().cloned().fold(f64::INFINITY, f64::min);
            let pmax = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            // Continuity check: streamwise mass flux per x-station vs the inlet.
            let m_in = rho * u_in * ly;
            let imb = (0..nx)
                .map(|i| {
                    let m: f64 = (0..ny).map(|j| u[j * nx + i].0).sum::<f64>() * dy * rho;
                    ((m - m_in) / m_in).abs()
                })
                .fold(0.0, f64::max);
            // Wake asymmetry (shedding signal): mean uy over a box behind the cylinder.
            let (mut asym, mut acnt) = (0.0, 0usize);
            for c in 0..n {
                let (x, y) = solver.grid().cell_center(c);
                if x > cx + r && x < cx + 6.0 * r && (y - cy).abs() < 2.0 * r && !solid[c] {
                    asym += u[c].1;
                    acnt += 1;
                }
            }
            asym /= acnt.max(1) as f64 * u_in;
            // Discrete continuity residual straight from the solver's own face-flux
            // buffer: dirs are 0=S(-y) 1=W(-x) 2=E(+x) 3=N(+y), outward-positive.
            let fl = solver.read_buffer("fluxes");
            let fs = fl.len() / (4 * n);
            let scale = rho * u_in * dx;
            let mut divmax = 0.0f64;
            let mut divmax_fluid = 0.0f64;
            for c in 0..n {
                let d: f64 = (0..4).map(|k| fl[(c * 4 + k) * fs] as f64).sum();
                divmax = divmax.max(d.abs() / scale);
                if !solid[c] {
                    divmax_fluid = divmax_fluid.max(d.abs() / scale);
                }
            }
            // Streamwise mass flux across vertical face-line i+1/2 (east faces).
            let station = |i: usize| -> f64 {
                (0..ny).map(|j| fl[((j * nx + i) * 4 + 2) * fs] as f64).sum()
            };
            let m_in_f = station(0);
            let mut fimb = 0.0f64;
            for i in 0..nx - 1 {
                fimb = fimb.max((station(i) - m_in_f).abs() / m_in.abs());
            }
            println!(
                "step={st:5} t={:.3} dt={:.3e} |U|max={:.4e} |U|solid={:.2e} p=[{:.3e},{:.3e}] cb_p={:.3} cb_U={:.3} massimb={:.1}% fluximb={:.1}% divmax={:.2e} divfluid={:.2e} asym={:+.4} outers={} lin={} res={:.1e}",
                solver.time(),
                solver.dt(),
                prev_max_vel,
                u_solid,
                pmin,
                pmax,
                cb,
                vcb,
                imb * 100.0,
                fimb * 100.0,
                divmax,
                divmax_fluid,
                asym,
                stt.outer_iters,
                stt.linear_iters,
                stt.linear_res
            );
        }
        if !prev_max_vel.is_finite() {
            println!("DIVERGED at step {st}");
            break;
        }
    }

    // Final field dump (CSV): i,j,x,y,ux,uy,p,solid
    let u = solver.get_u(u_off);
    let p = solver.get_scalar(p_off);
    for f in ["d_p", "psi_precond", "psi", "rho", "T"] {
        if let Some(o) = solver.field_offset(f) {
            let v = solver.get_scalar(o);
            let (mn, mx) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                (a.min(x), b.max(x))
            });
            // Value in a far-field fluid cell and in a solid cell.
            let fluid_c = 5 * nx + 5;
            let solid_c = (0..n).find(|&c| solid[c]).unwrap_or(0);
            println!(
                "field {f:12} min={mn:.4e} max={mx:.4e} fluid={:.4e} solid={:.4e}",
                v[fluid_c], v[solid_c]
            );
        }
    }
    // Cell-centred divergence (central differences), fluid cells away from the solid.
    let mut dmax: (f64, usize) = (0.0, 0);
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let c = j * nx + i;
            if solid[c] || solid[c - 1] || solid[c + 1] || solid[c - nx] || solid[c + nx] {
                continue;
            }
            let d = (u[c + 1].0 - u[c - 1].0) / (2.0 * dx) + (u[c + nx].1 - u[c - nx].1) / (2.0 * dy);
            if d.abs() > dmax.0 {
                dmax = (d.abs(), c);
            }
        }
    }
    println!(
        "cell-centred div: max={:.3e} (scaled by u_in/dx={:.3e} -> {:.3}) at (i={},j={})",
        dmax.0,
        u_in / dx,
        dmax.0 / (u_in / dx),
        dmax.1 % nx,
        dmax.1 / nx
    );
    let out = env_str("CSV", "");
    if !out.is_empty() {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
        writeln!(f, "i,j,x,y,ux,uy,p,solid").unwrap();
        for c in 0..n {
            let (x, y) = solver.grid().cell_center(c);
            writeln!(
                f,
                "{},{},{:.6},{:.6},{:.9e},{:.9e},{:.9e},{}",
                c % nx,
                c / nx,
                x,
                y,
                u[c].0,
                u[c].1,
                p[c],
                solid[c] as u8
            )
            .unwrap();
        }
        eprintln!("wrote {out}");
    }

    // Streamwise mass flux per x-station: sum_j rho*ux*dy. Should equal the inlet
    // value at every station once continuity is satisfied.
    println!("\n-- mass flux per x-station (rho*int(ux)dy), inlet={:.6e} --", rho * u_in * ly);
    for i in (0..nx).step_by(6) {
        let m: f64 = (0..ny).map(|j| u[j * nx + i].0).sum::<f64>() * dy * rho;
        print!("{:.4e} ", m);
    }
    println!();
    println!("\n-- centreline row j={} : |U| (x1e3) every cell --", ny / 2);
    let j = ny / 2;
    for i in 0..nx {
        print!("{:.2} ", u[j * nx + i].0.hypot(u[j * nx + i].1) * 1000.0);
    }
    println!("\n-- centreline p (x1e6) every cell --");
    for i in 0..nx {
        print!("{:.1} ", p[j * nx + i] * 1e6);
    }
    println!();
}
