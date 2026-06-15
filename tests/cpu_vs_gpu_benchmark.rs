//! Performance benchmark: every CPU backend variant (interpreter / transpiled ×
//! thread count × SIMD) vs the GPU, on an incompressible (Taylor-Green, coupled
//! Schur saddle-point) and a compressible (full KT-flux / EOS-recovery) flow.
//!
//! Run explicitly (it is `#[ignore]` — wall-clock timings, needs a GPU adapter for
//! the GPU row):
//!   cargo test --features cpu,meshgen --test cpu_vs_gpu_benchmark -- --ignored --nocapture
//!
//! Every variant is driven through the SAME `UnifiedSolver` path (the GUI / public
//! API), selected via the `CFD2_BACKEND` / `CFD2_CPU_*` env vars, so the numbers
//! reflect the real dispatch. The error column is a sanity check that each variant
//! computes the same physics (not a divergent/garbage run), not an order study.
#![cfg(feature = "cpu")]

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    compressible_mms_model, incompressible_momentum_mms_model,
    COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, COMPRESSIBLE_MMS_SOURCE_RHO_FIELD,
    COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, INCOMPRESSIBLE_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, TimeScheme, UnifiedSolver};
use std::f64::consts::PI;
use std::time::Instant;

/// (label, Some((engine, threads, simd)) for CPU variants; None for the GPU).
fn variants() -> Vec<(&'static str, Option<(&'static str, usize, bool)>)> {
    vec![
        ("GPU", None),
        ("CPU interp  x1", Some(("interpreter", 1, false))),
        ("CPU interp  x8", Some(("interpreter", 8, false))),
        ("CPU transp  x1", Some(("transpiled", 1, false))),
        ("CPU transp  x8", Some(("transpiled", 8, false))),
        ("CPU transp  x8+SIMD", Some(("transpiled", 8, true))),
    ]
}

fn set_backend_env(cpu: Option<(&'static str, usize, bool)>) {
    match cpu {
        None => {
            std::env::remove_var("CFD2_BACKEND");
        }
        Some((engine, threads, simd)) => {
            std::env::set_var("CFD2_BACKEND", "cpu");
            std::env::set_var("CFD2_CPU_ENGINE", engine);
            std::env::set_var("CFD2_CPU_THREADS", threads.to_string());
            std::env::set_var("CFD2_CPU_SIMD", if simd { "1" } else { "0" });
        }
    }
}

fn clear_backend_env() {
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    std::env::remove_var("CFD2_CPU_THREADS");
    std::env::remove_var("CFD2_CPU_SIMD");
}

fn tg_u(x: f64, y: f64) -> (f64, f64) {
    ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
}

/// Build + configure an incompressible Taylor-Green MMS solver on the CURRENT
/// backend (selected via env). Returns None if backend construction failed
/// (e.g. no GPU adapter).
fn build_incompressible(mesh: &Mesh, outer: usize) -> Option<UnifiedSolver> {
    let model = incompressible_momentum_mms_model().ok()?;
    let config = SolverConfig {
        advection_scheme: Scheme::SecondOrderUpwind,
        time_scheme: TimeScheme::BDF2,
        stepping: SteppingMode::Coupled,
        ..SolverConfig::default()
    };
    let mut s = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None)).ok()?;
    s.set_dt(0.05);
    s.set_density(1.0).ok()?;
    s.set_viscosity(1.0).ok()?;
    s.set_alpha_u(0.7).ok()?;
    s.set_alpha_p(0.3).ok()?;
    s.set_outer_iters(outer).ok()?;
    let (fx, fy) = (mesh.face_cx.clone(), mesh.face_cy.clone());
    for c in 0..2u32 {
        let (fx, fy) = (fx.clone(), fy.clone());
        let wall = move |i: u32| {
            let (ux, uy) = tg_u(fx[i as usize], fy[i as usize]);
            (if c == 0 { ux } else { uy }) as f32
        };
        s.set_boundary_values_per_face(GpuBoundaryType::Wall, "U", c, &wall).ok()?;
    }
    let src: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| {
            let (ux, uy) = tg_u(mesh.cell_cx[i], mesh.cell_cy[i]);
            (2.0 * PI * PI * ux, 2.0 * PI * PI * uy)
        })
        .collect();
    s.set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src).ok()?;
    s.set_field_vec2("U", &vec![(0.0, 0.0); mesh.num_cells()]).ok()?;
    s.set_field_scalar("p", &vec![0.0; mesh.num_cells()]).ok()?;
    s.initialize_history();
    Some(s)
}

/// Build + configure a compressible MMS solver (uniform stagnant gas, matching
/// inlets, zero source — an exact steady state that still exercises the full
/// KT-flux / multi-gradient / EOS-recovery / block-solve path every step).
fn build_compressible(mesh: &Mesh) -> Option<UnifiedSolver> {
    let model = compressible_mms_model().ok()?;
    let cells = mesh.num_cells();
    let (rho0, p0) = (1.0f64, 1.0f64);
    let rho_e0 = p0 / 0.4; // gamma-1 = 0.4, u = 0
    let t0 = p0 / rho0; // R = 1
    let config = SolverConfig {
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        stepping: SteppingMode::Implicit { outer_iters: 1 },
        ..SolverConfig::default()
    };
    let mut s = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None)).ok()?;
    s.set_dt(0.01);
    s.set_viscosity(0.05).ok()?;
    s.set_density(rho0 as f32).ok()?;
    s.set_outer_iters(1).ok()?;
    s.set_field_scalar("rho", &vec![rho0; cells]).ok()?;
    s.set_field_vec2("rho_u", &vec![(0.0, 0.0); cells]).ok()?;
    s.set_field_scalar("rho_e", &vec![rho_e0; cells]).ok()?;
    s.set_field_scalar("p", &vec![p0; cells]).ok()?;
    s.set_field_scalar("T", &vec![t0; cells]).ok()?;
    s.set_field_vec2("u", &vec![(0.0, 0.0); cells]).ok()?;
    s.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, &vec![0.0; cells]).ok()?;
    s.set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &vec![(0.0, 0.0); cells]).ok()?;
    s.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &vec![0.0; cells]).ok()?;
    for (field, val) in [("rho", rho0), ("p", p0), ("T", t0), ("rho_e", rho_e0)] {
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, field, 0, &|_| val as f32).ok()?;
    }
    for c in 0..2u32 {
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "u", c, &|_| 0.0).ok()?;
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_u", c, &|_| 0.0).ok()?;
    }
    s.initialize_history();
    Some(s)
}

/// Warm up, then time `steps` of `step()` followed by a field readback (which
/// forces GPU completion, so the GPU row is not just submit latency). Returns
/// (ms_per_step, max|field|, finite).
fn time_steps(solver: &mut UnifiedSolver, read_field: &str, steps: usize) -> (f64, f64, bool) {
    for _ in 0..3 {
        solver.step();
    }
    let _ = pollster::block_on(solver.get_field_scalar(read_field));
    let t0 = Instant::now();
    for _ in 0..steps {
        solver.step();
    }
    let field = pollster::block_on(solver.get_field_scalar(read_field)).unwrap_or_default();
    let elapsed = t0.elapsed();
    let finite = field.iter().all(|v| v.is_finite());
    let maxabs = field.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    ((elapsed.as_secs_f64() * 1e3) / steps as f64, maxabs, finite)
}

fn run_suite(
    title: &str,
    mesh: &Mesh,
    steps: usize,
    read_field: &str,
    build: impl Fn(&Mesh) -> Option<UnifiedSolver>,
) {
    println!("\n=== {title}  (cells={}, steps={steps}) ===", mesh.num_cells());
    println!("(rel.GPU = ms/step ÷ GPU ms/step; <1.0 = faster than GPU)");
    println!("{:<22} {:>12} {:>12} {:>10}  {}", "variant", "ms/step", "rel.GPU", "max|f|", "status");
    let mut gpu_ms: Option<f64> = None;
    for (label, cpu) in variants() {
        set_backend_env(cpu);
        let Some(mut solver) = build(mesh) else {
            println!("{label:<22} {:>12} {:>12} {:>10}  unavailable", "-", "-", "-");
            continue;
        };
        let is_cpu = solver.is_cpu();
        // Guard: the env must have taken effect.
        if cpu.is_some() != is_cpu {
            println!("{label:<22} backend mismatch (is_cpu={is_cpu})");
            continue;
        }
        let (ms, maxf, finite) = time_steps(&mut solver, read_field, steps);
        if cpu.is_none() {
            gpu_ms = Some(ms);
        }
        let speedup = match gpu_ms {
            Some(g) if g > 0.0 => format!("{:.2}x", ms / g),
            _ => "-".to_string(),
        };
        let status = if finite { "ok" } else { "NON-FINITE" };
        println!("{label:<22} {ms:>12.3} {speedup:>12} {maxf:>10.3e}  {status}");
    }
    clear_backend_env();
}

#[ignore]
#[test]
fn benchmark_cpu_variants_vs_gpu() {
    // Incompressible: coupled Schur saddle-point (the solve dominates).
    let inc_mesh = generate_structured_rect_mesh(24, 24, 1.0, 1.0, BoundarySides::wall());
    run_suite("Incompressible (Taylor-Green, Coupled/Schur)", &inc_mesh, 15, "p", |m| {
        build_incompressible(m, 10)
    });

    // Compressible: full KT-flux / EOS-recovery (the flux + assembly dominate).
    let cmp_mesh = generate_structured_rect_mesh(
        24,
        24,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Inlet,
            bottom: BoundaryType::Inlet,
            top: BoundaryType::Inlet,
        },
    );
    run_suite("Compressible (KT flux, Implicit)", &cmp_mesh, 15, "rho", build_compressible);

    clear_backend_env();
}
