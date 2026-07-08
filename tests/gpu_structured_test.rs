//! GPU structured-grid solver parity against the CPU structured path.
//!
//! The GPU `StructuredGpuSolver` runs the identical `TopologyMode::Structured2D`
//! codegen kernels the CPU interpreter runs, then closes the loop with a
//! matrix-free banded GPU solve. These tests assert bit-close parity of the
//! assembled banded operator and of the converged field.
//!
//! Requires the `cpu` feature (for the CPU reference solver) and a GPU device.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::structured::{StructuredCpuSolver, StructuredModelSolver};
use cfd2::solver::gpu::structured::{BcComp, Edge, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::model::{
    allmach_thermal_structured_model, compressible_structured_model,
    generic_diffusion_demo_structured_model, incompressible_momentum_structured_model,
};

/// The GPU-assembled banded 5-point operator must match the CPU one (same model,
/// grid, IC and Dirichlet BC) — isolates the GPU kernel/dispatch/grid-uniform
/// plumbing from the linear solve.
#[test]
fn gpu_structured_diffusion_assembly_matches_cpu() {
    let (nx, ny) = (16usize, 12usize);
    let (lx, ly) = (1.3, 1.0);
    let dt = 0.5;
    let model = generic_diffusion_demo_structured_model().expect("model");

    let ic = |x: f64, y: f64| (x + 0.3 * y).sin();
    let dir = |x: f64, y: f64| x * x - y;

    // CPU reference.
    let mut cpu = StructuredCpuSolver::new(StructuredGrid::new(nx, ny, lx, ly), &model, dt)
        .expect("cpu solver");
    cpu.set_scalar(ic);
    cpu.set_dirichlet(dir);
    cpu.assemble_only();
    let cpu_mat = cpu.matrix_bands();

    // GPU.
    let mut gpu = StructuredGpuSolver::new(StructuredGrid::new(nx, ny, lx, ly), &model, dt, 1)
        .expect("gpu solver");
    gpu.set_named_field("phi", ic);
    // Dirichlet on every edge, boundary type 3 (Wall) — matching the CPU scalar
    // solver's default face_boundary tagging + set_dirichlet.
    gpu.set_boundaries(|_edge: Edge, fx, fy| {
        (
            3,
            vec![BcComp {
                kind: 1,
                value: dir(fx, fy) as f32,
            }],
        )
    });
    gpu.assemble_only();
    let gpu_mat = gpu.matrix_values();

    assert_eq!(cpu_mat.len(), gpu_mat.len(), "band count mismatch");
    let mut max_abs = 0.0f32;
    for (i, (a, b)) in cpu_mat.iter().zip(&gpu_mat).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(
            d <= 1e-4 + 1e-4 * a.abs(),
            "band {i}: cpu {a} gpu {b} (|Δ|={d})"
        );
    }
    println!("[gpu-structured] assembly max |Δ| = {max_abs:e}");
}

/// End-to-end: the GPU structured scalar-diffusion solve (banded block-Jacobi CG
/// on the device) must recover the same steady field as the CPU banded CG.
#[test]
fn gpu_structured_diffusion_solve_matches_cpu() {
    let (nx, ny) = (24usize, 20usize);
    let (lx, ly) = (1.0, 1.0);
    let dt = 1.0e6; // near-steady in one implicit step
    let model = generic_diffusion_demo_structured_model().expect("model");

    // Harmonic Dirichlet data phi = x (linear) — the steady solution is phi = x.
    let dir = |x: f64, _y: f64| x;

    let mut cpu = StructuredCpuSolver::new(StructuredGrid::new(nx, ny, lx, ly), &model, dt)
        .expect("cpu solver");
    cpu.set_scalar(|_, _| 0.0);
    cpu.set_dirichlet(dir);
    cpu.solve_to_steady(1e-12, 200);
    let cpu_field = cpu.scalar_field();

    let mut gpu = StructuredGpuSolver::new(StructuredGrid::new(nx, ny, lx, ly), &model, dt, 1)
        .expect("gpu solver");
    gpu.set_named_field("phi", |_, _| 0.0);
    gpu.set_boundaries(|_edge: Edge, fx, fy| {
        (
            3,
            vec![BcComp {
                kind: 1,
                value: dir(fx, fy) as f32,
            }],
        )
    });
    for _ in 0..20 {
        gpu.step();
    }
    let gpu_field = gpu.state_field(0);

    assert_eq!(cpu_field.len(), gpu_field.len());
    let mut max_abs = 0.0f64;
    for (p, (a, b)) in cpu_field.iter().zip(&gpu_field).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(d < 2e-3, "cell {p}: cpu {a} gpu {b} (|Δ|={d})");
    }
    // And both must recover phi = x to a couple digits.
    let mut max_exact = 0.0f64;
    for p in 0..gpu_field.len() {
        let x = (p % nx) as f64 / nx as f64 + 0.5 / nx as f64;
        max_exact = max_exact.max((gpu_field[p] - x * lx).abs());
    }
    println!("[gpu-structured] solve max |Δ_cpu| = {max_abs:e}, |Δ_exact| = {max_exact:e}");
    assert!(max_exact < 5e-3, "GPU field must recover phi=x, max err {max_exact}");
}

/// DEBUG: stage-by-stage buffer diff after one momentum assembly pass, to locate
/// the first diverging kernel. Not a hard gate (prints max diffs).
#[test]
fn gpu_structured_momentum_stage_diff() {
    let (nx, ny) = (12usize, 12usize);
    let model = incompressible_momentum_structured_model().expect("model");
    let s = 3usize;
    let sstride = model.state_layout.stride() as usize;
    let n = nx * ny;
    let bc = |edge: Edge| {
        let u_wall = if matches!(edge, Edge::Top) { 1.0 } else { 0.0 };
        let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
        (
            btype,
            vec![
                BcComp { kind: 1, value: u_wall },
                BcComp { kind: 1, value: 0.0 },
                BcComp { kind: 2, value: 0.0 },
            ],
        )
    };
    let mut cpu =
        StructuredModelSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3).unwrap();
    cpu.set_fluid(1.0, 0.01);
    cpu.set_boundaries(|e, _x, _y| bc(e));
    cpu.assemble_only();
    let mut gpu =
        StructuredGpuSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3).unwrap();
    gpu.set_fluid(1.0, 0.01);
    gpu.set_boundaries(|e, _x, _y| bc(e));
    gpu.assemble_only();

    let cmp = |name: &str, a: &[f32], b: &[f32]| {
        let mut md = 0.0f32;
        let mut idx = 0;
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            if d > md {
                md = d;
                idx = i;
            }
        }
        println!(
            "  {name:<26} max|Δ|={md:e} at {idx} (cpu {} gpu {})",
            a.get(idx).copied().unwrap_or(0.0),
            b.get(idx).copied().unwrap_or(0.0)
        );
    };
    println!("[stage-diff]");
    let cs = cpu.read_buffer("state");
    let gs = gpu.read_named("state", n * sstride);
    cmp("state", &cs, &gs);
    cmp("grad_state", &cpu.read_buffer("grad_state"), &gpu.read_named("grad_state", n * sstride * 2));
    cmp("fluxes", &cpu.read_buffer("fluxes"), &gpu.read_named("fluxes", n * 4 * 3));
    let cm = cpu.matrix_values();
    let gm = gpu.matrix_values();
    cmp("matrix_values", &cm, &gm);
    cmp("rhs", &cpu.rhs(), &gpu.rhs());
    // Detail for cell 0: full state row + diagonal block (band 2).
    println!("  cell0 state cpu={:?}", &cs[0..sstride]);
    println!("  cell0 state gpu={:?}", &gs[0..sstride]);
    let db = |m: &[f32]| {
        let base = 0 * 5 * s * s;
        (0..s)
            .map(|r| (0..s).map(|c| m[base + 5 * s * r + 2 * s + c]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    println!("  cell0 diag cpu={:?}", db(&cm));
    println!("  cell0 diag gpu={:?}", db(&gm));
    let cf = cpu.read_buffer("fluxes");
    let gf = gpu.read_named("fluxes", n * 4 * 3);
    println!("  cell0 fluxes cpu={:?}", &cf[0..12]);
    println!("  cell0 fluxes gpu={:?}", &gf[0..12]);
}

/// COUPLED assembly isolation: the GPU-assembled momentum banded operator (with
/// the structured flux + Green–Gauss gradient kernels feeding it) must match the
/// CPU StructuredModelSolver's — isolates the GPU kernel path from the solve.
#[test]
fn gpu_structured_momentum_assembly_matches_cpu() {
    let (nx, ny) = (12usize, 12usize);
    let model = incompressible_momentum_structured_model().expect("model");
    let bc = |edge: Edge| {
        let u_wall = if matches!(edge, Edge::Top) { 1.0 } else { 0.0 };
        let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
        (
            btype,
            vec![
                BcComp { kind: 1, value: u_wall },
                BcComp { kind: 1, value: 0.0 },
                BcComp { kind: 2, value: 0.0 },
            ],
        )
    };

    let mut cpu =
        StructuredModelSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3)
            .expect("cpu");
    cpu.set_fluid(1.0, 0.01);
    cpu.set_boundaries(|edge, _x, _y| bc(edge));
    cpu.assemble_only();
    let cpu_mat = cpu.matrix_values();
    let cpu_rhs = cpu.rhs();

    let mut gpu = StructuredGpuSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3)
        .expect("gpu");
    gpu.set_fluid(1.0, 0.01);
    gpu.set_boundaries(|edge, _x, _y| bc(edge));
    gpu.assemble_only();
    let gpu_mat = gpu.matrix_values();
    let gpu_rhs = gpu.rhs();

    assert_eq!(cpu_mat.len(), gpu_mat.len(), "matrix band count mismatch");
    let mut max_m = 0.0f32;
    for (i, (a, b)) in cpu_mat.iter().zip(&gpu_mat).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "non-finite mat[{i}]: cpu {a} gpu {b}");
        let d = (a - b).abs();
        max_m = max_m.max(d);
        assert!(d <= 1e-4 + 1e-4 * a.abs(), "mat[{i}]: cpu {a} gpu {b} (|Δ|={d})");
    }
    let mut max_r = 0.0f32;
    for (i, (a, b)) in cpu_rhs.iter().zip(&gpu_rhs).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "non-finite rhs[{i}]: cpu {a} gpu {b}");
        let d = (a - b).abs();
        max_r = max_r.max(d);
        assert!(d <= 1e-4 + 1e-4 * a.abs(), "rhs[{i}]: cpu {a} gpu {b} (|Δ|={d})");
    }
    println!("[gpu-structured] momentum assembly max|Δmat|={max_m:e} max|Δrhs|={max_r:e}");
}

/// BC TYPES: exercise the SlipWall (type 4) free-slip boundary on the structured
/// path (flux + assembly reflect the velocity, removing its normal component) and
/// confirm GPU == CPU. Together with the lid-cavity (Wall type 3 + MovingWall type
/// 5) this covers every velocity boundary type structured.
#[test]
fn gpu_structured_slipwall_matches_cpu() {
    let (nx, ny) = (16usize, 16usize);
    let steps = 12;
    let model = incompressible_momentum_structured_model().expect("model");

    // Top lid slides (MovingWall 5); the other three walls are SLIP (type 4).
    let bc = |edge: Edge| {
        if matches!(edge, Edge::Top) {
            (
                5u32,
                vec![
                    BcComp { kind: 1, value: 1.0 },
                    BcComp { kind: 1, value: 0.0 },
                    BcComp { kind: 2, value: 0.0 },
                ],
            )
        } else {
            (
                4u32,
                vec![
                    BcComp { kind: 0, value: 0.0 },
                    BcComp { kind: 0, value: 0.0 },
                    BcComp { kind: 2, value: 0.0 },
                ],
            )
        }
    };

    let mut cpu =
        StructuredModelSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3).unwrap();
    cpu.set_fluid(1.0, 0.01);
    cpu.set_boundaries(|e, _x, _y| bc(e));
    for _ in 0..steps {
        cpu.step();
    }
    let cpu_ux = cpu.state_field(0);

    let mut gpu =
        StructuredGpuSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3).unwrap();
    gpu.set_fluid(1.0, 0.01);
    gpu.set_boundaries(|e, _x, _y| bc(e));
    for _ in 0..steps {
        gpu.step();
    }
    let gpu_ux = gpu.state_field(0);
    let gpu_uy = gpu.state_field(1);

    let mut umax = 0.0f64;
    let mut max_d = 0.0f64;
    for p in 0..cpu_ux.len() {
        assert!(gpu_ux[p].is_finite() && gpu_uy[p].is_finite(), "slip GPU diverged");
        umax = umax.max(gpu_ux[p].hypot(gpu_uy[p]));
        max_d = max_d.max((cpu_ux[p] - gpu_ux[p]).abs());
    }
    assert!(umax > 0.05 && umax < 5.0, "unphysical slip-wall speed {umax}");
    let side_col_speed: f64 = (0..ny).map(|j| gpu_ux[j * nx + 1].abs()).sum::<f64>() / ny as f64;
    println!("[gpu-structured] slipwall: umax={umax:.4}, side_col_ux={side_col_speed:.4}, max|Δcpu|={max_d:e}");
    assert!(max_d < 3e-2, "GPU vs CPU slip-wall mismatch {max_d}");
}

/// COUPLED: the GPU structured incompressible-momentum lid cavity (block SpMV +
/// block-Jacobi BiCGStab on the indefinite U–p system, plus the structured
/// flux/gradients/assembly kernels on the device) must reproduce the CPU
/// StructuredModelSolver's lid-driven flow.
#[test]
fn gpu_structured_momentum_lid_cavity_matches_cpu() {
    let (nx, ny) = (16usize, 16usize);
    let steps = 15;
    let model = incompressible_momentum_structured_model().expect("model");

    // Lid-cavity BCs: no-slip walls, top wall slides at u=1, p Neumann 0.
    // Components: 0=Ux, 1=Uy, 2=p.
    let bc = |edge: Edge| {
        let u_wall = if matches!(edge, Edge::Top) { 1.0 } else { 0.0 };
        let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
        (
            btype,
            vec![
                BcComp { kind: 1, value: u_wall },
                BcComp { kind: 1, value: 0.0 },
                BcComp { kind: 2, value: 0.0 },
            ],
        )
    };

    // CPU reference.
    let mut cpu =
        StructuredModelSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3)
            .expect("cpu solver");
    cpu.set_fluid(1.0, 0.01);
    cpu.set_boundaries(|edge, _x, _y| bc(edge));
    for _ in 0..steps {
        cpu.step();
    }
    let cpu_ux = cpu.state_field(0);
    let cpu_uy = cpu.state_field(1);

    // GPU.
    let mut gpu = StructuredGpuSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.05, 3)
        .expect("gpu solver");
    assert_eq!(gpu.unknowns(), 3, "coupled Ux/Uy/p");
    gpu.set_fluid(1.0, 0.01);
    gpu.set_boundaries(|edge, _x, _y| bc(edge));
    for _ in 0..steps {
        gpu.step();
    }
    let gpu_ux = gpu.state_field(0);
    let gpu_uy = gpu.state_field(1);

    // Physics sanity on the GPU field (same asserts as the CPU test).
    let mut umax = 0.0f64;
    for (&a, &b) in gpu_ux.iter().zip(&gpu_uy) {
        assert!(a.is_finite() && b.is_finite(), "GPU velocity diverged");
        umax = umax.max(a.hypot(b));
    }
    assert!(umax > 0.05 && umax < 5.0, "unphysical GPU lid-cavity speed {umax}");
    let top_row_mean_ux: f64 = (0..nx).map(|i| gpu_ux[(ny - 1) * nx + i]).sum::<f64>() / nx as f64;
    assert!(
        top_row_mean_ux > 0.1,
        "GPU near-lid x-velocity did not develop ({top_row_mean_ux})"
    );

    // Parity to the CPU field. Two different Krylov solvers (GMRES vs BiCGStab)
    // over 40 nonlinear steps in f32 — a modest tolerance on the developed flow.
    let mut max_d = 0.0f64;
    for p in 0..cpu_ux.len() {
        max_d = max_d
            .max((cpu_ux[p] - gpu_ux[p]).abs())
            .max((cpu_uy[p] - gpu_uy[p]).abs());
    }
    println!("[gpu-structured] momentum lid: umax={umax:.4}, top_ux={top_row_mean_ux:.4}, max|Δcpu|={max_d:e}");
    assert!(max_d < 3e-2, "GPU vs CPU lid-cavity velocity mismatch {max_d}");
}

/// EOS FAMILY: the all-Mach THERMAL structured pipeline (Ux/Uy/p/T + on-device
/// EOS density recovery + low_mach_params uniform) runs its full kernel schedule
/// on the GPU and matches the CPU StructuredModelSolver's lid-driven flow.
#[test]
fn gpu_structured_thermal_lid_matches_cpu() {
    let (nx, ny) = (16usize, 16usize);
    let steps = 10;
    let model = allmach_thermal_structured_model().expect("model");
    let s = model.system.unknowns_per_cell() as usize;
    let psi = 0.5f64;
    let seed = |solver: &mut dyn ThermalSeed| {
        solver.set_fluid(1.0, 0.02);
        solver.seed("psi", psi);
        solver.seed("psi_precond", psi.max(1.0));
        solver.seed("rho", 1.0);
        solver.seed("rho_t_ref", 1.0);
        solver.seed("T", 1.0);
        solver.seed_if("t_ref", 1.0);
        solver.seed_if("rho_floor", psi * 1.0e-5);
    };
    let bc = move |edge: Edge| {
        let u_wall = if matches!(edge, Edge::Top) { 1.0 } else { 0.0 };
        let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
        let mut v = vec![
            BcComp { kind: 1, value: u_wall },
            BcComp { kind: 1, value: 0.0 },
            BcComp { kind: 2, value: 0.0 },
        ];
        if s >= 4 {
            v.push(BcComp { kind: 2, value: 0.0 });
        }
        (btype, v)
    };

    let mut cpu =
        StructuredModelSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.02, 3).unwrap();
    seed(&mut CpuThermal(&mut cpu));
    cpu.set_boundaries(|e, _x, _y| bc(e));
    for _ in 0..steps {
        cpu.step();
    }
    let cpu_ux = cpu.state_field(0);

    let mut gpu =
        StructuredGpuSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.02, 3).unwrap();
    seed(&mut GpuThermal(&mut gpu));
    gpu.set_boundaries(|e, _x, _y| bc(e));
    for _ in 0..steps {
        gpu.step();
    }
    let gpu_ux = gpu.state_field(0);
    let gpu_uy = gpu.state_field(1);

    let mut umax = 0.0f64;
    let mut max_d = 0.0f64;
    for p in 0..cpu_ux.len() {
        assert!(gpu_ux[p].is_finite() && gpu_uy[p].is_finite(), "thermal GPU diverged");
        umax = umax.max(gpu_ux[p].hypot(gpu_uy[p]));
        max_d = max_d.max((cpu_ux[p] - gpu_ux[p]).abs());
    }
    assert!(umax > 0.02 && umax < 5.0, "unphysical thermal speed {umax}");
    println!("[gpu-structured] thermal lid: umax={umax:.4}, max|Δcpu|={max_d:e}");
    assert!(max_d < 3e-2, "GPU vs CPU thermal mismatch {max_d}");
}

/// EOS FAMILY: the density-based COMPRESSIBLE structured pipeline (s conserved
/// unknowns + central-upwind flux + on-device primitive recovery + the
/// expression-BC closure `bc_expr`, whose per-kernel EOS constants are packed to
/// its reduced Constants layout) runs on the GPU and matches the CPU on a uniform
/// gas box that must stay bounded.
#[test]
fn gpu_structured_compressible_box_matches_cpu() {
    let (nx, ny) = (16usize, 16usize);
    let steps = 20;
    let model = compressible_structured_model().expect("model");
    let s = model.system.unknowns_per_cell() as usize;
    let (rho0, e0, p0) = (1.0, 2.5, 1.0);
    let seed = |g: &mut StructuredGpuSolver| {
        g.set_fluid(1.0, 0.0);
        g.set_named_field("rho", |_, _| rho0);
        g.set_named_field("rho_e", |_, _| e0);
        g.set_named_field("p", |_, _| p0);
        g.set_named_field("T", |_, _| 1.0);
        if g.field_offset("mu").is_some() {
            g.set_named_field("mu", |_, _| 0.0);
        }
    };
    let bc = move |_edge: Edge| {
        let mut v = vec![
            BcComp { kind: 1, value: rho0 as f32 },
            BcComp { kind: 1, value: 0.0 },
            BcComp { kind: 1, value: 0.0 },
        ];
        if s >= 4 {
            v.push(BcComp { kind: 1, value: e0 as f32 });
        }
        (3u32, v) // Wall
    };

    let mut cpu =
        StructuredModelSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.01, 1).unwrap();
    cpu.set_fluid(1.0, 0.0);
    cpu.set_named_field("rho", |_, _| rho0);
    cpu.set_named_field("rho_e", |_, _| e0);
    cpu.set_named_field("p", |_, _| p0);
    cpu.set_named_field("T", |_, _| 1.0);
    if cpu.field_offset("mu").is_some() {
        cpu.set_named_field("mu", |_, _| 0.0);
    }
    cpu.set_boundaries(|e, _x, _y| bc(e));
    for _ in 0..steps {
        cpu.step();
    }
    let cpu_rho = cpu.state_field(cpu.field_offset("rho").unwrap());

    let mut gpu =
        StructuredGpuSolver::new(StructuredGrid::new(nx, ny, 1.0, 1.0), &model, 0.01, 1).unwrap();
    seed(&mut gpu);
    gpu.set_boundaries(|e, _x, _y| bc(e));
    for _ in 0..steps {
        gpu.step();
    }
    let rho_off = gpu.field_offset("rho").unwrap();
    let gpu_rho = gpu.state_field(rho_off);
    let gpu_re = gpu.state_field(gpu.field_offset("rho_e").unwrap());

    let mut max_d = 0.0f64;
    for (p, (&cr, &gr)) in cpu_rho.iter().zip(&gpu_rho).enumerate() {
        assert!(gr.is_finite() && gpu_re[p].is_finite(), "compressible GPU diverged");
        assert!(gr > 0.5 && gr < 2.0, "compressible density drifted: {gr}");
        max_d = max_d.max((cr - gr).abs());
    }
    println!("[gpu-structured] compressible box: max|Δcpu rho|={max_d:e}");
    assert!(max_d < 5e-3, "GPU vs CPU compressible mismatch {max_d}");
}

// Small shims so the thermal seeding is shared between the CPU and GPU solvers.
trait ThermalSeed {
    fn set_fluid(&mut self, d: f64, v: f64);
    fn seed(&mut self, name: &str, val: f64);
    fn seed_if(&mut self, name: &str, val: f64);
}
struct CpuThermal<'a>(&'a mut StructuredModelSolver);
impl ThermalSeed for CpuThermal<'_> {
    fn set_fluid(&mut self, d: f64, v: f64) {
        self.0.set_fluid(d, v);
    }
    fn seed(&mut self, name: &str, val: f64) {
        self.0.set_named_field(name, |_, _| val);
    }
    fn seed_if(&mut self, name: &str, val: f64) {
        if self.0.field_offset(name).is_some() {
            self.0.set_named_field(name, |_, _| val);
        }
    }
}
struct GpuThermal<'a>(&'a mut StructuredGpuSolver);
impl ThermalSeed for GpuThermal<'_> {
    fn set_fluid(&mut self, d: f64, v: f64) {
        self.0.set_fluid(d, v);
    }
    fn seed(&mut self, name: &str, val: f64) {
        self.0.set_named_field(name, |_, _| val);
    }
    fn seed_if(&mut self, name: &str, val: f64) {
        if self.0.field_offset(name).is_some() {
            self.0.set_named_field(name, |_, _| val);
        }
    }
}

/// GUI surface: the accessors the GUI worker drives — `get_u` (paired velocity),
/// `get_scalar`, `time`/`dt`, `model_id`, `state_layout`, `copy_state_to_buffer`
/// (the viz feed) — behave correctly on a lid-driven cavity built like the GUI's
/// `build_structured_init`.
#[test]
fn gpu_structured_gui_accessors_work() {
    use cfd2::solver::UiPortSet;
    let ctx = pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None)).unwrap();
    let device = ctx.device.clone();
    let queue = ctx.queue.clone();
    let model = incompressible_momentum_structured_model().unwrap();
    let (nx, ny) = (12usize, 12usize);
    let mut s = StructuredGpuSolver::with_context(
        ctx,
        StructuredGrid::new(nx, ny, 1.0, 1.0),
        &model,
        0.05,
        3,
    )
    .unwrap();
    s.set_fluid(1.0, 0.01);
    let u_lid = 1.0f64;
    s.set_boundaries(move |edge, _x, _y| {
        let uw = if matches!(edge, Edge::Top) { u_lid as f32 } else { 0.0 };
        let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
        (
            btype,
            vec![
                BcComp { kind: 1, value: uw },
                BcComp { kind: 1, value: 0.0 },
                BcComp { kind: 2, value: 0.0 },
            ],
        )
    });

    assert_eq!(s.model_id(), "incompressible_momentum_structured");
    assert!((s.dt() - 0.05).abs() < 1e-12);
    let t0 = s.time();
    for _ in 0..8 {
        s.step();
    }
    assert!((s.time() - (t0 + 8.0 * 0.05)).abs() < 1e-9, "sim time accumulates");

    let ports = UiPortSet::from_layout(s.state_layout());
    let u = s.get_u(ports.u_offset.unwrap() as usize);
    assert_eq!(u.len(), nx * ny, "get_u one pair per cell");
    let mut umax = 0.0f64;
    for &(ux, uy) in &u {
        assert!(ux.is_finite() && uy.is_finite(), "finite velocity");
        umax = umax.max(ux.hypot(uy));
    }
    assert!(umax > 0.05 && umax < 5.0, "lid-driven flow developed (umax={umax})");

    // copy_state_to_buffer (viz feed): copy packed state to a same-device buffer,
    // read it back, and confirm component 0 matches state_field(0).
    let sstride = model.state_layout.stride() as usize;
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_viz"),
        size: s.state_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    s.copy_state_to_buffer(&dst);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_viz_staging"),
        size: s.state_size_bytes(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, s.state_size_bytes());
    let idx = queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let viz: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    let ux0 = s.get_scalar(0);
    for p in 0..nx * ny {
        assert!(
            (viz[p * sstride] as f64 - ux0[p]).abs() < 1e-6,
            "viz buffer matches state at cell {p}"
        );
    }
    println!("[gpu-structured] GUI accessors ok: umax={umax:.4}");
}
