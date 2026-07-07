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
