//! GPU structured-grid solver parity against the CPU structured path.
//!
//! The GPU `StructuredGpuSolver` runs the identical `TopologyMode::Structured2D`
//! codegen kernels the CPU interpreter runs, then closes the loop with a
//! matrix-free banded GPU solve. These tests assert bit-close parity of the
//! assembled banded operator and of the converged field.
//!
//! Requires the `cpu` feature (for the CPU reference solver) and a GPU device.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::structured::StructuredCpuSolver;
use cfd2::solver::gpu::structured::{BcComp, Edge, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::model::generic_diffusion_demo_structured_model;

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
