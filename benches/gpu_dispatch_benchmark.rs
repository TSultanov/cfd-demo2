/// GPU Dispatch Performance Benchmark
///
/// This benchmark measures GPU solver performance with focus on dispatch operations.
/// It provides a baseline for tracking performance improvements when optimizing
/// the number of GPU dispatches.
use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::Vector2;

/// Setup a solver for benchmarking
fn setup_solver(cell_size: f64) -> (GpuSolver, usize) {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let num_cells = mesh.num_cells();

    let mut solver = pollster::block_on(GpuSolver::new(&mesh));
    solver.set_dt(0.001);
    solver.set_viscosity(0.001);
    solver.set_density(1.0);
    solver.set_alpha_p(0.3);
    solver.set_alpha_u(0.7);
    solver.set_scheme(0); // Upwind for stability

    // Set initial conditions - inlet velocity
    let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        let cx = mesh.cell_cx[i];
        let cy = mesh.cell_cy[i];
        if cx < 0.05 && cy > 0.5 {
            u_init[i] = (1.0, 0.0);
        }
    }
    solver.set_u(&u_init);
    solver.initialize_history();

    (solver, num_cells)
}

/// Benchmark single step performance with medium mesh
fn bench_gpu_step_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_dispatch");
    group.sample_size(10);

    // Medium mesh - good balance for benchmarking
    let cell_size = 0.02;
    let (mut solver, num_cells) = setup_solver(cell_size);

    group.throughput(Throughput::Elements(num_cells as u64));
    group.bench_function(BenchmarkId::new("step_medium", num_cells), |b| {
        b.iter(|| {
            solver.step();
        });
    });

    group.finish();
}

/// Benchmark single step performance across different mesh sizes
fn bench_gpu_step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_dispatch_scaling");
    group.sample_size(10);

    // Test with different mesh sizes
    let cell_sizes = [0.05, 0.02, 0.01];

    for &cell_size in &cell_sizes {
        let (mut solver, num_cells) = setup_solver(cell_size);

        group.throughput(Throughput::Elements(num_cells as u64));
        group.bench_function(BenchmarkId::new("step", num_cells), |b| {
            b.iter(|| {
                solver.step();
            });
        });
    }

    group.finish();
}

/// Benchmark with profiling enabled to track dispatch statistics
fn bench_with_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_dispatch_profiled");
    group.sample_size(10);

    let cell_size = 0.02;
    let (mut solver, num_cells) = setup_solver(cell_size);
    solver.enable_detailed_profiling(true);

    group.throughput(Throughput::Elements(num_cells as u64));
    group.bench_function(BenchmarkId::new("step_profiled", num_cells), |b| {
        b.iter(|| {
            solver.start_profiling_session();
            solver.step();
            solver.end_profiling_session();
        });
    });

    // Print final profiling stats after benchmark
    solver.start_profiling_session();
    solver.step();
    solver.end_profiling_session();
    solver.print_profiling_report();

    group.finish();
}

/// Report dispatch statistics for a single run
fn report_dispatch_stats() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("GPU Dispatch Statistics Report");
    println!("{}", "=".repeat(80));

    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    // Medium mesh for detailed analysis
    let cell_size = 0.01;
    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    println!("\nMesh Statistics:");
    println!("  Cells: {}", mesh.num_cells());
    println!("  Faces: {}", mesh.face_owner.len());
    println!("  Cell Size: {}", cell_size);

    pollster::block_on(async {
        let mut solver = GpuSolver::new(&mesh).await;
        solver.set_dt(0.001);
        solver.set_viscosity(0.001);
        solver.set_density(1.0);
        solver.set_alpha_p(0.3);
        solver.set_alpha_u(0.7);
        solver.set_scheme(0);

        let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
        for i in 0..mesh.num_cells() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];
            if cx < 0.05 && cy > 0.5 {
                u_init[i] = (1.0, 0.0);
            }
        }
        solver.set_u(&u_init);
        solver.initialize_history();

        solver.enable_detailed_profiling(true);
        solver.start_profiling_session();

        // Run a single step for analysis
        solver.step();

        solver.end_profiling_session();
        solver.print_profiling_report();

        // Extract and display dispatch-specific stats
        let stats = solver.get_profiling_stats();
        let location_stats = stats.get_location_stats();

        let mut dispatch_stats: Vec<_> = location_stats
            .iter()
            .filter(|(k, _)| k.contains("GPU Dispatch"))
            .collect();
        dispatch_stats.sort_by(|a, b| b.1.call_count.cmp(&a.1.call_count));

        println!("\n");
        println!("{}", "=".repeat(80));
        println!("Dispatch Operations by Call Count (Top 20)");
        println!("{}", "=".repeat(80));
        println!(
            "{:<50} {:>10} {:>12}",
            "Operation", "Calls", "Total Time"
        );
        println!("{}", "-".repeat(74));

        for (location, stats) in dispatch_stats.iter().take(20) {
            let name = location.replace("GPU Dispatch:", "");
            println!(
                "{:<50} {:>10} {:>12?}",
                name, stats.call_count, stats.total_time
            );
        }

        // Summary
        let total_dispatches: u64 = dispatch_stats.iter().map(|(_, s)| s.call_count).sum();
        let total_dispatch_time: std::time::Duration =
            dispatch_stats.iter().map(|(_, s)| s.total_time).sum();

        println!("\n{}", "-".repeat(74));
        println!(
            "{:<50} {:>10} {:>12?}",
            "TOTAL", total_dispatches, total_dispatch_time
        );

        println!("\nAnalysis:");
        println!("  Total GPU dispatches per step: {}", total_dispatches);
        println!(
            "  Average dispatch overhead: {:?}",
            total_dispatch_time / total_dispatches as u32
        );
        println!(
            "  Estimated reduction potential: Look for operations with >100 calls"
        );
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets = bench_gpu_step_medium, bench_gpu_step_scaling
}

criterion_group! {
    name = profiled_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_with_profiling
}

criterion_main!(benches, profiled_benches);

// Standalone function for quick dispatch analysis
// Run with: cargo bench --bench gpu_dispatch_benchmark -- --nocapture dispatch_stats
#[test]
fn dispatch_stats() {
    report_dispatch_stats();
}
