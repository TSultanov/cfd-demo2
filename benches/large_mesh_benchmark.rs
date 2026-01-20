use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Point2, Vector2};
use std::time::Duration;

fn large_mesh_benchmark(c: &mut Criterion) {
    let geo = ChannelWithObstacle {
        length: 2.0,
        height: 1.0,
        obstacle_center: Point2::new(0.5, 0.5),
        obstacle_radius: 0.1,
    };
    let domain_size = Vector2::new(2.0, 1.0);

    // Target: ~2-3 million cells.
    // Area = 2.0.
    // 2,000,000 cells => Area/cell = 1e-6 => size = 0.001.
    // Let's try a slightly larger size first to ensure it runs in reasonable time for a test run,
    // then we can decrease it.
    // 0.01 => 20,000 cells.
    // 0.005 => 80,000 cells.
    // 0.002 => 500,000 cells.
    // 0.001 => 2,000,000 cells.

    let min_cell_size = 0.001; // 2 million cells
    let max_cell_size = 0.001;

    let mut group = c.benchmark_group("large_mesh");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("generate_cut_cell_mesh_2M", |b| {
        b.iter(|| {
            generate_cut_cell_mesh(
                black_box(&geo),
                black_box(min_cell_size),
                black_box(max_cell_size),
                black_box(1.2),
                black_box(domain_size),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, large_mesh_benchmark);
criterion_main!(benches);
