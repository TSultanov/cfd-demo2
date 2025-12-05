use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Point2, Vector2};
use std::time::Duration;

fn mesh_smoothing_benchmark(c: &mut Criterion) {
    let geo = ChannelWithObstacle {
        length: 2.0,
        height: 1.0,
        obstacle_center: Point2::new(0.5, 0.5),
        obstacle_radius: 0.1,
    };
    let domain_size = Vector2::new(2.0, 1.0);

    // Requested cell size
    let min_cell_size = 0.00175;
    let max_cell_size = 0.00175;

    // Generate mesh once
    let mesh = generate_cut_cell_mesh(
        &geo,
        min_cell_size,
        max_cell_size,
        1.2,
        domain_size,
    );

    let mut group = c.benchmark_group("mesh_smoothing");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("smooth_mesh_0.00175", |b| {
        b.iter(|| {
            let mut m = mesh.clone();
            m.smooth(black_box(&geo), black_box(0.0), black_box(10));
        })
    });
    group.finish();
}

criterion_group!(benches, mesh_smoothing_benchmark);
criterion_main!(benches);
