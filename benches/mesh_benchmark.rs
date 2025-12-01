use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Point2, Vector2};

fn mesh_generation_benchmark(c: &mut Criterion) {
    let geo = ChannelWithObstacle {
        length: 2.0,
        height: 1.0,
        obstacle_center: Point2::new(0.5, 0.5),
        obstacle_radius: 0.1,
    };
    let domain_size = Vector2::new(2.0, 1.0);
    // Use a reasonably fine mesh to make it slow enough to measure
    let min_cell_size = 0.02;
    let max_cell_size = 0.05;

    c.bench_function("generate_cut_cell_mesh", |b| {
        b.iter(|| {
            generate_cut_cell_mesh(
                black_box(&geo),
                black_box(min_cell_size),
                black_box(max_cell_size),
                black_box(1.2), // growth_rate
                black_box(domain_size),
            )
        })
    });
}

criterion_group!(benches, mesh_generation_benchmark);
criterion_main!(benches);
