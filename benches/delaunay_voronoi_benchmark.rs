use cfd2::solver::mesh::{generate_delaunay_mesh, generate_voronoi_mesh, BackwardsStep};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::Vector2;

fn delaunay_benchmark(c: &mut Criterion) {
    let geo = BackwardsStep {
        length: 3.5,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let domain_size = Vector2::new(3.5, 1.0);
    let min_cell_size = 0.01;
    let max_cell_size = 0.01;
    let growth_rate = 1.2;

    c.bench_function("generate_delaunay_mesh_0.01", |b| {
        b.iter(|| {
            generate_delaunay_mesh(
                black_box(&geo),
                black_box(min_cell_size),
                black_box(max_cell_size),
                black_box(growth_rate),
                black_box(domain_size),
            )
        })
    });
}

fn voronoi_benchmark(c: &mut Criterion) {
    let geo = BackwardsStep {
        length: 3.5,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let domain_size = Vector2::new(3.5, 1.0);
    let min_cell_size = 0.01;
    let max_cell_size = 0.01;
    let growth_rate = 1.2;

    c.bench_function("generate_voronoi_mesh_0.01", |b| {
        b.iter(|| {
            generate_voronoi_mesh(
                black_box(&geo),
                black_box(min_cell_size),
                black_box(max_cell_size),
                black_box(growth_rate),
                black_box(domain_size),
            )
        })
    });
}

criterion_group!(benches, delaunay_benchmark, voronoi_benchmark);
criterion_main!(benches);
