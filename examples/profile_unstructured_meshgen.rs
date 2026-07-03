//! Timing probe for the unstructured (Delaunay / Voronoi) mesh generators.
//!
//! ```sh
//! cargo run --release --features meshgen --example profile_unstructured_meshgen
//! ```

fn main() {
    use cfd2::solver::mesh::{
        generate_delaunay_mesh, generate_voronoi_mesh, ChannelWithObstacle,
    };
    use nalgebra::{Point2, Vector2};
    use std::time::Instant;

    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let domain_size = Vector2::new(3.0, 1.0);

    for h in [0.025, 0.01, 0.005] {
        let t0 = Instant::now();
        let mesh_d = generate_delaunay_mesh(&geo, h, h, 1.2, domain_size);
        let t_d = t0.elapsed();

        let t1 = Instant::now();
        let mesh_v = generate_voronoi_mesh(&geo, h, h, 1.2, domain_size);
        let t_v = t1.elapsed();

        println!(
            "h={h}: delaunay {:>8.1?} ({} cells) | voronoi {:>8.1?} ({} cells)",
            t_d,
            mesh_d.num_cells(),
            t_v,
            mesh_v.num_cells()
        );
    }
}
