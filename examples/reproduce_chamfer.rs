#[cfg(not(feature = "meshgen"))]
fn main() {
    eprintln!("This example requires the `meshgen` feature.");
    eprintln!(
        "Try: cargo run --no-default-features --features meshgen --example reproduce_chamfer"
    );
}

#[cfg(feature = "meshgen")]
fn main() {
    use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, Geometry};
    use nalgebra::{Point2, Vector2};

    let geo = BackwardsStep {
        length: 2.0,
        height_inlet: 0.501,
        height_outlet: 1.0,
        step_x: 0.501,
    };

    let domain_size = Vector2::new(2.0, 1.0);

    let p00 = Point2::new(0.5, 0.5);
    println!("SDF at (0.5, 0.5): {}", geo.sdf(&p00));
    let p10 = Point2::new(0.6, 0.5);
    println!("SDF at (0.6, 0.5): {}", geo.sdf(&p10));
    let p01 = Point2::new(0.5, 0.6);
    println!("SDF at (0.5, 0.6): {}", geo.sdf(&p01));

    let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, 1.2, domain_size);

    println!("Generated mesh with {} cells", mesh.num_cells());

    let p_probe = Point2::new(0.55, 0.45);
    let cell_idx = mesh
        .get_cell_at_pos(p_probe)
        .expect("Should find a cell at probe position");

    let start = mesh.cell_vertex_offsets[cell_idx];
    let end = mesh.cell_vertex_offsets[cell_idx + 1];
    let num_vertices = end - start;

    println!("Corner cell has {} vertices", num_vertices);
    for k in 0..num_vertices {
        let idx = mesh.cell_vertices[start + k];
        println!("v{}: ({}, {})", k, mesh.vx[idx], mesh.vy[idx]);
    }

    assert!(
        num_vertices == 5 || num_vertices == 6,
        "Unexpected vertex count: {}",
        num_vertices
    );

    if num_vertices == 5 {
        println!("Reproduced chamfer issue: Cell has 5 vertices.");
    } else {
        println!("Issue not reproduced? Cell has 6 vertices.");
    }
}

