#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::generate_structured_backwards_step_mesh;

#[test]
fn backwards_step_mesh_cell_centers_match_structured_layout() {
    let mesh = generate_structured_backwards_step_mesh(30, 10, 3.0, 1.0, 0.5, 1.0);
    assert_eq!(mesh.num_cells(), 250);

    let mut pts: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| (mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    pts.sort_by_key(|(x, y)| common::yx_key(*x, *y));

    // First row is the downstream bottom band, starting at x=1.05, y=0.05 for dx=dy=0.1.
    let (x0, y0) = pts[0];
    assert!(
        (x0 - 1.05).abs() < 1e-12,
        "unexpected first x: {x0} (y={y0})"
    );
    assert!(
        (y0 - 0.05).abs() < 1e-12,
        "unexpected first y: {y0} (x={x0})"
    );

    let (x_last, y_last) = pts[pts.len() - 1];
    assert!((x_last - 2.95).abs() < 1e-12, "unexpected max x: {x_last}");
    assert!((y_last - 0.95).abs() < 1e-12, "unexpected max y: {y_last}");
}
