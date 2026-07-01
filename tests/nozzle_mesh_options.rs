//! The GUI nozzle case can now be meshed with every mesh type, not just the
//! body-fitted structured grid. This test validates the mesh-generation side of
//! that feature (`src/ui/app.rs` `build_mesh_with`, `GeometryType::Nozzle`):
//!
//!  - the `Fitted` option reproduces the validated `generate_structured_nozzle_mesh`;
//!  - the unstructured options (CutCell / Delaunay / Voronoi) conform to the same
//!    wall profile via the `Nozzle` SDF geometry and, after the same open-face
//!    → wall fallback the GUI applies, form a closed nozzle domain with the
//!    left edge tagged `Inlet`, the right edge `Outlet`, and every other
//!    boundary face a `Wall` (flat bottom + curved contour).
//!
//! Physics (the pressure-inlet supersonic solve) is covered by
//! `allmach_thermal_supersonic_test`; here we only assert the meshes are valid
//! and correctly tagged so those boundary conditions have faces to bind to.
#![cfg(feature = "meshgen")]

use cfd2::solver::mesh::{
    generate_cut_cell_mesh, generate_delaunay_mesh, generate_structured_nozzle_mesh,
    generate_voronoi_mesh, BoundarySides, BoundaryType, Mesh, Nozzle,
};
use nalgebra::Vector2;

// The GUI nozzle geometry (src/ui/app.rs `GeometryType::Nozzle`).
const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80;

fn nozzle_geo() -> Nozzle {
    Nozzle {
        length: LENGTH,
        height: HEIGHT,
        throat_height: THROAT_H,
        throat_frac: THROAT_FRAC,
        exit_height: EXIT_H,
    }
}

/// Mirror of the GUI's open-face fallback: any boundary face left untyped by
/// `classify_boundary` (the curved top wall sits below the bounding box) is a
/// solid no-slip wall.
fn close_open_faces_as_walls(mesh: &mut Mesh) {
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_none() && mesh.face_boundary[f].is_none() {
            mesh.face_boundary[f] = Some(BoundaryType::Wall);
        }
    }
}

fn count_boundary(mesh: &Mesh, kind: BoundaryType) -> usize {
    (0..mesh.num_faces())
        .filter(|&f| mesh.face_boundary[f] == Some(kind))
        .count()
}

/// Every boundary face (no neighbour) must be tagged, and interior faces must
/// not be tagged. Returns the (inlet, outlet, wall) face counts.
fn assert_closed_and_tagged(mesh: &Mesh, label: &str) -> (usize, usize, usize) {
    for f in 0..mesh.num_faces() {
        let is_boundary = mesh.face_neighbor[f].is_none();
        let is_tagged = mesh.face_boundary[f].is_some();
        assert_eq!(
            is_boundary, is_tagged,
            "{label}: face {f} boundary/tag mismatch (open faces must be walls, \
             interior faces untagged)"
        );
    }
    let inlet = count_boundary(mesh, BoundaryType::Inlet);
    let outlet = count_boundary(mesh, BoundaryType::Outlet);
    let wall = count_boundary(mesh, BoundaryType::Wall);
    assert!(inlet > 0, "{label}: no Inlet faces (left edge)");
    assert!(outlet > 0, "{label}: no Outlet faces (right edge)");
    assert!(wall > 0, "{label}: no Wall faces (contour)");
    (inlet, outlet, wall)
}

/// Inlet faces sit on x≈0, outlet faces on x≈length; walls are elsewhere.
fn assert_boundary_positions(mesh: &Mesh, label: &str) {
    let eps = 1e-3 * LENGTH;
    for f in 0..mesh.num_faces() {
        match mesh.face_boundary[f] {
            Some(BoundaryType::Inlet) => assert!(
                mesh.face_cx[f].abs() < eps,
                "{label}: Inlet face {f} not on left edge (x={})",
                mesh.face_cx[f]
            ),
            Some(BoundaryType::Outlet) => assert!(
                (mesh.face_cx[f] - LENGTH).abs() < eps,
                "{label}: Outlet face {f} not on right edge (x={})",
                mesh.face_cx[f]
            ),
            _ => {}
        }
    }
}

/// All cells sit inside the nozzle contour: 0 ≤ y ≤ wall(x) at the centroid.
fn assert_cells_inside(mesh: &Mesh, label: &str) {
    let geo = nozzle_geo();
    for i in 0..mesh.num_cells() {
        let x = mesh.cell_cx[i];
        let y = mesh.cell_cy[i];
        let wall = cfd2::solver::mesh::structured::nozzle_height(
            (x / LENGTH).clamp(0.0, 1.0),
            HEIGHT,
            THROAT_H,
            THROAT_FRAC,
            EXIT_H,
        );
        // A small tolerance for cut-cell centroids that hug the wall.
        assert!(
            y > -1e-6 && y < wall + 0.05 * HEIGHT,
            "{label}: cell {i} centroid ({x:.3},{y:.3}) outside nozzle (wall={wall:.3})"
        );
        let _ = &geo;
    }
}

fn check_unstructured(label: &str, mut mesh: Mesh) {
    assert!(mesh.num_cells() > 100, "{label}: too few cells");
    close_open_faces_as_walls(&mut mesh);
    let (inlet, outlet, wall) = assert_closed_and_tagged(&mesh, label);
    assert_boundary_positions(&mesh, label);
    assert_cells_inside(&mesh, label);
    // The curved + flat walls dominate the perimeter of a long thin channel.
    assert!(
        wall > inlet && wall > outlet,
        "{label}: expected the contour walls to outnumber inlet/outlet \
         (inlet={inlet} outlet={outlet} wall={wall})"
    );
}

#[test]
fn nozzle_cutcell_mesh_is_valid_and_tagged() {
    let geo = nozzle_geo();
    let domain = Vector2::new(LENGTH, HEIGHT);
    let mut mesh = generate_cut_cell_mesh(&geo, 0.02, 0.05, 1.2, domain);
    mesh.smooth(&geo, 0.3, 100);
    check_unstructured("cutcell", mesh);
}

#[test]
fn nozzle_delaunay_mesh_is_valid_and_tagged() {
    let geo = nozzle_geo();
    let domain = Vector2::new(LENGTH, HEIGHT);
    let mut mesh = generate_delaunay_mesh(&geo, 0.02, 0.05, 1.2, domain);
    mesh.smooth(&geo, 0.3, 50);
    check_unstructured("delaunay", mesh);
}

#[test]
fn nozzle_voronoi_mesh_is_valid_and_tagged() {
    let geo = nozzle_geo();
    let domain = Vector2::new(LENGTH, HEIGHT);
    let mut mesh = generate_voronoi_mesh(&geo, 0.02, 0.05, 1.2, domain);
    mesh.smooth(&geo, 0.3, 50);
    check_unstructured("voronoi", mesh);
}

#[test]
fn nozzle_fitted_structured_mesh_still_valid() {
    // The `Fitted` option is unchanged from the validated structured path.
    let mesh = generate_structured_nozzle_mesh(
        96,
        32,
        LENGTH,
        HEIGHT,
        THROAT_H,
        THROAT_FRAC,
        EXIT_H,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    assert_eq!(mesh.num_cells(), 96 * 32);
    let (inlet, outlet, _wall) = assert_closed_and_tagged(&mesh, "fitted");
    assert_boundary_positions(&mesh, "fitted");
    assert_cells_inside(&mesh, "fitted");
    assert_eq!(inlet, 32, "fitted: one inlet face per row");
    assert_eq!(outlet, 32, "fitted: one outlet face per row");
}
