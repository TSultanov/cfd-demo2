//! Bridge: assemble a solver-ready [`crate::solver::mesh::Mesh`] from the
//! GPU-built device outputs (Phase C stage D4), with NO CPU `assemble_mesh`.
//!
//! The device passes ([`super::EmitFaces`] → face topology + geometry,
//! [`super::GpuCsr`] → cell↔face connectivity, [`super::SweptFluxGeometry`] →
//! per-cell canonical volumes) all address faces by the SAME global id (the
//! `derive` owned-offset scan), so their readback arrays are mutually
//! consistent. This assembles them into the classic `Mesh` the solver's refresh
//! stack consumes — WITHOUT the union-find vertex arrays, because
//! [`crate::solver::mesh::refresh::mesh_geometry_f32`],
//! [`crate::solver::mesh::csr::build_sorted_scalar_csr`], and `MeshTopology`
//! read only the STORED geometry + connectivity (never `vx`/`vy`/`face_v1`/
//! `cell_vertices`). The moving loop feeds the device swept fluxes separately, so
//! the vertex-based CPU swept path is never invoked either.
//!
//! `cell_vol` uses the CANONICAL polygon area (`GpuSweptAreas::canon_area_new`),
//! NOT the engine's clipped `b_cell_area`: the swept quads telescope to the
//! canonical area, so cell_vols + fluxes stay self-consistent and the GCL closes
//! (see `swept_gpu`). Boundary faces are left untagged (`None`) HERE — the
//! caller must tag them before the mesh reaches the solver ([`super::regen`]
//! maps the device `face_bc` tags through `tag_boundary_type`, byte-identical
//! to the CPU assembler; an untagged open face would silently scatter to bc
//! row 0 with a zeroed kind).

use crate::solver::mesh::Mesh;

use super::emit::GpuFaceGeometry;
use super::cell_geom::GpuCellGeometry;
use super::csr_gpu::GpuCsrArrays;
use super::swept_gpu::GpuSweptAreas;

/// Assemble a vertex-less solver [`Mesh`] from the GPU-built face geometry
/// (`faces`), cell↔face connectivity (`csr`), cell centres (`cells`), and
/// per-cell canonical volumes (`swept.canon_area_new`).
///
/// All four must come from the SAME regen (same face global-id ordering). The
/// returned mesh has empty vertex arrays (`vx`/`vy`/`face_v1`/`face_v2`/
/// `cell_vertices`/`cell_vertex_offsets`) — the refresh path never reads them;
/// `num_faces()`/`num_cells()` are driven by the geometry arrays. Boundary faces
/// are `None`; the caller tags them.
pub fn assemble_solver_mesh(
    faces: &GpuFaceGeometry,
    csr: &GpuCsrArrays,
    cells: &GpuCellGeometry,
    swept: &GpuSweptAreas,
) -> Result<Mesh, String> {
    let num_faces = faces.num_faces as usize;
    let num_cells = cells.cell_vols.len();
    if csr.cell_face_offsets.len() != num_cells + 1 {
        return Err(format!(
            "assemble_solver_mesh: cell_face_offsets len {} != num_cells+1 {}",
            csr.cell_face_offsets.len(),
            num_cells + 1
        ));
    }
    if swept.canon_area_new.len() != num_cells {
        return Err(format!(
            "assemble_solver_mesh: canon_area_new len {} != num_cells {}",
            swept.canon_area_new.len(),
            num_cells
        ));
    }
    if faces.face_owner.len() != num_faces
        || faces.face_neighbor.len() != num_faces
        || faces.face_area.len() != num_faces
    {
        return Err("assemble_solver_mesh: face arrays length mismatch".into());
    }

    let face_owner: Vec<usize> = faces.face_owner.iter().map(|&o| o as usize).collect();
    let face_neighbor: Vec<Option<usize>> = faces
        .face_neighbor
        .iter()
        .map(|&n| if n < 0 { None } else { Some(n as usize) })
        .collect();
    let face_nx: Vec<f64> = faces.face_normal.iter().map(|n| n[0] as f64).collect();
    let face_ny: Vec<f64> = faces.face_normal.iter().map(|n| n[1] as f64).collect();
    let face_area: Vec<f64> = faces.face_area.iter().map(|&a| a as f64).collect();
    let face_cx: Vec<f64> = faces.face_center.iter().map(|c| c[0] as f64).collect();
    let face_cy: Vec<f64> = faces.face_center.iter().map(|c| c[1] as f64).collect();

    let cell_cx: Vec<f64> = cells.cell_centers.iter().map(|c| c[0] as f64).collect();
    let cell_cy: Vec<f64> = cells.cell_centers.iter().map(|c| c[1] as f64).collect();
    // Canonical polygon area — self-consistent with the swept fluxes (NOT the
    // engine's clipped area) so the GCL closes.
    let cell_vol: Vec<f64> = swept.canon_area_new.iter().map(|&v| v as f64).collect();

    let cell_faces: Vec<usize> = csr.cell_faces.iter().map(|&f| f as usize).collect();
    let cell_face_offsets: Vec<usize> =
        csr.cell_face_offsets.iter().map(|&o| o as usize).collect();

    Ok(Mesh {
        // Vertex arrays intentionally empty (rendering/CPU-swept only).
        vx: Vec::new(),
        vy: Vec::new(),
        v_fixed: Vec::new(),
        face_v1: Vec::new(),
        face_v2: Vec::new(),
        face_owner,
        face_neighbor,
        face_boundary: vec![None; num_faces],
        face_nx,
        face_ny,
        face_area,
        face_cx,
        face_cy,
        face_wrap_shift: Vec::new(),
        cell_cx,
        cell_cy,
        cell_vol,
        cell_faces,
        cell_face_offsets,
        cell_vertices: Vec::new(),
        cell_vertex_offsets: Vec::new(),
    })
}
