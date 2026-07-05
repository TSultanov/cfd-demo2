#![cfg(all(feature = "ui", feature = "dev-tests"))]

//! Regression test for the GUI Voronoi crash: a polygonal (Voronoi) cell needs
//! `3*(n-2)` fan-triangulation vertices and `2*n` wireframe vertices (~12+ for
//! the typical hexagon), which overran the old `num_cells * 10` vertex-buffer
//! heuristic (fatal wgpu validation error). Buffers must be sized from the
//! actual triangulated data; this test performs the same upload the GUI worker
//! does.

use cfd2::solver::mesh::{generate_voronoi_mesh, ChannelWithObstacle, Mesh};
use cfd2::ui::cfd_renderer;
use cfd2::ui::cfd_renderer::CfdVertex;
use nalgebra::{Point2, Vector2};

fn cell_polygons(mesh: &Mesh) -> Vec<Vec<[f64; 2]>> {
    (0..mesh.num_cells())
        .map(|i| {
            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            (start..end)
                .map(|k| {
                    let v = mesh.cell_vertices[k];
                    [mesh.vx[v], mesh.vy[v]]
                })
                .collect()
        })
        .collect()
}

#[test]
fn voronoi_mesh_upload_fits_render_buffers() {
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    // GUI default sizing — the original crash configuration.
    let mut mesh = generate_voronoi_mesh(&geo, 0.025, 0.025, 1.2, Vector2::new(3.0, 1.0));
    mesh.smooth(&geo, 0.3, 50);
    let cells = cell_polygons(&mesh);

    let vertices = cfd_renderer::build_mesh_vertices(&cells);
    let line_vertices = cfd_renderer::build_line_vertices(&cells);

    // Guard: the num_cells*10 heuristic is insufficient for polygonal meshes.
    assert!(
        vertices.len() > mesh.num_cells() * 10 || line_vertices.len() > mesh.num_cells() * 10,
        "expected a Voronoi mesh to exceed the old num_cells*10 vertex heuristic \
         (tri {} / line {} vs {})",
        vertices.len(),
        line_vertices.len(),
        mesh.num_cells() * 10
    );

    // Perform the actual GPU upload the GUI worker does at init.
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor::default()),
    )
    .expect("device");

    let max_vertices = vertices.len().max(line_vertices.len()).max(1);
    // Static upload path: allocate exactly, as the GUI static path does.
    let mut renderer = cfd_renderer::CfdRenderResources::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
        max_vertices,
        cfd_renderer::NO_HEADROOM,
    );
    renderer.update_mesh(&device, &queue, &vertices, &line_vertices);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

/// Build synthetic cell polygons: `n_cells` polygons, each with `verts_per_cell`
/// vertices arranged on a small circle. This lets us drive `build_mesh_vertices`
/// with a KNOWN, controllable per-cell vertex count so we can force the
/// changing-topology growth path deterministically without a full mesh solve.
fn synthetic_cells(n_cells: usize, verts_per_cell: usize) -> Vec<Vec<[f64; 2]>> {
    (0..n_cells)
        .map(|c| {
            let cx = c as f64;
            (0..verts_per_cell)
                .map(|k| {
                    let theta = std::f64::consts::TAU * (k as f64) / (verts_per_cell as f64);
                    [cx + theta.cos(), theta.sin()]
                })
                .collect()
        })
        .collect()
}

/// The moving-mesh (ALE) crash-regression gate.
///
/// A moving Voronoi mesh re-tessellates every step and per-cell vertex counts
/// drift as the topology changes — so the renderer's fixed-size vertex buffers
/// (allocated once from the INITIAL mesh) would overrun on any step whose
/// tessellation is larger than the first. This drives `update_mesh` with a
/// sequence of meshes A -> B -> C of strictly INCREASING per-cell vertex counts
/// (well past the initial capacity), asserting: no panic/overflow, the buffers
/// grow to fit, `num_vertices` tracks the data exactly (no truncation), and the
/// cell count stays invariant (so the field bind path is unaffected).
#[test]
fn moving_mesh_growing_topology_never_overflows() {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device");

    // Invariant cell count across the whole sequence (only topology grows).
    let n_cells = 64;

    // Mesh A: triangles (3 verts/cell). Size the renderer from A ONLY — this is
    // exactly what the GUI does at init, so A defines the initial capacity.
    let cells_a = synthetic_cells(n_cells, 3);
    let va = cfd_renderer::build_mesh_vertices(&cells_a);
    let la = cfd_renderer::build_line_vertices(&cells_a);
    let max_vertices = va.len().max(la.len()).max(1);

    // Moving path — headroom over the initial count, exactly as the GUI ALE
    // path allocates. Mesh B is chosen to exceed even this headroom'd capacity.
    let mut renderer = cfd_renderer::CfdRenderResources::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
        max_vertices,
        cfd_renderer::VERTEX_HEADROOM,
    );
    renderer.update_mesh(&device, &queue, &va, &la);
    assert_eq!(renderer.num_vertices as usize, va.len());
    assert_eq!(renderer.num_line_vertices as usize, la.len());
    let cap_a = renderer.capacity_vertices;

    // Mesh B: octagons (8 verts/cell) — fan tri = 3*(8-2)=18 verts/cell,
    // wireframe = 16 verts/cell. Far more than A's 3 tri / 6 line per cell, and
    // chosen to strictly exceed A's headroom'd initial capacity, forcing a grow.
    let cells_b = synthetic_cells(n_cells, 8);
    let vb = cfd_renderer::build_mesh_vertices(&cells_b);
    let lb = cfd_renderer::build_line_vertices(&cells_b);
    assert!(
        vb.len() > cap_a,
        "test precondition: mesh B tri verts ({}) must exceed A's initial capacity ({}) \
         to exercise the grow path",
        vb.len(),
        cap_a
    );
    assert!(!renderer.can_fit(vb.len(), lb.len()));
    renderer.update_mesh(&device, &queue, &vb, &lb);
    assert_eq!(
        renderer.num_vertices as usize,
        vb.len(),
        "num_vertices must track the larger mesh exactly (no truncation)"
    );
    assert_eq!(renderer.num_line_vertices as usize, lb.len());
    assert!(
        renderer.capacity_vertices >= vb.len(),
        "vertex buffer must have grown to fit mesh B"
    );
    assert!(renderer.can_fit(vb.len(), lb.len()));
    assert!(
        renderer.capacity_vertices > cap_a,
        "capacity must have strictly grown from A ({}) for B ({})",
        cap_a,
        renderer.capacity_vertices
    );

    // Mesh C: 16-gons (even bigger) — grow again from B.
    let cells_c = synthetic_cells(n_cells, 16);
    let vc = cfd_renderer::build_mesh_vertices(&cells_c);
    let lc = cfd_renderer::build_line_vertices(&cells_c);
    let cap_b = renderer.capacity_vertices;
    assert!(vc.len() > cap_b);
    renderer.update_mesh(&device, &queue, &vc, &lc);
    assert_eq!(renderer.num_vertices as usize, vc.len());
    assert_eq!(renderer.num_line_vertices as usize, lc.len());
    assert!(renderer.capacity_vertices >= vc.len());

    // Shrinking back to a smaller topology must NOT reallocate (capacity is a
    // high-water mark) and must still report the correct, smaller draw count.
    let cap_c = renderer.capacity_vertices;
    let cells_d = synthetic_cells(n_cells, 4);
    let vd = cfd_renderer::build_mesh_vertices(&cells_d);
    let ld = cfd_renderer::build_line_vertices(&cells_d);
    assert!(renderer.can_fit(vd.len(), ld.len()));
    renderer.update_mesh(&device, &queue, &vd, &ld);
    assert_eq!(renderer.num_vertices as usize, vd.len());
    assert_eq!(
        renderer.capacity_vertices, cap_c,
        "shrinking must not shrink the allocation (no needless realloc)"
    );

    // Force the GPU to actually service every write_buffer we issued; if any had
    // overrun its allocation, validation would have already errored out.
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Sanity on the data itself: fan triangulation of an m-gon yields 3*(m-2)
    // vertices per cell; nothing was silently dropped.
    let _ = std::mem::size_of::<CfdVertex>();
    assert_eq!(vc.len(), n_cells * 3 * (16 - 2));
}
