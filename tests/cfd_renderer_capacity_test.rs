#![cfg(all(feature = "ui", feature = "dev-tests"))]

//! Regression test for the GUI Voronoi crash: the renderer's vertex buffers
//! were sized with a `num_cells * 10` heuristic, but a polygonal (Voronoi)
//! cell needs `3*(n-2)` fan-triangulation vertices and `2*n` wireframe
//! vertices — ~12+ for the typical hexagon — so `Queue::write_buffer`
//! overran the buffer (fatal wgpu validation error) on every Voronoi mesh.
//! The buffers are now sized from the actual triangulated data; this test
//! performs the same upload the GUI worker does.

use cfd2::solver::mesh::{generate_voronoi_mesh, ChannelWithObstacle, Mesh};
use cfd2::ui::cfd_renderer;
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

    // The old heuristic really is insufficient for polygonal meshes — keep
    // this assertion so the sizing can never quietly go back to it.
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
    let mut renderer = cfd_renderer::CfdRenderResources::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
        max_vertices,
    );
    renderer.update_mesh(&queue, &vertices, &line_vertices);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}
