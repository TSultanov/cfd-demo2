//! GPU-resident cell geometry — the last constructive solver-mesh buffer built
//! on device (Phase C stage 2c). One thread per cell turns the engine's
//! per-cell clip outputs into the two cell-indexed arrays the GPU solver binds:
//!
//!  - `cell_centers` = `seed + centroid_rel` (the engine stores the centroid
//!    SEED-RELATIVE; absolute = seed + rel — same reconstruction `read_cells`
//!    does), interleaved `vec2<f32>` to match `mesh_geometry_f32`'s
//!    `interleave_f32(cell_cx, cell_cy)` byte layout;
//!  - `cell_vols` = `area` (the clip-polygon area = cell volume in 2D).
//!
//! Runs after [`GpuVoronoiEngine::resolve_flagged`], which patches
//! `b_cell_centroid` / `b_cell_area` in f64 for any epsilon-flagged cell, so the
//! inputs are already the final per-cell geometry. Cell-indexed (n cells, no
//! scan): the solver mesh has exactly one cell per seed, and the fixed-seed v1
//! moving mesh keeps the cell count invariant.

use crate::solver::gpu::buffers::create_buffer;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

use super::engine::GpuVoronoiEngine;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CellGeomParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Cell-indexed geometry read back for parity/inspection. In the live loop
/// these live only as device buffers (`b_cell_centers` / `b_cell_vols` in
/// `MeshResources`); the readback here is the test/validation path.
#[derive(Clone, Debug)]
pub struct GpuCellGeometry {
    /// Absolute cell centre per cell (`seed + centroid_rel`).
    pub cell_centers: Vec<[f32; 2]>,
    /// Cell volume (= clip-polygon area) per cell.
    pub cell_vols: Vec<f32>,
}

/// The cell-geometry emit pass: `seed + centroid_rel` → `cell_centers`, `area`
/// → `cell_vols`, entirely on device.
pub struct CellGeometry {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl CellGeometry {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cell_geom:bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),   // seeds
                storage(2, true),   // centroid_rel
                storage(3, true),   // area
                storage(4, false),  // out: cell_centers
                storage(5, false),  // out: cell_vols
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cell_geom:pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cell_geom:shader"),
            source: wgpu::ShaderSource::Wgsl(cell_geom_shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cell_geom"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("cell_geom"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bgl }
    }

    /// Run the pass over the engine's current cell-major outputs and read back
    /// the cell-indexed geometry (produced on device; the readback is the
    /// test/validation path).
    pub fn build(
        &self,
        ctx: &GpuContext,
        cache: &StagingBufferCache,
        engine: &GpuVoronoiEngine,
    ) -> GpuCellGeometry {
        use wgpu::BufferUsages as U;
        let n = engine.n_seeds();
        assert!(n > 0, "cell_geom called before a regen (n_seeds == 0)");

        let params = CellGeomParams { n, _pad0: 0, _pad1: 0, _pad2: 0 };
        let b_params = create_buffer(
            &ctx.device,
            "cell_geom:params",
            std::mem::size_of::<CellGeomParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        ctx.queue.write_buffer(&b_params, 0, bytemuck::bytes_of(&params));

        let nn = n as u64;
        let b_centers =
            create_buffer(&ctx.device, "cell_geom:centers", nn * 8, U::STORAGE | U::COPY_SRC);
        let b_vols =
            create_buffer(&ctx.device, "cell_geom:vols", nn * 4, U::STORAGE | U::COPY_SRC);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cell_geom:bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: engine.b_seeds.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: engine.outputs.b_cell_centroid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: engine.outputs.b_cell_area.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 4, resource: b_centers.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_vols.as_entire_binding() },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cell_geom:enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cell_geom:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y) = dispatch_2d(n.div_ceil(WORKGROUP_SIZE).max(1));
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let prof = ProfilingStats::new();
        let centers_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_centers, nn * 8, "cell_geom:rb_centers",
        ));
        let vols_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_vols, nn * 4, "cell_geom:rb_vols",
        ));
        GpuCellGeometry {
            cell_centers: bytemuck::cast_slice(&centers_b).to_vec(),
            cell_vols: bytemuck::cast_slice(&vols_b).to_vec(),
        }
    }
}

fn cell_geom_shader() -> String {
    format!(
        r#"
struct Params {{ n: u32, pad0: u32, pad1: u32, pad2: u32 }};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read>       seeds: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       centroid_rel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read>       area: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_centers: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> out_vols: array<f32>;

@compute @workgroup_size({wg})
fn cell_geom(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * nwg.x + wid.x) * {wg}u + lid.x;
    if (i >= P.n) {{ return; }}
    // Absolute centre = seed + seed-relative centroid (same reconstruction the
    // CPU read path does); area is the clip-polygon area = cell volume in 2D.
    out_centers[i] = seeds[i] + centroid_rel[i];
    out_vols[i] = area[i];
}}
"#,
        wg = WORKGROUP_SIZE,
    )
}
