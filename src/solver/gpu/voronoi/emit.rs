//! GPU-resident face-major EMIT — the pass the `derive_faces` doc flagged as
//! "not landed". Given the M1 engine's cell-major clip slots + the count→scan
//! per-cell owned-face offsets ([`super::derive::DeriveFaces`]), one thread per
//! cell writes its OWNED faces (canonical `owner = min(i,j)`; every boundary/box
//! face owned by its cell) into the face-major arrays the GPU SOLVER binds
//! (`face_owner` / `face_neighbor` / `face_areas` / `face_normals` /
//! `face_centers` / `face_bc`) — entirely on device, no CPU `assemble_mesh`
//! round-trip. Only the scalar `num_faces` is read back (to size the arrays);
//! the geometry never leaves the GPU.
//!
//! This is the SOLVER mesh, which needs no vertex arrays (the union-find vertex
//! merge in `assemble_mesh` produces only the RENDERING geometry). The cell→face
//! CSR (adjacency + matrix indices) is the next stage.

use crate::solver::gpu::buffers::create_buffer;
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

use super::derive::DeriveFaces;
use super::engine::GpuVoronoiEngine;
use super::K_FACE_MAX;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EmitParams {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Face-major solver geometry read back for parity/inspection. Length =
/// `num_faces` (the scan total). In the live loop these live only as device
/// buffers; the readback here is the test/validation path.
#[derive(Clone, Debug)]
pub struct GpuFaceGeometry {
    pub num_faces: u32,
    /// Owner cell id per face (`owner = min(i,j)`).
    pub face_owner: Vec<u32>,
    /// Neighbour cell id, or `-1` for a boundary/box face.
    pub face_neighbor: Vec<i32>,
    /// Face length (= area in 2D).
    pub face_area: Vec<f32>,
    /// Unit normal (owner-relative orientation, as the clip kernel emits it).
    pub face_normal: Vec<[f32; 2]>,
    /// Absolute face centre (`seed[owner] + face_mid`).
    pub face_center: Vec<[f32; 2]>,
    /// Boundary tag (bbox side id, `BC_SEG_FLAG|seg`, or `BC_NONE` interior).
    pub face_bc: Vec<u32>,
}

/// The face-major emit pass. Composes [`DeriveFaces`] (count → scan) and the
/// emit kernel; the whole chain runs in one encoder with no intermediate mesh
/// readback.
pub struct EmitFaces {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    derive: DeriveFaces,
}

impl EmitFaces {
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
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("emit:bgl"),
            entries: &[
                uniform(0),        // params
                storage(1, true),  // seeds
                storage(2, true),  // nbr_ids
                storage(3, true),  // face_bc
                storage(4, true),  // face_geom  [nx,ny,len,0]
                storage(5, true),  // face_mid   (seed-relative)
                storage(6, true),  // cell_nfaces
                storage(7, true),  // status
                storage(8, true),  // offsets    (scan output)
                storage(9, false),  // out: face_owner
                storage(10, false), // out: face_neighbor
                storage(11, false), // out: face_area
                storage(12, false), // out: face_normal
                storage(13, false), // out: face_center
                storage(14, false), // out: face_bc
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("emit:pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("emit:shader"),
            source: wgpu::ShaderSource::Wgsl(emit_shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("emit:faces"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("emit_faces"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bgl, derive: DeriveFaces::new(device) }
    }

    /// Run count → scan → emit over the engine's current cell-major outputs and
    /// read back the face-major solver geometry (the geometry itself is produced
    /// entirely on device; only the scalar `num_faces` and the parity readback
    /// touch the CPU).
    pub fn emit(
        &self,
        ctx: &GpuContext,
        cache: &StagingBufferCache,
        engine: &GpuVoronoiEngine,
    ) -> GpuFaceGeometry {
        use wgpu::BufferUsages as U;
        let n = engine.n_seeds();
        assert!(n > 0, "emit called before a regen (n_seeds == 0)");

        // count → scan → per-cell offsets + num_faces, kept on device.
        let scan = self.derive.encode_offsets(ctx, cache, engine);
        let num_faces = scan.total;
        assert!(num_faces > 0, "emit: zero faces");

        let params = EmitParams { n, k: K_FACE_MAX as u32, _pad0: 0, _pad1: 0 };
        let b_params = create_buffer(
            &ctx.device,
            "emit:params",
            std::mem::size_of::<EmitParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        ctx.queue.write_buffer(&b_params, 0, bytemuck::bytes_of(&params));

        let nf = num_faces as u64;
        let mk = |label: &str, bytes: u64| {
            create_buffer(&ctx.device, label, bytes, U::STORAGE | U::COPY_SRC)
        };
        let b_owner = mk("emit:owner", nf * 4);
        let b_neighbor = mk("emit:neighbor", nf * 4);
        let b_area = mk("emit:area", nf * 4);
        let b_normal = mk("emit:normal", nf * 8);
        let b_center = mk("emit:center", nf * 8);
        let b_bc = mk("emit:bc", nf * 4);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emit:bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: engine.b_seeds.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: engine.outputs.b_nbr_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: engine.outputs.b_face_bc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: engine.outputs.b_face_geom.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: engine.outputs.b_face_mid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: engine.outputs.b_cell_nfaces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: engine.outputs.b_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 8, resource: scan.b_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: b_owner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: b_neighbor.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: b_area.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: b_normal.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: b_center.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: b_bc.as_entire_binding() },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("emit:enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emit:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y) = dispatch_2d(n.div_ceil(WORKGROUP_SIZE).max(1));
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let prof = ProfilingStats::new();
        let owner_b =
            pollster::block_on(read_buffer_cached(ctx, cache, &prof, &b_owner, nf * 4, "emit:rb_owner"));
        let neigh_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_neighbor, nf * 4, "emit:rb_neighbor",
        ));
        let area_b =
            pollster::block_on(read_buffer_cached(ctx, cache, &prof, &b_area, nf * 4, "emit:rb_area"));
        let normal_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_normal, nf * 8, "emit:rb_normal",
        ));
        let center_b = pollster::block_on(read_buffer_cached(
            ctx, cache, &prof, &b_center, nf * 8, "emit:rb_center",
        ));
        let bc_b =
            pollster::block_on(read_buffer_cached(ctx, cache, &prof, &b_bc, nf * 4, "emit:rb_bc"));
        GpuFaceGeometry {
            num_faces,
            face_owner: bytemuck::cast_slice(&owner_b).to_vec(),
            face_neighbor: bytemuck::cast_slice(&neigh_b).to_vec(),
            face_area: bytemuck::cast_slice(&area_b).to_vec(),
            face_normal: bytemuck::cast_slice(&normal_b).to_vec(),
            face_center: bytemuck::cast_slice(&center_b).to_vec(),
            face_bc: bytemuck::cast_slice(&bc_b).to_vec(),
        }
    }
}

fn emit_shader() -> String {
    format!(
        r#"
struct Params {{ n: u32, k: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read>       seeds: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       nbr_ids: array<u32>;
@group(0) @binding(3) var<storage, read>       face_bc: array<u32>;
@group(0) @binding(4) var<storage, read>       face_geom: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       face_mid: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read>       cell_nfaces: array<u32>;
@group(0) @binding(7) var<storage, read>       cell_status: array<u32>;
@group(0) @binding(8) var<storage, read>       offsets: array<u32>;
@group(0) @binding(9)  var<storage, read_write> out_owner: array<u32>;
@group(0) @binding(10) var<storage, read_write> out_neighbor: array<i32>;
@group(0) @binding(11) var<storage, read_write> out_area: array<f32>;
@group(0) @binding(12) var<storage, read_write> out_normal: array<vec2<f32>>;
@group(0) @binding(13) var<storage, read_write> out_center: array<vec2<f32>>;
@group(0) @binding(14) var<storage, read_write> out_bc: array<u32>;

const NBR_NONE: u32 = 0xffffffffu;
const SUCCESS: u32  = 0u;

@compute @workgroup_size({wg})
fn emit_faces(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * nwg.x + wid.x) * {wg}u + lid.x;
    if (i >= P.n) {{ return; }}
    if (cell_status[i] != SUCCESS) {{ return; }}
    let nf = cell_nfaces[i];
    let seed = seeds[i];
    var rank: u32 = offsets[i];
    for (var e: u32 = 0u; e < nf; e = e + 1u) {{
        let slot = i * P.k + e;
        let nbr = nbr_ids[slot];
        let owned = (nbr == NBR_NONE) || (i < nbr);
        if (!owned) {{ continue; }}
        let g = rank;
        rank = rank + 1u;
        out_owner[g] = i;
        if (nbr == NBR_NONE) {{ out_neighbor[g] = -1; }}
        else {{ out_neighbor[g] = i32(nbr); }}
        let geom = face_geom[slot];        // [nx, ny, len, 0]
        out_area[g] = geom.z;
        out_normal[g] = vec2<f32>(geom.x, geom.y);
        out_center[g] = seed + face_mid[slot];
        out_bc[g] = face_bc[slot];
    }}
}}
"#,
        wg = WORKGROUP_SIZE,
    )
}
