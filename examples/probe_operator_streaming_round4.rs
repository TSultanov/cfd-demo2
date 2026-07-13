//! Isolated GPU probe for cell-owner operator streaming.
//!
//! This is deliberately not wired into the solver.  It compares a conventional
//! gradient -> physical-face flux -> cell residual -> update pipeline with one
//! workgroup dispatch that loads a two-ring closure, computes Green--Gauss-like
//! gradients and canonical face fluxes privately, and writes the next state.
//! The graph case randomly permutes global cell ids and uses an explicit block
//! closure map, exercising the same gather pattern an unstructured partitioner
//! would emit.

use bytemuck::{Pod, Zeroable};
use cfd2::solver::gpu::context::GpuContext;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

#[cfg(feature = "meshgen")]
use cfd2::solver::mesh::{generate_voronoi_mesh, Mesh, RectangularChannel};
#[cfg(feature = "meshgen")]
use nalgebra::Vector2;

const TILE: usize = 8;
const HALO: usize = 2;
const EXT: usize = TILE + 2 * HALO;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Config {
    nx: u32,
    ny: u32,
    blocks_x: u32,
    n: u32,
    neighbor_off: u32,
    face_off: u32,
    incidence_off: u32,
    block_map_off: u32,
    dt: f32,
    ux: f32,
    uy: f32,
    nu: f32,
}

#[derive(Clone, Copy)]
struct Options {
    nx: usize,
    ny: usize,
    stages: usize,
    warmup: usize,
}

fn options() -> Options {
    let mut out = Options {
        nx: 1024,
        ny: 1024,
        stages: 80,
        warmup: 8,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("missing value after {arg}"))
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid value after {arg}"))
        };
        match arg.as_str() {
            "--nx" => out.nx = value(),
            "--ny" => out.ny = value(),
            "--stages" => out.stages = value(),
            "--warmup" => out.warmup = value(),
            "-h" | "--help" => {
                println!(
                    "probe_operator_streaming_round4 [--nx N] [--ny N] \
                     [--stages N] [--warmup N]"
                );
                std::process::exit(0);
            }
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(out.nx % TILE == 0 && out.ny % TILE == 0);
    assert!(out.warmup % 2 == 0, "warmup must preserve ping-pong parity");
    assert!(out.stages > 0);
    out
}

fn wait(ctx: &GpuContext, submission: wgpu::SubmissionIndex) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("wait for GPU");
}

fn upload<T: Pod>(ctx: &GpuContext, label: &str, data: &[T]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
}

fn empty(ctx: &GpuContext, label: &str, bytes: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn read_vec4(ctx: &GpuContext, src: &wgpu::Buffer, n: usize) -> Vec<[f32; 4]> {
    let bytes = (n * 16) as u64;
    let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("operator-streaming readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("operator-streaming readback encoder"),
        });
    encoder.copy_buffer_to_buffer(src, 0, &read, 0, bytes);
    let submission = ctx.queue.submit(Some(encoder.finish()));
    let slice = read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    wait(ctx, submission);
    rx.recv().expect("map callback").expect("map readback");
    let mapped = slice.get_mapped_range();
    let out = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
    drop(mapped);
    read.unmap();
    out
}

fn initial_state(nx: usize, ny: usize) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(nx * ny);
    for y in 0..ny {
        for x in 0..nx {
            let xf = (x as f32 + 0.5) / nx as f32;
            let yf = (y as f32 + 0.5) / ny as f32;
            out.push([
                (std::f32::consts::TAU * xf).sin() * (std::f32::consts::TAU * yf).cos(),
                (4.0 * std::f32::consts::PI * xf + 0.3).cos(),
                (2.0 * std::f32::consts::PI * yf - 0.7).sin(),
                0.25 + 0.1 * (std::f32::consts::TAU * (xf + yf)).cos(),
            ]);
        }
    }
    out
}

fn permutation(n: usize) -> Vec<usize> {
    let mut p: Vec<_> = (0..n).collect();
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        p.swap(i, state as usize % (i + 1));
    }
    // p[logical cell] = deliberately nonlocal storage id.
    p
}

fn graph_topology(nx: usize, ny: usize, perm: &[usize]) -> (Vec<u32>, Config) {
    let n = nx * ny;
    let logical = |x: usize, y: usize| (y % ny) * nx + (x % nx);
    let mut topo = vec![0u32; 4 * n];
    let neighbor_off = 0usize;
    for y in 0..ny {
        for x in 0..nx {
            let l = logical(x, y);
            let g = perm[l];
            let ids = [
                perm[logical((x + nx - 1) % nx, y)],
                perm[logical(x + 1, y)],
                perm[logical(x, (y + ny - 1) % ny)],
                perm[logical(x, y + 1)],
            ];
            for d in 0..4 {
                topo[neighbor_off + 4 * g + d] = ids[d] as u32;
            }
        }
    }

    let face_off = topo.len();
    topo.resize(face_off + 4 * n, 0);
    for y in 0..ny {
        for x in 0..nx {
            let l = logical(x, y);
            let east = logical(x + 1, y);
            let north = logical(x, y + 1);
            topo[face_off + 2 * l] = perm[l] as u32;
            topo[face_off + 2 * l + 1] = perm[east] as u32;
            topo[face_off + 2 * (n + l)] = perm[l] as u32;
            topo[face_off + 2 * (n + l) + 1] = perm[north] as u32;
        }
    }

    let incidence_off = topo.len();
    topo.resize(incidence_off + 4 * n, 0);
    for y in 0..ny {
        for x in 0..nx {
            let l = logical(x, y);
            let g = perm[l];
            let west_face = logical((x + nx - 1) % nx, y);
            let east_face = l;
            let south_face = n + logical(x, (y + ny - 1) % ny);
            let north_face = n + l;
            // Low bit is 1 when this cell is the canonical face neighbor.
            let inc = [
                ((west_face as u32) << 1) | 1,
                (east_face as u32) << 1,
                ((south_face as u32) << 1) | 1,
                (north_face as u32) << 1,
            ];
            topo[incidence_off + 4 * g..incidence_off + 4 * g + 4].copy_from_slice(&inc);
        }
    }

    let block_map_off = topo.len();
    let blocks_x = nx / TILE;
    let blocks_y = ny / TILE;
    topo.reserve(blocks_x * blocks_y * EXT * EXT);
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            for ty in 0..EXT {
                for tx in 0..EXT {
                    let x = (bx * TILE + tx + nx - HALO) % nx;
                    let y = (by * TILE + ty + ny - HALO) % ny;
                    topo.push(perm[logical(x, y)] as u32);
                }
            }
        }
    }

    let config = Config {
        nx: nx as u32,
        ny: ny as u32,
        blocks_x: blocks_x as u32,
        n: n as u32,
        neighbor_off: neighbor_off as u32,
        face_off: face_off as u32,
        incidence_off: incidence_off as u32,
        block_map_off: block_map_off as u32,
        dt: 0.01,
        ux: 0.37,
        uy: -0.23,
        nu: 0.013,
    };
    (topo, config)
}

const SHADER: &str = r#"
struct Config {
    nx: u32,
    ny: u32,
    blocks_x: u32,
    n: u32,
    neighbor_off: u32,
    face_off: u32,
    incidence_off: u32,
    block_map_off: u32,
    dt: f32,
    ux: f32,
    uy: f32,
    nu: f32,
}

struct Grad { x: vec4<f32>, y: vec4<f32> }

@group(0) @binding(0) var<storage, read> q_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> q_out: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> grads: array<Grad>;
@group(0) @binding(3) var<storage, read_write> face_flux: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> rhs: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> topo: array<u32>;
@group(0) @binding(6) var<uniform> cfg: Config;

fn cell(x: u32, y: u32) -> u32 {
    return (y % cfg.ny) * cfg.nx + (x % cfg.nx);
}

fn numerical_flux(qp: vec4<f32>, qn: vec4<f32>, gp: Grad, gn: Grad, axis: u32) -> vec4<f32> {
    let un = select(cfg.ux, cfg.uy, axis == 1u);
    let dp = select(gp.x, gp.y, axis == 1u);
    let dn = select(gn.x, gn.y, axis == 1u);
    let ql = qp + 0.5 * dp;
    let qr = qn - 0.5 * dn;
    let adv = un * select(qr, ql, un >= 0.0);
    return adv - cfg.nu * (qn - qp);
}

@compute @workgroup_size(256)
fn structured_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let x = i % cfg.nx;
    let y = i / cfg.nx;
    let w = cell(x + cfg.nx - 1u, y);
    let e = cell(x + 1u, y);
    let s = cell(x, y + cfg.ny - 1u);
    let n = cell(x, y + 1u);
    grads[i].x = 0.5 * (q_in[e] - q_in[w]);
    grads[i].y = 0.5 * (q_in[n] - q_in[s]);
}

@compute @workgroup_size(256)
fn graph_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let b = cfg.neighbor_off + 4u * i;
    let w = topo[b];
    let e = topo[b + 1u];
    let s = topo[b + 2u];
    let n = topo[b + 3u];
    grads[i].x = 0.5 * (q_in[e] - q_in[w]);
    grads[i].y = 0.5 * (q_in[n] - q_in[s]);
}

@compute @workgroup_size(256)
fn structured_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= 2u * cfg.n) { return; }
    let axis = select(0u, 1u, f >= cfg.n);
    let p = select(f, f - cfg.n, f >= cfg.n);
    let x = p % cfg.nx;
    let y = p / cfg.nx;
    let n = select(cell(x + 1u, y), cell(x, y + 1u), axis == 1u);
    face_flux[f] = numerical_flux(q_in[p], q_in[n], grads[p], grads[n], axis);
}

@compute @workgroup_size(256)
fn graph_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= 2u * cfg.n) { return; }
    let p = topo[cfg.face_off + 2u * f];
    let n = topo[cfg.face_off + 2u * f + 1u];
    let axis = select(0u, 1u, f >= cfg.n);
    face_flux[f] = numerical_flux(q_in[p], q_in[n], grads[p], grads[n], axis);
}

@compute @workgroup_size(256)
fn structured_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let x = i % cfg.nx;
    let y = i / cfg.nx;
    let fw = cell(x + cfg.nx - 1u, y);
    let fe = i;
    let fs = cfg.n + cell(x, y + cfg.ny - 1u);
    let fnorth = cfg.n + i;
    rhs[i] = face_flux[fw] - face_flux[fe] + face_flux[fs] - face_flux[fnorth];
}

@compute @workgroup_size(256)
fn graph_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    var r = vec4<f32>(0.0);
    let b = cfg.incidence_off + 4u * i;
    for (var d = 0u; d < 4u; d++) {
        let encoded = topo[b + d];
        let f = encoded >> 1u;
        let sign = select(1.0, -1.0, (encoded & 1u) != 0u);
        r -= sign * face_flux[f];
    }
    rhs[i] = r;
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    q_out[i] = q_in[i] + cfg.dt * rhs[i];
}

var<workgroup> tile_q: array<vec4<f32>, 144>;
// 72 canonical x-faces (8 rows x 9 faces) followed by 72 canonical
// y-faces (9 rows x 8 faces). Faces crossing a block cut are evaluated in
// both endpoint blocks; every block-interior face is evaluated exactly once.
var<workgroup> tile_flux: array<vec4<f32>, 144>;

fn tile_gradient(p: u32) -> Grad {
    var g: Grad;
    g.x = 0.5 * (tile_q[p + 1u] - tile_q[p - 1u]);
    g.y = 0.5 * (tile_q[p + 12u] - tile_q[p - 12u]);
    return g;
}

fn fill_tile_flux(tid: u32) {
    for (var k = tid; k < 144u; k += 64u) {
        if (k < 72u) {
            let row = k / 9u;
            let col = k % 9u;
            let p = (row + 2u) * 12u + col + 1u;
            tile_flux[k] = numerical_flux(
                tile_q[p], tile_q[p + 1u], tile_gradient(p), tile_gradient(p + 1u), 0u
            );
        } else {
            let f = k - 72u;
            let row = f / 8u;
            let col = f % 8u;
            let p = (row + 1u) * 12u + col + 2u;
            tile_flux[k] = numerical_flux(
                tile_q[p], tile_q[p + 12u], tile_gradient(p), tile_gradient(p + 12u), 1u
            );
        }
    }
}

fn tile_rhs(lx: u32, ly: u32) -> vec4<f32> {
    let fw = tile_flux[ly * 9u + lx];
    let fe = tile_flux[ly * 9u + lx + 1u];
    let fs = tile_flux[72u + ly * 8u + lx];
    let fnorth = tile_flux[72u + (ly + 1u) * 8u + lx];
    return fw - fe + fs - fnorth;
}

@compute @workgroup_size(8, 8, 1)
fn stream_structured(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    for (var k = tid; k < 144u; k += 64u) {
        let tx = k % 12u;
        let ty = k / 12u;
        let x = (wid.x * 8u + tx + cfg.nx - 2u) % cfg.nx;
        let y = (wid.y * 8u + ty + cfg.ny - 2u) % cfg.ny;
        tile_q[k] = q_in[cell(x, y)];
    }
    workgroupBarrier();
    fill_tile_flux(tid);
    workgroupBarrier();
    let p = (lid.y + 2u) * 12u + lid.x + 2u;
    let x = wid.x * 8u + lid.x;
    let y = wid.y * 8u + lid.y;
    let i = cell(x, y);
    q_out[i] = tile_q[p] + cfg.dt * tile_rhs(lid.x, lid.y);
}

@compute @workgroup_size(8, 8, 1)
fn stream_graph(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let block = wid.y * cfg.blocks_x + wid.x;
    let map = cfg.block_map_off + block * 144u;
    for (var k = tid; k < 144u; k += 64u) {
        tile_q[k] = q_in[topo[map + k]];
    }
    workgroupBarrier();
    fill_tile_flux(tid);
    workgroupBarrier();
    let p = (lid.y + 2u) * 12u + lid.x + 2u;
    let i = topo[map + p];
    q_out[i] = tile_q[p] + cfg.dt * tile_rhs(lid.x, lid.y);
}
"#;

struct Pipelines {
    structured_gradient: wgpu::ComputePipeline,
    graph_gradient: wgpu::ComputePipeline,
    structured_faces: wgpu::ComputePipeline,
    graph_faces: wgpu::ComputePipeline,
    structured_residual: wgpu::ComputePipeline,
    graph_residual: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    stream_structured: wgpu::ComputePipeline,
    stream_graph: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
}

fn pipelines(ctx: &GpuContext) -> Pipelines {
    let entries = [
        (0, true),
        (1, false),
        (2, false),
        (3, false),
        (4, false),
        (5, true),
    ]
    .map(|(binding, read_only)| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    let mut layout_entries = entries.to_vec();
    layout_entries.push(wgpu::BindGroupLayoutEntry {
        binding: 6,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    let bind_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("operator-streaming bind layout"),
            entries: &layout_entries,
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("operator-streaming pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("operator-streaming probe shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
    let make = |entry: &str| {
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    Pipelines {
        structured_gradient: make("structured_gradient"),
        graph_gradient: make("graph_gradient"),
        structured_faces: make("structured_faces"),
        graph_faces: make("graph_faces"),
        structured_residual: make("structured_residual"),
        graph_residual: make("graph_residual"),
        update: make("update"),
        stream_structured: make("stream_structured"),
        stream_graph: make("stream_graph"),
        bind_layout,
    }
}

struct Case {
    q0: wgpu::Buffer,
    q1: wgpu::Buffer,
    grad: wgpu::Buffer,
    flux: wgpu::Buffer,
    rhs: wgpu::Buffer,
    topo: wgpu::Buffer,
    config: wgpu::Buffer,
    bind_01: wgpu::BindGroup,
    bind_10: wgpu::BindGroup,
    initial: Vec<[f32; 4]>,
}

impl Case {
    fn new(
        ctx: &GpuContext,
        pipes: &Pipelines,
        initial: Vec<[f32; 4]>,
        topo: &[u32],
        config: Config,
    ) -> Self {
        let n = initial.len();
        let q0 = upload(ctx, "operator q0", &initial);
        let q1 = empty(ctx, "operator q1", (16 * n) as u64);
        let grad = empty(ctx, "operator gradients", (32 * n) as u64);
        let flux = empty(ctx, "operator face flux", (32 * n) as u64);
        let rhs = empty(ctx, "operator rhs", (16 * n) as u64);
        let topo = upload(ctx, "operator topology", topo);
        let config = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("operator config"),
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = |label: &str, input: &wgpu::Buffer, output: &wgpu::Buffer| {
            let buffers = [input, output, &grad, &flux, &rhs, &topo, &config];
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipes.bind_layout,
                entries: &entries,
            })
        };
        let bind_01 = bind("operator bind 0->1", &q0, &q1);
        let bind_10 = bind("operator bind 1->0", &q1, &q0);
        Self {
            q0,
            q1,
            grad,
            flux,
            rhs,
            topo,
            config,
            bind_01,
            bind_10,
            initial,
        }
    }

    fn reset(&self, ctx: &GpuContext) {
        ctx.queue
            .write_buffer(&self.q0, 0, bytemuck::cast_slice(&self.initial));
        // q1 is always completely overwritten before being consumed.
    }
}

fn encode_baseline(
    ctx: &GpuContext,
    pipes: &Pipelines,
    case: &Case,
    n: usize,
    stages: usize,
    graph: bool,
) -> wgpu::CommandBuffer {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("materialized operator encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("materialized operator pass"),
            timestamp_writes: None,
        });
        for stage in 0..stages {
            pass.set_bind_group(
                0,
                if stage & 1 == 0 {
                    &case.bind_01
                } else {
                    &case.bind_10
                },
                &[],
            );
            pass.set_pipeline(if graph {
                &pipes.graph_gradient
            } else {
                &pipes.structured_gradient
            });
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
            pass.set_pipeline(if graph {
                &pipes.graph_faces
            } else {
                &pipes.structured_faces
            });
            pass.dispatch_workgroups((2 * n as u32).div_ceil(256), 1, 1);
            pass.set_pipeline(if graph {
                &pipes.graph_residual
            } else {
                &pipes.structured_residual
            });
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
            pass.set_pipeline(&pipes.update);
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
        }
    }
    encoder.finish()
}

fn encode_streaming(
    ctx: &GpuContext,
    pipes: &Pipelines,
    case: &Case,
    options: Options,
    stages: usize,
    graph: bool,
) -> wgpu::CommandBuffer {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("streaming operator encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("streaming operator pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(if graph {
            &pipes.stream_graph
        } else {
            &pipes.stream_structured
        });
        for stage in 0..stages {
            pass.set_bind_group(
                0,
                if stage & 1 == 0 {
                    &case.bind_01
                } else {
                    &case.bind_10
                },
                &[],
            );
            pass.dispatch_workgroups((options.nx / TILE) as u32, (options.ny / TILE) as u32, 1);
        }
    }
    encoder.finish()
}

fn submit_time(ctx: &GpuContext, command: wgpu::CommandBuffer) -> Duration {
    let start = Instant::now();
    let submission = ctx.queue.submit(Some(command));
    wait(ctx, submission);
    start.elapsed()
}

fn compare(reference: &[[f32; 4]], candidate: &[[f32; 4]]) -> (f32, f64, f64) {
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut sum_delta = 0.0f64;
    for (a, b) in reference.iter().zip(candidate) {
        for c in 0..4 {
            let d = (a[c] - b[c]).abs();
            max_abs = max_abs.max(d);
            sum_sq += (d as f64) * (d as f64);
            sum_delta += b[c] as f64 - a[c] as f64;
        }
    }
    (
        max_abs,
        (sum_sq / (4 * reference.len()) as f64).sqrt(),
        sum_delta,
    )
}

#[cfg(feature = "meshgen")]
fn partition_metrics(mesh: &Mesh, block_size: usize) -> (f64, f64, f64, usize, usize) {
    let n = mesh.num_cells();
    let blocks = n.div_ceil(block_size);
    let mut adjacency = vec![Vec::<usize>::new(); n];
    for f in 0..mesh.num_faces() {
        if let Some(neighbor) = mesh.face_neighbor[f] {
            let owner = mesh.face_owner[f];
            adjacency[owner].push(neighbor);
            adjacency[neighbor].push(owner);
        }
    }
    let mut sum_h1 = 0usize;
    let mut sum_h2 = 0usize;
    let mut max_h1 = 0usize;
    let mut max_h2 = 0usize;
    let mut mark = vec![0u32; n];
    let mut epoch = 0u32;
    for block in 0..blocks {
        epoch += 1;
        let start = block * block_size;
        let end = (start + block_size).min(n);
        let mut ring = Vec::with_capacity(4 * block_size);
        for cell in start..end {
            mark[cell] = epoch;
            ring.push(cell);
        }
        let core_len = ring.len();
        for at in 0..core_len {
            for &neighbor in &adjacency[ring[at]] {
                if mark[neighbor] != epoch {
                    mark[neighbor] = epoch;
                    ring.push(neighbor);
                }
            }
        }
        let h1 = ring.len();
        for at in core_len..h1 {
            for &neighbor in &adjacency[ring[at]] {
                if mark[neighbor] != epoch {
                    mark[neighbor] = epoch;
                    ring.push(neighbor);
                }
            }
        }
        let h2 = ring.len();
        sum_h1 += h1;
        sum_h2 += h2;
        max_h1 = max_h1.max(h1);
        max_h2 = max_h2.max(h2);
    }
    let mut interior = 0usize;
    let mut cut = 0usize;
    for f in 0..mesh.num_faces() {
        if let Some(neighbor) = mesh.face_neighbor[f] {
            interior += 1;
            if mesh.face_owner[f] / block_size != neighbor / block_size {
                cut += 1;
            }
        }
    }
    (
        sum_h1 as f64 / n as f64,
        sum_h2 as f64 / n as f64,
        cut as f64 / interior.max(1) as f64,
        max_h1,
        max_h2,
    )
}

#[cfg(feature = "meshgen")]
fn report_irregular_partition_metrics() {
    if std::env::var_os("CFD2_PROBE_PARTITIONS").is_none() {
        return;
    }
    let geometry = RectangularChannel {
        length: 1.0,
        height: 1.0,
    };
    let original = generate_voronoi_mesh(&geometry, 0.012, 0.018, 1.15, Vector2::new(1.0, 1.0));
    for (name, mesh) in [
        ("generator", original.clone()),
        ("hilbert", {
            let mut mesh = original.clone();
            let order = mesh.hilbert_cell_order();
            mesh.reorder_cells(&order);
            mesh
        }),
        ("random", {
            let mut mesh = original.clone();
            let order = mesh.random_cell_order();
            mesh.reorder_cells(&order);
            mesh
        }),
    ] {
        let (a1, a2, cut, h1_max, h2_max) = partition_metrics(&mesh, 64);
        let degree = mesh.cell_faces.len() as f64 / mesh.num_cells() as f64;
        println!(
            "partition: order={name} cells={} mean_degree={degree:.3} alpha1={a1:.3} alpha2={a2:.3} cut_fraction={cut:.3} max_h1={h1_max} max_h2={h2_max}",
            mesh.num_cells(),
        );
    }
}

fn verify(
    ctx: &GpuContext,
    pipes: &Pipelines,
    case: &Case,
    options: Options,
    graph: bool,
) -> (f32, f64, f64) {
    let n = options.nx * options.ny;
    case.reset(ctx);
    wait(
        ctx,
        ctx.queue
            .submit(Some(encode_baseline(ctx, pipes, case, n, 1, graph))),
    );
    let baseline = read_vec4(ctx, &case.q1, n);
    case.reset(ctx);
    wait(
        ctx,
        ctx.queue
            .submit(Some(encode_streaming(ctx, pipes, case, options, 1, graph))),
    );
    let streaming = read_vec4(ctx, &case.q1, n);
    compare(&baseline, &streaming)
}

fn benchmark(
    ctx: &GpuContext,
    pipes: &Pipelines,
    case: &Case,
    options: Options,
    graph: bool,
) -> (Duration, Duration) {
    let n = options.nx * options.ny;
    case.reset(ctx);
    wait(
        ctx,
        ctx.queue.submit(Some(encode_baseline(
            ctx,
            pipes,
            case,
            n,
            options.warmup,
            graph,
        ))),
    );
    let baseline = submit_time(
        ctx,
        encode_baseline(ctx, pipes, case, n, options.stages, graph),
    );

    case.reset(ctx);
    wait(
        ctx,
        ctx.queue.submit(Some(encode_streaming(
            ctx,
            pipes,
            case,
            options,
            options.warmup,
            graph,
        ))),
    );
    let streaming = submit_time(
        ctx,
        encode_streaming(ctx, pipes, case, options, options.stages, graph),
    );
    (baseline, streaming)
}

fn main() {
    #[cfg(feature = "meshgen")]
    report_irregular_partition_metrics();
    let options = options();
    let n = options.nx * options.ny;
    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("GPU context");
    let pipes = pipelines(&ctx);
    let logical = initial_state(options.nx, options.ny);
    let perm = permutation(n);
    let (graph_topo, config) = graph_topology(options.nx, options.ny, &perm);
    let structured_case = Case::new(&ctx, &pipes, logical.clone(), &[0], config);
    let mut graph_initial = vec![[0.0; 4]; n];
    for l in 0..n {
        graph_initial[perm[l]] = logical[l];
    }
    let graph_case = Case::new(&ctx, &pipes, graph_initial, &graph_topo, config);

    let (s_max, s_rms, s_sum) = verify(&ctx, &pipes, &structured_case, options, false);
    let (g_max, g_rms, g_sum) = verify(&ctx, &pipes, &graph_case, options, true);
    assert!(s_max <= 2.0e-6 && g_max <= 2.0e-6);
    let (s_base, s_stream) = benchmark(&ctx, &pipes, &structured_case, options, false);
    let (g_base, g_stream) = benchmark(&ctx, &pipes, &graph_case, options, true);

    let stage_cells = (n * options.stages) as f64;
    println!(
        "device_probe: grid={}x{} cells={} stages={} tile={} halo={} closure_ratio={:.4}",
        options.nx,
        options.ny,
        n,
        options.stages,
        TILE,
        HALO,
        (EXT * EXT) as f64 / (TILE * TILE) as f64,
    );
    println!(
        "correctness: structured max_abs={s_max:.3e} rms={s_rms:.3e} sum_delta={s_sum:.3e}; graph-permuted max_abs={g_max:.3e} rms={g_rms:.3e} sum_delta={g_sum:.3e}"
    );
    println!(
        "structured: materialized={:.3} ms ({:.1} Mcell-stage/s) streaming={:.3} ms ({:.1} Mcell-stage/s) speedup={:.3}x",
        s_base.as_secs_f64() * 1e3,
        stage_cells / s_base.as_secs_f64() / 1e6,
        s_stream.as_secs_f64() * 1e3,
        stage_cells / s_stream.as_secs_f64() / 1e6,
        s_base.as_secs_f64() / s_stream.as_secs_f64(),
    );
    println!(
        "graph-permuted: materialized={:.3} ms ({:.1} Mcell-stage/s) streaming={:.3} ms ({:.1} Mcell-stage/s) speedup={:.3}x",
        g_base.as_secs_f64() * 1e3,
        stage_cells / g_base.as_secs_f64() / 1e6,
        g_stream.as_secs_f64() * 1e3,
        stage_cells / g_stream.as_secs_f64() / 1e6,
        g_base.as_secs_f64() / g_stream.as_secs_f64(),
    );
    println!(
        "logical_bytes_per_cell_stage: structured_materialized=448 structured_streaming=52 graph_materialized=496 graph_streaming=61 (vec4 state; physical faces stored once)"
    );
    println!(
        "intermediate_buffers_retained_only_by_baseline: grad={} MiB flux={} MiB rhs={} MiB",
        32 * n / (1024 * 1024),
        32 * n / (1024 * 1024),
        16 * n / (1024 * 1024),
    );
    // Keep resources visibly live through the final queue completion; these
    // fields also document the exact materialized working set in the probe.
    std::hint::black_box((
        &structured_case.grad,
        &structured_case.flux,
        &structured_case.rhs,
        &structured_case.topo,
        &structured_case.config,
    ));
}
