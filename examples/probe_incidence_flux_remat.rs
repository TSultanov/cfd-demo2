//! Isolated A/B for incidence-owned flux rematerialization.
//!
//! The materialized schedule evaluates a canonical/directed face flux, writes
//! eight `f32` channels, then gathers the four differential channels in cell
//! order.  The pull schedule evaluates the exact same function and canonical
//! argument order inside that cell loop.  It deliberately does not use
//! workgroup tiling, atomics, face coloring, or a mesh partition.
//!
//! This is a mechanism probe, not a production solver path:
//!
//! ```text
//! cargo run --release --example probe_incidence_flux_remat -- \
//!     --n 1024 --iterations 24 --samples 7 --rounds 0,1,2,4
//! ```

use bytemuck::{Pod, Zeroable};
use cfd2::solver::gpu::context::GpuContext;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cell {
    q: [f32; 4],
    grad_x: [f32; 4],
    grad_y: [f32; 4],
    primitive: [f32; 4],
    visc_x: [f32; 4],
    visc_y: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Face {
    owner: u32,
    neighbor: u32,
    axis: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Config {
    n: u32,
    nx: u32,
    ny: u32,
    nonlinear_rounds: u32,
    bit_mask: u32,
    pad: [u32; 3],
}

#[derive(Clone)]
struct Options {
    n: usize,
    iterations: usize,
    samples: usize,
    rounds: Vec<u32>,
}

fn options() -> Options {
    let mut out = Options {
        n: 1024,
        iterations: 24,
        samples: 7,
        rounds: vec![0, 1, 2, 4],
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("missing value after {flag}"))
        };
        match flag.as_str() {
            "--n" => out.n = value().parse().expect("integer --n"),
            "--iterations" => {
                out.iterations = value().parse().expect("integer --iterations")
            }
            "--samples" => out.samples = value().parse().expect("integer --samples"),
            "--rounds" => {
                out.rounds = value()
                    .split(',')
                    .map(|v| v.parse().expect("comma-separated integer --rounds"))
                    .collect()
            }
            "-h" | "--help" => {
                println!(
                    "probe_incidence_flux_remat [--n N] [--iterations N] \
                     [--samples N] [--rounds 0,1,2,4]"
                );
                std::process::exit(0);
            }
            _ => panic!("unknown option {flag}"),
        }
    }
    assert!(out.n >= 8 && out.iterations > 0 && out.samples >= 3);
    assert!(!out.rounds.is_empty());
    out
}

const SHADER: &str = r#"
struct Cell {
    q: vec4<f32>,
    grad_x: vec4<f32>,
    grad_y: vec4<f32>,
    primitive: vec4<f32>,
    visc_x: vec4<f32>,
    visc_y: vec4<f32>,
}

struct Face {
    owner: u32,
    neighbor: u32,
    axis: u32,
    pad: u32,
}

struct Flux8 {
    differential: vec4<f32>,
    algebraic: vec4<f32>,
}

struct Config {
    n: u32,
    nx: u32,
    ny: u32,
    nonlinear_rounds: u32,
    bit_mask: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> state: array<Cell>;
@group(0) @binding(1) var<storage, read> faces: array<Face>;
@group(0) @binding(2) var<storage, read> incidences: array<u32>;
@group(0) @binding(3) var<storage, read_write> face_flux: array<Flux8>;
@group(0) @binding(4) var<storage, read_write> rhs_materialized: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> rhs_pull: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> cfg: Config;

fn cell(x: u32, y: u32) -> u32 {
    return (y % cfg.ny) * cfg.nx + (x % cfg.nx);
}

fn normal_for_direction(direction: u32) -> vec2<f32> {
    switch direction {
        case 0u: { return vec2<f32>(-1.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u: { return vec2<f32>(0.0, -1.0); }
        default: { return vec2<f32>(0.0, 1.0); }
    }
}

fn structured_neighbor(i: u32, direction: u32) -> u32 {
    let x = i % cfg.nx;
    let y = i / cfg.nx;
    switch direction {
        case 0u: { return cell(x + cfg.nx - 1u, y); }
        case 1u: { return cell(x + 1u, y); }
        case 2u: { return cell(x, y + cfg.ny - 1u); }
        default: { return cell(x, y + 1u); }
    }
}

// A bounded, production-shaped nonlinear flux.  The fixed core uses all six
// vec4 records at both endpoints.  Each optional round adds exactly eight
// scalar sqrt, eight scalar divide, and 73 scalar add/multiply operations
// (including the scalar round seed).
// The sweep therefore exposes the ALU/traffic crossover instead of assuming
// rematerialization is free.
fn numerical_flux(owner: Cell, neighbor: Cell, normal: vec2<f32>) -> Flux8 {
    let go = owner.grad_x * normal.x + owner.grad_y * normal.y;
    let gn = neighbor.grad_x * normal.x + neighbor.grad_y * normal.y;
    var left = owner.q + 0.25 * go + 0.03125 * (owner.visc_x + owner.visc_y);
    var right = neighbor.q - 0.25 * gn + 0.03125 * (neighbor.visc_x + neighbor.visc_y);

    for (var round = 0u; round < cfg.nonlinear_rounds; round++) {
        let seed = 0.0001 * f32(round + 1u);
        let root_l = sqrt(abs(left) + vec4<f32>(0.25));
        let root_r = sqrt(abs(right) + vec4<f32>(0.25));
        let next_l = (left + 0.125 * right * root_l + vec4<f32>(seed)) /
            (vec4<f32>(1.0) + 0.01 * root_r);
        let next_r = (right - 0.125 * left * root_r - vec4<f32>(seed)) /
            (vec4<f32>(1.0) + 0.01 * root_l);
        left = next_l + 0.03 * next_l.yzwx;
        right = next_r - 0.03 * next_r.wxyz;
    }

    let rho_l = abs(left.x) + 0.5;
    let rho_r = abs(right.x) + 0.5;
    let inv_l = 1.0 / rho_l;
    let inv_r = 1.0 / rho_r;
    let un_l = (left.y * normal.x + left.z * normal.y) * inv_l;
    let un_r = (right.y * normal.x + right.z * normal.y) * inv_r;
    let p_l = abs(owner.primitive.x + 0.4 * left.w -
        0.2 * (left.y * left.y + left.z * left.z) * inv_l) + 0.05;
    let p_r = abs(neighbor.primitive.x + 0.4 * right.w -
        0.2 * (right.y * right.y + right.z * right.z) * inv_r) + 0.05;
    let c_l = sqrt(1.4 * p_l * inv_l);
    let c_r = sqrt(1.4 * p_r * inv_r);
    let ap = max(max(un_l + c_l, un_r + c_r), 0.0);
    let am = min(min(un_l - c_l, un_r - c_r), 0.0);
    let inv_wave = 1.0 / max(ap - am, 0.000001);
    let physical_l = vec4<f32>(
        left.x * un_l,
        left.y * un_l + p_l * normal.x,
        left.z * un_l + p_l * normal.y,
        (left.w + p_l) * un_l,
    );
    let physical_r = vec4<f32>(
        right.x * un_r,
        right.y * un_r + p_r * normal.x,
        right.z * un_r + p_r * normal.y,
        (right.w + p_r) * un_r,
    );
    let differential_raw = (ap * physical_l - am * physical_r + ap * am * (right - left)) * inv_wave;
    // A storage dispatch is an observable f32 materialization point.  Make
    // that point explicit in the rematerialized function too, so Metal fast
    // math cannot contract the returned flux into the cell accumulation.  The
    // runtime mask is zero; keeping it uniform prevents compile-time erasure.
    let differential = bitcast<vec4<f32>>(
        bitcast<vec4<u32>>(differential_raw) ^ vec4<u32>(cfg.bit_mask)
    );
    let algebraic = 0.5 * (owner.primitive + neighbor.primitive) +
        0.0625 * (owner.visc_x - neighbor.visc_x + owner.visc_y - neighbor.visc_y) +
        differential.wxyz * vec4<f32>(0.003, -0.002, 0.004, -0.001);
    return Flux8(differential, algebraic);
}

fn local_diffusion(current: Cell, other: Cell) -> vec4<f32> {
    return 0.0007 * (other.q - current.q) +
        0.00003 * (other.visc_x + other.visc_y - current.visc_x - current.visc_y);
}

fn materialize_f32(value: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(
        bitcast<vec4<u32>>(value) ^ vec4<u32>(cfg.bit_mask)
    );
}

@compute @workgroup_size(256)
fn structured_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let owner = gid.x;
    if (owner >= cfg.n) { return; }
    let current = state[owner];
    for (var direction = 0u; direction < 4u; direction++) {
        let neighbor = structured_neighbor(owner, direction);
        face_flux[4u * owner + direction] = numerical_flux(
            current, state[neighbor], normal_for_direction(direction)
        );
    }
}

@compute @workgroup_size(256)
fn structured_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let current = state[i];
    var residual = vec4<f32>(0.0);
    for (var direction = 0u; direction < 4u; direction++) {
        let other = state[structured_neighbor(i, direction)];
        residual = materialize_f32(
            residual + face_flux[4u * i + direction].differential
        );
        residual = materialize_f32(residual + local_diffusion(current, other));
    }
    rhs_materialized[i] = residual;
}

@compute @workgroup_size(256)
fn structured_pull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let current = state[i];
    var residual = vec4<f32>(0.0);
    for (var direction = 0u; direction < 4u; direction++) {
        let other = state[structured_neighbor(i, direction)];
        let flux = numerical_flux(current, other, normal_for_direction(direction));
        residual = materialize_f32(residual + flux.differential);
        residual = materialize_f32(residual + local_diffusion(current, other));
    }
    rhs_pull[i] = residual;
}

@compute @workgroup_size(256)
fn csr_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= 2u * cfg.n) { return; }
    let face = faces[f];
    let normal = select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), face.axis == 1u);
    face_flux[f] = numerical_flux(state[face.owner], state[face.neighbor], normal);
}

@compute @workgroup_size(256)
fn csr_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let current = state[i];
    var residual = vec4<f32>(0.0);
    for (var direction = 0u; direction < 4u; direction++) {
        let encoded = incidences[4u * i + direction];
        let f = encoded >> 1u;
        let is_neighbor = (encoded & 1u) != 0u;
        let face = faces[f];
        let other_index = select(face.neighbor, face.owner, is_neighbor);
        let sign = select(1.0, -1.0, is_neighbor);
        residual = materialize_f32(residual + sign * face_flux[f].differential);
        residual = materialize_f32(
            residual + local_diffusion(current, state[other_index])
        );
    }
    rhs_materialized[i] = residual;
}

@compute @workgroup_size(256)
fn csr_pull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let current = state[i];
    var residual = vec4<f32>(0.0);
    for (var direction = 0u; direction < 4u; direction++) {
        let encoded = incidences[4u * i + direction];
        let f = encoded >> 1u;
        let is_neighbor = (encoded & 1u) != 0u;
        let face = faces[f];
        let other_index = select(face.neighbor, face.owner, is_neighbor);
        let other = state[other_index];
        let normal = select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), face.axis == 1u);
        var flux: Flux8;
        if (is_neighbor) {
            // Canonical arguments remain owner, neighbor.  Only the final
            // incidence contribution changes sign.
            flux = numerical_flux(other, current, normal);
        } else {
            flux = numerical_flux(current, other, normal);
        }
        let sign = select(1.0, -1.0, is_neighbor);
        residual = materialize_f32(residual + sign * flux.differential);
        residual = materialize_f32(residual + local_diffusion(current, other));
    }
    rhs_pull[i] = residual;
}
"#;

#[derive(Clone, Copy)]
enum Topology {
    Structured,
    CsrOrdered,
    CsrPermuted,
}

impl Topology {
    fn label(self) -> &'static str {
        match self {
            Self::Structured => "structured-directed",
            Self::CsrOrdered => "csr-ordered",
            Self::CsrPermuted => "csr-permuted",
        }
    }

    fn is_structured(self) -> bool {
        matches!(self, Self::Structured)
    }
}

struct Pipelines {
    structured_faces: wgpu::ComputePipeline,
    structured_residual: wgpu::ComputePipeline,
    structured_pull: wgpu::ComputePipeline,
    csr_faces: wgpu::ComputePipeline,
    csr_residual: wgpu::ComputePipeline,
    csr_pull: wgpu::ComputePipeline,
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &'static str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn make_pipelines(ctx: &GpuContext) -> (Pipelines, wgpu::BindGroupLayout) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("incidence flux rematerialization probe"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
    let entries = (0..=6)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 6 {
                    wgpu::BufferBindingType::Uniform
                } else if binding <= 2 {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect::<Vec<_>>();
    let bind_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("incidence flux rematerialization bind layout"),
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("incidence flux rematerialization pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
    (
        Pipelines {
            structured_faces: pipeline(&ctx.device, &layout, &module, "structured_faces"),
            structured_residual: pipeline(
                &ctx.device,
                &layout,
                &module,
                "structured_residual",
            ),
            structured_pull: pipeline(&ctx.device, &layout, &module, "structured_pull"),
            csr_faces: pipeline(&ctx.device, &layout, &module, "csr_faces"),
            csr_residual: pipeline(&ctx.device, &layout, &module, "csr_residual"),
            csr_pull: pipeline(&ctx.device, &layout, &module, "csr_pull"),
        },
        bind_layout,
    )
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
    p
}

fn make_state(nx: usize, permutation: &[usize]) -> Vec<Cell> {
    let n = nx * nx;
    let mut state = vec![Cell::zeroed(); n];
    for y in 0..nx {
        for x in 0..nx {
            let logical = y * nx + x;
            let physical = permutation[logical];
            let xf = (x as f32 + 0.5) / nx as f32;
            let yf = (y as f32 + 0.5) / nx as f32;
            let s = (std::f32::consts::TAU * xf).sin();
            let c = (std::f32::consts::TAU * yf).cos();
            state[physical] = Cell {
                q: [1.0 + 0.1 * s, 0.31 + 0.07 * c, -0.23 + 0.04 * s, 2.7 + 0.08 * c],
                grad_x: [0.03 * c, -0.02 * s, 0.04 * c, -0.01 * s],
                grad_y: [-0.02 * s, 0.01 * c, 0.015 * s, 0.025 * c],
                primitive: [1.0 + 0.02 * c, 0.31, -0.23, 1.02 + 0.03 * s],
                visc_x: [0.011 * s, -0.013 * c, 0.017 * s, -0.019 * c],
                visc_y: [-0.007 * c, 0.023 * s, -0.029 * c, 0.031 * s],
            };
        }
    }
    state
}

fn csr_topology(nx: usize, permutation: &[usize]) -> (Vec<Face>, Vec<u32>) {
    let n = nx * nx;
    let logical = |x: usize, y: usize| (y % nx) * nx + (x % nx);
    let mut faces = Vec::with_capacity(2 * n);
    for y in 0..nx {
        for x in 0..nx {
            faces.push(Face {
                owner: permutation[logical(x, y)] as u32,
                neighbor: permutation[logical(x + 1, y)] as u32,
                axis: 0,
                pad: 0,
            });
        }
    }
    for y in 0..nx {
        for x in 0..nx {
            faces.push(Face {
                owner: permutation[logical(x, y)] as u32,
                neighbor: permutation[logical(x, y + 1)] as u32,
                axis: 1,
                pad: 0,
            });
        }
    }
    let mut incidences = vec![0u32; 4 * n];
    for y in 0..nx {
        for x in 0..nx {
            let physical = permutation[logical(x, y)];
            let west = logical(x + nx - 1, y);
            let east = logical(x, y);
            let south = n + logical(x, y + nx - 1);
            let north = n + logical(x, y);
            incidences[4 * physical..4 * physical + 4].copy_from_slice(&[
                ((west as u32) << 1) | 1,
                (east as u32) << 1,
                ((south as u32) << 1) | 1,
                (north as u32) << 1,
            ]);
        }
    }
    (faces, incidences)
}

fn upload<T: Pod>(ctx: &GpuContext, label: &str, values: &[T]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
}

fn scratch(ctx: &GpuContext, label: &str, bytes: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

struct Case {
    topology: Topology,
    n: usize,
    config: wgpu::Buffer,
    bind: wgpu::BindGroup,
    rhs_materialized: wgpu::Buffer,
    rhs_pull: wgpu::Buffer,
    _state: wgpu::Buffer,
    _faces: wgpu::Buffer,
    _incidences: wgpu::Buffer,
    _face_flux: wgpu::Buffer,
}

impl Case {
    fn new(
        ctx: &GpuContext,
        bind_layout: &wgpu::BindGroupLayout,
        topology: Topology,
        nx: usize,
    ) -> Self {
        let n = nx * nx;
        let identity: Vec<_> = (0..n).collect();
        let permutation = if matches!(topology, Topology::CsrPermuted) {
            permutation(n)
        } else {
            identity
        };
        let state = upload(ctx, "remat state", &make_state(nx, &permutation));
        let (face_values, incidence_values) = csr_topology(nx, &permutation);
        let faces = upload(ctx, "remat faces", &face_values);
        let incidences = upload(ctx, "remat incidences", &incidence_values);
        // Structured directed faces are the largest flux domain: 4N * 32 B.
        let face_flux = scratch(ctx, "remat face flux", (4 * n * 32) as u64);
        let rhs_materialized = scratch(ctx, "remat materialized rhs", (n * 16) as u64);
        let rhs_pull = scratch(ctx, "remat pull rhs", (n * 16) as u64);
        let config = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("remat config"),
                contents: bytemuck::bytes_of(&Config {
                    n: n as u32,
                    nx: nx as u32,
                        ny: nx as u32,
                        nonlinear_rounds: 0,
                        bit_mask: 0,
                        pad: [0; 3],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("remat bind"),
            layout: bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: faces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: incidences.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: face_flux.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: rhs_materialized.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: rhs_pull.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: config.as_entire_binding(),
                },
            ],
        });
        Self {
            topology,
            n,
            config,
            bind,
            rhs_materialized,
            rhs_pull,
            _state: state,
            _faces: faces,
            _incidences: incidences,
            _face_flux: face_flux,
        }
    }

    fn set_rounds(&self, ctx: &GpuContext, nx: usize, rounds: u32) {
        ctx.queue.write_buffer(
            &self.config,
            0,
            bytemuck::bytes_of(&Config {
                n: self.n as u32,
                nx: nx as u32,
                ny: nx as u32,
                nonlinear_rounds: rounds,
                bit_mask: 0,
                pad: [0; 3],
            }),
        );
    }
}

fn encode(
    ctx: &GpuContext,
    pipes: &Pipelines,
    case: &Case,
    iterations: usize,
    pull: bool,
) -> wgpu::CommandBuffer {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("remat timing encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("remat timing pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &case.bind, &[]);
        for _ in 0..iterations {
            if case.topology.is_structured() {
                if pull {
                    pass.set_pipeline(&pipes.structured_pull);
                    pass.dispatch_workgroups((case.n as u32).div_ceil(WG_SIZE), 1, 1);
                } else {
                    pass.set_pipeline(&pipes.structured_faces);
                    pass.dispatch_workgroups((case.n as u32).div_ceil(WG_SIZE), 1, 1);
                    pass.set_pipeline(&pipes.structured_residual);
                    pass.dispatch_workgroups((case.n as u32).div_ceil(WG_SIZE), 1, 1);
                }
            } else if pull {
                pass.set_pipeline(&pipes.csr_pull);
                pass.dispatch_workgroups((case.n as u32).div_ceil(WG_SIZE), 1, 1);
            } else {
                pass.set_pipeline(&pipes.csr_faces);
                pass.dispatch_workgroups((2 * case.n as u32).div_ceil(WG_SIZE), 1, 1);
                pass.set_pipeline(&pipes.csr_residual);
                pass.dispatch_workgroups((case.n as u32).div_ceil(WG_SIZE), 1, 1);
            }
        }
    }
    encoder.finish()
}

fn wait(ctx: &GpuContext, submission: wgpu::SubmissionIndex) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("wait for remat probe");
}

fn timed(ctx: &GpuContext, command: wgpu::CommandBuffer) -> Duration {
    let start = Instant::now();
    let submission = ctx.queue.submit(Some(command));
    wait(ctx, submission);
    start.elapsed()
}

fn read_rhs(ctx: &GpuContext, source: &wgpu::Buffer, n: usize) -> Vec<[f32; 4]> {
    let bytes = (n * 16) as u64;
    let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("remat readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("remat readback encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &read, 0, bytes);
    let submission = ctx.queue.submit(Some(encoder.finish()));
    let slice = read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    wait(ctx, submission);
    rx.recv().expect("map callback").expect("map rhs");
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
    drop(mapped);
    read.unmap();
    result
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn compare_bits(a: &[[f32; 4]], b: &[[f32; 4]]) -> (usize, f32) {
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for (left, right) in a.iter().zip(b) {
        for lane in 0..4 {
            mismatches += usize::from(left[lane].to_bits() != right[lane].to_bits());
            max_abs = max_abs.max((left[lane] - right[lane]).abs());
        }
    }
    (mismatches, max_abs)
}

fn run_case(
    ctx: &GpuContext,
    pipes: &Pipelines,
    bind_layout: &wgpu::BindGroupLayout,
    topology: Topology,
    options: &Options,
) {
    let case = Case::new(ctx, bind_layout, topology, options.n);
    for &rounds in &options.rounds {
        case.set_rounds(ctx, options.n, rounds);
        wait(
            ctx,
            ctx.queue.submit(Some(
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("remat uniform fence"),
                    })
                    .finish(),
            )),
        );

        wait(ctx, ctx.queue.submit(Some(encode(ctx, pipes, &case, 1, false))));
        wait(ctx, ctx.queue.submit(Some(encode(ctx, pipes, &case, 1, true))));
        let materialized = read_rhs(ctx, &case.rhs_materialized, case.n);
        let pull = read_rhs(ctx, &case.rhs_pull, case.n);
        let (mismatches, max_abs) = compare_bits(&materialized, &pull);
        if mismatches != 0 {
            for (index, (left, right)) in materialized.iter().zip(&pull).enumerate().take(3) {
                eprintln!("parity sample {index}: materialized={left:?} pull={right:?}");
            }
        }
        assert_eq!(mismatches, 0, "canonical pull changed residual bits");

        // Warm each schedule with the complete working set.
        wait(ctx, ctx.queue.submit(Some(encode(ctx, pipes, &case, 2, false))));
        wait(ctx, ctx.queue.submit(Some(encode(ctx, pipes, &case, 2, true))));

        let mut materialized_times = Vec::with_capacity(options.samples);
        let mut pull_times = Vec::with_capacity(options.samples);
        for sample in 0..options.samples {
            let baseline_command = encode(ctx, pipes, &case, options.iterations, false);
            let pull_command = encode(ctx, pipes, &case, options.iterations, true);
            if sample & 1 == 0 {
                materialized_times.push(timed(ctx, baseline_command));
                pull_times.push(timed(ctx, pull_command));
            } else {
                pull_times.push(timed(ctx, pull_command));
                materialized_times.push(timed(ctx, baseline_command));
            }
        }
        let materialized_time = median(materialized_times).as_secs_f64()
            / options.iterations as f64;
        let pull_time = median(pull_times).as_secs_f64() / options.iterations as f64;
        println!(
            "topology={:<19} rounds={} materialized={:>8.3} ms/stage pull={:>8.3} ms/stage speedup={:>6.3}x bit_mismatches={} max_abs={:.3e}",
            topology.label(),
            rounds,
            materialized_time * 1.0e3,
            pull_time * 1.0e3,
            materialized_time / pull_time,
            mismatches,
            max_abs,
        );
    }
}

fn main() {
    let options = options();
    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("GPU context");
    let (pipes, bind_layout) = make_pipelines(&ctx);
    println!(
        "device={:?} cells={} iterations={} samples={} rounds={:?}",
        ctx.device.limits(),
        options.n * options.n,
        options.iterations,
        options.samples,
        options.rounds,
    );
    println!(
        "ALU: fixed flux core ~= 184 scalar add/mul + 2 sqrt + 3 divide; each round = 73 add/mul + 8 sqrt + 8 divide; materialized writes 8 floats/face and gathers 4"
    );
    println!(
        "declared flux-buffer traffic removed: structured=192 B/cell-stage, CSR=128 B/cell-stage (actual compressible 8-write/4-read layout)"
    );
    println!(
        "probe active-field traffic: structured=928 -> 496 B/cell-stage, CSR=880 -> 576 B/cell-stage (materialized -> pull; excludes cache effects)"
    );
    for topology in [
        Topology::Structured,
        Topology::CsrOrdered,
        Topology::CsrPermuted,
    ] {
        run_case(&ctx, &pipes, &bind_layout, topology, &options);
    }
}
