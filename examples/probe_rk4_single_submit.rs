//! Isolated benchmark for whole-step RK4 command batching.
//!
//! This intentionally does not use or modify the production solver.  It models
//! the production dependency chain with a neighbor-reading residual dispatch
//! followed by the exact compact classical-RK4 update.  `split4` is the
//! structured schedule (one residual+update submission per stage), `split8` is
//! the current generic/CSR schedule (separate residual and update submissions),
//! and `single` binds four aligned slices of one uniform buffer and records the
//! complete step in one compute pass/command buffer. Dispatch boundaries are
//! retained as the grid-wide memory barriers required by both stencils.
//!
//! Run in release mode, for example:
//!
//! ```text
//! cargo run --release --example probe_rk4_single_submit -- --nx 128 --steps 400
//! ```

use bytemuck::{Pod, Zeroable};
use cfd2::solver::gpu::context::GpuContext;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WG: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nx: u32,
    ny: u32,
    stage: u32,
    _pad0: u32,
    dt: f32,
    time: f32,
    lambda: f32,
    _pad1: f32,
}

const STRUCTURED_RESIDUAL: &str = r#"
struct Params {
    nx: u32,
    ny: u32,
    stage: u32,
    _pad0: u32,
    dt: f32,
    time: f32,
    lambda: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = p.nx * p.ny;
    if (i >= n) { return; }
    let x = i % p.nx;
    let y = i / p.nx;
    let xm = select(x - 1u, p.nx - 1u, x == 0u);
    let xp = select(x + 1u, 0u, x + 1u == p.nx);
    let ym = select(y - 1u, p.ny - 1u, y == 0u);
    let yp = select(y + 1u, 0u, y + 1u == p.ny);
    let l = y * p.nx + xm;
    let r = y * p.nx + xp;
    let d = ym * p.nx + x;
    let u = yp * p.nx + x;
    let q = state[i];
    rhs[i] = p.lambda * (state[l] + state[r] + state[d] + state[u] - 4.0 * q)
        + 0.001 * sin(p.time + f32(i & 31u));
}
"#;

const CSR_RESIDUAL: &str = r#"
struct Params {
    nx: u32,
    ny: u32,
    stage: u32,
    _pad0: u32,
    dt: f32,
    time: f32,
    lambda: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> neighbors: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = p.nx * p.ny;
    if (i >= n) { return; }
    var sum = 0.0;
    for (var k = offsets[i]; k < offsets[i + 1u]; k++) {
        sum += state[neighbors[k]];
    }
    let degree = f32(offsets[i + 1u] - offsets[i]);
    rhs[i] = p.lambda * (sum - degree * state[i])
        + 0.001 * sin(p.time + f32(i & 31u));
}
"#;

const UPDATE: &str = r#"
struct Params {
    nx: u32,
    ny: u32,
    stage: u32,
    _pad0: u32,
    dt: f32,
    time: f32,
    lambda: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> base: array<f32>;
@group(0) @binding(3) var<storage, read_write> accum: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.nx * p.ny) { return; }
    let k = rhs[i];
    if (p.stage == 0u) {
        base[i] = state[i];
        accum[i] = k / 6.0;
        state[i] = base[i] + 0.5 * p.dt * k;
    } else if (p.stage == 1u) {
        accum[i] += k / 3.0;
        state[i] = base[i] + 0.5 * p.dt * k;
    } else if (p.stage == 2u) {
        accum[i] += k / 3.0;
        state[i] = base[i] + p.dt * k;
    } else {
        state[i] = base[i] + p.dt * (accum[i] + k / 6.0);
    }
}
"#;

#[derive(Clone, Copy, Debug)]
enum Topology {
    Structured,
    Csr,
}

#[derive(Clone, Copy, Debug)]
enum Schedule {
    Split4,
    Split8,
    Single,
}

struct Probe<'a> {
    ctx: &'a GpuContext,
    nx: u32,
    ny: u32,
    dt: f32,
    state: wgpu::Buffer,
    _rhs: wgpu::Buffer,
    _base: wgpu::Buffer,
    _accum: wgpu::Buffer,
    live_params: wgpu::Buffer,
    stage_params: wgpu::Buffer,
    stage_param_stride: u64,
    residual_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    residual_bg: wgpu::BindGroup,
    update_bg: wgpu::BindGroup,
    residual_stage_bgs: Vec<wgpu::BindGroup>,
    update_stage_bgs: Vec<wgpu::BindGroup>,
    next_step: u64,
}

fn pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn storage(device: &wgpu::Device, label: &str, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn csr(nx: u32, ny: u32) -> (Vec<u32>, Vec<u32>) {
    let n = nx * ny;
    let mut offsets = Vec::with_capacity(n as usize + 1);
    let mut neighbors = Vec::with_capacity(n as usize * 4);
    offsets.push(0);
    for i in 0..n {
        let x = i % nx;
        let y = i / nx;
        let xm = if x == 0 { nx - 1 } else { x - 1 };
        let xp = if x + 1 == nx { 0 } else { x + 1 };
        let ym = if y == 0 { ny - 1 } else { y - 1 };
        let yp = if y + 1 == ny { 0 } else { y + 1 };
        // Deliberately non-geometric order: only the CSR adjacency contract is
        // relevant to the schedule optimization.
        neighbors.extend([yp * nx + x, y * nx + xm, ym * nx + x, y * nx + xp]);
        offsets.push(neighbors.len() as u32);
    }
    (offsets, neighbors)
}

impl<'a> Probe<'a> {
    fn new(ctx: &'a GpuContext, topology: Topology, nx: u32, ny: u32) -> Self {
        let device = &ctx.device;
        let n = (nx * ny) as usize;
        let initial: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as u32 % nx) as f32 / nx as f32;
                let y = (i as u32 / nx) as f32 / ny as f32;
                (std::f32::consts::TAU * x).sin() * (std::f32::consts::TAU * y).cos()
            })
            .collect();
        let zero = vec![0.0f32; n];
        let state = storage(device, "rk batch probe state", &initial);
        let rhs = storage(device, "rk batch probe rhs", &zero);
        let base = storage(device, "rk batch probe base", &zero);
        let accum = storage(device, "rk batch probe accum", &zero);
        let live_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rk batch live params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let param_bytes = std::mem::size_of::<Params>() as u64;
        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stage_param_stride = param_bytes.div_ceil(uniform_alignment) * uniform_alignment;
        let stage_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rk batch staged params"),
            size: 4 * stage_param_stride,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let residual_pipeline = pipeline(
            device,
            "rk batch residual",
            match topology {
                Topology::Structured => STRUCTURED_RESIDUAL,
                Topology::Csr => CSR_RESIDUAL,
            },
        );
        let update_pipeline = pipeline(device, "rk batch update", UPDATE);

        let csr_buffers = if matches!(topology, Topology::Csr) {
            let (offsets, neighbors) = csr(nx, ny);
            Some((
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rk batch csr offsets"),
                    contents: bytemuck::cast_slice(&offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rk batch csr neighbors"),
                    contents: bytemuck::cast_slice(&neighbors),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            ))
        } else {
            None
        };

        let residual_layout = residual_pipeline.get_bind_group_layout(0);
        let make_residual_bg = |label: &str, params: wgpu::BindingResource<'_>| {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params,
                },
            ];
            if let Some((offset_buffer, neighbor_buffer)) = &csr_buffers {
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offset_buffer.as_entire_binding(),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: neighbor_buffer.as_entire_binding(),
                });
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &residual_layout,
                entries: &entries,
            })
        };
        let residual_bg =
            make_residual_bg("rk batch residual live bg", live_params.as_entire_binding());
        let residual_stage_bgs = (0..4u64)
            .map(|stage| {
                make_residual_bg(
                    "rk batch residual stage bg",
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &stage_params,
                        offset: stage * stage_param_stride,
                        size: NonZeroU64::new(param_bytes),
                    }),
                )
            })
            .collect();
        // Bind groups retain their resources. Keeping the local handles until
        // creation completes also makes the intended ownership explicit.
        drop(csr_buffers);

        let update_layout = update_pipeline.get_bind_group_layout(0);
        let make_update_bg = |label: &str, params: wgpu::BindingResource<'_>| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &update_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: base.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: accum.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params,
                    },
                ],
            })
        };
        let update_bg = make_update_bg("rk batch update live bg", live_params.as_entire_binding());
        let update_stage_bgs = (0..4u64)
            .map(|stage| {
                make_update_bg(
                    "rk batch update stage bg",
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &stage_params,
                        offset: stage * stage_param_stride,
                        size: NonZeroU64::new(param_bytes),
                    }),
                )
            })
            .collect();

        Self {
            ctx,
            nx,
            ny,
            dt: 1.0e-3,
            state,
            _rhs: rhs,
            _base: base,
            _accum: accum,
            live_params,
            stage_params,
            stage_param_stride,
            residual_pipeline,
            update_pipeline,
            residual_bg,
            update_bg,
            residual_stage_bgs,
            update_stage_bgs,
            next_step: 0,
        }
    }

    fn params(&self, step: u64, stage: u32) -> Params {
        const C: [f32; 4] = [0.0, 0.5, 0.5, 1.0];
        Params {
            nx: self.nx,
            ny: self.ny,
            stage,
            _pad0: 0,
            dt: self.dt,
            time: (step as f32 + C[stage as usize]) * self.dt,
            lambda: 0.125,
            _pad1: 0.0,
        }
    }

    fn dispatch_residual(&self, pass: &mut wgpu::ComputePass<'_>, bind_group: &wgpu::BindGroup) {
        pass.set_pipeline(&self.residual_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups((self.nx * self.ny).div_ceil(WG), 1, 1);
    }

    fn dispatch_update(&self, pass: &mut wgpu::ComputePass<'_>, bind_group: &wgpu::BindGroup) {
        pass.set_pipeline(&self.update_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups((self.nx * self.ny).div_ceil(WG), 1, 1);
    }

    fn run_steps(&mut self, schedule: Schedule, steps: usize) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        for _ in 0..steps {
            let step = self.next_step;
            match schedule {
                Schedule::Split4 | Schedule::Split8 => {
                    for stage in 0..4 {
                        let params = self.params(step, stage);
                        queue.write_buffer(&self.live_params, 0, bytemuck::bytes_of(&params));
                        if matches!(schedule, Schedule::Split8) {
                            let mut residual =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("rk split residual"),
                                });
                            {
                                let mut pass =
                                    residual.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: Some("rk split residual"),
                                        timestamp_writes: None,
                                    });
                                self.dispatch_residual(&mut pass, &self.residual_bg);
                            }
                            queue.submit(Some(residual.finish()));

                            let mut update =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("rk split update"),
                                });
                            {
                                let mut pass =
                                    update.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: Some("rk split update"),
                                        timestamp_writes: None,
                                    });
                                self.dispatch_update(&mut pass, &self.update_bg);
                            }
                            queue.submit(Some(update.finish()));
                        } else {
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("rk split stage"),
                                });
                            {
                                let mut pass =
                                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: Some("rk split stage"),
                                        timestamp_writes: None,
                                    });
                                self.dispatch_residual(&mut pass, &self.residual_bg);
                                self.dispatch_update(&mut pass, &self.update_bg);
                            }
                            queue.submit(Some(encoder.finish()));
                        }
                    }
                }
                Schedule::Single => {
                    let records = [
                        self.params(step, 0),
                        self.params(step, 1),
                        self.params(step, 2),
                        self.params(step, 3),
                    ];
                    let mut packed = vec![0u8; (4 * self.stage_param_stride) as usize];
                    for (stage, record) in records.iter().enumerate() {
                        let offset = stage * self.stage_param_stride as usize;
                        let bytes = bytemuck::bytes_of(record);
                        packed[offset..offset + bytes.len()].copy_from_slice(bytes);
                    }
                    queue.write_buffer(&self.stage_params, 0, &packed);
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("rk complete step"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("rk complete step"),
                            timestamp_writes: None,
                        });
                        for stage in 0..4 {
                            self.dispatch_residual(&mut pass, &self.residual_stage_bgs[stage]);
                            self.dispatch_update(&mut pass, &self.update_stage_bgs[stage]);
                        }
                    }
                    queue.submit(Some(encoder.finish()));
                }
            }
            self.next_step += 1;
        }
    }

    fn sync(&self) {
        let encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rk batch fence"),
            });
        let submission = self.ctx.queue.submit(Some(encoder.finish()));
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("rk batch fence");
    }

    fn read_state(&self) -> Vec<f32> {
        let bytes = self.nx as u64 * self.ny as u64 * 4;
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rk batch readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rk batch readback"),
            });
        encoder.copy_buffer_to_buffer(&self.state, 0, &read, 0, bytes);
        let submission = self.ctx.queue.submit(Some(encoder.finish()));
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("send map result");
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("readback fence");
        rx.recv().expect("map callback").expect("map state");
        let mapped = slice.get_mapped_range();
        let out = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        read.unmap();
        out
    }
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn benchmark(
    ctx: &GpuContext,
    topology: Topology,
    schedule: Schedule,
    nx: u32,
    ny: u32,
    steps: usize,
    rounds: usize,
) -> Duration {
    let mut probe = Probe::new(ctx, topology, nx, ny);
    probe.run_steps(schedule, 4);
    probe.sync();
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        probe.run_steps(schedule, steps);
        probe.sync();
        samples.push(start.elapsed());
    }
    median(samples)
}

fn equivalence(ctx: &GpuContext, topology: Topology, reference: Schedule) {
    let mut split = Probe::new(ctx, topology, 37, 29);
    let mut single = Probe::new(ctx, topology, 37, 29);
    split.run_steps(reference, 17);
    single.run_steps(Schedule::Single, 17);
    split.sync();
    single.sync();
    let a = split.read_state();
    let b = single.read_state();
    let bit_mismatches = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "equivalence topology={topology:?} reference={reference:?} bit_mismatches={bit_mismatches} max_abs={max_abs:e}"
    );
    assert_eq!(bit_mismatches, 0, "whole-step batching changed RK4 values");
}

fn main() {
    let mut nx = 128u32;
    let mut ny = 128u32;
    let mut steps = 400usize;
    let mut rounds = 5usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("missing value after {name}"))
        };
        match flag.as_str() {
            "--nx" => nx = value(&mut args, &flag).parse().expect("integer nx"),
            "--ny" => ny = value(&mut args, &flag).parse().expect("integer ny"),
            "--steps" => steps = value(&mut args, &flag).parse().expect("integer steps"),
            "--rounds" => rounds = value(&mut args, &flag).parse().expect("integer rounds"),
            "--help" | "-h" => {
                println!("probe_rk4_single_submit [--nx N] [--ny N] [--steps N] [--rounds N]");
                return;
            }
            _ => panic!("unknown option {flag}"),
        }
    }
    assert!(nx > 1 && ny > 1 && steps > 0 && rounds > 0);

    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("GPU context");
    println!(
        "device limits: max_storage_buffers_per_shader_stage={} min_uniform_buffer_offset_alignment={} max_compute_workgroup_storage_size={} cells={} steps={} rounds={}",
        ctx.device.limits().max_storage_buffers_per_shader_stage,
        ctx.device.limits().min_uniform_buffer_offset_alignment,
        ctx.device.limits().max_compute_workgroup_storage_size,
        nx * ny,
        steps,
        rounds,
    );

    equivalence(&ctx, Topology::Structured, Schedule::Split4);
    equivalence(&ctx, Topology::Csr, Schedule::Split8);

    for (topology, incumbent) in [
        (Topology::Structured, Schedule::Split4),
        (Topology::Csr, Schedule::Split8),
    ] {
        let split = benchmark(&ctx, topology, incumbent, nx, ny, steps, rounds);
        let single = benchmark(&ctx, topology, Schedule::Single, nx, ny, steps, rounds);
        println!(
            "topology={topology:?} incumbent={incumbent:?} split={:.3}ms ({:.3}us/step) single={:.3}ms ({:.3}us/step) speedup={:.3}x",
            split.as_secs_f64() * 1e3,
            split.as_secs_f64() * 1e6 / steps as f64,
            single.as_secs_f64() * 1e3,
            single.as_secs_f64() * 1e6 / steps as f64,
            split.as_secs_f64() / single.as_secs_f64(),
        );
    }
}
