//! Temporary/adversarial probe for a radius-two structured workgroup tile.
//!
//! This deliberately contains no CFD model equations.  It compares a three-pass
//! global-memory stencil with the strongest plausible workgroup-tiled version of
//! the same dependency graph, so it is an upper bound on the benefit available
//! before real flux arithmetic and register pressure are introduced.

use cfd2::solver::gpu::context::GpuContext;
use std::time::{Duration, Instant};

const GLOBAL: &str = r#"
struct Grid { nx: u32, ny: u32 }
@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> grad: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> rhs: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

fn clamp_idx(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(grid.nx) - 1);
    let cy = clamp(y, 0, i32(grid.ny) - 1);
    return u32(cy) * grid.nx + u32(cx);
}

@compute @workgroup_size(256)
fn gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= grid.nx * grid.ny) { return; }
    let x = i32(i % grid.nx);
    let y = i32(i / grid.nx);
    grad[i] = (q[clamp_idx(x-1,y)] + q[clamp_idx(x+1,y)]
             + q[clamp_idx(x,y-1)] + q[clamp_idx(x,y+1)]) * 0.25;
}

@compute @workgroup_size(256)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= grid.nx * grid.ny) { return; }
    let x = i32(i % grid.nx);
    let y = i32(i / grid.nx);
    rhs[i] = q[i] * 0.125 + grad[i] * 0.25
           + grad[clamp_idx(x-1,y)] + grad[clamp_idx(x+1,y)]
           + grad[clamp_idx(x,y-1)] + grad[clamp_idx(x,y+1)];
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < grid.nx * grid.ny) { out[i] = q[i] + rhs[i] * 0.001; }
}
"#;

const TILED: &str = r#"
struct Grid { nx: u32, ny: u32 }
@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

var<workgroup> tile_q: array<vec4<f32>, 400>;  // (16+2r)^2, r=2
var<workgroup> tile_g: array<vec4<f32>, 324>;  // (16+2)^2

fn global_q(x: i32, y: i32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(grid.nx) - 1);
    let cy = clamp(y, 0, i32(grid.ny) - 1);
    return q[u32(cy) * grid.nx + u32(cx)];
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ox = i32(wid.x * 16u);
    let oy = i32(wid.y * 16u);
    for (var t = lane; t < 400u; t += 256u) {
        let hx = i32(t % 20u);
        let hy = i32(t / 20u);
        tile_q[t] = global_q(ox + hx - 2, oy + hy - 2);
    }
    workgroupBarrier();

    for (var t = lane; t < 324u; t += 256u) {
        let gx = t % 18u;
        let gy = t / 18u;
        let c = (gy + 1u) * 20u + gx + 1u;
        tile_g[t] = (tile_q[c-1u] + tile_q[c+1u]
                   + tile_q[c-20u] + tile_q[c+20u]) * 0.25;
    }
    workgroupBarrier();

    let x = wid.x * 16u + lid.x;
    let y = wid.y * 16u + lid.y;
    if (x < grid.nx && y < grid.ny) {
        let c = (lid.y + 2u) * 20u + lid.x + 2u;
        let g = (lid.y + 1u) * 18u + lid.x + 1u;
        let value = tile_q[c] * 0.125 + tile_g[g] * 0.25
                  + tile_g[g-1u] + tile_g[g+1u]
                  + tile_g[g-18u] + tile_g[g+18u];
        out[y * grid.nx + x] = tile_q[c] + value * 0.001;
    }
}
"#;

fn wait(ctx: &GpuContext, submission: wgpu::SubmissionIndex) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
}

fn buffer(ctx: &GpuContext, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage,
        mapped_at_creation: false,
    })
}

fn timed(ctx: &GpuContext, command: wgpu::CommandBuffer) -> Duration {
    let start = Instant::now();
    let submission = ctx.queue.submit(Some(command));
    wait(ctx, submission);
    start.elapsed()
}

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(2048u32);
    let iterations = std::env::args()
        .nth(2)
        .and_then(|x| x.parse().ok())
        .unwrap_or(20u32);
    let ctx = pollster::block_on(GpuContext::new(None, None)).unwrap();
    println!("device limits: {:?}", ctx.device.limits());
    let bytes = u64::from(n) * u64::from(n) * 16;
    let q0 = buffer(&ctx, "q0", bytes, wgpu::BufferUsages::STORAGE);
    let q1 = buffer(&ctx, "q1", bytes, wgpu::BufferUsages::STORAGE);
    let grad = buffer(&ctx, "grad", bytes, wgpu::BufferUsages::STORAGE);
    let rhs = buffer(&ctx, "rhs", bytes, wgpu::BufferUsages::STORAGE);
    let grid = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid"),
            contents: bytemuck::cast_slice(&[n, n]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let global_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("global probe"),
            source: wgpu::ShaderSource::Wgsl(GLOBAL.into()),
        });
    let tiled_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tiled probe"),
            source: wgpu::ShaderSource::Wgsl(TILED.into()),
        });
    // 6 persistent gradient scalars on the 18x18 halo plus one scalar for each
    // canonical tile face bring the declaration to exactly the device's
    // downlevel 16,352-byte workgroup-storage limit.
    let tiled_max_source = TILED
        .replace(
            "var<workgroup> tile_g: array<vec4<f32>, 324>;  // (16+2)^2",
            "var<workgroup> tile_g: array<vec4<f32>, 324>;  // four gradient scalars\n\
             var<workgroup> tile_gp: array<vec2<f32>, 324>; // two gradient scalars\n\
             var<workgroup> tile_face: array<f32, 544>;     // canonical tile faces",
        )
        .replace(
            "tile_g[t] = (tile_q[c-1u] + tile_q[c+1u]\n\
                   + tile_q[c-20u] + tile_q[c+20u]) * 0.25;",
            "tile_g[t] = (tile_q[c-1u] + tile_q[c+1u]\n\
                   + tile_q[c-20u] + tile_q[c+20u]) * 0.25;\n\
             tile_gp[t] = tile_g[t].xy;",
        )
        .replace(
            "workgroupBarrier();\n\n    let x = wid.x * 16u + lid.x;",
            "workgroupBarrier();\n\
             for (var t = lane; t < 544u; t += 256u) {\n\
                 let g = t % 324u;\n\
                 tile_face[t] = tile_g[g].x + tile_gp[g].y;\n\
             }\n\
             workgroupBarrier();\n\n\
             let x = wid.x * 16u + lid.x;",
        )
        .replace(
            "let value = tile_q[c] * 0.125 + tile_g[g] * 0.25",
            "let value = tile_q[c] * 0.125 + tile_g[g] * (0.25 + tile_face[lid.y * 17u + lid.x] * 0.000001)",
        );
    let tiled_max_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("max-storage tiled probe"),
            source: wgpu::ShaderSource::Wgsl(tiled_max_source.into()),
        });
    let global_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let global_pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("global pipeline layout"),
                bind_group_layouts: &[&global_layout],
                push_constant_ranges: &[],
            });
    let make_global = |label, entry| {
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&global_pipeline_layout),
                module: &global_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    let gp = make_global("gradient", "gradient");
    let rp = make_global("residual", "residual");
    let up = make_global("update", "update");
    let tp = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tiled"),
            layout: None,
            module: &tiled_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let tmp = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("max-storage tiled"),
            layout: None,
            module: &tiled_max_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let global_bind = |src: &wgpu::Buffer, out: &wgpu::Buffer| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global bindings"),
            layout: &global_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out.as_entire_binding(),
                },
            ],
        })
    };
    let tiled_layout = tp.get_bind_group_layout(0);
    let tiled_bind = |src: &wgpu::Buffer, out: &wgpu::Buffer| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiled bindings"),
            layout: &tiled_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        })
    };
    let gb = [global_bind(&q0, &q1), global_bind(&q1, &q0)];
    let tb = [tiled_bind(&q0, &q1), tiled_bind(&q1, &q0)];
    let tiled_max_layout = tmp.get_bind_group_layout(0);
    let tiled_max_bind = |src: &wgpu::Buffer, out: &wgpu::Buffer| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("max-storage tiled bindings"),
            layout: &tiled_max_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        })
    };
    let tmb = [tiled_max_bind(&q0, &q1), tiled_max_bind(&q1, &q0)];

    let encode_global = || {
        let mut e = ctx.device.create_command_encoder(&Default::default());
        {
            let mut p = e.begin_compute_pass(&Default::default());
            for i in 0..iterations {
                p.set_bind_group(0, &gb[(i & 1) as usize], &[]);
                for pipe in [&gp, &rp, &up] {
                    p.set_pipeline(pipe);
                    p.dispatch_workgroups((n * n).div_ceil(256), 1, 1);
                }
            }
        }
        e.finish()
    };
    let encode_tiled = || {
        let mut e = ctx.device.create_command_encoder(&Default::default());
        {
            let mut p = e.begin_compute_pass(&Default::default());
            p.set_pipeline(&tp);
            for i in 0..iterations {
                p.set_bind_group(0, &tb[(i & 1) as usize], &[]);
                p.dispatch_workgroups(n.div_ceil(16), n.div_ceil(16), 1);
            }
        }
        e.finish()
    };
    let encode_tiled_max = || {
        let mut e = ctx.device.create_command_encoder(&Default::default());
        {
            let mut p = e.begin_compute_pass(&Default::default());
            p.set_pipeline(&tmp);
            for i in 0..iterations {
                p.set_bind_group(0, &tmb[(i & 1) as usize], &[]);
                p.dispatch_workgroups(n.div_ceil(16), n.div_ceil(16), 1);
            }
        }
        e.finish()
    };
    wait(&ctx, ctx.queue.submit(Some(encode_global())));
    wait(&ctx, ctx.queue.submit(Some(encode_tiled())));
    wait(&ctx, ctx.queue.submit(Some(encode_tiled_max())));
    let g = timed(&ctx, encode_global()) / iterations;
    let t = timed(&ctx, encode_tiled()) / iterations;
    let tm = timed(&ctx, encode_tiled_max()) / iterations;
    println!(
        "{n}x{n}: global={:.3} ms tiled={:.3} ms max-storage-tiled={:.3} ms speedup={:.2}x/{:.2}x",
        g.as_secs_f64() * 1e3,
        t.as_secs_f64() * 1e3,
        tm.as_secs_f64() * 1e3,
        g.as_secs_f64() / t.as_secs_f64(),
        g.as_secs_f64() / tm.as_secs_f64()
    );
}

use wgpu::util::DeviceExt;
