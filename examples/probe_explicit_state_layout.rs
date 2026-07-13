//! Isolated GPU probe for explicit-state physical layouts.
//!
//! This does not participate in solver routing.  It compares the current
//! packed state record (`AoS23`) with a semantic-live compact record and a
//! 32-cell AoSoA block under two deliberately incompatible access classes:
//!
//! * affine cell dispatch, representative of structured stages/stencils;
//! * indirect cell gather, representative of unstructured face endpoints.
//!
//! The 8-component live set is compressible D+A.  The 20-component live set is
//! the high-order compressible face set after deleting the unused rho_e
//! gradients (full offsets 14 and 15).

use cfd2::solver::gpu::context::GpuContext;
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Params { n: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> state: array<f32>;
@group(0) @binding(2) var<storage, read> gather: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn linear_id(gid: vec3<u32>) -> u32 {
    return gid.y * 65535u * 256u + gid.x;
}

fn consume(i: u32, value: f32) {
    output[i] = value * 0.000001 + f32(i & 1u);
}

fn sum_aos23_8(cell: u32) -> f32 {
    let b = cell * 23u;
    return state[b] + state[b+1u] + state[b+2u] + state[b+7u]
         + state[b+8u] + state[b+9u] + state[b+10u] + state[b+11u];
}

fn sum_compact8(cell: u32) -> f32 {
    let b = cell * 8u;
    return state[b] + state[b+1u] + state[b+2u] + state[b+3u]
         + state[b+4u] + state[b+5u] + state[b+6u] + state[b+7u];
}

fn sum_aosoa8(cell: u32) -> f32 {
    let b = (cell / 32u) * 256u + cell % 32u;
    return state[b] + state[b+32u] + state[b+64u] + state[b+96u]
         + state[b+128u] + state[b+160u] + state[b+192u] + state[b+224u];
}

fn sum_aos23_20(cell: u32) -> f32 {
    let b = cell * 23u;
    var value = 0.0;
    for (var rank = 0u; rank < 14u; rank++) {
        value += state[b + rank];
    }
    for (var rank = 16u; rank < 22u; rank++) {
        value += state[b + rank];
    }
    return value;
}

fn sum_compact20(cell: u32) -> f32 {
    let b = cell * 20u;
    var value = 0.0;
    for (var rank = 0u; rank < 20u; rank++) {
        value += state[b + rank];
    }
    return value;
}

fn sum_compact20_first8(cell: u32) -> f32 {
    let b = cell * 20u;
    return state[b] + state[b+1u] + state[b+2u] + state[b+3u]
         + state[b+4u] + state[b+5u] + state[b+6u] + state[b+7u];
}

fn sum_split8_12(cell: u32) -> f32 {
    let evolved = cell * 8u;
    let producer = params.n * 8u + cell * 12u;
    var value = 0.0;
    for (var rank = 0u; rank < 8u; rank++) {
        value += state[evolved + rank];
    }
    for (var rank = 0u; rank < 12u; rank++) {
        value += state[producer + rank];
    }
    return value;
}

fn sum_aosoa20(cell: u32) -> f32 {
    let b = (cell / 32u) * 640u + cell % 32u;
    var value = 0.0;
    for (var rank = 0u; rank < 20u; rank++) {
        value += state[b + rank * 32u];
    }
    return value;
}

fn sum_aosoa20_first8(cell: u32) -> f32 {
    let b = (cell / 32u) * 640u + cell % 32u;
    return state[b] + state[b+32u] + state[b+64u] + state[b+96u]
         + state[b+128u] + state[b+160u] + state[b+192u] + state[b+224u];
}

@compute @workgroup_size(256)
fn affine_aos23_8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aos23_8(i)); }
}

@compute @workgroup_size(256)
fn affine_compact8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_compact8(i)); }
}

@compute @workgroup_size(256)
fn affine_aosoa8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aosoa8(i)); }
}

@compute @workgroup_size(256)
fn affine_aosoa20_first8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aosoa20_first8(i)); }
}

@compute @workgroup_size(256)
fn indirect_aos23_8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aos23_8(gather[i])); }
}

@compute @workgroup_size(256)
fn indirect_compact8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_compact8(gather[i])); }
}

@compute @workgroup_size(256)
fn indirect_aosoa8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aosoa8(gather[i])); }
}

@compute @workgroup_size(256)
fn affine_aos23_20(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aos23_20(i)); }
}

@compute @workgroup_size(256)
fn affine_compact20(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_compact20(i)); }
}

@compute @workgroup_size(256)
fn affine_compact20_first8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_compact20_first8(i)); }
}

@compute @workgroup_size(256)
fn affine_aosoa20(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aosoa20(i)); }
}

@compute @workgroup_size(256)
fn affine_split8_12(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_split8_12(i)); }
}

@compute @workgroup_size(256)
fn indirect_aos23_20(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aos23_20(gather[i])); }
}

@compute @workgroup_size(256)
fn indirect_compact20(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_compact20(gather[i])); }
}

@compute @workgroup_size(256)
fn indirect_aosoa20(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_aosoa20(gather[i])); }
}

@compute @workgroup_size(256)
fn indirect_split8_12(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = linear_id(gid);
    if (i < params.n) { consume(i, sum_split8_12(gather[i])); }
}
"#;

fn wait(ctx: &GpuContext, submission: wgpu::SubmissionIndex) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("GPU fence");
}

fn storage(ctx: &GpuContext, label: &str, floats: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: floats * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

struct Case {
    label: &'static str,
    entry: &'static str,
    state: usize,
    useful_read_bytes: u64,
}

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_097_152u32);
    let iterations = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(20u32);
    assert!(n.is_power_of_two(), "probe uses a power-of-two permutation");
    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("GPU context");

    let params = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layout probe params"),
            contents: bytemuck::cast_slice(&[n, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    // Multiplication by an odd number is a bijection modulo a power of two.
    // It intentionally destroys subgroup locality to expose layouts that add
    // an independent memory transaction for every gathered component.
    let permutation: Vec<u32> = (0..n)
        .map(|i| i.wrapping_mul(1_048_573) & (n - 1))
        .collect();
    let gather = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layout probe indirect indices"),
            contents: bytemuck::cast_slice(&permutation),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let states = [
        storage(&ctx, "AoS23", u64::from(n) * 23),
        storage(&ctx, "compact8", u64::from(n) * 8),
        storage(&ctx, "AoSoA8", u64::from(n.next_multiple_of(32)) * 8),
        storage(&ctx, "compact20", u64::from(n) * 20),
        storage(&ctx, "AoSoA20", u64::from(n.next_multiple_of(32)) * 20),
    ];
    let output = storage(&ctx, "layout probe output", u64::from(n));
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit state layout probe"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
    // Keep one stable interface for affine and indirect entry points.  An
    // inferred layout legitimately prunes `gather` from affine-only pipelines,
    // which would make the common benchmark bind group invalid.
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit state layout probe bindings"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit state layout probe pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let cases = [
        Case {
            label: "affine AoS23 K8",
            entry: "affine_aos23_8",
            state: 0,
            useful_read_bytes: 32,
        },
        Case {
            label: "affine compact-AoS8",
            entry: "affine_compact8",
            state: 1,
            useful_read_bytes: 32,
        },
        Case {
            label: "affine hot-AoS20 K8",
            entry: "affine_compact20_first8",
            state: 3,
            useful_read_bytes: 32,
        },
        Case {
            label: "affine hot-AoSoA20 K8",
            entry: "affine_aosoa20_first8",
            state: 4,
            useful_read_bytes: 32,
        },
        Case {
            label: "affine AoSoA32x8",
            entry: "affine_aosoa8",
            state: 2,
            useful_read_bytes: 32,
        },
        Case {
            label: "indirect AoS23 K8",
            entry: "indirect_aos23_8",
            state: 0,
            useful_read_bytes: 36,
        },
        Case {
            label: "indirect compact-AoS8",
            entry: "indirect_compact8",
            state: 1,
            useful_read_bytes: 36,
        },
        Case {
            label: "indirect AoSoA32x8",
            entry: "indirect_aosoa8",
            state: 2,
            useful_read_bytes: 36,
        },
        Case {
            label: "affine AoS23 K20",
            entry: "affine_aos23_20",
            state: 0,
            useful_read_bytes: 80,
        },
        Case {
            label: "affine compact-AoS20",
            entry: "affine_compact20",
            state: 3,
            useful_read_bytes: 80,
        },
        Case {
            label: "affine split-AoS8+12",
            entry: "affine_split8_12",
            state: 3,
            useful_read_bytes: 80,
        },
        Case {
            label: "affine AoSoA32x20",
            entry: "affine_aosoa20",
            state: 4,
            useful_read_bytes: 80,
        },
        Case {
            label: "indirect AoS23 K20",
            entry: "indirect_aos23_20",
            state: 0,
            useful_read_bytes: 84,
        },
        Case {
            label: "indirect compact-AoS20",
            entry: "indirect_compact20",
            state: 3,
            useful_read_bytes: 84,
        },
        Case {
            label: "indirect split-AoS8+12",
            entry: "indirect_split8_12",
            state: 3,
            useful_read_bytes: 84,
        },
        Case {
            label: "indirect AoSoA32x20",
            entry: "indirect_aosoa20",
            state: 4,
            useful_read_bytes: 84,
        },
    ];

    let groups = n.div_ceil(256);
    println!("explicit state layout probe: n={n}, iterations={iterations}");
    for case in cases {
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(case.label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(case.entry),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(case.label),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: states[case.state].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gather.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let encode = || {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                for _ in 0..iterations {
                    pass.dispatch_workgroups(groups.min(65_535), groups.div_ceil(65_535), 1);
                }
            }
            encoder.finish()
        };
        wait(&ctx, ctx.queue.submit(Some(encode())));
        let start = Instant::now();
        wait(&ctx, ctx.queue.submit(Some(encode())));
        let elapsed = start.elapsed() / iterations;
        let useful = (case.useful_read_bytes + 4) as f64 * f64::from(n);
        let useful_gbps = useful / elapsed.as_secs_f64() / 1.0e9;
        println!(
            "{:<28} {:>8.3} ms  {:>7.1} GB/s useful",
            case.label,
            elapsed.as_secs_f64() * 1.0e3,
            useful_gbps
        );
    }
}
