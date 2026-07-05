//! EVIDENCE probe for the GPU zero-flux BDF2 tolerance gate
//! (tests/ale_zero_flux_equivalence_test.rs): the moving-volume BDF2 ddt of
//! the ALE kernels — with the volume-ratio weights pinned to EXACT 1.0 by the
//! `select(vol_old/vol, 1.0, vol_old == vol)` guard — still compiles to
//! bitwise-different arithmetic than the static ddt on Metal, because the
//! shader compiler's fast-math reassociation of the (textually different) rhs
//! chains rounds differently. This kernel reproduces the generated static and
//! ALE ddt blocks verbatim on equal volume histories and measures the
//! difference: ~23% of lanes differ by exactly 1 ulp (maxd = 2.4e-7 on rhs
//! values of order 1-10, Apple M-series).
//!
//! Contract: target byte-identical; if reassociation genuinely prevents it,
//! gate at <= 1 ulp. Consequences:
//!   * CPU zero-flux equivalence stays BITWISE (strict IEEE IR/transpiled Rust).
//!   * GPU/Euler zero-flux equivalence stays BITWISE (the Euler path's ALE
//!     deltas are IEEE identities the compiler cannot reassociate away).
//!   * GPU/BDF2 zero-flux equivalence is gated at a small tolerance (the
//!     1-ulp/step assembly difference amplifies through the nonlinear solve
//!     over 20 steps to ~6e-5).
//! This test asserts the per-evaluation difference stays at the <= 1-ulp
//! scale; if it grows beyond that, the reassociation excuse no longer holds
//! and the ALE ddt emission must be revisited. The companion
//! `transcription_matches_generated_ddt_shape` test re-derives the key
//! transcribed lines from the committed generated WGSL so any ddt-shape change
//! fails here too, prompting a transcription refresh.
#![cfg(feature = "meshgen")]

const WGSL: &str = r#"
struct Constants {
    dt: f32,
    dt_old: f32,
    density: f32,
    time_scheme: u32,
}
@group(0) @binding(0) var<storage, read> vols: array<f32>;
@group(0) @binding(1) var<storage, read> vols_old: array<f32>;
@group(0) @binding(2) var<storage, read> vols_old_old: array<f32>;
@group(0) @binding(3) var<storage, read> so: array<f32>;
@group(0) @binding(4) var<storage, read> soo: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_static: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_ale: array<f32>;
@group(0) @binding(7) var<uniform> constants: Constants;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&vols)) { return; }
    let vol = vols[idx];

    // ---- static path (verbatim shape of the generated static ddt) ----
    var s_diag_0: f32 = 0.0;
    var s_rhs_0: f32 = 0.0;
    s_diag_0 += vol * constants.density / constants.dt;
    s_rhs_0 += vol * constants.density / constants.dt * so[idx];
    if (constants.time_scheme == 1u) {
        let r = constants.dt / constants.dt_old;
        let diag_bdf2 = vol * constants.density / constants.dt * (r * 2.0 + 1.0) / (r + 1.0);
        let factor_n = r + 1.0;
        let factor_nm1 = r * r / (r + 1.0);
        s_diag_0 = s_diag_0 - vol * constants.density / constants.dt + diag_bdf2;
        s_rhs_0 = s_rhs_0 - vol * constants.density / constants.dt * so[idx] + vol * constants.density / constants.dt * (factor_n * so[idx] - factor_nm1 * soo[idx]);
    }

    // ---- ALE path (verbatim shape of the generated ALE ddt) ----
    let vol_old = vols_old[idx];
    let vol_old_old = vols_old_old[idx];
    let ale_vol_ratio_n = select(vol_old / vol, 1.0, vol_old == vol);
    let ale_vol_ratio_nm1 = select(vol_old_old / vol, 1.0, vol_old_old == vol);
    let ale_dvdt_scl = (vol - vol_old) / constants.dt;
    var ale_dvdt_ddt: f32 = ale_dvdt_scl;
    if (constants.time_scheme == 1u) {
        let r_ale = constants.dt / constants.dt_old;
        ale_dvdt_ddt = ((r_ale * 2.0 + 1.0) / (r_ale + 1.0) * (vol - vol_old) - r_ale * r_ale / (r_ale + 1.0) * (vol_old - vol_old_old)) / constants.dt;
    }
    var a_diag_0: f32 = 0.0;
    var a_rhs_0: f32 = 0.0;
    a_diag_0 += vol * constants.density / constants.dt;
    a_rhs_0 += vol * constants.density / constants.dt * ale_vol_ratio_n * so[idx];
    if (constants.time_scheme == 1u) {
        let r = constants.dt / constants.dt_old;
        let diag_bdf2 = vol * constants.density / constants.dt * (r * 2.0 + 1.0) / (r + 1.0);
        let factor_n = r + 1.0;
        let factor_nm1 = r * r / (r + 1.0);
        a_diag_0 = a_diag_0 - vol * constants.density / constants.dt + diag_bdf2;
        a_rhs_0 = a_rhs_0 - vol * constants.density / constants.dt * ale_vol_ratio_n * so[idx] + vol * constants.density / constants.dt * (factor_n * ale_vol_ratio_n * so[idx] - factor_nm1 * ale_vol_ratio_nm1 * soo[idx]);
    }
    var a_bounded: f32 = 0.0;
    a_bounded += constants.density * ale_dvdt_ddt;
    a_diag_0 -= a_bounded;
    var s_bounded: f32 = 0.0;
    s_diag_0 -= s_bounded;

    out_static[idx] = s_rhs_0 + 1.7 * s_diag_0;
    out_ale[idx] = a_rhs_0 + 1.7 * a_diag_0;
}
"#;

/// Freshness pin for the hand-transcribed kernel above: the load-bearing ddt
/// lines must still appear VERBATIM (modulo the state-array -> probe-buffer
/// renames `state_old[idx * 8u + Cu]` -> `so[idx]` / `state_old_old[...]` ->
/// `soo[idx]`) in the committed generated WGSL. Runs without a GPU adapter.
#[test]
fn transcription_matches_generated_ddt_shape() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/solver/gpu/shaders/generated");
    let static_wgsl = std::fs::read_to_string(
        root.join("generic_coupled_assembly_incompressible_momentum.wgsl"),
    )
    .expect("read static generated WGSL");
    let ale_wgsl = std::fs::read_to_string(
        root.join("generic_coupled_assembly_incompressible_momentum_ale.wgsl"),
    )
    .expect("read ALE generated WGSL");

    // The transcription's probe-buffer lines, mapped back to the generated
    // state-array form. Each must appear verbatim in the generated file, or
    // the evidence kernel is testing a stale ddt shape.
    let static_expected = [
        "let diag_bdf2 = vol * constants.density / constants.dt * (r * 2.0 + 1.0) / (r + 1.0);",
        "rhs_0 = rhs_0 - vol * constants.density / constants.dt * state_old[idx * 8u + 0u] + \
         vol * constants.density / constants.dt * (factor_n * state_old[idx * 8u + 0u] - \
         factor_nm1 * state_old_old[idx * 8u + 0u]);",
    ];
    let ale_expected = [
        "let ale_vol_ratio_n = select(vol_old / vol, 1.0, vol_old == vol);",
        "let ale_vol_ratio_nm1 = select(vol_old_old / vol, 1.0, vol_old_old == vol);",
        "let ale_dvdt_scl = (vol - vol_old) / constants.dt;",
        "ale_dvdt_ddt = ((r_ale * 2.0 + 1.0) / (r_ale + 1.0) * (vol - vol_old) - r_ale * \
         r_ale / (r_ale + 1.0) * (vol_old - vol_old_old)) / constants.dt;",
        "rhs_0 = rhs_0 - vol * constants.density / constants.dt * ale_vol_ratio_n * \
         state_old[idx * 8u + 0u] + vol * constants.density / constants.dt * (factor_n * \
         ale_vol_ratio_n * state_old[idx * 8u + 0u] - factor_nm1 * ale_vol_ratio_nm1 * \
         state_old_old[idx * 8u + 0u]);",
    ];
    for line in static_expected {
        assert!(
            static_wgsl.contains(line),
            "generated STATIC ddt shape drifted from the hand-transcribed evidence kernel; \
             refresh the WGSL transcription in this file. Missing line:\n{line}"
        );
    }
    for line in ale_expected {
        assert!(
            ale_wgsl.contains(line),
            "generated ALE ddt shape drifted from the hand-transcribed evidence kernel; \
             refresh the WGSL transcription in this file. Missing line:\n{line}"
        );
    }
    println!("[ale-fastmath-evidence] transcription matches the generated ddt shape");
}

#[test]
fn gpu_bdf2_ddt_reassociation_at_ulp_scale() {
    let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[ale-fastmath-evidence] no GPU adapter ({e}); skipping");
            return;
        }
    };
    let device = &ctx.device;
    let queue = &ctx.queue;

    let n = 4096usize;
    let vols: Vec<f32> = (0..n).map(|i| 0.001 + (i as f32) * 1.7e-7).collect();
    let so: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 3.3).collect();
    let soo: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 2.1).collect();
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Constants { dt: f32, dt_old: f32, density: f32, time_scheme: u32 }
    let consts = Constants { dt: 0.005, dt_old: 0.005, density: 1.0, time_scheme: 1 };

    use wgpu::util::DeviceExt;
    let b_vols = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vols),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_vols_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vols),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_so = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&so),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_vols_old_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vols),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_soo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&soo),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_consts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&consts),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mk_out = || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let b_out_static = mk_out();
    let b_out_ale = mk_out();

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ratio probe"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b_vols.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_vols_old.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_vols_old_old.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_so.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: b_soo.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: b_out_static.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: b_out_ale.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: b_consts.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&b_out_static, 0, &staging, 0, (n * 4) as u64);
    enc.copy_buffer_to_buffer(&b_out_ale, 0, &staging, (n * 4) as u64, (n * 4) as u64);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let data = slice.get_mapped_range();
    let all: &[f32] = bytemuck::cast_slice(&data);
    let (a, b) = all.split_at(n);
    let ndiff = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let maxd = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "[ale-fastmath-evidence] BDF2 ddt static-vs-ALE at equal volume history: \
         ndiff = {ndiff}/{n}, maxd = {maxd:.3e}"
    );
    // rhs values here are O(1..10); 1 ulp is ~2.4e-7 rel. If this grows past
    // the ulp scale, the GPU/BDF2 tolerance gate's justification is void.
    assert!(
        maxd <= 1.5e-6,
        "static-vs-ALE ddt difference {maxd:.3e} exceeds the 1-ulp-scale bound \
         the GPU/BDF2 zero-flux tolerance gate is justified by"
    );
}
