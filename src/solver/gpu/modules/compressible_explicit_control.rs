//! Autonomous GPU health and timestep control for the production,
//! density-based ideal-gas compressible RK4 solver.
//!
//! This module is intentionally separate from the pressure-based all-Mach
//! controller. Its admissibility test is reconstructed exclusively from the
//! conserved variables and its adaptive policy is the density-based host
//! wave/diffusion oracle.

use super::explicit_control::{ExplicitControlCapabilities, ExplicitControlStatus};
use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::structs::GpuConstants;
use crate::solver::model::ModelSpec;
use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub(crate) const CELLS_INDIRECT_OFFSET: u64 = 0;
pub(crate) const FACES_INDIRECT_OFFSET: u64 = 3 * 4;
const ROLLBACK_INDIRECT_OFFSET: u64 = 6 * 4;
const STATUS_RING_SLOTS: usize = 3;
const Q64_HI_SCALE: f64 = 1.0 / 4_294_967_296.0;
const Q64_LO_SCALE: f64 = 1.0 / 18_446_744_073_709_551_616.0;

fn split_q64_time(time: f64) -> (u32, u32, u32) {
    if !time.is_finite() || time <= 0.0 {
        return (0, 0, 0);
    }
    if time >= u32::MAX as f64 {
        return (u32::MAX, 0, 0);
    }
    let mut seconds = time.floor() as u32;
    let scaled_hi = (time - f64::from(seconds)) * 4_294_967_296.0;
    let mut hi = scaled_hi.floor() as u64;
    let mut lo = ((scaled_hi - hi as f64) * 4_294_967_296.0).round() as u64;
    if lo == (1_u64 << 32) {
        lo = 0;
        hi += 1;
    }
    if hi == (1_u64 << 32) {
        hi = 0;
        seconds = seconds.saturating_add(1);
    }
    (seconds, hi as u32, lo as u32)
}

pub(crate) struct CompressibleStatusStagingLease {
    buffer: wgpu::Buffer,
    slot: usize,
    busy: Arc<[AtomicBool; STATUS_RING_SLOTS]>,
}

impl CompressibleStatusStagingLease {
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for CompressibleStatusStagingLease {
    fn drop(&mut self) {
        self.busy[self.slot].store(false, Ordering::Release);
    }
}

/// Byte-for-byte common autonomous-controller status ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuExplicitControl {
    time_seconds: u32,
    time_fraction_hi: u32,
    time_fraction_lo: u32,
    next_dt: f32,
    step_dt: f32,
    dt_old: f32,
    target_cfl: f32,
    safety: f32,
    growth: f32,
    max_dt: f32,
    accepted_total_lo: u32,
    accepted_total_hi: u32,
    accepted_batch: u32,
    halt: u32,
    invalid_count: u32,
    max_base_bits: u32,
    max_turnover_bits: u32,
    last_dt: f32,
    total_rate: f32,
    fail_after_accepted: u32,
    cells_x: u32,
    cells_y: u32,
    faces_x: u32,
    faces_y: u32,
    adaptive: u32,
    initial_phase: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuCompressibleParams {
    num_cells: u32,
    num_faces: u32,
    state_stride: u32,
    dispatch_stride: u32,
    rho_off: u32,
    rho_u_off: u32,
    rho_e_off: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuCompressibleMetrics {
    max_velocity_bits: u32,
    max_characteristic_speed_bits: u32,
    max_inv_h_bits: u32,
}

struct CompressibleFrameCopyTargets {
    destinations: [wgpu::Buffer; 3],
    bind_groups: [wgpu::BindGroup; 3],
}

pub(crate) struct CompressibleExplicitControl {
    initialized: AtomicBool,
    eos_supported: AtomicBool,
    adaptive_enabled: AtomicBool,
    bootstrap_pending: AtomicBool,
    volume_audit_pending: AtomicBool,
    status_faulted: Arc<AtomicBool>,
    status_staging: [wgpu::Buffer; STATUS_RING_SLOTS],
    status_slots_busy: Arc<[AtomicBool; STATUS_RING_SLOTS]>,
    next_status_slot: AtomicUsize,
    initial_phase: AtomicUsize,
    test_fail_after_accepted: AtomicU32,
    initial_step_count: u64,
    b_control: wgpu::Buffer,
    b_params: wgpu::Buffer,
    b_history_backup: wgpu::Buffer,
    pub(crate) b_indirect_args: Arc<wgpu::Buffer>,
    control_bg: wgpu::BindGroup,
    phase_bgs: [wgpu::BindGroup; 3],
    stage_pipelines: [wgpu::ComputePipeline; 4],
    batch_reset_pipeline: wgpu::ComputePipeline,
    volume_audit_pipeline: wgpu::ComputePipeline,
    volume_finalize_pipeline: wgpu::ComputePipeline,
    history_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,
    audit_pipeline: wgpu::ComputePipeline,
    bootstrap_audit_pipeline: wgpu::ComputePipeline,
    bootstrap_finalize_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    rollback_pipeline: wgpu::ComputePipeline,
    frame_copy_layout: wgpu::BindGroupLayout,
    frame_copy_pipeline: wgpu::ComputePipeline,
    frame_copy_targets: Option<CompressibleFrameCopyTargets>,
    state_buffers: [wgpu::Buffer; 3],
    cells_dispatch: (u32, u32),
    faces_dispatch: (u32, u32),
    frame_copy_dispatch: (u32, u32),
    state_size_bytes: u64,
}

fn ideal_gas_constants_supported(eos: &GpuConstants) -> bool {
    eos.eos_gamma.is_finite()
        && eos.eos_gamma > 1.0
        && eos.eos_gm1.is_finite()
        && eos.eos_gm1 > 0.0
        && eos.eos_r.is_finite()
        && eos.eos_r > 0.0
        && eos.eos_theta_ref.is_finite()
        && eos.eos_theta_ref > 0.0
        && eos.eos_dp_drho == 0.0
        && eos.eos_p_ref == 0.0
        && eos.eos_rho_ref == 0.0
}

fn observe_eos_support(
    initialized: &AtomicBool,
    previous_support: &AtomicBool,
    eos: &GpuConstants,
) -> bool {
    let supported = ideal_gas_constants_supported(eos);
    if previous_support.swap(supported, Ordering::AcqRel) != supported {
        // The controller's clock and candidate dt belong to the old physical
        // closure.  Re-seed both from the reconciled host shadows if this EOS
        // becomes eligible again; never carry a stale ideal-gas dt through an
        // unsupported interval.
        initialized.store(false, Ordering::Release);
    }
    supported
}

impl CompressibleExplicitControl {
    pub(crate) const fn capabilities(&self) -> ExplicitControlCapabilities {
        ExplicitControlCapabilities {
            fixed_health: true,
            adaptive_health: true,
            accepted_state_copy: true,
        }
    }

    /// Whether the live constants still describe the deliberately supported
    /// calorically-perfect ideal-gas closure. This is intentionally evaluated
    /// at capability/submission time rather than construction time: GUI EOS
    /// changes do not rebuild the solver.
    pub(crate) fn supports_constants(&self, constants: &GpuConstants) -> bool {
        observe_eos_support(&self.initialized, &self.eos_supported, constants)
    }

    pub(crate) fn capabilities_for_constants(
        &self,
        constants: &GpuConstants,
    ) -> Option<ExplicitControlCapabilities> {
        self.supports_constants(constants)
            .then(|| self.capabilities())
    }

    /// Returns None for every model outside the production density-based route.
    /// EOS eligibility is live policy (see [`Self::supports_constants`]), not a
    /// construction-time decision, because the GUI can change fluids in place.
    pub(crate) fn new(
        device: &wgpu::Device,
        mesh: &MeshResources,
        fields: &UnifiedFieldResources,
        _bc_kind: &wgpu::Buffer,
        _bc_value: &wgpu::Buffer,
        model: &ModelSpec,
    ) -> Result<Option<Self>, String> {
        if model.id != "compressible" || model.system.is_ale() {
            return Ok(None);
        }

        // Allocate the route independently of the current runtime EOS. The
        // live-EOS gate above decides whether a batch may be submitted, which
        // permits linear -> ideal without rebuilding while ideal -> linear
        // cleanly falls back before phase/controller mutation.
        let initial_eos_supported = ideal_gas_constants_supported(fields.constants.values());

        let offset = |name: &str| {
            model.state_layout.offset_for(name).ok_or_else(|| {
                format!("autonomous compressible controller missing state field '{name}'")
            })
        };
        let rho_off = offset("rho")?;
        let rho_u_off = offset("rho_u")?;
        let rho_e_off = offset("rho_e")?;
        if rho_off >= fields.state_stride
            || rho_u_off.checked_add(1).is_none_or(|y| y >= fields.state_stride)
            || rho_e_off >= fields.state_stride
        {
            return Err("autonomous compressible conserved-state offsets exceed stride".to_string());
        }

        let max_groups = device.limits().max_compute_workgroups_per_dimension;
        let dispatch = |items: u32| {
            let groups = items.div_ceil(64);
            if groups <= max_groups {
                (groups, 1)
            } else {
                (max_groups, groups.div_ceil(max_groups))
            }
        };
        let cells_dispatch = dispatch(fields.num_cells);
        let faces_dispatch = dispatch(fields.num_faces);
        let state_words = fields
            .num_cells
            .checked_mul(fields.state_stride)
            .ok_or_else(|| "autonomous compressible frame size overflows u32".to_string())?;
        let frame_copy_dispatch = dispatch(state_words);
        let params = GpuCompressibleParams {
            num_cells: fields.num_cells,
            num_faces: fields.num_faces,
            state_stride: fields.state_stride,
            dispatch_stride: max_groups.saturating_mul(64),
            rho_off,
            rho_u_off,
            rho_e_off,
            _pad: 0,
        };

        let b_control = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compressible_explicit_control:status"),
            size: std::mem::size_of::<GpuExplicitControl>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let status_staging = std::array::from_fn(|slot| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match slot {
                    0 => "compressible_explicit_control:status_ring_0",
                    1 => "compressible_explicit_control:status_ring_1",
                    _ => "compressible_explicit_control:status_ring_2",
                }),
                size: std::mem::size_of::<GpuExplicitControl>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let status_slots_busy = Arc::new(std::array::from_fn(|_| AtomicBool::new(false)));
        let b_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compressible_explicit_control:params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let b_metrics = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compressible_explicit_control:metrics"),
            contents: bytemuck::bytes_of(&GpuCompressibleMetrics::zeroed()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_history_backup = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compressible_explicit_control:history_backup"),
            size: fields.state_size_bytes().max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b_indirect_args = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compressible_explicit_control:indirect_args"),
            size: 9 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));

        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let control_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compressible_explicit_control:control_bgl"),
            entries: &[
                storage(0, false),
                storage(1, false),
                storage(2, false),
                uniform(3),
                storage(4, false),
            ],
        });
        let phase_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compressible_explicit_control:phase_bgl"),
            entries: &[
                storage(0, false),
                storage(1, true),
                storage(2, true),
                storage(3, false),
            ],
        });
        let frame_copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compressible_explicit_control:frame_copy_bgl"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                uniform(5),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compressible_explicit_control:pipeline_layout"),
            bind_group_layouts: &[&control_bgl, &phase_bgl],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compressible_explicit_control:shader"),
            source: wgpu::ShaderSource::Wgsl(COMPRESSIBLE_EXPLICIT_CONTROL_WGSL.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let stage_pipelines = [
            pipeline("stage_1"),
            pipeline("stage_2"),
            pipeline("stage_3"),
            pipeline("stage_4"),
        ];
        let batch_reset_pipeline = pipeline("batch_reset");
        let volume_audit_pipeline = pipeline("volume_audit");
        let volume_finalize_pipeline = pipeline("volume_finalize");
        let history_pipeline = pipeline("history_prepare");
        let clear_pipeline = pipeline("health_clear");
        let audit_pipeline = pipeline("audit_state");
        let bootstrap_audit_pipeline = pipeline("audit_accepted_state");
        let bootstrap_finalize_pipeline = pipeline("bootstrap_finalize");
        let finalize_pipeline = pipeline("health_finalize");
        let rollback_pipeline = pipeline("rollback_invalid");

        let frame_copy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compressible_explicit_control:frame_copy_pipeline_layout"),
                bind_group_layouts: &[&frame_copy_layout],
                push_constant_ranges: &[],
            });
        let frame_copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compressible_explicit_control:frame_copy_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPRESSIBLE_EXPLICIT_FRAME_COPY_WGSL.into()),
        });
        let frame_copy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compressible_explicit_control:frame_copy_pipeline"),
                layout: Some(&frame_copy_pipeline_layout),
                module: &frame_copy_shader,
                entry_point: Some("copy_accepted_state"),
                compilation_options: Default::default(),
                cache: None,
            });

        let control_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compressible_explicit_control:control_bg"),
            layout: &control_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_control.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fields.constants.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_indirect_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_metrics.as_entire_binding(),
                },
            ],
        });
        let phase_bgs = std::array::from_fn(|phase| {
            let (current, old, _) =
                crate::solver::gpu::modules::state::ping_pong_indices(phase);
            let buffers = fields.state_buffers();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compressible_explicit_control:phase_bg"),
                layout: &phase_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers[current].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers[old].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: mesh.b_cell_vols.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_history_backup.as_entire_binding(),
                    },
                ],
            })
        });

        Ok(Some(Self {
            initialized: AtomicBool::new(false),
            eos_supported: AtomicBool::new(initial_eos_supported),
            adaptive_enabled: AtomicBool::new(false),
            bootstrap_pending: AtomicBool::new(false),
            volume_audit_pending: AtomicBool::new(false),
            status_faulted: Arc::new(AtomicBool::new(false)),
            status_staging,
            status_slots_busy,
            next_status_slot: AtomicUsize::new(0),
            initial_phase: AtomicUsize::new(0),
            test_fail_after_accepted: AtomicU32::new(u32::MAX),
            initial_step_count: 0,
            b_control,
            b_params,
            b_history_backup,
            b_indirect_args,
            control_bg,
            phase_bgs,
            stage_pipelines,
            batch_reset_pipeline,
            volume_audit_pipeline,
            volume_finalize_pipeline,
            history_pipeline,
            clear_pipeline,
            audit_pipeline,
            bootstrap_audit_pipeline,
            bootstrap_finalize_pipeline,
            finalize_pipeline,
            rollback_pipeline,
            frame_copy_layout,
            frame_copy_pipeline,
            frame_copy_targets: None,
            state_buffers: std::array::from_fn(|index| fields.state_buffers()[index].clone()),
            cells_dispatch,
            faces_dispatch,
            frame_copy_dispatch,
            state_size_bytes: fields.state_size_bytes(),
        }))
    }

    pub(crate) fn invalidate(&self) {
        self.initialized.store(false, Ordering::Release);
    }

    pub(crate) fn configure(
        &mut self,
        queue: &wgpu::Queue,
        time: f64,
        dt: f32,
        dt_old: f32,
        step_count: u64,
        initial_phase: usize,
        target_cfl: f32,
        adaptive: bool,
        dt_is_accepted_candidate: bool,
    ) {
        let target_cfl = if target_cfl.is_finite() && target_cfl > 0.0 {
            target_cfl
        } else {
            1.0e-6
        };
        let newly_initialized = !self.initialized.swap(true, Ordering::AcqRel);
        if newly_initialized {
            // Cell volumes are immutable on this static route. Audit them once
            // per controller initialization/topology refresh, queue-ordered
            // ahead of stage_1.
            self.volume_audit_pending.store(true, Ordering::Release);
            self.initial_step_count = step_count;
            self.initial_phase
                .store(initial_phase % 3, Ordering::Release);
            let (seconds, fraction_hi, fraction_lo) = split_q64_time(time);
            let value = GpuExplicitControl {
                time_seconds: seconds,
                time_fraction_hi: fraction_hi,
                time_fraction_lo: fraction_lo,
                next_dt: dt,
                step_dt: dt,
                dt_old,
                target_cfl,
                safety: 1.0,
                growth: 1.2,
                max_dt: 100.0,
                accepted_total_lo: 0,
                accepted_total_hi: 0,
                accepted_batch: 0,
                halt: 0,
                invalid_count: 0,
                max_base_bits: 0,
                max_turnover_bits: 0,
                last_dt: 0.0,
                total_rate: 0.0,
                fail_after_accepted: self.test_fail_after_accepted.load(Ordering::Acquire),
                cells_x: self.cells_dispatch.0,
                cells_y: self.cells_dispatch.1,
                faces_x: self.faces_dispatch.0,
                faces_y: self.faces_dispatch.1,
                adaptive: u32::from(adaptive),
                initial_phase: (initial_phase % 3) as u32,
            };
            queue.write_buffer(&self.b_control, 0, bytemuck::bytes_of(&value));
        } else {
            queue.write_buffer(
                &self.b_control,
                std::mem::offset_of!(GpuExplicitControl, target_cfl) as u64,
                bytemuck::bytes_of(&target_cfl),
            );
            let adaptive = u32::from(adaptive);
            queue.write_buffer(
                &self.b_control,
                std::mem::offset_of!(GpuExplicitControl, adaptive) as u64,
                bytemuck::bytes_of(&adaptive),
            );
            if adaptive == 0 {
                queue.write_buffer(
                    &self.b_control,
                    std::mem::offset_of!(GpuExplicitControl, next_dt) as u64,
                    bytemuck::bytes_of(&dt),
                );
            }
        }
        let was_adaptive = self.adaptive_enabled.swap(adaptive, Ordering::AcqRel);
        if adaptive && (!was_adaptive || newly_initialized) && !dt_is_accepted_candidate {
            // The first adaptive candidate must be sized from the accepted
            // conserved state when the host supplied only a raw requested
            // seed. A driver-sampled/reconciled candidate already represents
            // that same state and must not receive the growth limiter twice.
            // A raw freshly re-seeded controller also covers a live EOS
            // transition back into the supported ideal-gas family.
            self.bootstrap_pending.store(true, Ordering::Release);
        } else if !adaptive || dt_is_accepted_candidate {
            self.bootstrap_pending.store(false, Ordering::Release);
        }
    }

    pub(crate) fn encode_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: usize,
        phase: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compressible_explicit_control:stage"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.stage_pipelines[stage]);
        pass.set_bind_group(0, &self.control_bg, &[]);
        pass.set_bind_group(1, &self.phase_bgs[phase % 3], &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub(crate) fn encode_batch_reset(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) {
        let bg = &self.phase_bgs[phase % 3];
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compressible_explicit_control:batch_reset"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.batch_reset_pipeline);
            pass.set_bind_group(0, &self.control_bg, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let bootstrap_pending = self.bootstrap_pending.swap(false, Ordering::AcqRel);
        let volume_audit_pending = self.volume_audit_pending.swap(false, Ordering::AcqRel);
        if bootstrap_pending {
            // `phase` is the first destination phase. Its state_old binding is
            // therefore the authoritative accepted state. Audit that state and
            // select the wave+diffusion CFL before stage_1 snapshots next_dt.
            // This audit includes the raw-volume invariant, so it also consumes
            // the one-time geometry audit without a duplicate cell pass.
            for (pipeline, x, y, label) in [
                (&self.clear_pipeline, 1, 1, "bootstrap_clear"),
                (
                    &self.bootstrap_audit_pipeline,
                    self.cells_dispatch.0,
                    self.cells_dispatch.1,
                    "bootstrap_audit",
                ),
                (
                    &self.bootstrap_finalize_pipeline,
                    1,
                    1,
                    "bootstrap_finalize",
                ),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.control_bg, &[]);
                pass.set_bind_group(1, bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        } else if volume_audit_pending {
            // Fixed mode does not need the full thermodynamic CFL bootstrap,
            // but geometry must still halt the indirect dispatch before the
            // first RK stage.
            for (pipeline, x, y, label) in [
                (
                    &self.volume_audit_pipeline,
                    self.cells_dispatch.0,
                    self.cells_dispatch.1,
                    "compressible_explicit_control:volume_audit",
                ),
                (
                    &self.volume_finalize_pipeline,
                    1,
                    1,
                    "compressible_explicit_control:volume_finalize",
                ),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.control_bg, &[]);
                pass.set_bind_group(1, bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
    }

    pub(crate) fn encode_history_prepare(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compressible_explicit_control:history_prepare"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.history_pipeline);
        pass.set_bind_group(0, &self.control_bg, &[]);
        pass.set_bind_group(1, &self.phase_bgs[phase % 3], &[]);
        pass.dispatch_workgroups_indirect(&self.b_indirect_args, CELLS_INDIRECT_OFFSET);
    }

    pub(crate) fn encode_health(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) {
        let bg = &self.phase_bgs[phase % 3];
        let mut run = |pipeline: &wgpu::ComputePipeline, x: u32, y: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compressible_explicit_control:health"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.control_bg, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.dispatch_workgroups(x, y, 1);
        };
        run(&self.clear_pipeline, 1, 1);
        run(
            &self.audit_pipeline,
            self.cells_dispatch.0,
            self.cells_dispatch.1,
        );
        run(&self.finalize_pipeline, 1, 1);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compressible_explicit_control:rollback"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.rollback_pipeline);
        pass.set_bind_group(0, &self.control_bg, &[]);
        pass.set_bind_group(1, bg, &[]);
        pass.dispatch_workgroups_indirect(&self.b_indirect_args, ROLLBACK_INDIRECT_OFFSET);
    }

    pub(crate) fn status_buffer(&self) -> &wgpu::Buffer {
        &self.b_control
    }

    pub(crate) fn ensure_status_recoverable(&self) -> Result<(), String> {
        if self.status_faulted.load(Ordering::Acquire) {
            Err(
                "autonomous compressible controller is non-resumable after a status map/decode failure; rebuild the solver"
                    .to_string(),
            )
        } else {
            Ok(())
        }
    }

    pub(crate) fn status_fault_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.status_faulted)
    }

    pub(crate) fn acquire_status_staging(
        &self,
    ) -> Result<CompressibleStatusStagingLease, String> {
        let start = self.next_status_slot.fetch_add(1, Ordering::Relaxed) % STATUS_RING_SLOTS;
        for offset in 0..STATUS_RING_SLOTS {
            let slot = (start + offset) % STATUS_RING_SLOTS;
            if self.status_slots_busy[slot]
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(CompressibleStatusStagingLease {
                    buffer: self.status_staging[slot].clone(),
                    slot,
                    busy: Arc::clone(&self.status_slots_busy),
                });
            }
        }
        Err(
            "all three autonomous compressible status staging slots are busy; poll GPU completions before submitting another batch"
                .to_string(),
        )
    }

    pub(crate) fn status_size(&self) -> u64 {
        std::mem::size_of::<GpuExplicitControl>() as u64
    }

    pub(crate) fn encode_status_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::Buffer,
    ) {
        encoder.copy_buffer_to_buffer(&self.b_control, 0, destination, 0, self.status_size());
    }

    pub(crate) fn install_frame_copy_targets(
        &mut self,
        device: &wgpu::Device,
        destinations: &[wgpu::Buffer; 3],
    ) -> Result<(), String> {
        for (slot, destination) in destinations.iter().enumerate() {
            if destination.size() < self.state_size_bytes {
                return Err(format!(
                    "autonomous compressible frame destination {slot} is {} bytes, needs {}",
                    destination.size(),
                    self.state_size_bytes
                ));
            }
            if !destination.usage().contains(wgpu::BufferUsages::STORAGE) {
                return Err(format!(
                    "autonomous compressible frame destination {slot} lacks STORAGE usage"
                ));
            }
        }
        let bind_groups = std::array::from_fn(|slot| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compressible_explicit_control:frame_copy_bg"),
                layout: &self.frame_copy_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.b_control.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.state_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.state_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.state_buffers[2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: destinations[slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.b_params.as_entire_binding(),
                    },
                ],
            })
        });
        self.frame_copy_targets = Some(CompressibleFrameCopyTargets {
            destinations: destinations.clone(),
            bind_groups,
        });
        Ok(())
    }

    pub(crate) fn frame_copy_targets(&self) -> Option<[wgpu::Buffer; 3]> {
        self.frame_copy_targets
            .as_ref()
            .map(|targets| targets.destinations.clone())
    }

    pub(crate) fn encode_accepted_state_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_slot: usize,
    ) -> Result<(), String> {
        let targets = self.frame_copy_targets.as_ref().ok_or_else(|| {
            "autonomous compressible frame-copy targets were not installed for this solver"
                .to_string()
        })?;
        let bind_group = targets.bind_groups.get(target_slot).ok_or_else(|| {
            format!("autonomous compressible frame-copy slot {target_slot} is outside 0..3")
        })?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compressible_explicit_control:copy_accepted_state"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.frame_copy_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(self.frame_copy_dispatch.0, self.frame_copy_dispatch.1, 1);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn cached_frame_copy_bind_group_count(&self) -> usize {
        self.frame_copy_targets.as_ref().map_or(0, |_| 3)
    }

    pub(crate) fn decode_status(bytes: &[u8]) -> Result<ExplicitControlStatus, String> {
        let size = std::mem::size_of::<GpuExplicitControl>();
        if bytes.len() < size {
            return Err(format!(
                "compressible explicit control status is {} bytes, expected {size}",
                bytes.len()
            ));
        }
        let value = *bytemuck::from_bytes::<GpuExplicitControl>(&bytes[..size]);
        let fraction = value.time_fraction_hi as f64 * Q64_HI_SCALE
            + value.time_fraction_lo as f64 * Q64_LO_SCALE;
        Ok(ExplicitControlStatus {
            time: value.time_seconds as f64 + fraction,
            next_dt: value.next_dt,
            last_dt: value.last_dt,
            accepted_total: (u64::from(value.accepted_total_hi) << 32)
                | u64::from(value.accepted_total_lo),
            accepted_batch: value.accepted_batch,
            halted: value.halt != 0,
            invalid_count: value.invalid_count,
            max_base_rate: f32::from_bits(value.max_base_bits),
            max_rhie_chow_turnover: f32::from_bits(value.max_turnover_bits),
            total_rate: value.total_rate,
        })
    }

    pub(crate) fn initial_step_count(&self) -> u64 {
        self.initial_step_count
    }

    pub(crate) fn accepted_phase(&self, accepted_total: u64) -> usize {
        let lo = accepted_total as u32;
        let hi = (accepted_total >> 32) as u32;
        let total_mod3 = (lo % 3 + hi % 3) % 3;
        (self.initial_phase.load(Ordering::Acquire) + total_mod3 as usize) % 3
    }

    #[cfg(test)]
    pub(crate) fn inject_failure_after(&self, accepted_steps: u32) {
        self.test_fail_after_accepted
            .store(accepted_steps, Ordering::Release);
        self.invalidate();
    }

    #[cfg(test)]
    pub(crate) fn inject_accepted_total(&self, queue: &wgpu::Queue, accepted_total: u64) {
        let accepted_total_lo = accepted_total as u32;
        let accepted_total_hi = (accepted_total >> 32) as u32;
        queue.write_buffer(
            &self.b_control,
            std::mem::offset_of!(GpuExplicitControl, accepted_total_lo) as u64,
            bytemuck::bytes_of(&accepted_total_lo),
        );
        queue.write_buffer(
            &self.b_control,
            std::mem::offset_of!(GpuExplicitControl, accepted_total_hi) as u64,
            bytemuck::bytes_of(&accepted_total_hi),
        );
    }
}

const COMPRESSIBLE_EXPLICIT_FRAME_COPY_WGSL: &str = r#"
struct Control {
    time_seconds: u32, time_fraction_hi: u32, time_fraction_lo: u32,
    next_dt: f32, step_dt: f32, dt_old: f32,
    target_cfl: f32, safety: f32, growth: f32, max_dt: f32,
    accepted_total_lo: u32, accepted_total_hi: u32, accepted_batch: u32, halt: u32,
    invalid_count: atomic<u32>, max_base_bits: atomic<u32>,
    max_turnover_bits: atomic<u32>, last_dt: f32, total_rate: f32,
    fail_after_accepted: u32, cells_x: u32, cells_y: u32,
    faces_x: u32, faces_y: u32, adaptive: u32, initial_phase: u32,
};
struct ParamsPrefix {
    num_cells: u32, num_faces: u32, state_stride: u32, dispatch_stride: u32,
};
@group(0) @binding(0) var<storage, read> control: Control;
@group(0) @binding(1) var<storage, read> state_0: array<f32>;
@group(0) @binding(2) var<storage, read> state_1: array<f32>;
@group(0) @binding(3) var<storage, read> state_2: array<f32>;
@group(0) @binding(4) var<storage, read_write> destination: array<f32>;
@group(0) @binding(5) var<uniform> params: ParamsPrefix;

@compute @workgroup_size(64)
fn copy_accepted_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x + gid.y * params.dispatch_stride;
    let words = params.num_cells * params.state_stride;
    if index >= words || index >= arrayLength(&destination) { return; }
    let accepted_mod3 =
        ((control.accepted_total_lo % 3u) + (control.accepted_total_hi % 3u)) % 3u;
    let logical_phase = (control.initial_phase + accepted_mod3) % 3u;
    if logical_phase == 0u { destination[index] = state_0[index]; }
    else if logical_phase == 1u { destination[index] = state_2[index]; }
    else { destination[index] = state_1[index]; }
}
"#;

const COMPRESSIBLE_EXPLICIT_CONTROL_WGSL: &str = r#"
struct Control {
    time_seconds: u32, time_fraction_hi: u32, time_fraction_lo: u32,
    next_dt: f32, step_dt: f32, dt_old: f32,
    target_cfl: f32, safety: f32, growth: f32, max_dt: f32,
    accepted_total_lo: u32, accepted_total_hi: u32, accepted_batch: u32, halt: u32,
    invalid_count: atomic<u32>, max_base_bits: atomic<u32>,
    max_turnover_bits: atomic<u32>, last_dt: f32, total_rate: f32,
    fail_after_accepted: u32, cells_x: u32, cells_y: u32,
    faces_x: u32, faces_y: u32, adaptive: u32, initial_phase: u32,
};
struct Params {
    num_cells: u32, num_faces: u32, state_stride: u32, dispatch_stride: u32,
    rho_off: u32, rho_u_off: u32, rho_e_off: u32, pad: u32,
};
struct Metrics {
    max_velocity_bits: atomic<u32>,
    max_characteristic_speed_bits: atomic<u32>,
    max_inv_h_bits: atomic<u32>,
};
@group(0) @binding(0) var<storage, read_write> control: Control;
@group(0) @binding(1) var<storage, read_write> constants_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_args: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> metrics: Metrics;

@group(1) @binding(0) var<storage, read_write> state: array<f32>;
@group(1) @binding(1) var<storage, read> state_old: array<f32>;
@group(1) @binding(2) var<storage, read> cell_vols: array<f32>;
@group(1) @binding(3) var<storage, read_write> history_backup: array<f32>;

fn finite(x: f32) -> bool { return x == x && abs(x) <= 3.402823e38; }
fn idx2(gid: vec3<u32>) -> u32 { return gid.x + gid.y * params.dispatch_stride; }
fn state_value(cell: u32, off: u32, accepted: bool) -> f32 {
    let index = cell * params.state_stride + off;
    return select(state[index], state_old[index], accepted);
}

struct Clock { seconds: u32, fraction_hi: u32, fraction_lo: u32 };
fn clock_parts() -> Clock {
    return Clock(control.time_seconds, control.time_fraction_hi, control.time_fraction_lo);
}
fn highest_bit(v0: u32) -> u32 {
    var v = v0;
    var bit = 0u;
    if v >= 65536u { v >>= 16u; bit += 16u; }
    if v >= 256u { v >>= 8u; bit += 8u; }
    if v >= 16u { v >>= 4u; bit += 4u; }
    if v >= 4u { v >>= 2u; bit += 2u; }
    if v >= 2u { bit += 1u; }
    return bit;
}
fn clock_bit(c: Clock, bit: u32) -> u32 {
    if bit < 32u { return (c.fraction_lo >> bit) & 1u; }
    if bit < 64u { return (c.fraction_hi >> (bit - 32u)) & 1u; }
    return (c.seconds >> (bit - 64u)) & 1u;
}
fn low_mask(bits: u32) -> u32 { return (1u << bits) - 1u; }
fn any_bits_below(c: Clock, count: u32) -> bool {
    if count == 0u { return false; }
    if count < 32u { return (c.fraction_lo & low_mask(count)) != 0u; }
    if count == 32u { return c.fraction_lo != 0u; }
    if count < 64u {
        return c.fraction_lo != 0u
            || (c.fraction_hi & low_mask(count - 32u)) != 0u;
    }
    if count == 64u { return c.fraction_lo != 0u || c.fraction_hi != 0u; }
    return c.fraction_lo != 0u || c.fraction_hi != 0u
        || (c.seconds & low_mask(count - 64u)) != 0u;
}
fn clock_to_f32(c: Clock) -> f32 {
    if c.seconds == 0u && c.fraction_hi == 0u && c.fraction_lo == 0u {
        return 0.0;
    }
    var top = 0u;
    if c.seconds != 0u { top = 64u + highest_bit(c.seconds); }
    else if c.fraction_hi != 0u { top = 32u + highest_bit(c.fraction_hi); }
    else { top = highest_bit(c.fraction_lo); }
    var significand = 0u;
    var discarded = 0u;
    if top < 23u {
        significand = c.fraction_lo << (23u - top);
    } else {
        discarded = top - 23u;
        for (var k = 0u; k < 24u; k += 1u) {
            significand |= clock_bit(c, discarded + k) << k;
        }
    }
    var exponent = top + 63u;
    if discarded != 0u {
        let guard = clock_bit(c, discarded - 1u) != 0u;
        let sticky = any_bits_below(c, discarded - 1u);
        if guard && (sticky || (significand & 1u) != 0u) {
            significand += 1u;
            if significand >= 0x01000000u {
                significand >>= 1u;
                exponent += 1u;
            }
        }
    }
    return bitcast<f32>((exponent << 23u) | (significand & 0x007fffffu));
}
fn add_clock_parts(base: Clock, dt: f32) -> Clock {
    let bits = bitcast<u32>(dt);
    let raw_exp = (bits >> 23u) & 0xffu;
    let significand = (bits & 0x007fffffu) | 0x00800000u;
    var add_seconds = 0u;
    var add_hi = 0u;
    var add_lo = 0u;
    if (bits & 0x80000000u) == 0u && raw_exp != 0u && raw_exp < 0xffu {
        if raw_exp >= 86u {
            let shift = raw_exp - 86u;
            if shift < 32u {
                add_lo = significand << shift;
                if shift > 8u { add_hi = significand >> (32u - shift); }
            } else if shift < 64u {
                add_hi = significand << (shift - 32u);
                if shift > 40u { add_seconds = significand >> (64u - shift); }
            } else if shift <= 72u {
                add_seconds = significand << (shift - 64u);
            }
        } else {
            let right = 86u - raw_exp;
            if right < 24u {
                add_lo = significand >> right;
                let remainder = significand & low_mask(right);
                let halfway = 1u << (right - 1u);
                if remainder > halfway
                    || (remainder == halfway && (add_lo & 1u) != 0u) {
                    add_lo += 1u;
                }
            }
        }
    }
    let next_low = base.fraction_lo + add_lo;
    let carry_low = select(0u, 1u, next_low < base.fraction_lo);
    let high_partial = base.fraction_hi + add_hi;
    let carry_high0 = select(0u, 1u, high_partial < base.fraction_hi);
    let next_high = high_partial + carry_low;
    let carry_high1 = select(0u, 1u, next_high < high_partial);
    return Clock(
        base.seconds + add_seconds + carry_high0 + carry_high1,
        next_high,
        next_low,
    );
}
fn clock_seconds() -> f32 { return clock_to_f32(clock_parts()); }
fn clock_plus(dt: f32) -> f32 {
    return clock_to_f32(add_clock_parts(clock_parts(), dt));
}
fn add_clock(dt: f32) {
    let next = add_clock_parts(clock_parts(), dt);
    control.time_seconds = next.seconds;
    control.time_fraction_hi = next.fraction_hi;
    control.time_fraction_lo = next.fraction_lo;
}

fn set_active(enabled: bool) {
    if enabled {
        dispatch_args[0] = control.cells_x;
        dispatch_args[1] = control.cells_y;
        dispatch_args[2] = 1u;
        dispatch_args[3] = control.faces_x;
        dispatch_args[4] = control.faces_y;
        dispatch_args[5] = 1u;
    } else {
        dispatch_args[0] = 0u;
        dispatch_args[1] = 1u;
        dispatch_args[2] = 1u;
        dispatch_args[3] = 0u;
        dispatch_args[4] = 1u;
        dispatch_args[5] = 1u;
    }
}
fn write_stage_time(a: f32, first: bool) {
    if control.halt != 0u {
        set_active(false);
        dispatch_args[6] = 0u;
        dispatch_args[7] = 1u;
        dispatch_args[8] = 1u;
        return;
    }
    if first {
        control.step_dt = control.next_dt;
        set_active(true);
    }
    constants_words[0] = bitcast<u32>(control.step_dt);
    constants_words[1] = bitcast<u32>(control.dt_old);
    constants_words[2] = bitcast<u32>(0.0);
    constants_words[3] = bitcast<u32>(clock_plus(a * control.step_dt));
}

@compute @workgroup_size(1) fn stage_1() { write_stage_time(0.0, true); }
@compute @workgroup_size(1) fn stage_2() { write_stage_time(0.5, false); }
@compute @workgroup_size(1) fn stage_3() { write_stage_time(0.5, false); }
@compute @workgroup_size(1) fn stage_4() { write_stage_time(1.0, false); }

@compute @workgroup_size(1)
fn batch_reset() {
    control.accepted_batch = 0u;
}

@compute @workgroup_size(64)
fn volume_audit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = idx2(gid);
    if cell >= params.num_cells { return; }
    let volume = cell_vols[cell];
    if !finite(volume) || !(volume > 0.0) {
        atomicAdd(&control.invalid_count, 1u);
    }
}

@compute @workgroup_size(1)
fn volume_finalize() {
    if atomicLoad(&control.invalid_count) != 0u {
        control.halt = 1u;
        set_active(false);
        dispatch_args[6] = 0u;
        dispatch_args[7] = 1u;
        dispatch_args[8] = 1u;
    }
}

@compute @workgroup_size(1)
fn health_clear() {
    if control.halt != 0u { return; }
    atomicStore(&control.invalid_count, 0u);
    atomicStore(&control.max_base_bits, 0u);
    atomicStore(&control.max_turnover_bits, 0u);
    atomicStore(&metrics.max_velocity_bits, 0u);
    atomicStore(&metrics.max_characteristic_speed_bits, 0u);
    atomicStore(&metrics.max_inv_h_bits, 0u);
    dispatch_args[6] = 0u;
    dispatch_args[7] = 1u;
    dispatch_args[8] = 1u;
}

@compute @workgroup_size(64)
fn history_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = idx2(gid);
    if cell >= params.num_cells { return; }
    let base = cell * params.state_stride;
    for (var k = 0u; k < params.state_stride; k += 1u) {
        history_backup[base + k] = state[base + k];
        state[base + k] = state_old[base + k];
    }
}

fn audit_cell(cell: u32, accepted: bool) {
    if control.halt != 0u || cell >= params.num_cells { return; }
    let base = cell * params.state_stride;
    for (var k = 0u; k < params.state_stride; k += 1u) {
        if !finite(select(state[base + k], state_old[base + k], accepted)) {
            atomicAdd(&control.invalid_count, 1u);
            return;
        }
    }

    let gamma = bitcast<f32>(constants_words[14]);
    let gm1 = bitcast<f32>(constants_words[15]);
    let gas_r = bitcast<f32>(constants_words[16]);
    let dp_drho = bitcast<f32>(constants_words[17]);
    let p_ref = bitcast<f32>(constants_words[18]);
    let theta_ref = bitcast<f32>(constants_words[19]);
    let rho_ref = bitcast<f32>(constants_words[20]);
    if !finite(gamma) || !(gamma > 1.0)
        || !finite(gm1) || !(gm1 > 0.0)
        || !finite(gas_r) || !(gas_r > 0.0)
        || !finite(theta_ref) || !(theta_ref > 0.0)
        || dp_drho != 0.0 || p_ref != 0.0 || rho_ref != 0.0 {
        atomicAdd(&control.invalid_count, 1u);
        return;
    }

    let rho = state_value(cell, params.rho_off, accepted);
    let mx = state_value(cell, params.rho_u_off, accepted);
    let my = state_value(cell, params.rho_u_off + 1u, accepted);
    let rho_e = state_value(cell, params.rho_e_off, accepted);
    if !(rho > 0.0) {
        atomicAdd(&control.invalid_count, 1u);
        return;
    }
    let kinetic_density = 0.5 * (mx * mx + my * my) / rho;
    let internal_density = rho_e - kinetic_density;
    let pressure = gm1 * internal_density;
    let temperature = pressure / (rho * gas_r);
    let sound_sq = gamma * pressure / rho;
    let ux = mx / rho;
    let uy = my / rho;
    let speed = length(vec2<f32>(ux, uy));
    if !finite(kinetic_density) || !finite(internal_density) || !(internal_density > 0.0)
        || !finite(pressure) || !(pressure > 0.0)
        || !finite(temperature) || !(temperature > 0.0)
        || !finite(sound_sq) || !(sound_sq > 0.0)
        || !finite(speed) {
        atomicAdd(&control.invalid_count, 1u);
        return;
    }

    if control.adaptive != 0u {
        let volume = cell_vols[cell];
        if !finite(volume) || !(volume > 0.0) {
            atomicAdd(&control.invalid_count, 1u);
            return;
        }
        let h = sqrt(volume);
        let inv_h = 1.0 / h;
        if !finite(h) || !(h > 1.0e-12)
            || !finite(inv_h) || !(inv_h > 0.0) {
            atomicAdd(&control.invalid_count, 1u);
            return;
        }
        atomicMax(&metrics.max_velocity_bits, bitcast<u32>(speed));
        atomicMax(
            &metrics.max_characteristic_speed_bits,
            bitcast<u32>(speed + sqrt(sound_sq)),
        );
        atomicMax(&metrics.max_inv_h_bits, bitcast<u32>(inv_h));
    }
}

@compute @workgroup_size(64)
fn audit_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    audit_cell(idx2(gid), false);
}

@compute @workgroup_size(64)
fn audit_accepted_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    audit_cell(idx2(gid), true);
}

@compute @workgroup_size(1)
fn bootstrap_finalize() {
    if control.halt != 0u { set_active(false); return; }
    var reject = atomicLoad(&control.invalid_count) != 0u;
    control.total_rate = 0.0;
    if control.adaptive != 0u && !reject {
        let max_velocity = bitcast<f32>(atomicLoad(&metrics.max_velocity_bits));
        let max_characteristic_speed = bitcast<f32>(
            atomicLoad(&metrics.max_characteristic_speed_bits));
        let max_inv_h = bitcast<f32>(atomicLoad(&metrics.max_inv_h_bits));
        let target_cfl_local = control.target_cfl;
        let gamma = bitcast<f32>(constants_words[14]);
        let theta_ref = bitcast<f32>(constants_words[19]);
        let sound_ref_sq = gamma * theta_ref;
        let inlet_velocity = abs(bitcast<f32>(constants_words[12]));
        let min_h = 1.0 / max_inv_h;
        // The accepted conserved state is authoritative.  A reference-T sound
        // speed underestimates the CFL rate after compression/heating and can
        // accept an otherwise valid state only to overstep on the next batch.
        let wave_speed = max(
            max_characteristic_speed,
            inlet_velocity + sqrt(sound_ref_sq),
        );
        var proposed = target_cfl_local * min_h / wave_speed;
        var has_bound = finite(proposed) && proposed > 0.0;
        let rho_ref = max(abs(bitcast<f32>(constants_words[5])), 1.0e-12);
        let alpha = abs(bitcast<f32>(constants_words[4])) / rho_ref;
        if finite(alpha) && alpha > 1.0e-14 {
            let diffusion_dt = 0.25 * target_cfl_local * min_h * min_h / alpha;
            if finite(diffusion_dt) && diffusion_dt > 0.0 {
                proposed = select(diffusion_dt, min(proposed, diffusion_dt), has_bound);
                has_bound = true;
            }
        }
        reject = !finite(max_velocity) || !finite(max_inv_h) || !(max_inv_h > 0.0)
            || !finite(target_cfl_local) || !(target_cfl_local > 0.0)
            || !finite(control.growth) || !(control.growth > 0.0)
            || !finite(control.max_dt) || !(control.max_dt >= 1.0e-9)
            || !finite(max_characteristic_speed) || !(max_characteristic_speed > 0.0)
            || !finite(sound_ref_sq) || !(sound_ref_sq > 0.0)
            || !finite(wave_speed) || !(wave_speed > 1.0e-12)
            || !has_bound;
        if !reject {
            let rate = target_cfl_local / proposed;
            reject = !finite(rate) || !(rate > 0.0);
            if !reject {
                atomicStore(&control.max_base_bits, bitcast<u32>(rate));
                control.total_rate = rate;
                control.next_dt = clamp(
                    min(proposed, control.next_dt * control.growth),
                    1.0e-9,
                    control.max_dt,
                );
                reject = !finite(control.next_dt) || !(control.next_dt > 0.0);
            }
        }
    }
    if reject {
        control.halt = 1u;
        set_active(false);
    }
    constants_words[0] = bitcast<u32>(control.next_dt);
    constants_words[1] = bitcast<u32>(control.dt_old);
    constants_words[3] = bitcast<u32>(clock_seconds());
}

@compute @workgroup_size(1)
fn health_finalize() {
    if control.halt != 0u {
        set_active(false);
        return;
    }
    if control.fail_after_accepted != 0xffffffffu
        && (control.accepted_total_hi != 0u
            || control.accepted_total_lo >= control.fail_after_accepted) {
        atomicAdd(&control.invalid_count, 1u);
    }
    if (control.accepted_total_lo == 0xffffffffu
            && control.accepted_total_hi == 0xffffffffu)
        || control.accepted_batch == 0xffffffffu {
        atomicAdd(&control.invalid_count, 1u);
    }

    var reject = atomicLoad(&control.invalid_count) != 0u
        || !finite(control.step_dt) || !(control.step_dt > 0.0);
    control.total_rate = 0.0;
    if control.adaptive != 0u && !reject {
        let max_velocity = bitcast<f32>(atomicLoad(&metrics.max_velocity_bits));
        let max_characteristic_speed = bitcast<f32>(
            atomicLoad(&metrics.max_characteristic_speed_bits));
        let max_inv_h = bitcast<f32>(atomicLoad(&metrics.max_inv_h_bits));
        let target_cfl_local = control.target_cfl;
        let gamma = bitcast<f32>(constants_words[14]);
        let theta_ref = bitcast<f32>(constants_words[19]);
        let sound_ref_sq = gamma * theta_ref;
        let inlet_velocity = abs(bitcast<f32>(constants_words[12]));
        let min_h = 1.0 / max_inv_h;
        let wave_speed = max(
            max_characteristic_speed,
            inlet_velocity + sqrt(sound_ref_sq),
        );
        var proposed = target_cfl_local * min_h / wave_speed;
        var has_bound = finite(proposed) && proposed > 0.0;
        let rho_ref = max(abs(bitcast<f32>(constants_words[5])), 1.0e-12);
        let alpha = abs(bitcast<f32>(constants_words[4])) / rho_ref;
        if finite(alpha) && alpha > 1.0e-14 {
            let diffusion_dt = 0.25 * target_cfl_local * min_h * min_h / alpha;
            if finite(diffusion_dt) && diffusion_dt > 0.0 {
                proposed = select(diffusion_dt, min(proposed, diffusion_dt), has_bound);
                has_bound = true;
            }
        }
        reject = !finite(max_velocity) || !finite(max_inv_h) || !(max_inv_h > 0.0)
            || !finite(target_cfl_local) || !(target_cfl_local > 0.0)
            || !finite(control.growth) || !(control.growth > 0.0)
            || !finite(control.max_dt) || !(control.max_dt >= 1.0e-9)
            || !finite(max_characteristic_speed) || !(max_characteristic_speed > 0.0)
            || !finite(sound_ref_sq) || !(sound_ref_sq > 0.0)
            || !finite(wave_speed) || !(wave_speed > 1.0e-12)
            || !has_bound;
        if !reject {
            let rate = target_cfl_local / proposed;
            reject = !finite(rate) || !(rate > 0.0);
            if !reject {
                atomicStore(&control.max_base_bits, bitcast<u32>(rate));
                control.total_rate = rate;
                control.next_dt = clamp(
                    min(proposed, control.step_dt * control.growth),
                    1.0e-9,
                    control.max_dt,
                );
                reject = !finite(control.next_dt) || !(control.next_dt > 0.0);
            }
        }
    }

    if reject {
        control.halt = 1u;
        set_active(false);
        dispatch_args[6] = control.cells_x;
        dispatch_args[7] = control.cells_y;
        dispatch_args[8] = 1u;
        constants_words[0] = bitcast<u32>(control.next_dt);
        constants_words[3] = bitcast<u32>(clock_seconds());
        return;
    }

    add_clock(control.step_dt);
    control.dt_old = control.step_dt;
    control.last_dt = control.step_dt;
    let next_total_lo = control.accepted_total_lo + 1u;
    if next_total_lo == 0u {
        control.accepted_total_hi += 1u;
    }
    control.accepted_total_lo = next_total_lo;
    control.accepted_batch += 1u;
    if control.adaptive == 0u {
        control.next_dt = control.step_dt;
    }
    constants_words[0] = bitcast<u32>(control.next_dt);
    constants_words[1] = bitcast<u32>(control.dt_old);
    constants_words[3] = bitcast<u32>(clock_seconds());
    dispatch_args[6] = 0u;
    dispatch_args[7] = 1u;
    dispatch_args[8] = 1u;
}

@compute @workgroup_size(64)
fn rollback_invalid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = idx2(gid);
    if cell >= params.num_cells { return; }
    let base = cell * params.state_stride;
    for (var k = 0u; k < params.state_stride; k += 1u) {
        state[base + k] = history_backup[base + k];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn conserved_ideal_gas_valid(
        rho: f32,
        momentum: [f32; 2],
        rho_e: f32,
        gamma: f32,
        gm1: f32,
        gas_r: f32,
    ) -> bool {
        let kinetic = 0.5 * (momentum[0] * momentum[0] + momentum[1] * momentum[1]) / rho;
        let internal = rho_e - kinetic;
        let pressure = gm1 * internal;
        let temperature = pressure / (rho * gas_r);
        let sound_sq = gamma * pressure / rho;
        rho.is_finite()
            && rho > 0.0
            && kinetic.is_finite()
            && internal.is_finite()
            && internal > 0.0
            && pressure.is_finite()
            && pressure > 0.0
            && temperature.is_finite()
            && temperature > 0.0
            && sound_sq.is_finite()
            && sound_sq > 0.0
    }

    fn host_oracle(
        current_dt: f32,
        target_cfl: f32,
        min_h: f32,
        max_characteristic_speed: f32,
        inlet_characteristic_speed: f32,
        viscosity: f32,
        rho_ref: f32,
    ) -> f32 {
        let wave = max_characteristic_speed.max(inlet_characteristic_speed);
        let mut proposed = target_cfl * min_h / wave;
        let alpha = viscosity.abs() / rho_ref.abs().max(1.0e-12);
        if alpha > 1.0e-14 {
            proposed = proposed.min(0.25 * target_cfl * min_h * min_h / alpha);
        }
        proposed.min(current_dt * 1.2).clamp(1.0e-9, 100.0)
    }

    #[test]
    fn common_status_abi_is_pinned() {
        assert_eq!(std::mem::size_of::<GpuExplicitControl>(), 104);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, next_dt), 12);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, target_cfl), 24);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, accepted_total_lo), 40);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, accepted_total_hi), 44);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, invalid_count), 56);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, last_dt), 68);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, cells_x), 80);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, adaptive), 96);
        assert_eq!(std::mem::offset_of!(GpuExplicitControl, initial_phase), 100);
    }

    #[test]
    fn raw_eos_constant_word_indices_are_pinned() {
        assert_eq!(std::mem::size_of::<GpuConstants>(), 24 * 4);
        assert_eq!(std::mem::offset_of!(GpuConstants, eos_gamma), 14 * 4);
        assert_eq!(std::mem::offset_of!(GpuConstants, eos_gm1), 15 * 4);
        assert_eq!(std::mem::offset_of!(GpuConstants, eos_r), 16 * 4);
        assert_eq!(std::mem::offset_of!(GpuConstants, eos_dp_drho), 17 * 4);
        assert_eq!(std::mem::offset_of!(GpuConstants, eos_p_ref), 18 * 4);
        assert_eq!(std::mem::offset_of!(GpuConstants, eos_theta_ref), 19 * 4);
        assert_eq!(std::mem::offset_of!(GpuConstants, eos_rho_ref), 20 * 4);
    }

    #[test]
    fn status_decodes_cumulative_counter_across_low_word_wrap() {
        let raw = GpuExplicitControl {
            accepted_total_lo: 7,
            accepted_total_hi: 3,
            ..GpuExplicitControl::zeroed()
        };
        let status =
            CompressibleExplicitControl::decode_status(bytemuck::bytes_of(&raw)).unwrap();
        assert_eq!(status.accepted_total, (3_u64 << 32) | 7);
    }

    #[test]
    fn controller_shaders_compile() {
        let Ok(context) = pollster::block_on(crate::solver::gpu::context::GpuContext::new(
            None, None,
        )) else {
            return;
        };
        let device = &context.device;
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let control_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compressible controller test control"),
            entries: &[
                storage(0, false),
                storage(1, false),
                storage(2, false),
                uniform(3),
                storage(4, false),
            ],
        });
        let phase_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compressible controller test phase"),
            entries: &[
                storage(0, false),
                storage(1, true),
                storage(2, true),
                storage(3, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compressible controller test layout"),
            bind_group_layouts: &[&control_bgl, &phase_bgl],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compressible controller test shader"),
            source: wgpu::ShaderSource::Wgsl(COMPRESSIBLE_EXPLICIT_CONTROL_WGSL.into()),
        });
        for entry in [
            "stage_1",
            "stage_2",
            "stage_3",
            "stage_4",
            "batch_reset",
            "health_clear",
            "history_prepare",
            "audit_state",
            "audit_accepted_state",
            "bootstrap_finalize",
            "health_finalize",
            "rollback_invalid",
        ] {
            let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        }

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compressible controller frame test"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                uniform(5),
            ],
        });
        let frame_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compressible controller frame test layout"),
            bind_group_layouts: &[&frame_bgl],
            push_constant_ranges: &[],
        });
        let frame_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compressible controller frame test shader"),
            source: wgpu::ShaderSource::Wgsl(COMPRESSIBLE_EXPLICIT_FRAME_COPY_WGSL.into()),
        });
        let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("copy_accepted_state"),
            layout: Some(&frame_layout),
            module: &frame_shader,
            entry_point: Some("copy_accepted_state"),
            compilation_options: Default::default(),
            cache: None,
        });
    }

    #[test]
    fn conserved_energy_domain_ignores_stale_auxiliary_primitives() {
        assert!(conserved_ideal_gas_valid(
            1.0,
            [2.0, 0.0],
            4.5,
            1.4,
            0.4,
            287.0,
        ));
        assert!(!conserved_ideal_gas_valid(
            1.0,
            [2.0, 0.0],
            1.5,
            1.4,
            0.4,
            287.0,
        ));
    }

    #[test]
    fn adaptive_oracle_selects_wave_diffusion_and_growth_bounds() {
        // Counterexample to post-candidate-only adaptation: a 20 ms GUI seed on
        // a 1 cm air cell has acoustic CFL ~700. The pre-stage bootstrap must
        // replace it with the accepted-state acoustic oracle before RK stage 1.
        let oversized = host_oracle(
            0.020,
            0.9,
            0.010,
            (1.4_f32 * 86_070.0).sqrt(),
            0.011 + (1.4_f32 * 86_070.0).sqrt(),
            1.8e-5,
            1.225,
        );
        let expected_oversized = 0.9 * 0.010 / (0.011 + (1.4_f32 * 86_070.0).sqrt());
        assert!(oversized < 0.020);
        assert!((oversized - expected_oversized).abs() <= expected_oversized * 2.0e-6);

        let wave = host_oracle(
            1.0,
            0.9,
            0.1,
            2.0 + 1.4_f32.sqrt(),
            1.0 + 1.4_f32.sqrt(),
            0.0,
            1.0,
        );
        let expected_wave = 0.9 * 0.1 / (2.0 + 1.4_f32.sqrt());
        assert!((wave - expected_wave).abs() <= 2.0 * f32::EPSILON);

        let diffusion = host_oracle(1.0, 0.9, 0.1, 1.4_f32.sqrt(), 1.4_f32.sqrt(), 10.0, 1.0);
        assert!((diffusion - 0.000225).abs() <= 2.0 * f32::EPSILON);

        let growth = host_oracle(
            0.01,
            0.9,
            100.0,
            1.4_f32.sqrt(),
            1.4_f32.sqrt(),
            0.0,
            1.0,
        );
        assert!((growth - 0.012).abs() <= 2.0 * f32::EPSILON);
    }

    #[test]
    fn adaptive_oracle_uses_accepted_local_sound_speed_after_heating() {
        let reference_sound = (1.4_f32 * 287.0 * 300.0).sqrt();
        let hot_sound = (1.4_f32 * 287.0 * 1200.0).sqrt();
        let reference_dt = host_oracle(
            1.0,
            0.9,
            0.01,
            5.0 + reference_sound,
            reference_sound,
            0.0,
            1.225,
        );
        let hot_dt = host_oracle(
            1.0,
            0.9,
            0.01,
            5.0 + hot_sound,
            reference_sound,
            0.0,
            1.225,
        );
        assert!(hot_dt < reference_dt * 0.51, "{hot_dt} vs {reference_dt}");
    }

    #[test]
    fn live_eos_gate_rejects_before_mutation_and_rearms_on_ideal_return() {
        let ideal = GpuConstants::default();
        let mut linear = ideal;
        linear.eos_gamma = 0.0;
        linear.eos_gm1 = 0.0;
        linear.eos_dp_drho = 2.2e9;
        linear.eos_theta_ref = 0.0;

        let initialized = AtomicBool::new(true);
        let supported = AtomicBool::new(true);
        assert!(!observe_eos_support(&initialized, &supported, &linear));
        assert!(
            !initialized.load(Ordering::Acquire),
            "ideal -> linear must invalidate before a submit can mutate phase"
        );

        // Model resources still exist while the unsupported EOS takes the
        // ordinary fallback. Returning to an ideal gas must therefore become
        // eligible without a solver rebuild, but with a fresh CFL bootstrap.
        initialized.store(true, Ordering::Release);
        assert!(observe_eos_support(&initialized, &supported, &ideal));
        assert!(
            !initialized.load(Ordering::Acquire),
            "linear -> ideal must re-seed the autonomous controller"
        );
    }

    #[test]
    fn live_eos_gate_rejects_partial_runtime_transition() {
        let mut half_written = GpuConstants::default();
        half_written.eos_gamma = 0.0;
        half_written.eos_gm1 = 0.0;
        // r still happens to be positive while a sequence of runtime writes is
        // in progress; that alone must not misclassify the closure as ideal.
        assert!(!ideal_gas_constants_supported(&half_written));
    }
}
