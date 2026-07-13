//! GPU-resident health/CFL controller for autonomous unstructured RK4 batches.
//!
//! The controller intentionally mirrors the host all-Mach policy as two
//! independently reduced non-negative rates:
//!
//! `lambda = max_i(lambda_base_i) + max_i(lambda_RC_i)`.
//!
//! Keeping those maxima separate is important: taking a maximum of their
//! per-cell sum changes the accepted timestep when the acoustic/diffusive and
//! Rhie--Chow hot spots occur in different cells.

use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExplicitControlCapabilities {
    pub fixed_health: bool,
    pub adaptive_health: bool,
    pub accepted_state_copy: bool,
}

pub(crate) struct ExplicitStatusStagingLease {
    buffer: wgpu::Buffer,
    slot: usize,
    busy: Arc<[AtomicBool; STATUS_RING_SLOTS]>,
}

impl ExplicitStatusStagingLease {
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for ExplicitStatusStagingLease {
    fn drop(&mut self) {
        self.busy[self.slot].store(false, Ordering::Release);
    }
}

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

/// Tiny completion telemetry copied from the controller after a batch fence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ExplicitControlStatus {
    pub time: f64,
    pub next_dt: f32,
    pub last_dt: f32,
    pub accepted_total: u64,
    pub accepted_batch: u32,
    pub halted: bool,
    pub invalid_count: u32,
    pub max_base_rate: f32,
    pub max_rhie_chow_turnover: f32,
    pub total_rate: f32,
}

impl From<GpuExplicitControl> for ExplicitControlStatus {
    fn from(value: GpuExplicitControl) -> Self {
        let fraction = value.time_fraction_hi as f64 * Q64_HI_SCALE
            + value.time_fraction_lo as f64 * Q64_LO_SCALE;
        Self {
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
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuExplicitHealthParams {
    num_cells: u32,
    num_faces: u32,
    state_stride: u32,
    dispatch_stride: u32,
    unknown_stride: u32,
    p_component: u32,
    u_off: u32,
    p_off: u32,
    rho_off: u32,
    psi_precond_off: u32,
    d_p_off: u32,
    rho_dt_off: u32,
    temperature_off: u32,
    rho_floor_off: u32,
    psi_ref_off: u32,
    t_ref_off: u32,
    penalty_off: u32,
    u_ref_off: u32,
    precond_mask_off: u32,
    dt_local_off: u32,
    psi_off: u32,
    thermal: u32,
}

struct ExplicitFrameCopyTargets {
    destinations: [wgpu::Buffer; 3],
    bind_groups: [wgpu::BindGroup; 3],
}

pub(crate) struct ExplicitAdaptiveControl {
    initialized: AtomicBool,
    adaptive_mode: AtomicBool,
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
    b_grad_p: wgpu::Buffer,
    b_face_magnitude: wgpu::Buffer,
    b_history_backup: wgpu::Buffer,
    pub(crate) b_indirect_args: Arc<wgpu::Buffer>,
    /// INDIRECT-usage mirror of `b_indirect_args`. wgpu (>=28, per WebGPU
    /// usage-scope rules) rejects a dispatch whose bound bind groups include
    /// the indirect-args buffer as read-write storage. Control kernels keep
    /// writing `b_indirect_args`; each indirect dispatch snapshots it into
    /// this mirror with a 36-byte copy just before its pass.
    b_indirect_dispatch: wgpu::Buffer,
    control_bg: wgpu::BindGroup,
    phase_bgs: [wgpu::BindGroup; 3],
    stage_pipelines: [wgpu::ComputePipeline; 4],
    batch_reset_pipeline: wgpu::ComputePipeline,
    volume_audit_pipeline: wgpu::ComputePipeline,
    volume_finalize_pipeline: wgpu::ComputePipeline,
    history_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,
    gradient_pipeline: wgpu::ComputePipeline,
    face_pipeline: wgpu::ComputePipeline,
    cell_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    rollback_pipeline: wgpu::ComputePipeline,
    frame_copy_layout: wgpu::BindGroupLayout,
    frame_copy_pipeline: wgpu::ComputePipeline,
    frame_copy_targets: Option<ExplicitFrameCopyTargets>,
    state_buffers: [wgpu::Buffer; 3],
    #[cfg(test)]
    time_probe_pipeline: wgpu::ComputePipeline,
    #[cfg(test)]
    time_random_probe_pipeline: wgpu::ComputePipeline,
    cells_dispatch: (u32, u32),
    faces_dispatch: (u32, u32),
    frame_copy_dispatch: (u32, u32),
    state_size_bytes: u64,
}

impl ExplicitAdaptiveControl {
    pub(crate) const fn capabilities(&self) -> ExplicitControlCapabilities {
        ExplicitControlCapabilities {
            fixed_health: true,
            adaptive_health: true,
            accepted_state_copy: true,
        }
    }

    pub(crate) fn new(
        device: &wgpu::Device,
        mesh: &MeshResources,
        fields: &UnifiedFieldResources,
        bc_kind: &wgpu::Buffer,
        bc_value: &wgpu::Buffer,
        model: &ModelSpec,
    ) -> Result<Option<Self>, String> {
        // The controller is deliberately limited to the two production static
        // pressure-based all-Mach closures. Other models need their own domain
        // and spectral derivation before they can enter this route.
        let thermal = match model.id {
            "allmach_thermal" => true,
            "allmach_pressure" => false,
            _ => return Ok(None),
        };
        if model.system.is_ale() {
            return Ok(None);
        }

        let offset = |name: &str| {
            model.state_layout.offset_for(name).ok_or_else(|| {
                format!("autonomous all-Mach controller missing state field '{name}'")
            })
        };
        let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
        let p_component = flux_layout
            .offset_for("p")
            .ok_or_else(|| "autonomous all-Mach controller missing coupled p slot".to_string())?;
        let absent = u32::MAX;
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
            .ok_or_else(|| "autonomous accepted-state copy size overflows u32".to_string())?;
        let frame_copy_dispatch = dispatch(state_words);
        let optional = |name: &str| model.state_layout.offset_for(name).unwrap_or(absent);
        let thermal_offset = |name: &str| {
            if thermal {
                offset(name)
            } else {
                Ok(absent)
            }
        };
        let params = GpuExplicitHealthParams {
            num_cells: fields.num_cells,
            num_faces: fields.num_faces,
            state_stride: fields.state_stride,
            dispatch_stride: max_groups.saturating_mul(64),
            unknown_stride: model.system.unknowns_per_cell(),
            p_component,
            u_off: offset("U")?,
            p_off: offset("p")?,
            rho_off: offset("rho")?,
            psi_precond_off: offset("psi_precond")?,
            d_p_off: offset("d_p")?,
            rho_dt_off: thermal_offset("rho_dT")?,
            temperature_off: thermal_offset("T")?,
            rho_floor_off: thermal_offset("rho_floor")?,
            psi_ref_off: thermal_offset("psi_ref")?,
            t_ref_off: thermal_offset("t_ref")?,
            penalty_off: optional("ibm_penalty_U"),
            u_ref_off: offset("u_ref")?,
            precond_mask_off: offset("precond_mask")?,
            dt_local_off: offset("dt_local")?,
            psi_off: if thermal {
                optional("psi")
            } else {
                offset("psi")?
            },
            thermal: u32::from(thermal),
        };

        let b_control = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit_control:status"),
            size: std::mem::size_of::<GpuExplicitControl>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let status_staging = std::array::from_fn(|slot| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match slot {
                    0 => "explicit_control:status_ring_0",
                    1 => "explicit_control:status_ring_1",
                    _ => "explicit_control:status_ring_2",
                }),
                size: std::mem::size_of::<GpuExplicitControl>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let status_slots_busy = Arc::new(std::array::from_fn(|_| AtomicBool::new(false)));
        let b_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("explicit_control:params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let b_grad_p = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit_control:grad_p"),
            size: (fields.num_cells.max(1) as u64) * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b_face_magnitude = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit_control:face_rc_magnitude"),
            size: (fields.num_faces.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b_history_backup = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit_control:history_backup"),
            size: fields.state_size_bytes().max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b_indirect_args = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit_control:indirect_args"),
            size: 9 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let b_indirect_dispatch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit_control:indirect_dispatch"),
            size: 9 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

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
        let control_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit_control:control_bgl"),
            entries: &[
                storage(0, false),
                storage(1, false),
                storage(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let phase_entries: Vec<_> = (0..=17)
            .map(|binding| storage(binding, !matches!(binding, 0 | 2 | 3 | 17)))
            .collect();
        let phase_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit_control:phase_bgl"),
            entries: &phase_entries,
        });
        let frame_copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit_control:frame_copy_bgl"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit_control:pipeline_layout"),
            bind_group_layouts: &[Some(&control_bgl), Some(&phase_bgl)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_control:shader"),
            source: wgpu::ShaderSource::Wgsl(EXPLICIT_CONTROL_WGSL.into()),
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
        let gradient_pipeline = pipeline("pressure_gradient");
        let face_pipeline = pipeline("face_turnover");
        let cell_pipeline = pipeline("cell_rate");
        let finalize_pipeline = pipeline("health_finalize");
        let rollback_pipeline = pipeline("rollback_invalid");
        #[cfg(test)]
        let time_probe_pipeline = pipeline("time_accumulation_probe");
        #[cfg(test)]
        let time_random_probe_pipeline = pipeline("time_random_sequence_probe");
        let frame_copy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("explicit_control:frame_copy_pipeline_layout"),
                bind_group_layouts: &[Some(&frame_copy_layout)],
                immediate_size: 0,
            });
        let frame_copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_control:frame_copy_shader"),
            source: wgpu::ShaderSource::Wgsl(EXPLICIT_FRAME_COPY_WGSL.into()),
        });
        let frame_copy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("explicit_control:frame_copy_pipeline"),
                layout: Some(&frame_copy_pipeline_layout),
                module: &frame_copy_shader,
                entry_point: Some("copy_accepted_state"),
                compilation_options: Default::default(),
                cache: None,
            });

        let control_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit_control:control_bg"),
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
            ],
        });

        let phase_bgs = std::array::from_fn(|phase| {
            let (current, old, _) = crate::solver::gpu::modules::state::ping_pong_indices(phase);
            let buffers = fields.state_buffers();
            let resources: [&wgpu::Buffer; 18] = [
                &buffers[current],
                &buffers[old],
                &b_grad_p,
                &b_face_magnitude,
                &mesh.b_face_owner,
                &mesh.b_face_neighbor,
                &mesh.b_face_boundary,
                &mesh.b_face_areas,
                &mesh.b_face_normals,
                &mesh.b_face_centers,
                &mesh.b_face_wrap_shift,
                &mesh.b_cell_centers,
                &mesh.b_cell_vols,
                &mesh.b_cell_face_offsets,
                &mesh.b_cell_faces,
                bc_kind,
                bc_value,
                &b_history_backup,
            ];
            let entries: Vec<_> = resources
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("explicit_control:phase_bg"),
                layout: &phase_bgl,
                entries: &entries,
            })
        });

        Ok(Some(Self {
            initialized: AtomicBool::new(false),
            adaptive_mode: AtomicBool::new(false),
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
            b_grad_p,
            b_face_magnitude,
            b_history_backup,
            b_indirect_args,
            b_indirect_dispatch,
            control_bg,
            phase_bgs,
            stage_pipelines,
            batch_reset_pipeline,
            volume_audit_pipeline,
            volume_finalize_pipeline,
            history_pipeline,
            clear_pipeline,
            gradient_pipeline,
            face_pipeline,
            cell_pipeline,
            finalize_pipeline,
            rollback_pipeline,
            frame_copy_layout,
            frame_copy_pipeline,
            frame_copy_targets: None,
            state_buffers: std::array::from_fn(|index| fields.state_buffers()[index].clone()),
            #[cfg(test)]
            time_probe_pipeline,
            #[cfg(test)]
            time_random_probe_pipeline,
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
        _dt_is_accepted_candidate: bool,
    ) {
        let target_cfl = target_cfl.clamp(1.0e-6, 1.0);
        self.adaptive_mode.store(adaptive, Ordering::Release);
        let newly_initialized = !self.initialized.swap(true, Ordering::AcqRel);
        if newly_initialized {
            // Static geometry is immutable between topology refreshes, so a
            // single queue-ordered audit is sufficient. Re-seeding the
            // controller re-arms it before the first RK stage.
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
                safety: 0.8,
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
            // Queue-ordered between batches: update only controller policy,
            // never overwrite GPU-produced time/rate/halt state.
            queue.write_buffer(&self.b_control, 6 * 4, bytemuck::bytes_of(&target_cfl));
            let adaptive = u32::from(adaptive);
            queue.write_buffer(
                &self.b_control,
                std::mem::offset_of!(GpuExplicitControl, adaptive) as u64,
                bytemuck::bytes_of(&adaptive),
            );
            if adaptive == 0 {
                // Fixed mode treats the configured RuntimeParams timestep as
                // policy, not the previous adaptive controller candidate.
                queue.write_buffer(
                    &self.b_control,
                    std::mem::offset_of!(GpuExplicitControl, next_dt) as u64,
                    bytemuck::bytes_of(&dt),
                );
            }
        }
    }

    pub(crate) fn encode_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: usize,
        phase: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("explicit_control:stage"),
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
                label: Some("explicit_control:batch_reset"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.batch_reset_pipeline);
            pass.set_bind_group(0, &self.control_bg, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        if self.volume_audit_pending.swap(false, Ordering::AcqRel) {
            // Both passes precede stage_1 in the same command buffer. An
            // invalid raw volume therefore zeros every indirect dispatch
            // before any history or residual kernel can touch state.
            for (pipeline, x, y, label) in [
                (
                    &self.volume_audit_pipeline,
                    self.cells_dispatch.0,
                    self.cells_dispatch.1,
                    "explicit_control:volume_audit",
                ),
                (
                    &self.volume_finalize_pipeline,
                    1,
                    1,
                    "explicit_control:volume_finalize",
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

    /// Save the history slot that an ordinary prepare-step overwrites, then
    /// seed the candidate current buffer from the last accepted state. Both
    /// operations are indirect-gated, so an already halted batch preserves all
    /// accepted-prefix history bytes.
    pub(crate) fn encode_history_prepare(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) {
        encoder.copy_buffer_to_buffer(&self.b_indirect_args, 0, &self.b_indirect_dispatch, 0, 9 * 4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("explicit_control:history_prepare"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.history_pipeline);
        pass.set_bind_group(0, &self.control_bg, &[]);
        pass.set_bind_group(1, &self.phase_bgs[phase % 3], &[]);
        pass.dispatch_workgroups_indirect(&self.b_indirect_dispatch, CELLS_INDIRECT_OFFSET);
    }

    pub(crate) fn encode_health(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) {
        let bg = &self.phase_bgs[phase % 3];
        let mut run = |pipeline: &wgpu::ComputePipeline, x: u32, y: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_control:health"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.control_bg, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.dispatch_workgroups(x, y, 1);
        };
        run(&self.clear_pipeline, 1, 1);
        if self.adaptive_mode.load(Ordering::Acquire) {
            run(
                &self.gradient_pipeline,
                self.cells_dispatch.0,
                self.cells_dispatch.1,
            );
            run(
                &self.face_pipeline,
                self.faces_dispatch.0,
                self.faces_dispatch.1,
            );
        }
        run(
            &self.cell_pipeline,
            self.cells_dispatch.0,
            self.cells_dispatch.1,
        );
        run(&self.finalize_pipeline, 1, 1);

        encoder.copy_buffer_to_buffer(&self.b_indirect_args, 0, &self.b_indirect_dispatch, 0, 9 * 4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("explicit_control:rollback"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.rollback_pipeline);
        pass.set_bind_group(0, &self.control_bg, &[]);
        pass.set_bind_group(1, bg, &[]);
        pass.dispatch_workgroups_indirect(&self.b_indirect_dispatch, ROLLBACK_INDIRECT_OFFSET);
    }

    pub(crate) fn status_buffer(&self) -> &wgpu::Buffer {
        &self.b_control
    }

    pub(crate) fn ensure_status_recoverable(&self) -> Result<(), String> {
        if self.status_faulted.load(Ordering::Acquire) {
            Err(
                "autonomous explicit controller is non-resumable after a status map/decode failure; rebuild the solver"
                    .to_string(),
            )
        } else {
            Ok(())
        }
    }

    pub(crate) fn status_fault_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.status_faulted)
    }

    pub(crate) fn acquire_status_staging(&self) -> Result<ExplicitStatusStagingLease, String> {
        let start = self.next_status_slot.fetch_add(1, Ordering::Relaxed) % STATUS_RING_SLOTS;
        for offset in 0..STATUS_RING_SLOTS {
            let slot = (start + offset) % STATUS_RING_SLOTS;
            if self.status_slots_busy[slot]
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(ExplicitStatusStagingLease {
                    buffer: self.status_staging[slot].clone(),
                    slot,
                    busy: Arc::clone(&self.status_slots_busy),
                });
            }
        }
        Err(
            "all three autonomous status staging slots are busy; poll GPU completions before submitting another batch"
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
                    "autonomous frame destination {slot} is {} bytes, needs {}",
                    destination.size(),
                    self.state_size_bytes
                ));
            }
            if !destination.usage().contains(wgpu::BufferUsages::STORAGE) {
                return Err(format!(
                    "autonomous frame destination {slot} lacks STORAGE usage"
                ));
            }
        }
        let bind_groups = std::array::from_fn(|slot| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("explicit_control:frame_copy_bg"),
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
        self.frame_copy_targets = Some(ExplicitFrameCopyTargets {
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

    /// Queue-ordered visualization snapshot of the latest accepted state. The
    /// GPU controller selects the physical triple-buffer slot from its original
    /// phase plus cumulative accepted count, so no host phase reconciliation is
    /// needed and rejected candidates can never leak into a frame.
    pub(crate) fn encode_accepted_state_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_slot: usize,
    ) -> Result<(), String> {
        let targets = self.frame_copy_targets.as_ref().ok_or_else(|| {
            "autonomous frame-copy targets were not installed for this solver".to_string()
        })?;
        let bind_group = targets.bind_groups.get(target_slot).ok_or_else(|| {
            format!("autonomous frame-copy slot {target_slot} is outside 0..3")
        })?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("explicit_control:copy_accepted_state"),
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
                "explicit control status is {} bytes, expected {size}",
                bytes.len()
            ));
        }
        Ok((*bytemuck::from_bytes::<GpuExplicitControl>(&bytes[..size])).into())
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
    pub(crate) fn inject_failure_after(&self, accepted_total_threshold: u32) {
        self.test_fail_after_accepted
            .store(accepted_total_threshold, Ordering::Release);
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

    #[cfg(test)]
    pub(crate) fn encode_time_accumulation_probe(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        phase: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("explicit_control:time_accumulation_probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.time_probe_pipeline);
        pass.set_bind_group(0, &self.control_bg, &[]);
        pass.set_bind_group(1, &self.phase_bgs[phase % 3], &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    #[cfg(test)]
    pub(crate) fn encode_time_random_sequence_probe(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        phase: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("explicit_control:time_random_sequence_probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.time_random_probe_pipeline);
        pass.set_bind_group(0, &self.control_bg, &[]);
        pass.set_bind_group(1, &self.phase_bgs[phase % 3], &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

const EXPLICIT_FRAME_COPY_WGSL: &str = r#"
struct Control {
    time_seconds: u32, time_fraction_hi: u32, time_fraction_lo: u32,
    next_dt: f32, step_dt: f32, dt_old: f32,
    target_cfl: f32, safety: f32, growth: f32, max_dt: f32,
    accepted_total_lo: u32, accepted_total_hi: u32, accepted_batch: u32,
    halt: u32, invalid_count: atomic<u32>, max_base_bits: atomic<u32>,
    max_turnover_bits: atomic<u32>, last_dt: f32, total_rate: f32, fail_after_accepted: u32,
    cells_x: u32, cells_y: u32, faces_x: u32, faces_y: u32,
    adaptive: u32, initial_phase: u32,
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
    let index=gid.x+gid.y*params.dispatch_stride;
    let words=params.num_cells*params.state_stride;
    if index>=words || index>=arrayLength(&destination) { return; }
    // 2^32 mod 3 == 1, so a portable lo/hi counter has this phase residue.
    let total_mod3=((control.accepted_total_lo%3u)+(control.accepted_total_hi%3u))%3u;
    let logical_phase=(control.initial_phase+total_mod3)%3u;
    if logical_phase==0u { destination[index]=state_0[index]; }
    else if logical_phase==1u { destination[index]=state_2[index]; }
    else { destination[index]=state_1[index]; }
}
"#;

const EXPLICIT_CONTROL_WGSL: &str = r#"
struct Control {
    time_seconds: u32, time_fraction_hi: u32, time_fraction_lo: u32,
    next_dt: f32, step_dt: f32, dt_old: f32,
    target_cfl: f32, safety: f32, growth: f32, max_dt: f32,
    accepted_total_lo: u32, accepted_total_hi: u32, accepted_batch: u32,
    halt: u32, invalid_count: atomic<u32>, max_base_bits: atomic<u32>,
    max_turnover_bits: atomic<u32>, last_dt: f32, total_rate: f32, fail_after_accepted: u32,
    cells_x: u32, cells_y: u32, faces_x: u32, faces_y: u32,
    adaptive: u32, initial_phase: u32,
};
struct Params {
    num_cells: u32, num_faces: u32, state_stride: u32, dispatch_stride: u32,
    unknown_stride: u32, p_component: u32, u_off: u32, p_off: u32,
    rho_off: u32, psi_precond_off: u32, d_p_off: u32, rho_dt_off: u32,
    temperature_off: u32, rho_floor_off: u32, psi_ref_off: u32, t_ref_off: u32,
    penalty_off: u32, u_ref_off: u32, precond_mask_off: u32, dt_local_off: u32,
    psi_off: u32, thermal: u32,
};
@group(0) @binding(0) var<storage, read_write> control: Control;
@group(0) @binding(1) var<storage, read_write> constants_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_args: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> state: array<f32>;
@group(1) @binding(1) var<storage, read> state_old: array<f32>;
@group(1) @binding(2) var<storage, read_write> grad_p: array<vec2<f32>>;
@group(1) @binding(3) var<storage, read_write> face_magnitude: array<f32>;
@group(1) @binding(4) var<storage, read> face_owner: array<u32>;
@group(1) @binding(5) var<storage, read> face_neighbor: array<u32>;
@group(1) @binding(6) var<storage, read> face_boundary: array<u32>;
@group(1) @binding(7) var<storage, read> face_areas: array<f32>;
@group(1) @binding(8) var<storage, read> face_normals: array<vec2<f32>>;
@group(1) @binding(9) var<storage, read> face_centers: array<vec2<f32>>;
@group(1) @binding(10) var<storage, read> face_wrap_shift: array<vec2<f32>>;
@group(1) @binding(11) var<storage, read> cell_centers: array<vec2<f32>>;
@group(1) @binding(12) var<storage, read> cell_vols: array<f32>;
@group(1) @binding(13) var<storage, read> cell_face_offsets: array<u32>;
@group(1) @binding(14) var<storage, read> cell_faces: array<u32>;
@group(1) @binding(15) var<storage, read> bc_kind: array<u32>;
@group(1) @binding(16) var<storage, read> bc_value: array<f32>;
@group(1) @binding(17) var<storage, read_write> history_backup: array<f32>;

fn finite(x: f32) -> bool { return x == x && abs(x) <= 3.402823e38; }
fn idx2(gid: vec3<u32>) -> u32 { return gid.x + gid.y * params.dispatch_stride; }
fn s(cell: u32, off: u32) -> f32 { return state[cell * params.state_stride + off]; }
struct Clock { seconds: u32, fraction_hi: u32, fraction_lo: u32 };
fn clock_parts() -> Clock {
    return Clock(control.time_seconds, control.time_fraction_hi, control.time_fraction_lo);
}
fn highest_bit(v0: u32) -> u32 {
    var v=v0; var bit=0u;
    if v>=65536u { v>>=16u; bit+=16u; }
    if v>=256u { v>>=8u; bit+=8u; }
    if v>=16u { v>>=4u; bit+=4u; }
    if v>=4u { v>>=2u; bit+=2u; }
    if v>=2u { bit+=1u; }
    return bit;
}
fn clock_bit(c: Clock, bit: u32) -> u32 {
    if bit<32u { return (c.fraction_lo>>bit)&1u; }
    if bit<64u { return (c.fraction_hi>>(bit-32u))&1u; }
    return (c.seconds>>(bit-64u))&1u;
}
fn low_mask(bits: u32) -> u32 { return (1u<<bits)-1u; }
fn any_bits_below(c: Clock, count: u32) -> bool {
    if count==0u { return false; }
    if count<32u { return (c.fraction_lo&low_mask(count))!=0u; }
    if count==32u { return c.fraction_lo!=0u; }
    if count<64u {
        return c.fraction_lo!=0u || (c.fraction_hi&low_mask(count-32u))!=0u;
    }
    if count==64u { return c.fraction_lo!=0u || c.fraction_hi!=0u; }
    return c.fraction_lo!=0u || c.fraction_hi!=0u
        || (c.seconds&low_mask(count-64u))!=0u;
}
fn clock_to_f32(c: Clock) -> f32 {
    if c.seconds==0u && c.fraction_hi==0u && c.fraction_lo==0u { return 0.0; }
    var top=0u;
    if c.seconds!=0u { top=64u+highest_bit(c.seconds); }
    else if c.fraction_hi!=0u { top=32u+highest_bit(c.fraction_hi); }
    else { top=highest_bit(c.fraction_lo); }
    var significand=0u; var discarded=0u;
    if top<23u { significand=c.fraction_lo<<(23u-top); }
    else {
        discarded=top-23u;
        for (var k=0u; k<24u; k+=1u) {
            significand|=clock_bit(c,discarded+k)<<k;
        }
    }
    var exponent=top+63u;
    if discarded!=0u {
        let guard=clock_bit(c,discarded-1u)!=0u;
        let sticky=any_bits_below(c,discarded-1u);
        if guard && (sticky || (significand&1u)!=0u) {
            significand+=1u;
            if significand>=0x01000000u { significand>>=1u; exponent+=1u; }
        }
    }
    return bitcast<f32>((exponent<<23u)|(significand&0x007fffffu));
}
fn add_clock_parts(base: Clock, dt: f32) -> Clock {
    // dt = significand*2^(raw_exp-150), hence its Q64 tick count is the
    // integer significand shifted by raw_exp-86. For dt>=1 ns raw_exp>=97.
    let bits=bitcast<u32>(dt);
    let raw_exp=(bits>>23u)&0xffu;
    let significand=(bits&0x007fffffu)|0x00800000u;
    var add_seconds=0u; var add_hi=0u; var add_lo=0u;
    if (bits&0x80000000u)==0u && raw_exp!=0u && raw_exp<0xffu {
        if raw_exp>=86u {
            let shift=raw_exp-86u;
            if shift<32u {
                add_lo=significand<<shift;
                if shift>8u { add_hi=significand>>(32u-shift); }
            } else if shift<64u {
                add_hi=significand<<(shift-32u);
                if shift>40u { add_seconds=significand>>(64u-shift); }
            } else if shift<=72u {
                add_seconds=significand<<(shift-64u);
            }
        } else {
            let right=86u-raw_exp;
            if right<24u {
                add_lo=significand>>right;
                let remainder=significand&low_mask(right);
                let halfway=1u<<(right-1u);
                if remainder>halfway || (remainder==halfway && (add_lo&1u)!=0u) { add_lo+=1u; }
            }
        }
    }
    let next_low=base.fraction_lo+add_lo;
    let carry_low=select(0u,1u,next_low<base.fraction_lo);
    let high_partial=base.fraction_hi+add_hi;
    let carry_high0=select(0u,1u,high_partial<base.fraction_hi);
    let next_high=high_partial+carry_low;
    let carry_high1=select(0u,1u,next_high<high_partial);
    return Clock(base.seconds+add_seconds+carry_high0+carry_high1,next_high,next_low);
}
fn clock_seconds() -> f32 { return clock_to_f32(clock_parts()); }
fn clock_plus(dt: f32) -> f32 { return clock_to_f32(add_clock_parts(clock_parts(),dt)); }
fn add_clock(dt: f32) {
    let next=add_clock_parts(clock_parts(),dt);
    control.time_seconds=next.seconds;
    control.time_fraction_hi=next.fraction_hi;
    control.time_fraction_lo=next.fraction_lo;
}
// Accepted-state closure. The RK residual's algebraic slots belong to its
// stage-4 input, so health derives every coefficient anew from the solved
// fields. Return (rho, chi pressure mass, rho_dT, d_p).
fn allmach_closure(cell: u32, for_rhie_chow: bool) -> vec4<f32> {
    let p=s(cell,params.p_off); let rho_ref=bitcast<f32>(constants_words[5]);
    let speed2=s(cell,params.u_off)*s(cell,params.u_off)+s(cell,params.u_off+1u)*s(cell,params.u_off+1u);
    let uref=s(cell,params.u_ref_off); let beta2=max(max(speed2,uref*uref),1.0e-12);
    let mask=s(cell,params.precond_mask_off);
    if params.thermal==0u {
        let psi=max(s(cell,params.psi_off),0.0);
        let rho_raw=rho_ref+psi*p;
        let rho=select(rho_raw,max(rho_raw,psi*1.0e-5),for_rhie_chow);
        let mass_pp=psi+mask*(max(psi,1.0/beta2)-psi);
        var dp=max(s(cell,params.dt_local_off),0.0)/max(rho,1.0e-12);
        if params.penalty_off != 0xffffffffu { dp=dp/(1.0+abs(s(cell,params.penalty_off))*dp); }
        return vec4<f32>(rho,mass_pp,0.0,dp);
    }
    let temp=s(cell,params.temperature_off);
    let psi0=max(s(cell,params.psi_ref_off),0.0); let tref=s(cell,params.t_ref_off);
    let gamma=bitcast<f32>(constants_words[14]);
    let gm1=bitcast<f32>(constants_words[15]);
    let numerator=rho_ref*tref+gamma*psi0*tref*p;
    let rho=max(numerator/temp,psi0*1.0e-5);
    let b=-numerator/(temp*temp);
    let psi_local=max(psi0*tref/temp,psi0);
    let mass_pp=mask*(gm1*psi0*tref/temp+max(psi_local,1.0/beta2));
    var dp=max(s(cell,params.dt_local_off),0.0)/max(rho,1.0e-30);
    if params.penalty_off != 0xffffffffu { dp=dp/(1.0+abs(s(cell,params.penalty_off))*dp); }
    return vec4<f32>(rho,mass_pp,b,dp);
}
fn pressure_bc(face: u32) -> vec2<f32> {
    let i = face * params.unknown_stride + params.p_component;
    return vec2<f32>(f32(bc_kind[i]), bc_value[i]);
}
fn set_active(enabled: bool) {
    if enabled {
        dispatch_args[0] = control.cells_x; dispatch_args[1] = control.cells_y; dispatch_args[2] = 1u;
        dispatch_args[3] = control.faces_x; dispatch_args[4] = control.faces_y; dispatch_args[5] = 1u;
    } else {
        dispatch_args[0] = 0u; dispatch_args[1] = 1u; dispatch_args[2] = 1u;
        dispatch_args[3] = 0u; dispatch_args[4] = 1u; dispatch_args[5] = 1u;
    }
}
fn write_stage_time(a: f32, first: bool) {
    if control.halt != 0u {
        set_active(false);
        dispatch_args[6]=0u; dispatch_args[7]=1u; dispatch_args[8]=1u;
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

@compute @workgroup_size(1) fn batch_reset() { control.accepted_batch=0u; }

@compute @workgroup_size(64) fn volume_audit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell=idx2(gid); if cell>=params.num_cells { return; }
    let volume=cell_vols[cell];
    if !finite(volume) || !(volume>0.0) { atomicAdd(&control.invalid_count,1u); }
}

@compute @workgroup_size(1) fn volume_finalize() {
    if atomicLoad(&control.invalid_count)!=0u {
        control.halt=1u;
        set_active(false);
        dispatch_args[6]=0u; dispatch_args[7]=1u; dispatch_args[8]=1u;
    }
}

// Test-only pipeline is compiled only by cfg(test) host code. Keeping the
// arithmetic text identical to health_finalize catches layout or shader drift.
@compute @workgroup_size(1) fn time_accumulation_probe() {
    control.time_seconds=0u; control.time_fraction_hi=0u; control.time_fraction_lo=0u;
    let tiny_dt=bitcast<f32>(0x30a9ad7fu);
    for (var i=0u; i<100003u; i++) { add_clock(tiny_dt); }
    let probe_dt=bitcast<f32>(0x3157e37cu);
    control.next_dt=clock_plus(0.0);
    control.last_dt=clock_plus(0.5*probe_dt);
    atomicStore(&control.max_base_bits,bitcast<u32>(clock_plus(probe_dt)));
}

// Exercise awkward f32 mantissas around 1 ns while crossing an integer-second
// boundary. The host reproduces this wrapping LCG and checks both the exact
// Q64 status value and the f32 stage clock bit-for-bit.
@compute @workgroup_size(1) fn time_random_sequence_probe() {
    control.time_seconds=31u; control.time_fraction_hi=0xfffff000u; control.time_fraction_lo=0u;
    var seed=0x6d2b79f5u;
    for (var i=0u; i<4096u; i++) {
        seed=seed*1664525u+1013904223u;
        add_clock(bitcast<f32>(0x3089705fu+(seed&0x001fffffu)));
    }
    seed=seed*1664525u+1013904223u;
    let probe_dt=bitcast<f32>(0x3089705fu+(seed&0x001fffffu));
    control.next_dt=clock_plus(0.0);
    control.last_dt=clock_plus(0.5*probe_dt);
    atomicStore(&control.max_base_bits,bitcast<u32>(clock_plus(probe_dt)));
}

@compute @workgroup_size(1) fn health_clear() {
    if control.halt != 0u { return; }
    atomicStore(&control.invalid_count, 0u);
    atomicStore(&control.max_base_bits, 0u);
    atomicStore(&control.max_turnover_bits, 0u);
    dispatch_args[6] = 0u; dispatch_args[7] = 1u; dispatch_args[8] = 1u;
}

@compute @workgroup_size(64) fn history_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell=idx2(gid); if cell>=params.num_cells { return; }
    let base=cell*params.state_stride;
    for (var k=0u; k<params.state_stride; k++) {
        history_backup[base+k]=state[base+k];
        state[base+k]=state_old[base+k];
    }
}

@compute @workgroup_size(64) fn pressure_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = idx2(gid);
    if control.halt != 0u || cell >= params.num_cells { return; }
    let p0 = s(cell, params.p_off);
    var mxx = 0.0; var mxy = 0.0; var myy = 0.0; var bx = 0.0; var by = 0.0;
    let begin = cell_face_offsets[cell]; let end = cell_face_offsets[cell + 1u];
    for (var q = begin; q < end; q++) {
        let face = cell_faces[q]; let owner = face_owner[face]; let neighbor = face_neighbor[face];
        var ux = 0.0; var uy = 0.0; var rhs = 0.0; var have = true;
        if neighbor != 0xffffffffu {
            var other = owner; var d = cell_centers[owner] - cell_centers[cell];
            if owner == cell { other = neighbor; d = cell_centers[neighbor] + face_wrap_shift[face] - cell_centers[cell]; }
            else { d = cell_centers[owner] - face_wrap_shift[face] - cell_centers[cell]; }
            let distance = length(d);
            if distance > 1.0e-12 { ux = d.x / distance; uy = d.y / distance; rhs = (s(other, params.p_off) - p0) / distance; }
            else { have = false; }
        } else {
            let bc = pressure_bc(face);
            if u32(bc.x) == 1u {
                let d = face_centers[face] - cell_centers[cell]; let distance = length(d);
                if distance > 1.0e-12 { ux = d.x / distance; uy = d.y / distance; rhs = (bc.y - p0) / distance; }
                else { have = false; }
            } else {
                let sign = select(-1.0, 1.0, owner == cell);
                ux = sign * face_normals[face].x; uy = sign * face_normals[face].y;
            }
        }
        if have { mxx += ux*ux; mxy += ux*uy; myy += uy*uy; bx += ux*rhs; by += uy*rhs; }
    }
    let det = mxx*myy - mxy*mxy; let trace = mxx + myy; let floor = 1.0e-5 * max(trace*trace, 1.0e-12);
    var g = vec2<f32>(0.0);
    if det > floor { g = vec2<f32>((myy*bx-mxy*by)/det, (mxx*by-mxy*bx)/det); }
    else if trace > 1.0e-12 { g = vec2<f32>(bx/trace, by/trace); }
    if !finite(g.x) || !finite(g.y) { atomicAdd(&control.invalid_count, 1u); g = vec2<f32>(0.0); }
    grad_p[cell] = g;
}

@compute @workgroup_size(64) fn face_turnover(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face = idx2(gid);
    if control.halt != 0u || face >= params.num_faces { return; }
    let owner = face_owner[face]; let neighbor = face_neighbor[face]; let area = abs(face_areas[face]);
    let closure_o=allmach_closure(owner,true); let rho_o=closure_o.x; let dp_o=closure_o.w; let k_o=rho_o*dp_o;
    var flux = 0.0;
    if neighbor != 0xffffffffu {
        let d = cell_centers[neighbor] + face_wrap_shift[face] - cell_centers[owner];
        let projected = abs(dot(d, face_normals[face])); let distance = select(max(length(d),1.0e-6), projected, projected > 1.0e-6);
        let do_ = abs(dot(face_centers[face]-cell_centers[owner], face_normals[face]));
        let dn = abs(dot(cell_centers[neighbor]+face_wrap_shift[face]-face_centers[face], face_normals[face]));
        let lambda = select(0.5, dn/(do_+dn), do_+dn > 1.0e-6); let other = 1.0-lambda;
        let closure_n=allmach_closure(neighbor,true); let rho_n=closure_n.x; let dp_n=closure_n.w; let k_n=rho_n*dp_n;
        let kappa = lambda*k_o + other*k_n; let q = lambda*k_o*grad_p[owner] + other*k_n*grad_p[neighbor];
        flux = area * (dot(q,d)/distance - kappa*(s(neighbor,params.p_off)-s(owner,params.p_off))/distance);
        if params.penalty_off != 0xffffffffu {
            let po = abs(s(owner,params.penalty_off)); let pn = abs(s(neighbor,params.penalty_off));
            let seal = 1.0-min(po+pn,1.0); let kp = (lambda*rho_o+other*rho_n)*min(dp_o,dp_n);
            let g = lambda*grad_p[owner]+other*grad_p[neighbor];
            flux = seal*area*(kp*dot(g,d)/distance-kp*(s(neighbor,params.p_off)-s(owner,params.p_off))/distance);
        }
    } else {
        let bc = pressure_bc(face);
        if u32(bc.x) == 1u {
            let d = face_centers[face]-cell_centers[owner]; let projected=abs(dot(d,face_normals[face]));
            let distance=select(max(length(d),1.0e-6),projected,projected>1.0e-6); var seal=1.0;
            if params.penalty_off != 0xffffffffu { seal=1.0-min(2.0*abs(s(owner,params.penalty_off)),1.0); }
            flux=seal*area*(dot(k_o*grad_p[owner],d)/distance-k_o*(bc.y-s(owner,params.p_off))/distance);
        }
    }
    let mag=abs(flux); face_magnitude[face]=select(0.0,mag,finite(mag));
    if !finite(mag) { atomicAdd(&control.invalid_count,1u); }
}

@compute @workgroup_size(64) fn cell_rate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell=idx2(gid); if control.halt != 0u || cell >= params.num_cells { return; }
    let base=cell*params.state_stride;
    for (var k=0u; k<params.state_stride; k++) { if !finite(state[base+k]) { atomicAdd(&control.invalid_count,1u); return; } }
    let ux=s(cell,params.u_off); let uy=s(cell,params.u_off+1u); let speed=length(vec2<f32>(ux,uy));
    let closure=allmach_closure(cell,false); let rho=closure.x; let mass_pp=closure.y; let b=closure.z; let local_dp=closure.w;
    var chi=mass_pp; var domain_valid=rho>0.0 && finite(rho); var alpha_t=0.0;
    if params.thermal!=0u {
        let psi_ref=s(cell,params.psi_ref_off); let t_ref=s(cell,params.t_ref_off); let temp=s(cell,params.temperature_off);
        let gm1=bitcast<f32>(constants_words[15]); let inv_cp=gm1*psi_ref*t_ref;
        chi=mass_pp+b*inv_cp/max(rho,1.0e-30);
        let rho_floor=s(cell,params.rho_floor_off); let gamma=bitcast<f32>(constants_words[14]);
        let rho_ref=bitcast<f32>(constants_words[5]);
        let rho_raw=(rho_ref*t_ref+gamma*psi_ref*t_ref*s(cell,params.p_off))/temp;
        domain_valid=domain_valid && temp>0.0 && rho_raw>rho_floor*(1.0+1.0e-6);
        alpha_t=0.01*mass_pp/(rho*chi);
    }
    if !(domain_valid && mass_pp>0.0 && finite(chi) && chi>abs(mass_pp)*1.0e-6) {
        atomicAdd(&control.invalid_count,1u); return;
    }
    if control.adaptive==0u { return; }
    var area_sum=0.0; var area_over_distance=0.0; var turnover_sum=0.0;
    let begin=cell_face_offsets[cell]; let end=cell_face_offsets[cell+1u];
    for (var q=begin; q<end; q++) {
        let face=cell_faces[q]; let area=abs(face_areas[face]); area_sum+=area; turnover_sum+=face_magnitude[face];
        let owner=face_owner[face]; let neighbor=face_neighbor[face]; var d=face_centers[face]-cell_centers[cell];
        if neighbor != 0xffffffffu {
            if owner==cell { d=cell_centers[neighbor]+face_wrap_shift[face]-cell_centers[cell]; }
            else { d=cell_centers[owner]-face_wrap_shift[face]-cell_centers[cell]; }
        }
        area_over_distance += area/max(abs(dot(d,face_normals[face])),1.0e-12);
    }
    let vol=cell_vols[cell];
    if !finite(vol) || !(vol>0.0) { atomicAdd(&control.invalid_count,1u); return; }
    let hyper=0.5*area_sum/vol; let diff=area_over_distance/vol;
    let sound=inverseSqrt(chi); let viscosity=abs(bitcast<f32>(constants_words[4]));
    let nu_long=4.0*viscosity/(3.0*rho); let alpha_p=rho*abs(local_dp)/chi;
    let rate=hyper*(speed+sound)+2.0*diff*max(nu_long,max(alpha_t,alpha_p));
    let turnover=turnover_sum/(rho*vol);
    if finite(rate) && rate>=0.0 && finite(turnover) && turnover>=0.0 {
        atomicMax(&control.max_base_bits,bitcast<u32>(rate)); atomicMax(&control.max_turnover_bits,bitcast<u32>(turnover));
    } else { atomicAdd(&control.invalid_count,1u); }
}

@compute @workgroup_size(1) fn health_finalize() {
    if control.halt != 0u { set_active(false); return; }
    if control.fail_after_accepted != 0xffffffffu
        && (control.accepted_total_hi != 0u || control.accepted_total_lo >= control.fail_after_accepted) {
        atomicAdd(&control.invalid_count,1u);
    }
    if (control.accepted_total_lo==0xffffffffu && control.accepted_total_hi==0xffffffffu)
        || control.accepted_batch==0xffffffffu {
        atomicAdd(&control.invalid_count,1u);
    }
    let invalid=atomicLoad(&control.invalid_count); let base=bitcast<f32>(atomicLoad(&control.max_base_bits));
    let turnover=bitcast<f32>(atomicLoad(&control.max_turnover_bits)); let total=base+turnover;
    control.total_rate=total;
    if invalid != 0u || !finite(total) || (control.adaptive!=0u && !(total>0.0)) {
        control.halt=1u; set_active(false);
        dispatch_args[6]=control.cells_x; dispatch_args[7]=control.cells_y; dispatch_args[8]=1u;
        constants_words[0]=bitcast<u32>(control.next_dt); constants_words[3]=bitcast<u32>(clock_seconds());
        return;
    }
    // Q64 fractional seconds represent every operational f32 timestep down to
    // 1 ns exactly while the u32 seconds word supplies ~136 years of range;
    // accumulation is integer and immune to backend fast-math reassociation.
    add_clock(control.step_dt);
    control.dt_old=control.step_dt; control.last_dt=control.step_dt;
    let old_total_lo=control.accepted_total_lo;
    control.accepted_total_lo+=1u;
    if old_total_lo==0xffffffffu { control.accepted_total_hi+=1u; }
    control.accepted_batch += 1u;
    if control.adaptive != 0u {
        let stable=control.safety*clamp(control.target_cfl,1.0e-6,1.0)/total;
        control.next_dt=min(control.max_dt,min(control.step_dt*control.growth,stable));
    } else {
        control.next_dt=control.step_dt;
    }
    constants_words[0]=bitcast<u32>(control.next_dt); constants_words[1]=bitcast<u32>(control.dt_old); constants_words[3]=bitcast<u32>(clock_seconds());
    dispatch_args[6]=0u; dispatch_args[7]=1u; dispatch_args[8]=1u;
}

@compute @workgroup_size(64) fn rollback_invalid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell=idx2(gid); if cell>=params.num_cells { return; }
    let base=cell*params.state_stride;
    for (var k=0u; k<params.state_stride; k++) { state[base+k]=history_backup[base+k]; }
}
"#;
