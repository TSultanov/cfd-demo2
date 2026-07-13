//! Structured (uniform Cartesian, dense-array) **GPU** solver path.
//!
//! This is the GPU counterpart of [`crate::solver::cpu::structured`]. Where the
//! unstructured GPU backend ([`crate::solver::gpu::GpuUnifiedSolver`]) builds a
//! face-connectivity mesh, a block-CSR and the FGMRES/AMG stack, this path holds
//! the grid as a *dense array with no connectivity indirection*: `state[j*nx+i]`,
//! neighbours `p±1`/`p±nx`, and a fixed 5-point **band** operator
//! (`matrix_values` sized `N*5*s*s`, ranks `[S, W, diag, E, N]`).
//!
//! It runs the *same* codegen-emitted `TopologyMode::Structured2D` kernels the CPU
//! interpreter and the checked-in WGSL use — assembly / flux / gradients /
//! `bc_expr` / update — on the GPU, then closes the loop with a **matrix-free
//! banded** linear solve (block-Jacobi preconditioned CG for the SPD scalar
//! system, block BiCGStab for the indefinite coupled U–p system). No CSR buffers,
//! no `row_offsets`/`col_indices`, no mesh topology — the operator expansion is
//! produced arithmetically by the IR, exactly as on the CPU path.
//!
//! Obstacles are immersed (Brinkman penalisation), never cut out of the grid.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::recipe::{KernelPhase, SolverRecipe, SteppingMode};
use crate::solver::gpu::structs::GpuConstants;
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::model::eos::EosRuntimeParams;
use crate::solver::model::kernel::ModelKernelArtifact;
use crate::solver::model::module::ModelModule;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use crate::solver::{PreconditionerType, TimeScheme};
use cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl;
use cfd2_ir::kernel::BindingAccess;

const WG: u32 = 64;

/// Runtime-EOS families covered by the density-based structured controller's
/// conserved-domain proof. Keep this predicate separate from presentation
/// policy so unsupported fluids cannot become autonomous merely by selecting
/// the Direct renderer.
fn structured_conserved_audit_certifies_eos(params: EosRuntimeParams) -> bool {
    params.gamma.is_finite()
        && params.gamma > 0.0
        && params.gm1.is_finite()
        && params.gm1 > 0.0
        && params.r.is_finite()
        && params.r > 0.0
        && params.theta_ref.is_finite()
        && params.theta_ref > 0.0
        && params.dp_drho == 0.0
        && params.p_ref == 0.0
        && params.rho_ref == 0.0
}

#[cfg(test)]
mod structured_eos_audit_tests {
    use super::*;
    use crate::solver::model::eos::EosSpec;

    #[test]
    fn conserved_autonomous_audit_advertises_only_certified_ideal_gas_eos() {
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let water = EosSpec::LinearCompressibility {
            bulk_modulus: 2.2e9,
            rho_ref: 1000.0,
            p_ref: 1.0e5,
        };
        assert!(structured_conserved_audit_certifies_eos(
            air.runtime_params()
        ));
        assert!(!structured_conserved_audit_certifies_eos(
            water.runtime_params()
        ));
        assert!(!structured_conserved_audit_certifies_eos(
            EosSpec::Constant.runtime_params()
        ));
    }
}

// Band ranks in the fixed 5-point stencil row (ascending column order for the
// row-major cell numbering `p = j*nx + i`): `[S, W, diag, E, N]`. Must match the
// codegen structured assembly + the CPU `BandedBlockOperator`.
const BAND_STRIDE: usize = 5;

// ===========================================================================
// Shared dense-grid geometry (topology-agnostic; used by both CPU and GPU).
// ===========================================================================

/// A uniform Cartesian grid of `nx * ny` cells over `[0,length] x [0,height]`.
/// Dense: cell `p = j*nx + i`, neighbours `p±1` / `p±nx`, no per-cell/face arrays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructuredGrid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
}

impl StructuredGrid {
    /// A grid spanning `[0, length] x [0, height]` with `nx * ny` cells.
    pub fn new(nx: usize, ny: usize, length: f64, height: f64) -> Self {
        assert!(
            nx > 0 && ny > 0,
            "grid must have at least one cell per axis"
        );
        assert!(
            length > 0.0 && height > 0.0,
            "grid extents must be positive"
        );
        Self {
            nx,
            ny,
            dx: length / nx as f64,
            dy: height / ny as f64,
        }
    }

    #[inline]
    pub fn num_cells(&self) -> usize {
        self.nx * self.ny
    }

    /// Cell centre coordinates for linear index `p = j*nx + i`.
    #[inline]
    pub fn cell_center(&self, p: usize) -> (f64, f64) {
        let i = p % self.nx;
        let j = p / self.nx;
        ((i as f64 + 0.5) * self.dx, (j as f64 + 0.5) * self.dy)
    }
}

/// Which domain edge a boundary face sits on (used to key the per-face BC).
#[derive(Clone, Copy, Debug)]
pub enum Edge {
    Left,
    Right,
    Bottom,
    Top,
}

/// Boundary condition (kind, value) for one unknown component on one face.
/// `kind`: 0 = none/interior, 1 = Dirichlet, 2 = Neumann (matches `GpuBcKind`).
#[derive(Clone, Copy)]
pub struct BcComp {
    pub kind: u32,
    pub value: f32,
}

/// The `grid: StructuredGrid` uniform the codegen structured kernels bind at
/// `@group(0) @binding(0)`. Layout must match the WGSL struct exactly
/// (`nx:u32, ny:u32, dx:f32, dy:f32`, 16 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuStructuredGrid {
    pub nx: u32,
    pub ny: u32,
    pub dx: f32,
    pub dy: f32,
}

// ===========================================================================
// A single codegen kernel compiled to a GPU pipeline, with its (group,binding)
// -> buffer-name reflection from the emitted WGSL.
// ===========================================================================

/// One codegen kernel binding: group/binding slot, buffer name, and buffer kind.
#[derive(Clone)]
struct BindInfo {
    group: u32,
    binding: u32,
    name: String,
    access: BindingAccess,
}

struct CompiledKernel {
    label: String,
    pipeline: wgpu::ComputePipeline,
    bindings: Vec<BindInfo>,
    /// Explicit per-group bind-group layouts (ascending group). Explicit (rather
    /// than pipeline auto-layout) so bindings the shader declares but does not
    /// statically use are still present — the codegen assembly binds e.g.
    /// `state_iter`/`state_old_old` that pure diffusion never reads.
    layouts: Vec<(u32, wgpu::BindGroupLayout)>,
    /// This kernel's `constants` uniform, packed to ITS `Constants` struct layout
    /// (base fields + this kernel's declared EOS params in order). Most kernels
    /// declare the full canonical EOS block (= a `GpuConstants` prefix), but
    /// `bc_expr` declares only the EOS params its expressions reference, so a
    /// shared buffer would misalign its `eos_*` reads. Per-kernel packing fixes it.
    constants_buf: wgpu::Buffer,
    eos_fields: Vec<String>,
    /// Buffer topology is immutable for the lifetime of a structured solver.
    /// Cache reflected bind groups once instead of rebuilding 1-3 groups for
    /// every kernel dispatch of every RK stage.
    bind_groups: std::sync::OnceLock<Vec<(u32, wgpu::BindGroup)>>,
}

/// One generated-kernel dispatch schedule together with immutable offsets into
/// an RK batch's constant-snapshot buffer.  A generated kernel owns a distinct
/// `Constants` layout (notably `bc_expr` may declare only a subset of EOS
/// fields), so every copy retains that kernel's exact packed byte count.
struct RkStageSnapshot {
    schedule: Vec<String>,
    constant_copies: Vec<(String, u64, u64)>,
}

struct RkStepSnapshot {
    stages: Vec<RkStageSnapshot>,
}

/// Tiny GPU-resident control/status block for bounded autonomous RK4 batches.
/// Atomic WGSL fields are represented by their underlying `u32` host storage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StructuredAutonomousControlRaw {
    dt: f32,
    dt_old: f32,
    time_seconds: u32,
    time_fraction_hi: u32,
    time_fraction_lo: u32,
    max_base_bits: u32,
    max_rc_bits: u32,
    max_vel_bits: u32,
    invalid_cells: u32,
    halted: u32,
    accepted_batch: u32,
    accepted_total: u32,
    accepted_total_hi: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StructuredAutonomousParams {
    target_cfl: f32,
    density: f32,
    viscosity: f32,
    safety: f32,
    dt_max: f32,
    adaptive: u32,
    bootstrap_dt: u32,
    state_stride: u32,
    inject_nan_after: u32,
    inject_negative_energy_after: u32,
    adaptive_policy: u32,
    has_ibm_penalty: u32,
    inlet_velocity: f32,
    eos_gamma: f32,
    eos_gm1: f32,
    eos_r: f32,
    eos_dp_drho: f32,
    eos_p_ref: f32,
    eos_theta_ref: f32,
    eos_rho_ref: f32,
    // Gauge-storage references: the conserved-state audit reconstructs the
    // ABSOLUTE thermodynamic state as rho + gauge_rho_ref / rho_e + gauge_e_ref
    // (zero = absolute storage).
    eos_gauge_rho_ref: f32,
    eos_gauge_e_ref: f32,
    /// ABSOLUTE floored-recovery thermodynamic floors (1 Pa / 1 K on gauged
    /// production runs; hugely negative = inert otherwise). The audit clamps
    /// its reconstructed p/T with these instead of rejecting the step.
    eos_p_floor_abs: f32,
    eos_t_floor: f32,
    /// STORED-form conserved-density floor for the in-place repair.
    eos_rho_floor: f32,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructuredAdaptivePolicy {
    Unsupported = 0,
    AllMachSpectral = 1,
    CompressibleHostCfl = 2,
}

const STRUCTURED_STATUS_RING_SLOTS: usize = 3;

struct StructuredAutonomousBatchCache {
    schedules: Vec<Vec<String>>,
    records: HashMap<String, (u64, u64)>,
    snapshot_bytes: std::sync::Mutex<Vec<u8>>,
    snapshots: wgpu::Buffer,
    _record_words: wgpu::Buffer,
    patch_bg: wgpu::BindGroup,
    patch_groups: u32,
    status_staging: [std::sync::Arc<wgpu::Buffer>; STRUCTURED_STATUS_RING_SLOTS],
    status_slots_busy:
        std::sync::Arc<[std::sync::atomic::AtomicBool; STRUCTURED_STATUS_RING_SLOTS]>,
    next_status_slot: std::sync::atomic::AtomicUsize,
}

struct StructuredStatusStagingLease {
    buffer: std::sync::Arc<wgpu::Buffer>,
    busy: std::sync::Arc<
        [std::sync::atomic::AtomicBool; STRUCTURED_STATUS_RING_SLOTS],
    >,
    slot: usize,
}

impl Drop for StructuredStatusStagingLease {
    fn drop(&mut self) {
        self.busy[self.slot].store(false, std::sync::atomic::Ordering::Release);
    }
}

impl StructuredAutonomousBatchCache {
    fn refresh_snapshots(
        &self,
        queue: &wgpu::Queue,
        constants: &GpuConstants,
        kernels: &HashMap<String, CompiledKernel>,
    ) {
        let mut bytes = self
            .snapshot_bytes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (id, &(offset, size)) in &self.records {
            let start = offset as usize;
            let end = start + size as usize;
            write_kernel_constants_bytes(
                constants,
                &kernels[id].eos_fields,
                &mut bytes[start..end],
            );
        }
        queue.write_buffer(&self.snapshots, 0, &bytes);
    }

    fn acquire_status_staging(&self) -> Result<StructuredStatusStagingLease, String> {
        let start = self
            .next_status_slot
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % STRUCTURED_STATUS_RING_SLOTS;
        for offset in 0..STRUCTURED_STATUS_RING_SLOTS {
            let slot = (start + offset) % STRUCTURED_STATUS_RING_SLOTS;
            if self.status_slots_busy[slot]
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(StructuredStatusStagingLease {
                    buffer: std::sync::Arc::clone(&self.status_staging[slot]),
                    busy: std::sync::Arc::clone(&self.status_slots_busy),
                    slot,
                });
            }
        }
        Err("structured autonomous status readback ring is full (3 batches still in flight)"
            .to_string())
    }
}

/// Completion-fenced telemetry for a structured autonomous batch. It is only
/// 52 bytes on-device; reading it does not transfer the cell state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StructuredAutonomousStatus {
    pub dt: f32,
    pub dt_old: f32,
    pub time: f64,
    pub max_base_rate: f32,
    pub max_rhie_chow_rate: f32,
    pub max_velocity: f32,
    pub invalid_cells: u32,
    pub halted: bool,
    pub accepted_steps: u32,
    pub accepted_total: u64,
}

struct StructuredAutonomousControl {
    control: wgpu::Buffer,
    params: wgpu::Buffer,
    /// Device cap on workgroups per dispatch dimension; control-kernel
    /// dispatches split into (x, y) rows above it, exactly like the stage
    /// kernels' indirect args, and the WGSL flattens via `launch_index`.
    max_workgroups_per_dim: u32,
    _grad_p: wgpu::Buffer,
    _rho_work: wgpu::Buffer,
    _dp_work: wgpu::Buffer,
    _penalty_work: wgpu::Buffer,
    indirect_args: wgpu::Buffer,
    common_bg: wgpu::BindGroup,
    reset_batch: wgpu::ComputePipeline,
    reset_metrics: wgpu::ComputePipeline,
    gradient: wgpu::ComputePipeline,
    sample: wgpu::ComputePipeline,
    rhie_chow: wgpu::ComputePipeline,
    bootstrap: wgpu::ComputePipeline,
    history: wgpu::ComputePipeline,
    accept: wgpu::ComputePipeline,
    rollback: wgpu::ComputePipeline,
    patch_stage_0: wgpu::ComputePipeline,
    patch_stage_half: wgpu::ComputePipeline,
    patch_stage_1: wgpu::ComputePipeline,
    batch_cache: StructuredAutonomousBatchCache,
    #[cfg(feature = "dev-tests")]
    clock_fixed_probe: wgpu::ComputePipeline,
    #[cfg(feature = "dev-tests")]
    clock_random_probe: wgpu::ComputePipeline,
    adaptive_policy: StructuredAdaptivePolicy,
}

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

fn decode_q64_time(raw: &StructuredAutonomousControlRaw) -> f64 {
    let fraction = f64::from(raw.time_fraction_hi) * (1.0 / 4_294_967_296.0)
        + f64::from(raw.time_fraction_lo) * (1.0 / 18_446_744_073_709_551_616.0);
    f64::from(raw.time_seconds) + fraction
}

/// Pack the `constants` uniform for a kernel whose `Constants` struct is the canonical
/// base fields followed by `eos_fields` (in declared order), padded to the WGSL
/// 16-byte uniform alignment.
fn packed_kernel_constants_len(eos_fields: &[String]) -> usize {
    let base_bytes =
        cfd2_codegen::solver::codegen::constants::base_constant_field_names().len() * 4;
    (base_bytes + eos_fields.len() * 4).next_multiple_of(16)
}

fn write_kernel_constants_bytes(c: &GpuConstants, eos_fields: &[String], bytes: &mut [u8]) {
    let base_bytes =
        cfd2_codegen::solver::codegen::constants::base_constant_field_names().len() * 4;
    assert_eq!(bytes.len(), packed_kernel_constants_len(eos_fields));
    bytes.fill(0);
    bytes[..base_bytes].copy_from_slice(&bytemuck::bytes_of(c)[..base_bytes]);
    for (index, field) in eos_fields.iter().enumerate() {
        let value: f32 = match field.as_str() {
            "eos_gamma" => c.eos_gamma,
            "eos_gm1" => c.eos_gm1,
            "eos_r" => c.eos_r,
            "eos_dp_drho" => c.eos_dp_drho,
            "eos_p_ref" => c.eos_p_ref,
            "eos_theta_ref" => c.eos_theta_ref,
            "eos_rho_ref" => c.eos_rho_ref,
            "eos_gauge_rho_ref" => c.eos_gauge_rho_ref,
            "eos_gauge_p_ref" => c.eos_gauge_p_ref,
            "eos_gauge_e_ref" => c.eos_gauge_e_ref,
            "eos_gauge_p_bias" => c.eos_gauge_p_bias,
            "bc_pressure_inlet" => c.bc_pressure_inlet,
            "eos_p_floor" => c.eos_p_floor,
            "eos_t_floor" => c.eos_t_floor,
            "eos_rho_floor" => c.eos_rho_floor,
            "buoyant_beta_g" => c.buoyant_beta_g,
            "buoyant_t0" => c.buoyant_t0,
            "buoyant_k_over_cp" => c.buoyant_k_over_cp,
            _ => 0.0,
        };
        let offset = base_bytes + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn pack_kernel_constants(c: &GpuConstants, eos_fields: &[String]) -> Vec<u8> {
    let mut bytes = vec![0; packed_kernel_constants_len(eos_fields)];
    write_kernel_constants_bytes(c, eos_fields, &mut bytes);
    bytes
}

impl CompiledKernel {
    fn write_constants(&self, queue: &wgpu::Queue, c: &GpuConstants) {
        queue.write_buffer(
            &self.constants_buf,
            0,
            &pack_kernel_constants(c, &self.eos_fields),
        );
    }

    fn build(
        device: &wgpu::Device,
        id: &str,
        wgsl: &str,
        bindings: Vec<BindInfo>,
        eos_fields: Vec<String>,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(id),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let mut groups: Vec<u32> = bindings.iter().map(|b| b.group).collect();
        groups.sort_unstable();
        groups.dedup();
        let max_g = groups.iter().copied().max().unwrap_or(0);

        let entry_ty = |access: BindingAccess| wgpu::BindingType::Buffer {
            ty: match access {
                BindingAccess::Uniform => wgpu::BufferBindingType::Uniform,
                BindingAccess::ReadOnlyStorage => {
                    wgpu::BufferBindingType::Storage { read_only: true }
                }
                BindingAccess::ReadWriteStorage => {
                    wgpu::BufferBindingType::Storage { read_only: false }
                }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        };

        let mut layouts: Vec<(u32, wgpu::BindGroupLayout)> = Vec::new();
        for &g in &groups {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
                .iter()
                .filter(|b| b.group == g)
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: entry_ty(b.access),
                    count: None,
                })
                .collect();
            layouts.push((
                g,
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(id),
                    entries: &entries,
                }),
            ));
        }
        // The pipeline layout is indexed by group number; fill any gap with an
        // empty layout so the array is contiguous 0..=max_g.
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("structured:empty"),
            entries: &[],
        });
        let ordered: Vec<Option<&wgpu::BindGroupLayout>> = (0..=max_g)
            .map(|g| {
                Some(
                    layouts
                        .iter()
                        .find(|(lg, _)| *lg == g)
                        .map(|(_, l)| l)
                        .unwrap_or(&empty),
                )
            })
            .collect();
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(id),
            bind_group_layouts: &ordered,
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(id),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let base_bytes =
            cfd2_codegen::solver::codegen::constants::base_constant_field_names().len() * 4;
        let cbytes = base_bytes + eos_fields.len() * 4;
        let csize = ((cbytes + 15) / 16 * 16).max(16) as u64;
        let constants_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(id),
            size: csize,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            label: id.to_string(),
            pipeline,
            bindings,
            layouts,
            constants_buf,
            eos_fields,
            bind_groups: std::sync::OnceLock::new(),
        }
    }

    fn cached_bind_groups<'kernel, 'buffer>(
        &'kernel self,
        device: &wgpu::Device,
        resolve: &impl Fn(&str) -> &'buffer wgpu::Buffer,
    ) -> &'kernel [(u32, wgpu::BindGroup)] {
        self.bind_groups
            .get_or_init(|| {
                self.layouts
                    .iter()
                    .map(|(g, layout)| {
                        let entries: Vec<wgpu::BindGroupEntry> = self
                            .bindings
                            .iter()
                            .filter(|b| b.group == *g)
                            .map(|b| wgpu::BindGroupEntry {
                                binding: b.binding,
                                resource: if b.name == "constants" {
                                    self.constants_buf.as_entire_binding()
                                } else {
                                    resolve(&b.name).as_entire_binding()
                                },
                            })
                            .collect();
                        (
                            *g,
                            device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("structured:bg"),
                                layout,
                                entries: &entries,
                            }),
                        )
                    })
                    .collect()
            })
            .as_slice()
    }

    fn dispatch_in_pass<'pass, 'buffer>(
        &'pass self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'pass>,
        n_threads: u32,
        resolve: &impl Fn(&str) -> &'buffer wgpu::Buffer,
    ) {
        let bind_groups = self.cached_bind_groups(device, resolve);
        pass.set_pipeline(&self.pipeline);
        for (g, bg) in bind_groups {
            pass.set_bind_group(*g, bg, &[]);
        }
        let groups = n_threads.div_ceil(WG).max(1);
        let max_x = device.limits().max_compute_workgroups_per_dimension;
        let groups_x = groups.min(max_x);
        let groups_y = groups.div_ceil(max_x);
        assert!(
            groups_y <= max_x,
            "structured dispatch requires more than a 2D workgroup grid: groups={groups}, max={max_x}"
        );
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        crate::count_dispatch!("Structured Kernel", &self.label);
    }

    fn dispatch_in_pass_indirect<'pass, 'buffer>(
        &'pass self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'pass>,
        indirect_args: &'pass wgpu::Buffer,
        resolve: &impl Fn(&str) -> &'buffer wgpu::Buffer,
    ) {
        let bind_groups = self.cached_bind_groups(device, resolve);
        pass.set_pipeline(&self.pipeline);
        for (group, bind_group) in bind_groups {
            pass.set_bind_group(*group, bind_group, &[]);
        }
        pass.dispatch_workgroups_indirect(indirect_args, 0);
        crate::count_dispatch!("Structured Kernel", &self.label);
    }

    /// Exact pre-batching dispatch path retained as a benchmark ablation.  It
    /// deliberately rebuilds the reflected bind groups for every invocation,
    /// matching the old structured router rather than sharing the cache above.
    fn dispatch_uncached<'a>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        n_threads: u32,
        resolve: &impl Fn(&str) -> &'a wgpu::Buffer,
    ) {
        let mut bind_groups = Vec::new();
        for (g, layout) in &self.layouts {
            let entries: Vec<wgpu::BindGroupEntry> = self
                .bindings
                .iter()
                .filter(|b| b.group == *g)
                .map(|b| wgpu::BindGroupEntry {
                    binding: b.binding,
                    resource: if b.name == "constants" {
                        self.constants_buf.as_entire_binding()
                    } else {
                        resolve(&b.name).as_entire_binding()
                    },
                })
                .collect();
            bind_groups.push((
                *g,
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("structured:bg:legacy"),
                    layout,
                    entries: &entries,
                }),
            ));
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("structured:pass:legacy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (g, bg) in &bind_groups {
            pass.set_bind_group(*g, bg, &[]);
        }
        let groups = n_threads.div_ceil(WG).max(1);
        let max_x = device.limits().max_compute_workgroups_per_dimension;
        let groups_x = groups.min(max_x);
        let groups_y = groups.div_ceil(max_x);
        assert!(
            groups_y <= max_x,
            "structured dispatch requires more than a 2D workgroup grid: groups={groups}, max={max_x}"
        );
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        crate::count_dispatch!("Structured Kernel", &self.label);
    }

    /// Record a dispatch of this kernel over `n_threads` (1D, workgroup 64),
    /// binding each declared name via `resolve`.
    fn dispatch<'a>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        n_threads: u32,
        resolve: &impl Fn(&str) -> &'a wgpu::Buffer,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("structured:pass"),
            timestamp_writes: None,
        });
        self.dispatch_in_pass(device, &mut pass, n_threads, resolve);
    }
}

fn structured_autonomous_wgsl(
    model_id: &str,
    layout: &crate::solver::model::backend::state_layout::StateLayout,
    unknowns: usize,
) -> (String, StructuredAdaptivePolicy) {
    let off = |name: &str| layout.offset_for(name).map(|value| value as usize);
    let allmach = if model_id == "allmach_thermal_structured" {
        match (
            off("U"),
            off("p"),
            off("T"),
            off("t_ref"),
            off("dt_local"),
            off("psi_ref"),
            off("u_ref"),
            off("ibm_penalty_U"),
        ) {
            (
                Some(u),
                Some(p),
                Some(t),
                Some(t_ref),
                Some(dt_local),
                Some(psi),
                Some(u_ref),
                Some(penalty),
            ) => Some((u, p, t, t_ref, dt_local, psi, u_ref, penalty)),
            _ => None,
        }
    } else {
        None
    };
    let compressible = if model_id == "compressible_structured" {
        match (off("rho"), off("rho_u"), off("rho_e")) {
            (Some(rho), Some(rho_u), Some(rho_e)) => {
                Some((rho, rho_u, rho_e, off("ibm_penalty_U").is_some()))
            }
            _ => None,
        }
    } else {
        None
    };
    let adaptive_policy = if allmach.is_some() {
        StructuredAdaptivePolicy::AllMachSpectral
    } else if compressible.is_some() {
        StructuredAdaptivePolicy::CompressibleHostCfl
    } else {
        StructuredAdaptivePolicy::Unsupported
    };

    let sample_body = if let Some((u, p, t, t_ref, dt_local, psi, u_ref, penalty)) = allmach {
        format!(
            r#"
    let base = cell * params.state_stride;
    var bad = false;
    for (var slot = 0u; slot < params.state_stride; slot += 1u) {{
        bad = bad || !finite_f32(state.data[base + slot]);
    }}
    if (bad) {{
        atomicAdd(&control.invalid_cells, 1u);
        rho_work.data[cell] = 0.0;
        dp_work.data[cell] = 0.0;
        penalty_work.data[cell] = 0.0;
        return;
    }}
    let ux = state.data[base + {u}u];
    let uy = state.data[base + {uy}u];
    let speed = sqrt(ux * ux + uy * uy);
    let pressure = state.data[base + {p}u];
    let temperature = state.data[base + {t}u];
    let reference_t = state.data[base + {t_ref}u];
    let psi0 = max(state.data[base + {psi}u], 0.0);
    let rho_numer = params.density * reference_t + 1.4 * psi0 * reference_t * pressure;
    let density_floor = psi0 * 1.0e-5;
    let rho_raw = rho_numer / temperature;
    let density = max(rho_raw, density_floor);
    let rho_dt = -rho_numer / (temperature * temperature);
    let psi_local = max(psi0 * reference_t / temperature, psi0);
    let uref = max(state.data[base + {u_ref}u], 0.0);
    let beta2 = max(max(speed * speed, uref * uref), 1.0e-12);
    var mass_pp = 0.0;
    if (psi0 > 0.0) {{
        mass_pp = 0.4 * psi0 * reference_t / temperature + max(psi_local, 1.0 / beta2);
    }}
    let inv_cp = 0.4 * psi0 * reference_t;
    let chi = mass_pp + rho_dt * inv_cp / max(density, 1.0e-30);
    let chi_floor = max(abs(mass_pp), 1.0e-30) * 1.0e-6;
    if (!(temperature > 0.0) || !(density > 0.0) || !(mass_pp > 0.0)
        || !finite_f32(chi) || !(chi > chi_floor)
        || !(rho_raw > density_floor * (1.0 + 1.0e-6))) {{
        atomicAdd(&control.invalid_cells, 1u);
        rho_work.data[cell] = 0.0;
        dp_work.data[cell] = 0.0;
        penalty_work.data[cell] = 0.0;
        return;
    }}
    if (params.adaptive == 0u) {{ return; }}
    let sound = 1.0 / sqrt(chi);
    let nu_long = 4.0 * abs(params.viscosity) / (3.0 * density);
    let alpha_t = 1.0e-2 * mass_pp / (density * chi);
    let dp0 = max(state.data[base + {dt_local}u], 0.0) / density;
    let penalty = abs(state.data[base + {penalty}u]);
    let local_dp = dp0 / (1.0 + penalty * dp0);
    let alpha_p = density * abs(local_dp) / chi;
    let gh = 1.0 / grid.dx + 1.0 / grid.dy;
    let gd = 2.0 / (grid.dx * grid.dx) + 2.0 / (grid.dy * grid.dy);
    let rate = gh * (speed + sound) + 2.0 * gd * max(max(nu_long, alpha_t), alpha_p);
    if (!finite_f32(rate) || !(rate >= 0.0)) {{
        atomicAdd(&control.invalid_cells, 1u);
        return;
    }}
    rho_work.data[cell] = density;
    dp_work.data[cell] = local_dp;
    penalty_work.data[cell] = penalty;
    atomicMax(&control.max_base_bits, bitcast<u32>(rate));
    atomicMax(&control.max_vel_bits, bitcast<u32>(speed));
"#,
            uy = u + 1,
        )
    } else if let Some((rho, rho_u, rho_e, _)) = compressible {
        format!(
            r#"
    let base = cell * params.state_stride;
    var bad = false;
    for (var slot = 0u; slot < params.state_stride; slot += 1u) {{
        bad = bad || !finite_f32(state.data[base + slot]);
    }}
    if (bad) {{
        atomicAdd(&control.invalid_cells, 1u);
        return;
    }}

    // The acceptance contract is defined entirely on the conserved state.
    // Primitive u/p/T fields are stage-local caches and may be stale here, so
    // none of them participate in the thermodynamic-domain decision.
    // Gauge storage: the packed state holds deviations from the constant
    // reference; the thermodynamic-domain audit runs on ABSOLUTE values.
    // FLOORED THERMODYNAMICS (matches the primitive recovery): p/T clamp to
    // the runtime floors (1 Pa / 1 K on gauged production runs) instead of
    // rejecting the step — a vacuum-crossing cell survives with a clamped
    // recovery. Only a NON-FINITE state (or an out-of-domain constants set)
    // invalidates. Floored values feed the adaptive-CFL characteristic.
    // CONSERVED-STATE REPAIR (inert at the f32::MIN defaults): clamp the
    // stored density and total energy in place so a vacuum-crossing cell
    // cannot feed progressively wilder fluxes until the state reaches Inf.
    // Post-step, between RK steps — the stage arithmetic stays untouched.
    state.data[base + {rho}u] = max(state.data[base + {rho}u], params.eos_rho_floor);
    let repair_rho = max(state.data[base + {rho}u] + params.eos_gauge_rho_ref, 1.0e-8);
    let repair_ke = 0.5
        * (state.data[base + {rho_u}u] * state.data[base + {rho_u}u]
            + state.data[base + {rho_u_y}u] * state.data[base + {rho_u_y}u])
        / repair_rho;
    let rho_e_min = (params.eos_p_floor_abs - params.eos_p_ref)
        / max(params.eos_gm1, 1.0e-6)
        + repair_ke - params.eos_gauge_e_ref;
    state.data[base + {rho_e}u] = max(state.data[base + {rho_e}u], rho_e_min);

    let density = max(state.data[base + {rho}u] + params.eos_gauge_rho_ref, 1.0e-8);
    let momentum_x = state.data[base + {rho_u}u];
    let momentum_y = state.data[base + {rho_u_y}u];
    let total_energy_density = state.data[base + {rho_e}u] + params.eos_gauge_e_ref;
    let inv_density = 1.0 / density;
    let velocity_x = momentum_x * inv_density;
    let velocity_y = momentum_y * inv_density;
    let velocity_sq = velocity_x * velocity_x + velocity_y * velocity_y;
    let speed = sqrt(velocity_sq);
    let kinetic_energy_density = 0.5
        * (momentum_x * momentum_x + momentum_y * momentum_y) * inv_density;
    let internal_energy_density = total_energy_density - kinetic_energy_density;
    let pressure = max(
        params.eos_gm1 * internal_energy_density
            + params.eos_dp_drho * (density - params.eos_rho_ref) + params.eos_p_ref,
        params.eos_p_floor_abs,
    );
    let temperature = max(pressure / (density * params.eos_r), params.eos_t_floor);
    let sound_speed_sq = params.eos_gamma * max(pressure, 1.0e-30) * inv_density
        + params.eos_dp_drho;
    if (!(params.eos_gamma > 0.0) || !(params.eos_gm1 > 0.0)
        || !(params.eos_r > 0.0)
        || !(sound_speed_sq > 0.0) || !finite_f32(inv_density)
        || !finite_f32(speed) || !finite_f32(kinetic_energy_density)
        || !finite_f32(internal_energy_density) || !finite_f32(pressure)
        || !finite_f32(temperature) || !finite_f32(sound_speed_sq)) {{
        atomicAdd(&control.invalid_cells, 1u);
        return;
    }}
    if (params.adaptive != 0u) {{
        atomicMax(&control.max_vel_bits, bitcast<u32>(speed));
        // Retain the hottest accepted-state characteristic, not merely |u|.
        // The reference EOS bound in update_dt is still a floor for cold or
        // quiescent states, while this closes the hot/compressed-state CFL gap.
        atomicMax(&control.max_base_bits, bitcast<u32>(speed + sqrt(sound_speed_sq)));
    }}
"#,
            rho_u_y = rho_u + 1,
        )
    } else {
        r#"
    let base = cell * params.state_stride;
    for (var slot = 0u; slot < params.state_stride; slot += 1u) {
        if (!finite_f32(state.data[base + slot])) {
            atomicAdd(&control.invalid_cells, 1u);
            return;
        }
    }
"#
        .to_string()
    };

    let negative_energy_injection = if let Some((rho, rho_u, rho_e, _)) = compressible {
        format!(
            r#"
    if (cell == 0u && params.inject_negative_energy_after != 0xffffffffu
        && accepted_total_at_least_u32(params.inject_negative_energy_after)) {{
        let base = cell * params.state_stride;
        state.data[base + {rho}u] = 1.0 - params.eos_gauge_rho_ref;
        state.data[base + {rho_u}u] = 0.0;
        state.data[base + {rho_u_y}u] = 0.0;
        state.data[base + {rho_e}u] = -1.0 - params.eos_gauge_e_ref;
    }}
"#,
            rho_u_y = rho_u + 1,
        )
    } else {
        String::new()
    };

    let gradient_body = if let Some((_, p, _, _, _, _, _, _)) = allmach {
        format!(
            r#"
    let i = cell % grid.nx;
    let j = cell / grid.nx;
    let base = cell * params.state_stride;
    let pc = state.data[base + {p}u];
    var pw = pc;
    var pe = pc;
    var ps = pc;
    var pn = pc;
    if (i > 0u) {{
        pw = 0.5 * (pc + state.data[(cell - 1u) * params.state_stride + {p}u]);
    }} else {{
        let bi = (cell * 4u + 1u) * {unknowns}u + {p_rank}u;
        if (bc_kind.data[bi] == 1u) {{ pw = bc_value.data[bi]; }}
    }}
    if (i + 1u < grid.nx) {{
        pe = 0.5 * (pc + state.data[(cell + 1u) * params.state_stride + {p}u]);
    }} else {{
        let bi = (cell * 4u + 2u) * {unknowns}u + {p_rank}u;
        if (bc_kind.data[bi] == 1u) {{ pe = bc_value.data[bi]; }}
    }}
    if (j > 0u) {{
        ps = 0.5 * (pc + state.data[(cell - grid.nx) * params.state_stride + {p}u]);
    }}
    if (j + 1u < grid.ny) {{
        pn = 0.5 * (pc + state.data[(cell + grid.nx) * params.state_stride + {p}u]);
    }}
    grad_p.data[cell] = vec2<f32>((pe - pw) / grid.dx, (pn - ps) / grid.dy);
"#,
            p_rank = p.min(unknowns.saturating_sub(1)),
        )
    } else {
        String::new()
    };

    let rc_body = if let Some((_, p, _, _, _, _, _, _)) = allmach {
        format!(
            r#"
    if (atomicLoad(&control.invalid_cells) != 0u) {{ return; }}
    let i = cell % grid.nx;
    let j = cell / grid.nx;
    var sum = 0.0;
    if (i > 0u) {{ sum += interior_rc(cell - 1u, cell, 0u); }}
    if (i + 1u < grid.nx) {{ sum += interior_rc(cell, cell + 1u, 0u); }}
    if (j > 0u) {{ sum += interior_rc(cell - grid.nx, cell, 1u); }}
    if (j + 1u < grid.ny) {{ sum += interior_rc(cell, cell + grid.nx, 1u); }}
    if (i == 0u) {{
        let bi = (cell * 4u + 1u) * {unknowns}u + {p_rank}u;
        if (bc_kind.data[bi] == 1u) {{
            sum += boundary_rc(cell, bc_value.data[bi], 0u, -1.0);
        }}
    }}
    if (i + 1u == grid.nx) {{
        let bi = (cell * 4u + 2u) * {unknowns}u + {p_rank}u;
        if (bc_kind.data[bi] == 1u) {{
            sum += boundary_rc(cell, bc_value.data[bi], 0u, 1.0);
        }}
    }}
    let density = rho_work.data[cell];
    let rate = sum / (density * grid.dx * grid.dy);
    if (!finite_f32(rate) || !(rate >= 0.0)) {{
        atomicAdd(&control.invalid_cells, 1u);
        return;
    }}
    atomicMax(&control.max_rc_bits, bitcast<u32>(rate));
"#,
            p_rank = p.min(unknowns.saturating_sub(1)),
        )
    } else {
        String::new()
    };

    let p_offset = allmach.map_or(0, |(_, p, _, _, _, _, _, _)| p);
    let source = format!(
        r#"
struct Grid {{ nx: u32, ny: u32, dx: f32, dy: f32 }};
struct Control {{
    dt: f32, dt_old: f32, time_seconds: u32,
    time_fraction_hi: u32, time_fraction_lo: u32,
    max_base_bits: atomic<u32>, max_rc_bits: atomic<u32>,
    max_vel_bits: atomic<u32>, invalid_cells: atomic<u32>,
    halted: atomic<u32>, accepted_batch: atomic<u32>,
    accepted_total: atomic<u32>, accepted_total_hi: atomic<u32>,
}};
struct Clock {{ seconds: u32, fraction_hi: u32, fraction_lo: u32 }};
struct Params {{
    target_cfl: f32, density: f32, viscosity: f32, safety: f32,
    dt_max: f32, adaptive: u32, bootstrap_dt: u32, state_stride: u32,
    inject_nan_after: u32, inject_negative_energy_after: u32,
    adaptive_policy: u32, has_ibm_penalty: u32,
    inlet_velocity: f32, eos_gamma: f32, eos_gm1: f32, eos_r: f32,
    eos_dp_drho: f32, eos_p_ref: f32, eos_theta_ref: f32, eos_rho_ref: f32,
    eos_gauge_rho_ref: f32, eos_gauge_e_ref: f32,
    eos_p_floor_abs: f32, eos_t_floor: f32, eos_rho_floor: f32,
}};
struct F32Buffer {{ data: array<f32> }};
struct U32Buffer {{ data: array<u32> }};
struct Vec2Buffer {{ data: array<vec2<f32>> }};
struct IndirectArgs {{ x: u32, y: u32, z: u32 }};
@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> state: F32Buffer;
@group(0) @binding(3) var<storage, read_write> state_old: F32Buffer;
@group(0) @binding(4) var<storage, read_write> state_old_old: F32Buffer;
@group(0) @binding(5) var<storage, read_write> accepted_state: F32Buffer;
@group(0) @binding(6) var<storage, read_write> control: Control;
@group(0) @binding(7) var<storage, read_write> grad_p: Vec2Buffer;
@group(0) @binding(8) var<storage, read_write> rho_work: F32Buffer;
@group(0) @binding(9) var<storage, read_write> dp_work: F32Buffer;
@group(0) @binding(10) var<storage, read_write> penalty_work: F32Buffer;
@group(0) @binding(11) var<storage, read> bc_kind: U32Buffer;
@group(0) @binding(12) var<storage, read> bc_value: F32Buffer;
@group(0) @binding(13) var<storage, read_write> accepted_history: F32Buffer;
@group(0) @binding(14) var<storage, read_write> indirect_args: IndirectArgs;

fn finite_f32(v: f32) -> bool {{ return v == v && abs(v) <= 3.402823e38; }}
// Flatten the 2D-split dispatch (rows of `workgroups.x` groups, capped at the
// device's 65,535 workgroups-per-dimension limit) back to a linear thread
// index. Un-split dispatches have gid.y == 0, so this reduces to gid.x.
fn launch_index(gid: vec3<u32>, workgroups: vec3<u32>) -> u32 {{
    return gid.y * (workgroups.x * 64u) + gid.x;
}}
fn clock_parts() -> Clock {{
    return Clock(control.time_seconds, control.time_fraction_hi, control.time_fraction_lo);
}}
fn highest_bit(v0: u32) -> u32 {{
    var v = v0; var bit = 0u;
    if (v >= 65536u) {{ v >>= 16u; bit += 16u; }}
    if (v >= 256u) {{ v >>= 8u; bit += 8u; }}
    if (v >= 16u) {{ v >>= 4u; bit += 4u; }}
    if (v >= 4u) {{ v >>= 2u; bit += 2u; }}
    if (v >= 2u) {{ bit += 1u; }}
    return bit;
}}
fn clock_bit(c: Clock, bit: u32) -> u32 {{
    if (bit < 32u) {{ return (c.fraction_lo >> bit) & 1u; }}
    if (bit < 64u) {{ return (c.fraction_hi >> (bit - 32u)) & 1u; }}
    return (c.seconds >> (bit - 64u)) & 1u;
}}
fn low_mask(bits: u32) -> u32 {{ return (1u << bits) - 1u; }}
fn any_bits_below(c: Clock, count: u32) -> bool {{
    if (count == 0u) {{ return false; }}
    if (count < 32u) {{ return (c.fraction_lo & low_mask(count)) != 0u; }}
    if (count == 32u) {{ return c.fraction_lo != 0u; }}
    if (count < 64u) {{
        return c.fraction_lo != 0u
            || (c.fraction_hi & low_mask(count - 32u)) != 0u;
    }}
    if (count == 64u) {{ return c.fraction_lo != 0u || c.fraction_hi != 0u; }}
    return c.fraction_lo != 0u || c.fraction_hi != 0u
        || (c.seconds & low_mask(count - 64u)) != 0u;
}}
// Exact round-to-nearest-even conversion of the 96-bit unsigned Q64 clock.
// Building IEEE-754 bits avoids both integer-to-f32 and staged-addition double
// rounding at RK abscissae.
fn clock_to_f32(c: Clock) -> f32 {{
    if (c.seconds == 0u && c.fraction_hi == 0u && c.fraction_lo == 0u) {{ return 0.0; }}
    var top = 0u;
    if (c.seconds != 0u) {{ top = 64u + highest_bit(c.seconds); }}
    else if (c.fraction_hi != 0u) {{ top = 32u + highest_bit(c.fraction_hi); }}
    else {{ top = highest_bit(c.fraction_lo); }}
    var significand = 0u;
    var discarded = 0u;
    if (top < 23u) {{
        significand = c.fraction_lo << (23u - top);
    }} else {{
        discarded = top - 23u;
        for (var k = 0u; k < 24u; k += 1u) {{
            significand |= clock_bit(c, discarded + k) << k;
        }}
    }}
    var exponent = top + 63u;
    if (discarded != 0u) {{
        let guard = clock_bit(c, discarded - 1u) != 0u;
        let sticky = any_bits_below(c, discarded - 1u);
        if (guard && (sticky || (significand & 1u) != 0u)) {{
            significand += 1u;
            if (significand >= 0x01000000u) {{ significand >>= 1u; exponent += 1u; }}
        }}
    }}
    return bitcast<f32>((exponent << 23u) | (significand & 0x007fffffu));
}}
fn add_clock_parts(base: Clock, dt: f32) -> Clock {{
    // dt = significand*2^(raw_exp-150), hence its Q64 tick count is the
    // integer significand shifted by raw_exp-86. For dt>=1 ns raw_exp>=97.
    let bits = bitcast<u32>(dt);
    let raw_exp = (bits >> 23u) & 0xffu;
    let significand = (bits & 0x007fffffu) | 0x00800000u;
    var add_seconds = 0u; var add_hi = 0u; var add_lo = 0u;
    if ((bits & 0x80000000u) == 0u && raw_exp != 0u && raw_exp < 0xffu) {{
        if (raw_exp >= 86u) {{
            let shift = raw_exp - 86u;
            if (shift < 32u) {{
                add_lo = significand << shift;
                if (shift > 8u) {{ add_hi = significand >> (32u - shift); }}
            }} else if (shift < 64u) {{
                add_hi = significand << (shift - 32u);
                if (shift > 40u) {{ add_seconds = significand >> (64u - shift); }}
            }} else if (shift <= 72u) {{
                add_seconds = significand << (shift - 64u);
            }}
        }} else {{
            let right = 86u - raw_exp;
            if (right < 24u) {{
                add_lo = significand >> right;
                let remainder = significand & low_mask(right);
                let halfway = 1u << (right - 1u);
                if (remainder > halfway || (remainder == halfway && (add_lo & 1u) != 0u)) {{ add_lo += 1u; }}
            }}
        }}
    }}
    let next_low = base.fraction_lo + add_lo;
    let carry_low = select(0u, 1u, next_low < base.fraction_lo);
    let high_partial = base.fraction_hi + add_hi;
    let carry_high0 = select(0u, 1u, high_partial < base.fraction_hi);
    let next_high = high_partial + carry_low;
    let carry_high1 = select(0u, 1u, next_high < high_partial);
    return Clock(base.seconds + add_seconds + carry_high0 + carry_high1, next_high, next_low);
}}
fn clock_seconds() -> f32 {{ return clock_to_f32(clock_parts()); }}
fn clock_plus(dt: f32) -> f32 {{ return clock_to_f32(add_clock_parts(clock_parts(), dt)); }}
fn add_clock(dt: f32) {{
    let next = add_clock_parts(clock_parts(), dt);
    control.time_seconds = next.seconds;
    control.time_fraction_hi = next.fraction_hi;
    control.time_fraction_lo = next.fraction_lo;
}}
fn halt_batch() {{
    atomicStore(&control.halted, 1u);
    indirect_args.x = 0u;
    indirect_args.y = 0u;
    indirect_args.z = 0u;
}}

fn accepted_total_at_least_u32(threshold: u32) -> bool {{
    return atomicLoad(&control.accepted_total_hi) != 0u
        || atomicLoad(&control.accepted_total) >= threshold;
}}

fn interior_rc(a: u32, b: u32, axis: u32) -> f32 {{
    let rho = 0.5 * (rho_work.data[a] + rho_work.data[b]);
    let local_dp = min(dp_work.data[a], dp_work.data[b]);
    let kappa = rho * local_dp;
    var q = 0.0;
    var compact = 0.0;
    var area = grid.dy;
    if (axis == 0u) {{
        q = kappa * 0.5 * (grad_p.data[a].x + grad_p.data[b].x);
        compact = kappa * (state.data[b * params.state_stride + {p_offset}u]
            - state.data[a * params.state_stride + {p_offset}u]) / grid.dx;
    }} else {{
        area = grid.dx;
        q = kappa * 0.5 * (grad_p.data[a].y + grad_p.data[b].y);
        compact = kappa * (state.data[b * params.state_stride + {p_offset}u]
            - state.data[a * params.state_stride + {p_offset}u]) / grid.dy;
    }}
    let seal = 1.0 - min(penalty_work.data[a] + penalty_work.data[b], 1.0);
    return abs(seal * area * (q - compact));
}}

fn boundary_rc(cell: u32, boundary_p: f32, axis: u32, normal_sign: f32) -> f32 {{
    let kappa = rho_work.data[cell] * dp_work.data[cell];
    var normal_gradient = normal_sign * grad_p.data[cell].x;
    var area = grid.dy;
    var distance = 0.5 * grid.dx;
    if (axis == 1u) {{
        normal_gradient = normal_sign * grad_p.data[cell].y;
        area = grid.dx;
        distance = 0.5 * grid.dy;
    }}
    let pressure = state.data[cell * params.state_stride + {p_offset}u];
    let compact = kappa * (boundary_p - pressure) / distance;
    let seal = 1.0 - min(2.0 * penalty_work.data[cell], 1.0);
    return abs(seal * area * (kappa * normal_gradient - compact));
}}

@compute @workgroup_size(1)
fn reset_batch(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    atomicStore(&control.accepted_batch, 0u);
    if (atomicLoad(&control.halted) != 0u) {{ return; }}
    atomicStore(&control.max_base_bits, 0u);
    atomicStore(&control.max_rc_bits, 0u);
    atomicStore(&control.max_vel_bits, 0u);
    atomicStore(&control.invalid_cells, 0u);
}}

@compute @workgroup_size(1)
fn reset_metrics(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    if (atomicLoad(&control.halted) != 0u) {{ return; }}
    atomicStore(&control.max_base_bits, 0u);
    atomicStore(&control.max_rc_bits, 0u);
    atomicStore(&control.max_vel_bits, 0u);
    atomicStore(&control.invalid_cells, 0u);
}}

@compute @workgroup_size(64)
fn history(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
) {{
    let index = launch_index(gid, workgroups);
    if (index >= arrayLength(&state.data) || atomicLoad(&control.halted) != 0u) {{ return; }}
    accepted_state.data[index] = state.data[index];
    accepted_history.data[index] = state_old.data[index];
    accepted_history.data[arrayLength(&state.data) + index] = state_old_old.data[index];
    state_old_old.data[index] = state_old.data[index];
    state_old.data[index] = state.data[index];
}}

@compute @workgroup_size(64)
fn pressure_gradient(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
) {{
    let cell = launch_index(gid, workgroups);
    if (cell >= grid.nx * grid.ny || atomicLoad(&control.halted) != 0u) {{ return; }}
{gradient_body}
}}

@compute @workgroup_size(64)
fn sample_state(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
) {{
    let cell = launch_index(gid, workgroups);
    if (cell >= grid.nx * grid.ny || atomicLoad(&control.halted) != 0u) {{ return; }}
    if (cell == 0u && params.inject_nan_after != 0xffffffffu
        && accepted_total_at_least_u32(params.inject_nan_after)) {{
        state.data[0] = bitcast<f32>(0x7fc00000u);
        atomicAdd(&control.invalid_cells, 1u);
        return;
    }}
{negative_energy_injection}
{sample_body}
}}

@compute @workgroup_size(64)
fn sample_rhie_chow(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
) {{
    let cell = launch_index(gid, workgroups);
    if (cell >= grid.nx * grid.ny || atomicLoad(&control.halted) != 0u) {{ return; }}
{rc_body}
}}

fn update_dt() {{
    if (params.adaptive == 0u) {{ return; }}
    if (params.adaptive_policy == 2u) {{
        // Host-policy parity for density-based structured RK4:
        //   dt_wave = CFL h_min /
        //             max(max_i(|u_i|+c_i), |u_in|+c_ref)
        //   dt_diff = 0.25 CFL h_min^2 / (|mu|/|rho_ref|)
        //   (the Brinkman reaction is projected out per stage; no dt_ibm).
        // The reference term protects a cold/quiescent inlet; the accepted-
        // state maximum protects hot/compressed cells whose local c exceeds it.
        let min_h = min(grid.dx, grid.dy);
        let sound_ref_sq = params.eos_gamma * params.eos_theta_ref;
        if (!finite_f32(min_h) || !(min_h > 1.0e-12)
            || !finite_f32(sound_ref_sq) || !(sound_ref_sq > 0.0)) {{
            halt_batch();
            return;
        }}
        let accepted_characteristic = bitcast<f32>(atomicLoad(&control.max_base_bits));
        let reference_characteristic = abs(params.inlet_velocity) + sqrt(sound_ref_sq);
        let wave_speed = max(accepted_characteristic, reference_characteristic);
        var proposed = params.target_cfl * min_h / wave_speed;
        var has_bound = finite_f32(proposed);
        let rho_ref = max(abs(params.density), 1.0e-12);
        let alpha = abs(params.viscosity) / rho_ref;
        if (finite_f32(alpha) && alpha > 1.0e-14) {{
            let diffusion_dt = 0.25 * params.target_cfl * min_h * min_h / alpha;
            proposed = select(diffusion_dt, min(proposed, diffusion_dt), has_bound);
            has_bound = has_bound || finite_f32(diffusion_dt);
        }}
        // No Brinkman reaction bound: the RK4 stage kernels enforce the
        // immersed solid as an algebraic momentum projection (rho_u = 0 in
        // solid cells at every abscissa), so the -1e5 penalty never enters
        // the explicit stability spectrum and dt is acoustic/diffusion-bound.
        if (!has_bound) {{
            halt_batch();
            return;
        }}
        proposed = min(proposed, control.dt * 1.2);
        control.dt = clamp(proposed, 1.0e-9, params.dt_max);
        if (!finite_f32(control.dt) || !(control.dt > 0.0)) {{ halt_batch(); }}
        return;
    }}
    let rate = bitcast<f32>(atomicLoad(&control.max_base_bits))
        + bitcast<f32>(atomicLoad(&control.max_rc_bits));
    if (!finite_f32(rate) || !(rate > 0.0)) {{
        halt_batch();
        return;
    }}
    let bounded_cfl = clamp(params.target_cfl, 1.0e-6, 1.0);
    let proposed = params.safety * bounded_cfl / rate;
    control.dt = min(params.dt_max, min(control.dt * 1.2, proposed));
}}

@compute @workgroup_size(1)
fn bootstrap(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    if (atomicLoad(&control.halted) != 0u || atomicLoad(&control.invalid_cells) != 0u) {{
        halt_batch();
        return;
    }}
    if (params.bootstrap_dt != 0u) {{ update_dt(); }}
}}

@compute @workgroup_size(1)
fn accept(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    if (atomicLoad(&control.halted) != 0u || atomicLoad(&control.invalid_cells) != 0u) {{
        halt_batch();
        return;
    }}
    let accepted_dt = control.dt;
    add_clock(accepted_dt);
    control.dt_old = accepted_dt;
    atomicAdd(&control.accepted_batch, 1u);
    let old_total_lo = atomicAdd(&control.accepted_total, 1u);
    if (old_total_lo == 0xffffffffu) {{
        atomicAdd(&control.accepted_total_hi, 1u);
    }}
    update_dt();
}}

@compute @workgroup_size(1)
fn clock_fixed_probe(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    control.time_seconds = 0u; control.time_fraction_hi = 0u; control.time_fraction_lo = 0u;
    let tiny_dt = bitcast<f32>(0x30a9ad7fu);
    for (var i = 0u; i < 100003u; i += 1u) {{ add_clock(tiny_dt); }}
    let probe_dt = bitcast<f32>(0x3157e37cu);
    control.dt = clock_plus(0.0);
    control.dt_old = clock_plus(0.5 * probe_dt);
    atomicStore(&control.max_base_bits, bitcast<u32>(clock_plus(probe_dt)));
}}

@compute @workgroup_size(1)
fn clock_random_probe(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    control.time_seconds = 31u; control.time_fraction_hi = 0xfffff000u; control.time_fraction_lo = 0u;
    var seed = 0x6d2b79f5u;
    for (var i = 0u; i < 4096u; i += 1u) {{
        seed = seed * 1664525u + 1013904223u;
        add_clock(bitcast<f32>(0x3089705fu + (seed & 0x001fffffu)));
    }}
    seed = seed * 1664525u + 1013904223u;
    let probe_dt = bitcast<f32>(0x3089705fu + (seed & 0x001fffffu));
    control.dt = clock_plus(0.0);
    control.dt_old = clock_plus(0.5 * probe_dt);
    atomicStore(&control.max_base_bits, bitcast<u32>(clock_plus(probe_dt)));
}}

@compute @workgroup_size(64)
fn rollback(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
) {{
    let index = launch_index(gid, workgroups);
    if (index >= arrayLength(&state.data) || atomicLoad(&control.halted) == 0u) {{ return; }}
    state.data[index] = accepted_state.data[index];
    state_old.data[index] = accepted_history.data[index];
    state_old_old.data[index] = accepted_history.data[arrayLength(&state.data) + index];
}}
"#
    );
    (source, adaptive_policy)
}

fn structured_patch_constants_wgsl() -> &'static str {
    r#"
struct Control {
    dt: f32, dt_old: f32, time_seconds: u32,
    time_fraction_hi: u32, time_fraction_lo: u32,
    max_base_bits: atomic<u32>, max_rc_bits: atomic<u32>,
    max_vel_bits: atomic<u32>, invalid_cells: atomic<u32>,
    halted: atomic<u32>, accepted_batch: atomic<u32>,
    accepted_total: atomic<u32>, accepted_total_hi: atomic<u32>,
};
struct Clock { seconds: u32, fraction_hi: u32, fraction_lo: u32 };
struct U32Buffer { data: array<u32> };
@group(0) @binding(0) var<storage, read> control: Control;
@group(0) @binding(1) var<storage, read> record_words: U32Buffer;
@group(0) @binding(2) var<storage, read_write> snapshots: U32Buffer;

fn clock_parts() -> Clock {
    return Clock(control.time_seconds, control.time_fraction_hi, control.time_fraction_lo);
}
fn highest_bit(v0: u32) -> u32 {
    var v = v0; var bit = 0u;
    if (v >= 65536u) { v >>= 16u; bit += 16u; }
    if (v >= 256u) { v >>= 8u; bit += 8u; }
    if (v >= 16u) { v >>= 4u; bit += 4u; }
    if (v >= 4u) { v >>= 2u; bit += 2u; }
    if (v >= 2u) { bit += 1u; }
    return bit;
}
fn clock_bit(c: Clock, bit: u32) -> u32 {
    if (bit < 32u) { return (c.fraction_lo >> bit) & 1u; }
    if (bit < 64u) { return (c.fraction_hi >> (bit - 32u)) & 1u; }
    return (c.seconds >> (bit - 64u)) & 1u;
}
fn low_mask(bits: u32) -> u32 { return (1u << bits) - 1u; }
fn any_bits_below(c: Clock, count: u32) -> bool {
    if (count == 0u) { return false; }
    if (count < 32u) { return (c.fraction_lo & low_mask(count)) != 0u; }
    if (count == 32u) { return c.fraction_lo != 0u; }
    if (count < 64u) {
        return c.fraction_lo != 0u
            || (c.fraction_hi & low_mask(count - 32u)) != 0u;
    }
    if (count == 64u) { return c.fraction_lo != 0u || c.fraction_hi != 0u; }
    return c.fraction_lo != 0u || c.fraction_hi != 0u
        || (c.seconds & low_mask(count - 64u)) != 0u;
}
fn clock_to_f32(c: Clock) -> f32 {
    if (c.seconds == 0u && c.fraction_hi == 0u && c.fraction_lo == 0u) { return 0.0; }
    var top = 0u;
    if (c.seconds != 0u) { top = 64u + highest_bit(c.seconds); }
    else if (c.fraction_hi != 0u) { top = 32u + highest_bit(c.fraction_hi); }
    else { top = highest_bit(c.fraction_lo); }
    var significand = 0u;
    var discarded = 0u;
    if (top < 23u) {
        significand = c.fraction_lo << (23u - top);
    } else {
        discarded = top - 23u;
        for (var k = 0u; k < 24u; k += 1u) {
            significand |= clock_bit(c, discarded + k) << k;
        }
    }
    var exponent = top + 63u;
    if (discarded != 0u) {
        let guard = clock_bit(c, discarded - 1u) != 0u;
        let sticky = any_bits_below(c, discarded - 1u);
        if (guard && (sticky || (significand & 1u) != 0u)) {
            significand += 1u;
            if (significand >= 0x01000000u) { significand >>= 1u; exponent += 1u; }
        }
    }
    return bitcast<f32>((exponent << 23u) | (significand & 0x007fffffu));
}
fn add_clock_parts(base: Clock, dt: f32) -> Clock {
    // dt = significand*2^(raw_exp-150), hence its Q64 tick count is the
    // integer significand shifted by raw_exp-86. For dt>=1 ns raw_exp>=97.
    let bits = bitcast<u32>(dt);
    let raw_exp = (bits >> 23u) & 0xffu;
    let significand = (bits & 0x007fffffu) | 0x00800000u;
    var add_seconds = 0u; var add_hi = 0u; var add_lo = 0u;
    if ((bits & 0x80000000u) == 0u && raw_exp != 0u && raw_exp < 0xffu) {
        if (raw_exp >= 86u) {
            let shift = raw_exp - 86u;
            if (shift < 32u) {
                add_lo = significand << shift;
                if (shift > 8u) { add_hi = significand >> (32u - shift); }
            } else if (shift < 64u) {
                add_hi = significand << (shift - 32u);
                if (shift > 40u) { add_seconds = significand >> (64u - shift); }
            } else if (shift <= 72u) {
                add_seconds = significand << (shift - 64u);
            }
        } else {
            let right = 86u - raw_exp;
            if (right < 24u) {
                add_lo = significand >> right;
                let remainder = significand & low_mask(right);
                let halfway = 1u << (right - 1u);
                if (remainder > halfway || (remainder == halfway && (add_lo & 1u) != 0u)) { add_lo += 1u; }
            }
        }
    }
    let next_low = base.fraction_lo + add_lo;
    let carry_low = select(0u, 1u, next_low < base.fraction_lo);
    let high_partial = base.fraction_hi + add_hi;
    let carry_high0 = select(0u, 1u, high_partial < base.fraction_hi);
    let next_high = high_partial + carry_low;
    let carry_high1 = select(0u, 1u, next_high < high_partial);
    return Clock(base.seconds + add_seconds + carry_high0 + carry_high1, next_high, next_low);
}
fn clock_plus(dt: f32) -> f32 { return clock_to_f32(add_clock_parts(clock_parts(), dt)); }

fn patch_constants(index: u32, c: f32) {
    if (index >= arrayLength(&record_words.data)) { return; }
    let word = record_words.data[index];
    snapshots.data[word] = bitcast<u32>(control.dt);
    snapshots.data[word + 1u] = bitcast<u32>(control.dt_old);
    snapshots.data[word + 2u] = bitcast<u32>(0.0);
    snapshots.data[word + 3u] = bitcast<u32>(clock_plus(c * control.dt));
}
@compute @workgroup_size(64)
fn stage_0(@builtin(global_invocation_id) gid: vec3<u32>) { patch_constants(gid.x, 0.0); }
@compute @workgroup_size(64)
fn stage_half(@builtin(global_invocation_id) gid: vec3<u32>) { patch_constants(gid.x, 0.5); }
@compute @workgroup_size(64)
fn stage_1(@builtin(global_invocation_id) gid: vec3<u32>) { patch_constants(gid.x, 1.0); }
"#
}

impl StructuredAutonomousControl {
    fn new(
        device: &wgpu::Device,
        grid_buf: &wgpu::Buffer,
        buffers: &HashMap<String, wgpu::Buffer>,
        layout: &crate::solver::model::backend::state_layout::StateLayout,
        model_id: &str,
        n: usize,
        unknowns: usize,
        prep: &[String],
        residual: &[String],
        kernels: &HashMap<String, CompiledKernel>,
        constants: &GpuConstants,
    ) -> Self {
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let common_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("structured:autonomous-common-layout"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
                storage_entry(9, false),
                storage_entry(10, false),
                storage_entry(11, true),
                storage_entry(12, true),
                storage_entry(13, false),
                storage_entry(14, false),
            ],
        });
        let common_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("structured:autonomous-common-pipeline-layout"),
                bind_group_layouts: &[Some(&common_layout)],
                immediate_size: 0,
            });
        let (common_wgsl, adaptive_policy) = structured_autonomous_wgsl(model_id, layout, unknowns);
        let common_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("structured:autonomous-control"),
            source: wgpu::ShaderSource::Wgsl(common_wgsl.into()),
        });
        let common_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&common_pipeline_layout),
                module: &common_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let initial_control = StructuredAutonomousControlRaw::zeroed();
        let initial_params = StructuredAutonomousParams {
            target_cfl: 0.5,
            density: 1.0,
            viscosity: 0.0,
            safety: 0.8,
            dt_max: 100.0,
            adaptive: 0,
            bootstrap_dt: 0,
            state_stride: layout.stride(),
            inject_nan_after: u32::MAX,
            inject_negative_energy_after: u32::MAX,
            adaptive_policy: adaptive_policy as u32,
            has_ibm_penalty: u32::from(layout.offset_for("ibm_penalty_U").is_some()),
            inlet_velocity: 0.0,
            eos_gamma: 1.4,
            eos_gm1: 0.4,
            eos_r: 1.0,
            eos_dp_drho: 0.0,
            eos_p_ref: 0.0,
            eos_theta_ref: 1.0,
            eos_rho_ref: 0.0,
            eos_gauge_rho_ref: 0.0,
            eos_p_floor_abs: f32::MIN,
            eos_t_floor: f32::MIN,
            eos_rho_floor: f32::MIN,
            eos_gauge_e_ref: 0.0,
        };
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("structured:autonomous-status"),
            contents: bytemuck::bytes_of(&initial_control),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("structured:autonomous-params"),
            contents: bytemuck::bytes_of(&initial_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scratch = |label, words| storage_buffer(device, label, words);
        let grad_p = scratch("structured:autonomous-grad-p", n * 2);
        let rho_work = scratch("structured:autonomous-rho", n);
        let dp_work = scratch("structured:autonomous-dp", n);
        let penalty_work = scratch("structured:autonomous-penalty", n);
        let groups = (n as u32).div_ceil(WG).max(1);
        let max_x = device.limits().max_compute_workgroups_per_dimension;
        let indirect_init = [groups.min(max_x), groups.div_ceil(max_x), 1u32];
        let indirect_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("structured:autonomous-indirect-args"),
            contents: bytemuck::cast_slice(&indirect_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        });
        let common_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("structured:autonomous-common-bg"),
            layout: &common_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers["state"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers["state_old"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers["state_old_old"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers["accepted_state"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: control.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grad_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: rho_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: dp_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: penalty_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers["bc_kind"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers["bc_value"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers["accepted_history"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: indirect_args.as_entire_binding(),
                },
            ],
        });

        let patch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("structured:autonomous-patch-layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let patch_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("structured:autonomous-patch-pipeline-layout"),
                bind_group_layouts: &[Some(&patch_layout)],
                immediate_size: 0,
            });
        let patch_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("structured:autonomous-patch-constants"),
            source: wgpu::ShaderSource::Wgsl(structured_patch_constants_wgsl().into()),
        });
        let patch_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&patch_pipeline_layout),
                module: &patch_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let patch_stage_0 = patch_pipeline("stage_0");
        let patch_stage_half = patch_pipeline("stage_half");
        let patch_stage_1 = patch_pipeline("stage_1");

        let stage_ids = [
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_1.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_2.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_3.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_4.as_str(),
        ];
        let stage_prep: Vec<String> = prep
            .iter()
            .filter(|id| id.contains("bc_expr"))
            .cloned()
            .collect();
        let mut schedules = Vec::with_capacity(4);
        for stage_id in stage_ids {
            let mut schedule = Vec::with_capacity(stage_prep.len() + residual.len() + 1);
            schedule.extend(stage_prep.iter().cloned());
            schedule.extend(residual.iter().cloned());
            schedule.push(stage_id.to_string());
            schedules.push(schedule);
        }

        let mut snapshot_bytes = Vec::<u8>::new();
        let mut records = HashMap::<String, (u64, u64)>::new();
        let mut record_words = Vec::<u32>::new();
        for id in schedules.iter().flatten() {
            if records.contains_key(id) {
                continue;
            }
            let kernel = &kernels[id];
            let offset = snapshot_bytes.len() as u64;
            let size = packed_kernel_constants_len(&kernel.eos_fields) as u64;
            let word = u32::try_from(offset / 4)
                .expect("structured autonomous constant snapshot exceeds u32 words");
            snapshot_bytes.resize(snapshot_bytes.len() + size as usize, 0);
            write_kernel_constants_bytes(
                constants,
                &kernel.eos_fields,
                &mut snapshot_bytes[offset as usize..(offset + size) as usize],
            );
            record_words.push(word);
            records.insert(id.clone(), (offset, size));
        }
        let snapshots = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("structured:autonomous-constant-snapshots"),
            contents: &snapshot_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let record_words_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("structured:autonomous-constant-records"),
                contents: bytemuck::cast_slice(&record_words),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let patch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("structured:autonomous-patch-bg"),
            layout: &patch_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: control.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: record_words_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: snapshots.as_entire_binding(),
                },
            ],
        });
        let patch_groups = (record_words.len() as u32).div_ceil(WG).max(1);
        let status_staging = std::array::from_fn(|slot| {
            std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match slot {
                    0 => "structured:autonomous-status-staging-0",
                    1 => "structured:autonomous-status-staging-1",
                    _ => "structured:autonomous-status-staging-2",
                }),
                size: std::mem::size_of::<StructuredAutonomousControlRaw>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        let batch_cache = StructuredAutonomousBatchCache {
            schedules,
            records,
            snapshot_bytes: std::sync::Mutex::new(snapshot_bytes),
            snapshots,
            _record_words: record_words_buffer,
            patch_bg,
            patch_groups,
            status_staging,
            status_slots_busy: std::sync::Arc::new(std::array::from_fn(|_| {
                std::sync::atomic::AtomicBool::new(false)
            })),
            next_status_slot: std::sync::atomic::AtomicUsize::new(0),
        };
        #[cfg(feature = "dev-tests")]
        let clock_fixed_probe = common_pipeline("clock_fixed_probe");
        #[cfg(feature = "dev-tests")]
        let clock_random_probe = common_pipeline("clock_random_probe");

        Self {
            control,
            params,
            max_workgroups_per_dim: max_x,
            _grad_p: grad_p,
            _rho_work: rho_work,
            _dp_work: dp_work,
            _penalty_work: penalty_work,
            indirect_args,
            common_bg,
            reset_batch: common_pipeline("reset_batch"),
            reset_metrics: common_pipeline("reset_metrics"),
            gradient: common_pipeline("pressure_gradient"),
            sample: common_pipeline("sample_state"),
            rhie_chow: common_pipeline("sample_rhie_chow"),
            bootstrap: common_pipeline("bootstrap"),
            history: common_pipeline("history"),
            accept: common_pipeline("accept"),
            rollback: common_pipeline("rollback"),
            patch_stage_0,
            patch_stage_half,
            patch_stage_1,
            batch_cache,
            #[cfg(feature = "dev-tests")]
            clock_fixed_probe,
            #[cfg(feature = "dev-tests")]
            clock_random_probe,
            adaptive_policy,
        }
    }

    fn encode_common(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        threads: u32,
        label: &str,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.common_bg, &[]);
        // Same 2D split as the stage kernels' indirect args: the state-word
        // kernels (history/rollback) exceed 65,535 workgroups from ~190k cells
        // (n * state_stride threads); the WGSL flattens via `launch_index`.
        let groups = threads.div_ceil(WG).max(1);
        let groups_x = groups.min(self.max_workgroups_per_dim);
        let groups_y = groups.div_ceil(self.max_workgroups_per_dim);
        assert!(
            groups_y <= self.max_workgroups_per_dim,
            "structured autonomous dispatch requires more than a 2D workgroup grid: groups={groups}, max={}",
            self.max_workgroups_per_dim
        );
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    fn encode_sample(&self, encoder: &mut wgpu::CommandEncoder, cells: u32, adaptive: bool) {
        let rhie_chow =
            adaptive && self.adaptive_policy == StructuredAdaptivePolicy::AllMachSpectral;
        if rhie_chow {
            self.encode_common(encoder, &self.gradient, cells, "structured:auto-gradient");
        }
        self.encode_common(encoder, &self.sample, cells, "structured:auto-sample");
        if rhie_chow {
            self.encode_common(encoder, &self.rhie_chow, cells, "structured:auto-rhie-chow");
        }
    }
}

// ===========================================================================
// Selective conserved-field filter (explicit RK4, density-based compressible).
// ===========================================================================

/// Hand-written WGSL for the dimension-split selective low-pass filter over
/// the conserved fields (`rho`, `rho_u`, `rho_e`) of the density-based
/// structured compressible family. Per direction, per component:
///
/// `q_i <- q_i - sigma * D_i`, `D_i = w0*q_i + sum_{j=1..N} wj*(q_{i-j}+q_{i+j})`
///
/// with the binomial weight sets (transfer `1 - sigma*sin^(2N)(k*h/2)`):
/// N=4 `[70,-56,28,-8,1]/256`, N=3 `[20,-15,6,-1]/64`, N=2 `[6,-4,1]/16`.
/// `N` per cell/direction is the largest in `{4,3,2}` whose full `i-N..i+N`
/// stencil stays inside the grid AND clear of solid cells (`ibm_penalty_U < 0`,
/// the [`append_ibm_velocity_projection`] convention); clearance < 2 or a solid
/// centre cell is identity. The evaluation order is pinned pairwise-symmetric
/// (highest ring first, `w0*q0` last) so the hand-written CPU implementation
/// (`cpu::structured`) computes the identical f32 sequence.
///
/// Two entry points ping-pong through `filter_scratch`: `filter_x` reads
/// `state` and writes the scratch (copying every non-filtered component
/// through bit-exactly), `filter_y` reads the scratch back into `state`.
/// Both flatten a possibly 2D-split dispatch via `launch_index`, so they run
/// under the autonomous route's cells indirect-args slot (halted batch =>
/// zeroed args => no filtering of a rolled-back state).
fn structured_conserved_filter_wgsl(
    stride: usize,
    rho: usize,
    rho_u: usize,
    rho_e: usize,
    penalty: Option<usize>,
) -> String {
    let solid_body = match penalty {
        Some(offset) => format!("return src.data[cell * {stride}u + {offset}u] < 0.0;"),
        None => "return false;".to_string(),
    };
    format!(
        r#"
struct Grid {{ nx: u32, ny: u32, dx: f32, dy: f32 }};
struct FilterParams {{ sigma: f32, _pad0: f32, _pad1: f32, _pad2: f32 }};
struct F32Buffer {{ data: array<f32> }};
@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<uniform> params: FilterParams;
@group(0) @binding(2) var<storage, read> src: F32Buffer;
@group(0) @binding(3) var<storage, read_write> dst: F32Buffer;

// Flatten the 2D-split dispatch (rows of `workgroups.x` groups, capped at the
// device's 65,535 workgroups-per-dimension limit) back to a linear cell index.
fn launch_index(gid: vec3<u32>, workgroups: vec3<u32>) -> u32 {{
    return gid.y * (workgroups.x * 64u) + gid.x;
}}

fn is_solid(cell: u32) -> bool {{
    {solid_body}
}}

// The filtered value of component `comp` of `cell` along stride `step`
// (1 = x, nx = y) with stencil half-width `clearance` (2, 3 or 4). All
// binomial weights are exact binary fractions; the pinned pairwise order
// (highest ring first, centre last) matches the CPU implementation exactly.
fn filtered_component(cell: u32, comp: u32, step: u32, clearance: u32) -> f32 {{
    let q0 = src.data[cell * {stride}u + comp];
    let s1 = src.data[(cell - step) * {stride}u + comp]
        + src.data[(cell + step) * {stride}u + comp];
    let s2 = src.data[(cell - 2u * step) * {stride}u + comp]
        + src.data[(cell + 2u * step) * {stride}u + comp];
    var d = 0.0;
    if (clearance >= 4u) {{
        let s3 = src.data[(cell - 3u * step) * {stride}u + comp]
            + src.data[(cell + 3u * step) * {stride}u + comp];
        let s4 = src.data[(cell - 4u * step) * {stride}u + comp]
            + src.data[(cell + 4u * step) * {stride}u + comp];
        d = 0.00390625 * s4;
        d = d - 0.03125 * s3;
        d = d + 0.109375 * s2;
        d = d - 0.21875 * s1;
        d = d + 0.2734375 * q0;
    }} else if (clearance == 3u) {{
        let s3 = src.data[(cell - 3u * step) * {stride}u + comp]
            + src.data[(cell + 3u * step) * {stride}u + comp];
        d = -0.015625 * s3;
        d = d + 0.09375 * s2;
        d = d - 0.234375 * s1;
        d = d + 0.3125 * q0;
    }} else {{
        d = 0.0625 * s2;
        d = d - 0.25 * s1;
        d = d + 0.375 * q0;
    }}
    return q0 - params.sigma * d;
}}

// One directional pass for `cell` at 1D position `pos` of `count` cells with
// neighbour stride `step`. Copies the complete state row through, then
// overwrites the four conserved components with their filtered values when the
// cell is fluid and has clearance >= 2.
fn apply_pass(cell: u32, pos: u32, count: u32, step: u32) {{
    let base = cell * {stride}u;
    for (var slot = 0u; slot < {stride}u; slot = slot + 1u) {{
        dst.data[base + slot] = src.data[base + slot];
    }}
    if (is_solid(cell)) {{ return; }}
    let room = min(pos, count - 1u - pos);
    let limit = min(room, 4u);
    var clearance = 0u;
    for (var ring = 1u; ring <= limit; ring = ring + 1u) {{
        if (is_solid(cell - ring * step) || is_solid(cell + ring * step)) {{ break; }}
        clearance = ring;
    }}
    if (clearance < 2u) {{ return; }}
    dst.data[base + {rho}u] = filtered_component(cell, {rho}u, step, clearance);
    dst.data[base + {rho_u}u] = filtered_component(cell, {rho_u}u, step, clearance);
    dst.data[base + {rho_u_y}u] = filtered_component(cell, {rho_u_y}u, step, clearance);
    dst.data[base + {rho_e}u] = filtered_component(cell, {rho_e}u, step, clearance);
}}

@compute @workgroup_size(64)
fn filter_x(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
) {{
    let cell = launch_index(gid, workgroups);
    if (cell >= grid.nx * grid.ny) {{ return; }}
    apply_pass(cell, cell % grid.nx, grid.nx, 1u);
}}

@compute @workgroup_size(64)
fn filter_y(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
) {{
    let cell = launch_index(gid, workgroups);
    if (cell >= grid.nx * grid.ny) {{ return; }}
    apply_pass(cell, cell / grid.nx, grid.ny, grid.nx);
}}
"#,
        rho_u_y = rho_u + 1,
    )
}

/// GPU resources for the selective conserved-field filter: the ping-pong
/// scratch (`n * state_stride`), the `sigma` uniform, and the two directional
/// pipelines with their pre-built bind groups (`x`: state -> scratch, `y`:
/// scratch -> state). Built only for explicit-RK4 models whose state layout
/// carries all of `rho`/`rho_u`/`rho_e`; `sigma == 0.0` is a host-side skip at
/// every dispatch site.
struct StructuredConservedFilter {
    /// Host cache of the runtime filter strength; the uniform holds the same
    /// value. Every dispatch site skips encoding when this is exactly 0.0.
    sigma: f32,
    params: wgpu::Buffer,
    _scratch: wgpu::Buffer,
    pipeline_x: wgpu::ComputePipeline,
    pipeline_y: wgpu::ComputePipeline,
    bg_x: wgpu::BindGroup,
    bg_y: wgpu::BindGroup,
    max_workgroups_per_dim: u32,
}

impl StructuredConservedFilter {
    fn new(
        device: &wgpu::Device,
        grid_buf: &wgpu::Buffer,
        state_buf: &wgpu::Buffer,
        n: usize,
        state_stride: usize,
        layout: &crate::solver::model::backend::state_layout::StateLayout,
    ) -> Option<Self> {
        let rho = layout.offset_for("rho")? as usize;
        let rho_u = layout.offset_for("rho_u")? as usize;
        let rho_e = layout.offset_for("rho_e")? as usize;
        let penalty = layout.offset_for("ibm_penalty_U").map(|o| o as usize);
        let wgsl = structured_conserved_filter_wgsl(state_stride, rho, rho_u, rho_e, penalty);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("structured:conserved-filter"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("structured:conserved-filter-layout"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_entry(1, wgpu::BufferBindingType::Uniform),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("structured:conserved-filter-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("structured:conserved-filter-params"),
            contents: bytemuck::cast_slice(&[0.0_f32; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scratch = storage_buffer(device, "filter_scratch", n * state_stride);
        let bind_group = |label: &str, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: grid_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dst.as_entire_binding(),
                    },
                ],
            })
        };
        let bg_x = bind_group("structured:conserved-filter-x", state_buf, &scratch);
        let bg_y = bind_group("structured:conserved-filter-y", &scratch, state_buf);
        Some(Self {
            sigma: 0.0,
            params,
            _scratch: scratch,
            pipeline_x: pipeline("filter_x"),
            pipeline_y: pipeline("filter_y"),
            bg_x,
            bg_y,
            max_workgroups_per_dim: device.limits().max_compute_workgroups_per_dimension,
        })
    }

    /// Encode both directional passes with a direct per-cell dispatch (the
    /// plot/host routes). One compute pass: each dispatch owns its usage
    /// scope, so `filter_y` sees `filter_x`'s scratch writes.
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, cells: u32) {
        let groups = cells.div_ceil(WG).max(1);
        let groups_x = groups.min(self.max_workgroups_per_dim);
        let groups_y = groups.div_ceil(self.max_workgroups_per_dim);
        assert!(
            groups_y <= self.max_workgroups_per_dim,
            "structured filter dispatch requires more than a 2D workgroup grid: groups={groups}, max={}",
            self.max_workgroups_per_dim
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("structured:conserved-filter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_x);
        pass.set_bind_group(0, &self.bg_x, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        pass.set_pipeline(&self.pipeline_y);
        pass.set_bind_group(0, &self.bg_y, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        crate::count_dispatch!("Structured Kernel", "conserved_filter");
    }

    /// Encode both directional passes through the autonomous route's per-cell
    /// indirect-args slot. A halted batch zeroes those args, so a rolled-back
    /// accepted state is never filtered again.
    fn encode_indirect(&self, encoder: &mut wgpu::CommandEncoder, indirect_args: &wgpu::Buffer) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("structured:conserved-filter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_x);
        pass.set_bind_group(0, &self.bg_x, &[]);
        pass.dispatch_workgroups_indirect(indirect_args, 0);
        pass.set_pipeline(&self.pipeline_y);
        pass.set_bind_group(0, &self.bg_y, &[]);
        pass.dispatch_workgroups_indirect(indirect_args, 0);
        crate::count_dispatch!("Structured Kernel", "conserved_filter");
    }
}

/// Lower every `Structured2D` codegen kernel of `model` to `(id, wgsl, bindings,
/// dispatch)`. Mirrors `cpu::lowering::model_kernel_programs` but at the WGSL
/// level and without the `cpu` feature (walks the model modules directly).
fn lower_structured_kernels(
    model: &ModelSpec,
    schemes: &SchemeRegistry,
) -> Result<Vec<(String, String, Vec<BindInfo>, Vec<String>)>, String> {
    let mut out = Vec::new();
    for module in &model.modules {
        let module: &dyn ModelModule = module;
        for spec in module.kernel_generators() {
            let art = (spec.generator)(model, schemes)?;
            if let ModelKernelArtifact::DslProgram(p) = art {
                let wgsl = lower_kernel_program_to_wgsl(&p)?;
                // Bindings from the typed program (carry access kind); the emitted
                // WGSL `@group/@binding` slots are derived from these same specs.
                let bindings: Vec<BindInfo> = p
                    .bindings
                    .iter()
                    .map(|b| BindInfo {
                        group: b.group,
                        binding: b.binding,
                        name: b.name.clone(),
                        access: b.access,
                    })
                    .collect();
                // The kernel's `Constants` tail (EOS params beyond the canonical base
                // fields), read from the EMITTED WGSL — authoritative, since the
                // codegen derives them from referenced params, not p.eos_params
                // (which is empty for e.g. bc_expr yet its struct has eos_gm1/eos_r).
                let src = wgsl.to_wgsl();
                let eos_fields = parse_constants_eos_fields(&src);
                out.push((spec.id.as_str().to_string(), src, bindings, eos_fields));
            }
        }
    }
    Ok(out)
}

/// The `struct Constants` fields beyond the canonical base fields, in order,
/// parsed from emitted WGSL. These are the EOS/buoyant params a kernel appends;
/// they drive per-kernel `constants` uniform packing so `eos_*` reads land at the
/// right offset (bc_expr appends only the params it references, not the full set).
fn parse_constants_eos_fields(wgsl: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut in_struct = false;
    for line in wgsl.lines() {
        let t = line.trim();
        if t.starts_with("struct Constants") {
            in_struct = true;
            continue;
        }
        if in_struct {
            if t.starts_with('}') {
                break;
            }
            // Field line: `name: type,`
            if let Some(colon) = t.find(':') {
                let name = t[..colon].trim();
                if !name.is_empty() {
                    fields.push(name.to_string());
                }
            }
        }
    }
    let base_len = cfd2_codegen::solver::codegen::constants::base_constant_field_names().len();
    if fields.len() > base_len {
        fields.split_off(base_len)
    } else {
        Vec::new()
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ===========================================================================
// StructuredGpuSolver
// ===========================================================================

/// A structured-grid GPU solver: runs the codegen `Structured2D` kernels on the
/// device and closes the loop with a matrix-free banded solve.
pub struct StructuredGpuSolver {
    ctx: GpuContext,
    grid: StructuredGrid,
    n: usize,
    s: usize,
    state_stride: usize,
    layout: crate::solver::model::backend::state_layout::StateLayout,

    prep: Vec<String>,
    per_iter: Vec<String>,
    update: Vec<String>,
    kernels: HashMap<String, CompiledKernel>,

    buffers: HashMap<String, wgpu::Buffer>,
    constants: GpuConstants,
    constants_buf: wgpu::Buffer,
    grid_buf: wgpu::Buffer,
    /// The `low_mach_params` uniform the all-Mach/compressible kernels bind.
    /// Held at the CPU structured solver's neutral values (`model=0`, others 0).
    low_mach_buf: wgpu::Buffer,

    solver: BandedGpuLinAlg,

    /// The model's declared Schur block layout (`None` if the model declares no
    /// Schur preconditioner, e.g. the density-based compressible model). Lets
    /// `set_preconditioner` rebuild the host coupled solve's preconditioner for
    /// any of {block-Jacobi, Schur, Schur+AMG} without re-reading the model.
    schur_layout: Option<crate::solver::banded_schur::SchurLayout>,

    /// State-layout offsets of the solved unknowns (not auxiliaries like
    /// `ibm_penalty_U`), GROUPED by solved field. Drives the applied Picard
    /// residual — same set and grouping the CPU structured solver uses.
    unknown_state_groups: Vec<Vec<usize>>,

    outer_iters: usize,
    /// When true, the Picard loop may exit early once every unknown's
    /// per-field scaled applied residual falls below [`Self::outer_tol`]
    /// (GUI `outer_auto_converge`). Residual is `|state − state_iter| /
    /// max(|state|, 1)` after under-relaxation — CPU parity; not absolute `|x|`.
    outer_auto_converge: bool,
    /// Per-unknown relative applied-change threshold for outer early-exit
    /// (`1e-3` default). Matches CPU structured / unstructured outer tol.
    outer_tol: f32,
    dt: f64,
    /// Previous step's `dt` — BDF2 variable-step coefficients read `r = dt/dt_old`
    /// (must lag `dt` under adaptive CFL; unstructured TimeIntegrationModule).
    dt_old: f64,
    /// Committed steps — BDF2→Euler startup when `step_count == 0`.
    step_count: u64,
    /// Requested time scheme (BDF2 falls back to Euler on step 0).
    time_scheme: crate::solver::TimeScheme,
    /// Model id (for the GUI's model-echo / caps) and accumulated sim time.
    model_id: &'static str,
    time: f64,
    /// Convergence telemetry from the last [`step`](Self::step) (GUI readout;
    /// parity with the CPU `StructuredModelSolver`).
    last_stats: crate::solver::banded_schur::StructuredStepStats,
    /// Selective conserved-field low-pass filter (explicit RK4, density-based
    /// compressible family only; `None` otherwise). Runs once per logical RK4
    /// step when its runtime `sigma` is nonzero.
    filter: Option<StructuredConservedFilter>,
    /// GPU-resident health/adaptive controller for bounded autonomous explicit
    /// batches. `None` for implicit program families.
    autonomous: Option<StructuredAutonomousControl>,
    autonomous_initialized: bool,
    autonomous_test_nan_after: u32,
    autonomous_test_negative_energy_after: u32,
}

impl StructuredGpuSolver {
    /// Build a structured GPU solver for a coupled model on `grid`. Creates its
    /// own headless device/queue.
    pub fn new(
        grid: StructuredGrid,
        model: &ModelSpec,
        dt: f64,
        outer_iters: usize,
    ) -> Result<Self, String> {
        let ctx = pollster::block_on(GpuContext::new(None, None))?;
        Self::with_context(ctx, grid, model, dt, outer_iters)
    }

    /// Build against an existing device/queue with the default schemes
    /// (Upwind / backward-Euler).
    pub fn with_context(
        ctx: GpuContext,
        grid: StructuredGrid,
        model: &ModelSpec,
        dt: f64,
        outer_iters: usize,
    ) -> Result<Self, String> {
        Self::with_config(
            ctx,
            grid,
            model,
            dt,
            outer_iters,
            Scheme::Upwind,
            TimeScheme::Euler,
        )
    }

    /// Build against an existing device/queue with an explicit advection scheme
    /// and time-integration scheme. Euler/BDF2 use the coupled banded program;
    /// RK4 selects the fully explicit matrix-free program.
    pub fn with_config(
        ctx: GpuContext,
        grid: StructuredGrid,
        model: &ModelSpec,
        dt: f64,
        outer_iters: usize,
        scheme: Scheme,
        time_scheme: TimeScheme,
    ) -> Result<Self, String> {
        if model.system.topology() != cfd2_ir::equation::TopologyMode::Structured2D {
            return Err("StructuredGpuSolver requires a Structured2D model".to_string());
        }
        let stepping = if time_scheme == TimeScheme::RK4 {
            SteppingMode::Explicit
        } else {
            SteppingMode::Coupled
        };
        let recipe = SolverRecipe::from_model(
            model,
            scheme,
            time_scheme,
            PreconditionerType::Jacobi,
            stepping,
        )?;
        let schemes = SchemeRegistry::new(scheme);

        let lowered = lower_structured_kernels(model, &schemes)?;
        let wgsl_by_id: HashMap<String, (String, Vec<BindInfo>, Vec<String>)> = lowered
            .into_iter()
            .map(|(id, wgsl, bindings, eos)| (id, (wgsl, bindings, eos)))
            .collect();

        let n = grid.num_cells();
        let s = model.system.unknowns_per_cell() as usize;
        let state_stride = model.state_layout.stride() as usize;
        let flux_stride = recipe.flux.map(|f| f.stride as usize).unwrap_or(1);

        // Group the schedule exactly like the CPU structured solver: prep once;
        // per-iter gradients/flux/assembly; update/recovery. LinearSolve is
        // replaced by the banded solve; Apply (monitor matvec) is skipped.
        let mut prep = Vec::new();
        let mut per_iter = Vec::new();
        let mut update = Vec::new();
        let mut kernels: HashMap<String, CompiledKernel> = HashMap::new();
        let mut compile = |id: &str| -> Result<(), String> {
            if kernels.contains_key(id) {
                return Ok(());
            }
            let (wgsl, bindings, eos_fields) = wgsl_by_id
                .get(id)
                .ok_or_else(|| format!("structured GPU model missing kernel `{id}`"))?;
            kernels.insert(
                id.to_string(),
                CompiledKernel::build(&ctx.device, id, wgsl, bindings.clone(), eos_fields.clone()),
            );
            Ok(())
        };
        for k in &recipe.kernels {
            let id = k.id.as_str().to_string();
            match k.phase {
                KernelPhase::Preparation => {
                    compile(&id)?;
                    prep.push(id);
                }
                KernelPhase::Gradients | KernelPhase::FluxComputation | KernelPhase::Assembly => {
                    compile(&id)?;
                    per_iter.push(id);
                }
                KernelPhase::Update | KernelPhase::PrimitiveRecovery => {
                    compile(&id)?;
                    update.push(id);
                }
                _ => continue,
            }
        }
        if std::env::var("CFD2_STRUCTGPU_DEBUG").is_ok() {
            eprintln!("[structgpu] s={s} state_stride={state_stride} flux_stride={flux_stride}");
            eprintln!("[structgpu] prep={prep:?}");
            eprintln!("[structgpu] per_iter={per_iter:?}");
            eprintln!("[structgpu] update={update:?}");
            for (id, k) in &kernels {
                let names: Vec<&str> = k.bindings.iter().map(|b| b.name.as_str()).collect();
                eprintln!("[structgpu]   {id}: {names:?}");
            }
        }

        // Buffers (by name), sized as on the CPU structured solver.
        let mut buffers = HashMap::new();
        let dev = &ctx.device;
        for name in ["state", "state_old", "state_old_old", "state_iter"] {
            buffers.insert(
                name.to_string(),
                storage_buffer(dev, name, n * state_stride),
            );
        }
        buffers.insert("rhs".to_string(), storage_buffer(dev, "rhs", n * s));
        if stepping == SteppingMode::Explicit {
            buffers.insert("rk_base".to_string(), storage_buffer(dev, "rk_base", n * s));
            buffers.insert(
                "rk_accum".to_string(),
                storage_buffer(dev, "rk_accum", n * s),
            );
            buffers.insert(
                "accepted_state".to_string(),
                storage_buffer(dev, "accepted_state", n * state_stride),
            );
            buffers.insert(
                "accepted_history".to_string(),
                storage_buffer(dev, "accepted_history", 2 * n * state_stride),
            );
        } else {
            buffers.insert(
                "matrix_values".to_string(),
                storage_buffer(dev, "matrix_values", n * BAND_STRIDE * s * s),
            );
            buffers.insert("x".to_string(), storage_buffer(dev, "x", n * s));
            buffers.insert("y".to_string(), storage_buffer(dev, "y", n * s));
        }
        buffers.insert(
            "fluxes".to_string(),
            storage_buffer(dev, "fluxes", n * 4 * flux_stride),
        );
        buffers.insert(
            "grad_state".to_string(),
            storage_buffer(dev, "grad_state", n * state_stride * 2),
        );
        buffers.insert(
            "bc_kind".to_string(),
            storage_buffer(dev, "bc_kind", n * 4 * s),
        );
        buffers.insert(
            "bc_value".to_string(),
            storage_buffer(dev, "bc_value", n * 4 * s),
        );
        buffers.insert(
            "face_boundary".to_string(),
            storage_buffer(dev, "face_boundary", n * 4),
        );

        // `recipe.initial_constants` already carries the advection `scheme` and
        // `time_scheme` (from `from_model`); keep them so the runtime kernels honour
        // the requested schemes. Only the per-run knobs are overridden here.
        let mut constants = recipe.initial_constants;
        constants.dt = dt as f32;
        constants.dt_old = dt as f32;
        constants.dtau = 0.0;
        // Generated kernels flatten a potentially two-dimensional dispatch as
        // `global_id.y * stride_x + global_id.x`.  Use the same launch-row
        // stride as the generic GPU runtime so grids above WebGPU's 65,535
        // workgroups-per-dimension limit remain addressable.
        constants.stride_x = dev
            .limits()
            .max_compute_workgroups_per_dimension
            .checked_mul(WG)
            .expect("structured dispatch-row stride overflow");
        // Pressure under-relaxation for the coupled Schur-layout models (mirrors the
        // CPU `StructuredModelSolver` + the driver's alpha_p=0.3): the update kernel
        // applies `phi = phi_old + alpha*(x-phi_old)`, damping the saddle outer loop.
        if crate::solver::banded_schur::schur_layout_from_model(model).is_some() {
            constants.alpha_p = 0.3;
            constants.alpha_u = 0.7;
        }
        let constants_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("constants"),
            contents: bytemuck::bytes_of(&constants),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let grid_gpu = GpuStructuredGrid {
            nx: grid.nx as u32,
            ny: grid.ny as u32,
            dx: grid.dx as f32,
            dy: grid.dy as f32,
        };
        let grid_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid"),
            contents: bytemuck::bytes_of(&grid_gpu),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Neutral low-Mach params (model=Off/0), matching the CPU structured
        // solver's build_ctx — no preconditioning, no biharmonic dissipation.
        let low_mach = crate::solver::gpu::structs::GpuLowMachParams {
            model: 0,
            theta_floor: 0.0,
            pressure_coupling_alpha: 0.0,
            eps4: 0.0,
        };
        let low_mach_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("low_mach_params"),
            contents: bytemuck::bytes_of(&low_mach),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let solver = BandedGpuLinAlg::new(dev, grid.nx as u32, grid.ny as u32, s as u32);

        let unknown_state_groups: Vec<Vec<usize>> =
            crate::solver::model::kernel::model_unknown_state_offset_groups(model)?
                .into_iter()
                .map(|g| g.into_iter().map(|o| o as usize).collect())
                .collect();

        // Pack each kernel's `constants` uniform to its own `Constants` layout.
        for k in kernels.values() {
            k.write_constants(&ctx.queue, &constants);
        }

        // Selective conserved-field filter: explicit RK4 + a density-based
        // conserved layout only. Other families build no buffers/pipelines.
        let filter = (stepping == SteppingMode::Explicit)
            .then(|| {
                StructuredConservedFilter::new(
                    dev,
                    &grid_buf,
                    &buffers["state"],
                    n,
                    state_stride,
                    &model.state_layout,
                )
            })
            .flatten();

        let autonomous = (stepping == SteppingMode::Explicit).then(|| {
            StructuredAutonomousControl::new(
                dev,
                &grid_buf,
                &buffers,
                &model.state_layout,
                model.id,
                n,
                s,
                &prep,
                &per_iter,
                &kernels,
                &constants,
            )
        });

        Ok(Self {
            ctx,
            grid,
            n,
            s,
            state_stride,
            layout: model.state_layout.clone(),
            prep,
            per_iter,
            update,
            kernels,
            buffers,
            constants,
            constants_buf,
            grid_buf,
            low_mach_buf,
            solver,
            schur_layout: crate::solver::banded_schur::schur_layout_from_model(model),
            unknown_state_groups,
            outer_iters: outer_iters.max(1),
            outer_auto_converge: false,
            outer_tol: 1e-3,
            dt,
            dt_old: dt,
            step_count: 0,
            time_scheme,
            model_id: model.id,
            time: 0.0,
            last_stats: crate::solver::banded_schur::StructuredStepStats::default(),
            filter,
            autonomous,
            autonomous_initialized: false,
            autonomous_test_nan_after: u32::MAX,
            autonomous_test_negative_energy_after: u32::MAX,
        })
    }

    fn buf(&self, name: &str) -> &wgpu::Buffer {
        // `constants`/`grid`/`low_mach_params` are uniforms; everything else is a
        // named storage buffer.
        match name {
            "constants" => &self.constants_buf,
            "grid" => &self.grid_buf,
            "low_mach_params" => &self.low_mach_buf,
            other => self
                .buffers
                .get(other)
                .unwrap_or_else(|| panic!("structured GPU solver: no buffer `{other}`")),
        }
    }

    fn upload_f32(&self, name: &str, data: &[f32]) {
        self.ctx
            .queue
            .write_buffer(self.buf(name), 0, bytemuck::cast_slice(data));
    }
    fn upload_u32(&self, name: &str, data: &[u32]) {
        self.ctx
            .queue
            .write_buffer(self.buf(name), 0, bytemuck::cast_slice(data));
    }

    /// Read a named storage buffer back as `f32` (length `len`).
    fn read_f32(&self, name: &str, len: usize) -> Vec<f32> {
        read_buffer_f32(&self.ctx, self.buf(name), len)
    }

    /// Seed a named state field from a closure of cell-centre coords (writes all
    /// history buffers — IC semantics).
    pub fn set_named_field<F: Fn(f64, f64) -> f64>(&mut self, name: &str, f: F) {
        let off = self
            .layout
            .offset_for(name)
            .unwrap_or_else(|| panic!("structured GPU solver: no state field `{name}`"))
            as usize;
        // Read-modify-write the packed state (component `off` of stride).
        let mut state = self.read_f32("state", self.n * self.state_stride);
        for p in 0..self.n {
            let (x, y) = self.grid.cell_center(p);
            state[p * self.state_stride + off] = f(x, y) as f32;
        }
        self.upload_f32("state", &state);
        self.upload_f32("state_old", &state);
        self.upload_f32("state_old_old", &state);
        if self.buffers.contains_key("accepted_state") {
            self.upload_f32("accepted_state", &state);
            let mut accepted_history = Vec::with_capacity(state.len() * 2);
            accepted_history.extend_from_slice(&state);
            accepted_history.extend_from_slice(&state);
            self.upload_f32("accepted_history", &accepted_history);
            self.autonomous_initialized = false;
        }
    }

    /// Seed a packed state component (by state-layout offset) from cell-centre
    /// coords — writes all history buffers (IC semantics). Mirrors the CPU
    /// `StructuredModelSolver::set_state`.
    pub fn set_state_component<F: Fn(f64, f64) -> f64>(&mut self, offset: usize, f: F) {
        let mut state = self.read_f32("state", self.n * self.state_stride);
        for p in 0..self.n {
            let (x, y) = self.grid.cell_center(p);
            state[p * self.state_stride + offset] = f(x, y) as f32;
        }
        self.upload_f32("state", &state);
        self.upload_f32("state_old", &state);
        self.upload_f32("state_old_old", &state);
        if self.buffers.contains_key("accepted_state") {
            self.upload_f32("accepted_state", &state);
            let mut accepted_history = Vec::with_capacity(state.len() * 2);
            accepted_history.extend_from_slice(&state);
            accepted_history.extend_from_slice(&state);
            self.upload_f32("accepted_history", &accepted_history);
            self.autonomous_initialized = false;
        }
    }

    /// Seed the complete packed state in one upload and initialize every
    /// history buffer to the same values. This is the bulk, math-layout-driven
    /// counterpart of [`set_named_field`](Self::set_named_field): callers build
    /// the row from the model's [`StateLayout`] offsets instead of paying one
    /// read/modify/write round trip per field.
    pub fn set_packed_state_f32(&mut self, state: &[f32]) -> Result<(), String> {
        let expected = self.n * self.state_stride;
        if state.len() != expected {
            return Err(format!(
                "structured packed state length mismatch: got {}, expected {} cells * {} stride = {expected}",
                state.len(), self.n, self.state_stride
            ));
        }
        for name in ["state", "state_old", "state_old_old", "state_iter"] {
            self.upload_f32(name, state);
        }
        if self.buffers.contains_key("accepted_state") {
            self.upload_f32("accepted_state", state);
            let mut accepted_history = Vec::with_capacity(state.len() * 2);
            accepted_history.extend_from_slice(state);
            accepted_history.extend_from_slice(state);
            self.upload_f32("accepted_history", &accepted_history);
            self.autonomous_initialized = false;
        }
        Ok(())
    }

    /// Coupled unknowns per cell (banded block stride).
    pub fn unknowns(&self) -> usize {
        self.s
    }

    pub fn field_offset(&self, name: &str) -> Option<usize> {
        self.layout.offset_for(name).map(|o| o as usize)
    }

    pub fn set_fluid(&mut self, density: f64, viscosity: f64) {
        self.constants.density = density as f32;
        self.constants.viscosity = viscosity as f32;
        self.write_kernel_constants();
        self.autonomous_initialized = false;
    }

    /// (Re)pack every kernel's `constants` uniform from the current `GpuConstants`
    /// — each to its own `Constants` struct layout (base fields + its EOS params).
    fn write_kernel_constants(&self) {
        for k in self.kernels.values() {
            k.write_constants(&self.ctx.queue, &self.constants);
        }
    }

    /// Runtime strength of the selective conserved-field low-pass filter
    /// applied once per logical explicit-RK4 step (see
    /// [`structured_conserved_filter_wgsl`]). `0.0` (the default) skips the
    /// pass entirely at every dispatch site, so it is trivially bit-inert.
    /// Takes effect immediately (uniform write, no rebuild). A no-op on
    /// models without the conserved `rho`/`rho_u`/`rho_e` layout.
    pub fn set_filter_sigma(&mut self, sigma: f32) {
        if let Some(filter) = self.filter.as_mut() {
            filter.sigma = sigma;
            self.ctx.queue.write_buffer(
                &filter.params,
                0,
                bytemuck::cast_slice(&[sigma, 0.0_f32, 0.0, 0.0]),
            );
        }
    }

    /// Apply the complete runtime EOS block used by generated recovery, flux,
    /// boundary-expression, and autonomous-control kernels.
    pub fn set_eos(&mut self, params: EosRuntimeParams) {
        self.constants.eos_gamma = params.gamma;
        self.constants.eos_gm1 = params.gm1;
        self.constants.eos_r = params.r;
        self.constants.eos_dp_drho = params.dp_drho;
        self.constants.eos_p_ref = params.p_ref;
        self.constants.eos_theta_ref = params.theta_ref;
        self.constants.eos_rho_ref = params.rho_ref;
        self.constants.eos_gauge_rho_ref = params.gauge_rho_ref;
        self.constants.eos_gauge_p_ref = params.gauge_p_ref;
        self.constants.eos_gauge_e_ref = params.gauge_e_ref;
        self.constants.eos_gauge_p_bias = params.gauge_p_bias;
        self.constants.bc_pressure_inlet = params.bc_pressure_inlet;
        self.constants.eos_p_floor = params.p_floor;
        self.constants.eos_t_floor = params.t_floor;
        self.constants.eos_rho_floor = params.rho_floor;
        self.write_kernel_constants();
        // A live EOS switch changes the conserved-domain oracle and its first
        // stable dt. Force the next autonomous entry to seed both from the
        // accepted host metadata instead of continuing a stale control tail.
        self.autonomous_initialized = false;
    }

    /// Impose a per-`(cell, dir)` boundary type + per-unknown BC. Mirrors the CPU
    /// `StructuredModelSolver::set_boundaries`: `f(edge, fx, fy) -> (btype, comps)`.
    pub fn set_boundaries<F: Fn(Edge, f64, f64) -> (u32, Vec<BcComp>)>(&mut self, f: F) {
        let (nx, ny, s) = (self.grid.nx, self.grid.ny, self.s);
        let (dx, dy) = (self.grid.dx, self.grid.dy);
        let n = self.n;
        let mut bc_kind = vec![0u32; n * 4 * s];
        let mut bc_value = vec![0.0f32; n * 4 * s];
        let mut face_boundary = vec![0u32; n * 4];
        for j in 0..ny {
            for i in 0..nx {
                let p = j * nx + i;
                let (cx, cy) = self.grid.cell_center(p);
                // Direction order S=0, W=1, E=2, N=3 (the codegen `k`).
                let faces = [
                    (0usize, j == 0, Edge::Bottom, cx, cy - 0.5 * dy),
                    (1, i == 0, Edge::Left, cx - 0.5 * dx, cy),
                    (2, i == nx - 1, Edge::Right, cx + 0.5 * dx, cy),
                    (3, j == ny - 1, Edge::Top, cx, cy + 0.5 * dy),
                ];
                for (k, is_b, edge, fx, fy) in faces {
                    if !is_b {
                        continue;
                    }
                    let (btype, comps) = f(edge, fx, fy);
                    face_boundary[p * 4 + k] = btype;
                    for (c, bc) in comps.iter().enumerate().take(s) {
                        let idx = (p * 4 + k) * s + c;
                        bc_kind[idx] = bc.kind;
                        bc_value[idx] = bc.value;
                    }
                }
            }
        }
        self.upload_u32("bc_kind", &bc_kind);
        self.upload_f32("bc_value", &bc_value);
        self.upload_u32("face_boundary", &face_boundary);
    }

    /// Copy `src -> dst` in its own submission.
    fn copy_submit(&self, src: &str, dst: &str, len: usize) {
        self.copy_many_submit(&[(src, dst, len)]);
    }

    /// Record dependency-ordered buffer copies in one command buffer. Copy
    /// commands are ordered, so history rotation (`old -> old_old`, then
    /// `state -> old`) is identical to separate queue submissions.
    fn copy_many_submit(&self, copies: &[(&str, &str, usize)]) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy"),
            });
        for &(src, dst, len) in copies {
            enc.copy_buffer_to_buffer(self.buf(src), 0, self.buf(dst), 0, (len * 4) as u64);
        }
        self.ctx.queue.submit(Some(enc.finish()));
        crate::count_submission!("Structured", "copy_many");
    }

    /// Dispatch each kernel in its own submission. Separate command buffers are
    /// strictly ordered on the queue with full memory visibility between them, so
    /// the chained schedule (flux -> gradients -> assembly) sees each prior
    /// kernel's storage writes.
    fn dispatch_ids(&self, ids: &[String]) {
        let resolve = |name: &str| self.buf(name);
        for id in ids {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(id) });
            let k = &self.kernels[id];
            k.dispatch(&self.ctx.device, &mut enc, self.n as u32, &resolve);
            self.ctx.queue.submit(Some(enc.finish()));
            crate::count_submission!("Structured", "kernel");
        }
    }

    fn dispatch_ids_uncached(&self, ids: &[String]) {
        let resolve = |name: &str| self.buf(name);
        for id in ids {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(id) });
            self.kernels[id].dispatch_uncached(&self.ctx.device, &mut enc, self.n as u32, &resolve);
            self.ctx.queue.submit(Some(enc.finish()));
            crate::count_submission!("Structured", "kernel_legacy");
        }
    }

    /// Encode a dependency-ordered generated schedule into one compute pass and
    /// one queue submission. WebGPU gives each dispatch its own usage scope, so
    /// storage writes from gradients/flux/residual are visible to the next
    /// dispatch while avoiding per-kernel pass transitions, submissions, and
    /// bind-group construction.
    fn dispatch_ids_batched(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("structured:batched"),
            });
        self.encode_dispatch_ids(&mut enc, ids, "structured:batched-pass");
        self.ctx.queue.submit(Some(enc.finish()));
        crate::count_submission!("Structured", "batched_schedule");
    }

    /// Encode a dependency-ordered generated schedule as one compute pass in an
    /// existing command buffer.  Ending the pass is significant for autonomous
    /// RK batches: it creates a legal usage boundary before the next stage
    /// copies a new immutable constant snapshot into the same uniform buffers.
    fn encode_dispatch_ids(&self, encoder: &mut wgpu::CommandEncoder, ids: &[String], label: &str) {
        if ids.is_empty() {
            return;
        }
        let resolve = |name: &str| self.buf(name);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        for id in ids {
            self.kernels[id].dispatch_in_pass(&self.ctx.device, &mut pass, self.n as u32, &resolve);
        }
    }

    fn encode_dispatch_ids_indirect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ids: &[String],
        indirect_args: &wgpu::Buffer,
        label: &str,
    ) {
        if ids.is_empty() {
            return;
        }
        let resolve = |name: &str| self.buf(name);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        for id in ids {
            self.kernels[id].dispatch_in_pass_indirect(
                &self.ctx.device,
                &mut pass,
                indirect_args,
                &resolve,
            );
        }
    }

    /// Assemble the banded operator once (advancing history, running prep +
    /// per-iter kernels) WITHOUT solving — for operator-parity tests.
    pub fn assemble_only(&mut self) {
        let n = self.n;
        let sstride = self.state_stride;
        // Match `step()` (and the CPU `assemble_only`): the time-step size drives
        // the `ddt` diagonal — without this re-sync a `set_dt` after construction
        // assembled with the stale constructor dt.
        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt_old as f32;
        self.write_kernel_constants();
        self.copy_submit("state", "state_old", n * sstride);
        self.copy_submit("state", "state_iter", n * sstride);
        let ids: Vec<String> = self
            .prep
            .iter()
            .chain(self.per_iter.iter())
            .cloned()
            .collect();
        self.dispatch_ids(&ids);
    }

    /// Read the assembled banded operator (`N*5*s*s`).
    pub fn matrix_values(&self) -> Vec<f32> {
        self.read_f32("matrix_values", self.n * BAND_STRIDE * self.s * self.s)
    }
    /// Read the assembled RHS (`N*s`).
    pub fn rhs(&self) -> Vec<f32> {
        self.read_f32("rhs", self.n * self.s)
    }

    /// Read a named storage buffer back as `f32` (debug/parity).
    pub fn read_named(&self, name: &str, len: usize) -> Vec<f32> {
        self.read_f32(name, len)
    }

    /// Host-routed RK4 retained for benchmark ablations.  Production uses
    /// [`Self::step_explicit_rk4_batch`], including for a one-step batch.
    fn step_explicit_rk4_host_routed(&mut self, legacy_router: bool) {
        // The host clock becomes authoritative again on any ordinary step.
        // A later autonomous entry must reseed its device Q64 clock instead of
        // continuing from a stale controller tail.
        self.autonomous_initialized = false;
        let state_len = self.n * self.state_stride;
        let separate_cached = std::env::var_os("CFD2_STRUCTGPU_NO_EXPLICIT_BATCH").is_some();

        // Rotate history once for the complete RK step. RK4 itself is one-step,
        // but the public history contract remains identical to Euler/BDF2.
        if legacy_router {
            self.copy_submit("state_old", "state_old_old", state_len);
            self.copy_submit("state", "state_old", state_len);
        } else {
            self.copy_many_submit(&[
                ("state_old", "state_old_old", state_len),
                ("state", "state_old", state_len),
            ]);
        }

        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt_old as f32;
        self.constants.dtau = 0.0;
        self.constants.time_scheme = TimeScheme::RK4 as u32;

        let stage_ids = [
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_1.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_2.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_3.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_4.as_str(),
        ];
        let stage_times = [0.0_f64, 0.5, 0.5, 1.0];
        let base_time = self.time;
        // Expression-valued BCs depend on the evolving stage state, so refresh
        // them every stage. Other Preparation kernels retain their one-time
        // semantics and are intentionally excluded, matching the CPU oracle.
        let prep: Vec<String> = self
            .prep
            .iter()
            .filter(|id| id.contains("bc_expr"))
            .cloned()
            .collect();
        let residual = self.per_iter.clone();

        for (&stage_id, &c) in stage_ids.iter().zip(stage_times.iter()) {
            self.constants.time = (base_time + c * self.dt) as f32;
            self.write_kernel_constants();
            if legacy_router {
                self.dispatch_ids_uncached(&prep);
                self.dispatch_ids_uncached(&residual);
                self.dispatch_ids_uncached(&[stage_id.to_string()]);
            } else if separate_cached {
                self.dispatch_ids(&prep);
                self.dispatch_ids(&residual);
                self.dispatch_ids(&[stage_id.to_string()]);
            } else {
                let mut stage_schedule = Vec::with_capacity(prep.len() + residual.len() + 1);
                stage_schedule.extend(prep.iter().cloned());
                stage_schedule.extend(residual.iter().cloned());
                stage_schedule.push(stage_id.to_string());
                self.dispatch_ids_batched(&stage_schedule);
            }
        }

        // Selective conserved-field filter, once per logical step after RK
        // stage 4 (host-side skip at sigma == 0.0 keeps the default bit-inert).
        if let Some(filter) = self.filter.as_ref().filter(|f| f.sigma != 0.0) {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("structured:conserved-filter"),
                });
            filter.encode(&mut enc, self.n as u32);
            self.ctx.queue.submit(Some(enc.finish()));
            crate::count_submission!("Structured", "conserved_filter");
        }

        self.dt_old = self.dt;
        self.step_count = self.step_count.saturating_add(1);
        self.time = base_time + self.dt;
        self.constants.time = self.time as f32;
        self.write_kernel_constants();
        self.last_stats = crate::solver::banded_schur::StructuredStepStats::default();
    }

    /// Encode `steps` complete fixed-`dt` classical RK4 steps into one command
    /// buffer and submit it once.  History rotation occurs at the start of every
    /// encoded step, and every stage reevaluates the generated preparation and
    /// residual schedule at its exact physical abscissa.
    ///
    /// Generated kernels bind distinct uniform buffers and, in general,
    /// distinct `Constants` layouts.  Mutating those buffers with
    /// `Queue::write_buffer` between submissions is therefore not available
    /// inside an autonomous batch.  Instead, this method packs every
    /// `(step,stage,kernel)` constant value into one immutable `COPY_SRC` buffer,
    /// then encodes ordered copies into each kernel's uniform immediately before
    /// its stage compute pass.  There are no per-stage host writes or queue
    /// submissions.  The returned submission index is the completion fence for
    /// the whole batch; `None` denotes the specified zero-step no-op.
    fn step_explicit_rk4_batch(&mut self, steps: usize) -> Option<wgpu::SubmissionIndex> {
        if steps == 0 {
            return None;
        }
        self.autonomous_initialized = false;

        let stage_ids = [
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_1.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_2.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_3.as_str(),
            crate::solver::model::KernelId::EXPLICIT_RK4_STAGE_4.as_str(),
        ];
        let stage_times = [0.0_f64, 0.5, 0.5, 1.0];
        let prep: Vec<String> = self
            .prep
            .iter()
            .filter(|id| id.contains("bc_expr"))
            .cloned()
            .collect();
        let residual = self.per_iter.clone();

        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt_old as f32;
        self.constants.dtau = 0.0;
        self.constants.time_scheme = TimeScheme::RK4 as u32;

        // Build immutable, kernel-layout-specific uniform snapshots before
        // encoding.  Advance the f64 cursor by repeated addition exactly as B
        // calls to `step()` do; using `start + step*dt` can differ after long
        // runs and would perturb an f32 stage-time boundary evaluation.
        let mut constant_bytes = Vec::<u8>::new();
        let mut encoded_steps = Vec::with_capacity(steps);
        let mut time_cursor = self.time;
        for _ in 0..steps {
            let base_time = time_cursor;
            let mut stages = Vec::with_capacity(4);
            for (&stage_id, &c) in stage_ids.iter().zip(stage_times.iter()) {
                self.constants.time = (base_time + c * self.dt) as f32;
                let mut schedule = Vec::with_capacity(prep.len() + residual.len() + 1);
                schedule.extend(prep.iter().cloned());
                schedule.extend(residual.iter().cloned());
                schedule.push(stage_id.to_string());

                let mut constant_copies = Vec::with_capacity(schedule.len());
                for id in &schedule {
                    let kernel = &self.kernels[id];
                    let packed = pack_kernel_constants(&self.constants, &kernel.eos_fields);
                    let offset = constant_bytes.len() as u64;
                    let size = packed.len() as u64;
                    debug_assert_eq!(offset % wgpu::COPY_BUFFER_ALIGNMENT, 0);
                    debug_assert_eq!(size % wgpu::COPY_BUFFER_ALIGNMENT, 0);
                    constant_bytes.extend_from_slice(&packed);
                    constant_copies.push((id.clone(), offset, size));
                }
                stages.push(RkStageSnapshot {
                    schedule,
                    constant_copies,
                });
            }
            time_cursor = base_time + self.dt;
            encoded_steps.push(RkStepSnapshot { stages });
        }

        let snapshots = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("structured:rk4-constant-snapshots"),
                contents: &constant_bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let state_len_bytes = (self.n * self.state_stride * 4) as u64;
        // Selective conserved-field filter: encoded after stage 4 of every
        // logical step; host-side skip at sigma == 0.0 (bit-inert default).
        let step_filter = self.filter.as_ref().filter(|f| f.sigma != 0.0);
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("structured:rk4-step-batch"),
            });
        for step in &encoded_steps {
            // Preserve public history semantics after every logical step, not
            // merely at batch boundaries.
            encoder.copy_buffer_to_buffer(
                self.buf("state_old"),
                0,
                self.buf("state_old_old"),
                0,
                state_len_bytes,
            );
            encoder.copy_buffer_to_buffer(
                self.buf("state"),
                0,
                self.buf("state_old"),
                0,
                state_len_bytes,
            );

            for stage in &step.stages {
                for (id, offset, size) in &stage.constant_copies {
                    encoder.copy_buffer_to_buffer(
                        &snapshots,
                        *offset,
                        &self.kernels[id].constants_buf,
                        0,
                        *size,
                    );
                }
                self.encode_dispatch_ids(&mut encoder, &stage.schedule, "structured:rk4-stage");
            }
            if let Some(filter) = step_filter {
                filter.encode(&mut encoder, self.n as u32);
            }
        }

        let submission = self.ctx.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Structured", "rk4_step_batch");

        self.dt_old = self.dt;
        self.step_count = self
            .step_count
            .saturating_add(u64::try_from(steps).unwrap_or(u64::MAX));
        self.time = time_cursor;
        self.constants.time = self.time as f32;
        self.last_stats = crate::solver::banded_schur::StructuredStepStats::default();
        Some(submission)
    }

    /// One classical RK4 step. Every stage refreshes expression-valued boundary
    /// data and the spatial residual from that stage's state and physical time.
    fn step_explicit_rk4(&mut self) {
        // Benchmark-only ablations. The default is the autonomous one-submission
        // path; neither environment variable changes generated mathematics.
        let legacy_router = std::env::var_os("CFD2_STRUCTGPU_LEGACY_EXPLICIT_ROUTER").is_some();
        let separate_cached = std::env::var_os("CFD2_STRUCTGPU_NO_EXPLICIT_BATCH").is_some();
        if legacy_router || separate_cached {
            self.step_explicit_rk4_host_routed(legacy_router);
        } else {
            let _ = self.step_explicit_rk4_batch(1);
        }
    }

    /// Advance `steps` logical time steps.  Explicit RK4 is encoded as one GPU
    /// command buffer and one queue submission.  Other schemes retain their
    /// existing host-routed semantics and return no single completion fence.
    /// This conservative fallback keeps the method safe for generic callers;
    /// autonomous batching is currently a fixed-`dt`, static-mesh RK4 feature.
    pub fn step_batch(&mut self, steps: usize) -> Option<wgpu::SubmissionIndex> {
        if steps == 0 {
            return None;
        }
        if self.time_scheme == TimeScheme::RK4 {
            return self.step_explicit_rk4_batch(steps);
        }
        for _ in 0..steps {
            self.step();
        }
        None
    }

    /// Submit a bounded, GPU-controlled explicit RK4 batch with accepted-state
    /// health checking and rollback. `target_cfl = Some(cfl)` enables the
    /// model-derived structured adaptive policy; `None` retains fixed `dt`
    /// while still rejecting an invalid accepted state. No cell-state
    /// readback or host queue operation occurs between logical steps.
    ///
    /// The GPU control block is authoritative for `dt`, time and accepted-step
    /// count. Call [`Self::autonomous_status`] only at a presentation/completion
    /// fence when the host needs those tiny telemetry values.
    fn submit_autonomous_batch_inner(
        &mut self,
        steps: usize,
        target_cfl: Option<f32>,
        status_callback: Option<
            Box<dyn FnOnce(Result<StructuredAutonomousStatus, String>) + Send + 'static>,
        >,
    ) -> Result<Option<wgpu::SubmissionIndex>, String> {
        if steps == 0 {
            return Ok(None);
        }
        if self.time_scheme != TimeScheme::RK4 {
            return Err("structured autonomous batching requires explicit RK4".to_string());
        }
        if !self.autonomous_eos_is_certified() {
            return Err(format!(
                "GPU-resident conserved-state audit is not certified for the runtime EOS of structured model `{}`",
                self.model_id
            ));
        }
        let control = self
            .autonomous
            .as_ref()
            .ok_or_else(|| "structured autonomous control was not constructed".to_string())?;
        if target_cfl.is_some() && control.adaptive_policy == StructuredAdaptivePolicy::Unsupported
        {
            return Err(format!(
                "GPU-resident adaptive control is not derived for structured model `{}`",
                self.model_id
            ));
        }
        let status_lease = if status_callback.is_some() {
            Some(control.batch_cache.acquire_status_staging()?)
        } else {
            None
        };
        let bootstrap_dt = !self.autonomous_initialized;
        if bootstrap_dt {
            let (time_seconds, time_fraction_hi, time_fraction_lo) = split_q64_time(self.time);
            let initial = StructuredAutonomousControlRaw {
                dt: self.dt as f32,
                dt_old: self.dt_old as f32,
                time_seconds,
                time_fraction_hi,
                time_fraction_lo,
                max_base_bits: 0,
                max_rc_bits: 0,
                max_vel_bits: 0,
                invalid_cells: 0,
                halted: 0,
                accepted_batch: 0,
                accepted_total: self.step_count as u32,
                accepted_total_hi: (self.step_count >> 32) as u32,
            };
            self.ctx
                .queue
                .write_buffer(&control.control, 0, bytemuck::bytes_of(&initial));
            let groups = (self.n as u32).div_ceil(WG).max(1);
            let max_x = self
                .ctx
                .device
                .limits()
                .max_compute_workgroups_per_dimension;
            let indirect = [groups.min(max_x), groups.div_ceil(max_x), 1u32];
            self.ctx
                .queue
                .write_buffer(&control.indirect_args, 0, bytemuck::cast_slice(&indirect));
            self.autonomous_initialized = true;
        }
        let params = StructuredAutonomousParams {
            target_cfl: target_cfl.unwrap_or(0.0),
            density: self.constants.density,
            viscosity: self.constants.viscosity,
            safety: 0.8,
            dt_max: 100.0,
            adaptive: u32::from(target_cfl.is_some()),
            bootstrap_dt: u32::from(bootstrap_dt),
            state_stride: self.state_stride as u32,
            inject_nan_after: self.autonomous_test_nan_after,
            inject_negative_energy_after: self.autonomous_test_negative_energy_after,
            adaptive_policy: control.adaptive_policy as u32,
            has_ibm_penalty: u32::from(self.layout.offset_for("ibm_penalty_U").is_some()),
            inlet_velocity: self.constants.inlet_velocity,
            eos_gamma: self.constants.eos_gamma,
            eos_gm1: self.constants.eos_gm1,
            eos_r: self.constants.eos_r,
            eos_dp_drho: self.constants.eos_dp_drho,
            eos_p_ref: self.constants.eos_p_ref,
            eos_theta_ref: self.constants.eos_theta_ref,
            eos_rho_ref: self.constants.eos_rho_ref,
            eos_gauge_rho_ref: self.constants.eos_gauge_rho_ref,
            eos_gauge_e_ref: self.constants.eos_gauge_e_ref,
            // Absolute floor: stored-form floor + gauge reference (exactly
            // 1.0 Pa on gauged production runs; ~f32::MIN = inert otherwise).
            eos_p_floor_abs: self.constants.eos_p_floor + self.constants.eos_gauge_p_ref,
            eos_t_floor: self.constants.eos_t_floor,
            eos_rho_floor: self.constants.eos_rho_floor,
        };
        self.ctx
            .queue
            .write_buffer(&control.params, 0, bytemuck::bytes_of(&params));

        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt_old as f32;
        self.constants.dtau = 0.0;
        self.constants.time_scheme = TimeScheme::RK4 as u32;
        control.batch_cache.refresh_snapshots(
            &self.ctx.queue,
            &self.constants,
            &self.kernels,
        );

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("structured:autonomous-rk4-batch"),
            });
        control.encode_common(
            &mut encoder,
            &control.reset_batch,
            1,
            "structured:auto-reset-batch",
        );
        control.encode_sample(&mut encoder, self.n as u32, target_cfl.is_some());
        control.encode_common(
            &mut encoder,
            &control.bootstrap,
            1,
            "structured:auto-bootstrap",
        );

        let state_threads = (self.n * self.state_stride) as u32;
        // Selective conserved-field filter: dispatched through the SAME
        // per-cell indirect-args slot as the stage kernels, so a halted batch
        // (zeroed args) never filters a rolled-back accepted state. Encoded
        // after stage 4 and BEFORE the audit, so acceptance judges the
        // filtered candidate; rollback restores `state` (the only buffer
        // `filter_y` writes) from the accepted prefix.
        let step_filter = self.filter.as_ref().filter(|f| f.sigma != 0.0);
        for _ in 0..steps {
            control.encode_common(
                &mut encoder,
                &control.history,
                state_threads,
                "structured:auto-history",
            );
            for (stage, schedule) in control.batch_cache.schedules.iter().enumerate() {
                let patch_pipeline = match stage {
                    0 => &control.patch_stage_0,
                    1 | 2 => &control.patch_stage_half,
                    _ => &control.patch_stage_1,
                };
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("structured:auto-patch-constants"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(patch_pipeline);
                    pass.set_bind_group(0, &control.batch_cache.patch_bg, &[]);
                    pass.dispatch_workgroups(control.batch_cache.patch_groups, 1, 1);
                }
                for id in schedule {
                    let (offset, size) = control.batch_cache.records[id];
                    encoder.copy_buffer_to_buffer(
                        &control.batch_cache.snapshots,
                        offset,
                        &self.kernels[id].constants_buf,
                        0,
                        size,
                    );
                }
                self.encode_dispatch_ids_indirect(
                    &mut encoder,
                    schedule,
                    &control.indirect_args,
                    "structured:autonomous-rk4-stage",
                );
            }
            if let Some(filter) = step_filter {
                filter.encode_indirect(&mut encoder, &control.indirect_args);
            }
            control.encode_common(
                &mut encoder,
                &control.reset_metrics,
                1,
                "structured:auto-reset-metrics",
            );
            control.encode_sample(&mut encoder, self.n as u32, target_cfl.is_some());
            control.encode_common(&mut encoder, &control.accept, 1, "structured:auto-accept");
            control.encode_common(
                &mut encoder,
                &control.rollback,
                state_threads,
                "structured:auto-rollback",
            );
        }
        if let Some(lease) = &status_lease {
            encoder.copy_buffer_to_buffer(
                &control.control,
                0,
                &lease.buffer,
                0,
                Self::autonomous_status_size_bytes(),
            );
        }
        let submission = self.ctx.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Structured", "autonomous_rk4_batch");
        if let (Some(callback), Some(lease)) = (status_callback, status_lease) {
            let request_buffer = std::sync::Arc::clone(&lease.buffer);
            let mapped = std::sync::Arc::clone(&lease.buffer);
            request_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let status = result
                        .map_err(|error| format!("structured status map failed: {error}"))
                        .and_then(|()| {
                            let bytes = mapped.slice(..).get_mapped_range();
                            let raw =
                                bytemuck::try_from_bytes::<StructuredAutonomousControlRaw>(&bytes)
                                    .map_err(|error| {
                                        format!("structured status decode failed: {error}")
                                    })
                                    .copied();
                            drop(bytes);
                            mapped.unmap();
                            raw.map(StructuredGpuSolver::decode_autonomous_status)
                        });
                    callback(status);
                    drop(lease);
                });
        }
        Ok(Some(submission))
    }

    pub fn step_autonomous_batch(
        &mut self,
        steps: usize,
        target_cfl: Option<f32>,
    ) -> Result<Option<wgpu::SubmissionIndex>, String> {
        self.submit_autonomous_batch_inner(steps, target_cfl, None)
    }

    /// Submit an autonomous batch and deliver its exact tail status through a
    /// completion-fenced, nonblocking map callback. The tail copy belongs to
    /// this command buffer, so two batches may be in flight without the first
    /// callback observing the second batch's control state.
    pub fn submit_autonomous_batch<F>(
        &mut self,
        steps: usize,
        target_cfl: Option<f32>,
        on_completed: F,
    ) -> Result<Option<wgpu::SubmissionIndex>, String>
    where
        F: FnOnce(Result<StructuredAutonomousStatus, String>) + Send + 'static,
    {
        self.submit_autonomous_batch_inner(steps, target_cfl, Some(Box::new(on_completed)))
    }

    /// Register a callback for completion of every submission queued before
    /// this call. Callback delivery is driven by [`Self::poll_gpu_completions`].
    pub fn on_submitted_work_done<F>(&self, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.ctx.queue.on_submitted_work_done(callback);
    }

    /// Drive GPU completion callbacks without blocking.
    pub fn poll_gpu_completions(&self) -> Result<(), String> {
        self.ctx
            .device
            .poll(wgpu::PollType::Poll)
            .map(|_| ())
            .map_err(|error| format!("nonblocking structured GPU poll failed: {error}"))
    }

    pub const fn autonomous_status_size_bytes() -> u64 {
        std::mem::size_of::<StructuredAutonomousControlRaw>() as u64
    }

    fn decode_autonomous_status(raw: StructuredAutonomousControlRaw) -> StructuredAutonomousStatus {
        StructuredAutonomousStatus {
            dt: raw.dt,
            dt_old: raw.dt_old,
            time: decode_q64_time(&raw),
            max_base_rate: f32::from_bits(raw.max_base_bits),
            max_rhie_chow_rate: f32::from_bits(raw.max_rc_bits),
            max_velocity: f32::from_bits(raw.max_vel_bits),
            invalid_cells: raw.invalid_cells,
            halted: raw.halted != 0,
            accepted_steps: raw.accepted_batch,
            accepted_total: (u64::from(raw.accepted_total_hi) << 32)
                | u64::from(raw.accepted_total),
        }
    }

    /// Reconcile host metadata after a completion callback. This never touches
    /// cell state: the batch tail has already rolled `state` and both history
    /// levels back to the accepted prefix before capturing `status`.
    pub fn reconcile_autonomous_status(&mut self, status: StructuredAutonomousStatus) {
        self.dt = status.dt as f64;
        self.dt_old = status.dt_old as f64;
        self.time = status.time;
        self.step_count = status.accepted_total;
        self.constants.dt = status.dt;
        self.constants.dt_old = status.dt_old;
        self.constants.time = status.time as f32;
    }

    /// Blocking diagnostic/status read. Autonomous marching itself never calls
    /// this; presentation code should use it only behind a completion fence.
    pub fn autonomous_status(&self) -> Option<StructuredAutonomousStatus> {
        let control = self.autonomous.as_ref()?;
        let words = read_buffer_f32(
            &self.ctx,
            &control.control,
            std::mem::size_of::<StructuredAutonomousControlRaw>() / 4,
        );
        let bytes = bytemuck::cast_slice::<f32, u8>(&words);
        bytemuck::try_from_bytes::<StructuredAutonomousControlRaw>(bytes)
            .ok()
            .copied()
            .map(Self::decode_autonomous_status)
    }

    /// Device-only Q64 accumulation and RK-abscissa rounding probe. The three
    /// emitted f32 values are returned through `dt` (a=0), `dt_old` (a=1/2),
    /// and `max_base_rate` (a=1); `time` is the exact decoded Q64 base.
    #[cfg(feature = "dev-tests")]
    pub fn autonomous_clock_probe(
        &self,
        random_sequence: bool,
    ) -> Option<StructuredAutonomousStatus> {
        let control = self.autonomous.as_ref()?;
        let pipeline = if random_sequence {
            &control.clock_random_probe
        } else {
            &control.clock_fixed_probe
        };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("structured:autonomous-clock-probe"),
            });
        control.encode_common(
            &mut encoder,
            pipeline,
            1,
            "structured:autonomous-clock-probe",
        );
        self.ctx.queue.submit(Some(encoder.finish()));
        self.autonomous_status()
    }

    /// Advance one time step. RK4 runs the matrix-free four-stage program;
    /// Euler/BDF2 run `outer_iters` coupled banded solves. All work stays on the
    /// GPU.
    pub fn step(&mut self) {
        use crate::solver::TimeScheme;
        if self.time_scheme == TimeScheme::RK4 {
            self.step_explicit_rk4();
            return;
        }
        self.autonomous_initialized = false;
        let n = self.n;
        let sstride = self.state_stride;
        // Variable-dt BDF2: `dt_old` lags `dt` (unstructured TimeIntegrationModule).
        // Overwriting `dt_old = dt` every step forced r=1 and broke adaptive-dt BDF2.
        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt_old as f32;
        // BDF2 OpenFOAM-style startup: first step is Euler (no valid n-2 state).
        self.constants.time_scheme = if self.time_scheme == TimeScheme::BDF2 && self.step_count == 0
        {
            TimeScheme::Euler as u32
        } else {
            self.time_scheme as u32
        };
        self.write_kernel_constants();

        // Advance history: old_old <- old, old <- state.
        self.copy_submit("state_old", "state_old_old", n * sstride);
        self.copy_submit("state", "state_old", n * sstride);

        // Prep once.
        let prep = self.prep.clone();
        self.dispatch_ids(&prep);

        let mut solve_stats = crate::solver::banded_schur::StructuredStepStats::default();
        let mut outers_done = 0u32;
        // Pressure state offset for the GUI "P residual" line (schur rank ==
        // state offset for current models; fallback 2).
        let p_slot: Option<usize> = self.schur_layout.as_ref().map(|l| l.p);
        // Shared tolerance/plateau exit (CPU parity via the one shared type).
        let mut outer_exit = crate::solver::banded_schur::StructuredOuterExit::default();
        // Eisenstat–Walker outer forcing (CPU `StructuredModelSolver::step`
        // parity — identical formula, so the two backends run identical solves):
        // middle outers at `clamp(0.1 * prev_err, default, 1e-2)`; outer 1 and
        // the LAST outer at the full default tolerance.
        let default_tol = crate::solver::banded_schur::default_step_tol();
        let ew_hi = 1e-2f64.max(default_tol);
        let mut prev_err: Option<f64> = None;
        for outer in 0..self.outer_iters {
            // state_iter <- state, then flux/gradients/assembly.
            self.copy_submit("state", "state_iter", n * sstride);
            let per = self.per_iter.clone();
            self.dispatch_ids(&per);

            let lin_tol = match prev_err {
                Some(e) if outer > 0 && outer + 1 < self.outer_iters && e.is_finite() => {
                    (0.1 * e).clamp(default_tol, ew_hi)
                }
                _ => default_tol,
            };
            // Banded solve: x = A^{-1} rhs. Keep last outer's *linear* stats;
            // Picard residual is measured from applied state change below.
            let mut solve_failed = false;
            if let Some(st) = self.solver.solve(
                &self.ctx,
                self.buf("grid"),
                self.buf("matrix_values"),
                self.buf("rhs"),
                self.buf("x"),
                lin_tol,
            ) {
                solve_stats.linear_iters = st.linear_iters;
                solve_stats.linear_res = st.linear_res;
                solve_failed = !st.linear_res.is_finite();
            }
            // NON-FINITE system: no usable correction was produced (and `x` was
            // not uploaded). Applying the update would blend in a stale/zero
            // iterate and destroy the state — FREEZE the step instead and
            // surface the failure via `linear_res = inf` (CPU structured parity).
            if solve_failed {
                outers_done += 1;
                break;
            }

            // Update: state <- phi + alpha*(x - phi) under-relaxation.
            let upd = self.update.clone();
            self.dispatch_ids(&upd);
            outers_done += 1;

            // Applied Picard residual (CPU parity): per-FIELD |state − state_iter|
            // relative to that field's own scale. Absolute |x| is NOT a residual —
            // freestream U ~ O(0.01) and T≈1 both look "converged" or "huge" for the
            // wrong reasons. See `structured_outer_residuals`. Host-side banded solve
            // already read the matrix back this outer, so the extra state/state_iter
            // readback is not the cost.
            let st = self.read_f32("state", n * sstride);
            let it = self.read_f32("state_iter", n * sstride);
            let scaled = crate::solver::banded_schur::structured_outer_residuals(
                &st,
                &it,
                sstride,
                &self.unknown_state_groups,
            );

            // GUI "Coupled: U / P": the velocity field's residual (state offs 0,1),
            // not T; pressure via schur p rank / offset 2.
            let (mut du, mut dp) = (0.0f32, 0.0f32);
            for (g, comps) in self.unknown_state_groups.iter().enumerate() {
                if comps.first() == Some(&0) {
                    du = du.max(scaled[g]);
                }
                let p_off = p_slot.unwrap_or(2);
                if comps.first() == Some(&p_off) {
                    dp = dp.max(scaled[g]);
                }
            }
            solve_stats.outer_du = du;
            solve_stats.outer_dp = dp;
            // EW forcing input: the worst per-field scaled residual of this outer.
            prev_err = Some(scaled.iter().fold(0.0f32, |m, &v| m.max(v)) as f64);

            // Early-exit when EVERY unknown is under tol (from outer 2), or
            // under tol / stalled / provably unable to reach tol within the
            // cap (from outer 5) — see `StructuredOuterExit`.
            if self.outer_auto_converge
                && self.outer_tol > 0.0
                && outer_exit.should_break(
                    &scaled,
                    outers_done,
                    self.outer_iters as u32,
                    self.outer_tol,
                )
            {
                break;
            }
        }
        solve_stats.outer_iters = outers_done;
        self.last_stats = solve_stats;
        // Commit dt → dt_old and bump step counter (unstructured finalize_step).
        self.dt_old = self.dt;
        self.step_count = self.step_count.saturating_add(1);
        self.time += self.dt;
    }

    /// Convergence telemetry from the most recent [`step`](Self::step) (GUI
    /// readout; parity with the CPU `StructuredModelSolver::last_stats`).
    pub fn last_stats(&self) -> crate::solver::banded_schur::StructuredStepStats {
        self.last_stats
    }

    /// Read a state component field (length `nx*ny`).
    pub fn state_field(&self, offset: usize) -> Vec<f64> {
        let st = self.read_f32("state", self.n * self.state_stride);
        (0..self.n)
            .map(|p| st[p * self.state_stride + offset] as f64)
            .collect()
    }

    // ---- GUI integration surface --------------------------------------------

    /// The packed `state` GPU buffer (cell-major `state[p*stride + off]`) — the
    /// renderer's viz-color source. Lives on the solver's device; build the solver
    /// via `with_context` on the GUI's device so a same-device copy is legal.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        self.buf("state")
    }

    /// Size of the packed `state` buffer in bytes (`n * stride * 4`).
    pub fn state_size_bytes(&self) -> u64 {
        (self.n * self.state_stride * 4) as u64
    }

    /// GPU-to-GPU copy of the packed `state` into a renderer viz buffer (same
    /// device). Mirrors `GpuUnifiedSolver::copy_state_to_buffer`.
    pub(crate) fn encode_state_copy_to_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
    ) -> Result<(), String> {
        let size = self.state_size_bytes();
        if dst.size() < size {
            return Err(format!(
                "structured visualization destination is {} bytes, needs {size}",
                dst.size()
            ));
        }
        if !dst.usage().contains(wgpu::BufferUsages::COPY_DST) {
            return Err("structured visualization destination lacks COPY_DST usage".to_string());
        }
        encoder.copy_buffer_to_buffer(self.buf("state"), 0, dst, 0, size);
        Ok(())
    }

    pub fn copy_state_to_buffer(&self, dst: &wgpu::Buffer) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("structgpu:viz"),
            });
        self.encode_state_copy_to_buffer(&mut enc, dst)
            .expect("structured visualization destination");
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// The state layout (for `UiPortSet::from_layout` in the GUI caps path).
    pub fn state_layout(&self) -> &crate::solver::model::backend::state_layout::StateLayout {
        &self.layout
    }

    /// The model id this solver runs (GUI model-echo).
    pub fn model_id(&self) -> &'static str {
        self.model_id
    }

    /// Whether this solver owns the GPU-resident accepted-state audit and
    /// rollback controller used by fixed-dt autonomous RK4 batches.
    pub fn supports_autonomous_fixed(&self) -> bool {
        self.autonomous.is_some() && self.autonomous_eos_is_certified()
    }

    /// Whether this model has a derived GPU-resident adaptive controller, as
    /// opposed to only the generic fixed-dt finite-state audit.
    pub fn supports_autonomous_adaptive(&self) -> bool {
        self.autonomous_eos_is_certified()
            && self
                .autonomous
                .as_ref()
                .is_some_and(|control| {
                    control.adaptive_policy != StructuredAdaptivePolicy::Unsupported
                })
    }

    /// The density-based controller currently proves the calorically-perfect
    /// ideal-gas domain (`rho>0`, positive internal energy/p/T/c^2). Linear and
    /// constant EOS families require different energy/domain lemmas and must
    /// use the ordinary accepted-step path until those are implemented. Other
    /// structured models have their own, non-conserved audit.
    fn autonomous_eos_is_certified(&self) -> bool {
        if self.model_id != "compressible_structured" {
            return true;
        }
        structured_conserved_audit_certifies_eos(self.runtime_eos())
    }

    fn runtime_eos(&self) -> EosRuntimeParams {
        EosRuntimeParams {
            gamma: self.constants.eos_gamma,
            gm1: self.constants.eos_gm1,
            r: self.constants.eos_r,
            dp_drho: self.constants.eos_dp_drho,
            p_ref: self.constants.eos_p_ref,
            theta_ref: self.constants.eos_theta_ref,
            rho_ref: self.constants.eos_rho_ref,
            gauge_rho_ref: self.constants.eos_gauge_rho_ref,
            gauge_p_ref: self.constants.eos_gauge_p_ref,
            gauge_e_ref: self.constants.eos_gauge_e_ref,
            gauge_p_bias: self.constants.eos_gauge_p_bias,
            bc_pressure_inlet: self.constants.bc_pressure_inlet,
            p_floor: self.constants.eos_p_floor,
            t_floor: self.constants.eos_t_floor,
            rho_floor: self.constants.eos_rho_floor,
        }
    }

    #[cfg(test)]
    pub(crate) fn runtime_eos_for_test(&self) -> EosRuntimeParams {
        self.runtime_eos()
    }

    #[cfg(test)]
    pub(crate) fn inlet_velocity_for_test(&self) -> f32 {
        self.constants.inlet_velocity
    }

    #[cfg(test)]
    pub(crate) fn inlet_ramp_time_for_test(&self) -> f32 {
        self.constants.inlet_ramp_time
    }

    /// Accumulated simulation time (`sum of dt`).
    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn committed_steps(&self) -> u64 {
        self.step_count
    }

    /// Deterministic device-side invalid-candidate injection used only by the
    /// dev-test rollback gate. `accepted_prefix=1` injects a quiet NaN while
    /// auditing the second candidate step, after one step has been accepted.
    #[cfg(feature = "dev-tests")]
    pub fn set_autonomous_test_nan_after(&mut self, accepted_prefix: Option<u32>) {
        self.autonomous_test_nan_after = accepted_prefix.unwrap_or(u32::MAX);
    }

    /// Inject a finite conserved state with negative internal energy while
    /// auditing the candidate after `accepted_prefix` committed steps. This
    /// proves that acceptance is an EOS-domain check, not merely a NaN scan.
    #[cfg(feature = "dev-tests")]
    pub fn set_autonomous_test_negative_energy_after(&mut self, accepted_prefix: Option<u32>) {
        self.autonomous_test_negative_energy_after = accepted_prefix.unwrap_or(u32::MAX);
    }

    /// Seed the logical accepted-step clock near a word boundary without
    /// executing billions of steps. The next autonomous entry uploads the
    /// split low/high words and exercises the device carry path.
    #[cfg(feature = "dev-tests")]
    pub fn set_autonomous_test_accepted_total(&mut self, accepted_total: u64) {
        self.step_count = accepted_total;
        self.autonomous_initialized = false;
    }

    /// The dense grid (for the GUI's structured cell-polygon adapter).
    pub fn grid(&self) -> StructuredGrid {
        self.grid
    }

    /// Set the implicit time-step size (GUI timestep slider / adaptive CFL).
    /// On step 0 also seeds `dt_old` so BDF2 starts with `r = 1`.
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
        if self.step_count == 0 {
            self.dt_old = dt;
        }
        self.autonomous_initialized = false;
    }

    /// Velocity under-relaxation for the coupled update.
    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        self.constants.alpha_u = alpha_u;
        self.write_kernel_constants();
    }

    /// Pressure under-relaxation for the coupled update.
    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        self.constants.alpha_p = alpha_p;
        self.write_kernel_constants();
    }

    /// Stage-time velocity-inlet soft-start inputs. A zero duration disables
    /// the ramp and applies `velocity` immediately.
    pub fn set_inlet_ramp(&mut self, velocity: f32, duration: f32) {
        self.constants.inlet_velocity = velocity;
        self.constants.inlet_ramp_time = duration.max(0.0);
        self.write_kernel_constants();
        self.autonomous_initialized = false;
    }

    /// Change the time scheme without changing solver-program families.
    /// Euler↔BDF2 is live; transitions to/from RK4 require reconstruction.
    pub fn try_set_time_scheme(&mut self, scheme: crate::solver::TimeScheme) -> Result<(), String> {
        let built_explicit = self.buffers.contains_key("rk_base");
        if (scheme == crate::solver::TimeScheme::RK4) != built_explicit {
            return Err(
                "switching to or from RK4 requires rebuilding the structured solver program"
                    .to_string(),
            );
        }
        self.time_scheme = scheme;
        self.constants.time_scheme = scheme as u32;
        self.write_kernel_constants();
        Ok(())
    }

    pub fn set_time_scheme(&mut self, scheme: crate::solver::TimeScheme) {
        if let Err(error) = self.try_set_time_scheme(scheme) {
            log::warn!("structured time-scheme change ignored: {error}");
        }
    }

    /// Cap on Picard (outer) sweeps per time step.
    pub fn set_outer_iters(&mut self, n: usize) {
        self.outer_iters = n.max(1);
    }

    /// Enable opportunistic Picard early-exit on small applied residuals.
    pub fn set_outer_auto_converge(&mut self, enable: bool) {
        self.outer_auto_converge = enable;
    }

    /// Per-unknown relative applied-change threshold for outer early-exit
    /// (`1e-3` default). Matches CPU structured.
    pub fn set_outer_tolerance(&mut self, tol: f32) {
        self.outer_tol = tol.max(0.0);
    }

    /// Select the coupled-solve preconditioner: block-Jacobi, the model-owned
    /// SIMPLE Schur, or Schur with an AMG pressure solve. Requesting a Schur
    /// variant on a model without a `SchurBlockLayout` (e.g. compressible)
    /// silently keeps block-Jacobi. Returns the EFFECTIVE kind.
    pub fn set_preconditioner(
        &mut self,
        kind: crate::solver::banded_schur::CoupledPrecondKind,
    ) -> crate::solver::banded_schur::CoupledPrecondKind {
        use crate::solver::banded_schur::{BandedPrecond, CoupledPrecondKind as K};
        self.solver.precond = match (kind, &self.schur_layout) {
            (K::Schur, Some(l)) => BandedPrecond::schur(l, false),
            (K::SchurAmg, Some(l)) => BandedPrecond::schur(l, true),
            _ => BandedPrecond::BlockJacobi,
        };
        // Re-arm the adaptive AMG latch on a precond change (CPU parity).
        self.solver
            .amg_active
            .store(false, std::sync::atomic::Ordering::Relaxed);
        crate::solver::banded_schur::kind_of(&self.solver.precond)
    }

    /// Whether the adaptive AMG latch has activated (diagnostics/tests; CPU
    /// `StructuredModelSolver::amg_is_active` parity).
    pub fn amg_is_active(&self) -> bool {
        self.solver
            .amg_active
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether this model declares a Schur preconditioner (drives the GUI to
    /// offer the Schur / Schur+AMG choices only where they are meaningful).
    pub fn supports_schur(&self) -> bool {
        self.schur_layout.is_some()
    }

    /// The implicit time-step size.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Paired velocity `(Ux, Uy)` per cell — the GUI readback shape.
    pub fn get_u(&self, u_offset: usize) -> Vec<(f64, f64)> {
        let st = self.read_f32("state", self.n * self.state_stride);
        (0..self.n)
            .map(|p| {
                let b = p * self.state_stride + u_offset;
                (st[b] as f64, st[b + 1] as f64)
            })
            .collect()
    }

    /// Scalar field (e.g. pressure) per cell — the GUI readback shape.
    pub fn get_scalar(&self, offset: usize) -> Vec<f64> {
        self.state_field(offset)
    }

    /// Read the complete packed state in one transfer. Explicit thermal
    /// stability and positivity checks use this instead of one map per field.
    pub fn packed_state_f32(&self) -> Vec<f32> {
        self.read_f32("state", self.n * self.state_stride)
    }
}

// ===========================================================================
// Matrix-free banded GPU linear algebra: SpMV + block-Jacobi + CG / BiCGStab.
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LaDims {
    s: u32,
    n: u32,    // cells
    ndof: u32, // n * s
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LaScalar {
    a: f32,
    b: f32,
    _p0: f32,
    _p1: f32,
}

/// Owns the hand-written banded LA kernels + work buffers, and orchestrates the
/// solve on the host (dispatching GPU kernels; only scalars cross the bus).
struct BandedGpuLinAlg {
    n: u32,
    nx: u32,
    ny: u32,
    s: u32,
    ndof: u32,
    restart: usize,
    /// Active preconditioner for the HOST coupled solve (`host_solve`):
    /// block-Jacobi (default) or the model-owned SIMPLE Schur. The on-device
    /// GMRES/CG paths always use the block-Jacobi GPU kernels.
    precond: crate::solver::banded_schur::BandedPrecond,
    /// Cross-step cache for the Schur AMG pressure hierarchy (built once from the
    /// grid sparsity; only the Galerkin values re-assemble each host solve). Shared
    /// borrow works via `OnceCell` interior mutability under `host_solve(&self)`.
    amg_cache: crate::solver::banded_schur::StructuredAmgCache,
    /// One-way adaptive-AMG latch (CPU parity, `StructuredModelSolver::
    /// effective_precond` + its step-loop latch): a SchurAmg request runs the
    /// robust heavy-ball Schur until an inner solve stalls (`res > 0.7`) or
    /// burns a full restart cycle (`iters > 60`). From-rest startup makes
    /// `A_pp` transiently indefinite, and the SPD-assuming AMG V-cycle
    /// amplifies on it — mapping SchurAmg straight in blew the startup up.
    amg_active: std::sync::atomic::AtomicBool,
    /// WARM START for the host coupled solve: the previous solve's RAW solution
    /// (the device `x` buffer is overwritten by the update kernel with the
    /// APPLIED under-relaxed value, so the raw solution is kept host-side).
    /// Mirror of the CPU `StructuredModelSolver::prev_x` — identical update
    /// policy (only on a finite solve), so CPU/GPU bit-parity holds. `Mutex`
    /// because `host_solve` runs under `&self`.
    prev_x: std::sync::Mutex<Option<Vec<f32>>>,
    dims_buf: wgpu::Buffer,
    scalar_buf: wgpu::Buffer,

    p_spmv: wgpu::ComputePipeline,
    p_vscale: wgpu::ComputePipeline,
    p_binv: wgpu::ComputePipeline,
    p_papply: wgpu::ComputePipeline,
    p_axpy: wgpu::ComputePipeline,
    p_xpby: wgpu::ComputePipeline,
    p_copy: wgpu::ComputePipeline,
    p_dot: wgpu::ComputePipeline,

    /// On-device work buffers, allocated LAZILY on first use: only the `s == 1`
    /// on-device CG and the env-gated (`CFD2_STRUCTGPU_ONDEVICE_SOLVE`) GMRES
    /// touch them. The shipped coupled path host-solves and never does — eagerly
    /// allocating the 61-vector Krylov basis + work vectors used to pin
    /// 600MB–2.4GB of dead VRAM on fine grids (on the SHARED GUI renderer
    /// device).
    work: std::sync::OnceLock<LaWork>,
}

/// The on-device solve's work set: block-Jacobi inverse (`n*s*s`), work vectors
/// (`ndof` each), the GMRES Krylov basis (`restart+1` vectors) and the
/// dot-product partial buffers. See [`BandedGpuLinAlg::work`].
struct LaWork {
    dinv: wgpu::Buffer,
    r: wgpu::Buffer,
    z: wgpu::Buffer,
    p: wgpu::Buffer,
    ap: wgpu::Buffer,
    // GMRES Arnoldi scratch: `v` holds w = M^{-1} A v_k.
    v: wgpu::Buffer,
    // GMRES Krylov basis (restart+1 vectors) — the robust solver for the
    // indefinite saddle-point system whose block-Jacobi diagonal is singular.
    basis: Vec<wgpu::Buffer>,
    partials: wgpu::Buffer,
    partials_staging: wgpu::Buffer,
    n_partials: u32,
}

impl BandedGpuLinAlg {
    fn new(device: &wgpu::Device, nx: u32, ny: u32, s: u32) -> Self {
        let n = nx * ny;
        let ndof = n * s;
        // Each LA kernel is its own module (distinct `@group(0)` bindings would
        // collide inside one module), all sharing the struct header.
        let make = |body: &str, entry: &str| {
            let src = format!("{LA_HEADER}\n{body}");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let dims = LaDims {
            s,
            n,
            ndof,
            _pad: 0,
        };
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("la_dims"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scalar_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("la_scalar"),
            contents: bytemuck::bytes_of(&LaScalar {
                a: 0.0,
                b: 0.0,
                _p0: 0.0,
                _p1: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // GMRES restart: capped by the DOF count so tiny systems don't
        // over-allocate. 60 matches the CPU banded_block_gmres.
        let restart = (60).min(ndof as usize).max(1);
        Self {
            n,
            nx,
            ny,
            s,
            ndof,
            restart,
            precond: crate::solver::banded_schur::BandedPrecond::BlockJacobi,
            amg_cache: crate::solver::banded_schur::StructuredAmgCache::default(),
            amg_active: std::sync::atomic::AtomicBool::new(false),
            prev_x: std::sync::Mutex::new(None),
            dims_buf,
            scalar_buf,
            p_spmv: make(LA_SPMV, "spmv"),
            p_vscale: make(LA_VSCALE, "vscale"),
            p_binv: make(LA_BLOCK_INVERT, "block_invert"),
            p_papply: make(LA_PRECOND, "precond_apply"),
            p_axpy: make(LA_AXPY, "axpy"),
            p_xpby: make(LA_XPBY, "xpby"),
            p_copy: make(LA_VCOPY, "vcopy"),
            p_dot: make(LA_DOT, "dot_partial"),
            work: std::sync::OnceLock::new(),
        }
    }

    /// The on-device work set, allocated on FIRST use (the `s == 1` CG and the
    /// env-gated on-device GMRES). The shipped coupled path host-solves and
    /// never allocates these.
    fn work(&self, device: &wgpu::Device) -> &LaWork {
        self.work.get_or_init(|| {
            let (n, s, ndof) = (self.n, self.s, self.ndof);
            let n_partials = ndof.div_ceil(WG).max(1);
            let partials = storage_buffer(device, "la_partials", n_partials as usize);
            let partials_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("la_partials_staging"),
                size: (n_partials * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let basis: Vec<wgpu::Buffer> = (0..=self.restart)
                .map(|i| storage_buffer(device, &format!("la_basis_{i}"), ndof as usize))
                .collect();
            LaWork {
                dinv: storage_buffer(device, "la_dinv", (n * s * s) as usize),
                r: storage_buffer(device, "la_r", ndof as usize),
                z: storage_buffer(device, "la_z", ndof as usize),
                p: storage_buffer(device, "la_p", ndof as usize),
                ap: storage_buffer(device, "la_ap", ndof as usize),
                v: storage_buffer(device, "la_v", ndof as usize),
                basis,
                partials,
                partials_staging,
                n_partials,
            }
        })
    }

    fn bg<'a>(
        &self,
        device: &wgpu::Device,
        pipeline: &wgpu::ComputePipeline,
        entries: &[(u32, &'a wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let e: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("la:bg"),
            layout: &layout,
            entries: &e,
        })
    }

    fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        entries: &[(u32, &wgpu::Buffer)],
        n_threads: u32,
    ) {
        let bind = self.bg(device, pipeline, entries);
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("la") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("la:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n_threads.div_ceil(WG).max(1), 1, 1);
        }
        queue.submit(Some(enc.finish()));
    }

    /// Dot product `a·b` (dispatch the partial-reduction kernel, then sum the
    /// per-workgroup partials on the host — only `n_partials` floats cross).
    fn dot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: &LaWork,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
    ) -> f64 {
        self.run(
            device,
            queue,
            &self.p_dot,
            &[(0, &self.dims_buf), (1, a), (2, b), (3, &w.partials)],
            self.ndof,
        );
        let parts = read_buffer_f32_via(
            device,
            queue,
            &w.partials,
            &w.partials_staging,
            w.n_partials as usize,
        );
        parts.iter().map(|&v| v as f64).sum()
    }

    fn set_scalar(&self, queue: &wgpu::Queue, a: f32, b: f32) {
        queue.write_buffer(
            &self.scalar_buf,
            0,
            bytemuck::bytes_of(&LaScalar {
                a,
                b,
                _p0: 0.0,
                _p1: 0.0,
            }),
        );
    }

    /// Solve `A x = rhs` (matrix-free banded, block-Jacobi preconditioned). Picks
    /// CG for the SPD scalar system (`s == 1`) and BiCGStab for the indefinite
    /// coupled U–p system (`s > 1`). `x` is overwritten (initial guess zero).
    /// Returns per-solve convergence telemetry (linear residual + coupled-increment
    /// residuals) for the host coupled path; `None` for the on-device CG/GMRES
    /// paths, which don't surface it to the GUI readout.
    fn solve(
        &self,
        ctx: &GpuContext,
        grid: &wgpu::Buffer,
        mat: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        x: &wgpu::Buffer,
        tol: f64,
    ) -> Option<crate::solver::banded_schur::StructuredStepStats> {
        let device = &ctx.device;
        let queue = &ctx.queue;
        if self.s == 1 {
            // SPD scalar: fully on-device block-Jacobi CG.
            let w = self.work(device);
            self.run(
                device,
                queue,
                &self.p_binv,
                &[(0, &self.dims_buf), (1, mat), (2, &w.dinv)],
                self.n,
            );
            self.cg(device, queue, w, grid, mat, rhs, x);
            None
        } else if std::env::var("CFD2_STRUCTGPU_ONDEVICE_SOLVE").is_ok() {
            // Opt-in fully on-device GMRES (correct but per-dot readback makes it
            // slow for the many-iteration saddle-point solve).
            let w = self.work(device);
            self.run(
                device,
                queue,
                &self.p_binv,
                &[(0, &self.dims_buf), (1, mat), (2, &w.dinv)],
                self.n,
            );
            self.gmres(device, queue, w, grid, mat, rhs, x);
            None
        } else {
            // Coupled indefinite U-p: the assembly runs on the GPU, but the
            // banded block-GMRES inner solve is done host-side in f64 (one matrix
            // + rhs readback, one x upload per solve) — robust and fast, avoiding
            // O(iters^2) GPU dot-product round-trips on a weakly-preconditioned,
            // often-hundreds-of-iterations saddle-point system.
            Some(self.host_solve(ctx, mat, rhs, x, tol))
        }
    }

    /// Host-side preconditioned banded GMRES (f64) — the GPU analog path reads the
    /// assembled banded operator + rhs back, solves via the shared
    /// `banded_schur::banded_gmres` (block-Jacobi or the model-owned Schur, per
    /// `self.precond`), and uploads the correction. Bit-identical to the CPU
    /// `StructuredModelSolver` coupled solve (same shared routine).
    fn host_solve(
        &self,
        ctx: &GpuContext,
        mat: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        x: &wgpu::Buffer,
        tol: f64,
    ) -> crate::solver::banded_schur::StructuredStepStats {
        let (nx, ny, s) = (self.nx as usize, self.ny as usize, self.s as usize);
        let a = read_buffer_f32(ctx, mat, nx * ny * BAND_STRIDE * s * s);
        let b = read_buffer_f32(ctx, rhs, nx * ny * s);
        // Parallelize the host-side banded solve over the available cores (the
        // GUI is a wgpu app but the coupled solve is host-side; parallel only
        // takes effect under the `cpu` feature — the GUI build has it).
        let threads = std::env::var("CFD2_CPU_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        // Adaptive AMG latch (CPU parity, `StructuredModelSolver::effective_precond`):
        // a SchurAmg request runs the robust heavy-ball Schur until the one-way
        // latch fires — from-rest startup makes A_pp transiently indefinite and
        // the SPD-assuming AMG V-cycle amplifies on it.
        use crate::solver::banded_schur::BandedPrecond;
        use std::sync::atomic::Ordering;
        let amg_requested = matches!(
            &self.precond,
            BandedPrecond::Schur {
                pressure_amg: true,
                ..
            }
        );
        let effective = match &self.precond {
            BandedPrecond::Schur {
                u_idx,
                p,
                omega,
                sweeps_cap,
                pressure_amg: true,
            } if !self.amg_active.load(Ordering::Relaxed) => {
                BandedPrecond::Schur {
                    u_idx: u_idx.clone(),
                    p: *p,
                    omega: *omega,
                    sweeps_cap: *sweeps_cap,
                    pressure_amg: false, // heavy-ball until the latch activates AMG
                }
            }
            other => other.clone(),
        };
        // Warm start from the previous solve's raw solution (CPU parity: the
        // CPU Picard loop keeps the identical `prev_x` cache).
        let mut prev_x = self.prev_x.lock().unwrap();
        let (xh, res, iters) = crate::solver::banded_schur::banded_gmres_opts(
            &a,
            nx,
            ny,
            s,
            &b,
            &effective,
            &crate::solver::banded_schur::BandedSolveOpts {
                // coupled_restart (not the on-device `self.restart`): CPU
                // Picard-loop parity — see `banded_schur::coupled_restart`.
                restart: crate::solver::banded_schur::coupled_restart(&effective)
                    .min(self.ndof as usize)
                    .max(1),
                max_outer: 200,
                tol,
                threads,
                amg_cache: Some(&self.amg_cache),
                x0: prev_x.as_deref(),
            },
        );
        // Flip AMG once heavy-ball either fails to reduce the residual or
        // converges only after burning a full GMRES restart cycle (iters > 60)
        // — the h-dependent regime AMG cures. Mirrors cpu/structured.rs. A
        // non-finite residual (poisoned system) must NOT latch AMG.
        if amg_requested
            && !self.amg_active.load(Ordering::Relaxed)
            && res.is_finite()
            && (res > 0.7 || iters > 60)
        {
            self.amg_active.store(true, Ordering::Relaxed);
        }
        // NON-FINITE system: the solve bailed without a usable correction —
        // do NOT upload the zero iterate (the caller skips the update and
        // freezes the step; see `StructuredGpuSolver::step`, CPU parity), and
        // keep the last good warm-start cache.
        if res.is_finite() {
            // Picard outer residual is measured by `StructuredGpuSolver::step` from
            // applied |state − state_iter| after under-relaxation — not from |xh|.
            // Absolute |x| is freestream-scale (or T≈1) and is not a residual.
            ctx.queue.write_buffer(x, 0, bytemuck::cast_slice(&xh));
            *prev_x = Some(xh);
        }
        crate::solver::banded_schur::StructuredStepStats {
            outer_iters: 0, // filled in by the caller (`step`), which owns the count
            linear_iters: iters,
            linear_res: res as f32,
            outer_du: 0.0,
            outer_dp: 0.0,
        }
    }

    /// Block-Jacobi preconditioned CG (SPD scalar system). Assumes `w.dinv` is set.
    #[allow(clippy::too_many_arguments)]
    fn cg(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: &LaWork,
        grid: &wgpu::Buffer,
        mat: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        x: &wgpu::Buffer,
    ) {
        // Initial guess x <- 0, so the initial residual r <- rhs. Then
        // z <- M^{-1} r and the search direction p <- z.
        self.zero(queue, x);
        self.run(
            device,
            queue,
            &self.p_copy,
            &[(0, &self.dims_buf), (1, rhs), (2, &w.r)],
            self.ndof,
        ); // r <- rhs
        self.run(
            device,
            queue,
            &self.p_papply,
            &[(0, &self.dims_buf), (1, &w.dinv), (2, &w.r), (3, &w.z)],
            self.n,
        ); // z <- Minv r
        self.run(
            device,
            queue,
            &self.p_copy,
            &[(0, &self.dims_buf), (1, &w.z), (2, &w.p)],
            self.ndof,
        ); // p <- z

        let bnorm = self.dot(device, queue, w, rhs, rhs).sqrt().max(1e-30);
        let mut rz = self.dot(device, queue, w, &w.r, &w.z);
        let tol = 1e-10_f64;
        let maxit = (self.ndof as usize * 4).max(200);

        for _ in 0..maxit {
            // Ap <- A p
            self.run(
                device,
                queue,
                &self.p_spmv,
                &[
                    (0, grid),
                    (1, &self.dims_buf),
                    (2, mat),
                    (3, &w.p),
                    (4, &w.ap),
                ],
                self.n,
            );
            let pap = self.dot(device, queue, w, &w.p, &w.ap);
            if pap.abs() < 1e-300 {
                break;
            }
            let alpha = rz / pap;
            // x <- x + alpha p ; r <- r - alpha Ap
            self.set_scalar(queue, alpha as f32, 0.0);
            self.run(
                device,
                queue,
                &self.p_axpy,
                &[
                    (0, &self.dims_buf),
                    (1, &self.scalar_buf),
                    (2, &w.p),
                    (3, x),
                ],
                self.ndof,
            );
            self.set_scalar(queue, -(alpha as f32), 0.0);
            self.run(
                device,
                queue,
                &self.p_axpy,
                &[
                    (0, &self.dims_buf),
                    (1, &self.scalar_buf),
                    (2, &w.ap),
                    (3, &w.r),
                ],
                self.ndof,
            );

            let rr = self.dot(device, queue, w, &w.r, &w.r);
            if rr.sqrt() / bnorm <= tol {
                break;
            }
            // z <- Minv r ; rz_new ; beta ; p <- z + beta p
            self.run(
                device,
                queue,
                &self.p_papply,
                &[(0, &self.dims_buf), (1, &w.dinv), (2, &w.r), (3, &w.z)],
                self.n,
            );
            let rz_new = self.dot(device, queue, w, &w.r, &w.z);
            let beta = rz_new / rz;
            self.set_scalar(queue, 0.0, beta as f32);
            self.run(
                device,
                queue,
                &self.p_xpby,
                &[
                    (0, &self.dims_buf),
                    (1, &self.scalar_buf),
                    (2, &w.z),
                    (3, &w.p),
                ],
                self.ndof,
            );
            rz = rz_new;
        }
    }

    /// Restarted, block-Jacobi (left)-preconditioned GMRES — the GPU analog of the
    /// CPU `banded_block_gmres`. Arnoldi/Givens run on the host over the small
    /// Hessenberg; all vector work (SpMV, precond, dot, axpy, scale) is on the
    /// device, with only scalars crossing the bus. Assumes `dinv` is set.
    #[allow(clippy::too_many_arguments)]
    fn gmres(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: &LaWork,
        grid: &wgpu::Buffer,
        mat: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        x: &wgpu::Buffer,
    ) {
        let axpy = |src: &wgpu::Buffer, dst: &wgpu::Buffer, a: f64| {
            self.set_scalar(queue, a as f32, 0.0);
            self.run(
                device,
                queue,
                &self.p_axpy,
                &[
                    (0, &self.dims_buf),
                    (1, &self.scalar_buf),
                    (2, src),
                    (3, dst),
                ],
                self.ndof,
            );
        };
        let scale = |src: &wgpu::Buffer, dst: &wgpu::Buffer, a: f64| {
            self.set_scalar(queue, a as f32, 0.0);
            self.run(
                device,
                queue,
                &self.p_vscale,
                &[
                    (0, &self.dims_buf),
                    (1, &self.scalar_buf),
                    (2, src),
                    (3, dst),
                ],
                self.ndof,
            );
        };
        let copy = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            self.run(
                device,
                queue,
                &self.p_copy,
                &[(0, &self.dims_buf), (1, src), (2, dst)],
                self.ndof,
            );
        };
        let precond = |rin: &wgpu::Buffer, zout: &wgpu::Buffer| {
            self.run(
                device,
                queue,
                &self.p_papply,
                &[(0, &self.dims_buf), (1, &w.dinv), (2, rin), (3, zout)],
                self.n,
            );
        };
        let spmv = |xin: &wgpu::Buffer, yout: &wgpu::Buffer| {
            self.run(
                device,
                queue,
                &self.p_spmv,
                &[
                    (0, grid),
                    (1, &self.dims_buf),
                    (2, mat),
                    (3, xin),
                    (4, yout),
                ],
                self.n,
            );
        };
        let dot = |a: &wgpu::Buffer, b: &wgpu::Buffer| self.dot(device, queue, w, a, b);
        let norm = |a: &wgpu::Buffer| dot(a, a).sqrt();

        let m = self.restart;
        let bnorm = norm(rhs).max(1e-30);
        let tol = 1e-8_f64;
        let max_outer = 50usize;
        let debug = std::env::var("CFD2_STRUCTGPU_DEBUG").is_ok();
        self.zero(queue, x);

        let mut total_k = 0usize;
        let mut final_rel = 1.0_f64;
        for _outer in 0..max_outer {
            // r = b - A x ; z = M^{-1} r ; beta = ||z||.
            spmv(x, &w.ap);
            copy(rhs, &w.r);
            axpy(&w.ap, &w.r, -1.0);
            precond(&w.r, &w.z);
            let beta = norm(&w.z);
            if beta / bnorm <= tol {
                final_rel = beta / bnorm;
                break;
            }
            scale(&w.z, &w.basis[0], 1.0 / beta);

            let mut h = vec![vec![0.0f64; m]; m + 1];
            let mut g = vec![0.0f64; m + 1];
            g[0] = beta;
            let mut cs = vec![0.0f64; m];
            let mut sn = vec![0.0f64; m];
            let mut k_used = 0usize;

            for k in 0..m {
                // w = M^{-1} A v_k  (use `v` as w).
                spmv(&w.basis[k], &w.ap);
                precond(&w.ap, &w.v);
                // Modified Gram-Schmidt against the existing basis.
                for i in 0..=k {
                    h[i][k] = dot(&w.v, &w.basis[i]);
                    axpy(&w.basis[i], &w.v, -h[i][k]);
                }
                h[k + 1][k] = norm(&w.v);
                if h[k + 1][k] > 1e-14 {
                    scale(&w.v, &w.basis[k + 1], 1.0 / h[k + 1][k]);
                } else {
                    self.zero(queue, &w.basis[k + 1]);
                }
                // Apply previous Givens rotations, then a new one.
                for i in 0..k {
                    let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
                    h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
                    h[i][k] = temp;
                }
                let denom = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
                if denom < 1e-300 {
                    k_used = k;
                    break;
                }
                cs[k] = h[k][k] / denom;
                sn[k] = h[k + 1][k] / denom;
                h[k][k] = cs[k] * h[k][k] + sn[k] * h[k + 1][k];
                h[k + 1][k] = 0.0;
                g[k + 1] = -sn[k] * g[k];
                g[k] = cs[k] * g[k];
                k_used = k + 1;
                if g[k + 1].abs() / bnorm <= tol {
                    break;
                }
            }

            // Back-substitute y, then x += sum y_i v_i.
            let kk = k_used;
            let mut y = vec![0.0f64; kk];
            for i in (0..kk).rev() {
                let mut sum = g[i];
                for j in (i + 1)..kk {
                    sum -= h[i][j] * y[j];
                }
                y[i] = if h[i][i].abs() > 1e-300 {
                    sum / h[i][i]
                } else {
                    0.0
                };
            }
            for i in 0..kk {
                axpy(&w.basis[i], x, y[i]);
            }
            total_k += kk;

            // True-residual convergence check.
            spmv(x, &w.ap);
            copy(rhs, &w.r);
            axpy(&w.ap, &w.r, -1.0);
            final_rel = norm(&w.r) / bnorm;
            if final_rel <= tol {
                break;
            }
        }
        if debug {
            eprintln!("[structgpu] gmres: {total_k} inner iters, rel_res = {final_rel:e}");
        }
    }

    fn zero(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer) {
        queue.write_buffer(buf, 0, &vec![0u8; (self.ndof * 4) as usize]);
    }
}

fn read_buffer_f32(ctx: &GpuContext, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_staging"),
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    read_buffer_f32_via(&ctx.device, &ctx.queue, buf, &staging, len)
}

fn read_buffer_f32_via(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    enc.copy_buffer_to_buffer(buf, 0, staging, 0, (len * 4) as u64);
    let idx = queue.submit(Some(enc.finish()));
    let slice = staging.slice(..(len * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
    rx.recv().expect("recv").expect("map");
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

const LA_HEADER: &str = r#"
struct SGrid { nx: u32, ny: u32, dx: f32, dy: f32 }
struct Dims { s: u32, n: u32, ndof: u32, pad: u32 }
struct Scalar { a: f32, b: f32, p0: f32, p1: f32 }
"#;

// ---- banded block SpMV: y = A x -------------------------------------------
const LA_SPMV: &str = r#"
@group(0) @binding(0) var<uniform> grid: SGrid;
@group(0) @binding(1) var<uniform> dims: Dims;
@group(0) @binding(2) var<storage, read> mat: array<f32>;
@group(0) @binding(3) var<storage, read> xin: array<f32>;
@group(0) @binding(4) var<storage, read_write> yout: array<f32>;

@compute @workgroup_size(64)
fn spmv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= dims.n) { return; }
    let s = dims.s;
    let nx = grid.nx;
    let ny = grid.ny;
    let gi = p % nx;
    let gj = p / nx;
    let blk = 5u * s * s;
    for (var r: u32 = 0u; r < s; r = r + 1u) {
        var acc: f32 = 0.0;
        let rowbase = p * blk + 5u * s * r;
        // South band 0 (q = p - nx)
        if (gj > 0u) {
            let q = p - nx;
            for (var c: u32 = 0u; c < s; c = c + 1u) { acc = acc + mat[rowbase + 0u * s + c] * xin[q * s + c]; }
        }
        // West band 1 (q = p - 1)
        if (gi > 0u) {
            let q = p - 1u;
            for (var c: u32 = 0u; c < s; c = c + 1u) { acc = acc + mat[rowbase + 1u * s + c] * xin[q * s + c]; }
        }
        // Diagonal band 2 (q = p)
        for (var c: u32 = 0u; c < s; c = c + 1u) { acc = acc + mat[rowbase + 2u * s + c] * xin[p * s + c]; }
        // East band 3 (q = p + 1)
        if (gi + 1u < nx) {
            let q = p + 1u;
            for (var c: u32 = 0u; c < s; c = c + 1u) { acc = acc + mat[rowbase + 3u * s + c] * xin[q * s + c]; }
        }
        // North band 4 (q = p + nx)
        if (gj + 1u < ny) {
            let q = p + nx;
            for (var c: u32 = 0u; c < s; c = c + 1u) { acc = acc + mat[rowbase + 4u * s + c] * xin[q * s + c]; }
        }
        yout[p * s + r] = acc;
    }
}
"#;

// ---- vscale: dst = a * src -------------------------------------------------
const LA_VSCALE: &str = r#"
@group(0) @binding(0) var<uniform> vsdims: Dims;
@group(0) @binding(1) var<uniform> vssc: Scalar;
@group(0) @binding(2) var<storage, read> vssrc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vsdst: array<f32>;

@compute @workgroup_size(64)
fn vscale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vsdims.ndof) { return; }
    vsdst[i] = vssc.a * vssrc[i];
}
"#;

// ---- per-cell block-Jacobi inverse: dinv = (diag block)^{-1} ---------------
const LA_BLOCK_INVERT: &str = r#"
@group(0) @binding(0) var<uniform> bdims: Dims;
@group(0) @binding(1) var<storage, read> bmat: array<f32>;
@group(0) @binding(2) var<storage, read_write> bdinv: array<f32>;

@compute @workgroup_size(64)
fn block_invert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= bdims.n) { return; }
    let s = bdims.s;
    let blk = 5u * s * s;
    // Augmented [M | I], sized MAX_S x 2*MAX_S, row-major with width 2*s.
    var aug: array<f32, 128>; // MAX_S * 2 * MAX_S = 8*16
    let w = 2u * s;
    for (var r: u32 = 0u; r < s; r = r + 1u) {
        for (var c: u32 = 0u; c < s; c = c + 1u) {
            aug[r * w + c] = bmat[p * blk + 5u * s * r + 2u * s + c];
        }
        for (var c: u32 = 0u; c < s; c = c + 1u) {
            aug[r * w + s + c] = select(0.0, 1.0, r == c);
        }
    }
    // Gauss-Jordan with partial pivoting.
    var singular = false;
    for (var col: u32 = 0u; col < s; col = col + 1u) {
        var piv = col;
        var best = abs(aug[col * w + col]);
        for (var rr: u32 = col + 1u; rr < s; rr = rr + 1u) {
            let v = abs(aug[rr * w + col]);
            if (v > best) { best = v; piv = rr; }
        }
        if (best < 1e-30) { singular = true; break; }
        if (piv != col) {
            for (var c: u32 = 0u; c < w; c = c + 1u) {
                let t = aug[col * w + c];
                aug[col * w + c] = aug[piv * w + c];
                aug[piv * w + c] = t;
            }
        }
        let d = aug[col * w + col];
        for (var c: u32 = 0u; c < w; c = c + 1u) { aug[col * w + c] = aug[col * w + c] / d; }
        for (var rr: u32 = 0u; rr < s; rr = rr + 1u) {
            if (rr == col) { continue; }
            let f = aug[rr * w + col];
            if (f != 0.0) {
                for (var c: u32 = 0u; c < w; c = c + 1u) { aug[rr * w + c] = aug[rr * w + c] - f * aug[col * w + c]; }
            }
        }
    }
    let obase = p * s * s;
    if (singular) {
        // Diagonal fallback: inverse of the diagonal entries only.
        for (var r: u32 = 0u; r < s; r = r + 1u) {
            for (var c: u32 = 0u; c < s; c = c + 1u) {
                var val = 0.0;
                if (r == c) {
                    let dd = bmat[p * blk + 5u * s * r + 2u * s + c];
                    if (abs(dd) > 1e-30) { val = 1.0 / dd; }
                }
                bdinv[obase + r * s + c] = val;
            }
        }
    } else {
        for (var r: u32 = 0u; r < s; r = r + 1u) {
            for (var c: u32 = 0u; c < s; c = c + 1u) {
                bdinv[obase + r * s + c] = aug[r * w + s + c];
            }
        }
    }
}
"#;

// ---- block-Jacobi apply: z = Minv r ---------------------------------------
const LA_PRECOND: &str = r#"
@group(0) @binding(0) var<uniform> pdims: Dims;
@group(0) @binding(1) var<storage, read> pdinv: array<f32>;
@group(0) @binding(2) var<storage, read> prin: array<f32>;
@group(0) @binding(3) var<storage, read_write> pzout: array<f32>;

@compute @workgroup_size(64)
fn precond_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= pdims.n) { return; }
    let s = pdims.s;
    let obase = p * s * s;
    for (var i: u32 = 0u; i < s; i = i + 1u) {
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < s; k = k + 1u) {
            acc = acc + pdinv[obase + i * s + k] * prin[p * s + k];
        }
        pzout[p * s + i] = acc;
    }
}
"#;

// ---- axpy: y = y + a*x -----------------------------------------------------
const LA_AXPY: &str = r#"
@group(0) @binding(0) var<uniform> adims: Dims;
@group(0) @binding(1) var<uniform> asc: Scalar;
@group(0) @binding(2) var<storage, read> axin: array<f32>;
@group(0) @binding(3) var<storage, read_write> ayio: array<f32>;

@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= adims.ndof) { return; }
    ayio[i] = ayio[i] + asc.a * axin[i];
}
"#;

// ---- xpby: y = x + b*y -----------------------------------------------------
const LA_XPBY: &str = r#"
@group(0) @binding(0) var<uniform> xdims: Dims;
@group(0) @binding(1) var<uniform> xsc: Scalar;
@group(0) @binding(2) var<storage, read> xxin: array<f32>;
@group(0) @binding(3) var<storage, read_write> xyio: array<f32>;

@compute @workgroup_size(64)
fn xpby(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= xdims.ndof) { return; }
    xyio[i] = xxin[i] + xsc.b * xyio[i];
}
"#;

// ---- vcopy: dst = src ------------------------------------------------------
const LA_VCOPY: &str = r#"
@group(0) @binding(0) var<uniform> cdims: Dims;
@group(0) @binding(1) var<storage, read> csrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> cdst: array<f32>;

@compute @workgroup_size(64)
fn vcopy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cdims.ndof) { return; }
    cdst[i] = csrc[i];
}
"#;

// ---- dot_partial: per-workgroup partial sums of a·b -----------------------
const LA_DOT: &str = r#"
@group(0) @binding(0) var<uniform> ddims: Dims;
@group(0) @binding(1) var<storage, read> da: array<f32>;
@group(0) @binding(2) var<storage, read> db: array<f32>;
@group(0) @binding(3) var<storage, read_write> dpart: array<f32>;

var<workgroup> sdata: array<f32, 64>;

@compute @workgroup_size(64)
fn dot_partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    var v: f32 = 0.0;
    if (gid.x < ddims.ndof) { v = da[gid.x] * db[gid.x]; }
    sdata[lid.x] = v;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { dpart[wid.x] = sdata[0]; }
}
"#;
