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
use crate::solver::model::kernel::ModelKernelArtifact;
use crate::solver::model::module::ModelModule;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use crate::solver::{PreconditionerType, TimeScheme};
use cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl;
use cfd2_ir::kernel::BindingAccess;

const WG: u32 = 64;

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
        assert!(nx > 0 && ny > 0, "grid must have at least one cell per axis");
        assert!(length > 0.0 && height > 0.0, "grid extents must be positive");
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
}

/// Pack the `constants` uniform for a kernel whose `Constants` struct is the 12
/// base fields followed by `eos_fields` (in declared order), padded to the WGSL
/// 16-byte uniform alignment.
fn pack_kernel_constants(c: &GpuConstants, eos_fields: &[String]) -> Vec<u8> {
    // First 48 bytes of GpuConstants are the 12 canonical base fields.
    let mut bytes = bytemuck::bytes_of(c)[0..48].to_vec();
    for f in eos_fields {
        let v: f32 = match f.as_str() {
            "eos_gamma" => c.eos_gamma,
            "eos_gm1" => c.eos_gm1,
            "eos_r" => c.eos_r,
            "eos_dp_drho" => c.eos_dp_drho,
            "eos_p_offset" => c.eos_p_offset,
            "eos_theta_ref" => c.eos_theta_ref,
            "buoyant_beta_g" => c.buoyant_beta_g,
            "buoyant_t0" => c.buoyant_t0,
            "buoyant_k_over_cp" => c.buoyant_k_over_cp,
            _ => 0.0,
        };
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    while bytes.len() % 16 != 0 {
        bytes.push(0);
    }
    bytes
}

impl CompiledKernel {
    fn write_constants(&self, queue: &wgpu::Queue, c: &GpuConstants) {
        queue.write_buffer(&self.constants_buf, 0, &pack_kernel_constants(c, &self.eos_fields));
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
        let ordered: Vec<&wgpu::BindGroupLayout> = (0..=max_g)
            .map(|g| {
                layouts
                    .iter()
                    .find(|(lg, _)| *lg == g)
                    .map(|(_, l)| l)
                    .unwrap_or(&empty)
            })
            .collect();
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(id),
            bind_group_layouts: &ordered,
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(id),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // `Constants` = 12 base fields (48 B) + eos_fields, padded to 16 B.
        let cbytes = 48 + eos_fields.len() * 4;
        let csize = ((cbytes + 15) / 16 * 16).max(16) as u64;
        let constants_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(id),
            size: csize,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bindings,
            layouts,
            constants_buf,
            eos_fields,
        }
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
        let mut bind_groups = Vec::new();
        for (g, layout) in &self.layouts {
            let entries: Vec<wgpu::BindGroupEntry> = self
                .bindings
                .iter()
                .filter(|b| b.group == *g)
                .map(|b| wgpu::BindGroupEntry {
                    binding: b.binding,
                    // `constants` binds THIS kernel's per-layout uniform (correct
                    // eos field offsets); everything else resolves by name.
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
                    label: Some("structured:bg"),
                    layout,
                    entries: &entries,
                }),
            ));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("structured:pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (g, bg) in &bind_groups {
            pass.set_bind_group(*g, bg, &[]);
        }
        pass.dispatch_workgroups(n_threads.div_ceil(WG).max(1), 1, 1);
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
                // The kernel's `Constants` tail (EOS params beyond the 12 base
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

/// The `struct Constants` fields BEYOND the 12 canonical base fields, in order,
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
    // Drop the 12 base fields; the remainder is the EOS/buoyant tail.
    if fields.len() > 12 {
        fields.split_off(12)
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

    outer_iters: usize,
    dt: f64,
    /// Model id (for the GUI's model-echo / caps) and accumulated sim time.
    model_id: &'static str,
    time: f64,
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
    /// and time-integration scheme. Both are honoured at RUNTIME by the codegen
    /// structured kernels (`constants.scheme` selects the deferred-correction
    /// reconstruction; `constants.time_scheme==1` is BDF2) — no kernel change.
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
        let recipe = SolverRecipe::from_model(
            model,
            scheme,
            time_scheme,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
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
            buffers.insert(name.to_string(), storage_buffer(dev, name, n * state_stride));
        }
        buffers.insert(
            "matrix_values".to_string(),
            storage_buffer(dev, "matrix_values", n * BAND_STRIDE * s * s),
        );
        buffers.insert("rhs".to_string(), storage_buffer(dev, "rhs", n * s));
        buffers.insert("x".to_string(), storage_buffer(dev, "x", n * s));
        buffers.insert("y".to_string(), storage_buffer(dev, "y", n * s));
        buffers.insert(
            "fluxes".to_string(),
            storage_buffer(dev, "fluxes", n * 4 * flux_stride),
        );
        buffers.insert(
            "grad_state".to_string(),
            storage_buffer(dev, "grad_state", n * state_stride * 2),
        );
        buffers.insert("bc_kind".to_string(), storage_buffer(dev, "bc_kind", n * 4 * s));
        buffers.insert("bc_value".to_string(), storage_buffer(dev, "bc_value", n * 4 * s));
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
        constants.stride_x = grid.nx as u32;
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

        // Pack each kernel's `constants` uniform to its own `Constants` layout.
        for k in kernels.values() {
            k.write_constants(&ctx.queue, &constants);
        }

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
            outer_iters: outer_iters.max(1),
            dt,
            model_id: model.id,
            time: 0.0,
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
    }

    /// (Re)pack every kernel's `constants` uniform from the current `GpuConstants`
    /// — each to its own `Constants` struct layout (base fields + its EOS params).
    fn write_kernel_constants(&self) {
        for k in self.kernels.values() {
            k.write_constants(&self.ctx.queue, &self.constants);
        }
    }

    /// Set an EOS runtime parameter (e.g. gamma via `eos_gm1`) on the constants.
    pub fn set_eos(&mut self, gamma: f32, gas_constant: f32) {
        self.constants.eos_gamma = gamma;
        self.constants.eos_gm1 = gamma - 1.0;
        self.constants.eos_r = gas_constant;
        self.write_kernel_constants();
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
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
        enc.copy_buffer_to_buffer(self.buf(src), 0, self.buf(dst), 0, (len * 4) as u64);
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Dispatch each kernel in its own submission. Separate command buffers are
    /// strictly ordered on the queue with full memory visibility between them, so
    /// the chained schedule (flux -> gradients -> assembly) sees each prior
    /// kernel's storage writes.
    fn dispatch_ids(&self, ids: &[String]) {
        let resolve = |name: &str| self.buf(name);
        for id in ids {
            let mut enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some(id) },
            );
            let k = &self.kernels[id];
            k.dispatch(&self.ctx.device, &mut enc, self.n as u32, &resolve);
            self.ctx.queue.submit(Some(enc.finish()));
        }
    }

    /// Assemble the banded operator once (advancing history, running prep +
    /// per-iter kernels) WITHOUT solving — for operator-parity tests.
    pub fn assemble_only(&mut self) {
        let n = self.n;
        let sstride = self.state_stride;
        self.copy_submit("state", "state_old", n * sstride);
        self.copy_submit("state", "state_iter", n * sstride);
        let ids: Vec<String> = self.prep.iter().chain(self.per_iter.iter()).cloned().collect();
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

    /// One implicit (backward-Euler) time step: advance history, then
    /// `outer_iters` sweeps of prep/flux/gradients/assembly → banded solve →
    /// update. All on the GPU.
    pub fn step(&mut self) {
        let n = self.n;
        let sstride = self.state_stride;
        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt as f32;
        self.write_kernel_constants();

        // Advance history: old_old <- old, old <- state.
        self.copy_submit("state_old", "state_old_old", n * sstride);
        self.copy_submit("state", "state_old", n * sstride);

        // Prep once.
        let prep = self.prep.clone();
        self.dispatch_ids(&prep);

        for _outer in 0..self.outer_iters {
            // state_iter <- state, then flux/gradients/assembly.
            self.copy_submit("state", "state_iter", n * sstride);
            let per = self.per_iter.clone();
            self.dispatch_ids(&per);

            // Banded solve: x = A^{-1} rhs (fully on the GPU).
            self.solver.solve(
                &self.ctx,
                self.buf("grid"),
                self.buf("matrix_values"),
                self.buf("rhs"),
                self.buf("x"),
            );

            // Update: state <- f(x).
            let upd = self.update.clone();
            self.dispatch_ids(&upd);
        }
        self.time += self.dt;
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
    pub fn copy_state_to_buffer(&self, dst: &wgpu::Buffer) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("structgpu:viz") });
        enc.copy_buffer_to_buffer(self.buf("state"), 0, dst, 0, self.state_size_bytes());
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

    /// Accumulated simulation time (`sum of dt`).
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The dense grid (for the GUI's structured cell-polygon adapter).
    pub fn grid(&self) -> StructuredGrid {
        self.grid
    }

    /// Set the implicit time-step size (GUI timestep slider).
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
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
        crate::solver::banded_schur::kind_of(&self.solver.precond)
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

    // Work vectors (ndof each) + block-Jacobi inverse (n*s*s) + dot partials.
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
        let n_partials = ndof.div_ceil(WG).max(1);
        let partials = storage_buffer(device, "la_partials", n_partials as usize);
        let partials_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("la_partials_staging"),
            size: (n_partials * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // GMRES restart: capped by the DOF count so tiny systems don't
        // over-allocate. 60 matches the CPU banded_block_gmres.
        let restart = (60).min(ndof as usize).max(1);
        let basis: Vec<wgpu::Buffer> = (0..=restart)
            .map(|i| storage_buffer(device, &format!("la_basis_{i}"), ndof as usize))
            .collect();
        Self {
            n,
            nx,
            ny,
            s,
            ndof,
            restart,
            precond: crate::solver::banded_schur::BandedPrecond::BlockJacobi,
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
    fn dot(&self, device: &wgpu::Device, queue: &wgpu::Queue, a: &wgpu::Buffer, b: &wgpu::Buffer) -> f64 {
        self.run(
            device,
            queue,
            &self.p_dot,
            &[
                (0, &self.dims_buf),
                (1, a),
                (2, b),
                (3, &self.partials),
            ],
            self.ndof,
        );
        let parts = read_buffer_f32_via(device, queue, &self.partials, &self.partials_staging, self.n_partials as usize);
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
    fn solve(
        &self,
        ctx: &GpuContext,
        grid: &wgpu::Buffer,
        mat: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        x: &wgpu::Buffer,
    ) {
        let device = &ctx.device;
        let queue = &ctx.queue;
        if self.s == 1 {
            // SPD scalar: fully on-device block-Jacobi CG.
            self.run(
                device,
                queue,
                &self.p_binv,
                &[(0, &self.dims_buf), (1, mat), (2, &self.dinv)],
                self.n,
            );
            self.cg(device, queue, grid, mat, rhs, x);
        } else if std::env::var("CFD2_STRUCTGPU_ONDEVICE_SOLVE").is_ok() {
            // Opt-in fully on-device GMRES (correct but per-dot readback makes it
            // slow for the many-iteration saddle-point solve).
            self.run(
                device,
                queue,
                &self.p_binv,
                &[(0, &self.dims_buf), (1, mat), (2, &self.dinv)],
                self.n,
            );
            self.gmres(device, queue, grid, mat, rhs, x);
        } else {
            // Coupled indefinite U-p: the assembly runs on the GPU, but the
            // banded block-GMRES inner solve is done host-side in f64 (one matrix
            // + rhs readback, one x upload per solve) — robust and fast, avoiding
            // O(iters^2) GPU dot-product round-trips on a weakly-preconditioned,
            // often-hundreds-of-iterations saddle-point system.
            self.host_solve(ctx, mat, rhs, x);
        }
    }

    /// Host-side preconditioned banded GMRES (f64) — the GPU analog path reads the
    /// assembled banded operator + rhs back, solves via the shared
    /// `banded_schur::banded_gmres` (block-Jacobi or the model-owned Schur, per
    /// `self.precond`), and uploads the correction. Bit-identical to the CPU
    /// `StructuredModelSolver` coupled solve (same shared routine).
    fn host_solve(&self, ctx: &GpuContext, mat: &wgpu::Buffer, rhs: &wgpu::Buffer, x: &wgpu::Buffer) {
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
        let (xh, _res) = crate::solver::banded_schur::banded_gmres_t(
            &a,
            nx,
            ny,
            s,
            &b,
            &self.precond,
            self.restart.max(1),
            200,
            crate::solver::banded_schur::default_step_tol(),
            threads,
        );
        ctx.queue.write_buffer(x, 0, bytemuck::cast_slice(&xh));
    }

    /// Block-Jacobi preconditioned CG (SPD scalar system). Assumes `dinv` is set.
    fn cg(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &wgpu::Buffer,
        mat: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        x: &wgpu::Buffer,
    ) {
        // Initial guess x <- 0, so the initial residual r <- rhs. Then
        // z <- M^{-1} r and the search direction p <- z.
        self.zero(queue, x);
        self.run(device, queue, &self.p_copy, &[(0, &self.dims_buf), (1, rhs), (2, &self.r)], self.ndof); // r <- rhs
        self.run(
            device,
            queue,
            &self.p_papply,
            &[(0, &self.dims_buf), (1, &self.dinv), (2, &self.r), (3, &self.z)],
            self.n,
        ); // z <- Minv r
        self.run(device, queue, &self.p_copy, &[(0, &self.dims_buf), (1, &self.z), (2, &self.p)], self.ndof); // p <- z

        let bnorm = self.dot(device, queue, rhs, rhs).sqrt().max(1e-30);
        let mut rz = self.dot(device, queue, &self.r, &self.z);
        let tol = 1e-10_f64;
        let maxit = (self.ndof as usize * 4).max(200);

        for _ in 0..maxit {
            // Ap <- A p
            self.run(
                device,
                queue,
                &self.p_spmv,
                &[(0, grid), (1, &self.dims_buf), (2, mat), (3, &self.p), (4, &self.ap)],
                self.n,
            );
            let pap = self.dot(device, queue, &self.p, &self.ap);
            if pap.abs() < 1e-300 {
                break;
            }
            let alpha = rz / pap;
            // x <- x + alpha p ; r <- r - alpha Ap
            self.set_scalar(queue, alpha as f32, 0.0);
            self.run(device, queue, &self.p_axpy, &[(0, &self.dims_buf), (1, &self.scalar_buf), (2, &self.p), (3, x)], self.ndof);
            self.set_scalar(queue, -(alpha as f32), 0.0);
            self.run(device, queue, &self.p_axpy, &[(0, &self.dims_buf), (1, &self.scalar_buf), (2, &self.ap), (3, &self.r)], self.ndof);

            let rr = self.dot(device, queue, &self.r, &self.r);
            if rr.sqrt() / bnorm <= tol {
                break;
            }
            // z <- Minv r ; rz_new ; beta ; p <- z + beta p
            self.run(device, queue, &self.p_papply, &[(0, &self.dims_buf), (1, &self.dinv), (2, &self.r), (3, &self.z)], self.n);
            let rz_new = self.dot(device, queue, &self.r, &self.z);
            let beta = rz_new / rz;
            self.set_scalar(queue, 0.0, beta as f32);
            self.run(device, queue, &self.p_xpby, &[(0, &self.dims_buf), (1, &self.scalar_buf), (2, &self.z), (3, &self.p)], self.ndof);
            rz = rz_new;
        }
    }

    /// Restarted, block-Jacobi (left)-preconditioned GMRES — the GPU analog of the
    /// CPU `banded_block_gmres`. Arnoldi/Givens run on the host over the small
    /// Hessenberg; all vector work (SpMV, precond, dot, axpy, scale) is on the
    /// device, with only scalars crossing the bus. Assumes `dinv` is set.
    fn gmres(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &wgpu::Buffer,
        mat: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        x: &wgpu::Buffer,
    ) {
        let axpy = |src: &wgpu::Buffer, dst: &wgpu::Buffer, a: f64| {
            self.set_scalar(queue, a as f32, 0.0);
            self.run(device, queue, &self.p_axpy, &[(0, &self.dims_buf), (1, &self.scalar_buf), (2, src), (3, dst)], self.ndof);
        };
        let scale = |src: &wgpu::Buffer, dst: &wgpu::Buffer, a: f64| {
            self.set_scalar(queue, a as f32, 0.0);
            self.run(device, queue, &self.p_vscale, &[(0, &self.dims_buf), (1, &self.scalar_buf), (2, src), (3, dst)], self.ndof);
        };
        let copy = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            self.run(device, queue, &self.p_copy, &[(0, &self.dims_buf), (1, src), (2, dst)], self.ndof);
        };
        let precond = |rin: &wgpu::Buffer, zout: &wgpu::Buffer| {
            self.run(device, queue, &self.p_papply, &[(0, &self.dims_buf), (1, &self.dinv), (2, rin), (3, zout)], self.n);
        };
        let spmv = |xin: &wgpu::Buffer, yout: &wgpu::Buffer| {
            self.run(device, queue, &self.p_spmv, &[(0, grid), (1, &self.dims_buf), (2, mat), (3, xin), (4, yout)], self.n);
        };
        let dot = |a: &wgpu::Buffer, b: &wgpu::Buffer| self.dot(device, queue, a, b);
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
            spmv(x, &self.ap);
            copy(rhs, &self.r);
            axpy(&self.ap, &self.r, -1.0);
            precond(&self.r, &self.z);
            let beta = norm(&self.z);
            if beta / bnorm <= tol {
                final_rel = beta / bnorm;
                break;
            }
            scale(&self.z, &self.basis[0], 1.0 / beta);

            let mut h = vec![vec![0.0f64; m]; m + 1];
            let mut g = vec![0.0f64; m + 1];
            g[0] = beta;
            let mut cs = vec![0.0f64; m];
            let mut sn = vec![0.0f64; m];
            let mut k_used = 0usize;

            for k in 0..m {
                // w = M^{-1} A v_k  (use `v` as w).
                spmv(&self.basis[k], &self.ap);
                precond(&self.ap, &self.v);
                // Modified Gram-Schmidt against the existing basis.
                for i in 0..=k {
                    h[i][k] = dot(&self.v, &self.basis[i]);
                    axpy(&self.basis[i], &self.v, -h[i][k]);
                }
                h[k + 1][k] = norm(&self.v);
                if h[k + 1][k] > 1e-14 {
                    scale(&self.v, &self.basis[k + 1], 1.0 / h[k + 1][k]);
                } else {
                    self.zero(queue, &self.basis[k + 1]);
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
                y[i] = if h[i][i].abs() > 1e-300 { sum / h[i][i] } else { 0.0 };
            }
            for i in 0..kk {
                axpy(&self.basis[i], x, y[i]);
            }
            total_k += kk;

            // True-residual convergence check.
            spmv(x, &self.ap);
            copy(rhs, &self.r);
            axpy(&self.ap, &self.r, -1.0);
            final_rel = norm(&self.r) / bnorm;
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
    let mut enc =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
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
