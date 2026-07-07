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
use cfd2_ir::kernel::{BindingAccess, DispatchDomain};

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
}

impl CompiledKernel {
    fn build(device: &wgpu::Device, id: &str, wgsl: &str, bindings: Vec<BindInfo>) -> Self {
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
        Self {
            pipeline,
            bindings,
            layouts,
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
                    resource: resolve(&b.name).as_entire_binding(),
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
) -> Result<Vec<(String, String, Vec<BindInfo>, DispatchDomain)>, String> {
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
                out.push((spec.id.as_str().to_string(), wgsl.to_wgsl(), bindings, p.dispatch));
            }
        }
    }
    Ok(out)
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

    solver: BandedGpuLinAlg,

    outer_iters: usize,
    dt: f64,
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

    /// Build against an existing device/queue.
    pub fn with_context(
        ctx: GpuContext,
        grid: StructuredGrid,
        model: &ModelSpec,
        dt: f64,
        outer_iters: usize,
    ) -> Result<Self, String> {
        if model.system.topology() != cfd2_ir::equation::TopologyMode::Structured2D {
            return Err("StructuredGpuSolver requires a Structured2D model".to_string());
        }
        let scheme = Scheme::Upwind;
        let recipe = SolverRecipe::from_model(
            model,
            scheme,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
        )?;
        let schemes = SchemeRegistry::new(scheme);

        let lowered = lower_structured_kernels(model, &schemes)?;
        let wgsl_by_id: HashMap<String, (String, Vec<BindInfo>)> = lowered
            .into_iter()
            .map(|(id, wgsl, bindings, _)| (id, (wgsl, bindings)))
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
            let (wgsl, bindings) = wgsl_by_id
                .get(id)
                .ok_or_else(|| format!("structured GPU model missing kernel `{id}`"))?;
            kernels.insert(
                id.to_string(),
                CompiledKernel::build(&ctx.device, id, wgsl, bindings.clone()),
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

        let mut constants = recipe.initial_constants;
        constants.dt = dt as f32;
        constants.dt_old = dt as f32;
        constants.dtau = 0.0;
        constants.time_scheme = 0; // Euler
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

        let solver = BandedGpuLinAlg::new(dev, grid.nx as u32, grid.ny as u32, s as u32);

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
            solver,
            outer_iters: outer_iters.max(1),
            dt,
        })
    }

    fn buf(&self, name: &str) -> &wgpu::Buffer {
        // `constants` and `grid` are the two uniforms; everything else is a
        // named storage buffer.
        match name {
            "constants" => &self.constants_buf,
            "grid" => &self.grid_buf,
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
        self.ctx
            .queue
            .write_buffer(&self.constants_buf, 0, bytemuck::bytes_of(&self.constants));
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
        self.ctx
            .queue
            .write_buffer(&self.constants_buf, 0, bytemuck::bytes_of(&self.constants));

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
    }

    /// Read a state component field (length `nx*ny`).
    pub fn state_field(&self, offset: usize) -> Vec<f64> {
        let st = self.read_f32("state", self.n * self.state_stride);
        (0..self.n)
            .map(|p| st[p * self.state_stride + offset] as f64)
            .collect()
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

    /// Host-side block-Jacobi-preconditioned banded GMRES (f64) — the GPU analog
    /// path reads the assembled banded operator + rhs back, solves, and uploads
    /// the correction. Mirrors the CPU `banded_block_gmres`.
    fn host_solve(&self, ctx: &GpuContext, mat: &wgpu::Buffer, rhs: &wgpu::Buffer, x: &wgpu::Buffer) {
        let (nx, ny, s) = (self.nx as usize, self.ny as usize, self.s as usize);
        let a = read_buffer_f32(ctx, mat, nx * ny * BAND_STRIDE * s * s);
        let b = read_buffer_f32(ctx, rhs, nx * ny * s);
        let xh = host_banded_gmres(&a, nx, ny, s, &b, self.restart.max(1), 200, 1e-9);
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

// ===========================================================================
// Host-side banded block GMRES (f64) over a read-back `matrix_values` slice.
// The GPU assembles the operator; this solves the coupled (indefinite) system
// in double precision. Mirrors the CPU `banded_block_gmres` exactly (bands
// [S,W,diag,E,N]; matrix_values[p*5*s*s + 5*s*r + band*s + c]).
// ===========================================================================

#[inline]
fn hb_block(a: &[f32], s: usize, p: usize, band: usize, r: usize, c: usize) -> f64 {
    a[p * 5 * s * s + 5 * s * r + band * s + c] as f64
}

fn hb_spmv(a: &[f32], nx: usize, ny: usize, s: usize, x: &[f64]) -> Vec<f64> {
    let n = nx * ny;
    let mut y = vec![0.0f64; n * s];
    for j in 0..ny {
        for i in 0..nx {
            let p = j * nx + i;
            // (band, neighbour) pairs present; edge bands are zero (closed via RHS).
            let mut nbrs: Vec<(usize, usize)> = vec![(2, p)];
            if j > 0 {
                nbrs.push((0, p - nx));
            }
            if i > 0 {
                nbrs.push((1, p - 1));
            }
            if i + 1 < nx {
                nbrs.push((3, p + 1));
            }
            if j + 1 < ny {
                nbrs.push((4, p + nx));
            }
            for r in 0..s {
                let mut acc = 0.0;
                for &(band, q) in &nbrs {
                    for c in 0..s {
                        acc += hb_block(a, s, p, band, r, c) * x[q * s + c];
                    }
                }
                y[p * s + r] = acc;
            }
        }
    }
    y
}

/// Invert an `s x s` matrix (row-major) via Gauss–Jordan with partial pivoting;
/// singular blocks fall back to the (pseudo-)diagonal inverse.
fn hb_invert(m: &[f64], s: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; s * 2 * s];
    for r in 0..s {
        for c in 0..s {
            a[r * 2 * s + c] = m[r * s + c];
        }
        a[r * 2 * s + s + r] = 1.0;
    }
    for col in 0..s {
        let mut piv = col;
        let mut best = a[col * 2 * s + col].abs();
        for r in (col + 1)..s {
            let v = a[r * 2 * s + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-30 {
            let mut out = vec![0.0f64; s * s];
            for k in 0..s {
                let d = m[k * s + k];
                out[k * s + k] = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };
            }
            return out;
        }
        if piv != col {
            for c in 0..(2 * s) {
                a.swap(col * 2 * s + c, piv * 2 * s + c);
            }
        }
        let d = a[col * 2 * s + col];
        for c in 0..(2 * s) {
            a[col * 2 * s + c] /= d;
        }
        for r in 0..s {
            if r == col {
                continue;
            }
            let f = a[r * 2 * s + col];
            if f != 0.0 {
                for c in 0..(2 * s) {
                    a[r * 2 * s + c] -= f * a[col * 2 * s + c];
                }
            }
        }
    }
    let mut out = vec![0.0f64; s * s];
    for r in 0..s {
        for c in 0..s {
            out[r * s + c] = a[r * 2 * s + s + c];
        }
    }
    out
}

fn hb_diag_inverses(a: &[f32], nx: usize, ny: usize, s: usize) -> Vec<Vec<f64>> {
    (0..nx * ny)
        .map(|p| {
            let mut m = vec![0.0f64; s * s];
            for r in 0..s {
                for c in 0..s {
                    m[r * s + c] = hb_block(a, s, p, 2, r, c);
                }
            }
            hb_invert(&m, s)
        })
        .collect()
}

fn hb_apply_jacobi(minv: &[Vec<f64>], s: usize, r: &[f64]) -> Vec<f64> {
    let ncells = minv.len();
    let mut z = vec![0.0f64; ncells * s];
    for p in 0..ncells {
        for i in 0..s {
            let mut acc = 0.0;
            for k in 0..s {
                acc += minv[p][i * s + k] * r[p * s + k];
            }
            z[p * s + i] = acc;
        }
    }
    z
}

fn hb_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}
fn hb_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Restarted block-Jacobi-preconditioned GMRES on the banded block operator.
fn host_banded_gmres(
    a: &[f32],
    nx: usize,
    ny: usize,
    s: usize,
    b: &[f32],
    restart: usize,
    max_outer: usize,
    tol: f64,
) -> Vec<f32> {
    let n = nx * ny * s;
    let minv = hb_diag_inverses(a, nx, ny, s);
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = hb_norm(&b64).max(1e-30);
    let mut x = vec![0.0f64; n];
    let m = restart;

    for _ in 0..max_outer {
        let ax = hb_spmv(a, nx, ny, s, &x);
        let r0: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        let mut r = hb_apply_jacobi(&minv, s, &r0);
        let beta = hb_norm(&r);
        if beta / bnorm <= tol {
            return x.iter().map(|&v| v as f32).collect();
        }
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(r.iter().map(|&x| x / beta).collect());
        let mut h = vec![vec![0.0f64; m]; m + 1];
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];
        let mut k_used = 0;
        for k in 0..m {
            let av = hb_spmv(a, nx, ny, s, &v[k]);
            let mut w = hb_apply_jacobi(&minv, s, &av);
            for i in 0..=k {
                h[i][k] = hb_dot(&w, &v[i]);
                for t in 0..n {
                    w[t] -= h[i][k] * v[i][t];
                }
            }
            h[k + 1][k] = hb_norm(&w);
            if h[k + 1][k] > 1e-14 {
                v.push(w.iter().map(|&x| x / h[k + 1][k]).collect());
            } else {
                v.push(vec![0.0f64; n]);
            }
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
            for t in 0..n {
                x[t] += y[i] * v[i][t];
            }
        }
        let ax = hb_spmv(a, nx, ny, s, &x);
        r = (0..n).map(|i| b64[i] - ax[i]).collect();
        if hb_norm(&r) / bnorm <= tol {
            return x.iter().map(|&v| v as f32).collect();
        }
    }
    x.iter().map(|&v| v as f32).collect()
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
