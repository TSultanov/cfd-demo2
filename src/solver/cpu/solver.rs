//! `CpuSolver`: the CPU backend's solve driver.
//!
//! Reuses the model's backend-agnostic IR end-to-end: it builds the same
//! `SolverRecipe` the GPU path builds, regenerates the model's kernels as typed
//! `KernelProgram`s, allocates CPU-side buffers from the mesh + state layout, and
//! runs the coupled solve loop by *interpreting* the kernels (flux → gradients →
//! assembly) and solving the assembled CSR system with a CPU iterative solver
//! (replacing the GPU FGMRES/AMG kernel stack), then interpreting the update
//! kernel.
//!
//! The public surface mirrors the subset of `GpuUnifiedSolver` the MMS tests use,
//! so the same manufactured-solution references validate this backend.

use std::collections::HashMap;

use cfd2_ir::ast::Stmt;

use crate::solver::cpu::interpreter::{Buffers, Ctx, Frame, Interpreter, Value};
use crate::solver::cpu::linalg::{bicgstab, CsrView};
use crate::solver::cpu::lowering::model_kernel_programs;
use crate::solver::cpu::CpuBackendConfig;
use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};
use crate::solver::gpu::structs::{GpuConstants, PreconditionerType};
use crate::solver::ir::DispatchDomain;
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use crate::solver::TimeScheme;

const DIRICHLET: u32 = 1;
/// Linear-solve budget for the CPU BiCGSTAB (per outer iteration).
const LINEAR_MAX_ITERS: usize = 5000;
const LINEAR_TOL: f64 = 1e-9;

/// A model kernel prepared for interpretation: its dispatch domain plus the
/// concatenated `indexing ++ preamble ++ body` statement list.
struct CpuKernel {
    domain: DispatchDomain,
    stmts: Vec<Stmt>,
}

pub struct CpuSolver {
    num_cells: usize,
    num_faces: usize,
    state_stride: u32,
    t_offset: u32,
    unknowns_per_cell: usize,

    buffers: Buffers,
    kernels: HashMap<String, CpuKernel>,

    // CSR topology (fixed for the mesh); also mirrored into buffers for kernels.
    row_offsets: Vec<u32>,
    col_indices: Vec<u32>,

    // Faces grouped by boundary type index (`GpuBoundaryType as u32`).
    boundary_faces: Vec<Vec<u32>>,

    constants: GpuConstants,
    state_layout: crate::solver::model::backend::StateLayout,

    outer_iters: usize,
    dt: f32,
    dt_old: f32,
    time: f32,
    time_scheme: TimeScheme,
    needs_gradients: bool,
    #[allow(dead_code)]
    config: CpuBackendConfig,
}

impl CpuSolver {
    pub fn new(
        mesh: &Mesh,
        model: ModelSpec,
        advection_scheme: Scheme,
        time_scheme: TimeScheme,
        config: CpuBackendConfig,
    ) -> Result<Self, String> {
        let recipe = SolverRecipe::from_model(
            &model,
            advection_scheme,
            time_scheme,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
        )?;
        let schemes = SchemeRegistry::new(advection_scheme);
        let (programs, wgsl_only) = model_kernel_programs(&model, &schemes)?;
        if !wgsl_only.is_empty() {
            // For scalar transport every model kernel is typed AST; if a model
            // brings WGSL-only kernels they must be migrated before CPU support.
            let names: Vec<&str> = wgsl_only.iter().map(|k| k.as_str()).collect();
            return Err(format!(
                "model `{}` has WGSL-only kernels (not CPU-executable): {}",
                model.id,
                names.join(", ")
            ));
        }

        let mut kernels = HashMap::new();
        for (id, prog) in programs {
            let mut stmts = Vec::with_capacity(
                prog.indexing.len() + prog.preamble.len() + prog.body.len(),
            );
            stmts.extend_from_slice(&prog.indexing);
            stmts.extend_from_slice(&prog.preamble);
            stmts.extend_from_slice(&prog.body);
            kernels.insert(
                id.as_str().to_string(),
                CpuKernel {
                    domain: prog.dispatch.clone(),
                    stmts,
                },
            );
        }

        let state_layout = model.state_layout.clone();
        let state_stride = state_layout.stride();
        let t_offset = state_layout
            .offset_for(crate::solver::model::SCALAR_TRANSPORT_FIELD)
            .ok_or("model has no scalar transport field `T`")?;
        let unknowns_per_cell = recipe.unknowns_per_cell;
        if unknowns_per_cell != 1 {
            return Err(format!(
                "CpuSolver currently supports scalar systems only (unknowns_per_cell={unknowns_per_cell})"
            ));
        }

        let num_cells = mesh.num_cells();
        let num_faces = mesh.num_faces();

        let (row_offsets, col_indices, diagonal_indices, cell_face_matrix_indices) =
            build_csr_topology(mesh);
        let nnz = *row_offsets.last().unwrap() as usize;

        let mut buffers = Buffers::new();
        upload_mesh(&mut buffers, mesh);
        buffers.insert_u32("scalar_row_offsets", row_offsets.clone());
        buffers.insert_u32("row_offsets", row_offsets.clone());
        buffers.insert_u32("diagonal_indices", diagonal_indices);
        buffers.insert_u32("cell_face_matrix_indices", cell_face_matrix_indices);

        // State + history + per-iteration snapshot.
        let state_len = num_cells * state_stride as usize;
        buffers.insert_f32("state", vec![0.0; state_len]);
        buffers.insert_f32("state_old", vec![0.0; state_len]);
        buffers.insert_f32("state_old_old", vec![0.0; state_len]);
        buffers.insert_f32("state_iter", vec![0.0; state_len]);

        // Flux (one value per face), gradients, linear system, solution.
        buffers.insert_f32("fluxes", vec![0.0; num_faces]);
        // grad_state mirrors the state layout: one Vector2 gradient per state slot
        // (indexed `grad_state[cell * stride + component]`), so it holds
        // `num_cells * stride` Vector2 elements (× 2 floats each).
        buffers.insert_vec2(
            "grad_state",
            vec![0.0; num_cells * state_stride as usize * 2],
        );
        buffers.insert_f32("matrix_values", vec![0.0; nnz]);
        buffers.insert_f32("rhs", vec![0.0; num_cells]);
        buffers.insert_f32("x", vec![0.0; num_cells]);

        // Boundary conditions: kind per face (scalar stride 1), values per face.
        let mut bc_kind = vec![0u32; num_faces];
        for f in 0..num_faces {
            if mesh.face_neighbor[f].is_none() {
                // scalar_transport declares Dirichlet on every boundary type used.
                bc_kind[f] = DIRICHLET;
            }
        }
        buffers.insert_u32("bc_kind", bc_kind);
        buffers.insert_f32("bc_value", vec![0.0; num_faces]);

        let boundary_faces = group_boundary_faces(mesh);

        let mut constants = recipe.initial_constants;
        constants.dtau = 0.0; // coupled solve: no dual-time term
        constants.time_scheme = time_scheme as u32;

        Ok(Self {
            num_cells,
            num_faces,
            state_stride,
            t_offset,
            unknowns_per_cell,
            buffers,
            kernels,
            row_offsets,
            col_indices,
            boundary_faces,
            constants,
            state_layout,
            outer_iters: 2,
            dt: 0.01,
            dt_old: 0.01,
            time: 0.0,
            time_scheme,
            needs_gradients: recipe.needs_gradients(),
            config,
        })
    }

    // ── configuration ────────────────────────────────────────────────────

    pub fn set_outer_iters(&mut self, n: usize) {
        self.outer_iters = n.max(1);
    }
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }
    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.time_scheme = scheme;
        self.constants.time_scheme = scheme as u32;
    }
    pub fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.constants.scheme = scheme.gpu_id();
    }

    // ── field I/O ─────────────────────────────────────────────────────────

    pub fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        let off = self
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("unknown field `{field}`"))? as usize;
        let stride = self.state_stride as usize;
        for (i, &v) in values.iter().enumerate() {
            self.buffers.set_f32("state", i * stride + off, v as f32);
        }
        Ok(())
    }

    pub fn set_field_scalar_current(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        self.set_field_scalar(field, values)
    }

    pub fn set_field_vec2(&mut self, field: &str, values: &[(f64, f64)]) -> Result<(), String> {
        let off = self
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("unknown field `{field}`"))? as usize;
        let stride = self.state_stride as usize;
        for (i, &(x, y)) in values.iter().enumerate() {
            self.buffers.set_f32("state", i * stride + off, x as f32);
            self.buffers.set_f32("state", i * stride + off + 1, y as f32);
        }
        Ok(())
    }

    pub fn get_field_scalar(&self, field: &str) -> Result<Vec<f64>, String> {
        let off = self
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("unknown field `{field}`"))? as usize;
        let stride = self.state_stride as usize;
        Ok((0..self.num_cells)
            .map(|i| self.buffers.get_f32("state", i * stride + off) as f64)
            .collect())
    }

    // ── boundary conditions ───────────────────────────────────────────────

    pub fn set_boundary_values_per_face(
        &mut self,
        boundary: GpuBoundaryType,
        _field: &str,
        component: u32,
        value_for_face: &dyn Fn(u32) -> f32,
    ) -> Result<(), String> {
        let bidx = boundary as usize;
        let stride = self.unknowns_per_cell;
        let faces = self
            .boundary_faces
            .get(bidx)
            .ok_or_else(|| format!("invalid boundary index {bidx}"))?
            .clone();
        for face in faces {
            self.buffers
                .set_f32("bc_value", face as usize * stride + component as usize, value_for_face(face));
        }
        Ok(())
    }

    pub fn set_boundary_scalar(
        &mut self,
        boundary: GpuBoundaryType,
        field: &str,
        value: f32,
    ) -> Result<(), String> {
        self.set_boundary_values_per_face(boundary, field, 0, &|_| value)
    }

    // ── stepping ──────────────────────────────────────────────────────────

    pub fn initialize_history(&mut self) {
        let state = self.buffers.f32_vec("state");
        self.buffers.copy_into_f32("state_old", &state);
        self.buffers.copy_into_f32("state_old_old", &state);
        self.buffers.copy_into_f32("state_iter", &state);
    }

    pub fn step(&mut self) {
        // Rotate time history: old_old <- old, old <- current state.
        let old = self.buffers.f32_vec("state_old");
        self.buffers.copy_into_f32("state_old_old", &old);
        let cur = self.buffers.f32_vec("state");
        self.buffers.copy_into_f32("state_old", &cur);

        self.constants.dt = self.dt;
        self.constants.dt_old = self.dt_old;
        self.time += self.dt;
        self.constants.time = self.time;
        self.constants.time_scheme = self.time_scheme as u32;
        let ctx = constants_ctx(&self.constants);

        let threads = self.config.threads;
        let nf = self.num_faces;
        let nc = self.num_cells;

        // Advective flux from the (frozen) advecting velocity — once per step.
        run_kernel(&self.buffers, &ctx, &self.kernels, "flux_module", nf, nc, threads);

        let assembly_id = if self.needs_gradients {
            "generic_coupled_assembly_grad_state"
        } else {
            "generic_coupled_assembly"
        };

        for _ in 0..self.outer_iters {
            // Snapshot current iterate (used only when dual-time is active).
            let cur = self.buffers.f32_vec("state");
            self.buffers.copy_into_f32("state_iter", &cur);

            if self.needs_gradients {
                run_kernel(
                    &self.buffers,
                    &ctx,
                    &self.kernels,
                    "packed_state_gradients",
                    nf,
                    nc,
                    threads,
                );
            }

            run_kernel(&self.buffers, &ctx, &self.kernels, assembly_id, nf, nc, threads);

            self.linear_solve();

            run_kernel(
                &self.buffers,
                &ctx,
                &self.kernels,
                "generic_coupled_update",
                nf,
                nc,
                threads,
            );
        }

        self.dt_old = self.dt;
    }

    /// Solve `A x = b` (the assembled CSR system) on the CPU; result lands in the
    /// `x` buffer that the update kernel consumes.
    fn linear_solve(&mut self) {
        // Initial guess = current T.
        let stride = self.state_stride as usize;
        let off = self.t_offset as usize;
        let state = self.buffers.f32_vec("state");
        let mut x: Vec<f32> = (0..self.num_cells)
            .map(|i| state[i * stride + off])
            .collect();

        let matrix = self.buffers.f32_vec("matrix_values");
        let rhs = self.buffers.f32_vec("rhs");
        let a = CsrView {
            row_offsets: &self.row_offsets,
            col_indices: &self.col_indices,
            values: &matrix,
        };
        bicgstab(&a, &rhs, &mut x, LINEAR_MAX_ITERS, LINEAR_TOL, self.config.simd);

        self.buffers.copy_into_f32("x", &x);
    }
}

// ── helpers ───────────────────────────────────────────────────────────────

fn run_kernel(
    buffers: &Buffers,
    ctx: &Ctx,
    kernels: &HashMap<String, CpuKernel>,
    id: &str,
    num_faces: usize,
    num_cells: usize,
    threads: usize,
) {
    let kernel = kernels
        .get(id)
        .unwrap_or_else(|| panic!("CPU backend missing kernel `{id}`"));
    let domain = match kernel.domain {
        DispatchDomain::Faces => num_faces,
        DispatchDomain::Cells => num_cells,
        DispatchDomain::Custom(_) => panic!("custom dispatch domain unsupported on CPU"),
    };
    let stmts = &kernel.stmts;
    crate::solver::cpu::parallel::parallel_for(domain, threads, |idx| {
        // The launch wrapper's `let idx = <invocation_index_expr>;` and the
        // `if (idx >= bound) return;` guard are synthesized by the WGSL emitter
        // from LaunchSemantics, not stored in the kernel body. On CPU we own the
        // dispatch loop, so bind `idx` directly (and `global_id` for any kernel
        // that reads it). We iterate exactly `[0, domain)`, so the bound guard is
        // always false and can be skipped.
        let mut frame = Frame::new()
            .with_local("idx", Value::U32(idx as u32))
            .with_local("global_id", Value::Vec3([idx as f32, 0.0, 0.0]));
        Interpreter::new(buffers, ctx).run(stmts, &mut frame);
    });
}

/// Build the constants uniform as an interpreter struct value (`constants.field`).
fn constants_ctx(c: &GpuConstants) -> Ctx {
    Ctx::new()
        .with_constant("constants", "dt", Value::F32(c.dt))
        .with_constant("constants", "dt_old", Value::F32(c.dt_old))
        .with_constant("constants", "dtau", Value::F32(c.dtau))
        .with_constant("constants", "time", Value::F32(c.time))
        .with_constant("constants", "viscosity", Value::F32(c.viscosity))
        .with_constant("constants", "density", Value::F32(c.density))
        .with_constant("constants", "component", Value::U32(c.component))
        .with_constant("constants", "alpha_p", Value::F32(c.alpha_p))
        .with_constant("constants", "scheme", Value::U32(c.scheme))
        .with_constant("constants", "alpha_u", Value::F32(c.alpha_u))
        .with_constant("constants", "stride_x", Value::U32(c.stride_x))
        .with_constant("constants", "time_scheme", Value::U32(c.time_scheme))
        .with_constant("constants", "eos_gamma", Value::F32(c.eos_gamma))
        .with_constant("constants", "eos_gm1", Value::F32(c.eos_gm1))
        .with_constant("constants", "eos_r", Value::F32(c.eos_r))
        .with_constant("constants", "eos_dp_drho", Value::F32(c.eos_dp_drho))
        .with_constant("constants", "eos_p_offset", Value::F32(c.eos_p_offset))
        .with_constant("constants", "eos_theta_ref", Value::F32(c.eos_theta_ref))
}

/// Upload mesh geometry/topology into named CPU buffers matching kernel bindings.
fn upload_mesh(buffers: &mut Buffers, mesh: &Mesh) {
    let nf = mesh.num_faces();
    buffers.insert_u32(
        "face_owner",
        mesh.face_owner.iter().map(|&o| o as u32).collect(),
    );
    buffers.insert_i32(
        "face_neighbor",
        mesh.face_neighbor
            .iter()
            .map(|n| n.map(|v| v as i32).unwrap_or(-1))
            .collect(),
    );
    buffers.insert_f32("face_areas", mesh.face_area.iter().map(|&a| a as f32).collect());
    buffers.insert_vec2("face_normals", interleave(&mesh.face_nx, &mesh.face_ny));
    buffers.insert_vec2("cell_centers", interleave(&mesh.cell_cx, &mesh.cell_cy));
    buffers.insert_vec2("face_centers", interleave(&mesh.face_cx, &mesh.face_cy));
    buffers.insert_f32("cell_vols", mesh.cell_vol.iter().map(|&v| v as f32).collect());
    buffers.insert_u32(
        "cell_face_offsets",
        mesh.cell_face_offsets.iter().map(|&o| o as u32).collect(),
    );
    buffers.insert_u32(
        "cell_faces",
        mesh.cell_faces.iter().map(|&f| f as u32).collect(),
    );
    buffers.insert_u32(
        "face_boundary",
        mesh.face_boundary
            .iter()
            .map(|b| b.map(|t| t.bc_table_index() as u32).unwrap_or(0))
            .collect(),
    );
    // face_wrap_shift is empty on non-periodic meshes (treat as zeros).
    let wrap: Vec<f32> = if mesh.face_wrap_shift.is_empty() {
        vec![0.0; nf * 2]
    } else {
        mesh.face_wrap_shift
            .iter()
            .flat_map(|s| [s[0] as f32, s[1] as f32])
            .collect()
    };
    buffers.insert_vec2("face_wrap_shift", wrap);
}

fn interleave(xs: &[f64], ys: &[f64]) -> Vec<f32> {
    xs.iter()
        .zip(ys)
        .flat_map(|(&x, &y)| [x as f32, y as f32])
        .collect()
}

/// Construct the scalar CSR topology consistent with the assembly kernel's index
/// maps: each row holds the diagonal (rank 0) followed by one entry per interior
/// face. Returns `(row_offsets, col_indices, diagonal_indices,
/// cell_face_matrix_indices)`.
fn build_csr_topology(mesh: &Mesh) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = mesh.num_cells();
    let mut row_offsets = vec![0u32; n + 1];
    for i in 0..n {
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        let interior = (start..end)
            .filter(|&k| mesh.face_neighbor[mesh.cell_faces[k]].is_some())
            .count();
        row_offsets[i + 1] = row_offsets[i] + 1 + interior as u32;
    }
    let nnz = *row_offsets.last().unwrap() as usize;
    let mut col_indices = vec![0u32; nnz];
    let mut diagonal_indices = vec![0u32; n];
    let mut cell_face_matrix_indices = vec![0u32; mesh.cell_faces.len()];

    for i in 0..n {
        let base = row_offsets[i] as usize;
        col_indices[base] = i as u32;
        diagonal_indices[i] = base as u32;
        let mut pos = base + 1;
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        for k in start..end {
            let f = mesh.cell_faces[k];
            match mesh.face_neighbor[f] {
                Some(nb) => {
                    let other = if mesh.face_owner[f] == i { nb } else { mesh.face_owner[f] };
                    col_indices[pos] = other as u32;
                    cell_face_matrix_indices[k] = pos as u32;
                    pos += 1;
                }
                None => {
                    // Boundary face: no column entry; point at the diagonal so any
                    // stray read is harmless (the kernel guards with is_boundary).
                    cell_face_matrix_indices[k] = base as u32;
                }
            }
        }
    }
    (row_offsets, col_indices, diagonal_indices, cell_face_matrix_indices)
}

/// Group boundary faces by `BoundaryType::bc_table_index()` (== `GpuBoundaryType
/// as u32`); index 0 is "None"/interior.
fn group_boundary_faces(mesh: &Mesh) -> Vec<Vec<u32>> {
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); 8];
    for f in 0..mesh.num_faces() {
        if let Some(bt) = mesh.face_boundary[f] {
            groups[bt.bc_table_index()].push(f as u32);
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
    use crate::solver::model::{
        scalar_transport_model, ADVECTING_VELOCITY_FIELD, SCALAR_TRANSPORT_FIELD,
        SCALAR_TRANSPORT_KAPPA, SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
    };
    use std::f64::consts::PI;

    fn unit_square(n: usize) -> Mesh {
        generate_structured_rect_mesh(
            n,
            n,
            1.0,
            1.0,
            BoundarySides {
                left: BoundaryType::Inlet,
                right: BoundaryType::Outlet,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        )
    }

    // Manufactured solution: T* = sin(pi x) cos(pi y), U = (4, 2) constant,
    // S = U.grad(T*) - kappa lap(T*).  (Mirrors mms_scalar_transport_order_test.)
    fn exact(x: f64, y: f64) -> f64 {
        (PI * x).sin() * (PI * y).cos()
    }
    fn source(x: f64, y: f64) -> f64 {
        let (ux, uy) = (4.0, 2.0);
        let k = SCALAR_TRANSPORT_KAPPA;
        let dtdx = PI * (PI * x).cos() * (PI * y).cos();
        let dtdy = -PI * (PI * x).sin() * (PI * y).sin();
        let lap = -2.0 * PI * PI * exact(x, y);
        ux * dtdx + uy * dtdy - k * lap
    }

    fn l2_error(mesh: &Mesh, t: &[f64]) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..mesh.num_cells() {
            let e = t[i] - exact(mesh.cell_cx[i], mesh.cell_cy[i]);
            num += e * e * mesh.cell_vol[i];
            den += mesh.cell_vol[i];
        }
        (num / den).sqrt()
    }

    /// Least-squares slope of log(err) vs log(h).
    fn fit_order(hs: &[f64], errs: &[f64]) -> f64 {
        let n = hs.len() as f64;
        let lx: Vec<f64> = hs.iter().map(|h| h.ln()).collect();
        let ly: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
        let sx: f64 = lx.iter().sum();
        let sy: f64 = ly.iter().sum();
        let sxx: f64 = lx.iter().map(|x| x * x).sum();
        let sxy: f64 = lx.iter().zip(&ly).map(|(x, y)| x * y).sum();
        (n * sxy - sx * sy) / (n * sxx - sx * sx)
    }

    fn solve_steady(n: usize, scheme: Scheme) -> (Mesh, Vec<f64>) {
        solve_steady_cfg(n, scheme, CpuBackendConfig::default())
    }

    fn solve_steady_cfg(n: usize, scheme: Scheme, config: CpuBackendConfig) -> (Mesh, Vec<f64>) {
        let mesh = unit_square(n);
        let model = scalar_transport_model().expect("model");
        let mut solver = CpuSolver::new(&mesh, model, scheme, TimeScheme::Euler, config)
            .expect("create cpu solver");
        solver.set_outer_iters(2);
        solver.set_dt(0.2);

        let face_value = |face: u32| exact(mesh.face_cx[face as usize], mesh.face_cy[face as usize]) as f32;
        for b in [
            GpuBoundaryType::Inlet,
            GpuBoundaryType::Outlet,
            GpuBoundaryType::Wall,
        ] {
            solver
                .set_boundary_values_per_face(b, SCALAR_TRANSPORT_FIELD, 0, &face_value)
                .expect("bc");
        }

        let u: Vec<(f64, f64)> = vec![(4.0, 2.0); mesh.num_cells()];
        solver.set_field_vec2(ADVECTING_VELOCITY_FIELD, &u).expect("U");
        let src: Vec<f64> = (0..mesh.num_cells())
            .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
            .collect();
        solver
            .set_field_scalar(SCALAR_TRANSPORT_MMS_SOURCE_FIELD, &src)
            .expect("src");
        solver
            .set_field_scalar(SCALAR_TRANSPORT_FIELD, &vec![0.0; mesh.num_cells()])
            .expect("init");
        solver.initialize_history();

        let mut prev = solver.get_field_scalar(SCALAR_TRANSPORT_FIELD).unwrap();
        for _ in 0..400 {
            solver.step();
            let cur = solver.get_field_scalar(SCALAR_TRANSPORT_FIELD).unwrap();
            let delta = cur
                .iter()
                .zip(&prev)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            prev = cur;
            if delta < 4e-6 {
                break;
            }
        }
        (mesh, prev)
    }

    #[test]
    fn cpu_scalar_transport_upwind_converges() {
        let levels = [16usize, 32, 64];
        let mut hs = Vec::new();
        let mut errs = Vec::new();
        for &n in &levels {
            let (mesh, t) = solve_steady(n, Scheme::Upwind);
            let e = l2_error(&mesh, &t);
            println!("[cpu-mms][upwind] n={n} l2={e:.3e}");
            hs.push(1.0 / n as f64);
            errs.push(e);
        }
        // Errors must decrease monotonically and the finest must be small.
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "errors not decreasing: {errs:?}");
        assert!(*errs.last().unwrap() < 1e-2, "finest error too large: {errs:?}");
        let order = fit_order(&hs, &errs);
        println!("[cpu-mms][upwind] observed order = {order:.3}");
        // Upwind advection + 2nd-order diffusion: mixed order, ~1 (matches the
        // GPU MMS test's 1.0 ± 0.2 window, with slack for the coarse 16..64 fit).
        assert!(
            (0.6..=1.6).contains(&order),
            "implausible order {order:.3} for upwind advection-diffusion"
        );
    }

    #[test]
    fn cpu_scalar_transport_sou_converges() {
        // Second-order upwind exercises the gradient path (packed_state_gradients
        // + generic_coupled_assembly_grad_state). Expect ~2nd order.
        let levels = [8usize, 16, 32, 64];
        let mut hs = Vec::new();
        let mut errs = Vec::new();
        for &n in &levels {
            let (mesh, t) = solve_steady(n, Scheme::SecondOrderUpwind);
            let e = l2_error(&mesh, &t);
            println!("[cpu-mms][sou] n={n} l2={e:.3e}");
            hs.push(1.0 / n as f64);
            errs.push(e);
        }
        let order = fit_order(&hs, &errs);
        println!("[cpu-mms][sou] observed order = {order:.3}");
        assert!(
            (1.6..=2.4).contains(&order),
            "implausible SOU order {order:.3} (expected ~2)"
        );
        assert!(*errs.last().unwrap() < 1e-3, "finest SOU error too large: {errs:?}");
    }

    #[test]
    fn cpu_multithread_matches_singlethread() {
        // Runtime-switchable multithreading must not change results: relaxed-atomic
        // buffers + disjoint per-cell writes make the parallel run deterministic and
        // identical to the serial run.
        let n = 32;
        let (_, t1) = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { threads: 1, simd: false });
        let (_, t4) = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { threads: 4, simd: false });
        let max_diff = t1
            .iter()
            .zip(&t4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!("[cpu-mt] n={n} max|1thread - 4thread| = {max_diff:.3e}");
        assert!(
            max_diff == 0.0,
            "multithreaded result differs from serial: max|diff|={max_diff:.3e}"
        );
    }

    #[test]
    fn cpu_compute_options_all_agree() {
        // Validate every runtime CPU computation option against the reference
        // {1 thread, scalar}: {1,4 threads} × {scalar, SIMD}. SIMD reorders the
        // reduction summation so it matches to rounding (not bit-exact); threads
        // are bit-identical.
        let n = 32;
        let base = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { threads: 1, simd: false }).1;
        for (threads, simd, tol) in [
            (4usize, false, 0.0f64),
            (1, true, 1e-4),
            (4, true, 1e-4),
        ] {
            let t = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { threads, simd }).1;
            let max_diff = base
                .iter()
                .zip(&t)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            println!("[cpu-opts] threads={threads} simd={simd} max|diff vs ref|={max_diff:.3e}");
            assert!(
                max_diff <= tol,
                "option (threads={threads}, simd={simd}) diverges: {max_diff:.3e} > {tol:.1e}"
            );
        }
    }
}
