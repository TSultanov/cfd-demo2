//! State Redistribution (SRD) for cut-cell small cells (Berger & Giuliani, JCP 2021).
//!
//! A non-merging, conservative, geometry-preserving treatment for the cut-cell
//! *small-cell instability*: tiny "sliver" control volumes — produced where an
//! immersed boundary cuts a background Cartesian cell — destabilize the implicit
//! coupled solve (a single ~6%-volume sliver on the default channel-with-obstacle
//! mesh runs away in the base operator, independent of dt, viscosity, or the
//! transpose-stress term). SRD applies a fixed linear map `u <- S u` after each
//! step, where `S` is a partition-of-unity averaging operator that is *exactly
//! identity* for every cell outside a sliver neighborhood, so the simulation keeps
//! the true geometry and every degree of freedom (no merging, no mesh edit, no
//! viscosity floor).
//!
//! **Status: opt-in, OFF by default.** The cut-cell small-cell instability is
//! properly cured by the immersed no-slip wall BC on the embedded geometry (see
//! `generate_cut_cell_mesh`), which both stabilizes the tiny cut cells — a
//! no-slip face adds a `mu/dist` wall-shear damping that grows as cells shrink —
//! and produces the physical boundary layer SRD's averaging would smear. SRD is
//! retained as an opt-in stabilizer (`GpuUnifiedSolver::set_srd_enabled`); it is
//! built when slivers exist but not applied unless enabled.
//!
//! `S` is conservative by construction: `sum_j V_j (S u)_j == sum_j V_j u_j`
//! (proven in `conservation_error` / the unit tests).
//!
//! **No-op safety.** [`build_srd_operator`] only returns `Some` for meshes that
//! carry the cut-cell *signature* — a dominant plateau of equal-volume "full"
//! background cells plus a small minority of sub-half-nominal slivers. Structured
//! and graded meshes lack that signature, so the operator is not even built there
//! (`None`), and SRD is a guaranteed no-op: Ghia, the compressible-lid reference,
//! and the graded MMS order tests are byte-for-byte untouched.
//!
//! The CPU neighborhood construction + operator mirror the validated prototype in
//! `tests/gui_default_convergence_test.rs` (`srd_neighborhoods` / `srd_apply`,
//! which hold the true-geometry obstacle bounded over 200 steps at the physical
//! velocity scale). The GPU path applies the same operator in two compute passes
//! (gather into a temp buffer, then copy back) to avoid the read-after-write race
//! a cell would otherwise hit reading neighbors' already-overwritten velocity.

use crate::solver::gpu::buffers::{create_buffer, create_buffer_init};
use crate::solver::mesh::Mesh;
use std::collections::{HashMap, HashSet};

/// Sparse row-compressed State-Redistribution operator `S` (`u_new = S u`).
///
/// Built only for cut-cell meshes that actually contain slivers. Rows for
/// non-sliver cells are the single identity entry `(j, 1.0)` (so applying `S`
/// leaves them bit-exact in f32).
#[derive(Debug, Clone)]
pub struct SrdCsr {
    pub n_cells: usize,
    /// CSR row start offsets, length `n_cells + 1`.
    pub row_offsets: Vec<u32>,
    /// Column index per non-zero, length `nnz`.
    pub col_indices: Vec<u32>,
    /// Weight per non-zero (f32, as applied on the GPU), length `nnz`.
    pub weights: Vec<f32>,
    /// Number of cells that received a redistribution neighborhood (|N_i| > 1).
    pub num_small_cells: usize,
}

impl SrdCsr {
    /// Apply `S` on the CPU: `u <- S u`. Reference for the GPU cross-check and
    /// for callers without a device. Accumulates in f64 over the f32 weights.
    pub fn apply_cpu(&self, u: &mut [(f64, f64)]) {
        debug_assert_eq!(u.len(), self.n_cells);
        let mut out = vec![(0.0f64, 0.0f64); self.n_cells];
        for j in 0..self.n_cells {
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            for e in self.row_offsets[j] as usize..self.row_offsets[j + 1] as usize {
                let k = self.col_indices[e] as usize;
                let w = self.weights[e] as f64;
                sx += w * u[k].0;
                sy += w * u[k].1;
            }
            out[j] = (sx, sy);
        }
        u.copy_from_slice(&out);
    }

    /// Worst absolute relative conservation error over all columns: `S` is
    /// conservative iff `sum_j V_j S_jk == V_k` for every column `k`
    /// (equivalently `sum_j V_j u_new_j == sum_j V_j u_j` for all `u`). Returns
    /// the max over `k` of `|sum_j V_j S_jk - V_k| / V_k`.
    pub fn conservation_error(&self, cell_vol: &[f64]) -> f64 {
        let mut colsum = vec![0.0f64; self.n_cells];
        for j in 0..self.n_cells {
            for e in self.row_offsets[j] as usize..self.row_offsets[j + 1] as usize {
                let k = self.col_indices[e] as usize;
                colsum[k] += cell_vol[j] * self.weights[e] as f64;
            }
        }
        let mut worst = 0.0f64;
        for k in 0..self.n_cells {
            if cell_vol[k] > 0.0 {
                let rel = (colsum[k] - cell_vol[k]).abs() / cell_vol[k];
                worst = worst.max(rel);
            }
        }
        worst
    }
}

/// Build State-Redistribution neighborhoods (Berger & Giuliani 2021): every cell
/// owns a neighborhood `N_i` (itself); each *sliver* cell grows `N_i` by adding
/// the largest face-adjacent cells until the neighborhood volume reaches
/// `target`. Returns `(neighborhoods, theta)` where `theta_j` = number of
/// neighborhoods containing `j` (overlap count, `>= 1`).
///
/// A cell is a sliver when it is abruptly smaller than its largest face-neighbor
/// (`cell_vol[i] < relative_frac * max_neighbor_vol[i]`) — the cut-cell
/// signature. This relative test is what makes SRD safe on graded meshes:
/// smoothly graded cells differ from their neighbors only by the (near-1) grading
/// ratio, so they are never flagged, whereas a cut sliver is a sudden drop. On
/// the validated obstacle mesh each sliver has a near-full neighbor, so this
/// reproduces the prototype's global `0.5 * nominal` flagging
/// (`tests/gui_default_convergence_test.rs::srd_neighborhoods`).
fn srd_neighborhoods(
    mesh: &Mesh,
    relative_frac: f64,
    target: f64,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = mesh.num_cells();
    let other = |fi: usize, c: usize| -> Option<usize> {
        if mesh.face_owner[fi] == c {
            mesh.face_neighbor[fi]
        } else {
            Some(mesh.face_owner[fi])
        }
    };
    // Largest face-neighbor volume per cell (0 if the cell has no interior
    // neighbor — then it can never be flagged, which is correct).
    let mut max_nbr_vol = vec![0.0f64; n];
    for c in 0..n {
        for &fi in &mesh.cell_faces[mesh.cell_face_offsets[c]..mesh.cell_face_offsets[c + 1]] {
            if let Some(o) = other(fi, c) {
                max_nbr_vol[c] = max_nbr_vol[c].max(mesh.cell_vol[o]);
            }
        }
    }
    let mut neigh: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    for i in 0..n {
        if !(mesh.cell_vol[i] < relative_frac * max_nbr_vol[i]) {
            continue;
        }
        let mut members = vec![i];
        let mut set: HashSet<usize> = HashSet::from([i]);
        let mut vol = mesh.cell_vol[i];
        while vol < target {
            let mut best: Option<(f64, usize)> = None;
            for &m in &members {
                for &fi in &mesh.cell_faces[mesh.cell_face_offsets[m]..mesh.cell_face_offsets[m + 1]]
                {
                    if let Some(o) = other(fi, m) {
                        if !set.contains(&o) && best.map_or(true, |(bv, _)| mesh.cell_vol[o] > bv) {
                            best = Some((mesh.cell_vol[o], o));
                        }
                    }
                }
            }
            match best {
                Some((v, nb)) => {
                    members.push(nb);
                    set.insert(nb);
                    vol += v;
                }
                None => break,
            }
        }
        neigh[i] = members;
    }
    let mut theta = vec![0usize; n];
    for ni in &neigh {
        for &j in ni {
            theta[j] += 1;
        }
    }
    (neigh, theta)
}

/// Build the SRD operator `S` for `mesh`, or `None` if the mesh is not a
/// sliver-bearing cut-cell mesh (structured / graded → no-op, references
/// untouched).
///
/// The detection + neighborhood thresholds match the validated prototype: a cell
/// is a sliver if its volume is below `0.5 * nominal`, and neighborhoods grow to
/// `nominal`, where `nominal = max(cell_vol)` is the uniform background-cell
/// volume of the cut mesh. The extra cut-cell *signature* guard (a full-cell
/// plateau plus a small sliver minority) keeps SRD from ever firing on graded
/// meshes, whose legitimately small fine-region cells are not slivers.
pub fn build_srd_operator(mesh: &Mesh) -> Option<SrdCsr> {
    let n = mesh.num_cells();
    if n == 0 {
        return None;
    }
    let nominal = mesh.cell_vol.iter().copied().fold(0.0f64, f64::max);
    if !(nominal > 0.0) {
        return None;
    }

    // A sliver is a cell abruptly smaller than its largest face-neighbor (below
    // half its volume); neighborhoods grow to the background volume `nominal`.
    // The relative sliver test (inside `srd_neighborhoods`) is what keeps
    // structured AND graded meshes sliver-free — they then yield only identity
    // rows and we return `None` below, so references stay byte-identical.
    let (neigh, theta) = srd_neighborhoods(mesh, 0.5, nominal);
    let num_small_cells = neigh.iter().filter(|ni| ni.len() > 1).count();
    if num_small_cells == 0 {
        return None; // no slivers (structured / graded) → SRD is a pure no-op
    }

    // Inverse map: inv[j] = { i : j ∈ N_i }.
    let mut inv: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, ni) in neigh.iter().enumerate() {
        for &j in ni {
            inv[j].push(i);
        }
    }
    // M_i = Σ_{m ∈ N_i} V_m / theta_m.
    let m: Vec<f64> = neigh
        .iter()
        .map(|ni| ni.iter().map(|&mm| mesh.cell_vol[mm] / theta[mm] as f64).sum())
        .collect();

    // Assemble S row by row. From the gather/scatter form,
    //   u_new_j = (1/theta_j) Σ_{i: j∈N_i} Q̂_i,   Q̂_i = (1/M_i) Σ_{k∈N_i} (V_k/theta_k) u_k
    // ⇒ S_jk = Σ_{i: j∈N_i ∧ k∈N_i} (1/theta_j) · V_k / (theta_k · M_i).
    // Non-sliver cells have inv[j] = {j}, theta_j = 1, M_j = V_j ⇒ S_jj = 1.
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices: Vec<u32> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    row_offsets.push(0u32);
    let mut row: HashMap<usize, f64> = HashMap::new();
    for j in 0..n {
        row.clear();
        let inv_theta_j = 1.0 / theta[j] as f64;
        for &i in &inv[j] {
            for &k in &neigh[i] {
                // S_jk contribution from neighborhood i: (1/theta_j)·V_k/(theta_k·M_i).
                *row.entry(k).or_insert(0.0) +=
                    inv_theta_j * mesh.cell_vol[k] / (theta[k] as f64 * m[i]);
            }
        }
        // Emit in ascending column order for deterministic, cache-friendly rows.
        let mut entries: Vec<(usize, f64)> = row.iter().map(|(&k, &w)| (k, w)).collect();
        entries.sort_unstable_by_key(|&(k, _)| k);
        for (k, w) in entries {
            col_indices.push(k as u32);
            weights.push(w as f32);
        }
        row_offsets.push(col_indices.len() as u32);
    }

    Some(SrdCsr {
        n_cells: n,
        row_offsets,
        col_indices,
        weights,
        num_small_cells,
    })
}

// =============================================================================
// GPU application
// =============================================================================

const SRD_WGSL: &str = r#"
struct Params {
    n_cells: u32,
    stride: u32,
    u_offset: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<storage, read_write> temp_vel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> col_indices: array<u32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// Pass A: gather S·u from the (old) velocities in `state` into `temp_vel`.
@compute @workgroup_size(64)
fn srd_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.n_cells) { return; }
    let start = row_offsets[j];
    let end = row_offsets[j + 1u];
    var acc = vec2<f32>(0.0, 0.0);
    for (var e = start; e < end; e = e + 1u) {
        let k = col_indices[e];
        let w = weights[e];
        let base = k * params.stride + params.u_offset;
        acc = acc + w * vec2<f32>(state[base], state[base + 1u]);
    }
    temp_vel[j] = acc;
}

// Pass B: copy the redistributed velocities back into `state`.
@compute @workgroup_size(64)
fn srd_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.n_cells) { return; }
    let base = j * params.stride + params.u_offset;
    let v = temp_vel[j];
    state[base] = v.x;
    state[base + 1u] = v.y;
}
"#;

const WORKGROUP_SIZE: u32 = 64;

/// GPU resources to apply the precomputed SRD operator to the solver's live
/// velocity field after each step. Velocity-only (the pressure solve
/// re-establishes continuity the following step, matching the validated
/// prototype).
pub struct SrdGpu {
    n_cells: u32,
    bind_group_layout: wgpu::BindGroupLayout,
    gather: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    b_temp_vel: wgpu::Buffer,
    b_row_offsets: wgpu::Buffer,
    b_col_indices: wgpu::Buffer,
    b_weights: wgpu::Buffer,
    b_params: wgpu::Buffer,
}

impl SrdGpu {
    /// Create the GPU operator. `u_offset`/`stride` locate the vec2 velocity
    /// within each cell's interleaved f32 state slot; `n_cells` is the mesh cell
    /// count (== `csr.n_cells`).
    pub fn new(
        device: &wgpu::Device,
        csr: &SrdCsr,
        u_offset: u32,
        stride: u32,
        n_cells: u32,
    ) -> Self {
        use wgpu::BufferUsages as U;

        let b_row_offsets =
            create_buffer_init(device, "srd:row_offsets", &csr.row_offsets, U::STORAGE);
        let b_col_indices =
            create_buffer_init(device, "srd:col_indices", &csr.col_indices, U::STORAGE);
        let b_weights = create_buffer_init(device, "srd:weights", &csr.weights, U::STORAGE);
        let b_params = create_buffer_init(
            device,
            "srd:params",
            &[n_cells, stride, u_offset, 0u32],
            U::UNIFORM,
        );
        // vec2<f32> per cell.
        let b_temp_vel = create_buffer(
            device,
            "srd:temp_vel",
            n_cells as u64 * 8,
            U::STORAGE,
        );

        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("srd:bgl"),
                entries: &[
                    storage_rw(0), // state
                    storage_rw(1), // temp_vel
                    storage_ro(2), // row_offsets
                    storage_ro(3), // col_indices
                    storage_ro(4), // weights
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
            label: Some("srd:pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("srd:shader"),
            source: wgpu::ShaderSource::Wgsl(SRD_WGSL.into()),
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let gather = make("srd_gather", "srd:gather");
        let scatter = make("srd_scatter", "srd:scatter");

        Self {
            n_cells,
            bind_group_layout,
            gather,
            scatter,
            b_temp_vel,
            b_row_offsets,
            b_col_indices,
            b_weights,
            b_params,
        }
    }

    /// Apply `u <- S u` to the velocity in `state_buffer` (the solver's live,
    /// current state). Encodes two compute passes (gather then copy-back) so the
    /// stencil reads the OLD neighbor velocities, and submits.
    pub fn apply(&self, device: &wgpu::Device, queue: &wgpu::Queue, state_buffer: &wgpu::Buffer) {
        // Rebuild the bind group each call against the CURRENT state buffer
        // (solvers may ping-pong their state allocation between steps).
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("srd:bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_temp_vel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.b_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.b_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.b_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.b_params.as_entire_binding(),
                },
            ],
        });

        let groups = self.n_cells.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("srd:encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("srd:gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("srd:scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::mesh::{
        generate_graded_rect_mesh, generate_structured_rect_mesh, AxisGrading, BoundarySides,
    };

    #[test]
    fn structured_mesh_is_noop_none() {
        // Uniform structured grid: no slivers → operator not built.
        let mesh = generate_structured_rect_mesh(16, 16, 1.0, 1.0, BoundarySides::wall());
        assert!(
            build_srd_operator(&mesh).is_none(),
            "structured mesh must not trigger SRD"
        );
    }

    #[test]
    fn graded_mesh_is_noop_none() {
        // Strongly graded mesh (8× one-sided on both axes): the small fine-region
        // cells are legitimate, NOT slivers — SRD must not fire (else it would
        // corrupt the graded MMS order tests).
        let mesh = generate_graded_rect_mesh(
            24,
            24,
            1.0,
            1.0,
            AxisGrading::OneSided { ratio: 8.0 },
            AxisGrading::OneSided { ratio: 8.0 },
            BoundarySides::wall(),
        );
        assert!(
            build_srd_operator(&mesh).is_none(),
            "graded mesh must not trigger SRD (min/max vol = {:e}/{:e})",
            mesh.cell_vol.iter().copied().fold(f64::INFINITY, f64::min),
            mesh.cell_vol.iter().copied().fold(0.0, f64::max),
        );
    }

    // Cut-cell (sliver-bearing) tests need the meshgen geometry generators.
    #[cfg(feature = "meshgen")]
    mod cut_cell {
        use crate::solver::gpu::srd::build_srd_operator;
        use crate::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
        use nalgebra::{Point2, Vector2};

        /// The default GUI channel-with-obstacle mesh (mirrors the gate test's
        /// `channel_obstacle_mesh`).
        fn obstacle_mesh() -> Mesh {
            let length = 3.0;
            let geo = ChannelWithObstacle {
                length,
                height: 1.0,
                obstacle_center: Point2::new(1.0, 0.51),
                obstacle_radius: 0.1,
            };
            let mut mesh =
                generate_cut_cell_mesh(&geo, 0.025, 0.025, 1.2, Vector2::new(length, 1.0));
            mesh.smooth(&geo, 0.3, 100);
            mesh
        }

        #[test]
        fn obstacle_builds_operator_with_slivers() {
            let mesh = obstacle_mesh();
            let csr = build_srd_operator(&mesh).expect("obstacle mesh must yield an SRD operator");
            assert_eq!(csr.n_cells, mesh.num_cells());
            assert!(
                csr.num_small_cells > 0,
                "expected sliver neighborhoods, got {}",
                csr.num_small_cells
            );
            // Every cell has at least its identity row.
            assert!(csr.col_indices.len() >= csr.n_cells);
        }

        #[test]
        fn backstep_has_no_slivers_noop_none() {
            // The cut-cell backwards-step (the GUI compressible default geometry)
            // has no slivers — its step edges align with the grid. SRD must stay
            // inert there (also why it never redistributes the compressible
            // model's *derived* velocity field).
            use crate::solver::mesh::BackwardsStep;
            let length = 3.5;
            let geo = BackwardsStep {
                length,
                height_inlet: 0.5,
                height_outlet: 1.0,
                step_x: 0.5,
            };
            let mut mesh =
                generate_cut_cell_mesh(&geo, 0.025, 0.025, 1.2, Vector2::new(length, 1.0));
            mesh.smooth(&geo, 0.3, 50);
            assert!(
                build_srd_operator(&mesh).is_none(),
                "backstep mesh unexpectedly triggered SRD"
            );
        }

        #[test]
        fn operator_is_conservative() {
            let mesh = obstacle_mesh();
            let csr = build_srd_operator(&mesh).expect("operator");
            let err = csr.conservation_error(&mesh.cell_vol);
            assert!(err < 1e-6, "SRD must be conservative; worst rel error = {err:e}");
        }

        #[test]
        fn constant_field_is_preserved() {
            let mesh = obstacle_mesh();
            let csr = build_srd_operator(&mesh).expect("operator");
            // A spatially constant field is a fixed point of any
            // partition-of-unity operator (row sums == 1). Preservation is to
            // f32 weight precision (sliver rows have multiple f32 weights);
            // identity rows for non-sliver cells are exact.
            let mut u = vec![(1.0f64, -2.0f64); csr.n_cells];
            csr.apply_cpu(&mut u);
            for (x, y) in u {
                assert!(
                    (x - 1.0).abs() < 1e-6 && (y + 2.0).abs() < 1e-6,
                    "constant not preserved: ({x}, {y})"
                );
            }
        }
    }
}
