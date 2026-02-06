use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub row_offsets: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f32>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl CsrMatrix {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            row_offsets: vec![0; num_rows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
            num_rows,
            num_cols,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        let start = self.row_offsets[row] as usize;
        let end = self.row_offsets[row + 1] as usize;
        for i in start..end {
            if self.col_indices[i] as usize == col {
                return self.values[i];
            }
        }
        0.0
    }
}

pub struct AmgLevel {
    pub size: u32,
    pub b_matrix_row_offsets: wgpu::Buffer,
    pub b_matrix_col_indices: wgpu::Buffer,
    pub b_matrix_values: wgpu::Buffer,

    // Prolongation P (Coarse -> Fine)
    // For aggregation, P is boolean/simple. We store it as CSR.
    pub b_p_row_offsets: wgpu::Buffer,
    pub b_p_col_indices: wgpu::Buffer,
    pub b_p_values: wgpu::Buffer,

    // Restriction R = P^T (Fine -> Coarse)
    pub b_r_row_offsets: wgpu::Buffer,
    pub b_r_col_indices: wgpu::Buffer,
    pub b_r_values: wgpu::Buffer,

    // State vectors
    pub b_x: wgpu::Buffer, // Solution
    pub b_b: wgpu::Buffer, // RHS
    pub b_params: wgpu::Buffer,

    // Bind groups
    pub bg_matrix: wgpu::BindGroup,
    pub bg_p: wgpu::BindGroup,
    pub bg_r: wgpu::BindGroup,
    pub bg_state: wgpu::BindGroup, // x, b, r

    // Cross-level bind groups (interaction with next coarser level)
    pub bg_restrict: Option<wgpu::BindGroup>, // Binds coarse.b_b
    pub bg_prolongate: Option<wgpu::BindGroup>, // Binds coarse.b_x
}

pub struct AmgResources {
    pub levels: Vec<AmgLevel>,
    pub pipeline_smooth: wgpu::ComputePipeline,
    pub pipeline_restrict_residual: wgpu::ComputePipeline,
    pub pipeline_prolongate: wgpu::ComputePipeline,
    pub pipeline_clear: wgpu::ComputePipeline,

    pub bgl_matrix: wgpu::BindGroupLayout,
    pub bgl_op: wgpu::BindGroupLayout, // For P and R
    pub bgl_state: wgpu::BindGroupLayout,
    pub bgl_cross: wgpu::BindGroupLayout,
    b_control_scalars: wgpu::Buffer,
    bg_cross_dummy: wgpu::BindGroup,
    bindings: &'static [wgsl_reflect::WgslBindingDesc],
}

// Simple greedy aggregation
pub fn aggregate(matrix: &CsrMatrix) -> (Vec<usize>, usize) {
    let n = matrix.num_rows;
    let mut aggregates = vec![usize::MAX; n];
    let mut num_aggregates = 0;

    // Simple greedy pass
    for i in 0..n {
        if aggregates[i] != usize::MAX {
            continue;
        }

        // Start new aggregate
        aggregates[i] = num_aggregates;

        // Add unaggregated neighbors (strong connections)
        let start = matrix.row_offsets[i] as usize;
        let end = matrix.row_offsets[i + 1] as usize;

        for k in start..end {
            let j = matrix.col_indices[k] as usize;
            if j != i && aggregates[j] == usize::MAX {
                // Check strength? For now just connectivity
                aggregates[j] = num_aggregates;
            }
        }

        num_aggregates += 1;
    }

    // Cleanup singletons (optional: merge into neighbors)

    (aggregates, num_aggregates)
}

pub fn build_prolongation(
    aggregates: &[usize],
    num_aggregates: usize,
    fine_size: usize,
) -> CsrMatrix {
    // P is fine_size x num_aggregates
    // P_ij = 1 if fine node i is in aggregate j
    let mut p = CsrMatrix::new(fine_size, num_aggregates);

    let mut count = 0;
    for (i, &agg) in aggregates.iter().take(fine_size).enumerate() {
        p.row_offsets[i] = count as u32;
        if agg < num_aggregates {
            p.col_indices.push(agg as u32);
            p.values.push(1.0);
            count += 1;
        }
    }
    p.row_offsets[fine_size] = count as u32;
    p
}

pub fn transpose(matrix: &CsrMatrix) -> CsrMatrix {
    let mut t = CsrMatrix::new(matrix.num_cols, matrix.num_rows);

    // Count entries per row in T (cols in M)
    let mut row_counts = vec![0; matrix.num_cols];
    for col in &matrix.col_indices {
        row_counts[*col as usize] += 1;
    }

    // Build row offsets
    let mut count = 0;
    for (i, &row_count) in row_counts.iter().enumerate() {
        t.row_offsets[i] = count;
        count += row_count;
    }
    t.row_offsets[matrix.num_cols] = count;

    // Fill
    // Re-do with vec of vecs for simplicity
    let mut rows = vec![Vec::new(); matrix.num_cols];
    for (i, (&start, &end)) in matrix.row_offsets[..matrix.num_rows]
        .iter()
        .zip(&matrix.row_offsets[1..])
        .enumerate()
    {
        let start = start as usize;
        let end = end as usize;
        for k in start..end {
            let j = matrix.col_indices[k] as usize;
            let val = matrix.values[k];
            rows[j].push((i as u32, val));
        }
    }

    t.col_indices.clear();
    t.values.clear();
    let mut offset = 0;
    for (i, row) in rows.iter().enumerate() {
        t.row_offsets[i] = offset;
        for (col, val) in row {
            t.col_indices.push(*col);
            t.values.push(*val);
            offset += 1;
        }
    }
    t.row_offsets[matrix.num_cols] = offset;

    t
}

pub fn mat_mat_mult(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    // C = A * B
    let mut c = CsrMatrix::new(a.num_rows, b.num_cols);

    let mut offset = 0;
    for i in 0..a.num_rows {
        c.row_offsets[i] = offset;

        // Row i of A
        let start_a = a.row_offsets[i] as usize;
        let end_a = a.row_offsets[i + 1] as usize;

        let mut row_vals: HashMap<u32, f32> = HashMap::new();

        for k_a in start_a..end_a {
            let j = a.col_indices[k_a] as usize;
            let val_a = a.values[k_a];

            // Row j of B
            let start_b = b.row_offsets[j] as usize;
            let end_b = b.row_offsets[j + 1] as usize;

            for k_b in start_b..end_b {
                let k = b.col_indices[k_b];
                let val_b = b.values[k_b];

                *row_vals.entry(k).or_insert(0.0) += val_a * val_b;
            }
        }

        // Sort by column index
        let mut sorted_cols: Vec<_> = row_vals.into_iter().collect();
        sorted_cols.sort_by_key(|k| k.0);

        for (col, val) in sorted_cols {
            c.col_indices.push(col);
            c.values.push(val);
            offset += 1;
        }
    }
    c.row_offsets[a.num_rows] = offset;
    c
}

pub fn galerkin_product(r: &CsrMatrix, a: &CsrMatrix, p: &CsrMatrix) -> CsrMatrix {
    // R * A * P
    let ra = mat_mat_mult(r, a);
    mat_mat_mult(&ra, p)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AmgParams {
    pub n: u32,
    pub omega: f32,
    pub padding: [u32; 2],
}

impl AmgResources {
    pub fn new(device: &wgpu::Device, fine_matrix: &CsrMatrix, max_levels: usize) -> Self {
        let mut levels = Vec::new();
        let mut current_matrix = fine_matrix.clone();

        let smooth_src = kernel_registry::kernel_source_by_id("", KernelId::AMG_SMOOTH_OP)
            .expect("amg/smooth_op shader missing from kernel registry");
        let restrict_src =
            kernel_registry::kernel_source_by_id("", KernelId::AMG_RESTRICT_RESIDUAL)
                .expect("amg/restrict_residual shader missing from kernel registry");
        let bindings = restrict_src.bindings;
        let prolongate_src = kernel_registry::kernel_source_by_id("", KernelId::AMG_PROLONGATE_OP)
            .expect("amg/prolongate_op shader missing from kernel registry");
        let clear_src = kernel_registry::kernel_source_by_id("", KernelId::AMG_CLEAR)
            .expect("amg/clear shader missing from kernel registry");

        let pipeline_smooth = (smooth_src.create_pipeline)(device);
        let pipeline_restrict_residual = (restrict_src.create_pipeline)(device);
        let pipeline_prolongate = (prolongate_src.create_pipeline)(device);
        let pipeline_clear = (clear_src.create_pipeline)(device);

        let bgl_matrix = pipeline_restrict_residual.get_bind_group_layout(0);
        let bgl_state = pipeline_restrict_residual.get_bind_group_layout(1);
        let bgl_op = pipeline_restrict_residual.get_bind_group_layout(2);
        let bgl_cross = pipeline_restrict_residual.get_bind_group_layout(3);

        let b_control_scalars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AMG Control Scalars"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Build Hierarchy
        for level_idx in 0..max_levels {
            let n = current_matrix.num_rows;

            // Create buffers for matrix
            let b_matrix_row_offsets =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("AMG L{} Matrix Row Offsets", level_idx)),
                    contents: bytemuck::cast_slice(&current_matrix.row_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let b_matrix_col_indices =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("AMG L{} Matrix Col Indices", level_idx)),
                    contents: bytemuck::cast_slice(&current_matrix.col_indices),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let b_matrix_values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("AMG L{} Matrix Values", level_idx)),
                contents: bytemuck::cast_slice(&current_matrix.values),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            // Create state buffers
            let b_x = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("AMG L{} x", level_idx)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let b_b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("AMG L{} b", level_idx)),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Bind Groups
            let bg_matrix = {
                let registry = ResourceRegistry::new()
                    .with_buffer("row_offsets", &b_matrix_row_offsets)
                    .with_buffer("col_indices", &b_matrix_col_indices)
                    .with_buffer("values", &b_matrix_values);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("AMG L{level_idx} Matrix BG"),
                    &bgl_matrix,
                    bindings,
                    0,
                    |name| registry.resolve(name),
                )
                .unwrap_or_else(|err| panic!("AMG L{level_idx} matrix BG creation failed: {err}"))
            };

            // Params buffer
            let params = AmgParams {
                n: n as u32,
                omega: 0.8,
                padding: [0; 2],
            };
            let b_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("AMG L{} Params", level_idx)),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bg_state = {
                let registry = ResourceRegistry::new()
                    .with_buffer("x", &b_x)
                    .with_buffer("b", &b_b)
                    .with_buffer("params", &b_params);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("AMG L{level_idx} State BG"),
                    &bgl_state,
                    bindings,
                    1,
                    |name| registry.resolve(name),
                )
                .unwrap_or_else(|err| panic!("AMG L{level_idx} state BG creation failed: {err}"))
            };

            // Coarsening (if not last level)
            let (p, r) = if level_idx < max_levels - 1 && n > 100 {
                let (aggregates, num_aggs) = aggregate(&current_matrix);
                if num_aggs < n {
                    let p_mat = build_prolongation(&aggregates, num_aggs, n);
                    let r_mat = transpose(&p_mat);

                    // Galerkin product for next level
                    let next_matrix = galerkin_product(&r_mat, &current_matrix, &p_mat);
                    current_matrix = next_matrix;

                    (Some(p_mat), Some(r_mat))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            // Create P and R buffers (dummy if None)
            let create_csr_buffers = |mat: Option<&CsrMatrix>| {
                if let Some(m) = mat {
                    (
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("AMG Op Row"),
                            contents: bytemuck::cast_slice(&m.row_offsets),
                            usage: wgpu::BufferUsages::STORAGE,
                        }),
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("AMG Op Col"),
                            contents: bytemuck::cast_slice(&m.col_indices),
                            usage: wgpu::BufferUsages::STORAGE,
                        }),
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("AMG Op Val"),
                            contents: bytemuck::cast_slice(&m.values),
                            usage: wgpu::BufferUsages::STORAGE,
                        }),
                    )
                } else {
                    // Dummy buffers
                    (
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: None,
                            size: 4,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        }),
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: None,
                            size: 4,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        }),
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: None,
                            size: 4,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        }),
                    )
                }
            };

            let (b_p_row, b_p_col, b_p_val) = create_csr_buffers(p.as_ref());
            let (b_r_row, b_r_col, b_r_val) = create_csr_buffers(r.as_ref());

            let bg_p = {
                let registry = ResourceRegistry::new()
                    .with_buffer("op_row_offsets", &b_p_row)
                    .with_buffer("op_col_indices", &b_p_col)
                    .with_buffer("op_values", &b_p_val);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("AMG L{level_idx} P BG"),
                    &bgl_op,
                    bindings,
                    2,
                    |name| registry.resolve(name),
                )
                .unwrap_or_else(|err| panic!("AMG L{level_idx} P BG creation failed: {err}"))
            };

            let bg_r = {
                let registry = ResourceRegistry::new()
                    .with_buffer("op_row_offsets", &b_r_row)
                    .with_buffer("op_col_indices", &b_r_col)
                    .with_buffer("op_values", &b_r_val);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("AMG L{level_idx} R BG"),
                    &bgl_op,
                    bindings,
                    2,
                    |name| registry.resolve(name),
                )
                .unwrap_or_else(|err| panic!("AMG L{level_idx} R BG creation failed: {err}"))
            };

            levels.push(AmgLevel {
                size: n as u32,
                b_matrix_row_offsets,
                b_matrix_col_indices,
                b_matrix_values,
                b_p_row_offsets: b_p_row,
                b_p_col_indices: b_p_col,
                b_p_values: b_p_val,
                b_r_row_offsets: b_r_row,
                b_r_col_indices: b_r_col,
                b_r_values: b_r_val,
                b_x,
                b_b,
                b_params,
                bg_matrix,
                bg_p,
                bg_r,
                bg_state,
                bg_restrict: None,
                bg_prolongate: None,
            });

            if p.is_none() {
                break;
            }
        }

        // Create cross-level bind groups
        for i in 0..levels.len() - 1 {
            let coarse_b = &levels[i + 1].b_b;
            let coarse_x = &levels[i + 1].b_x;

            let bg_restrict = {
                let registry = ResourceRegistry::new()
                    .with_buffer("coarse_vec", coarse_b)
                    .with_buffer("scalars", &b_control_scalars);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("AMG L{i} Cross Restrict"),
                    &bgl_cross,
                    bindings,
                    3,
                    |name| registry.resolve(name),
                )
                .unwrap_or_else(|err| panic!("AMG L{i} cross restrict BG creation failed: {err}"))
            };

            let bg_prolongate = {
                let registry = ResourceRegistry::new()
                    .with_buffer("coarse_vec", coarse_x)
                    .with_buffer("scalars", &b_control_scalars);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("AMG L{i} Cross Prolongate"),
                    &bgl_cross,
                    bindings,
                    3,
                    |name| registry.resolve(name),
                )
                .unwrap_or_else(|err| panic!("AMG L{i} cross prolongate BG creation failed: {err}"))
            };

            levels[i].bg_restrict = Some(bg_restrict);
            levels[i].bg_prolongate = Some(bg_prolongate);
        }

        let bg_cross_dummy = {
            let coarsest = levels
                .last()
                .expect("AMG hierarchy must contain at least one level");
            let registry = ResourceRegistry::new()
                .with_buffer("coarse_vec", &coarsest.b_x)
                .with_buffer("scalars", &b_control_scalars);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "AMG Cross Dummy BG",
                &bgl_cross,
                bindings,
                3,
                |name| registry.resolve(name),
            )
            .unwrap_or_else(|err| panic!("AMG cross dummy BG creation failed: {err}"))
        };

        AmgResources {
            levels,
            pipeline_smooth,
            pipeline_restrict_residual,
            pipeline_prolongate,
            pipeline_clear,
            bgl_matrix,
            bgl_op,
            bgl_state,
            bgl_cross,
            b_control_scalars,
            bg_cross_dummy,
            bindings,
        }
    }

    pub(crate) fn sync_control_scalars(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scalars: &wgpu::Buffer,
    ) {
        encoder.copy_buffer_to_buffer(scalars, 0, &self.b_control_scalars, 0, 64);
    }

    pub(crate) fn create_state_override_bind_group(
        &self,
        device: &wgpu::Device,
        x: &wgpu::Buffer,
        b: &wgpu::Buffer,
        params: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        let registry = ResourceRegistry::new()
            .with_buffer("x", x)
            .with_buffer("b", b)
            .with_buffer("params", params);
        wgsl_reflect::create_bind_group_from_bindings(
            device,
            label,
            &self.bgl_state,
            self.bindings,
            1,
            |name| registry.resolve(name),
        )
        .unwrap_or_else(|err| panic!("{label} creation failed: {err}"))
    }

    pub fn v_cycle(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level0_state_override: Option<&wgpu::BindGroup>,
    ) {
        let num_levels = self.levels.len();
        if num_levels == 0 {
            return;
        }

        let level0_state = level0_state_override.unwrap_or(&self.levels[0].bg_state);
        let state_bg = |idx: usize| -> &wgpu::BindGroup {
            if idx == 0 {
                level0_state
            } else {
                &self.levels[idx].bg_state
            }
        };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("AMG V-Cycle"),
            timestamp_writes: None,
        });

        // Downward cycle
        for i in 0..num_levels - 1 {
            let fine = &self.levels[i];
            let coarse = &self.levels[i + 1];

            let fine_groups = fine.size.div_ceil(64);
            let coarse_groups = coarse.size.div_ceil(64);
            let (fine_dispatch_x, fine_dispatch_y) = dispatch_2d(fine_groups);
            let (coarse_dispatch_x, coarse_dispatch_y) = dispatch_2d(coarse_groups);

            pass.push_debug_group(&format!("AMG Down L{}->L{}", i, i + 1));

            // 1. Pre-smooth
            pass.set_pipeline(&self.pipeline_smooth);
            pass.set_bind_group(0, &fine.bg_matrix, &[]);
            pass.set_bind_group(1, state_bg(i), &[]);
            pass.set_bind_group(2, &fine.bg_r, &[]);
            pass.set_bind_group(
                3,
                fine.bg_restrict.as_ref().unwrap_or(&self.bg_cross_dummy),
                &[],
            );
            pass.dispatch_workgroups(fine_dispatch_x, fine_dispatch_y, 1);

            // 2. Fused Residual + Restrict (r_fine implicit, r_coarse = R * (b - Ax))
            if let Some(bg_cross) = &fine.bg_restrict {
                pass.set_pipeline(&self.pipeline_restrict_residual);
                // Bind groups used:
                // 0: Matrix (Fine)
                // 1: State (Fine x, b)
                // 2: Op (R)
                // 3: Cross (Coarse b)
                pass.set_bind_group(0, &fine.bg_matrix, &[]);
                pass.set_bind_group(1, state_bg(i), &[]);
                pass.set_bind_group(2, &fine.bg_r, &[]); // bg_r contains R op buffers
                pass.set_bind_group(3, bg_cross, &[]);
                pass.dispatch_workgroups(coarse_dispatch_x, coarse_dispatch_y, 1);
            }

            // Clear coarse solution x (ready for solve at next level)
            pass.set_pipeline(&self.pipeline_clear);
            pass.set_bind_group(0, &coarse.bg_matrix, &[]); // Dummy bind, just for layout
            pass.set_bind_group(1, state_bg(i + 1), &[]); // Binds coarse x
            pass.set_bind_group(2, &coarse.bg_r, &[]);
            pass.set_bind_group(
                3,
                coarse.bg_restrict.as_ref().unwrap_or(&self.bg_cross_dummy),
                &[],
            );
            pass.dispatch_workgroups(coarse_dispatch_x, coarse_dispatch_y, 1);

            pass.pop_debug_group();
        }

        // Coarsest solve
        let coarsest = &self.levels[num_levels - 1];
        let coarsest_groups = coarsest.size.div_ceil(64);
        let (coarsest_dispatch_x, coarsest_dispatch_y) = dispatch_2d(coarsest_groups);
        {
            pass.push_debug_group("AMG Coarse Solve");
            pass.set_pipeline(&self.pipeline_smooth);
            pass.set_bind_group(0, &coarsest.bg_matrix, &[]);
            pass.set_bind_group(1, state_bg(num_levels - 1), &[]);
            pass.set_bind_group(2, &coarsest.bg_r, &[]);
            pass.set_bind_group(3, &self.bg_cross_dummy, &[]);
            for _ in 0..10 {
                pass.dispatch_workgroups(coarsest_dispatch_x, coarsest_dispatch_y, 1);
            }
            pass.pop_debug_group();
        }

        // Upward cycle
        for i in (0..num_levels - 1).rev() {
            let fine = &self.levels[i];
            let _coarse = &self.levels[i + 1];
            let fine_groups = fine.size.div_ceil(64);
            let (fine_dispatch_x, fine_dispatch_y) = dispatch_2d(fine_groups);

            pass.push_debug_group(&format!("AMG Up L{}->L{}", i + 1, i));

            // Prolongate (x_fine += P * x_coarse)
            if let Some(bg_cross) = &fine.bg_prolongate {
                pass.set_pipeline(&self.pipeline_prolongate);
                pass.set_bind_group(0, &fine.bg_matrix, &[]);
                pass.set_bind_group(1, state_bg(i), &[]);
                pass.set_bind_group(2, &fine.bg_p, &[]); // P op
                pass.set_bind_group(3, bg_cross, &[]); // Coarse x
                pass.dispatch_workgroups(fine_dispatch_x, fine_dispatch_y, 1);
            }

            // Post-smooth
            pass.set_pipeline(&self.pipeline_smooth);
            pass.set_bind_group(0, &fine.bg_matrix, &[]);
            pass.set_bind_group(1, state_bg(i), &[]);
            pass.set_bind_group(2, &fine.bg_p, &[]);
            pass.set_bind_group(
                3,
                fine.bg_prolongate.as_ref().unwrap_or(&self.bg_cross_dummy),
                &[],
            );
            pass.dispatch_workgroups(fine_dispatch_x, fine_dispatch_y, 1);

            pass.pop_debug_group();
        }
    }
}
