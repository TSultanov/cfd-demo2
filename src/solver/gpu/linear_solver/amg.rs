use crate::solver::gpu::bindings;
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
    for i in 0..fine_size {
        p.row_offsets[i] = count as u32;
        let agg = aggregates[i];
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
    for i in 0..matrix.num_cols {
        t.row_offsets[i] = count;
        count += row_counts[i];
    }
    t.row_offsets[matrix.num_cols] = count;

    // Fill
    // Re-do with vec of vecs for simplicity
    let mut rows = vec![Vec::new(); matrix.num_cols];
    for i in 0..matrix.num_rows {
        let start = matrix.row_offsets[i] as usize;
        let end = matrix.row_offsets[i + 1] as usize;
        for k in start..end {
            let j = matrix.col_indices[k] as usize;
            let val = matrix.values[k];
            rows[j].push((i as u32, val));
        }
    }

    t.col_indices.clear();
    t.values.clear();
    let mut offset = 0;
    for i in 0..matrix.num_cols {
        t.row_offsets[i] = offset;
        for (col, val) in &rows[i] {
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

        // Create Bind Group Layouts
        let bgl_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Matrix BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let bgl_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG State BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let bgl_op = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Op BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let bgl_cross = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AMG Cross Level BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
                usage: wgpu::BufferUsages::STORAGE,
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
            let bg_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("AMG L{} Matrix BG", level_idx)),
                layout: &bgl_matrix,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_matrix_row_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_matrix_col_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_matrix_values.as_entire_binding(),
                    },
                ],
            });

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

            let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("AMG L{} State BG", level_idx)),
                layout: &bgl_state,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_params.as_entire_binding(),
                    },
                ],
            });

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

            let bg_p = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("AMG L{} P BG", level_idx)),
                layout: &bgl_op,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_p_row.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_p_col.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_p_val.as_entire_binding(),
                    },
                ],
            });

            let bg_r = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("AMG L{} R BG", level_idx)),
                layout: &bgl_op,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b_r_row.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_r_col.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_r_val.as_entire_binding(),
                    },
                ],
            });

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

            let bg_restrict = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("AMG L{} Cross Restrict", i)),
                layout: &bgl_cross,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: coarse_b.as_entire_binding(),
                }],
            });

            let bg_prolongate = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("AMG L{} Cross Prolongate", i)),
                layout: &bgl_cross,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: coarse_x.as_entire_binding(),
                }],
            });

            levels[i].bg_restrict = Some(bg_restrict);
            levels[i].bg_prolongate = Some(bg_prolongate);
        }

        // Pipelines
        let shader = bindings::amg::create_shader_module_embed_source(device);

        // Layouts
        // bgl_cross moved up

        let layout_smooth = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("AMG Smooth Layout"),
            bind_group_layouts: &[&bgl_matrix, &bgl_state],
            push_constant_ranges: &[],
        });

        let layout_op = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("AMG Op Layout"),
            bind_group_layouts: &[&bgl_matrix, &bgl_state, &bgl_op, &bgl_cross],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("AMG {}", entry)),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        AmgResources {
            levels,
            pipeline_smooth: make_pipeline("smooth_op", &layout_smooth),
            pipeline_restrict_residual: make_pipeline("restrict_residual", &layout_op),
            pipeline_prolongate: make_pipeline("prolongate_op", &layout_op),
            pipeline_clear: make_pipeline("clear", &layout_smooth),
            bgl_matrix,
            bgl_op,
            bgl_state,
            bgl_cross,
        }
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

            pass.push_debug_group(&format!("AMG Down L{}->L{}", i, i + 1));

            // 1. Pre-smooth
            pass.set_pipeline(&self.pipeline_smooth);
            pass.set_bind_group(0, &fine.bg_matrix, &[]);
            pass.set_bind_group(1, state_bg(i), &[]);
            pass.dispatch_workgroups(fine_groups, 1, 1);

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
                pass.dispatch_workgroups(coarse_groups, 1, 1);
            }

            // Clear coarse solution x (ready for solve at next level)
            pass.set_pipeline(&self.pipeline_clear);
            pass.set_bind_group(0, &coarse.bg_matrix, &[]); // Dummy bind, just for layout
            pass.set_bind_group(1, state_bg(i + 1), &[]); // Binds coarse x
            pass.dispatch_workgroups(coarse_groups, 1, 1);

            pass.pop_debug_group();
        }

        // Coarsest solve
        let coarsest = &self.levels[num_levels - 1];
        let coarsest_groups = coarsest.size.div_ceil(64);
        {
            pass.push_debug_group("AMG Coarse Solve");
            pass.set_pipeline(&self.pipeline_smooth);
            pass.set_bind_group(0, &coarsest.bg_matrix, &[]);
            pass.set_bind_group(1, state_bg(num_levels - 1), &[]);
            for _ in 0..10 {
                pass.dispatch_workgroups(coarsest_groups, 1, 1);
            }
            pass.pop_debug_group();
        }

        // Upward cycle
        for i in (0..num_levels - 1).rev() {
            let fine = &self.levels[i];
            let _coarse = &self.levels[i + 1];
            let fine_groups = fine.size.div_ceil(64);

            pass.push_debug_group(&format!("AMG Up L{}->L{}", i + 1, i));

            // Prolongate (x_fine += P * x_coarse)
            if let Some(bg_cross) = &fine.bg_prolongate {
                pass.set_pipeline(&self.pipeline_prolongate);
                pass.set_bind_group(0, &fine.bg_matrix, &[]);
                pass.set_bind_group(1, state_bg(i), &[]);
                pass.set_bind_group(2, &fine.bg_p, &[]); // P op
                pass.set_bind_group(3, bg_cross, &[]); // Coarse x
                pass.dispatch_workgroups(fine_groups, 1, 1);
            }

            // Post-smooth
            pass.set_pipeline(&self.pipeline_smooth);
            pass.set_bind_group(0, &fine.bg_matrix, &[]);
            pass.set_bind_group(1, state_bg(i), &[]);
            pass.dispatch_workgroups(fine_groups, 1, 1);

            pass.pop_debug_group();
        }
    }
}
