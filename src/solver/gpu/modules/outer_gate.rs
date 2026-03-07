//! GPU-side adaptive outer break gate for one-submission batched outer loops.
//!
//! Provides the gate kernel (which conditionally zeroes indirect dispatch args
//! when the solver has converged) and the STOP-inject kernels (which write a
//! convergence flag into the FGMRES/CG solver scalars buffer so subsequent
//! linear solver iterations become near-zero-cost).

use crate::solver::gpu::linear_solver::fgmres::FGMRES_SCALAR_STOP;
use crate::solver::gpu::modules::scalar_cg::CG_SCALAR_STOP;
use cfd2_codegen::solver::codegen::wgsl_ast::*;
use cfd2_codegen::solver::codegen::wgsl_dsl::*;

/// Builds the outer gate kernel WGSL via the structured DSL.
///
/// Single-thread kernel that reads `break_status[0]`:
/// - If NOT converged (0): copies real dispatch args to indirect args for cells/faces,
///   and atomically increments the iteration counter.
/// - If converged (1): writes zeros to indirect args (zero-cost dispatch).
fn build_outer_gate_wgsl() -> String {
    let mut m = Module::new();

    m.push(Item::GlobalVar(GlobalVar::new(
        "break_status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "real_args_cells",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args_cells",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "real_args_faces",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args_faces",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(4)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "iter_counter",
        Type::array(Type::atomic(Type::U32)),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(5)],
    )));

    let global_id = Expr::ident("global_id");
    let break_status = Expr::ident("break_status");
    let real_args_cells = Expr::ident("real_args_cells");
    let indirect_args_cells = Expr::ident("indirect_args_cells");
    let real_args_faces = Expr::ident("real_args_faces");
    let indirect_args_faces = Expr::ident("indirect_args_faces");
    let iter_counter = Expr::ident("iter_counter");
    let converged = Expr::ident("converged");

    let body = block(vec![
        // if (global_id.x != 0u) { return; }
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        // let converged = break_status[0];
        let_expr("converged", break_status.clone().index(Expr::lit_u32(0))),
        // if (converged == 0u) { ... } else { ... }
        if_block_expr(
            converged.clone().eq(Expr::lit_u32(0)),
            block(vec![
                // Not converged: enable dispatches with real args
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(0)),
                    real_args_cells.clone().index(Expr::lit_u32(0)),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(1)),
                    real_args_cells.clone().index(Expr::lit_u32(1)),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(2)),
                    real_args_cells.clone().index(Expr::lit_u32(2)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(0)),
                    real_args_faces.clone().index(Expr::lit_u32(0)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(1)),
                    real_args_faces.clone().index(Expr::lit_u32(1)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(2)),
                    real_args_faces.clone().index(Expr::lit_u32(2)),
                ),
                // Increment iteration counter
                call_stmt_expr(atomic_add(
                    iter_counter.clone().index(Expr::lit_u32(0)).addr_of(),
                    Expr::lit_u32(1),
                )),
            ]),
            Some(block(vec![
                // Converged: zero out dispatches
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(0)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(1)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(2)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(0)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(1)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(2)),
                    Expr::lit_u32(0),
                ),
            ])),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    m.to_wgsl()
}

/// Builds the STOP-inject kernel WGSL via the structured DSL.
///
/// Single-thread kernel that reads `break_status[0]` and writes `f32(break_status[0])`
/// into the scalars buffer at the given `scalar_stop` index.
fn build_outer_stop_inject_wgsl(scalar_stop: usize) -> String {
    let mut m = Module::new();

    m.push(Item::GlobalVar(GlobalVar::new(
        "break_status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));

    let global_id = Expr::ident("global_id");
    let break_status = Expr::ident("break_status");
    let scalars = Expr::ident("scalars");

    let body = block(vec![
        // if (global_id.x != 0u) { return; }
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        // scalars[SCALAR_STOP] = f32(break_status[0]);
        assign_expr(
            scalars.clone().index(Expr::lit_u32(scalar_stop as u32)),
            f32_cast(break_status.clone().index(Expr::lit_u32(0))),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    m.to_wgsl()
}

/// GPU-side adaptive outer break gate resources.
///
/// This struct holds the indirect dispatch argument buffers, the gate kernel
/// pipeline, and the iteration counter. It is created when one-submission mode
/// with adaptive break is enabled.
pub(crate) struct OuterAdaptiveGate {
    /// Gate kernel pipeline (reads break_status, writes indirect args / counter)
    gate_pipeline: wgpu::ComputePipeline,
    gate_bg: wgpu::BindGroup,
    /// STOP-inject kernel pipeline (writes break_status into FGMRES scalars[SCALAR_STOP])
    stop_inject_pipeline: wgpu::ComputePipeline,
    /// STOP-inject kernel pipeline for CG (writes break_status into CG scalars[CG_SCALAR_STOP])
    cg_stop_inject_pipeline: wgpu::ComputePipeline,
    /// Indirect dispatch args for Cells-dispatched kernels (3 × u32)
    pub(crate) b_indirect_args_cells: std::sync::Arc<wgpu::Buffer>,
    /// Indirect dispatch args for Faces-dispatched kernels (3 × u32)
    pub(crate) b_indirect_args_faces: std::sync::Arc<wgpu::Buffer>,
    /// GPU-side iteration counter (atomically incremented by gate kernel)
    pub(crate) b_iter_counter: std::sync::Arc<wgpu::Buffer>,
}

impl OuterAdaptiveGate {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_cells: u32,
        num_faces: u32,
        max_workgroups_per_dim: u32,
        b_break_status: &wgpu::Buffer,
    ) -> Self {
        // Compute real dispatch args using the same formula as GeneratedKernelsModule
        const WORKGROUP_SIZE_X: u32 = 64;
        let compute_dispatch = |items: u32| -> [u32; 3] {
            let groups = items.div_ceil(WORKGROUP_SIZE_X);
            if groups <= max_workgroups_per_dim {
                [groups.max(1), 1, 1]
            } else {
                let x = max_workgroups_per_dim;
                let y = groups.div_ceil(x);
                [x, y, 1]
            }
        };
        let real_cells = compute_dispatch(num_cells);
        let real_faces = compute_dispatch(num_faces);

        let b_real_args_cells = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_gate:real_args_cells"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_real_args_cells, 0, bytemuck::cast_slice(&real_cells));

        let b_real_args_faces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_gate:real_args_faces"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_real_args_faces, 0, bytemuck::cast_slice(&real_faces));

        let b_indirect_args_cells =
            std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("outer_gate:indirect_args_cells"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }));

        let b_indirect_args_faces =
            std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("outer_gate:indirect_args_faces"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }));

        let b_iter_counter = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_gate:iter_counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Gate pipeline
        let gate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_gate:gate"),
            source: wgpu::ShaderSource::Wgsl(build_outer_gate_wgsl().into()),
        });
        let gate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("outer_gate:gate"),
            layout: None,
            module: &gate_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let gate_bgl = gate_pipeline.get_bind_group_layout(0);
        let gate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_gate:gate_bg"),
            layout: &gate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_real_args_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_indirect_args_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_real_args_faces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_indirect_args_faces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_iter_counter.as_entire_binding(),
                },
            ],
        });

        // STOP-inject pipeline (bind group created per-FGMRES workspace)
        let stop_inject_src = build_outer_stop_inject_wgsl(FGMRES_SCALAR_STOP);
        let stop_inject_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_gate:stop_inject"),
            source: wgpu::ShaderSource::Wgsl(stop_inject_src.into()),
        });
        let stop_inject_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("outer_gate:stop_inject"),
                layout: None,
                module: &stop_inject_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // STOP-inject pipeline for CG (same shader, different scalar offset)
        let cg_stop_inject_src = build_outer_stop_inject_wgsl(CG_SCALAR_STOP);
        let cg_stop_inject_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_gate:cg_stop_inject"),
            source: wgpu::ShaderSource::Wgsl(cg_stop_inject_src.into()),
        });
        let cg_stop_inject_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("outer_gate:cg_stop_inject"),
                layout: None,
                module: &cg_stop_inject_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            gate_pipeline,
            gate_bg,
            stop_inject_pipeline,
            cg_stop_inject_pipeline,
            b_indirect_args_cells,
            b_indirect_args_faces,
            b_iter_counter,
        }
    }

    /// Create a bind group for the STOP-inject kernel with a specific FGMRES scalars buffer.
    pub(crate) fn create_stop_inject_bind_group(
        &self,
        device: &wgpu::Device,
        b_break_status: &wgpu::Buffer,
        b_scalars: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bgl = self.stop_inject_pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_gate:stop_inject_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scalars.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode the gate kernel dispatch into a command encoder.
    pub(crate) fn encode_gate_into(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("outer_gate:gate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.gate_pipeline);
        pass.set_bind_group(0, &self.gate_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Encode the STOP-inject kernel dispatch into a command encoder.
    pub(crate) fn encode_stop_inject_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stop_inject_bg: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("outer_gate:stop_inject"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.stop_inject_pipeline);
        pass.set_bind_group(0, stop_inject_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Create a bind group for the CG STOP-inject kernel with a specific CG scalars buffer.
    pub(crate) fn create_stop_inject_bind_group_cg(
        &self,
        device: &wgpu::Device,
        b_break_status: &wgpu::Buffer,
        b_scalars: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bgl = self.cg_stop_inject_pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_gate:cg_stop_inject_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scalars.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode the CG STOP-inject kernel dispatch into a command encoder.
    pub(crate) fn encode_stop_inject_cg_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stop_inject_bg: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("outer_gate:cg_stop_inject"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.cg_stop_inject_pipeline);
        pass.set_bind_group(0, stop_inject_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
