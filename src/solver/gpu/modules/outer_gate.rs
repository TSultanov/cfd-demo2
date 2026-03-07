//! GPU-side adaptive outer break gate for one-submission batched outer loops.
//!
//! Provides the gate kernel (which conditionally zeroes indirect dispatch args
//! when the solver has converged) and the STOP-inject kernels (which write a
//! convergence flag into the FGMRES/CG solver scalars buffer so subsequent
//! linear solver iterations become near-zero-cost).

use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::model::KernelId;

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

        // Gate pipeline (pre-compiled infrastructure kernel)
        let gate_src = kernel_registry::kernel_source_by_id("", KernelId::OUTER_GATE)
            .expect("missing outer_gate infrastructure kernel");
        let gate_pipeline = (gate_src.create_pipeline)(device);
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

        // STOP-inject pipeline for FGMRES (pre-compiled infrastructure kernel)
        let stop_inject_src = kernel_registry::kernel_source_by_id(
            "",
            KernelId::OUTER_STOP_INJECT_FGMRES,
        )
        .expect("missing outer_stop_inject_fgmres infrastructure kernel");
        let stop_inject_pipeline = (stop_inject_src.create_pipeline)(device);

        // STOP-inject pipeline for CG (pre-compiled infrastructure kernel)
        let cg_stop_inject_src =
            kernel_registry::kernel_source_by_id("", KernelId::OUTER_STOP_INJECT_CG)
                .expect("missing outer_stop_inject_cg infrastructure kernel");
        let cg_stop_inject_pipeline = (cg_stop_inject_src.create_pipeline)(device);

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
