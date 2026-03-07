//! Outer convergence monitoring for coupled solver outer iterations.
//!
//! Provides GPU-side reduction kernels that compute per-field correction norms
//! (delta maxima) and state scale values, then evaluate a convergence break
//! criterion.  Used by both host-driven and one-submission batched outer loops.

use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::program::plan::GpuProgramPlan;
use crate::solver::model::{KernelId, ModelSpec};
use bytemuck::{bytes_of, Pod, Zeroable};

use super::super::lowering::programs::generic_coupled::ResolvedUnknownMapping;

pub(crate) const OUTER_CONVERGENCE_WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOuterConvergenceParams {
    num_cells: u32,
    stride: u32,
    num_targets: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOuterConvergenceTargetDesc {
    offsets: [u32; 4],
    num_comps: u32,
    _pad0: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOuterConvergenceBreakParams {
    count: u32,
    tol_rel: f32,
    tol_abs: f32,
    _pad0: u32,
}

/// Builds the outer convergence break kernel WGSL via the structured DSL.
///
pub(crate) struct OuterConvergenceMonitor {
    target_names: Vec<String>,
    pipeline: wgpu::ComputePipeline,
    break_pipeline: wgpu::ComputePipeline,
    _b_params_x: wgpu::Buffer,
    b_params_state: wgpu::Buffer,
    _b_descs_x: wgpu::Buffer,
    b_descs_state: wgpu::Buffer,
    b_out_bits: wgpu::Buffer,
    pub(crate) b_delta: wgpu::Buffer,
    b_scale: wgpu::Buffer,
    pub(crate) b_break_status: wgpu::Buffer,
    b_break_params: wgpu::Buffer,
    bg_x: wgpu::BindGroup,
    break_bg: wgpu::BindGroup,
    zero_out_words: Vec<u32>,
    dispatch_cells: u32,
    state_scale: Option<Vec<f32>>,
    state_scale_ready: bool,
}

impl OuterConvergenceMonitor {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &ModelSpec,
        num_cells: u32,
        x: &wgpu::Buffer,
        unknown_mapping: &ResolvedUnknownMapping,
    ) -> Result<Option<Self>, String> {
        let stride_x = model.system.unknowns_per_cell();
        let stride_state = model.state_layout.stride();
        if stride_x == 0 || stride_state == 0 || num_cells == 0 {
            return Ok(None);
        }

        let mut target_names: Vec<String> = Vec::new();
        let mut target_descs_x: Vec<GpuOuterConvergenceTargetDesc> = Vec::new();
        let mut target_descs_state: Vec<GpuOuterConvergenceTargetDesc> = Vec::new();

        let mut unknown_offset_cursor: u32 = 0;
        for (eq_idx, eqn) in model.system.equations().iter().enumerate() {
            let target = eqn.target();
            let name = target.name();

            let kind = target.kind();
            let comps = kind.component_count();
            if comps == 0 {
                continue;
            }
            if comps > 4 {
                return Err(format!(
                    "outer convergence monitor only supports up to 4 components per target (got {comps} for '{name}')"
                ));
            }

            let mut offsets_x = [0u32; 4];
            for (comp, offset) in offsets_x.iter_mut().enumerate().take(comps) {
                *offset = unknown_offset_cursor + comp as u32;
            }
            unknown_offset_cursor += comps as u32;

            // Get state offsets from the pre-resolved mapping
            let mut offsets_state = [0u32; 4];
            let mut has_all_offsets = true;
            for (comp, offset) in offsets_state.iter_mut().enumerate().take(comps) {
                match unknown_mapping.get_offset(eq_idx, comp) {
                    Some(off) => *offset = off,
                    None => {
                        has_all_offsets = false;
                        break;
                    }
                }
            }

            if !has_all_offsets {
                continue;
            }

            target_names.push(name.to_string());
            target_descs_x.push(GpuOuterConvergenceTargetDesc {
                offsets: offsets_x,
                num_comps: comps as u32,
                _pad0: [0u32; 3],
            });
            target_descs_state.push(GpuOuterConvergenceTargetDesc {
                offsets: offsets_state,
                num_comps: comps as u32,
                _pad0: [0u32; 3],
            });
        }

        let num_targets = target_descs_x.len() as u32;
        if num_targets == 0 {
            return Ok(None);
        }
        if target_descs_state.len() != target_descs_x.len() {
            return Err(format!(
                "outer convergence monitor target count mismatch: x_descs={} state_descs={}",
                target_descs_x.len(),
                target_descs_state.len()
            ));
        }

        let pipeline = {
            let src = kernel_registry::kernel_source_by_id(
                "",
                crate::solver::model::KernelId::OUTER_CONVERGENCE,
            )?;
            (src.create_pipeline)(device)
        };
        let bgl = pipeline.get_bind_group_layout(0);
        let break_src = kernel_registry::kernel_source_by_id(
            "",
            KernelId::OUTER_CONVERGENCE_BREAK,
        )
        .map_err(|e| format!("missing outer_convergence_break infrastructure kernel: {e}"))?;
        let break_pipeline = (break_src.create_pipeline)(device);

        let b_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:params_x"),
            size: std::mem::size_of::<GpuOuterConvergenceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_x = GpuOuterConvergenceParams {
            num_cells,
            stride: stride_x,
            num_targets,
            _pad0: 0,
        };
        queue.write_buffer(&b_params, 0, bytemuck::bytes_of(&params_x));

        let b_params_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:params_state"),
            size: std::mem::size_of::<GpuOuterConvergenceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_state = GpuOuterConvergenceParams {
            num_cells,
            stride: stride_state,
            num_targets,
            _pad0: 0,
        };
        queue.write_buffer(&b_params_state, 0, bytemuck::bytes_of(&params_state));

        let b_descs_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:target_descs_x"),
            size: (target_descs_x.len() as u64)
                * std::mem::size_of::<GpuOuterConvergenceTargetDesc>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_descs_x, 0, bytemuck::cast_slice(&target_descs_x));

        let b_descs_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:target_descs_state"),
            size: (target_descs_state.len() as u64)
                * std::mem::size_of::<GpuOuterConvergenceTargetDesc>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &b_descs_state,
            0,
            bytemuck::cast_slice(&target_descs_state),
        );

        let b_out_bits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:out_bits"),
            size: (num_targets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_delta = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:delta"),
            size: (num_targets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_scale = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:scale"),
            size: (num_targets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_break_status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:break_status"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_break_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:break_params"),
            size: std::mem::size_of::<GpuOuterConvergenceBreakParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_x = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_convergence:bg_x"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_descs_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_out_bits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_params.as_entire_binding(),
                },
            ],
        });
        let break_bgl = break_pipeline.get_bind_group_layout(0);
        let break_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_convergence:break_bg"),
            layout: &break_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scale.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_break_params.as_entire_binding(),
                },
            ],
        });

        let zero_out_words = vec![0u32; target_descs_x.len()];
        let dispatch_cells = num_cells.div_ceil(OUTER_CONVERGENCE_WORKGROUP_SIZE);

        Ok(Some(Self {
            target_names,
            pipeline,
            break_pipeline,
            _b_params_x: b_params,
            b_params_state,
            _b_descs_x: b_descs_x,
            b_descs_state,
            b_out_bits,
            b_delta,
            b_scale,
            b_break_status,
            b_break_params,
            bg_x,
            break_bg,
            zero_out_words,
            dispatch_cells,
            state_scale: None,
            state_scale_ready: false,
        }))
    }

    pub(crate) fn reset_step(&mut self) {
        self.state_scale = None;
        self.state_scale_ready = false;
    }

    pub(crate) fn ensure_state_scale(
        &mut self,
        plan: &GpuProgramPlan,
        state: &wgpu::Buffer,
    ) -> Result<(), String> {
        if self.state_scale.is_some() {
            return Ok(());
        }
        let scale = self.compute_maxima_with_params_and_descs(
            plan,
            state,
            &self.b_descs_state,
            &self.b_params_state,
            "outer_convergence:state",
        )?;
        if !scale.is_empty() {
            plan.context
                .queue
                .write_buffer(&self.b_scale, 0, bytemuck::cast_slice(&scale));
        }
        self.state_scale = Some(scale);
        self.state_scale_ready = true;
        Ok(())
    }

    pub(crate) fn delta_maxima(&self, plan: &GpuProgramPlan) -> Result<Vec<f32>, String> {
        self.compute_maxima_from_bind_group(plan, &self.bg_x, "outer_convergence:delta")
    }

    fn compute_maxima_with_params_and_descs(
        &self,
        plan: &GpuProgramPlan,
        input: &wgpu::Buffer,
        descs: &wgpu::Buffer,
        params: &wgpu::Buffer,
        label_prefix: &'static str,
    ) -> Result<Vec<f32>, String> {
        let bgl = self.pipeline.get_bind_group_layout(0);
        let bg = plan
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label_prefix),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.b_out_bits.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        self.compute_maxima_from_bind_group(plan, &bg, label_prefix)
    }

    fn compute_maxima_from_bind_group(
        &self,
        plan: &GpuProgramPlan,
        bind_group: &wgpu::BindGroup,
        label_prefix: &'static str,
    ) -> Result<Vec<f32>, String> {
        if self.zero_out_words.is_empty() {
            return Ok(Vec::new());
        }

        plan.context.queue.write_buffer(
            &self.b_out_bits,
            0,
            bytemuck::cast_slice(&self.zero_out_words),
        );

        let mut encoder =
            plan.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(label_prefix),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label_prefix),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.dispatch_cells.max(1), 1, 1);
        }

        let out_bytes = (self.zero_out_words.len() as u64) * 4;
        let staging_buffer = plan.staging_cache.take_or_create(
            &plan.context.device,
            out_bytes,
            "outer_convergence:out_bits (cached)",
        );
        encoder.copy_buffer_to_buffer(&self.b_out_bits, 0, &staging_buffer, 0, out_bytes);
        let submission_index = plan.context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", label_prefix);

        let raw_result: Result<Vec<u8>, String> = (|| {
            let slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = plan.context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });

            let map_result = rx
                .recv()
                .map_err(|_| "outer convergence readback map channel closed".to_string())?;
            map_result.map_err(|err| format!("outer convergence readback map failed: {err:?}"))?;

            let data = slice.get_mapped_range();
            let raw = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(raw)
        })();
        plan.staging_cache.put(out_bytes, staging_buffer);
        let raw = raw_result?;

        if raw.len() != out_bytes as usize {
            return Err(format!(
                "outer convergence readback size mismatch: got {} expected {}",
                raw.len(),
                out_bytes
            ));
        }
        let words: &[u32] = bytemuck::cast_slice(&raw);
        Ok(words.iter().map(|&w| f32::from_bits(w)).collect())
    }

    pub(crate) fn target_names(&self) -> &[String] {
        &self.target_names
    }

    pub(crate) fn state_scale(&self) -> Option<&[f32]> {
        self.state_scale.as_deref()
    }

    pub(crate) fn submit_break_eval_and_read_status(
        &self,
        plan: &GpuProgramPlan,
        mut encoder: wgpu::CommandEncoder,
        submission_label: &'static str,
    ) -> Result<bool, String> {
        let out_bytes = 4u64;
        let staging_buffer = plan.staging_cache.take_or_create(
            &plan.context.device,
            out_bytes,
            "outer_convergence:break_status (cached)",
        );
        encoder.copy_buffer_to_buffer(&self.b_break_status, 0, &staging_buffer, 0, out_bytes);
        let submission_index = plan.context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", submission_label);

        let raw_result: Result<Vec<u8>, String> = (|| {
            let slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = plan.context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });

            let map_result = rx
                .recv()
                .map_err(|_| "outer convergence break map channel closed".to_string())?;
            map_result.map_err(|err| format!("outer convergence break map failed: {err:?}"))?;

            let data = slice.get_mapped_range();
            let raw = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(raw)
        })();
        plan.staging_cache.put(out_bytes, staging_buffer);
        let raw = raw_result?;

        if raw.len() != out_bytes as usize {
            return Err(format!(
                "outer convergence break readback size mismatch: got {} expected {}",
                raw.len(),
                out_bytes
            ));
        }
        let words: &[u32] = bytemuck::cast_slice(&raw);
        Ok(words.first().copied().unwrap_or(0) != 0)
    }

    pub(crate) fn evaluate_break_from_current_buffers(
        &self,
        plan: &GpuProgramPlan,
        tol_rel: f32,
        tol_abs: f32,
    ) -> Result<bool, String> {
        let params = GpuOuterConvergenceBreakParams {
            count: self.target_names.len() as u32,
            tol_rel,
            tol_abs,
            _pad0: 0,
        };
        plan.context
            .queue
            .write_buffer(&self.b_break_params, 0, bytes_of(&params));

        let mut encoder =
            plan.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("outer_convergence:break_eval_buffered"),
                });
        self.encode_break_eval_into(&mut encoder);
        self.submit_break_eval_and_read_status(
            plan,
            encoder,
            "outer_convergence:break_eval_buffered",
        )
    }

    pub(crate) fn evaluate_break_from_state_on_gpu(
        &mut self,
        plan: &GpuProgramPlan,
        state: &wgpu::Buffer,
        tol_rel: f32,
        tol_abs: f32,
    ) -> Result<bool, String> {
        let params = GpuOuterConvergenceBreakParams {
            count: self.target_names.len() as u32,
            tol_rel,
            tol_abs,
            _pad0: 0,
        };
        plan.context
            .queue
            .write_buffer(&self.b_break_params, 0, bytes_of(&params));

        let mut encoder =
            plan.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("outer_convergence:break_eval_no_readback"),
                });

        let seeded_scale = if self.state_scale_ready {
            false
        } else {
            let bg_state = self.create_state_bind_group(&plan.context.device, state);
            self.encode_state_scale_into(&mut encoder, &bg_state);
            true
        };

        self.encode_delta_maxima_into(&mut encoder);
        self.encode_break_eval_into(&mut encoder);

        let converged = self.submit_break_eval_and_read_status(
            plan,
            encoder,
            "outer_convergence:break_eval_no_readback",
        )?;
        if seeded_scale {
            self.state_scale_ready = true;
        }
        Ok(converged)
    }

    // --- Encode-only methods for GPU-driven adaptive outer break ---

    /// Create a bind group for the reduction pipeline bound to the state buffer.
    /// Call this once before the one-submission encoder loop.
    pub(crate) fn create_state_bind_group(
        &self,
        device: &wgpu::Device,
        state: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bgl = self.pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_convergence:bg_state"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_descs_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.b_out_bits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.b_params_state.as_entire_binding(),
                },
            ],
        })
    }

    /// Upload break parameters into the GPU buffer. Call once before the encoder loop.
    pub(crate) fn upload_break_params(&self, queue: &wgpu::Queue, tol_rel: f32, tol_abs: f32) {
        let params = GpuOuterConvergenceBreakParams {
            count: self.target_names.len() as u32,
            tol_rel,
            tol_abs,
            _pad0: 0,
        };
        queue.write_buffer(&self.b_break_params, 0, bytes_of(&params));
    }

    /// Encode the delta-maxima reduction: clear out_bits, dispatch reduction, copy → b_delta.
    pub(crate) fn encode_delta_maxima_into(&self, encoder: &mut wgpu::CommandEncoder) {
        let out_bytes = (self.zero_out_words.len() as u64) * 4;
        encoder.clear_buffer(&self.b_out_bits, 0, Some(out_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("outer_convergence:delta_encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg_x, &[]);
            pass.dispatch_workgroups(self.dispatch_cells.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.b_out_bits, 0, &self.b_delta, 0, out_bytes);
    }

    /// Encode the state-scale reduction: clear out_bits, dispatch reduction, copy → b_scale.
    pub(crate) fn encode_state_scale_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_state: &wgpu::BindGroup,
    ) {
        let out_bytes = (self.zero_out_words.len() as u64) * 4;
        encoder.clear_buffer(&self.b_out_bits, 0, Some(out_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("outer_convergence:scale_encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg_state, &[]);
            pass.dispatch_workgroups(self.dispatch_cells.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.b_out_bits, 0, &self.b_scale, 0, out_bytes);
    }

    /// Encode the break evaluation: clear break_status, dispatch break kernel.
    pub(crate) fn encode_break_eval_into(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.b_break_status, 0, Some(4));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("outer_convergence:break_eval_encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.break_pipeline);
            pass.set_bind_group(0, &self.break_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Encode a full convergence check sequence for one outer iteration.
    ///
    /// If `first_iter` is true, also encodes the state-scale reduction.
    /// After this, `b_break_status` contains the convergence result on the GPU.
    pub(crate) fn encode_convergence_check(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_state: &wgpu::BindGroup,
        first_iter: bool,
    ) {
        if first_iter {
            self.encode_state_scale_into(encoder, bg_state);
        }
        self.encode_delta_maxima_into(encoder);
        self.encode_break_eval_into(encoder);
    }
}
