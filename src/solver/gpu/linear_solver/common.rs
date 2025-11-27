use crate::solver::gpu::structs::GpuSolver;

impl GpuSolver {
    pub fn zero_buffer(&self, buffer: &wgpu::Buffer, size: u64) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Zero Buffer Encoder"),
                });
        encoder.clear_buffer(buffer, 0, Some(size));
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub fn encode_compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_mesh, &[]);
        cpass.set_bind_group(1, bind_group, &[]); // linear_state
        cpass.set_bind_group(2, &self.bg_linear_matrix, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    pub fn encode_spmv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        num_groups: u32,
    ) {
        self.encode_compute(encoder, pipeline, &self.bg_linear_state, num_groups);
    }

    pub fn encode_dot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_dot_inputs: &wgpu::BindGroup,
        num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Dot Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline_dot);
        cpass.set_bind_group(0, &self.bg_dot_params, &[]);
        cpass.set_bind_group(1, bg_dot_inputs, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    pub fn encode_dot_pair(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Dot Pair Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline_dot_pair);
        cpass.set_bind_group(0, &self.bg_dot_params, &[]);
        cpass.set_bind_group(1, bind_group, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    pub fn encode_scalar_compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scalar Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_scalars, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    pub fn encode_scalar_reduce(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        _num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scalar Reduce Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_scalars, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    pub async fn read_scalar_r_r(&self) -> f32 {
        // Read r_r from b_scalars.
        // Offset of r_r is 8 * 4 = 32 bytes.
        let offset = 32;
        let size = 4;

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Read Scalar Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_scalars, offset, &self.b_staging_scalar, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = self.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.context.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let val = f32::from_ne_bytes(data[0..4].try_into().unwrap());
        drop(data);
        self.b_staging_scalar.unmap();

        val
    }

    pub async fn read_scalar_rho_new(&self) -> f32 {
        let offset = 4;
        let size = 4;
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Read Scalar Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_scalars, offset, &self.b_staging_scalar, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = self.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.context.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let val = f32::from_ne_bytes(data[0..4].try_into().unwrap());
        drop(data);
        self.b_staging_scalar.unmap();
        val
    }
}
