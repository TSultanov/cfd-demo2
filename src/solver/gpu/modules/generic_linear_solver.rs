//! Identity preconditioner used by the Krylov solve infrastructure.

use crate::solver::gpu::modules::krylov_precond::{PrecondContext, PreconditionerModule};

/// Identity preconditioner (no preconditioning).
///
/// Simply copies the input vector to the output vector unchanged.
/// Uses `encoder.copy_buffer_to_buffer()` — no compute pipeline needed.
#[derive(Default)]
pub struct IdentityPreconditioner;

impl IdentityPreconditioner {
    pub fn new() -> Self {
        Self
    }
}

impl PreconditionerModule for IdentityPreconditioner {
    fn encode_apply(
        &mut self,
        _device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PrecondContext<'_>,
        input: wgpu::BindingResource<'_>,
        output: wgpu::BindingResource<'_>,
    ) {
        let wgpu::BindingResource::Buffer(in_buf) = &input else {
            return;
        };
        let wgpu::BindingResource::Buffer(out_buf) = &output else {
            return;
        };
        let size = in_buf
            .size
            .map(|s| s.get())
            .unwrap_or((ctx.num_dofs as u64) * 4);
        encoder.copy_buffer_to_buffer(
            in_buf.buffer,
            in_buf.offset,
            out_buf.buffer,
            out_buf.offset,
            size,
        );
    }
}
