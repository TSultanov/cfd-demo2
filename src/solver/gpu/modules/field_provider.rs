//! Field Provider Trait
//!
//! This module provides a trait abstraction for field buffer access, allowing
//! different field storage implementations to be used interchangeably.

use crate::solver::gpu::modules::constants::ConstantsModule;
use crate::solver::gpu::modules::state::PingPongState;
use crate::solver::gpu::structs::{GpuConstants, GpuLowMachParams};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// Trait for providing field buffers to shaders.
///
/// This abstraction allows solver kernels to be agnostic to the concrete field
/// storage implementation (currently backed by `UnifiedFieldResources`).
pub trait FieldProvider {
    /// Get the ping-pong state.
    fn state(&self) -> &PingPongState;

    /// Get the step handle for ping-pong indexing.
    fn step_handle(&self) -> Arc<AtomicUsize>;

    /// Get the constants module.
    fn constants(&self) -> &ConstantsModule;

    /// Get mutable access to the constants module.
    fn constants_mut(&mut self) -> &mut ConstantsModule;

    /// Get the current constants values.
    fn constants_values(&self) -> &GpuConstants {
        self.constants().values()
    }

    /// Get mutable access to constants values.
    fn constants_values_mut(&mut self) -> &mut GpuConstants {
        self.constants_mut().values_mut()
    }

    /// Get the iteration snapshot buffer (for implicit solvers).
    fn iteration_buffer(&self) -> Option<&wgpu::Buffer>;

    /// Get the flux buffer.
    fn flux_buffer(&self) -> Option<&wgpu::Buffer>;

    /// Get a gradient buffer by field name.
    fn gradient_buffer(&self, field_name: &str) -> Option<&wgpu::Buffer>;

    /// Get the low-mach params buffer.
    fn low_mach_buffer(&self) -> Option<&wgpu::Buffer>;

    /// Get low-mach params values.
    fn low_mach_params(&self) -> &GpuLowMachParams;

    /// Get mutable low-mach params values.
    fn low_mach_params_mut(&mut self) -> &mut GpuLowMachParams;

    /// Get buffer by binding name for shader reflection.
    /// The `ping_pong_phase` determines which state buffer is current.
    fn buffer_for_binding(&self, name: &str, ping_pong_phase: usize) -> Option<&wgpu::Buffer> {
        let (idx_cur, idx_old, idx_old_old) =
            crate::solver::gpu::modules::state::ping_pong_indices(ping_pong_phase);

        match name {
            "state" => Some(&self.state().buffers()[idx_cur]),
            "state_old" => Some(&self.state().buffers()[idx_old]),
            "state_old_old" => Some(&self.state().buffers()[idx_old_old]),
            "state_iter" => self.iteration_buffer(),
            "fluxes" => self.flux_buffer(),
            "low_mach" | "low_mach_params" => self.low_mach_buffer(),
            "constants" => Some(self.constants().buffer()),
            _ => {
                // Check for gradient field names (grad_rho, grad_rho_u_x, etc.)
                if let Some(field) = name.strip_prefix("grad_") {
                    self.gradient_buffer(field)
                } else {
                    None
                }
            }
        }
    }

    /// Number of cells.
    fn num_cells(&self) -> u32;

    /// State stride (floats per cell).
    fn state_stride(&self) -> u32;

    /// Write state data to all ping-pong buffers.
    fn write_state_bytes(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        self.state().write_all(queue, bytes);
    }

    /// Advance to next time step.
    fn advance_step(&self) -> usize {
        self.state().advance()
    }

    /// Get state size in bytes.
    fn state_size_bytes(&self) -> u64 {
        (self.num_cells() as u64) * (self.state_stride() as u64) * 4
    }
}
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;

impl FieldProvider for UnifiedFieldResources {
    fn state(&self) -> &PingPongState {
        &self.state
    }

    fn step_handle(&self) -> Arc<AtomicUsize> {
        self.state.step_handle()
    }

    fn constants(&self) -> &ConstantsModule {
        &self.constants
    }

    fn constants_mut(&mut self) -> &mut ConstantsModule {
        &mut self.constants
    }

    fn iteration_buffer(&self) -> Option<&wgpu::Buffer> {
        self.iteration_snapshot.as_ref()
    }

    fn flux_buffer(&self) -> Option<&wgpu::Buffer> {
        self.flux_buffer.as_ref()
    }

    fn gradient_buffer(&self, field_name: &str) -> Option<&wgpu::Buffer> {
        self.gradients.get(field_name)
    }

    fn low_mach_buffer(&self) -> Option<&wgpu::Buffer> {
        self.low_mach_params_buffer.as_ref()
    }

    fn low_mach_params(&self) -> &GpuLowMachParams {
        &self.low_mach_params
    }

    fn low_mach_params_mut(&mut self) -> &mut GpuLowMachParams {
        &mut self.low_mach_params
    }

    fn num_cells(&self) -> u32 {
        self.num_cells
    }

    fn state_stride(&self) -> u32 {
        self.state_stride
    }
}
