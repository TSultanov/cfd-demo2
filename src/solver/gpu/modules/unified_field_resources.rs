//! Unified Field Resources Module
//!
//! This module provides a common abstraction for field storage across all solver families.
//! It combines ping-pong state management, gradient buffers, and other auxiliary storage
//! into a single cohesive module that can be parameterized by the SolverRecipe.

use crate::solver::gpu::modules::constants::ConstantsModule;
use crate::solver::gpu::modules::state::PingPongState;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::structs::{GpuConstants, GpuLowMachParams};
use bytemuck::cast_slice;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Unified field resources for any solver family.
///
/// This module owns:
/// - Ping-pong state buffers for time integration
/// - Gradient buffers (if required by the recipe)
/// - Constants buffer
/// - Flux buffer (for face-based fluxes)
/// - Additional auxiliary buffers as specified by the recipe
pub struct UnifiedFieldResources {
    /// Ping-pong state for time stepping
    pub state: PingPongState,

    /// Solver constants (dt, viscosity, etc.)
    pub constants: ConstantsModule,

    /// Gradient buffers keyed by field name
    pub gradients: HashMap<String, wgpu::Buffer>,

    /// History buffers for multi-step time integration
    pub history_buffers: Vec<wgpu::Buffer>,

    /// Iteration snapshot buffer (for implicit solvers)
    pub iteration_snapshot: Option<wgpu::Buffer>,

    /// Flux buffer for face-based flux storage (optional, for compressible/transport solvers)
    pub flux_buffer: Option<wgpu::Buffer>,

    /// Low-Mach preconditioning parameters (optional, for compressible solvers)
    pub low_mach_params_buffer: Option<wgpu::Buffer>,
    pub low_mach_params: GpuLowMachParams,

    /// Number of cells
    pub num_cells: u32,

    /// Number of faces (for flux buffer sizing)
    pub num_faces: u32,

    /// Flux stride (floats per face, for compressible: 4 for [rho, rho_u_x, rho_u_y, rho_e])
    pub flux_stride: u32,

    /// State stride (floats per cell)
    pub state_stride: u32,
}

impl UnifiedFieldResources {
    /// Allocate unified field resources directly from the solver recipe.
    ///
    /// This is the single allocation entrypoint: plans should not decide which
    /// optional buffers exist.
    pub fn from_recipe(
        device: &wgpu::Device,
        recipe: &SolverRecipe,
        num_cells: u32,
        num_faces: u32,
        state_stride: u32,
        initial_constants: GpuConstants,
    ) -> Self {
        let state_size = num_cells as usize * state_stride as usize;
        let zero_state = vec![0.0f32; state_size];

        let state = PingPongState::new([
            Self::create_state_buffer(device, &zero_state, "UnifiedField state buffer 0"),
            Self::create_state_buffer(device, &zero_state, "UnifiedField state buffer 1"),
            Self::create_state_buffer(device, &zero_state, "UnifiedField state buffer 2"),
        ]);

        let constants = ConstantsModule::new(device, initial_constants, "UnifiedField constants");

        let mut gradients = HashMap::new();
        for field_name in &recipe.gradient_fields {
            let size_per_cell = if field_name == "state" {
                state_stride as usize * 2
            } else {
                2
            };
            let grad_size = num_cells as usize * size_per_cell;
            let zero_grad = vec![0.0f32; grad_size];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("UnifiedField grad_{field_name}")),
                contents: cast_slice(&zero_grad),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            gradients.insert(field_name.clone(), buffer);
        }

        // Retained for compatibility: time history beyond PingPongState is not
        // used by current kernels, but the recipe can request it.
        let mut history_buffers = Vec::new();
        let history_levels = recipe.time_integration.history_levels;
        if history_levels > 2 {
            for i in 0..(history_levels - 2) {
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("UnifiedField history buffer {i}")),
                    contents: cast_slice(&zero_state),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });
                history_buffers.push(buffer);
            }
        }

        let iteration_snapshot = if recipe.is_implicit() {
            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("UnifiedField iteration snapshot"),
                    contents: cast_slice(&zero_state),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
            )
        } else {
            None
        };

        let (flux_buffer, flux_stride) = if let Some(flux) = recipe.flux {
            if num_faces == 0 {
                (None, flux.stride)
            } else {
                let flux_size = num_faces as usize * flux.stride as usize;
                let zero_flux = vec![0.0f32; flux_size];
                (
                    Some(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("UnifiedField flux buffer"),
                            contents: cast_slice(&zero_flux),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        }),
                    ),
                    flux.stride,
                )
            }
        } else {
            (None, 0)
        };

        let low_mach_params = GpuLowMachParams::default();
        let low_mach_params_buffer = if recipe.requires_low_mach_params {
            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("UnifiedField low-mach params"),
                    contents: bytemuck::bytes_of(&low_mach_params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
            )
        } else {
            None
        };

        Self {
            state,
            constants,
            gradients,
            history_buffers,
            iteration_snapshot,
            flux_buffer,
            low_mach_params_buffer,
            low_mach_params,
            num_cells,
            num_faces,
            flux_stride,
            state_stride,
        }
    }

    fn create_state_buffer(device: &wgpu::Device, data: &[f32], label: &str) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    /// Get the current state buffer.
    pub fn current_state(&self) -> &wgpu::Buffer {
        self.state.state()
    }

    /// Get the previous state buffer.
    pub fn previous_state(&self) -> &wgpu::Buffer {
        self.state.state_old()
    }

    /// Get a gradient buffer by field name.
    pub fn gradient_for(&self, field_name: &str) -> Option<&wgpu::Buffer> {
        self.gradients.get(field_name)
    }

    /// Check if gradients are available.
    pub fn has_gradients(&self) -> bool {
        !self.gradients.is_empty()
    }

    /// Write state data to all ping-pong buffers.
    pub fn write_state_bytes(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        self.state.write_all(queue, bytes);
    }

    /// Advance the ping-pong state to the next step.
    pub fn advance_step(&self) -> usize {
        self.state.advance()
    }

    /// Get the state buffer size in bytes.
    pub fn state_size_bytes(&self) -> u64 {
        (self.num_cells as u64) * (self.state_stride as u64) * 4
    }

    /// Copy current state to iteration snapshot (for implicit solver restore).
    pub fn snapshot_for_iteration(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(snapshot) = &self.iteration_snapshot {
            encoder.copy_buffer_to_buffer(
                self.state.state(),
                0,
                snapshot,
                0,
                self.state_size_bytes(),
            );
        }
    }

    /// Restore state from iteration snapshot.
    pub fn restore_from_snapshot(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(snapshot) = &self.iteration_snapshot {
            encoder.copy_buffer_to_buffer(
                snapshot,
                0,
                self.state.state(),
                0,
                self.state_size_bytes(),
            );
        }
    }

    /// Get all state buffers for ping-pong indexing.
    pub fn state_buffers(&self) -> &[wgpu::Buffer; 3] {
        self.state.buffers()
    }

    /// Get buffer by binding name (for shader reflection).
    /// Returns the appropriate buffer based on the current ping-pong phase.
    /// Note: "constants" is not returned here since it's typically managed by the runtime.
    /// Use buffer_for_binding_with_constants() if you want to include constants.
    pub fn buffer_for_binding(&self, name: &str, ping_pong_phase: usize) -> Option<&wgpu::Buffer> {
        let (idx_cur, idx_old, idx_old_old) =
            crate::solver::gpu::modules::state::ping_pong_indices(ping_pong_phase);

        match name {
            "state" => Some(&self.state.buffers()[idx_cur]),
            "state_old" => Some(&self.state.buffers()[idx_old]),
            "state_old_old" => Some(&self.state.buffers()[idx_old_old]),
            "state_iter" => self.iteration_snapshot.as_ref(),
            "fluxes" => self.flux_buffer.as_ref(),
            "low_mach_params" => self.low_mach_params_buffer.as_ref(),
            "grad_state" => self.gradients.get("state"),
            _ => {
                // Check for gradient field names
                if let Some(field) = name.strip_prefix("grad_") {
                    self.gradients.get(field)
                } else {
                    None
                }
            }
        }
    }

    /// Get the step handle for the ping-pong state.
    pub fn step_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
        self.state.step_handle()
    }

    /// Update constants buffer with new values.
    pub fn update_constants(&mut self, queue: &wgpu::Queue) {
        self.constants.write(queue);
    }

    /// Get mutable access to constants values.
    pub fn constants_mut(&mut self) -> &mut GpuConstants {
        self.constants.values_mut()
    }

    /// Get read access to constants values.
    pub fn constants(&self) -> &GpuConstants {
        self.constants.values()
    }

    /// Get flux buffer if available.
    pub fn flux_buffer(&self) -> Option<&wgpu::Buffer> {
        self.flux_buffer.as_ref()
    }

    /// Get low-mach params buffer if available.
    pub fn low_mach_params_buffer(&self) -> Option<&wgpu::Buffer> {
        self.low_mach_params_buffer.as_ref()
    }

    /// Update low-mach parameters and write to GPU.
    pub fn update_low_mach_params(&mut self, queue: &wgpu::Queue) {
        if let Some(buf) = &self.low_mach_params_buffer {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&self.low_mach_params));
        }
    }

    /// Get mutable access to low-mach params.
    pub fn low_mach_params_mut(&mut self) -> &mut GpuLowMachParams {
        &mut self.low_mach_params
    }

    /// Add a gradient buffer to the resources.
    /// This can be used to add gradients that weren't specified in the recipe.
    pub fn add_gradient(&mut self, device: &wgpu::Device, field_name: &str) {
        if self.gradients.contains_key(field_name) {
            return;
        }
        let size_per_cell = if field_name == "state" {
            self.state_stride as usize * 2
        } else {
            2
        };
        let grad_size = self.num_cells as usize * size_per_cell;
        let zero_grad = vec![0.0f32; grad_size];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("UnifiedField grad_{field_name}")),
            contents: cast_slice(&zero_grad),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.gradients.insert(field_name.to_string(), buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests would require a GPU device to run.
    // For now, we just verify the types compile correctly.

    #[test]
    fn unified_field_resources_compiles() {
        // Type check only - no GPU available in unit tests
        fn _check_types(resources: &UnifiedFieldResources) {
            let _ = resources.current_state();
            let _ = resources.previous_state();
            let _ = resources.has_gradients();
        }
    }
}
