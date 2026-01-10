//! Unified Field Resources Module
//!
//! This module provides a common abstraction for field storage across all solver families.
//! It combines ping-pong state management, gradient buffers, and other auxiliary storage
//! into a single cohesive module that can be parameterized by the SolverRecipe.

use crate::solver::gpu::modules::constants::ConstantsModule;
use crate::solver::gpu::modules::state::PingPongState;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::structs::GpuConstants;
use bytemuck::cast_slice;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Unified field resources for any solver family.
///
/// This module owns:
/// - Ping-pong state buffers for time integration
/// - Gradient buffers (if required by the recipe)
/// - Constants buffer
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

    /// Number of cells
    pub num_cells: u32,

    /// State stride (floats per cell)
    pub state_stride: u32,
}

impl UnifiedFieldResources {
    /// Create unified field resources from a solver recipe.
    pub fn new(
        device: &wgpu::Device,
        recipe: &SolverRecipe,
        num_cells: u32,
        state_stride: u32,
        initial_constants: GpuConstants,
    ) -> Self {
        let state_size = num_cells as usize * state_stride as usize;

        // Create ping-pong state buffers
        let zero_state = vec![0.0f32; state_size];
        let state = PingPongState::new([
            Self::create_state_buffer(device, &zero_state, "UnifiedField state buffer 0"),
            Self::create_state_buffer(device, &zero_state, "UnifiedField state buffer 1"),
            Self::create_state_buffer(device, &zero_state, "UnifiedField state buffer 2"),
        ]);

        // Create constants buffer
        let constants = ConstantsModule::new(device, initial_constants, "UnifiedField constants");

        // Create gradient buffers based on recipe
        let mut gradients = HashMap::new();
        for field_name in &recipe.gradient_fields {
            let grad_size = num_cells as usize * 2; // 2D gradient (dx, dy)
            let zero_grad = vec![0.0f32; grad_size];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("UnifiedField grad_{field_name}")),
                contents: cast_slice(&zero_grad),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            gradients.insert(field_name.clone(), buffer);
        }

        // Create history buffers for multi-step time integration
        let mut history_buffers = Vec::new();
        let history_levels = recipe.time_integration.history_levels;
        if history_levels > 1 {
            // Need additional history beyond what ping-pong provides
            for i in 0..(history_levels.saturating_sub(2)) {
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

        // Create iteration snapshot buffer for implicit solvers
        let iteration_snapshot = if recipe.is_implicit() {
            Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UnifiedField iteration snapshot"),
                contents: cast_slice(&zero_state),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }))
        } else {
            None
        };

        Self {
            state,
            constants,
            gradients,
            history_buffers,
            iteration_snapshot,
            num_cells,
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
    pub fn buffer_for_binding(&self, name: &str, ping_pong_phase: usize) -> Option<&wgpu::Buffer> {
        let (idx_cur, idx_old, idx_old_old) =
            crate::solver::gpu::modules::state::ping_pong_indices(ping_pong_phase);

        match name {
            "state" => Some(&self.state.buffers()[idx_cur]),
            "state_old" => Some(&self.state.buffers()[idx_old]),
            "state_old_old" => Some(&self.state.buffers()[idx_old_old]),
            "constants" => Some(self.constants.buffer()),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gpu::enums::TimeScheme;
    use crate::solver::gpu::recipe::SolverRecipe;
    use crate::solver::gpu::structs::PreconditionerType;
    use crate::solver::model::generic_diffusion_demo_model;
    use crate::solver::scheme::Scheme;

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
