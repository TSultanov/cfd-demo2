//! Bind Group Builder Module
//!
//! Provides utilities for building shader bind groups from UnifiedFieldResources.
//! This module simplifies the process of wiring GPU buffers to shader bindings
//! by providing a declarative way to map buffer names to binding slots.

use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::wgsl_reflect::{create_bind_group_from_bindings, WgslBindingLike};

/// A builder for creating shader bind groups from various resource sources.
/// 
/// This builder aggregates buffers from UnifiedFieldResources and additional
/// external buffers, then creates bind groups that match shader expectations.
pub struct BindGroupBuilder<'a> {
    device: &'a wgpu::Device,
    unified_fields: Option<&'a UnifiedFieldResources>,
    external_buffers: Vec<(&'static str, &'a wgpu::Buffer)>,
    ping_pong_phase: usize,
}

impl<'a> BindGroupBuilder<'a> {
    /// Create a new bind group builder.
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            unified_fields: None,
            external_buffers: Vec::new(),
            ping_pong_phase: 0,
        }
    }

    /// Add unified field resources as a buffer source.
    pub fn with_unified_fields(mut self, fields: &'a UnifiedFieldResources) -> Self {
        self.unified_fields = Some(fields);
        self
    }

    /// Set the current ping-pong phase for state buffer resolution.
    pub fn at_ping_pong_phase(mut self, phase: usize) -> Self {
        self.ping_pong_phase = phase;
        self
    }

    /// Add an external buffer that can be resolved by name.
    pub fn with_buffer(mut self, name: &'static str, buffer: &'a wgpu::Buffer) -> Self {
        self.external_buffers.push((name, buffer));
        self
    }

    /// Build a bind group for the specified shader bindings and group index.
    pub fn build_bind_group<B: WgslBindingLike>(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        bindings: &[B],
        group: u32,
    ) -> Result<wgpu::BindGroup, String> {
        create_bind_group_from_bindings(
            self.device,
            label,
            layout,
            bindings,
            group,
            |name| self.resolve_buffer(name),
        )
    }

    /// Resolve a buffer by name, checking all available sources.
    fn resolve_buffer(&self, name: &str) -> Option<wgpu::BindingResource<'a>> {
        // First check external buffers
        for (buf_name, buffer) in &self.external_buffers {
            if *buf_name == name {
                return Some(wgpu::BindingResource::Buffer(
                    buffer.as_entire_buffer_binding(),
                ));
            }
        }

        // Then check unified field resources
        if let Some(fields) = self.unified_fields {
            if let Some(buffer) = fields.buffer_for_binding(name, self.ping_pong_phase) {
                return Some(wgpu::BindingResource::Buffer(
                    buffer.as_entire_buffer_binding(),
                ));
            }
        }

        None
    }
}

/// Create a ping-pong series of bind groups (for state buffers that rotate each step).
/// 
/// This creates 3 bind groups, one for each ping-pong phase, allowing the solver
/// to advance time by simply switching which bind group is used.
pub fn create_ping_pong_bind_groups<'a, B: WgslBindingLike>(
    device: &'a wgpu::Device,
    label_prefix: &str,
    layout: &wgpu::BindGroupLayout,
    bindings: &[B],
    group: u32,
    unified_fields: &'a UnifiedFieldResources,
    external_buffers: &[(&'static str, &'a wgpu::Buffer)],
) -> Result<Vec<wgpu::BindGroup>, String> {
    let mut bind_groups = Vec::with_capacity(3);
    
    for phase in 0..3 {
        let mut builder = BindGroupBuilder::new(device)
            .with_unified_fields(unified_fields)
            .at_ping_pong_phase(phase);
        
        for (name, buffer) in external_buffers {
            builder = builder.with_buffer(name, buffer);
        }
        
        bind_groups.push(builder.build_bind_group(
            &format!("{label_prefix} phase {phase}"),
            layout,
            bindings,
            group,
        )?);
    }
    
    Ok(bind_groups)
}

#[cfg(test)]
mod tests {
    // Note: These tests require a GPU device to run.
    // The module is tested through integration tests.
}
