use crate::solver::gpu::init::mesh::MeshResources;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;

#[derive(Clone)]
pub struct ResourceRegistry<'a> {
    mesh: Option<&'a MeshResources>,
    unified_fields: Option<&'a UnifiedFieldResources>,
    ping_pong_phase: usize,
    constants: Option<&'a wgpu::Buffer>,
    named_buffers: Vec<(&'static str, &'a wgpu::Buffer)>,
}

impl<'a> ResourceRegistry<'a> {
    pub fn new() -> Self {
        Self {
            mesh: None,
            unified_fields: None,
            ping_pong_phase: 0,
            constants: None,
            named_buffers: Vec::new(),
        }
    }

    pub fn with_mesh(mut self, mesh: &'a MeshResources) -> Self {
        self.mesh = Some(mesh);
        self
    }

    pub fn with_unified_fields(mut self, fields: &'a UnifiedFieldResources) -> Self {
        self.unified_fields = Some(fields);
        self
    }

    pub fn at_ping_pong_phase(mut self, phase: usize) -> Self {
        self.ping_pong_phase = phase;
        self
    }

    pub fn with_constants_buffer(mut self, buffer: &'a wgpu::Buffer) -> Self {
        self.constants = Some(buffer);
        self
    }

    pub fn with_buffer(mut self, name: &'static str, buffer: &'a wgpu::Buffer) -> Self {
        self.named_buffers.push((name, buffer));
        self
    }

    pub fn resolve(&self, name: &str) -> Option<wgpu::BindingResource<'a>> {
        for (buf_name, buffer) in &self.named_buffers {
            if *buf_name == name {
                return Some(wgpu::BindingResource::Buffer(
                    buffer.as_entire_buffer_binding(),
                ));
            }
        }

        if name == "constants" {
            if let Some(constants) = self.constants {
                return Some(wgpu::BindingResource::Buffer(
                    constants.as_entire_buffer_binding(),
                ));
            }
            if let Some(fields) = self.unified_fields {
                return Some(wgpu::BindingResource::Buffer(
                    fields.constants.buffer().as_entire_buffer_binding(),
                ));
            }
        }

        if let Some(mesh) = self.mesh {
            if let Some(buffer) = mesh.buffer_for_binding_name(name) {
                return Some(wgpu::BindingResource::Buffer(
                    buffer.as_entire_buffer_binding(),
                ));
            }
        }

        if let Some(fields) = self.unified_fields {
            if let Some(buf) = fields.buffer_for_binding(name, self.ping_pong_phase) {
                return Some(wgpu::BindingResource::Buffer(
                    buf.as_entire_buffer_binding(),
                ));
            }
        }

        None
    }
}

impl<'a> Default for ResourceRegistry<'a> {
    fn default() -> Self {
        Self::new()
    }
}
