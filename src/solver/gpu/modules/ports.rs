use std::collections::HashMap;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufF32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Port<T> {
    id: PortId,
    name: &'static str,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Port<T> {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub(crate) fn id(&self) -> PortId {
        self.id
    }
}

pub struct PortRegistry {
    next: u32,
}

impl Default for PortRegistry {
    fn default() -> Self {
        Self { next: 0 }
    }
}

impl PortRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn port<T>(&mut self, name: &'static str) -> Port<T> {
        let id = PortId(self.next);
        self.next = self.next.wrapping_add(1);
        Port {
            id,
            name,
            _marker: PhantomData,
        }
    }
}

pub struct LoweredBuffers {
    buffers: HashMap<PortId, wgpu::Buffer>,
}

impl Default for LoweredBuffers {
    fn default() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }
}

impl LoweredBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T>(&mut self, port: Port<T>, buffer: wgpu::Buffer) {
        self.buffers.insert(port.id(), buffer);
    }

    pub fn buffer<T>(&self, port: Port<T>) -> &wgpu::Buffer {
        let id = port.id();
        let name = port.name();
        self.buffers
            .get(&id)
            .unwrap_or_else(|| panic!("missing buffer for port {}", name))
    }

    pub fn clone_buffer<T>(&self, port: Port<T>) -> wgpu::Buffer {
        self.buffer(port).clone()
    }
}

pub struct Lowerer<'a> {
    device: &'a wgpu::Device,
    registry: PortRegistry,
    buffers: LoweredBuffers,
}

impl<'a> Lowerer<'a> {
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            registry: PortRegistry::new(),
            buffers: LoweredBuffers::new(),
        }
    }

    pub fn buffer_f32(
        &mut self,
        name: &'static str,
        size_bytes: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Port<BufF32> {
        let port = self.registry.port::<BufF32>(name);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage,
            mapped_at_creation: false,
        });
        self.buffers.insert(port, buffer);
        port
    }

    pub fn buffer_u32_init(
        &mut self,
        name: &'static str,
        data: &[u32],
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Port<BufU32> {
        let port = self.registry.port::<BufU32>(name);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            });
        self.buffers.insert(port, buffer);
        port
    }

    pub fn finish(self) -> LoweredBuffers {
        self.buffers
    }
}
