use super::ports::{BufF32, BufU32, Port, PortSpace};

#[derive(Clone, Copy, Debug)]
pub struct LinearSystemPorts {
    pub row_offsets: Port<BufU32>,
    pub col_indices: Port<BufU32>,
    pub values: Port<BufF32>,
    pub rhs: Port<BufF32>,
    pub x: Port<BufF32>,
}

#[derive(Clone, Copy)]
pub struct LinearSystemView<'a> {
    pub ports: LinearSystemPorts,
    pub space: &'a PortSpace,
}

impl<'a> LinearSystemView<'a> {
    pub fn row_offsets(&self) -> &'a wgpu::Buffer {
        self.space.buffer(self.ports.row_offsets)
    }

    pub fn col_indices(&self) -> &'a wgpu::Buffer {
        self.space.buffer(self.ports.col_indices)
    }

    pub fn values(&self) -> &'a wgpu::Buffer {
        self.space.buffer(self.ports.values)
    }

    pub fn rhs(&self) -> &'a wgpu::Buffer {
        self.space.buffer(self.ports.rhs)
    }

    pub fn x(&self) -> &'a wgpu::Buffer {
        self.space.buffer(self.ports.x)
    }
}
