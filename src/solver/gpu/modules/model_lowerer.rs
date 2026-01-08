use crate::solver::gpu::init::mesh;
use crate::solver::mesh::Mesh;

use super::ports::{Lowerer, PortSpace};

pub struct LoweredCommon {
    pub num_cells: u32,
    pub num_faces: u32,
    pub mesh: mesh::MeshResources,
    pub ports: PortSpace,
}

impl LoweredCommon {
    pub fn new(device: &wgpu::Device, mesh_input: &Mesh) -> Self {
        let num_cells = mesh_input.cell_cx.len() as u32;
        let num_faces = mesh_input.face_owner.len() as u32;
        let mesh = mesh::init_mesh(device, mesh_input);
        let ports = PortSpace::new();
        Self {
            num_cells,
            num_faces,
            mesh,
            ports,
        }
    }

    pub fn lowerer<'a>(&'a mut self, device: &'a wgpu::Device) -> Lowerer<'a> {
        self.ports.lowerer(device)
    }
}

