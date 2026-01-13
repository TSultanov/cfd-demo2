use crate::solver::mesh::{BoundaryType, Mesh};
use wgpu::util::DeviceExt;

pub struct MeshResources {
    pub b_face_owner: wgpu::Buffer,
    pub b_face_neighbor: wgpu::Buffer,
    pub b_face_boundary: wgpu::Buffer,
    pub b_face_areas: wgpu::Buffer,
    pub b_face_normals: wgpu::Buffer,
    pub b_face_centers: wgpu::Buffer,
    pub b_cell_centers: wgpu::Buffer,
    pub b_cell_vols: wgpu::Buffer,
    pub b_cell_face_offsets: wgpu::Buffer,
    pub b_cell_faces: wgpu::Buffer,
    pub b_cell_face_matrix_indices: wgpu::Buffer,
    pub b_diagonal_indices: wgpu::Buffer,
    pub b_scalar_row_offsets: wgpu::Buffer,
    pub b_scalar_col_indices: wgpu::Buffer,
    pub scalar_row_offsets: Vec<u32>,
    pub scalar_col_indices: Vec<u32>,
}

impl MeshResources {
    pub fn buffer_for_binding_name(&self, name: &str) -> Option<&wgpu::Buffer> {
        match name {
            "face_owner" => Some(&self.b_face_owner),
            "face_neighbor" => Some(&self.b_face_neighbor),
            "face_boundary" => Some(&self.b_face_boundary),
            "face_areas" => Some(&self.b_face_areas),
            "face_normals" => Some(&self.b_face_normals),
            "face_centers" => Some(&self.b_face_centers),
            "cell_centers" => Some(&self.b_cell_centers),
            "cell_vols" => Some(&self.b_cell_vols),
            "cell_face_offsets" => Some(&self.b_cell_face_offsets),
            "cell_faces" => Some(&self.b_cell_faces),
            "cell_face_matrix_indices" => Some(&self.b_cell_face_matrix_indices),
            "diagonal_indices" => Some(&self.b_diagonal_indices),
            "scalar_row_offsets" => Some(&self.b_scalar_row_offsets),
            "scalar_col_indices" => Some(&self.b_scalar_col_indices),
            _ => None,
        }
    }
}

pub fn init_mesh(device: &wgpu::Device, mesh: &Mesh) -> MeshResources {
    let num_cells = mesh.cell_cx.len() as u32;

    // --- CSR Matrix Structure ---
    let mut scalar_row_offsets = vec![0u32; num_cells as usize + 1];
    let mut scalar_col_indices = Vec::new();

    let mut adj = vec![Vec::new(); num_cells as usize];
    for (i, &owner) in mesh.face_owner.iter().enumerate() {
        if let Some(neighbor) = mesh.face_neighbor[i] {
            adj[owner].push(neighbor);
            adj[neighbor].push(owner);
        }
    }

    for (i, list) in adj.iter_mut().enumerate() {
        list.push(i); // Add diagonal
        list.sort();
        list.dedup();
    }

    let mut current_offset = 0;
    for (i, list) in adj.iter().enumerate() {
        scalar_row_offsets[i] = current_offset;
        for &neighbor in list {
            scalar_col_indices.push(neighbor as u32);
        }
        current_offset += list.len() as u32;
    }
    scalar_row_offsets[num_cells as usize] = current_offset;

    let b_scalar_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh scalar_row_offsets"),
        contents: bytemuck::cast_slice(&scalar_row_offsets),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_scalar_col_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh scalar_col_indices"),
        contents: bytemuck::cast_slice(&scalar_col_indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // --- Mesh Buffers ---
    let face_owner: Vec<u32> = mesh.face_owner.iter().map(|&x| x as u32).collect();
    let b_face_owner = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Owner Buffer"),
        contents: bytemuck::cast_slice(&face_owner),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let face_neighbor: Vec<u32> = mesh
        .face_neighbor
        .iter()
        .map(|&x| match x {
            Some(n) => n as u32,
            None => u32::MAX,
        })
        .collect();
    let b_face_neighbor = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Neighbor Buffer"),
        contents: bytemuck::cast_slice(&face_neighbor),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let face_boundary: Vec<u32> = mesh
        .face_boundary
        .iter()
        .map(|b| match b {
            None => 0,
            Some(BoundaryType::Inlet) => 1,
            Some(BoundaryType::Outlet) => 2,
            Some(BoundaryType::Wall) => 3,
        })
        .collect();
    let b_face_boundary = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Boundary Buffer"),
        contents: bytemuck::cast_slice(&face_boundary),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let face_areas: Vec<f32> = mesh.face_area.iter().map(|&x| x as f32).collect();
    let b_face_areas = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Areas Buffer"),
        contents: bytemuck::cast_slice(&face_areas),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let face_normals: Vec<[f32; 2]> = mesh
        .face_nx
        .iter()
        .zip(mesh.face_ny.iter())
        .map(|(&nx, &ny)| [nx as f32, ny as f32])
        .collect();
    let b_face_normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Normals Buffer"),
        contents: bytemuck::cast_slice(&face_normals),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let face_centers: Vec<[f32; 2]> = mesh
        .face_cx
        .iter()
        .zip(mesh.face_cy.iter())
        .map(|(&cx, &cy)| [cx as f32, cy as f32])
        .collect();
    let b_face_centers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Centers Buffer"),
        contents: bytemuck::cast_slice(&face_centers),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let cell_centers: Vec<[f32; 2]> = mesh
        .cell_cx
        .iter()
        .zip(mesh.cell_cy.iter())
        .map(|(&cx, &cy)| [cx as f32, cy as f32])
        .collect();
    let b_cell_centers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Centers Buffer"),
        contents: bytemuck::cast_slice(&cell_centers),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let cell_vols: Vec<f32> = mesh.cell_vol.iter().map(|&x| x as f32).collect();
    let b_cell_vols = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Volumes Buffer"),
        contents: bytemuck::cast_slice(&cell_vols),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let cell_face_offsets: Vec<u32> = mesh.cell_face_offsets.iter().map(|&x| x as u32).collect();
    let b_cell_face_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Face Offsets Buffer"),
        contents: bytemuck::cast_slice(&cell_face_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let cell_faces: Vec<u32> = mesh.cell_faces.iter().map(|&x| x as u32).collect();
    let b_cell_faces = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Faces Buffer"),
        contents: bytemuck::cast_slice(&cell_faces),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // --- Cell Face Matrix Indices ---
    let mut cell_face_matrix_indices = Vec::new();
    for i in 0..num_cells {
        let start = mesh.cell_face_offsets[i as usize];
        let end = mesh.cell_face_offsets[i as usize + 1];

        for k in start..end {
            let face_idx = mesh.cell_faces[k];
            let owner = mesh.face_owner[face_idx];
            let neighbor_opt = mesh.face_neighbor[face_idx];

            let neighbor = if owner == i as usize {
                neighbor_opt
            } else {
                Some(owner)
            };

            let target_col = match neighbor {
                Some(n) => n as u32,
                None => u32::MAX,
            };

            if target_col == u32::MAX {
                cell_face_matrix_indices.push(u32::MAX);
            } else {
                let row_start = scalar_row_offsets[i as usize] as usize;
                let row_end = scalar_row_offsets[i as usize + 1] as usize;
                let cols = &scalar_col_indices[row_start..row_end];

                if let Ok(idx) = cols.binary_search(&target_col) {
                    cell_face_matrix_indices.push((row_start + idx) as u32);
                } else {
                    cell_face_matrix_indices.push(u32::MAX);
                }
            }
        }
    }

    let b_cell_face_matrix_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Face Matrix Indices Buffer"),
        contents: bytemuck::cast_slice(&cell_face_matrix_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let mut diagonal_indices = Vec::with_capacity(num_cells as usize);
    for i in 0..num_cells {
        let row_start = scalar_row_offsets[i as usize] as usize;
        let row_end = scalar_row_offsets[i as usize + 1] as usize;
        let cols = &scalar_col_indices[row_start..row_end];

        if let Ok(idx) = cols.binary_search(&i) {
            diagonal_indices.push((row_start + idx) as u32);
        } else {
            panic!("Diagonal not found in CSR cols");
        }
    }

    let b_diagonal_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Diagonal Indices Buffer"),
        contents: bytemuck::cast_slice(&diagonal_indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    MeshResources {
        b_face_owner,
        b_face_neighbor,
        b_face_boundary,
        b_face_areas,
        b_face_normals,
        b_face_centers,
        b_cell_centers,
        b_cell_vols,
        b_cell_face_offsets,
        b_cell_faces,
        b_cell_face_matrix_indices,
        b_diagonal_indices,
        b_scalar_row_offsets,
        b_scalar_col_indices,
        scalar_row_offsets,
        scalar_col_indices,
    }
}
