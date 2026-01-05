use crate::solver::gpu::bindings::generated::coupled_assembly_merged;
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
    pub bg_mesh: wgpu::BindGroup,
    pub bgl_mesh: wgpu::BindGroupLayout,
    pub row_offsets: Vec<u32>,
    pub col_indices: Vec<u32>,
}

pub fn init_mesh(device: &wgpu::Device, mesh: &Mesh) -> MeshResources {
    let num_cells = mesh.cell_cx.len() as u32;

    // --- CSR Matrix Structure ---
    let mut row_offsets = vec![0u32; num_cells as usize + 1];
    let mut col_indices = Vec::new();

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
        row_offsets[i] = current_offset;
        for &neighbor in list {
            col_indices.push(neighbor as u32);
        }
        current_offset += list.len() as u32;
    }
    row_offsets[num_cells as usize] = current_offset;

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
                let row_start = row_offsets[i as usize] as usize;
                let row_end = row_offsets[i as usize + 1] as usize;
                let cols = &col_indices[row_start..row_end];

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
        let row_start = row_offsets[i as usize] as usize;
        let row_end = row_offsets[i as usize + 1] as usize;
        let cols = &col_indices[row_start..row_end];

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

    // Group 0: Mesh (Read Only)
    let bgl_mesh = device
        .create_bind_group_layout(&coupled_assembly_merged::WgpuBindGroup0::LAYOUT_DESCRIPTOR);

    let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh Bind Group"),
        layout: &bgl_mesh,
        entries: &coupled_assembly_merged::WgpuBindGroup0Entries::new(
            coupled_assembly_merged::WgpuBindGroup0EntriesParams {
                face_owner: b_face_owner.as_entire_buffer_binding(),
                face_neighbor: b_face_neighbor.as_entire_buffer_binding(),
                face_areas: b_face_areas.as_entire_buffer_binding(),
                face_normals: b_face_normals.as_entire_buffer_binding(),
                cell_centers: b_cell_centers.as_entire_buffer_binding(),
                cell_vols: b_cell_vols.as_entire_buffer_binding(),
                cell_face_offsets: b_cell_face_offsets.as_entire_buffer_binding(),
                cell_faces: b_cell_faces.as_entire_buffer_binding(),
                cell_face_matrix_indices: b_cell_face_matrix_indices.as_entire_buffer_binding(),
                diagonal_indices: b_diagonal_indices.as_entire_buffer_binding(),
                face_boundary: b_face_boundary.as_entire_buffer_binding(),
                face_centers: b_face_centers.as_entire_buffer_binding(),
            },
        )
        .into_array(),
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
        bg_mesh,
        bgl_mesh,
        row_offsets,
        col_indices,
    }
}

impl MeshResources {
    pub fn as_bind_group_0_entries(
        &self,
    ) -> coupled_assembly_merged::WgpuBindGroup0EntriesParams<'_> {
        coupled_assembly_merged::WgpuBindGroup0EntriesParams {
            face_owner: self.b_face_owner.as_entire_buffer_binding(),
            face_neighbor: self.b_face_neighbor.as_entire_buffer_binding(),
            face_areas: self.b_face_areas.as_entire_buffer_binding(),
            face_normals: self.b_face_normals.as_entire_buffer_binding(),
            cell_centers: self.b_cell_centers.as_entire_buffer_binding(),
            cell_vols: self.b_cell_vols.as_entire_buffer_binding(),
            cell_face_offsets: self.b_cell_face_offsets.as_entire_buffer_binding(),
            cell_faces: self.b_cell_faces.as_entire_buffer_binding(),
            cell_face_matrix_indices: self.b_cell_face_matrix_indices.as_entire_buffer_binding(),
            diagonal_indices: self.b_diagonal_indices.as_entire_buffer_binding(),
            face_boundary: self.b_face_boundary.as_entire_buffer_binding(),
            face_centers: self.b_face_centers.as_entire_buffer_binding(),
        }
    }
}
