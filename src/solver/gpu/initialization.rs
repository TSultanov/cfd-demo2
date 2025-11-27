use crate::solver::mesh::{BoundaryType, Mesh};
use std::borrow::Cow;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

use super::structs::{GpuConstants, GpuSolver, SolverParams};

impl GpuSolver {
    pub async fn new(mesh: &Mesh) -> Self {
        let context = super::context::GpuContext::new().await;
        let device = &context.device;

        let num_cells = mesh.cell_cx.len() as u32;
        let num_faces = mesh.face_owner.len() as u32;

        // --- Mesh Buffers ---
        let face_owner: Vec<u32> = mesh.face_owner.iter().map(|&x| x as u32).collect();
        let b_face_owner = super::buffers::create_buffer_init(
            device,
            "Face Owner Buffer",
            &face_owner,
            wgpu::BufferUsages::STORAGE,
        );

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

        let cell_face_offsets: Vec<u32> =
            mesh.cell_face_offsets.iter().map(|&x| x as u32).collect();
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
        let num_nonzeros = current_offset;

        let b_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Row Offsets Buffer"),
            contents: bytemuck::cast_slice(&row_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let b_col_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Col Indices Buffer"),
            contents: bytemuck::cast_slice(&col_indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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

        let b_cell_face_matrix_indices =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

        // --- Field Buffers ---
        let zero_vecs = vec![[0.0f32; 2]; num_cells as usize];
        let b_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("U Buffer"),
            contents: bytemuck::cast_slice(&zero_vecs),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let b_u_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("U Old Buffer"),
            contents: bytemuck::cast_slice(&zero_vecs),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let b_u_old_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("U Old Old Buffer"),
            contents: bytemuck::cast_slice(&zero_vecs),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let zero_scalars = vec![0.0f32; num_cells as usize];
        let b_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("P Buffer"),
            contents: bytemuck::cast_slice(&zero_scalars),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let b_p_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("P Old Buffer"),
            contents: bytemuck::cast_slice(&zero_scalars),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let b_d_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("D_P Buffer"),
            contents: bytemuck::cast_slice(&zero_scalars),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let zero_fluxes = vec![0.0f32; num_faces as usize];
        let b_fluxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluxes Buffer"),
            contents: bytemuck::cast_slice(&zero_fluxes),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let b_grad_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grad P Buffer"),
            contents: bytemuck::cast_slice(&zero_vecs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let b_grad_component = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grad Component Buffer"),
            contents: bytemuck::cast_slice(&zero_vecs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let b_grad_p_prime = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grad P Prime Buffer"),
            size: (num_cells as u64 * std::mem::size_of::<[f32; 2]>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let constants = GpuConstants {
            dt: 0.0001, // Reduced dt
            dt_old: 0.0001,
            time: 0.0,
            viscosity: 0.01,
            density: 1.0,
            component: 0,
            alpha_p: 1.0, // Default pressure relaxation
            scheme: 0,    // Upwind
            alpha_u: 0.7, // Default velocity under-relaxation
            stride_x: 65535 * 64,
            time_scheme: 0,
        };
        let b_constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Constants Buffer"),
            contents: bytemuck::bytes_of(&constants),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Solver Buffers ---
        let b_matrix_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix Values Buffer"),
            size: (num_nonzeros as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_rhs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RHS Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("X Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_r = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("R Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_r0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("R0 Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_p_solver = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P Solver Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_s = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("S Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let b_t = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Dot Product & Params
        let workgroup_size = 64;
        let num_groups = num_cells.div_ceil(workgroup_size);

        let b_dot_result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Result Buffer"),
            size: (num_groups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_dot_result_2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Result Buffer 2"),
            size: (num_groups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_scalars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scalars Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_staging_scalar = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer Scalar"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solver_params = SolverParams {
            n: num_cells,
            num_groups,
            padding: [0; 2],
        };
        let b_solver_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Solver Params Buffer"),
            contents: bytemuck::bytes_of(&solver_params),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Bind Group Layouts

        // Group 0: Mesh (Read Only)
        let bgl_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Group 1: Fields (Read/Write)
        let bgl_fields = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fields Bind Group Layout"),
            entries: &[
                // 0: U
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: P
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: Fluxes
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Constants
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Grad P
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: D_P
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: Grad Component
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: U Old (for under-relaxation)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: Grad P Prime
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9: U Old Old
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Group 2: Solver (Read/Write)
        let bgl_solver = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Solver Bind Group Layout"),
            entries: &[
                // 0: Matrix Values
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: RHS
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: X
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: R
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: P_Solver
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: V
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: S
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: T
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Linear Solver Layouts
        let bgl_linear_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear Matrix Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bgl_linear_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear State Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bgl_linear_state_ro =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Linear State RO Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bgl_dot_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dot Params Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bgl_dot_inputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dot Inputs Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bgl_dot_pair_inputs =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Dot Pair Inputs Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Bind Group"),
            layout: &bgl_mesh,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_face_owner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_face_neighbor.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_face_areas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_face_normals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_cell_centers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_cell_vols.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_cell_face_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_cell_faces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: b_cell_face_matrix_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: b_diagonal_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: b_face_boundary.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: b_face_centers.as_entire_binding(),
                },
            ],
        });

        let bg_fields = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fields Bind Group"),
            layout: &bgl_fields,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_fluxes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_constants.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_grad_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_d_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_grad_component.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_u_old.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: b_grad_p_prime.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: b_u_old_old.as_entire_binding(),
                },
            ],
        });

        let bg_solver = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Solver Bind Group"),
            layout: &bgl_solver,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_matrix_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_p_solver.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_s.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_t.as_entire_binding(),
                },
            ],
        });

        let bg_dot_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Params Bind Group"),
            layout: &bgl_dot_params,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b_solver_params.as_entire_binding(),
            }],
        });

        let bg_linear_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Linear Matrix Bind Group"),
            layout: &bgl_linear_matrix,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_matrix_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_scalars.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_solver_params.as_entire_binding(),
                },
            ],
        });

        let bg_linear_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Linear State Bind Group"),
            layout: &bgl_linear_state,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_p_solver.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_s.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_t.as_entire_binding(),
                },
            ],
        });

        let bg_linear_state_ro = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Linear State RO Bind Group"),
            layout: &bgl_linear_state_ro,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_p_solver.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_s.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_t.as_entire_binding(),
                },
            ],
        });

        let bg_dot_r0_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot R0 V Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_dot_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_r0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_v.as_entire_binding(),
                },
            ],
        });

        let bg_dot_p_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot P V Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_dot_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_p_solver.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_v.as_entire_binding(),
                },
            ],
        });

        let bg_dot_r_r = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot R R Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_dot_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_r.as_entire_binding(),
                },
            ],
        });

        let bg_dot_pair_r0r_rr = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Pair R0R & RR Bind Group"),
            layout: &bgl_dot_pair_inputs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_dot_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_dot_result_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_r0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_r.as_entire_binding(),
                },
            ],
        });

        let bg_dot_pair_tstt = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Pair TS & TT Bind Group"),
            layout: &bgl_dot_pair_inputs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_dot_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_dot_result_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_t.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_s.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_t.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_t.as_entire_binding(),
                },
            ],
        });

        // Shaders
        let shader_gradient = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gradient Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/gradient.wgsl"
            ))),
        });

        let pl_gradient = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient Pipeline Layout"),
            bind_group_layouts: &[&bgl_mesh, &bgl_fields], // Needs mesh and fields
            push_constant_ranges: &[],
        });

        let pl_mesh_fields_state = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Fields State Pipeline Layout"),
            bind_group_layouts: &[&bgl_mesh, &bgl_fields, &bgl_linear_state_ro],
            push_constant_ranges: &[],
        });

        let pipeline_gradient = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gradient Pipeline"),
            layout: Some(&pl_gradient),
            module: &shader_gradient,
            entry_point: "main",
        });

        let pl_matrix = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Assembly Pipeline Layout"),
            bind_group_layouts: &[&bgl_mesh, &bgl_fields, &bgl_solver],
            push_constant_ranges: &[],
        });

        let shader_linear = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Linear Solver Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/linear_solver.wgsl"
            ))),
        });

        let bgl_empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Empty Bind Group Layout"),
            entries: &[],
        });

        let bg_empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Empty Bind Group"),
            layout: &bgl_empty,
            entries: &[],
        });

        let pl_linear = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Linear Solver Pipeline Layout"),
            bind_group_layouts: &[&bgl_mesh, &bgl_linear_state, &bgl_linear_matrix],
            push_constant_ranges: &[],
        });

        let pl_dot = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dot Product Pipeline Layout"),
            bind_group_layouts: &[&bgl_dot_params, &bgl_dot_inputs],
            push_constant_ranges: &[],
        });

        let pl_dot_pair = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dot Pair Pipeline Layout"),
            bind_group_layouts: &[&bgl_dot_params, &bgl_dot_pair_inputs],
            push_constant_ranges: &[],
        });

        let pipeline_spmv_p_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPMV P->V Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "spmv_p_v",
        });

        let pipeline_spmv_s_t = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPMV S->T Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "spmv_s_t",
        });

        let shader_dot = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dot Product Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/dot_product.wgsl"
            ))),
        });

        let pipeline_dot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dot Product Pipeline"),
            layout: Some(&pl_dot),
            module: &shader_dot,
            entry_point: "main",
        });

        let shader_dot_pair = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dot Pair Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/dot_product_pair.wgsl"
            ))),
        });

        let pipeline_dot_pair = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dot Pair Pipeline"),
            layout: Some(&pl_dot_pair),
            module: &shader_dot_pair,
            entry_point: "main",
        });

        let pipeline_bicgstab_update_x_r =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BiCGStab Update X and R Pipeline"),
                layout: Some(&pl_linear),
                module: &shader_linear,
                entry_point: "bicgstab_update_x_r",
            });

        let pipeline_bicgstab_update_p =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BiCGStab Update P Pipeline"),
                layout: Some(&pl_linear),
                module: &shader_linear,
                entry_point: "bicgstab_update_p",
            });

        let pipeline_bicgstab_update_s =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BiCGStab Update S Pipeline"),
                layout: Some(&pl_linear),
                module: &shader_linear,
                entry_point: "bicgstab_update_s",
            });

        let pipeline_cg_update_x_r =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CG Update X R Pipeline"),
                layout: Some(&pl_linear),
                module: &shader_linear,
                entry_point: "cg_update_x_r",
            });

        let pipeline_cg_update_p =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CG Update P Pipeline"),
                layout: Some(&pl_linear),
                module: &shader_linear,
                entry_point: "cg_update_p",
            });

        let shader_flux = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flux Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/flux.wgsl"))),
        });

        let pipeline_flux = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flux Pipeline"),
            layout: Some(&pl_gradient),
            module: &shader_flux,
            entry_point: "main",
        });

        let shader_momentum = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Momentum Assembly Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/momentum_assembly_v2.wgsl"
            ))),
        });

        let pipeline_momentum_assembly =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Momentum Assembly Pipeline"),
                layout: Some(&pl_matrix),
                module: &shader_momentum,
                entry_point: "main",
            });

        let shader_pressure = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure Assembly Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/pressure_assembly.wgsl"
            ))),
        });

        let pipeline_pressure_assembly =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pressure Assembly Pipeline"),
                layout: Some(&pl_matrix),
                module: &shader_pressure,
                entry_point: "main",
            });

        // Combined pressure assembly + gradient shader (optimization)
        let shader_pressure_with_grad = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure Assembly With Grad Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/pressure_assembly_with_grad.wgsl"
            ))),
        });

        let pipeline_pressure_assembly_with_grad =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pressure Assembly With Grad Pipeline"),
                layout: Some(&pl_matrix),
                module: &shader_pressure_with_grad,
                entry_point: "main",
            });

        let shader_flux_rhie_chow = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flux Rhie-Chow Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/flux_rhie_chow.wgsl"
            ))),
        });

        let pipeline_flux_rhie_chow =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flux Rhie-Chow Pipeline"),
                layout: Some(&pl_gradient), // Uses mesh and fields (Group 0, 1)
                module: &shader_flux_rhie_chow,
                entry_point: "main",
            });

        let shader_velocity_correction =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Velocity Correction Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/velocity_correction.wgsl"
                ))),
            });

        let pipeline_velocity_correction =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Velocity Correction Pipeline"),
                layout: Some(&pl_mesh_fields_state),
                module: &shader_velocity_correction,
                entry_point: "main",
            });

        let shader_update_u = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Update U Component Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/update_u_component.wgsl"
            ))),
        });

        let pipeline_update_u_component =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Update U Component Pipeline"),
                layout: Some(&pl_mesh_fields_state),
                module: &shader_update_u,
                entry_point: "main",
            });

        // Scalar Pipelines
        let shader_scalars = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scalars Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/scalars.wgsl"
            ))),
        });

        let bgl_scalars = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scalars Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bg_scalars = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scalars Bind Group"),
            layout: &bgl_scalars,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_scalars.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_dot_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_dot_result_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_solver_params.as_entire_binding(),
                },
            ],
        });

        let pl_scalars = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scalars Pipeline Layout"),
            bind_group_layouts: &[&bgl_scalars],
            push_constant_ranges: &[],
        });

        let pipeline_init_scalars =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Init Scalars Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "init_scalars",
            });

        let pipeline_init_cg_scalars =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Init CG Scalars Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "init_cg_scalars",
            });

        let pipeline_reduce_rho_new_r_r =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Reduce Rho New R R Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "reduce_rho_new_r_r",
            });

        let pipeline_reduce_r0_v =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Reduce R0 V Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "reduce_r0_v",
            });

        let pipeline_reduce_t_s_t_t =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Reduce T S T T Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "reduce_t_s_t_t",
            });

        let pipeline_update_cg_alpha =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Update CG Alpha Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "update_cg_alpha",
            });

        let pipeline_update_cg_beta =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Update CG Beta Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "update_cg_beta",
            });

        let pipeline_update_rho_old =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Update Rho Old Pipeline"),
                layout: Some(&pl_scalars),
                module: &shader_scalars,
                entry_point: "update_rho_old",
            });

        Self {
            context,
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
            b_row_offsets,
            b_col_indices,
            num_nonzeros,
            b_matrix_values,
            b_rhs,
            b_x,
            b_r,
            b_r0,
            b_p_solver,
            b_v,
            b_s,
            b_t,
            b_dot_result,
            b_dot_result_2,
            b_scalars,
            b_staging_scalar,
            b_solver_params,

            b_constants,
            b_u,
            b_u_old,
            b_u_old_old,
            b_p,
            b_p_old,
            b_d_p,
            b_fluxes,
            b_grad_p,
            b_grad_component,
            b_grad_p_prime,
            bg_mesh,
            bg_fields,
            bg_solver,
            bg_linear_matrix,
            bg_linear_state,
            bg_linear_state_ro,
            bg_dot_params,
            bg_dot_r0_v,
            bg_dot_p_v,
            bg_dot_r_r,
            bg_dot_pair_r0r_rr,
            bg_dot_pair_tstt,
            bg_scalars,
            bg_empty,
            pipeline_gradient,
            pipeline_spmv_p_v,
            pipeline_spmv_s_t,
            pipeline_dot,
            pipeline_dot_pair,
            pipeline_bicgstab_update_x_r,
            pipeline_bicgstab_update_p,
            pipeline_bicgstab_update_s,
            pipeline_cg_update_x_r,
            pipeline_cg_update_p,
            pipeline_init_scalars,
            pipeline_init_cg_scalars,
            pipeline_update_cg_alpha,
            pipeline_update_cg_beta,
            pipeline_update_rho_old,
            pipeline_reduce_rho_new_r_r,
            pipeline_reduce_r0_v,
            pipeline_reduce_t_s_t_t,
            pipeline_flux,
            pipeline_momentum_assembly,
            pipeline_pressure_assembly,
            pipeline_pressure_assembly_with_grad,
            pipeline_flux_rhie_chow,
            pipeline_velocity_correction,
            pipeline_update_u_component,
            num_cells,
            num_faces,
            constants,
            profiling_enabled: AtomicBool::new(false),
            time_compute: Mutex::new(std::time::Duration::new(0, 0)),
            time_spmv: Mutex::new(std::time::Duration::new(0, 0)),
            time_dot: Mutex::new(std::time::Duration::new(0, 0)),
            stats_ux: Mutex::new(Default::default()),
            stats_uy: Mutex::new(Default::default()),
            stats_p: Mutex::new(Default::default()),
            outer_residual_u: Mutex::new(0.0),
            outer_residual_p: Mutex::new(0.0),
            outer_iterations: Mutex::new(0),
        }
    }
}
