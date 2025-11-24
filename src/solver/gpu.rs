use wgpu::util::DeviceExt;
use crate::solver::mesh::{Mesh, BoundaryType};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuConstants {
    dt: f32,
    time: f32,
    viscosity: f32,
    density: f32,
    component: u32, // 0: x, 1: y
    alpha_p: f32,   // Pressure relaxation
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SolverParams {
    n: u32,
    num_groups: u32,
    padding: [u32; 2],
}

pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    
    // Mesh buffers
    b_face_owner: wgpu::Buffer,
    b_face_neighbor: wgpu::Buffer,
    b_face_boundary: wgpu::Buffer, // Added
    b_face_areas: wgpu::Buffer,
    b_face_normals: wgpu::Buffer,
    b_face_centers: wgpu::Buffer, // Added
    b_cell_centers: wgpu::Buffer,
    b_cell_vols: wgpu::Buffer,
    
    // Connectivity
    b_cell_face_offsets: wgpu::Buffer,
    b_cell_faces: wgpu::Buffer,
    b_cell_face_matrix_indices: wgpu::Buffer, // Maps cell_face to matrix index
    b_diagonal_indices: wgpu::Buffer, // Maps cell to matrix index of diagonal
    
    // Field buffers
    b_u: wgpu::Buffer,      // Velocity (Vector2)
    b_p: wgpu::Buffer,      // Pressure (Scalar)
    b_d_p: wgpu::Buffer,    // Inverse diagonal (Scalar)
    b_fluxes: wgpu::Buffer, // Face fluxes
    b_grad_p: wgpu::Buffer, // Pressure Gradient
    
    // Matrix Structure (CSR)
    b_row_offsets: wgpu::Buffer,
    b_col_indices: wgpu::Buffer,
    num_nonzeros: u32,
    
    // Linear Solver Buffers
    b_matrix_values: wgpu::Buffer,
    b_rhs: wgpu::Buffer,
    b_x: wgpu::Buffer,
    b_r: wgpu::Buffer,
    b_r0: wgpu::Buffer, // Added
    b_p_solver: wgpu::Buffer, // p vector in BiCGStab
    b_v: wgpu::Buffer,
    b_s: wgpu::Buffer,
    b_t: wgpu::Buffer,
    
    // Dot Product & Params
    b_dot_result: wgpu::Buffer,
    b_dot_result_2: wgpu::Buffer,
    b_scalars: wgpu::Buffer,
    b_staging_scalar: wgpu::Buffer, // Added
    b_solver_params: wgpu::Buffer,
    
    // Constants
    b_constants: wgpu::Buffer,

    // Bind Groups & Pipelines
    bg_mesh: wgpu::BindGroup,
    bg_fields: wgpu::BindGroup,
    bg_solver: wgpu::BindGroup,
    
    // Linear Solver Bind Groups
    bg_linear_matrix: wgpu::BindGroup,
    bg_linear_state: wgpu::BindGroup,
    bg_linear_state_ro: wgpu::BindGroup, // Added
    bg_dot_r0_r: wgpu::BindGroup,
    bg_dot_r0_v: wgpu::BindGroup,
    bg_dot_t_s: wgpu::BindGroup,
    bg_dot_t_t: wgpu::BindGroup,
    bg_dot_r_r: wgpu::BindGroup,
    bg_scalars: wgpu::BindGroup,
    bg_empty: wgpu::BindGroup,
    
    pipeline_gradient: wgpu::ComputePipeline,
    // pipeline_matrix_assembly: wgpu::ComputePipeline, // Removed
    pipeline_spmv_p_v: wgpu::ComputePipeline,
    pipeline_spmv_s_t: wgpu::ComputePipeline,
    pipeline_dot: wgpu::ComputePipeline,
    pipeline_bicgstab_update_x_r: wgpu::ComputePipeline,
    pipeline_bicgstab_update_p: wgpu::ComputePipeline,
    pipeline_bicgstab_update_s: wgpu::ComputePipeline,

    // Scalar Pipelines
    pipeline_init_scalars: wgpu::ComputePipeline,
    pipeline_reduce_rho_new_r_r: wgpu::ComputePipeline,
    pipeline_update_beta: wgpu::ComputePipeline,
    pipeline_update_alpha: wgpu::ComputePipeline,
    pipeline_reduce_r0_v: wgpu::ComputePipeline,
    pipeline_reduce_t_s_t_t: wgpu::ComputePipeline,
    pipeline_update_omega: wgpu::ComputePipeline,
    pipeline_update_rho_old: wgpu::ComputePipeline,

    pipeline_flux: wgpu::ComputePipeline, // Added
    pipeline_momentum_assembly: wgpu::ComputePipeline,
    pipeline_pressure_assembly: wgpu::ComputePipeline,
    pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pipeline_velocity_correction: wgpu::ComputePipeline,
    pipeline_flux_correction: wgpu::ComputePipeline,
    pipeline_update_p_field: wgpu::ComputePipeline,
    pipeline_compute_d_p: wgpu::ComputePipeline,
    pipeline_update_u_component: wgpu::ComputePipeline,
    
    bg_fields_prime: wgpu::BindGroup,
    
    num_cells: u32,
    num_faces: u32,
    
    constants: GpuConstants,
    
    // Profiling
    profiling_enabled: AtomicBool,
    time_compute: Mutex<std::time::Duration>,
    time_spmv: Mutex<std::time::Duration>,
    time_dot: Mutex<std::time::Duration>,
}

impl GpuSolver {
    pub async fn new(mesh: &Mesh) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 16,
                ..wgpu::Limits::downlevel_defaults()
            },
        }, None).await.unwrap();

        let num_cells = mesh.cell_cx.len() as u32;
        let num_faces = mesh.face_owner.len() as u32;

        // --- Mesh Buffers ---
        let face_owner: Vec<u32> = mesh.face_owner.iter().map(|&x| x as u32).collect();
        let b_face_owner = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Face Owner Buffer"),
            contents: bytemuck::cast_slice(&face_owner),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let face_neighbor: Vec<u32> = mesh.face_neighbor.iter().map(|&x| {
            match x {
                Some(n) => n as u32,
                None => u32::MAX,
            }
        }).collect();
        let b_face_neighbor = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Face Neighbor Buffer"),
            contents: bytemuck::cast_slice(&face_neighbor),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let face_boundary: Vec<u32> = mesh.face_boundary.iter().map(|b| {
            match b {
                None => 0,
                Some(BoundaryType::Inlet) => 1,
                Some(BoundaryType::Outlet) => 2,
                Some(BoundaryType::Wall) => 3,
                Some(BoundaryType::ParallelInterface(_, _)) => 4,
            }
        }).collect();
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

        let face_normals: Vec<[f32; 2]> = mesh.face_nx.iter().zip(mesh.face_ny.iter())
            .map(|(&nx, &ny)| [nx as f32, ny as f32]).collect();
        let b_face_normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Face Normals Buffer"),
            contents: bytemuck::cast_slice(&face_normals),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let face_centers: Vec<[f32; 2]> = mesh.face_cx.iter().zip(mesh.face_cy.iter())
            .map(|(&cx, &cy)| [cx as f32, cy as f32]).collect();
        let b_face_centers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Face Centers Buffer"),
            contents: bytemuck::cast_slice(&face_centers),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cell_centers: Vec<[f32; 2]> = mesh.cell_cx.iter().zip(mesh.cell_cy.iter())
            .map(|(&cx, &cy)| [cx as f32, cy as f32]).collect();
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

        // --- Field Buffers ---
        let zero_vecs = vec![[0.0f32; 2]; num_cells as usize];
        let b_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("U Buffer"),
            contents: bytemuck::cast_slice(&zero_vecs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let zero_scalars = vec![0.0f32; num_cells as usize];
        let b_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("P Buffer"),
            contents: bytemuck::cast_slice(&zero_scalars),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let b_d_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("D_P Buffer"),
            contents: bytemuck::cast_slice(&zero_scalars),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let zero_fluxes = vec![0.0f32; num_faces as usize];
        let b_fluxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluxes Buffer"),
            contents: bytemuck::cast_slice(&zero_fluxes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        
        let b_grad_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grad P Buffer"),
            contents: bytemuck::cast_slice(&zero_vecs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let constants = GpuConstants {
            dt: 0.001, // Default
            time: 0.0,
            viscosity: 0.01,
            density: 1.0,
            component: 0,
            alpha_p: 0.1, // Default pressure relaxation
            padding: [0; 2],
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_rhs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RHS Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("X Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_r = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("R Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_r0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("R0 Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_p_solver = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P Solver Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Buffer"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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
        let num_groups = (num_cells + workgroup_size - 1) / workgroup_size;
        let b_dot_result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Result Buffer"),
            size: (num_groups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_dot_result_2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Result Buffer 2"),
            size: (num_groups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_scalars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scalars Buffer"),
            size: 64, 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_staging_scalar = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer Scalar"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solver_params = SolverParams { n: num_cells, num_groups, padding: [0; 2] };
        let b_solver_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Solver Params Buffer"),
            contents: bytemuck::bytes_of(&solver_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Bind Group Layouts
        
        // Group 0: Mesh (Read Only)
        let bgl_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 11, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 12, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 13, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Group 1: Fields (Read/Write)
        let bgl_fields = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fields Bind Group Layout"),
            entries: &[
                // 0: U
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 1: P
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 2: Fluxes
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 3: Constants
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 4: Grad P
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 5: D_P
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Group 2: Solver (Read/Write)
        let bgl_solver = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Solver Bind Group Layout"),
            entries: &[
                // 0: Matrix Values
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 1: RHS
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 2: X
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 3: R
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 4: P_Solver
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 5: V
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 6: S
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 7: T
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Linear Solver Layouts
        let bgl_linear_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear Matrix Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bgl_linear_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear State Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bgl_linear_state_ro = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear State RO Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bgl_dot_inputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dot Inputs Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Bind Group"),
            layout: &bgl_mesh,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_face_owner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_face_neighbor.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_face_areas.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_face_normals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_cell_centers.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_cell_vols.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b_cell_face_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: b_cell_faces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: b_row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: b_col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: b_cell_face_matrix_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: b_diagonal_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: b_face_boundary.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: b_face_centers.as_entire_binding() },
            ],
        });

        let bg_fields = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fields Bind Group"),
            layout: &bgl_fields,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_fluxes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_constants.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_grad_p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_d_p.as_entire_binding() },
            ],
        });

        let bg_solver = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Solver Bind Group"),
            layout: &bgl_solver,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_matrix_values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_rhs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_p_solver.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b_s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: b_t.as_entire_binding() },
            ],
        });

        let bg_linear_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Linear Matrix Bind Group"),
            layout: &bgl_linear_matrix,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_matrix_values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_scalars.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_solver_params.as_entire_binding() },
            ],
        });

        let bg_linear_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Linear State Bind Group"),
            layout: &bgl_linear_state,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_p_solver.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_t.as_entire_binding() },
            ],
        });

        let bg_linear_state_ro = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Linear State RO Bind Group"),
            layout: &bgl_linear_state_ro,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_p_solver.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_t.as_entire_binding() },
            ],
        });

        let bg_dot_r0_r = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Product R0 R Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_dot_result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_r0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_r.as_entire_binding() },
            ],
        });

        let bg_dot_r0_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Product R0 V Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_dot_result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_r0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_v.as_entire_binding() },
            ],
        });

        let bg_dot_t_s = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Product T S Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_dot_result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_s.as_entire_binding() },
            ],
        });

        let bg_dot_t_t = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Product T T Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_dot_result_2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_t.as_entire_binding() },
            ],
        });

        let bg_dot_r_r = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Product R R Bind Group"),
            layout: &bgl_dot_inputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_dot_result_2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_r.as_entire_binding() },
            ],
        });

        // Shaders
        let shader_gradient = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gradient Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/gradient.wgsl"))),
        });

        let pl_gradient = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient Pipeline Layout"),
            bind_group_layouts: &[&bgl_mesh, &bgl_fields], // Needs mesh and fields
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/linear_solver.wgsl"))),
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
            bind_group_layouts: &[&bgl_mesh, &bgl_linear_state_ro, &bgl_linear_matrix, &bgl_dot_inputs],
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/dot_product.wgsl"))),
        });

        let pipeline_dot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dot Product Pipeline"),
            layout: Some(&pl_dot),
            module: &shader_dot,
            entry_point: "main",
        });

        let pipeline_bicgstab_update_x_r = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BiCGStab Update X and R Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "bicgstab_update_x_r",
        });

        let pipeline_bicgstab_update_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BiCGStab Update P Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "bicgstab_update_p",
        });

        let pipeline_bicgstab_update_s = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BiCGStab Update S Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "bicgstab_update_s",
        });

        let shader_flux = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flux Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/flux.wgsl"))),
        });

        let pipeline_flux = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flux Pipeline"),
            layout: Some(&pl_gradient),
            module: &shader_flux,
            entry_point: "main",
        });

        let shader_momentum = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Momentum Assembly Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/momentum_assembly.wgsl"))),
        });

        let pipeline_momentum_assembly = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Momentum Assembly Pipeline"),
            layout: Some(&pl_matrix),
            module: &shader_momentum,
            entry_point: "main",
        });

        let shader_pressure = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure Assembly Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/pressure_assembly.wgsl"))),
        });

        let pipeline_pressure_assembly = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pressure Assembly Pipeline"),
            layout: Some(&pl_matrix),
            module: &shader_pressure,
            entry_point: "main",
        });

        let shader_flux_rhie_chow = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flux Rhie-Chow Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/flux_rhie_chow.wgsl"))),
        });

        let pipeline_flux_rhie_chow = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flux Rhie-Chow Pipeline"),
            layout: Some(&pl_gradient), // Uses mesh and fields (Group 0, 1)
            module: &shader_flux_rhie_chow,
            entry_point: "main",
        });

        let shader_velocity_correction = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Velocity Correction Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/velocity_correction.wgsl"))),
        });

        let pipeline_velocity_correction = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Velocity Correction Pipeline"),
            layout: Some(&pl_gradient), // Uses mesh and fields
            module: &shader_velocity_correction,
            entry_point: "main",
        });

        let shader_flux_correction = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flux Correction Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/flux_correction.wgsl"))),
        });

        let pl_flux_correction = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flux Correction Pipeline Layout"),
            bind_group_layouts: &[&bgl_mesh, &bgl_fields, &bgl_linear_state_ro], // Group 0, 1, 2 (x is in state_ro)
            push_constant_ranges: &[],
        });

        let pipeline_flux_correction = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flux Correction Pipeline"),
            layout: Some(&pl_flux_correction),
            module: &shader_flux_correction,
            entry_point: "main",
        });

        let shader_update_p = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Update P Field Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/update_p_field.wgsl"))),
        });

        let pipeline_update_p_field = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update P Field Pipeline"),
            layout: Some(&pl_flux_correction), // Uses fields and state_ro (x)
            module: &shader_update_p,
            entry_point: "main",
        });

        let shader_compute_d_p = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute D_P Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/compute_d_p.wgsl"))),
        });

        let pl_compute_d_p = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute D_P Pipeline Layout"),
            bind_group_layouts: &[&bgl_mesh, &bgl_fields, &bgl_solver], // Group 0, 1, 2 (matrix)
            push_constant_ranges: &[],
        });

        let pipeline_compute_d_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute D_P Pipeline"),
            layout: Some(&pl_compute_d_p),
            module: &shader_compute_d_p,
            entry_point: "main",
        });

        let shader_update_u = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Update U Component Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/update_u_component.wgsl"))),
        });

        let pipeline_update_u_component = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update U Component Pipeline"),
            layout: Some(&pl_flux_correction), // Uses fields and state_ro (x)
            module: &shader_update_u,
            entry_point: "main",
        });

        let bg_fields_prime = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fields Prime Bind Group"),
            layout: &bgl_fields,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_x.as_entire_binding() }, // Use X as P
                wgpu::BindGroupEntry { binding: 2, resource: b_fluxes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_constants.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_grad_p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_d_p.as_entire_binding() },
            ],
        });

        // Scalar Pipelines
        let shader_scalars = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scalars Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/scalars.wgsl"))),
        });

        let bgl_scalars = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scalars Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bg_scalars = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scalars Bind Group"),
            layout: &bgl_scalars,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_scalars.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_dot_result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_dot_result_2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_solver_params.as_entire_binding() },
            ],
        });

        let pl_scalars = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scalars Pipeline Layout"),
            bind_group_layouts: &[&bgl_scalars],
            push_constant_ranges: &[],
        });

        let pipeline_init_scalars = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Init Scalars Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "init_scalars",
        });

        let pipeline_reduce_rho_new_r_r = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce Rho New R R Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "reduce_rho_new_r_r",
        });

        let pipeline_update_beta = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Beta Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "update_beta",
        });

        let pipeline_update_alpha = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Alpha Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "update_alpha",
        });

        let pipeline_reduce_r0_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce R0 V Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "reduce_r0_v",
        });

        let pipeline_reduce_t_s_t_t = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce T S T T Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "reduce_t_s_t_t",
        });

        let pipeline_update_omega = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Omega Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "update_omega",
        });

        let pipeline_update_rho_old = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Rho Old Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "update_rho_old",
        });

        Self {
            device,
            queue,
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
            b_p,
            b_d_p,
            b_fluxes,
            b_grad_p,
            bg_mesh,
            bg_fields,
            bg_solver,
            bg_linear_matrix,
            bg_linear_state,
            bg_linear_state_ro,
            bg_dot_r0_r,
            bg_dot_r0_v,
            bg_dot_t_s,
            bg_dot_t_t,
            bg_dot_r_r,
            bg_scalars,
            bg_empty,
            pipeline_gradient,
            pipeline_spmv_p_v,
            pipeline_spmv_s_t,
            pipeline_dot,
            pipeline_bicgstab_update_x_r,
            pipeline_bicgstab_update_p,
            pipeline_bicgstab_update_s,
            pipeline_init_scalars,
            pipeline_reduce_rho_new_r_r,
            pipeline_update_beta,
            pipeline_update_alpha,
            pipeline_reduce_r0_v,
            pipeline_reduce_t_s_t_t,
            pipeline_update_omega,
            pipeline_update_rho_old,
            pipeline_flux,
            pipeline_momentum_assembly,
            pipeline_pressure_assembly,
            pipeline_flux_rhie_chow,
            pipeline_velocity_correction,
            pipeline_flux_correction,
            pipeline_update_p_field,
            pipeline_compute_d_p,
            pipeline_update_u_component,
            bg_fields_prime,
            num_cells,
            num_faces,
            constants,
            profiling_enabled: AtomicBool::new(false),
            time_compute: Mutex::new(std::time::Duration::new(0, 0)),
            time_spmv: Mutex::new(std::time::Duration::new(0, 0)),
            time_dot: Mutex::new(std::time::Duration::new(0, 0)),
        }
    }
    pub fn set_u(&self, u: &[(f64, f64)]) {
        let u_f32: Vec<[f32; 2]> = u.iter().map(|&(x, y)| [x as f32, y as f32]).collect();
        self.queue.write_buffer(&self.b_u, 0, bytemuck::cast_slice(&u_f32));
    }

    pub fn set_p(&self, p: &[f64]) {
        let p_f32: Vec<f32> = p.iter().map(|&x| x as f32).collect();
        self.queue.write_buffer(&self.b_p, 0, bytemuck::cast_slice(&p_f32));
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.constants.dt = dt;
        self.update_constants();
    }

    pub fn set_viscosity(&mut self, nu: f32) {
        self.constants.viscosity = nu;
        self.update_constants();
    }

    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        self.constants.alpha_p = alpha_p;
        self.update_constants();
    }

    pub fn set_density(&mut self, rho: f32) {
        self.constants.density = rho;
        self.update_constants();
    }

    fn update_constants(&self) {
        self.queue.write_buffer(&self.b_constants, 0, bytemuck::bytes_of(&self.constants));
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let data = self.read_buffer(&self.b_u, (self.num_cells as u64) * 8).await; // 2 floats * 4 bytes = 8
        let u_f32: &[f32] = bytemuck::cast_slice(&data);
        u_f32.chunks(2).map(|c| (c[0] as f64, c[1] as f64)).collect()
    }

    pub async fn get_grad_p(&self) -> Vec<(f64, f64)> {
        let data = self.read_buffer(&self.b_grad_p, (self.num_cells as u64) * 8).await;
        let u_f32: &[f32] = bytemuck::cast_slice(&data);
        u_f32.chunks(2).map(|c| (c[0] as f64, c[1] as f64)).collect()
    }

    pub async fn get_matrix_values(&self) -> Vec<f64> {
        let data = self.read_buffer(&self.b_matrix_values, (self.num_nonzeros as u64) * 4).await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.iter().map(|&x| x as f64).collect()
    }

    pub async fn get_x(&self) -> Vec<f32> {
        let data = self.read_buffer(&self.b_x, (self.num_cells as u64) * 4).await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.to_vec()
    }

    pub async fn get_p(&self) -> Vec<f64> {
        let data = self.read_buffer(&self.b_p, (self.num_cells as u64) * 4).await;
        let p_f32: &[f32] = bytemuck::cast_slice(&data);
        p_f32.iter().map(|&x| x as f64).collect()
    }

    pub async fn get_rhs(&self) -> Vec<f64> {
        let data = self.read_buffer(&self.b_rhs, (self.num_cells as u64) * 4).await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.iter().map(|&x| x as f64).collect()
    }

    pub async fn get_cell_vols(&self) -> Vec<f32> {
        let data = self.read_buffer(&self.b_cell_vols, (self.num_cells as u64) * 4).await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.to_vec()
    }

    pub async fn get_diagonal_indices(&self) -> Vec<u32> {
        let data = self.read_buffer(&self.b_diagonal_indices, (self.num_cells as u64) * 4).await;
        let vals_u32: &[u32] = bytemuck::cast_slice(&data);
        vals_u32.to_vec()
    }

    pub async fn get_d_p(&self) -> Vec<f64> {
        let data = self.read_buffer(&self.b_d_p, (self.num_cells as u64) * 4).await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.iter().map(|&x| x as f64).collect()
    }

    async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn compute_gradient(&mut self) {
        let workgroup_size = 64;
        let num_groups_cells = (self.num_cells + workgroup_size - 1) / workgroup_size;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Gradient Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Gradient Pass"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline_gradient);
            cpass.set_bind_group(0, &self.bg_mesh, &[]);
            cpass.set_bind_group(1, &self.bg_fields, &[]);
            cpass.dispatch_workgroups(num_groups_cells, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn step(&mut self) {
        let workgroup_size = 64;
        let num_groups_cells = (self.num_cells + workgroup_size - 1) / workgroup_size;
        let num_groups_faces = (self.num_faces + workgroup_size - 1) / workgroup_size;

        // 1. Momentum Predictor
        
        // Compute Gradient P
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Gradient Encoder") });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Gradient Pass"), timestamp_writes: None });
                cpass.set_pipeline(&self.pipeline_gradient);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.dispatch_workgroups(num_groups_cells, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        
        // Solve Ux (Component 0)
        self.solve_momentum(0, num_groups_cells);
        
        // Solve Uy (Component 1)
        self.solve_momentum(1, num_groups_cells);
        
        // 2. Pressure Corrector
        
        // Compute d_p
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute D_P Encoder") });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute D_P Pass"), timestamp_writes: None });
                cpass.set_pipeline(&self.pipeline_compute_d_p);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.set_bind_group(2, &self.bg_solver, &[]);
                cpass.dispatch_workgroups(num_groups_cells, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        
        // PISO Loop
        for _ in 0..2 {
            // Compute Gradient P
            {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Gradient Encoder") });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Gradient Pass"), timestamp_writes: None });
                    cpass.set_pipeline(&self.pipeline_gradient);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
            
            // Flux Rhie-Chow
            {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Flux RC Encoder") });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Flux RC Pass"), timestamp_writes: None });
                    cpass.set_pipeline(&self.pipeline_flux_rhie_chow);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.dispatch_workgroups(num_groups_faces, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
            
            // Assemble Pressure Matrix
            {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Pressure Assembly Encoder") });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Pressure Assembly Pass"), timestamp_writes: None });
                    cpass.set_pipeline(&self.pipeline_pressure_assembly);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.set_bind_group(2, &self.bg_solver, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
            
            // Solve Pressure (p_prime)
            self.zero_buffer(&self.b_x, (self.num_cells as u64) * 4);
            pollster::block_on(self.solve());
            
            // Update P
            {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Update P Encoder") });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Update P Pass"), timestamp_writes: None });
                    cpass.set_pipeline(&self.pipeline_update_p_field);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.set_bind_group(2, &self.bg_linear_state_ro, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
            
            // Compute Grad P Prime
            {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Grad P Prime Encoder") });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Grad P Prime Pass"), timestamp_writes: None });
                    cpass.set_pipeline(&self.pipeline_gradient);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields_prime, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
            
            // Velocity Correction
            {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Velocity Correction Encoder") });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Velocity Correction Pass"), timestamp_writes: None });
                    cpass.set_pipeline(&self.pipeline_velocity_correction);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
            
            // Flux Correction
            {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Flux Correction Encoder") });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Flux Correction Pass"), timestamp_writes: None });
                    cpass.set_pipeline(&self.pipeline_flux_correction);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.set_bind_group(2, &self.bg_linear_state_ro, &[]);
                    cpass.dispatch_workgroups(num_groups_faces, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
        }
        
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn solve_momentum(&mut self, component: u32, num_groups: u32) {
        self.constants.component = component;
        self.update_constants();
        
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Momentum Assembly Encoder") });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Momentum Assembly Pass"), timestamp_writes: None });
                cpass.set_pipeline(&self.pipeline_momentum_assembly);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.set_bind_group(2, &self.bg_solver, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        
        self.zero_buffer(&self.b_x, (self.num_cells as u64) * 4);
        pollster::block_on(self.solve());
        
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Update U Encoder") });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Update U Pass"), timestamp_writes: None });
                cpass.set_pipeline(&self.pipeline_update_u_component);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.set_bind_group(2, &self.bg_linear_state_ro, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
    }
    
    fn zero_buffer(&self, buffer: &wgpu::Buffer, size: u64) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Zero Buffer Encoder") });
        encoder.clear_buffer(buffer, 0, Some(size));
        self.queue.submit(Some(encoder.finish()));
    }

    pub async fn solve(&self) {
        let max_iter = 1000;
        let tol = 1e-6;
        let n = self.num_cells;
        let workgroup_size = 64;
        let num_groups = (n + workgroup_size - 1) / workgroup_size;

        // Initialize r = b - Ax. Since x=0, r = b.
        let size = (n as u64) * 4;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Init Solver Encoder") });
        encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_r, 0, size);
        encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_r0, 0, size);
        encoder.clear_buffer(&self.b_p_solver, 0, None);
        encoder.clear_buffer(&self.b_v, 0, None);
        
        // Init scalars
        self.encode_scalar_compute(&mut encoder, &self.pipeline_init_scalars);
        
        self.queue.submit(Some(encoder.finish()));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Solver Loop Encoder") });
        let mut pending_commands = false;

        for _iter in 0..max_iter {
            // 1. rho_new = (r0, r) -> b_dot_result
            // 2. r_r = (r, r) -> b_dot_result_2
            self.encode_dot_paired(&mut encoder, &self.bg_dot_r0_r, &self.bg_dot_r_r, num_groups);
            
            // Reduce rho_new and r_r
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_rho_new_r_r, num_groups);
            
            pending_commands = true;

            // Check convergence every 50 iterations
            if _iter % 50 == 0 {
                // Submit pending commands before reading back
                if pending_commands {
                    // println!("Submitting batch at iter {}", _iter);
                    let start = if self.profiling_enabled.load(Ordering::Relaxed) { Some(std::time::Instant::now()) } else { None };
                    self.queue.submit(Some(encoder.finish()));
                    if let Some(start) = start {
                        *self.time_compute.lock().unwrap() += start.elapsed();
                    }
                    encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Solver Loop Encoder") });
                    pending_commands = false;
                }

                let r_r = self.read_scalar_r_r().await;
                if r_r.sqrt() < tol {
                    // println!("Converged at iter {} with residual {:e}", _iter, r_r.sqrt());
                    break;
                }
            }
            
            // Update beta
            self.encode_scalar_compute(&mut encoder, &self.pipeline_update_beta);
            
            // p = r + beta * (p - omega * v)
            self.encode_compute(&mut encoder, &self.pipeline_bicgstab_update_p, &self.bg_linear_state, num_groups);
            
            // v = A * p
            self.encode_spmv(&mut encoder, &self.pipeline_spmv_p_v, num_groups);
            
            // r0_v = (r0, v)
            self.encode_dot(&mut encoder, &self.bg_dot_r0_v, num_groups);
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_r0_v, num_groups);
            
            // Update alpha
            self.encode_scalar_compute(&mut encoder, &self.pipeline_update_alpha);
            
            // s = r - alpha * v
            self.encode_compute(&mut encoder, &self.pipeline_bicgstab_update_s, &self.bg_linear_state, num_groups);
            
            // t = A * s
            self.encode_spmv(&mut encoder, &self.pipeline_spmv_s_t, num_groups);
            
            // t_s = (t, s) -> b_dot_result
            // t_t = (t, t) -> b_dot_result_2
            self.encode_dot_paired(&mut encoder, &self.bg_dot_t_s, &self.bg_dot_t_t, num_groups);
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_t_s_t_t, num_groups);
            
            // Update omega
            self.encode_scalar_compute(&mut encoder, &self.pipeline_update_omega);
            
            // x = x + alpha * p + omega * s
            // r = s - omega * t
            self.encode_compute(&mut encoder, &self.pipeline_bicgstab_update_x_r, &self.bg_linear_state, num_groups);
            
            // Update rho_old
            self.encode_scalar_compute(&mut encoder, &self.pipeline_update_rho_old);
            
            pending_commands = true;
        }
        
        if pending_commands {
             self.queue.submit(Some(encoder.finish()));
        }
    }

    fn encode_compute(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, num_groups: u32) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_mesh, &[]);
        cpass.set_bind_group(1, bind_group, &[]); // linear_state
        cpass.set_bind_group(2, &self.bg_linear_matrix, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    fn encode_spmv(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, num_groups: u32) {
        self.encode_compute(encoder, pipeline, &self.bg_linear_state, num_groups);
    }

    fn encode_dot(&self, encoder: &mut wgpu::CommandEncoder, bg_dot_inputs: &wgpu::BindGroup, num_groups: u32) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Dot Pass"), timestamp_writes: None });
        cpass.set_pipeline(&self.pipeline_dot);
        cpass.set_bind_group(0, &self.bg_mesh, &[]);
        cpass.set_bind_group(1, &self.bg_linear_state_ro, &[]);
        cpass.set_bind_group(2, &self.bg_linear_matrix, &[]);
        cpass.set_bind_group(3, bg_dot_inputs, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    fn encode_dot_paired(&self, encoder: &mut wgpu::CommandEncoder, bg1: &wgpu::BindGroup, bg2: &wgpu::BindGroup, num_groups: u32) {
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Dot Pass 1"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline_dot);
            cpass.set_bind_group(0, &self.bg_mesh, &[]);
            cpass.set_bind_group(1, &self.bg_linear_state_ro, &[]);
            cpass.set_bind_group(2, &self.bg_linear_matrix, &[]);
            cpass.set_bind_group(3, bg1, &[]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Dot Pass 2"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline_dot);
            cpass.set_bind_group(0, &self.bg_mesh, &[]);
            cpass.set_bind_group(1, &self.bg_linear_state_ro, &[]);
            cpass.set_bind_group(2, &self.bg_linear_matrix, &[]);
            cpass.set_bind_group(3, bg2, &[]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }
    }

    fn encode_scalar_compute(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Scalar Compute Pass"), timestamp_writes: None });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_scalars, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    fn encode_scalar_reduce(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, _num_groups: u32) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Scalar Reduce Pass"), timestamp_writes: None });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_scalars, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    async fn read_scalar_r_r(&self) -> f32 {
        // Read r_r from b_scalars.
        // Offset of r_r is 8 * 4 = 32 bytes.
        let offset = 32;
        let size = 4;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Read Scalar Encoder") });
        encoder.copy_buffer_to_buffer(&self.b_scalars, offset, &self.b_staging_scalar, 0, size);
        self.queue.submit(Some(encoder.finish()));
        
        let slice = self.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let val = f32::from_ne_bytes(data[0..4].try_into().unwrap());
        drop(data);
        self.b_staging_scalar.unmap();
        
        val
    }

    pub fn enable_profiling(&self, enable: bool) {
        self.profiling_enabled.store(enable, Ordering::Relaxed);
    }

    pub fn get_profiling_data(&self) -> (std::time::Duration, std::time::Duration, std::time::Duration, std::time::Duration) {
        let compute = *self.time_compute.lock().unwrap();
        let spmv = *self.time_spmv.lock().unwrap();
        let dot = *self.time_dot.lock().unwrap();
        (dot, compute, spmv, std::time::Duration::new(0, 0))
    }
}
