use crate::solver::mesh::refresh::{mesh_geometry_f32, MeshTopology};
use crate::solver::mesh::Mesh;
use wgpu::util::DeviceExt;

pub struct MeshResources {
    pub b_face_owner: wgpu::Buffer,
    pub b_face_neighbor: wgpu::Buffer,
    pub b_face_boundary: wgpu::Buffer,
    pub b_face_areas: wgpu::Buffer,
    pub b_face_normals: wgpu::Buffer,
    pub b_face_centers: wgpu::Buffer,
    pub b_face_wrap_shift: wgpu::Buffer,
    pub b_cell_centers: wgpu::Buffer,
    pub b_cell_vols: wgpu::Buffer,
    pub b_cell_face_offsets: wgpu::Buffer,
    pub b_cell_faces: wgpu::Buffer,
    pub b_cell_face_matrix_indices: wgpu::Buffer,
    pub b_diagonal_indices: wgpu::Buffer,
    pub b_scalar_row_offsets: wgpu::Buffer,
    pub b_scalar_col_indices: wgpu::Buffer,
    /// ALE mesh face fluxes: per-face volumetric swept rate `V̇_f` (f32,
    /// Volume/Time, owner-signed like `fluxes`). Allocated zero-filled ALWAYS
    /// — non-ALE models never bind it, and ALE models over a static mesh bind
    /// zeros so the mesh-relative subtraction vanishes bitwise (mirrors the
    /// `face_wrap_shift` empty-means-zero convention). Written per step by
    /// the moving-mesh loop (M4) from swept-face geometry.
    pub b_mesh_fluxes: wgpu::Buffer,
    /// ALE volume history `V^n` (f32, cells). Seeded equal to `cell_vols` at
    /// creation and by `seed_volume_history`; rotated by the refresh/ALE-step
    /// seam (single owner — NOT `host_prepare_step`, which runs after new
    /// volumes are uploaded).
    pub b_cell_vols_old: wgpu::Buffer,
    /// ALE volume history `V^{n-1}` (f32, cells); see `b_cell_vols_old`.
    pub b_cell_vols_old_old: wgpu::Buffer,
    pub scalar_row_offsets: Vec<u32>,
    pub scalar_col_indices: Vec<u32>,
    /// Host snapshot of the topology this solver was built on, kept so a
    /// `Geometry`-level `refresh_mesh` can validate the incoming mesh is
    /// topology-identical before overwriting the geometry buffers in place.
    pub topology: MeshTopology,
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
            "face_wrap_shift" => Some(&self.b_face_wrap_shift),
            "cell_centers" => Some(&self.b_cell_centers),
            "cell_vols" => Some(&self.b_cell_vols),
            "cell_face_offsets" => Some(&self.b_cell_face_offsets),
            "cell_faces" => Some(&self.b_cell_faces),
            "cell_face_matrix_indices" => Some(&self.b_cell_face_matrix_indices),
            "diagonal_indices" => Some(&self.b_diagonal_indices),
            "scalar_row_offsets" => Some(&self.b_scalar_row_offsets),
            "scalar_col_indices" => Some(&self.b_scalar_col_indices),
            "mesh_fluxes" => Some(&self.b_mesh_fluxes),
            "cell_vols_old" => Some(&self.b_cell_vols_old),
            "cell_vols_old_old" => Some(&self.b_cell_vols_old_old),
            _ => None,
        }
    }

    /// Tier A geometry-only refresh: overwrite the six geometry buffers
    /// (face areas/normals/centers/wrap shifts, cell centers/volumes) with the
    /// f32 casts of `mesh`'s geometry, produced by the same shared
    /// [`mesh_geometry_f32`] helper `init_mesh` uses — init and refresh can
    /// never drift. Topology must be identical to the build-time mesh
    /// (validated against the stored snapshot; full array compare — see
    /// [`MeshTopology::validate_matches`] for the cost note).
    ///
    /// The buffer objects themselves are unchanged (contents rewritten via
    /// `queue.write_buffer`), so every bind group referencing them stays valid.
    pub fn refresh_geometry(&self, queue: &wgpu::Queue, mesh: &Mesh) -> Result<(), String> {
        self.topology.validate_matches(mesh)?;
        let geo = mesh_geometry_f32(mesh);
        queue.write_buffer(&self.b_face_areas, 0, bytemuck::cast_slice(&geo.face_areas));
        queue.write_buffer(&self.b_face_normals, 0, bytemuck::cast_slice(&geo.face_normals));
        queue.write_buffer(&self.b_face_centers, 0, bytemuck::cast_slice(&geo.face_centers));
        queue.write_buffer(
            &self.b_face_wrap_shift,
            0,
            bytemuck::cast_slice(&geo.face_wrap_shift),
        );
        queue.write_buffer(&self.b_cell_centers, 0, bytemuck::cast_slice(&geo.cell_centers));
        queue.write_buffer(&self.b_cell_vols, 0, bytemuck::cast_slice(&geo.cell_vols));
        Ok(())
    }

    /// Seed the ALE volume history: `cell_vols_old = cell_vols_old_old =
    /// cell_vols` (on-device copies). Called from `initialize_history` so a
    /// geometry refresh before initialization cannot leave stale history; a
    /// no-op numerically for static runs (the buffers are created equal and
    /// only *_ale kernels ever bind them).
    pub fn seed_volume_history(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let size = self.b_cell_vols.size();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ALE volume history seed"),
        });
        encoder.copy_buffer_to_buffer(&self.b_cell_vols, 0, &self.b_cell_vols_old, 0, size);
        encoder.copy_buffer_to_buffer(&self.b_cell_vols, 0, &self.b_cell_vols_old_old, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Return the list of all binding names this resource can resolve.
    pub fn binding_names(&self) -> &'static [&'static str] {
        &[
            "cell_centers",
            "cell_face_matrix_indices",
            "cell_face_offsets",
            "cell_faces",
            "cell_vols",
            "cell_vols_old",
            "cell_vols_old_old",
            "diagonal_indices",
            "face_areas",
            "face_boundary",
            "face_centers",
            "face_neighbor",
            "face_normals",
            "face_owner",
            "face_wrap_shift",
            "mesh_fluxes",
            "scalar_col_indices",
            "scalar_row_offsets",
        ]
    }
}

pub fn init_mesh(device: &wgpu::Device, mesh: &Mesh) -> Result<MeshResources, String> {
    let num_cells = mesh.cell_cx.len() as u32;

    // Shared f64->f32 geometry cast (also consumed by the CPU backend's
    // `upload_mesh` and by `MeshResources::refresh_geometry`): one source of
    // truth so init, refresh, and both backends see bit-identical geometry.
    let geo = mesh_geometry_f32(mesh);

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
            Some(bt) => bt.bc_table_index() as u32,
        })
        .collect();
    let b_face_boundary = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Boundary Buffer"),
        contents: bytemuck::cast_slice(&face_boundary),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // The six geometry buffers carry COPY_DST so a Geometry-level
    // `refresh_mesh` can overwrite them in place via `queue.write_buffer`
    // (usage flags have no effect on results; see `refresh_geometry`).
    let b_face_areas = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Areas Buffer"),
        contents: bytemuck::cast_slice(&geo.face_areas),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_face_normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Normals Buffer"),
        contents: bytemuck::cast_slice(&geo.face_normals),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_face_centers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Centers Buffer"),
        contents: bytemuck::cast_slice(&geo.face_centers),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Periodic wrap shift, one vec2 per face (zero on ordinary faces; an empty
    // mesh field means all-zero, so non-periodic meshes are unaffected).
    let b_face_wrap_shift = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Face Wrap Shift Buffer"),
        contents: bytemuck::cast_slice(&geo.face_wrap_shift),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_cell_centers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Centers Buffer"),
        contents: bytemuck::cast_slice(&geo.cell_centers),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_cell_vols = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Volumes Buffer"),
        contents: bytemuck::cast_slice(&geo.cell_vols),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
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

            // For boundary faces, map to the diagonal entry.
            //
            // Assembly kernels rely on `cell_face_matrix_indices` to produce a valid CSR rank for
            // every (cell, face) pair. Using the diagonal for boundary faces keeps neighbor-rank
            // indexing well-defined and matches the intended "ghost equals owner" convention.
            let target_col = match neighbor {
                Some(n) => n as u32,
                None => i,
            };

            let row_start = scalar_row_offsets[i as usize] as usize;
            let row_end = scalar_row_offsets[i as usize + 1] as usize;
            let cols = &scalar_col_indices[row_start..row_end];

            if let Ok(idx) = cols.binary_search(&target_col) {
                cell_face_matrix_indices.push((row_start + idx) as u32);
            } else {
                // Should not happen: scalar CSR always contains the diagonal and any true neighbor.
                cell_face_matrix_indices.push(u32::MAX);
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
            return Err(format!("diagonal not found in CSR cols for cell {i}"));
        }
    }

    let b_diagonal_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Diagonal Indices Buffer"),
        contents: bytemuck::cast_slice(&diagonal_indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // --- ALE buffers (always allocated; bound only by *_ale model kernels) ---
    // Zero-filled mesh face fluxes: a static mesh has zero swept rate, so an
    // ALE model that never uploads reproduces static physics bitwise.
    let mesh_fluxes = vec![0.0f32; mesh.face_owner.len()];
    let b_mesh_fluxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh Fluxes Buffer (ALE)"),
        contents: bytemuck::cast_slice(&mesh_fluxes),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    // Volume history, seeded equal to the current volumes (COPY_SRC so the
    // old -> old_old rotation can run on-device; COPY_DST for uploads/seeding).
    let vols_history_usage = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST;
    let b_cell_vols_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Volumes Old Buffer (ALE)"),
        contents: bytemuck::cast_slice(&geo.cell_vols),
        usage: vols_history_usage,
    });
    let b_cell_vols_old_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Volumes Old-Old Buffer (ALE)"),
        contents: bytemuck::cast_slice(&geo.cell_vols),
        usage: vols_history_usage,
    });

    Ok(MeshResources {
        b_face_wrap_shift,
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
        b_mesh_fluxes,
        b_cell_vols_old,
        b_cell_vols_old_old,
        scalar_row_offsets,
        scalar_col_indices,
        topology: MeshTopology::from_mesh(mesh),
    })
}
