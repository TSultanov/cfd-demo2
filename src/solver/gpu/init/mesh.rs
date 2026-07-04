use crate::solver::gpu::capacity::{create_buffer_with_capacity, sized_binding, CapacityPlan};
use crate::solver::mesh::csr::build_sorted_scalar_csr;
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
    /// Allocation policy the topology-sized buffers were created with
    /// (default = exact size). Kept so a Tier-B refresh can decide whether a
    /// new topology still fits the reserved capacity.
    pub capacity: CapacityPlan,
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

    /// Logical byte length of a topology-sized binding, or `None` for
    /// bindings whose logical size always equals the allocation (cell-sized
    /// buffers — the cell count is invariant under every refresh level).
    ///
    /// Element strides: u32/f32 = 4 bytes, vec2<f32> = 8 bytes.
    fn logical_binding_bytes(&self, name: &str) -> Option<u64> {
        let faces = self.topology.num_faces() as u64;
        let cell_faces = self.topology.cell_faces_len() as u64;
        let nnz = self.scalar_col_indices.len() as u64;
        match name {
            "face_owner" | "face_neighbor" | "face_boundary" | "face_areas" | "mesh_fluxes" => {
                Some(faces * 4)
            }
            "face_normals" | "face_centers" | "face_wrap_shift" => Some(faces * 8),
            "cell_faces" | "cell_face_matrix_indices" => Some(cell_faces * 4),
            "scalar_col_indices" => Some(nnz * 4),
            _ => None,
        }
    }

    /// Resolve a binding name to a **sized** binding resource: topology-sized
    /// buffers are bound as `BufferBinding { offset: 0, size: logical }` so
    /// their WGSL `arrayLength` guards see the logical length even when the
    /// allocation carries capacity headroom (see `gpu::capacity`); cell-sized
    /// buffers bind entire. With the default exact-capacity plan the two
    /// shapes are equivalent (size == full buffer size).
    pub fn binding_resource_for(&self, name: &str) -> Option<wgpu::BindingResource<'_>> {
        let buffer = self.buffer_for_binding_name(name)?;
        Some(match self.logical_binding_bytes(name) {
            Some(bytes) => sized_binding(buffer, bytes),
            None => wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
        })
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

    /// Tier B topology refresh (M2): the incoming mesh keeps the SAME cell
    /// count (invariant) but may have a different face set, adjacency and
    /// boundary classification. Every topology-derived buffer is rebuilt from
    /// `mesh` at its new exact size (per design §1.4: reallocate rather than
    /// pad — bind groups and `arrayLength` guards are size-load-bearing) by
    /// delegating to [`init_mesh`], which is the single source of truth for the
    /// CSR builders and buffer layout. Only the two ALE volume-history buffers
    /// (`cell_vols_old{,_old}`) are carried over untouched — they are
    /// cell-indexed, so a cell keeps its `V^n`/`V^{n-1}` across a topology
    /// change (the cell count is invariant).
    ///
    /// After this returns, every buffer object in `self` (except the two
    /// preserved history buffers) is a FRESH allocation: callers that hold bind
    /// groups over these buffers MUST rebuild them (the topology-refresh path in
    /// `generic_coupled` does exactly that).
    pub fn refresh_topology(&mut self, device: &wgpu::Device, mesh: &Mesh) -> Result<(), String> {
        if mesh.num_cells() != self.topology.num_cells() {
            return Err(format!(
                "topology refresh requires an invariant cell count ({} -> {})",
                self.topology.num_cells(),
                mesh.num_cells()
            ));
        }
        // Rebuild the entire resource set from the new mesh (reuses the CSR
        // builders + buffer layout — init and refresh can never drift), at the
        // SAME capacity plan this solver was built with.
        let mut fresh = init_mesh(device, mesh, self.capacity)?;
        // Preserve the ALE volume history across the topology change: swap the
        // freshly-seeded history buffers OUT of `fresh` (they will be dropped)
        // and our existing ones IN, so `*self = fresh` carries them over.
        std::mem::swap(&mut self.b_cell_vols_old, &mut fresh.b_cell_vols_old);
        std::mem::swap(&mut self.b_cell_vols_old_old, &mut fresh.b_cell_vols_old_old);
        *self = fresh;
        Ok(())
    }

    /// ALE step entry (M3.2): rotate the volume history, upload the new
    /// geometry, upload the (f32-closed) mesh face fluxes — in that order.
    ///
    /// Ordering is the whole point (review F3): the rotation must capture the
    /// CURRENT `cell_vols` as `V^n` **before** `refresh_geometry` overwrites
    /// them with `V^{n+1}`. That is why the rotation lives here, in the
    /// refresh/ALE-step seam, and NOT in `host_prepare_step`
    /// (generic_coupled.rs): `host_prepare_step` runs inside `step()`, i.e.
    /// AFTER the caller's refresh has already uploaded the new volumes —
    /// rotating there would copy `V^{n+1}` into the history and corrupt the
    /// moving-volume ddt. Single owner = this method.
    ///
    /// `mesh_fluxes` are the per-face volumetric swept rates, already
    /// f32-closed against `(V^{n+1}-V^n)/dt` per cell (see
    /// `solver::mesh::ale::swept_mesh_fluxes_closed`); they are uploaded
    /// verbatim (no cast) so the closure survives byte-exactly.
    pub fn begin_ale_step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &Mesh,
        mesh_fluxes: &[f32],
    ) -> Result<(), String> {
        if mesh_fluxes.len() != mesh.num_faces() {
            return Err(format!(
                "begin_ale_step: mesh_fluxes has {} entries, mesh has {} faces",
                mesh_fluxes.len(),
                mesh.num_faces()
            ));
        }
        // 1. Rotate the volume history: old_old <- old, old <- current.
        //    (cell_vols_old carries COPY_SRC|COPY_DST; cell_vols COPY_SRC.)
        let size = self.b_cell_vols.size();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ALE volume history rotate"),
        });
        encoder.copy_buffer_to_buffer(&self.b_cell_vols_old, 0, &self.b_cell_vols_old_old, 0, size);
        encoder.copy_buffer_to_buffer(&self.b_cell_vols, 0, &self.b_cell_vols_old, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
        // 2. Upload the new geometry (validates topology-identity; writes the
        //    new cell_vols = V^{n+1}).
        self.refresh_geometry(queue, mesh)?;
        // 3. Upload the closed mesh fluxes.
        queue.write_buffer(&self.b_mesh_fluxes, 0, bytemuck::cast_slice(mesh_fluxes));
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

pub fn init_mesh(
    device: &wgpu::Device,
    mesh: &Mesh,
    capacity: CapacityPlan,
) -> Result<MeshResources, String> {
    // Shared f64->f32 geometry cast (also consumed by the CPU backend's
    // `upload_mesh` and by `MeshResources::refresh_geometry`): one source of
    // truth so init, refresh, and both backends see bit-identical geometry.
    let geo = mesh_geometry_f32(mesh);

    // --- CSR Matrix Structure (factored builder — Tier B refresh reuses it;
    // byte-equivalence to the historical inlined logic is gated by
    // tests/csr_builder_equivalence_test.rs) ---
    let csr = build_sorted_scalar_csr(mesh)?;
    let scalar_row_offsets = csr.row_offsets;
    let scalar_col_indices = csr.col_indices;
    let diagonal_indices = csr.diagonal_indices;
    let cell_face_matrix_indices = csr.cell_face_matrix_indices;

    // Capacity-reserved element counts for the topology-sized buffers
    // (default plan = exact size; see `gpu::capacity`). Cell-sized buffers
    // are always exact — the cell count is invariant under refresh.
    let faces_cap = capacity.capacity_elems(mesh.num_faces());
    let cell_faces_cap = capacity.capacity_elems(mesh.cell_faces.len());
    let nnz_cap = capacity.capacity_elems(scalar_col_indices.len());

    let b_scalar_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh scalar_row_offsets"),
        contents: bytemuck::cast_slice(&scalar_row_offsets),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_scalar_col_indices = create_buffer_with_capacity(
        device,
        "Mesh scalar_col_indices",
        bytemuck::cast_slice(&scalar_col_indices),
        (nnz_cap * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    // --- Mesh Buffers ---
    let face_owner: Vec<u32> = mesh.face_owner.iter().map(|&x| x as u32).collect();
    let b_face_owner = create_buffer_with_capacity(
        device,
        "Face Owner Buffer",
        bytemuck::cast_slice(&face_owner),
        (faces_cap * 4) as u64,
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
    let b_face_neighbor = create_buffer_with_capacity(
        device,
        "Face Neighbor Buffer",
        bytemuck::cast_slice(&face_neighbor),
        (faces_cap * 4) as u64,
        wgpu::BufferUsages::STORAGE,
    );

    let face_boundary: Vec<u32> = mesh
        .face_boundary
        .iter()
        .map(|b| match b {
            None => 0,
            Some(bt) => bt.bc_table_index() as u32,
        })
        .collect();
    let b_face_boundary = create_buffer_with_capacity(
        device,
        "Face Boundary Buffer",
        bytemuck::cast_slice(&face_boundary),
        (faces_cap * 4) as u64,
        wgpu::BufferUsages::STORAGE,
    );

    // The six geometry buffers carry COPY_DST so a Geometry-level
    // `refresh_mesh` can overwrite them in place via `queue.write_buffer`
    // (usage flags have no effect on results; see `refresh_geometry`).
    let b_face_areas = create_buffer_with_capacity(
        device,
        "Face Areas Buffer",
        bytemuck::cast_slice(&geo.face_areas),
        (faces_cap * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    let b_face_normals = create_buffer_with_capacity(
        device,
        "Face Normals Buffer",
        bytemuck::cast_slice(&geo.face_normals),
        (faces_cap * 8) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    let b_face_centers = create_buffer_with_capacity(
        device,
        "Face Centers Buffer",
        bytemuck::cast_slice(&geo.face_centers),
        (faces_cap * 8) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    // Periodic wrap shift, one vec2 per face (zero on ordinary faces; an empty
    // mesh field means all-zero, so non-periodic meshes are unaffected).
    let b_face_wrap_shift = create_buffer_with_capacity(
        device,
        "Face Wrap Shift Buffer",
        bytemuck::cast_slice(&geo.face_wrap_shift),
        (faces_cap * 8) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

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
    let b_cell_faces = create_buffer_with_capacity(
        device,
        "Cell Faces Buffer",
        bytemuck::cast_slice(&cell_faces),
        (cell_faces_cap * 4) as u64,
        wgpu::BufferUsages::STORAGE,
    );

    // COPY_DST: rewritten in place by a Tier-B topology refresh (and by the
    // M5 GPU-resident regeneration path).
    let b_cell_face_matrix_indices = create_buffer_with_capacity(
        device,
        "Cell Face Matrix Indices Buffer",
        bytemuck::cast_slice(&cell_face_matrix_indices),
        (cell_faces_cap * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    let b_diagonal_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Diagonal Indices Buffer"),
        contents: bytemuck::cast_slice(&diagonal_indices),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // --- ALE buffers (always allocated; bound only by *_ale model kernels) ---
    // Zero-filled mesh face fluxes: a static mesh has zero swept rate, so an
    // ALE model that never uploads reproduces static physics bitwise.
    let mesh_fluxes = vec![0.0f32; mesh.face_owner.len()];
    let b_mesh_fluxes = create_buffer_with_capacity(
        device,
        "Mesh Fluxes Buffer (ALE)",
        bytemuck::cast_slice(&mesh_fluxes),
        (faces_cap * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
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
        capacity,
    })
}
