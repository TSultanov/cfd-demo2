use crate::solver::gpu::capacity::{create_buffer_with_capacity, CapacityPlan};
use wgpu::util::DeviceExt;

pub struct MatrixResources {
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub b_matrix_values: wgpu::Buffer,
    pub num_nonzeros: u32,
}

/// Allocate the (block-expanded) CSR matrix buffers.
///
/// `num_nonzeros` stays the LOGICAL count; `capacity` only widens the
/// `col_indices`/`matrix_values` allocations (a topology refresh rewrites them
/// in place instead of reallocating; the block CSR is ~S² x the scalar nnz and
/// dominates uploads). `row_offsets` is dof-sized (cell count invariant) and
/// stays exact.
pub fn init_matrix(
    device: &wgpu::Device,
    row_offsets: &[u32],
    col_indices: &[u32],
    capacity: CapacityPlan,
) -> MatrixResources {
    // GUARD: the block-CSR `col_indices`/`matrix_values` are bound ENTIRE
    // (`ResourceRegistry::with_buffer` → `as_entire_buffer_binding`) by every
    // LA-stack consumer (FGMRES, AMG, Schur). Unlike the scalar mesh CSR — whose
    // buffers resolve through `binding_resource_for` as sized ranges — there is
    // no sized-binding path for these named buffers yet, so a capacity headroom
    // would make their WGSL `arrayLength` over-report into the zero-padded tail
    // and SpMV would iterate phantom nnz. Refuse headroom here until the
    // block-CSR named buffers are wired to sized bindings. Exact capacity (the
    // default) is byte-neutral.
    assert!(
        capacity.headroom == 1.0,
        "init_matrix: block-CSR buffers are bound entire; capacity headroom \
         (got {}) would corrupt arrayLength. Wire sized bindings for the \
         matrix_values/col_indices named buffers before enabling headroom (M4).",
        capacity.headroom
    );
    let num_nonzeros = row_offsets.last().cloned().unwrap_or(0);
    let nnz_cap = capacity.capacity_elems(num_nonzeros as usize) as u64;

    let b_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Row Offsets Buffer"),
        contents: bytemuck::cast_slice(row_offsets),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let b_col_indices = create_buffer_with_capacity(
        device,
        "Col Indices Buffer",
        bytemuck::cast_slice(col_indices),
        nnz_cap * 4,
        wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    );

    let b_matrix_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix Values Buffer"),
        size: nnz_cap * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    MatrixResources {
        b_row_offsets,
        b_col_indices,
        b_matrix_values,
        num_nonzeros,
    }
}
