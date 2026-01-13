This document defines the **canonical CSR representations** used by the GPU solver and the binding-name contract that kernels rely on.

## Two CSR structures

### 1) Scalar / mesh CSR (cell adjacency)
- **Rows:** `num_cells`
- **Meaning:** graph adjacency of cells (including diagonal).
- **Source:** built once from the mesh topology in `src/solver/gpu/init/mesh.rs`.
- **Binding names (WGSL):**
  - `scalar_row_offsets`: `array<u32>` of length `num_cells + 1`
  - `scalar_col_indices`: `array<u32>` of length `scalar_nnz`

This CSR is also the canonical source for:
- `diagonal_indices`: absolute index of the diagonal entry within `scalar_col_indices` for each cell.
- `cell_face_matrix_indices`: mapping from `(cell, face)` to the **scalar CSR rank** (an absolute index into the scalar CSR row segment) used by assembly kernels.

### 2) DOF / system CSR (expanded by unknowns-per-cell)
- **Rows:** `num_dofs = num_cells * unknowns_per_cell`
- **Meaning:** the linear system CSR used by Krylov solvers and SpMV kernels.
- **Source:** derived deterministically from the scalar CSR via `build_block_csr(...)` in `src/solver/gpu/csr.rs`.
- **Binding names (WGSL):**
  - `row_offsets`: `array<u32>` of length `num_dofs + 1`
  - `col_indices`: `array<u32>` of length `dof_nnz`

## Matrix values layout

The buffer bound as `matrix_values` is sized for the **DOF/system CSR** (`dof_nnz` elements), but kernels may index it using the scalar CSR plus `unknowns_per_cell` by treating it as a block-CSR derived from the scalar adjacency.

For a cell `c`, let:
- `scalar_offset = scalar_row_offsets[c]`
- `num_neighbors = scalar_row_offsets[c + 1] - scalar_offset`
- `block_size = unknowns_per_cell`
- `block_stride = block_size * block_size`
- `row_stride = num_neighbors * block_size`

Then the start offset for the DOF-row corresponding to variable-row `u` in cell `c` is:
`start_row_u = scalar_offset * block_stride + u * row_stride`.

This matches the nested-loop ordering used by `build_block_csr`:
`for cell in cells { for u in 0..block_size { for neighbor_rank in row { for v in 0..block_size { ... }}}}`.

## What to bind where

- Assembly kernels typically consume:
  - mesh topology buffers + `scalar_row_offsets`/`diagonal_indices`/`cell_face_matrix_indices`
  - and write into `matrix_values`/`rhs` using the block-CSR indexing scheme above.
- Apply/SpMV kernels consume:
  - `row_offsets`/`col_indices`/`matrix_values` as a conventional CSR over `num_dofs`.
- Schur helpers consume:
  - scalar CSR for the extracted pressure block (`p_*`) and/or `scalar_row_offsets` for block addressing into `matrix_values`.

If a kernel expects scalar adjacency, it must use `scalar_row_offsets` / `scalar_col_indices` (never `row_offsets` / `col_indices`).
