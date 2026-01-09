pub fn build_block_csr(
    row_offsets: &[u32],
    col_indices: &[u32],
    block_size: u32,
) -> (Vec<u32>, Vec<u32>) {
    let num_rows = row_offsets.len().saturating_sub(1);
    let block_rows = num_rows * block_size as usize;
    let mut block_row_offsets = vec![0u32; block_rows + 1];
    let mut block_col_indices = Vec::new();
    let mut current_offset = 0u32;

    for row in 0..num_rows {
        let start = row_offsets[row] as usize;
        let end = row_offsets[row + 1] as usize;
        let neighbors = &col_indices[start..end];

        for block_row in 0..block_size {
            block_row_offsets[row * block_size as usize + block_row as usize] = current_offset;
            for &neighbor in neighbors {
                for block_col in 0..block_size {
                    block_col_indices.push(neighbor * block_size + block_col);
                }
            }
            current_offset += neighbors.len() as u32 * block_size;
        }
    }
    block_row_offsets[block_rows] = current_offset;

    (block_row_offsets, block_col_indices)
}
