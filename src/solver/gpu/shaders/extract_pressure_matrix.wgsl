// Extract pressure matrix (A_pp) from coupled block matrix
// Used for AMG setup

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> coupled_matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure_matrix: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= arrayLength(&row_offsets) - 1u) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    let num_neighbors = end - start;
    
    // Calculate start of pressure block rows in coupled matrix
    // See coupled_assembly.wgsl for layout details
    // start_row_0 = 9 * start
    // start_row_2 = start_row_0 + 6 * num_neighbors
    let start_row_2 = 9u * start + 6u * num_neighbors;

    for (var k = 0u; k < num_neighbors; k++) {
        // A_pp is at index 2 in the pressure row block
        let coupled_idx = start_row_2 + 3u * k + 2u;
        let val = coupled_matrix[coupled_idx];
        
        // Write to scalar matrix
        pressure_matrix[start + k] = val;
    }
}

@group(0) @binding(0) var<storage, read> coupled_vector: array<f32>;
@group(0) @binding(1) var<storage, read_write> scalar_vector: array<f32>;

@compute @workgroup_size(64)
fn extract_vector(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&scalar_vector)) {
        return;
    }
    
    // Pressure is at index 3*idx + 2
    scalar_vector[idx] = coupled_vector[3u * idx + 2u];
}

@group(0) @binding(0) var<storage, read> scalar_vector_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> coupled_vector_out: array<f32>;

@compute @workgroup_size(64)
fn insert_vector(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&scalar_vector_in)) {
        return;
    }
    
    // Pressure is at index 3*idx + 2
    coupled_vector_out[3u * idx + 2u] = scalar_vector_in[idx];
}
