struct AmgParams {
    n: u32,
    omega: f32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;

@group(1) @binding(0) var<storage, read_write> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> r: array<f32>;
@group(1) @binding(3) var<uniform> params: AmgParams;

// For restriction/prolongation
@group(2) @binding(0) var<storage, read> op_row_offsets: array<u32>;
@group(2) @binding(1) var<storage, read> op_col_indices: array<u32>;
@group(2) @binding(2) var<storage, read> op_values: array<f32>;

// Additional bindings for cross-level operations
@group(3) @binding(0) var<storage, read_write> coarse_vec: array<f32>; // r_coarse or x_coarse

@compute @workgroup_size(64)
fn smooth_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    let start = row_offsets[i];
    let end = row_offsets[i + 1u];
    
    var sigma = 0.0;
    var diag = 1.0;
    
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = values[k];
        if (col == i) {
            diag = val;
        } else {
            sigma += val * x[col];
        }
    }
    
    if (abs(diag) < 1e-14) { diag = 1.0; }
    
    let x_new = (b[i] - sigma) / diag;
    x[i] = mix(x[i], x_new, params.omega);
}

@compute @workgroup_size(64)
fn residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    let start = row_offsets[i];
    let end = row_offsets[i + 1u];
    
    var ax = 0.0;
    for (var k = start; k < end; k++) {
        ax += values[k] * x[col_indices[k]];
    }
    
    r[i] = b[i] - ax;
}

// r_coarse = R * r_fine
// Run on coarse threads (one per coarse row)
@compute @workgroup_size(64)
fn restrict_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x; // Coarse row index
    if (i >= params.n) { // params.n is coarse size here
        return;
    }

    let start = op_row_offsets[i];
    let end = op_row_offsets[i + 1u];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let fine_idx = op_col_indices[k];
        let val = op_values[k];
        // Read from fine residual (bound as 'r' in group 1? No, need flexible binding)
        // We assume group 1 is FINE level state, group 3 is COARSE level state.
        // Wait, if we run on coarse threads, we write to coarse vector.
        // Input is fine vector.
        // Let's assume:
        // Group 1: Fine vector (source) - bound as 'r'
        // Group 3: Coarse vector (dest) - bound as 'coarse_vec'
        sum += val * r[fine_idx];
    }
    
    coarse_vec[i] = sum;
}

// x_fine += P * x_coarse
// Run on fine threads (one per fine row)
@compute @workgroup_size(64)
fn prolongate_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x; // Fine row index
    if (i >= params.n) { // params.n is fine size here
        return;
    }

    let start = op_row_offsets[i];
    let end = op_row_offsets[i + 1u];
    
    var correction = 0.0;
    for (var k = start; k < end; k++) {
        let coarse_idx = op_col_indices[k];
        let val = op_values[k];
        // Read from coarse solution (bound as 'coarse_vec' in group 3)
        correction += val * coarse_vec[coarse_idx];
    }
    
    x[i] += correction;
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }
    x[i] = 0.0;
}
