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
@group(1) @binding(2) var<uniform> params: AmgParams;

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

// r_coarse = R * (b - A * x)
// Run on coarse threads (one per coarse row)
@compute @workgroup_size(64)
fn restrict_residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x; // Coarse row index
    if (i >= params.n) { // params.n is COLUMNS of R (coarse size)
        return;
    }

    let start = op_row_offsets[i];
    let end = op_row_offsets[i + 1u];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let fine_idx = op_col_indices[k]; // Row of A
        let r_val = op_values[k];
        
        // Compute residual for fine_idx directly here
        // r_fine = b[fine_idx] - A * x
        
        // A row for fine_idx
        let a_start = row_offsets[fine_idx];
        let a_end = row_offsets[fine_idx + 1u];
        
        var ax = 0.0;
        for (var j = a_start; j < a_end; j++) {
            ax += values[j] * x[col_indices[j]];
        }
        
        let fine_r = b[fine_idx] - ax;
        sum += r_val * fine_r;
    }
    
    coarse_vec[i] = sum;
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }
    x[i] = 0.0;
}
