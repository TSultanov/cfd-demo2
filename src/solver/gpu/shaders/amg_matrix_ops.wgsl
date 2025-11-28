// AMG Matrix Operations

// Group 0: Matrix
@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> matrix_values: array<f32>;
@group(0) @binding(3) var<storage, read> inv_diagonal: array<f32>; // For smoother

// Group 1: State (x, b, r)
@group(1) @binding(0) var<storage, read_write> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> r: array<f32>;

// Group 2: Params
struct Params {
    n: u32,
    omega: f32, // Relaxation factor
}
@group(2) @binding(0) var<uniform> params: Params;

// Compute Residual: r = b - Ax
@compute @workgroup_size(64)
fn residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.n) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * x[col];
    }
    
    r[row] = b[row] - sum;
}

// Jacobi Smooth: x_new = x + omega * D^-1 * (b - Ax)
// Note: This is a read-write update on x. For true Jacobi, we need x_old and x_new.
// But for damped Jacobi with small omega, in-place might be acceptable or we use a temp buffer.
// Standard parallel Jacobi requires double buffering.
// Here we assume x is read-write, effectively Gauss-Seidel-ish if race conditions (but random).
// To do it correctly, we should use a temp buffer or just accept the hybrid nature.
// For simplicity, we'll do in-place for now, which is effectively a chaotic relaxation.
@compute @workgroup_size(64)
fn smooth_jacobi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.n) {
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    // Compute Ax
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * x[col];
    }
    
    let resid = b[row] - sum;
    x[row] += params.omega * inv_diagonal[row] * resid;
}

// Restriction: r_c = R * r
// Group 0: R Matrix
// Group 1: Fine State (r is input)
// Group 2: Coarse State (r (actually b_c) is output)
// This requires different bind group layout.
// Let's define specific bind groups for restriction/prolongation.

// Restriction Bind Groups
// Group 0: R Matrix
// Group 1: Input Vector (Fine r)
// Group 2: Params
// Group 3: Output Vector (Coarse b/r)
@group(1) @binding(2) var<storage, read_write> input_r: array<f32>; // Reusing slot 2 from State
@group(3) @binding(1) var<storage, read_write> output_b: array<f32>; // Reusing slot 1 from State (b_c)

@compute @workgroup_size(64)
fn restrict_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Global ID corresponds to Coarse Rows (rows of R)
    let row = global_id.x;
    if (row >= params.n) { // params.n is coarse size
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * input_r[col];
    }
    
    output_b[row] = sum;
}

// Prolongation and Add: x += P * x_c
// Group 0: P Matrix
// Group 1: Input Vector (Coarse x)
// Group 2: Params
// Group 3: Output Vector (Fine x)
@group(1) @binding(0) var<storage, read_write> input_xc: array<f32>; // Reusing slot 0 from State
@group(3) @binding(0) var<storage, read_write> output_x: array<f32>; // Reusing slot 0 from State

@compute @workgroup_size(64)
fn prolongate_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Global ID corresponds to Fine Rows (rows of P)
    let row = global_id.x;
    if (row >= params.n) { // params.n is fine size
        return;
    }

    let start = row_offsets[row];
    let end = row_offsets[row + 1];
    
    var sum = 0.0;
    for (var k = start; k < end; k++) {
        let col = col_indices[k];
        let val = matrix_values[k];
        sum += val * input_xc[col];
    }
    
    output_x[row] += sum;
}
