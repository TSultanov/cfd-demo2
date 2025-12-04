struct Params {
    n: u32,
    num_cells: u32,
    num_iters: u32, // Used for 'j' (current restart index)
    omega: f32,
    dispatch_x: u32,
    max_restart: u32,
    column_offset: u32,
    pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> b_basis: array<f32>;
@group(0) @binding(2) var<storage, read_write> b_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> b_dot_partial: array<f32>;
@group(0) @binding(4) var<storage, read_write> b_hessenberg: array<f32>;

var<workgroup> sdata: array<f32, 64>;

const WORKGROUP_SIZE: u32 = 64u;

// Kernel 1: Calculate partial dot products for all i in 0..=j
// Dispatch: (num_groups_n, j+1, 1)
// Global ID x: group_id_n * 64 + local_id
// Global ID y: i (vector index)
@compute @workgroup_size(64)
fn calc_dots_cgs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let idx = global_id.x;
    let i = global_id.y; // Which basis vector V_i
    let j = params.num_iters; // Current iteration index

    // Basis stride (aligned to 256 bytes)
    // We need to pass stride in params or calculate it.
    // Since we can't easily pass u64 in uniform, we assume it's passed or calculated.
    // But stride depends on alignment.
    // Let's assume we pass stride in params._pad1 or similar, or just calculate it if we know the rule.
    // Better to pass it. Let's use _pad1 for stride_words (stride / 4).
    let stride_words = params.dispatch_x; // Reusing dispatch_x field for stride in words? No, dispatch_x is used for 2D dispatch logic.
    // We need a new field or reuse unused pads.
    // Let's assume we add `stride_words` to Params.
    
    // Wait, we can't easily change Params struct layout without breaking other shaders.
    // But we can use a separate bind group or push constants (not available in WGPU easily).
    // Or just calculate it: (n * 4 + 255) & !255 / 4.
    // But n is in params.
    
    let n = params.n;
    let stride_bytes = (n * 4u + 255u) & 4294967040u; // & !255
    let stride_words_calc = stride_bytes / 4u;
    
    var sum = 0.0;
    if (idx < n) {
        let val_w = b_w[idx];
        let val_v = b_basis[i * stride_words_calc + idx];
        sum = val_w * val_v;
    }

    // Workgroup reduction
    sdata[local_id.x] = sum;
    workgroupBarrier();

    if (local_id.x < 32u) { sdata[local_id.x] += sdata[local_id.x + 32u]; } workgroupBarrier();
    if (local_id.x < 16u) { sdata[local_id.x] += sdata[local_id.x + 16u]; } workgroupBarrier();
    if (local_id.x < 8u) { sdata[local_id.x] += sdata[local_id.x + 8u]; } workgroupBarrier();
    if (local_id.x < 4u) { sdata[local_id.x] += sdata[local_id.x + 4u]; } workgroupBarrier();
    if (local_id.x < 2u) { sdata[local_id.x] += sdata[local_id.x + 2u]; } workgroupBarrier();
    if (local_id.x < 1u) { 
        sdata[local_id.x] += sdata[local_id.x + 1u]; 
        // Write partial sum
        // b_dot_partial is size [num_groups_n * (max_restart + 1)]
        // Index: i * num_groups_n + group_id_n
        let group_id_n = group_id.x; // Since we dispatch (groups, j+1)
        // Wait, if we use dispatch_2d for n, then group_id.x is not linear group index.
        // But here we dispatch (num_groups_linear, j+1).
        // We need to ensure we dispatch correctly.
        
        let out_idx = i * num_groups.x + group_id_n;
        b_dot_partial[out_idx] = sdata[0];
    }
}

// Kernel 2: Reduce partials to final dot products
// Dispatch: (j+1, 1, 1)
@compute @workgroup_size(64)
fn reduce_dots_cgs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let i = group_id.x; // Which vector V_i
    let j = params.num_iters; // Current iteration index
    let num_groups_n = params.dispatch_x; // Passed as param (number of partials per vector)

    var sum = 0.0;
    // Loop over all partials for this vector
    // Each thread sums a chunk
    for (var k = local_id.x; k < num_groups_n; k += 64u) {
        sum += b_dot_partial[i * num_groups_n + k];
    }

    // Workgroup reduction
    sdata[local_id.x] = sum;
    workgroupBarrier();

    if (local_id.x < 32u) { sdata[local_id.x] += sdata[local_id.x + 32u]; } workgroupBarrier();
    if (local_id.x < 16u) { sdata[local_id.x] += sdata[local_id.x + 16u]; } workgroupBarrier();
    if (local_id.x < 8u) { sdata[local_id.x] += sdata[local_id.x + 8u]; } workgroupBarrier();
    if (local_id.x < 4u) { sdata[local_id.x] += sdata[local_id.x + 4u]; } workgroupBarrier();
    if (local_id.x < 2u) { sdata[local_id.x] += sdata[local_id.x + 2u]; } workgroupBarrier();
    if (local_id.x < 1u) { 
        sdata[local_id.x] += sdata[local_id.x + 1u]; 
        
        let max_restart = params.max_restart;
        let h_idx = j * (max_restart + 1u) + i;
        b_hessenberg[h_idx] = sdata[0];
    }
}

// Kernel 3: Update w
// Dispatch: (num_groups_n, 1, 1)
@compute @workgroup_size(64)
fn update_w_cgs(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let j = params.num_iters;
    let n = params.n;
    let max_restart = params.max_restart;
    
    let stride_bytes = (n * 4u + 255u) & 4294967040u;
    let stride_words = stride_bytes / 4u;

    if (idx >= n) { return; }

    var correction = 0.0;
    // Loop over i = 0..=j
    for (var i = 0u; i <= j; i++) {
        // Read h[i, j]
        let h_idx = j * (max_restart + 1u) + i;
        let h_val = b_hessenberg[h_idx];
        
        // Read V_i[idx]
        let v_val = b_basis[i * stride_words + idx];
        
        correction += h_val * v_val;
    }
    
    b_w[idx] = b_w[idx] - correction;
}
