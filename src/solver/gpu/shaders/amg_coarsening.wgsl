// AMG Coarsening Shaders

// Bind Groups:
// Group 0: Matrix (Graph)
// Group 1: State (MIS status, random values)
// Group 2: Params

struct MatrixParams {
    n: u32, // Number of rows/cells
    padding: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
// We don't need values for connectivity, just the graph structure.
// But we might need strength of connection later. For now, standard aggregation.

@group(1) @binding(0) var<storage, read_write> state: array<u32>; // 0: Undecided, 1: C-point, 2: F-point
@group(1) @binding(1) var<storage, read_write> random_values: array<f32>; // Random values for MIS

@group(2) @binding(0) var<uniform> params: MatrixParams;

// Constants for State
const UNDECIDED: u32 = 0u;
const C_POINT: u32 = 1u;
const F_POINT: u32 = 2u;

// 1. Initialize Random Values (if not done on CPU)
// We can do this on CPU or use a simple hash in shader.
// Let's assume passed from CPU for now or generated here.
// Simple hash function
fn hash(id: u32) -> f32 {
    var x = id;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return f32(x) / 4294967295.0;
}

@compute @workgroup_size(64)
fn init_random(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }
    random_values[i] = hash(i); // Or combine with seed
    state[i] = UNDECIDED;
}

// 2. MIS Iteration: Select C-points
// A point i becomes C-point if it has the maximal random value among its undecided neighbors.
@compute @workgroup_size(64)
fn mis_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    if (state[i] != UNDECIDED) {
        return;
    }

    let my_val = random_values[i];
    var is_max = true;

    let start = row_offsets[i];
    let end = row_offsets[i + 1];

    for (var k = start; k < end; k++) {
        let neighbor = col_indices[k];
        if (neighbor == i) { continue; } // Ignore self-loop

        // Only consider neighbors that are UNDECIDED or C-POINTS?
        // Standard MIS: check all neighbors.
        // If neighbor is already C-point, I might become F-point (handled in next pass).
        // Here we check if I am strictly greater than all neighbors.
        
        // Optimization: Only check undecided neighbors?
        // If a neighbor is already C, I can't be C.
        if (state[neighbor] == C_POINT) {
            is_max = false;
            break;
        }
        
        // If neighbor is F, it doesn't compete.
        if (state[neighbor] == F_POINT) {
            continue;
        }

        // Compare random values
        let neighbor_val = random_values[neighbor];
        if (neighbor_val > my_val) {
            is_max = false;
            break;
        } else if (neighbor_val == my_val && neighbor > i) {
            // Tie-breaking: use index
            is_max = false;
            break;
        }
    }

    if (is_max) {
        state[i] = C_POINT;
    }
}

// 3. MIS Update: Mark F-points
// If a neighbor is a C-point, I become an F-point.
@compute @workgroup_size(64)
fn mis_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    if (state[i] != UNDECIDED) {
        return;
    }

    let start = row_offsets[i];
    let end = row_offsets[i + 1];

    for (var k = start; k < end; k++) {
        let neighbor = col_indices[k];
        if (state[neighbor] == C_POINT) {
            state[i] = F_POINT;
            break;
        }
    }
}

// 4. Count C-points (Prefix Sum helper, or atomic counter)
// We need to know the size of the coarse grid.
// We can use an atomic counter in a separate buffer.

@group(3) @binding(0) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(64)
fn count_c_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    if (state[i] == C_POINT) {
        atomicAdd(&counter, 1u);
    }
}

// 5. Generate Coarse Indices (Mapping from Fine to Coarse)
// We need a mapping array: fine_to_coarse[fine_idx] = coarse_idx (or -1 if F-point)
// This requires a prefix sum scan on the boolean array (is_c_point).
// For simplicity on GPU without prefix sum lib, we can use a single-threaded dispatch or atomic counter (slow but works for now).
// Or we can just use the atomic counter to assign indices linearly.

@group(1) @binding(2) var<storage, read_write> fine_to_coarse: array<i32>; // -1 for F-point

@compute @workgroup_size(64)
fn assign_coarse_indices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    if (state[i] == C_POINT) {
        let idx = atomicAdd(&counter, 1u); // Re-using counter, must be reset to 0 before this
        fine_to_coarse[i] = i32(idx);
    } else {
        fine_to_coarse[i] = -1;
    }
}

// 6. Form Aggregates (Assign F-points to C-points)
@compute @workgroup_size(64)
fn form_aggregates(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    if (state[i] == F_POINT) {
        let start = row_offsets[i];
        let end = row_offsets[i + 1];
        var best_c = -1;
        var max_val = -1.0;

        for (var k = start; k < end; k++) {
            let neighbor = col_indices[k];
            if (state[neighbor] == C_POINT) {
                let val = random_values[neighbor];
                if (val > max_val) {
                    max_val = val;
                    best_c = i32(neighbor);
                }
            }
        }

        if (best_c != -1) {
            // Assign to the coarse index of the C-point
            // Note: fine_to_coarse[best_c] must be already set (run assign_coarse_indices first)
            fine_to_coarse[i] = fine_to_coarse[best_c];
        } else {
            // Orphan F-point. Should not happen in connected graph.
            // Fallback: keep as -1 or assign to self (new C-point)?
            // For now, leave as -1 (will be dropped or handled).
        }
    }
}

// 7. Generate P Matrix (CSR)
// P is fine_n x coarse_n.
// row_offsets: 0, 1, 2, ... fine_n
// col_indices: fine_to_coarse
// values: 1.0

@group(3) @binding(1) var<storage, read_write> p_row_offsets: array<u32>;
@group(3) @binding(2) var<storage, read_write> p_col_indices: array<u32>;
@group(3) @binding(3) var<storage, read_write> p_values: array<f32>;

@compute @workgroup_size(64)
fn generate_p_csr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        if (i == params.n) {
            p_row_offsets[i] = i; // Last offset
        }
        return;
    }

    p_row_offsets[i] = i;
    let c = fine_to_coarse[i];
    if (c != -1) {
        p_col_indices[i] = u32(c);
        p_values[i] = 1.0;
    } else {
        // Orphan point, what to do?
        // Point to 0 with 0 value?
        p_col_indices[i] = 0u;
        p_values[i] = 0.0;
    }
}

// 8. Count Aggregates (for R Matrix Row Offsets)
// R is coarse_n x fine_n.
// R row offsets needs size of each aggregate.
// We use atomic add to count.

@group(3) @binding(4) var<storage, read_write> r_row_offsets: array<atomic<u32>>; // Using atomics

@compute @workgroup_size(64)
fn count_aggregates(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    let c = fine_to_coarse[i];
    if (c != -1) {
        atomicAdd(&r_row_offsets[u32(c) + 1u], 1u);
    }
}

// 9. Fill R Matrix (CSR)
// We need a temp counter to track current position in each row.
@group(3) @binding(5) var<storage, read_write> temp_counter: array<atomic<u32>>;
@group(3) @binding(6) var<storage, read_write> r_col_indices: array<u32>;
@group(3) @binding(7) var<storage, read_write> r_values: array<f32>;
// r_row_offsets is now read-only (u32 array, not atomic)
@group(3) @binding(8) var<storage, read> r_row_offsets_final: array<u32>;

@compute @workgroup_size(64)
fn fill_r_csr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    let c = fine_to_coarse[i];
    if (c != -1) {
        let idx = atomicAdd(&temp_counter[u32(c)], 1u);
        let offset = r_row_offsets_final[u32(c)];
        r_col_indices[offset + idx] = i;
        r_values[offset + idx] = 1.0;
    }
}
