@group(0) @binding(0) 
var<storage, read> break_status: array<u32>;

@group(0) @binding(1) 
var<storage, read> real_args_cells: array<u32>;

@group(0) @binding(2) 
var<storage, read_write> indirect_args_cells: array<u32>;

@group(0) @binding(3) 
var<storage, read> real_args_faces: array<u32>;

@group(0) @binding(4) 
var<storage, read_write> indirect_args_faces: array<u32>;

@group(0) @binding(5) 
var<storage, read_write> iter_counter: array<atomic<u32>>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }
    let converged = break_status[0u];
    if (converged == 0u) {
        indirect_args_cells[0u] = real_args_cells[0u];
        indirect_args_cells[1u] = real_args_cells[1u];
        indirect_args_cells[2u] = real_args_cells[2u];
        indirect_args_faces[0u] = real_args_faces[0u];
        indirect_args_faces[1u] = real_args_faces[1u];
        indirect_args_faces[2u] = real_args_faces[2u];
        atomicAdd(&iter_counter[0u], 1u);
    } else {
        indirect_args_cells[0u] = 0u;
        indirect_args_cells[1u] = 0u;
        indirect_args_cells[2u] = 0u;
        indirect_args_faces[0u] = 0u;
        indirect_args_faces[1u] = 0u;
        indirect_args_faces[2u] = 0u;
    }
}
