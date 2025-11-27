@group(0) @binding(0)
var<storage, read> face_owner: array<u32>;

@group(0) @binding(1)
var<storage, read> face_neighbor: array<i32>;

struct Vector2 {
    x: f32,
    y: f32,
}

@group(0) @binding(2)
var<storage, read_write> u: array<Vector2>;

@group(0) @binding(3)
var<storage, read> cell_face_offsets: array<u32>;

@group(0) @binding(4)
var<storage, read> cell_faces: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&u)) {
        return;
    }
    
    let start = cell_face_offsets[idx];
    let end = cell_face_offsets[idx + 1];
    
    var sum_u = Vector2(0.0, 0.0);
    var count = 0.0;
    
    for (var k = start; k < end; k++) {
        let face_idx = cell_faces[k];
        let owner = face_owner[face_idx];
        let neigh = face_neighbor[face_idx];
        
        var other_idx = -1;
        if (owner == idx) {
            other_idx = neigh;
        } else {
            other_idx = i32(owner);
        }
        
        if (other_idx != -1) {
            let val = u[other_idx];
            sum_u.x += val.x;
            sum_u.y += val.y;
            count += 1.0;
        }
    }
    
    if (count > 0.0) {
        // Simple relaxation: u = 0.5 * u + 0.5 * avg
        let avg_x = sum_u.x / count;
        let avg_y = sum_u.y / count;
        
        u[idx].x = u[idx].x * 0.5 + avg_x * 0.5;
        u[idx].y = u[idx].y * 0.5 + avg_y * 0.5;
    }
}
