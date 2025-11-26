struct Vector2 {
    x: f32,
    y: f32,
}

struct Constants {
    dt: f32,
    time: f32,
    viscosity: f32,
    density: f32,
    component: u32,
    alpha_p: f32,
    scheme: u32,
    alpha_u: f32,
    stride_x: u32,
    padding: u32,
}

// Mesh data (matches gradient.wgsl layout)
@group(0) @binding(0) var<storage, read> face_owner: array<u32>;
@group(0) @binding(1) var<storage, read> face_neighbor: array<i32>;
@group(0) @binding(2) var<storage, read> face_areas: array<f32>;
@group(0) @binding(3) var<storage, read> face_normals: array<Vector2>;
@group(0) @binding(4) var<storage, read> cell_centers: array<Vector2>;
@group(0) @binding(5) var<storage, read> cell_vols: array<f32>;
@group(0) @binding(6) var<storage, read> cell_face_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> cell_faces: array<u32>;
@group(0) @binding(12) var<storage, read> face_boundary: array<u32>;
@group(0) @binding(13) var<storage, read> face_centers: array<Vector2>;

// Field data
@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(1) var<storage, read_write> p: array<f32>;
@group(1) @binding(2) var<storage, read_write> fluxes: array<f32>;
@group(1) @binding(3) var<uniform> constants: Constants;
@group(1) @binding(4) var<storage, read_write> grad_p: array<Vector2>; // Keep binding 4 as grad_p (unused here or maybe used? No, previously it was overwriting it)
@group(1) @binding(5) var<storage, read_write> d_p: array<f32>;
@group(1) @binding(8) var<storage, read_write> grad_p_prime: array<Vector2>; // New binding

// Linear solver state (p_prime sits in x)
@group(2) @binding(0) var<storage, read> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&u)) {
        return;
    }

    let start = cell_face_offsets[idx];
    let end = cell_face_offsets[idx + 1u];
    let vol = cell_vols[idx];
    let center = cell_centers[idx];
    let val_c = x[idx];

    var grad = Vector2(0.0, 0.0);

    for (var k = start; k < end; k = k + 1u) {
        let face_idx = cell_faces[k];
        let owner = face_owner[face_idx];
        let neigh_idx = face_neighbor[face_idx];

        var normal = face_normals[face_idx];
        let area = face_areas[face_idx];
        let face_center = face_centers[face_idx];

        if (owner != idx) {
            normal.x = -normal.x;
            normal.y = -normal.y;
        }

        var val_f = 0.0;

        if (neigh_idx != -1) {
            var other_idx = u32(neigh_idx);
            if (owner != idx) {
                other_idx = owner;
            }

            let val_other = x[other_idx];
            let center_other = cell_centers[other_idx];

            let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(face_center.x, face_center.y));
            let d_o = distance(vec2<f32>(center_other.x, center_other.y), vec2<f32>(face_center.x, face_center.y));

            let total_dist = d_c + d_o;
            if (total_dist > 1e-6) {
                let lambda = d_o / total_dist;
                val_f = lambda * val_c + (1.0 - lambda) * val_other;
            } else {
                val_f = 0.5 * (val_c + val_other);
            }
        } else {
            let boundary_type = face_boundary[face_idx];
            if (boundary_type == 2u) {
                val_f = 0.0;
            } else {
                val_f = val_c;
            }
        }

        grad.x += val_f * normal.x * area;
        grad.y += val_f * normal.y * area;

        if (owner == idx) {
            var d_p_face = d_p[idx];

            if (neigh_idx != -1) {
                let neigh = u32(neigh_idx);
                
                // Distance weighted interpolation for d_p_face to match pressure assembly
                let c_neigh = cell_centers[neigh];
                let d_c = distance(vec2<f32>(center.x, center.y), vec2<f32>(face_center.x, face_center.y));
                let d_n = distance(vec2<f32>(c_neigh.x, c_neigh.y), vec2<f32>(face_center.x, face_center.y));
                let total_dist = d_c + d_n;
                
                var lambda = 0.5;
                if (total_dist > 1e-6) {
                    lambda = d_n / total_dist;
                }
                d_p_face = lambda * d_p[idx] + (1.0 - lambda) * d_p[neigh];

                let dx = c_neigh.x - center.x;
                let dy = c_neigh.y - center.y;
                let dist = sqrt(dx * dx + dy * dy);
                if (dist > 1e-6) {
                    let grad_p_f = (x[neigh] - val_c) / dist;
                    fluxes[face_idx] -= constants.density * d_p_face * area * grad_p_f;
                }
            } else {
                let boundary_type = face_boundary[face_idx];
                if (boundary_type == 2u) {
                    let f_center = face_centers[face_idx];
                    let dx = f_center.x - center.x;
                    let dy = f_center.y - center.y;
                    let dist = sqrt(dx * dx + dy * dy);
                    if (dist > 1e-6) {
                        let grad_p_f = (0.0 - val_c) / dist;
                        fluxes[face_idx] -= constants.density * d_p_face * area * grad_p_f;
                    }
                }
            }
        }
    }

    grad.x /= vol;
    grad.y /= vol;
    grad_p_prime[idx] = grad;

    let dp = d_p[idx];
    u[idx].x -= dp * grad.x;
    u[idx].y -= dp * grad.y;

    p[idx] += constants.alpha_p * val_c;
}
