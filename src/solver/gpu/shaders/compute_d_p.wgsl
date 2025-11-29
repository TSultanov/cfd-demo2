struct Constants {
    dt: f32,
    dt_old: f32,
    time: f32,
    viscosity: f32,
    density: f32,
    component: u32,
    alpha_p: f32,
    time_scheme: u32,
    inlet_velocity: f32,
    ramp_time: f32,
    alpha_u: f32,
    stride_x: u32,
    padding: u32,
}

// Group 0: Mesh
@group(0) @binding(5) var<storage, read> cell_vols: array<f32>;
@group(0) @binding(11) var<storage, read> diagonal_indices: array<u32>;

// Group 1: Fields
@group(1) @binding(3) var<uniform> constants: Constants;
@group(1) @binding(5) var<storage, read_write> d_p: array<f32>;

// Group 2: Solver
@group(2) @binding(0) var<storage, read_write> matrix_values: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&cell_vols)) {
        return;
    }
    
    let diag_idx = diagonal_indices[idx];
    let a_p = matrix_values[diag_idx];
    let vol = cell_vols[idx];
    
    if (abs(a_p) > 1e-20) {
        d_p[idx] = vol / a_p;
    } else {
        d_p[idx] = 0.0;
    }
}
