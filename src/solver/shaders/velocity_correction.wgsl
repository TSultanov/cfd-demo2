struct Vector2 {
    x: f32,
    y: f32,
}

@group(1) @binding(0) var<storage, read_write> u: array<Vector2>;
@group(1) @binding(4) var<storage, read_write> grad_p_prime: array<Vector2>;
@group(1) @binding(5) var<storage, read_write> d_p: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&u)) {
        return;
    }
    
    let dp = d_p[idx];
    let gp = grad_p_prime[idx];
    
    u[idx].x -= dp * gp.x;
    u[idx].y -= dp * gp.y;
}
