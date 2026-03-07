struct BreakParams {
    count: u32,
    tol_rel: f32,
    tol_abs: f32,
    _pad0: u32,
}

@group(0) @binding(0) 
var<storage, read> delta: array<f32>;

@group(0) @binding(1) 
var<storage, read> scale: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> status: array<u32>;

@group(0) @binding(3) 
var<uniform> params: BreakParams;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }
    var converged: u32 = 1u;
    for (var i: u32 = 0u; i < params.count; i++) {
        let d = delta[i];
        let s_raw = scale[i];
        let bad_d = !(d <= d) || abs(d) > 1000000000000000000000000000000.0;
        let bad_s = !(s_raw <= s_raw) || abs(s_raw) > 1000000000000000000000000000000.0;
        if (bad_d || bad_s) {
            converged = 0u;
            break;
        }
        let s = max(s_raw, 1.0);
        let tol = params.tol_abs + params.tol_rel * s;
        if (d > tol) {
            converged = 0u;
            break;
        }
    }
    status[0u] = converged;
}
