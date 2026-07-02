struct BreakParams {
    count: u32,
    tol_rel: f32,
    tol_abs: f32,
    plateau_factor: f32,
    plateau_ceiling: f32,
    min_iters_tol: u32,
    min_iters_stall: u32,
    plateau_mode: u32,
}

@group(0) @binding(0) 
var<storage, read> delta: array<f32>;

@group(0) @binding(1) 
var<storage, read> scale: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> status: array<u32>;

@group(0) @binding(3) 
var<uniform> params: BreakParams;

@group(0) @binding(4) 
var<storage, read_write> delta_prev: array<f32>;

@group(0) @binding(5) 
var<storage, read_write> eval_count: array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }
    let it = eval_count[0u] + 1u;
    eval_count[0u] = it;
    if (params.plateau_mode == 0u) {
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
    } else {
        var all_under: u32 = 1u;
        var all_band: u32 = 1u;
        for (var i: u32 = 0u; i < params.count; i++) {
            let d = delta[i];
            let s_raw = scale[i];
            let bad_d = !(d <= d) || abs(d) > 1000000000000000000000000000000.0;
            let bad_s = !(s_raw <= s_raw) || abs(s_raw) > 1000000000000000000000000000000.0;
            if (bad_d || bad_s) {
                all_under = 0u;
                all_band = 0u;
                break;
            }
            let s = max(s_raw, 1.0);
            let r_cur = d / s;
            if (!(r_cur <= params.tol_rel || r_cur <= params.tol_abs)) {
                all_under = 0u;
                let r_prev = delta_prev[i] / s;
                let ratio = r_cur / max(r_prev, 0.000000000000000000000000000001);
                if (ratio < params.plateau_factor || ratio > params.plateau_ceiling) {
                    all_band = 0u;
                }
            }
        }
        var st: u32 = 0u;
        if (it >= params.min_iters_tol) {
            if (all_under == 1u) {
                st = 1u;
            }
        }
        if (it >= params.min_iters_stall) {
            if (all_band == 1u) {
                st = 1u;
            }
        }
        status[0u] = st;
    }
    for (var i: u32 = 0u; i < params.count; i++) {
        delta_prev[i] = delta[i];
    }
}
