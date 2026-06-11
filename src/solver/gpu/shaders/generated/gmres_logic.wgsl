// GMRES Logic Shaders (Small system operations)

struct IterParams {
    current_idx: u32,
    max_restart: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) 
var<storage, read_write> hessenberg: array<f32>;

@group(0) @binding(1) 
var<storage, read_write> givens: array<vec2<f32>>;

@group(0) @binding(2) 
var<storage, read_write> g_rhs: array<f32>;

@group(0) @binding(3) 
var<storage, read_write> y_sol: array<f32>;

@group(1) @binding(0) 
var<uniform> iter_params: IterParams;

@group(1) @binding(1) 
var<storage, read_write> scalars: array<f32>;

@group(1) @binding(2) 
var<storage, read_write> indirect_args: array<vec4<u32>>;

const SCALAR_STOP: u32 = 8u;

const SCALAR_CONVERGED: u32 = 9u;

const SCALAR_ITERS_USED: u32 = 10u;

const SCALAR_RESIDUAL_EST: u32 = 11u;

const SCALAR_TOL_REL_RHS: u32 = 12u;

const SCALAR_TOL_ABS: u32 = 13u;

const SCALAR_RHS_NORM: u32 = 14u;

const SCALAR_SKIP_UPDATE: u32 = 15u;

const SCALAR_BEST_RESID: u32 = 16u;

const SCALAR_GUARD_FLAG: u32 = 17u;

const SCALAR_PREV_RESID: u32 = 18u;

const SCALAR_STALL_REL: u32 = 19u;

const SCALAR_STALL_COUNT: u32 = 20u;

const SCALAR_PREV_EST: u32 = 21u;

const SCALAR_STALL_COUNT_ITER: u32 = 22u;

const SCALAR_TOTAL_ITERS: u32 = 23u;

fn h_idx(row: u32, col: u32) -> u32 {
    return col * (iter_params.max_restart + 1u) + row;
}

@compute
@workgroup_size(1)
fn update_hessenberg_givens(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        return;
    }
    scalars[SCALAR_TOTAL_ITERS] = scalars[SCALAR_TOTAL_ITERS] + 1.0;
    let j = iter_params.current_idx;
    // Apply previous Givens rotations to the new column H[:, j]
    for (var i: u32 = 0u; i < j; i++) {
        let idx_i = h_idx(i, j);
        let idx_i1 = h_idx(i + 1u, j);
        let h_ij = hessenberg[idx_i];
        let h_i1j = hessenberg[idx_i1];
        let cs = givens[i];
        let c = cs.x;
        let s = cs.y;
        hessenberg[idx_i] = c * h_ij + s * h_i1j;
        hessenberg[idx_i1] = -s * h_ij + c * h_i1j;
    }
    // Compute new Givens rotation for H[j, j] and H[j+1, j]
    let idx_jj = h_idx(j, j);
    let idx_j1j = h_idx(j + 1u, j);
    let h_jj = hessenberg[idx_jj];
    let h_j1j = hessenberg[idx_j1j];
    var c = 1.0;
    var s = 0.0;
    var rho = sqrt(h_jj * h_jj + h_j1j * h_j1j);
    if (abs(rho) > 0.00000000000000000001) {
        c = h_jj / rho;
        s = h_j1j / rho;
    }
    // Store rotation
    givens[j] = vec2<f32>(c, s);
    // Apply rotation to H
    hessenberg[idx_jj] = rho;
    hessenberg[idx_j1j] = 0.0;
    // Apply rotation to RHS vector g
    let g_j = g_rhs[j];
    let g_j1 = g_rhs[j + 1u];
    g_rhs[j] = c * g_j + s * g_j1;
    g_rhs[j + 1u] = -s * g_j + c * g_j1;
    let residual = abs(g_rhs[j + 1u]);
    scalars[SCALAR_RESIDUAL_EST] = residual;
    let tol_rel_rhs = scalars[SCALAR_TOL_REL_RHS] * scalars[SCALAR_RHS_NORM];
    let tol_abs = scalars[SCALAR_TOL_ABS];
    if (residual <= tol_rel_rhs || residual <= tol_abs) {
        scalars[SCALAR_STOP] = 1.0;
        scalars[SCALAR_CONVERGED] = 1.0;
        scalars[SCALAR_ITERS_USED] = f32(j + 1u);
        // Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.
        indirect_args[0u] = vec4<u32>(0u, 0u, 0u, 0u);
        indirect_args[1u] = vec4<u32>(0u, 0u, 0u, 0u);
        indirect_args[2u] = vec4<u32>(0u, 0u, 0u, 0u);
    } else {
        let stall_rel = scalars[SCALAR_STALL_REL];
        if (stall_rel > 0.0) {
            let prev_est = scalars[SCALAR_PREV_EST];
            let no_improve = prev_est > 0.0 && residual > prev_est * 0.995;
            let level_ok = residual <= stall_rel * scalars[SCALAR_RHS_NORM];
            if (no_improve && level_ok) {
                scalars[SCALAR_STALL_COUNT_ITER] = scalars[SCALAR_STALL_COUNT_ITER] + 1.0;
                if (scalars[SCALAR_STALL_COUNT_ITER] > 9.5) {
                    scalars[SCALAR_STOP] = 1.0;
                    scalars[SCALAR_ITERS_USED] = f32(j + 1u);
                    // Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.
                    indirect_args[0u] = vec4<u32>(0u, 0u, 0u, 0u);
                    indirect_args[1u] = vec4<u32>(0u, 0u, 0u, 0u);
                    indirect_args[2u] = vec4<u32>(0u, 0u, 0u, 0u);
                }
            } else {
                scalars[SCALAR_STALL_COUNT_ITER] = 0.0;
            }
            scalars[SCALAR_PREV_EST] = residual;
        }
    }
}

@compute
@workgroup_size(1)
fn solve_triangular(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (scalars[SCALAR_SKIP_UPDATE] > 0.5) {
        return;
    }
    let k = u32(clamp(round(scalars[SCALAR_ITERS_USED]), 1.0, f32(iter_params.max_restart)));
    // Backward substitution
    for (var loop_i: u32 = 0u; loop_i < k; loop_i++) {
        let i = k - 1u - loop_i;
        var sum = g_rhs[i];
        for (var j: u32 = i + 1u; j < k; j++) {
            sum -= hessenberg[h_idx(i, j)] * y_sol[j];
        }
        let diag = hessenberg[h_idx(i, i)];
        if (abs(diag) > 0.000000000001) {
            y_sol[i] = sum / diag;
        } else {
            y_sol[i] = 0.0;
        }
    }
}

@compute
@workgroup_size(1)
fn finish_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let norm_sq = scalars[0u];
    let norm = sqrt(norm_sq);
    hessenberg[iter_params.current_idx] = norm;
    if (norm > 0.00000000000000000001) {
        scalars[0u] = 1.0 / norm;
    } else {
        scalars[0u] = 0.0;
    }
}

@compute
@workgroup_size(1)
fn restart_guard(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (scalars[SCALAR_STOP] > 0.5) {
        scalars[SCALAR_GUARD_FLAG] = 0.0;
        return;
    }
    let r = hessenberg[0u];
    let best = scalars[SCALAR_BEST_RESID];
    let grew = r != r || best > 0.0 && r > best * 1.25;
    if (grew) {
        scalars[SCALAR_GUARD_FLAG] = 2.0;
        scalars[SCALAR_STOP] = 1.0;
        scalars[SCALAR_SKIP_UPDATE] = 1.0;
        scalars[SCALAR_RESIDUAL_EST] = best;
        // Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.
        indirect_args[0u] = vec4<u32>(0u, 0u, 0u, 0u);
        indirect_args[1u] = vec4<u32>(0u, 0u, 0u, 0u);
        indirect_args[2u] = vec4<u32>(0u, 0u, 0u, 0u);
    } else {
        if (best <= 0.0 || r < best) {
            scalars[SCALAR_BEST_RESID] = r;
            scalars[SCALAR_GUARD_FLAG] = 1.0;
        } else {
            scalars[SCALAR_GUARD_FLAG] = 0.0;
        }
        let prev = scalars[SCALAR_PREV_RESID];
        let stall_rel = scalars[SCALAR_STALL_REL];
        let no_improve = prev > 0.0 && r > prev * 0.98;
        let level_ok = r <= stall_rel * scalars[SCALAR_RHS_NORM];
        if (stall_rel > 0.0 && no_improve && level_ok) {
            scalars[SCALAR_STALL_COUNT] = scalars[SCALAR_STALL_COUNT] + 1.0;
            if (scalars[SCALAR_STALL_COUNT] > 1.5) {
                let best_now = scalars[SCALAR_BEST_RESID];
                scalars[SCALAR_GUARD_FLAG] = select(2.0, 1.0, r <= best_now);
                scalars[SCALAR_STOP] = 1.0;
                scalars[SCALAR_SKIP_UPDATE] = 1.0;
                scalars[SCALAR_RESIDUAL_EST] = min(r, best_now);
                // Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.
                indirect_args[0u] = vec4<u32>(0u, 0u, 0u, 0u);
                indirect_args[1u] = vec4<u32>(0u, 0u, 0u, 0u);
                indirect_args[2u] = vec4<u32>(0u, 0u, 0u, 0u);
            }
        } else {
            scalars[SCALAR_STALL_COUNT] = 0.0;
        }
        scalars[SCALAR_PREV_RESID] = r;
    }
}

@compute
@workgroup_size(1)
fn clamp_rel_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let beta = hessenberg[0u];
    if (beta < scalars[SCALAR_RHS_NORM]) {
        scalars[SCALAR_RHS_NORM] = beta;
    }
}
