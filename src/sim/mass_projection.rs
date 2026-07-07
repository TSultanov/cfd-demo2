//! Mass-row transfer projection: make a transferred/re-seeded state
//! DISCRETELY mass-consistent before the next step.
//!
//! Two seams interpolate state onto cells whose surroundings disagree with
//! it, and both leave a residual defect in the solver's own continuity row
//! that the next coupled step balances with a `defect/dt` pressure response:
//!
//!  - a mesh-adaptation RESIZE (cell birth/kill/wall-split) transfers the
//!    whole field first-order onto the new mesh — the "phantom dipoles" at
//!    split/merge sites ([`project_transferred_state`]);
//!  - a seed RECYCLE re-seeds outflow parcels at the inlet with a
//!    zeroth-order copy of the nearest survivor's row — per-parcel pressure
//!    speckle in the inlet strip, which the |grad p| adaptation indicator
//!    can then chase ([`project_recycled_state`]).
//!
//! Two remedies were refuted BY MEASUREMENT before this one (commit
//! 8719bde): settle sub-steps (projection pressure ~ 1/dt, worse) and an
//! average-face-flux patch projection (the solver's mass operator is the
//! RHIE-CHOW flux, not the average flux — correcting the wrong divergence
//! adds noise). The lesson is encoded here: the ONLY defect worth removing
//! is the one the solver itself sees, so both the residual and the
//! correction operator come from the solver's own assembly.
//!
//! Algorithm (all f64, matrix-free):
//!  1. `eps = r_p(x*)` — the pressure-row (continuity) residual of the
//!     assembled coupled system at the current state, via
//!     [`CpuSolver::debug_assemble`]. At a resize the fresh solver's flat
//!     history makes the ddt/ALE terms cancel, so `eps` is the pure spatial
//!     mass defect; at a recycle (mid-run, armed ALE seam) the target is
//!     MASKED to the recycled cells' 2-ring neighborhood — the only rows
//!     the re-seed changed.
//!  2. `G` — the Jacobian of the mass flux w.r.t. CELL velocities. It is NOT
//!     in the assembled matrix (the div-flux term is Picard-deferred to the
//!     RHS), so it is reconstructed from the flux-module formula: per interior
//!     face `dphi/dU_own = rho_f*lambda*n*A`, `dphi/dU_neigh =
//!     rho_f*(1-lambda)*n*A` with the flux module's OpenFOAM-style
//!     face-normal-projected `lambda`; boundary faces use the ghost rules
//!     (Dirichlet: no sensitivity; ZeroGradient/Neumann: full owner weight;
//!     SlipWall: analytically zero normal flux).
//!  3. Least-norm velocity correction through the momentum diagonal:
//!     `delta_u = W G^T lambda` with `(G W G^T) lambda = target` solved by
//!     Jacobi-PCG (`W = 1/diag(A_uu)` — the same scale the solver's own
//!     pressure correction would push through, so the momentum disturbance is
//!     minimal). Pressure is NOT touched: the dipole IS the pressure response,
//!     so the correction must be carried entirely by the velocity.
//!  4. Verify by RE-ASSEMBLY: the correction is kept only if the measured
//!     residual actually dropped. Any convention drift between step 2 and
//!     the real flux kernel degrades the drop instead of corrupting the run.

use crate::solver::cpu::CpuSolver;
use crate::solver::mesh::{BoundaryType, Mesh};

/// What the projection did — surfaced in [`super::MovingMeshStats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct MassProjectionOutcome {
    /// Residual inf-norm at the pre-correction state (over the target set:
    /// all cells for a resize, the affected neighborhood for a recycle).
    pub pre: f64,
    /// ... after the correction (== `pre` when it was rejected).
    pub post: f64,
    /// PCG iterations spent on the `(G W G^T) lambda = target` solve.
    pub cg_iters: usize,
    /// Whether the correction was kept (re-assembly measured an improvement).
    pub applied: bool,
}

/// State-layout facts the projection needs (the caller reads them off the
/// model once; the CPU solver only knows coupled ranks).
#[derive(Clone, Copy, Debug)]
pub struct ProjectionLayout {
    /// State stride (floats per cell row).
    pub stride: usize,
    /// State offset of the velocity field "U" (2 components).
    pub u_off: usize,
    /// State offset of the density field "rho", when it is a state field
    /// (all-Mach families); `None` = uniform `constants.density`.
    pub rho_off: Option<usize>,
    /// Density upwinding marker: the flux module upwinds `rho_f` by the sign
    /// of the face-normal velocity iff the model carries `t_ref` (the real-EOS
    /// thermal family). Frozen at the transferred state, Picard-style.
    pub upwind_rho: bool,
}

/// Per-face sensitivity of the (predicted) mass flux to the two cell
/// velocities. `neighbor == usize::MAX` marks a boundary face (owner-only).
struct FaceJac {
    owner: usize,
    neighbor: usize,
    g_own: [f64; 2],
    g_nei: [f64; 2],
}

const NO_NEIGHBOR: usize = usize::MAX;
/// PCG relative tolerance on `||r||_2 / ||target||_2`. The assembly is f32,
/// so anything past ~1e-7 relative is below the operator's own noise floor.
const CG_RTOL: f64 = 1e-10;
const CG_MAX_ITERS: usize = 2000;

/// The coupled-system ranks of ("U", "p"), `None` for models without a
/// momentum/pressure system.
fn coupled_ranks(cpu: &CpuSolver) -> Option<(usize, usize)> {
    Some((
        cpu.coupled_rank_of("U")? as usize,
        cpu.coupled_rank_of("p")? as usize,
    ))
}

/// One assembly at the given state: the per-cell pressure-row residual
/// `r_p = rhs_p - (A x)_p` (f64) and the momentum-diagonal weights
/// `W = 1/diag(A_uu)` (2 per cell). `state` must be what the solver holds.
fn assemble_eps_w(
    cpu: &mut CpuSolver,
    state: &[f32],
    stride: usize,
    p_rank: usize,
    u_rank: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let threads = cpu.threads();
    let (mat, rhs) = cpu.debug_assemble();
    let (row_offsets, col_indices, diag_indices, s) = cpu.debug_topology();
    let offsets: Vec<usize> = cpu.unknown_state_offsets().iter().map(|&o| o as usize).collect();
    let mut eps = vec![0.0f64; n];
    let mut w = vec![0.0f64; 2 * n];
    // Per-cell CSR read: each `eps[c]` / `w[c*2..]` depends only on cell `c`'s
    // matrix row, so the fill parallelizes bit-identically (disjoint writes,
    // no reduction — same arithmetic per cell regardless of `threads`).
    crate::solver::cpu::parallel::parallel_cell_chunks_mut2(
        n,
        1,
        2,
        threads,
        &mut eps,
        &mut w,
        |start, er, wr| {
            for (li, (ec, wc)) in er.iter_mut().zip(wr.chunks_exact_mut(2)).enumerate() {
                let c = start + li;
                let so = row_offsets[c] as usize;
                let nn = row_offsets[c + 1] as usize - so;
                let start_p = so * s * s + nn * s * p_rank;
                let mut acc = rhs[c * s + p_rank] as f64;
                for k in 0..nn {
                    let col_cell = col_indices[so + k] as usize;
                    let base = start_p + k * s;
                    for col in 0..s {
                        let off = offsets[col];
                        acc -= mat[base + col] as f64 * state[col_cell * stride + off] as f64;
                    }
                }
                *ec = acc;
                let drank = (diag_indices[c] as usize) - so;
                for comp in 0..2 {
                    let r = u_rank + comp;
                    let a = mat[so * s * s + nn * s * r + drank * s + r] as f64;
                    // Momentum diagonals carry vol*rho/dt and are positive; a
                    // degenerate one freezes that dof out of the correction.
                    wc[comp] = if a > 1e-30 { 1.0 / a } else { 0.0 };
                }
            }
        },
    );
    (eps, w)
}

/// The mass-flux velocity Jacobian, face by face, from the flux-module
/// formula evaluated at `state`. Returns the face list and whether any
/// boundary face carries a u-sensitivity (an outlet "grounds" the least-norm
/// system; without one it has the constant null space).
fn build_face_jacobian(
    cpu: &CpuSolver,
    mesh: &Mesh,
    layout: ProjectionLayout,
    state: &[f32],
    u_rank: usize,
) -> (Vec<FaceJac>, bool) {
    let stride = layout.stride;
    let bc_kind = cpu.debug_bc_kind();
    let s_coupled = cpu.debug_topology().3;
    let rho_at = |c: usize| -> f64 {
        match layout.rho_off {
            Some(off) => state[c * stride + off] as f64,
            None => cpu.density_constant() as f64,
        }
    };
    let mut jac: Vec<FaceJac> = Vec::with_capacity(mesh.num_faces());
    let mut grounded = false;
    for f in 0..mesh.num_faces() {
        let o = mesh.face_owner[f];
        let area = mesh.face_area[f];
        let (nx, ny) = (mesh.face_nx[f], mesh.face_ny[f]);
        if area <= 0.0 {
            continue;
        }
        match mesh.face_neighbor[f] {
            Some(nb) => {
                // Flux-module lambda: OpenFOAM surfaceInterpolation::weights(),
                // face-normal-projected distances (NOT the assembly's
                // Euclidean weights), 0.5 fallback under the 1e-6 guard.
                let (fcx, fcy) = (mesh.face_cx[f], mesh.face_cy[f]);
                let wrap = mesh.face_wrap_shift.get(f).copied().unwrap_or([0.0, 0.0]);
                let d_own = ((fcx - mesh.cell_cx[o]) * nx + (fcy - mesh.cell_cy[o]) * ny).abs();
                let d_nei = ((mesh.cell_cx[nb] + wrap[0] - fcx) * nx
                    + (mesh.cell_cy[nb] + wrap[1] - fcy) * ny)
                    .abs();
                let total = d_own + d_nei;
                let lambda = if total > 1e-6 { d_nei / total } else { 0.5 };
                let rho_f = match layout.rho_off {
                    None => cpu.density_constant() as f64,
                    Some(_) => {
                        let (ro, rn) = (rho_at(o), rho_at(nb));
                        if layout.upwind_rho {
                            // rho_f = avg + 0.5*sgn(u_n)*(rho_o - rho_n),
                            // sgn frozen at the current state.
                            let uo = &state[o * stride + layout.u_off..];
                            let un = &state[nb * stride + layout.u_off..];
                            let ufx = lambda * uo[0] as f64 + (1.0 - lambda) * un[0] as f64;
                            let ufy = lambda * uo[1] as f64 + (1.0 - lambda) * un[1] as f64;
                            let u_n = ufx * nx + ufy * ny;
                            let sgn = u_n / u_n.abs().max(1.0e-12);
                            0.5 * (ro + rn) + 0.5 * sgn * (ro - rn)
                        } else {
                            lambda * ro + (1.0 - lambda) * rn
                        }
                    }
                };
                let k_own = rho_f * lambda * area;
                let k_nei = rho_f * (1.0 - lambda) * area;
                jac.push(FaceJac {
                    owner: o,
                    neighbor: nb,
                    g_own: [k_own * nx, k_own * ny],
                    g_nei: [k_nei * nx, k_nei * ny],
                });
            }
            None => {
                // Boundary face: both flux sides evaluate the GHOST. SlipWall
                // projects out the normal component (zero normal-flux
                // sensitivity); per-component BC kinds decide the rest —
                // Dirichlet (wall/inlet/moving wall) has no sensitivity,
                // ZeroGradient/Neumann (outlet) passes the owner through at
                // full weight.
                if matches!(mesh.face_boundary[f], Some(BoundaryType::SlipWall)) {
                    continue;
                }
                let rho_f = rho_at(o);
                let mut g = [0.0f64; 2];
                let mut any = false;
                for comp in 0..2 {
                    let kind = bc_kind[f * s_coupled + u_rank + comp];
                    // GpuBcKind: 0 = ZeroGradient, 1 = Dirichlet, 2 = Neumann.
                    if kind != 1 {
                        g[comp] = rho_f * area * if comp == 0 { nx } else { ny };
                        any = any || g[comp] != 0.0;
                    }
                }
                if any {
                    grounded = true;
                    jac.push(FaceJac {
                        owner: o,
                        neighbor: NO_NEIGHBOR,
                        g_own: g,
                        g_nei: [0.0; 2],
                    });
                }
            }
        }
    }
    (jac, grounded)
}

/// Least-norm velocity correction: Jacobi-PCG on `S = G W G^T` (SPD,
/// matrix-free — `G^T` scatters a cell potential to velocity dofs, `W`
/// scales, `G` gathers back to divergences), then `du = W G^T lambda`.
/// Returns `(du, cg_iters)`; `du` solves `G du = target` in the least-norm
/// sense at convergence.
fn solve_du(
    jac: &[FaceJac],
    w: &[f64],
    target: &[f64],
    grounded: bool,
    n: usize,
    threads: usize,
) -> (Vec<f64>, usize) {
    use crate::solver::cpu::parallel::parallel_cell_chunks_mut;

    // Per-cell CSR of the `jac` entries that touch each cell, in ASCENDING jac
    // index. Because `jac` is built in ascending face order, this reproduces
    // the exact order the serial scatter (`for fj in jac`) would add each
    // cell's contributions — so the gather operators below are BIT-IDENTICAL
    // to the serial scatter regardless of `threads`. Built once per solve
    // (integer counting sort, deterministic).
    let mut off = vec![0usize; n + 1];
    for fj in jac {
        off[fj.owner + 1] += 1;
        if fj.neighbor != NO_NEIGHBOR {
            off[fj.neighbor + 1] += 1;
        }
    }
    for c in 0..n {
        off[c + 1] += off[c];
    }
    let mut ent = vec![0u32; off[n]];
    {
        let mut cur = off[..n].to_vec();
        for (j, fj) in jac.iter().enumerate() {
            let o = fj.owner;
            ent[cur[o]] = j as u32;
            cur[o] += 1;
            if fj.neighbor != NO_NEIGHBOR {
                let nb = fj.neighbor;
                ent[cur[nb]] = j as u32;
                cur[nb] += 1;
            }
        }
    }
    let cell_entries = |c: usize| -> &[u32] { &ent[off[c]..off[c + 1]] };

    // t = W · G^T lam, gathered per cell (owner: +, neighbor: −), then scaled
    // by the momentum diagonal `w`. Disjoint per-cell writes.
    let apply_gt_w = |lam: &[f64], t: &mut [f64]| {
        parallel_cell_chunks_mut(n, 2, threads, t, |start, tr| {
            for (li, tc) in tr.chunks_exact_mut(2).enumerate() {
                let c = start + li;
                let (mut a0, mut a1) = (0.0f64, 0.0f64);
                for &je in cell_entries(c) {
                    let fj = &jac[je as usize];
                    if fj.neighbor == NO_NEIGHBOR {
                        let lo = lam[fj.owner];
                        a0 += lo * fj.g_own[0];
                        a1 += lo * fj.g_own[1];
                    } else {
                        let d = lam[fj.owner] - lam[fj.neighbor];
                        if fj.owner == c {
                            a0 += d * fj.g_own[0];
                            a1 += d * fj.g_own[1];
                        } else {
                            a0 += d * fj.g_nei[0];
                            a1 += d * fj.g_nei[1];
                        }
                    }
                }
                tc[0] = a0 * w[c * 2];
                tc[1] = a1 * w[c * 2 + 1];
            }
        });
    };
    // y = G t, gathered per cell. The per-face divergence `val` is recomputed
    // for both incident cells (owner adds it, neighbor subtracts it) — 2× the
    // flops of the scatter, no shared state.
    let apply_g = |t: &[f64], y: &mut [f64]| {
        parallel_cell_chunks_mut(n, 1, threads, y, |start, yr| {
            for (li, yc) in yr.iter_mut().enumerate() {
                let c = start + li;
                let mut acc = 0.0f64;
                for &je in cell_entries(c) {
                    let fj = &jac[je as usize];
                    if fj.neighbor == NO_NEIGHBOR {
                        acc += fj.g_own[0] * t[fj.owner * 2] + fj.g_own[1] * t[fj.owner * 2 + 1];
                    } else {
                        let val = fj.g_own[0] * t[fj.owner * 2]
                            + fj.g_own[1] * t[fj.owner * 2 + 1]
                            + fj.g_nei[0] * t[fj.neighbor * 2]
                            + fj.g_nei[1] * t[fj.neighbor * 2 + 1];
                        if fj.owner == c {
                            acc += val;
                        } else {
                            acc -= val;
                        }
                    }
                }
                *yc = acc;
            }
        });
    };
    let mut diag = vec![0.0f64; n];
    parallel_cell_chunks_mut(n, 1, threads, &mut diag, |start, dr| {
        for (li, dc) in dr.iter_mut().enumerate() {
            let c = start + li;
            let mut acc = 0.0f64;
            for &je in cell_entries(c) {
                let fj = &jac[je as usize];
                let own2 = fj.g_own[0] * fj.g_own[0] * w[fj.owner * 2]
                    + fj.g_own[1] * fj.g_own[1] * w[fj.owner * 2 + 1];
                if fj.neighbor == NO_NEIGHBOR {
                    acc += own2;
                } else {
                    let nei2 = fj.g_nei[0] * fj.g_nei[0] * w[fj.neighbor * 2]
                        + fj.g_nei[1] * fj.g_nei[1] * w[fj.neighbor * 2 + 1];
                    // Both incident cells receive own2+nei2 (see the scatter).
                    acc += own2 + nei2;
                }
            }
            *dc = acc;
        }
    });
    let deflate = |v: &mut [f64]| {
        // Closed domain (no outlet u-sensitivity): S has the constant null
        // space; keep everything mean-free so PCG stays on the range.
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.iter_mut().for_each(|x| *x -= mean);
    };

    let mut b = target.to_vec();
    if !grounded {
        deflate(&mut b);
    }
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut lam = vec![0.0f64; n];
    let mut iters = 0usize;
    if bnorm > 0.0 {
        let mut t = vec![0.0f64; 2 * n];
        let mut r = b.clone();
        let mut z: Vec<f64> = r
            .iter()
            .zip(diag.iter())
            .map(|(ri, di)| if *di > 0.0 { ri / di } else { 0.0 })
            .collect();
        if !grounded {
            deflate(&mut z);
        }
        let mut p = z.clone();
        let mut rz = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
        let mut ap = vec![0.0f64; n];
        for it in 0..CG_MAX_ITERS {
            apply_gt_w(&p, &mut t);
            apply_g(&t, &mut ap);
            let pap = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum::<f64>();
            if pap <= 0.0 || !pap.is_finite() {
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                lam[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            iters = it + 1;
            let rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rnorm <= CG_RTOL * bnorm {
                break;
            }
            for i in 0..n {
                z[i] = if diag[i] > 0.0 { r[i] / diag[i] } else { 0.0 };
            }
            if !grounded {
                deflate(&mut z);
            }
            let rz_new = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
            let beta = rz_new / rz;
            rz = rz_new;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }
    }
    let mut du = vec![0.0f64; 2 * n];
    apply_gt_w(&lam, &mut du);
    (du, iters)
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

/// Project RESIZE-transferred rows onto the solver's own mass-row constraint.
///
/// Contract: `cpu` must already hold `rows` (a full-mesh `reinit_cells` — all
/// time levels flat, `x` in sync) and have dt/BC values applied via the same
/// params path as the production solver. On return `cpu` holds the FINAL rows
/// (corrected when `applied`, the originals otherwise), and `rows` matches.
pub fn project_transferred_state(
    cpu: &mut CpuSolver,
    mesh: &Mesh,
    layout: ProjectionLayout,
    rows: &mut [f32],
    vols: &[f64],
) -> Result<MassProjectionOutcome, String> {
    let n = mesh.num_cells();
    let stride = layout.stride;
    if rows.len() != n * stride {
        return Err(format!(
            "mass projection: rows length {} != n_cells*stride {}",
            rows.len(),
            n * stride
        ));
    }
    let Some((u_rank, p_rank)) = coupled_ranks(cpu) else {
        // Not a momentum/pressure system — nothing to project.
        return Ok(MassProjectionOutcome::default());
    };

    let profile = super::moving_mesh_driver::ale_profile_level() >= 2;
    let eps_t = std::time::Instant::now();
    let (eps, w) = assemble_eps_w(cpu, rows, stride, p_rank, u_rank, n);
    let eps_ms = eps_t.elapsed().as_secs_f32() * 1000.0;
    let pre = inf_norm(&eps);
    let mut out = MassProjectionOutcome {
        pre,
        post: pre,
        cg_iters: 0,
        applied: false,
    };
    if !pre.is_finite() {
        return Err("mass projection: non-finite mass residual at transferred state".into());
    }
    if pre == 0.0 {
        return Ok(out);
    }

    let threads = cpu.threads();
    let jac_t = std::time::Instant::now();
    let (jac, grounded) = build_face_jacobian(cpu, mesh, layout, rows, u_rank);
    let jac_ms = jac_t.elapsed().as_secs_f32() * 1000.0;
    let cg_t = std::time::Instant::now();
    let (du, iters) = solve_du(&jac, &w, &eps, grounded, n, threads);
    let cg_ms = cg_t.elapsed().as_secs_f32() * 1000.0;
    out.cg_iters = iters;
    if du.iter().any(|v| !v.is_finite()) {
        return Ok(out);
    }

    // Apply + verify by re-assembly; revert unless the measured residual
    // actually improved.
    let verify_t = std::time::Instant::now();
    let original: Vec<f32> = rows.to_vec();
    for c in 0..n {
        rows[c * stride + layout.u_off] =
            (rows[c * stride + layout.u_off] as f64 + du[c * 2]) as f32;
        rows[c * stride + layout.u_off + 1] =
            (rows[c * stride + layout.u_off + 1] as f64 + du[c * 2 + 1]) as f32;
    }
    let cells: Vec<usize> = (0..n).collect();
    cpu.reinit_cells(&cells, rows, vols)?;
    let (r2, _) = assemble_eps_w(cpu, rows, stride, p_rank, u_rank, n);
    let post = inf_norm(&r2);
    if post.is_finite() && post < pre {
        out.post = post;
        out.applied = true;
    } else {
        rows.copy_from_slice(&original);
        cpu.reinit_cells(&cells, rows, vols)?;
    }
    if profile {
        eprintln!(
            "[ale-profile]   proj(transfer): eps_w={eps_ms:.2} jac={jac_ms:.2} \
             cg={cg_ms:.2}({iters}it) verify={:.2}ms n={n} faces={}",
            verify_t.elapsed().as_secs_f32() * 1000.0,
            mesh.num_faces(),
        );
    }
    Ok(out)
}

/// Project a RECYCLE re-seed onto the solver's own mass-row constraint, on
/// the LIVE mid-run solver (armed ALE seam, real time history).
///
/// Contract: the recycle `reinit_cells` already ran (`cpu` holds the fresh
/// parcel rows); `recycled`/`recycled_vols` are the re-seeded cells and the
/// volumes that reinit used. The target is the mass-row residual MASKED to
/// the recycled cells' 2-ring face neighborhood — the only rows the re-seed
/// changed (shared-face fluxes + the lagged-gradient second ring); the rest
/// of the field keeps its physical pre-step residual untouched. The
/// correction is applied HISTORY-PRESERVING: live cells get a current-buffer
/// velocity update (the CPU `write_state_f32` semantics), recycled cells are
/// re-reinit'd (their history is the fresh-parcel row and stays flat at the
/// corrected value). CPU backend only — the recycle step already falls back
/// to the CPU regen path, and the GPU-solver variant would need a mid-run
/// state/history/mesh-flux mirror that no seam provides.
pub fn project_recycled_state(
    cpu: &mut CpuSolver,
    mesh: &Mesh,
    layout: ProjectionLayout,
    recycled: &[usize],
    recycled_vols: &[f64],
) -> Result<MassProjectionOutcome, String> {
    let n = mesh.num_cells();
    let stride = layout.stride;
    let Some((u_rank, p_rank)) = coupled_ranks(cpu) else {
        return Ok(MassProjectionOutcome::default());
    };
    if recycled.is_empty() {
        return Ok(MassProjectionOutcome::default());
    }
    let mut state = cpu.read_state_f32();
    if state.len() != n * stride {
        return Err(format!(
            "recycle projection: state length {} != n_cells*stride {}",
            state.len(),
            n * stride
        ));
    }

    // The affected set: recycled cells + 2 face-adjacency rings.
    let mut affected = vec![false; n];
    for &c in recycled {
        if c >= n {
            return Err(format!("recycle projection: cell {c} out of range ({n})"));
        }
        affected[c] = true;
    }
    for _ in 0..2 {
        let mut next = affected.clone();
        for f in 0..mesh.num_faces() {
            if let Some(nb) = mesh.face_neighbor[f] {
                let o = mesh.face_owner[f];
                if affected[o] {
                    next[nb] = true;
                }
                if affected[nb] {
                    next[o] = true;
                }
            }
        }
        affected = next;
    }
    let inf_masked = |v: &[f64]| -> f64 {
        v.iter()
            .zip(affected.iter())
            .filter(|(_, &m)| m)
            .fold(0.0f64, |m, (&x, _)| m.max(x.abs()))
    };

    let profile = super::moving_mesh_driver::ale_profile_level() >= 2;
    let eps_t = std::time::Instant::now();
    let (r1, w) = assemble_eps_w(cpu, &state, stride, p_rank, u_rank, n);
    let eps_ms = eps_t.elapsed().as_secs_f32() * 1000.0;
    let pre = inf_masked(&r1);
    let pre_global = inf_norm(&r1);
    let mut out = MassProjectionOutcome {
        pre,
        post: pre,
        cg_iters: 0,
        applied: false,
    };
    if !pre.is_finite() {
        return Err("recycle projection: non-finite mass residual".into());
    }
    if pre == 0.0 {
        return Ok(out);
    }
    let target: Vec<f64> = r1
        .iter()
        .zip(affected.iter())
        .map(|(&r, &m)| if m { r } else { 0.0 })
        .collect();

    let threads = cpu.threads();
    let jac_t = std::time::Instant::now();
    let (jac, grounded) = build_face_jacobian(cpu, mesh, layout, &state, u_rank);
    let jac_ms = jac_t.elapsed().as_secs_f32() * 1000.0;
    let cg_t = std::time::Instant::now();
    let (du, iters) = solve_du(&jac, &w, &target, grounded, n, threads);
    let cg_ms = cg_t.elapsed().as_secs_f32() * 1000.0;
    out.cg_iters = iters;
    if du.iter().any(|v| !v.is_finite()) {
        return Ok(out);
    }

    let verify_t = std::time::Instant::now();
    let original = state.clone();
    for c in 0..n {
        state[c * stride + layout.u_off] =
            (state[c * stride + layout.u_off] as f64 + du[c * 2]) as f32;
        state[c * stride + layout.u_off + 1] =
            (state[c * stride + layout.u_off + 1] as f64 + du[c * 2 + 1]) as f32;
    }
    let write_all = |cpu: &mut CpuSolver, snapshot: &[f32]| -> Result<(), String> {
        // Live cells: CURRENT buffer only (the CPU `write_state_f32`
        // semantics) — the correction is part of the t^n state and must not
        // rewrite t^{n-1}. Recycled cells: full re-reinit (all levels + x),
        // keeping their fresh-parcel history flat at the corrected row.
        cpu.write_state_f32(snapshot)?;
        let mut rec_rows: Vec<f32> = Vec::with_capacity(recycled.len() * stride);
        for &c in recycled {
            rec_rows.extend_from_slice(&snapshot[c * stride..(c + 1) * stride]);
        }
        cpu.reinit_cells(recycled, &rec_rows, recycled_vols)
    };
    write_all(cpu, &state)?;

    let (r2, _) = assemble_eps_w(cpu, &state, stride, p_rank, u_rank, n);
    let post = inf_masked(&r2);
    let post_global = inf_norm(&r2);
    // Keep only if the recycle neighborhood measurably improved AND the
    // rest of the field was not degraded (the spread correction may shift
    // far-field residuals by roundoff-scale amounts; 0.1% headroom).
    if post.is_finite() && post < pre && post_global <= pre_global.max(pre) * 1.001 {
        out.post = post;
        out.applied = true;
    } else {
        write_all(cpu, &original)?;
    }
    if profile {
        eprintln!(
            "[ale-profile]   proj(recycle): eps_w={eps_ms:.2} jac={jac_ms:.2} \
             cg={cg_ms:.2}({iters}it) verify={:.2}ms n={n} faces={} recycled={}",
            verify_t.elapsed().as_secs_f32() * 1000.0,
            mesh.num_faces(),
            recycled.len(),
        );
    }
    Ok(out)
}
