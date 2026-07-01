//! Scalar algebraic multigrid (plain aggregation) for the CPU backend's Schur
//! pressure block.
//!
//! The Schur preconditioner needs an approximate solve of the pressure block
//! `A_pp p = g` every FGMRES iteration. A Jacobi-preconditioned BiCGSTAB works
//! at coarse resolution but its iteration count grows like the Poisson
//! condition number (~h^-2): on fine meshes it burns its whole budget without
//! converging and the outer FGMRES pays for it in extra iterations. This module
//! provides the mesh-scalable alternative the GPU reaches for with its AMG
//! branch: a plain-aggregation V-cycle used as the preconditioner inside the
//! inner BiCGSTAB.
//!
//! Split in two so the expensive part is done once:
//!  * [`AmgHierarchy`] — TOPOLOGY ONLY (aggregation, coarse CSR patterns, the
//!    fine-nnz → coarse-nnz scatter map for the Galerkin product, restriction
//!    member lists). Built once per solver from the block topology; the matrix
//!    VALUES never affect it.
//!  * [`AmgSolver`] — per-solve values: Galerkin-coarsened `A_l` on every
//!    level (a single O(nnz) scatter-add per level), inverse diagonals, and a
//!    dense inverse of the coarsest level.
//!
//! Determinism: aggregation is a fixed-order greedy pass; the Galerkin
//! scatter-add and all reductions run in fixed serial order or over disjoint
//! per-node chunks, so results are bit-identical across thread counts (the
//! same contract as the rest of `cpu::linalg`).

use crate::solver::cpu::parallel::{par_map_into, parallel_cell_chunks_mut};

/// Stop coarsening when a level is at most this many unknowns; solve it densely.
const COARSEST_N: usize = 64;
/// Damped-Jacobi smoothing weight (standard 2/3 for aggregation AMG).
const JACOBI_OMEGA: f64 = 2.0 / 3.0;

/// One level's topology: the CSR pattern of `A_l`, plus the maps tying it to
/// the next-coarser level.
struct LevelTopo {
    /// CSR pattern of this level's matrix.
    row_offsets: Vec<u32>,
    col_indices: Vec<u32>,
    /// Aggregate id of each node on THIS level (length = this level's n).
    agg: Vec<u32>,
    /// For each nnz of THIS level's matrix, the destination nnz index in the
    /// next-coarser matrix (Galerkin scatter target).
    coarse_nnz_of: Vec<u32>,
    /// Restriction member lists: aggregate -> its fine nodes (CSR-ish).
    member_offsets: Vec<u32>,
    members: Vec<u32>,
    /// Next-coarser level's node count.
    coarse_n: usize,
}

/// Topology-only AMG hierarchy (values-independent; build once).
pub struct AmgHierarchy {
    levels: Vec<LevelTopo>,
    /// Coarsest level's CSR pattern (= the finest pattern when `levels` is
    /// empty) and node count; the coarsest matrix is dense-inverted per solve.
    coarsest_row_offsets: Vec<u32>,
    coarsest_col_indices: Vec<u32>,
    coarsest_n: usize,
}

/// Greedy plain aggregation over an undirected adjacency (CSR), restricted to
/// STRONG connections: |a_ij| >= theta * sqrt(|a_ii| * |a_jj|). On heterogeneous
/// operators (cut-cell pressure blocks: face-area/volume ratios vary by orders
/// of magnitude) aggregating across weak couplings destroys the coarse-grid
/// correction; strength filtering is the standard fix. Returns (aggregate id
/// per node, number of aggregates). Fixed visiting order = deterministic.
fn aggregate(
    n: usize,
    row_offsets: &[u32],
    col_indices: &[u32],
    values: &[f64],
    theta: f64,
) -> (Vec<u32>, usize) {
    const UNASSIGNED: u32 = u32::MAX;
    // Diagonal magnitudes for the strength test.
    let mut diag = vec![0.0f64; n];
    for i in 0..n {
        for k in row_offsets[i] as usize..row_offsets[i + 1] as usize {
            if col_indices[k] as usize == i {
                diag[i] = values[k].abs();
            }
        }
    }
    let strong = |i: usize, k: usize| -> bool {
        let j = col_indices[k] as usize;
        if j == i {
            return false;
        }
        let denom = (diag[i] * diag[j]).sqrt();
        denom > 0.0 && values[k].abs() >= theta * denom
    };

    let mut agg = vec![UNASSIGNED; n];
    let mut num_agg = 0u32;
    // Pass 1: seed aggregates from nodes whose strong neighbourhood is untouched.
    for i in 0..n {
        if agg[i] != UNASSIGNED {
            continue;
        }
        let (s, e) = (row_offsets[i] as usize, row_offsets[i + 1] as usize);
        let untouched = (s..e).all(|k| !strong(i, k) || agg[col_indices[k] as usize] == UNASSIGNED);
        if untouched {
            agg[i] = num_agg;
            for k in s..e {
                if strong(i, k) {
                    agg[col_indices[k] as usize] = num_agg;
                }
            }
            num_agg += 1;
        }
    }
    // Pass 2: attach leftovers to the first strongly-connected aggregated
    // neighbour; fall back to any aggregated neighbour; else singleton.
    for i in 0..n {
        if agg[i] != UNASSIGNED {
            continue;
        }
        let (s, e) = (row_offsets[i] as usize, row_offsets[i + 1] as usize);
        let joined = (s..e)
            .find(|&k| strong(i, k) && agg[col_indices[k] as usize] != UNASSIGNED)
            .or_else(|| {
                (s..e).find(|&k| {
                    col_indices[k] as usize != i && agg[col_indices[k] as usize] != UNASSIGNED
                })
            })
            .map(|k| agg[col_indices[k] as usize]);
        agg[i] = joined.unwrap_or_else(|| {
            num_agg += 1;
            num_agg - 1
        });
    }
    (agg, num_agg as usize)
}

impl AmgHierarchy {
    /// Build the hierarchy from the finest CSR pattern plus REPRESENTATIVE
    /// matrix values (used only for strength-of-connection aggregation — the
    /// coefficient PATTERN is dominated by static mesh geometry, so a
    /// hierarchy aggregated from the first assembled matrix stays good for
    /// the whole run; the per-solve values are re-Galerkin'd every
    /// [`AmgSolver::assemble`]).
    pub fn build(row_offsets: &[u32], col_indices: &[u32], values: &[f32]) -> Self {
        let theta = std::env::var("CFD2_CPU_AMG_THETA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.08);
        let mut levels = Vec::new();
        let mut ro: Vec<u32> = row_offsets.to_vec();
        let mut ci: Vec<u32> = col_indices.to_vec();
        let mut vals: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let mut n = ro.len() - 1;

        while n > COARSEST_N && levels.len() < 24 {
            let (agg, nc) = aggregate(n, &ro, &ci, &vals, theta);
            if nc >= n {
                break; // no reduction possible (degenerate graph)
            }

            // Coarse CSR pattern + fine-nnz -> coarse-nnz scatter map.
            // Row-by-row: collect the coarse columns touched by each coarse row.
            let nnz = ci.len();
            let mut coarse_rows: Vec<Vec<u32>> = vec![Vec::new(); nc];
            for i in 0..n {
                let (s, e) = (ro[i] as usize, ro[i + 1] as usize);
                let ci_row = &ci[s..e];
                let coarse_i = agg[i] as usize;
                for &j in ci_row {
                    let cj = agg[j as usize];
                    if !coarse_rows[coarse_i].contains(&cj) {
                        coarse_rows[coarse_i].push(cj);
                    }
                }
            }
            let mut cro = Vec::with_capacity(nc + 1);
            cro.push(0u32);
            let mut cci = Vec::new();
            for row in coarse_rows.iter_mut() {
                row.sort_unstable();
                cci.extend_from_slice(row);
                cro.push(cci.len() as u32);
            }
            // Scatter map: fine nnz k in row i, col j -> coarse entry
            // (agg[i], agg[j]) located by binary search in the sorted row.
            let mut coarse_nnz_of = vec![0u32; nnz];
            for i in 0..n {
                let (s, e) = (ro[i] as usize, ro[i + 1] as usize);
                let coarse_i = agg[i] as usize;
                let (cs, ce) = (cro[coarse_i] as usize, cro[coarse_i + 1] as usize);
                for k in s..e {
                    let cj = agg[ci[k] as usize];
                    let pos = cci[cs..ce].binary_search(&cj).expect("coarse col present");
                    coarse_nnz_of[k] = (cs + pos) as u32;
                }
            }
            // Restriction member lists (aggregate -> fine nodes, ascending).
            let mut member_offsets = vec![0u32; nc + 1];
            for &a in &agg {
                member_offsets[a as usize + 1] += 1;
            }
            for i in 0..nc {
                member_offsets[i + 1] += member_offsets[i];
            }
            let mut cursor = member_offsets[..nc].to_vec();
            let mut members = vec![0u32; n];
            for (i, &a) in agg.iter().enumerate() {
                members[cursor[a as usize] as usize] = i as u32;
                cursor[a as usize] += 1;
            }

            // Coarse values for the NEXT level's strength test (same Galerkin
            // scatter the per-solve assemble applies).
            let mut cv = vec![0.0f64; cci.len()];
            for (k, &dst) in coarse_nnz_of.iter().enumerate() {
                cv[dst as usize] += vals[k];
            }

            levels.push(LevelTopo {
                row_offsets: ro.clone(),
                col_indices: ci.clone(),
                agg,
                coarse_nnz_of,
                member_offsets,
                members,
                coarse_n: nc,
            });
            ro = cro;
            ci = cci;
            vals = cv;
            n = nc;
        }

        Self {
            levels,
            coarsest_row_offsets: ro,
            coarsest_col_indices: ci,
            coarsest_n: n,
        }
    }

    /// Number of coarsening levels (0 = the finest matrix is already coarsest).
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }
}

/// Per-solve AMG operator: Galerkin values on all levels + smoother data.
pub struct AmgSolver<'h> {
    hier: &'h AmgHierarchy,
    /// values[l] = CSR values of level l's matrix (f64). values[0] = fine.
    values: Vec<Vec<f64>>,
    /// inv_diag[l][i] = 1 / A_l[i][i] (0 if the diagonal vanishes).
    inv_diag: Vec<Vec<f64>>,
    /// Dense row-major inverse of the coarsest matrix.
    coarsest_inv: Vec<f64>,
    threads: usize,
    /// Damped-Jacobi sweeps before/after the coarse correction
    /// (`CFD2_CPU_AMG_SWEEPS`, default 1 → V(1,1)).
    sweeps: usize,
}

/// `y = A x` for an f64 CSR level (parallel over disjoint row chunks).
fn spmv_f64(
    row_offsets: &[u32],
    col_indices: &[u32],
    values: &[f64],
    x: &[f64],
    y: &mut [f64],
    threads: usize,
) {
    let n = row_offsets.len() - 1;
    parallel_cell_chunks_mut(n, 1, threads, y, |row0, yc| {
        for (li, out) in yc.iter_mut().enumerate() {
            let row = row0 + li;
            let (s, e) = (row_offsets[row] as usize, row_offsets[row + 1] as usize);
            let mut sum = 0.0f64;
            for k in s..e {
                sum += values[k] * x[col_indices[k] as usize];
            }
            *out = sum;
        }
    });
}

impl<'h> AmgSolver<'h> {
    /// Galerkin-assemble all levels from the finest values (f32, as the
    /// assembly writes them).
    pub fn assemble(hier: &'h AmgHierarchy, fine_values: &[f32], threads: usize) -> Self {
        let nlev = hier.levels.len();
        let mut values: Vec<Vec<f64>> = Vec::with_capacity(nlev + 1);
        values.push(fine_values.iter().map(|&v| v as f64).collect());
        for (l, topo) in hier.levels.iter().enumerate() {
            let coarse_nnz = if l + 1 < nlev {
                hier.levels[l + 1].col_indices.len()
            } else {
                hier.coarsest_col_indices.len()
            };
            // Galerkin product with piecewise-constant P: a fixed-order serial
            // scatter-add (deterministic; O(nnz) per level).
            let mut cv = vec![0.0f64; coarse_nnz];
            let fv = &values[l];
            for (k, &dst) in topo.coarse_nnz_of.iter().enumerate() {
                cv[dst as usize] += fv[k];
            }
            values.push(cv);
        }

        // Inverse diagonals per smoothed level (all but the coarsest).
        let mut inv_diag = Vec::with_capacity(nlev);
        for (l, topo) in hier.levels.iter().enumerate() {
            let n = topo.row_offsets.len() - 1;
            let (ro, ci, v) = (&topo.row_offsets, &topo.col_indices, &values[l]);
            let mut d = vec![0.0f64; n];
            for i in 0..n {
                for k in ro[i] as usize..ro[i + 1] as usize {
                    if ci[k] as usize == i {
                        let dv = v[k];
                        d[i] = if dv.abs() > 1e-300 { 1.0 / dv } else { 0.0 };
                    }
                }
            }
            inv_diag.push(d);
        }

        // Dense inverse of the coarsest level.
        let nc = hier.coarsest_n;
        let (cro, cci) = (&hier.coarsest_row_offsets, &hier.coarsest_col_indices);
        let cvals = &values[nlev];
        let mut dense = vec![0.0f64; nc * nc];
        for i in 0..nc {
            for k in cro[i] as usize..cro[i + 1] as usize {
                dense[i * nc + cci[k] as usize] = cvals[k];
            }
        }
        let mut coarsest_inv = vec![0.0f64; nc * nc];
        super::linalg::invert_dense(nc, &dense, &mut coarsest_inv);

        let sweeps = std::env::var("CFD2_CPU_AMG_SWEEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .max(1);
        Self {
            hier,
            values,
            inv_diag,
            coarsest_inv,
            threads,
            sweeps,
        }
    }

    /// One V(1,1) cycle: `z ≈ A^{-1} b` from a ZERO initial guess (the
    /// preconditioner-apply contract). `z` and `b` are finest-level length.
    pub fn vcycle(&self, b: &[f64], z: &mut [f64]) {
        self.cycle_level(0, b, z);
    }

    fn cycle_level(&self, l: usize, b: &[f64], x: &mut [f64]) {
        let nlev = self.hier.levels.len();
        if l == nlev {
            // Coarsest: dense multiply x = A^{-1} b.
            let n = self.hier.coarsest_n;
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..n {
                    s += self.coarsest_inv[i * n + j] * b[j];
                }
                x[i] = s;
            }
            return;
        }
        let topo = &self.hier.levels[l];
        let n = topo.row_offsets.len() - 1;
        let (ro, ci, v) = (&topo.row_offsets, &topo.col_indices, &self.values[l]);
        let dinv = &self.inv_diag[l];
        let threads = self.threads;

        // Pre-smooth from zero guess: x = omega * Dinv * b, then further damped
        // Jacobi refinements when sweeps > 1.
        par_map_into(threads, x, |i| JACOBI_OMEGA * dinv[i] * b[i]);
        let mut ax = vec![0.0f64; n];
        for _ in 1..self.sweeps {
            spmv_f64(ro, ci, v, x, &mut ax, threads);
            let ax_r = &ax;
            crate::solver::cpu::parallel::par_update(threads, x, |i, xi| {
                *xi += JACOBI_OMEGA * dinv[i] * (b[i] - ax_r[i])
            });
        }

        // Residual r = b - A x.
        spmv_f64(ro, ci, v, x, &mut ax, threads);
        let mut r = vec![0.0f64; n];
        par_map_into(threads, &mut r, |i| b[i] - ax[i]);

        // Restrict: r_c[I] = sum of r over members of I (fixed member order).
        let nc = topo.coarse_n;
        let mut bc = vec![0.0f64; nc];
        parallel_cell_chunks_mut(nc, 1, threads, &mut bc, |c0, out| {
            for (li, o) in out.iter_mut().enumerate() {
                let cidx = c0 + li;
                let (s, e) = (
                    topo.member_offsets[cidx] as usize,
                    topo.member_offsets[cidx + 1] as usize,
                );
                let mut sum = 0.0f64;
                for &m in &topo.members[s..e] {
                    sum += r[m as usize];
                }
                *o = sum;
            }
        });

        // Coarse solve (recursive).
        let mut xc = vec![0.0f64; nc];
        self.cycle_level(l + 1, &bc, &mut xc);

        // Prolong: x[i] += xc[agg[i]].
        {
            let agg = &topo.agg;
            let xc = &xc;
            crate::solver::cpu::parallel::par_update(threads, x, |i, xi| {
                *xi += xc[agg[i] as usize]
            });
        }

        // Post-smooth: x += omega * Dinv * (b - A x), `sweeps` times.
        for _ in 0..self.sweeps {
            spmv_f64(ro, ci, v, x, &mut ax, threads);
            let ax_r = &ax;
            crate::solver::cpu::parallel::par_update(threads, x, |i, xi| {
                *xi += JACOBI_OMEGA * dinv[i] * (b[i] - ax_r[i])
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2D 5-point Poisson (Dirichlet) on an nx*ny grid, CSR with diagonal-first
    /// rows — Poisson-like, the exact shape of the Schur pressure block.
    fn poisson2d(nx: usize, ny: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let n = nx * ny;
        let mut ro = vec![0u32];
        let mut ci = Vec::new();
        let mut vals = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                ci.push(idx as u32);
                vals.push(4.0f32);
                let mut push = |c: usize| {
                    ci.push(c as u32);
                    vals.push(-1.0f32);
                };
                if i > 0 {
                    push(idx - 1);
                }
                if i + 1 < nx {
                    push(idx + 1);
                }
                if j > 0 {
                    push(idx - nx);
                }
                if j + 1 < ny {
                    push(idx + nx);
                }
                ro.push(ci.len() as u32);
            }
        }
        (ro, ci, vals)
    }

    #[test]
    fn hierarchy_coarsens_poisson() {
        let (ro, ci, vals) = poisson2d(64, 64);
        let h = AmgHierarchy::build(&ro, &ci, &vals);
        assert!(h.num_levels() >= 2, "expected >=2 levels, got {}", h.num_levels());
        assert!(h.coarsest_n <= COARSEST_N);
    }

    #[test]
    fn vcycle_preconditioned_bicgstab_converges_fast_and_thread_invariant() {
        use crate::solver::cpu::linalg::{bicgstab_pc, CsrView};
        let (nx, ny) = (96, 96);
        let (ro, ci, vals) = poisson2d(nx, ny);
        let n = nx * ny;
        let h = AmgHierarchy::build(&ro, &ci, &vals);
        let b: Vec<f32> = (0..n).map(|k| ((k * 13 % 31) as f32) - 15.0).collect();

        let solve = |threads: usize| {
            let amg = AmgSolver::assemble(&h, &vals, threads);
            let a = CsrView {
                row_offsets: &ro,
                col_indices: &ci,
                values: &vals,
                threads,
            };
            let mut x = vec![0.0f32; n];
            let stats =
                bicgstab_pc(&a, &b, &mut x, 30, 1e-8, &|r, z| amg.vcycle(r, z));
            (x, stats)
        };
        let (x1, s1) = solve(1);
        // AMG-preconditioned BiCGSTAB should crack a 9216-unknown Poisson in a
        // handful of iterations (Jacobi-BiCGSTAB needs hundreds).
        assert!(s1.converged, "did not converge: {s1:?}");
        assert!(s1.iters <= 25, "too many iterations: {}", s1.iters);

        // Thread-count invariance (the cpu-backend determinism contract).
        let (x8, s8) = solve(8);
        assert!(s8.converged);
        assert_eq!(s1.iters, s8.iters, "iteration count must not depend on threads");
        let maxd = x1
            .iter()
            .zip(&x8)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(maxd, 0.0, "threads=1 vs 8 differ: max|diff|={maxd:e}");
    }
}
