//! CPU sparse linear algebra for the CPU backend.
//!
//! The GPU backend implements the linear solve as a stack of WGSL kernels
//! (FGMRES/CG/AMG/Schur over a CSR matrix). The CPU backend instead reads the
//! *same assembled CSR system* — `row_offsets` / `col_indices` / `matrix_values`
//! / `rhs`, produced by the generic-coupled assembly kernel — and solves it with
//! a plain Rust iterative solver. The matrix is generally nonsymmetric (the
//! convection operator), so we use **BiCGSTAB** with a **Jacobi (diagonal)**
//! preconditioner. We solve in `f64` for robustness even though the assembled
//! values are `f32`; results are compared to references at tolerance, not
//! bit-exactly against the GPU.

/// A borrowed CSR matrix (single scalar unknown per row).
pub struct CsrView<'a> {
    pub row_offsets: &'a [u32],
    pub col_indices: &'a [u32],
    pub values: &'a [f32],
}

impl CsrView<'_> {
    pub fn n(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// `y = A x` (both length n), computed in f64.
    fn spmv(&self, x: &[f64], y: &mut [f64]) {
        for row in 0..self.n() {
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            let mut sum = 0.0f64;
            for k in start..end {
                sum += self.values[k] as f64 * x[self.col_indices[k] as usize];
            }
            y[row] = sum;
        }
    }

    /// Diagonal entries (for Jacobi preconditioning).
    fn diagonal(&self) -> Vec<f64> {
        let n = self.n();
        let mut d = vec![0.0f64; n];
        for row in 0..n {
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            for k in start..end {
                if self.col_indices[k] as usize == row {
                    d[row] = self.values[k] as f64;
                }
            }
        }
        d
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Outcome of a linear solve.
#[derive(Debug, Clone, Copy)]
pub struct SolveStats {
    pub iters: usize,
    /// Final relative residual `||b - A x|| / ||b||`.
    pub rel_residual: f64,
    pub converged: bool,
}

/// Solve `A x = b` with preconditioned BiCGSTAB. `x` is used as the initial guess
/// and overwritten with the solution. Returns convergence statistics.
pub fn bicgstab(a: &CsrView, b: &[f32], x: &mut [f32], max_iter: usize, tol: f64) -> SolveStats {
    let n = a.n();
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);

    let diag = a.diagonal();
    let minv = |v: &[f64], out: &mut [f64]| {
        for i in 0..n {
            // Jacobi; guard a (pathological) zero diagonal.
            out[i] = if diag[i] != 0.0 { v[i] / diag[i] } else { v[i] };
        }
    };

    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = norm(&bf).max(1e-300);

    let mut xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();

    // r = b - A x
    let mut ax = vec![0.0f64; n];
    a.spmv(&xf, &mut ax);
    let mut r: Vec<f64> = (0..n).map(|i| bf[i] - ax[i]).collect();

    let finish = |xf: &[f64], x: &mut [f32], iters: usize, res: f64, conv: bool| -> SolveStats {
        for i in 0..n {
            x[i] = xf[i] as f32;
        }
        SolveStats {
            iters,
            rel_residual: res / bnorm,
            converged: conv,
        }
    };

    let mut res = norm(&r);
    if res / bnorm <= tol {
        return finish(&xf, x, 0, res, true);
    }

    let rhat = r.clone();
    let mut rho = 1.0f64;
    let mut alpha = 1.0f64;
    let mut omega = 1.0f64;
    let mut v = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];
    let (mut phat, mut shat, mut t) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);

    for iter in 1..=max_iter {
        let rho_new = dot(&rhat, &r);
        if rho_new.abs() < 1e-300 {
            // Breakdown; restart from the current residual.
            return finish(&xf, x, iter, res, res / bnorm <= tol);
        }
        let beta = (rho_new / rho) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        minv(&p, &mut phat);
        a.spmv(&phat, &mut v);
        let rhat_v = dot(&rhat, &v);
        alpha = rho_new / rhat_v;

        // s = r - alpha v  (reuse r as s after recording)
        let mut s = vec![0.0f64; n];
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let snorm = norm(&s);
        if snorm / bnorm <= tol {
            for i in 0..n {
                xf[i] += alpha * phat[i];
            }
            return finish(&xf, x, iter, snorm, true);
        }

        minv(&s, &mut shat);
        a.spmv(&shat, &mut t);
        let tt = dot(&t, &t).max(1e-300);
        omega = dot(&t, &s) / tt;

        for i in 0..n {
            xf[i] += alpha * phat[i] + omega * shat[i];
            r[i] = s[i] - omega * t[i];
        }
        res = norm(&r);
        if res / bnorm <= tol {
            return finish(&xf, x, iter, res, true);
        }
        if omega.abs() < 1e-300 {
            return finish(&xf, x, iter, res, res / bnorm <= tol);
        }
        rho = rho_new;
    }

    finish(&xf, x, max_iter, res, res / bnorm <= tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual_norm(a: &CsrView, b: &[f32], x: &[f32]) -> f64 {
        let n = a.n();
        let mut max = 0.0f64;
        for row in 0..n {
            let start = a.row_offsets[row] as usize;
            let end = a.row_offsets[row + 1] as usize;
            let mut s = 0.0f64;
            for k in start..end {
                s += a.values[k] as f64 * x[a.col_indices[k] as usize] as f64;
            }
            max = max.max((s - b[row] as f64).abs());
        }
        max
    }

    #[test]
    fn bicgstab_solves_nonsymmetric_m_matrix() {
        // [ 4 -1  0 ] [x0]   [1]
        // [-2  4 -1 ] [x1] = [2]
        // [ 0 -1  3 ] [x2]   [3]
        let row_offsets = [0u32, 2, 5, 7];
        let col_indices = [0u32, 1, 0, 1, 2, 1, 2];
        let values = [4.0f32, -1.0, -2.0, 4.0, -1.0, -1.0, 3.0];
        let a = CsrView {
            row_offsets: &row_offsets,
            col_indices: &col_indices,
            values: &values,
        };
        let b = [1.0f32, 2.0, 3.0];
        let mut x = [0.0f32; 3];
        let stats = bicgstab(&a, &b, &mut x, 100, 1e-10);
        assert!(stats.converged, "did not converge: {stats:?}");
        assert!(
            residual_norm(&a, &b, &x) < 1e-5,
            "residual too large: {}",
            residual_norm(&a, &b, &x)
        );
    }

    #[test]
    fn bicgstab_solves_larger_diffusion_like_system() {
        // 1D Laplacian-ish tridiagonal with Dirichlet-ish diagonal boost: SPD,
        // well-conditioned — exercises convergence on a bigger system.
        let n = 200usize;
        let mut row_offsets = vec![0u32];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push((i - 1) as u32);
                values.push(-1.0f32);
            }
            col_indices.push(i as u32);
            values.push(4.0f32);
            if i + 1 < n {
                col_indices.push((i + 1) as u32);
                values.push(-1.0f32);
            }
            row_offsets.push(col_indices.len() as u32);
        }
        let a = CsrView {
            row_offsets: &row_offsets,
            col_indices: &col_indices,
            values: &values,
        };
        let b = vec![1.0f32; n];
        let mut x = vec![0.0f32; n];
        let stats = bicgstab(&a, &b, &mut x, 500, 1e-10);
        assert!(stats.converged, "did not converge: {stats:?}");
        assert!(residual_norm(&a, &b, &x) < 1e-4);
    }
}
