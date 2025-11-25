use crate::solver::float::{Float, Simd};

pub trait SolverOps<T: Float> {
    fn dot(&self, a: &[T], b: &[T]) -> T;
    fn mat_vec_mul(&self, matrix: &SparseMatrix<T>, x: &[T], y: &mut [T]);
    fn norm(&self, a: &[T]) -> T {
        self.dot(a, a).sqrt()
    }
    fn exchange_halo(&self, _data: &[T]) -> Vec<T> {
        Vec::new()
    }
}

pub struct SerialOps;
impl<T: Float> SolverOps<T> for SerialOps {
    fn dot(&self, a: &[T], b: &[T]) -> T {
        dot(a, b)
    }
    fn mat_vec_mul(&self, matrix: &SparseMatrix<T>, x: &[T], y: &mut [T]) {
        matrix.mat_vec_mul(x, y);
    }
}

#[derive(Clone, Debug)]
pub struct SparseMatrix<T: Float> {
    pub values: Vec<T>,
    pub col_indices: Vec<usize>,
    pub row_offsets: Vec<usize>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl<T: Float> SparseMatrix<T> {
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_offsets: vec![0; n_rows + 1],
            n_rows,
            n_cols,
        }
    }

    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: &[(usize, usize, T)]) -> Self {
        let mut row_counts = vec![0; n_rows];
        for &(r, _, _) in triplets {
            row_counts[r] += 1;
        }
        
        let mut row_offsets = vec![0; n_rows + 1];
        for i in 0..n_rows {
            row_offsets[i+1] = row_offsets[i] + row_counts[i];
        }
        
        let mut mat = Self::new(n_rows, n_cols);
        mat.row_offsets = row_offsets.clone();
        mat.values = vec![T::zero(); triplets.len()];
        mat.col_indices = vec![0; triplets.len()];
        
        let mut current_row_indices = row_offsets.clone();
        
        for &(r, c, v) in triplets {
            let idx = current_row_indices[r];
            mat.values[idx] = v;
            mat.col_indices[idx] = c;
            current_row_indices[r] += 1;
        }
        
        mat
    }

    pub fn mat_vec_mul(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.n_cols);
        assert_eq!(y.len(), self.n_rows);
        
        for i in 0..self.n_rows {
            let mut sum = T::zero();
            for j in self.row_offsets[i]..self.row_offsets[i+1] {
                sum = sum + self.values[j] * x[self.col_indices[j]];
            }
            y[i] = sum;
        }
    }
}

pub fn solve_bicgstab<T: Float, O: SolverOps<T>>(
    a: &SparseMatrix<T>,
    b: &[T],
    x: &mut [T],
    max_iter: usize,
    tol: T,
    ops: &O,
) -> (usize, T, T) {
    let n = b.len();
    let mut r = vec![T::zero(); n];
    ops.mat_vec_mul(a, x, &mut r);
    
    // r = b - Ax
    let mut i = 0;
    let lanes = T::Simd::LANES;
    while i + lanes <= n {
        let vb = T::Simd::from_slice(&b[i..i+lanes]);
        let _vr = T::Simd::from_slice(&r[i..i+lanes]);
        let res = vb - _vr;
        res.write_to_slice(&mut r[i..i+lanes]);
        i += lanes;
    }
    while i < n {
        r[i] = b[i] - r[i];
        i += 1;
    }
    
    let init_resid = ops.norm(&r);
    if init_resid < tol {
        return (0, init_resid, init_resid);
    }

    let r0 = r.clone();
    let mut rho_old = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    let mut v = vec![T::zero(); n];
    let mut p = vec![T::zero(); n];
    
    let mut rho_new;
    let mut s = vec![T::zero(); n];
    let mut t = vec![T::zero(); n];
    
    for iter in 0..max_iter {
        rho_new = ops.dot(&r0, &r);
        
        if rho_new.is_nan() {
            println!("BiCGStab: rho_new is NaN at iter {}", iter);
            return (iter, T::nan(), init_resid);
        }
        
        if rho_new.abs() < T::val_from_f64(1e-20) {
            break;
        }
        
        if iter == 0 {
            // p = r
            let mut i = 0;
            while i + lanes <= n {
                let vr = T::Simd::from_slice(&r[i..i+lanes]);
                vr.write_to_slice(&mut p[i..i+lanes]);
                i += lanes;
            }
            while i < n {
                p[i] = r[i];
                i += 1;
            }
        } else {
            let beta = (rho_new / rho_old) * (alpha / omega);
            let v_beta = T::Simd::splat(beta);
            let v_omega = T::Simd::splat(omega);
            
            let mut i = 0;
            while i + lanes <= n {
                let vr = T::Simd::from_slice(&r[i..i+lanes]);
                let vp = T::Simd::from_slice(&p[i..i+lanes]);
                let vv = T::Simd::from_slice(&v[i..i+lanes]);
                let res = vr + v_beta * (vp - v_omega * vv);
                res.write_to_slice(&mut p[i..i+lanes]);
                i += lanes;
            }
            while i < n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
                i += 1;
            }
        }

        ops.mat_vec_mul(a, &p, &mut v);
        let r0_v = ops.dot(&r0, &v);
        if r0_v.abs() < T::val_from_f64(1e-20) {
            break;
        }
        alpha = rho_new / r0_v;
        
        // s = r - alpha * v
        let v_alpha = T::Simd::splat(alpha);
        let mut i = 0;
        while i + lanes <= n {
            let vr = T::Simd::from_slice(&r[i..i+lanes]);
            let vv = T::Simd::from_slice(&v[i..i+lanes]);
            let res = vr - v_alpha * vv;
            res.write_to_slice(&mut s[i..i+lanes]);
            i += lanes;
        }
        while i < n {
            s[i] = r[i] - alpha * v[i];
            i += 1;
        }
        
        let norm_s = ops.norm(&s);
        if norm_s < tol {
            // x = x + alpha * p
            let mut i = 0;
            while i + lanes <= n {
                let vx = T::Simd::from_slice(&x[i..i+lanes]);
                let vp = T::Simd::from_slice(&p[i..i+lanes]);
                let res = vx + v_alpha * vp;
                res.write_to_slice(&mut x[i..i+lanes]);
                i += lanes;
            }
            while i < n {
                x[i] = x[i] + alpha * p[i];
                i += 1;
            }
            return (iter + 1, norm_s, init_resid);
        }
        
        ops.mat_vec_mul(a, &s, &mut t);
        let t_s = ops.dot(&t, &s);
        let t_t = ops.dot(&t, &t);
        
        if t_t.abs() < T::val_from_f64(1e-20) {
             omega = T::zero();
        } else {
             omega = t_s / t_t;
        }
        
        // x = x + alpha * p + omega * s
        // r = s - omega * t
        let v_omega = T::Simd::splat(omega);
        let mut i = 0;
        while i + lanes <= n {
            let vx = T::Simd::from_slice(&x[i..i+lanes]);
            let vp = T::Simd::from_slice(&p[i..i+lanes]);
            let vs = T::Simd::from_slice(&s[i..i+lanes]);
            let _vr = T::Simd::from_slice(&r[i..i+lanes]); // r becomes new r
            let vt = T::Simd::from_slice(&t[i..i+lanes]);
            
            let res_x = vx + v_alpha * vp + v_omega * vs;
            let res_r = vs - v_omega * vt; // r = s - omega * t
            
            res_x.write_to_slice(&mut x[i..i+lanes]);
            res_r.write_to_slice(&mut r[i..i+lanes]);
            i += lanes;
        }
        while i < n {
            x[i] = x[i] + alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
            i += 1;
        }
        
        let resid = ops.norm(&r);
        if resid < tol {
            return (iter + 1, resid, init_resid);
        }
        
        if omega.abs() < T::val_from_f64(1e-20) {
            break;
        }
        
        rho_old = rho_new;
    }
    
    (max_iter, ops.norm(&r), init_resid)
}

pub fn solve_cg<T: Float, O: SolverOps<T>>(
    a: &SparseMatrix<T>,
    b: &[T],
    x: &mut [T],
    max_iter: usize,
    tol: T,
    ops: &O,
) -> (usize, T, T) {
    let n = b.len();
    let mut r = vec![T::zero(); n];
    ops.mat_vec_mul(a, x, &mut r);
    
    // r = b - Ax
    let mut i = 0;
    let lanes = T::Simd::LANES;
    while i + lanes <= n {
        let vb = T::Simd::from_slice(&b[i..i+lanes]);
        let _vr = T::Simd::from_slice(&r[i..i+lanes]);
        let res = vb - _vr;
        res.write_to_slice(&mut r[i..i+lanes]);
        i += lanes;
    }
    while i < n {
        r[i] = b[i] - r[i];
        i += 1;
    }
    
    let init_resid = ops.norm(&r);
    
    let mut p = r.clone();
    let mut rsold = ops.dot(&r, &r);
    let mut q = vec![T::zero(); n];
    
    for iter in 0..max_iter {
        if rsold.sqrt() < tol {
            return (iter, rsold.sqrt(), init_resid);
        }
        
        ops.mat_vec_mul(a, &p, &mut q);
        let p_q = ops.dot(&p, &q);
        if p_q.abs() < T::val_from_f64(1e-20) {
            break;
        }
        let alpha = rsold / p_q;
        
        let v_alpha = T::Simd::splat(alpha);
        let mut i = 0;
        while i + lanes <= n {
            let vx = T::Simd::from_slice(&x[i..i+lanes]);
            let vp = T::Simd::from_slice(&p[i..i+lanes]);
            let vr = T::Simd::from_slice(&r[i..i+lanes]);
            let vq = T::Simd::from_slice(&q[i..i+lanes]);
            
            let res_x = vx + v_alpha * vp;
            let res_r = vr - v_alpha * vq;
            
            res_x.write_to_slice(&mut x[i..i+lanes]);
            res_r.write_to_slice(&mut r[i..i+lanes]);
            i += lanes;
        }
        while i < n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * q[i];
            i += 1;
        }
        
        let rsnew = ops.dot(&r, &r);
        if rsnew.sqrt() < tol {
            return (iter + 1, rsnew.sqrt(), init_resid);
        }
        
        let p_val = rsnew / rsold;
        let v_pval = T::Simd::splat(p_val);
        let mut i = 0;
        while i + lanes <= n {
            let vr = T::Simd::from_slice(&r[i..i+lanes]);
            let vp = T::Simd::from_slice(&p[i..i+lanes]);
            let res = vr + v_pval * vp;
            res.write_to_slice(&mut p[i..i+lanes]);
            i += lanes;
        }
        while i < n {
            p[i] = r[i] + p_val * p[i];
            i += 1;
        }
        rsold = rsnew;
    }
    
    (max_iter, rsold.sqrt(), init_resid)
}

pub fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    let mut sum = T::Simd::splat(T::zero());
    let mut i = 0;
    let n = a.len();
    let lanes = T::Simd::LANES;
    while i + lanes <= n {
        let va = T::Simd::from_slice(&a[i..i+lanes]);
        let vb = T::Simd::from_slice(&b[i..i+lanes]);
        sum = sum + va * vb;
        i += lanes;
    }
    
    // Reduce sum
    let mut arr = vec![T::zero(); lanes];
    sum.write_to_slice(&mut arr);
    let mut s = arr.iter().cloned().sum();
    
    while i < n {
        s = s + a[i] * b[i];
        i += 1;
    }
    s
}

#[allow(dead_code)]
fn norm<T: Float>(a: &[T]) -> T {
    dot(a, a).sqrt()
}
