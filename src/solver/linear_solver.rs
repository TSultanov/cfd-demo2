use wide::f64x4;

#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_offsets: Vec<usize>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl SparseMatrix {
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_offsets: vec![0; n_rows + 1],
            n_rows,
            n_cols,
        }
    }

    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
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
        mat.values = vec![0.0; triplets.len()];
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

    pub fn mat_vec_mul(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols);
        assert_eq!(y.len(), self.n_rows);
        
        for i in 0..self.n_rows {
            let mut sum = 0.0;
            for j in self.row_offsets[i]..self.row_offsets[i+1] {
                sum += self.values[j] * x[self.col_indices[j]];
            }
            y[i] = sum;
        }
    }
}

pub fn solve_bicgstab(
    a: &SparseMatrix,
    b: &[f64],
    x: &mut [f64],
    max_iter: usize,
    tol: f64,
) -> (usize, f64, f64) {
    let n = b.len();
    let mut r = vec![0.0; n];
    a.mat_vec_mul(x, &mut r);
    
    // r = b - Ax
    let mut i = 0;
    while i + 4 <= n {
        let vb = f64x4::from(&b[i..i+4]);
        let vr = f64x4::from(&r[i..i+4]);
        let res = vb - vr;
        let res_arr: [f64; 4] = res.into();
        r[i..i+4].copy_from_slice(&res_arr);
        i += 4;
    }
    while i < n {
        r[i] = b[i] - r[i];
        i += 1;
    }
    
    let init_resid = norm(&r);
    if init_resid < tol {
        return (0, init_resid, init_resid);
    }

    let r0 = r.clone();
    let mut rho_old = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    
    let mut rho_new = 0.0;
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];
    
    let mut resid = init_resid;
    
    for iter in 0..max_iter {
        rho_new = dot(&r0, &r);
        
        if rho_new.is_nan() {
            println!("BiCGStab: rho_new is NaN at iter {}", iter);
            return (iter, f64::NAN, init_resid);
        }
        
        if rho_new.abs() < 1e-20 {
            break;
        }
        
        if iter == 0 {
            p.copy_from_slice(&r);
        } else {
            let beta = (rho_new / rho_old) * (alpha / omega);
            let v_beta = f64x4::splat(beta);
            let v_omega = f64x4::splat(omega);
            
            let mut i = 0;
            while i + 4 <= n {
                let vr = f64x4::from(&r[i..i+4]);
                let vp = f64x4::from(&p[i..i+4]);
                let vv = f64x4::from(&v[i..i+4]);
                let res = vr + v_beta * (vp - v_omega * vv);
                let res_arr: [f64; 4] = res.into();
                p[i..i+4].copy_from_slice(&res_arr);
                i += 4;
            }
            while i < n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
                i += 1;
            }
        }

        a.mat_vec_mul(&p, &mut v);
        let r0_v = dot(&r0, &v);
        if r0_v.abs() < 1e-20 {
            break;
        }
        alpha = rho_new / r0_v;
        
        let v_alpha = f64x4::splat(alpha);
        let mut i = 0;
        while i + 4 <= n {
            let vr = f64x4::from(&r[i..i+4]);
            let vv = f64x4::from(&v[i..i+4]);
            let res = vr - v_alpha * vv;
            let res_arr: [f64; 4] = res.into();
            s[i..i+4].copy_from_slice(&res_arr);
            i += 4;
        }
        while i < n {
            s[i] = r[i] - alpha * v[i];
            i += 1;
        }
        
        if norm(&s) < tol {
            let v_alpha = f64x4::splat(alpha);
            let mut i = 0;
            while i + 4 <= n {
                let vx = f64x4::from(&x[i..i+4]);
                let vp = f64x4::from(&p[i..i+4]);
                let res = vx + v_alpha * vp;
                let res_arr: [f64; 4] = res.into();
                x[i..i+4].copy_from_slice(&res_arr);
                i += 4;
            }
            while i < n {
                x[i] += alpha * p[i];
                i += 1;
            }
            return (iter, norm(&s), norm(&s) / init_resid);
        }

        a.mat_vec_mul(&s, &mut t);
        let t_t = dot(&t, &t);
        if t_t.abs() < 1e-20 {
            omega = 0.0;
        } else {
            omega = dot(&t, &s) / t_t;
        }
        
        let v_alpha = f64x4::splat(alpha);
        let v_omega = f64x4::splat(omega);
        let mut i = 0;
        while i + 4 <= n {
            let vx = f64x4::from(&x[i..i+4]);
            let vp = f64x4::from(&p[i..i+4]);
            let vs = f64x4::from(&s[i..i+4]);
            let vt = f64x4::from(&t[i..i+4]);
            
            let res_x = vx + v_alpha * vp + v_omega * vs;
            let res_r = vs - v_omega * vt;
            
            let res_x_arr: [f64; 4] = res_x.into();
            let res_r_arr: [f64; 4] = res_r.into();
            
            x[i..i+4].copy_from_slice(&res_x_arr);
            r[i..i+4].copy_from_slice(&res_r_arr);
            i += 4;
        }
        while i < n {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
            i += 1;
        }
        
        resid = norm(&r);
        if resid > 1e10 {
             println!("BiCGStab diverging at iter {}: resid={}", iter, resid);
             return (iter, resid, init_resid);
        }
        if resid < tol {
            return (iter + 1, resid, init_resid);
        }
        
        if omega.abs() < 1e-20 {
            break;
        }
        
        rho_old = rho_new;
    }
    
    (max_iter, resid, init_resid)
}

pub fn solve_cg(
    a: &SparseMatrix,
    b: &[f64],
    x: &mut [f64],
    max_iter: usize,
    tol: f64,
) -> (usize, f64, f64) {
    let n = b.len();
    let mut r = vec![0.0; n];
    a.mat_vec_mul(x, &mut r);
    // r = b - Ax
    let mut i = 0;
    while i + 4 <= n {
        let vb = f64x4::from(&b[i..i+4]);
        let vr = f64x4::from(&r[i..i+4]);
        let res = vb - vr;
        let res_arr: [f64; 4] = res.into();
        r[i..i+4].copy_from_slice(&res_arr);
        i += 4;
    }
    while i < n {
        r[i] = b[i] - r[i];
        i += 1;
    }
    
    let init_resid = norm(&r); // Use norm instead of manual dot
    
    let mut p = r.clone();
    let mut rsold = dot(&r, &r);
    let mut q = vec![0.0; n];
    
    for iter in 0..max_iter {
        if rsold.sqrt() < tol {
            return (iter, rsold.sqrt(), init_resid);
        }
        
        a.mat_vec_mul(&p, &mut q);
        let p_q = dot(&p, &q);
        if p_q.abs() < 1e-20 {
            break;
        }
        let alpha = rsold / p_q;
        
        let v_alpha = f64x4::splat(alpha);
        let mut i = 0;
        while i + 4 <= n {
            let vx = f64x4::from(&x[i..i+4]);
            let vp = f64x4::from(&p[i..i+4]);
            let vr = f64x4::from(&r[i..i+4]);
            let vq = f64x4::from(&q[i..i+4]);
            
            let res_x = vx + v_alpha * vp;
            let res_r = vr - v_alpha * vq;
            
            let res_x_arr: [f64; 4] = res_x.into();
            let res_r_arr: [f64; 4] = res_r.into();
            
            x[i..i+4].copy_from_slice(&res_x_arr);
            r[i..i+4].copy_from_slice(&res_r_arr);
            i += 4;
        }
        while i < n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
            i += 1;
        }
        
        let rsnew = dot(&r, &r);
        if rsnew.sqrt() < tol {
            return (iter + 1, rsnew.sqrt(), init_resid);
        }
        
        let p_val = rsnew / rsold;
        let v_pval = f64x4::splat(p_val);
        let mut i = 0;
        while i + 4 <= n {
            let vr = f64x4::from(&r[i..i+4]);
            let vp = f64x4::from(&p[i..i+4]);
            let res = vr + v_pval * vp;
            let res_arr: [f64; 4] = res.into();
            p[i..i+4].copy_from_slice(&res_arr);
            i += 4;
        }
        while i < n {
            p[i] = r[i] + p_val * p[i];
            i += 1;
        }
        rsold = rsnew;
    }
    
    (max_iter, rsold.sqrt(), init_resid)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = f64x4::splat(0.0);
    let mut i = 0;
    let n = a.len();
    while i + 4 <= n {
        let va = f64x4::from(&a[i..i+4]);
        let vb = f64x4::from(&b[i..i+4]);
        sum += va * vb;
        i += 4;
    }
    let mut s = sum.reduce_add();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
