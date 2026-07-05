//! Shared support for Method of Manufactured Solutions (MMS) convergence tests.
//!
//! MMS is the primary correctness oracle for the math-driven solver: a manufactured
//! exact solution `phi*(x, y, t)` is substituted into the PDE to derive a source term,
//! the source is added to the model as one more equation term (see the `_mms` model
//! variants), and the discrete solution is compared against `phi*` on a sequence of
//! refined meshes. The observed convergence order must match the design order of the
//! discretization.
//!
//! Error norms are accumulated in f64 on the host; state itself is f32 on the GPU, so
//! manufactured solutions should have O(1) amplitude and coarse-enough meshes that
//! discretization error stays well above the f32 noise floor (~1e-6).

use cfd2::solver::mesh::Mesh;
use cfd2::solver::UnifiedSolver;

/// Volume-weighted L2 and Linf errors of `sol` against `exact` at cell centers.
pub struct FieldErrors {
    pub l2: f64,
    pub linf: f64,
}

pub fn field_errors(mesh: &Mesh, sol: &[f64], exact: impl Fn(f64, f64) -> f64) -> FieldErrors {
    assert_eq!(sol.len(), mesh.num_cells(), "solution length != num_cells");
    let mut sum_sq = 0.0f64;
    let mut vol_sum = 0.0f64;
    let mut linf = 0.0f64;
    for i in 0..mesh.num_cells() {
        let e = sol[i] - exact(mesh.cell_cx[i], mesh.cell_cy[i]);
        let v = mesh.cell_vol[i];
        sum_sq += v * e * e;
        vol_sum += v;
        linf = linf.max(e.abs());
    }
    FieldErrors {
        l2: (sum_sq / vol_sum).sqrt(),
        linf,
    }
}

/// Volume-weighted L2/Linf errors of a Vector2 field against an exact
/// solution, accumulated over both components.
#[allow(dead_code)]
pub fn field_errors_vec2(
    mesh: &Mesh,
    sol: &[(f64, f64)],
    exact: impl Fn(f64, f64) -> (f64, f64),
) -> FieldErrors {
    assert_eq!(sol.len(), mesh.num_cells(), "solution length != num_cells");
    let mut sum_sq = 0.0f64;
    let mut vol_sum = 0.0f64;
    let mut linf = 0.0f64;
    for i in 0..mesh.num_cells() {
        let (ex, ey) = exact(mesh.cell_cx[i], mesh.cell_cy[i]);
        let dx = sol[i].0 - ex;
        let dy = sol[i].1 - ey;
        let v = mesh.cell_vol[i];
        sum_sq += v * (dx * dx + dy * dy);
        vol_sum += v;
        linf = linf.max(dx.abs()).max(dy.abs());
    }
    FieldErrors {
        l2: (sum_sq / vol_sum).sqrt(),
        linf,
    }
}

/// March to steady state watching a Vector2 field AND additional scalar
/// fields: the criterion fires only when every watched field's per-step max
/// delta is below `steady_tol`. Watching only the fast field (e.g. velocity)
/// stops the run while slower fields (e.g. an advected temperature) are
/// still mid-transient, and the leftover transient masquerades as
/// first-order discretization error in convergence studies.
#[allow(dead_code)]
pub fn run_to_steady_vec2_with_scalars(
    solver: &mut UnifiedSolver,
    field: &str,
    scalar_fields: &[&str],
    max_steps: usize,
    steady_tol: f64,
) -> Vec<(f64, f64)> {
    let read_scalars = |solver: &UnifiedSolver| -> Vec<Vec<f64>> {
        scalar_fields
            .iter()
            .map(|f| pollster::block_on(solver.get_field_scalar(f)).expect("read scalar field"))
            .collect()
    };
    let mut prev = pollster::block_on(solver.get_field_vec2(field)).expect("read field");
    let mut prev_scalars = read_scalars(solver);
    let mut last_delta = f64::INFINITY;
    for step in 0..max_steps {
        solver.step();
        let cur = pollster::block_on(solver.get_field_vec2(field)).expect("read field");
        let cur_scalars = read_scalars(solver);
        let mut max_delta = cur
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
            .fold(0.0f64, f64::max);
        for (c, p) in cur_scalars.iter().zip(prev_scalars.iter()) {
            max_delta = c
                .iter()
                .zip(p.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(max_delta, f64::max);
        }
        if max_delta < steady_tol {
            println!(
                "[mms] steady after {} steps (max_delta={max_delta:.3e})",
                step + 1
            );
            return cur;
        }
        if step % 10 == 0 {
            println!("[mms] step {step}: max_delta={max_delta:.3e}");
        }
        last_delta = max_delta;
        prev = cur;
        prev_scalars = cur_scalars;
    }
    panic!("did not reach steady state within {max_steps} steps (tol {steady_tol:.1e}, last max_delta={last_delta:.3e})");
}

/// Like `run_to_steady`, for a Vector2 field.
#[allow(dead_code)]
pub fn run_to_steady_vec2(
    solver: &mut UnifiedSolver,
    field: &str,
    max_steps: usize,
    steady_tol: f64,
) -> Vec<(f64, f64)> {
    let mut prev = pollster::block_on(solver.get_field_vec2(field)).expect("read field");
    let mut last_delta = f64::INFINITY;
    for step in 0..max_steps {
        solver.step();
        let cur = pollster::block_on(solver.get_field_vec2(field)).expect("read field");
        let max_delta = cur
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
            .fold(0.0f64, f64::max);
        if max_delta < steady_tol {
            println!(
                "[mms] steady after {} steps (max_delta={max_delta:.3e})",
                step + 1
            );
            return cur;
        }
        if step % 10 == 0 {
            println!("[mms] step {step}: max_delta={max_delta:.3e}");
        }
        last_delta = max_delta;
        prev = cur;
    }
    panic!("did not reach steady state within {max_steps} steps (tol {steady_tol:.1e}, last max_delta={last_delta:.3e})");
}

/// Effective mesh spacing for convergence fits on non-uniform meshes: the
/// maximum cell extent (width or height) over all cells.
///
/// On a graded mesh the LARGEST cell bounds the truncation error, so fitting
/// orders against it is the conservative choice — a mean or minimum spacing
/// would inflate apparent orders. On a uniform n x n unit square this reduces
/// to 1/n, so graded and uniform studies share one fit convention.
#[allow(dead_code)]
pub fn max_cell_extent(mesh: &Mesh) -> f64 {
    let mut h_max = 0.0f64;
    for c in 0..mesh.num_cells() {
        let start = mesh.cell_vertex_offsets[c];
        let end = mesh.cell_vertex_offsets[c + 1];
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for &v in &mesh.cell_vertices[start..end] {
            x_min = x_min.min(mesh.vx[v]);
            x_max = x_max.max(mesh.vx[v]);
            y_min = y_min.min(mesh.vy[v]);
            y_max = y_max.max(mesh.vy[v]);
        }
        h_max = h_max.max(x_max - x_min).max(y_max - y_min);
    }
    h_max
}

/// Column/row indexing of a tensor-product (structured rectangular) mesh,
/// derived from the unique vertex coordinates — valid on uniform AND graded
/// meshes, replacing the `round(coord / h)` grouping that assumes uniform
/// spacing.
#[allow(dead_code)]
pub struct TensorGrid {
    /// Sorted unique vertex x coordinates (column boundaries, len nx+1).
    pub x_bounds: Vec<f64>,
    /// Sorted unique vertex y coordinates (row boundaries, len ny+1).
    pub y_bounds: Vec<f64>,
    /// Per-cell column index (0..nx).
    pub cell_col: Vec<usize>,
    /// Per-cell row index (0..ny).
    pub cell_row: Vec<usize>,
}

#[allow(dead_code)]
impl TensorGrid {
    pub fn nx(&self) -> usize {
        self.x_bounds.len() - 1
    }
    pub fn ny(&self) -> usize {
        self.y_bounds.len() - 1
    }
    /// Cell index at (col, row); panics if the mesh has holes there.
    pub fn cell_at(&self, col: usize, row: usize) -> usize {
        for c in 0..self.cell_col.len() {
            if self.cell_col[c] == col && self.cell_row[c] == row {
                return c;
            }
        }
        panic!("no cell at column {col}, row {row}");
    }
}

#[allow(dead_code)]
pub fn tensor_grid(mesh: &Mesh) -> TensorGrid {
    let unique_sorted = |vals: &[f64]| -> Vec<f64> {
        let mut v = vals.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for x in v {
            if out.last().map_or(true, |&last| x - last > 1e-9) {
                out.push(x);
            }
        }
        out
    };
    let x_bounds = unique_sorted(&mesh.vx);
    let y_bounds = unique_sorted(&mesh.vy);
    let locate = |bounds: &[f64], c: f64| -> usize {
        match bounds.binary_search_by(|b| b.partial_cmp(&c).unwrap()) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    };
    let cell_col = (0..mesh.num_cells())
        .map(|i| locate(&x_bounds, mesh.cell_cx[i]))
        .collect();
    let cell_row = (0..mesh.num_cells())
        .map(|i| locate(&y_bounds, mesh.cell_cy[i]))
        .collect();
    TensorGrid {
        x_bounds,
        y_bounds,
        cell_col,
        cell_row,
    }
}

/// Least-squares slope of log(err) vs log(h): the observed convergence order.
pub fn fit_order(hs: &[f64], errors: &[f64]) -> f64 {
    assert_eq!(hs.len(), errors.len());
    assert!(hs.len() >= 2, "need at least two refinement levels");
    let xs: Vec<f64> = hs.iter().map(|h| h.ln()).collect();
    let ys: Vec<f64> = errors.iter().map(|e| e.max(1e-300).ln()).collect();
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        num += (x - mean_x) * (y - mean_y);
        den += (x - mean_x) * (x - mean_x);
    }
    num / den
}

/// Assert the observed convergence order and the finest-level absolute error.
///
/// The absolute cap guards against "consistent garbage": a wrong-but-self-consistent
/// discretization can converge at design order to the wrong answer.
pub fn assert_convergence_order(
    name: &str,
    hs: &[f64],
    errors: &[f64],
    expected_order: f64,
    slope_tol: f64,
    finest_error_cap: f64,
) {
    let order = fit_order(hs, errors);
    let finest = *errors.last().expect("at least one level");
    println!("[mms][{name}] h={hs:?}");
    println!("[mms][{name}] l2_err={errors:?}");
    println!("[mms][{name}] observed_order={order:.3} (expected >= {expected_order} - {slope_tol}), finest_err={finest:.3e} (cap {finest_error_cap:.3e})");
    assert!(
        order >= expected_order - slope_tol,
        "[mms][{name}] convergence order {order:.3} below expected {expected_order} - {slope_tol} (errors: {errors:?}, h: {hs:?})"
    );
    assert!(
        finest <= finest_error_cap,
        "[mms][{name}] finest-level error {finest:.3e} above cap {finest_error_cap:.3e}"
    );
}

/// March the (transient-form) solver until the field stops changing: steady-state solve.
///
/// Returns the steady field. Panics if the change between consecutive steps does not
/// drop below `steady_tol` (absolute, in field units) within `max_steps`.
pub fn run_to_steady(
    solver: &mut UnifiedSolver,
    field: &str,
    max_steps: usize,
    steady_tol: f64,
) -> Vec<f64> {
    let mut prev = pollster::block_on(solver.get_field_scalar(field)).expect("read field");
    let mut last_delta = f64::INFINITY;
    for step in 0..max_steps {
        solver.step();
        let cur = pollster::block_on(solver.get_field_scalar(field)).expect("read field");
        let max_delta = cur
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        if max_delta < steady_tol {
            println!("[mms] steady after {} steps (max_delta={max_delta:.3e})", step + 1);
            return cur;
        }
        if step % 10 == 0 {
            println!("[mms] step {step}: max_delta={max_delta:.3e}");
        }
        last_delta = max_delta;
        prev = cur;
    }
    panic!("did not reach steady state within {max_steps} steps (tol {steady_tol:.1e}, last max_delta={last_delta:.3e})");
}
