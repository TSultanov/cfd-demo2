//! Cell-ordering locality probe.
//!
//! Measures how cache-friendly each mesh generator's CELL NUMBERING is for the
//! solver's neighbor gathers (block-CSR spmv `x[col*S]`, Schur `psol[col_cell]`,
//! GPU `state[other_idx*stride]`), and how much a renumbering pass (RCM /
//! Hilbert) would improve it. Also reports FACE-order locality (per-face flux
//! kernels gather `state[owner]`/`state[neighbor]`; per-cell assembly gathers
//! `face_*[cell_faces[k]]`).
//!
//! Run:
//!   cargo run --release --features meshgen --example profile_mesh_ordering
//! Env:
//!   CFD2_ORDER_SIZES  comma list of target cell sizes (default 0.025,0.01,0.005)
//!   CFD2_ORDER_MESHES comma list of cutcell,voronoi,delaunay (default all)

use cfd2::solver::mesh::{
    generate_cut_cell_mesh, generate_delaunay_mesh, generate_structured_nozzle_mesh,
    generate_voronoi_mesh, BoundarySides, BoundaryType, ChannelWithObstacle, Mesh,
};
use nalgebra::{Point2, Vector2};

fn adjacency(mesh: &Mesh) -> Vec<Vec<usize>> {
    let n = mesh.num_cells();
    let mut adj = vec![Vec::new(); n];
    for f in 0..mesh.num_faces() {
        if let Some(nb) = mesh.face_neighbor[f] {
            let o = mesh.face_owner[f];
            adj[o].push(nb);
            adj[nb].push(o);
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    adj
}

/// Reverse Cuthill-McKee: returns new_of_old.
fn rcm_order(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n); // order[k] = old id visited k-th
    let mut queue = std::collections::VecDeque::new();
    // Process every component, seeded at its min-degree node.
    let mut by_degree: Vec<usize> = (0..n).collect();
    by_degree.sort_by_key(|&i| adj[i].len());
    for &seed in &by_degree {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        queue.push_back(seed);
        while let Some(c) = queue.pop_front() {
            order.push(c);
            let mut nbrs: Vec<usize> = adj[c].iter().cloned().filter(|&x| !visited[x]).collect();
            nbrs.sort_by_key(|&x| adj[x].len());
            for x in nbrs {
                visited[x] = true;
                queue.push_back(x);
            }
        }
    }
    order.reverse();
    let mut new_of_old = vec![0usize; n];
    for (new, &old) in order.iter().enumerate() {
        new_of_old[old] = new;
    }
    new_of_old
}

fn hilbert_d(order: u32, mut x: u32, mut y: u32) -> u64 {
    let mut d: u64 = 0;
    let mut s: u32 = 1 << (order - 1);
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x) & (s.wrapping_mul(2).wrapping_sub(1));
                y = s.wrapping_sub(1).wrapping_sub(y) & (s.wrapping_mul(2).wrapping_sub(1));
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// Hilbert-curve order on cell centroids: returns new_of_old.
fn hilbert_order(mesh: &Mesh) -> Vec<usize> {
    let n = mesh.num_cells();
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for i in 0..n {
        x0 = x0.min(mesh.cell_cx[i]);
        x1 = x1.max(mesh.cell_cx[i]);
        y0 = y0.min(mesh.cell_cy[i]);
        y1 = y1.max(mesh.cell_cy[i]);
    }
    const ORDER: u32 = 16;
    let scale = ((1u64 << ORDER) - 1) as f64;
    let sx = if x1 > x0 { scale / (x1 - x0) } else { 0.0 };
    let sy = if y1 > y0 { scale / (y1 - y0) } else { 0.0 };
    let mut keyed: Vec<(u64, usize)> = (0..n)
        .map(|i| {
            let gx = ((mesh.cell_cx[i] - x0) * sx) as u32;
            let gy = ((mesh.cell_cy[i] - y0) * sy) as u32;
            (hilbert_d(ORDER, gx, gy), i)
        })
        .collect();
    keyed.sort_unstable();
    let mut new_of_old = vec![0usize; n];
    for (new, &(_, old)) in keyed.iter().enumerate() {
        new_of_old[old] = new;
    }
    new_of_old
}

/// Deterministic shuffle (control / worst case): returns new_of_old.
fn random_order(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut s: u64 = 0x5EED_CFD2;
    for i in (1..n).rev() {
        // splitmix64
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        v.swap(i, (z % (i as u64 + 1)) as usize);
    }
    v
}

struct Stats {
    mean: f64,
    p50: usize,
    p95: usize,
    max: usize,
    within16: f64,
    within512: f64,
}

/// Neighbor-index distance |new(o) - new(n)| over interior faces under a
/// given renumbering (identity = current ordering).
fn neighbor_stats(mesh: &Mesh, new_of_old: &[usize]) -> Stats {
    let mut d: Vec<usize> = Vec::new();
    for f in 0..mesh.num_faces() {
        if let Some(nb) = mesh.face_neighbor[f] {
            let a = new_of_old[mesh.face_owner[f]];
            let b = new_of_old[nb];
            d.push(a.abs_diff(b));
        }
    }
    d.sort_unstable();
    let n = d.len();
    let mean = d.iter().sum::<usize>() as f64 / n as f64;
    let count_le = |k: usize| d.partition_point(|&x| x <= k) as f64 / n as f64 * 100.0;
    Stats {
        mean,
        p50: d[n / 2],
        p95: d[n * 95 / 100],
        max: d[n - 1],
        within16: count_le(16),
        within512: count_le(512),
    }
}

/// Face-order locality: mean |owner(f) - owner(f-1)| over the face sweep
/// (per-face GPU flux kernels), and mean face-id spread within a cell's face
/// list (per-cell assembly gathering face data).
fn face_stats(mesh: &Mesh, new_of_old: &[usize], face_perm: Option<&[usize]>) -> (f64, f64) {
    let nf = mesh.num_faces();
    // face_perm: new_face_of_old; identity if None.
    let owner_at_new: Vec<usize> = {
        let mut v = vec![0usize; nf];
        for f in 0..nf {
            let nfid = face_perm.map_or(f, |p| p[f]);
            v[nfid] = new_of_old[mesh.face_owner[f]];
        }
        v
    };
    let sweep_jump = owner_at_new.windows(2).map(|w| w[0].abs_diff(w[1])).sum::<usize>() as f64
        / (nf - 1) as f64;
    let mut spread_sum = 0.0;
    let n = mesh.num_cells();
    for c in 0..n {
        let s = mesh.cell_face_offsets[c];
        let e = mesh.cell_face_offsets[c + 1];
        let ids: Vec<usize> =
            (s..e).map(|k| face_perm.map_or(mesh.cell_faces[k], |p| p[mesh.cell_faces[k]])).collect();
        let mn = *ids.iter().min().unwrap();
        let mx = *ids.iter().max().unwrap();
        spread_sum += (mx - mn) as f64;
    }
    (sweep_jump, spread_sum / n as f64)
}

fn report(label: &str, mesh: &Mesh) {
    let n = mesh.num_cells();
    let adj = adjacency(mesh);
    let ident: Vec<usize> = (0..n).collect();
    let orders: Vec<(&str, Vec<usize>)> = vec![
        ("current", ident),
        ("random", random_order(n)),
        ("rcm", rcm_order(&adj)),
        ("hilbert", hilbert_order(mesh)),
    ];
    println!("\n=== {label}: {n} cells, {} faces ===", mesh.num_faces());
    println!(
        "{:<8} {:>10} {:>8} {:>8} {:>9} {:>8} {:>8}",
        "order", "mean|d|", "p50", "p95", "max(bw)", "<=16 %", "<=512 %"
    );
    for (name, perm) in &orders {
        let s = neighbor_stats(mesh, perm);
        let (sweep, spread) = face_stats(mesh, perm, None);
        println!(
            "{:<8} {:>10.1} {:>8} {:>8} {:>9} {:>8.1} {:>8.1}   face-sweep-jump {:>8.1}  cell-face-spread {:>9.1}",
            name, s.mean, s.p50, s.p95, s.max, s.within16, s.within512, sweep, spread
        );
    }
}

fn main() {
    let sizes: Vec<f64> = std::env::var("CFD2_ORDER_SIZES")
        .unwrap_or_else(|_| "0.025,0.01,0.005".into())
        .split(',')
        .map(|s| s.trim().parse().expect("bad size"))
        .collect();
    let meshes: Vec<String> = std::env::var("CFD2_ORDER_MESHES")
        .unwrap_or_else(|_| "cutcell,voronoi,delaunay".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let domain = Vector2::new(3.0, 1.0);

    for &size in &sizes {
        for kind in &meshes {
            let mut mesh = match kind.as_str() {
                "cutcell" => generate_cut_cell_mesh(&geo, size, size, 1.2, domain),
                "voronoi" => generate_voronoi_mesh(&geo, size, size, 1.2, domain),
                "delaunay" => generate_delaunay_mesh(&geo, size, size, 1.2, domain),
                other => panic!("unknown mesh kind {other}"),
            };
            let smooth_iters = if kind == "cutcell" { 100 } else { 50 };
            mesh.smooth(&geo, 0.3, smooth_iters);
            report(&format!("obstacle/{kind} size={size}"), &mesh);
        }
    }

    // Structured nozzle (the allmach bench mesh): expected near-optimal row-major.
    let mesh = generate_structured_nozzle_mesh(
        300,
        100,
        3.0,
        1.0,
        0.40,
        0.40,
        0.80,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    report("nozzle/structured 300x100", &mesh);
}
