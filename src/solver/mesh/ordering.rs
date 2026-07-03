//! Cell/face renumbering for cache locality.
//!
//! The solvers gather neighbor data by cell id everywhere (block-CSR spmv
//! `x[col*S]`, Schur `psol[col_cell]`, GPU `state[other_idx*stride]`), so the
//! distance between a cell's index and its neighbors' indices decides how
//! local those gathers are. Generators already produce reasonable orders
//! (structured/cut-cell: row-major; Delaunay/Voronoi: Morton-sorted points),
//! but the unstructured ones carry a fat tail of far-apart neighbor pairs.
//!
//! `reorder_cells` applies a permutation to every cell-indexed array and
//! remaps faces; it changes nothing topological — solvers rebuild CSR/buffers
//! from the mesh, so a renumbered mesh is transparently consumed. Face order
//! is normalized to first-use-by-owner so per-face arrays stay coherent with
//! the cell sweep.

use super::structs::Mesh;

impl Mesh {
    /// Renumber cells: `new_of_old[old_cell] = new_cell`. Also reorders faces
    /// to first-use order under the new cell numbering (keeps the per-cell
    /// face lists and the face sweep coherent with the cell sweep).
    ///
    /// `new_of_old` must be a permutation of `0..num_cells`.
    pub fn reorder_cells(&mut self, new_of_old: &[usize]) {
        let n = self.num_cells();
        assert_eq!(new_of_old.len(), n, "permutation length != num_cells");
        let mut old_of_new = vec![usize::MAX; n];
        for (old, &new) in new_of_old.iter().enumerate() {
            assert!(new < n && old_of_new[new] == usize::MAX, "not a permutation");
            old_of_new[new] = old;
        }

        // Per-cell scalars.
        let permute_f64 = |a: &[f64]| -> Vec<f64> { old_of_new.iter().map(|&o| a[o]).collect() };
        self.cell_cx = permute_f64(&self.cell_cx);
        self.cell_cy = permute_f64(&self.cell_cy);
        self.cell_vol = permute_f64(&self.cell_vol);

        // Per-cell CSR lists (faces, vertices) rebuilt in new cell order.
        let rebuild_csr = |offsets: &[usize], items: &[usize]| -> (Vec<usize>, Vec<usize>) {
            let mut new_offsets = Vec::with_capacity(n + 1);
            let mut new_items = Vec::with_capacity(items.len());
            new_offsets.push(0);
            for &o in &old_of_new {
                new_items.extend_from_slice(&items[offsets[o]..offsets[o + 1]]);
                new_offsets.push(new_items.len());
            }
            (new_offsets, new_items)
        };
        let (cfo, cf) = rebuild_csr(&self.cell_face_offsets, &self.cell_faces);
        self.cell_face_offsets = cfo;
        self.cell_faces = cf;
        let (cvo, cv) = rebuild_csr(&self.cell_vertex_offsets, &self.cell_vertices);
        self.cell_vertex_offsets = cvo;
        self.cell_vertices = cv;

        // Remap face -> cell references.
        for o in self.face_owner.iter_mut() {
            *o = new_of_old[*o];
        }
        for nb in self.face_neighbor.iter_mut() {
            if let Some(c) = nb {
                *c = new_of_old[*c];
            }
        }

        // Reorder faces to first-use order in the new cell sweep, so face
        // arrays are read near-sequentially by both the per-cell assembly
        // (via cell_faces) and the per-face kernels (owner ids ascend).
        let nf = self.num_faces();
        let mut new_of_old_face = vec![usize::MAX; nf];
        let mut next = 0usize;
        for &f in &self.cell_faces {
            if new_of_old_face[f] == usize::MAX {
                new_of_old_face[f] = next;
                next += 1;
            }
        }
        // Faces never referenced by any cell (shouldn't exist) keep tail slots.
        for slot in new_of_old_face.iter_mut() {
            if *slot == usize::MAX {
                *slot = next;
                next += 1;
            }
        }
        debug_assert_eq!(next, nf);
        let mut old_of_new_face = vec![0usize; nf];
        for (old, &new) in new_of_old_face.iter().enumerate() {
            old_of_new_face[new] = old;
        }
        fn permute<T: Clone>(a: &[T], old_of_new: &[usize]) -> Vec<T> {
            old_of_new.iter().map(|&o| a[o].clone()).collect()
        }
        self.face_v1 = permute(&self.face_v1, &old_of_new_face);
        self.face_v2 = permute(&self.face_v2, &old_of_new_face);
        self.face_owner = permute(&self.face_owner, &old_of_new_face);
        self.face_neighbor = permute(&self.face_neighbor, &old_of_new_face);
        self.face_boundary = permute(&self.face_boundary, &old_of_new_face);
        self.face_nx = permute(&self.face_nx, &old_of_new_face);
        self.face_ny = permute(&self.face_ny, &old_of_new_face);
        self.face_area = permute(&self.face_area, &old_of_new_face);
        self.face_cx = permute(&self.face_cx, &old_of_new_face);
        self.face_cy = permute(&self.face_cy, &old_of_new_face);
        if !self.face_wrap_shift.is_empty() {
            self.face_wrap_shift = permute(&self.face_wrap_shift, &old_of_new_face);
        }
        for f in self.cell_faces.iter_mut() {
            *f = new_of_old_face[*f];
        }
    }

    /// Reverse Cuthill-McKee cell order (minimizes index bandwidth between
    /// neighbors). Returns `new_of_old`.
    pub fn rcm_cell_order(&self) -> Vec<usize> {
        let n = self.num_cells();
        let mut adj = vec![Vec::new(); n];
        for f in 0..self.num_faces() {
            if let Some(nb) = self.face_neighbor[f] {
                let o = self.face_owner[f];
                adj[o].push(nb);
                adj[nb].push(o);
            }
        }
        for a in adj.iter_mut() {
            a.sort_unstable();
            a.dedup();
        }
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut queue = std::collections::VecDeque::new();
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
                let mut nbrs: Vec<usize> =
                    adj[c].iter().cloned().filter(|&x| !visited[x]).collect();
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

    /// Hilbert space-filling-curve cell order on centroids (keeps 2D-nearby
    /// cells nearby in index space in both directions). Returns `new_of_old`.
    pub fn hilbert_cell_order(&self) -> Vec<usize> {
        fn hilbert_d(order: u32, mut x: u32, mut y: u32) -> u64 {
            let mut d: u64 = 0;
            let mut s: u32 = 1 << (order - 1);
            while s > 0 {
                let rx = u32::from((x & s) > 0);
                let ry = u32::from((y & s) > 0);
                d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
                if ry == 0 {
                    if rx == 1 {
                        let mask = s.wrapping_mul(2).wrapping_sub(1);
                        x = s.wrapping_sub(1).wrapping_sub(x) & mask;
                        y = s.wrapping_sub(1).wrapping_sub(y) & mask;
                    }
                    std::mem::swap(&mut x, &mut y);
                }
                s /= 2;
            }
            d
        }
        let n = self.num_cells();
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for i in 0..n {
            x0 = x0.min(self.cell_cx[i]);
            x1 = x1.max(self.cell_cx[i]);
            y0 = y0.min(self.cell_cy[i]);
            y1 = y1.max(self.cell_cy[i]);
        }
        const ORDER: u32 = 16;
        let scale = ((1u64 << ORDER) - 1) as f64;
        let sx = if x1 > x0 { scale / (x1 - x0) } else { 0.0 };
        let sy = if y1 > y0 { scale / (y1 - y0) } else { 0.0 };
        let mut keyed: Vec<(u64, usize)> = (0..n)
            .map(|i| {
                let gx = ((self.cell_cx[i] - x0) * sx) as u32;
                let gy = ((self.cell_cy[i] - y0) * sy) as u32;
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

    /// Deterministic random cell order (locality worst-case control for
    /// benchmarks). Returns `new_of_old`.
    pub fn random_cell_order(&self) -> Vec<usize> {
        let n = self.num_cells();
        let mut v: Vec<usize> = (0..n).collect();
        let mut s: u64 = 0x5EED_CFD2;
        for i in (1..n).rev() {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            v.swap(i, (z % (i as u64 + 1)) as usize);
        }
        v
    }

    /// Env-driven reorder hook for benchmarks: `CFD2_MESH_ORDER=rcm|hilbert|random`.
    /// Unset or `none` leaves the generator order untouched.
    pub fn apply_env_cell_order(&mut self) {
        let Ok(kind) = std::env::var("CFD2_MESH_ORDER") else {
            return;
        };
        let perm = match kind.as_str() {
            "" | "none" => return,
            "rcm" => self.rcm_cell_order(),
            "hilbert" => self.hilbert_cell_order(),
            "random" => self.random_cell_order(),
            other => panic!("unknown CFD2_MESH_ORDER: {other}"),
        };
        self.reorder_cells(&perm);
    }
}

#[cfg(test)]
mod tests {
    use super::super::structured::{generate_structured_rect_mesh, BoundarySides};
    use super::super::structs::BoundaryType;

    fn sides() -> BoundarySides {
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        }
    }

    /// Reordering must preserve the mesh as a set of cells: volumes/centroids
    /// travel with the cell, faces keep the same owner/neighbor cells (by
    /// centroid identity), per-cell closure holds.
    #[test]
    fn reorder_preserves_topology_and_geometry() {
        let base = generate_structured_rect_mesh(20, 13, 2.0, 1.0, sides());
        for order in ["rcm", "hilbert", "random"] {
            let mut m = base.clone();
            let perm = match order {
                "rcm" => m.rcm_cell_order(),
                "hilbert" => m.hilbert_cell_order(),
                _ => m.random_cell_order(),
            };
            m.reorder_cells(&perm);
            assert_eq!(m.num_cells(), base.num_cells());
            assert_eq!(m.num_faces(), base.num_faces());

            // Cell data follows the permutation.
            for old in 0..base.num_cells() {
                let new = perm[old];
                assert_eq!(m.cell_cx[new], base.cell_cx[old], "{order}: cx moved");
                assert_eq!(m.cell_cy[new], base.cell_cy[old], "{order}: cy moved");
                assert_eq!(m.cell_vol[new], base.cell_vol[old], "{order}: vol moved");
            }

            // Face set is identical up to face permutation: match faces by
            // center and compare owner/neighbor centroids.
            let key = |cx: f64, cy: f64| ((cx * 1e9) as i64, (cy * 1e9) as i64);
            let mut base_faces = std::collections::HashMap::new();
            for f in 0..base.num_faces() {
                let o = base.face_owner[f];
                let nb = base.face_neighbor[f].map(|c| key(base.cell_cx[c], base.cell_cy[c]));
                base_faces.insert(
                    key(base.face_cx[f], base.face_cy[f]),
                    (key(base.cell_cx[o], base.cell_cy[o]), nb, base.face_boundary[f]),
                );
            }
            for f in 0..m.num_faces() {
                let o = m.face_owner[f];
                let nb = m.face_neighbor[f].map(|c| key(m.cell_cx[c], m.cell_cy[c]));
                let got = (key(m.cell_cx[o], m.cell_cy[o]), nb, m.face_boundary[f]);
                let want = base_faces[&key(m.face_cx[f], m.face_cy[f])];
                assert_eq!(got, want, "{order}: face owner/neighbor changed");
            }

            // Per-cell closure: faces listed for a cell touch that cell.
            for c in 0..m.num_cells() {
                for k in m.cell_face_offsets[c]..m.cell_face_offsets[c + 1] {
                    let f = m.cell_faces[k];
                    assert!(
                        m.face_owner[f] == c || m.face_neighbor[f] == Some(c),
                        "{order}: cell_faces lists a face not touching the cell"
                    );
                }
            }
        }
    }
}
