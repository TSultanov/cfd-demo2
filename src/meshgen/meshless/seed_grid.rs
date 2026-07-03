//! Uniform CSR seed grid + exact k-nearest-neighbor ring search (meshless
//! engine M0.1, design §2).
//!
//! Seeds are binned into a uniform grid (~`TARGET_OCCUPANCY` seeds/bin) laid
//! out CSR-style by a counting sort — O(n), deterministic, and exactly the
//! layout the M1 GPU stage-1 kernel consumes. Queries visit bins in
//! concentric Chebyshev rings keeping the k best candidates in a bounded
//! max-heap ordered by the `(d², id)` *total order* (id tie-break), and stop
//! once the next ring is provably farther than the current k-th distance.
//! The result is the exact k smallest neighbors under that total order —
//! bit-deterministic even for cocircular/equidistant seed sets — which is
//! load-bearing: the clip kernel's oracle-equality guarantee assumes the
//! neighbor list is an exact distance-ordered prefix of all other seeds.

use nalgebra::{Point2, Vector2};
use std::collections::BinaryHeap;

/// Target mean seeds per grid bin (the paper's ~5/cell adapted to 2D).
const TARGET_OCCUPANCY: f64 = 4.0;

/// Squared distance, kept as THE d² formula for the whole meshless module:
/// `knn`, the brute-force oracle, and the clip driver must agree bitwise.
#[inline]
pub(crate) fn dist2(a: Point2<f64>, b: Point2<f64>) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    dx * dx + dy * dy
}

/// Heap candidate ordered by the `(d², id)` total order. `d²` is always
/// finite here, so `total_cmp` coincides with the numeric order.
#[derive(Clone, Copy, Debug)]
struct Cand {
    d2: f64,
    id: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.d2.total_cmp(&other.d2).then(self.id.cmp(&other.id))
    }
}

#[inline]
fn bin_coords(cell_size: f64, gw: usize, gh: usize, p: &Point2<f64>) -> (usize, usize) {
    // Clamp so points exactly on the far domain edges land in the last bin
    // (same policy as the Poisson conflict grid). Negative coordinates
    // saturate to 0 through the `as usize` cast.
    let gx = ((p.x / cell_size) as usize).min(gw - 1);
    let gy = ((p.y / cell_size) as usize).min(gh - 1);
    (gx, gy)
}

/// Uniform grid over the domain bbox with seeds stored CSR-style:
/// `ids[offsets[b]..offsets[b+1]]` are the seeds in bin `b`, ascending by id.
pub struct SeedGrid {
    pub cell_size: f64,
    pub gw: usize,
    pub gh: usize,
    /// CSR bin offsets, `gw*gh + 1` entries.
    pub offsets: Vec<u32>,
    /// Seed ids sorted by `(bin, id)`.
    pub ids: Vec<u32>,
}

impl SeedGrid {
    /// Build by counting sort over `(bin, id)`. Seeds are expected inside
    /// `[0, domain.x] × [0, domain.y]`; out-of-range points are clamped into
    /// edge bins (which would void the ring-search distance bound, so keep
    /// the contract).
    pub fn build(seeds: &[Point2<f64>], domain: Vector2<f64>) -> Self {
        assert!(!seeds.is_empty(), "SeedGrid::build needs at least one seed");
        let n = seeds.len();
        let raw = (domain.x * domain.y * TARGET_OCCUPANCY / n as f64).sqrt();
        let cell_size = if raw.is_finite() && raw > 0.0 { raw } else { 1.0 };
        let gw = ((domain.x / cell_size).ceil() as usize).max(1);
        let gh = ((domain.y / cell_size).ceil() as usize).max(1);
        let nbins = gw * gh;

        let mut offsets = vec![0u32; nbins + 1];
        for p in seeds {
            let (gx, gy) = bin_coords(cell_size, gw, gh, p);
            offsets[gy * gw + gx + 1] += 1;
        }
        for b in 0..nbins {
            offsets[b + 1] += offsets[b];
        }
        // Stable counting sort over ascending seed ids ⇒ ids within each bin
        // come out ascending (the (bin, id) sort order, no comparator).
        let mut cursor: Vec<u32> = offsets[..nbins].to_vec();
        let mut ids = vec![0u32; n];
        for (id, p) in seeds.iter().enumerate() {
            let (gx, gy) = bin_coords(cell_size, gw, gh, p);
            let b = gy * gw + gx;
            ids[cursor[b] as usize] = id as u32;
            cursor[b] += 1;
        }

        Self {
            cell_size,
            gw,
            gh,
            offsets,
            ids,
        }
    }

    #[inline]
    fn bin_of(&self, p: &Point2<f64>) -> (usize, usize) {
        bin_coords(self.cell_size, self.gw, self.gh, p)
    }

    /// Lower bound on the distance from `p` (in bin `(bx, by)`) to any seed
    /// in a bin of Chebyshev ring `r` **or beyond**: those bins lie outside
    /// the box of bins within Chebyshev distance `r − 1`, so the distance
    /// from `p` to that box's boundary bounds them all at once. Grid
    /// truncation only removes bins, so the bound stays conservative.
    fn ring_lower_bound(&self, p: &Point2<f64>, bx: usize, by: usize, r: usize) -> f64 {
        if r == 0 {
            return 0.0;
        }
        let cs = self.cell_size;
        let x_lo = (bx as f64 - (r as f64 - 1.0)) * cs;
        let x_hi = (bx as f64 + r as f64) * cs;
        let y_lo = (by as f64 - (r as f64 - 1.0)) * cs;
        let y_hi = (by as f64 + r as f64) * cs;
        (p.x - x_lo)
            .min(x_hi - p.x)
            .min(p.y - y_lo)
            .min(y_hi - p.y)
            .max(0.0)
    }

    /// Visit every bin at exactly Chebyshev distance `r` from `(bx, by)`,
    /// skipping bins outside the grid. Visit order does not affect results
    /// (the bounded heap keeps the k smallest under a total order).
    fn for_each_ring_bin(&self, bx: usize, by: usize, r: usize, mut f: impl FnMut(usize)) {
        let (gw, gh) = (self.gw as isize, self.gh as isize);
        let (bx, by) = (bx as isize, by as isize);
        let r = r as isize;
        if r == 0 {
            f((by * gw + bx) as usize);
            return;
        }
        for gx in (bx - r)..=(bx + r) {
            if gx < 0 || gx >= gw {
                continue;
            }
            for gy in [by - r, by + r] {
                if gy >= 0 && gy < gh {
                    f((gy * gw + gx) as usize);
                }
            }
        }
        for gy in (by - r + 1)..=(by + r - 1) {
            if gy < 0 || gy >= gh {
                continue;
            }
            for gx in [bx - r, bx + r] {
                if gx >= 0 && gx < gw {
                    f((gy * gw + gx) as usize);
                }
            }
        }
    }

    /// Exact k nearest neighbors of seed `query` under the `(d², id)` total
    /// order, written to `out` ascending. Fewer than `k` entries come back
    /// only when fewer than `k` other seeds exist (that is the exhaustive
    /// case: the caller then knows the list covers every other seed).
    pub fn knn(&self, seeds: &[Point2<f64>], query: u32, k: usize, out: &mut Vec<(f64, u32)>) {
        out.clear();
        if k == 0 {
            return;
        }
        let p = seeds[query as usize];
        let (bx, by) = self.bin_of(&p);
        let r_max = bx
            .max(self.gw - 1 - bx)
            .max(by)
            .max(self.gh - 1 - by);
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        for r in 0..=r_max {
            if heap.len() == k {
                let dmin = self.ring_lower_bound(&p, bx, by, r);
                // Strict '>': a seed at exactly the heap-max distance but
                // with a smaller id would still displace the current max.
                if dmin * dmin > heap.peek().unwrap().d2 {
                    break;
                }
            }
            self.for_each_ring_bin(bx, by, r, |bin| {
                let lo = self.offsets[bin] as usize;
                let hi = self.offsets[bin + 1] as usize;
                for &id in &self.ids[lo..hi] {
                    if id == query {
                        continue;
                    }
                    let cand = Cand {
                        d2: dist2(p, seeds[id as usize]),
                        id,
                    };
                    if heap.len() < k {
                        heap.push(cand);
                    } else if cand < *heap.peek().unwrap() {
                        heap.pop();
                        heap.push(cand);
                    }
                }
            });
        }
        out.extend(heap.into_iter().map(|c| (c.d2, c.id)));
        out.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_layout_covers_all_seeds_in_bin_id_order() {
        let seeds: Vec<Point2<f64>> = (0..25)
            .map(|i| Point2::new(0.1 + 0.19 * (i % 5) as f64, 0.1 + 0.19 * (i / 5) as f64))
            .collect();
        let grid = SeedGrid::build(&seeds, Vector2::new(1.0, 1.0));
        assert_eq!(*grid.offsets.last().unwrap() as usize, seeds.len());
        for b in 0..(grid.gw * grid.gh) {
            let lo = grid.offsets[b] as usize;
            let hi = grid.offsets[b + 1] as usize;
            assert!(grid.ids[lo..hi].windows(2).all(|w| w[0] < w[1]));
        }
        let mut all: Vec<u32> = grid.ids.clone();
        all.sort_unstable();
        assert_eq!(all, (0..25u32).collect::<Vec<_>>());
    }

    #[test]
    fn knn_on_a_line_returns_ordered_neighbors() {
        let seeds: Vec<Point2<f64>> =
            (0..10).map(|i| Point2::new(0.05 + 0.1 * i as f64, 0.5)).collect();
        let grid = SeedGrid::build(&seeds, Vector2::new(1.0, 1.0));
        let mut out = Vec::new();
        grid.knn(&seeds, 0, 3, &mut out);
        let ids: Vec<u32> = out.iter().map(|&(_, id)| id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
        assert!(out.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[test]
    fn knn_with_k_exceeding_set_returns_everyone() {
        let seeds = vec![
            Point2::new(0.2, 0.2),
            Point2::new(0.8, 0.8),
            Point2::new(0.2, 0.8),
        ];
        let grid = SeedGrid::build(&seeds, Vector2::new(1.0, 1.0));
        let mut out = Vec::new();
        grid.knn(&seeds, 1, 64, &mut out);
        assert_eq!(out.len(), 2);
    }
}
