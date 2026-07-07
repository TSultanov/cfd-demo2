//! Lloyd/CVT relaxation for the meshless engine.
//!
//! Each iteration rebuilds the diagram from the current seed set (grid +
//! per-cell clipping only — no assembly) and moves every `Interior` seed to
//! the density-weighted centroid of its cell; boundary seeds stay fixed. The
//! density is the graded-CVT weight ρ(x) = h(x)⁻⁴ in 2D (energy-optimal cell
//! capacity ∝ h²·ρ = const), where `h` is the same sizing function the
//! Poisson sampler uses; the weighted centroid is evaluated by fanning the
//! ring from the seed with centroid-point quadrature of ρ per triangle —
//! exact for constant ρ, O(h) otherwise.
//!
//! CVT replaces generator smoothing entirely: `generate_cvt_mesh` runs no
//! `triangulate`/`smooth_generators` and no `Mesh::smooth` afterwards —
//! vertex smoothing would move Voronoi vertices off the bisectors and destroy
//! the engine's defining property.
//!
//! Determinism: per-seed moves are computed in parallel into an index-ordered
//! `Vec` (disjoint outputs, no reductions), and the convergence measure is an
//! f64 `max` — order-insensitive — so the relaxation is byte-identical for any
//! rayon thread count.

use nalgebra::{Point2, Vector2};
use rayon::prelude::*;

use super::super::geometry::Geometry;
use super::super::tolerances::MeshgenTolerances;
use super::boundary::{meshless_seed_points, BoundarySpec, SeedKind};
use super::{
    assemble_mesh, build_diagram, CellStatus, EngineConfig, MeshlessDiagram, MeshlessInput,
    MAX_CLIP_VERTS,
};
use crate::solver::mesh::Mesh;

/// Lloyd relaxation knobs.
#[derive(Clone, Copy, Debug)]
pub struct LloydConfig {
    /// Iteration cap; Lloyd converges linearly, so the cap binds on large
    /// uniform sets long before quality stops improving.
    pub max_iters: usize,
    /// Convergence threshold on `max_i |Δx_i| / h(x_i)` (h-relative
    /// displacement).
    pub tol_disp: f64,
    /// Under-/over-relaxation factor on the centroid move. Plain Lloyd
    /// (ω = 1) is unconditionally stable.
    pub omega: f64,
    /// ρ(x) = h(x)^-exponent. 4 = the 2D energy-CVT grading exponent.
    pub density_exponent: f64,
}

impl Default for LloydConfig {
    fn default() -> Self {
        Self {
            max_iters: 30,
            tol_disp: 0.01,
            omega: 1.0,
            density_exponent: 4.0,
        }
    }
}

/// Relaxation telemetry: iterations actually run, the final h-relative max
/// displacement (`< tol_disp` ⇔ converged), and the escalated-cell count of
/// the last diagram build.
#[derive(Clone, Copy, Debug)]
pub struct LloydStats {
    pub iters: usize,
    pub max_disp: f64,
    pub escalated: usize,
}

/// Ring of cell `i` in absolute coordinates: the padded SoA slot for packed
/// cells, the (rare) spill ring for `RingOverflow` ones. `None` for cells
/// with no geometry (`EmptyCell`/`SecurityRadiusFailed` — broken inputs the
/// caller surfaces at assembly).
fn cell_ring_xy(d: &MeshlessDiagram, i: usize) -> Option<Vec<[f64; 2]>> {
    match d.status[i] {
        CellStatus::EmptyCell | CellStatus::SecurityRadiusFailed => None,
        CellStatus::RingOverflow => {
            let idx = d
                .overflow
                .binary_search_by_key(&(i as u32), |(c, _)| *c)
                .expect("RingOverflow cell must have a spill ring");
            Some(d.overflow[idx].1.iter().map(|(xy, _)| *xy).collect())
        }
        CellStatus::Ok | CellStatus::OkEscalated(_) => {
            let len = d.ring_len[i] as usize;
            Some(d.ring_xy[i * MAX_CLIP_VERTS..i * MAX_CLIP_VERTS + len].to_vec())
        }
    }
}

/// Density-weighted centroid of the convex cell ring, fanned from the seed:
/// per triangle (seed, v_e, v_{e+1}) the weight is `area · ρ(centroid)` with
/// ρ = h⁻ᵉˣᵖ evaluated at the triangle centroid (centroid-point quadrature).
/// Falls back to the seed itself if the total weight degenerates (zero-area
/// ring — cannot happen for a valid cell, but stay total).
fn weighted_centroid(
    seed: Point2<f64>,
    ring: &[[f64; 2]],
    sizing: &(impl Fn(Point2<f64>) -> f64 + Sync),
    exponent: f64,
) -> Point2<f64> {
    let m = ring.len();
    let mut w_sum = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for e in 0..m {
        let a = ring[e];
        let b = ring[(e + 1) % m];
        // Signed fan-triangle area (positive: the ring is CCW and the seed
        // is inside its own convex cell).
        let ax = a[0] - seed.x;
        let ay = a[1] - seed.y;
        let bx = b[0] - seed.x;
        let by = b[1] - seed.y;
        let area = 0.5 * (ax * by - ay * bx);
        let tx = (seed.x + a[0] + b[0]) / 3.0;
        let ty = (seed.y + a[1] + b[1]) / 3.0;
        let w = area * sizing(Point2::new(tx, ty)).powf(-exponent);
        w_sum += w;
        cx += w * tx;
        cy += w * ty;
    }
    if w_sum > 0.0 {
        Point2::new(cx / w_sum, cy / w_sum)
    } else {
        seed
    }
}

/// Lloyd/CVT relaxation: per iteration, rebuild the seed grid + diagram (no
/// assembly) and move every `Interior` seed toward its density-weighted cell
/// centroid; `Boundary` seeds are fixed. Converged when
/// `max_i |Δx_i| / h(x_i) < lcfg.tol_disp`. The centroid of a convex cell is
/// strictly inside it, so seeds stay inside the domain (and, by shielding,
/// inside the fluid) and pairwise distinct — every intermediate seed set is
/// a valid engine input.
#[allow(clippy::too_many_arguments)]
pub fn lloyd_relax(
    seeds: &mut Vec<Point2<f64>>,
    kinds: &[SeedKind],
    boundary: &BoundarySpec,
    sizing: &(impl Fn(Point2<f64>) -> f64 + Sync),
    domain: Vector2<f64>,
    tol: &MeshgenTolerances,
    cfg: &EngineConfig,
    lcfg: &LloydConfig,
) -> LloydStats {
    assert!(
        kinds.is_empty() || kinds.len() == seeds.len(),
        "kinds must be empty or one per seed"
    );
    let mut stats = LloydStats {
        iters: 0,
        max_disp: 0.0,
        escalated: 0,
    };
    for _ in 0..lcfg.max_iters {
        let input = MeshlessInput {
            seeds,
            kinds,
            boundary,
            domain,
            tol,
            cfg: *cfg,
        };
        let d = build_diagram(&input);
        stats.escalated = d.status_counts().1;

        // Deterministic parallel move: an index-ordered collect of pure
        // per-seed results (position, h-relative displacement).
        let moved: Vec<(Point2<f64>, f64)> = (0..seeds.len())
            .into_par_iter()
            .with_min_len(super::meshless_min_len())
            .map(|i| {
                let p = seeds[i];
                let interior = kinds.is_empty() || kinds[i] == SeedKind::Interior;
                if !interior {
                    return (p, 0.0);
                }
                match cell_ring_xy(&d, i) {
                    None => (p, 0.0),
                    Some(ring) => {
                        let c = weighted_centroid(p, &ring, sizing, lcfg.density_exponent);
                        let np = p + (c - p) * lcfg.omega;
                        ((np), (np - p).norm() / sizing(p))
                    }
                }
            })
            .collect();

        // f64 max is order-insensitive; a serial fold over the ordered Vec
        // is byte-equal to any parallel reduction of it.
        let mut max_disp = 0.0f64;
        for (i, &(np, disp)) in moved.iter().enumerate() {
            seeds[i] = np;
            max_disp = max_disp.max(disp);
        }
        stats.iters += 1;
        stats.max_disp = max_disp;
        if max_disp < lcfg.tol_disp {
            break;
        }
    }
    stats
}

/// A CVT mesh together with the authoritative seed set that produced it —
/// the input the moving-mesh driver owns so it can advect the seeds and
/// regenerate the mesh deterministically. `seeds`/`kinds` are the POST-Lloyd
/// relaxed set (seed `i` == cell `i`), `spec` the boundary loops,
/// and `domain`/`min_cell_size` the two scalars `MeshgenTolerances` and the
/// engine config are derived from — everything [`assemble_meshless_from_seeds`]
/// needs to reproduce `mesh` byte-for-byte.
pub struct CvtMeshSeeds {
    pub mesh: Mesh,
    pub seeds: Vec<Point2<f64>>,
    pub kinds: Vec<SeedKind>,
    pub spec: BoundarySpec,
    pub domain: Vector2<f64>,
    pub min_cell_size: f64,
}

/// Regenerate the meshless Voronoi `Mesh` from a fixed seed set: build the
/// diagram (per-cell clip, deterministic) and canonically assemble it — the
/// exact tail of [`generate_cvt_mesh`]. Because `MeshgenTolerances`,
/// `EngineConfig`, `build_diagram` and `assemble_mesh` are all deterministic
/// pure functions of `(seeds, kinds, spec, domain, min_cell_size)`, calling
/// this with the [`CvtMeshSeeds`] the mesh was generated from reproduces that
/// mesh BYTE-FOR-BYTE. That byte-reproducibility is the frozen-seed
/// do-no-harm foundation: a moving-mesh step that has not moved the seeds
/// regenerates the identical mesh, so the swept-quad fluxes are exactly zero.
pub fn assemble_meshless_from_seeds(
    seeds: &[Point2<f64>],
    kinds: &[SeedKind],
    spec: &BoundarySpec,
    domain: Vector2<f64>,
    min_cell_size: f64,
) -> Mesh {
    let tol = MeshgenTolerances::from_geometry(min_cell_size, domain);
    let input = MeshlessInput {
        seeds,
        kinds,
        boundary: spec,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let diagram = build_diagram(&input);
    assemble_mesh(&input, &diagram)
}

/// CVT entry point: loop-derived boundary seeds + Poisson interior fill
/// (Morton-sorted inside `meshless_seed_points`) → Lloyd relaxation →
/// diagram → canonical assembly. The sizing closure is the exact Poisson
/// grading rule (`min + (growth−1)·|sdf|`, capped at `max`), so the CVT
/// density targets the same h-field the sampler seeded. NO `triangulate`,
/// NO `smooth_generators`, NO `Mesh::smooth` afterwards — see the module
/// docs; smoothing an assembled CVT mesh would UNDO the Voronoi property.
pub fn generate_cvt_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
    lloyd: &LloydConfig,
) -> Mesh {
    generate_cvt_mesh_with_seeds(
        geo,
        min_cell_size,
        max_cell_size,
        growth_rate,
        domain_size,
        lloyd,
    )
    .mesh
}

/// [`generate_cvt_mesh`] that also RETURNS the authoritative seed set (and the
/// boundary spec / tolerance scalars) alongside the `Mesh`, so a caller can
/// own the seeds and regenerate via [`assemble_meshless_from_seeds`]. The
/// `mesh` field is produced through that same helper, so it is byte-identical
/// to what `generate_cvt_mesh` returns and to a later regen from `seeds`.
pub fn generate_cvt_mesh_with_seeds(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
    lloyd: &LloydConfig,
) -> CvtMeshSeeds {
    let (mut seeds, kinds, spec) =
        meshless_seed_points(geo, min_cell_size, max_cell_size, growth_rate, domain_size);
    let tol = MeshgenTolerances::from_geometry(min_cell_size, domain_size);
    // Same sizing rule as `generate_poisson_points`' `get_radius`.
    let sizing = |p: Point2<f64>| -> f64 {
        let dist = geo.sdf(&p).abs();
        (min_cell_size + (growth_rate - 1.0).max(0.0) * dist).min(max_cell_size)
    };
    let cfg = EngineConfig::default();
    lloyd_relax(
        &mut seeds, &kinds, &spec, &sizing, domain_size, &tol, &cfg, lloyd,
    );
    // Assemble through the shared seed→mesh helper so the returned `mesh` is
    // byte-identical to a later `assemble_meshless_from_seeds(&seeds, …)`.
    let mesh = assemble_meshless_from_seeds(&seeds, &kinds, &spec, domain_size, min_cell_size);
    CvtMeshSeeds {
        mesh,
        seeds,
        kinds,
        spec,
        domain: domain_size,
        min_cell_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::boundary::polyline_loop;

    /// One Lloyd step on a single interior seed in an empty box moves it to
    /// the box centroid (uniform density ⇒ plain centroid), and the next
    /// step is a fixed point.
    #[test]
    fn single_seed_moves_to_box_centroid() {
        let domain = Vector2::new(2.0, 1.0);
        let tol = MeshgenTolerances::from_geometry(0.1, domain);
        let spec = BoundarySpec::empty();
        let mut seeds = vec![Point2::new(0.3, 0.8)];
        let kinds = [SeedKind::Interior];
        let sizing = |_: Point2<f64>| 0.1;
        let cfg = EngineConfig::default();
        let lcfg = LloydConfig {
            tol_disp: 1e-12,
            ..LloydConfig::default()
        };
        let stats = lloyd_relax(
            &mut seeds, &kinds, &spec, &sizing, domain, &tol, &cfg, &lcfg,
        );
        assert!((seeds[0] - Point2::new(1.0, 0.5)).norm() < 1e-12);
        assert!(stats.iters <= 3, "fixed point after the first move");
        assert!(stats.max_disp < lcfg.tol_disp);
    }

    /// Boundary-kind seeds never move, whatever the diagram does.
    #[test]
    fn boundary_seeds_stay_fixed() {
        let domain = Vector2::new(1.0, 1.0);
        let tol = MeshgenTolerances::from_geometry(0.25, domain);
        let corners = [
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let spec =
            BoundarySpec::from_loops(vec![polyline_loop(&corners, 0.25, domain, &tol)]);
        let (mut seeds, mut kinds) = super::super::boundary::boundary_seeds(&spec, &tol);
        let before = seeds.clone();
        // One off-center interior seed so the diagram is non-trivial.
        seeds.push(Point2::new(0.4, 0.6));
        kinds.push(SeedKind::Interior);
        let sizing = |_: Point2<f64>| 0.25;
        lloyd_relax(
            &mut seeds,
            &kinds,
            &spec,
            &sizing,
            domain,
            &tol,
            &EngineConfig::default(),
            &LloydConfig::default(),
        );
        for (i, p) in before.iter().enumerate() {
            assert_eq!(
                (seeds[i].x.to_bits(), seeds[i].y.to_bits()),
                (p.x.to_bits(), p.y.to_bits()),
                "boundary seed {i} moved"
            );
        }
    }
}
