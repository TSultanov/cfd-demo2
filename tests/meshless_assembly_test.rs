//! Meshless engine assembly gates:
//!
//! 1. **Equivalence vs the incumbent Voronoi generator on identical seeds**:
//!    the incumbent's exact post-smoothing point set (via the re-exported
//!    `triangulate` seed seam) is fed to BOTH the incumbent's dualization
//!    (`generate_voronoi_mesh`, which re-triangulates the same deterministic
//!    points) and the meshless from-seeds path (pure-bbox `MeshlessInput` +
//!    `assemble_mesh`). Compared INTERIOR-only, matched by generator
//!    coordinate — the incumbent culls dead generators and splits concave
//!    cells, so index comparison is invalid: neighbor sets equal after
//!    dropping faces shorter than the `vertex_merge`-scale eps (1e-6·h, the
//!    incumbent's own quantization — NOT 1e-9), cell area rel diff < 1e-6,
//!    centroid within 1e-6·h. Boundary handling is compared by
//!    per-`BoundaryType` wall LENGTH sums (face counts differ legitimately
//!    between the seeding protocols).
//! 2. Face reciprocity + seed identity on the full loops-based generator
//!    (`generate_meshless_voronoi_mesh`): every interior face is listed by
//!    exactly its owner and neighbor cells (owner < neighbor), every
//!    boundary face exactly once; cell i contains seed i (strictly for
//!    interior seeds, touch-tolerant for boundary seeds).
//! 3. Byte determinism of the whole assembled `Mesh` across explicit rayon
//!    thread pools (1/2/8).
//!
//! Run with:
//!
//! ```sh
//! cargo test --features meshgen --test meshless_assembly_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use std::collections::BTreeSet;

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, BoundarySpec, EngineConfig, MeshlessInput,
    SeedKind,
};
use cfd2::meshgen::{
    generate_meshless_voronoi_mesh, generate_voronoi_mesh, triangulate, BackwardsStep,
    ChannelWithObstacle, Geometry, MeshgenTolerances, Nozzle, RectangularChannel,
};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use nalgebra::{Point2, Vector2};

const H: f64 = 0.05;
const GROWTH: f64 = 1.2;

fn rect() -> (RectangularChannel, Vector2<f64>) {
    (RectangularChannel { length: 3.0, height: 1.0 }, Vector2::new(3.0, 1.0))
}

fn obstacle() -> (ChannelWithObstacle, Vector2<f64>) {
    (
        ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        },
        Vector2::new(3.0, 1.0),
    )
}

fn backstep() -> (BackwardsStep, Vector2<f64>) {
    (
        BackwardsStep {
            length: 3.5,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        },
        Vector2::new(3.5, 1.0),
    )
}

fn nozzle() -> (Nozzle, Vector2<f64>) {
    (
        Nozzle {
            length: 3.0,
            height: 1.0,
            throat_height: 0.40,
            throat_frac: 0.40,
            exit_height: 0.80,
        },
        Vector2::new(3.0, 1.0),
    )
}

/// Does cell `c` of `mesh` touch the boundary (any neighbor-less face)?
fn has_boundary_face(mesh: &Mesh, c: usize) -> bool {
    mesh.cell_faces[mesh.cell_face_offsets[c]..mesh.cell_face_offsets[c + 1]]
        .iter()
        .any(|&f| mesh.face_neighbor[f].is_none())
}

/// Neighbor set of cell `c` as *generator* indices (`cell_to_gen` maps cell
/// index -> generator index; identity for the meshless mesh), dropping faces
/// shorter than `eps_face`.
fn neighbor_gens(mesh: &Mesh, c: usize, cell_to_gen: &[usize], eps_face: f64) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for &f in &mesh.cell_faces[mesh.cell_face_offsets[c]..mesh.cell_face_offsets[c + 1]] {
        if mesh.face_area[f] < eps_face {
            continue;
        }
        if let Some(nb) = mesh.face_neighbor[f] {
            let other = if mesh.face_owner[f] == c { nb } else { mesh.face_owner[f] };
            out.insert(cell_to_gen[other]);
        }
    }
    out
}

/// Per-`BoundaryType` sums of boundary-face lengths.
fn wall_length_census(mesh: &Mesh) -> (f64, f64, f64) {
    let (mut inlet, mut outlet, mut wall) = (0.0, 0.0, 0.0);
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_some() {
            continue;
        }
        match mesh.face_boundary[f] {
            Some(BoundaryType::Inlet) => inlet += mesh.face_area[f],
            Some(BoundaryType::Outlet) => outlet += mesh.face_area[f],
            Some(BoundaryType::Wall) => wall += mesh.face_area[f],
            other => panic!("face {f}: unexpected boundary tag {other:?}"),
        }
    }
    (inlet, outlet, wall)
}

// ---------------------------------------------------------------------------
// 1. Equivalence vs incumbent
// ---------------------------------------------------------------------------

#[test]
fn interior_cells_match_incumbent_on_identical_seeds() {
    fn run(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>) {
        // The seed seam: the incumbent's exact post-smoothing generator set.
        // `generate_voronoi_mesh` re-runs the same deterministic pipeline
        // internally (fixed RNG 0x5EED_CFD2), so these ARE its generators.
        let (points, _tris, fixed) = triangulate(geo, H, H, GROWTH, domain);
        let incumbent = generate_voronoi_mesh(geo, H, H, GROWTH, domain);

        // Meshless from-seeds path: pure-bbox clipping (the incumbent seeds
        // carry no loop metadata), every seed Interior.
        let spec = BoundarySpec::empty();
        let tol = MeshgenTolerances::from_geometry(H, domain);
        let input = MeshlessInput {
            seeds: &points,
            kinds: &[],
            boundary: &spec,
            domain,
            tol: &tol,
            cfg: EngineConfig::default(),
        };
        let diagram = build_diagram(&input);
        let ml = assemble_mesh(&input, &diagram);
        assert_eq!(ml.num_cells(), points.len(), "{name}: seed i != cell i");

        // Match every incumbent cell to its generator: any point of a
        // Voronoi cell (in particular its centroid) is nearest to its own
        // generator; split sub-cells are subsets of the parent cell, so
        // they inherit the parent's generator.
        let nearest = |p: Point2<f64>| -> usize {
            let mut best = (f64::INFINITY, usize::MAX);
            for (g, q) in points.iter().enumerate() {
                let d2 = (p - q).norm_squared();
                if d2 < best.0 {
                    best = (d2, g);
                }
            }
            best.1
        };
        let cell_to_gen: Vec<usize> = (0..incumbent.num_cells())
            .map(|c| nearest(Point2::new(incumbent.cell_cx[c], incumbent.cell_cy[c])))
            .collect();
        let mut gen_cells: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
        for (c, &g) in cell_to_gen.iter().enumerate() {
            gen_cells[g].push(c);
        }

        // Strict-comparison set: non-fixed (interior-seeded) generators with
        // exactly one incumbent cell (not split, not dead) and no boundary
        // faces on either side (the incumbent hull-closes cells whose
        // Delaunay fan lost a culled triangle; those legitimately differ).
        let eps_face = 1e-6 * H;
        let ml_ident: Vec<usize> = (0..ml.num_cells()).collect();
        let n_interior = fixed.iter().filter(|&&f| !f).count();
        let (mut strict, mut excluded_split_or_dead, mut excluded_wall) = (0usize, 0usize, 0usize);
        let mut mismatches = Vec::new();
        for g in 0..points.len() {
            if fixed[g] {
                continue;
            }
            if gen_cells[g].len() != 1 {
                excluded_split_or_dead += 1;
                continue;
            }
            let c = gen_cells[g][0];
            if has_boundary_face(&incumbent, c) || has_boundary_face(&ml, g) {
                excluded_wall += 1;
                continue;
            }
            strict += 1;

            let nb_inc = neighbor_gens(&incumbent, c, &cell_to_gen, eps_face);
            let nb_ml = neighbor_gens(&ml, g, &ml_ident, eps_face);
            if nb_inc != nb_ml {
                mismatches.push(format!(
                    "gen {g}: neighbor sets differ (incumbent {nb_inc:?} vs meshless {nb_ml:?})"
                ));
                continue;
            }
            let area_rel = ((incumbent.cell_vol[c] - ml.cell_vol[g]) / incumbent.cell_vol[c]).abs();
            if area_rel >= 1e-6 {
                mismatches.push(format!("gen {g}: area rel diff {area_rel:.3e}"));
            }
            let cd = ((incumbent.cell_cx[c] - ml.cell_cx[g]).powi(2)
                + (incumbent.cell_cy[c] - ml.cell_cy[g]).powi(2))
            .sqrt();
            if cd >= 1e-6 * H {
                mismatches.push(format!("gen {g}: centroid diff {cd:.3e}"));
            }
        }
        println!(
            "[equiv/{name}] gens={} interior={n_interior} strict-compared={strict} \
             excluded: split/dead={excluded_split_or_dead} wall-touching={excluded_wall} \
             mismatches={}",
            points.len(),
            mismatches.len()
        );
        for m in mismatches.iter().take(10) {
            println!("    - {m}");
        }
        assert!(mismatches.is_empty(), "{name}: {} mismatches", mismatches.len());
        // The comparison must actually cover the bulk of the interior.
        assert!(
            strict * 2 > n_interior,
            "{name}: strict set {strict} covers < 50% of {n_interior} interior generators"
        );
    }

    let (geo, domain) = rect();
    run("rect", &geo, domain);
    let (geo, domain) = obstacle();
    run("obstacle", &geo, domain);
    let (geo, domain) = backstep();
    run("backstep", &geo, domain);
    let (geo, domain) = nozzle();
    run("nozzle", &geo, domain);
}

// ---------------------------------------------------------------------------
// 1b. Boundary comparison by per-type wall length
// ---------------------------------------------------------------------------

#[test]
fn wall_length_census_matches_incumbent_per_boundary_type() {
    fn run(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>, tol_rel: f64) {
        let incumbent = generate_voronoi_mesh(geo, H, H, GROWTH, domain);
        let ml = generate_meshless_voronoi_mesh(geo, H, H, GROWTH, domain);
        let (ii, io, iw) = wall_length_census(&incumbent);
        let (mi, mo, mw) = wall_length_census(&ml);
        println!(
            "[census/{name}] inlet {ii:.9}/{mi:.9} outlet {io:.9}/{mo:.9} wall {iw:.9}/{mw:.9} (incumbent/meshless)"
        );
        for (label, a, b) in [("inlet", ii, mi), ("outlet", io, mo), ("wall", iw, mw)] {
            let rel = (a - b).abs() / a.max(b).max(1e-300);
            assert!(
                rel < tol_rel,
                "{name}/{label}: wall-length sums differ (incumbent {a:.12}, meshless {b:.12}, rel {rel:.3e})"
            );
        }
    }

    // Both pipelines subdivide the box sides, the obstacle circle and the
    // nozzle top wall on the same parameter grid, so the per-type sums must
    // agree to fp roundoff.
    let (geo, domain) = rect();
    run("rect", &geo, domain, 1e-9);
    let (geo, domain) = obstacle();
    run("obstacle", &geo, domain, 1e-9);
    let (geo, domain) = backstep();
    run("backstep", &geo, domain, 1e-9);
    let (geo, domain) = nozzle();
    run("nozzle", &geo, domain, 1e-9);
}

// ---------------------------------------------------------------------------
// 2. Face reciprocity + seed identity
// ---------------------------------------------------------------------------

#[test]
fn faces_are_reciprocal_and_seed_identity_holds() {
    fn run(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>) {
        let mesh = generate_meshless_voronoi_mesh(geo, H, 2.0 * H, GROWTH, domain);
        // Seeds are reproducible (fixed RNG): cell i must be seed i's cell.
        let (seeds, kinds, _spec) = meshless_seed_points(geo, H, 2.0 * H, GROWTH, domain);
        assert_eq!(mesh.num_cells(), seeds.len(), "{name}: cell count != seed count");

        // Face incidence census: interior faces appear in exactly the
        // owner's and neighbor's lists, boundary faces exactly once.
        let mut incidence = vec![0usize; mesh.num_faces()];
        for c in 0..mesh.num_cells() {
            for &f in &mesh.cell_faces[mesh.cell_face_offsets[c]..mesh.cell_face_offsets[c + 1]] {
                incidence[f] += 1;
                assert!(
                    mesh.face_owner[f] == c || mesh.face_neighbor[f] == Some(c),
                    "{name}: cell {c} lists face {f} it is not incident to"
                );
            }
        }
        for f in 0..mesh.num_faces() {
            match mesh.face_neighbor[f] {
                Some(nb) => {
                    assert_ne!(nb, mesh.face_owner[f], "{name}: face {f} owner==neighbor");
                    // Meshless convention: owner is the smaller seed id.
                    assert!(
                        mesh.face_owner[f] < nb,
                        "{name}: face {f} violates owner<neighbor"
                    );
                    assert_eq!(incidence[f], 2, "{name}: interior face {f} not reciprocal");
                    assert!(mesh.face_boundary[f].is_none());
                }
                None => {
                    assert_eq!(incidence[f], 1, "{name}: boundary face {f} incidence");
                    assert!(
                        mesh.face_boundary[f].is_some(),
                        "{name}: boundary face {f} untagged"
                    );
                }
            }
        }

        // Seed identity: seed i inside ring i — strictly for interior seeds,
        // touch-tolerant (the seed lies ON its wall) for boundary seeds.
        for i in 0..mesh.num_cells() {
            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            let ring = &mesh.cell_vertices[start..end];
            let s = seeds[i];
            let mut min_dist = f64::INFINITY;
            for e in 0..ring.len() {
                let a = ring[e];
                let b = ring[(e + 1) % ring.len()];
                let (ex, ey) = (mesh.vx[b] - mesh.vx[a], mesh.vy[b] - mesh.vy[a]);
                let len = (ex * ex + ey * ey).sqrt();
                if len < 1e-14 {
                    continue;
                }
                let cross = ex * (s.y - mesh.vy[a]) - ey * (s.x - mesh.vx[a]);
                min_dist = min_dist.min(cross / len);
            }
            match kinds[i] {
                SeedKind::Interior => assert!(
                    min_dist > 1e-9,
                    "{name} cell {i}: interior seed not strictly inside (min dist {min_dist:.3e})"
                ),
                SeedKind::Boundary { .. } => assert!(
                    min_dist > -1e-9,
                    "{name} cell {i}: boundary seed outside its cell (min dist {min_dist:.3e})"
                ),
            }
        }
        println!(
            "[recip+identity/{name}] cells={} faces={} OK",
            mesh.num_cells(),
            mesh.num_faces()
        );
    }

    let (geo, domain) = rect();
    run("rect", &geo, domain);
    let (geo, domain) = obstacle();
    run("obstacle", &geo, domain);
    let (geo, domain) = backstep();
    run("backstep", &geo, domain);
    let (geo, domain) = nozzle();
    run("nozzle", &geo, domain);
}

// ---------------------------------------------------------------------------
// 3. Byte determinism across thread pools
// ---------------------------------------------------------------------------

fn assert_meshes_bit_identical(a: &Mesh, b: &Mesh, label: &str) {
    let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&a.vx), bits(&b.vx), "{label}: vx");
    assert_eq!(bits(&a.vy), bits(&b.vy), "{label}: vy");
    assert_eq!(a.v_fixed, b.v_fixed, "{label}: v_fixed");
    assert_eq!(a.face_v1, b.face_v1, "{label}: face_v1");
    assert_eq!(a.face_v2, b.face_v2, "{label}: face_v2");
    assert_eq!(a.face_owner, b.face_owner, "{label}: face_owner");
    assert_eq!(a.face_neighbor, b.face_neighbor, "{label}: face_neighbor");
    assert_eq!(a.face_boundary, b.face_boundary, "{label}: face_boundary");
    assert_eq!(bits(&a.face_nx), bits(&b.face_nx), "{label}: face_nx");
    assert_eq!(bits(&a.face_ny), bits(&b.face_ny), "{label}: face_ny");
    assert_eq!(bits(&a.face_area), bits(&b.face_area), "{label}: face_area");
    assert_eq!(bits(&a.face_cx), bits(&b.face_cx), "{label}: face_cx");
    assert_eq!(bits(&a.face_cy), bits(&b.face_cy), "{label}: face_cy");
    assert_eq!(a.face_wrap_shift.len(), 0, "{label}: face_wrap_shift not empty");
    assert_eq!(b.face_wrap_shift.len(), 0, "{label}: face_wrap_shift not empty");
    assert_eq!(bits(&a.cell_cx), bits(&b.cell_cx), "{label}: cell_cx");
    assert_eq!(bits(&a.cell_cy), bits(&b.cell_cy), "{label}: cell_cy");
    assert_eq!(bits(&a.cell_vol), bits(&b.cell_vol), "{label}: cell_vol");
    assert_eq!(a.cell_faces, b.cell_faces, "{label}: cell_faces");
    assert_eq!(a.cell_face_offsets, b.cell_face_offsets, "{label}: cell_face_offsets");
    assert_eq!(a.cell_vertices, b.cell_vertices, "{label}: cell_vertices");
    assert_eq!(
        a.cell_vertex_offsets, b.cell_vertex_offsets,
        "{label}: cell_vertex_offsets"
    );
}

#[test]
fn assembled_mesh_is_byte_identical_across_thread_counts() {
    let (geo, domain) = obstacle();
    let reference = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| generate_meshless_voronoi_mesh(&geo, H, H, GROWTH, domain));
    for threads in [2usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mesh = pool.install(|| generate_meshless_voronoi_mesh(&geo, H, H, GROWTH, domain));
        assert_meshes_bit_identical(&reference, &mesh, &format!("threads={threads}"));
    }
    println!(
        "[determinism/assembled] cells={} faces={} pools 1/2/8 byte-identical",
        reference.num_cells(),
        reference.num_faces()
    );
}
