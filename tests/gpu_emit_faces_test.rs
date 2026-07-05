//! GPU face-major EMIT parity ([`cfd2::solver::gpu::voronoi::EmitFaces`]).
//!
//! The emit pass turns the M1 engine's cell-major clip slots + the count→scan
//! per-cell owned-face offsets into the face-major SOLVER geometry
//! (`face_owner`/`face_neighbor`/`face_area`/`face_normal`/`face_center`/
//! `face_bc`) entirely on device. Two gates:
//!
//! 1. **Emit logic (bit-exact):** the GPU emit reproduces a CPU reorganization
//!    of the SAME engine cell-major output — same `owner = min(i,j)` dedup, same
//!    exclusive-scan addressing, same `seed + face_mid` arithmetic. This isolates
//!    the emit kernel from any f32/vertex-merge confound: it must be bitwise.
//! 2. **Solver-mesh topology:** the emit's interior `(owner, neighbour)` face set
//!    equals the CPU `assemble_mesh` solver mesh's — the emit produces the same
//!    face adjacency the GPU solver would otherwise get via the CPU round-trip.
//!
//! ```sh
//! cargo test --features meshgen --test gpu_emit_faces_test -- --nocapture
//! ```
#![cfg(feature = "meshgen")]

use std::collections::BTreeSet;

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, EngineConfig, MeshlessInput, SeedKind,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{
    boundary_spec_f32, EmitFaces, GpuVoronoiCells, GpuVoronoiEngine, K_FACE_MAX, NBR_NONE,
};
use cfd2::solver::mesh::{Geometry, Mesh, RectangularChannel};
use nalgebra::{Point2, Vector2};

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

/// CPU reorganization of the engine's cell-major output into the face-major
/// arrays the emit kernel is supposed to produce — the bit-exact reference.
struct ExpectedFaces {
    owner: Vec<u32>,
    neighbor: Vec<i32>,
    area: Vec<f32>,
    normal: Vec<[f32; 2]>,
    center: Vec<[f32; 2]>,
    bc: Vec<u32>,
}

fn expected_from_cells(gpu: &GpuVoronoiCells, seeds_f32: &[f32]) -> ExpectedFaces {
    let n = gpu.n;
    // Per-cell owned-face count (owner = min(i,j); boundary owned by its cell) —
    // only for SUCCESS cells (status 0), matching the count kernel.
    let mut owned = vec![0u32; n];
    for i in 0..n {
        if gpu.status[i] != 0 {
            continue;
        }
        let nf = gpu.nfaces[i] as usize;
        let mut c = 0u32;
        for e in 0..nf.min(K_FACE_MAX) {
            let nbr = gpu.nbr_ids[i * K_FACE_MAX + e];
            if nbr == NBR_NONE || (i as u32) < nbr {
                c += 1;
            }
        }
        owned[i] = c;
    }
    // Exclusive scan → per-cell base offset + total.
    let mut offsets = vec![0u32; n];
    let mut acc = 0u32;
    for i in 0..n {
        offsets[i] = acc;
        acc += owned[i];
    }
    let total = acc as usize;
    let mut ex = ExpectedFaces {
        owner: vec![0; total],
        neighbor: vec![0; total],
        area: vec![0.0; total],
        normal: vec![[0.0; 2]; total],
        center: vec![[0.0; 2]; total],
        bc: vec![0; total],
    };
    for i in 0..n {
        if gpu.status[i] != 0 {
            continue;
        }
        let nf = gpu.nfaces[i] as usize;
        let seed = [seeds_f32[2 * i], seeds_f32[2 * i + 1]];
        let mut rank = offsets[i] as usize;
        for e in 0..nf.min(K_FACE_MAX) {
            let slot = i * K_FACE_MAX + e;
            let nbr = gpu.nbr_ids[slot];
            let is_owned = nbr == NBR_NONE || (i as u32) < nbr;
            if !is_owned {
                continue;
            }
            let g = rank;
            rank += 1;
            ex.owner[g] = i as u32;
            ex.neighbor[g] = if nbr == NBR_NONE { -1 } else { nbr as i32 };
            let geom = gpu.face_geom[slot]; // [nx, ny, len, 0]
            ex.area[g] = geom[2];
            ex.normal[g] = [geom[0], geom[1]];
            let mid = gpu.face_mid[slot];
            ex.center[g] = [seed[0] + mid[0], seed[1] + mid[1]];
            ex.bc[g] = gpu.face_bc[slot];
        }
    }
    ex
}

/// Interior `(min,max)` seed-id pairs of a solver mesh (owner/neighbour), for
/// the topology comparison.
fn mesh_interior_pairs(mesh: &Mesh) -> BTreeSet<(usize, usize)> {
    let mut s = BTreeSet::new();
    for f in 0..mesh.num_faces() {
        if let Some(nb) = mesh.face_neighbor[f] {
            let o = mesh.face_owner[f];
            s.insert((o.min(nb), o.max(nb)));
        }
    }
    s
}

fn emit_interior_pairs(faces: &cfd2::solver::gpu::voronoi::GpuFaceGeometry) -> BTreeSet<(usize, usize)> {
    let mut s = BTreeSet::new();
    for f in 0..faces.num_faces as usize {
        let nb = faces.face_neighbor[f];
        if nb >= 0 {
            let o = faces.face_owner[f] as usize;
            let nb = nb as usize;
            s.insert((o.min(nb), o.max(nb)));
        }
    }
    s
}

fn run_case(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>, hmin: f64) {
    let Some(ctx) = gpu_context() else { return };
    let (seeds, kinds, spec) = meshless_seed_points(geo, hmin, hmin, 1.2, domain);
    let n = seeds.len();
    let n_boundary = kinds.iter().filter(|k| matches!(k, SeedKind::Boundary { .. })).count();
    let tol = MeshgenTolerances::from_geometry(hmin, domain);

    let seeds_f32: Vec<f32> = seeds.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
    let rounded: Vec<Point2<f64>> =
        (0..n).map(|i| Point2::new(seeds_f32[2 * i] as f64, seeds_f32[2 * i + 1] as f64)).collect();
    let spec_r = boundary_spec_f32(&spec);
    let flags = vec![0u32; n];

    let input = MeshlessInput {
        seeds: &rounded,
        kinds: &kinds,
        boundary: &spec_r,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let cpu = build_diagram(&input);

    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_case(&ctx.device, &ctx.queue, &seeds_f32, &flags, &kinds, &spec);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
    let cache = StagingBufferCache::default();
    // Resolve any f32-flagged cells so the engine output is reciprocal + matches
    // the CPU oracle (the topology gate needs it; the emit-logic gate does not).
    let report = engine.resolve_flagged(&ctx, &cache);
    let gpu = engine.read_cells(&ctx, &cache);

    // ---- Gate 1: emit logic is bit-exact vs the CPU reorganization. ----------
    let emitter = EmitFaces::new(&ctx.device);
    let faces = emitter.emit(&ctx, &cache, &engine);
    let ex = expected_from_cells(&gpu, &seeds_f32);
    assert_eq!(
        faces.num_faces as usize,
        ex.owner.len(),
        "[{name}] emit num_faces {} != CPU-reorganized {}",
        faces.num_faces,
        ex.owner.len()
    );
    let nf = faces.num_faces as usize;
    for f in 0..nf {
        assert_eq!(faces.face_owner[f], ex.owner[f], "[{name}] face {f} owner");
        assert_eq!(faces.face_neighbor[f], ex.neighbor[f], "[{name}] face {f} neighbor");
        assert_eq!(faces.face_bc[f], ex.bc[f], "[{name}] face {f} bc");
        assert_eq!(faces.face_area[f].to_bits(), ex.area[f].to_bits(), "[{name}] face {f} area");
        for c in 0..2 {
            assert_eq!(
                faces.face_normal[f][c].to_bits(),
                ex.normal[f][c].to_bits(),
                "[{name}] face {f} normal[{c}]"
            );
            assert_eq!(
                faces.face_center[f][c].to_bits(),
                ex.center[f][c].to_bits(),
                "[{name}] face {f} center[{c}]"
            );
        }
    }

    // ---- Gate 2: interior face set == CPU assemble_mesh solver mesh. ----------
    let cpu_mesh = assemble_mesh(&input, &cpu);
    let cpu_pairs = mesh_interior_pairs(&cpu_mesh);
    let gpu_pairs = emit_interior_pairs(&faces);
    let only_cpu: Vec<_> = cpu_pairs.difference(&gpu_pairs).take(8).collect();
    let only_gpu: Vec<_> = gpu_pairs.difference(&cpu_pairs).take(8).collect();
    assert!(
        cpu_pairs == gpu_pairs,
        "[{name}] interior face-set mismatch: {} only-CPU (e.g. {:?}), {} only-GPU (e.g. {:?})",
        cpu_pairs.difference(&gpu_pairs).count(),
        only_cpu,
        gpu_pairs.difference(&cpu_pairs).count(),
        only_gpu,
    );

    println!(
        "[gpu-emit] {name}: n={n} ({n_boundary} boundary), num_faces={} \
         (interior pairs {}), resolve patched={}, bit-exact emit + face-set parity OK",
        faces.num_faces,
        gpu_pairs.len(),
        report.patched.len(),
    );
}

#[test]
fn emit_faces_parity_rect_channel_coarse() {
    run_case("rect_coarse", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.12);
}

#[test]
fn emit_faces_parity_rect_channel_fine() {
    run_case("rect_fine", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.06);
}
