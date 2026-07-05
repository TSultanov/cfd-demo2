//! GPU exclusive-scan primitive parity.
//!
//! Pins the two-level exclusive scan against a CPU reference across boundary
//! cases that exercise every level: sub-block, exactly one block, block+1
//! (forces the block-sums level), a large multi-block count, and a randomized
//! count. Deterministic (`u32` adds are exact) so the match is bit-exact.
#![cfg(feature = "meshgen")]

use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{GpuScan, ELEMS_PER_BLOCK};
use rand::{Rng, SeedableRng};

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

/// CPU reference exclusive scan + total.
fn cpu_exclusive_scan(data: &[u32]) -> (Vec<u32>, u32) {
    let mut out = Vec::with_capacity(data.len());
    let mut run = 0u32;
    for &v in data {
        out.push(run);
        run += v;
    }
    (out, run)
}

#[test]
fn gpu_scan_matches_cpu_exclusive_scan() {
    let Some(ctx) = gpu_context() else { return };
    let cache = StagingBufferCache::default();
    let epb = ELEMS_PER_BLOCK as usize;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5CA_1AB1E);
    // Deterministic per-size payloads; small values keep sums well under u32.
    let cases: Vec<(String, Vec<u32>)> = vec![
        ("empty".into(), vec![]),
        ("one".into(), vec![7]),
        ("sub_block".into(), (0..300u32).map(|i| (i % 5) + 1).collect()),
        ("exact_block".into(), vec![1u32; epb]),
        ("block_plus_one".into(), vec![2u32; epb + 1]),
        ("two_blocks".into(), (0..2 * epb as u32).map(|i| i % 7).collect()),
        (
            // Many blocks: forces the block-sums level.
            "large_300k".into(),
            (0..300_000u32).map(|i| (i % 13) + 1).collect(),
        ),
        (
            "random_50k".into(),
            (0..50_000).map(|_| rng.gen_range(0..17u32)).collect(),
        ),
    ];

    for (name, data) in &cases {
        let (want, want_total) = cpu_exclusive_scan(data);
        let (got, got_total) = GpuScan::scan_to_vec(&ctx, &cache, data);
        assert_eq!(
            got.len(),
            want.len(),
            "case {name}: length mismatch ({} vs {})",
            got.len(),
            want.len()
        );
        assert_eq!(got_total, want_total, "case {name}: total sum mismatch");
        if got != want {
            let idx = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
            panic!(
                "case {name}: exclusive scan diverges at index {idx}: gpu={} cpu={}",
                got[idx], want[idx]
            );
        }
        println!("[gpu-scan][{name}] n={} total={want_total} — bit-exact", data.len());
    }
}
