//! Timing regression harness for kernel-program generation.
//!
//! The compressible flux-module lowering once took ~5.9 s (1.53M
//! `lower_scalar` calls for a 111 KB WGSL output) because
//! `derive_central_upwind` deep-clones shared subtrees and the lowering
//! materialized every duplicate before `CseBuilder::eliminate` collapsed
//! them. Kernel generation runs at CPU-solver construction (a GUI freeze)
//! and in build.rs for every compressible-family model, so a regression
//! here must fail loudly.
//!
//! Per-phase breakdown:
//!   CFD2_KGEN_PROFILE=1 cargo test --features "cpu meshgen" \
//!       --test zz_transpile_perf_probe -- --nocapture
#![cfg(feature = "cpu")]

use std::time::Instant;

/// Generous bound: post-fix the hash-consed lowering runs in ~0.1 s on an
/// M-series laptop (tests build with the release-inheriting `test`
/// profile); the regression this guards was a ~40x blowout past this.
const BUDGET_SECS: f64 = 1.5;

#[test]
fn time_compressible_kernel_generation() {
    let model = cfd2::solver::model::compressible_model().expect("model");
    let schemes = cfd2::solver::model::backend::SchemeRegistry::new(
        cfd2::solver::scheme::Scheme::Upwind,
    );
    let t = Instant::now();
    let (programs, _wgsl_only) =
        cfd2::solver::cpu::lowering::model_kernel_programs(&model, &schemes).expect("programs");
    let elapsed = t.elapsed().as_secs_f64();
    println!(
        "[kgen] TOTAL model_kernel_programs: {:.2} s ({} programs)",
        elapsed,
        programs.len()
    );
    assert!(
        elapsed < BUDGET_SECS,
        "compressible model_kernel_programs took {elapsed:.2} s (budget {BUDGET_SECS} s): \
         the flux-module lowering stall is back — rerun with CFD2_KGEN_PROFILE=1 \
         and check the hash-consed lowering (FaceCse) in flux_module_wgsl.rs"
    );
}
