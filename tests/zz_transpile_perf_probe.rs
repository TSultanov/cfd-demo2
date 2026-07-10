//! PROBE (temporary): time compressible kernel-program generation with
//! CFD2_KGEN_PROFILE instrumentation live in flux_module.rs.
//! Run: CFD2_KGEN_PROFILE=1 cargo test --features "cpu meshgen" --test zz_transpile_perf_probe -- --nocapture
#![cfg(feature = "cpu")]

use std::time::Instant;

#[test]
fn time_compressible_kernel_generation() {
    let model = cfd2::solver::model::compressible_model().expect("model");
    let schemes = cfd2::solver::model::backend::SchemeRegistry::new(
        cfd2::solver::scheme::Scheme::Upwind,
    );
    let t = Instant::now();
    let (programs, _wgsl_only) =
        cfd2::solver::cpu::lowering::model_kernel_programs(&model, &schemes).expect("programs");
    println!(
        "[kgen] TOTAL model_kernel_programs: {:.2} s ({} programs)",
        t.elapsed().as_secs_f64(),
        programs.len()
    );
}
