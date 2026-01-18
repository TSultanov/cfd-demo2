use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_utf8(path: impl AsRef<Path>) -> String {
    fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read {}: {err}",
            path.as_ref().display()
        )
    })
}

fn assert_not_contains(haystack: &str, needle: &str, context: &str) {
    assert!(
        !haystack.contains(needle),
        "{context}: must not contain '{needle}'"
    );
}

fn assert_contains(haystack: &str, needle: &str, context: &str) {
    assert!(haystack.contains(needle), "{context}: missing '{needle}'");
}

#[test]
fn contract_kernel_registry_has_no_special_case_kernel_matches() {
    let path = repo_root().join("src/solver/gpu/lowering/kernel_registry.rs");
    let src = read_utf8(&path);

    // Kernel registry lookups must go through the generated `(model_id, kernel_id)` mapping.
    assert_contains(
        &src,
        "generated::kernel_entry_by_id",
        "kernel_registry.rs",
    );

    // No KernelId-specific switches in the runtime registry lookup.
    assert_not_contains(&src, "match kernel_id", "kernel_registry.rs");
    assert_not_contains(&src, "KernelId::", "kernel_registry.rs");
    assert_not_contains(&src, "include_str!", "kernel_registry.rs");
}

#[test]
fn contract_generated_kernels_module_has_no_kernel_id_switches() {
    let path = repo_root().join("src/solver/gpu/modules/generated_kernels.rs");
    let src = read_utf8(&path);

    // Host bind groups/pipelines must be metadata-driven and not keyed off KernelId matches.
    assert_not_contains(&src, "match kernel_id", "generated_kernels.rs");
    assert_not_contains(&src, "KernelId::", "generated_kernels.rs");
}

#[test]
fn contract_model_driven_lowering_does_not_branch_on_model_identity() {
    let path = repo_root().join("src/solver/gpu/lowering/model_driven.rs");
    let src = read_utf8(&path);

    // Lowering selects runtime shape based on the derived recipe, not on model.id/method/eos.
    for needle in [
        "match model.id",
        "match model.method",
        "match model.eos",
        "if model.id",
        "if model.method",
        "if model.eos",
        "model.id ==",
        "model.method ==",
        "model.eos ==",
    ] {
        assert_not_contains(&src, needle, "model_driven.rs");
    }
}

#[test]
fn contract_build_rs_has_no_per_kernel_prefix_discovery_glue() {
    let path = repo_root().join("build.rs");
    let src = read_utf8(&path);

    // build.rs should be driven by `derive_kernel_specs_for_model` and not accrete prefix-based
    // discovery logic for each new numerical module.
    for needle in [
        "flux_kt",
        "flux_rhie_chow",
        "rhie_chow_correct_velocity",
        "dp_update_from_diag",
    ] {
        assert_not_contains(&src, needle, "build.rs");
    }
}

#[test]
fn contract_unified_solver_has_no_model_specific_helpers() {
    let path = repo_root().join("src/solver/gpu/unified_solver.rs");
    let src = read_utf8(&path);

    // Core solver API should not encode PDE/physics details (those belong in model helper layers).
    for needle in [
        "get_u(",
        "get_p(",
        "get_rho(",
        "set_uniform_state",
        "set_state_fields",
        "gamma=",
        "inlet_velocity",
        "ramp_time",
        "outer_correctors",
    ] {
        assert_not_contains(&src, needle, "unified_solver.rs");
    }
}
