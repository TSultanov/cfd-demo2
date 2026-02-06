use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn contract_dtau_is_used_by_compressible_assembly_wgsl() {
    let root = repo_root();
    for rel in [
        "src/solver/gpu/shaders/generated/generic_coupled_assembly_compressible.wgsl",
        "src/solver/gpu/shaders/generated/generic_coupled_assembly_grad_state_compressible.wgsl",
    ] {
        let path = root.join(rel);
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        assert!(
            src.contains("constants.dtau"),
            "{rel}: expected constants.dtau usage in WGSL body"
        );
        assert!(
            src.contains("select(constants.dt, constants.dtau, constants.dtau > 0.0)"),
            "{rel}: expected dtau select() lowering"
        );
    }
}
