//! Guards the core promise of `TopologyMode::Structured2D`: the generated
//! assembly kernels contain NO mesh-connectivity indirection — neighbours,
//! geometry and the matrix layout are all index arithmetic on the dense
//! Cartesian grid. This is a cheap, fast regression gate over the committed
//! generated WGSL (no GPU/CPU run needed).

use std::path::Path;

/// Identifiers that only appear when a kernel gathers over an unstructured
/// face-connectivity mesh. A structured kernel must reference none of them.
const CONNECTIVITY_TOKENS: &[&str] = &[
    "cell_faces",
    "face_owner",
    "face_neighbor",
    "cell_face_offsets",
    "cell_face_matrix_indices",
    "scalar_row_offsets",
    "face_areas",
    "face_normals",
    "face_centers",
    "cell_centers",
    "cell_vols",
    "diagonal_indices",
];

fn generated_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/solver/gpu/shaders/generated")
}

fn read_shader(name: &str) -> String {
    let path = generated_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read generated shader {}: {e}", path.display()))
}

fn assert_no_connectivity(name: &str) {
    let src = read_shader(name);
    for tok in CONNECTIVITY_TOKENS {
        assert!(
            !src.contains(tok),
            "structured kernel `{name}` still references connectivity token `{tok}` \
             — the operator expansion must be indirection-free"
        );
    }
    // Positive check: it must use the dense-grid uniform + banded arithmetic.
    assert!(
        src.contains("grid.nx") && src.contains("StructuredGrid"),
        "structured kernel `{name}` does not use the dense `grid` uniform"
    );
    assert!(
        src.contains("idx % grid.nx") && src.contains("idx * 5u"),
        "structured kernel `{name}` is missing the arithmetic index / 5-point band"
    );
}

#[test]
fn structured_diffusion_assembly_has_no_connectivity() {
    assert_no_connectivity("generic_coupled_assembly_generic_diffusion_demo_structured.wgsl");
}

#[test]
fn structured_diffusion_ibm_assembly_has_no_connectivity() {
    assert_no_connectivity("generic_coupled_assembly_generic_diffusion_demo_structured_ibm.wgsl");
}

#[test]
fn structured_momentum_coupled_assembly_has_no_connectivity() {
    // The coupled Navier–Stokes assembly (velocity + pressure, block stride 3)
    // is the same indirection-free operator, banded as N*5 blocks of 3x3.
    let name = "generic_coupled_assembly_incompressible_momentum_structured.wgsl";
    assert_no_connectivity(name);
    let src = read_shader(name);
    // Block-CSR SoA row splits prove the coupled (stride-3) banded layout.
    assert!(
        src.contains("scalar_offset * 9u") && src.contains("start_row_2"),
        "structured momentum assembly is missing the coupled 3x3 banded block layout"
    );
}

/// The `Structured2D` family must be strictly additive: the incumbent
/// (unstructured) assembly for the same model is unchanged and still gathers
/// over connectivity (so the two paths are genuinely distinct code).
#[test]
fn unstructured_assembly_still_uses_connectivity() {
    let src = read_shader("generic_coupled_assembly_generic_diffusion_demo.wgsl");
    assert!(
        src.contains("cell_faces") && src.contains("face_owner"),
        "the unstructured path unexpectedly lost its connectivity gather"
    );
}
