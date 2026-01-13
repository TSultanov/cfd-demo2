use cfd2::solver::model;
use cfd2::solver::scheme::Scheme;

fn main() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let schemes = model::backend::SchemeRegistry::new(Scheme::Upwind);

    for spec in model::all_models() {
        let paths = model::kernel::emit_model_kernels_wgsl(base_dir, &spec, &schemes)
            .unwrap_or_else(|err| {
                eprintln!("Codegen failed for model '{}': {err}", spec.id);
                std::process::exit(1);
            });
        for path in paths {
            println!("Wrote {}", path.display());
        }
    }
}
