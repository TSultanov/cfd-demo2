use cfd2::solver::compiler::emit_incompressible_momentum_wgsl;

fn main() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    match emit_incompressible_momentum_wgsl(base_dir) {
        Ok(path) => {
            println!("Wrote {}", path.display());
        }
        Err(err) => {
            eprintln!("Codegen failed: {}", err);
            std::process::exit(1);
        }
    }
}
