use cfd2::solver::compiler::emit_system_main_wgsl;

fn main() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    match emit_system_main_wgsl(base_dir) {
        Ok(path) => {
            println!("Wrote {}", path.display());
        }
        Err(err) => {
            eprintln!("Codegen failed: {}", err);
            std::process::exit(1);
        }
    }
}
