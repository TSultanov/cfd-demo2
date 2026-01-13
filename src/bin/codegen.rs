use cfd2::solver::model::incompressible_momentum_model;
use cfd2_codegen::compiler::write_generated_wgsl;
use cfd2_codegen::solver::codegen::{generate_wgsl, lower_system};
use cfd2_ir::solver::ir::SchemeRegistry;
use cfd2_ir::solver::scheme::Scheme;

fn main() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let model = incompressible_momentum_model();
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    let discrete = lower_system(&model.system, &schemes).unwrap_or_else(|err| {
        eprintln!("Codegen failed: {err}");
        std::process::exit(1);
    });
    let wgsl = generate_wgsl(&discrete);
    let path = write_generated_wgsl(base_dir, "system_main.wgsl", &wgsl).unwrap_or_else(|err| {
        eprintln!("Codegen failed: {err}");
        std::process::exit(1);
    });
    println!("Wrote {}", path.display());
}
