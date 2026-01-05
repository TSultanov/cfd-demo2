use glob::glob;
use wgsl_bindgen::{WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

#[allow(dead_code)]
mod solver {
    pub mod scheme {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/solver/scheme.rs"));
    }
    pub mod codegen {
        pub mod ast {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/ast.rs"
            ));
        }
        pub mod coupled_assembly {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/coupled_assembly.rs"
            ));
        }
        pub mod prepare_coupled {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/prepare_coupled.rs"
            ));
        }
        pub mod pressure_assembly {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/pressure_assembly.rs"
            ));
        }
        pub mod update_fields_from_coupled {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/update_fields_from_coupled.rs"
            ));
        }
        pub mod ir {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/ir.rs"
            ));
        }
        pub mod state_access {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/state_access.rs"
            ));
        }
        pub mod state_layout {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/state_layout.rs"
            ));
        }
        pub mod wgsl {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/wgsl.rs"
            ));
        }
        pub mod wgsl_ast {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/wgsl_ast.rs"
            ));
        }
        pub mod wgsl_dsl {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/wgsl_dsl.rs"
            ));
        }
        pub mod scheme {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/scheme.rs"
            ));
        }
        pub mod model {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/model.rs"
            ));
        }
        pub mod emit {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/emit.rs"
            ));
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/solver/scheme.rs");
    for entry in glob("src/solver/codegen/*.rs").expect("Failed to read codegen glob") {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }
    println!("cargo:rerun-if-changed=src/solver/gpu/shaders");

    let mut builder = WgslBindgenOptionBuilder::default();
    builder
        .workspace_root("src/solver/gpu/shaders")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .derive_serde(false)
        .output("src/solver/gpu/bindings.rs");

    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    if let Err(err) = solver::codegen::emit::emit_incompressible_momentum_wgsl(&manifest_dir) {
        panic!("codegen failed: {}", err);
    }
    if let Err(err) = solver::codegen::emit::emit_coupled_assembly_codegen_wgsl(&manifest_dir) {
        panic!("codegen failed: {}", err);
    }
    if let Err(err) = solver::codegen::emit::emit_prepare_coupled_codegen_wgsl(&manifest_dir) {
        panic!("codegen failed: {}", err);
    }
    if let Err(err) = solver::codegen::emit::emit_pressure_assembly_codegen_wgsl(&manifest_dir) {
        panic!("codegen failed: {}", err);
    }
    if let Err(err) = solver::codegen::emit::emit_update_fields_from_coupled_codegen_wgsl(&manifest_dir) {
        panic!("codegen failed: {}", err);
    }

    for entry in glob("src/solver/gpu/shaders/**/*.wgsl").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                let path_str = path.to_str().unwrap();
                builder.add_entry_point(path_str);
                println!("cargo:rerun-if-changed={}", path_str);
            }
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }

    builder.build().unwrap().generate().unwrap();
}
