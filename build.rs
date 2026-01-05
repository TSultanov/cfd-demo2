use glob::glob;
use wgsl_bindgen::{WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

#[allow(dead_code)]
mod solver {
    pub mod scheme {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/solver/scheme.rs"));
    }
    pub mod model {
        pub mod backend {
            pub mod ast {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/model/backend/ast.rs"
                ));
            }
            pub mod scheme {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/model/backend/scheme.rs"
                ));
            }
            pub mod state_layout {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/model/backend/state_layout.rs"
                ));
            }
            #[allow(unused_imports)]
            pub use ast::{
                fvc, fvm, Coefficient, Discretization, Equation, EquationSystem, FieldKind,
                FieldRef, FluxRef, Term, TermOp,
            };
            #[allow(unused_imports)]
            pub use scheme::{SchemeRegistry, TermKey};
            #[allow(unused_imports)]
            pub use state_layout::{StateField, StateLayout};
        }
        pub mod definitions {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/definitions.rs"
            ));
        }
        pub mod kernel {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/kernel.rs"
            ));
        }
        #[allow(unused_imports)]
        pub use definitions::{
            compressible_model, compressible_system, incompressible_momentum_model,
            incompressible_momentum_system, CompressibleFields, IncompressibleMomentumFields,
            ModelFields, ModelSpec,
        };
        #[allow(unused_imports)]
        pub use kernel::{KernelKind, KernelPlan};
    }
    pub mod codegen {
        pub mod compressible_assembly {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/compressible_assembly.rs"
            ));
        }
        pub mod compressible_apply {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/compressible_apply.rs"
            ));
        }
        pub mod compressible_flux_kt {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/compressible_flux_kt.rs"
            ));
        }
        pub mod compressible_update {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/compressible_update.rs"
            ));
        }
        pub mod coupled_assembly {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/coupled_assembly.rs"
            ));
        }
        pub mod coeff_expr {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/coeff_expr.rs"
            ));
        }
        pub mod flux_rhie_chow {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/flux_rhie_chow.rs"
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
        pub mod plan {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/plan.rs"
            ));
        }
        pub mod state_access {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/state_access.rs"
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
    for entry in glob("src/solver/model/**/*.rs").expect("Failed to read model glob") {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }
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
    let model = solver::model::incompressible_momentum_model();
    let schemes =
        solver::model::backend::SchemeRegistry::new(solver::scheme::Scheme::Upwind);
    if let Err(err) = solver::codegen::emit::emit_model_kernels_wgsl(
        &manifest_dir,
        &model,
        &schemes,
    ) {
        panic!("codegen failed: {}", err);
    }
    let compressible_model = solver::model::compressible_model();
    if let Err(err) = solver::codegen::emit::emit_model_kernels_wgsl(
        &manifest_dir,
        &compressible_model,
        &schemes,
    ) {
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
