use glob::glob;
use wgsl_bindgen::{WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

fn main() {
    println!("cargo:rerun-if-changed=src/solver/gpu/shaders");

    let mut builder = WgslBindgenOptionBuilder::default();
    builder
        .workspace_root("src/solver/gpu/shaders")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .derive_serde(false)
        .output("src/solver/gpu/bindings.rs");

    for entry in glob("src/solver/gpu/shaders/*.wgsl").expect("Failed to read glob pattern") {
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
