use glob::glob;
use std::fs;
use std::path::PathBuf;
use wgsl_bindgen::{WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

#[allow(dead_code)]
mod solver {
    pub mod units {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/solver/units.rs"));
    }
    pub mod scheme {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/solver/scheme.rs"));
    }
    pub mod gpu {
        pub mod enums {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/gpu/enums.rs"
            ));
        }
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
            pub mod scheme_expansion {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/model/backend/scheme_expansion.rs"
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
            pub use scheme_expansion::{expand_schemes, SchemeExpansion};
            #[allow(unused_imports)]
            pub use state_layout::{StateField, StateLayout};
        }
        pub mod gpu_spec {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/gpu_spec.rs"
            ));
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
            compressible_model, compressible_system, generic_diffusion_demo_model,
            generic_diffusion_demo_neumann_model, incompressible_momentum_model,
            incompressible_momentum_system, CompressibleFields, GenericCoupledFields,
            IncompressibleMomentumFields, ModelSpec,
        };
        #[allow(unused_imports)]
        pub use gpu_spec::{expand_field_components, FluxSpec, GradientStorage, ModelGpuSpec};
        #[allow(unused_imports)]
        pub use kernel::{KernelKind, KernelPlan};
    }
    pub mod codegen {
        pub mod dsl {
            pub mod enums {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/codegen/dsl/enums.rs"
                ));
            }
            pub mod expr {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/codegen/dsl/expr.rs"
                ));
            }
            pub mod matrix {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/codegen/dsl/matrix.rs"
                ));
            }
            pub mod types {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/codegen/dsl/types.rs"
                ));
            }
            pub mod tensor {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/codegen/dsl/tensor.rs"
                ));
            }
            pub mod units {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/codegen/dsl/units.rs"
                ));
            }

            #[allow(unused_imports)]
            pub use enums::{EnumExpr, WgslEnum};
            #[allow(unused_imports)]
            pub use expr::{DslError, TypedExpr};
            #[allow(unused_imports)]
            pub use matrix::{
                BlockCsrMatrix, BlockCsrSoaEntry, BlockCsrSoaMatrix, BlockShape, CsrMatrix,
                CsrPattern,
            };
            pub use tensor::{
                AxisCons, AxisXY, Cons, MatExpr, NamedMatExpr, NamedVecExpr, VecExpr, XY,
            };
            pub use types::{DslType, ScalarType, Shape};
            pub use units::UnitDim;
        }
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
        pub mod compressible_explicit_update {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/compressible_explicit_update.rs"
            ));
        }
        pub mod compressible_gradients {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/compressible_gradients.rs"
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
        pub mod generic_coupled_kernels {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/generic_coupled_kernels.rs"
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
        pub mod reconstruction {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/reconstruction.rs"
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
        pub mod unified_assembly {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/codegen/unified_assembly.rs"
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
    for entry in glob("src/solver/codegen/**/*.rs").expect("Failed to read codegen glob") {
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

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    if let Err(err) = solver::codegen::emit::emit_system_main_wgsl(&manifest_dir) {
        panic!("codegen failed: {}", err);
    }
    let model = solver::model::incompressible_momentum_model();
    let schemes = solver::model::backend::SchemeRegistry::new(solver::scheme::Scheme::Upwind);
    if let Err(err) =
        solver::codegen::emit::emit_model_kernels_wgsl(&manifest_dir, &model, &schemes)
    {
        panic!("codegen failed: {}", err);
    }
    let compressible_model = solver::model::compressible_model();
    if let Err(err) =
        solver::codegen::emit::emit_model_kernels_wgsl(&manifest_dir, &compressible_model, &schemes)
    {
        panic!("codegen failed: {}", err);
    }
    let generic_model = solver::model::generic_diffusion_demo_model();
    if let Err(err) =
        solver::codegen::emit::emit_model_kernels_wgsl(&manifest_dir, &generic_model, &schemes)
    {
        panic!("codegen failed: {}", err);
    }
    let generic_neumann_model = solver::model::generic_diffusion_demo_neumann_model();
    if let Err(err) = solver::codegen::emit::emit_model_kernels_wgsl(
        &manifest_dir,
        &generic_neumann_model,
        &schemes,
    ) {
        panic!("codegen failed: {}", err);
    }

    generate_generic_coupled_registry(&manifest_dir);
    generate_wgsl_binding_meta(&manifest_dir);
    generate_kernel_registry_map();

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

fn generate_generic_coupled_registry(manifest_dir: &str) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("generic_coupled_registry.rs");

    let gen_dir = solver::codegen::emit::generated_dir_for(manifest_dir);
    let mut model_ids = Vec::new();

    let pattern = gen_dir
        .join("generic_coupled_assembly_*.wgsl")
        .to_string_lossy()
        .to_string();
    for entry in glob(&pattern).expect("Failed to read generic coupled assembly glob") {
        let Ok(path) = entry else { continue };
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(model_id) = stem.strip_prefix("generic_coupled_assembly_") else {
            continue;
        };
        let update = gen_dir.join(format!("generic_coupled_update_{model_id}.wgsl"));
        if update.exists() {
            model_ids.push(model_id.to_string());
        }
    }

    model_ids.sort();
    model_ids.dedup();

    let mut code = String::new();
    code.push_str("// @generated by build.rs\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");
    code.push_str("#[allow(dead_code)]\n");
    code.push_str("pub(crate) const GENERIC_COUPLED_MODEL_IDS: &[&str] = &[\n");
    for id in &model_ids {
        code.push_str(&format!("    \"{id}\",\n"));
    }
    code.push_str("];\n\n");

    code.push_str("pub(crate) fn generic_coupled_pair(\n");
    code.push_str("    model_id: &str,\n");
    code.push_str(") -> Option<(\n");
    code.push_str("    &'static str,\n");
    code.push_str("    fn(&wgpu::Device) -> wgpu::ComputePipeline,\n");
    code.push_str("    &'static [crate::solver::gpu::wgsl_reflect::WgslBindingDesc],\n");
    code.push_str("    &'static str,\n");
    code.push_str("    fn(&wgpu::Device) -> wgpu::ComputePipeline,\n");
    code.push_str("    &'static [crate::solver::gpu::wgsl_reflect::WgslBindingDesc],\n");
    code.push_str(")> {\n");
    code.push_str("    match model_id {\n");

    for id in &model_ids {
        let mod_id = sanitize_rust_ident(id);
        let assembly_path = gen_dir.join(format!("generic_coupled_assembly_{id}.wgsl"));
        let update_path = gen_dir.join(format!("generic_coupled_update_{id}.wgsl"));
        let assembly_src = fs::read_to_string(&assembly_path).unwrap_or_else(|err| {
            panic!(
                "failed to read generated WGSL '{}': {err}",
                assembly_path.display()
            )
        });
        let update_src = fs::read_to_string(&update_path).unwrap_or_else(|err| {
            panic!(
                "failed to read generated WGSL '{}': {err}",
                update_path.display()
            )
        });
        let assembly_bindings = parse_wgsl_bindings(&assembly_src);
        let update_bindings = parse_wgsl_bindings(&update_src);

        code.push_str(&format!("        \"{id}\" => {{\n"));
        code.push_str("            use crate::solver::gpu::bindings::generated::");
        code.push_str(&format!("generic_coupled_assembly_{mod_id} as assembly;\n"));
        code.push_str("            use crate::solver::gpu::bindings::generated::");
        code.push_str(&format!("generic_coupled_update_{mod_id} as update;\n"));
        code.push_str("            Some((\n");
        code.push_str("                assembly::SHADER_STRING,\n");
        code.push_str("                assembly::compute::create_main_pipeline_embed_source,\n");
        code.push_str("                &[\n");
        for (group, binding, name) in &assembly_bindings {
            code.push_str(&format!(
                "                    crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("                ],\n");
        code.push_str("                update::SHADER_STRING,\n");
        code.push_str("                update::compute::create_main_pipeline_embed_source,\n");
        code.push_str("                &[\n");
        for (group, binding, name) in &update_bindings {
            code.push_str(&format!(
                "                    crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("                ],\n");
        code.push_str("            ))\n");
        code.push_str("        }\n");
    }

    code.push_str("        _ => None,\n");
    code.push_str("    }\n");
    code.push_str("}\n");

    write_if_changed(&out_path, &code);
}

fn generate_wgsl_binding_meta(manifest_dir: &str) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("wgsl_binding_meta.rs");

    let gen_dir = solver::codegen::emit::generated_dir_for(manifest_dir);
    let shader_dir = PathBuf::from(manifest_dir)
        .join("src")
        .join("solver")
        .join("gpu")
        .join("shaders");
    let shader_paths = [
        ("amg", shader_dir.join("amg.wgsl")),
        ("amg_pack", shader_dir.join("amg_pack.wgsl")),
        ("block_precond", shader_dir.join("block_precond.wgsl")),
        ("dot_product", shader_dir.join("dot_product.wgsl")),
        ("dot_product_pair", shader_dir.join("dot_product_pair.wgsl")),
        ("gmres_cgs", shader_dir.join("gmres_cgs.wgsl")),
        ("gmres_logic", shader_dir.join("gmres_logic.wgsl")),
        ("gmres_ops", shader_dir.join("gmres_ops.wgsl")),
        ("linear_solver", shader_dir.join("linear_solver.wgsl")),
        ("preconditioner", shader_dir.join("preconditioner.wgsl")),
        ("scalars", shader_dir.join("scalars.wgsl")),
        ("schur_precond", shader_dir.join("schur_precond.wgsl")),
        (
            "system_main",
            gen_dir.join("system_main.wgsl"),
        ),
        (
            "ei_assembly",
            gen_dir.join("ei_assembly.wgsl"),
        ),
        (
            "ei_apply",
            gen_dir.join("ei_apply.wgsl"),
        ),
        (
            "ei_explicit_update",
            gen_dir.join("ei_explicit_update.wgsl"),
        ),
        (
            "ei_flux_kt",
            gen_dir.join("ei_flux_kt.wgsl"),
        ),
        (
            "ei_gradients",
            gen_dir.join("ei_gradients.wgsl"),
        ),
        (
            "ei_update",
            gen_dir.join("ei_update.wgsl"),
        ),
        (
            "coupled_assembly_merged",
            gen_dir.join("coupled_assembly_merged.wgsl"),
        ),
        ("flux_rhie_chow", gen_dir.join("flux_rhie_chow.wgsl")),
        (
            "generic_coupled_apply",
            gen_dir.join("generic_coupled_apply.wgsl"),
        ),
        ("prepare_coupled", gen_dir.join("prepare_coupled.wgsl")),
        ("pressure_assembly", gen_dir.join("pressure_assembly.wgsl")),
        (
            "update_fields_from_coupled",
            gen_dir.join("update_fields_from_coupled.wgsl"),
        ),
    ];

    let mut code = String::new();
    code.push_str("// @generated by build.rs\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");

    for (shader_name, path) in shader_paths {
        let src = fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("failed to read generated WGSL '{}': {err}", path.display())
        });
        let bindings = parse_wgsl_bindings(&src);

        let ident = sanitize_rust_ident(shader_name).to_ascii_uppercase();
        let const_name = format!("{ident}_BINDINGS");
        code.push_str(&format!(
            "pub(crate) const {const_name}: &[crate::solver::gpu::wgsl_reflect::WgslBindingDesc] = &[\n"
        ));
        for (group, binding, name) in bindings {
            code.push_str(&format!(
                "    crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("];\n\n");
    }

    write_if_changed(&out_path, &code);
}

fn generate_kernel_registry_map() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("kernel_registry_map.rs");

    // (KernelKind variant, generated module name, stable KernelId string)
    let entries: &[(&str, &str, &str)] = &[
        ("PrepareCoupled", "prepare_coupled", "prepare_coupled"),
        // Note: kernel kind is `CoupledAssembly`, but current shader module is `coupled_assembly_merged`.
        (
            "CoupledAssembly",
            "coupled_assembly_merged",
            "coupled_assembly",
        ),
        ("PressureAssembly", "pressure_assembly", "pressure_assembly"),
        (
            "UpdateFieldsFromCoupled",
            "update_fields_from_coupled",
            "update_fields_from_coupled",
        ),
        ("FluxRhieChow", "flux_rhie_chow", "flux_rhie_chow"),
        (
            "EiAssembly",
            "ei_assembly",
            "ei_assembly",
        ),
        (
            "EiApply",
            "ei_apply",
            "ei_apply",
        ),
        (
            "EiExplicitUpdate",
            "ei_explicit_update",
            "ei_explicit_update",
        ),
        (
            "EiGradients",
            "ei_gradients",
            "ei_gradients",
        ),
        (
            "EiUpdate",
            "ei_update",
            "ei_update",
        ),
        (
            "EiFluxKt",
            "ei_flux_kt",
            "ei_flux_kt",
        ),
        (
            "GenericCoupledApply",
            "generic_coupled_apply",
            "generic_coupled_apply",
        ),
    ];

    // (stable KernelId string, bindings module name, compute pipeline ctor function name)
    // These are handwritten WGSL kernels that we want to route through the same registry path.
    let id_only_entries: &[(&str, &str, &str)] = &[
        (
            "dot_product",
            "dot_product",
            "create_main_pipeline_embed_source",
        ),
        (
            "dot_product_pair",
            "dot_product_pair",
            "create_main_pipeline_embed_source",
        ),
        (
            "scalars/init_scalars",
            "scalars",
            "create_init_scalars_pipeline_embed_source",
        ),
        (
            "scalars/init_cg_scalars",
            "scalars",
            "create_init_cg_scalars_pipeline_embed_source",
        ),
        (
            "scalars/reduce_rho_new_r_r",
            "scalars",
            "create_reduce_rho_new_r_r_pipeline_embed_source",
        ),
        (
            "scalars/reduce_r0_v",
            "scalars",
            "create_reduce_r0_v_pipeline_embed_source",
        ),
        (
            "scalars/reduce_t_s_t_t",
            "scalars",
            "create_reduce_t_s_t_t_pipeline_embed_source",
        ),
        (
            "scalars/update_cg_alpha",
            "scalars",
            "create_update_cg_alpha_pipeline_embed_source",
        ),
        (
            "scalars/update_cg_beta",
            "scalars",
            "create_update_cg_beta_pipeline_embed_source",
        ),
        (
            "scalars/update_rho_old",
            "scalars",
            "create_update_rho_old_pipeline_embed_source",
        ),
        (
            "linear_solver/spmv_p_v",
            "linear_solver",
            "create_spmv_p_v_pipeline_embed_source",
        ),
        (
            "linear_solver/spmv_s_t",
            "linear_solver",
            "create_spmv_s_t_pipeline_embed_source",
        ),
        (
            "linear_solver/bicgstab_update_x_r",
            "linear_solver",
            "create_bicgstab_update_x_r_pipeline_embed_source",
        ),
        (
            "linear_solver/bicgstab_update_p",
            "linear_solver",
            "create_bicgstab_update_p_pipeline_embed_source",
        ),
        (
            "linear_solver/bicgstab_update_s",
            "linear_solver",
            "create_bicgstab_update_s_pipeline_embed_source",
        ),
        (
            "linear_solver/cg_update_x_r",
            "linear_solver",
            "create_cg_update_x_r_pipeline_embed_source",
        ),
        (
            "linear_solver/cg_update_p",
            "linear_solver",
            "create_cg_update_p_pipeline_embed_source",
        ),
    ];

    let mut code = String::new();
    code.push_str("// @generated by build.rs\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");
    code.push_str("use crate::solver::gpu::{bindings, wgsl_meta};\n");
    code.push_str("use crate::solver::model::KernelKind;\n\n");
    code.push_str(
        "pub(crate) type KernelPipelineCtor = fn(&wgpu::Device) -> wgpu::ComputePipeline;\n\n",
    );
    code.push_str("pub(crate) fn kernel_entry(\n");
    code.push_str("    kind: KernelKind,\n");
    code.push_str(") -> Option<(\n");
    code.push_str("    &'static str,\n");
    code.push_str("    KernelPipelineCtor,\n");
    code.push_str("    &'static [crate::solver::gpu::wgsl_reflect::WgslBindingDesc],\n");
    code.push_str(")> {\n");
    code.push_str("    match kind {\n");

    for (variant, module, _kernel_id) in entries {
        let bindings_const = format!("{}_BINDINGS", module.to_ascii_uppercase());
        code.push_str(&format!("        KernelKind::{variant} => Some((\n"));
        code.push_str(&format!(
            "            bindings::generated::{module}::SHADER_STRING,\n"
        ));
        code.push_str(&format!(
            "            bindings::generated::{module}::compute::create_main_pipeline_embed_source,\n"
        ));
        code.push_str(&format!("            wgsl_meta::{bindings_const},\n"));
        code.push_str("        )),\n");
    }

    code.push_str("        _ => None,\n");
    code.push_str("    }\n");
    code.push_str("}\n");

    code.push_str("\n");
    code.push_str("pub(crate) fn kernel_entry_by_id(\n");
    code.push_str("    kernel_id: &str,\n");
    code.push_str(") -> Option<(\n");
    code.push_str("    &'static str,\n");
    code.push_str("    KernelPipelineCtor,\n");
    code.push_str("    &'static [crate::solver::gpu::wgsl_reflect::WgslBindingDesc],\n");
    code.push_str(")> {\n");
    code.push_str("    match kernel_id {\n");

    for (_variant, module, kernel_id) in entries {
        let bindings_const = format!("{}_BINDINGS", module.to_ascii_uppercase());
        code.push_str(&format!("        \"{kernel_id}\" => Some((\n"));
        code.push_str(&format!(
            "            bindings::generated::{module}::SHADER_STRING,\n"
        ));
        code.push_str(&format!(
            "            bindings::generated::{module}::compute::create_main_pipeline_embed_source,\n"
        ));
        code.push_str(&format!("            wgsl_meta::{bindings_const},\n"));
        code.push_str("        )),\n");
    }

    for (kernel_id, module, ctor) in id_only_entries {
        let bindings_const = format!("{}_BINDINGS", module.to_ascii_uppercase());
        code.push_str(&format!("        \"{kernel_id}\" => Some((\n"));
        code.push_str(&format!("            bindings::{module}::SHADER_STRING,\n"));
        code.push_str(&format!("            bindings::{module}::compute::{ctor},\n"));
        code.push_str(&format!("            wgsl_meta::{bindings_const},\n"));
        code.push_str("        )),\n");
    }

    code.push_str("        _ => None,\n");
    code.push_str("    }\n");
    code.push_str("}\n");

    write_if_changed(&out_path, &code);
}

fn sanitize_rust_ident(raw: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in raw.chars().enumerate() {
        let valid = ch.is_ascii_alphanumeric() || ch == '_';
        let mut mapped = if valid { ch } else { '_' };
        if idx == 0 && mapped.is_ascii_digit() {
            out.push('_');
        }
        // avoid empty id
        if idx == 0 && mapped == '_' && raw.len() == 1 {
            mapped = 'x';
        }
        out.push(mapped);
    }
    if out.is_empty() {
        out.push('x');
    }
    out
}

fn parse_wgsl_bindings(shader: &str) -> Vec<(u32, u32, String)> {
    let mut out = Vec::new();
    let mut pending: Option<(u32, u32)> = None;

    for raw in shader.lines() {
        let line = raw.trim();

        if pending.is_none() && line.contains("@group(") && line.contains("@binding(") {
            let group = parse_attr_u32(line, "@group(");
            let binding = parse_attr_u32(line, "@binding(");
            if let (Some(group), Some(binding)) = (group, binding) {
                pending = Some((group, binding));
            }
            continue;
        }

        let Some((group, binding)) = pending else {
            continue;
        };

        if line.starts_with("var") {
            if let Some(name) = parse_var_name(line) {
                out.push((group, binding, name));
            }
            pending = None;
        }
    }

    out.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    out.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    out
}

fn parse_attr_u32(line: &str, prefix: &str) -> Option<u32> {
    let start = line.find(prefix)? + prefix.len();
    let rest = &line[start..];
    let end = rest.find(')')?;
    rest[..end].trim().parse().ok()
}

fn parse_var_name(line: &str) -> Option<String> {
    // Expected patterns:
    // - var<storage, read> face_owner: array<u32>;
    // - var<uniform> constants: Constants;
    // - var state: array<f32>;
    let after_var = line.strip_prefix("var")?.trim_start();
    let after_decl = if let Some(idx) = after_var.find('>') {
        after_var[idx + 1..].trim_start()
    } else {
        after_var
    };
    let name_end = after_decl.find(':')?;
    let name = after_decl[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    Some(name.to_string())
}

fn write_if_changed(path: &PathBuf, contents: &str) {
    if let Ok(existing) = fs::read_to_string(path) {
        if existing == contents {
            return;
        }
    }
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, contents).expect("Failed to write generated registry file");
}
