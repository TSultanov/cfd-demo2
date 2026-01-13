use glob::glob;
use std::fs;
use std::path::PathBuf;
use wgsl_bindgen::{WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};
use cfd2_codegen::compiler as codegen_compiler;

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

    pub mod shared {
        pub use cfd2_ir::solver::shared::*;
    }

    pub mod ir {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/solver/ir/mod.rs"));
    }
    pub mod model {
        pub mod linear_solver {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/linear_solver.rs"
            ));
        }
        pub mod method {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/method.rs"
            ));
        }
        pub mod eos {
            include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/solver/model/eos.rs"));
        }
        pub mod flux_layout {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/flux_layout.rs"
            ));
        }
        pub mod flux_module {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/flux_module.rs"
            ));
        }
        pub mod primitives {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/primitives.rs"
            ));
        }
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
            all_models,
            compressible_model, compressible_system, generic_diffusion_demo_model,
            generic_diffusion_demo_neumann_model, incompressible_momentum_model,
            incompressible_momentum_generic_model,
            incompressible_momentum_system, CompressibleFields, GenericCoupledFields,
            IncompressibleMomentumFields, ModelSpec,
        };
        #[allow(unused_imports)]
        pub use flux_layout::{FluxComponent, FluxLayout};
        #[allow(unused_imports)]
        pub use flux_module::{FluxModuleSpec, ReconstructionSpec};
        #[allow(unused_imports)]
        pub use primitives::PrimitiveDerivations;
        #[allow(unused_imports)]
        pub use eos::EosSpec;
        #[allow(unused_imports)]
        pub use method::MethodSpec;
        #[allow(unused_imports)]
        pub use gpu_spec::{expand_field_components, FluxSpec, GradientStorage, ModelGpuSpec};
        #[allow(unused_imports)]
        pub use kernel::{KernelKind, KernelPlan};
    }
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    enforce_codegen_ir_boundary(&manifest_dir);

    println!("cargo:rerun-if-changed=src/solver/scheme.rs");
    for entry in glob("src/solver/model/**/*.rs").expect("Failed to read model glob") {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }
    for entry in glob("crates/cfd2_codegen/src/solver/codegen/**/*.rs")
        .expect("Failed to read codegen glob")
    {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }
    for entry in glob("crates/cfd2_ir/src/solver/**/*.rs").expect("Failed to read ir glob") {
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

    let schemes = solver::model::backend::SchemeRegistry::new(solver::scheme::Scheme::Upwind);

    for model in solver::model::all_models() {
        solver::model::kernel::emit_model_kernels_wgsl(&manifest_dir, &model, &schemes)
            .unwrap_or_else(|err| panic!("codegen failed for model '{}': {err}", model.id));
    }

    generate_wgsl_binding_meta(&manifest_dir);
    generate_kernel_registry_map(&manifest_dir);

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

fn enforce_codegen_ir_boundary(manifest_dir: &str) {
    let codegen_dir = PathBuf::from(manifest_dir)
        .join("crates")
        .join("cfd2_codegen")
        .join("src")
        .join("solver")
        .join("codegen");

    let pattern = codegen_dir.join("**/*.rs").to_string_lossy().to_string();
    let mut violations = Vec::new();

    // Make it difficult to accidentally punch through the IR boundary via path imports.
    //
    // If codegen needs additional inputs, expand `crate::solver::ir` (the facade) or move
    // model-dependent orchestration into build-time code (e.g. `build.rs`).
    let needles = [
        "crate::solver::model",
        "crate::solver::{model",
        "crate::solver::{ model",
        "super::model",
        "super::{model",
        "super::{ model",
        // Catch `use crate::solver as s; use s::model;` and similar aliasing.
        "::model::",
        "::model;",
        "::model,",
        "::model}",
        "::model)",
        "::model as",
    ];

    for entry in glob(&pattern).expect("Failed to read codegen glob") {
        let Ok(path) = entry else {
            continue;
        };
        let Ok(src) = fs::read_to_string(&path) else {
            continue;
        };

        for (idx, line) in src.lines().enumerate() {
            if needles.iter().any(|needle| line.contains(needle)) {
                let rel = path.strip_prefix(manifest_dir).unwrap_or(&path);
                violations.push(format!(
                    "{}:{}: {}",
                    rel.display(),
                    idx + 1,
                    line.trim_end()
                ));
            }
        }
    }

    if !violations.is_empty() {
        let mut msg = String::new();
        msg.push_str("IR boundary violation: codegen must not reference model types.\n");
        msg.push_str("Use crate::solver::ir::{...} or move orchestration to build-time code.\n\n");
        for v in violations {
            msg.push_str(&v);
            msg.push('\n');
        }
        panic!("{msg}");
    }
}

fn generate_wgsl_binding_meta(manifest_dir: &str) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("wgsl_binding_meta.rs");

    let gen_dir = codegen_compiler::generated_dir_for(manifest_dir);
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
        (
            "generic_coupled_schur_setup",
            shader_dir.join("generic_coupled_schur_setup.wgsl"),
        ),
        ("schur_precond", shader_dir.join("schur_precond.wgsl")),
        (
            "schur_precond_generic",
            shader_dir.join("schur_precond_generic.wgsl"),
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

        // Transitional KT flux module kernels.
        ("kt_gradients", gen_dir.join("kt_gradients.wgsl")),
        ("flux_kt", gen_dir.join("flux_kt.wgsl")),
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

fn generate_kernel_registry_map(manifest_dir: &str) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("kernel_registry_map.rs");

    // Discover which models have per-model generic-coupled kernels.
    //
    // These kernels are emitted as:
    // - generic_coupled_assembly_<model_id>.wgsl
    // - generic_coupled_update_<model_id>.wgsl
    //
    // We include them in the same registry mapping so runtime lookup goes through a single path.
    let gen_dir = codegen_compiler::generated_dir_for(manifest_dir);
    let mut generic_coupled_models: Vec<(
        String,
        String,
        Vec<(u32, u32, String)>,
        Vec<(u32, u32, String)>,
    )> = Vec::new();

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
        let update_path = gen_dir.join(format!("generic_coupled_update_{model_id}.wgsl"));
        if !update_path.exists() {
            continue;
        }

        let assembly_src = fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("failed to read generated WGSL '{}': {err}", path.display())
        });
        let update_src = fs::read_to_string(&update_path).unwrap_or_else(|err| {
            panic!(
                "failed to read generated WGSL '{}': {err}",
                update_path.display()
            )
        });

        let assembly_bindings = parse_wgsl_bindings(&assembly_src);
        let update_bindings = parse_wgsl_bindings(&update_src);

        generic_coupled_models.push((
            model_id.to_string(),
            sanitize_rust_ident(model_id),
            assembly_bindings,
            update_bindings,
        ));
    }

    generic_coupled_models.sort_by(|a, b| a.0.cmp(&b.0));
    generic_coupled_models.dedup_by(|a, b| a.0 == b.0);

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
            "GenericCoupledApply",
            "generic_coupled_apply",
            "generic_coupled_apply",
        ),

        // KT flux module kernels for generic-coupled models.
        ("KtGradients", "kt_gradients", "kt_gradients"),
        ("FluxKt", "flux_kt", "flux_kt"),
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

        // AMG
        (
            "amg/smooth_op",
            "amg",
            "create_smooth_op_pipeline_embed_source",
        ),
        (
            "amg/restrict_residual",
            "amg",
            "create_restrict_residual_pipeline_embed_source",
        ),
        (
            "amg/prolongate_op",
            "amg",
            "create_prolongate_op_pipeline_embed_source",
        ),
        (
            "amg/clear",
            "amg",
            "create_clear_pipeline_embed_source",
        ),

        // Packing helpers
        (
            "amg_pack/pack_component",
            "amg_pack",
            "create_pack_component_pipeline_embed_source",
        ),
        (
            "amg_pack/unpack_component",
            "amg_pack",
            "create_unpack_component_pipeline_embed_source",
        ),

        // Block preconditioner
        (
            "block_precond/build_block_inv",
            "block_precond",
            "create_build_block_inv_pipeline_embed_source",
        ),
        (
            "block_precond/apply_block_precond",
            "block_precond",
            "create_apply_block_precond_pipeline_embed_source",
        ),

        // Schur preconditioner
        (
            "schur_precond/predict_and_form_schur",
            "schur_precond",
            "create_predict_and_form_schur_pipeline_embed_source",
        ),
        (
            "schur_precond/relax_pressure",
            "schur_precond",
            "create_relax_pressure_pipeline_embed_source",
        ),
        (
            "schur_precond/correct_velocity",
            "schur_precond",
            "create_correct_velocity_pipeline_embed_source",
        ),

        // Generic schur preconditioner (model-owned layout)
        (
            "schur_precond_generic/predict_and_form_schur",
            "schur_precond_generic",
            "create_predict_and_form_schur_pipeline_embed_source",
        ),
        (
            "schur_precond_generic/relax_pressure",
            "schur_precond_generic",
            "create_relax_pressure_pipeline_embed_source",
        ),
        (
            "schur_precond_generic/correct_velocity",
            "schur_precond_generic",
            "create_correct_velocity_pipeline_embed_source",
        ),

        // Generic-coupled Schur setup (extract diagonal/pressure blocks).
        (
            "generic_coupled_schur_setup/build_diag_and_pressure",
            "generic_coupled_schur_setup",
            "create_build_diag_and_pressure_pipeline_embed_source",
        ),

        // Coupled-solver preconditioner
        (
            "preconditioner/build_schur_rhs",
            "preconditioner",
            "create_build_schur_rhs_pipeline_embed_source",
        ),
        (
            "preconditioner/finalize_precond",
            "preconditioner",
            "create_finalize_precond_pipeline_embed_source",
        ),
        (
            "preconditioner/spmv_phat_v",
            "preconditioner",
            "create_spmv_phat_v_pipeline_embed_source",
        ),
        (
            "preconditioner/spmv_shat_t",
            "preconditioner",
            "create_spmv_shat_t_pipeline_embed_source",
        ),

        // GMRES ops/logic/CGS
        (
            "gmres_ops/spmv",
            "gmres_ops",
            "create_spmv_pipeline_embed_source",
        ),
        (
            "gmres_ops/axpy",
            "gmres_ops",
            "create_axpy_pipeline_embed_source",
        ),
        (
            "gmres_ops/axpy_from_y",
            "gmres_ops",
            "create_axpy_from_y_pipeline_embed_source",
        ),
        (
            "gmres_ops/axpby",
            "gmres_ops",
            "create_axpby_pipeline_embed_source",
        ),
        (
            "gmres_ops/scale",
            "gmres_ops",
            "create_scale_pipeline_embed_source",
        ),
        (
            "gmres_ops/scale_in_place",
            "gmres_ops",
            "create_scale_in_place_pipeline_embed_source",
        ),
        (
            "gmres_ops/copy",
            "gmres_ops",
            "create_copy_pipeline_embed_source",
        ),
        (
            "gmres_ops/dot_product_partial",
            "gmres_ops",
            "create_dot_product_partial_pipeline_embed_source",
        ),
        (
            "gmres_ops/norm_sq_partial",
            "gmres_ops",
            "create_norm_sq_partial_pipeline_embed_source",
        ),
        (
            "gmres_ops/reduce_final",
            "gmres_ops",
            "create_reduce_final_pipeline_embed_source",
        ),
        (
            "gmres_ops/reduce_final_and_finish_norm",
            "gmres_ops",
            "create_reduce_final_and_finish_norm_pipeline_embed_source",
        ),
        (
            "gmres_logic/update_hessenberg_givens",
            "gmres_logic",
            "create_update_hessenberg_givens_pipeline_embed_source",
        ),
        (
            "gmres_logic/solve_triangular",
            "gmres_logic",
            "create_solve_triangular_pipeline_embed_source",
        ),
        (
            "gmres_cgs/calc_dots_cgs",
            "gmres_cgs",
            "create_calc_dots_cgs_pipeline_embed_source",
        ),
        (
            "gmres_cgs/reduce_dots_cgs",
            "gmres_cgs",
            "create_reduce_dots_cgs_pipeline_embed_source",
        ),
        (
            "gmres_cgs/update_w_cgs",
            "gmres_cgs",
            "create_update_w_cgs_pipeline_embed_source",
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
    code.push_str("    model_id: &str,\n");
    code.push_str("    kernel_id: &str,\n");
    code.push_str(") -> Option<(\n");
    code.push_str("    &'static str,\n");
    code.push_str("    KernelPipelineCtor,\n");
    code.push_str("    &'static [crate::solver::gpu::wgsl_reflect::WgslBindingDesc],\n");
    code.push_str(")> {\n");
    code.push_str("    match (model_id, kernel_id) {\n");

    for (model_id, mod_id, assembly_bindings, update_bindings) in &generic_coupled_models {
        code.push_str(&format!(
            "        (\"{model_id}\", \"generic_coupled_assembly\") => {{\n"
        ));
        code.push_str("            use crate::solver::gpu::bindings::generated::");
        code.push_str(&format!("generic_coupled_assembly_{mod_id} as kernel;\n"));
        code.push_str("            Some((\n");
        code.push_str("                kernel::SHADER_STRING,\n");
        code.push_str("                kernel::compute::create_main_pipeline_embed_source,\n");
        code.push_str("                &[\n");
        for (group, binding, name) in assembly_bindings {
            code.push_str(&format!(
                "                    crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("                ],\n");
        code.push_str("            ))\n");
        code.push_str("        }\n");

        code.push_str(&format!(
            "        (\"{model_id}\", \"generic_coupled_update\") => {{\n"
        ));
        code.push_str("            use crate::solver::gpu::bindings::generated::");
        code.push_str(&format!("generic_coupled_update_{mod_id} as kernel;\n"));
        code.push_str("            Some((\n");
        code.push_str("                kernel::SHADER_STRING,\n");
        code.push_str("                kernel::compute::create_main_pipeline_embed_source,\n");
        code.push_str("                &[\n");
        for (group, binding, name) in update_bindings {
            code.push_str(&format!(
                "                    crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("                ],\n");
        code.push_str("            ))\n");
        code.push_str("        }\n");
    }

    for (_variant, module, kernel_id) in entries {
        let bindings_const = format!("{}_BINDINGS", module.to_ascii_uppercase());
        code.push_str(&format!("        (_, \"{kernel_id}\") => Some((\n"));
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
        code.push_str(&format!("        (_, \"{kernel_id}\") => Some((\n"));
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

                // Handle inline declarations like:
                // `@group(0) @binding(0) var<storage, read> foo: array<u32>;`
                if let Some(var_idx) = line.find("var") {
                    if let Some(name) = parse_var_name(line[var_idx..].trim_start()) {
                        out.push((group, binding, name));
                    }
                    pending = None;
                }
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
