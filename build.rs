use glob::glob;
use std::fs;
use std::path::{Path, PathBuf};
use wgsl_bindgen::{WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

#[allow(dead_code)]
mod solver {
    pub mod dimensions {
        pub use cfd2_ir::solver::dimensions::*;
    }
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
        pub mod ports {
            // Include the full ports module (intern module is cfg-gated for runtime only)
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/ports/mod.rs"
            ));
        }
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
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/eos.rs"
            ));
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
        pub mod flux_schemes {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/flux_schemes.rs"
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
            pub mod typed_ast {
                include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/crates/cfd2_ir/src/solver/model/backend/typed_ast.rs"
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
            // Re-export typed_ast items
            #[allow(unused_imports)]
            pub use typed_ast::{
                typed_fvc, typed_fvm, Kind, Scalar, TypedCoeff, TypedEquation, TypedEquationSystem,
                TypedFieldRef, TypedFluxRef, TypedTerm, TypedTermSum, Vector2, Vector3,
            };
        }
        pub mod gpu_spec {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/gpu_spec.rs"
            ));
        }
        pub mod modules {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/modules/mod.rs"
            ));
        }
        pub mod definitions {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/definitions.rs"
            ));
        }
        pub mod module {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/module.rs"
            ));
        }
        pub mod invariants {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/invariants.rs"
            ));
        }
        pub mod kernel {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/solver/model/kernel.rs"
            ));
        }
        #[allow(unused_imports)]
        pub use super::ir::LimiterSpec;
        #[allow(unused_imports)]
        pub use definitions::{
            all_models, compressible_model, compressible_system, generic_diffusion_demo_model,
            generic_diffusion_demo_neumann_model, incompressible_momentum_model,
            incompressible_momentum_system, CompressibleFields, GenericCoupledFields,
            IncompressibleMomentumFields, ModelSpec,
        };
        #[allow(unused_imports)]
        pub use eos::EosSpec;
        #[allow(unused_imports)]
        pub use flux_layout::{FluxComponent, FluxLayout};
        #[allow(unused_imports)]
        pub use flux_module::FluxModuleSpec;
        #[allow(unused_imports)]
        pub use gpu_spec::{expand_field_components, GradientStorage};
        #[allow(unused_imports)]
        pub use kernel::KernelId;
        #[allow(unused_imports)]
        pub use method::MethodSpec;
        #[allow(unused_imports)]
        pub use primitives::PrimitiveDerivations;
    }
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    enforce_codegen_ir_boundary(&manifest_dir);

    let mut model_codegen_inputs: Vec<PathBuf> = Vec::new();

    println!("cargo:rerun-if-changed=src/solver/scheme.rs");
    model_codegen_inputs.push(PathBuf::from("src/solver/scheme.rs"));
    for entry in glob("src/solver/model/**/*.rs").expect("Failed to read model glob") {
        match entry {
            Ok(path) => {
                println!("cargo:rerun-if-changed={}", path.display());
                model_codegen_inputs.push(path);
            }
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }
    for entry in
        glob("crates/cfd2_codegen/src/solver/codegen/**/*.rs").expect("Failed to read codegen glob")
    {
        match entry {
            Ok(path) => {
                println!("cargo:rerun-if-changed={}", path.display());
                model_codegen_inputs.push(path);
            }
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }
    for entry in glob("crates/cfd2_ir/src/solver/**/*.rs").expect("Failed to read ir glob") {
        match entry {
            Ok(path) => {
                println!("cargo:rerun-if-changed={}", path.display());
                model_codegen_inputs.push(path);
            }
            Err(e) => println!("cargo:warning=Glob error: {:?}", e),
        }
    }
    println!("cargo:rerun-if-changed=src/solver/gpu/shaders");
    println!("cargo:rerun-if-changed=src/solver/gpu/lowering/named_params");

    let mut builder = WgslBindgenOptionBuilder::default();
    builder
        .workspace_root("src/solver/gpu/shaders")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .derive_serde(false)
        .output("src/solver/gpu/bindings.rs");

    let schemes = solver::model::backend::SchemeRegistry::new(solver::scheme::Scheme::Upwind);

    let models = solver::model::all_models();

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let model_codegen_fingerprint = fingerprint_files(&model_codegen_inputs);
    let model_codegen_fingerprint_path =
        PathBuf::from(&out_dir).join("cfd2_model_codegen_fingerprint.txt");
    let prev_model_codegen_fingerprint = read_u64_hex(&model_codegen_fingerprint_path);

    let shader_dir = PathBuf::from(&manifest_dir)
        .join("src")
        .join("solver")
        .join("gpu")
        .join("shaders");
    let generated_dir = shader_dir.join("generated");
    let has_generated_kernels = !list_wgsl_files_recursive(&generated_dir).is_empty();

    let should_regenerate_model_kernels = !has_generated_kernels
        || prev_model_codegen_fingerprint
            .map(|prev| prev != model_codegen_fingerprint)
            .unwrap_or(true);

    if should_regenerate_model_kernels {
        solver::model::kernel::emit_shared_kernels_wgsl_with_ids_for_models(
            &manifest_dir,
            &models,
            &schemes,
        )
        .unwrap_or_else(|err| panic!("codegen failed for shared kernels: {err}"));

        for model in &models {
            solver::model::kernel::emit_model_kernels_wgsl_with_ids(&manifest_dir, model, &schemes)
                .unwrap_or_else(|err| panic!("codegen failed for model '{}': {err}", model.id));
        }

        write_u64_hex(&model_codegen_fingerprint_path, model_codegen_fingerprint);
    }

    generate_kernel_registry_map(&manifest_dir, &models);
    generate_fusion_schedule_registry(&manifest_dir, &models);
    generate_named_param_registry(&manifest_dir);

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

    // Post-process: add clippy::too_many_arguments to the allow list in the generated file
    postprocess_bindings_for_clippy("src/solver/gpu/bindings.rs");
}

/// Idempotently adds clippy::too_many_arguments to the allow list in the generated bindings file.
fn postprocess_bindings_for_clippy(bindings_path: &str) {
    let content = fs::read_to_string(bindings_path).unwrap_or_else(|err| {
        panic!("Failed to read {bindings_path}: {err}");
    });

    // Check if already contains too_many_arguments
    if content.contains("clippy::too_many_arguments") {
        return;
    }

    // Find the existing #![allow(...)] line and add clippy::too_many_arguments to it
    // The line looks like: #![allow(unused, non_snake_case, non_camel_case_types, non_upper_case_globals)]
    let updated = if let Some(start_pos) = content.find("#![allow(") {
        // Insert inside the allow-list parentheses: before the closing `)]`.
        if let Some(relative_end) = content[start_pos..].find(")]") {
            let end_pos = start_pos + relative_end; // points at the `)` in `)]`
            let before = &content[..end_pos];
            let after = &content[end_pos..];
            format!("{before}, clippy::too_many_arguments{after}")
        } else {
            content.clone()
        }
    } else {
        // Fallback: insert a new allow after the header comment block
        let after_header = content.find("#![allow(").unwrap_or(0);
        let before = &content[..after_header];
        let after = &content[after_header..];
        format!("{before}#![allow(clippy::too_many_arguments)]\n{after}")
    };

    fs::write(bindings_path, updated).unwrap_or_else(|err| {
        panic!("Failed to write {bindings_path}: {err}");
    });
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

type WgslBinding = (u32, u32, String);
type PerModelEntry = (String, String, String, Vec<WgslBinding>);
type SharedEntry = (String, String, Vec<WgslBinding>);

fn generate_kernel_registry_map(manifest_dir: &str, models: &[solver::model::ModelSpec]) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("kernel_registry_map.rs");

    let mut per_model_entries: Vec<PerModelEntry> = Vec::new();
    let mut shared_entries: Vec<SharedEntry> = Vec::new();

    let shader_dir = PathBuf::from(manifest_dir)
        .join("src")
        .join("solver")
        .join("gpu")
        .join("shaders");
    let generated_dir = shader_dir.join("generated");

    let mut shared_kernel_ids: std::collections::HashSet<solver::model::KernelId> =
        std::collections::HashSet::new();
    for model in models {
        for module in &model.modules {
            let module: &dyn solver::model::module::ModelModule = module;
            for gen in module.kernel_generators() {
                if gen.scope == solver::model::kernel::KernelWgslScope::Shared {
                    shared_kernel_ids.insert(gen.id);
                }
            }
        }
    }

    let mut shared_kernel_ids: Vec<solver::model::KernelId> =
        shared_kernel_ids.into_iter().collect();
    shared_kernel_ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));

    for kernel_id in shared_kernel_ids {
        let filename = solver::model::kernel::kernel_output_name_for_model("", kernel_id)
            .unwrap_or_else(|err| panic!("failed to compute shared kernel output name: {err}"));
        let path = generated_dir.join(filename);
        let src = fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("failed to read generated WGSL '{}': {err}", path.display())
        });
        let bindings = parse_wgsl_bindings(&src);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("bad generated WGSL file name: {}", path.display()));
        let module_name = sanitize_rust_ident(stem);
        shared_entries.push((kernel_id.as_str().to_string(), module_name, bindings));
    }

    for model in models {
        let mut per_model_ids: std::collections::HashSet<solver::model::KernelId> =
            std::collections::HashSet::new();
        for module in &model.modules {
            let module: &dyn solver::model::module::ModelModule = module;
            for gen in module.kernel_generators() {
                if gen.scope == solver::model::kernel::KernelWgslScope::PerModel {
                    per_model_ids.insert(gen.id);
                }
            }
        }
        for replacement_id in
            solver::model::kernel::derive_fusion_replacement_kernel_ids_for_model(model)
        {
            per_model_ids.insert(replacement_id);
        }

        let mut per_model_ids: Vec<solver::model::KernelId> = per_model_ids.into_iter().collect();
        per_model_ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));

        for kernel_id in per_model_ids {
            let filename = solver::model::kernel::kernel_output_name_for_model(model.id, kernel_id)
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to compute per-model kernel output name for model '{}': {err}",
                        model.id
                    )
                });
            let path = generated_dir.join(filename);
            let src = fs::read_to_string(&path).unwrap_or_else(|err| {
                panic!(
                    "failed to read generated WGSL '{}' for model '{}' kernel '{}': {err}",
                    path.display(),
                    model.id,
                    kernel_id.as_str()
                )
            });
            let bindings = parse_wgsl_bindings(&src);
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_else(|| panic!("bad generated WGSL file name: {}", path.display()));
            let module_name = sanitize_rust_ident(stem);
            per_model_entries.push((
                model.id.to_string(),
                kernel_id.as_str().to_string(),
                module_name,
                bindings,
            ));
        }
    }

    per_model_entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    per_model_entries.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    shared_entries.sort_by(|a, b| a.0.cmp(&b.0));
    shared_entries.dedup_by(|a, b| a.0 == b.0);

    let mut handwritten_entries: Vec<PerModelEntry> = Vec::new();
    for path in list_wgsl_files_recursive(&shader_dir) {
        // Handwritten kernels only: generated WGSL is mapped separately via per-model/shared entries.
        if path.starts_with(&generated_dir) {
            continue;
        }
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read WGSL '{}': {err}", path.display()));
        let bindings = parse_wgsl_bindings(&src);
        let entrypoints = parse_wgsl_compute_entrypoints(&src);
        if entrypoints.is_empty() {
            continue;
        }
        let rel_stem = wgsl_relative_stem(&shader_dir, &path);
        let module_path = wgsl_bindings_module_path(&rel_stem);
        if entrypoints.len() == 1 && entrypoints[0] == "main" {
            handwritten_entries.push((
                rel_stem,
                module_path,
                "create_main_pipeline_embed_source".to_string(),
                bindings.clone(),
            ));
            continue;
        }
        for entry in entrypoints {
            handwritten_entries.push((
                format!("{rel_stem}/{entry}"),
                module_path.clone(),
                format!("create_{entry}_pipeline_embed_source"),
                bindings.clone(),
            ));
        }
    }
    handwritten_entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    handwritten_entries.dedup_by(|a, b| a.0 == b.0);

    let mut code = String::new();
    code.push_str("// @generated by build.rs\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");
    code.push_str("use crate::solver::gpu::bindings;\n\n");
    code.push_str(
        "pub(crate) type KernelPipelineCtor = fn(&wgpu::Device) -> wgpu::ComputePipeline;\n\n",
    );
    code.push_str("pub(crate) fn kernel_entry_by_id(\n");
    code.push_str("    model_id: &str,\n");
    code.push_str("    kernel_id: &str,\n");
    code.push_str(") -> Option<(\n");
    code.push_str("    &'static str,\n");
    code.push_str("    KernelPipelineCtor,\n");
    code.push_str("    &'static [crate::solver::gpu::wgsl_reflect::WgslBindingDesc],\n");
    code.push_str(")> {\n");
    code.push_str("    match (model_id, kernel_id) {\n");

    for (model_id, kernel_id, module_name, bindings) in &per_model_entries {
        code.push_str(&format!(
            "        (\"{model_id}\", \"{kernel_id}\") => {{\n"
        ));
        code.push_str("            use crate::solver::gpu::bindings::generated::");
        code.push_str(&format!("{module_name} as kernel;\n"));
        code.push_str("            Some((\n");
        code.push_str("                kernel::SHADER_STRING,\n");
        code.push_str("                kernel::compute::create_main_pipeline_embed_source,\n");
        code.push_str("                &[\n");
        for (group, binding, name) in bindings {
            code.push_str(&format!(
                "                    crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("                ],\n");
        code.push_str("            ))\n");
        code.push_str("        }\n");
    }

    for (kernel_id, module_name, bindings) in &shared_entries {
        code.push_str(&format!("        (_, \"{kernel_id}\") => {{\n"));
        code.push_str("            use crate::solver::gpu::bindings::generated::");
        code.push_str(&format!("{module_name} as kernel;\n"));
        code.push_str("            Some((\n");
        code.push_str("                kernel::SHADER_STRING,\n");
        code.push_str("                kernel::compute::create_main_pipeline_embed_source,\n");
        code.push_str("                &[\n");
        for (group, binding, name) in bindings {
            code.push_str(&format!(
                "                    crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("                ],\n");
        code.push_str("            ))\n");
        code.push_str("        }\n");
    }

    for (kernel_id, module, ctor, bindings) in handwritten_entries {
        code.push_str(&format!("        (_, \"{kernel_id}\") => Some((\n"));
        code.push_str(&format!("            bindings::{module}::SHADER_STRING,\n"));
        code.push_str(&format!(
            "            bindings::{module}::compute::{ctor},\n"
        ));
        code.push_str("            &[\n");
        for (group, binding, name) in bindings {
            code.push_str(&format!(
                "                crate::solver::gpu::wgsl_reflect::WgslBindingDesc {{ group: {group}, binding: {binding}, name: \"{name}\" }},\n"
            ));
        }
        code.push_str("            ],\n");
        code.push_str("        )),\n");
    }

    code.push_str("        _ => None,\n");
    code.push_str("    }\n");
    code.push_str("}\n");

    write_if_changed(&out_path, &code);
}

type FusedKernelEntry = (String, u8, u8);
type FusedScheduleEntry = (String, u8, bool, u8, Vec<FusedKernelEntry>, Vec<String>);

fn generate_fusion_schedule_registry(manifest_dir: &str, models: &[solver::model::ModelSpec]) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("fusion_schedule_registry.rs");

    let shared_generated_ids = collect_shared_generated_kernel_ids(models);
    let per_model_generated_ids = collect_per_model_generated_kernel_ids(models);
    let handwritten_kernel_ids = collect_handwritten_kernel_ids(manifest_dir);

    let stepping_cases = [
        (
            0u8,
            solver::model::kernel::KernelFusionStepping::Explicit,
            false,
        ),
        (
            1u8,
            solver::model::kernel::KernelFusionStepping::Implicit,
            true,
        ),
        (
            2u8,
            solver::model::kernel::KernelFusionStepping::Coupled,
            false,
        ),
    ];
    let policies = [
        solver::model::kernel::KernelFusionPolicy::Off,
        solver::model::kernel::KernelFusionPolicy::Safe,
        solver::model::kernel::KernelFusionPolicy::Aggressive,
    ];

    let mut entries: Vec<FusedScheduleEntry> = Vec::new();

    for model in models {
        let model_kernel_specs = solver::model::kernel::derive_kernel_specs_for_model(model)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to derive kernel specs for model '{}': {err}",
                    model.id
                )
            });
        let fusion_rules = solver::model::kernel::derive_kernel_fusion_rules_for_model(model);
        let module_names: Vec<&'static str> = model.modules.iter().map(|m| m.name).collect();

        for (stepping_tag, fusion_stepping, satisfies_requires_implicit_stepping) in stepping_cases
        {
            for has_grad_state in [false, true] {
                let filtered_specs: Vec<solver::model::kernel::ModelKernelSpec> =
                    model_kernel_specs
                        .iter()
                        .copied()
                        .filter(|spec| {
                            include_kernel_in_schedule(
                                spec.condition,
                                has_grad_state,
                                satisfies_requires_implicit_stepping,
                            )
                        })
                        .collect();

                for policy in policies {
                    let ctx = solver::model::kernel::KernelFusionContext {
                        policy,
                        stepping: fusion_stepping,
                        has_grad_state,
                        module_names: &module_names,
                    };
                    let result = solver::model::kernel::apply_model_fusion_rules(
                        &filtered_specs,
                        &fusion_rules,
                        &ctx,
                    );

                    for spec in &result.kernels {
                        ensure_fusion_schedule_kernel_is_resolvable(
                            model.id,
                            spec.id.as_str(),
                            &shared_generated_ids,
                            &per_model_generated_ids,
                            &handwritten_kernel_ids,
                        );
                    }

                    let kernels: Vec<FusedKernelEntry> = result
                        .kernels
                        .iter()
                        .map(|spec| {
                            (
                                spec.id.as_str().to_string(),
                                kernel_phase_tag(spec.phase),
                                dispatch_kind_tag(spec.dispatch),
                            )
                        })
                        .collect();

                    let applied: Vec<String> = result
                        .applied
                        .iter()
                        .map(|rule| rule.name.to_string())
                        .collect();

                    entries.push((
                        model.id.to_string(),
                        stepping_tag,
                        has_grad_state,
                        kernel_fusion_policy_tag(policy),
                        kernels,
                        applied,
                    ));
                }
            }
        }
    }

    entries.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });

    let mut code = String::new();
    code.push_str("// @generated by build.rs\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");
    code.push_str("#[derive(Clone, Copy)]\n");
    code.push_str("pub(crate) struct FusedKernelSpecEntry {\n");
    code.push_str("    pub id: &'static str,\n");
    code.push_str("    pub phase: u8,\n");
    code.push_str("    pub dispatch: u8,\n");
    code.push_str("}\n\n");
    code.push_str("pub(crate) fn schedule_entry(\n");
    code.push_str("    model_id: &str,\n");
    code.push_str("    stepping: u8,\n");
    code.push_str("    has_grad_state: bool,\n");
    code.push_str("    policy: u8,\n");
    code.push_str(") -> Option<(&'static [FusedKernelSpecEntry], &'static [&'static str])> {\n");
    code.push_str("    match (model_id, stepping, has_grad_state, policy) {\n");

    for (model_id, stepping_tag, has_grad_state, policy_tag, kernels, applied) in &entries {
        let model_lit = rust_string_literal(model_id);
        code.push_str(&format!(
            "        ({model_lit}, {stepping_tag}u8, {has_grad_state}, {policy_tag}u8) => Some((&[\n"
        ));
        for (kernel_id, phase_tag, dispatch_tag) in kernels {
            let kernel_lit = rust_string_literal(kernel_id);
            code.push_str(&format!(
                "            FusedKernelSpecEntry {{ id: {kernel_lit}, phase: {phase_tag}u8, dispatch: {dispatch_tag}u8 }},\n"
            ));
        }
        code.push_str("        ], &[\n");
        for fusion_name in applied {
            let fusion_lit = rust_string_literal(fusion_name);
            code.push_str(&format!("            {fusion_lit},\n"));
        }
        code.push_str("        ])),\n");
    }

    code.push_str("        _ => None,\n");
    code.push_str("    }\n");
    code.push_str("}\n");

    write_if_changed(&out_path, &code);
}

fn include_kernel_in_schedule(
    condition: solver::model::kernel::KernelConditionId,
    has_grad_state: bool,
    satisfies_requires_implicit_stepping: bool,
) -> bool {
    match condition {
        solver::model::kernel::KernelConditionId::Always => true,
        solver::model::kernel::KernelConditionId::RequiresGradState => has_grad_state,
        solver::model::kernel::KernelConditionId::RequiresNoGradState => !has_grad_state,
        solver::model::kernel::KernelConditionId::RequiresImplicitStepping => {
            satisfies_requires_implicit_stepping
        }
    }
}

fn kernel_phase_tag(phase: solver::model::kernel::KernelPhaseId) -> u8 {
    match phase {
        solver::model::kernel::KernelPhaseId::Preparation => 0,
        solver::model::kernel::KernelPhaseId::Gradients => 1,
        solver::model::kernel::KernelPhaseId::FluxComputation => 2,
        solver::model::kernel::KernelPhaseId::Assembly => 3,
        solver::model::kernel::KernelPhaseId::Apply => 4,
        solver::model::kernel::KernelPhaseId::Update => 5,
    }
}

fn dispatch_kind_tag(dispatch: solver::model::kernel::DispatchKindId) -> u8 {
    match dispatch {
        solver::model::kernel::DispatchKindId::Cells => 0,
        solver::model::kernel::DispatchKindId::Faces => 1,
    }
}

fn kernel_fusion_policy_tag(policy: solver::model::kernel::KernelFusionPolicy) -> u8 {
    match policy {
        solver::model::kernel::KernelFusionPolicy::Off => 0,
        solver::model::kernel::KernelFusionPolicy::Safe => 1,
        solver::model::kernel::KernelFusionPolicy::Aggressive => 2,
    }
}

fn collect_shared_generated_kernel_ids(
    models: &[solver::model::ModelSpec],
) -> std::collections::HashSet<String> {
    let mut ids = std::collections::HashSet::<String>::new();
    for model in models {
        for module in &model.modules {
            let module: &dyn solver::model::module::ModelModule = module;
            for generator in module.kernel_generators() {
                if generator.scope == solver::model::kernel::KernelWgslScope::Shared {
                    ids.insert(generator.id.as_str().to_string());
                }
            }
        }
    }
    ids
}

fn collect_per_model_generated_kernel_ids(
    models: &[solver::model::ModelSpec],
) -> std::collections::HashMap<String, std::collections::HashSet<String>> {
    let mut out = std::collections::HashMap::<String, std::collections::HashSet<String>>::new();

    for model in models {
        let ids = out.entry(model.id.to_string()).or_default();
        for module in &model.modules {
            let module: &dyn solver::model::module::ModelModule = module;
            for generator in module.kernel_generators() {
                if generator.scope == solver::model::kernel::KernelWgslScope::PerModel {
                    ids.insert(generator.id.as_str().to_string());
                }
            }
        }
        for replacement_id in
            solver::model::kernel::derive_fusion_replacement_kernel_ids_for_model(model)
        {
            ids.insert(replacement_id.as_str().to_string());
        }
    }

    out
}

fn collect_handwritten_kernel_ids(manifest_dir: &str) -> std::collections::HashSet<String> {
    let shader_dir = PathBuf::from(manifest_dir)
        .join("src")
        .join("solver")
        .join("gpu")
        .join("shaders");
    let generated_dir = shader_dir.join("generated");

    let mut out = std::collections::HashSet::<String>::new();
    for path in list_wgsl_files_recursive(&shader_dir) {
        if path.starts_with(&generated_dir) {
            continue;
        }
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read WGSL '{}': {err}", path.display()));
        let entrypoints = parse_wgsl_compute_entrypoints(&src);
        if entrypoints.is_empty() {
            continue;
        }
        let rel_stem = wgsl_relative_stem(&shader_dir, &path);
        if entrypoints.len() == 1 && entrypoints[0] == "main" {
            out.insert(rel_stem);
            continue;
        }
        for entry in entrypoints {
            out.insert(format!("{rel_stem}/{entry}"));
        }
    }
    out
}

fn ensure_fusion_schedule_kernel_is_resolvable(
    model_id: &str,
    kernel_id: &str,
    shared_generated_ids: &std::collections::HashSet<String>,
    per_model_generated_ids: &std::collections::HashMap<String, std::collections::HashSet<String>>,
    handwritten_kernel_ids: &std::collections::HashSet<String>,
) {
    let model_has_kernel = per_model_generated_ids
        .get(model_id)
        .map(|ids| ids.contains(kernel_id))
        .unwrap_or(false);

    if model_has_kernel
        || shared_generated_ids.contains(kernel_id)
        || handwritten_kernel_ids.contains(kernel_id)
    {
        return;
    }

    panic!(
        "fusion schedule references kernel '{}' for model '{}' but no kernel registry source is available",
        kernel_id, model_id
    );
}

fn generate_named_param_registry(manifest_dir: &str) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("named_params_registry.rs");

    let dir = PathBuf::from(manifest_dir)
        .join("src")
        .join("solver")
        .join("gpu")
        .join("lowering")
        .join("named_params");

    let mut modules: Vec<(String, String)> = Vec::new();
    for entry in fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("failed to read named_params dir {}: {err}", dir.display()))
    {
        let entry = entry.unwrap_or_else(|err| panic!("failed to read named_params entry: {err}"));
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name().and_then(|s| s.to_str()) == Some("mod.rs") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("bad named_params file name: {}", path.display()));
        let ident = sanitize_rust_ident(stem);
        modules.push((stem.to_string(), ident));
    }

    modules.sort_by(|a, b| a.0.cmp(&b.0));
    modules.dedup_by(|a, b| a.0 == b.0);

    let mut code = String::new();
    code.push_str("// @generated by build.rs\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");

    for (name, ident) in &modules {
        let path = dir.join(format!("{name}.rs"));
        let lit = rust_string_literal(&path.display().to_string());
        code.push_str(&format!("#[path = {lit}]\nmod {ident};\n"));
    }

    code.push('\n');

    code.push_str("pub(crate) fn named_param_registry_exists(module: &str) -> bool {\n");
    if modules.is_empty() {
        code.push_str("    false\n");
    } else {
        code.push_str("    matches!(module, ");
        for (idx, (name, _ident)) in modules.iter().enumerate() {
            if idx > 0 {
                code.push_str(" | ");
            }
            code.push_str(&format!("\"{name}\""));
        }
        code.push_str(")\n");
    }
    code.push_str("}\n\n");

    code.push_str("pub(crate) fn named_param_handler_for_key(\n");
    code.push_str("    module: &str,\n");
    code.push_str("    key: &'static str,\n");
    code.push_str(") -> Option<crate::solver::gpu::program::plan::ProgramParamHandler> {\n");
    code.push_str("    match module {\n");
    for (name, ident) in &modules {
        code.push_str(&format!(
            "        \"{name}\" => {ident}::handler_for_key(key),\n"
        ));
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

fn rust_string_literal(raw: &str) -> String {
    let mut out = String::new();
    out.push('"');
    for ch in raw.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
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

fn list_wgsl_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let pattern = dir.join("**/*.wgsl").to_string_lossy().to_string();
    let mut out = Vec::new();
    for path in glob(&pattern)
        .unwrap_or_else(|err| {
            panic!("Failed to read glob pattern '{pattern}': {err:?}");
        })
        .flatten()
    {
        out.push(path);
    }
    out.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    out
}

fn wgsl_relative_stem(shader_dir: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(shader_dir).unwrap_or_else(|_| {
        panic!(
            "WGSL path '{}' is not under '{}'",
            path.display(),
            shader_dir.display()
        )
    });
    let rel_no_ext = rel.with_extension("");
    rel_no_ext.to_string_lossy().replace('\\', "/")
}

fn wgsl_bindings_module_path(rel_stem: &str) -> String {
    rel_stem
        .split('/')
        .filter(|s| !s.is_empty())
        .map(sanitize_rust_ident)
        .collect::<Vec<_>>()
        .join("::")
}

fn parse_wgsl_compute_entrypoints(shader: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut pending_compute = false;

    for raw in shader.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        if line.contains("@compute") {
            pending_compute = true;
        }

        if pending_compute {
            if let Some(name) = parse_fn_name(line) {
                out.push(name);
                pending_compute = false;
            }
        }
    }

    out.sort();
    out.dedup();
    out
}

fn parse_fn_name(line: &str) -> Option<String> {
    let fn_idx = line.find("fn")?;
    let after_fn = &line[fn_idx + 2..];
    if !after_fn
        .chars()
        .next()
        .map(|ch| ch.is_whitespace())
        .unwrap_or(false)
    {
        return None;
    }
    let after_fn = after_fn.trim_start();
    let mut end = 0usize;
    for ch in after_fn.chars() {
        let valid = ch.is_ascii_alphanumeric() || ch == '_';
        if !valid {
            break;
        }
        end += ch.len_utf8();
    }
    let name = after_fn[..end].trim();
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

fn fingerprint_files(paths: &[PathBuf]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    let mut sorted: Vec<&PathBuf> = paths.iter().collect();
    sorted.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));

    let mut hash = FNV_OFFSET;
    for path in sorted {
        let bytes = path.to_string_lossy();
        for b in bytes.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(FNV_PRIME);

        let data = fs::read(path).unwrap_or_else(|err| {
            panic!(
                "failed to read fingerprint input '{}': {err}",
                path.display()
            )
        });
        for b in &data {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= 0xfe;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn read_u64_hex(path: &PathBuf) -> Option<u64> {
    let raw = fs::read_to_string(path).ok()?;
    u64::from_str_radix(raw.trim(), 16).ok()
}

fn write_u64_hex(path: &PathBuf, value: u64) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, format!("{value:016x}\n")).expect("Failed to write fingerprint file");
}
