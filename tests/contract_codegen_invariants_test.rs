use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_utf8(path: impl AsRef<Path>) -> String {
    fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.as_ref().display()))
}

fn assert_not_contains(haystack: &str, needle: &str, context: &str) {
    assert!(
        !haystack.contains(needle),
        "{context}: must not contain '{needle}'"
    );
}

fn assert_contains(haystack: &str, needle: &str, context: &str) {
    assert!(haystack.contains(needle), "{context}: missing '{needle}'");
}

fn assert_contains_ident(haystack: &str, ident: &str, context: &str) {
    assert!(
        contains_ident(haystack, ident),
        "{context}: missing identifier '{ident}'"
    );
}

#[test]
fn contract_kernel_registry_has_no_special_case_kernel_matches() {
    let path = repo_root().join("src/solver/gpu/lowering/kernel_registry.rs");
    let src = read_utf8(&path);

    // Kernel registry lookups must go through the generated `(model_id, kernel_id)` mapping.
    assert_contains(&src, "generated::kernel_entry_by_id", "kernel_registry.rs");

    // No KernelId-specific switches in the runtime registry lookup.
    assert_not_contains(&src, "match kernel_id", "kernel_registry.rs");
    assert_not_contains(&src, "KernelId::", "kernel_registry.rs");
    assert!(
        !contains_macro_invocation(&src, "include_str"),
        "kernel_registry.rs: must not contain include_str! macro invocations"
    );
}

#[test]
fn contract_generated_kernels_module_has_no_kernel_id_switches() {
    let path = repo_root().join("src/solver/gpu/modules/generated_kernels.rs");
    let src = read_utf8(&path);

    // Host bind groups/pipelines must be metadata-driven and not keyed off KernelId matches.
    assert_not_contains(&src, "match kernel_id", "generated_kernels.rs");
    assert_not_contains(&src, "KernelId::", "generated_kernels.rs");
}

#[test]
fn contract_model_driven_lowering_does_not_branch_on_model_identity() {
    let path = repo_root().join("src/solver/gpu/lowering/model_driven.rs");
    let src = read_utf8(&path);

    // Lowering selects runtime shape based on the derived recipe, not on model.id/method/eos.
    for needle in [
        "match model.id",
        "match model.method",
        "match model.eos",
        "if model.id",
        "if model.method",
        "if model.eos",
        "model.id ==",
        "model.method ==",
        "model.eos ==",
    ] {
        assert_not_contains(&src, needle, "model_driven.rs");
    }
}

#[test]
fn contract_model_driven_lowering_is_single_path_for_stepping_modes() {
    let path = repo_root().join("src/solver/gpu/lowering/model_driven.rs");
    let src = read_utf8(&path);

    // Stepping mode is already encoded in the recipe and in the program spec emitted by it.
    // Avoid duplicating stepping-mode switches in lowering glue.
    assert_not_contains(&src, "match recipe.stepping", "model_driven.rs");
    assert_not_contains(&src, "SteppingMode::", "model_driven.rs");
}

#[test]
fn contract_recipe_fusion_is_compile_time_registry_driven() {
    let path = repo_root().join("src/solver/gpu/recipe.rs");
    let src = read_utf8(&path);

    // Fusion planning must be generated at compile-time and looked up by key.
    assert_contains(
        &src,
        "fusion_schedule_registry::schedule_for_model",
        "recipe.rs",
    );

    // Runtime recipe derivation must not execute the fusion pass directly.
    assert_not_contains(&src, "apply_model_fusion_rules", "recipe.rs");
    assert_not_contains(&src, "derive_kernel_fusion_rules_for_model", "recipe.rs");
    assert_not_contains(&src, "derive_kernel_specs_for_model", "recipe.rs");
}

#[test]
fn contract_build_rs_has_no_per_kernel_prefix_discovery_glue() {
    let path = repo_root().join("build.rs");
    let src = read_utf8(&path);

    // build.rs should be driven by `derive_kernel_specs_for_model` and not accrete prefix-based
    // discovery logic for each new numerical module.
    for needle in [
        "flux_kt",
        "flux_rhie_chow",
        "rhie_chow_correct_velocity",
        "dp_update_from_diag",
    ] {
        assert_not_contains(&src, needle, "build.rs");
    }

    // Gap 4: handwritten solver-infrastructure shaders should flow through the same registry
    // mechanism as generated kernels. Avoid a separate wgsl_meta binding table.
    assert_not_contains(&src, "wgsl_binding_meta", "build.rs");
    assert_not_contains(&src, "wgsl_meta", "build.rs");
}

#[test]
fn contract_build_rs_includes_fusion_replacement_kernels_in_generated_registries() {
    let path = repo_root().join("build.rs");
    let src = read_utf8(&path);

    // build.rs must include replacement KernelIds from fusion rules so synthesized fused kernels
    // participate in generated registry maps and schedule resolvability checks.
    assert_contains(
        &src,
        "derive_fusion_replacement_kernel_ids_for_model",
        "build.rs",
    );
}

#[test]
fn contract_unified_solver_has_no_model_specific_helpers() {
    let path = repo_root().join("src/solver/gpu/unified_solver.rs");
    let src = read_utf8(&path);

    // Core solver API should not encode PDE/physics details (those belong in model helper layers).
    for needle in [
        "get_u(",
        "get_p(",
        "get_rho(",
        "set_uniform_state",
        "set_state_fields",
        "gamma=",
        "inlet_velocity",
        "ramp_time",
        "outer_correctors",
    ] {
        assert_not_contains(&src, needle, "unified_solver.rs");
    }
}

#[test]
fn contract_named_param_handlers_are_not_centralized_in_generic_coupled() {
    let path = repo_root().join("src/solver/gpu/lowering/programs/generic_coupled.rs");
    let src = read_utf8(&path);

    // Gap 3: named-parameter handler registration must be module-owned and manifest-driven.
    // Do not reintroduce a centralized `all_named_param_handlers()` registry.
    assert_not_contains(&src, "all_named_param_handlers", "generic_coupled.rs");
}

#[test]
fn contract_preconditioner_named_param_is_not_ignored() {
    let path = repo_root().join("src/solver/gpu/lowering/programs/generic_coupled.rs");
    let src = read_utf8(&path);

    // Runtime preconditioner selection should be derived from (recipe + config) and the
    // `preconditioner` named param must update the active resources (not be a no-op).
    assert_contains(
        &src,
        "RuntimePreconditionerModule::new",
        "generic_coupled.rs",
    );
    assert_contains(
        &src,
        "linear_solver.preconditioner = preconditioner",
        "generic_coupled.rs",
    );
    assert_contains(&src, "set_kind(preconditioner)", "generic_coupled.rs");
}

#[test]
fn contract_jacobi_preconditioner_is_diagonal_scaled() {
    let runtime_path = repo_root().join("src/solver/gpu/modules/runtime_preconditioner.rs");
    let runtime_src = read_utf8(&runtime_path);

    // `PreconditionerType::Jacobi` should not silently degrade to identity for generic-coupled.
    assert_contains(
        &runtime_src,
        "gmres_ops/extract_diag_inv",
        "runtime_preconditioner.rs",
    );
    assert_contains(
        &runtime_src,
        "gmres_ops/apply_diag_inv",
        "runtime_preconditioner.rs",
    );

    let shader_path = repo_root().join("src/solver/gpu/shaders/gmres_ops.wgsl");
    let shader_src = read_utf8(&shader_path);
    assert_contains(&shader_src, "fn extract_diag_inv", "gmres_ops.wgsl");
    assert_contains(&shader_src, "fn apply_diag_inv", "gmres_ops.wgsl");

    let coupled_path = repo_root().join("src/solver/gpu/lowering/programs/generic_coupled.rs");
    let coupled_src = read_utf8(&coupled_path);
    assert_contains(
        &coupled_src,
        "generic_coupled:jacobi_diag_u_inv",
        "generic_coupled.rs",
    );
}

#[test]
fn contract_block_jacobi_preconditioner_is_wired() {
    let runtime_path = repo_root().join("src/solver/gpu/modules/runtime_preconditioner.rs");
    let runtime_src = read_utf8(&runtime_path);

    assert_contains(
        &runtime_src,
        "PreconditionerType::BlockJacobi",
        "runtime_preconditioner.rs",
    );
    assert_contains(
        &runtime_src,
        "BLOCK_PRECOND_BUILD_BLOCK_INV",
        "runtime_preconditioner.rs",
    );
    assert_contains(
        &runtime_src,
        "BLOCK_PRECOND_APPLY_BLOCK_PRECOND",
        "runtime_preconditioner.rs",
    );

    let shader_path = repo_root().join("src/solver/gpu/shaders/block_precond.wgsl");
    let shader_src = read_utf8(&shader_path);
    assert_contains(
        &shader_src,
        "params.n / params.num_cells",
        "block_precond.wgsl",
    );
}

#[test]
fn contract_block_jacobi_preconditioner_falls_back_for_unsupported_block_sizes() {
    let runtime_path = repo_root().join("src/solver/gpu/modules/runtime_preconditioner.rs");
    let runtime_src = read_utf8(&runtime_path);

    assert_contains(
        &runtime_src,
        "fn block_jacobi_block_size",
        "runtime_preconditioner.rs",
    );
    assert_contains(
        &runtime_src,
        "MAX_BLOCK_JACOBI",
        "runtime_preconditioner.rs",
    );

    assert_contains(
        &runtime_src,
        "if self.block_jacobi_block_size().is_some()",
        "runtime_preconditioner.rs",
    );
    assert_contains(
        &runtime_src,
        "self.refresh_jacobi_diag_inv",
        "runtime_preconditioner.rs",
    );
    assert_contains(
        &runtime_src,
        "block_jacobi_block_size().is_none()",
        "runtime_preconditioner.rs",
    );
    assert_contains(
        &runtime_src,
        "block_jacobi_fallback_jacobi_apply",
        "runtime_preconditioner.rs",
    );
}

#[test]
fn contract_named_param_handlers_are_module_discovered() {
    let path = repo_root().join("src/solver/gpu/lowering/named_params/mod.rs");
    let src = read_utf8(&path);

    // Named-param handler registries should be discovered automatically so adding a module does
    // not require editing a centralized module-name `match` table.
    assert_contains(
        &src,
        "include!(concat!(env!(\"OUT_DIR\"), \"/named_params_registry.rs\"));",
        "named_params/mod.rs",
    );

    assert_not_contains(&src, "match module.name", "named_params/mod.rs");
    assert_not_contains(&src, "mod eos;", "named_params/mod.rs");
    assert_not_contains(&src, "mod generic_coupled;", "named_params/mod.rs");
}

#[test]
fn contract_generic_coupled_graphs_have_no_assembly_fallback() {
    let path = repo_root().join("src/solver/gpu/lowering/programs/generic_coupled.rs");
    let src = read_utf8(&path);

    // Kernel scheduling must be recipe-driven. Avoid hard-coded fallback graphs that can mask
    // missing module wiring.
    assert_not_contains(&src, "build_assembly_graph_fallback", "generic_coupled.rs");
    assert_contains(&src, "build_graph_for_phases(", "generic_coupled.rs");
}

#[test]
fn contract_generated_wgsl_does_not_assume_1d_dispatch() {
    let dir = repo_root().join("src/solver/gpu/shaders/generated");
    let entries = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("failed to read generated WGSL dir {}: {err}", dir.display()));

    for entry in entries {
        let entry =
            entry.unwrap_or_else(|err| panic!("failed to read generated WGSL entry: {err}"));
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wgsl") {
            continue;
        }

        let src = read_utf8(&path);
        let context = path.display().to_string();

        assert_not_contains(&src, "let idx = global_id.x;", &context);
        assert_not_contains(&src, "let idx=global_id.x;", &context);
        assert_not_contains(&src, "let row = global_id.x;", &context);
        assert_not_contains(&src, "let row=global_id.x;", &context);
    }
}

#[test]
fn contract_handwritten_wgsl_does_not_assume_1d_dispatch() {
    let dir = repo_root().join("src/solver/gpu/shaders");
    let entries = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("failed to read WGSL dir {}: {err}", dir.display()));

    for entry in entries {
        let entry = entry.unwrap_or_else(|err| panic!("failed to read WGSL entry: {err}"));
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wgsl") {
            continue;
        }

        let src = read_utf8(&path);
        let context = path.display().to_string();

        for needle in [
            "let idx = global_id.x;",
            "let idx=global_id.x;",
            "let row = global_id.x;",
            "let row=global_id.x;",
            "let cell = global_id.x;",
            "let cell=global_id.x;",
            "let i = global_id.x;",
            "let i=global_id.x;",
        ] {
            assert_not_contains(&src, needle, &context);
        }
    }
}

#[test]
fn contract_handwritten_wgsl_does_not_use_dispatch_x_uniform_for_indexing() {
    let dir = repo_root().join("src/solver/gpu/shaders");
    let entries = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("failed to read WGSL dir {}: {err}", dir.display()));

    for entry in entries {
        let entry = entry.unwrap_or_else(|err| panic!("failed to read WGSL entry: {err}"));
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wgsl") {
            continue;
        }

        let src = read_utf8(&path);
        let context = path.display().to_string();
        assert_not_contains(&src, "params.dispatch_x", &context);
    }
}

fn try_consume_char_literal(bytes: &[u8], start: usize) -> Option<usize> {
    debug_assert_eq!(bytes.get(start), Some(&b'\''));

    let mut i = start + 1;
    if i >= bytes.len() {
        return None;
    }

    if bytes[i] == b'\\' {
        // Escape sequence.
        i += 1;
        if i >= bytes.len() {
            return None;
        }

        match bytes[i] {
            // Unicode escape: \u{...}
            b'u' if i + 1 < bytes.len() && bytes[i + 1] == b'{' => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'}' {
                    i += 1;
                }
                if i >= bytes.len() {
                    return None;
                }
                i += 1;
            }
            // Hex byte escape: \xNN
            b'x' => {
                i += 1;
                if i + 1 >= bytes.len() {
                    return None;
                }
                i += 2;
            }
            // Simple escape: \n, \t, \\', etc.
            _ => {
                i += 1;
            }
        }
    } else {
        // A single character (consume UTF-8 sequence conservatively).
        i += 1;
        while i < bytes.len() && (bytes[i] & 0b1100_0000) == 0b1000_0000 {
            i += 1;
        }
    }

    if i < bytes.len() && bytes[i] == b'\'' {
        Some(i + 1)
    } else {
        None
    }
}

fn contains_macro_invocation(src: &str, macro_name: &str) -> bool {
    let bytes = src.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip line comments.
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Skip nested block comments.
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            let mut depth = 1usize;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                    continue;
                }
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                    continue;
                }
                i += 1;
            }
            continue;
        }

        // Skip string literals.
        if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Skip char literals (e.g. '\n', 'a') and lifetimes (e.g. 'a, 'static, '_).
        // Important: do not treat lifetimes as unterminated char literals, or we may skip
        // over macro invocations and produce false negatives.
        if bytes[i] == b'\'' {
            if let Some(next) = try_consume_char_literal(bytes, i) {
                i = next;
                continue;
            }

            // Lifetime.
            i += 1;
            if i < bytes.len() {
                let c = bytes[i];
                let is_lifetime_start = c == b'_' || (c as char).is_ascii_alphabetic();
                if is_lifetime_start {
                    i += 1;
                    while i < bytes.len() {
                        let c = bytes[i];
                        if c == b'_' || (c as char).is_ascii_alphanumeric() {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            continue;
        }

        // Skip raw strings: r"...", r#"..."#, br#"..."#
        let mut raw_start = None;
        if bytes[i] == b'r' {
            raw_start = Some(i);
        } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'r' {
            raw_start = Some(i + 1);
        } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
            // Byte string b"..."
            i += 2;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        if let Some(r_idx) = raw_start {
            let mut j = r_idx + 1;
            let mut hashes = 0usize;
            while j < bytes.len() && bytes[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'"' {
                // Consume until closing quote + hashes.
                j += 1;
                while j < bytes.len() {
                    if bytes[j] == b'"' {
                        let mut k = j + 1;
                        let mut matched = 0usize;
                        while matched < hashes && k < bytes.len() && bytes[k] == b'#' {
                            matched += 1;
                            k += 1;
                        }
                        if matched == hashes {
                            i = k;
                            break;
                        }
                    }
                    j += 1;
                }
                continue;
            }
        }

        // Identifier scan.
        let ch = bytes[i];
        let is_ident_start = ch == b'_' || (ch as char).is_ascii_alphabetic();
        if !is_ident_start {
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < bytes.len() {
            let c = bytes[i];
            if c == b'_' || (c as char).is_ascii_alphanumeric() {
                i += 1;
            } else {
                break;
            }
        }
        let ident = &src[start..i];
        if ident != macro_name {
            continue;
        }

        // Skip whitespace/comments between the identifier and '!'.
        let mut j = i;
        loop {
            while j < bytes.len() && (bytes[j] as char).is_ascii_whitespace() {
                j += 1;
            }

            if j + 1 < bytes.len() && bytes[j] == b'/' && bytes[j + 1] == b'/' {
                j += 2;
                while j < bytes.len() && bytes[j] != b'\n' {
                    j += 1;
                }
                continue;
            }

            if j + 1 < bytes.len() && bytes[j] == b'/' && bytes[j + 1] == b'*' {
                j += 2;
                let mut depth = 1usize;
                while j + 1 < bytes.len() && depth > 0 {
                    if bytes[j] == b'/' && bytes[j + 1] == b'*' {
                        depth += 1;
                        j += 2;
                        continue;
                    }
                    if bytes[j] == b'*' && bytes[j + 1] == b'/' {
                        depth -= 1;
                        j += 2;
                        continue;
                    }
                    j += 1;
                }
                continue;
            }

            break;
        }

        if j < bytes.len() && bytes[j] == b'!' {
            return true;
        }
    }

    false
}

fn contains_ident(src: &str, ident_name: &str) -> bool {
    let bytes = src.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip line comments.
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Skip nested block comments.
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            let mut depth = 1usize;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                    continue;
                }
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                    continue;
                }
                i += 1;
            }
            continue;
        }

        // Skip string literals.
        if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Skip char literals and lifetimes.
        if bytes[i] == b'\'' {
            if let Some(next) = try_consume_char_literal(bytes, i) {
                i = next;
                continue;
            }

            // Lifetime.
            i += 1;
            if i < bytes.len() {
                let c = bytes[i];
                let is_lifetime_start = c == b'_' || (c as char).is_ascii_alphabetic();
                if is_lifetime_start {
                    i += 1;
                    while i < bytes.len() {
                        let c = bytes[i];
                        if c == b'_' || (c as char).is_ascii_alphanumeric() {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            continue;
        }

        // Skip raw strings: r"...", r#"..."#, br#"..."#
        let mut raw_start = None;
        if bytes[i] == b'r' {
            raw_start = Some(i);
        } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'r' {
            raw_start = Some(i + 1);
        } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
            // Byte string b"..."
            i += 2;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        if let Some(r_idx) = raw_start {
            let mut j = r_idx + 1;
            let mut hashes = 0usize;
            while j < bytes.len() && bytes[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'"' {
                // Consume until closing quote + hashes.
                j += 1;
                while j < bytes.len() {
                    if bytes[j] == b'"' {
                        let mut k = j + 1;
                        let mut matched = 0usize;
                        while matched < hashes && k < bytes.len() && bytes[k] == b'#' {
                            matched += 1;
                            k += 1;
                        }
                        if matched == hashes {
                            i = k;
                            break;
                        }
                    }
                    j += 1;
                }
                continue;
            }
        }

        // Identifier scan.
        let ch = bytes[i];
        let is_ident_start = ch == b'_' || (ch as char).is_ascii_alphabetic();
        if !is_ident_start {
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < bytes.len() {
            let c = bytes[i];
            if c == b'_' || (c as char).is_ascii_alphanumeric() {
                i += 1;
            } else {
                break;
            }
        }
        if &src[start..i] == ident_name {
            return true;
        }
    }

    false
}

#[test]
fn contract_contains_macro_invocation_detects_include_str_despite_lifetime_syntax() {
    let src = r#"
        fn f<'a>(x: &'a str) -> &'a str {
            let _ = include_str!("ok");
            x
        }
    "#;
    assert!(contains_macro_invocation(src, "include_str"));
}

#[test]
fn contract_contains_macro_invocation_detects_include_str_with_static_lifetime() {
    let src = r#"
        fn f() -> &'static str {
            include_str!("ok")
        }
    "#;
    assert!(contains_macro_invocation(src, "include_str"));
}

fn visit_rs_files_recursive<F: FnMut(&Path)>(dir: &Path, f: &mut F) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to read dir {}: {err}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|err| panic!("failed to read entry in {}: {err}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            visit_rs_files_recursive(&path, f);
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            f(&path);
        }
    }
}

#[test]
fn contract_solver_gpu_does_not_embed_wgsl_via_include_str() {
    // Gap 4: runtime should consume registry-provided WGSL, not ad-hoc `include_str!()`.
    // This contract focuses on the solver runtime (src/solver/gpu), not UI shaders.

    let solver_gpu_dir = repo_root().join("src/solver/gpu");
    visit_rs_files_recursive(&solver_gpu_dir, &mut |path| {
        let src = read_utf8(path);
        assert!(
            !contains_macro_invocation(&src, "include_str"),
            "solver/gpu/*.rs: must not contain include_str! macro invocations"
        );
    });
}

#[test]
fn contract_solver_gpu_does_not_depend_on_wgsl_bindgen_bindings() {
    // Gap 4: infrastructure shaders should be consumed via the kernel registry + binding
    // metadata path, not by reaching into wgsl_bindgen-generated Rust bindings at runtime.

    let solver_gpu_dir = repo_root().join("src/solver/gpu");
    visit_rs_files_recursive(&solver_gpu_dir, &mut |path| {
        if path.file_name().and_then(|s| s.to_str()) == Some("bindings.rs") {
            return;
        }

        let src = read_utf8(path);
        let context = path
            .strip_prefix(repo_root())
            .unwrap_or(path)
            .display()
            .to_string();
        assert_not_contains(&src, "crate::solver::gpu::bindings", &context);
    });
}

#[test]
fn contract_solver_gpu_does_not_handwire_pipeline_layouts() {
    // Gap 4: runtime should consume bind group layouts from registry-created pipelines, not by
    // manually constructing `BindGroupLayoutDescriptor` / `PipelineLayoutDescriptor` variants.
    let solver_gpu_dir = repo_root().join("src/solver/gpu");
    visit_rs_files_recursive(&solver_gpu_dir, &mut |path| {
        if path.file_name().and_then(|s| s.to_str()) == Some("bindings.rs") {
            return;
        }
        let src = read_utf8(path);
        let context = path
            .strip_prefix(repo_root())
            .unwrap_or(path)
            .display()
            .to_string();
        assert_not_contains(&src, "create_bind_group_layout", &context);
        assert_not_contains(&src, "create_pipeline_layout", &context);
        assert_not_contains(&src, "BindGroupLayoutDescriptor", &context);
        assert_not_contains(&src, "PipelineLayoutDescriptor", &context);
    });
}

#[test]
fn contract_recipe_does_not_inject_generic_coupled_apply_kernel_by_id() {
    // Apply-kernel presence in implicit stepping should come from composing a module into the
    // derived recipe, not from hard-coded `KernelId::GENERIC_COUPLED_APPLY` insertion.
    let path = repo_root().join("src/solver/gpu/recipe.rs");
    let src = read_utf8(&path);
    assert_not_contains(&src, "KernelId::GENERIC_COUPLED_APPLY", "recipe.rs");
}

#[test]
fn contract_recipe_low_mach_buffer_is_manifest_driven() {
    // Low-Mach buffer allocation should follow model-declared module manifests (named params),
    // not hard-coded EOS-variant matches inside the recipe.
    let path = repo_root().join("src/solver/gpu/recipe.rs");
    let src = read_utf8(&path);

    assert_contains(&src, "low_mach.", "recipe.rs");
    assert_not_contains(&src, "EosSpec::IdealGas", "recipe.rs");
    assert_not_contains(&src, "EosSpec::LinearCompressibility", "recipe.rs");
}

#[test]
fn contract_iteration_snapshot_stage_is_binding_driven() {
    // Implicit stepping should only include the snapshot stage when kernels actually bind `state_iter`.
    let path = repo_root().join("src/solver/gpu/recipe.rs");
    let src = read_utf8(&path);

    assert_contains_ident(&src, "requires_iteration_snapshot", "recipe.rs");
    assert_contains(&src, "if self.requires_iteration_snapshot", "recipe.rs");
    assert_contains(&src, "implicit:snapshot", "recipe.rs");
}

#[test]
fn contract_iteration_snapshot_allocation_is_recipe_driven() {
    let path = repo_root().join("src/solver/gpu/modules/unified_field_resources.rs");
    let src = read_utf8(&path);

    assert_contains(
        &src,
        "iteration_snapshot = if recipe.requires_iteration_snapshot",
        "unified_field_resources.rs",
    );
}

#[test]
fn contract_flux_buffer_allocation_is_binding_driven() {
    // Flux buffer allocation should follow kernel binding metadata (needs `fluxes`), not ad-hoc
    // model/module heuristics inside the recipe.
    let path = repo_root().join("src/solver/gpu/recipe.rs");
    let src = read_utf8(&path);

    assert_contains(&src, "\"fluxes\"", "recipe.rs");
    assert_contains(&src, "let flux = if binds_fluxes", "recipe.rs");
    assert_not_contains(&src, "flux_module(", "recipe.rs");
}

#[test]
fn contract_solver_gpu_has_no_low_mach_binding_aliases() {
    // ResourceRegistry/FieldProvider should resolve binding names exactly as reflected from WGSL;
    // avoid solver-side aliasing that hides mismatches between generated bindings and runtime.
    let registry_path = repo_root().join("src/solver/gpu/modules/resource_registry.rs");
    let registry_src = read_utf8(&registry_path);
    assert_not_contains(
        &registry_src,
        "if name == \"low_mach\"",
        "resource_registry.rs",
    );

    let provider_path = repo_root().join("src/solver/gpu/modules/field_provider.rs");
    let provider_src = read_utf8(&provider_path);
    assert_not_contains(&provider_src, "\"low_mach\" |", "field_provider.rs");
}

#[test]
fn contract_low_mach_named_params_are_consumed_by_generated_kernels() {
    // Low-Mach params are part of the model-declared named-parameter interface; ensure at least
    // one generated kernel consumes the corresponding uniform buffer so the knob is not a no-op.
    let shader_path =
        repo_root().join("src/solver/gpu/shaders/generated/flux_module_compressible.wgsl");
    let shader_src = read_utf8(&shader_path);

    assert_contains(
        &shader_src,
        "struct LowMachParams",
        "flux_module_compressible.wgsl",
    );
    assert_contains(
        &shader_src,
        "var<uniform> low_mach_params",
        "flux_module_compressible.wgsl",
    );
}

#[test]
fn contract_compressible_flux_module_reflects_velocity_on_walls() {
    // Compressible slip-wall boundaries must enforce no-penetration in the convective flux.
    // The generated flux module should reflect the normal component for SlipWall boundary types
    // when reading the neighbor primitive velocity (`u`).
    let shader_path =
        repo_root().join("src/solver/gpu/shaders/generated/flux_module_compressible.wgsl");
    let shader_src = read_utf8(&shader_path);

    assert_contains(
        &shader_src,
        "boundary_type == 4u",
        "flux_module_compressible.wgsl",
    );
}

#[test]
fn contract_reconstruction_paths_share_vanleer_eps_constant() {
    // Drift guard: ensure unified_assembly and flux-module reconstruction use the same shared
    // epsilon constant (no duplicated numeric literals in separate implementations).

    let ir_path = repo_root().join("crates/cfd2_ir/src/solver/ir/mod.rs");
    let ir_src = read_utf8(&ir_path);
    assert_contains_ident(&ir_src, "VANLEER_EPS", "cfd2_ir::ir/mod.rs");

    let recon_path = repo_root().join("crates/cfd2_ir/src/solver/ir/reconstruction.rs");
    let recon_src = read_utf8(&recon_path);
    assert_contains_ident(&recon_src, "VANLEER_EPS", "cfd2_ir::ir/reconstruction.rs");
    assert_not_contains(&recon_src, "1e-8", "cfd2_ir::ir/reconstruction.rs");

    let flux_path = repo_root().join("src/solver/model/flux_schemes.rs");
    let flux_src = read_utf8(&flux_path);
    let flux_impl = flux_src.split("#[cfg(test)]").next().unwrap_or(&flux_src);
    assert_not_contains(flux_impl, "1e-8", "flux_schemes.rs");
    assert_contains(flux_impl, "limited_linear_face_value", "flux_schemes.rs");

    let ua_path = repo_root().join("crates/cfd2_codegen/src/solver/codegen/reconstruction.rs");
    let ua_src = read_utf8(&ua_path);
    let ua_impl = ua_src.split("#[cfg(test)]").next().unwrap_or(&ua_src);
    assert_not_contains(ua_impl, "1e-8", "reconstruction.rs");
    assert_contains(ua_impl, "limited_linear_face_value", "reconstruction.rs");
}
