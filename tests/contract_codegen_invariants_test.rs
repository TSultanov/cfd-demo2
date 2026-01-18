use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_utf8(path: impl AsRef<Path>) -> String {
    fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read {}: {err}",
            path.as_ref().display()
        )
    })
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
    assert_contains(
        &src,
        "generated::kernel_entry_by_id",
        "kernel_registry.rs",
    );

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
    let path = repo_root().join("src/solver/gpu/lowering/models/generic_coupled.rs");
    let src = read_utf8(&path);

    // Gap 3: named-parameter handler registration must be module-owned and manifest-driven.
    // Do not reintroduce a centralized `all_named_param_handlers()` registry.
    assert_not_contains(&src, "all_named_param_handlers", "generic_coupled.rs");
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
    let entries = fs::read_dir(dir).unwrap_or_else(|err| {
        panic!("failed to read dir {}: {err}", dir.display())
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|err| {
            panic!("failed to read entry in {}: {err}", dir.display())
        });
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
    let flux_impl = flux_src
        .split("#[cfg(test)]")
        .next()
        .unwrap_or(&flux_src);
    assert_not_contains(&flux_impl, "1e-8", "flux_schemes.rs");
    assert_contains(
        &flux_impl,
        "limited_linear_face_value",
        "flux_schemes.rs",
    );

    let ua_path = repo_root().join("crates/cfd2_codegen/src/solver/codegen/reconstruction.rs");
    let ua_src = read_utf8(&ua_path);
    let ua_impl = ua_src
        .split("#[cfg(test)]")
        .next()
        .unwrap_or(&ua_src);
    assert_not_contains(&ua_impl, "1e-8", "reconstruction.rs");
    // Allow either direct `VANLEER_EPS` use or delegating to the shared reconstruction helpers.
    assert!(
        contains_ident(&ua_impl, "VANLEER_EPS")
            || ua_impl.contains("ir::reconstruction")
            || ua_impl.contains("solver::ir::reconstruction"),
        "reconstruction.rs should reference VANLEER_EPS or use shared reconstruction helpers"
    );
}
