use super::KernelWgsl;
use cfd2_ir::solver::ir::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelProgram, SideEffectMetadata,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionPatternRule {
    pub name: String,
    pub priority: i32,
    pub pattern: Vec<String>,
    pub replacement_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    pub rule_name: String,
    pub replacement_id: String,
    pub start_index: usize,
    pub pattern_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionSafetyPolicy {
    Safe,
    Aggressive,
}

pub fn match_fusion_candidates(
    ordered_kernel_ids: &[&str],
    rules: &[FusionPatternRule],
) -> Vec<FusionCandidate> {
    if ordered_kernel_ids.is_empty() || rules.is_empty() {
        return Vec::new();
    }

    let mut ranked_rules: Vec<(usize, &FusionPatternRule)> = rules
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.pattern.is_empty())
        .collect();
    ranked_rules.sort_by(|(ia, a), (ib, b)| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| b.pattern.len().cmp(&a.pattern.len()))
            .then_with(|| ia.cmp(ib))
    });

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < ordered_kernel_ids.len() {
        let mut matched: Option<&FusionPatternRule> = None;
        for (_, rule) in &ranked_rules {
            if pattern_matches_at(ordered_kernel_ids, &rule.pattern, i) {
                matched = Some(rule);
                break;
            }
        }

        if let Some(rule) = matched {
            out.push(FusionCandidate {
                rule_name: rule.name.clone(),
                replacement_id: rule.replacement_id.clone(),
                start_index: i,
                pattern_len: rule.pattern.len(),
            });
            i += rule.pattern.len();
        } else {
            i += 1;
        }
    }

    out
}

fn pattern_matches_at(ordered_kernel_ids: &[&str], pattern: &[String], start: usize) -> bool {
    if start + pattern.len() > ordered_kernel_ids.len() {
        return false;
    }
    for (offset, id) in pattern.iter().enumerate() {
        if ordered_kernel_ids[start + offset] != id {
            return false;
        }
    }
    true
}

pub fn synthesize_fused_program(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
) -> Result<KernelProgram, String> {
    if programs.is_empty() {
        return Err("fusion synthesis requires at least one input kernel".to_string());
    }
    ensure_safe_composition(programs, policy)?;

    let dispatch = programs[0].dispatch.clone();
    let launch = programs[0].launch.clone();
    let merged_bindings = merge_bindings(programs)?;

    let mut preamble = Vec::new();
    let mut body = Vec::new();
    let mut local_symbols = Vec::new();
    let mut side_effects = SideEffectMetadata::default();

    for (idx, program) in programs.iter().enumerate() {
        let rename_map = deterministic_symbol_rename_map(idx, &program.local_symbols);

        preamble.push(format!("// begin fused segment: {}", program.id));
        preamble.extend(rename_lines(&program.preamble, &rename_map));

        body.extend(rename_lines(&program.body, &rename_map));
        body.push(format!("// end fused segment: {}", program.id));

        local_symbols.extend(rename_symbols(&program.local_symbols, &rename_map));
        side_effects.read_set.extend(program.side_effects.read_set.clone());
        side_effects
            .write_set
            .extend(program.side_effects.write_set.clone());
        side_effects.uses_barriers |= program.side_effects.uses_barriers;
        side_effects.uses_atomics |= program.side_effects.uses_atomics;
    }

    let mut fused = KernelProgram::new(
        replacement_id.into(),
        dispatch,
        launch,
        merged_bindings.into_values().collect(),
    );
    fused.indexing = programs[0].indexing.clone();
    fused.preamble = preamble;
    fused.body = body;
    fused.local_symbols = local_symbols;
    fused.side_effects = side_effects;

    if policy == FusionSafetyPolicy::Aggressive {
        apply_aggressive_cleanup(&mut fused);
    }

    // Attach an explicit synthesis marker as a deterministic first preamble line.
    fused
        .preamble
        .insert(0, format!("// synthesized by fusion rule: {rule_name}"));

    Ok(fused)
}

fn ensure_safe_composition(
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
) -> Result<(), String> {
    let first = &programs[0];
    for program in programs.iter().skip(1) {
        if program.dispatch != first.dispatch {
            return Err(format!(
                "fusion rejected: dispatch mismatch ('{}' vs '{}')",
                dispatch_label(&first.dispatch),
                dispatch_label(&program.dispatch),
            ));
        }
        if program.launch != first.launch {
            return Err(format!(
                "fusion rejected: launch semantics mismatch between '{}' and '{}'",
                first.id, program.id
            ));
        }
        if program.indexing != first.indexing {
            return Err(format!(
                "fusion rejected: indexing mismatch between '{}' and '{}'",
                first.id, program.id
            ));
        }
    }

    if policy == FusionSafetyPolicy::Safe {
        if programs
            .iter()
            .any(|p| p.side_effects.uses_barriers || p.side_effects.uses_atomics)
        {
            return Err(
                "fusion rejected: barriers/atomics require aggressive policy and dedicated transforms"
                    .to_string(),
            );
        }

        let mut prior_reads = BTreeSet::<EffectResource>::new();
        let mut prior_writes = BTreeSet::<EffectResource>::new();
        for program in programs {
            let read = &program.side_effects.read_set;
            let write = &program.side_effects.write_set;

            if intersects(&prior_writes, read) {
                return Err(format!("fusion rejected: RAW hazard at kernel '{}'", program.id));
            }
            if intersects(&prior_reads, write) {
                return Err(format!("fusion rejected: WAR hazard at kernel '{}'", program.id));
            }
            if intersects(&prior_writes, write) {
                return Err(format!("fusion rejected: WAW hazard at kernel '{}'", program.id));
            }

            prior_reads.extend(read.iter().cloned());
            prior_writes.extend(write.iter().cloned());
        }
    }

    Ok(())
}

fn dispatch_label(dispatch: &DispatchDomain) -> String {
    match dispatch {
        DispatchDomain::Cells => "cells".to_string(),
        DispatchDomain::Faces => "faces".to_string(),
        DispatchDomain::Custom(name) => format!("custom:{name}"),
    }
}

fn intersects(left: &BTreeSet<EffectResource>, right: &BTreeSet<EffectResource>) -> bool {
    if left.len() < right.len() {
        left.iter().any(|r| right.contains(r))
    } else {
        right.iter().any(|r| left.contains(r))
    }
}

fn merge_bindings(programs: &[KernelProgram]) -> Result<BTreeMap<(u32, u32), KernelBinding>, String> {
    let mut merged = BTreeMap::<(u32, u32), KernelBinding>::new();
    for program in programs {
        for binding in &program.bindings {
            let key = (binding.group, binding.binding);
            if let Some(prev) = merged.get(&key) {
                if prev.name != binding.name
                    || prev.wgsl_type != binding.wgsl_type
                    || prev.access != binding.access
                {
                    return Err(format!(
                        "incompatible bind interface at @group({}) @binding({}) while fusing '{}': '{}'/'{}' vs '{}'/'{}'",
                        binding.group,
                        binding.binding,
                        program.id,
                        prev.name,
                        prev.wgsl_type,
                        binding.name,
                        binding.wgsl_type,
                    ));
                }
            } else {
                merged.insert(key, binding.clone());
            }
        }
    }
    Ok(merged)
}

fn deterministic_symbol_rename_map(
    program_index: usize,
    local_symbols: &[String],
) -> BTreeMap<String, String> {
    if program_index == 0 {
        return BTreeMap::new();
    }

    let mut symbols: Vec<String> = local_symbols.to_vec();
    symbols.sort();
    symbols.dedup();

    let mut out = BTreeMap::new();
    for symbol in symbols {
        out.insert(symbol.clone(), format!("k{program_index}_{symbol}"));
    }
    out
}

fn rename_symbols(symbols: &[String], rename_map: &BTreeMap<String, String>) -> Vec<String> {
    symbols
        .iter()
        .map(|s| rename_map.get(s).cloned().unwrap_or_else(|| s.clone()))
        .collect()
}

fn rename_lines(lines: &[String], rename_map: &BTreeMap<String, String>) -> Vec<String> {
    lines
        .iter()
        .map(|line| {
            rename_map
                .iter()
                .fold(line.clone(), |acc, (old, new)| rename_identifier(&acc, old, new))
        })
        .collect()
}

fn rename_identifier(src: &str, old: &str, new: &str) -> String {
    if old.is_empty() || old == new {
        return src.to_string();
    }
    let src_bytes = src.as_bytes();
    let old_bytes = old.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < src_bytes.len() {
        let end = i + old_bytes.len();
        if end <= src_bytes.len()
            && &src_bytes[i..end] == old_bytes
            && is_ident_boundary(src_bytes, i, end)
        {
            out.push_str(new);
            i = end;
            continue;
        }
        out.push(src_bytes[i] as char);
        i += 1;
    }
    out
}

fn is_ident_boundary(src: &[u8], start: usize, end: usize) -> bool {
    let left_ok = if start == 0 {
        true
    } else {
        !is_ident_char(src[start - 1])
    };
    let right_ok = if end == src.len() {
        true
    } else {
        !is_ident_char(src[end])
    };
    left_ok && right_ok
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn apply_aggressive_cleanup(program: &mut KernelProgram) {
    program.body.retain(|line| !is_noop_local_self_assignment(line));
}

fn is_noop_local_self_assignment(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.ends_with(';') {
        return false;
    }
    let stmt = trimmed.trim_end_matches(';').trim();
    let Some((lhs, rhs)) = stmt.split_once('=') else {
        return false;
    };
    let lhs = lhs.trim();
    let rhs = rhs.trim();
    lhs == rhs && is_identifier(lhs)
}

fn is_identifier(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

pub fn lower_kernel_program_to_wgsl(program: &KernelProgram) -> Result<KernelWgsl, String> {
    let mut lines = Vec::<String>::new();
    lines.push(format!(
        "// GENERATED BY CFD2 DSL FUSION ({})",
        program.id
    ));
    lines.push("// DO NOT EDIT MANUALLY".to_string());
    lines.push(String::new());

    let mut sorted_bindings = program.bindings.clone();
    sorted_bindings.sort_by(|a, b| {
        a.group
            .cmp(&b.group)
            .then(a.binding.cmp(&b.binding))
            .then(a.name.cmp(&b.name))
    });

    // Reject duplicate bind slots with incompatible definitions up front.
    let mut by_slot = BTreeMap::<(u32, u32), &KernelBinding>::new();
    for binding in &sorted_bindings {
        let key = (binding.group, binding.binding);
        if let Some(prev) = by_slot.get(&key) {
            if prev.name != binding.name
                || prev.wgsl_type != binding.wgsl_type
                || prev.access != binding.access
            {
                return Err(format!(
                    "duplicate bind slot mismatch at @group({}) @binding({})",
                    binding.group, binding.binding
                ));
            }
            continue;
        }
        by_slot.insert(key, binding);
    }

    // Most kernel DSL programs in this repo use the shared runtime `Constants` uniform.
    // Emit the canonical struct definition so lowered WGSL is self-contained.
    let needs_constants_struct = by_slot.values().any(|binding| binding.wgsl_type == "Constants");
    if needs_constants_struct {
        let mut constants_module = super::wgsl_ast::Module::new();
        constants_module.push(super::wgsl_ast::Item::Struct(
            super::constants::constants_struct(&[]),
        ));
        lines.push(constants_module.to_wgsl());
        lines.push(String::new());
    }

    for binding in by_slot.values() {
        let decl = match binding.access {
            BindingAccess::ReadOnlyStorage => {
                format!(
                    "@group({}) @binding({}) var<storage, read> {}: {};",
                    binding.group, binding.binding, binding.name, binding.wgsl_type
                )
            }
            BindingAccess::ReadWriteStorage => {
                format!(
                    "@group({}) @binding({}) var<storage, read_write> {}: {};",
                    binding.group, binding.binding, binding.name, binding.wgsl_type
                )
            }
            BindingAccess::Uniform => {
                format!(
                    "@group({}) @binding({}) var<uniform> {}: {};",
                    binding.group, binding.binding, binding.name, binding.wgsl_type
                )
            }
        };
        lines.push(decl);
    }

    lines.push(String::new());
    lines.push(format!(
        "@compute @workgroup_size({}, {}, {})",
        program.launch.workgroup_size[0], program.launch.workgroup_size[1], program.launch.workgroup_size[2]
    ));
    lines.push("fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {".to_string());
    lines.push(format!(
        "    let idx = {};",
        program.launch.invocation_index_expr
    ));
    if let Some(check) = &program.launch.bounds_check_expr {
        lines.push(format!("    if ({check}) {{ return; }}"));
    }

    for line in &program.indexing {
        lines.push(format!("    {line}"));
    }
    for line in &program.preamble {
        lines.push(format!("    {line}"));
    }
    for line in &program.body {
        lines.push(format!("    {line}"));
    }

    lines.push("}".to_string());
    lines.push(String::new());

    Ok(KernelWgsl::from_source(lines.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd2_ir::solver::ir::{DispatchDomain, LaunchSemantics};

    fn sample_program(id: &str) -> KernelProgram {
        let launch = LaunchSemantics::new(
            [64, 1, 1],
            "global_id.y * constants.stride_x + global_id.x",
            Some("idx >= num_cells"),
        );
        let mut program = KernelProgram::new(
            id,
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
            ],
        );
        program.preamble = vec!["var value: f32 = 0.0;".to_string()];
        program.body = vec![
            "value = value + 1.0;".to_string(),
            "state[idx] = value;".to_string(),
        ];
        program.local_symbols = vec!["value".to_string()];
        program
    }

    #[test]
    fn matcher_prefers_higher_priority_longer_pattern() {
        let ordered = ["a", "b", "c"];
        let rules = vec![
            FusionPatternRule {
                name: "ab".to_string(),
                priority: 10,
                pattern: vec!["a".to_string(), "b".to_string()],
                replacement_id: "ab_fused".to_string(),
            },
            FusionPatternRule {
                name: "abc".to_string(),
                priority: 10,
                pattern: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                replacement_id: "abc_fused".to_string(),
            },
        ];
        let out = match_fusion_candidates(&ordered, &rules);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].rule_name, "abc");
    }

    #[test]
    fn safe_policy_rejects_hazards() {
        let mut a = sample_program("a");
        a.side_effects.write_set.insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects.read_set.insert(EffectResource::binding(0, 0));

        let err = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Safe,
        )
        .unwrap_err();
        assert!(err.contains("RAW hazard"), "unexpected error: {err}");
    }

    #[test]
    fn merge_bindings_rejects_incompatible_interface() {
        let a = sample_program("a");
        let mut b = sample_program("b");
        b.bindings[0].name = "state_other".to_string();
        let err = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
        )
        .unwrap_err();
        assert!(
            err.contains("incompatible bind interface"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn synthesis_renames_symbols_deterministically() {
        let a = sample_program("a");
        let b = sample_program("b");
        let fused = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
        )
        .expect("fused synthesis");
        let joined = fused.body.join("\n");
        assert!(joined.contains("value = value + 1.0;"));
        assert!(joined.contains("k1_value = k1_value + 1.0;"));
    }

    #[test]
    fn lowering_to_wgsl_is_deterministic() {
        let a = sample_program("a");
        let b = sample_program("b");
        let fused = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
        )
        .expect("fused synthesis");

        let wgsl1 = lower_kernel_program_to_wgsl(&fused).expect("wgsl");
        let wgsl2 = lower_kernel_program_to_wgsl(&fused).expect("wgsl");
        assert_eq!(wgsl1.to_wgsl(), wgsl2.to_wgsl());
        assert!(wgsl1.to_wgsl().contains("@compute @workgroup_size(64, 1, 1)"));
        assert!(wgsl1.to_wgsl().contains("struct Constants"));
    }

    #[test]
    fn aggressive_cleanup_removes_noop_local_self_assignment() {
        let mut a = sample_program("a");
        a.body.insert(0, "value = value;".to_string());
        let b = sample_program("b");

        let safe = synthesize_fused_program("fused", "rule/a_b", &[a.clone(), b.clone()], FusionSafetyPolicy::Safe)
            .expect("safe fused synthesis");
        let aggressive = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
        )
        .expect("aggressive fused synthesis");

        assert!(
            safe.body.iter().any(|line| line.trim() == "value = value;"),
            "safe policy should preserve no-op local assignment"
        );
        assert!(
            aggressive
                .body
                .iter()
                .all(|line| line.trim() != "value = value;"),
            "aggressive policy should remove no-op local assignment"
        );
    }

    #[test]
    fn aggressive_differs_from_safe_only_when_cleanup_applies() {
        let a = sample_program("a");
        let b = sample_program("b");

        let safe = synthesize_fused_program("fused", "rule/a_b", &[a.clone(), b.clone()], FusionSafetyPolicy::Safe)
            .expect("safe fused synthesis");
        let aggressive = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
        )
        .expect("aggressive fused synthesis");

        assert_eq!(
            safe.preamble, aggressive.preamble,
            "aggressive cleanup should not mutate preamble when no cleanup candidates exist"
        );
        assert_eq!(
            safe.body, aggressive.body,
            "aggressive cleanup should match safe output when no cleanup candidates exist"
        );
    }
}
