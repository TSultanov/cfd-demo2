use super::KernelWgsl;
use cfd2_ir::solver::dimensions::{Dimensionless, UnitDimension};
use cfd2_ir::solver::ir::ports::ParamSpec;
use cfd2_ir::solver::ir::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelProgram, SideEffectMetadata,
};
use std::collections::{BTreeMap, BTreeSet};

/// A single binding slot remap for fusion synthesis.
///
/// When two kernels being fused have the same logical buffer (e.g. `bc_kind`)
/// at different `(group, binding)` slots, one kernel's bindings must be
/// remapped to match the other's layout before `merge_bindings` can succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingRemap {
    /// Index of the program in the fusion input sequence (0-based).
    pub program_index: usize,
    /// Original slot in this program's `KernelProgram.bindings`.
    pub from_group: u32,
    pub from_binding: u32,
    /// Target slot the binding should be moved to in the fused layout.
    pub to_group: u32,
    pub to_binding: u32,
}

/// Test-only pattern-rule type for matcher unit tests.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionPatternRule {
    pub name: String,
    pub priority: i32,
    pub pattern: Vec<String>,
    pub replacement_id: String,
}

/// Test-only matcher output type for matcher unit tests.
#[cfg(test)]
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

/// Classification of a data hazard detected during fusion safety analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardKind {
    /// Read-After-Write: a later kernel reads a resource written by an earlier kernel.
    RAW,
    /// Write-After-Read: a later kernel writes a resource read by an earlier kernel.
    WAR,
    /// Write-After-Write: a later kernel writes a resource also written by an earlier kernel.
    WAW,
    /// A kernel uses barriers or atomics, requiring dedicated transforms.
    BarriersOrAtomics,
}

impl std::fmt::Display for HazardKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HazardKind::RAW => write!(f, "RAW"),
            HazardKind::WAR => write!(f, "WAR"),
            HazardKind::WAW => write!(f, "WAW"),
            HazardKind::BarriersOrAtomics => write!(f, "barriers/atomics"),
        }
    }
}

/// A single hazard detected during fusion composition analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HazardReport {
    pub kind: HazardKind,
    pub kernel_id: String,
    pub resources: Vec<EffectResource>,
}

impl std::fmt::Display for HazardReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} hazard at kernel '{}'", self.kind, self.kernel_id)?;
        if !self.resources.is_empty() {
            write!(f, " (resources: ")?;
            for (i, r) in self.resources.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "@group({}) @binding({})", r.group, r.binding)?;
                if let Some(ref c) = r.component {
                    write!(f, ":{c}")?;
                }
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

#[cfg(test)]
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

#[cfg(test)]
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
    synthesize_fused_program_remapped(replacement_id, rule_name, programs, policy, &[])
}

/// Like [`synthesize_fused_program`], but applies binding slot remaps before
/// merging. Use this when the kernels being fused have conflicting bind group
/// layouts that must be reconciled (e.g. a gradients kernel using group 2 for
/// boundary conditions vs an assembly kernel using group 3).
pub fn synthesize_fused_program_remapped(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
    binding_remaps: &[BindingRemap],
) -> Result<KernelProgram, String> {
    let (program, _hazards) = synthesize_fused_program_with_report_remapped(
        replacement_id,
        rule_name,
        programs,
        policy,
        binding_remaps,
    )?;
    Ok(program)
}

/// Like [`synthesize_fused_program`], but also returns any hazard reports detected
/// during composition analysis. Under `Safe` policy, hazards cause rejection (Err).
/// Under `Aggressive` policy, hazards are collected and returned alongside the
/// successfully fused program so callers have visibility.
pub fn synthesize_fused_program_with_report(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
) -> Result<(KernelProgram, Vec<HazardReport>), String> {
    synthesize_fused_program_with_report_remapped(replacement_id, rule_name, programs, policy, &[])
}

/// Like [`synthesize_fused_program_with_report`], but applies binding slot
/// remaps before merging.
pub fn synthesize_fused_program_with_report_remapped(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
    binding_remaps: &[BindingRemap],
) -> Result<(KernelProgram, Vec<HazardReport>), String> {
    if programs.is_empty() {
        return Err("fusion synthesis requires at least one input kernel".to_string());
    }

    // Apply binding remaps to produce adjusted programs for merging.
    let remapped: Vec<KernelProgram> = if binding_remaps.is_empty() {
        programs.to_vec()
    } else {
        apply_binding_remaps(programs, binding_remaps)?
    };

    let hazards = ensure_safe_composition(&remapped, policy)?;

    let dispatch = remapped[0].dispatch.clone();
    let launch = remapped[0].launch.clone();
    let merged_bindings = merge_bindings(&remapped)?;

    let mut body = Vec::new();
    let mut local_symbols = Vec::new();
    let mut side_effects = SideEffectMetadata::default();
    let mut helper_functions = Vec::<String>::new();

    for (idx, program) in remapped.iter().enumerate() {
        let rename_map = deterministic_symbol_rename_map(idx, &program.local_symbols);

        // Merge helper functions (deduplicate by content).
        for helper in &program.helper_functions {
            if !helper_functions.contains(helper) {
                helper_functions.push(helper.clone());
            }
        }

        // Preserve per-kernel execution order by emitting each segment's preamble
        // immediately before that same segment's body.
        body.push(format!("// begin fused segment: {}", program.id));
        body.extend(rename_lines(&program.preamble, &rename_map));
        body.extend(rename_lines(&program.body, &rename_map));
        body.push(format!("// end fused segment: {}", program.id));

        local_symbols.extend(rename_symbols(&program.local_symbols, &rename_map));
        side_effects
            .read_set
            .extend(program.side_effects.read_set.clone());
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
    fused.helper_functions = helper_functions;
    fused.indexing = remapped[0].indexing.clone();
    fused.preamble = Vec::new();
    fused.body = body;
    fused.local_symbols = local_symbols;
    fused.side_effects = side_effects;
    fused.eos_params = merge_eos_params(&remapped);

    if policy == FusionSafetyPolicy::Aggressive {
        apply_aggressive_cleanup(&mut fused);
    }

    // Attach an explicit synthesis marker as a deterministic first preamble line.
    fused
        .preamble
        .insert(0, format!("// synthesized by fusion rule: {rule_name}"));

    Ok((fused, hazards))
}

/// Detect all data hazards across a sequence of programs without rejecting.
///
/// Returns a list of [`HazardReport`] entries describing every RAW, WAR, WAW,
/// and barrier/atomics hazard found in the program sequence. An empty list means
/// the composition is safe.
pub fn detect_hazards(programs: &[KernelProgram]) -> Vec<HazardReport> {
    let mut reports = Vec::new();

    for program in programs {
        if program.side_effects.uses_barriers || program.side_effects.uses_atomics {
            reports.push(HazardReport {
                kind: HazardKind::BarriersOrAtomics,
                kernel_id: program.id.clone(),
                resources: Vec::new(),
            });
        }
    }

    let mut prior_reads = BTreeSet::<EffectResource>::new();
    let mut prior_writes = BTreeSet::<EffectResource>::new();
    for program in programs {
        let read = &program.side_effects.read_set;
        let write = &program.side_effects.write_set;

        let raw_resources = intersection_list(&prior_writes, read);
        if !raw_resources.is_empty() {
            reports.push(HazardReport {
                kind: HazardKind::RAW,
                kernel_id: program.id.clone(),
                resources: raw_resources,
            });
        }

        let war_resources = intersection_list(&prior_reads, write);
        if !war_resources.is_empty() {
            reports.push(HazardReport {
                kind: HazardKind::WAR,
                kernel_id: program.id.clone(),
                resources: war_resources,
            });
        }

        let waw_resources = intersection_list(&prior_writes, write);
        if !waw_resources.is_empty() {
            reports.push(HazardReport {
                kind: HazardKind::WAW,
                kernel_id: program.id.clone(),
                resources: waw_resources,
            });
        }

        prior_reads.extend(read.iter().cloned());
        prior_writes.extend(write.iter().cloned());
    }

    reports
}

fn ensure_safe_composition(
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
) -> Result<Vec<HazardReport>, String> {
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

    let hazards = detect_hazards(programs);

    if policy == FusionSafetyPolicy::Safe {
        // Under Safe policy, any hazard is a hard rejection.
        if let Some(h) = hazards.first() {
            return Err(format!(
                "fusion rejected: {} hazard at kernel '{}'",
                h.kind, h.kernel_id
            ));
        }
    }

    // Under Aggressive policy, hazards are collected but do not block synthesis.
    Ok(hazards)
}

fn dispatch_label(dispatch: &DispatchDomain) -> String {
    match dispatch {
        DispatchDomain::Cells => "cells".to_string(),
        DispatchDomain::Faces => "faces".to_string(),
        DispatchDomain::Custom(name) => format!("custom:{name}"),
    }
}

fn intersection_list(
    left: &BTreeSet<EffectResource>,
    right: &BTreeSet<EffectResource>,
) -> Vec<EffectResource> {
    if left.len() < right.len() {
        left.iter().filter(|r| right.contains(r)).cloned().collect()
    } else {
        right.iter().filter(|r| left.contains(r)).cloned().collect()
    }
}

/// Apply binding slot remaps to a sequence of programs before fusion merging.
///
/// Each `BindingRemap` entry specifies that program N's binding at
/// `(from_group, from_binding)` should be moved to `(to_group, to_binding)`.
/// The corresponding `side_effects` read/write sets are updated as well.
fn apply_binding_remaps(
    programs: &[KernelProgram],
    remaps: &[BindingRemap],
) -> Result<Vec<KernelProgram>, String> {
    let mut result: Vec<KernelProgram> = programs.to_vec();

    for remap in remaps {
        if remap.program_index >= result.len() {
            return Err(format!(
                "binding remap references program index {} but only {} programs provided",
                remap.program_index,
                result.len()
            ));
        }

        let program = &mut result[remap.program_index];

        // Remap binding slots.
        let mut found = false;
        for binding in &mut program.bindings {
            if binding.group == remap.from_group && binding.binding == remap.from_binding {
                binding.group = remap.to_group;
                binding.binding = remap.to_binding;
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!(
                "binding remap: program '{}' has no binding at @group({}) @binding({})",
                program.id, remap.from_group, remap.from_binding
            ));
        }

        // Remap side-effect metadata.
        let from_res = EffectResource::binding(remap.from_group, remap.from_binding);
        let to_res = EffectResource::binding(remap.to_group, remap.to_binding);
        if program.side_effects.read_set.remove(&from_res) {
            program.side_effects.read_set.insert(to_res.clone());
        }
        if program.side_effects.write_set.remove(&from_res) {
            program.side_effects.write_set.insert(to_res);
        }

        // Also remap any component-level side-effects at the same slot.
        let read_to_remap: Vec<_> = program
            .side_effects
            .read_set
            .iter()
            .filter(|r| {
                r.group == remap.from_group
                    && r.binding == remap.from_binding
                    && r.component.is_some()
            })
            .cloned()
            .collect();
        for r in read_to_remap {
            program.side_effects.read_set.remove(&r);
            program.side_effects.read_set.insert(EffectResource {
                group: remap.to_group,
                binding: remap.to_binding,
                component: r.component,
            });
        }

        let write_to_remap: Vec<_> = program
            .side_effects
            .write_set
            .iter()
            .filter(|r| {
                r.group == remap.from_group
                    && r.binding == remap.from_binding
                    && r.component.is_some()
            })
            .cloned()
            .collect();
        for r in write_to_remap {
            program.side_effects.write_set.remove(&r);
            program.side_effects.write_set.insert(EffectResource {
                group: remap.to_group,
                binding: remap.to_binding,
                component: r.component,
            });
        }
    }

    Ok(result)
}

fn merge_bindings(
    programs: &[KernelProgram],
) -> Result<BTreeMap<(u32, u32), KernelBinding>, String> {
    let mut merged = BTreeMap::<(u32, u32), KernelBinding>::new();
    for program in programs {
        for binding in &program.bindings {
            let key = (binding.group, binding.binding);
            if let Some(prev) = merged.get_mut(&key) {
                if prev.name != binding.name || prev.wgsl_type != binding.wgsl_type {
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
                // Promote to most permissive access mode:
                // ReadOnlyStorage + ReadWriteStorage → ReadWriteStorage
                // Uniform is incompatible with storage modes.
                if prev.access != binding.access {
                    let promoted = promote_access(prev.access, binding.access).ok_or_else(|| {
                        format!(
                            "incompatible access modes at @group({}) @binding({}) while fusing '{}': {:?} vs {:?}",
                            binding.group,
                            binding.binding,
                            program.id,
                            prev.access,
                            binding.access,
                        )
                    })?;
                    prev.access = promoted;
                }
            } else {
                merged.insert(key, binding.clone());
            }
        }
    }
    Ok(merged)
}

/// Promote two access modes to the most permissive compatible mode.
/// Returns `None` for incompatible combinations (e.g. Uniform + Storage).
fn promote_access(a: BindingAccess, b: BindingAccess) -> Option<BindingAccess> {
    match (a, b) {
        (BindingAccess::ReadOnlyStorage, BindingAccess::ReadWriteStorage)
        | (BindingAccess::ReadWriteStorage, BindingAccess::ReadOnlyStorage) => {
            Some(BindingAccess::ReadWriteStorage)
        }
        (BindingAccess::Uniform, BindingAccess::Uniform) => Some(BindingAccess::Uniform),
        _ => None,
    }
}

/// Union all `eos_params` across input programs, deduplicated by `wgsl_field`.
///
/// Order is preserved by first-seen insertion.
fn merge_eos_params(programs: &[KernelProgram]) -> Vec<ParamSpec> {
    let mut seen = BTreeSet::new();
    let mut merged = Vec::new();
    for program in programs {
        for param in &program.eos_params {
            if seen.insert(param.wgsl_field) {
                merged.push(param.clone());
            }
        }
    }
    merged
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
            rename_map.iter().fold(line.clone(), |acc, (old, new)| {
                rename_identifier(&acc, old, new)
            })
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
            && !is_member_access_field(src_bytes, i)
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

fn is_member_access_field(src: &[u8], ident_start: usize) -> bool {
    if ident_start == 0 {
        return false;
    }
    let mut i = ident_start;
    while i > 0 {
        let prev = src[i - 1];
        if prev.is_ascii_whitespace() {
            i -= 1;
            continue;
        }
        return prev == b'.';
    }
    false
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn program_references_constants_field(program: &KernelProgram, field: &str) -> bool {
    let needle = format!("constants.{field}");
    let mut sections = program.indexing.iter().chain(program.preamble.iter());
    if sections.any(|line| line.contains(&needle)) {
        return true;
    }
    if program.body.iter().any(|line| line.contains(&needle)) {
        return true;
    }
    if program.launch.invocation_index_expr.contains(&needle) {
        return true;
    }
    program
        .launch
        .bounds_check_expr
        .as_ref()
        .map(|expr| expr.contains(&needle))
        .unwrap_or(false)
}

fn constants_extra_params_for_program(program: &KernelProgram) -> Vec<ParamSpec> {
    let known_eos_fields = [
        ("eos.gamma", "eos_gamma"),
        ("eos.gm1", "eos_gm1"),
        ("eos.r", "eos_r"),
        ("eos.dp_drho", "eos_dp_drho"),
        ("eos.p_offset", "eos_p_offset"),
        ("eos.theta_ref", "eos_theta_ref"),
    ];
    let mut extras = Vec::new();
    for (key, field) in known_eos_fields {
        if program_references_constants_field(program, field) {
            extras.push(ParamSpec {
                key,
                wgsl_field: field,
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            });
        }
    }
    extras
}

fn apply_aggressive_cleanup(program: &mut KernelProgram) {
    program
        .body
        .retain(|line| !is_noop_local_self_assignment(line));
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
    lines.push(format!("// GENERATED BY CFD2 DSL FUSION ({})", program.id));
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
    let mut by_slot = BTreeMap::<(u32, u32), KernelBinding>::new();
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
        by_slot.insert(key, binding.clone());
    }

    // Emit canonical shared struct definitions so lowered WGSL is self-contained.
    let needs_constants_struct = by_slot
        .values()
        .any(|binding| binding.wgsl_type == "Constants");
    let needs_vector2_struct = by_slot
        .values()
        .any(|binding| binding.wgsl_type.contains("Vector2"));
    let needs_low_mach_params_struct = by_slot
        .values()
        .any(|binding| binding.wgsl_type == "LowMachParams");
    if needs_constants_struct || needs_vector2_struct || needs_low_mach_params_struct {
        let mut shared_structs_module = super::wgsl_ast::Module::new();
        if needs_vector2_struct {
            shared_structs_module.push(super::wgsl_ast::Item::Struct(
                super::wgsl_bindings::vector2_struct(),
            ));
        }
        if needs_constants_struct {
            // Use the structured eos_params declaration from the KernelProgram IR
            // instead of scanning body strings for field references (§2d fix).
            // Falls back to the legacy string-scan heuristic when eos_params is
            // empty to keep backward compatibility with programs that haven't
            // been updated yet.
            let extra_constants = if program.eos_params.is_empty() {
                constants_extra_params_for_program(program)
            } else {
                program.eos_params.clone()
            };
            shared_structs_module.push(super::wgsl_ast::Item::Struct(
                super::constants::constants_struct(&extra_constants),
            ));
        }
        if needs_low_mach_params_struct {
            shared_structs_module.push(super::wgsl_ast::Item::Struct(
                super::wgsl_bindings::low_mach_params_struct(),
            ));
        }
        lines.push(shared_structs_module.to_wgsl());
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

    // Emit module-level helper functions between bindings and the compute entry point.
    if !program.helper_functions.is_empty() {
        lines.push(String::new());
        for helper in &program.helper_functions {
            lines.push(helper.clone());
        }
    }

    lines.push(String::new());
    lines.push(format!(
        "@compute @workgroup_size({}, {}, {})",
        program.launch.workgroup_size[0],
        program.launch.workgroup_size[1],
        program.launch.workgroup_size[2]
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
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));

        let err = synthesize_fused_program("fused", "rule/a_b", &[a, b], FusionSafetyPolicy::Safe)
            .unwrap_err();
        assert!(err.contains("RAW hazard"), "unexpected error: {err}");
    }

    #[test]
    fn merge_bindings_rejects_incompatible_interface() {
        let a = sample_program("a");
        let mut b = sample_program("b");
        b.bindings[0].name = "state_other".to_string();
        let err =
            synthesize_fused_program("fused", "rule/a_b", &[a, b], FusionSafetyPolicy::Aggressive)
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
        let fused =
            synthesize_fused_program("fused", "rule/a_b", &[a, b], FusionSafetyPolicy::Aggressive)
                .expect("fused synthesis");
        let joined = fused.body.join("\n");
        assert!(joined.contains("value = value + 1.0;"));
        assert!(joined.contains("k1_value = k1_value + 1.0;"));
    }

    #[test]
    fn lowering_to_wgsl_is_deterministic() {
        let a = sample_program("a");
        let b = sample_program("b");
        let fused =
            synthesize_fused_program("fused", "rule/a_b", &[a, b], FusionSafetyPolicy::Aggressive)
                .expect("fused synthesis");

        let wgsl1 = lower_kernel_program_to_wgsl(&fused).expect("wgsl");
        let wgsl2 = lower_kernel_program_to_wgsl(&fused).expect("wgsl");
        assert_eq!(wgsl1.to_wgsl(), wgsl2.to_wgsl());
        assert!(wgsl1
            .to_wgsl()
            .contains("@compute @workgroup_size(64, 1, 1)"));
        assert!(wgsl1.to_wgsl().contains("struct Constants"));
    }

    #[test]
    fn lowering_emits_eos_constants_when_program_references_them() {
        let launch = LaunchSemantics::new(
            [64, 1, 1],
            "global_id.y * constants.stride_x + global_id.x",
            Some("idx >= arrayLength(&state) / 2u"),
        );
        let mut program = KernelProgram::new(
            "eos_constants_case",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
            ],
        );
        program.body = vec![
            "let c = constants.eos_gamma / max(constants.eos_gm1, 1e-12);".to_string(),
            "state[idx] = c;".to_string(),
        ];

        let wgsl = lower_kernel_program_to_wgsl(&program).expect("lowering should succeed");
        let src = wgsl.to_wgsl();
        assert!(
            src.contains("eos_gamma: f32"),
            "missing eos_gamma in Constants"
        );
        assert!(src.contains("eos_gm1: f32"), "missing eos_gm1 in Constants");
        assert!(
            !src.contains("eos_dp_drho: f32"),
            "unused eos fields must stay absent"
        );
    }

    #[test]
    fn lowering_emits_vector2_struct_when_required_by_bindings() {
        let launch = LaunchSemantics::new(
            [64, 1, 1],
            "global_id.y * constants.stride_x + global_id.x",
            Some("idx >= arrayLength(&state) / 2u"),
        );
        let mut program = KernelProgram::new(
            "vector2_case",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(
                    0,
                    0,
                    "cell_centers",
                    "array<Vector2>",
                    BindingAccess::ReadOnlyStorage,
                ),
                KernelBinding::new(0, 1, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(0, 2, "constants", "Constants", BindingAccess::Uniform),
            ],
        );
        program.body = vec!["state[idx] = cell_centers[idx].x;".to_string()];

        let wgsl = lower_kernel_program_to_wgsl(&program)
            .expect("lowering with Vector2 binding should succeed");
        let src = wgsl.to_wgsl();
        assert!(
            src.contains("struct Vector2"),
            "missing Vector2 struct in WGSL"
        );
        assert!(
            src.contains("struct Constants"),
            "missing Constants struct in WGSL"
        );
    }

    #[test]
    fn aggressive_cleanup_removes_noop_local_self_assignment() {
        let mut a = sample_program("a");
        a.body.insert(0, "value = value;".to_string());
        let b = sample_program("b");

        let safe = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a.clone(), b.clone()],
            FusionSafetyPolicy::Safe,
        )
        .expect("safe fused synthesis");
        let aggressive =
            synthesize_fused_program("fused", "rule/a_b", &[a, b], FusionSafetyPolicy::Aggressive)
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

        let safe = synthesize_fused_program(
            "fused",
            "rule/a_b",
            &[a.clone(), b.clone()],
            FusionSafetyPolicy::Safe,
        )
        .expect("safe fused synthesis");
        let aggressive =
            synthesize_fused_program("fused", "rule/a_b", &[a, b], FusionSafetyPolicy::Aggressive)
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

    #[test]
    fn symbol_rename_skips_member_access_fields() {
        let src = "let dt = max(constants.dt, state[idx].dt);";
        let renamed = rename_identifier(src, "dt", "k1_dt");
        assert_eq!(renamed, "let k1_dt = max(constants.dt, state[idx].dt);");
    }

    #[test]
    fn detect_hazards_finds_raw() {
        let mut a = sample_program("a");
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));

        let hazards = detect_hazards(&[a, b]);
        assert_eq!(hazards.len(), 1);
        assert_eq!(hazards[0].kind, HazardKind::RAW);
        assert_eq!(hazards[0].kernel_id, "b");
        assert_eq!(hazards[0].resources.len(), 1);
        assert_eq!(hazards[0].resources[0], EffectResource::binding(0, 0));
    }

    #[test]
    fn detect_hazards_finds_war() {
        let mut a = sample_program("a");
        a.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let hazards = detect_hazards(&[a, b]);
        assert_eq!(hazards.len(), 1);
        assert_eq!(hazards[0].kind, HazardKind::WAR);
        assert_eq!(hazards[0].kernel_id, "b");
    }

    #[test]
    fn detect_hazards_finds_waw() {
        let mut a = sample_program("a");
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let hazards = detect_hazards(&[a, b]);
        assert_eq!(hazards.len(), 1);
        assert_eq!(hazards[0].kind, HazardKind::WAW);
        assert_eq!(hazards[0].kernel_id, "b");
    }

    #[test]
    fn detect_hazards_finds_barriers_atomics() {
        let mut a = sample_program("a");
        a.side_effects.uses_barriers = true;

        let b = sample_program("b");

        let hazards = detect_hazards(&[a, b]);
        assert_eq!(hazards.len(), 1);
        assert_eq!(hazards[0].kind, HazardKind::BarriersOrAtomics);
        assert_eq!(hazards[0].kernel_id, "a");
    }

    #[test]
    fn detect_hazards_empty_for_safe_programs() {
        let a = sample_program("a");
        let b = sample_program("b");
        let hazards = detect_hazards(&[a, b]);
        assert!(hazards.is_empty());
    }

    #[test]
    fn detect_hazards_multiple_hazards_in_chain() {
        let mut a = sample_program("a");
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));
        b.side_effects
            .write_set
            .insert(EffectResource::component(0, 0, "x"));

        let mut c = sample_program("c");
        c.side_effects
            .read_set
            .insert(EffectResource::component(0, 0, "x"));

        let hazards = detect_hazards(&[a, b, c]);
        assert!(
            hazards.len() >= 2,
            "expected at least 2 hazards, got {}",
            hazards.len()
        );
        assert_eq!(hazards[0].kind, HazardKind::RAW);
        assert_eq!(hazards[0].kernel_id, "b");
    }

    #[test]
    fn aggressive_policy_succeeds_with_hazards_and_returns_report() {
        let mut a = sample_program("a");
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));

        let result = synthesize_fused_program_with_report(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
        );
        let (program, hazards) =
            result.expect("aggressive synthesis should succeed despite hazards");
        assert!(!hazards.is_empty(), "hazards should be reported");
        assert_eq!(hazards[0].kind, HazardKind::RAW);
        assert!(!program.body.is_empty());
    }

    #[test]
    fn safe_policy_still_rejects_on_hazards() {
        let mut a = sample_program("a");
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));

        let err = synthesize_fused_program_with_report(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Safe,
        )
        .unwrap_err();
        assert!(err.contains("RAW hazard"), "unexpected error: {err}");
    }

    #[test]
    fn hazard_report_display_formatting() {
        let report = HazardReport {
            kind: HazardKind::WAW,
            kernel_id: "dp_update_from_diag".to_string(),
            resources: vec![EffectResource::component(0, 0, "state:d_p_offset")],
        };
        let display = format!("{report}");
        assert!(display.contains("WAW hazard at kernel 'dp_update_from_diag'"));
        assert!(display.contains("@group(0) @binding(0):state:d_p_offset"));
    }

    #[test]
    fn lowering_uses_structured_eos_params_from_ir() {
        use cfd2_ir::solver::dimensions::Dimensionless;
        let launch = LaunchSemantics::new(
            [64, 1, 1],
            "global_id.y * constants.stride_x + global_id.x",
            Some("idx >= arrayLength(&state) / 2u"),
        );
        let mut program = KernelProgram::new(
            "structured_eos_test",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
            ],
        );
        // Declare eos_gamma and eos_r via structured IR field
        program.eos_params = vec![
            ParamSpec {
                key: "eos.gamma",
                wgsl_field: "eos_gamma",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
            ParamSpec {
                key: "eos.r",
                wgsl_field: "eos_r",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
        ];
        // Body references eos_gamma only — but since it's declared, eos_r
        // should also appear in the Constants struct.
        program.body = vec![
            "let c = constants.eos_gamma;".to_string(),
            "state[idx] = c;".to_string(),
        ];

        let wgsl = lower_kernel_program_to_wgsl(&program).expect("lowering should succeed");
        let src = wgsl.to_wgsl();
        assert!(
            src.contains("eos_gamma: f32"),
            "declared eos_gamma must appear in Constants"
        );
        assert!(
            src.contains("eos_r: f32"),
            "declared eos_r must appear in Constants even if not in body"
        );
        assert!(
            !src.contains("eos_gm1: f32"),
            "undeclared eos_gm1 must not appear"
        );
    }

    #[test]
    fn lowering_falls_back_to_string_scan_when_eos_params_empty() {
        let launch = LaunchSemantics::new(
            [64, 1, 1],
            "global_id.y * constants.stride_x + global_id.x",
            Some("idx >= arrayLength(&state) / 2u"),
        );
        let mut program = KernelProgram::new(
            "fallback_eos_test",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
            ],
        );
        // No eos_params set (empty vec), but body references eos_gamma
        program.body = vec![
            "let c = constants.eos_gamma;".to_string(),
            "state[idx] = c;".to_string(),
        ];
        assert!(program.eos_params.is_empty());

        let wgsl = lower_kernel_program_to_wgsl(&program).expect("lowering should succeed");
        let src = wgsl.to_wgsl();
        assert!(
            src.contains("eos_gamma: f32"),
            "string-scan fallback should detect eos_gamma in body"
        );
    }

    #[test]
    fn fusion_merges_eos_params_from_input_programs() {
        use cfd2_ir::solver::dimensions::Dimensionless;
        let mut a = sample_program("a");
        a.eos_params = vec![ParamSpec {
            key: "eos.gamma",
            wgsl_field: "eos_gamma",
            wgsl_type: "f32",
            unit: Dimensionless::UNIT,
        }];

        let mut b = sample_program("b");
        b.eos_params = vec![
            ParamSpec {
                key: "eos.gamma",
                wgsl_field: "eos_gamma",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
            ParamSpec {
                key: "eos.r",
                wgsl_field: "eos_r",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
        ];

        let fused =
            synthesize_fused_program("fused", "rule/a_b", &[a, b], FusionSafetyPolicy::Safe)
                .expect("safe fused synthesis");

        assert_eq!(fused.eos_params.len(), 2, "should deduplicate eos_gamma");
        let fields: Vec<&str> = fused.eos_params.iter().map(|p| p.wgsl_field).collect();
        assert!(fields.contains(&"eos_gamma"));
        assert!(fields.contains(&"eos_r"));
    }
}
