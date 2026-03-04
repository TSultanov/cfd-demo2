use super::KernelWgsl;
use cfd2_ir::solver::dimensions::{Dimensionless, UnitDimension};
use cfd2_ir::solver::ir::ports::ParamSpec;
use cfd2_ir::solver::ir::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelBodyIrOp,
    KernelBufferAccess, KernelProgram, SideEffectMetadata,
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

    if policy == FusionSafetyPolicy::Aggressive {
        if let Some(program) = remapped
            .iter()
            .find(|p| (!p.preamble.is_empty() || !p.body.is_empty()) && p.body_ir_ops.is_empty())
        {
            return Err(format!(
                "fusion rejected: aggressive cleanup requires structured body_ir_ops for kernel '{}'",
                program.id
            ));
        }
    }

    let hazards = ensure_safe_composition(&remapped, policy)?;

    let dispatch = remapped[0].dispatch.clone();
    let launch = remapped[0].launch.clone();
    let merged_bindings = merge_bindings(&remapped)?;

    let mut body = Vec::new();
    let mut body_ir_ops = Vec::<KernelBodyIrOp>::new();
    let mut local_symbols = Vec::new();
    let mut side_effects = SideEffectMetadata::default();
    let mut helper_functions = Vec::<String>::new();

    // Track whether all programs have structured AST bodies.
    let all_have_body_ast = remapped.iter().all(|p| p.body_ast.is_some());
    let mut fused_body_ast: Vec<cfd2_ir::ast::Stmt> = Vec::new();

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
        let segment_exec_start = body.len();
        let renamed_preamble = rename_lines(&program.preamble, &rename_map);
        body.extend(renamed_preamble);
        let renamed_body = rename_lines(&program.body, &rename_map);
        body.extend(renamed_body);
        body.push(format!("// end fused segment: {}", program.id));

        // AST-based body concatenation (alongside string-based)
        if all_have_body_ast {
            fused_body_ast.push(cfd2_ir::ast::Stmt::Comment(format!("begin fused segment: {}", program.id)));
            if let Some(ref preamble_ast) = program.preamble_ast {
                fused_body_ast.extend(rename_stmts(preamble_ast, &rename_map));
            }
            if let Some(ref body_ast) = program.body_ast {
                fused_body_ast.extend(rename_stmts(body_ast, &rename_map));
            }
            fused_body_ast.push(cfd2_ir::ast::Stmt::Comment(format!("end fused segment: {}", program.id)));
        }

        if !program.body_ir_ops.is_empty() {
            body_ir_ops.extend(rename_and_offset_body_ir_ops(
                &program.body_ir_ops,
                &rename_map,
                segment_exec_start,
            ));
        }

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
    fused.indexing_ast = remapped[0].indexing_ast.clone();
    fused.preamble = Vec::new();
    fused.preamble_ast = Some(Vec::new());
    fused.body = body;
    fused.body_ast = if all_have_body_ast { Some(fused_body_ast) } else { None };
    fused.body_ir_ops = body_ir_ops;
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
    if let Some(ref mut preamble_ast) = fused.preamble_ast {
        preamble_ast.insert(0, cfd2_ir::ast::Stmt::Comment(format!("synthesized by fusion rule: {rule_name}")));
    }

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

fn rename_and_offset_body_ir_ops(
    ops: &[KernelBodyIrOp],
    rename_map: &BTreeMap<String, String>,
    body_line_offset: usize,
) -> Vec<KernelBodyIrOp> {
    ops.iter()
        .map(|op| match op {
            KernelBodyIrOp::Store {
                line_index,
                access,
                value_expr,
                value_reads,
            } => KernelBodyIrOp::Store {
                line_index: body_line_offset + line_index,
                access: KernelBufferAccess::new(
                    rename_text(&access.base, rename_map),
                    rename_text(&access.index_expr, rename_map),
                ),
                value_expr: rename_text(value_expr, rename_map),
                value_reads: value_reads
                    .iter()
                    .map(|read| {
                        KernelBufferAccess::new(
                            rename_text(&read.base, rename_map),
                            rename_text(&read.index_expr, rename_map),
                        )
                    })
                    .collect(),
            },
            KernelBodyIrOp::LetLoad {
                line_index,
                name,
                ty,
                access,
            } => KernelBodyIrOp::LetLoad {
                line_index: body_line_offset + line_index,
                name: rename_map
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| name.clone()),
                ty: ty.clone(),
                access: KernelBufferAccess::new(
                    rename_text(&access.base, rename_map),
                    rename_text(&access.index_expr, rename_map),
                ),
            },
            KernelBodyIrOp::NoopSelfAssign { line_index } => KernelBodyIrOp::NoopSelfAssign {
                line_index: body_line_offset + line_index,
            },
            KernelBodyIrOp::Invalidate { line_index } => KernelBodyIrOp::Invalidate {
                line_index: body_line_offset + line_index,
            },
        })
        .collect()
}

fn rename_text(src: &str, rename_map: &BTreeMap<String, String>) -> String {
    rename_map.iter().fold(src.to_string(), |acc, (old, new)| {
        rename_identifier(&acc, old, new)
    })
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
    // Prefer AST-based search when available.
    let needle = format!("constants.{field}");

    // Check AST sections first.
    let has_ast = program.indexing_ast.is_some()
        || program.preamble_ast.is_some()
        || program.body_ast.is_some();

    if has_ast {
        let all_stmts = program.indexing_ast.iter()
            .chain(program.preamble_ast.iter())
            .chain(program.body_ast.iter())
            .flat_map(|stmts| stmts.iter());
        for stmt in all_stmts {
            if ast_stmt_references_field(stmt, "constants", field) {
                return true;
            }
        }
    } else {
        // Legacy string-based search.
        let mut sections = program.indexing.iter().chain(program.preamble.iter());
        if sections.any(|line| line.contains(&needle)) {
            return true;
        }
        if program.body.iter().any(|line| line.contains(&needle)) {
            return true;
        }
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

/// Check if an expression references `base.field` (e.g. `constants.eos_gamma`).
fn ast_expr_references_field(expr: &cfd2_ir::ast::Expr, base: &str, field: &str) -> bool {
    use cfd2_ir::ast::ExprNode;
    match expr.node() {
        ExprNode::Field { base: base_expr, field: f } => {
            if f == field {
                if let ExprNode::Ident(name) = base_expr.node() {
                    if name == base {
                        return true;
                    }
                }
            }
            ast_expr_references_field(base_expr, base, field)
        }
        ExprNode::Binary { left, right, .. } => {
            ast_expr_references_field(left, base, field) || ast_expr_references_field(right, base, field)
        }
        ExprNode::Unary { expr: inner, .. } => ast_expr_references_field(inner, base, field),
        ExprNode::Call { callee, args } => {
            ast_expr_references_field(callee, base, field)
                || args.iter().any(|a| ast_expr_references_field(a, base, field))
        }
        ExprNode::Index { base: b, index } => {
            ast_expr_references_field(b, base, field) || ast_expr_references_field(index, base, field)
        }
        ExprNode::Ident(_) | ExprNode::Literal(_) => false,
    }
}

/// Check if a statement references `base.field`.
fn ast_stmt_references_field(stmt: &cfd2_ir::ast::Stmt, base: &str, field: &str) -> bool {
    use cfd2_ir::ast::Stmt;
    match stmt {
        Stmt::Let { expr, .. } => ast_expr_references_field(expr, base, field),
        Stmt::Var { expr, .. } => expr.as_ref().map_or(false, |e| ast_expr_references_field(e, base, field)),
        Stmt::Assign { target, value } => {
            ast_expr_references_field(target, base, field) || ast_expr_references_field(value, base, field)
        }
        Stmt::AssignOp { target, value, .. } => {
            ast_expr_references_field(target, base, field) || ast_expr_references_field(value, base, field)
        }
        Stmt::If { cond, then_block, else_block } => {
            ast_expr_references_field(cond, base, field)
                || then_block.stmts.iter().any(|s| ast_stmt_references_field(s, base, field))
                || else_block.as_ref().map_or(false, |b| b.stmts.iter().any(|s| ast_stmt_references_field(s, base, field)))
        }
        Stmt::For { cond, body, .. } => {
            ast_expr_references_field(cond, base, field)
                || body.stmts.iter().any(|s| ast_stmt_references_field(s, base, field))
        }
        Stmt::Loop { body } | Stmt::While { body, .. } => {
            body.stmts.iter().any(|s| ast_stmt_references_field(s, base, field))
        }
        Stmt::Call(expr) | Stmt::Increment(expr) | Stmt::Decrement(expr) => {
            ast_expr_references_field(expr, base, field)
        }
        Stmt::Return(expr) => expr.as_ref().map_or(false, |e| ast_expr_references_field(e, base, field)),
        Stmt::Comment(_) | Stmt::Break | Stmt::Continue => false,
    }
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

// ── AST-based rename and optimization passes ────────────────────────────

/// Rename identifiers in an expression tree using a rename map.
fn rename_expr(expr: &cfd2_ir::ast::Expr, rename_map: &BTreeMap<String, String>) -> cfd2_ir::ast::Expr {
    use cfd2_ir::ast::{Expr, ExprNode};
    match expr.node() {
        ExprNode::Ident(name) => {
            if let Some(new_name) = rename_map.get(name.as_str()) {
                Expr::ident(new_name.clone())
            } else {
                expr.clone()
            }
        }
        ExprNode::Literal(_) => expr.clone(),
        ExprNode::Field { base, field } => {
            rename_expr(base, rename_map).field(field.clone())
        }
        ExprNode::Index { base, index } => {
            rename_expr(base, rename_map).index(rename_expr(index, rename_map))
        }
        ExprNode::Unary { op, expr: inner } => {
            let new_inner = rename_expr(inner, rename_map);
            match op {
                cfd2_ir::ast::UnaryOp::Negate => -new_inner,
                cfd2_ir::ast::UnaryOp::Not => !new_inner,
                cfd2_ir::ast::UnaryOp::AddressOf => new_inner.addr_of(),
                cfd2_ir::ast::UnaryOp::Deref => new_inner.deref(),
            }
        }
        ExprNode::Binary { left, op, right } => {
            Expr::binary(
                rename_expr(left, rename_map),
                *op,
                rename_expr(right, rename_map),
            )
        }
        ExprNode::Call { callee, args } => {
            let new_callee = rename_expr(callee, rename_map);
            let new_args: Vec<Expr> = args.iter().map(|a| rename_expr(a, rename_map)).collect();
            Expr::call(new_callee, new_args)
        }
    }
}

/// Rename identifiers in a statement tree using a rename map.
fn rename_stmt(stmt: &cfd2_ir::ast::Stmt, rename_map: &BTreeMap<String, String>) -> cfd2_ir::ast::Stmt {
    use cfd2_ir::ast::{Block, ForInit, ForStep, Stmt};
    match stmt {
        Stmt::Comment(text) => Stmt::Comment(text.clone()),
        Stmt::Let { name, ty, expr } => Stmt::Let {
            name: rename_map.get(name).cloned().unwrap_or_else(|| name.clone()),
            ty: ty.clone(),
            expr: rename_expr(expr, rename_map),
        },
        Stmt::Var { name, ty, expr } => Stmt::Var {
            name: rename_map.get(name).cloned().unwrap_or_else(|| name.clone()),
            ty: ty.clone(),
            expr: expr.as_ref().map(|e| rename_expr(e, rename_map)),
        },
        Stmt::Assign { target, value } => Stmt::Assign {
            target: rename_expr(target, rename_map),
            value: rename_expr(value, rename_map),
        },
        Stmt::AssignOp { target, op, value } => Stmt::AssignOp {
            target: rename_expr(target, rename_map),
            op: *op,
            value: rename_expr(value, rename_map),
        },
        Stmt::If { cond, then_block, else_block } => Stmt::If {
            cond: rename_expr(cond, rename_map),
            then_block: rename_block(then_block, rename_map),
            else_block: else_block.as_ref().map(|b| rename_block(b, rename_map)),
        },
        Stmt::For { init, cond, step, body } => Stmt::For {
            init: rename_for_init(init, rename_map),
            cond: rename_expr(cond, rename_map),
            step: rename_for_step(step, rename_map),
            body: rename_block(body, rename_map),
        },
        Stmt::Loop { body } => Stmt::Loop {
            body: rename_block(body, rename_map),
        },
        Stmt::While { cond, body } => Stmt::While {
            cond: rename_expr(cond, rename_map),
            body: rename_block(body, rename_map),
        },
        Stmt::Break => Stmt::Break,
        Stmt::Continue => Stmt::Continue,
        Stmt::Return(expr) => Stmt::Return(expr.as_ref().map(|e| rename_expr(e, rename_map))),
        Stmt::Call(expr) => Stmt::Call(rename_expr(expr, rename_map)),
        Stmt::Increment(expr) => Stmt::Increment(rename_expr(expr, rename_map)),
        Stmt::Decrement(expr) => Stmt::Decrement(rename_expr(expr, rename_map)),
    }
}

fn rename_block(block: &cfd2_ir::ast::Block, rename_map: &BTreeMap<String, String>) -> cfd2_ir::ast::Block {
    cfd2_ir::ast::Block::new(block.stmts.iter().map(|s| rename_stmt(s, rename_map)).collect())
}

fn rename_for_init(init: &cfd2_ir::ast::ForInit, rename_map: &BTreeMap<String, String>) -> cfd2_ir::ast::ForInit {
    use cfd2_ir::ast::ForInit;
    match init {
        ForInit::Let { name, ty, expr } => ForInit::Let {
            name: rename_map.get(name).cloned().unwrap_or_else(|| name.clone()),
            ty: ty.clone(),
            expr: rename_expr(expr, rename_map),
        },
        ForInit::Var { name, ty, expr } => ForInit::Var {
            name: rename_map.get(name).cloned().unwrap_or_else(|| name.clone()),
            ty: ty.clone(),
            expr: rename_expr(expr, rename_map),
        },
        ForInit::Assign { target, value } => ForInit::Assign {
            target: rename_expr(target, rename_map),
            value: rename_expr(value, rename_map),
        },
    }
}

fn rename_for_step(step: &cfd2_ir::ast::ForStep, rename_map: &BTreeMap<String, String>) -> cfd2_ir::ast::ForStep {
    use cfd2_ir::ast::ForStep;
    match step {
        ForStep::Increment(expr) => ForStep::Increment(rename_expr(expr, rename_map)),
        ForStep::Decrement(expr) => ForStep::Decrement(rename_expr(expr, rename_map)),
        ForStep::Assign { target, value } => ForStep::Assign {
            target: rename_expr(target, rename_map),
            value: rename_expr(value, rename_map),
        },
        ForStep::AssignOp { target, op, value } => ForStep::AssignOp {
            target: rename_expr(target, rename_map),
            op: *op,
            value: rename_expr(value, rename_map),
        },
    }
}

/// Rename identifiers in a slice of statements using a rename map.
fn rename_stmts(stmts: &[cfd2_ir::ast::Stmt], rename_map: &BTreeMap<String, String>) -> Vec<cfd2_ir::ast::Stmt> {
    stmts.iter().map(|s| rename_stmt(s, rename_map)).collect()
}

fn apply_aggressive_cleanup(program: &mut KernelProgram) {
    if program.body_ast.is_some() {
        apply_ast_load_after_store_forwarding(program);
        apply_ast_noop_self_assign_cleanup(program);
    } else {
        apply_ir_load_after_store_forwarding(program);
        apply_ir_noop_self_assign_cleanup(program);
    }
}

// ── AST-based aggressive cleanup passes ────────────────────────────────

/// AST-based load-after-store forwarding.
///
/// Walks the flat `body_ast` statement list. For each store (`Assign` to a
/// buffer index), records the stored value expression. When a subsequent
/// `Let` loads from the same buffer+index, replaces the load expression with
/// the forwarded value. Control-flow and non-trivial ops conservatively
/// invalidate the store map.
fn apply_ast_load_after_store_forwarding(program: &mut KernelProgram) {
    use cfd2_ir::ast::{Expr, ExprNode, Stmt};

    let body_ast = match program.body_ast.as_mut() {
        Some(ast) => ast,
        None => return,
    };

    // Key: (base_name, index_expr_str) -> forwarded value Expr
    let mut last_store: BTreeMap<(String, String), Expr> = BTreeMap::new();

    for stmt in body_ast.iter_mut() {
        match stmt {
            Stmt::Assign { target, value } => {
                if let Some((base, idx_str)) = ast_buffer_access_key(target) {
                    let reads = ast_collect_buffer_accesses(value);
                    if reads.is_empty() {
                        last_store.insert((base, idx_str), value.clone());
                    } else {
                        last_store.remove(&(base, idx_str));
                    }
                } else {
                    // Unknown assignment: conservatively invalidate all.
                    last_store.clear();
                }
            }
            Stmt::Let { expr, .. } => {
                if let Some((base, idx_str)) = ast_buffer_access_key(expr) {
                    if let Some(forwarded) = last_store.get(&(base, idx_str)) {
                        *expr = forwarded.clone();
                    }
                }
            }
            // Control flow and calls conservatively invalidate.
            Stmt::If { .. }
            | Stmt::For { .. }
            | Stmt::Loop { .. }
            | Stmt::While { .. }
            | Stmt::AssignOp { .. }
            | Stmt::Return(_)
            | Stmt::Call(_)
            | Stmt::Increment(_)
            | Stmt::Decrement(_)
            | Stmt::Break
            | Stmt::Continue => {
                last_store.clear();
            }
            Stmt::Comment(_) | Stmt::Var { .. } => {}
        }
    }
}

/// AST-based noop self-assign cleanup.
///
/// Removes statements of the form `ident = ident` where both sides are the
/// same identifier. Also keeps `body_ast` consistent by filtering out the
/// removed statements (no line-index tracking needed).
fn apply_ast_noop_self_assign_cleanup(program: &mut KernelProgram) {
    use cfd2_ir::ast::{ExprNode, Stmt};

    let body_ast = match program.body_ast.as_mut() {
        Some(ast) => ast,
        None => return,
    };

    body_ast.retain(|stmt| {
        if let Stmt::Assign { target, value } = stmt {
            if let (ExprNode::Ident(lhs), ExprNode::Ident(rhs)) = (target.node(), value.node()) {
                if lhs == rhs {
                    return false;
                }
            }
        }
        true
    });

    // Also sync the string body and body_ir_ops if present (keep dual paths consistent).
    if !program.body_ir_ops.is_empty() {
        apply_ir_load_after_store_forwarding(program);
        apply_ir_noop_self_assign_cleanup(program);
    }
}

/// Extract (base_name, index_expr_string) from a buffer-index expression like `state[idx]`.
fn ast_buffer_access_key(expr: &cfd2_ir::ast::Expr) -> Option<(String, String)> {
    use cfd2_ir::ast::ExprNode;
    match expr.node() {
        ExprNode::Index { base, index } => match base.node() {
            ExprNode::Ident(name) => Some((name.clone(), index.to_string())),
            _ => None,
        },
        _ => None,
    }
}

/// Collect all buffer accesses (base[index]) from an expression tree.
fn ast_collect_buffer_accesses(expr: &cfd2_ir::ast::Expr) -> Vec<(String, String)> {
    use cfd2_ir::ast::ExprNode;
    let mut accesses = Vec::new();
    ast_collect_buffer_accesses_recursive(expr, &mut accesses);
    accesses
}

fn ast_collect_buffer_accesses_recursive(expr: &cfd2_ir::ast::Expr, out: &mut Vec<(String, String)>) {
    use cfd2_ir::ast::ExprNode;
    match expr.node() {
        ExprNode::Index { base, index } => {
            if let ExprNode::Ident(name) = base.node() {
                out.push((name.clone(), index.to_string()));
            }
            ast_collect_buffer_accesses_recursive(base, out);
            ast_collect_buffer_accesses_recursive(index, out);
        }
        ExprNode::Binary { left, right, .. } => {
            ast_collect_buffer_accesses_recursive(left, out);
            ast_collect_buffer_accesses_recursive(right, out);
        }
        ExprNode::Unary { expr: inner, .. } => {
            ast_collect_buffer_accesses_recursive(inner, out);
        }
        ExprNode::Call { callee, args } => {
            ast_collect_buffer_accesses_recursive(callee, out);
            for a in args {
                ast_collect_buffer_accesses_recursive(a, out);
            }
        }
        ExprNode::Field { base, .. } => {
            ast_collect_buffer_accesses_recursive(base, out);
        }
        ExprNode::Ident(_) | ExprNode::Literal(_) => {}
    }
}

fn apply_ir_load_after_store_forwarding(program: &mut KernelProgram) {
    if program.body_ir_ops.is_empty() {
        return;
    }

    let mut ops = program.body_ir_ops.clone();
    ops.sort_by_key(KernelBodyIrOp::line_index);

    let mut last_store_by_access = BTreeMap::<(String, String), String>::new();

    for op in ops {
        match op {
            KernelBodyIrOp::Store {
                access,
                value_expr,
                value_reads,
                ..
            } => {
                let key = (access.base, access.index_expr);
                if value_reads.is_empty() {
                    last_store_by_access.insert(key, value_expr);
                } else {
                    last_store_by_access.remove(&key);
                }
            }
            KernelBodyIrOp::LetLoad {
                line_index,
                name,
                ty,
                access,
            } => {
                let key = (access.base, access.index_expr);
                let Some(value_expr) = last_store_by_access.get(&key) else {
                    continue;
                };
                if line_index >= program.body.len() {
                    continue;
                }
                let mut line = format!("let {name}");
                if let Some(ty) = ty {
                    line.push_str(&format!(": {ty}"));
                }
                line.push_str(" = ");
                line.push_str(value_expr);
                line.push(';');
                program.body[line_index] = line;
            }
            KernelBodyIrOp::NoopSelfAssign { .. } => {}
            KernelBodyIrOp::Invalidate { .. } => {
                last_store_by_access.clear();
            }
        }
    }
}

fn apply_ir_noop_self_assign_cleanup(program: &mut KernelProgram) {
    if program.body_ir_ops.is_empty() {
        return;
    }

    let mut removal_indices: Vec<usize> = program
        .body_ir_ops
        .iter()
        .filter_map(|op| match op {
            KernelBodyIrOp::NoopSelfAssign { line_index } => Some(*line_index),
            _ => None,
        })
        .filter(|line_index| *line_index < program.body.len())
        .collect();

    if removal_indices.is_empty() {
        return;
    }

    removal_indices.sort_unstable();
    removal_indices.dedup();

    let remove_set: BTreeSet<usize> = removal_indices.iter().copied().collect();
    program.body = program
        .body
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            if remove_set.contains(&idx) {
                None
            } else {
                Some(line.clone())
            }
        })
        .collect();

    let remap_index = |old_idx: usize| -> Option<usize> {
        if remove_set.contains(&old_idx) {
            return None;
        }
        let removed_before = removal_indices.iter().filter(|idx| **idx < old_idx).count();
        Some(old_idx - removed_before)
    };

    program.body_ir_ops = program
        .body_ir_ops
        .iter()
        .filter_map(|op| match op {
            KernelBodyIrOp::Store {
                line_index,
                access,
                value_expr,
                value_reads,
            } => remap_index(*line_index).map(|line_index| KernelBodyIrOp::Store {
                line_index,
                access: access.clone(),
                value_expr: value_expr.clone(),
                value_reads: value_reads.clone(),
            }),
            KernelBodyIrOp::LetLoad {
                line_index,
                name,
                ty,
                access,
            } => remap_index(*line_index).map(|line_index| KernelBodyIrOp::LetLoad {
                line_index,
                name: name.clone(),
                ty: ty.clone(),
                access: access.clone(),
            }),
            KernelBodyIrOp::NoopSelfAssign { .. } => None,
            KernelBodyIrOp::Invalidate { line_index } => {
                remap_index(*line_index).map(|line_index| KernelBodyIrOp::Invalidate { line_index })
            }
        })
        .collect();
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

    // Emit indexing section
    if let Some(ref indexing_ast) = program.indexing_ast {
        for line in cfd2_ir::ast::stmt::render_stmt_lines(indexing_ast) {
            lines.push(format!("    {line}"));
        }
    } else {
        for line in &program.indexing {
            lines.push(format!("    {line}"));
        }
    }

    // Emit preamble section
    if let Some(ref preamble_ast) = program.preamble_ast {
        for line in cfd2_ir::ast::stmt::render_stmt_lines(preamble_ast) {
            lines.push(format!("    {line}"));
        }
    } else {
        for line in &program.preamble {
            lines.push(format!("    {line}"));
        }
    }

    // Emit body section
    if let Some(ref body_ast) = program.body_ast {
        for line in cfd2_ir::ast::stmt::render_stmt_lines(body_ast) {
            lines.push(format!("    {line}"));
        }
    } else {
        for line in &program.body {
            lines.push(format!("    {line}"));
        }
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
        use cfd2_ir::ast::{Expr, Stmt, Type};
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
        let preamble_stmts = vec![Stmt::Var {
            name: "value".to_string(),
            ty: Some(Type::F32),
            expr: Some(Expr::lit_f32(0.0)),
        }];
        let body_stmts = vec![
            Stmt::Assign {
                target: Expr::ident("value"),
                value: Expr::ident("value") + Expr::lit_f32(1.0),
            },
            Stmt::Assign {
                target: Expr::ident("state").index(Expr::ident("idx")),
                value: Expr::ident("value"),
            },
        ];
        program.preamble = super::super::wgsl_ast::render_block_lines(
            &cfd2_ir::ast::Block::new(preamble_stmts.clone()),
        );
        program.preamble_ast = Some(preamble_stmts);
        program.body = super::super::wgsl_ast::render_block_lines(
            &cfd2_ir::ast::Block::new(body_stmts.clone()),
        );
        program.body_ast = Some(body_stmts);
        program.body_ir_ops = vec![
            KernelBodyIrOp::Invalidate { line_index: 0 },
            KernelBodyIrOp::Store {
                line_index: 1,
                access: KernelBufferAccess::new("state", "idx"),
                value_expr: "value".to_string(),
                value_reads: Vec::new(),
            },
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
        program.body_ast = Some(vec![
            cfd2_ir::ast::Stmt::Let {
                name: "c".to_string(),
                ty: None,
                expr: cfd2_ir::ast::Expr::ident("constants").field("eos_gamma")
                    / cfd2_ir::ast::Expr::call(
                        cfd2_ir::ast::Expr::ident("max"),
                        vec![
                            cfd2_ir::ast::Expr::ident("constants").field("eos_gm1"),
                            cfd2_ir::ast::Expr::lit_f32(1e-12),
                        ],
                    ),
            },
            cfd2_ir::ast::Stmt::Assign {
                target: cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::ident("idx")),
                value: cfd2_ir::ast::Expr::ident("c"),
            },
        ]);

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
        program.body_ast = Some(vec![cfd2_ir::ast::Stmt::Assign {
            target: cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::ident("idx")),
            value: cfd2_ir::ast::Expr::ident("cell_centers")
                .index(cfd2_ir::ast::Expr::ident("idx"))
                .field("x"),
        }]);

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
        a.body_ir_ops = vec![
            KernelBodyIrOp::Invalidate { line_index: 0 },
            KernelBodyIrOp::NoopSelfAssign { line_index: 1 },
            KernelBodyIrOp::Invalidate { line_index: 2 },
            KernelBodyIrOp::Store {
                line_index: 3,
                access: KernelBufferAccess::new("state", "idx"),
                value_expr: "value".to_string(),
                value_reads: Vec::new(),
            },
        ];
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
    fn aggressive_cleanup_forwards_ir_store_to_load() {
        use cfd2_ir::ast::{Expr, Stmt};
        let launch = LaunchSemantics::new([64, 1, 1], "idx", Some("idx >= n"));
        let mut program = KernelProgram::new(
            "ir_forward_test",
            DispatchDomain::Cells,
            launch,
            vec![KernelBinding::new(
                0,
                0,
                "state",
                "array<f32>",
                BindingAccess::ReadWriteStorage,
            )],
        );
        let body_stmts = vec![
            Stmt::Assign {
                target: Expr::ident("state").index(Expr::ident("idx")),
                value: Expr::ident("value"),
            },
            Stmt::Let {
                name: "out".to_string(),
                ty: None,
                expr: Expr::ident("state").index(Expr::ident("idx")),
            },
        ];
        program.body = super::super::wgsl_ast::render_block_lines(
            &cfd2_ir::ast::Block::new(body_stmts.clone()),
        );
        program.body_ast = Some(body_stmts);
        program.body_ir_ops = vec![
            KernelBodyIrOp::Store {
                line_index: 0,
                access: KernelBufferAccess::new("state", "idx"),
                value_expr: "value".to_string(),
                value_reads: Vec::new(),
            },
            KernelBodyIrOp::LetLoad {
                line_index: 1,
                name: "out".to_string(),
                ty: None,
                access: KernelBufferAccess::new("state", "idx"),
            },
        ];

        apply_aggressive_cleanup(&mut program);
        // After forwarding, the let should load the stored value directly.
        let forwarded_body = program.body_ast.as_ref().expect("body_ast should be set");
        match &forwarded_body[1] {
            Stmt::Let { name, expr, .. } => {
                assert_eq!(name, "out");
                assert_eq!(expr.to_string(), "value");
            }
            other => panic!("expected Let, got {:?}", other),
        }
    }

    #[test]
    fn aggressive_cleanup_does_not_forward_when_store_value_reads_memory() {
        use cfd2_ir::ast::{Expr, Stmt};
        let launch = LaunchSemantics::new([64, 1, 1], "idx", Some("idx >= n"));
        let mut program = KernelProgram::new(
            "ir_forward_guard_test",
            DispatchDomain::Cells,
            launch,
            vec![KernelBinding::new(
                0,
                0,
                "state",
                "array<f32>",
                BindingAccess::ReadWriteStorage,
            )],
        );
        let body_stmts = vec![
            Stmt::Assign {
                target: Expr::ident("state").index(Expr::ident("idx")),
                value: Expr::ident("state").index(Expr::ident("idx2")),
            },
            Stmt::Let {
                name: "out".to_string(),
                ty: None,
                expr: Expr::ident("state").index(Expr::ident("idx")),
            },
        ];
        program.body = super::super::wgsl_ast::render_block_lines(
            &cfd2_ir::ast::Block::new(body_stmts.clone()),
        );
        program.body_ast = Some(body_stmts);
        program.body_ir_ops = vec![
            KernelBodyIrOp::Store {
                line_index: 0,
                access: KernelBufferAccess::new("state", "idx"),
                value_expr: "state[idx2]".to_string(),
                value_reads: vec![KernelBufferAccess::new("state", "idx2")],
            },
            KernelBodyIrOp::LetLoad {
                line_index: 1,
                name: "out".to_string(),
                ty: None,
                access: KernelBufferAccess::new("state", "idx"),
            },
        ];

        apply_aggressive_cleanup(&mut program);
        // Should NOT forward because the stored value reads from memory.
        let body = program.body_ast.as_ref().expect("body_ast should be set");
        match &body[1] {
            Stmt::Let { name, expr, .. } => {
                assert_eq!(name, "out");
                assert_eq!(expr.to_string(), "state[idx]");
            }
            other => panic!("expected Let, got {:?}", other),
        }
    }

    #[test]
    fn aggressive_cleanup_respects_invalidation_boundaries() {
        use cfd2_ir::ast::{Expr, Stmt};
        let launch = LaunchSemantics::new([64, 1, 1], "idx", Some("idx >= n"));
        let mut program = KernelProgram::new(
            "ir_forward_invalidate_test",
            DispatchDomain::Cells,
            launch,
            vec![KernelBinding::new(
                0,
                0,
                "state",
                "array<f32>",
                BindingAccess::ReadWriteStorage,
            )],
        );
        let body_stmts = vec![
            Stmt::Assign {
                target: Expr::ident("state").index(Expr::ident("idx")),
                value: Expr::ident("value"),
            },
            Stmt::If {
                cond: Expr::ident("cond"),
                then_block: cfd2_ir::ast::Block::new(vec![Stmt::Return(None)]),
                else_block: None,
            },
            Stmt::Let {
                name: "out".to_string(),
                ty: None,
                expr: Expr::ident("state").index(Expr::ident("idx")),
            },
        ];
        program.body = super::super::wgsl_ast::render_block_lines(
            &cfd2_ir::ast::Block::new(body_stmts.clone()),
        );
        program.body_ast = Some(body_stmts);
        program.body_ir_ops = vec![
            KernelBodyIrOp::Store {
                line_index: 0,
                access: KernelBufferAccess::new("state", "idx"),
                value_expr: "value".to_string(),
                value_reads: Vec::new(),
            },
            KernelBodyIrOp::Invalidate { line_index: 1 },
            KernelBodyIrOp::LetLoad {
                line_index: 2,
                name: "out".to_string(),
                ty: None,
                access: KernelBufferAccess::new("state", "idx"),
            },
        ];

        apply_aggressive_cleanup(&mut program);
        // Should NOT forward past the invalidation boundary (if stmt).
        let body = program.body_ast.as_ref().expect("body_ast should be set");
        match &body[2] {
            Stmt::Let { name, expr, .. } => {
                assert_eq!(name, "out");
                assert_eq!(expr.to_string(), "state[idx]");
            }
            other => panic!("expected Let, got {:?}", other),
        }
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
        program.body_ast = Some(vec![
            cfd2_ir::ast::Stmt::Let {
                name: "c".to_string(),
                ty: None,
                expr: cfd2_ir::ast::Expr::ident("constants").field("eos_gamma"),
            },
            cfd2_ir::ast::Stmt::Assign {
                target: cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::ident("idx")),
                value: cfd2_ir::ast::Expr::ident("c"),
            },
        ]);

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
