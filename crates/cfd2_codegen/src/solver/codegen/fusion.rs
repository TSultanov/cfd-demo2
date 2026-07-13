//! Kernel fusion synthesis engine.
//!
//! # Safety Model
//!
//! Fusion merges multiple kernel programs into a single GPU dispatch. This is
//! only safe when the kernels have no inter-kernel data dependencies that would
//! require memory barriers.
//!
//! ## Hazard Detection
//!
//! The [`detect_hazards`] function performs conservative side-effect analysis
//! across a sequence of [`KernelProgram`]s, using their `read_set` and
//! `write_set` metadata. It reports RAW (read-after-write), WAR
//! (write-after-read), WAW (write-after-write), and barrier/atomics hazards.
//!
//! This analysis is *conservative*: it reports hazards whenever two kernels
//! touch the same buffer binding slot, even if they access disjoint index
//! ranges at runtime (e.g. two per-cell kernels writing different state
//! components at `state[idx * stride + offset_a]` vs `state[idx * stride +
//! offset_b]`). False positives are common.
//!
//! ## Safety Policies
//!
//! - [`FusionSafetyPolicy::Safe`]: Any detected hazard rejects fusion. This
//!   is the default for production rules.
//!
//! - [`FusionSafetyPolicy::Aggressive`]: Hazards are checked against an
//!   explicit whitelist ([`ExpectedHazard`] entries). Only whitelisted hazards
//!   are tolerated; any non-whitelisted hazard still causes rejection. Each
//!   whitelist entry requires a human-readable justification explaining why
//!   the hazard is safe (e.g. "per-cell writes at disjoint offsets").
//!
//! ## Cleanup Passes
//!
//! Post-synthesis AST optimization passes are controlled by
//! [`FusionCleanupPolicy`], which is orthogonal to hazard policy:
//!
//! - `apply_ast_load_after_store_forwarding`: Replaces loads from a buffer
//!   with the previously stored value when the store is provably local.
//!   Conservative: invalidates on control flow, does not forward when the
//!   stored value itself reads from memory.
//!
//! - `apply_ast_noop_self_assign_cleanup`: Removes `x = x` statements that
//!   arise from symbol renaming during fusion.

use super::KernelWgsl;
use cfd2_ir::dimensions::{Dimensionless, UnitDimension};
use cfd2_ir::kernel::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelProgram, SideEffectMetadata,
};
use cfd2_ir::ports::ParamSpec;
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

/// Controls whether post-synthesis AST optimization passes run.
///
/// This is orthogonal to [`FusionSafetyPolicy`]: cleanup transforms are
/// semantics-preserving and can run independently of hazard policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionCleanupPolicy {
    /// No post-synthesis AST cleanup.
    None,
    /// Run conservative AST cleanup passes (store→load forwarding,
    /// noop self-assign removal). These passes do not change semantics.
    Standard,
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

/// A declared hazard that the fusion rule author has audited and confirmed is
/// safe (e.g. because the kernels operate on disjoint index ranges, or the
/// dependency is a false positive from conservative side-effect tracking).
///
/// Under `Safe` policy this is ignored (all hazards reject).
/// Under `Aggressive` policy, only hazards matching an `ExpectedHazard` entry
/// in the rule's whitelist are tolerated; any non-whitelisted hazard still
/// causes a hard rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedHazard {
    pub kind: HazardKind,
    pub kernel_id: &'static str,
    /// Human-readable justification for why this hazard is safe.
    pub justification: &'static str,
}

impl ExpectedHazard {
    /// Check whether this expected hazard matches a detected [`HazardReport`].
    pub fn matches(&self, report: &HazardReport) -> bool {
        self.kind == report.kind && self.kernel_id == report.kernel_id
    }
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

/// Like [`synthesize_fused_program`], but with an explicit hazard whitelist
/// for Aggressive policy rules.
pub fn synthesize_fused_program_whitelisted(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
    expected_hazards: &[ExpectedHazard],
) -> Result<KernelProgram, String> {
    synthesize_fused_program_remapped_whitelisted(
        replacement_id,
        rule_name,
        programs,
        policy,
        &[],
        expected_hazards,
    )
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
    synthesize_fused_program_remapped_whitelisted(
        replacement_id,
        rule_name,
        programs,
        policy,
        binding_remaps,
        &[],
    )
}

/// Like [`synthesize_fused_program_remapped`], but with an explicit hazard
/// whitelist for Aggressive policy rules.
pub fn synthesize_fused_program_remapped_whitelisted(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
    binding_remaps: &[BindingRemap],
    expected_hazards: &[ExpectedHazard],
) -> Result<KernelProgram, String> {
    let (program, _hazards) = synthesize_fused_program_with_report_remapped(
        replacement_id,
        rule_name,
        programs,
        policy,
        binding_remaps,
        expected_hazards,
    )?;
    Ok(program)
}

/// Like [`synthesize_fused_program`], but also returns any hazard reports detected
/// during composition analysis. Under `Safe` policy, hazards cause rejection (Err).
/// Under `Aggressive` policy, hazards are checked against the expected whitelist;
/// non-whitelisted hazards cause rejection, whitelisted ones are collected and
/// returned alongside the successfully fused program.
pub fn synthesize_fused_program_with_report(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
) -> Result<(KernelProgram, Vec<HazardReport>), String> {
    synthesize_fused_program_with_report_remapped(
        replacement_id,
        rule_name,
        programs,
        policy,
        &[],
        &[],
    )
}

/// Like [`synthesize_fused_program_with_report`], but applies binding slot
/// remaps before merging.
pub fn synthesize_fused_program_with_report_remapped(
    replacement_id: impl Into<String>,
    rule_name: &str,
    programs: &[KernelProgram],
    policy: FusionSafetyPolicy,
    binding_remaps: &[BindingRemap],
    expected_hazards: &[ExpectedHazard],
) -> Result<(KernelProgram, Vec<HazardReport>), String> {
    if programs.is_empty() {
        return Err("fusion synthesis requires at least one input kernel".to_string());
    }

    let remapped: Vec<KernelProgram> = if binding_remaps.is_empty() {
        programs.to_vec()
    } else {
        apply_binding_remaps(programs, binding_remaps)?
    };

    if policy == FusionSafetyPolicy::Aggressive {}

    let hazards = ensure_safe_composition(&remapped, policy, expected_hazards)?;

    let dispatch = remapped[0].dispatch.clone();
    let launch = remapped[0].launch.clone();
    let merged_bindings = merge_bindings(&remapped)?;

    let mut fused_body: Vec<cfd2_ir::ast::Stmt> = Vec::new();
    let mut side_effects = SideEffectMetadata::default();
    let mut helper_functions = Vec::<String>::new();

    for (idx, program) in remapped.iter().enumerate() {
        let local_syms = program.local_symbols();
        let rename_map = deterministic_symbol_rename_map(idx, &local_syms);

        for helper in &program.helper_functions {
            if !helper_functions.contains(helper) {
                helper_functions.push(helper.clone());
            }
        }

        fused_body.push(cfd2_ir::ast::Stmt::Comment(format!(
            "begin fused segment: {}",
            program.id
        )));
        fused_body.extend(rename_stmts(&program.preamble, &rename_map));
        fused_body.extend(rename_stmts(&program.body, &rename_map));
        fused_body.push(cfd2_ir::ast::Stmt::Comment(format!(
            "end fused segment: {}",
            program.id
        )));

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
    fused.body = fused_body;
    fused.side_effects = side_effects;
    fused.eos_params = merge_eos_params(&remapped);

    let cleanup_policy = match policy {
        FusionSafetyPolicy::Safe => FusionCleanupPolicy::None,
        FusionSafetyPolicy::Aggressive => FusionCleanupPolicy::Standard,
    };

    if cleanup_policy == FusionCleanupPolicy::Standard {
        apply_fusion_cleanup(&mut fused);
    }

    fused.preamble.insert(
        0,
        cfd2_ir::ast::Stmt::Comment(format!("synthesized by fusion rule: {rule_name}")),
    );

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
    expected_hazards: &[ExpectedHazard],
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

    match policy {
        FusionSafetyPolicy::Safe => {
            if let Some(h) = hazards.first() {
                return Err(format!(
                    "fusion rejected: {} hazard at kernel '{}'",
                    h.kind, h.kernel_id
                ));
            }
        }
        FusionSafetyPolicy::Aggressive => {
            // Only whitelisted hazards are tolerated; an empty whitelist still
            // rejects, so every hazard must be explicitly audited.
            for h in &hazards {
                if !expected_hazards.iter().any(|e| e.matches(h)) {
                    return Err(format!(
                        "fusion rejected: unexpected {} hazard at kernel '{}' \
                         (not in expected_hazards whitelist)",
                        h.kind, h.kernel_id
                    ));
                }
            }
        }
    }

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

        let from_res = EffectResource::binding(remap.from_group, remap.from_binding);
        let to_res = EffectResource::binding(remap.to_group, remap.to_binding);
        if program.side_effects.read_set.remove(&from_res) {
            program.side_effects.read_set.insert(to_res.clone());
        }
        if program.side_effects.write_set.remove(&from_res) {
            program.side_effects.write_set.insert(to_res);
        }

        // Component-level side-effects at the same slot.
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
    let mut slots_by_name = BTreeMap::<String, (u32, u32)>::new();
    for program in programs {
        for binding in &program.bindings {
            let key = (binding.group, binding.binding);
            if let Some(previous_slot) = slots_by_name.get(&binding.name) {
                if *previous_slot != key {
                    return Err(format!(
                        "binding name '{}' occupies both @group({}) @binding({}) and @group({}) @binding({}) while fusing '{}'",
                        binding.name,
                        previous_slot.0,
                        previous_slot.1,
                        key.0,
                        key.1,
                        program.id,
                    ));
                }
            } else {
                slots_by_name.insert(binding.name.clone(), key);
            }
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

fn program_references_constants_field(program: &KernelProgram, field: &str) -> bool {
    let needle = format!("constants.{field}");

    let all_stmts = program
        .indexing
        .iter()
        .chain(program.preamble.iter())
        .chain(program.body.iter());
    for stmt in all_stmts {
        if ast_stmt_references_field(stmt, "constants", field) {
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
        ExprNode::Field {
            base: base_expr,
            field: f,
        } => {
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
            ast_expr_references_field(left, base, field)
                || ast_expr_references_field(right, base, field)
        }
        ExprNode::Unary { expr: inner, .. } => ast_expr_references_field(inner, base, field),
        ExprNode::Call { callee, args } => {
            ast_expr_references_field(callee, base, field)
                || args
                    .iter()
                    .any(|a| ast_expr_references_field(a, base, field))
        }
        ExprNode::Index { base: b, index } => {
            ast_expr_references_field(b, base, field)
                || ast_expr_references_field(index, base, field)
        }
        ExprNode::Ident(_) | ExprNode::Literal(_) => false,
    }
}

/// Check if a statement references `base.field`.
fn ast_stmt_references_field(stmt: &cfd2_ir::ast::Stmt, base: &str, field: &str) -> bool {
    use cfd2_ir::ast::Stmt;
    match stmt {
        Stmt::Let { expr, .. } => ast_expr_references_field(expr, base, field),
        Stmt::Var { expr, .. } => expr
            .as_ref()
            .map_or(false, |e| ast_expr_references_field(e, base, field)),
        Stmt::Assign { target, value } => {
            ast_expr_references_field(target, base, field)
                || ast_expr_references_field(value, base, field)
        }
        Stmt::AssignOp { target, value, .. } => {
            ast_expr_references_field(target, base, field)
                || ast_expr_references_field(value, base, field)
        }
        Stmt::If {
            cond,
            then_block,
            else_block,
        } => {
            ast_expr_references_field(cond, base, field)
                || then_block
                    .stmts
                    .iter()
                    .any(|s| ast_stmt_references_field(s, base, field))
                || else_block.as_ref().map_or(false, |b| {
                    b.stmts
                        .iter()
                        .any(|s| ast_stmt_references_field(s, base, field))
                })
        }
        Stmt::For { cond, body, .. } => {
            ast_expr_references_field(cond, base, field)
                || body
                    .stmts
                    .iter()
                    .any(|s| ast_stmt_references_field(s, base, field))
        }
        Stmt::Loop { body } | Stmt::While { body, .. } => body
            .stmts
            .iter()
            .any(|s| ast_stmt_references_field(s, base, field)),
        Stmt::Call(expr) | Stmt::Increment(expr) | Stmt::Decrement(expr) => {
            ast_expr_references_field(expr, base, field)
        }
        Stmt::Return(expr) => expr
            .as_ref()
            .map_or(false, |e| ast_expr_references_field(e, base, field)),
        Stmt::Comment(_) | Stmt::Break | Stmt::Continue => false,
    }
}

fn constants_extra_params_for_program(program: &KernelProgram) -> Vec<ParamSpec> {
    // Unstructured GPU kernels bind the shared `GpuConstants` POD wholesale.
    // A WGSL struct may omit an unused suffix, but it must never omit a field in
    // the middle of that canonical tail: doing so shifts every later field (for
    // example a kernel using eos_gm1/eos_r would read gamma/gm1).  The AST fallback
    // therefore emits the complete prefix through the last referenced field.
    let canonical_tail_fields = [
        ("eos.gamma", "eos_gamma"),
        ("eos.gm1", "eos_gm1"),
        ("eos.r", "eos_r"),
        ("eos.dp_drho", "eos_dp_drho"),
        ("eos.p_ref", "eos_p_ref"),
        ("eos.theta_ref", "eos_theta_ref"),
        ("eos.rho_ref", "eos_rho_ref"),
        ("eos.gauge_rho_ref", "eos_gauge_rho_ref"),
        ("eos.gauge_p_ref", "eos_gauge_p_ref"),
        ("eos.gauge_e_ref", "eos_gauge_e_ref"),
        ("eos.gauge_p_bias", "eos_gauge_p_bias"),
        ("eos.bc_pressure_inlet", "bc_pressure_inlet"),
        ("buoyant.beta_g", "buoyant_beta_g"),
        ("buoyant.t0", "buoyant_t0"),
        ("buoyant.k_over_cp", "buoyant_k_over_cp"),
    ];
    let Some(last_referenced) = canonical_tail_fields
        .iter()
        .rposition(|(_, field)| program_references_constants_field(program, field))
    else {
        return Vec::new();
    };
    canonical_tail_fields[..=last_referenced]
        .iter()
        .map(|&(key, field)| ParamSpec {
            key,
            wgsl_field: field,
            wgsl_type: "f32",
            unit: Dimensionless::UNIT,
        })
        .collect()
}

/// Rename identifiers in an expression tree using a rename map.
fn rename_expr(
    expr: &cfd2_ir::ast::Expr,
    rename_map: &BTreeMap<String, String>,
) -> cfd2_ir::ast::Expr {
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
        ExprNode::Field { base, field } => rename_expr(base, rename_map).field(field.clone()),
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
        ExprNode::Binary { left, op, right } => Expr::binary(
            rename_expr(left, rename_map),
            *op,
            rename_expr(right, rename_map),
        ),
        ExprNode::Call { callee, args } => {
            let new_callee = rename_expr(callee, rename_map);
            let new_args: Vec<Expr> = args.iter().map(|a| rename_expr(a, rename_map)).collect();
            Expr::call(new_callee, new_args)
        }
    }
}

/// Rename identifiers in a statement tree using a rename map.
fn rename_stmt(
    stmt: &cfd2_ir::ast::Stmt,
    rename_map: &BTreeMap<String, String>,
) -> cfd2_ir::ast::Stmt {
    use cfd2_ir::ast::Stmt;
    match stmt {
        Stmt::Comment(text) => Stmt::Comment(text.clone()),
        Stmt::Let { name, ty, expr } => Stmt::Let {
            name: rename_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone()),
            ty: ty.clone(),
            expr: rename_expr(expr, rename_map),
        },
        Stmt::Var { name, ty, expr } => Stmt::Var {
            name: rename_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone()),
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
        Stmt::If {
            cond,
            then_block,
            else_block,
        } => Stmt::If {
            cond: rename_expr(cond, rename_map),
            then_block: rename_block(then_block, rename_map),
            else_block: else_block.as_ref().map(|b| rename_block(b, rename_map)),
        },
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => Stmt::For {
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

fn rename_block(
    block: &cfd2_ir::ast::Block,
    rename_map: &BTreeMap<String, String>,
) -> cfd2_ir::ast::Block {
    cfd2_ir::ast::Block::new(
        block
            .stmts
            .iter()
            .map(|s| rename_stmt(s, rename_map))
            .collect(),
    )
}

fn rename_for_init(
    init: &cfd2_ir::ast::ForInit,
    rename_map: &BTreeMap<String, String>,
) -> cfd2_ir::ast::ForInit {
    use cfd2_ir::ast::ForInit;
    match init {
        ForInit::Let { name, ty, expr } => ForInit::Let {
            name: rename_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone()),
            ty: ty.clone(),
            expr: rename_expr(expr, rename_map),
        },
        ForInit::Var { name, ty, expr } => ForInit::Var {
            name: rename_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone()),
            ty: ty.clone(),
            expr: rename_expr(expr, rename_map),
        },
        ForInit::Assign { target, value } => ForInit::Assign {
            target: rename_expr(target, rename_map),
            value: rename_expr(value, rename_map),
        },
    }
}

fn rename_for_step(
    step: &cfd2_ir::ast::ForStep,
    rename_map: &BTreeMap<String, String>,
) -> cfd2_ir::ast::ForStep {
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
fn rename_stmts(
    stmts: &[cfd2_ir::ast::Stmt],
    rename_map: &BTreeMap<String, String>,
) -> Vec<cfd2_ir::ast::Stmt> {
    stmts.iter().map(|s| rename_stmt(s, rename_map)).collect()
}

/// Post-synthesis AST cleanup passes (semantics-preserving).
///
/// Runs conservative optimizations on the fused program body:
/// 1. Load-after-store forwarding (CSE for buffer accesses)
/// 2. Noop self-assign removal (`x = x` → removed)
fn apply_fusion_cleanup(program: &mut KernelProgram) {
    apply_ast_load_after_store_forwarding(program);
    apply_ast_noop_self_assign_cleanup(program);
}

/// Clone a typed kernel while renaming one storage binding and every typed AST
/// reference to it.  This is used by explicit stage ping-pong lowering: the
/// generated mathematics still refers to the model's logical `state`, while
/// each stage pipeline binds a fixed physical input role.
///
/// Raw helper WGSL is deliberately rejected when it mentions the binding.  A
/// string substitution there would evade the typed proof this transform is
/// intended to provide.
pub fn rename_program_buffer_binding(
    program: &KernelProgram,
    replacement_id: impl Into<String>,
    from: &str,
    to: &str,
) -> Result<KernelProgram, String> {
    validate_wgsl_identifier(to)?;
    if from == to {
        let mut out = program.clone();
        out.id = replacement_id.into();
        return Ok(out);
    }
    if program.launch.invocation_index_expr.contains(from)
        || program
            .launch
            .bounds_check_expr
            .as_ref()
            .is_some_and(|bounds| bounds.contains(from))
    {
        return Err(format!(
            "buffer rename rejected for '{}': raw launch semantics reference '{from}'",
            program.id
        ));
    }
    if program
        .helper_functions
        .iter()
        .any(|helper| helper.contains(from))
    {
        return Err(format!(
            "buffer rename rejected for '{}': raw helper WGSL references '{from}'",
            program.id
        ));
    }

    let mut out = program.clone();
    out.id = replacement_id.into();
    if out.bindings.iter().any(|binding| binding.name == to) {
        return Err(format!(
            "buffer rename for '{}' would collide with existing binding '{to}'",
            program.id
        ));
    }
    let mut found = 0usize;
    for binding in &mut out.bindings {
        if binding.name == from {
            binding.name = to.to_string();
            found += 1;
        }
    }
    if found != 1 {
        return Err(format!(
            "buffer rename for '{}' expected one '{from}' binding, found {found}",
            program.id
        ));
    }

    let rename = BTreeMap::from([(from.to_string(), to.to_string())]);
    out.indexing = rename_stmts(&out.indexing, &rename);
    out.preamble = rename_stmts(&out.preamble, &rename);
    out.body = rename_stmts(&out.body, &rename);
    Ok(out)
}

/// Fuse one matrix-free cell residual with one cell-local RK stage while:
///
/// - replacing the global `rhs` hand-off with invocation-local expressions;
/// - reading the spatial operator from an immutable stage input buffer; and
/// - writing the updated/closed state to a distinct stage output buffer.
///
/// The distinct output is a correctness requirement, not merely an
/// optimization.  An in-place fused dispatch has no grid-wide barrier: one
/// invocation could update a cell while a neighboring invocation still reads
/// that cell for its residual.
pub fn synthesize_explicit_residual_rk_ping_pong(
    replacement_id: impl Into<String>,
    residual: &KernelProgram,
    stage: &KernelProgram,
    input_state_binding: &str,
    output_state_binding: &str,
) -> Result<KernelProgram, String> {
    let replacement_id = replacement_id.into();
    validate_wgsl_identifier(input_state_binding)?;
    validate_wgsl_identifier(output_state_binding)?;
    if input_state_binding == output_state_binding {
        return Err(
            "explicit residual/RK fusion requires distinct input and output state buffers"
                .to_string(),
        );
    }
    if residual.dispatch != DispatchDomain::Cells || stage.dispatch != DispatchDomain::Cells {
        return Err(
            "explicit residual/RK fusion requires two cell-dispatched programs".to_string(),
        );
    }
    let residual_locals: BTreeSet<String> = residual.local_symbols().into_iter().collect();
    let stage_locals: BTreeSet<String> = stage.local_symbols().into_iter().collect();
    for binding_name in [input_state_binding, output_state_binding] {
        if residual_locals.contains(binding_name) || stage_locals.contains(binding_name) {
            return Err(format!(
                "explicit residual/RK binding name '{binding_name}' collides with a local symbol"
            ));
        }
    }
    if !residual.helper_functions.is_empty() || !stage.helper_functions.is_empty() {
        return Err(
            "explicit residual/RK fusion requires helper-free typed programs; raw helper symbol capture is unproven"
                .to_string(),
        );
    }
    if residual.launch != stage.launch {
        return Err(format!(
            "explicit residual/RK fusion launch mismatch: '{}' vs '{}'",
            residual.id, stage.id
        ));
    }
    // The residual owns the structured/unstructured cell-indexing preamble.
    // RK stages normally need only `idx`; accepting a second, different
    // indexing program would silently discard locals when ordinary fusion
    // keeps the first program's indexing block.
    if !stage.indexing.is_empty() && residual.indexing != stage.indexing {
        return Err(format!(
            "explicit residual/RK fusion indexing mismatch: '{}' vs '{}'",
            residual.id, stage.id
        ));
    }

    let mut residual = rename_program_buffer_binding(
        residual,
        format!("{}:input", residual.id),
        "state",
        input_state_binding,
    )?;
    if residual
        .body
        .iter()
        .any(|statement| stmt_writes_buffer(statement, input_state_binding))
    {
        return Err(format!(
            "explicit residual '{}' writes its stage input state",
            residual.id
        ));
    }
    if let Some(input) = residual
        .bindings
        .iter_mut()
        .find(|binding| binding.name == input_state_binding)
    {
        input.access = BindingAccess::ReadOnlyStorage;
    }
    let rhs_binding = remove_named_binding(&mut residual, "rhs")?;
    let rhs_values = extract_top_level_buffer_stores(&mut residual.body, "rhs")?;
    if rhs_values.is_empty() {
        return Err(format!(
            "explicit residual '{}' produced no invocation-local rhs values",
            residual.id
        ));
    }
    if stmts_reference_ident(&residual.preamble, "rhs")
        || stmts_reference_ident(&residual.indexing, "rhs")
        || stmts_reference_ident(&residual.body, "rhs")
        || residual
            .helper_functions
            .iter()
            .any(|helper| helper.contains("rhs"))
    {
        return Err(format!(
            "explicit residual '{}' has non-output rhs accesses",
            residual.id
        ));
    }
    remove_effect_binding(&mut residual, rhs_binding.group, rhs_binding.binding);

    let mut stage = alpha_rename_stage_locals(stage, &residual)?;
    let stage_rhs = remove_named_binding(&mut stage, "rhs")?;
    remove_effect_binding(&mut stage, stage_rhs.group, stage_rhs.binding);
    let stage_state = remove_named_binding(&mut stage, "state")?;
    remove_effect_binding(&mut stage, stage_state.group, stage_state.binding);
    for binding in residual.bindings.iter().chain(stage.bindings.iter()) {
        if binding.name == output_state_binding {
            return Err(format!(
                "explicit residual/RK output name '{output_state_binding}' collides with retained binding @group({}) @binding({})",
                binding.group, binding.binding
            ));
        }
        if binding.name == input_state_binding && !(binding.group == 1 && binding.binding == 0) {
            return Err(format!(
                "explicit residual/RK input name '{input_state_binding}' collides with retained binding @group({}) @binding({})",
                binding.group, binding.binding
            ));
        }
    }

    stage.indexing = rewrite_buffer_reads_in_stmts(&stage.indexing, "rhs", &rhs_values)?;
    stage.preamble = rewrite_buffer_reads_in_stmts(&stage.preamble, "rhs", &rhs_values)?;
    stage.body = rewrite_buffer_reads_in_stmts(&stage.body, "rhs", &rhs_values)?;
    if stmts_reference_ident(&stage.indexing, "rhs")
        || stmts_reference_ident(&stage.preamble, "rhs")
        || stmts_reference_ident(&stage.body, "rhs")
    {
        return Err(format!(
            "RK stage '{}' contains an rhs access that could not be privatized",
            stage.id
        ));
    }

    let mut written_state = BTreeSet::<BufferIndexKey>::new();
    stage.indexing = rewrite_state_reads_in_stmts(
        &stage.indexing,
        &written_state,
        input_state_binding,
        output_state_binding,
    )?;
    stage.preamble = rewrite_state_reads_in_stmts(
        &stage.preamble,
        &written_state,
        input_state_binding,
        output_state_binding,
    )?;
    stage.body = rewrite_top_level_state_writes(
        &stage.body,
        &mut written_state,
        input_state_binding,
        output_state_binding,
    )?;
    if written_state.is_empty() {
        return Err(format!(
            "RK stage '{}' writes no state components",
            stage.id
        ));
    }
    if stmts_reference_ident(&stage.indexing, "state")
        || stmts_reference_ident(&stage.preamble, "state")
        || stmts_reference_ident(&stage.body, "state")
    {
        return Err(format!(
            "RK stage '{}' contains a state access that could not be ping-pong lowered",
            stage.id
        ));
    }

    // Reuse the removed RHS slot when possible.  This keeps the interface
    // within the same per-stage storage-binding budget; if a residual happens
    // to occupy that slot, select the first deterministic free slot instead.
    let mut occupied = BTreeSet::new();
    for binding in residual.bindings.iter().chain(stage.bindings.iter()) {
        if binding.group == stage_rhs.group {
            occupied.insert(binding.binding);
        }
    }
    let output_slot = if !occupied.contains(&stage_rhs.binding) {
        stage_rhs.binding
    } else {
        (0..=u32::MAX)
            .find(|slot| !occupied.contains(slot))
            .ok_or_else(|| "explicit residual/RK fusion found no free output binding".to_string())?
    };
    stage.bindings.push(KernelBinding::new(
        stage_rhs.group,
        output_slot,
        output_state_binding,
        stage_state.wgsl_type,
        BindingAccess::ReadWriteStorage,
    ));
    stage.side_effects.write_set.insert(EffectResource {
        group: stage_rhs.group,
        binding: output_slot,
        component: None,
    });

    // Shared coupled interfaces deliberately declare history/iteration
    // buffers that static explicit residuals do not use.  Remove only those
    // bindings whose absence is proven by both typed AST and raw-helper scans;
    // this also prevents accidental physical aliasing with ping-pong scratch.
    for dead_candidate in ["state_old", "state_old_old", "state_iter"] {
        remove_unreferenced_binding(&mut residual, dead_candidate)?;
    }

    // After the dedicated cross-invocation proof above, ordinary safe fusion
    // still checks dispatch/launch/indexing, interface compatibility, barriers,
    // atomics, and any side-effect metadata unrelated to the privatized streams.
    synthesize_fused_program(
        replacement_id,
        "explicit:residual_rk_ping_pong_v1",
        &[residual, stage],
        FusionSafetyPolicy::Safe,
    )
}

fn remove_named_binding(program: &mut KernelProgram, name: &str) -> Result<KernelBinding, String> {
    let positions: Vec<usize> = program
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| (binding.name == name).then_some(index))
        .collect();
    if positions.len() != 1 {
        return Err(format!(
            "kernel '{}' expected one '{name}' binding, found {}",
            program.id,
            positions.len()
        ));
    }
    Ok(program.bindings.remove(positions[0]))
}

fn validate_wgsl_identifier(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err("WGSL binding identifier must not be empty".to_string());
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return Err(format!("'{name}' is not a valid WGSL binding identifier"));
    }
    // Core WGSL keywords plus reserved scalar/control words. This transform is
    // intentionally conservative because names become module-scope globals.
    if matches!(
        name,
        "alias"
            | "array"
            | "atomic"
            | "bool"
            | "break"
            | "case"
            | "const"
            | "const_assert"
            | "continue"
            | "continuing"
            | "default"
            | "diagnostic"
            | "discard"
            | "else"
            | "enable"
            | "false"
            | "f16"
            | "f32"
            | "fn"
            | "for"
            | "i32"
            | "if"
            | "let"
            | "loop"
            | "override"
            | "ptr"
            | "requires"
            | "return"
            | "struct"
            | "switch"
            | "true"
            | "u32"
            | "var"
            | "while"
    ) {
        return Err(format!("'{name}' is reserved by WGSL"));
    }
    Ok(())
}

/// Give the second program a namespace that is disjoint from every residual
/// local before residual expressions are substituted into its AST. Ordinary
/// fusion alpha-renames program 1 later; doing this first prevents that pass
/// from capturing an ident that originated in program 0.
fn alpha_rename_stage_locals(
    stage: &KernelProgram,
    residual: &KernelProgram,
) -> Result<KernelProgram, String> {
    let mut out = stage.clone();
    let forbidden: BTreeSet<String> = residual
        .local_symbols()
        .into_iter()
        .chain(stage.local_symbols())
        .chain(residual.bindings.iter().map(|binding| binding.name.clone()))
        .chain(stage.bindings.iter().map(|binding| binding.name.clone()))
        .collect();
    let mut map = BTreeMap::new();
    for (ordinal, local) in stage.local_symbols().into_iter().enumerate() {
        let mut candidate = format!("cfd2_stage_local_{ordinal}_{local}");
        let mut discriminator = 0usize;
        while forbidden.contains(&candidate) || map.values().any(|value| value == &candidate) {
            discriminator += 1;
            candidate = format!("cfd2_stage_local_{ordinal}_{discriminator}_{local}");
        }
        validate_wgsl_identifier(&candidate)?;
        map.insert(local, candidate);
    }
    out.preamble = rename_stmts(&out.preamble, &map);
    out.body = rename_stmts(&out.body, &map);
    Ok(out)
}

fn remove_effect_binding(program: &mut KernelProgram, group: u32, binding: u32) {
    program
        .side_effects
        .read_set
        .retain(|resource| resource.group != group || resource.binding != binding);
    program
        .side_effects
        .write_set
        .retain(|resource| resource.group != group || resource.binding != binding);
}

fn remove_unreferenced_binding(program: &mut KernelProgram, name: &str) -> Result<(), String> {
    let Some(binding) = program
        .bindings
        .iter()
        .find(|binding| binding.name == name)
        .cloned()
    else {
        return Ok(());
    };
    let referenced = stmts_reference_ident(&program.indexing, name)
        || stmts_reference_ident(&program.preamble, name)
        || stmts_reference_ident(&program.body, name)
        || program
            .helper_functions
            .iter()
            .any(|helper| helper.contains(name));
    if referenced {
        return Ok(());
    }
    let _ = remove_named_binding(program, name)?;
    remove_effect_binding(program, binding.group, binding.binding);
    Ok(())
}

fn extract_top_level_buffer_stores(
    statements: &mut Vec<cfd2_ir::ast::Stmt>,
    buffer: &str,
) -> Result<BTreeMap<BufferIndexKey, cfd2_ir::ast::Expr>, String> {
    use cfd2_ir::ast::{ExprNode, Stmt};
    let mut values = BTreeMap::new();
    let mut kept = Vec::with_capacity(statements.len());
    for statement in std::mem::take(statements) {
        if let Stmt::Assign { target, value } = &statement {
            if let ExprNode::Index { base, index } = target.node() {
                if matches!(base.node(), ExprNode::Ident(name) if name == buffer) {
                    let key = BufferIndexKey::from_expr(index);
                    if values.insert(key.clone(), value.clone()).is_some() {
                        return Err(format!(
                            "buffer '{buffer}' component '{key}' is stored more than once"
                        ));
                    }
                    continue;
                }
            }
        }
        if stmt_writes_buffer(&statement, buffer) {
            return Err(format!(
                "buffer '{buffer}' has a nested or compound write; local scalar replacement is unproven"
            ));
        }
        kept.push(statement);
    }
    *statements = kept;
    Ok(values)
}

fn rewrite_buffer_reads_in_stmts(
    statements: &[cfd2_ir::ast::Stmt],
    buffer: &str,
    values: &BTreeMap<BufferIndexKey, cfd2_ir::ast::Expr>,
) -> Result<Vec<cfd2_ir::ast::Stmt>, String> {
    statements
        .iter()
        .map(|statement| rewrite_stmt_buffer_reads(statement, buffer, values))
        .collect()
}

fn rewrite_stmt_buffer_reads(
    statement: &cfd2_ir::ast::Stmt,
    buffer: &str,
    values: &BTreeMap<BufferIndexKey, cfd2_ir::ast::Expr>,
) -> Result<cfd2_ir::ast::Stmt, String> {
    use cfd2_ir::ast::{Block, ForInit, ForStep, Stmt};
    let expr = |value: &cfd2_ir::ast::Expr| rewrite_expr_buffer_reads(value, buffer, values);
    let block = |value: &Block| -> Result<Block, String> {
        Ok(Block::new(rewrite_buffer_reads_in_stmts(
            &value.stmts,
            buffer,
            values,
        )?))
    };
    Ok(match statement {
        Stmt::Comment(text) => Stmt::Comment(text.clone()),
        Stmt::Let {
            name,
            ty,
            expr: value,
        } => Stmt::Let {
            name: name.clone(),
            ty: ty.clone(),
            expr: expr(value)?,
        },
        Stmt::Var {
            name,
            ty,
            expr: value,
        } => Stmt::Var {
            name: name.clone(),
            ty: ty.clone(),
            expr: value.as_ref().map(expr).transpose()?,
        },
        Stmt::Assign { target, value } => Stmt::Assign {
            target: expr(target)?,
            value: expr(value)?,
        },
        Stmt::AssignOp { target, op, value } => Stmt::AssignOp {
            target: expr(target)?,
            op: *op,
            value: expr(value)?,
        },
        Stmt::If {
            cond,
            then_block,
            else_block,
        } => Stmt::If {
            cond: expr(cond)?,
            then_block: block(then_block)?,
            else_block: else_block.as_ref().map(block).transpose()?,
        },
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => Stmt::For {
            init: match init {
                ForInit::Let {
                    name,
                    ty,
                    expr: value,
                } => ForInit::Let {
                    name: name.clone(),
                    ty: ty.clone(),
                    expr: expr(value)?,
                },
                ForInit::Var {
                    name,
                    ty,
                    expr: value,
                } => ForInit::Var {
                    name: name.clone(),
                    ty: ty.clone(),
                    expr: expr(value)?,
                },
                ForInit::Assign { target, value } => ForInit::Assign {
                    target: expr(target)?,
                    value: expr(value)?,
                },
            },
            cond: expr(cond)?,
            step: match step {
                ForStep::Increment(value) => ForStep::Increment(expr(value)?),
                ForStep::Decrement(value) => ForStep::Decrement(expr(value)?),
                ForStep::Assign { target, value } => ForStep::Assign {
                    target: expr(target)?,
                    value: expr(value)?,
                },
                ForStep::AssignOp { target, op, value } => ForStep::AssignOp {
                    target: expr(target)?,
                    op: *op,
                    value: expr(value)?,
                },
            },
            body: block(body)?,
        },
        Stmt::Loop { body } => Stmt::Loop { body: block(body)? },
        Stmt::While { cond, body } => Stmt::While {
            cond: expr(cond)?,
            body: block(body)?,
        },
        Stmt::Break => Stmt::Break,
        Stmt::Continue => Stmt::Continue,
        Stmt::Return(value) => Stmt::Return(value.as_ref().map(expr).transpose()?),
        Stmt::Call(value) => Stmt::Call(expr(value)?),
        Stmt::Increment(value) => Stmt::Increment(expr(value)?),
        Stmt::Decrement(value) => Stmt::Decrement(expr(value)?),
    })
}

fn rewrite_expr_buffer_reads(
    expression: &cfd2_ir::ast::Expr,
    buffer: &str,
    values: &BTreeMap<BufferIndexKey, cfd2_ir::ast::Expr>,
) -> Result<cfd2_ir::ast::Expr, String> {
    use cfd2_ir::ast::{Expr, ExprNode};
    if let ExprNode::Index { base, index } = expression.node() {
        if matches!(base.node(), ExprNode::Ident(name) if name == buffer) {
            let key = BufferIndexKey::from_expr(index);
            return values.get(&key).cloned().ok_or_else(|| {
                format!("buffer '{buffer}' read '{key}' has no dominating local producer")
            });
        }
    }
    Ok(match expression.node() {
        ExprNode::Ident(_) | ExprNode::Literal(_) => expression.clone(),
        ExprNode::Field { base, field } => {
            rewrite_expr_buffer_reads(base, buffer, values)?.field(field.clone())
        }
        ExprNode::Index { base, index } => rewrite_expr_buffer_reads(base, buffer, values)?
            .index(rewrite_expr_buffer_reads(index, buffer, values)?),
        ExprNode::Unary { op, expr } => Expr::alloc_node(ExprNode::Unary {
            op: *op,
            expr: rewrite_expr_buffer_reads(expr, buffer, values)?,
        }),
        ExprNode::Binary { left, op, right } => Expr::binary(
            rewrite_expr_buffer_reads(left, buffer, values)?,
            *op,
            rewrite_expr_buffer_reads(right, buffer, values)?,
        ),
        ExprNode::Call { callee, args } => Expr::call(
            rewrite_expr_buffer_reads(callee, buffer, values)?,
            args.iter()
                .map(|arg| rewrite_expr_buffer_reads(arg, buffer, values))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

fn rewrite_top_level_state_writes(
    statements: &[cfd2_ir::ast::Stmt],
    written: &mut BTreeSet<BufferIndexKey>,
    input: &str,
    output: &str,
) -> Result<Vec<cfd2_ir::ast::Stmt>, String> {
    use cfd2_ir::ast::{Expr, ExprNode, Stmt};
    let mut result = Vec::with_capacity(statements.len());
    for statement in statements {
        if let Stmt::Assign { target, value } = statement {
            if let ExprNode::Index { base, index } = target.node() {
                if matches!(base.node(), ExprNode::Ident(name) if name == "state") {
                    let key = BufferIndexKey::from_expr(index);
                    let value = rewrite_state_read_expr(value, written, input, output)?;
                    result.push(Stmt::Assign {
                        target: Expr::ident(output).index(index.clone()),
                        value,
                    });
                    written.insert(key);
                    continue;
                }
            }
        }
        if stmt_writes_buffer(statement, "state") {
            return Err(
                "RK stage has a nested or compound state write; ping-pong dominance is unproven"
                    .to_string(),
            );
        }
        result.push(rewrite_state_read_stmt(statement, written, input, output)?);
    }
    Ok(result)
}

fn rewrite_state_reads_in_stmts(
    statements: &[cfd2_ir::ast::Stmt],
    written: &BTreeSet<BufferIndexKey>,
    input: &str,
    output: &str,
) -> Result<Vec<cfd2_ir::ast::Stmt>, String> {
    statements
        .iter()
        .map(|statement| rewrite_state_read_stmt(statement, written, input, output))
        .collect()
}

fn rewrite_state_read_stmt(
    statement: &cfd2_ir::ast::Stmt,
    written: &BTreeSet<BufferIndexKey>,
    input: &str,
    output: &str,
) -> Result<cfd2_ir::ast::Stmt, String> {
    rewrite_stmt_with_expr(statement, &|expr| {
        rewrite_state_read_expr(expr, written, input, output)
    })
}

fn rewrite_stmt_with_expr(
    statement: &cfd2_ir::ast::Stmt,
    rewrite: &impl Fn(&cfd2_ir::ast::Expr) -> Result<cfd2_ir::ast::Expr, String>,
) -> Result<cfd2_ir::ast::Stmt, String> {
    use cfd2_ir::ast::{Block, ForInit, ForStep, Stmt};
    let block = |value: &Block| -> Result<Block, String> {
        Ok(Block::new(
            value
                .stmts
                .iter()
                .map(|statement| rewrite_stmt_with_expr(statement, rewrite))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    };
    Ok(match statement {
        Stmt::Comment(text) => Stmt::Comment(text.clone()),
        Stmt::Let { name, ty, expr } => Stmt::Let {
            name: name.clone(),
            ty: ty.clone(),
            expr: rewrite(expr)?,
        },
        Stmt::Var { name, ty, expr } => Stmt::Var {
            name: name.clone(),
            ty: ty.clone(),
            expr: expr.as_ref().map(rewrite).transpose()?,
        },
        Stmt::Assign { target, value } => Stmt::Assign {
            target: rewrite(target)?,
            value: rewrite(value)?,
        },
        Stmt::AssignOp { target, op, value } => Stmt::AssignOp {
            target: rewrite(target)?,
            op: *op,
            value: rewrite(value)?,
        },
        Stmt::If {
            cond,
            then_block,
            else_block,
        } => Stmt::If {
            cond: rewrite(cond)?,
            then_block: block(then_block)?,
            else_block: else_block.as_ref().map(block).transpose()?,
        },
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => Stmt::For {
            init: match init {
                ForInit::Let { name, ty, expr } => ForInit::Let {
                    name: name.clone(),
                    ty: ty.clone(),
                    expr: rewrite(expr)?,
                },
                ForInit::Var { name, ty, expr } => ForInit::Var {
                    name: name.clone(),
                    ty: ty.clone(),
                    expr: rewrite(expr)?,
                },
                ForInit::Assign { target, value } => ForInit::Assign {
                    target: rewrite(target)?,
                    value: rewrite(value)?,
                },
            },
            cond: rewrite(cond)?,
            step: match step {
                ForStep::Increment(value) => ForStep::Increment(rewrite(value)?),
                ForStep::Decrement(value) => ForStep::Decrement(rewrite(value)?),
                ForStep::Assign { target, value } => ForStep::Assign {
                    target: rewrite(target)?,
                    value: rewrite(value)?,
                },
                ForStep::AssignOp { target, op, value } => ForStep::AssignOp {
                    target: rewrite(target)?,
                    op: *op,
                    value: rewrite(value)?,
                },
            },
            body: block(body)?,
        },
        Stmt::Loop { body } => Stmt::Loop { body: block(body)? },
        Stmt::While { cond, body } => Stmt::While {
            cond: rewrite(cond)?,
            body: block(body)?,
        },
        Stmt::Break => Stmt::Break,
        Stmt::Continue => Stmt::Continue,
        Stmt::Return(value) => Stmt::Return(value.as_ref().map(rewrite).transpose()?),
        Stmt::Call(value) => Stmt::Call(rewrite(value)?),
        Stmt::Increment(value) => Stmt::Increment(rewrite(value)?),
        Stmt::Decrement(value) => Stmt::Decrement(rewrite(value)?),
    })
}

fn rewrite_state_read_expr(
    expression: &cfd2_ir::ast::Expr,
    written: &BTreeSet<BufferIndexKey>,
    input: &str,
    output: &str,
) -> Result<cfd2_ir::ast::Expr, String> {
    use cfd2_ir::ast::{Expr, ExprNode};
    if let ExprNode::Index { base, index } = expression.node() {
        if matches!(base.node(), ExprNode::Ident(name) if name == "state") {
            let key = BufferIndexKey::from_expr(index);
            let binding = if written.contains(&key) {
                output
            } else {
                input
            };
            return Ok(Expr::ident(binding).index(index.clone()));
        }
    }
    Ok(match expression.node() {
        ExprNode::Ident(_) | ExprNode::Literal(_) => expression.clone(),
        ExprNode::Field { base, field } => {
            rewrite_state_read_expr(base, written, input, output)?.field(field.clone())
        }
        ExprNode::Index { base, index } => rewrite_state_read_expr(base, written, input, output)?
            .index(rewrite_state_read_expr(index, written, input, output)?),
        ExprNode::Unary { op, expr } => Expr::alloc_node(ExprNode::Unary {
            op: *op,
            expr: rewrite_state_read_expr(expr, written, input, output)?,
        }),
        ExprNode::Binary { left, op, right } => Expr::binary(
            rewrite_state_read_expr(left, written, input, output)?,
            *op,
            rewrite_state_read_expr(right, written, input, output)?,
        ),
        ExprNode::Call { callee, args } => Expr::call(
            rewrite_state_read_expr(callee, written, input, output)?,
            args.iter()
                .map(|arg| rewrite_state_read_expr(arg, written, input, output))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum BufferIndexKey {
    /// Canonical `cell * stride + offset`; an omitted `+ 0` has the same key.
    Affine {
        cell: String,
        stride: u32,
        offset: u32,
    },
    /// Conservative fallback: only syntactically identical unsupported forms
    /// are considered the same access.
    Exact(String),
}

impl BufferIndexKey {
    fn from_expr(expression: &cfd2_ir::ast::Expr) -> Self {
        use cfd2_ir::ast::{BinaryOp, ExprNode, Literal};

        let uint = |expr: &cfd2_ir::ast::Expr| match expr.node() {
            ExprNode::Literal(Literal::Uint(value)) => Some(*value),
            _ => None,
        };
        let product = |expr: &cfd2_ir::ast::Expr| match expr.node() {
            ExprNode::Binary {
                left,
                op: BinaryOp::Mul,
                right,
            } => {
                if let Some(stride) = uint(right) {
                    Some((left.to_string(), stride))
                } else {
                    uint(left).map(|stride| (right.to_string(), stride))
                }
            }
            _ => None,
        };

        match expression.node() {
            ExprNode::Binary {
                left,
                op: BinaryOp::Add,
                right,
            } => {
                if let (Some((cell, stride)), Some(offset)) = (product(left), uint(right)) {
                    Self::Affine {
                        cell,
                        stride,
                        offset,
                    }
                } else if let (Some(offset), Some((cell, stride))) = (uint(left), product(right)) {
                    Self::Affine {
                        cell,
                        stride,
                        offset,
                    }
                } else {
                    Self::Exact(expression.to_string())
                }
            }
            _ => product(expression)
                .map(|(cell, stride)| Self::Affine {
                    cell,
                    stride,
                    offset: 0,
                })
                .unwrap_or_else(|| Self::Exact(expression.to_string())),
        }
    }
}

impl std::fmt::Display for BufferIndexKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Affine {
                cell,
                stride,
                offset,
            } => write!(formatter, "{cell}*{stride}+{offset}"),
            Self::Exact(expression) => formatter.write_str(expression),
        }
    }
}

fn stmts_reference_ident(statements: &[cfd2_ir::ast::Stmt], ident: &str) -> bool {
    statements
        .iter()
        .any(|statement| stmt_references_ident(statement, ident))
}

fn stmt_writes_buffer(statement: &cfd2_ir::ast::Stmt, buffer: &str) -> bool {
    use cfd2_ir::ast::{ExprNode, ForInit, ForStep, Stmt};
    let target_is_buffer = |target: &cfd2_ir::ast::Expr| {
        matches!(
            target.node(),
            ExprNode::Index { base, .. }
                if matches!(base.node(), ExprNode::Ident(name) if name == buffer)
        )
    };
    match statement {
        Stmt::Assign { target, .. }
        | Stmt::AssignOp { target, .. }
        | Stmt::Increment(target)
        | Stmt::Decrement(target) => target_is_buffer(target),
        Stmt::If {
            then_block,
            else_block,
            ..
        } => {
            then_block
                .stmts
                .iter()
                .any(|s| stmt_writes_buffer(s, buffer))
                || else_block
                    .as_ref()
                    .is_some_and(|block| block.stmts.iter().any(|s| stmt_writes_buffer(s, buffer)))
        }
        Stmt::For {
            init, step, body, ..
        } => {
            let init_writes = matches!(
                init,
                ForInit::Assign { target, .. } if target_is_buffer(target)
            );
            let step_writes = match step {
                ForStep::Increment(target)
                | ForStep::Decrement(target)
                | ForStep::Assign { target, .. }
                | ForStep::AssignOp { target, .. } => target_is_buffer(target),
            };
            init_writes || step_writes || body.stmts.iter().any(|s| stmt_writes_buffer(s, buffer))
        }
        Stmt::Loop { body } | Stmt::While { body, .. } => {
            body.stmts.iter().any(|s| stmt_writes_buffer(s, buffer))
        }
        _ => false,
    }
}

fn stmt_references_ident(statement: &cfd2_ir::ast::Stmt, ident: &str) -> bool {
    use cfd2_ir::ast::{ForInit, ForStep, Stmt};
    let expr = |value: &cfd2_ir::ast::Expr| expr_references_ident(value, ident);
    match statement {
        Stmt::Comment(_) | Stmt::Break | Stmt::Continue => false,
        Stmt::Let { expr: value, .. } => expr(value),
        Stmt::Var { expr: value, .. } => value.as_ref().is_some_and(expr),
        Stmt::Assign { target, value } | Stmt::AssignOp { target, value, .. } => {
            expr(target) || expr(value)
        }
        Stmt::If {
            cond,
            then_block,
            else_block,
        } => {
            expr(cond)
                || stmts_reference_ident(&then_block.stmts, ident)
                || else_block
                    .as_ref()
                    .is_some_and(|block| stmts_reference_ident(&block.stmts, ident))
        }
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => {
            let init = match init {
                ForInit::Let { expr: value, .. } | ForInit::Var { expr: value, .. } => expr(value),
                ForInit::Assign { target, value } => expr(target) || expr(value),
            };
            let step = match step {
                ForStep::Increment(value) | ForStep::Decrement(value) => expr(value),
                ForStep::Assign { target, value } | ForStep::AssignOp { target, value, .. } => {
                    expr(target) || expr(value)
                }
            };
            init || expr(cond) || step || stmts_reference_ident(&body.stmts, ident)
        }
        Stmt::Loop { body } => stmts_reference_ident(&body.stmts, ident),
        Stmt::While { cond, body } => expr(cond) || stmts_reference_ident(&body.stmts, ident),
        Stmt::Return(value) => value.as_ref().is_some_and(expr),
        Stmt::Call(value) | Stmt::Increment(value) | Stmt::Decrement(value) => expr(value),
    }
}

fn expr_references_ident(expression: &cfd2_ir::ast::Expr, ident: &str) -> bool {
    use cfd2_ir::ast::ExprNode;
    match expression.node() {
        ExprNode::Ident(name) => name == ident,
        ExprNode::Literal(_) => false,
        ExprNode::Field { base, .. } | ExprNode::Unary { expr: base, .. } => {
            expr_references_ident(base, ident)
        }
        ExprNode::Index { base, index }
        | ExprNode::Binary {
            left: base,
            right: index,
            ..
        } => expr_references_ident(base, ident) || expr_references_ident(index, ident),
        ExprNode::Call { callee, args } => {
            expr_references_ident(callee, ident)
                || args.iter().any(|arg| expr_references_ident(arg, ident))
        }
    }
}

/// AST-based load-after-store forwarding.
///
/// Walks the flat body statement list. For each store (`Assign` to a
/// buffer index), records the stored value expression. When a subsequent
/// `Let` loads from the same buffer+index, replaces the load expression with
/// the forwarded value. Control-flow and non-trivial ops conservatively
/// invalidate the store map.
fn apply_ast_load_after_store_forwarding(program: &mut KernelProgram) {
    use cfd2_ir::ast::{Expr, Stmt};

    // Key: (base_name, index_expr_str) -> forwarded value Expr
    let mut last_store: BTreeMap<(String, String), Expr> = BTreeMap::new();

    for stmt in program.body.iter_mut() {
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
/// same identifier.
fn apply_ast_noop_self_assign_cleanup(program: &mut KernelProgram) {
    use cfd2_ir::ast::{ExprNode, Stmt};

    program.body.retain(|stmt| {
        if let Stmt::Assign { target, value } = stmt {
            if let (ExprNode::Ident(lhs), ExprNode::Ident(rhs)) = (target.node(), value.node()) {
                if lhs == rhs {
                    return false;
                }
            }
        }
        true
    });
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
    let mut accesses = Vec::new();
    ast_collect_buffer_accesses_recursive(expr, &mut accesses);
    accesses
}

fn ast_collect_buffer_accesses_recursive(
    expr: &cfd2_ir::ast::Expr,
    out: &mut Vec<(String, String)>,
) {
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
    // Structured (Cartesian) kernels bind the `grid: StructuredGrid` uniform in
    // place of the connectivity buffers.
    let needs_structured_grid_struct = by_slot
        .values()
        .any(|binding| binding.wgsl_type == "StructuredGrid");
    // A Vector2 is needed either by a Vector2-typed binding OR by the structured
    // kernels' arithmetic geometry (cell/face centres, normals), which construct
    // `Vector2(...)` in the body without any Vector2-typed binding.
    let needs_vector2_struct = needs_structured_grid_struct
        || by_slot
            .values()
            .any(|binding| binding.wgsl_type.contains("Vector2"));
    let needs_low_mach_params_struct = by_slot
        .values()
        .any(|binding| binding.wgsl_type == "LowMachParams");
    if needs_constants_struct
        || needs_vector2_struct
        || needs_low_mach_params_struct
        || needs_structured_grid_struct
    {
        let mut shared_structs_module = super::wgsl_ast::Module::new();
        if needs_vector2_struct {
            shared_structs_module.push(super::wgsl_ast::Item::Struct(
                super::wgsl_bindings::vector2_struct(),
            ));
        }
        if needs_structured_grid_struct {
            shared_structs_module.push(super::wgsl_ast::Item::Struct(
                super::coupled_common::structured_grid_struct(),
            ));
        }
        if needs_constants_struct {
            // Prefer the structured eos_params declaration; fall back to the
            // AST string-scan heuristic when it is empty.
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

    for line in cfd2_ir::ast::stmt::render_stmt_lines(&program.indexing) {
        lines.push(format!("    {line}"));
    }

    for line in cfd2_ir::ast::stmt::render_stmt_lines(&program.preamble) {
        lines.push(format!("    {line}"));
    }

    for line in cfd2_ir::ast::stmt::render_stmt_lines(&program.body) {
        lines.push(format!("    {line}"));
    }

    lines.push("}".to_string());
    lines.push(String::new());

    Ok(KernelWgsl::from_source(lines.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd2_ir::kernel::{DispatchDomain, LaunchSemantics};

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
        program.preamble = vec![Stmt::Var {
            name: "value".to_string(),
            ty: Some(Type::F32),
            expr: Some(Expr::lit_f32(0.0)),
        }];
        program.body = vec![
            Stmt::Assign {
                target: Expr::ident("value"),
                value: Expr::ident("value") + Expr::lit_f32(1.0),
            },
            Stmt::Assign {
                target: Expr::ident("state").index(Expr::ident("idx")),
                value: Expr::ident("value"),
            },
        ];
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
        let rendered = cfd2_ir::ast::stmt::render_stmt_lines(&fused.body);
        let joined = rendered.join("\n");
        assert!(
            joined.contains("value = value + 1.0;"),
            "missing original value assignment in:\n{joined}"
        );
        assert!(
            joined.contains("k1_value = k1_value + 1.0;"),
            "missing renamed value assignment in:\n{joined}"
        );
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
        program.body = vec![cfd2_ir::ast::Stmt::Assign {
            target: cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::ident("idx")),
            value: cfd2_ir::ast::Expr::ident("cell_centers")
                .index(cfd2_ir::ast::Expr::ident("idx"))
                .field("x"),
        }];

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
        use cfd2_ir::ast::{Expr, Stmt};
        let mut a = sample_program("a");
        a.body.insert(
            0,
            Stmt::Assign {
                target: Expr::ident("value"),
                value: Expr::ident("value"),
            },
        );
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

        let is_noop = |stmt: &Stmt| -> bool {
            matches!(stmt, Stmt::Assign { target, value }
                if matches!((target.node(), value.node()),
                    (cfd2_ir::ast::ExprNode::Ident(l), cfd2_ir::ast::ExprNode::Ident(r)) if l == r))
        };

        assert!(
            safe.body.iter().any(is_noop),
            "safe policy should preserve no-op local assignment"
        );
        assert!(
            aggressive.body.iter().all(|s| !is_noop(s)),
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
    fn aggressive_cleanup_forwards_store_to_load() {
        use cfd2_ir::ast::{Expr, Stmt};
        let launch = LaunchSemantics::new([64, 1, 1], "idx", Some("idx >= n"));
        let mut program = KernelProgram::new(
            "forward_test",
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
        program.body = vec![
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

        apply_fusion_cleanup(&mut program);
        match &program.body[1] {
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
            "forward_guard_test",
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
        program.body = vec![
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

        apply_fusion_cleanup(&mut program);
        match &program.body[1] {
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
            "forward_invalidate_test",
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
        program.body = vec![
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

        apply_fusion_cleanup(&mut program);
        match &program.body[2] {
            Stmt::Let { name, expr, .. } => {
                assert_eq!(name, "out");
                assert_eq!(expr.to_string(), "state[idx]");
            }
            other => panic!("expected Let, got {:?}", other),
        }
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
    fn aggressive_policy_succeeds_with_whitelisted_hazards_and_returns_report() {
        let mut a = sample_program("a");
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));

        let expected = vec![ExpectedHazard {
            kind: HazardKind::RAW,
            kernel_id: "b",
            justification: "test: disjoint index ranges",
        }];

        let result = synthesize_fused_program_with_report_remapped(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
            &[],
            &expected,
        );
        let (program, hazards) =
            result.expect("aggressive synthesis should succeed with whitelisted hazards");
        assert!(!hazards.is_empty(), "hazards should be reported");
        assert_eq!(hazards[0].kind, HazardKind::RAW);
        assert!(!program.body.is_empty());
    }

    #[test]
    fn aggressive_policy_rejects_non_whitelisted_hazards() {
        let mut a = sample_program("a");
        a.side_effects
            .write_set
            .insert(EffectResource::binding(0, 0));

        let mut b = sample_program("b");
        b.side_effects
            .read_set
            .insert(EffectResource::binding(0, 0));

        let err = synthesize_fused_program_with_report_remapped(
            "fused",
            "rule/a_b",
            &[a.clone(), b.clone()],
            FusionSafetyPolicy::Aggressive,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(
            err.contains("not in expected_hazards whitelist"),
            "unexpected error: {err}"
        );

        let wrong_whitelist = vec![ExpectedHazard {
            kind: HazardKind::WAW,
            kernel_id: "b",
            justification: "wrong kind",
        }];
        let err2 = synthesize_fused_program_with_report_remapped(
            "fused",
            "rule/a_b",
            &[a, b],
            FusionSafetyPolicy::Aggressive,
            &[],
            &wrong_whitelist,
        )
        .unwrap_err();
        assert!(
            err2.contains("not in expected_hazards whitelist"),
            "unexpected error: {err2}"
        );
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
        use cfd2_ir::dimensions::Dimensionless;
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
            cfd2_ir::ast::Stmt::Let {
                name: "c".to_string(),
                ty: None,
                expr: cfd2_ir::ast::Expr::ident("constants").field("eos_gamma"),
            },
            cfd2_ir::ast::Stmt::Assign {
                target: cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::ident("idx")),
                value: cfd2_ir::ast::Expr::ident("c"),
            },
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
    fn lowering_detects_eos_from_ast_when_eos_params_empty() {
        let launch = LaunchSemantics::new(
            [64, 1, 1],
            "global_id.y * constants.stride_x + global_id.x",
            Some("idx >= arrayLength(&state) / 2u"),
        );
        let mut program = KernelProgram::new(
            "ast_eos_test",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
            ],
        );
        // No eos_params set (empty vec), but body references eos_gamma via AST
        program.body = vec![
            cfd2_ir::ast::Stmt::Let {
                name: "c".to_string(),
                ty: None,
                expr: cfd2_ir::ast::Expr::ident("constants").field("eos_gamma"),
            },
            cfd2_ir::ast::Stmt::Assign {
                target: cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::ident("idx")),
                value: cfd2_ir::ast::Expr::ident("c"),
            },
        ];
        assert!(program.eos_params.is_empty());

        let wgsl = lower_kernel_program_to_wgsl(&program).expect("lowering should succeed");
        let src = wgsl.to_wgsl();
        assert!(
            src.contains("eos_gamma: f32"),
            "AST search should detect eos_gamma in body"
        );
    }

    #[test]
    fn ast_detected_constants_keep_the_shared_host_tail_prefix() {
        let launch = LaunchSemantics::new([64, 1, 1], "global_id.x", None::<String>);
        let mut program = KernelProgram::new(
            "ast_eos_prefix_test",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
            ],
        );
        program.body = vec![cfd2_ir::ast::Stmt::Assign {
            target: cfd2_ir::ast::Expr::ident("state").index(cfd2_ir::ast::Expr::ident("idx")),
            value: cfd2_ir::ast::Expr::ident("constants").field("eos_rho_ref"),
        }];

        let src = lower_kernel_program_to_wgsl(&program)
            .expect("lowering should succeed")
            .to_wgsl();
        let gamma = src.find("eos_gamma: f32").expect("gamma prefix");
        let gm1 = src.find("eos_gm1: f32").expect("gm1 prefix");
        let gas_constant = src.find("eos_r: f32").expect("R prefix");
        let dp_drho = src.find("eos_dp_drho: f32").expect("dp/drho prefix");
        let p_ref = src.find("eos_p_ref: f32").expect("pressure reference prefix");
        let theta_ref = src
            .find("eos_theta_ref: f32")
            .expect("theta reference prefix");
        let rho_ref = src
            .find("eos_rho_ref: f32")
            .expect("referenced density reference");
        assert!(
            gamma < gm1
                && gm1 < gas_constant
                && gas_constant < dp_drho
                && dp_drho < p_ref
                && p_ref < theta_ref
                && theta_ref < rho_ref,
            "noncanonical tail:\n{src}"
        );
    }

    #[test]
    fn fusion_merges_eos_params_from_input_programs() {
        use cfd2_ir::dimensions::Dimensionless;
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

    fn explicit_residual_stage_fixture() -> (KernelProgram, KernelProgram) {
        use cfd2_ir::ast::{Expr, Stmt, Type};

        let launch = LaunchSemantics::new([64, 1, 1], "global_id.x", None::<String>);
        let mut residual = KernelProgram::new(
            "residual",
            DispatchDomain::Cells,
            launch.clone(),
            vec![
                KernelBinding::new(1, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(
                    1,
                    1,
                    "state_old",
                    "array<f32>",
                    BindingAccess::ReadOnlyStorage,
                ),
                KernelBinding::new(
                    1,
                    2,
                    "state_old_old",
                    "array<f32>",
                    BindingAccess::ReadOnlyStorage,
                ),
                KernelBinding::new(
                    1,
                    4,
                    "state_iter",
                    "array<f32>",
                    BindingAccess::ReadOnlyStorage,
                ),
                KernelBinding::new(2, 0, "rhs", "array<f32>", BindingAccess::ReadWriteStorage),
            ],
        );
        let idx0 = Expr::ident("idx") * Expr::lit_u32(2) + Expr::lit_u32(0);
        let idx1 = Expr::ident("idx") * Expr::lit_u32(2) + Expr::lit_u32(1);
        residual.body = vec![
            Stmt::Let {
                name: "r0".into(),
                ty: Some(Type::F32),
                expr: Expr::ident("state").index(Expr::ident("idx") * Expr::lit_u32(3))
                    * Expr::lit_f32(2.0),
            },
            Stmt::Assign {
                target: Expr::ident("rhs").index(idx0.clone()),
                value: Expr::ident("r0"),
            },
            Stmt::Assign {
                target: Expr::ident("rhs").index(idx1.clone()),
                value: Expr::lit_f32(0.0),
            },
        ];

        let mut stage = KernelProgram::new(
            "stage",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(1, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
                KernelBinding::new(2, 0, "rhs", "array<f32>", BindingAccess::ReadOnlyStorage),
                KernelBinding::new(
                    2,
                    1,
                    "rk_base",
                    "array<f32>",
                    BindingAccess::ReadWriteStorage,
                ),
                KernelBinding::new(
                    2,
                    2,
                    "rk_accum",
                    "array<f32>",
                    BindingAccess::ReadWriteStorage,
                ),
            ],
        );
        let state0 = Expr::ident("idx") * Expr::lit_u32(3) + Expr::lit_u32(0);
        let state1 = Expr::ident("idx") * Expr::lit_u32(3) + Expr::lit_u32(1);
        let state2 = Expr::ident("idx") * Expr::lit_u32(3) + Expr::lit_u32(2);
        stage.body = vec![
            Stmt::Let {
                name: "rate".into(),
                ty: Some(Type::F32),
                expr: Expr::ident("rhs").index(idx0),
            },
            Stmt::Assign {
                target: Expr::ident("state").index(state0.clone()),
                value: Expr::ident("state").index(state0.clone()) + Expr::ident("rate"),
            },
            // A closure expression must observe the just-written differential
            // component from the output, but an immutable coefficient from the
            // input buffer.
            Stmt::Assign {
                target: Expr::ident("state").index(state1),
                value: Expr::ident("state").index(state0) + Expr::ident("state").index(state2),
            },
        ];
        (residual, stage)
    }

    #[test]
    fn explicit_ping_pong_fusion_privatises_rhs_and_routes_state_dominance() {
        let (residual, stage) = explicit_residual_stage_fixture();
        let fused = synthesize_explicit_residual_rk_ping_pong(
            "fused_stage",
            &residual,
            &stage,
            "rk_input_state",
            "rk_output_state",
        )
        .expect("ping-pong fusion");
        let src = lower_kernel_program_to_wgsl(&fused)
            .expect("lower fused WGSL")
            .to_wgsl();

        assert!(!fused.bindings.iter().any(|binding| binding.name == "rhs"));
        assert!(!fused.bindings.iter().any(|binding| {
            matches!(
                binding.name.as_str(),
                "state_old" | "state_old_old" | "state_iter"
            )
        }));
        assert!(src.contains("var<storage, read> rk_input_state: array<f32>"));
        assert!(src.contains("var<storage, read_write> rk_output_state: array<f32>"));
        assert!(
            !src.contains(" rhs:"),
            "global rhs hand-off survived:\n{src}"
        );
        assert!(
            src.contains("rk_output_state[idx * 3u + 0u] = rk_input_state[idx * 3u + 0u] + k1_cfd2_stage_local_"),
            "differential write did not read input state:\n{src}"
        );
        assert!(
            src.contains("rk_output_state[idx * 3u + 1u] = rk_output_state[idx * 3u + 0u] + rk_input_state[idx * 3u + 2u]"),
            "closure dominance routing is wrong:\n{src}"
        );
    }

    #[test]
    fn explicit_ping_pong_fusion_rejects_alias_and_unproven_control_flow() {
        use cfd2_ir::ast::{Block, Expr, Stmt};
        let (residual, stage) = explicit_residual_stage_fixture();
        let alias_error = synthesize_explicit_residual_rk_ping_pong(
            "bad_alias",
            &residual,
            &stage,
            "state_a",
            "state_a",
        )
        .unwrap_err();
        assert!(alias_error.contains("distinct input and output"));

        let mut nested = residual;
        let rhs_store = nested.body.pop().expect("fixture rhs store");
        nested.body.push(Stmt::If {
            cond: Expr::lit_bool(true),
            then_block: Block::new(vec![rhs_store]),
            else_block: None,
        });
        let nested_error = synthesize_explicit_residual_rk_ping_pong(
            "bad_nested",
            &nested,
            &stage,
            "state_a",
            "state_b",
        )
        .unwrap_err();
        assert!(nested_error.contains("nested or compound write"));
    }

    #[test]
    fn explicit_ping_pong_fusion_rejects_global_name_collisions_and_invalid_names() {
        let (residual, stage) = explicit_residual_stage_fixture();
        for (input, output) in [
            ("rk_base", "stage_out"),
            ("stage_in", "rk_base"),
            ("r0", "stage_out"),
            ("stage_in", "rate"),
        ] {
            let error = synthesize_explicit_residual_rk_ping_pong(
                "bad_name_collision",
                &residual,
                &stage,
                input,
                output,
            )
            .unwrap_err();
            assert!(
                error.contains("collide") || error.contains("binding name"),
                "unexpected collision error for {input}->{output}: {error}"
            );
        }
        for invalid in ["1stage", "var", "has-dash"] {
            let error = synthesize_explicit_residual_rk_ping_pong(
                "bad_identifier",
                &residual,
                &stage,
                invalid,
                "stage_out",
            )
            .unwrap_err();
            assert!(
                error.contains("WGSL") || error.contains("reserved"),
                "unexpected identifier error for {invalid}: {error}"
            );
        }
    }

    #[test]
    fn explicit_ping_pong_fusion_alpha_renames_before_rhs_substitution() {
        let (residual, mut stage) = explicit_residual_stage_fixture();
        let rename = BTreeMap::from([("rate".to_string(), "r0".to_string())]);
        stage.body = rename_stmts(&stage.body, &rename);
        let fused = synthesize_explicit_residual_rk_ping_pong(
            "capture_safe",
            &residual,
            &stage,
            "stage_in",
            "stage_out",
        )
        .expect("disjoint alpha-renaming must make same-spelled locals safe");
        let src = lower_kernel_program_to_wgsl(&fused)
            .expect("lower capture-safe program")
            .to_wgsl();
        assert!(
            src.contains("let r0: f32 = stage_in"),
            "residual local missing:\n{src}"
        );
        assert!(
            src.contains("let k1_cfd2_stage_local_0_r0: f32 = r0;"),
            "substituted residual expression was captured by the stage alpha-renamer:\n{src}"
        );
    }

    #[test]
    fn explicit_ping_pong_dominance_canonicalizes_omitted_zero_offset() {
        use cfd2_ir::ast::{Expr, Stmt};
        let (residual, mut stage) = explicit_residual_stage_fixture();
        let slot1 = Expr::ident("idx") * Expr::lit_u32(3) + Expr::lit_u32(1);
        stage.body[2] = Stmt::Assign {
            target: Expr::ident("state").index(slot1),
            value: Expr::ident("state").index(Expr::ident("idx") * Expr::lit_u32(3)),
        };
        let fused = synthesize_explicit_residual_rk_ping_pong(
            "zero_offset_canonical",
            &residual,
            &stage,
            "stage_in",
            "stage_out",
        )
        .expect("canonical zero-offset dominance");
        let src = lower_kernel_program_to_wgsl(&fused)
            .expect("lower canonical program")
            .to_wgsl();
        assert!(
            src.contains("stage_out[idx * 3u + 1u] = stage_out[idx * 3u]"),
            "algebraically identical prior write was routed to stale input:\n{src}"
        );
    }
}
