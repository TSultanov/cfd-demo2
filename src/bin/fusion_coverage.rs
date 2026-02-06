use cfd2_codegen::solver::codegen::fusion::{synthesize_fused_program, FusionSafetyPolicy};
use cfd2::solver::model;
use cfd2::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, KernelWgslScope, ModelKernelArtifact,
};
use cfd2::solver::scheme::Scheme;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
struct KernelCoverageRow {
    id: &'static str,
    phase: Option<KernelPhaseId>,
    dispatch: Option<DispatchKindId>,
    condition: Option<KernelConditionId>,
    scope: Option<KernelWgslScope>,
    implementation: &'static str,
    eligible_for_dsl_migration: bool,
    note: String,
}

#[derive(Debug, Clone)]
struct ModelCoverage {
    model_id: &'static str,
    rows: Vec<KernelCoverageRow>,
    opportunities: Vec<FusionOpportunityRow>,
}

#[derive(Debug, Clone)]
struct FusionOpportunityRow {
    lhs: &'static str,
    rhs: &'static str,
    phase: KernelPhaseId,
    dispatch: DispatchKindId,
    lhs_impl: &'static str,
    rhs_impl: &'static str,
    existing_rule: Option<&'static str>,
    safe_fuseable: bool,
    aggressive_fuseable: bool,
    safe_reason: String,
    aggressive_reason: String,
    note: String,
    blocked_by_wgsl: bool,
}

fn phase_name(phase: Option<KernelPhaseId>) -> &'static str {
    match phase {
        Some(KernelPhaseId::Preparation) => "Preparation",
        Some(KernelPhaseId::Gradients) => "Gradients",
        Some(KernelPhaseId::FluxComputation) => "FluxComputation",
        Some(KernelPhaseId::Assembly) => "Assembly",
        Some(KernelPhaseId::Apply) => "Apply",
        Some(KernelPhaseId::Update) => "Update",
        None => "-",
    }
}

fn dispatch_name(dispatch: Option<DispatchKindId>) -> &'static str {
    match dispatch {
        Some(DispatchKindId::Cells) => "Cells",
        Some(DispatchKindId::Faces) => "Faces",
        None => "-",
    }
}

fn condition_name(condition: Option<KernelConditionId>) -> &'static str {
    match condition {
        Some(KernelConditionId::Always) => "Always",
        Some(KernelConditionId::RequiresGradState) => "RequiresGradState",
        Some(KernelConditionId::RequiresNoGradState) => "RequiresNoGradState",
        Some(KernelConditionId::RequiresImplicitStepping) => "RequiresImplicitStepping",
        None => "-",
    }
}

fn scope_name(scope: Option<KernelWgslScope>) -> &'static str {
    match scope {
        Some(KernelWgslScope::PerModel) => "PerModel",
        Some(KernelWgslScope::Shared) => "Shared",
        None => "-",
    }
}

fn compact_error(err: &str) -> String {
    let mut compact = err.replace('\n', " ").replace('|', "/");
    if compact.len() > 120 {
        compact.truncate(117);
        compact.push_str("...");
    }
    compact
}

fn collect_model_coverage(
    spec: &model::ModelSpec,
    schemes: &model::backend::SchemeRegistry,
) -> Result<ModelCoverage, String> {
    let active_specs = model::kernel::derive_kernel_specs_for_model(spec)?;

    let mut phase_by_id = BTreeMap::<&'static str, KernelPhaseId>::new();
    let mut dispatch_by_id = BTreeMap::<&'static str, DispatchKindId>::new();
    let mut condition_by_id = BTreeMap::<&'static str, KernelConditionId>::new();
    for module in &spec.modules {
        for kernel in &module.kernels {
            phase_by_id.insert(kernel.id.as_str(), kernel.phase);
            dispatch_by_id.insert(kernel.id.as_str(), kernel.dispatch);
            condition_by_id.insert(kernel.id.as_str(), kernel.condition);
        }
    }
    for kernel in &active_specs {
        phase_by_id.insert(kernel.id.as_str(), kernel.phase);
        dispatch_by_id.insert(kernel.id.as_str(), kernel.dispatch);
        condition_by_id.insert(kernel.id.as_str(), kernel.condition);
    }
    for rule in model::kernel::derive_kernel_fusion_rules_for_model(spec) {
        let replacement = rule.replacement;
        phase_by_id.insert(replacement.id.as_str(), replacement.phase);
        dispatch_by_id.insert(replacement.id.as_str(), replacement.dispatch);
        condition_by_id.insert(replacement.id.as_str(), replacement.condition);
    }

    let replacement_ids = model::kernel::derive_fusion_replacement_kernel_ids_for_model(spec);
    let replacement_set = replacement_ids
        .iter()
        .map(|id| id.as_str())
        .collect::<BTreeSet<_>>();
    let rules = model::kernel::derive_kernel_fusion_rules_for_model(spec);

    let mut impl_by_id = BTreeMap::<&'static str, &'static str>::new();
    let mut dsl_program_by_id =
        BTreeMap::<&'static str, cfd2_ir::solver::ir::KernelProgram>::new();

    let mut by_id = BTreeMap::<&'static str, KernelCoverageRow>::new();
    for module in &spec.modules {
        for generator in &module.generators {
            let artifact = (generator.generator.as_ref())(spec, schemes).map_err(|e| {
                format!(
                    "model '{}' kernel '{}' generator failed during coverage collection: {e}",
                    spec.id,
                    generator.id.as_str()
                )
            })?;

            let implementation = match artifact {
                ModelKernelArtifact::DslProgram(program) => {
                    dsl_program_by_id.insert(generator.id.as_str(), program);
                    "DSL"
                }
                ModelKernelArtifact::Wgsl(_) => "WGSL",
            };
            impl_by_id.insert(generator.id.as_str(), implementation);

            let phase = phase_by_id.get(generator.id.as_str()).copied();
            let eligible_for_dsl_migration =
                implementation == "WGSL" && generator.scope == KernelWgslScope::PerModel
                    && matches!(phase, Some(KernelPhaseId::Assembly | KernelPhaseId::Update));

            let note = if replacement_set.contains(generator.id.as_str()) {
                "replacement kernel has explicit generator".to_string()
            } else {
                String::new()
            };

            by_id.insert(
                generator.id.as_str(),
                KernelCoverageRow {
                    id: generator.id.as_str(),
                    phase,
                    dispatch: dispatch_by_id.get(generator.id.as_str()).copied(),
                    condition: condition_by_id.get(generator.id.as_str()).copied(),
                    scope: Some(generator.scope),
                    implementation,
                    eligible_for_dsl_migration,
                    note,
                },
            );
        }
    }

    for replacement in replacement_ids {
        let id = replacement.as_str();
        by_id.entry(id).or_insert_with(|| KernelCoverageRow {
            id,
            phase: phase_by_id.get(id).copied(),
            dispatch: dispatch_by_id.get(id).copied(),
            condition: condition_by_id.get(id).copied(),
            scope: None,
            implementation: "Synthesized",
            eligible_for_dsl_migration: false,
            note: "replacement kernel synthesized from fusion DSL inputs".to_string(),
        });
    }

    let mut rows: Vec<KernelCoverageRow> = by_id.into_values().collect();
    rows.sort_by(|a, b| {
        phase_name(a.phase)
            .cmp(phase_name(b.phase))
            .then(a.id.cmp(b.id))
    });

    let mut opportunities = Vec::<FusionOpportunityRow>::new();
    for pair in active_specs.windows(2) {
        let lhs = pair[0];
        let rhs = pair[1];
        if lhs.phase != rhs.phase || lhs.dispatch != rhs.dispatch {
            continue;
        }

        let lhs_impl = impl_by_id.get(lhs.id.as_str()).copied().unwrap_or("Missing");
        let rhs_impl = impl_by_id.get(rhs.id.as_str()).copied().unwrap_or("Missing");

        let existing_rule = rules
            .iter()
            .find(|rule| {
                rule.phase == lhs.phase
                    && rule.pattern.len() == 2
                    && rule.pattern[0].id == lhs.id
                    && rule.pattern[1].id == rhs.id
            })
            .map(|rule| rule.name);

        let (
            safe_fuseable,
            aggressive_fuseable,
            safe_reason,
            aggressive_reason,
            note,
            blocked_by_wgsl,
        ) = if lhs_impl == "DSL" && rhs_impl == "DSL" {
            let lhs_program = dsl_program_by_id
                .get(lhs.id.as_str())
                .ok_or_else(|| format!("missing DSL program for '{}'", lhs.id.as_str()))?;
            let rhs_program = dsl_program_by_id
                .get(rhs.id.as_str())
                .ok_or_else(|| format!("missing DSL program for '{}'", rhs.id.as_str()))?;
            let pair_programs = vec![lhs_program.clone(), rhs_program.clone()];
            let (safe_fuseable, safe_reason) = match synthesize_fused_program(
                format!("reassess/{}_{}", lhs.id.as_str(), rhs.id.as_str()),
                "reassess_pair",
                &pair_programs,
                FusionSafetyPolicy::Safe,
            ) {
                Ok(_) => (true, "compatible".to_string()),
                Err(err) => (false, compact_error(&err)),
            };
            let (aggressive_fuseable, aggressive_reason) = match synthesize_fused_program(
                format!("reassess/{}_{}", lhs.id.as_str(), rhs.id.as_str()),
                "reassess_pair",
                &pair_programs,
                FusionSafetyPolicy::Aggressive,
            ) {
                Ok(_) => (true, "compatible".to_string()),
                Err(err) => (false, compact_error(&err)),
            };
            let note = if !safe_fuseable && aggressive_fuseable {
                "aggressive-only opportunity".to_string()
            } else {
                String::new()
            };
            (
                safe_fuseable,
                aggressive_fuseable,
                safe_reason,
                aggressive_reason,
                note,
                false,
            )
        } else {
            (
                false,
                false,
                "requires DSL migration".to_string(),
                "requires DSL migration".to_string(),
                format!("artifacts: {lhs_impl} + {rhs_impl}"),
                true,
            )
        };

        opportunities.push(FusionOpportunityRow {
            lhs: lhs.id.as_str(),
            rhs: rhs.id.as_str(),
            phase: lhs.phase,
            dispatch: lhs.dispatch,
            lhs_impl,
            rhs_impl,
            existing_rule,
            safe_fuseable,
            aggressive_fuseable,
            safe_reason,
            aggressive_reason,
            note,
            blocked_by_wgsl,
        });
    }

    Ok(ModelCoverage {
        model_id: spec.id,
        rows,
        opportunities,
    })
}

fn render_report(coverages: &[ModelCoverage]) -> String {
    let mut out = String::new();
    out.push_str("# Kernel Fusion Coverage Report\n\n");
    out.push_str("Generated by `cargo run --bin fusion_coverage -- --write FUSION_COVERAGE.md`.\n\n");
    out.push_str("## Summary\n\n");
    out.push_str(
        "| Model | DSL | WGSL | Synthesized | Legacy eligible for DSL migration | Pair opportunities (safe) | Pair opportunities (aggressive-only) | Pair opportunities blocked by WGSL |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");

    for coverage in coverages {
        let dsl = coverage
            .rows
            .iter()
            .filter(|row| row.implementation == "DSL")
            .count();
        let wgsl = coverage
            .rows
            .iter()
            .filter(|row| row.implementation == "WGSL")
            .count();
        let synthesized = coverage
            .rows
            .iter()
            .filter(|row| row.implementation == "Synthesized")
            .count();
        let eligible = coverage
            .rows
            .iter()
            .filter(|row| row.eligible_for_dsl_migration)
            .count();
        let safe_pairs = coverage
            .opportunities
            .iter()
            .filter(|row| row.safe_fuseable)
            .count();
        let aggressive_only_pairs = coverage
            .opportunities
            .iter()
            .filter(|row| !row.safe_fuseable && row.aggressive_fuseable)
            .count();
        let blocked_pairs = coverage
            .opportunities
            .iter()
            .filter(|row| row.blocked_by_wgsl)
            .count();

        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |\n",
            coverage.model_id, dsl, wgsl, synthesized, eligible, safe_pairs, aggressive_only_pairs, blocked_pairs
        ));
    }

    out.push_str("\n## Per-Model Details\n");

    for coverage in coverages {
        out.push_str(&format!("\n### `{}`\n\n", coverage.model_id));
        out.push_str(
            "| Kernel | Phase | Dispatch | Condition | Scope | Implementation | Eligible for DSL migration | Notes |\n",
        );
        out.push_str("|---|---|---|---|---|---|---|---|\n");
        for row in &coverage.rows {
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |\n",
                row.id,
                phase_name(row.phase),
                dispatch_name(row.dispatch),
                condition_name(row.condition),
                scope_name(row.scope),
                row.implementation,
                if row.eligible_for_dsl_migration {
                    "yes"
                } else {
                    "no"
                },
                if row.note.is_empty() { "-" } else { &row.note }
            ));
        }

        out.push_str("\n#### Adjacent Fusion Reassessment\n\n");
        out.push_str(
            "| Pair | Phase | Dispatch | Artifacts | Existing rule | Safe fuseable | Aggressive fuseable | Notes |\n",
        );
        out.push_str("|---|---|---|---|---|---|---|---|\n");
        for row in &coverage.opportunities {
            let safe = if row.safe_fuseable {
                "yes".to_string()
            } else {
                format!("no ({})", row.safe_reason)
            };
            let aggressive = if row.aggressive_fuseable {
                "yes".to_string()
            } else {
                format!("no ({})", row.aggressive_reason)
            };
            let note = if row.note.is_empty() { "-" } else { &row.note };
            out.push_str(&format!(
                "| `{} -> {}` | {} | {} | {} + {} | {} | {} | {} | {} |\n",
                row.lhs,
                row.rhs,
                phase_name(Some(row.phase)),
                dispatch_name(Some(row.dispatch)),
                row.lhs_impl,
                row.rhs_impl,
                row.existing_rule.unwrap_or("-"),
                safe,
                aggressive,
                note
            ));
        }
    }

    out
}

fn write_report(path: &std::path::Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, content)
}

fn parse_write_path(args: &[String]) -> Option<String> {
    args.windows(2).find_map(|w| {
        if w[0] == "--write" {
            Some(w[1].clone())
        } else {
            None
        }
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let write_path = parse_write_path(&args);

    let schemes = model::backend::SchemeRegistry::new(Scheme::Upwind);
    let mut coverages = Vec::new();

    for spec in model::all_models() {
        let coverage = collect_model_coverage(&spec, &schemes).unwrap_or_else(|err| {
            eprintln!("Failed to collect fusion coverage for model '{}': {err}", spec.id);
            std::process::exit(1);
        });
        coverages.push(coverage);
    }
    coverages.sort_by(|a, b| a.model_id.cmp(b.model_id));

    let report = render_report(&coverages);

    if let Some(path) = write_path {
        let path = std::path::PathBuf::from(path);
        write_report(&path, &report).unwrap_or_else(|err| {
            eprintln!("Failed to write coverage report to '{}': {err}", path.display());
            std::process::exit(1);
        });
        println!("Wrote {}", path.display());
    } else {
        println!("{report}");
    }
}
