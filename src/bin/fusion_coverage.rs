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
                ModelKernelArtifact::DslProgram(_) => "DSL",
                ModelKernelArtifact::Wgsl(_) => "WGSL",
            };

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

    Ok(ModelCoverage {
        model_id: spec.id,
        rows,
    })
}

fn render_report(coverages: &[ModelCoverage]) -> String {
    let mut out = String::new();
    out.push_str("# Kernel Fusion Coverage Report\n\n");
    out.push_str("Generated by `cargo run --bin fusion_coverage -- --write FUSION_COVERAGE.md`.\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Model | DSL | WGSL | Synthesized | Legacy eligible for DSL migration |\n");
    out.push_str("|---|---:|---:|---:|---:|\n");

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

        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |\n",
            coverage.model_id, dsl, wgsl, synthesized, eligible
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
